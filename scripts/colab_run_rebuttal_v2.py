#!/usr/bin/env python3
"""Colab L4 Rebuttal v2 Benchmark Suite for LSLC.

Unified single-checkpoint evaluation resolving every forensic review item:
1. Single unified Qwen3-1.7B BF16 adapter checkpoint trained once, frozen, and
   exported to SafeTensors BEFORE running all evaluations.
2. True Constant-k Raw-Text RAG baseline (512 raw tokens, ZERO TOC tokens in LLM KV)
   vs. Flat 1-Level LSLC vs. 2-Level Hierarchical Chapter->Page LSLC across
   32K (P=128), 128K (P=512), and 256K (P=1,024), measuring the exact memory crossover.
3. Non-oracle Okapi BM25 retriever shared identically between LSLC and Raw-Text RAG
   at 32K (P=128, 3 seeds x 60 episodes = 180 total episodes), plus Learned QueryToTOCRouter.
4. Fixed answer metrics: leading-whitespace-stripped `content_tok_acc`, autoregressive
   Exact Match (`em`), and non-teacher-forced character multiset F1 (`gen_char_f1`).
5. Head-to-head 4-way Multi-Turn State-Override evaluation (k=3 numeric, N=108;
   k=5 symbolic, N=90 across 3 seeds) comparing:
     - Raw-Text RAG (Immutable)
     - Raw-Text RAG (Append Errata)
     - Raw-Text RAG (In-Place LLM-Rewrite via frozen Qwen3-1.7B, + full-page rewrite profile)
     - LSLC (In-Place Convex DeltaUpdater, O(1) l=2 override + max|Delta_TOC|=0.0)
   with live CUDA per-edit latency (ms/edit) for every editable system.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lslc import (
    FrozenQwenLSLCWrapper,
    LSLCConfig,
    LivingBookStateOverrideDataset,
    RouterOutput,
    SyntheticExactSemanticNIAHDataset,
    compute_analytical_memory_stats,
    greedy_generate_from_prefix,
    load_frozen_qwen3,
    select_amp_dtype,
)
from lslc.encoder import SparseExactBank
from scripts.colab_run_hard_stress import build_k5_symbolic_override_episodes
from scripts.colab_run_m2 import (
    _cache_bytes,
    _load_wikitext_pools,
    train_stage,
)


def _mean_std_ci(vals: list[float]) -> dict[str, float]:
    n = max(1, len(vals))
    mean = sum(vals) / n
    if n <= 1:
        return {"mean": round(mean, 4), "std": 0.0, "ci95": 0.0}
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    std = math.sqrt(var)
    ci95 = 1.96 * std / math.sqrt(n)
    return {"mean": round(mean, 4), "std": round(std, 4), "ci95": round(ci95, 4)}


def _strip_leading_ws_answer(
    ans_logits: torch.Tensor, a_ids: torch.Tensor, tokenizer
) -> tuple[torch.Tensor, torch.Tensor]:
    if a_ids.shape[1] > 1:
        first_tok = tokenizer.decode([int(a_ids[0, 0].item())])
        if first_tok.strip() == "":
            return ans_logits[:, 1:, :], a_ids[:, 1:]
    return ans_logits, a_ids


def _char_f1(gen_text: str, answer_str: str) -> float:
    g_s = answer_str.strip().lower()
    if not g_s:
        return 1.0 if not gen_text.strip() else 0.0
    p_s = gen_text.strip().lower()[: max(1, len(g_s) * 2)]
    if not p_s:
        return 0.0
    p_cnt = Counter(p_s)
    g_cnt = Counter(g_s)
    common = sum((p_cnt & g_cnt).values())
    if common == 0:
        return 0.0
    return (2.0 * common) / float(len(p_s) + len(g_s))


def compute_answer_metrics(
    ans_logits: torch.Tensor,
    a_ids: torch.Tensor,
    gen_text: str,
    answer_str: str,
    tokenizer,
) -> tuple[float, float, float]:
    logits_c, a_c = _strip_leading_ws_answer(ans_logits, a_ids, tokenizer)
    content_tok_acc = float((logits_c.argmax(dim=-1) == a_c).float().mean().item())
    em = 1.0 if answer_str.lower() in gen_text.lower() else 0.0
    char_f1 = _char_f1(gen_text, answer_str)
    return em, content_tok_acc, char_f1


class BM25PageRetriever:
    """Non-oracle Okapi BM25 lexical retriever over decoded book pages."""

    _TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")

    def __init__(self, page_texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[Counter[str]] = []
        self.doc_lens: list[int] = []
        self.df: Counter[str] = Counter()
        for txt in page_texts:
            toks = self._tokenize(txt)
            self.docs.append(Counter(toks))
            self.doc_lens.append(len(toks))
        self._rebuild_stats()

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return cls._TOKEN_RE.findall(text.lower())

    def _rebuild_stats(self) -> None:
        n_docs = len(self.docs)
        self.avgdl = sum(self.doc_lens) / max(1, n_docs)
        self.df = Counter()
        for cnt in self.docs:
            for term in cnt.keys():
                self.df[term] += 1
        self.idf = {
            term: math.log(((n_docs - freq + 0.5) / (freq + 0.5)) + 1.0)
            for term, freq in self.df.items()
        }

    def update_page(self, page_idx: int, new_text: str) -> None:
        toks = self._tokenize(new_text)
        self.docs[page_idx] = Counter(toks)
        self.doc_lens[page_idx] = len(toks)
        self._rebuild_stats()

    def score_pages(self, query_text: str) -> list[float]:
        q_toks = self._tokenize(query_text)
        scores = [0.0] * len(self.docs)
        avgdl = max(1e-6, self.avgdl)
        for i, cnt in enumerate(self.docs):
            dl = self.doc_lens[i]
            norm = self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
            s = 0.0
            for t in q_toks:
                tf = cnt.get(t, 0)
                if tf > 0:
                    idf = self.idf.get(t, 0.0)
                    s += idf * (tf * (self.k1 + 1.0)) / (tf + norm)
            scores[i] = s
        return scores

    def route(
        self, query_text: str, k_deep: int, k_skim: int, device: torch.device
    ) -> RouterOutput:
        scores = self.score_pages(query_text)
        num_pages = len(scores)
        logits = torch.tensor([scores], dtype=torch.float32, device=device)
        std = float(logits.std().item()) if num_pages > 1 else 0.0
        temp = max(1.0, std)
        probs = F.softmax(logits / temp, dim=-1)

        kd = min(num_pages, max(0, k_deep))
        ks = min(max(0, num_pages - kd), max(0, k_skim))
        sorted_idx = torch.argsort(logits, dim=-1, descending=True)
        # Put highest-scoring page LAST in deep_idx so it sits right next to query.
        deep_idx = sorted_idx[:, :kd].flip(dims=[-1])
        skim_idx = torch.sort(sorted_idx[:, kd : kd + ks], dim=-1).values
        toc_idx = torch.sort(sorted_idx[:, kd + ks :], dim=-1).values
        return RouterOutput(
            logits=logits, probs=probs, deep_idx=deep_idx, skim_idx=skim_idx, toc_idx=toc_idx
        )


def _encode_book_chunked(
    wrapper: FrozenQwenLSLCWrapper,
    pages: torch.Tensor,
    embed_fn,
    vocab_size: int,
    chunk_pages: int = 32,
):
    """Encodes [1, P, Tp] pages in chunks of 32 pages to keep peak activation VRAM O(1)."""
    num_pages = pages.shape[1]
    z_list, toc_list, tok_id_chunks, h_chunks, w_chunks = [], [], [], [], []
    for s in range(0, num_pages, chunk_pages):
        e = min(num_pages, s + chunk_pages)
        p_c = pages[:, s:e]
        z_c, bank_c, toc_c = wrapper.lslc.encode_book(
            p_c, embed_fn, use_exact_bank=True, vocab_size=vocab_size
        )
        z_list.append(z_c)
        toc_list.append(toc_c)
        tok_id_chunks.append(bank_c.token_ids)
        h_chunks.append(bank_c.h_coords)
        w_chunks.append(bank_c.w_coords)

    page_idx = (
        torch.arange(num_pages, device=pages.device, dtype=torch.int16)
        .view(1, num_pages, 1)
        .expand(1, num_pages, wrapper.cfg.k_exact)
        .contiguous()
    )
    bank = SparseExactBank(
        token_ids=torch.cat(tok_id_chunks, dim=1),
        page_ids=page_idx,
        h_coords=torch.cat(h_chunks, dim=1),
        w_coords=torch.cat(w_chunks, dim=1),
    )
    return torch.cat(z_list, dim=1), bank, torch.cat(toc_list, dim=1)


def train_unified_checkpoint(
    wrapper: FrozenQwenLSLCWrapper,
    train_pool: list[int],
    tokenizer,
    device: torch.device,
) -> dict[str, Any]:
    """Train ONE unified Qwen3-1.7B adapter checkpoint cleanly and freeze for all evals."""
    embed_fn = wrapper.llm.get_input_embeddings()
    amp_dtype, _ = select_amp_dtype(device)
    vocab_size = len(embed_fn.weight)

    train_niah_4p = list(
        SyntheticExactSemanticNIAHDataset(
            num_episodes=120,
            num_pages=4,
            page_size=wrapper.cfg.page_size,
            tokenizer=tokenizer,
            seed=42,
            mode="mixed",
            background_token_pool=train_pool,
        )
    )
    train_niah_16p = list(
        SyntheticExactSemanticNIAHDataset(
            num_episodes=80,
            num_pages=16,
            page_size=wrapper.cfg.page_size,
            tokenizer=tokenizer,
            seed=142,
            mode="mixed",
            background_token_pool=train_pool,
        )
    )
    base_stats = train_stage(
        "V2_BASE",
        wrapper,
        train_niah_4p + train_niah_16p,
        steps=220,
        grad_accum=2,
        lr=3e-4,
        device=device,
        use_exact_bank=True,
        distill_salience=True,
        run_matryoshka=True,
    )

    # Freeze encoder/projector/salience_gate while curriculum-training DeltaUpdater (k=3 then k=5)
    frozen_delta_stage = []
    for name_p, p in wrapper.lslc.named_parameters():
        if not (name_p.startswith("delta_updater.") or name_p.startswith("router.")) and p.requires_grad:
            p.requires_grad_(False)
            frozen_delta_stage.append(p)

    train_k3 = list(
        LivingBookStateOverrideDataset(
            num_episodes=96,
            num_pages=4,
            page_size=wrapper.cfg.page_size,
            num_edits=3,
            tokenizer=tokenizer,
            seed=42,
            background_token_pool=train_pool,
        )
    )
    k3_stats = train_stage(
        "V2_DELTA_K3",
        wrapper,
        train_k3,
        steps=120,
        grad_accum=2,
        lr=3e-4,
        device=device,
        use_exact_bank=True,
        distill_salience=False,
        run_matryoshka=False,
        use_edits=True,
    )

    train_k5 = build_k5_symbolic_override_episodes(
        tokenizer, train_pool, num_episodes=60, num_pages=4, num_edits=5, seed=8001
    )
    k5_stats = train_stage(
        "V2_DELTA_K5",
        wrapper,
        train_k5,
        steps=110,
        grad_accum=2,
        lr=3e-4,
        device=device,
        use_exact_bank=True,
        distill_salience=False,
        run_matryoshka=False,
        use_edits=True,
    )
    for p in frozen_delta_stage:
        p.requires_grad_(True)

    # Fine-tune QueryToTOCRouter (plus toc_proj) on P=128 (32K) books WITHOUT altering encoder/delta_updater
    frozen_router_stage = []
    for name_p, p in wrapper.lslc.named_parameters():
        if not (name_p.startswith("router.") or name_p.startswith("projector.toc_proj.")) and p.requires_grad:
            p.requires_grad_(False)
            frozen_router_stage.append(p)

    ds_128 = SyntheticExactSemanticNIAHDataset(
        num_episodes=100,
        num_pages=128,
        page_size=wrapper.cfg.page_size,
        tokenizer=tokenizer,
        seed=777,
        mode="mixed",
        background_token_pool=train_pool,
    )
    opt = torch.optim.AdamW(
        list(wrapper.lslc.router.parameters())
        + list(wrapper.lslc.projector.toc_proj.parameters()),
        lr=1e-3,
        weight_decay=0.01,
    )
    wrapper.lslc.eval()
    wrapper.lslc.router.train()
    wrapper.lslc.projector.toc_proj.train()
    t_r0 = time.perf_counter()
    last_r_ce = 0.0
    for step_i in range(100):
        ep = ds_128[step_i]
        pages = ep.context_pages.unsqueeze(0).to(device)
        q_ids = ep.query_ids.unsqueeze(0).to(device)
        target_p = torch.tensor([ep.target_page_idx], dtype=torch.long, device=device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            z_list = []
            for s in range(0, 128, 32):
                z_c, _, _ = wrapper.lslc.encode_book(
                    pages[:, s : s + 32], embed_fn, use_exact_bank=False, vocab_size=vocab_size
                )
                z_list.append(z_c)
            z_book = torch.cat(z_list, dim=1)
            q_emb = embed_fn(q_ids).mean(dim=1)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            toc_tokens = wrapper.lslc.projector.project_toc(z_book)
            route_out = wrapper.lslc.router(q_emb, toc_tokens, k_deep=2, k_skim=4)
            loss = F.cross_entropy(route_out.logits, target_p)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(wrapper.lslc.router.parameters()) + list(wrapper.lslc.projector.toc_proj.parameters()),
            1.0,
        )
        opt.step()
        last_r_ce = float(loss.item())
        if (step_i + 1) in (1, 25, 50, 75, 100):
            print(f"  [V2_ROUTER_128P] step={step_i+1}/100 route_ce={last_r_ce:.4f}", flush=True)

    for p in frozen_router_stage:
        p.requires_grad_(True)
    wrapper.lslc.eval()

    return {
        "base": base_stats,
        "delta_k3": k3_stats,
        "delta_k5": k5_stats,
        "router_128p": {
            "steps": 100,
            "elapsed_sec": round(time.perf_counter() - t_r0, 2),
            "final_route_ce": round(last_r_ce, 4),
        },
        "trainable_params": wrapper.lslc.trainable_parameter_count(),
    }


@torch.no_grad()
def profile_memory_and_latency_scaling(
    wrapper: FrozenQwenLSLCWrapper,
    val_pool: list[int],
    device: torch.device,
) -> dict[str, Any]:
    """Profiles live CUDA memory & TTFT across 32K, 128K, and 256K for Constant-k RAG, Flat LSLC, and 2-Level Hier LSLC."""
    import inspect

    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    amp_dtype, _ = select_amp_dtype(device)
    vocab_size = len(embed_fn.weight)
    mib = float(1024 ** 2)
    fwd_params = inspect.signature(wrapper.llm.forward).parameters
    extra_kw = {"logits_to_keep": 1} if "logits_to_keep" in fwd_params else {}
    wrapper.lslc.eval()

    prev_cache = getattr(wrapper.llm.config, "use_cache", False)
    wrapper.llm.config.use_cache = True
    rows: dict[str, Any] = {}

    try:
        for tag, ctx_len in (("32k", 32768), ("128k", 131072), ("256k", 262144)):
            num_pages = ctx_len // wrapper.cfg.page_size
            stream = (val_pool * ((ctx_len // len(val_pool)) + 2))[:ctx_len]
            pages = torch.tensor(stream, dtype=torch.long, device=device).view(
                1, num_pages, wrapper.cfg.page_size
            )
            q_ids = pages[:, 0, :16]
            turn_emb = embed_fn(q_ids).to(llm_dtype)

            # Warmup + live chunked encode latency
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _ = _encode_book_chunked(wrapper, pages[:, : min(32, num_pages)], embed_fn, vocab_size, chunk_pages=32)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                z_book, bank, toc_all = _encode_book_chunked(
                    wrapper, pages, embed_fn, vocab_size, chunk_pages=32
                )
            torch.cuda.synchronize()
            encode_ms = (time.perf_counter() - t0) * 1000.0
            book_bytes = z_book.numel() * 2
            bank_bytes = bank.storage_bytes()

            # 1. Constant-k Raw-Text RAG: strictly 512 raw tokens + 16 query = 528 active tokens (NO TOC tokens in LLM KV)
            rag_pages = torch.cat((pages[:, 0, :], pages[:, -1, :]), dim=1)  # [1, 512]
            rag_inp = torch.cat((embed_fn(rag_pages).to(llm_dtype), turn_emb), dim=1)
            rag_out = wrapper.llm(inputs_embeds=rag_inp, use_cache=True, **extra_kw)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                rag_out = wrapper.llm(inputs_embeds=rag_inp, use_cache=True, **extra_kw)
            torch.cuda.synchronize()
            rag_ttft_ms = (time.perf_counter() - t0) * 1000.0 / 5.0
            rag_kv_bytes = _cache_bytes(rag_out.past_key_values)
            rag_store_bytes = num_pages * wrapper.cfg.page_size * 4 + num_pages * 128
            rag_net_mib = (rag_store_bytes + rag_kv_bytes) / mib
            del rag_out

            # 2. LSLC Flat 1-Level TOC
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                q_emb = embed_fn(q_ids).mean(dim=1)
                _, asm_flat = wrapper.lslc.route_and_assemble(
                    z_book, toc_all, q_emb, bank, embed_fn, k_deep=2, k_skim=4, mode="full", training_ste=False
                )
            flat_inp = torch.cat((asm_flat.prefix_embeds.to(llm_dtype), turn_emb), dim=1)
            flat_out = wrapper.llm(inputs_embeds=flat_inp, use_cache=True, **extra_kw)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    q_emb_i = embed_fn(q_ids).mean(dim=1)
                    _, asm_i = wrapper.lslc.route_and_assemble(
                        z_book, toc_all, q_emb_i, bank, embed_fn, k_deep=2, k_skim=4, mode="full", training_ste=False
                    )
                flat_out = wrapper.llm(
                    inputs_embeds=torch.cat((asm_i.prefix_embeds.to(llm_dtype), turn_emb), dim=1),
                    use_cache=True,
                    **extra_kw,
                )
            torch.cuda.synchronize()
            flat_cold_ttft_ms = (time.perf_counter() - t0) * 1000.0 / 5.0
            flat_kv_bytes = _cache_bytes(flat_out.past_key_values)
            flat_net_mib = (book_bytes + bank_bytes + flat_kv_bytes) / mib
            imm_f = asm_flat.immutable_prefix_len
            toc_kv_f = wrapper.llm(inputs_embeds=flat_inp[:, :imm_f], use_cache=True, **extra_kw).past_key_values
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                _ = wrapper.llm(inputs_embeds=flat_inp[:, imm_f:], past_key_values=toc_kv_f, use_cache=False, **extra_kw)
            torch.cuda.synchronize()
            flat_warm_ttft_ms = (time.perf_counter() - t0) * 1000.0 / 5.0
            del flat_out, toc_kv_f

            # 3. LSLC 2-Level Hierarchical Chapter->Page TOC (chapter_size=16)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                route_h, asm_hier = wrapper.lslc.route_and_assemble(
                    z_book,
                    toc_all,
                    q_emb,
                    bank,
                    embed_fn,
                    k_deep=2,
                    k_skim=4,
                    mode="full",
                    training_ste=False,
                    hierarchical_chapter_size=16,
                )
            hier_inp = torch.cat((asm_hier.prefix_embeds.to(llm_dtype), turn_emb), dim=1)
            hier_out = wrapper.llm(inputs_embeds=hier_inp, use_cache=True, **extra_kw)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    q_emb_h = embed_fn(q_ids).mean(dim=1)
                    _, asm_hi = wrapper.lslc.route_and_assemble(
                        z_book,
                        toc_all,
                        q_emb_h,
                        bank,
                        embed_fn,
                        k_deep=2,
                        k_skim=4,
                        mode="full",
                        training_ste=False,
                        hierarchical_chapter_size=16,
                    )
                hier_out = wrapper.llm(
                    inputs_embeds=torch.cat((asm_hi.prefix_embeds.to(llm_dtype), turn_emb), dim=1),
                    use_cache=True,
                    **extra_kw,
                )
            torch.cuda.synchronize()
            hier_cold_ttft_ms = (time.perf_counter() - t0) * 1000.0 / 5.0
            hier_kv_bytes = _cache_bytes(hier_out.past_key_values)
            hier_net_mib = (book_bytes + bank_bytes + hier_kv_bytes) / mib
            imm_h = asm_hier.immutable_prefix_len
            toc_kv_h = wrapper.llm(inputs_embeds=hier_inp[:, :imm_h], use_cache=True, **extra_kw).past_key_values
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                _ = wrapper.llm(inputs_embeds=hier_inp[:, imm_h:], past_key_values=toc_kv_h, use_cache=False, **extra_kw)
            torch.cuda.synchronize()
            hier_warm_ttft_ms = (time.perf_counter() - t0) * 1000.0 / 5.0
            del hier_out, toc_kv_h

            # Verify max|Delta_TOC| == 0.0 under hierarchical assembly
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                z_edited, _ = wrapper.lslc.apply_page_delta(
                    z_book,
                    torch.tensor([int(route_h.deep_idx[0, -1].item())], device=device),
                    q_ids,
                    embed_fn,
                )
                toc_edited = wrapper.lslc.projector.project_toc(z_edited)
                _, asm_hier_after = wrapper.lslc.route_and_assemble(
                    z_edited,
                    toc_edited,
                    q_emb,
                    bank,
                    embed_fn,
                    k_deep=2,
                    k_skim=4,
                    mode="full",
                    training_ste=False,
                    router_out_override=route_h,
                    hierarchical_chapter_size=16,
                )
            hier_invariance_diff = float(
                (asm_hier_after.prefix_embeds[:, :imm_h] - asm_hier.prefix_embeds[:, :imm_h]).abs().max().item()
            )
            del z_edited, toc_edited, asm_hier_after, z_book, bank, toc_all, pages
            torch.cuda.empty_cache()

            analytical = compute_analytical_memory_stats(wrapper.cfg, ctx_len, k_deep=2, k_skim=4)
            full_kv_mib = analytical["full_fp16_kv_mib"]

            rows[tag] = {
                "context_tokens": ctx_len,
                "num_pages": num_pages,
                "encode_total_ms": round(encode_ms, 2),
                "encode_per_page_ms": round(encode_ms / num_pages, 3),
                "full_fp16_kv_mib": round(full_kv_mib, 2),
                "rag_const_k2": {
                    "active_prefix_tokens": 512,
                    "active_seq_tokens_with_q": 528,
                    "net_vram_mib": round(rag_net_mib, 2),
                    "query_ttft_ms": round(rag_ttft_ms, 2),
                },
                "lslc_flat_toc": {
                    "active_prefix_tokens": int(asm_flat.prefix_embeds.shape[1]),
                    "net_vram_mib": round(flat_net_mib, 2),
                    "ratio_vs_full_kv": round(full_kv_mib / flat_net_mib, 2),
                    "ratio_vs_rag_const_k2": round(rag_net_mib / flat_net_mib, 2),
                    "cold_query_ttft_ms": round(flat_cold_ttft_ms, 2),
                    "warm_toc_ttft_ms": round(flat_warm_ttft_ms, 2),
                },
                "lslc_hier_toc_c16": {
                    "active_prefix_tokens": int(asm_hier.prefix_embeds.shape[1]),
                    "net_vram_mib": round(hier_net_mib, 2),
                    "ratio_vs_full_kv": round(full_kv_mib / hier_net_mib, 2),
                    "ratio_vs_rag_const_k2": round(rag_net_mib / hier_net_mib, 2),
                    "cold_query_ttft_ms": round(hier_cold_ttft_ms, 2),
                    "warm_toc_ttft_ms": round(hier_warm_ttft_ms, 2),
                    "max_toc_prefix_diff_on_edit": hier_invariance_diff,
                },
            }
            print(
                f"[SCALE {tag}] RAG_k2={rag_net_mib:.2f} MiB (528 tok) | "
                f"LSLC_Flat={flat_net_mib:.2f} MiB ({asm_flat.prefix_embeds.shape[1]} pref) | "
                f"LSLC_Hier16={hier_net_mib:.2f} MiB ({asm_hier.prefix_embeds.shape[1]} pref, diff={hier_invariance_diff})",
                flush=True,
            )

        # Exact analytical crossover page count P* for Qwen3-1.7B where Flat LSLC net VRAM == Constant-k=2 RAG net VRAM:
        # Flat LSLC bytes(P) = (12416 + 114688) * P + 90 * 114688
        # Const RAG bytes(P) = (1024 + 128) * P + 512 * 114688
        kv_tok = wrapper.cfg.kv_bytes_per_token_fp16  # 114,688 B for Qwen3-1.7B
        lslc_slope = 12_288 + 128 + kv_tok
        lslc_const = (16 + 48 + 32 - 6) * kv_tok
        rag_slope = 256 * 4 + 128
        rag_const = 512 * kv_tok
        p_star = (rag_const - lslc_const) / (lslc_slope - rag_slope)
        rows["analytical_flat_vs_const_k2_rag_crossover"] = {
            "crossover_pages": round(p_star, 1),
            "crossover_tokens": int(round(p_star * wrapper.cfg.page_size)),
            "note": "Exact page count P* where 1-level Flat LSLC active-KV + store overtakes Constant-k=2 RAG (eliminated by 2-level Hierarchical TOC).",
        }
    finally:
        wrapper.llm.config.use_cache = prev_cache

    return rows


@torch.no_grad()
def evaluate_32k_niah_rebuttal(
    wrapper: FrozenQwenLSLCWrapper,
    val_pool: list[int],
    tokenizer,
    device: torch.device,
    seeds: list[int],
    episodes_per_seed: int = 60,
) -> dict[str, Any]:
    """3-seed P=128 (32,768-token) NIAH benchmark with Shared Okapi BM25 vs Learned Router + Fixed Metrics."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    amp_dtype, _ = select_amp_dtype(device)
    vocab_size = len(embed_fn.weight)
    wrapper.lslc.eval()

    systems = (
        "lslc_learned_router_flat",
        "lslc_shared_bm25_flat",
        "lslc_shared_bm25_hier",
        "raw_rag_shared_bm25",
    )
    per_sys_exact_em = {s: [] for s in systems}
    per_sys_exact_tok = {s: [] for s in systems}
    per_sys_exact_f1 = {s: [] for s in systems}
    per_sys_sem_em = {s: [] for s in systems}
    per_sys_sem_tok = {s: [] for s in systems}
    per_sys_sem_f1 = {s: [] for s in systems}

    capture_rates = []
    lr_t1_all, lr_t2_all, lr_t1_ex, lr_t2_ex, lr_t1_sem, lr_t2_sem = [], [], [], [], [], []
    bm_t1_all, bm_t2_all, bm_t1_ex, bm_t2_ex, bm_t1_sem, bm_t2_sem = [], [], [], [], [], []

    def _eval_prefix(prefix_llm: torch.Tensor, q_ids: torch.Tensor, a_ids: torch.Tensor, turn_emb: torch.Tensor, ans_str: str):
        inp = torch.cat((prefix_llm, turn_emb), dim=1)
        logits = wrapper.llm(inputs_embeds=inp).logits
        start = prefix_llm.shape[1] + q_ids.shape[1] - 1
        ans_logits = logits[:, start : start + a_ids.shape[1], :]
        gen_ids = greedy_generate_from_prefix(
            wrapper.llm, prefix_llm, q_ids, max_new_tokens=24, eos_token_id=tokenizer.eos_token_id
        )
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return compute_answer_metrics(ans_logits, a_ids, gen_text, ans_str, tokenizer)

    for seed in seeds:
        ds = SyntheticExactSemanticNIAHDataset(
            num_episodes=episodes_per_seed,
            num_pages=128,
            page_size=wrapper.cfg.page_size,
            tokenizer=tokenizer,
            seed=seed,
            mode="mixed",
            background_token_pool=val_pool,
        )
        sys_e_em = {s: [] for s in systems}
        sys_e_tok = {s: [] for s in systems}
        sys_e_f1 = {s: [] for s in systems}
        sys_s_em = {s: [] for s in systems}
        sys_s_tok = {s: [] for s in systems}
        sys_s_f1 = {s: [] for s in systems}

        cap_hits, cap_tot = 0, 0
        lr_t1_e, lr_t2_e, lr_t1_s, lr_t2_s = [], [], [], []
        bm_t1_e, bm_t2_e, bm_t1_s, bm_t2_s = [], [], [], []

        for idx_ep, ep in enumerate(ds):
            pages = ep.context_pages.unsqueeze(0).to(device)
            q_ids = ep.query_ids.unsqueeze(0).to(device)
            a_ids = ep.answer_ids.unsqueeze(0).to(device)
            tp = ep.target_page_idx
            is_exact = ep.metadata["needle_type"] == "exact"
            ans_str = ep.metadata["answer_str"]
            turn_emb = embed_fn(torch.cat((q_ids, a_ids), dim=1)).to(llm_dtype)

            page_texts = [tokenizer.decode(ep.context_pages[p].tolist(), skip_special_tokens=True) for p in range(128)]
            query_text = tokenizer.decode(ep.query_ids.tolist(), skip_special_tokens=True)
            bm25 = BM25PageRetriever(page_texts)
            bm_route = bm25.route(query_text, k_deep=2, k_skim=4, device=device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                z_book, bank, toc = _encode_book_chunked(
                    wrapper, pages, embed_fn, vocab_size, chunk_pages=32
                )
                q_emb = embed_fn(q_ids).mean(dim=1)
                lr_route, asm_lr_flat = wrapper.lslc.route_and_assemble(
                    z_book, toc, q_emb, bank, embed_fn, k_deep=2, k_skim=4, mode="full", training_ste=False
                )
                _, asm_bm_flat = wrapper.lslc.route_and_assemble(
                    z_book,
                    toc,
                    q_emb,
                    bank,
                    embed_fn,
                    k_deep=2,
                    k_skim=4,
                    mode="full",
                    training_ste=False,
                    router_out_override=bm_route,
                )
                _, asm_bm_hier = wrapper.lslc.route_and_assemble(
                    z_book,
                    toc,
                    q_emb,
                    bank,
                    embed_fn,
                    k_deep=2,
                    k_skim=4,
                    mode="full",
                    training_ste=False,
                    router_out_override=bm_route,
                    hierarchical_chapter_size=16,
                )

            lr_deep = lr_route.deep_idx[0].tolist()
            bm_deep = bm_route.deep_idx[0].tolist()
            if is_exact:
                lr_t1_e.append(1.0 if lr_deep[-1] == tp else 0.0)
                lr_t2_e.append(1.0 if tp in lr_deep else 0.0)
                bm_t1_e.append(1.0 if bm_deep[-1] == tp else 0.0)
                bm_t2_e.append(1.0 if tp in bm_deep else 0.0)

                captured = set(bank.token_ids[0, tp].tolist())
                ans_only_ids = tokenizer.encode(ans_str, add_special_tokens=False)
                for tid in ans_only_ids:
                    cap_tot += 1
                    if tid in captured:
                        cap_hits += 1
            else:
                lr_t1_s.append(1.0 if lr_deep[-1] == tp else 0.0)
                lr_t2_s.append(1.0 if tp in lr_deep else 0.0)
                bm_t1_s.append(1.0 if bm_deep[-1] == tp else 0.0)
                bm_t2_s.append(1.0 if tp in bm_deep else 0.0)

            rag_pages = torch.gather(
                pages, 1, bm_route.deep_idx.unsqueeze(-1).expand(-1, -1, wrapper.cfg.page_size)
            ).reshape(1, -1)
            rag_prefix = embed_fn(rag_pages).to(llm_dtype)

            prefixes = {
                "lslc_learned_router_flat": asm_lr_flat.prefix_embeds.to(llm_dtype),
                "lslc_shared_bm25_flat": asm_bm_flat.prefix_embeds.to(llm_dtype),
                "lslc_shared_bm25_hier": asm_bm_hier.prefix_embeds.to(llm_dtype),
                "raw_rag_shared_bm25": rag_prefix,
            }
            for s_name, pref in prefixes.items():
                em, ctok, cf1 = _eval_prefix(pref, q_ids, a_ids, turn_emb, ans_str)
                if is_exact:
                    sys_e_em[s_name].append(em)
                    sys_e_tok[s_name].append(ctok)
                    sys_e_f1[s_name].append(cf1)
                else:
                    sys_s_em[s_name].append(em)
                    sys_s_tok[s_name].append(ctok)
                    sys_s_f1[s_name].append(cf1)

            if (idx_ep + 1) % 20 == 0:
                print(f"  [32K NIAH seed={seed}] {idx_ep+1}/{episodes_per_seed} done", flush=True)

        capture_rates.append(cap_hits / max(1, cap_tot))
        lr_t1_ex.append(sum(lr_t1_e) / max(1, len(lr_t1_e)))
        lr_t2_ex.append(sum(lr_t2_e) / max(1, len(lr_t2_e)))
        lr_t1_sem.append(sum(lr_t1_s) / max(1, len(lr_t1_s)))
        lr_t2_sem.append(sum(lr_t2_s) / max(1, len(lr_t2_s)))
        lr_t1_all.append(sum(lr_t1_e + lr_t1_s) / max(1, len(lr_t1_e) + len(lr_t1_s)))
        lr_t2_all.append(sum(lr_t2_e + lr_t2_s) / max(1, len(lr_t2_e) + len(lr_t2_s)))

        bm_t1_ex.append(sum(bm_t1_e) / max(1, len(bm_t1_e)))
        bm_t2_ex.append(sum(bm_t2_e) / max(1, len(bm_t2_e)))
        bm_t1_sem.append(sum(bm_t1_s) / max(1, len(bm_t1_s)))
        bm_t2_sem.append(sum(bm_t2_s) / max(1, len(bm_t2_s)))
        bm_t1_all.append(sum(bm_t1_e + bm_t1_s) / max(1, len(bm_t1_e) + len(bm_t1_s)))
        bm_t2_all.append(sum(bm_t2_e + bm_t2_s) / max(1, len(bm_t2_e) + len(bm_t2_s)))

        for s_name in systems:
            per_sys_exact_em[s_name].append(sum(sys_e_em[s_name]) / max(1, len(sys_e_em[s_name])))
            per_sys_exact_tok[s_name].append(sum(sys_e_tok[s_name]) / max(1, len(sys_e_tok[s_name])))
            per_sys_exact_f1[s_name].append(sum(sys_e_f1[s_name]) / max(1, len(sys_e_f1[s_name])))
            per_sys_sem_em[s_name].append(sum(sys_s_em[s_name]) / max(1, len(sys_s_em[s_name])))
            per_sys_sem_tok[s_name].append(sum(sys_s_tok[s_name]) / max(1, len(sys_s_tok[s_name])))
            per_sys_sem_f1[s_name].append(sum(sys_s_f1[s_name]) / max(1, len(sys_s_f1[s_name])))

    out = {
        "num_pages": 128,
        "context_tokens": 32768,
        "seeds": seeds,
        "episodes_per_seed": episodes_per_seed,
        "total_episodes": len(seeds) * episodes_per_seed,
        "salience_capture_rate_32k": _mean_std_ci(capture_rates),
        "retriever_recall": {
            "bm25_exact_top1": _mean_std_ci(bm_t1_ex),
            "bm25_exact_top2": _mean_std_ci(bm_t2_ex),
            "bm25_semantic_top1": _mean_std_ci(bm_t1_sem),
            "bm25_semantic_top2": _mean_std_ci(bm_t2_sem),
            "bm25_overall_top1": _mean_std_ci(bm_t1_all),
            "bm25_overall_top2": _mean_std_ci(bm_t2_all),
            "learned_router_exact_top1": _mean_std_ci(lr_t1_ex),
            "learned_router_exact_top2": _mean_std_ci(lr_t2_ex),
            "learned_router_semantic_top1": _mean_std_ci(lr_t1_sem),
            "learned_router_semantic_top2": _mean_std_ci(lr_t2_sem),
            "learned_router_overall_top1": _mean_std_ci(lr_t1_all),
            "learned_router_overall_top2": _mean_std_ci(lr_t2_all),
        },
        "systems": {
            s_name: {
                "exact_em": _mean_std_ci(per_sys_exact_em[s_name]),
                "exact_content_tok_acc": _mean_std_ci(per_sys_exact_tok[s_name]),
                "exact_gen_char_f1": _mean_std_ci(per_sys_exact_f1[s_name]),
                "semantic_em": _mean_std_ci(per_sys_sem_em[s_name]),
                "semantic_content_tok_acc": _mean_std_ci(per_sys_sem_tok[s_name]),
                "semantic_gen_char_f1": _mean_std_ci(per_sys_sem_f1[s_name]),
            }
            for s_name in systems
        },
    }
    return out


@torch.no_grad()
def _llm_rewrite_state_block(
    llm,
    tokenizer,
    embed_fn,
    llm_dtype,
    device: torch.device,
    header_text: str,
    edit_text: str,
    max_new_tokens: int = 24,
) -> tuple[list[int], int]:
    """Perform real frozen Qwen3-1.7B KV-cached in-place state-block rewrite for RAG edit."""
    prompt = (
        "Current: STATE BUDGET INITIAL : 25M .\n"
        "Revision: REVISION OVERRIDE : budget updated to 60M .\n"
        "Updated: STATE BUDGET INITIAL : 60M .\n\n"
        "Current: DEPLOYMENT STATE INITIAL : cold-standby .\n"
        "Revision: REVISION OVERRIDE : deployment state updated to shadow-mirror .\n"
        "Updated: DEPLOYMENT STATE INITIAL : shadow-mirror .\n\n"
        f"Current: {header_text.strip()}\n"
        f"Revision: {edit_text.strip()}\n"
        "Updated:"
    )
    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)],
        dtype=torch.long,
        device=device,
    )
    bsz, plen = prompt_ids.shape
    attn_mask = torch.ones((bsz, plen), dtype=torch.long, device=device)
    out = llm(inputs_embeds=embed_fn(prompt_ids).to(llm_dtype), attention_mask=attn_mask, use_cache=True)
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    pkv = out.past_key_values
    gen_list = [int(next_tok.item())]
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens - 1):
        if eos_id is not None and gen_list[-1] == eos_id:
            break
        piece = tokenizer.decode([gen_list[-1]])
        if "\n" in piece or (len(gen_list) >= 4 and "." in piece):
            break
        attn_mask = torch.cat((attn_mask, torch.ones((bsz, 1), dtype=torch.long, device=device)), dim=1)
        out = llm(input_ids=next_tok, attention_mask=attn_mask, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_list.append(int(next_tok.item()))

    rewritten_str = tokenizer.decode(gen_list, skip_special_tokens=True).split("\n")[0].strip()
    rewritten_ids = tokenizer.encode(f"\n{rewritten_str}\n", add_special_tokens=False)
    return rewritten_ids, len(gen_list)


@torch.no_grad()
def profile_full_page_llm_rewrite_latency(
    llm,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    trials: int = 5,
) -> dict[str, Any]:
    """Measure realistic full 256-token page LLM rewrite latency on frozen Qwen3-1.7B."""
    prev_cache = getattr(llm.config, "use_cache", False)
    llm.config.use_cache = True
    embed_fn = llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    try:
        sample_page = tokenizer.decode(val_pool[:256], skip_special_tokens=True)
        prompt = (
            "Rewrite the full 256-token document page below with the revision applied cleanly "
            "while preserving unrelated text:\n\n"
            f"PAGE:\n{sample_page}\n\n"
            "REVISION: REVISION OVERRIDE : budget updated to 80M .\n\n"
            "REWRITTEN PAGE:\n"
        )
        p_ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)],
            dtype=torch.long,
            device=device,
        )
        empty_pref = torch.zeros((1, 0, llm.config.hidden_size), dtype=llm_dtype, device=device)
        # Warmup
        _ = greedy_generate_from_prefix(
            llm, empty_pref, p_ids, max_new_tokens=32, eos_token_id=tokenizer.eos_token_id
        )
        latencies = []
        toks_out = []
        for _ in range(trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gen = greedy_generate_from_prefix(
                llm, empty_pref, p_ids, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id
            )
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            toks_out.append(int(gen.shape[1]))
        return {
            "trials": trials,
            "prompt_tokens": int(p_ids.shape[1]),
            "generated_tokens": _mean_std_ci([float(x) for x in toks_out]),
            "latency_ms_per_edit": _mean_std_ci(latencies),
        }
    finally:
        llm.config.use_cache = prev_cache


@torch.no_grad()
def evaluate_state_override_rebuttal(
    wrapper: FrozenQwenLSLCWrapper,
    val_pool: list[int],
    tokenizer,
    device: torch.device,
    seeds: list[int],
) -> dict[str, Any]:
    """Evaluates 4 editable/immutable systems across 3 seeds on k=3 numeric (N=108) and k=5 symbolic (N=90)."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    amp_dtype, _ = select_amp_dtype(device)
    vocab_size = len(embed_fn.weight)
    wrapper.lslc.eval()

    prev_cache = getattr(wrapper.llm.config, "use_cache", False)
    wrapper.llm.config.use_cache = True

    def _eval_raw_rag(page_toks_list: list[list[int]], bm25: BM25PageRetriever, q_ids, a_ids, ans_str: str):
        q_text = tokenizer.decode(q_ids[0].tolist(), skip_special_tokens=True)
        r_out = bm25.route(q_text, k_deep=2, k_skim=2, device=device)
        sel = [page_toks_list[int(p)] for p in r_out.deep_idx[0].tolist()]
        flat = torch.tensor([tok for chunk in sel for tok in chunk], dtype=torch.long, device=device).unsqueeze(0)
        pref = embed_fn(flat).to(llm_dtype)
        turn_emb = embed_fn(torch.cat((q_ids, a_ids), dim=1)).to(llm_dtype)
        inp = torch.cat((pref, turn_emb), dim=1)
        logits = wrapper.llm(inputs_embeds=inp).logits
        start = pref.shape[1] + q_ids.shape[1] - 1
        ans_logits = logits[:, start : start + a_ids.shape[1], :]
        gen_ids = greedy_generate_from_prefix(
            wrapper.llm, pref, q_ids, max_new_tokens=24, eos_token_id=tokenizer.eos_token_id
        )
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return compute_answer_metrics(ans_logits, a_ids, gen_text, ans_str, tokenizer)

    def _run_suite(num_edits: int, num_episodes_per_seed: int) -> dict[str, Any]:
        sys_names = (
            "raw_rag_immutable",
            "raw_rag_append_errata",
            "raw_rag_in_place_llm_rewrite",
            "lslc_in_place_delta",
        )
        ov_em = {s: [] for s in sys_names}
        ov_tok = {s: [] for s in sys_names}
        ov_f1 = {s: [] for s in sys_names}
        st_em = {s: [] for s in sys_names}
        edit_ms_lslc, edit_ms_rw, edit_ms_app, rw_dec_toks = [], [], [], []
        inv_diffs = []

        for seed in seeds:
            if num_edits == 3:
                ds = LivingBookStateOverrideDataset(
                    num_episodes=num_episodes_per_seed,
                    num_pages=4,
                    page_size=wrapper.cfg.page_size,
                    num_edits=3,
                    tokenizer=tokenizer,
                    seed=seed,
                    background_token_pool=val_pool,
                )
            else:
                ds = build_k5_symbolic_override_episodes(
                    tokenizer,
                    val_pool,
                    num_episodes=num_episodes_per_seed,
                    num_pages=4,
                    num_edits=5,
                    seed=seed,
                )

            s_ov_em = {s: [] for s in sys_names}
            s_ov_tok = {s: [] for s in sys_names}
            s_ov_f1 = {s: [] for s in sys_names}
            s_st_em = {s: [] for s in sys_names}
            s_ms_lslc, s_ms_rw, s_ms_app, s_rw_toks = [], [], [], []

            for ep in ds:
                pages = ep.context_pages.unsqueeze(0).to(device)
                q_ids = ep.query_ids.unsqueeze(0).to(device)
                a_ids = ep.answer_ids.unsqueeze(0).to(device)
                ans_str = ep.metadata["answer_str"]
                is_ov = bool(ep.metadata["is_overridden_query"])

                base_pages = [ep.context_pages[p].tolist() for p in range(pages.shape[1])]
                base_texts = [tokenizer.decode(pt, skip_special_tokens=True) for pt in base_pages]

                # 1. Immutable RAG
                bm_imm = BM25PageRetriever(base_texts)
                m_imm = _eval_raw_rag(base_pages, bm_imm, q_ids, a_ids, ans_str)

                # 2. Append-Errata RAG
                app_pages = [list(pt) for pt in base_pages]
                bm_app = BM25PageRetriever(list(base_texts))
                for p_idx, e_ids in (ep.edit_pages or []):
                    t0 = time.perf_counter()
                    e_list = e_ids.tolist()
                    app_pages[p_idx].extend(e_list)
                    bm_app.update_page(p_idx, tokenizer.decode(app_pages[p_idx], skip_special_tokens=True))
                    s_ms_app.append((time.perf_counter() - t0) * 1000.0)
                m_app = _eval_raw_rag(app_pages, bm_app, q_ids, a_ids, ans_str)

                # 3. In-Place LLM-Rewrite RAG (rewrites the 32-token state block at page head)
                rw_pages = [list(pt) for pt in base_pages]
                header_texts = {
                    p: tokenizer.decode(rw_pages[p][:32], skip_special_tokens=True)
                    for p in range(len(rw_pages))
                }
                bm_rw = BM25PageRetriever(list(base_texts))
                for p_idx, e_ids in (ep.edit_pages or []):
                    e_str = tokenizer.decode(e_ids.tolist(), skip_special_tokens=True)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    new_hdr_ids, dec_cnt = _llm_rewrite_state_block(
                        wrapper.llm,
                        tokenizer,
                        embed_fn,
                        llm_dtype,
                        device,
                        header_texts[p_idx],
                        e_str,
                        max_new_tokens=24,
                    )
                    header_texts[p_idx] = tokenizer.decode(new_hdr_ids, skip_special_tokens=True)
                    # Patch the top 32 state header tokens in-place without changing chunk length
                    patched_head = (new_hdr_ids + rw_pages[p_idx][32:])[:32]
                    if len(new_hdr_ids) < 32:
                        rw_pages[p_idx] = new_hdr_ids + rw_pages[p_idx][32:]
                    else:
                        rw_pages[p_idx] = new_hdr_ids[:32] + rw_pages[p_idx][32:]
                    bm_rw.update_page(p_idx, tokenizer.decode(rw_pages[p_idx], skip_special_tokens=True))
                    torch.cuda.synchronize()
                    s_ms_rw.append((time.perf_counter() - t0) * 1000.0)
                    s_rw_toks.append(float(dec_cnt))
                m_rw = _eval_raw_rag(rw_pages, bm_rw, q_ids, a_ids, ans_str)

                # 4. LSLC In-Place Convex DeltaUpdater
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    z_book, bank, toc0 = wrapper.lslc.encode_book(
                        pages, embed_fn, use_exact_bank=True, vocab_size=vocab_size
                    )
                    q_emb = embed_fn(q_ids).mean(dim=1)
                    _, asm_before = wrapper.lslc.route_and_assemble(
                        z_book, toc0, q_emb, bank, embed_fn, k_deep=2, k_skim=2, mode="full", training_ste=False
                    )

                for p_idx, e_ids in (ep.edit_pages or []):
                    p_t = torch.tensor([p_idx], dtype=torch.long, device=device)
                    e_t = e_ids.unsqueeze(0).to(device)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        z_book, _g, bank = wrapper.lslc.apply_page_delta(
                            z_book, p_t, e_t, embed_fn, exact_bank=bank, vocab_size=vocab_size
                        )
                    torch.cuda.synchronize()
                    s_ms_lslc.append((time.perf_counter() - t0) * 1000.0)

                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    toc_after = wrapper.lslc.projector.project_toc(z_book)
                    _, asm_after = wrapper.lslc.route_and_assemble(
                        z_book, toc_after, q_emb, bank, embed_fn, k_deep=2, k_skim=2, mode="full", training_ste=False
                    )
                imm_l = asm_before.immutable_prefix_len
                inv_diffs.append(
                    float((asm_after.prefix_embeds[:, :imm_l] - asm_before.prefix_embeds[:, :imm_l]).abs().max().item())
                )
                lslc_pref = asm_after.prefix_embeds.to(llm_dtype)
                turn_emb = embed_fn(torch.cat((q_ids, a_ids), dim=1)).to(llm_dtype)
                lslc_inp = torch.cat((lslc_pref, turn_emb), dim=1)
                lslc_logits = wrapper.llm(inputs_embeds=lslc_inp).logits
                lslc_start = lslc_pref.shape[1] + q_ids.shape[1] - 1
                lslc_ans_logits = lslc_logits[:, lslc_start : lslc_start + a_ids.shape[1], :]
                lslc_gen_ids = greedy_generate_from_prefix(
                    wrapper.llm, lslc_pref, q_ids, max_new_tokens=24, eos_token_id=tokenizer.eos_token_id
                )
                lslc_gen_txt = tokenizer.decode(lslc_gen_ids[0], skip_special_tokens=True)
                m_lslc = compute_answer_metrics(lslc_ans_logits, a_ids, lslc_gen_txt, ans_str, tokenizer)

                for s_name, (em_v, tok_v, f1_v) in (
                    ("raw_rag_immutable", m_imm),
                    ("raw_rag_append_errata", m_app),
                    ("raw_rag_in_place_llm_rewrite", m_rw),
                    ("lslc_in_place_delta", m_lslc),
                ):
                    if is_ov:
                        s_ov_em[s_name].append(em_v)
                        s_ov_tok[s_name].append(tok_v)
                        s_ov_f1[s_name].append(f1_v)
                    else:
                        s_st_em[s_name].append(em_v)

            for s_name in sys_names:
                ov_em[s_name].append(sum(s_ov_em[s_name]) / max(1, len(s_ov_em[s_name])))
                ov_tok[s_name].append(sum(s_ov_tok[s_name]) / max(1, len(s_ov_tok[s_name])))
                ov_f1[s_name].append(sum(s_ov_f1[s_name]) / max(1, len(s_ov_f1[s_name])))
                if s_st_em[s_name]:
                    st_em[s_name].append(sum(s_st_em[s_name]) / len(s_st_em[s_name]))
            edit_ms_lslc.append(sum(s_ms_lslc) / max(1, len(s_ms_lslc)))
            edit_ms_rw.append(sum(s_ms_rw) / max(1, len(s_ms_rw)))
            edit_ms_app.append(sum(s_ms_app) / max(1, len(s_ms_app)))
            rw_dec_toks.append(sum(s_rw_toks) / max(1, len(s_rw_toks)))
            print(
                f"  [STATE k={num_edits} seed={seed}] LSLC={ov_em['lslc_in_place_delta'][-1]:.3f} ({edit_ms_lslc[-1]:.2f}ms/edit) | "
                f"LLM-RW-RAG={ov_em['raw_rag_in_place_llm_rewrite'][-1]:.3f} ({edit_ms_rw[-1]:.1f}ms/edit) | "
                f"Append-RAG={ov_em['raw_rag_append_errata'][-1]:.3f}",
                flush=True,
            )

        res: dict[str, Any] = {
            "num_edits": num_edits,
            "seeds": seeds,
            "total_episodes": len(seeds) * num_episodes_per_seed,
            "lslc_max_toc_prefix_diff": max(inv_diffs) if inv_diffs else 0.0,
            "edit_cost": {
                "lslc_ms_per_edit": _mean_std_ci(edit_ms_lslc),
                "rag_llm_rewrite_block_ms_per_edit": _mean_std_ci(edit_ms_rw),
                "rag_llm_rewrite_block_decode_tokens": _mean_std_ci(rw_dec_toks),
                "rag_append_errata_ms_per_edit": _mean_std_ci(edit_ms_app),
                "rag_immutable_ms_per_edit": {"mean": 0.0, "std": 0.0, "ci95": 0.0},
            },
            "systems": {},
        }
        for s_name in sys_names:
            sys_entry = {
                "override_em": _mean_std_ci(ov_em[s_name]),
                "override_content_tok_acc": _mean_std_ci(ov_tok[s_name]),
                "override_gen_char_f1": _mean_std_ci(ov_f1[s_name]),
            }
            if st_em[s_name]:
                sys_entry["static_retention_em"] = _mean_std_ci(st_em[s_name])
            res["systems"][s_name] = sys_entry
        return res

    try:
        k3_res = _run_suite(num_edits=3, num_episodes_per_seed=36)
        k5_res = _run_suite(num_edits=5, num_episodes_per_seed=30)
        full_page_rw = profile_full_page_llm_rewrite_latency(wrapper.llm, tokenizer, val_pool, device, trials=5)
    finally:
        wrapper.llm.config.use_cache = prev_cache

    return {
        "k3_numeric_overrides": k3_res,
        "k5_symbolic_overrides": k5_res,
        "full_page_256tok_llm_rewrite_latency": full_page_rw,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--out-json", type=str, default="/content/lslc_results/rebuttal_v2_summary.json")
    args = parser.parse_args()

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"=== LSLC REBUTTAL V2 UNIFIED SUITE on {props.name} ===", flush=True)

    cfg = LSLCConfig.from_model_name(args.model, lr=3e-4)
    llm, tokenizer = load_frozen_qwen3(cfg, load_in_4bit=False, device=device)
    wrapper = FrozenQwenLSLCWrapper(cfg, llm)
    wrapper.lslc.to(device)

    train_pool, val_pool, corpus_src = _load_wikitext_pools(tokenizer)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "model": args.model,
        "gpu": props.name,
        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
        "corpus": corpus_src,
    }

    def _flush() -> None:
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Stage 1: Train ONE unified checkpoint & export immediately
    print("--- Stage 1: Training Unified LSLC Qwen3-1.7B Adapter Checkpoint ---", flush=True)
    summary["training"] = train_unified_checkpoint(wrapper, train_pool, tokenizer, device)
    clean_sd = {k: v.detach().cpu().contiguous().to(torch.bfloat16) for k, v in wrapper.lslc.state_dict().items()}
    st_path = out_path.parent / "lslc_qwen3_1.7b_adapter.safetensors"
    from safetensors.torch import save_file

    save_file(
        clean_sd,
        str(st_path),
        metadata={
            "format": "pt",
            "architecture": "LayeredSpatialLatentCache",
            "base_model": args.model,
            "trainable_parameters": str(wrapper.lslc.trainable_parameter_count()),
            "suite": "rebuttal_v2_unified",
        },
    )
    print(f"Exported unified checkpoint -> {st_path}", flush=True)
    _flush()

    # Stage 2: Live Memory & Latency Scaling (32K, 128K, 256K)
    print("--- Stage 2: Live Scaling Profile (32K, 128K, 256K) ---", flush=True)
    summary["memory_and_latency_scaling"] = profile_memory_and_latency_scaling(wrapper, val_pool, device)
    _flush()

    # Stage 3: 32K (P=128) NIAH with Shared Okapi BM25 vs. Learned Router (3 seeds x 60 = 180 episodes)
    print("--- Stage 3: 32K (P=128) Multi-Seed NIAH (BM25 + Learned Router, N=180) ---", flush=True)
    summary["niah_32k_multi_seed"] = evaluate_32k_niah_rebuttal(
        wrapper, val_pool, tokenizer, device, seeds=[42, 1337, 9999], episodes_per_seed=60
    )
    _flush()

    # Stage 4: 4-Way Editable RAG vs. LSLC State-Override Benchmark (k=3, k=5, 3 seeds)
    print("--- Stage 4: Multi-Seed State Overrides (4-Way RAG vs. LSLC + Edit Cost) ---", flush=True)
    summary["state_override_rebuttal"] = evaluate_state_override_rebuttal(
        wrapper, val_pool, tokenizer, device, seeds=[42, 1337, 9999]
    )
    _flush()

    print(f"=== SAVED REBUTTAL V2 SUMMARY -> {out_path} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
