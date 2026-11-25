#!/usr/bin/env python3
"""Gerbang 4 Runner: Pure-Paraphrase Contrastive Dense Routing (Pilar 4A) + Community Benchmarks (Pilar 4B).

Pilar 4A:
  - Optional contextual query encoding (`use_contextual_query_encoding=True`) + InfoNCE contrastive loss
    (intra-book + in-batch hard negatives) training ONLY `QueryToTOCRouter` (LLM & Encoder/Projector FROZEN).
  - Evaluates `ZeroOverlapParaphraseRoutingBenchmark` (strictly 0 content-word overlap, 100% disjoint splits)
    at P=16 and P=64 comparing: Okapi BM25 CPU, LSLC Dense-Only (w=0.0), and LSLC Hybrid (w=2.0).
  - Verifies zero regression on `router_stress_p64` / `router_stress_p128` at hybrid w=2.0.

Pilar 4B:
  - Benchmark 1 (RULER 4K / 16K / 32K): deterministic `s_niah`, `mk_niah` (48 distractors), and `vt`
    (multi-hop variable tracking) over disjoint WikiText-2/essay haystacks comparing `full_context_bf16`,
    `bm25_rag_k2`, `bm25_rag_k4`, and `lslc_v3_e2e_gpu` (2-level hierarchical TOC c=16, 0 BM25 CPU).
  - Benchmark 2 (LongMemEval `xiaowu0162/longmemeval-cleaned` / `longmemeval_s_cleaned.json`):
    deterministic stratified subset (N=60 across `knowledge-update`, `single-session-user`, `multi-session`)
    comparing `bm25_rag_k4`, `llm_auto_compact`, and `lslc_v3_e2e_gpu` (Frozen Mode A + blind DeltaUpdater
    + 2-level hierarchical TOC c=16).
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any
import urllib.request

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from colab_run_m2 import _load_wikitext_pools  # noqa: E402
from colab_run_rebuttal_v2 import BM25PageRetriever  # noqa: E402
from colab_run_v3_gerbang3_coding import (  # noqa: E402
    _MockCodeTokenizer,
    _load_lslc_checkpoint,
    _pad_or_chunk_tokens,
    _strip_think_blocks,
)
from colab_run_v3_phase1 import _save_lslc_checkpoint  # noqa: E402
from lslc import (  # noqa: E402
    AdversarialLivingBookDataset,
    FrozenQwenLSLCWrapper,
    LSLCConfig,
    LSLCSystem,
    RouterOutput,
    TinyMockCausalLM,
    ZeroOverlapParaphraseRoutingBenchmark,
    compute_router_infonce_loss,
    content_words,
    extract_contextual_edit_embeds,
    extract_contextual_page_keys,
    load_frozen_qwen3,
    select_amp_dtype,
)
from lslc.datasets import _encode_text_or_mock, _sample_background_page  # noqa: E402
from lslc.encoder import SparseExactBank  # noqa: E402


_USER_OPEN_STR = "<|im_start|>user\n# Reference Memory:\n"


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_vram_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2), 2)
    return 0.0


def _reset_peak_vram(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


# ---------------------------------------------------------------------------
# Fair Community QA Evaluation Metrics & Identical ChatML Greedy Generation
# ---------------------------------------------------------------------------

_QA_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_QA_PREAMBLE_RE = re.compile(
    r"^(?:the\s+(?:final\s+)?(?:answer|value|number|special\s+magic\s+number(?:\s+for\s+[^:]+)?)\s*(?:is|=|:)\s*|"
    r"answer\s*[:=]\s*)",
    flags=re.IGNORECASE,
)


def _normalize_qa_text(text: str) -> str:
    s = _strip_think_blocks(text).strip().lower()
    s = re.sub(r"[^\w\s-]", " ", s)
    s = _QA_ARTICLES_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_first_clause(text: str) -> str:
    cleaned = _strip_think_blocks(text).strip()
    if not cleaned:
        return ""
    first_line = next((ln.strip() for ln in cleaned.splitlines() if ln.strip()), "")
    first_sent = re.split(r"(?<=[.!?])\s+", first_line)[0].strip()
    first_sent = _QA_PREAMBLE_RE.sub("", first_sent).strip()
    return _normalize_qa_text(first_sent)


def compute_community_qa_metrics(gen_text: str, gold_answer: str) -> dict[str, float]:
    """Compute fair cross-system QA metrics: Extracted EM, Token F1, Substring Match, Token Recall."""
    cleaned = _strip_think_blocks(gen_text)
    norm_pred = _normalize_qa_text(cleaned)
    norm_gold = _normalize_qa_text(gold_answer)
    first_clause = _extract_first_clause(cleaned)

    if not norm_gold:
        is_empty = float(norm_pred == "")
        return {
            "extracted_em": is_empty,
            "token_f1": is_empty,
            "substring_match": is_empty,
            "token_recall": is_empty,
        }

    extracted_em = 1.0 if first_clause == norm_gold else 0.0
    if extracted_em == 0.0 and norm_gold.isdigit():
        nums = re.findall(r"\b\d+\b", first_clause)
        if nums and nums[0] == norm_gold:
            extracted_em = 1.0

    substring_match = 1.0 if norm_gold in norm_pred else 0.0

    pred_toks = norm_pred.split()
    gold_toks = norm_gold.split()
    common = sum((Counter(pred_toks) & Counter(gold_toks)).values())
    if common == 0:
        token_f1 = 0.0
        token_recall = 0.0
    else:
        prec = common / max(1, len(pred_toks))
        rec = common / max(1, len(gold_toks))
        token_f1 = (2.0 * prec * rec) / (prec + rec)
        token_recall = rec

    return {
        "extracted_em": float(extracted_em),
        "token_f1": float(token_f1),
        "substring_match": float(substring_match),
        "token_recall": float(token_recall),
    }


def _should_stop_qa_gen(decoded: str) -> bool:
    if "<think>" in decoded and "</think>" not in decoded:
        return False
    post = (decoded.split("</think>", 1)[1] if "</think>" in decoded else decoded).lstrip()
    if not post:
        return False
    if "\n\n" in post or "<|im_end|>" in post:
        return True
    first_ln = post.splitlines()[0].strip()
    if len(post.splitlines()) >= 2 and len(first_ln) > 0:
        return True
    return False


@torch.no_grad()
def _greedy_generate_qa(
    llm: torch.nn.Module,
    prefix_embeds: torch.Tensor,
    prompt_ids: torch.Tensor,
    tokenizer,
    max_new_tokens: int = 32,
    user_open_ids: torch.Tensor | None = None,
    stop_fn=None,
) -> str:
    """Identical KV-cached greedy QA generation across all systems."""
    embed_fn = llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    prefix_cast = prefix_embeds.to(llm_dtype)
    p_embeds = embed_fn(prompt_ids).to(llm_dtype)
    if user_open_ids is not None:
        u_embeds = embed_fn(user_open_ids).to(llm_dtype)
        inp_embeds = torch.cat((u_embeds, prefix_cast, p_embeds), dim=1)
    else:
        inp_embeds = torch.cat((prefix_cast, p_embeds), dim=1)

    bsz, init_len, _ = inp_embeds.shape
    device = inp_embeds.device
    attn_mask = torch.ones((bsz, init_len), device=device, dtype=torch.long)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    prev_use_cache = getattr(llm.config, "use_cache", False) if hasattr(llm, "config") else False
    if hasattr(llm, "config"):
        llm.config.use_cache = True

    active_stop_fn = _should_stop_qa_gen if stop_fn is None else stop_fn
    gen_tokens: list[int] = []
    try:
        out = llm(inputs_embeds=inp_embeds, attention_mask=attn_mask, use_cache=True)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        pkv = getattr(out, "past_key_values", None)
        for step_idx in range(max_new_tokens):
            tid = int(next_tok[0, 0].item())
            if eos_id is not None and tid == int(eos_id):
                break
            gen_tokens.append(tid)
            if len(gen_tokens) % 4 == 0 or step_idx >= max_new_tokens - 1:
                decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                if active_stop_fn(decoded):
                    break
            attn_mask = torch.cat((attn_mask, torch.ones((bsz, 1), device=device, dtype=torch.long)), dim=1)
            if pkv is not None:
                out = llm(input_ids=next_tok, attention_mask=attn_mask, past_key_values=pkv, use_cache=True)
                pkv = getattr(out, "past_key_values", None)
            else:
                inp_embeds = torch.cat((inp_embeds, embed_fn(next_tok).to(llm_dtype)), dim=1)
                out = llm(inputs_embeds=inp_embeds, attention_mask=attn_mask)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    finally:
        if hasattr(llm, "config"):
            llm.config.use_cache = prev_use_cache

    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


def _encode_unpadded_rag_ids(
    rag_txt: str,
    tokenizer,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode BM25 retrieved context strictly unpadded (zero pad_token_id / eos_token_id)."""
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    drop = {int(x) for x in (pad_id, eos_id) if x is not None}
    raw_ids = _encode_text_or_mock(rag_txt, tokenizer, vocab_size)
    clean_ids = [int(t) for t in raw_ids if int(t) not in drop]
    if not clean_ids:
        clean_ids = [2]
    if pad_id is not None:
        assert int(pad_id) not in clean_ids, "Unpadded BM25 RAG must never contain pad_token_id"
    return torch.tensor(clean_ids, dtype=torch.long, device=device).unsqueeze(0)


def _format_qa_prompt(question: str) -> str:
    return (
        f"\n\n# Question:\n{question.strip()}\n"
        f"Answer concisely with only the exact answer."
        f"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _get_special_token_ids(tokenizer) -> tuple[int, ...]:
    if tokenizer is None:
        return ()
    sids: set[int] = set()
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
        val = getattr(tokenizer, attr, None)
        if val is not None:
            sids.add(int(val))
    all_sp = getattr(tokenizer, "all_special_ids", None)
    if all_sp:
        for val in all_sp:
            if val is not None:
                sids.add(int(val))
    return tuple(sorted(sids))


@torch.no_grad()
def _encode_book_chunked(
    lslc,
    page_tensor: torch.Tensor,
    embed_fn,
    vocab_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    autocast_enabled: bool,
    chunk_size: int = 32,
    special_token_ids: tuple[int, ...] | None = None,
    salience_span_len: int | None = None,
) -> tuple[torch.Tensor, SparseExactBank, torch.Tensor]:
    z_chunks, toc_chunks, tok_chunks, h_chunks, w_chunks = [], [], [], [], []
    num_p = page_tensor.shape[1]
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
        for s in range(0, num_p, chunk_size):
            e = min(num_p, s + chunk_size)
            zc, bc, tc = lslc.encode_book(
                page_tensor[:, s:e],
                embed_fn,
                use_exact_bank=True,
                vocab_size=vocab_size,
                special_token_ids=special_token_ids,
                salience_span_len=salience_span_len,
            )
            z_chunks.append(zc)
            toc_chunks.append(tc)
            assert bc is not None
            tok_chunks.append(bc.token_ids)
            h_chunks.append(bc.h_coords)
            w_chunks.append(bc.w_coords)

    z_book = torch.cat(z_chunks, dim=1)
    toc_embeds = torch.cat(toc_chunks, dim=1)
    k_ex = tok_chunks[0].shape[-1] if tok_chunks else lslc.cfg.k_exact
    p_ids = (
        torch.arange(num_p, device=device, dtype=torch.int16)
        .view(1, num_p, 1)
        .expand(1, num_p, k_ex)
        .contiguous()
    )
    bank = SparseExactBank(
        token_ids=torch.cat(tok_chunks, dim=1),
        page_ids=p_ids,
        h_coords=torch.cat(h_chunks, dim=1),
        w_coords=torch.cat(w_chunks, dim=1),
    )
    return z_book, bank, toc_embeds


@torch.no_grad()
def _compute_query_ctx_embeds(
    wrapper: FrozenQwenLSLCWrapper,
    rq_ids: torch.Tensor,
    amp_dtype: torch.dtype,
    autocast_enabled: bool,
) -> torch.Tensor | None:
    if not getattr(wrapper.cfg, "use_contextual_query_encoding", False):
        return None
    layer_idx = (
        wrapper.cfg.contextual_query_layer
        if wrapper.cfg.contextual_query_layer is not None
        else wrapper.cfg.contextual_edit_layer
    )
    with torch.autocast(device_type=rq_ids.device.type, dtype=amp_dtype, enabled=autocast_enabled):
        return extract_contextual_edit_embeds(wrapper.llm, rq_ids, layer_idx=layer_idx)


@torch.no_grad()
def _compute_page_ctx_keys(
    wrapper: FrozenQwenLSLCWrapper,
    page_tensor: torch.Tensor,
    tokenizer,
    amp_dtype: torch.dtype,
    autocast_enabled: bool,
    chunk_size: int = 16,
) -> torch.Tensor | None:
    if not getattr(wrapper.cfg, "use_contextual_query_encoding", False):
        return None
    layer_idx = (
        wrapper.cfg.contextual_query_layer
        if wrapper.cfg.contextual_query_layer is not None
        else wrapper.cfg.contextual_edit_layer
    )
    pad_id = getattr(tokenizer, "pad_token_id", None)
    num_spans = int(getattr(wrapper.cfg, "contextual_page_spans", 1))
    with torch.autocast(device_type=page_tensor.device.type, dtype=amp_dtype, enabled=autocast_enabled):
        return extract_contextual_page_keys(
            wrapper.llm,
            page_tensor,
            layer_idx=layer_idx,
            pad_token_id=pad_id,
            chunk_size=chunk_size,
            num_page_spans=num_spans,
        )


# ---------------------------------------------------------------------------
# Pilar 4A: Pure-Paraphrase Contrastive Dense Routing (InfoNCE, Frozen LLM/Enc)
# ---------------------------------------------------------------------------


def _wilson_ci95(p: float, n: int) -> list[float]:
    """Return 95% Wilson score confidence interval [lo, hi] for proportion p in [0, 1]."""
    if n <= 0:
        return [0.0, 0.0]
    p_c = max(0.0, min(1.0, float(p)))
    z = 1.959963984540054
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_c + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((p_c * (1.0 - p_c) + z2 / (4.0 * n)) / n)) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def _normal_ci95(vals: list[float], min_val: float | None = 0.0, max_val: float | None = 1.0) -> list[float]:
    """Return 95% normal-approximation confidence interval [lo, hi] for sample values."""
    n = len(vals)
    if n <= 0:
        return [0.0, 0.0]
    mean_v = sum(vals) / n
    if n == 1:
        v = round(mean_v, 4)
        return [v, v]
    var = sum((x - mean_v) ** 2 for x in vals) / (n - 1)
    half = 1.959963984540054 * math.sqrt(max(0.0, var) / n)
    lo = mean_v - half
    hi = mean_v + half
    if min_val is not None:
        lo = max(min_val, lo)
    if max_val is not None:
        hi = min(max_val, hi)
    return [round(lo, 4), round(hi, 4)]


def _bootstrap_ci95(vals: list[float], n_boot: int = 400, seed: int = 2026) -> list[float]:
    """Deterministic percentile 95% bootstrap CI [lo, hi] for sample values."""
    n = len(vals)
    if n <= 0:
        return [0.0, 0.0]
    if n == 1:
        v = round(float(vals[0]), 4)
        return [v, v]
    rng = random.Random(seed + n * 17)
    means: list[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _j in range(n):
            s += float(vals[rng.randrange(n)])
        means.append(s / n)
    means.sort()
    lo_idx = max(0, int(0.025 * (n_boot - 1)))
    hi_idx = min(n_boot - 1, int(0.975 * (n_boot - 1)))
    return [round(means[lo_idx], 4), round(means[hi_idx], 4)]


def _build_wikitext_selfsup_episode(
    rng: random.Random,
    tokenizer,
    train_pool: list[int],
    num_pages: int,
    page_size: int,
    vocab_size: int,
):
    """Build a zero-overlap cross-sentence contrastive episode from WikiText-2 train pool."""
    stride = max(48, page_size // 2)
    win_size = min(len(train_pool) - 1, page_size * 2)
    chunks: list[tuple[list[int], str]] = []
    max_start = max(1, len(train_pool) - win_size - 1)
    for _ in range(num_pages * 2):
        st = rng.randrange(0, max_start, max(1, stride // 4))
        ids_full = list(train_pool[st : st + win_size])
        ids = ids_full[:page_size]
        txt = tokenizer.decode(ids_full, skip_special_tokens=True)
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt) if len(s.strip()) >= 32]
        if len(sents) >= 2:
            chunks.append((ids, txt))
        if len(chunks) >= num_pages:
            break
    if len(chunks) < num_pages:
        return None

    tgt_idx = rng.randrange(num_pages)
    tgt_ids, tgt_txt = chunks[tgt_idx]
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", tgt_txt) if len(s.strip()) >= 28]
    if len(sents) < 2:
        return None
    mid = len(sents) // 2
    ctx_part = " ".join(sents[:mid])
    nxt_part = " ".join(sents[mid:])
    q_words = content_words(nxt_part)
    # Remove any sentence from ctx_part that shares content words with nxt_part to enforce zero overlap
    filtered_ctx_sents = [s for s in sents[:mid] if not (content_words(s) & q_words)]
    if not filtered_ctx_sents:
        return None
    ctx_clean = " ".join(filtered_ctx_sents)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    tgt_clean_ids = _pad_or_chunk_tokens(_encode_text_or_mock(ctx_clean, tokenizer, vocab_size), page_size, pad_id)[0]

    pages = [c[0] for c in chunks]
    pages[tgt_idx] = tgt_clean_ids
    rq_ids = torch.tensor(_encode_text_or_mock(nxt_part[:160], tokenizer, vocab_size), dtype=torch.long)
    return {
        "pages": torch.tensor(pages, dtype=torch.long),
        "rq_ids": rq_ids,
        "target_page": tgt_idx,
        "topic_idx": 10000 + rng.randrange(100000),
    }


def train_pilar4a_contrastive_router(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    train_pool: list[int],
    device: torch.device,
    *,
    steps: int = 240,
    batch_size: int = 8,
    lr: float = 3e-4,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, object]:
    """Train ONLY QueryToTOCRouter with intra-book + in-batch InfoNCE (LLM, Encoder, Salience, Projector frozen)."""
    if steps <= 0:
        return {"steps": 0, "skipped": True, "max_frozen_param_diff": 0.0}

    rng = random.Random(seed)
    embed_fn = wrapper.llm.get_input_embeddings()
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32

    # Freeze EVERYTHING except wrapper.lslc.router
    saved_reqs: list[tuple[torch.nn.Parameter, bool]] = []
    router_param_ids = {id(p) for p in wrapper.lslc.router.parameters()}
    frozen_snap: dict[str, torch.Tensor] = {}
    for name, p in wrapper.named_parameters():
        saved_reqs.append((p, p.requires_grad))
        is_r = id(p) in router_param_ids
        p.requires_grad_(is_r)
        if not is_r:
            frozen_snap[name] = p.detach().cpu().clone()

    proj_norm_params: list[torch.nn.Parameter] = []
    ctx_gate_params: list[torch.nn.Parameter] = []
    for rname, rp in wrapper.lslc.router.named_parameters():
        if not rp.requires_grad:
            continue
        if any(rname.startswith(pref) for pref in ("q_proj", "k_proj", "q_norm", "k_norm")):
            proj_norm_params.append(rp)
        else:
            ctx_gate_params.append(rp)
    router_params = proj_norm_params + ctx_gate_params
    proj_lr = lr if is_mock else lr * 0.25
    wd = 0.01 if is_mock else 0.05
    opt_groups = []
    if proj_norm_params:
        opt_groups.append({"params": proj_norm_params, "lr": proj_lr})
    if ctx_gate_params:
        opt_groups.append({"params": ctx_gate_params, "lr": lr})
    optimizer = torch.optim.AdamW(opt_groups, lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps), eta_min=lr * 0.1)

    num_p_train = 8 if is_mock else 16
    n_train_eps = 48 if is_mock else 192
    train_ds = ZeroOverlapParaphraseRoutingBenchmark(
        num_episodes=n_train_eps,
        num_pages=num_p_train,
        page_size=wrapper.cfg.page_size,
        split="train",
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        seed=seed + 11,
        background_token_pool=train_pool,
    )
    entity_ds = AdversarialLivingBookDataset(
        num_episodes=16 if is_mock else 48,
        num_pages=num_p_train,
        page_size=wrapper.cfg.page_size,
        num_edits=2,
        mode="symbolic",
        split="train",
        style="mixed_natural",
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        seed=seed + 29,
        background_token_pool=train_pool,
    )

    # Pre-encode frozen book latents & contextual query embeddings once under torch.no_grad()
    # because Encoder, SalienceGate, Projector, and LLM are all strictly FROZEN!
    wrapper.eval()
    cached_items: list[dict[str, Any]] = []
    with torch.no_grad():
        for ep in train_ds:
            p_t = ep.context_pages.unsqueeze(0).to(device)
            z_b, bank_b, toc_b = _encode_book_chunked(
                wrapper.lslc, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled, chunk_size=16
            )
            page_ctx = _compute_page_ctx_keys(wrapper, p_t, tokenizer, amp_dtype, autocast_enabled, chunk_size=16)
            bank_emb = (
                embed_fn(bank_b.routing_token_ids().clamp(min=0).to(device))
                if bank_b is not None
                else None
            )
            rq_ids = ep.metadata["router_query_ids"].unsqueeze(0).to(device)
            q_raw = embed_fn(rq_ids)
            q_ctx = _compute_query_ctx_embeds(wrapper, rq_ids, amp_dtype, autocast_enabled)
            cached_items.append(
                {
                    "toc": toc_b.squeeze(0),
                    "bank_emb": None if bank_emb is None else bank_emb.squeeze(0),
                    "page_ctx": None if page_ctx is None else page_ctx.squeeze(0),
                    "q_mean": q_raw.mean(dim=1).squeeze(0),
                    "q_tok": q_raw.squeeze(0),
                    "q_ctx": None if q_ctx is None else q_ctx.squeeze(0),
                    "target_page": int(ep.target_page_idx),
                    "topic_idx": int(ep.metadata["topic_idx"]),
                    "source": "paraphrase",
                }
            )
        for idx_e, ep in enumerate(entity_ds):
            p_t = ep.metadata["verified_final_pages"].unsqueeze(0).to(device)
            z_b, bank_b, toc_b = _encode_book_chunked(
                wrapper.lslc, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled, chunk_size=16
            )
            page_ctx = _compute_page_ctx_keys(wrapper, p_t, tokenizer, amp_dtype, autocast_enabled, chunk_size=16)
            bank_emb = (
                embed_fn(bank_b.routing_token_ids().clamp(min=0).to(device))
                if bank_b is not None
                else None
            )
            rq_ids = ep.metadata["router_query_ids"].unsqueeze(0).to(device)
            q_raw = embed_fn(rq_ids)
            q_ctx = _compute_query_ctx_embeds(wrapper, rq_ids, amp_dtype, autocast_enabled)
            cached_items.append(
                {
                    "toc": toc_b.squeeze(0),
                    "bank_emb": None if bank_emb is None else bank_emb.squeeze(0),
                    "page_ctx": None if page_ctx is None else page_ctx.squeeze(0),
                    "q_mean": q_raw.mean(dim=1).squeeze(0),
                    "q_tok": q_raw.squeeze(0),
                    "q_ctx": None if q_ctx is None else q_ctx.squeeze(0),
                    "target_page": int(ep.target_page_idx),
                    "topic_idx": 5000 + idx_e,
                    "source": "entity_replay",
                }
            )
        if not is_mock and train_pool and len(train_pool) >= 8192:
            wiki_cnt = 0
            attempts = 0
            while wiki_cnt < 256 and attempts < 900:
                attempts += 1
                w_item = _build_wikitext_selfsup_episode(
                    rng, tokenizer, train_pool, num_p_train, wrapper.cfg.page_size, vocab_size
                )
                if w_item is None:
                    continue
                p_t = w_item["pages"].unsqueeze(0).to(device)
                _, bank_b, toc_b = _encode_book_chunked(
                    wrapper.lslc, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled, chunk_size=16
                )
                page_ctx = _compute_page_ctx_keys(wrapper, p_t, tokenizer, amp_dtype, autocast_enabled, chunk_size=16)
                bank_emb = (
                    embed_fn(bank_b.routing_token_ids().clamp(min=0).to(device))
                    if bank_b is not None
                    else None
                )
                rq_ids = w_item["rq_ids"].unsqueeze(0).to(device)
                q_raw = embed_fn(rq_ids)
                q_ctx = _compute_query_ctx_embeds(wrapper, rq_ids, amp_dtype, autocast_enabled)
                cached_items.append(
                    {
                        "toc": toc_b.squeeze(0),
                        "bank_emb": None if bank_emb is None else bank_emb.squeeze(0),
                        "page_ctx": None if page_ctx is None else page_ctx.squeeze(0),
                        "q_mean": q_raw.mean(dim=1).squeeze(0),
                        "q_tok": q_raw.squeeze(0),
                        "q_ctx": None if q_ctx is None else q_ctx.squeeze(0),
                        "target_page": int(w_item["target_page"]),
                        "topic_idx": int(w_item["topic_idx"]),
                        "source": "wikitext_selfsup",
                    }
                )
                wiki_cnt += 1

    para_indices = [i for i, x in enumerate(cached_items) if x["source"] == "paraphrase"]
    wiki_indices = [i for i, x in enumerate(cached_items) if x["source"] == "wikitext_selfsup"]
    entity_indices = [i for i, x in enumerate(cached_items) if x["source"] == "entity_replay"]

    wrapper.lslc.router.train()
    t0 = time.perf_counter()
    loss_hist: list[float] = []
    intra_top1_hist: list[float] = []
    inbatch_top1_hist: list[float] = []

    try:
        for _step in range(1, steps + 1):
            batch_items: list[dict[str, Any]] = []
            seen_topics: set[int] = set()
            eff_bs = min(batch_size, max(1, len(cached_items)))
            for _b in range(eff_bs):
                r_draw = rng.random()
                if wiki_indices:
                    if r_draw < 0.50:
                        pool_idx = para_indices
                    elif r_draw < 0.85:
                        pool_idx = wiki_indices
                    else:
                        pool_idx = entity_indices or para_indices
                elif entity_indices and r_draw >= 0.80:
                    pool_idx = entity_indices
                else:
                    pool_idx = para_indices
                picked = cached_items[rng.choice(pool_idx)]
                for _retry in range(6):
                    if picked["topic_idx"] not in seen_topics:
                        break
                    picked = cached_items[rng.choice(pool_idx)]
                seen_topics.add(int(picked["topic_idx"]))
                batch_items.append(picked)

            toc_b = torch.stack([x["toc"] for x in batch_items], dim=0)
            bank_b = (
                torch.stack([x["bank_emb"] for x in batch_items], dim=0)
                if batch_items[0]["bank_emb"] is not None
                else None
            )
            k_ctx_b = (
                torch.stack([x["page_ctx"] for x in batch_items], dim=0)
                if batch_items[0]["page_ctx"] is not None
                else None
            )
            q_vecs = torch.cat(
                [
                    wrapper.lslc.router.encode_query(
                        x["q_mean"].unsqueeze(0),
                        query_token_embeds=x["q_tok"].unsqueeze(0),
                        contextual_query_embeds=None if x["q_ctx"] is None else x["q_ctx"].unsqueeze(0),
                    )
                    for x in batch_items
                ],
                dim=0,
            )
            k_vecs = wrapper.lslc.router.encode_keys(
                toc_b,
                exact_bank_embeds=bank_b,
                center_across_pages=True,
                contextual_page_keys=k_ctx_b,
            )
            tgt_t = torch.tensor([x["target_page"] for x in batch_items], dtype=torch.long, device=device)
            topic_t = torch.tensor([x["topic_idx"] for x in batch_items], dtype=torch.long, device=device)

            loss, metrics = compute_router_infonce_loss(
                wrapper.lslc.router,
                q_vecs,
                k_vecs,
                tgt_t,
                topic_ids=topic_t,
                in_batch_weight=1.0,
                symmetric_in_batch=True,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router_params, wrapper.cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            loss_hist.append(float(loss.detach().item()))
            intra_top1_hist.append(float(metrics["intra_top1"]))
            inbatch_top1_hist.append(float(metrics["inbatch_top1"]))
    finally:
        for p, prev_req in saved_reqs:
            p.requires_grad_(prev_req)
        wrapper.eval()

    # Verify freeze contract: LLM, Encoder, SalienceGate, Projector, DeltaUpdater, hw_coord_embed unchanged
    max_frozen_diff = 0.0
    for name, p in wrapper.named_parameters():
        if name in frozen_snap:
            diff = float((p.detach().cpu() - frozen_snap[name]).abs().max().item())
            if diff > max_frozen_diff:
                max_frozen_diff = diff
    assert max_frozen_diff == 0.0, f"Frozen parameters mutated during Pilar 4A training: {max_frozen_diff}"

    window_n = max(1, min(20, len(loss_hist) // 4))
    return {
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "num_cached_items": len(cached_items),
        "num_paraphrase_items": len(para_indices),
        "num_wikitext_items": len(wiki_indices),
        "num_entity_items": len(entity_indices),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "step1_loss": round(loss_hist[0], 4) if loss_hist else 0.0,
        "last_step_loss": round(loss_hist[-1], 4) if loss_hist else 0.0,
        "initial_loss_mean": round(sum(loss_hist[:window_n]) / window_n, 4),
        "final_loss_mean": round(sum(loss_hist[-window_n:]) / window_n, 4),
        "final_intra_top1": round(sum(intra_top1_hist[-window_n:]) / window_n, 4),
        "final_inbatch_top1": round(sum(inbatch_top1_hist[-window_n:]) / window_n, 4),
        "trainable_router_params": sum(p.numel() for p in router_params),
        "max_frozen_param_diff": max_frozen_diff,
    }


@torch.no_grad()
def evaluate_pilar4a_zero_overlap(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    eval_pages: tuple[int, ...] = (16, 64),
    eval_n: int = 24,
    seed: int = 2026,
    is_mock: bool = False,
    eval_cache: dict[int, dict[str, Any]] | None = None,
) -> dict[str, object]:
    """Evaluate 0-overlap pure-paraphrase routing at P=16 and P=64: BM25 CPU vs Dense w=0 vs Hybrid w=2."""
    embed_fn = wrapper.llm.get_input_embeddings()
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    wrapper.eval()

    results: dict[str, object] = {}
    for p_cnt in eval_pages:
        if eval_cache is not None and p_cnt in eval_cache:
            cached_p = eval_cache[p_cnt]
            num_dist = int(cached_p["num_distractor_topics"])
            cached_eps = cached_p["items"]
            n_ep = len(cached_eps)
        else:
            ds_eval = ZeroOverlapParaphraseRoutingBenchmark(
                num_episodes=eval_n,
                num_pages=p_cnt,
                page_size=wrapper.cfg.page_size,
                split="eval",
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                seed=seed + p_cnt * 13,
                background_token_pool=val_pool,
            )
            # Verify hard 0-overlap invariant on every eval episode
            for ep in ds_eval:
                q_w = content_words(ep.metadata["query_text"])
                p_w = content_words(ep.metadata["target_page_text"])
                assert q_w & p_w == set(), f"Zero-overlap violated in eval P={p_cnt}: {q_w & p_w}"
            num_dist = ds_eval.num_distractor_topics
            n_ep = len(ds_eval)
            cached_eps = None

        stats: dict[str, dict[str, float]] = {
            "bm25_cpu": {"top1": 0.0, "top2": 0.0, "ms": 0.0},
            "lslc_dense_w0": {"top1": 0.0, "top2": 0.0, "ms": 0.0},
            "lslc_hybrid_w2": {"top1": 0.0, "top2": 0.0, "ms": 0.0},
        }
        _reset_peak_vram(device)
        new_cache_items: list[dict[str, Any]] | None = [] if (eval_cache is not None and cached_eps is None) else None

        iter_src = cached_eps if cached_eps is not None else ds_eval
        for item in iter_src:
            if cached_eps is not None:
                tgt = int(item["tgt"])
                stats["bm25_cpu"]["ms"] += float(item["bm25_ms"])
                stats["bm25_cpu"]["top1"] += float(item["bm25_top1"])
                stats["bm25_cpu"]["top2"] += float(item["bm25_top2"])
                z_book = item["z_book"].to(device)
                toc = item["toc"].to(device)
                bank = SparseExactBank(
                    token_ids=item["bank_tok"].to(device),
                    page_ids=item["bank_pid"].to(device),
                    h_coords=item["bank_h"].to(device),
                    w_coords=item["bank_w"].to(device),
                )
                page_ctx = None if item["page_ctx"] is None else item["page_ctx"].to(device)
                rq_ids = item["rq_ids"].to(device)
                q_ctx_pre = None if item["q_ctx"] is None else item["q_ctx"].to(device)
            else:
                ep = item
                tgt = int(ep.target_page_idx)
                q_txt = str(ep.metadata["query_text"])
                page_texts = list(ep.metadata["page_texts"])

                # 1) Okapi BM25 CPU baseline
                t_b0 = time.perf_counter()
                bm25 = BM25PageRetriever(page_texts)
                rr_bm25 = bm25.route(q_txt, k_deep=2, k_skim=0, device=device)
                b_ms = (time.perf_counter() - t_b0) * 1000.0
                b_t1 = 1.0 if int(torch.argmax(rr_bm25.logits[0], dim=-1).item()) == tgt else 0.0
                b_t2 = 1.0 if tgt in rr_bm25.deep_idx[0].tolist() else 0.0
                stats["bm25_cpu"]["ms"] += b_ms
                stats["bm25_cpu"]["top1"] += b_t1
                stats["bm25_cpu"]["top2"] += b_t2

                # Encode book once for GPU routers
                p_t = ep.context_pages.unsqueeze(0).to(device)
                z_book, bank, toc = _encode_book_chunked(
                    wrapper.lslc, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled, chunk_size=32
                )
                page_ctx = _compute_page_ctx_keys(wrapper, p_t, tokenizer, amp_dtype, autocast_enabled, chunk_size=16)
                rq_ids = ep.metadata["router_query_ids"].unsqueeze(0).to(device)
                q_ctx_pre = _compute_query_ctx_embeds(wrapper, rq_ids, amp_dtype, autocast_enabled)
                if new_cache_items is not None:
                    new_cache_items.append(
                        {
                            "tgt": tgt,
                            "bm25_ms": b_ms,
                            "bm25_top1": b_t1,
                            "bm25_top2": b_t2,
                            "z_book": z_book.detach().cpu(),
                            "toc": toc.detach().cpu(),
                            "bank_tok": bank.token_ids.detach().cpu(),
                            "bank_pid": bank.page_ids.detach().cpu(),
                            "bank_h": bank.h_coords.detach().cpu(),
                            "bank_w": bank.w_coords.detach().cpu(),
                            "page_ctx": None if page_ctx is None else page_ctx.detach().cpu(),
                            "rq_ids": rq_ids.detach().cpu(),
                            "q_ctx": None if q_ctx_pre is None else q_ctx_pre.detach().cpu(),
                        }
                    )

            # 2) LSLC Dense-Only (w=0.0)
            _sync_device(device)
            t_d0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                q_ctx = q_ctx_pre if q_ctx_pre is not None else _compute_query_ctx_embeds(wrapper, rq_ids, amp_dtype, autocast_enabled)
                q_emb = embed_fn(rq_ids).mean(dim=1)
                r0, _ = wrapper.lslc.route_and_assemble(
                    z_book,
                    toc,
                    q_emb,
                    bank,
                    embed_fn,
                    k_deep=2,
                    k_skim=min(2, max(0, p_cnt - 2)),
                    mode="full",
                    training_ste=False,
                    query_input_ids=rq_ids,
                    salience_route_weight=0.0,
                    contextual_query_embeds=q_ctx,
                    contextual_page_keys=page_ctx,
                )
            _sync_device(device)
            stats["lslc_dense_w0"]["ms"] += (time.perf_counter() - t_d0) * 1000.0
            if int(torch.argmax(r0.logits[0], dim=-1).item()) == tgt:
                stats["lslc_dense_w0"]["top1"] += 1.0
            if tgt in r0.deep_idx[0].tolist():
                stats["lslc_dense_w0"]["top2"] += 1.0

            # 3) LSLC Hybrid (w=2.0)
            _sync_device(device)
            t_h0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                q_ctx = q_ctx_pre if q_ctx_pre is not None else _compute_query_ctx_embeds(wrapper, rq_ids, amp_dtype, autocast_enabled)
                q_emb = embed_fn(rq_ids).mean(dim=1)
                r2, _ = wrapper.lslc.route_and_assemble(
                    z_book,
                    toc,
                    q_emb,
                    bank,
                    embed_fn,
                    k_deep=2,
                    k_skim=min(2, max(0, p_cnt - 2)),
                    mode="full",
                    training_ste=False,
                    query_input_ids=rq_ids,
                    salience_route_weight=2.0,
                    contextual_query_embeds=q_ctx,
                    contextual_page_keys=page_ctx,
                )
            _sync_device(device)
            stats["lslc_hybrid_w2"]["ms"] += (time.perf_counter() - t_h0) * 1000.0
            if int(torch.argmax(r2.logits[0], dim=-1).item()) == tgt:
                stats["lslc_hybrid_w2"]["top1"] += 1.0
            if tgt in r2.deep_idx[0].tolist():
                stats["lslc_hybrid_w2"]["top2"] += 1.0

        if new_cache_items is not None and eval_cache is not None:
            eval_cache[p_cnt] = {"num_distractor_topics": num_dist, "items": new_cache_items}

        d = float(max(1, n_ep))
        peak_mb = _peak_vram_mb(device)
        results[f"P={p_cnt}"] = {
            "num_episodes": n_ep,
            "num_pages": p_cnt,
            "num_distractor_topics": num_dist,
            "peak_vram_mb": peak_mb,
            **{
                arm: {
                    "top1_recall": round(vals["top1"] / d, 4),
                    "top1_ci95": _wilson_ci95(vals["top1"] / d, n_ep),
                    "top2_recall": round(vals["top2"] / d, 4),
                    "top2_ci95": _wilson_ci95(vals["top2"] / d, n_ep),
                    "per_query_ms": round(vals["ms"] / d, 3),
                }
                for arm, vals in stats.items()
            },
        }
    return results


@torch.no_grad()
def evaluate_router_stress_regression(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    is_mock: bool,
    seed: int = 9091,
    lslc_override: LSLCSystem | None = None,
    special_token_ids: tuple[int, ...] | None = None,
    salience_span_len: int | None = None,
    use_contextual_keys: bool | None = None,
) -> dict[str, object]:
    """Verify router_stress_p64 and router_stress_p128 stay 100% in hybrid (w=2.0)."""
    lslc_mod = wrapper.lslc if lslc_override is None else lslc_override
    embed_fn = wrapper.llm.get_input_embeddings()
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    vocab_size = len(embed_fn.weight)
    page_size = lslc_mod.cfg.page_size
    use_ctx = bool(getattr(lslc_mod.cfg, "use_contextual_query_encoding", False)) if use_contextual_keys is None else bool(use_contextual_keys)
    wrapper.eval()
    lslc_mod.eval()

    specs = (
        ("router_stress_p64", 16 if is_mock else 64, 8 if is_mock else 12),
        ("router_stress_p128", 32 if is_mock else 128, 8 if is_mock else 20),
    )
    out: dict[str, object] = {}
    for label, num_p, n_entities in specs:
        rng = random.Random(seed + num_p)
        chosen_pages = rng.sample(range(num_p), n_entities)
        filler_vocab = val_pool if val_pool else list(range(2, max(16, vocab_size - 2)))
        pages = [_sample_background_page(rng, page_size, val_pool, filler_vocab) for _ in range(num_p)]

        queries: list[tuple[int, str]] = []
        for idx_e, p_idx in enumerate(chosen_pages):
            proj = f"StressNode-{idx_e:02d}"
            val = f"stress-tier-{idx_e:02d}"
            fmt = "record" if (idx_e % 2 == 0) else ("eval_prose" if idx_e % 4 == 1 else "eval_code")
            stmt = AdversarialLivingBookDataset._format_fact_stmt("symbolic", fmt=fmt, proj=proj, val=val)
            toks = _encode_text_or_mock(stmt, tokenizer, vocab_size)
            offset = rng.randint(0, max(0, page_size - len(toks)))
            n_w = min(len(toks), page_size - offset)
            pages[p_idx][offset : offset + n_w] = toks[:n_w]
            queries.append((p_idx, f"What is the approved deployment state for {proj}?"))

        page_tensor = torch.tensor(pages, dtype=torch.long, device=device).unsqueeze(0)
        z_book, bank, toc = _encode_book_chunked(
            lslc_mod,
            page_tensor,
            embed_fn,
            vocab_size,
            device,
            amp_dtype,
            autocast_enabled,
            chunk_size=32,
            special_token_ids=special_token_ids,
            salience_span_len=salience_span_len,
        )
        page_ctx = _compute_page_ctx_keys(wrapper, page_tensor, tokenizer, amp_dtype, autocast_enabled, chunk_size=16) if use_ctx else None

        w0_t1, w0_t2, w2_t1, w2_t2 = 0, 0, 0, 0
        for p_idx, q_txt in queries:
            rq_ids = torch.tensor(
                _encode_text_or_mock(q_txt, tokenizer, vocab_size), dtype=torch.long, device=device
            ).unsqueeze(0)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                q_ctx = _compute_query_ctx_embeds(wrapper, rq_ids, amp_dtype, autocast_enabled) if use_ctx else None
                q_emb = embed_fn(rq_ids).mean(dim=1)
                r0, _ = lslc_mod.route_and_assemble(
                    z_book,
                    toc,
                    q_emb,
                    bank,
                    embed_fn,
                    mode="full",
                    training_ste=False,
                    query_input_ids=rq_ids,
                    salience_route_weight=0.0,
                    contextual_query_embeds=q_ctx,
                    contextual_page_keys=page_ctx,
                )
                r2, _ = lslc_mod.route_and_assemble(
                    z_book,
                    toc,
                    q_emb,
                    bank,
                    embed_fn,
                    mode="full",
                    training_ste=False,
                    query_input_ids=rq_ids,
                    salience_route_weight=2.0,
                    contextual_query_embeds=q_ctx,
                    contextual_page_keys=page_ctx,
                )
            if int(torch.argmax(r0.logits[0], dim=-1).item()) == p_idx:
                w0_t1 += 1
            if p_idx in r0.deep_idx[0].tolist():
                w0_t2 += 1
            if int(torch.argmax(r2.logits[0], dim=-1).item()) == p_idx:
                w2_t1 += 1
            if p_idx in r2.deep_idx[0].tolist():
                w2_t2 += 1

        d = max(1, len(queries))
        out[label] = {
            "num_pages": num_p,
            "num_entity_pages": n_entities,
            "dense_w0_top1_recall": round(w0_t1 / d, 4),
            "dense_w0_top2_recall": round(w0_t2 / d, 4),
            "hybrid_w2_top1_recall": round(w2_t1 / d, 4),
            "hybrid_w2_top2_recall": round(w2_t2 / d, 4),
        }
    return out


# ---------------------------------------------------------------------------
# Pilar 4B Benchmark 1: RULER (s_niah, mk_niah, vt) across 4K / 16K / 32K
# ---------------------------------------------------------------------------

_RULER_WORDS = (
    "amber", "basalt", "cedar", "delta", "ember", "falcon", "granite", "harbor",
    "indigo", "jasper", "kestrel", "lumen", "marble", "nebula", "obsidian", "quartz",
    "redwood", "saffron", "timber", "umbra", "velvet", "willow", "zenith", "cobalt",
    "topaz", "birch", "copper", "flint", "garnet", "heron", "ivory", "juniper",
    "kelp", "laurel", "maple", "nickel", "ochre", "pebble", "ripple", "spruce",
    "thistle", "verdigris", "walnut", "yarrow", "zircon", "agate", "banyan", "cinder",
    "dusk", "fern", "grove", "hazel", "isle", "jade", "knoll", "lichen",
)
_RULER_DEPTHS = (0.0, 0.25, 0.50, 0.75, 1.0)


def _decode_safe_haystack_pages(
    rng: random.Random,
    num_pages: int,
    page_size: int,
    tokenizer,
    val_pool: list[int],
    vocab_size: int,
) -> list[list[int]]:
    filler_vocab = val_pool if val_pool else _encode_text_or_mock(
        "archival ledger essay chronicle routine observation paragraph", tokenizer, vocab_size
    )
    return [_sample_background_page(rng, page_size, val_pool, filler_vocab) for _ in range(num_pages)]


def build_ruler_episodes(
    task_name: str,
    context_length: int,
    num_episodes: int,
    tokenizer,
    val_pool: list[int],
    *,
    page_size: int = 256,
    seed: int = 2026,
    mk_distractors: int = 48,
    vt_hops: int = 3,
    is_mock: bool = False,
) -> list[dict[str, Any]]:
    """Build deterministic RULER `s_niah`, `mk_niah`, or `vt` episodes over real WikiText-2/essay haystacks."""
    if task_name not in ("s_niah", "mk_niah", "vt"):
        raise ValueError(f"Unknown RULER task: {task_name!r}")
    vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else 512
    num_pages = max(4, context_length // page_size)
    episodes: list[dict[str, Any]] = []

    for idx in range(num_episodes):
        rng = random.Random(seed + context_length * 17 + idx * 1009 + sum(ord(c) for c in task_name))
        pages = _decode_safe_haystack_pages(rng, num_pages, page_size, tokenizer, val_pool, vocab_size)
        gold_pages: set[int] = set()

        if task_name == "s_niah":
            depth = _RULER_DEPTHS[idx % len(_RULER_DEPTHS)]
            tgt_p = min(num_pages - 1, int(round(depth * (num_pages - 1))))
            key_str = f"{_RULER_WORDS[idx % len(_RULER_WORDS)]}-{rng.randint(10, 99)}"
            val_str = str(rng.randint(1000000, 9999999))
            needle = f"\nOne of the special magic numbers for {key_str} is: {val_str}.\n"
            n_toks = _encode_text_or_mock(needle, tokenizer, vocab_size)
            offset = rng.randint(16, max(16, page_size - len(n_toks) - 8))
            n_w = min(len(n_toks), page_size - offset)
            pages[tgt_p][offset : offset + n_w] = n_toks[:n_w]
            gold_pages.add(tgt_p)
            question = f"What is the special magic number for {key_str} mentioned in the provided text?"
            router_query = f"special magic number for {key_str}"
            answer = val_str
            meta: dict[str, Any] = {"depth": depth, "key": key_str}

        elif task_name == "mk_niah":
            n_dist = min(mk_distractors, max(4, num_pages * 2)) if is_mock else int(mk_distractors)
            total_keys = n_dist + 1
            depth = _RULER_DEPTHS[idx % len(_RULER_DEPTHS)]
            tgt_p = min(num_pages - 1, int(round(depth * (num_pages - 1))))
            tgt_slot = rng.randrange(total_keys)

            # Distribute keys across pages using non-overlapping intra-page byte slots
            per_page_slots: dict[int, int] = {}
            used_vals: set[int] = set()
            keys_list: list[tuple[str, str]] = []
            for k_i in range(total_keys):
                w = _RULER_WORDS[k_i % len(_RULER_WORDS)]
                k_name = f"{w}-node-{k_i:02d}"
                v_num = rng.randint(1000000, 9999999)
                while v_num in used_vals:
                    v_num = rng.randint(1000000, 9999999)
                used_vals.add(v_num)
                keys_list.append((k_name, str(v_num)))

            for k_i, (k_name, v_str) in enumerate(keys_list):
                p_idx = tgt_p if k_i == tgt_slot else (k_i * 7 + idx) % num_pages
                slot_idx = per_page_slots.get(p_idx, 0)
                per_page_slots[p_idx] = slot_idx + 1
                needle = f"\nOne of the special magic numbers for {k_name} is: {v_str}.\n"
                n_toks = _encode_text_or_mock(needle, tokenizer, vocab_size)
                step_span = max(len(n_toks) + 4, page_size // max(1, (total_keys // num_pages) + 2))
                offset = min(max(0, page_size - len(n_toks) - 2), 8 + slot_idx * step_span)
                n_w = min(len(n_toks), page_size - offset)
                pages[p_idx][offset : offset + n_w] = n_toks[:n_w]

            tgt_key, tgt_val = keys_list[tgt_slot]
            gold_pages.add(tgt_p)
            question = f"What is the special magic number for {tgt_key} mentioned in the provided text?"
            router_query = f"special magic number for {tgt_key}"
            answer = tgt_val
            meta = {"depth": depth, "num_distractor_keys": n_dist, "target_key": tgt_key}

        else:  # "vt" (Variable Tracking multi-hop chains)
            n_vars = vt_hops + 1
            chain_pages = sorted(rng.sample(range(num_pages), min(n_vars, num_pages)))
            while len(chain_pages) < n_vars:
                chain_pages.append(chain_pages[-1])
            chain_pages.sort()

            # Build primary target chain + 2 distractor chains with distinct var names and values
            used_vals = set()
            chains_info: list[tuple[list[str], str, list[int]]] = []
            alpha_pool = [f"VAR_{chr(65 + i)}{idx % 10}" for i in range(26)]
            rng.shuffle(alpha_pool)
            for c_i in range(3):
                var_names = alpha_pool[c_i * n_vars : (c_i + 1) * n_vars]
                v_num = rng.randint(10000, 99999)
                while v_num in used_vals:
                    v_num = rng.randint(10000, 99999)
                used_vals.add(v_num)
                if c_i == 0:
                    c_pages = chain_pages
                else:
                    c_pages = sorted(rng.choices(range(num_pages), k=n_vars))
                chains_info.append((var_names, str(v_num), c_pages))

            per_page_cursor: dict[int, int] = {}
            for c_i, (var_names, v_str, c_pages) in enumerate(chains_info):
                for hop_i, v_name in enumerate(var_names):
                    stmt = (
                        f"\nASSIGN {v_name} = {v_str} .\n"
                        if hop_i == 0
                        else f"\nASSIGN {v_name} = {var_names[hop_i - 1]} .\n"
                    )
                    s_toks = _encode_text_or_mock(stmt, tokenizer, vocab_size)
                    p_idx = c_pages[hop_i]
                    cur_off = per_page_cursor.get(p_idx, 12)
                    offset = min(max(0, page_size - len(s_toks) - 2), cur_off)
                    per_page_cursor[p_idx] = offset + len(s_toks) + 8
                    n_w = min(len(s_toks), page_size - offset)
                    pages[p_idx][offset : offset + n_w] = s_toks[:n_w]
                    if c_i == 0:
                        gold_pages.add(p_idx)

            tgt_vars, tgt_val, _ = chains_info[0]
            final_var = tgt_vars[-1]
            question = f"Trace the variable assignments in the text above. What numeric value is assigned to {final_var}?"
            router_query = f"ASSIGN {final_var}"
            answer = tgt_val
            meta = {"vt_hops": vt_hops, "chain_vars": tgt_vars}

        page_texts = [tokenizer.decode(p, skip_special_tokens=True) for p in pages]
        episodes.append(
            {
                "task": task_name,
                "context_length": context_length,
                "num_pages": num_pages,
                "pages": torch.tensor(pages, dtype=torch.long),
                "page_texts": page_texts,
                "gold_pages": sorted(gold_pages),
                "question": question,
                "router_query": router_query,
                "answer": answer,
                "metadata": meta,
            }
        )
    return episodes


def _aggregate_eval_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)

    def _vals(key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None]

    def _mean(key: str) -> float:
        vals = _vals(key)
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    em_mean = _mean("extracted_em")
    sub_mean = _mean("substring_match")
    f1_vals = _vals("token_f1")
    rec_vals = _vals("token_recall")

    ci95: dict[str, list[float]] = {
        "extracted_em": _wilson_ci95(em_mean, n),
        "token_f1": _normal_ci95(f1_vals),
        "token_f1_bootstrap": _bootstrap_ci95(f1_vals),
        "substring_match": _wilson_ci95(sub_mean, n),
        "token_recall": _normal_ci95(rec_vals),
    }

    f1_mean = _mean("token_f1")
    active_tok_mean = _mean("active_prefix_tokens")
    upd_ms_mean = _mean("memory_update_ms")
    out: dict[str, Any] = {
        "num_episodes": n,
        "extracted_em": em_mean,
        "em": em_mean,
        "token_f1": f1_mean,
        "f1": f1_mean,
        "substring_match": sub_mean,
        "token_recall": _mean("token_recall"),
        "ci95": ci95,
        "active_prefix_tokens": active_tok_mean,
        "active_tokens_mean": active_tok_mean,
        "memory_update_ms": upd_ms_mean,
        "update_latency_ms": upd_ms_mean,
        "per_query_ms": _mean("per_query_ms"),
        "peak_vram_mb": round(max((float(r["peak_vram_mb"]) for r in rows), default=0.0), 2),
        "max_toc_diff": round(max((float(r["max_toc_diff"]) for r in rows), default=0.0), 6),
    }
    hit_vals = _vals("router_hit_rate")
    if hit_vals:
        out["router_hit_rate"] = _mean("router_hit_rate")
        ci95["router_hit_rate"] = _normal_ci95(hit_vals)
    return out


@torch.no_grad()
def evaluate_ruler_benchmark(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    lengths: tuple[int, ...] = (4096, 16384, 32768),
    episodes_per_task: int = 12,
    mk_distractors: int = 48,
    vt_hops: int = 3,
    lslc_k_deep: int = 4,
    lslc_k_skim: int = 2,
    chapter_size: int = 16,
    max_new_tokens: int = 24,
    seed: int = 2026,
    is_mock: bool = False,
    bge_toc: Any | None = None,
    salience_route_weight: float = 1.5,
    include_exact_only_ablation: bool = False,
    exact_only_max_length: int = 32768,
    systems_override: tuple[str, ...] | None = None,
    use_raw_question_router: bool = False,
) -> dict[str, Any]:
    """Run RULER benchmark (`s_niah`, `mk_niah`, `vt`) with Pareto iso-token BM25 k=1/2/4, full_context_bf16, LSLC E2E GPU, and optional exact_only."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    eff_span_len = int(getattr(wrapper.cfg, "salience_span_len", 1))
    wrapper.eval()

    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)
    tasks = ("s_niah", "mk_niah", "vt")
    out: dict[str, Any] = {
        "lengths": list(lengths),
        "tasks": list(tasks),
        "salience_span_len": eff_span_len,
        "k_exact": int(wrapper.cfg.k_exact),
        "salience_route_weight": float(salience_route_weight),
        "router_query_mode": "raw_question" if use_raw_question_router else "short_query",
        "bge_model": None if bge_toc is None else getattr(bge_toc, "model_name", None),
        "by_length": {},
    }

    for ctx_len in lengths:
        len_key = f"{ctx_len // 1024}K" if ctx_len >= 1024 else str(ctx_len)
        run_exact_only = bool(include_exact_only_ablation and (ctx_len <= exact_only_max_length or is_mock))
        if systems_override is not None:
            systems = tuple(systems_override)
        elif run_exact_only:
            systems = ("full_context_bf16", "bm25_rag_k1", "bm25_rag_k2", "bm25_rag_k4", "lslc_v3_e2e_gpu", "lslc_exact_only")
        else:
            systems = ("full_context_bf16", "bm25_rag_k1", "bm25_rag_k2", "bm25_rag_k4", "lslc_v3_e2e_gpu")
        len_bucket: dict[str, Any] = {"by_task": {}, "summary": {}}
        all_rows_by_sys: dict[str, list[dict[str, Any]]] = {s: [] for s in systems}
        oom_systems: set[str] = set()

        for task_name in tasks:
            eps = build_ruler_episodes(
                task_name=task_name,
                context_length=ctx_len,
                num_episodes=episodes_per_task,
                tokenizer=tokenizer,
                val_pool=val_pool,
                page_size=wrapper.cfg.page_size,
                seed=seed,
                mk_distractors=mk_distractors,
                vt_hops=vt_hops,
                is_mock=is_mock,
            )
            task_bucket: dict[str, Any] = {}
            lslc_ep_cache: list[tuple[torch.Tensor, SparseExactBank, torch.Tensor, torch.Tensor | None, torch.Tensor | None, float]] = []

            for sys_name in systems:
                if sys_name in oom_systems:
                    task_bucket[sys_name] = {"status": "oom_skipped"}
                    continue
                rows: list[dict[str, Any]] = []
                raw_samples: list[dict[str, Any]] = []
                hit_oom = False
                for ep_idx, ep in enumerate(eps):
                    _reset_peak_vram(device)
                    p_ids_cpu: torch.Tensor = ep["pages"]  # [P, Tp]
                    page_texts: list[str] = ep["page_texts"]
                    gold_set = set(ep["gold_pages"])
                    route_q_str = str(ep["question"] if use_raw_question_router else ep["router_query"])
                    p_prompt = torch.tensor(
                        tokenizer.encode(_format_qa_prompt(ep["question"]), add_special_tokens=False),
                        dtype=torch.long,
                        device=device,
                    ).unsqueeze(0)

                    try:
                        mem_ms = 0.0
                        max_toc_diff = 0.0
                        router_hit: float | None = None

                        if sys_name == "full_context_bf16":
                            _sync_device(device)
                            t_q0 = time.perf_counter()
                            flat_ids = p_ids_cpu.reshape(1, -1).to(device)
                            pref_embeds = embed_fn(flat_ids).to(llm_dtype)
                            gen_txt = _greedy_generate_qa(
                                wrapper.llm, pref_embeds, p_prompt, tokenizer, max_new_tokens=max_new_tokens, user_open_ids=user_open_ids
                            )
                            _sync_device(device)
                            q_ms = (time.perf_counter() - t_q0) * 1000.0
                            pref_toks = int(pref_embeds.shape[1])
                            del flat_ids, pref_embeds
                            if device.type == "cuda" and ctx_len >= 16384:
                                torch.cuda.empty_cache()

                        elif sys_name.startswith("bm25_rag_k"):
                            k_val = int(sys_name.rsplit("_k", 1)[-1])
                            t_m0 = time.perf_counter()
                            bm25 = BM25PageRetriever(page_texts)
                            mem_ms = (time.perf_counter() - t_m0) * 1000.0

                            _sync_device(device)
                            t_q0 = time.perf_counter()
                            rr = bm25.route(route_q_str, k_deep=min(k_val, len(page_texts)), k_skim=0, device=device)
                            chosen = rr.deep_idx[0].tolist()
                            router_hit = 1.0 if gold_set.issubset(set(chosen)) else (
                                len(gold_set & set(chosen)) / max(1, len(gold_set))
                            )
                            rag_txt = "\n\n".join(page_texts[i].strip() for i in chosen) + "\n"
                            r_ids = _encode_unpadded_rag_ids(rag_txt, tokenizer, vocab_size, device)
                            pref_embeds = embed_fn(r_ids).to(llm_dtype)
                            gen_txt = _greedy_generate_qa(
                                wrapper.llm, pref_embeds, p_prompt, tokenizer, max_new_tokens=max_new_tokens, user_open_ids=user_open_ids
                            )
                            _sync_device(device)
                            q_ms = (time.perf_counter() - t_q0) * 1000.0
                            pref_toks = int(pref_embeds.shape[1])

                        else:  # lslc_v3_e2e_gpu (chatml_full) or lslc_exact_only (chatml_exact_only)
                            is_exact_only = sys_name == "lslc_exact_only"
                            if is_exact_only and ep_idx < len(lslc_ep_cache):
                                z_book, bank, toc, page_ctx, bge_p4, mem_ms = lslc_ep_cache[ep_idx]
                            else:
                                p_t = p_ids_cpu.unsqueeze(0).to(device)
                                _sync_device(device)
                                t_m0 = time.perf_counter()
                                z_book, bank, toc = _encode_book_chunked(
                                    wrapper.lslc,
                                    p_t,
                                    embed_fn,
                                    vocab_size,
                                    device,
                                    amp_dtype,
                                    autocast_enabled,
                                    chunk_size=32,
                                    special_token_ids=special_ids,
                                    salience_span_len=eff_span_len,
                                )
                                if bge_toc is not None:
                                    bge_p4 = bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=128)
                                    page_ctx = None
                                else:
                                    bge_p4 = None
                                    page_ctx = _compute_page_ctx_keys(
                                        wrapper, p_t, tokenizer, amp_dtype, autocast_enabled, chunk_size=16
                                    )
                                _sync_device(device)
                                mem_ms = (time.perf_counter() - t_m0) * 1000.0
                                if run_exact_only and not is_exact_only:
                                    lslc_ep_cache.append((z_book, bank, toc, page_ctx, bge_p4, mem_ms))

                            rq_ids = torch.tensor(
                                _encode_text_or_mock(route_q_str, tokenizer, vocab_size),
                                dtype=torch.long,
                                device=device,
                            ).unsqueeze(0)
                            _sync_device(device)
                            t_q0 = time.perf_counter()
                            ext_logits = (
                                bge_toc.score_book(bge_toc.encode_texts([route_q_str], is_query=True), bge_p4)
                                if (bge_toc is not None and bge_p4 is not None)
                                else None
                            )
                            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                                q_ctx = (
                                    None
                                    if ext_logits is not None
                                    else _compute_query_ctx_embeds(wrapper, rq_ids, amp_dtype, autocast_enabled)
                                )
                                q_emb = embed_fn(rq_ids).mean(dim=1)
                                kd_use = min(lslc_k_deep, z_book.shape[1])
                                ks_use = min(lslc_k_skim, max(0, z_book.shape[1] - kd_use))
                                r_out, asm = wrapper.lslc.route_and_assemble(
                                    z_book=z_book,
                                    toc_embeds=toc,
                                    query_embed=q_emb,
                                    exact_bank=bank,
                                    embed_tokens_fn=embed_fn,
                                    k_deep=kd_use,
                                    k_skim=ks_use,
                                    mode="exact_only" if is_exact_only else "full",
                                    training_ste=False,
                                    hierarchical_chapter_size=chapter_size,
                                    query_input_ids=rq_ids,
                                    salience_route_weight=salience_route_weight,
                                    exact_bank_attenuation="none",
                                    contextual_query_embeds=q_ctx,
                                    contextual_page_keys=page_ctx,
                                    external_dense_logits=ext_logits,
                                )
                            chosen = r_out.deep_idx[0].tolist()
                            router_hit = 1.0 if gold_set.issubset(set(chosen)) else (
                                len(gold_set & set(chosen)) / max(1, len(gold_set))
                            )
                            gen_txt = _greedy_generate_qa(
                                wrapper.llm,
                                asm.prefix_embeds.to(llm_dtype),
                                p_prompt,
                                tokenizer,
                                max_new_tokens=max_new_tokens,
                                user_open_ids=user_open_ids,
                            )
                            _sync_device(device)
                            q_ms = (time.perf_counter() - t_q0) * 1000.0
                            pref_toks = int(asm.prefix_embeds.shape[1])

                        qm = compute_community_qa_metrics(gen_txt, ep["answer"])
                        if len(raw_samples) < 3:
                            raw_samples.append(
                                {
                                    "question": ep["question"],
                                    "gold": ep["answer"],
                                    "pred": gen_txt,
                                    "router_hit": router_hit,
                                    "extracted_em": qm["extracted_em"],
                                    "token_f1": qm["token_f1"],
                                    "substring_match": qm["substring_match"],
                                }
                            )
                        rows.append(
                            {
                                **qm,
                                "router_hit_rate": router_hit,
                                "active_prefix_tokens": pref_toks,
                                "memory_update_ms": round(mem_ms, 3),
                                "per_query_ms": round(q_ms, 3),
                                "peak_vram_mb": _peak_vram_mb(device),
                                "max_toc_diff": max_toc_diff,
                            }
                        )
                    except torch.cuda.OutOfMemoryError:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        hit_oom = True
                        oom_systems.add(sys_name)
                        break

                if hit_oom:
                    task_bucket[sys_name] = {"status": "oom"}
                else:
                    agg = _aggregate_eval_rows(rows)
                    agg["samples"] = raw_samples
                    task_bucket[sys_name] = agg
                    all_rows_by_sys[sys_name].extend(rows)

            lslc_ep_cache.clear()
            len_bucket["by_task"][task_name] = task_bucket

        for sys_name in systems:
            if sys_name in oom_systems:
                len_bucket["summary"][sys_name] = {"status": "oom"}
            else:
                len_bucket["summary"][sys_name] = _aggregate_eval_rows(all_rows_by_sys[sys_name])

        out["by_length"][len_key] = len_bucket
        print(
            f"[RULER {len_key}] "
            + " | ".join(
                f"{s}="
                + (
                    "OOM"
                    if len_bucket["summary"][s].get("status") == "oom"
                    else f"EM={len_bucket['summary'][s]['extracted_em']:.3f},F1={len_bucket['summary'][s]['token_f1']:.3f},tok={len_bucket['summary'][s]['active_prefix_tokens']:.0f}"
                )
                for s in systems
            ),
            flush=True,
        )

    return out


# ---------------------------------------------------------------------------
# Pilar 4B Benchmark 2: LongMemEval (`xiaowu0162/longmemeval-cleaned`)
# ---------------------------------------------------------------------------


def _build_verified_local_longmemeval_fallback(seed: int = 2026) -> list[dict[str, Any]]:
    """Deterministic verified local fallback matching exact `longmemeval_s_cleaned.json` schema (for offline/mock tests)."""
    rng = random.Random(seed + 4242)
    filler_topics = [
        ("Can you suggest three healthy weekday breakfasts?", "Try oatmeal with berries, Greek yogurt with walnuts, or veggie omelets."),
        ("How do I format a Python virtual environment?", "Run `python3 -m venv .venv` and activate it via `source .venv/bin/activate`."),
        ("What is the capital of Portugal?", "Lisbon is the capital and largest city of Portugal."),
        ("Give me a quick tip for kneading sourdough.", "Use stretch-and-fold intervals every thirty minutes for gentle gluten development."),
    ]
    items: list[dict[str, Any]] = []

    specs = (
        ("knowledge-update", "What coffee grinder brand do I currently use?", "Comandante C40",
         "I bought a Baratza Encore grinder for morning pour-overs.",
         "Actually I replaced my old grinder last week and switched to a Comandante C40."),
        ("knowledge-update", "Which city did I relocate to for my new job?", "Seattle",
         "I have been living in Austin for the past two years.",
         "Update on my move: I relocated to Seattle this Monday for my new engineering role."),
        ("knowledge-update", "What is my current weekly running mileage target?", "35 miles",
         "My weekly running mileage target is 20 miles right now.",
         "I bumped my training schedule up; my current weekly running mileage target is now 35 miles."),
        ("knowledge-update", "Which cloud provider hosts our staging cluster now?", "Cloudflare Workers",
         "Our staging cluster currently runs on Heroku.",
         "We migrated everything off Heroku yesterday; our staging cluster now runs on Cloudflare Workers."),
        ("single-session-user", "What breed is my rescue dog?", "Border Collie",
         "Last month I adopted a very energetic Border Collie named Pepper.",
         None),
        ("single-session-user", "Which musical instrument did I study in college?", "cello",
         "During my four years in college I played the cello in the chamber orchestra.",
         None),
        ("single-session-user", "What allergy must I avoid at restaurants?", "shellfish",
         "Please remember I have a severe shellfish allergy whenever recommending seafood spots.",
         None),
        ("single-session-user", "What model camera lens do I carry for street photography?", "35mm prime",
         "I always pack my lightweight 35mm prime lens when shooting street photography.",
         None),
        ("multi-session", "How many total hours did I spend on my pottery and woodworking workshops combined?", "14 hours",
         "I completed a 6 hours weekend pottery workshop at the community studio.",
         "I just wrapped up an 8 hours woodworking workshop across town."),
        ("multi-session", "Which two languages am I studying this semester?", "Japanese and Spanish",
         "This semester I enrolled in an evening Japanese conversation class.",
         "Alongside my other class, I also signed up for intermediate Spanish this semester."),
        ("multi-session", "How much did I spend altogether on my monitor and mechanical keyboard?", "520 dollars",
         "I ordered a 400 dollars ultrawide monitor for my home desk.",
         "To finish my desk setup, I bought a 120 dollars mechanical keyboard today."),
        ("multi-session", "Which two national parks did I hike in during May and August?", "Zion and Acadia",
         "In May I spent four days hiking slot canyons in Zion National Park.",
         "My August vacation took me to coastal granite trails in Acadia National Park."),
    )

    for idx, (qtype, question, answer, sess_a_txt, sess_b_txt) in enumerate(specs):
        sessions: list[list[dict[str, str]]] = []
        sess_ids: list[str] = []
        dates: list[str] = []
        ans_sids: list[str] = []
        n_sess = 8
        pos_a = 1 if sess_b_txt is not None else 3
        pos_b = 5 if sess_b_txt is not None else -1

        for s_i in range(n_sess):
            sid = f"local_fb_{idx:02d}_s{s_i:02d}"
            date_str = f"2024/0{(s_i % 8) + 1}/1{s_i % 9} (Mon) 10:0{s_i % 9}"
            if s_i == pos_a:
                turns = [
                    {"role": "user", "content": sess_a_txt, "has_answer": True},
                    {"role": "assistant", "content": "Got it, I have noted that detail for future reference."},
                ]
                ans_sids.append(sid)
            elif s_i == pos_b and sess_b_txt is not None:
                turns = [
                    {"role": "user", "content": sess_b_txt, "has_answer": True},
                    {"role": "assistant", "content": "Understood, I will keep that update in mind."},
                ]
                ans_sids.append(sid)
            else:
                fu, fa = filler_topics[(idx + s_i + rng.randrange(4)) % len(filler_topics)]
                turns = [{"role": "user", "content": fu}, {"role": "assistant", "content": fa}]
            sessions.append(turns)
            sess_ids.append(sid)
            dates.append(date_str)

        items.append(
            {
                "question_id": f"fb_{qtype}_{idx:02d}",
                "question_type": qtype,
                "question": question,
                "question_date": "2024/09/01 (Sun) 12:00",
                "answer": answer,
                "answer_session_ids": ans_sids,
                "haystack_dates": dates,
                "haystack_session_ids": sess_ids,
                "haystack_sessions": sessions,
            }
        )
    return items


def load_longmemeval_stratified(
    repo_id: str = "xiaowu0162/longmemeval-cleaned",
    filename: str = "longmemeval_s_cleaned.json",
    local_path: str = "",
    *,
    n_total: int = 60,
    categories: tuple[str, ...] = ("knowledge-update", "single-session-user", "multi-session"),
    seed: int = 2026,
    allow_fallback: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load official LongMemEval JSON (HF public or local JSON) and sample a deterministic stratified subset."""
    raw_items: list[dict[str, Any]] | None = None
    meta: dict[str, Any] = {
        "repo_id": repo_id,
        "filename": filename,
        "categories": list(categories),
        "seed": seed,
    }

    if local_path and Path(local_path).is_file():
        raw_items = json.loads(Path(local_path).read_text(encoding="utf-8"))
        meta["dataset_source"] = "local_json"
        meta["resolved_path"] = str(Path(local_path).resolve())
    elif allow_fallback:
        raw_items = _build_verified_local_longmemeval_fallback(seed=seed)
        meta["dataset_source"] = "verified_local_fallback"
        meta["fallback_reason"] = "explicit_allow_fallback_mock"
    else:
        load_err: str | None = None
        try:
            from huggingface_hub import hf_hub_download

            dl_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
            raw_items = json.loads(Path(dl_path).read_text(encoding="utf-8"))
            meta["dataset_source"] = f"hf_hub:{repo_id}/{filename}"
            meta["resolved_path"] = str(dl_path)
        except Exception as exc_hf:
            load_err = f"hf_hub_download failed: {exc_hf}"
            cache_dir = Path("/tmp/longmemeval_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached_file = cache_dir / filename
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
            try:
                if not cached_file.is_file() or cached_file.stat().st_size == 0:
                    urllib.request.urlretrieve(url, str(cached_file))
                raw_items = json.loads(cached_file.read_text(encoding="utf-8"))
                meta["dataset_source"] = f"hf_https:{url}"
                meta["resolved_path"] = str(cached_file)
            except Exception as exc_url:
                load_err = f"{load_err}; urlretrieve failed: {exc_url}"
                print(
                    f"[WARNING] LongMemEval HF download failed; using verified local fallback: {load_err}",
                    flush=True,
                )
                raw_items = _build_verified_local_longmemeval_fallback(seed=seed)
                meta["dataset_source"] = "verified_local_fallback"
                meta["fallback_reason"] = load_err

    assert raw_items is not None
    meta["raw_total_items"] = len(raw_items)

    rng = random.Random(seed)
    per_cat = max(1, n_total // max(1, len(categories)))
    selected: list[dict[str, Any]] = []
    counts_by_cat: dict[str, int] = {}

    for cat in categories:
        pool = [
            x
            for x in raw_items
            if str(x.get("question_type", "")) == cat and not str(x.get("question_id", "")).endswith("_abs")
        ]
        pool.sort(key=lambda x: str(x.get("question_id", "")))
        rng.shuffle(pool)
        take = pool[: min(per_cat, len(pool))]
        selected.extend(take)
        counts_by_cat[cat] = len(take)

    meta["selected_total"] = len(selected)
    meta["counts_by_category"] = counts_by_cat
    return selected, meta


def _format_longmemeval_session(sess_idx: int, date_str: str, turns: list[dict[str, Any]]) -> str:
    lines = [f"[Session {sess_idx + 1} | Date: {date_str}]"]
    for t in turns:
        role = str(t.get("role", "user")).capitalize()
        content = str(t.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines) + "\n"


def _pack_longmemeval_sessions(
    item: dict[str, Any],
    tokenizer,
    page_size: int,
    max_pages: int = 0,
) -> dict[str, Any]:
    """Pack chronological sessions into fixed page_size pages (preserving session boundaries for modular encoding)."""
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0) if tokenizer is not None else 0
    sessions: list[list[dict[str, Any]]] = list(item["haystack_sessions"])
    dates: list[str] = list(item.get("haystack_dates", [""] * len(sessions)))
    sess_ids: list[str] = list(item.get("haystack_session_ids", [f"s{i}" for i in range(len(sessions))]))
    ans_sids = set(str(x) for x in item.get("answer_session_ids", []))
    vocab_cap = len(tokenizer) if hasattr(tokenizer, "__len__") else 512

    pre_encoded: list[dict[str, Any]] = []
    for s_i, (turns, d_str, sid) in enumerate(zip(sessions, dates, sess_ids)):
        s_txt = _format_longmemeval_session(s_i, str(d_str), turns)
        s_toks = _encode_text_or_mock(s_txt, tokenizer, vocab_cap)
        raw_chunks = [s_toks[st : st + page_size] for st in range(0, max(1, len(s_toks)), page_size)]
        chunks = _pad_or_chunk_tokens(s_toks, page_size, pad_id)
        pre_encoded.append(
            {
                "orig_idx": s_i,
                "turns": turns,
                "sid": str(sid),
                "text": s_txt,
                "raw_chunks": raw_chunks,
                "chunks": chunks,
                "is_gold": str(sid) in ans_sids,
            }
        )

    total_pages_all = sum(len(x["chunks"]) for x in pre_encoded)
    if max_pages > 0 and total_pages_all > max_pages:
        gold_indices = [i for i, x in enumerate(pre_encoded) if x["is_gold"]]
        dist_indices = [i for i, x in enumerate(pre_encoded) if not x["is_gold"]]
        gold_pages = sum(len(pre_encoded[i]["chunks"]) for i in gold_indices)
        rem_budget = max(0, max_pages - gold_pages)
        chosen_dist: list[int] = []
        used_dist: set[int] = set()
        if dist_indices and rem_budget > 0:
            avg_p = sum(len(pre_encoded[i]["chunks"]) for i in dist_indices) / max(1, len(dist_indices))
            target_k = min(len(dist_indices), max(1, int(round(rem_budget / max(1.0, avg_p)))))
            if target_k == 1:
                positions = [len(dist_indices) // 2]
            else:
                positions = sorted({int(round(j * (len(dist_indices) - 1) / (target_k - 1))) for j in range(target_k)})
            for pos in positions:
                idx = dist_indices[pos]
                ch_n = len(pre_encoded[idx]["chunks"])
                if ch_n <= rem_budget:
                    chosen_dist.append(idx)
                    used_dist.add(idx)
                    rem_budget -= ch_n
            # Top up any remaining page budget evenly across unpicked distractors
            leftover = [idx for idx in dist_indices if idx not in used_dist]
            if leftover and rem_budget > 0:
                step = max(1, len(leftover) // max(1, rem_budget))
                for pos in range(0, len(leftover), step):
                    idx = leftover[pos]
                    ch_n = len(pre_encoded[idx]["chunks"])
                    if ch_n <= rem_budget:
                        chosen_dist.append(idx)
                        used_dist.add(idx)
                        rem_budget -= ch_n
                if rem_budget > 0:
                    for idx in leftover:
                        if idx in used_dist:
                            continue
                        ch_n = len(pre_encoded[idx]["chunks"])
                        if ch_n <= rem_budget:
                            chosen_dist.append(idx)
                            used_dist.add(idx)
                            rem_budget -= ch_n
        keep_indices = sorted(gold_indices + chosen_dist)
    else:
        keep_indices = list(range(len(pre_encoded)))

    session_records: list[dict[str, Any]] = []
    all_page_ids: list[list[int]] = []
    all_page_texts: list[str] = []
    page_session_ids: list[str] = []
    content_tokens = 0

    for new_i, k_idx in enumerate(keep_indices):
        rec_p = pre_encoded[k_idx]
        s_txt = rec_p["text"]
        raw_chunks = rec_p["raw_chunks"]
        chunks = rec_p["chunks"]
        sid = rec_p["sid"]
        turns = rec_p["turns"]
        content_tokens += sum(len(ch) for ch in raw_chunks)
        start_p = len(all_page_ids)
        for raw_ch, ch in zip(raw_chunks, chunks):
            all_page_ids.append(ch)
            ch_txt = s_txt if len(chunks) == 1 else tokenizer.decode(raw_ch, skip_special_tokens=True)
            all_page_texts.append(ch_txt)
            page_session_ids.append(sid)
        user_txt = " ".join(str(t.get("content", "")).strip() for t in turns if str(t.get("role", "")).lower() == "user")
        session_records.append(
            {
                "session_idx": new_i,
                "orig_session_idx": rec_p["orig_idx"],
                "session_id": sid,
                "text": s_txt,
                "user_text": user_txt,
                "page_range": (start_p, len(all_page_ids)),
                "chunks": chunks,
            }
        )

    return {
        "session_records": session_records,
        "page_tensor": torch.tensor(all_page_ids, dtype=torch.long).unsqueeze(0),
        "page_texts": all_page_texts,
        "page_session_ids": page_session_ids,
        "answer_session_ids": ans_sids,
        "num_sessions": len(session_records),
        "num_pages": len(all_page_ids),
        "content_tokens": content_tokens,
    }


def _truncate_text_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Truncate text cleanly to at most max_tokens without clipping outer prompt tags."""
    if not text or max_tokens <= 0:
        return ""
    toks = tokenizer.encode(text, add_special_tokens=False)
    if len(toks) <= max_tokens:
        return text
    return tokenizer.decode(toks[:max_tokens], skip_special_tokens=True).strip()


def _compose_longmemeval_compact_prompt(
    rolling_summary: str,
    chunk_str: str,
    tokenizer,
    *,
    max_summary_tokens: int = 256,
    max_chunk_tokens: int = 2000,
) -> str:
    """Truncate summary & chunk BEFORE wrapping ChatML turn markers so <|im_start|>assistant never gets cut."""
    safe_sum = _truncate_text_to_tokens(rolling_summary.strip(), tokenizer, max_summary_tokens)
    safe_chunk = _truncate_text_to_tokens(chunk_str.strip(), tokenizer, max_chunk_tokens)
    return (
        f"<|im_start|>user\n"
        f"# Rolling Memory Compaction\n"
        f"Merge the existing memory summary and new dialogue sessions into concise factual bullet points "
        f"(keep all user preferences, updates, dates, numbers, and names; supersede outdated facts).\n\n"
        f"# Prior Summary:\n{safe_sum or '(none)'}\n\n"
        f"# New Sessions:\n{safe_chunk}\n"
        f"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n- "
    )


def split_sessions_into_lossless_chunks(
    session_records: list[dict[str, Any]],
    tokenizer,
    *,
    max_chunk_tokens: int = 2000,
) -> list[dict[str, Any]]:
    """Pack chronological sessions into compaction windows strictly <= max_chunk_tokens with ZERO dropped tokens.

    Flushes the window BEFORE `window_tok_cnt + piece_len > max_chunk_tokens`, and splits any single
    session longer than `max_chunk_tokens` into sequential sub-chunks of at most `max_chunk_tokens`.
    """
    cap = max(1, int(max_chunk_tokens))
    pieces: list[tuple[list[int], str]] = []
    for rec in session_records:
        s_txt = str(rec.get("text", "")).strip()
        if not s_txt:
            continue
        toks = list(tokenizer.encode(s_txt, add_special_tokens=False))
        if not toks:
            continue
        if len(toks) <= cap:
            pieces.append((toks, s_txt))
        else:
            for st in range(0, len(toks), cap):
                sub_toks = toks[st : st + cap]
                sub_txt = tokenizer.decode(sub_toks, skip_special_tokens=True).strip()
                pieces.append((sub_toks, sub_txt))

    windows: list[dict[str, Any]] = []
    cur_texts: list[str] = []
    cur_toks: list[int] = []
    for sub_toks, sub_txt in pieces:
        if cur_toks and (len(cur_toks) + len(sub_toks) > cap):
            windows.append({"text": "\n".join(cur_texts), "token_ids": list(cur_toks), "num_tokens": len(cur_toks)})
            cur_texts = []
            cur_toks = []
        cur_texts.append(sub_txt)
        cur_toks.extend(sub_toks)
    if cur_toks:
        windows.append({"text": "\n".join(cur_texts), "token_ids": list(cur_toks), "num_tokens": len(cur_toks)})
    return windows


def _answer_in_chosen_pages(gold_answer: str, chosen_texts: list[str]) -> float:
    """Check whether normalized gold answer substring appears in the retrieved chosen pages."""
    norm_gold = _normalize_qa_text(gold_answer)
    if not norm_gold:
        return 0.0
    joined = _normalize_qa_text("\n".join(chosen_texts))
    if norm_gold in joined:
        return 1.0
    gold_toks = [w for w in norm_gold.split() if len(w) >= 3 or any(ch.isdigit() for ch in w)]
    if gold_toks and all(w in joined.split() for w in gold_toks):
        return 1.0
    return 0.0


@torch.no_grad()
def _build_longmemeval_auto_compact_prefix(
    session_records: list[dict[str, Any]],
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    device: torch.device,
    *,
    compact_budget_tokens: int = 256,
    window_trigger_tokens: int = 2000,
    max_chunk_tokens: int = 2000,
    return_summary_text: bool = False,
) -> tuple[torch.Tensor, float] | tuple[torch.Tensor, float, str, int]:
    """Lossless rolling real-LLM context compaction over chronological LongMemEval sessions into iso-token budget."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    _sync_device(device)
    t0 = time.perf_counter()

    if isinstance(wrapper.llm, TinyMockCausalLM):
        joined = "\n".join(rec["text"].strip() for rec in session_records)
        ids = _encode_text_or_mock(joined, tokenizer, len(embed_fn.weight))[:compact_budget_tokens]
        if not ids:
            ids = [2]
        t_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        pref = embed_fn(t_ids).to(llm_dtype)
        _sync_device(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if return_summary_text:
            return pref, elapsed_ms, tokenizer.decode(ids, skip_special_tokens=True).strip(), 1
        return pref, elapsed_ms

    # Guarantee 0 silently dropped tokens: flush BEFORE exceeding effective cap and split oversized sessions
    eff_cap = max(1, min(int(max_chunk_tokens), int(window_trigger_tokens)))
    windows = split_sessions_into_lossless_chunks(session_records, tokenizer, max_chunk_tokens=eff_cap)
    rolling_summary = ""
    num_windows = len(windows)
    for win in windows:
        chunk_str = str(win["text"])
        compact_prompt = _compose_longmemeval_compact_prompt(
            rolling_summary,
            chunk_str,
            tokenizer,
            max_summary_tokens=compact_budget_tokens,
            max_chunk_tokens=eff_cap + 64,
        )
        c_ids = torch.tensor(
            tokenizer.encode(compact_prompt, add_special_tokens=False),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        empty_pref = torch.empty((1, 0, int(embed_fn.weight.shape[1])), device=device, dtype=llm_dtype)
        raw_sum = _greedy_generate_qa(
            wrapper.llm,
            empty_pref,
            c_ids,
            tokenizer,
            max_new_tokens=compact_budget_tokens,
            stop_fn=lambda s: "<|im_end|>" in s,
        )
        cleaned = _strip_think_blocks(raw_sum)
        sum_ids = tokenizer.encode(cleaned, add_special_tokens=False)[:compact_budget_tokens]
        rolling_summary = tokenizer.decode(sum_ids, skip_special_tokens=True).strip()

    final_ids = tokenizer.encode(rolling_summary or "No prior sessions.", add_special_tokens=False)[:compact_budget_tokens]
    if not final_ids:
        final_ids = [2]
    pref = embed_fn(torch.tensor(final_ids, dtype=torch.long, device=device).unsqueeze(0)).to(llm_dtype)
    _sync_device(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if return_summary_text:
        return pref, elapsed_ms, rolling_summary, num_windows
    return pref, elapsed_ms


@torch.no_grad()
def evaluate_longmemeval_benchmark(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    items: list[dict[str, Any]],
    dataset_meta: dict[str, Any],
    device: torch.device,
    *,
    lslc_k_deep: int = 4,
    lslc_k_skim: int = 2,
    chapter_size: int = 16,
    compact_budget_tokens: int = 256,
    compact_window_trigger: int = 2000,
    compact_max_chunk_tokens: int = 2000,
    max_pages: int = 0,
    max_new_tokens: int = 32,
    bge_toc: Any | None = None,
    salience_route_weight: float = 1.5,
    systems_override: tuple[str, ...] | None = None,
    flush_callback: Any | None = None,
    flush_every: int = 10,
) -> dict[str, Any]:
    """Evaluate LongMemEval Pareto curve: bm25_rag_k1/k2/k4, llm_auto_compact, and lslc_v3_e2e_gpu."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    eff_span_len = int(getattr(wrapper.cfg, "salience_span_len", 1))
    wrapper.eval()

    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)
    systems = (
        tuple(systems_override)
        if systems_override is not None
        else ("bm25_rag_k1", "bm25_rag_k2", "bm25_rag_k4", "llm_auto_compact", "lslc_v3_e2e_gpu")
    )
    rows_by_sys_cat: dict[str, dict[str, list[dict[str, Any]]]] = {s: {} for s in systems}
    all_rows_by_sys: dict[str, list[dict[str, Any]]] = {s: [] for s in systems}
    pack_num_sessions: list[int] = []
    pack_num_pages: list[int] = []
    pack_content_tokens: list[int] = []
    t_lme_start = time.perf_counter()

    def _build_lme_snapshot(completed_items: int, *, log_final: bool = False) -> dict[str, Any]:
        snap_by_sys: dict[str, Any] = {}
        for sys_name in systems:
            s_rows = all_rows_by_sys[sys_name]
            overall = _aggregate_eval_rows(s_rows)
            hit_vals = [float(r["session_hit_at_k"]) for r in s_rows if r.get("session_hit_at_k") is not None]
            if hit_vals:
                hit_m = round(sum(hit_vals) / len(hit_vals), 4)
                overall["session_hit_at_k"] = hit_m
                overall["gold_session_hit_rate"] = hit_m
            ans_in_vals = [
                float(r["answer_substr_in_chosen_pages_rate"])
                for r in s_rows
                if r.get("answer_substr_in_chosen_pages_rate") is not None
            ]
            if ans_in_vals:
                overall["answer_substr_in_chosen_pages_rate"] = round(sum(ans_in_vals) / len(ans_in_vals), 4)
            if sys_name == "lslc_v3_e2e_gpu":
                sg_vals = [float(r["mean_strip_gate"]) for r in s_rows if r.get("mean_strip_gate") is not None]
                bw_vals = [float(r["num_bank_writes"]) for r in s_rows if r.get("num_bank_writes") is not None]
                overall["mean_strip_gate"] = round(sum(sg_vals) / len(sg_vals), 4) if sg_vals else 0.0
                overall["num_bank_writes_mean"] = round(sum(bw_vals) / len(bw_vals), 3) if bw_vals else 0.0
            by_cat = {
                c: _aggregate_eval_rows(rlist) for c, rlist in sorted(rows_by_sys_cat[sys_name].items())
            }
            samples: list[dict[str, Any]] = []
            seen_qids: set[str] = set()
            for c_name in sorted(rows_by_sys_cat[sys_name]):
                if len(samples) >= 6:
                    break
                r0 = rows_by_sys_cat[sys_name][c_name][0]
                seen_qids.add(r0["question_id"])
                samples.append(
                    {
                        "question_id": r0["question_id"],
                        "question_type": r0["question_type"],
                        "question": r0["question"],
                        "gold": r0["gold"],
                        "pred": r0["pred"],
                        "extracted_em": r0["extracted_em"],
                        "token_f1": r0["token_f1"],
                        "substring_match": r0["substring_match"],
                    }
                )
            for r in s_rows:
                if len(samples) >= 6:
                    break
                if r["question_id"] in seen_qids:
                    continue
                seen_qids.add(r["question_id"])
                samples.append(
                    {
                        "question_id": r["question_id"],
                        "question_type": r["question_type"],
                        "question": r["question"],
                        "gold": r["gold"],
                        "pred": r["pred"],
                        "extracted_em": r["extracted_em"],
                        "token_f1": r["token_f1"],
                        "substring_match": r["substring_match"],
                    }
                )
            snap_by_sys[sys_name] = {"overall": overall, "by_category": by_cat, "samples": samples}
            if log_final:
                print(
                    f"[LongMemEval {sys_name}] EM={overall['extracted_em']:.3f} F1={overall['token_f1']:.3f} "
                    f"Sub={overall['substring_match']:.3f} tok={overall['active_prefix_tokens']:.1f} "
                    f"upd={overall['memory_update_ms']:.1f}ms q={overall['per_query_ms']:.1f}ms toc_diff={overall['max_toc_diff']}",
                    flush=True,
                )

        packed_meta_s = {
            **dataset_meta,
            "completed_items": int(completed_items),
            "total_items": len(items),
            "max_pages": max_pages,
            "mean_num_sessions": round(sum(pack_num_sessions) / max(1, len(pack_num_sessions)), 2),
            "mean_num_pages": round(sum(pack_num_pages) / max(1, len(pack_num_pages)), 2),
            "mean_content_tokens": round(sum(pack_content_tokens) / max(1, len(pack_content_tokens)), 1),
            "elapsed_sec": round(time.perf_counter() - t_lme_start, 1),
        }
        return {"dataset": packed_meta_s, "systems": snap_by_sys}

    for item_idx, item in enumerate(items, start=1):
        cat = str(item.get("question_type", "unknown"))
        question = str(item["question"])
        gold_answer = str(item["answer"])
        pack = _pack_longmemeval_sessions(item, tokenizer, wrapper.cfg.page_size, max_pages=max_pages)
        pack_num_sessions.append(int(pack.get("num_sessions", len(pack["session_records"]))))
        pack_num_pages.append(int(pack.get("num_pages", len(pack["page_texts"]))))
        pack_content_tokens.append(int(pack.get("content_tokens", 0)))
        page_texts: list[str] = pack["page_texts"]
        page_sids: list[str] = pack["page_session_ids"]
        ans_sids: set[str] = pack["answer_session_ids"]
        session_records: list[dict[str, Any]] = pack["session_records"]

        p_prompt = torch.tensor(
            tokenizer.encode(_format_qa_prompt(question), add_special_tokens=False),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        for sys_name in systems:
            _reset_peak_vram(device)
            max_toc_diff = 0.0
            session_hit: float | None = None
            ans_in_pages: float | None = None
            ep_strip_gates: list[float] = []
            ep_bank_writes = 0

            if sys_name.startswith("bm25_rag_k"):
                k_val = int(sys_name.rsplit("_k", 1)[-1])
                t_m0 = time.perf_counter()
                bm25 = BM25PageRetriever(page_texts)
                mem_ms = (time.perf_counter() - t_m0) * 1000.0

                _sync_device(device)
                t_q0 = time.perf_counter()
                rr = bm25.route(question, k_deep=min(k_val, len(page_texts)), k_skim=0, device=device)
                chosen = rr.deep_idx[0].tolist()
                chosen_sids = {page_sids[i] for i in chosen if 0 <= i < len(page_sids)}
                session_hit = float(bool(ans_sids & chosen_sids)) if ans_sids else None
                chosen_texts = [page_texts[i] for i in chosen if 0 <= i < len(page_texts)]
                ans_in_pages = _answer_in_chosen_pages(gold_answer, chosen_texts)
                rag_txt = "\n\n".join(t.strip() for t in chosen_texts) + "\n"
                r_ids = _encode_unpadded_rag_ids(rag_txt, tokenizer, vocab_size, device)
                pref_embeds = embed_fn(r_ids).to(llm_dtype)
                gen_txt = _greedy_generate_qa(
                    wrapper.llm, pref_embeds, p_prompt, tokenizer, max_new_tokens=max_new_tokens, user_open_ids=user_open_ids
                )
                _sync_device(device)
                q_ms = (time.perf_counter() - t_q0) * 1000.0
                pref_toks = int(pref_embeds.shape[1])

            elif sys_name == "llm_auto_compact":
                pref_embeds, mem_ms = _build_longmemeval_auto_compact_prefix(
                    session_records,
                    wrapper,
                    tokenizer,
                    device,
                    compact_budget_tokens=compact_budget_tokens,
                    window_trigger_tokens=compact_window_trigger,
                    max_chunk_tokens=compact_max_chunk_tokens,
                )
                _sync_device(device)
                t_q0 = time.perf_counter()
                gen_txt = _greedy_generate_qa(
                    wrapper.llm, pref_embeds, p_prompt, tokenizer, max_new_tokens=max_new_tokens, user_open_ids=user_open_ids
                )
                _sync_device(device)
                q_ms = (time.perf_counter() - t_q0) * 1000.0
                pref_toks = int(pref_embeds.shape[1])

            else:  # lslc_v3_e2e_gpu (Frozen Mode A + modular session encoding + blind DeltaUpdater + 2-level TOC)
                _sync_device(device)
                t_m0 = time.perf_counter()
                z_book: torch.Tensor | None = None
                toc_embeds: torch.Tensor | None = None
                exact_bank: SparseExactBank | None = None
                page_ctx: torch.Tensor | None = None
                bge_pages: torch.Tensor | None = None

                # Modular per-session encoding + blind GPU-routed DeltaUpdater (zero oracle flags!)
                for rec in session_records:
                    s_tensor = torch.tensor(rec["chunks"], dtype=torch.long, device=device).unsqueeze(0)  # [1, Ps, Tp]
                    zc, bc, tc = _encode_book_chunked(
                        wrapper.lslc,
                        s_tensor,
                        embed_fn,
                        vocab_size,
                        device,
                        amp_dtype,
                        autocast_enabled,
                        chunk_size=16,
                        special_token_ids=special_ids,
                        salience_span_len=eff_span_len,
                    )
                    if bge_toc is not None:
                        bge_sc = bge_toc.encode_book_pages(s_tensor, None, tokenizer, num_spans=4, batch_size=128)
                        pc_ctx = None
                    else:
                        bge_sc = None
                        pc_ctx = _compute_page_ctx_keys(
                            wrapper, s_tensor, tokenizer, amp_dtype, autocast_enabled, chunk_size=16
                        )
                    if z_book is None:
                        z_book = zc
                        toc_embeds = tc
                        page_ctx = pc_ctx
                        bge_pages = bge_sc
                        num_p_cur = z_book.shape[1]
                        k_cur = bc.token_ids.shape[-1]
                        exact_bank = SparseExactBank(
                            token_ids=bc.token_ids,
                            page_ids=torch.arange(num_p_cur, device=device, dtype=torch.int16)
                            .view(1, num_p_cur, 1)
                            .expand(1, num_p_cur, k_cur)
                            .contiguous(),
                            h_coords=bc.h_coords,
                            w_coords=bc.w_coords,
                        )
                    else:
                        assert toc_embeds is not None and exact_bank is not None
                        prev_num_p = z_book.shape[1]
                        user_txt = str(rec["user_text"]).strip()
                        if user_txt and prev_num_p >= 1:
                            edit_ids = torch.tensor(
                                _encode_text_or_mock(user_txt, tokenizer, vocab_size)[:96],
                                dtype=torch.long,
                                device=device,
                            ).unsqueeze(0)
                            ext_upd = (
                                bge_toc.score_book(bge_toc.encode_texts([user_txt], is_query=True), bge_pages)
                                if (bge_toc is not None and bge_pages is not None)
                                else None
                            )
                            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                                q_ctx_e = (
                                    None
                                    if ext_upd is not None
                                    else _compute_query_ctx_embeds(wrapper, edit_ids, amp_dtype, autocast_enabled)
                                )
                                r_upd, _ = wrapper.lslc.route_and_assemble(
                                    z_book=z_book,
                                    toc_embeds=toc_embeds,
                                    query_embed=embed_fn(edit_ids).mean(dim=1),
                                    exact_bank=exact_bank,
                                    embed_tokens_fn=embed_fn,
                                    k_deep=1,
                                    k_skim=0,
                                    mode="toc_only",
                                    training_ste=False,
                                    query_input_ids=edit_ids,
                                    salience_route_weight=salience_route_weight,
                                    contextual_query_embeds=q_ctx_e,
                                    contextual_page_keys=page_ctx,
                                    external_dense_logits=ext_upd,
                                )
                                tgt_p_upd = r_upd.deep_idx[:, -1]
                                p_upd_i = int(tgt_p_upd[0].item())
                                toc_before_page = wrapper.lslc.projector.project_toc(z_book[:, p_upd_i : p_upd_i + 1]).clone()
                                override_e = (
                                    extract_contextual_edit_embeds(
                                        wrapper.llm, edit_ids, layer_idx=wrapper.cfg.contextual_edit_layer
                                    )
                                    if wrapper.cfg.use_contextual_edit_encoding
                                    else None
                                )
                                prev_bank_tok = exact_bank.token_ids[:, p_upd_i].clone()
                                z_book, g_upd, exact_bank = wrapper.lslc.apply_page_delta(
                                    z_book,
                                    tgt_p_upd,
                                    edit_ids,
                                    embed_tokens_fn=embed_fn,
                                    exact_bank=exact_bank,
                                    vocab_size=vocab_size,
                                    edit_embeds_override=override_e,
                                    in_place=True,
                                    bank_write_threshold=0.45,
                                )
                                ep_strip_gates.append(float(g_upd.float().mean().item()))
                                if not torch.equal(exact_bank.token_ids[:, p_upd_i], prev_bank_tok):
                                    ep_bank_writes += 1
                                toc_after_page = wrapper.lslc.projector.project_toc(z_book[:, p_upd_i : p_upd_i + 1])
                            diff = float((toc_after_page.float() - toc_before_page.float()).abs().max().item())
                            if diff > max_toc_diff:
                                max_toc_diff = diff

                        z_book = torch.cat((z_book, zc), dim=1)
                        toc_embeds = torch.cat((toc_embeds, tc), dim=1)
                        if page_ctx is not None and pc_ctx is not None:
                            page_ctx = torch.cat((page_ctx, pc_ctx), dim=1)
                        if bge_pages is not None and bge_sc is not None:
                            bge_pages = torch.cat((bge_pages, bge_sc), dim=1)
                        new_num_p = z_book.shape[1]
                        k_cur = exact_bank.token_ids.shape[-1]
                        new_tok = torch.cat((exact_bank.token_ids, bc.token_ids), dim=1)
                        new_h = torch.cat((exact_bank.h_coords, bc.h_coords), dim=1)
                        new_w = torch.cat((exact_bank.w_coords, bc.w_coords), dim=1)
                        new_base = (
                            None
                            if exact_bank.base_token_ids is None
                            else torch.cat((exact_bank.base_token_ids, bc.token_ids), dim=1)
                        )
                        exact_bank = SparseExactBank(
                            token_ids=new_tok,
                            page_ids=torch.arange(new_num_p, device=device, dtype=torch.int16)
                            .view(1, new_num_p, 1)
                            .expand(1, new_num_p, k_cur)
                            .contiguous(),
                            h_coords=new_h,
                            w_coords=new_w,
                            base_token_ids=new_base,
                        )
                _sync_device(device)
                mem_ms = (time.perf_counter() - t_m0) * 1000.0
                assert z_book is not None and toc_embeds is not None and exact_bank is not None

                rq_ids = torch.tensor(
                    _encode_text_or_mock(question, tokenizer, vocab_size), dtype=torch.long, device=device
                ).unsqueeze(0)
                _sync_device(device)
                t_q0 = time.perf_counter()
                ext_q = (
                    bge_toc.score_book(bge_toc.encode_texts([question], is_query=True), bge_pages)
                    if (bge_toc is not None and bge_pages is not None)
                    else None
                )
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    q_ctx = (
                        None
                        if ext_q is not None
                        else _compute_query_ctx_embeds(wrapper, rq_ids, amp_dtype, autocast_enabled)
                    )
                    q_emb = embed_fn(rq_ids).mean(dim=1)
                    kd_use = min(lslc_k_deep, z_book.shape[1])
                    ks_use = min(lslc_k_skim, max(0, z_book.shape[1] - kd_use))
                    r_out, asm = wrapper.lslc.route_and_assemble(
                        z_book=z_book,
                        toc_embeds=toc_embeds,
                        query_embed=q_emb,
                        exact_bank=exact_bank,
                        embed_tokens_fn=embed_fn,
                        k_deep=kd_use,
                        k_skim=ks_use,
                        mode="full",
                        training_ste=False,
                        hierarchical_chapter_size=chapter_size,
                        query_input_ids=rq_ids,
                        salience_route_weight=salience_route_weight,
                        exact_bank_attenuation="none",
                        contextual_query_embeds=q_ctx,
                        contextual_page_keys=page_ctx,
                        external_dense_logits=ext_q,
                    )
                chosen = r_out.deep_idx[0].tolist()
                chosen_sids = {page_sids[i] for i in chosen if 0 <= i < len(page_sids)}
                session_hit = float(bool(ans_sids & chosen_sids)) if ans_sids else None
                chosen_texts = [page_texts[i] for i in chosen if 0 <= i < len(page_texts)]
                ans_in_pages = _answer_in_chosen_pages(gold_answer, chosen_texts)
                gen_txt = _greedy_generate_qa(
                    wrapper.llm,
                    asm.prefix_embeds.to(llm_dtype),
                    p_prompt,
                    tokenizer,
                    max_new_tokens=max_new_tokens,
                    user_open_ids=user_open_ids,
                )
                _sync_device(device)
                q_ms = (time.perf_counter() - t_q0) * 1000.0
                pref_toks = int(asm.prefix_embeds.shape[1])

            qm = compute_community_qa_metrics(gen_txt, gold_answer)
            row = {
                **qm,
                "question_id": str(item.get("question_id", "")),
                "question_type": cat,
                "question": question,
                "gold": gold_answer,
                "pred": gen_txt,
                "session_hit_at_k": session_hit,
                "gold_session_hit_rate": session_hit,
                "answer_substr_in_chosen_pages_rate": ans_in_pages,
                "active_prefix_tokens": pref_toks,
                "memory_update_ms": round(mem_ms, 3),
                "per_query_ms": round(q_ms, 3),
                "peak_vram_mb": _peak_vram_mb(device),
                "max_toc_diff": round(max_toc_diff, 6),
            }
            if sys_name == "lslc_v3_e2e_gpu":
                row["mean_strip_gate"] = round(sum(ep_strip_gates) / len(ep_strip_gates), 4) if ep_strip_gates else 0.0
                row["num_bank_writes"] = ep_bank_writes
            rows_by_sys_cat[sys_name].setdefault(cat, []).append(row)
            all_rows_by_sys[sys_name].append(row)

        if (flush_every > 0 and item_idx % flush_every == 0) or item_idx == len(items):
            el = time.perf_counter() - t_lme_start
            parts_p = [
                f"{s}_F1={sum(r['token_f1'] for r in all_rows_by_sys[s])/len(all_rows_by_sys[s]):.3f}"
                for s in systems
                if all_rows_by_sys[s]
            ]
            print(f"[LongMemEval {item_idx}/{len(items)} {el:.0f}s] " + " | ".join(parts_p), flush=True)
            if flush_callback is not None:
                flush_callback(_build_lme_snapshot(item_idx, log_final=False))

    return _build_lme_snapshot(len(items), log_final=True)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_diag_probe(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    val_pool: list[int],
    lme_items: list[dict[str, Any]],
    lme_meta: dict[str, Any],
    device: torch.device,
    *,
    g2_checkpoint: str = "",
    g3_checkpoint: str = "",
    k_exact: int = 24,
    episodes_per_task: int = 4,
    mk_distractors: int = 48,
    lslc_k_deep: int = 4,
    lslc_k_skim: int = 2,
    chapter_size: int = 16,
    compact_budget_tokens: int = 256,
    max_pages: int = 64,
    max_new_tokens: int = 32,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """2-minute diagnostic probe: (2a) exact_bank 7-digit capture G2 vs G3, (2b) G2 prompt ablation on RULER, (2c) LongMemEval rolling_summary[:250]."""
    import copy

    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32

    cfg_base = copy.deepcopy(wrapper.cfg)
    cfg_base.use_contextual_query_encoding = False
    cfg_base.contextual_query_layer = None
    cfg_base.k_exact = k_exact

    g2_sys = LSLCSystem(copy.deepcopy(cfg_base)).to(device).eval()
    g2_info = _load_lslc_checkpoint(g2_sys, g2_checkpoint) if g2_checkpoint else {"path": None, "note": "mock_init"}
    g3_sys = LSLCSystem(copy.deepcopy(cfg_base)).to(device).eval()
    g3_info = _load_lslc_checkpoint(g3_sys, g3_checkpoint) if g3_checkpoint else {"path": None, "note": "mock_init"}

    ctx_len = 1024 if is_mock else 4096
    diag_tasks = ("s_niah", "mk_niah")
    eps_by_task = {
        t_name: build_ruler_episodes(
            task_name=t_name,
            context_length=ctx_len,
            num_episodes=episodes_per_task,
            tokenizer=tokenizer,
            val_pool=val_pool,
            page_size=cfg_base.page_size,
            seed=seed,
            mk_distractors=mk_distractors,
            is_mock=is_mock,
        )
        for t_name in diag_tasks
    }

    # 2a. Exact-bank 7-digit audit for G2 vs G3
    bank_audit: dict[str, Any] = {}
    for sys_label, sys_mod in (("v3_g2_cond_b", g2_sys), ("v3_g3_aligned", g3_sys)):
        per_task_audit: dict[str, Any] = {}
        all_rows_a: list[dict[str, Any]] = []
        for t_name in diag_tasks:
            t_rows: list[dict[str, Any]] = []
            examples: list[dict[str, Any]] = []
            for ep in eps_by_task[t_name]:
                p_t = ep["pages"].unsqueeze(0).to(device)
                _, bank, _ = _encode_book_chunked(
                    sys_mod, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled, chunk_size=32
                )
                gold_p = int(ep["gold_pages"][0])
                bank_toks = [int(t) for t in bank.token_ids[0, gold_p].tolist() if int(t) >= 0]
                bank_decoded = tokenizer.decode(bank_toks, skip_special_tokens=True)
                ans = str(ep["answer"])
                ans_toks = [int(t) for t in _encode_text_or_mock(ans, tokenizer, vocab_size)]
                c_ans = Counter(ans_toks)
                c_bank = Counter(bank_toks)
                contiguous_ok = 1.0 if ans in bank_decoded else 0.0
                multiset_ok = 1.0 if (ans_toks and all(c_bank[k] >= v for k, v in c_ans.items())) else 0.0
                tok_rec = sum(min(c_bank[k], v) for k, v in c_ans.items()) / max(1, len(ans_toks))
                key_str = str(ep["metadata"].get("key") or ep["metadata"].get("target_key") or "")
                key_ok = 1.0 if (not key_str or key_str.lower() in bank_decoded.lower()) else 0.0
                rec_a = {
                    "all_7_digits_contiguous": contiguous_ok,
                    "all_7_digits_multiset": multiset_ok,
                    "digit_token_recall": tok_rec,
                    "key_in_bank": key_ok,
                }
                t_rows.append(rec_a)
                all_rows_a.append(rec_a)
                if len(examples) < 2:
                    examples.append(
                        {
                            "gold": ans,
                            "key": key_str,
                            "all_7_digits_contiguous": contiguous_ok,
                            "all_7_digits_multiset": multiset_ok,
                            "bank_decoded": bank_decoded,
                            "bank_pieces": [tokenizer.decode([t], skip_special_tokens=True) for t in bank_toks],
                        }
                    )
            per_task_audit[t_name] = {
                "num_episodes": len(t_rows),
                "all_7_digits_contiguous_rate": round(sum(r["all_7_digits_contiguous"] for r in t_rows) / max(1, len(t_rows)), 4),
                "all_7_digits_multiset_rate": round(sum(r["all_7_digits_multiset"] for r in t_rows) / max(1, len(t_rows)), 4),
                "digit_token_recall_mean": round(sum(r["digit_token_recall"] for r in t_rows) / max(1, len(t_rows)), 4),
                "key_in_bank_rate": round(sum(r["key_in_bank"] for r in t_rows) / max(1, len(t_rows)), 4),
                "examples": examples,
            }
        bank_audit[sys_label] = {
            "overall": {
                "num_episodes": len(all_rows_a),
                "all_7_digits_contiguous_rate": round(sum(r["all_7_digits_contiguous"] for r in all_rows_a) / max(1, len(all_rows_a)), 4),
                "all_7_digits_multiset_rate": round(sum(r["all_7_digits_multiset"] for r in all_rows_a) / max(1, len(all_rows_a)), 4),
                "digit_token_recall_mean": round(sum(r["digit_token_recall"] for r in all_rows_a) / max(1, len(all_rows_a)), 4),
                "key_in_bank_rate": round(sum(r["key_in_bank"] for r in all_rows_a) / max(1, len(all_rows_a)), 4),
            },
            "by_task": per_task_audit,
        }
        print(f"[DIAG exact_bank {sys_label}] {bank_audit[sys_label]['overall']}", flush=True)

    # 2b. G2 RULER 4K prompt ablation: chatml_full, plain_stage1_full, chatml_exact_only
    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)
    ablation_specs = (
        ("chatml_full", "full", True),
        ("plain_stage1_full", "full", False),
        ("chatml_exact_only", "exact_only", True),
    )
    prompt_ablation: dict[str, Any] = {}
    for arm_name, asm_mode, use_chatml in ablation_specs:
        by_task_res: dict[str, Any] = {}
        all_rows_b: list[dict[str, Any]] = []
        for t_name in diag_tasks:
            t_rows_b: list[dict[str, Any]] = []
            raw_samples_b: list[dict[str, Any]] = []
            for ep in eps_by_task[t_name]:
                _reset_peak_vram(device)
                p_t = ep["pages"].unsqueeze(0).to(device)
                gold_set = set(ep["gold_pages"])
                t_m0 = time.perf_counter()
                z_book, bank, toc = _encode_book_chunked(
                    g2_sys, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled, chunk_size=32
                )
                mem_ms = (time.perf_counter() - t_m0) * 1000.0
                rq_ids = torch.tensor(
                    _encode_text_or_mock(ep["router_query"], tokenizer, vocab_size), dtype=torch.long, device=device
                ).unsqueeze(0)
                if use_chatml:
                    q_str = _format_qa_prompt(ep["question"])
                    u_open = user_open_ids
                else:
                    q_str = f"\nQuestion: {ep['question'].strip()}\nAnswer:"
                    u_open = None
                p_prompt = torch.tensor(tokenizer.encode(q_str, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)

                _sync_device(device)
                t_q0 = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    q_emb = embed_fn(rq_ids).mean(dim=1)
                    kd_use = min(lslc_k_deep, p_t.shape[1])
                    ks_use = min(lslc_k_skim, max(0, p_t.shape[1] - kd_use))
                    r_out, asm = g2_sys.route_and_assemble(
                        z_book=z_book,
                        toc_embeds=toc,
                        query_embed=q_emb,
                        exact_bank=bank,
                        embed_tokens_fn=embed_fn,
                        k_deep=kd_use,
                        k_skim=ks_use,
                        mode=asm_mode,
                        training_ste=False,
                        hierarchical_chapter_size=chapter_size,
                        query_input_ids=rq_ids,
                        salience_route_weight=2.0,
                        exact_bank_attenuation="none",
                    )
                chosen = r_out.deep_idx[0].tolist()
                router_hit = 1.0 if gold_set.issubset(set(chosen)) else (len(gold_set & set(chosen)) / max(1, len(gold_set)))
                gen_txt = _greedy_generate_qa(
                    wrapper.llm,
                    asm.prefix_embeds.to(llm_dtype),
                    p_prompt,
                    tokenizer,
                    max_new_tokens=max_new_tokens,
                    user_open_ids=u_open,
                )
                _sync_device(device)
                q_ms = (time.perf_counter() - t_q0) * 1000.0
                qm = compute_community_qa_metrics(gen_txt, ep["answer"])
                t_rows_b.append(
                    {
                        **qm,
                        "router_hit_rate": router_hit,
                        "active_prefix_tokens": int(asm.prefix_embeds.shape[1]),
                        "memory_update_ms": round(mem_ms, 3),
                        "per_query_ms": round(q_ms, 3),
                        "peak_vram_mb": _peak_vram_mb(device),
                        "max_toc_diff": 0.0,
                    }
                )
                raw_samples_b.append(
                    {
                        "question": ep["question"],
                        "gold": ep["answer"],
                        "pred": gen_txt,
                        "router_hit": router_hit,
                        **qm,
                    }
                )
            agg_t = _aggregate_eval_rows(t_rows_b)
            agg_t["samples"] = raw_samples_b
            by_task_res[t_name] = agg_t
            all_rows_b.extend(t_rows_b)
        prompt_ablation[arm_name] = {
            "overall": _aggregate_eval_rows(all_rows_b),
            "by_task": by_task_res,
        }
        ov_b = prompt_ablation[arm_name]["overall"]
        print(
            f"[DIAG G2 {arm_name}] EM={ov_b['extracted_em']:.3f} F1={ov_b['token_f1']:.3f} "
            f"hit={ov_b.get('router_hit_rate', 0.0):.3f} tok={ov_b['active_prefix_tokens']:.1f}",
            flush=True,
        )

    # 2c. LongMemEval first 2 items rolling_summary[:250] probe
    compact_probe: list[dict[str, Any]] = []
    for item in lme_items[:2]:
        qid = str(item.get("question_id", ""))
        cat = str(item.get("question_type", ""))
        question = str(item["question"])
        gold = str(item["answer"])
        pack = _pack_longmemeval_sessions(item, tokenizer, cfg_base.page_size, max_pages=max_pages)
        pref_embeds, upd_ms, rolling_summary, num_windows = _build_longmemeval_auto_compact_prefix(
            pack["session_records"],
            wrapper,
            tokenizer,
            device,
            compact_budget_tokens=compact_budget_tokens,
            return_summary_text=True,
        )
        p_prompt = torch.tensor(
            tokenizer.encode(_format_qa_prompt(question), add_special_tokens=False),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        gen_txt = _greedy_generate_qa(
            wrapper.llm,
            pref_embeds.to(llm_dtype),
            p_prompt,
            tokenizer,
            max_new_tokens=max_new_tokens,
            user_open_ids=user_open_ids,
        )
        qm = compute_community_qa_metrics(gen_txt, gold)
        summary_250 = rolling_summary[:250]
        print(
            f"[DIAG LongMemEval compact qid={qid}] windows={num_windows} len={len(rolling_summary)} "
            f"summary[:250]={summary_250!r} | gold={gold!r} | pred={gen_txt!r}",
            flush=True,
        )
        compact_probe.append(
            {
                "question_id": qid,
                "question_type": cat,
                "question": question,
                "gold": gold,
                "pred": gen_txt,
                **qm,
                "num_windows": num_windows,
                "memory_update_ms": round(upd_ms, 2),
                "active_prefix_tokens": int(pref_embeds.shape[1]),
                "rolling_summary_len_chars": len(rolling_summary),
                "rolling_summary_first_250": summary_250,
                "rolling_summary_full": rolling_summary,
            }
        )

    return {
        "context_length": ctx_len,
        "g2_checkpoint": g2_info,
        "g3_checkpoint": g3_info,
        "longmemeval_meta": lme_meta,
        "ruler_exact_bank_audit": bank_audit,
        "ruler_g2_prompt_ablation": prompt_ablation,
        "longmemeval_compact_probe": compact_probe,
    }


@torch.no_grad()
def run_span_salience_probe(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    g2_checkpoint: str = "",
    episodes_per_task: int = 4,
    mk_distractors: int = 48,
    lslc_k_deep: int = 4,
    lslc_k_skim: int = 2,
    chapter_size: int = 16,
    max_new_tokens: int = 32,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """0-training probe comparing SalienceGate contiguous span_len=8 at k_exact=24 vs k_exact=48 on RULER 4K + router_stress."""
    import copy

    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    special_set = set(special_ids)
    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)

    ctx_len = 1024 if is_mock else 4096
    diag_tasks = ("s_niah", "mk_niah")
    eps_by_task = {
        t_name: build_ruler_episodes(
            task_name=t_name,
            context_length=ctx_len,
            num_episodes=episodes_per_task,
            tokenizer=tokenizer,
            val_pool=val_pool,
            page_size=wrapper.cfg.page_size,
            seed=seed,
            mk_distractors=mk_distractors,
            is_mock=is_mock,
        )
        for t_name in diag_tasks
    }

    configs_to_test = (
        ("span8_k24", 8, 24),
        ("span8_k48", 8, 48),
    )
    results_by_cfg: dict[str, Any] = {}
    for cfg_label, span_len, k_ex in configs_to_test:
        cfg_g2 = copy.deepcopy(wrapper.cfg)
        cfg_g2.use_contextual_query_encoding = False
        cfg_g2.contextual_query_layer = None
        cfg_g2.contextual_page_spans = 1
        cfg_g2.k_exact = min(k_ex, cfg_g2.page_size)
        cfg_g2.salience_span_len = span_len
        g2_sys = LSLCSystem(cfg_g2).to(device).eval()
        g2_info = _load_lslc_checkpoint(g2_sys, g2_checkpoint) if g2_checkpoint else {"path": None, "note": "mock_init"}

        # 1) Exact-bank capture audit
        audit_by_task: dict[str, Any] = {}
        all_audit_rows: list[dict[str, Any]] = []
        encoded_cache: dict[str, list[tuple[dict[str, Any], torch.Tensor, SparseExactBank, torch.Tensor, float]]] = {}
        for t_name in diag_tasks:
            t_audit: list[dict[str, Any]] = []
            examples: list[dict[str, Any]] = []
            enc_list = []
            for ep in eps_by_task[t_name]:
                p_t = ep["pages"].unsqueeze(0).to(device)
                _sync_device(device)
                t_m0 = time.perf_counter()
                z_book, bank, toc = _encode_book_chunked(
                    g2_sys,
                    p_t,
                    embed_fn,
                    vocab_size,
                    device,
                    amp_dtype,
                    autocast_enabled,
                    chunk_size=32,
                    special_token_ids=special_ids,
                    salience_span_len=span_len,
                )
                _sync_device(device)
                mem_ms = (time.perf_counter() - t_m0) * 1000.0
                enc_list.append((ep, z_book, bank, toc, mem_ms))

                gold_p = int(ep["gold_pages"][0])
                bank_toks = [int(t) for t in bank.token_ids[0, gold_p].tolist() if int(t) >= 0]
                sp_cnt = sum(1 for t in bank_toks if t in special_set)
                bank_decoded = tokenizer.decode(bank_toks, skip_special_tokens=True)
                ans = str(ep["answer"])
                ans_toks = [int(t) for t in _encode_text_or_mock(ans, tokenizer, vocab_size)]
                c_ans = Counter(ans_toks)
                c_bank = Counter(bank_toks)
                contiguous_ok = 1.0 if ans in bank_decoded else 0.0
                multiset_ok = 1.0 if (ans_toks and all(c_bank[k] >= v for k, v in c_ans.items())) else 0.0
                tok_rec = sum(min(c_bank[k], v) for k, v in c_ans.items()) / max(1, len(ans_toks))
                key_str = str(ep["metadata"].get("key") or ep["metadata"].get("target_key") or "")
                key_ok = 1.0 if (not key_str or key_str.lower() in bank_decoded.lower()) else 0.0
                rec_a = {
                    "all_7_digits_contiguous": contiguous_ok,
                    "all_7_digits_multiset": multiset_ok,
                    "digit_token_recall": tok_rec,
                    "key_in_bank": key_ok,
                    "special_tokens_in_bank": sp_cnt,
                }
                t_audit.append(rec_a)
                all_audit_rows.append(rec_a)
                if len(examples) < 2:
                    examples.append(
                        {
                            "gold": ans,
                            "key": key_str,
                            "all_7_digits_contiguous": contiguous_ok,
                            "digit_token_recall": round(tok_rec, 4),
                            "bank_decoded": bank_decoded,
                        }
                    )
            encoded_cache[t_name] = enc_list
            audit_by_task[t_name] = {
                "num_episodes": len(t_audit),
                "all_7_digits_contiguous_rate": round(sum(r["all_7_digits_contiguous"] for r in t_audit) / max(1, len(t_audit)), 4),
                "all_7_digits_multiset_rate": round(sum(r["all_7_digits_multiset"] for r in t_audit) / max(1, len(t_audit)), 4),
                "digit_token_recall_mean": round(sum(r["digit_token_recall"] for r in t_audit) / max(1, len(t_audit)), 4),
                "key_in_bank_rate": round(sum(r["key_in_bank"] for r in t_audit) / max(1, len(t_audit)), 4),
                "special_tokens_in_bank_total": int(sum(r["special_tokens_in_bank"] for r in t_audit)),
                "examples": examples,
            }
        overall_audit = {
            "num_episodes": len(all_audit_rows),
            "all_7_digits_contiguous_rate": round(sum(r["all_7_digits_contiguous"] for r in all_audit_rows) / max(1, len(all_audit_rows)), 4),
            "all_7_digits_multiset_rate": round(sum(r["all_7_digits_multiset"] for r in all_audit_rows) / max(1, len(all_audit_rows)), 4),
            "digit_token_recall_mean": round(sum(r["digit_token_recall"] for r in all_audit_rows) / max(1, len(all_audit_rows)), 4),
            "key_in_bank_rate": round(sum(r["key_in_bank"] for r in all_audit_rows) / max(1, len(all_audit_rows)), 4),
            "special_tokens_in_bank_total": int(sum(r["special_tokens_in_bank"] for r in all_audit_rows)),
        }

        # 2) E2E RULER generation on plain_stage1_full and chatml_exact_only
        modes_res: dict[str, Any] = {}
        for mode_label, asm_mode, use_chatml in (
            ("plain_stage1_full", "full", False),
            ("chatml_exact_only", "exact_only", True),
        ):
            by_task_m: dict[str, Any] = {}
            all_rows_m: list[dict[str, Any]] = []
            for t_name in diag_tasks:
                t_rows_m: list[dict[str, Any]] = []
                raw_samples_m: list[dict[str, Any]] = []
                for ep, z_book, bank, toc, mem_ms in encoded_cache[t_name]:
                    _reset_peak_vram(device)
                    gold_set = set(ep["gold_pages"])
                    rq_ids = torch.tensor(
                        _encode_text_or_mock(ep["router_query"], tokenizer, vocab_size), dtype=torch.long, device=device
                    ).unsqueeze(0)
                    if use_chatml:
                        q_str = _format_qa_prompt(ep["question"])
                        u_open = user_open_ids
                    else:
                        q_str = f"\nQuestion: {ep['question'].strip()}\nAnswer:"
                        u_open = None
                    p_prompt = torch.tensor(tokenizer.encode(q_str, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)

                    _sync_device(device)
                    t_q0 = time.perf_counter()
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        q_emb = embed_fn(rq_ids).mean(dim=1)
                        kd_use = min(lslc_k_deep, z_book.shape[1])
                        ks_use = min(lslc_k_skim, max(0, z_book.shape[1] - kd_use))
                        r_out, asm = g2_sys.route_and_assemble(
                            z_book=z_book,
                            toc_embeds=toc,
                            query_embed=q_emb,
                            exact_bank=bank,
                            embed_tokens_fn=embed_fn,
                            k_deep=kd_use,
                            k_skim=ks_use,
                            mode=asm_mode,
                            training_ste=False,
                            hierarchical_chapter_size=chapter_size,
                            query_input_ids=rq_ids,
                            salience_route_weight=2.0,
                            exact_bank_attenuation="none",
                        )
                    chosen = r_out.deep_idx[0].tolist()
                    router_hit = 1.0 if gold_set.issubset(set(chosen)) else (len(gold_set & set(chosen)) / max(1, len(gold_set)))
                    gen_txt = _greedy_generate_qa(
                        wrapper.llm,
                        asm.prefix_embeds.to(llm_dtype),
                        p_prompt,
                        tokenizer,
                        max_new_tokens=max_new_tokens,
                        user_open_ids=u_open,
                    )
                    _sync_device(device)
                    q_ms = (time.perf_counter() - t_q0) * 1000.0
                    qm = compute_community_qa_metrics(gen_txt, ep["answer"])
                    t_rows_m.append(
                        {
                            **qm,
                            "router_hit_rate": router_hit,
                            "active_prefix_tokens": int(asm.prefix_embeds.shape[1]),
                            "memory_update_ms": round(mem_ms, 3),
                            "per_query_ms": round(q_ms, 3),
                            "peak_vram_mb": _peak_vram_mb(device),
                            "max_toc_diff": 0.0,
                        }
                    )
                    raw_samples_m.append(
                        {
                            "question": ep["question"],
                            "gold": ep["answer"],
                            "pred": gen_txt,
                            "router_hit": router_hit,
                            **qm,
                        }
                    )
                agg_m = _aggregate_eval_rows(t_rows_m)
                agg_m["samples"] = raw_samples_m
                by_task_m[t_name] = agg_m
                all_rows_m.extend(t_rows_m)
            modes_res[mode_label] = {"overall": _aggregate_eval_rows(all_rows_m), "by_task": by_task_m}

        # 3) router_stress_p64 & p128 under G2 AND under active wrapper.lslc (4-span router)
        stress_g2 = evaluate_router_stress_regression(
            wrapper=wrapper,
            tokenizer=tokenizer,
            val_pool=val_pool,
            device=device,
            is_mock=is_mock,
            seed=seed + 700,
            lslc_override=g2_sys,
            special_token_ids=special_ids,
            salience_span_len=span_len,
            use_contextual_keys=False,
        )
        prev_k_exact = wrapper.lslc.cfg.k_exact
        prev_span_len = wrapper.lslc.cfg.salience_span_len
        wrapper.lslc.cfg.k_exact = min(k_ex, wrapper.lslc.cfg.page_size)
        wrapper.lslc.cfg.salience_span_len = span_len
        try:
            stress_p4a = evaluate_router_stress_regression(
                wrapper=wrapper,
                tokenizer=tokenizer,
                val_pool=val_pool,
                device=device,
                is_mock=is_mock,
                seed=seed + 700,
                special_token_ids=special_ids,
                salience_span_len=span_len,
            )
        finally:
            wrapper.lslc.cfg.k_exact = prev_k_exact
            wrapper.lslc.cfg.salience_span_len = prev_span_len

        results_by_cfg[cfg_label] = {
            "salience_span_len": span_len,
            "k_exact": k_ex,
            "g2_checkpoint": g2_info,
            "exact_bank_audit": {"overall": overall_audit, "by_task": audit_by_task},
            "ruler_modes": modes_res,
            "router_stress_g2": stress_g2,
            "router_stress_active_router": stress_p4a,
        }
        print(
            f"[SPAN PROBE {cfg_label}] bank_7d={overall_audit['all_7_digits_contiguous_rate']:.3f} "
            f"dig_rec={overall_audit['digit_token_recall_mean']:.3f} key={overall_audit['key_in_bank_rate']:.3f} | "
            f"plain_EM={modes_res['plain_stage1_full']['overall']['extracted_em']:.3f} "
            f"exact_only_EM={modes_res['chatml_exact_only']['overall']['extracted_em']:.3f} | "
            f"stress_g2_p128={stress_g2['router_stress_p128']['hybrid_w2_top1_recall']:.3f} "
            f"stress_p4a_p128={stress_p4a['router_stress_p128']['hybrid_w2_top1_recall']:.3f}",
            flush=True,
        )

    return {
        "context_length": ctx_len,
        "configs": results_by_cfg,
    }


class _PlugInBiEncoderTOC:
    """Plug-in 33M semantic bi-encoder TOC (e.g. BAAI/bge-small-en-v1.5, 0 KV tokens added, 0 Qwen3 weight changes)."""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, device: torch.device, *, is_mock: bool = False) -> None:
        self.model_name = model_name
        self.device = device
        self.is_mock = is_mock
        self.dim = 384
        if is_mock:
            self.tokenizer = None
            self.model = None
        else:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else (
                torch.float16 if device.type == "cuda" else torch.float32
            )
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.dim = int(getattr(self.model.config, "hidden_size", 384))

    def _mock_embed(self, texts: list[str]) -> torch.Tensor:
        out = torch.zeros((len(texts), self.dim), dtype=torch.float32, device=self.device)
        for i, txt in enumerate(texts):
            toks = re.findall(r"[a-z0-9]+", txt.lower())
            if not toks:
                out[i, 0] = 1.0
                continue
            for t in toks:
                h1 = sum((k + 1) * ord(ch) for k, ch in enumerate(t)) % self.dim
                out[i, h1] += 1.0
                if len(t) >= 3:
                    for j in range(len(t) - 2):
                        tri = t[j : j + 3]
                        h2 = (ord(tri[0]) * 131 + ord(tri[1]) * 17 + ord(tri[2])) % self.dim
                        out[i, h2] += 0.5
        return F.normalize(out, dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], *, is_query: bool = False, batch_size: int = 64, max_length: int = 256) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, self.dim), dtype=torch.float32, device=self.device)
        prefixed = [f"{self.QUERY_PREFIX}{t.strip()}" if is_query else t.strip() for t in texts]
        if self.is_mock or self.model is None or self.tokenizer is None:
            return self._mock_embed(prefixed)

        outs: list[torch.Tensor] = []
        for s in range(0, len(prefixed), batch_size):
            batch = prefixed[s : s + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            res = self.model(**enc)
            cls_emb = res.last_hidden_state[:, 0].float()
            outs.append(F.normalize(cls_emb, p=2, dim=-1))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def encode_book_pages(
        self,
        page_tensor: torch.Tensor | None,
        page_texts: list[str] | None,
        llm_tokenizer,
        *,
        num_spans: int = 4,
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Encode P pages into [1, P, D] (num_spans=1) or [1, P, S, D] (num_spans>1) normalized vectors."""
        spans = max(1, int(num_spans))
        if spans == 1:
            if page_texts is not None:
                texts = list(page_texts)
            else:
                assert page_tensor is not None
                p_2d = page_tensor[0] if page_tensor.dim() == 3 else page_tensor
                texts = [llm_tokenizer.decode(p_2d[i].tolist(), skip_special_tokens=True) for i in range(p_2d.shape[0])]
            vecs = self.encode_texts(texts, is_query=False, batch_size=batch_size, max_length=512)
            return vecs.unsqueeze(0)  # [1, P, D]

        span_texts: list[str] = []
        if page_tensor is not None:
            p_2d = page_tensor[0] if page_tensor.dim() == 3 else page_tensor
            num_p, p_len = int(p_2d.shape[0]), int(p_2d.shape[1])
            pad_id = getattr(llm_tokenizer, "pad_token_id", None)
            for i in range(num_p):
                row_ids = p_2d[i].tolist()
                full_txt = llm_tokenizer.decode([t for t in row_ids if pad_id is None or t != pad_id], skip_special_tokens=True).strip()
                for sp_i in range(spans):
                    st = (sp_i * p_len) // spans
                    en = ((sp_i + 1) * p_len) // spans
                    seg_ids = [t for t in row_ids[st:en] if pad_id is None or t != pad_id]
                    seg_txt = llm_tokenizer.decode(seg_ids, skip_special_tokens=True).strip() if seg_ids else full_txt
                    span_texts.append(seg_txt or full_txt or ".")
        else:
            assert page_texts is not None
            num_p = len(page_texts)
            for txt in page_texts:
                words = txt.split()
                if len(words) < spans:
                    for _ in range(spans):
                        span_texts.append(txt or ".")
                else:
                    for sp_i in range(spans):
                        st = (sp_i * len(words)) // spans
                        en = max(st + 1, ((sp_i + 1) * len(words)) // spans)
                        span_texts.append(" ".join(words[st:en]))
        vecs = self.encode_texts(span_texts, is_query=False, batch_size=batch_size, max_length=160)
        return vecs.view(1, num_p, spans, self.dim)

    @staticmethod
    def score_book(q_vec: torch.Tensor, page_vecs: torch.Tensor) -> torch.Tensor:
        """Compute [B, P] cosine similarity (or MaxSim over S spans) in float32."""
        q_f = q_vec.float()
        if q_f.dim() == 1:
            q_f = q_f.unsqueeze(0)
        p_f = page_vecs.float()
        if p_f.dim() == 4:
            return torch.einsum("bd,bpsd->bps", q_f, p_f).amax(dim=-1)
        return torch.einsum("bd,bpd->bp", q_f, p_f)


def _finetune_salience_gate_span16(
    lslc_sys: LSLCSystem,
    embed_fn,
    tokenizer,
    train_pool: list[int],
    device: torch.device,
    *,
    steps: int = 150,
    batch_size: int = 8,
    lr: float = 1e-3,
    span_len: int = 16,
    mixed_natural_replay_ratio: float = 0.30,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Fine-tune ONLY SalienceGate (~182K params) on 70% disjoint general factual/numeric clauses + 30% Stage-1 mixed_natural train replay."""
    rng = random.Random(seed + 5151)
    vocab_size = len(embed_fn.weight)
    page_size = lslc_sys.cfg.page_size
    filler_vocab = train_pool if train_pool else list(range(2, max(16, vocab_size - 2)))

    # Pre-build Stage-1 mixed_natural train replay pages with full entity+value span masks (0 RULER / 0 LongMemEval!)
    replay_pages: list[tuple[list[int], list[float]]] = []
    if mixed_natural_replay_ratio > 0.0:
        n_rep_eps = 12 if is_mock else 48
        for mode_r, sd_off in (("budget", 6101), ("symbolic", 6203)):
            ds_r = AdversarialLivingBookDataset(
                num_episodes=n_rep_eps // 2,
                num_pages=6,
                page_size=page_size,
                num_edits=1,
                mode=mode_r,
                split="train",
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                seed=seed + sd_off,
                background_token_pool=train_pool,
                style="mixed_natural",
            )
            for ep in ds_r:
                sal_m = ep.metadata["context_salience_span_mask"]  # [P, Tp]
                for p_i in range(ep.context_pages.shape[0]):
                    if float(sal_m[p_i].sum().item()) > 0.5:
                        replay_pages.append(
                            (ep.context_pages[p_i].tolist(), (sal_m[p_i] > 0.0).float().tolist())
                        )

    # Freeze EVERYTHING except lslc_sys.salience_gate
    sal_ids = {id(p) for p in lslc_sys.salience_gate.parameters()}
    saved_reqs: list[tuple[torch.nn.Parameter, bool]] = []
    frozen_snap: dict[str, torch.Tensor] = {}
    for name, p in lslc_sys.named_parameters():
        saved_reqs.append((p, p.requires_grad))
        is_s = id(p) in sal_ids
        p.requires_grad_(is_s)
        if not is_s:
            frozen_snap[name] = p.detach().cpu().clone()

    sal_params = [p for p in lslc_sys.salience_gate.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(sal_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps), eta_min=lr * 0.1)

    # 100% disjoint general domain templates (ZERO RULER or LongMemEval wording)
    months = ("March", "April", "July", "October", "November", "January")
    cities = ("Rotterdam", "Osaka", "Zurich", "Halifax", "Bergen", "Lyon", "Bilbao", "Graz")
    prefixes = ("VX", "K7", "RT", "M4", "sector-b", "lot", "ref", "unit-c", "bay", "dock")

    def _make_disjoint_clause() -> str:
        kind = rng.randrange(7)
        tag = f"{rng.choice(prefixes)}-{rng.randint(10, 999)}"
        n7 = str(rng.randint(1000000, 9999999))
        n5 = str(rng.randint(10000, 99999))
        n3 = str(rng.randint(12, 980))
        if kind == 0:
            return f" Invoice #{tag} recorded settlement ${n5} on {rng.choice(months)} {rng.randint(1, 28)}. "
        if kind == 1:
            return f" Calibration gauge {tag} registered {n7} units during shift {rng.randint(1, 4)}. "
        if kind == 2:
            return f" Parcel {tag} cleared customs at {rng.choice(cities)} weighing {n3}.{rng.randint(1, 9)} kg. "
        if kind == 3:
            return f" Account {tag} raised reserve threshold from {n3} to {n5} credits. "
        if kind == 4:
            return f" Station {tag} logged serial {n7} at pressure {n3} hPa. "
        if kind == 5:
            return f" Dispatch {tag} assigned {rng.randint(2, 48)} technicians to depot {rng.choice(cities)}. "
        return f" Registry record {tag} lists code {n7} valid until {rng.randint(2027, 2036)}. "

    t0 = time.perf_counter()
    loss_hist: list[float] = []
    lslc_sys.salience_gate.train()
    eff_steps = min(steps, 16) if is_mock else steps
    eff_bs = min(batch_size, 4) if is_mock else batch_size
    pos_weight = torch.tensor([4.0], device=device, dtype=torch.float32)

    try:
        with torch.enable_grad():
            for _step in range(1, eff_steps + 1):
                batch_ids: list[list[int]] = []
                batch_masks: list[list[float]] = []
                for _b in range(eff_bs):
                    if replay_pages and rng.random() < mixed_natural_replay_ratio:
                        r_ids, r_mask = replay_pages[rng.randrange(len(replay_pages))]
                        batch_ids.append(list(r_ids))
                        batch_masks.append(list(r_mask))
                        continue
                    p_ids = _sample_background_page(rng, page_size, train_pool, filler_vocab)
                    mask = [0.0] * page_size
                    n_ins = rng.randint(1, 3)
                    occupied: list[tuple[int, int]] = []
                    for _k in range(n_ins):
                        clause = _make_disjoint_clause()
                        c_toks = _encode_text_or_mock(clause, tokenizer, vocab_size)
                        if len(c_toks) >= page_size - 16:
                            continue
                        for _try in range(8):
                            off = rng.randint(8, page_size - len(c_toks) - 8)
                            if all(off + len(c_toks) + 4 <= s or off >= e + 4 for s, e in occupied):
                                p_ids[off : off + len(c_toks)] = c_toks
                                for j in range(off, off + len(c_toks)):
                                    mask[j] = 1.0
                                occupied.append((off, off + len(c_toks)))
                                break
                    batch_ids.append(p_ids)
                    batch_masks.append(mask)

                ids_t = torch.tensor(batch_ids, dtype=torch.long, device=device)
                y_t = torch.tensor(batch_masks, dtype=torch.float32, device=device)
                with torch.no_grad():
                    emb_t = embed_fn(ids_t).detach()
                logits = lslc_sys.salience_gate.score_logits(emb_t, token_ids=ids_t, vocab_size=vocab_size).float()
                loss_bce = F.binary_cross_entropy_with_logits(logits, y_t, pos_weight=pos_weight)

                # Align 16-token sliding window scores directly with span_len=16 greedy selection
                s_w = min(span_len, page_size)
                win_mean = F.avg_pool1d(logits.unsqueeze(1), kernel_size=s_w, stride=1).squeeze(1)
                win_max = F.max_pool1d(logits.unsqueeze(1), kernel_size=s_w, stride=1).squeeze(1)
                win_score = 0.7 * win_mean + 0.3 * win_max
                win_y = F.avg_pool1d(y_t.unsqueeze(1), kernel_size=s_w, stride=1).squeeze(1)
                pos_w_mask = win_y >= 0.45
                neg_w_mask = win_y <= 0.01
                loss_rank = torch.zeros((), device=device, dtype=torch.float32)
                n_rank = 0
                for b_i in range(eff_bs):
                    if bool(pos_w_mask[b_i].any()) and bool(neg_w_mask[b_i].any()):
                        best_pos = win_score[b_i][pos_w_mask[b_i]].mean()
                        hard_neg = win_score[b_i][neg_w_mask[b_i]].amax()
                        loss_rank = loss_rank + F.relu(2.5 - (best_pos - hard_neg))
                        n_rank += 1
                if n_rank > 0:
                    loss_rank = loss_rank / float(n_rank)

                loss = loss_bce + 1.0 * loss_rank
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sal_params, 1.0)
                optimizer.step()
                scheduler.step()
                loss_hist.append(float(loss.detach().item()))
    finally:
        for p, prev_req in saved_reqs:
            p.requires_grad_(prev_req)
        lslc_sys.eval()

    max_frozen_diff = 0.0
    for name, p in lslc_sys.named_parameters():
        if name in frozen_snap:
            diff = float((p.detach().cpu() - frozen_snap[name]).abs().max().item())
            if diff > max_frozen_diff:
                max_frozen_diff = diff
    assert max_frozen_diff == 0.0, f"Frozen parameters mutated during SalienceGate fine-tune: {max_frozen_diff}"

    win_n = max(1, min(20, len(loss_hist) // 4))
    return {
        "steps": eff_steps,
        "batch_size": eff_bs,
        "lr": lr,
        "span_len": span_len,
        "mixed_natural_replay_ratio": mixed_natural_replay_ratio,
        "num_replay_pages": len(replay_pages),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "initial_loss_mean": round(sum(loss_hist[:win_n]) / win_n, 4),
        "final_loss_mean": round(sum(loss_hist[-win_n:]) / win_n, 4),
        "trainable_salience_params": sum(p.numel() for p in sal_params),
        "max_frozen_param_diff": max_frozen_diff,
    }


def _build_wikitext_selfsup_qa_episodes(
    train_pool: list[int],
    tokenizer,
    vocab_size: int,
    *,
    num_episodes: int = 48,
    num_pages: int = 8,
    page_size: int = 256,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Build 100% disjoint WikiText-train self-supervised factual QA episodes (ZERO RULER / LongMemEval wording)."""
    rng = random.Random(seed + 8383)
    filler_vocab = train_pool if train_pool else list(range(2, max(16, vocab_size - 2)))
    q_templates = (
        'According to the passage, what exact words immediately follow "{cue}"?',
        'In the text provided above, what follows "{cue}"?',
        'Complete the exact next phrase after "{cue}" based strictly on the passage.',
    )
    episodes: list[dict[str, Any]] = []
    for idx in range(num_episodes):
        pages = [_sample_background_page(rng, page_size, train_pool, filler_vocab) for _ in range(num_pages)]
        tgt_p = rng.randrange(num_pages)
        txt = tokenizer.decode(pages[tgt_p], skip_special_tokens=True)
        words = [w for w in txt.split() if w not in ("=", "==", "===", "@-@", "@,@", "@.@")]
        cue_str = ""
        ans_str = ""
        if len(words) >= 12:
            cands: list[tuple[str, str]] = []
            for j in range(5, len(words) - 3):
                ans_w = words[j]
                if any(ch.isdigit() for ch in ans_w) or (len(ans_w) >= 4 and ans_w[0].isupper()):
                    span_len_a = rng.randint(1, min(3, len(words) - j))
                    cue_c = " ".join(words[j - 5 : j]).strip()
                    ans_c = " ".join(words[j : j + span_len_a]).strip(" ,.;:!?\"'()")
                    if len(cue_c) >= 10 and len(ans_c) >= 2:
                        cands.append((cue_c, ans_c))
            if cands:
                cue_str, ans_str = rng.choice(cands)
        if not cue_str or not ans_str:
            mid = max(5, min(len(words) - 3, len(words) // 2))
            cue_str = " ".join(words[max(0, mid - 5) : mid]).strip() or "the passage"
            ans_str = " ".join(words[mid : mid + 2]).strip(" ,.;:!?\"'()") or "value"
        q_txt = q_templates[idx % len(q_templates)].format(cue=cue_str)
        episodes.append(
            {
                "pages": torch.tensor(pages, dtype=torch.long),
                "target_page": tgt_p,
                "question": q_txt,
                "router_query": cue_str,
                "answer": ans_str,
                "source": "wikitext_train_selfsup",
            }
        )
    return episodes


def _align_projector_chatml_qa(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    train_pool: list[int],
    device: torch.device,
    *,
    bge_toc: Any | None = None,
    steps: int = 120,
    lr: float = 1e-4,
    lslc_k_deep: int = 4,
    lslc_k_skim: int = 2,
    chapter_size: int = 16,
    salience_route_weight: float = 1.5,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Align ONLY SpatialLatentProjector on 50% WikiText-train self-sup QA + 50% Stage-1 mixed_natural train under identical ChatML QA prompt."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    eff_span_len = int(getattr(wrapper.cfg, "salience_span_len", 1))
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    stop_tail = im_end_ids if im_end_ids else ([int(eos_id)] if eos_id is not None else [])

    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        user_open_embeds = embed_fn(user_open_ids).to(llm_dtype).detach()

    # Build 50/50 disjoint training cache (ZERO RULER / ZERO LongMemEval data!)
    n_each = 12 if is_mock else 48
    wiki_raw = _build_wikitext_selfsup_qa_episodes(
        train_pool, tokenizer, vocab_size, num_episodes=n_each, num_pages=8, page_size=wrapper.cfg.page_size, seed=seed
    )
    mn_raw: list[dict[str, Any]] = []
    for mode_m, sd_off in (("budget", 7101), ("symbolic", 7207)):
        ds_m = AdversarialLivingBookDataset(
            num_episodes=n_each // 2,
            num_pages=8,
            page_size=wrapper.cfg.page_size,
            num_edits=1,
            mode=mode_m,
            split="train",
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            seed=seed + sd_off,
            background_token_pool=train_pool,
            style="mixed_natural",
        )
        for ep in ds_m:
            clean_q = tokenizer.decode(ep.metadata["router_query_ids"].tolist(), skip_special_tokens=True).strip()
            mn_raw.append(
                {
                    "pages": ep.metadata["verified_final_pages"],
                    "target_page": int(ep.target_page_idx),
                    "question": clean_q,
                    "router_query": clean_q,
                    "answer": str(ep.metadata["answer_str"]).strip(),
                    "source": f"mixed_natural_{mode_m}",
                }
            )

    wrapper.eval()
    wiki_cache: list[dict[str, Any]] = []
    mn_cache: list[dict[str, Any]] = []
    with torch.no_grad():
        for raw_list, out_cache in ((wiki_raw, wiki_cache), (mn_raw, mn_cache)):
            for ep in raw_list:
                p_t = ep["pages"].unsqueeze(0).to(device)
                z_book, bank, _ = _encode_book_chunked(
                    wrapper.lslc,
                    p_t,
                    embed_fn,
                    vocab_size,
                    device,
                    amp_dtype,
                    autocast_enabled,
                    chunk_size=16,
                    special_token_ids=special_ids,
                    salience_span_len=eff_span_len,
                )
                ext_s = None
                if bge_toc is not None:
                    bge_p4 = bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=128)
                    ext_s = bge_toc.score_book(bge_toc.encode_texts([ep["router_query"]], is_query=True), bge_p4).detach()
                rq_ids = torch.tensor(
                    _encode_text_or_mock(ep["router_query"], tokenizer, vocab_size), dtype=torch.long, device=device
                ).unsqueeze(0)
                q_ids = torch.tensor(
                    tokenizer.encode(_format_qa_prompt(ep["question"]), add_special_tokens=False),
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                ans_toks = _encode_text_or_mock(ep["answer"], tokenizer, vocab_size) + list(stop_tail)
                if not ans_toks:
                    ans_toks = [2]
                t_ids = torch.tensor(ans_toks, dtype=torch.long, device=device).unsqueeze(0)
                out_cache.append(
                    {
                        "z_book": z_book.detach(),
                        "exact_bank": bank,
                        "rq_ids": rq_ids,
                        "q_ids": q_ids,
                        "t_ids": t_ids,
                        "target_page": int(ep["target_page"]),
                        "ext_logits": ext_s,
                    }
                )

    # Freeze EVERYTHING except wrapper.lslc.projector (LLM r=0, Encoder, SalienceGate, Router, DeltaUpdater frozen!)
    proj_ids = {id(p) for p in wrapper.lslc.projector.parameters()}
    saved_reqs: list[tuple[torch.nn.Parameter, bool]] = []
    frozen_snap: dict[str, torch.Tensor] = {}
    for name, p in wrapper.named_parameters():
        saved_reqs.append((p, p.requires_grad))
        is_proj = id(p) in proj_ids
        p.requires_grad_(is_proj)
        if not is_proj:
            frozen_snap[name] = p.detach().cpu().clone()

    proj_params = [p for p in wrapper.lslc.projector.parameters() if p.requires_grad]
    non_proj = [n for n, p in wrapper.named_parameters() if p.requires_grad and not n.startswith("lslc.projector.")]
    assert not non_proj, f"Unexpected trainable params in ChatML projector alignment: {non_proj}"

    prev_use_cache = getattr(wrapper.llm.config, "use_cache", False) if hasattr(wrapper.llm, "config") else False
    if hasattr(wrapper.llm, "config"):
        wrapper.llm.config.use_cache = False
    wrapper.llm.eval()
    wrapper.lslc.eval()
    wrapper.lslc.projector.train()

    eff_steps = min(steps, 10) if is_mock else steps
    optimizer = torch.optim.AdamW(proj_params, lr=lr, weight_decay=wrapper.cfg.weight_decay, betas=wrapper.cfg.betas)
    warmup = max(1, int(eff_steps * 0.08))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: float(s + 1) / warmup
        if s < warmup
        else max(0.1, 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(1, eff_steps - warmup)))),
    )

    rng = random.Random(seed + 9191)
    t0 = time.perf_counter()
    wiki_losses: list[float] = []
    mn_losses: list[float] = []

    def _forward_chatml_ce(entry: dict[str, Any]) -> torch.Tensor:
        z_b = entry["z_book"]
        num_p = z_b.shape[1]
        kd_use = min(lslc_k_deep, num_p)
        ks_use = min(lslc_k_skim, max(0, num_p - kd_use))
        tgt_p = int(entry["target_page"])
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            toc = wrapper.lslc.projector.project_toc(z_b)
            q_emb = embed_fn(entry["rq_ids"]).mean(dim=1)
            r_out, asm = wrapper.lslc.route_and_assemble(
                z_book=z_b,
                toc_embeds=toc,
                query_embed=q_emb,
                exact_bank=entry["exact_bank"],
                embed_tokens_fn=embed_fn,
                k_deep=kd_use,
                k_skim=ks_use,
                mode="full",
                training_ste=False,
                hierarchical_chapter_size=chapter_size,
                query_input_ids=entry["rq_ids"],
                salience_route_weight=salience_route_weight,
                exact_bank_attenuation="none",
                external_dense_logits=entry["ext_logits"],
            )
            # Guarantee target page is present in deep slots during projector alignment (teacher-forced deep set if missed)
            if tgt_p not in r_out.deep_idx[0].tolist():
                deep_list = [p for p in r_out.deep_idx[0].tolist() if p != tgt_p][-(kd_use - 1) :] + [tgt_p]
                rem = [p for p in range(num_p) if p not in set(deep_list)]
                skim_list = rem[:ks_use]
                toc_list = rem[ks_use:]
                r_override = RouterOutput(
                    logits=r_out.logits,
                    probs=r_out.probs,
                    deep_idx=torch.tensor([deep_list], dtype=torch.long, device=device),
                    skim_idx=torch.tensor([skim_list], dtype=torch.long, device=device),
                    toc_idx=torch.tensor([toc_list], dtype=torch.long, device=device),
                )
                _, asm = wrapper.lslc.route_and_assemble(
                    z_book=z_b,
                    toc_embeds=toc,
                    query_embed=q_emb,
                    exact_bank=entry["exact_bank"],
                    embed_tokens_fn=embed_fn,
                    k_deep=kd_use,
                    k_skim=ks_use,
                    mode="full",
                    training_ste=False,
                    hierarchical_chapter_size=chapter_size,
                    router_out_override=r_override,
                    exact_bank_attenuation="none",
                )
            q_emb_seq = embed_fn(entry["q_ids"]).to(llm_dtype)
            t_emb_seq = embed_fn(entry["t_ids"]).to(llm_dtype)
            inp_e = torch.cat((user_open_embeds, asm.prefix_embeds.to(llm_dtype), q_emb_seq, t_emb_seq), dim=1)
            attn_m = torch.ones((1, inp_e.shape[1]), device=device, dtype=torch.long)
            logits = wrapper.llm(inputs_embeds=inp_e, attention_mask=attn_m).logits
            s_start = user_open_embeds.shape[1] + asm.prefix_embeds.shape[1] + q_emb_seq.shape[1] - 1
            pred = logits[:, s_start : s_start + entry["t_ids"].shape[1], :].float()
            return F.cross_entropy(pred.reshape(-1, pred.shape[-1]), entry["t_ids"].reshape(-1))

    try:
        with torch.enable_grad():
            for _step in range(1, eff_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                l_wiki = _forward_chatml_ce(wiki_cache[rng.randrange(len(wiki_cache))])
                (0.5 * l_wiki).backward()
                l_mn = _forward_chatml_ce(mn_cache[rng.randrange(len(mn_cache))])
                (0.5 * l_mn).backward()
                torch.nn.utils.clip_grad_norm_(proj_params, 1.0)
                optimizer.step()
                scheduler.step()
                wiki_losses.append(float(l_wiki.detach().item()))
                mn_losses.append(float(l_mn.detach().item()))
    finally:
        if hasattr(wrapper.llm, "config"):
            wrapper.llm.config.use_cache = prev_use_cache
        for p, prev_req in saved_reqs:
            p.requires_grad_(prev_req)
        wrapper.eval()

    max_frozen_diff = 0.0
    for name, p in wrapper.named_parameters():
        if name in frozen_snap:
            diff = float((p.detach().cpu() - frozen_snap[name]).abs().max().item())
            if diff > max_frozen_diff:
                max_frozen_diff = diff
    assert max_frozen_diff == 0.0, f"Frozen parameters mutated during ChatML projector alignment: {max_frozen_diff}"

    win_n = max(1, min(16, len(wiki_losses) // 4))
    return {
        "steps": eff_steps,
        "lr": lr,
        "num_wikitext_items": len(wiki_cache),
        "num_mixed_natural_items": len(mn_cache),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "wiki_initial_loss": round(sum(wiki_losses[:win_n]) / win_n, 4),
        "wiki_final_loss": round(sum(wiki_losses[-win_n:]) / win_n, 4),
        "mixed_natural_initial_loss": round(sum(mn_losses[:win_n]) / win_n, 4),
        "mixed_natural_final_loss": round(sum(mn_losses[-win_n:]) / win_n, 4),
        "trainable_projector_params": sum(p.numel() for p in proj_params),
        "max_frozen_param_diff": max_frozen_diff,
    }


@torch.no_grad()
def evaluate_bge_pilar4a_final(
    wrapper: FrozenQwenLSLCWrapper,
    bge_toc: _PlugInBiEncoderTOC,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    eval_n: int = 72,
    salience_route_weight: float = 1.5,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Evaluate Pilar 4A Final: BM25 CPU vs BGE-4span Dense (w=0) vs Adaptive Hybrid BGE-4span+SparseExactBank + Router Stress."""
    embed_fn = wrapper.llm.get_input_embeddings()
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    eff_span_len = int(getattr(wrapper.cfg, "salience_span_len", 16))
    wrapper.eval()

    p4_eval_n = min(eval_n, 4) if is_mock else eval_n
    hybrid_key = f"lslc_bge_hybrid_w{salience_route_weight:g}"
    zero_overlap_eval: dict[str, Any] = {}

    for p_cnt in (16, 64):
        ds_eval = ZeroOverlapParaphraseRoutingBenchmark(
            num_episodes=p4_eval_n,
            num_pages=p_cnt,
            page_size=wrapper.cfg.page_size,
            split="eval",
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            seed=seed + p_cnt * 13,
            background_token_pool=val_pool,
        )
        for ep in ds_eval:
            assert content_words(ep.metadata["query_text"]) & content_words(ep.metadata["target_page_text"]) == set()

        stats_b: dict[str, dict[str, float]] = {
            "bm25_cpu": {"top1": 0.0, "top2": 0.0, "ms": 0.0},
            "bge_dense_1span_w0": {"top1": 0.0, "top2": 0.0, "ms": 0.0},
            "bge_dense_4span_w0": {"top1": 0.0, "top2": 0.0, "ms": 0.0},
            hybrid_key: {"top1": 0.0, "top2": 0.0, "ms": 0.0},
        }
        _reset_peak_vram(device)

        for ep in ds_eval:
            tgt = int(ep.target_page_idx)
            q_txt = str(ep.metadata["query_text"])
            page_texts = list(ep.metadata["page_texts"])

            t0 = time.perf_counter()
            rr_bm25 = BM25PageRetriever(page_texts).route(q_txt, k_deep=2, k_skim=0, device=device)
            stats_b["bm25_cpu"]["ms"] += (time.perf_counter() - t0) * 1000.0
            if int(torch.argmax(rr_bm25.logits[0], dim=-1).item()) == tgt:
                stats_b["bm25_cpu"]["top1"] += 1.0
            if tgt in rr_bm25.deep_idx[0].tolist():
                stats_b["bm25_cpu"]["top2"] += 1.0

            p_t = ep.context_pages.unsqueeze(0).to(device)
            z_book, bank, toc = _encode_book_chunked(
                wrapper.lslc,
                p_t,
                embed_fn,
                vocab_size,
                device,
                amp_dtype,
                autocast_enabled,
                chunk_size=32,
                special_token_ids=special_ids,
                salience_span_len=eff_span_len,
            )
            bge_p1 = bge_toc.encode_book_pages(None, page_texts, tokenizer, num_spans=1, batch_size=128)
            bge_p4 = bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=128)
            rq_ids = ep.metadata["router_query_ids"].unsqueeze(0).to(device)
            q_emb = embed_fn(rq_ids).mean(dim=1)

            for arm_name, p_vecs, sw in (
                ("bge_dense_1span_w0", bge_p1, 0.0),
                ("bge_dense_4span_w0", bge_p4, 0.0),
                (hybrid_key, bge_p4, salience_route_weight),
            ):
                _sync_device(device)
                tq = time.perf_counter()
                q_bge = bge_toc.encode_texts([q_txt], is_query=True)
                ext_s = bge_toc.score_book(q_bge, p_vecs)
                r_out, _ = wrapper.lslc.route_and_assemble(
                    z_book,
                    toc,
                    q_emb,
                    bank,
                    embed_fn,
                    k_deep=2,
                    k_skim=min(2, max(0, p_cnt - 2)),
                    mode="toc_only",
                    training_ste=False,
                    query_input_ids=rq_ids,
                    salience_route_weight=sw,
                    external_dense_logits=ext_s,
                )
                _sync_device(device)
                stats_b[arm_name]["ms"] += (time.perf_counter() - tq) * 1000.0
                if int(torch.argmax(r_out.logits[0], dim=-1).item()) == tgt:
                    stats_b[arm_name]["top1"] += 1.0
                if tgt in r_out.deep_idx[0].tolist():
                    stats_b[arm_name]["top2"] += 1.0

        d = float(max(1, len(ds_eval)))
        n_ep = len(ds_eval)
        zero_overlap_eval[f"P={p_cnt}"] = {
            "num_episodes": n_ep,
            "num_pages": p_cnt,
            "peak_vram_mb": _peak_vram_mb(device),
            **{
                arm: {
                    "top1_recall": round(vals["top1"] / d, 4),
                    "top1_ci95": _wilson_ci95(vals["top1"] / d, n_ep),
                    "top2_recall": round(vals["top2"] / d, 4),
                    "top2_ci95": _wilson_ci95(vals["top2"] / d, n_ep),
                    "per_query_ms": round(vals["ms"] / d, 3),
                }
                for arm, vals in stats_b.items()
            },
        }
        print(
            f"[PILAR 4A FINAL P={p_cnt}] bm25={zero_overlap_eval[f'P={p_cnt}']['bm25_cpu']['top1_recall']:.3f} | "
            f"bge1={zero_overlap_eval[f'P={p_cnt}']['bge_dense_1span_w0']['top1_recall']:.3f} | "
            f"bge4={zero_overlap_eval[f'P={p_cnt}']['bge_dense_4span_w0']['top1_recall']:.3f} | "
            f"{hybrid_key}={zero_overlap_eval[f'P={p_cnt}'][hybrid_key]['top1_recall']:.3f}",
            flush=True,
        )

    # Router stress P=64 & P=128: BGE-4span dense w=0 vs Adaptive Hybrid w=1.5 vs Native Hybrid w=1.5
    router_stress: dict[str, Any] = {}
    for label, num_p, n_entities in (
        ("router_stress_p64", 16 if is_mock else 64, 8 if is_mock else 12),
        ("router_stress_p128", 32 if is_mock else 128, 8 if is_mock else 20),
    ):
        rng_s = random.Random(seed + 700 + num_p)
        chosen_pages = rng_s.sample(range(num_p), n_entities)
        filler_vocab = val_pool if val_pool else list(range(2, max(16, vocab_size - 2)))
        pages = [_sample_background_page(rng_s, wrapper.cfg.page_size, val_pool, filler_vocab) for _ in range(num_p)]
        queries: list[tuple[int, str]] = []
        for idx_e, p_idx in enumerate(chosen_pages):
            proj = f"StressNode-{idx_e:02d}"
            val = f"stress-tier-{idx_e:02d}"
            fmt = "record" if (idx_e % 2 == 0) else ("eval_prose" if idx_e % 4 == 1 else "eval_code")
            stmt = AdversarialLivingBookDataset._format_fact_stmt("symbolic", fmt=fmt, proj=proj, val=val)
            toks = _encode_text_or_mock(stmt, tokenizer, vocab_size)
            offset = rng_s.randint(0, max(0, wrapper.cfg.page_size - len(toks)))
            n_w = min(len(toks), wrapper.cfg.page_size - offset)
            pages[p_idx][offset : offset + n_w] = toks[:n_w]
            queries.append((p_idx, f"What is the approved deployment state for {proj}?"))

        page_tensor = torch.tensor(pages, dtype=torch.long, device=device).unsqueeze(0)
        z_book, bank, toc = _encode_book_chunked(
            wrapper.lslc,
            page_tensor,
            embed_fn,
            vocab_size,
            device,
            amp_dtype,
            autocast_enabled,
            chunk_size=32,
            special_token_ids=special_ids,
            salience_span_len=eff_span_len,
        )
        bge_p4 = bge_toc.encode_book_pages(page_tensor, None, tokenizer, num_spans=4, batch_size=128)
        w0_t1, w0_t2, wh_t1, wh_t2, nat_t1, nat_t2 = 0, 0, 0, 0, 0, 0
        for p_idx, q_txt in queries:
            rq_ids = torch.tensor(_encode_text_or_mock(q_txt, tokenizer, vocab_size), dtype=torch.long, device=device).unsqueeze(0)
            q_bge = bge_toc.encode_texts([q_txt], is_query=True)
            s4 = bge_toc.score_book(q_bge, bge_p4)
            q_emb = embed_fn(rq_ids).mean(dim=1)
            r0, _ = wrapper.lslc.route_and_assemble(
                z_book, toc, q_emb, bank, embed_fn, mode="toc_only", training_ste=False,
                query_input_ids=rq_ids, salience_route_weight=0.0, external_dense_logits=s4,
            )
            rh, _ = wrapper.lslc.route_and_assemble(
                z_book, toc, q_emb, bank, embed_fn, mode="toc_only", training_ste=False,
                query_input_ids=rq_ids, salience_route_weight=salience_route_weight, external_dense_logits=s4,
            )
            rn, _ = wrapper.lslc.route_and_assemble(
                z_book, toc, q_emb, bank, embed_fn, mode="toc_only", training_ste=False,
                query_input_ids=rq_ids, salience_route_weight=salience_route_weight,
            )
            if int(torch.argmax(r0.logits[0], dim=-1).item()) == p_idx:
                w0_t1 += 1
            if p_idx in r0.deep_idx[0].tolist():
                w0_t2 += 1
            if int(torch.argmax(rh.logits[0], dim=-1).item()) == p_idx:
                wh_t1 += 1
            if p_idx in rh.deep_idx[0].tolist():
                wh_t2 += 1
            if int(torch.argmax(rn.logits[0], dim=-1).item()) == p_idx:
                nat_t1 += 1
            if p_idx in rn.deep_idx[0].tolist():
                nat_t2 += 1
        d = max(1, len(queries))
        router_stress[label] = {
            "num_pages": num_p,
            "num_entity_pages": n_entities,
            "bge_dense_w0_top1_recall": round(w0_t1 / d, 4),
            "bge_dense_w0_top2_recall": round(w0_t2 / d, 4),
            f"{hybrid_key}_top1_recall": round(wh_t1 / d, 4),
            f"{hybrid_key}_top2_recall": round(wh_t2 / d, 4),
            "lslc_native_hybrid_top1_recall": round(nat_t1 / d, 4),
            "lslc_native_hybrid_top2_recall": round(nat_t2 / d, 4),
        }
        print(f"[PILAR 4A FINAL Stress {label}] {router_stress[label]}", flush=True)

    return {
        "bge_model": bge_toc.model_name,
        "salience_span_len": eff_span_len,
        "k_exact": int(wrapper.cfg.k_exact),
        "salience_route_weight": float(salience_route_weight),
        "zero_overlap_eval": zero_overlap_eval,
        "router_stress": router_stress,
        "prior_qwen3_only_4span_control": {
            "source_artifact": "evals/m2_colab_l4/v3_gerbang4_4span_and_spansalience.json",
            "P=16": {"top1_recall": 0.1806, "top2_recall": 0.2500},
            "P=64": {"top1_recall": 0.0972, "top2_recall": 0.1667},
        },
    }


@torch.no_grad()
def run_span16_bge_probe(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    train_pool: list[int],
    val_pool: list[int],
    lme_items: list[dict[str, Any]],
    lme_meta: dict[str, Any],
    device: torch.device,
    *,
    g2_checkpoint: str = "",
    save_salience_checkpoint: str = "",
    bge_model_name: str = "BAAI/bge-small-en-v1.5",
    pilar4a_eval_n: int = 72,
    episodes_per_task: int = 4,
    mk_distractors: int = 48,
    lslc_k_deep: int = 4,
    lslc_k_skim: int = 2,
    chapter_size: int = 16,
    max_pages: int = 64,
    max_new_tokens: int = 32,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Combined final lock-in probe: (1) RULER 4K span16_k48 + conditional Jalur A, (2) BGE-small 2x2 table, (3) LongMemEval N=6 3-mode sanity."""
    import copy

    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    special_set = set(special_ids)
    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)

    cfg_g2 = copy.deepcopy(wrapper.cfg)
    cfg_g2.use_contextual_query_encoding = False
    cfg_g2.contextual_query_layer = None
    cfg_g2.contextual_page_spans = 1
    cfg_g2.k_exact = min(48, cfg_g2.page_size)
    cfg_g2.salience_span_len = 16
    g2_sys = LSLCSystem(cfg_g2).to(device).eval()
    g2_info = _load_lslc_checkpoint(g2_sys, g2_checkpoint) if g2_checkpoint else {"path": None, "note": "mock_init"}

    ctx_len = 1024 if is_mock else 4096
    diag_tasks = ("s_niah", "mk_niah")
    eps_by_task = {
        t_name: build_ruler_episodes(
            task_name=t_name,
            context_length=ctx_len,
            num_episodes=episodes_per_task,
            tokenizer=tokenizer,
            val_pool=val_pool,
            page_size=cfg_g2.page_size,
            seed=seed,
            mk_distractors=mk_distractors,
            is_mock=is_mock,
        )
        for t_name in diag_tasks
    }

    def _eval_ruler_span16(sys_mod: LSLCSystem, label_tag: str) -> dict[str, Any]:
        audit_by_task: dict[str, Any] = {}
        all_audit: list[dict[str, Any]] = []
        enc_cache: dict[str, list[tuple[dict[str, Any], torch.Tensor, SparseExactBank, torch.Tensor, float]]] = {}
        for t_name in diag_tasks:
            t_a: list[dict[str, Any]] = []
            examples: list[dict[str, Any]] = []
            enc_l = []
            for ep in eps_by_task[t_name]:
                p_t = ep["pages"].unsqueeze(0).to(device)
                _sync_device(device)
                t_m0 = time.perf_counter()
                z_book, bank, toc = _encode_book_chunked(
                    sys_mod,
                    p_t,
                    embed_fn,
                    vocab_size,
                    device,
                    amp_dtype,
                    autocast_enabled,
                    chunk_size=32,
                    special_token_ids=special_ids,
                    salience_span_len=16,
                )
                _sync_device(device)
                mem_ms = (time.perf_counter() - t_m0) * 1000.0
                enc_l.append((ep, z_book, bank, toc, mem_ms))

                gold_p = int(ep["gold_pages"][0])
                bank_toks = [int(t) for t in bank.token_ids[0, gold_p].tolist() if int(t) >= 0]
                sp_cnt = sum(1 for t in bank_toks if t in special_set)
                bank_decoded = tokenizer.decode(bank_toks, skip_special_tokens=True)
                ans = str(ep["answer"])
                ans_toks = [int(t) for t in _encode_text_or_mock(ans, tokenizer, vocab_size)]
                c_ans = Counter(ans_toks)
                c_bank = Counter(bank_toks)
                contiguous_ok = 1.0 if ans in bank_decoded else 0.0
                multiset_ok = 1.0 if (ans_toks and all(c_bank[k] >= v for k, v in c_ans.items())) else 0.0
                tok_rec = sum(min(c_bank[k], v) for k, v in c_ans.items()) / max(1, len(ans_toks))
                key_str = str(ep["metadata"].get("key") or ep["metadata"].get("target_key") or "")
                key_ok = 1.0 if (not key_str or key_str.lower() in bank_decoded.lower()) else 0.0
                rec_a = {
                    "all_7_digits_contiguous": contiguous_ok,
                    "all_7_digits_multiset": multiset_ok,
                    "digit_token_recall": tok_rec,
                    "key_in_bank": key_ok,
                    "special_tokens_in_bank": sp_cnt,
                }
                t_a.append(rec_a)
                all_audit.append(rec_a)
                if len(examples) < 2:
                    examples.append(
                        {
                            "gold": ans,
                            "key": key_str,
                            "all_7_digits_contiguous": contiguous_ok,
                            "digit_token_recall": round(tok_rec, 4),
                            "bank_decoded": bank_decoded,
                        }
                    )
            enc_cache[t_name] = enc_l
            audit_by_task[t_name] = {
                "num_episodes": len(t_a),
                "all_7_digits_contiguous_rate": round(sum(r["all_7_digits_contiguous"] for r in t_a) / max(1, len(t_a)), 4),
                "all_7_digits_multiset_rate": round(sum(r["all_7_digits_multiset"] for r in t_a) / max(1, len(t_a)), 4),
                "digit_token_recall_mean": round(sum(r["digit_token_recall"] for r in t_a) / max(1, len(t_a)), 4),
                "key_in_bank_rate": round(sum(r["key_in_bank"] for r in t_a) / max(1, len(t_a)), 4),
                "special_tokens_in_bank_total": int(sum(r["special_tokens_in_bank"] for r in t_a)),
                "examples": examples,
            }
        ov_audit = {
            "num_episodes": len(all_audit),
            "all_7_digits_contiguous_rate": round(sum(r["all_7_digits_contiguous"] for r in all_audit) / max(1, len(all_audit)), 4),
            "all_7_digits_multiset_rate": round(sum(r["all_7_digits_multiset"] for r in all_audit) / max(1, len(all_audit)), 4),
            "digit_token_recall_mean": round(sum(r["digit_token_recall"] for r in all_audit) / max(1, len(all_audit)), 4),
            "key_in_bank_rate": round(sum(r["key_in_bank"] for r in all_audit) / max(1, len(all_audit)), 4),
            "special_tokens_in_bank_total": int(sum(r["special_tokens_in_bank"] for r in all_audit)),
        }

        modes_out: dict[str, Any] = {}
        for mode_label, asm_mode, use_chatml in (
            ("plain_stage1_full", "full", False),
            ("chatml_exact_only", "exact_only", True),
        ):
            by_task_m: dict[str, Any] = {}
            all_rows_m: list[dict[str, Any]] = []
            for t_name in diag_tasks:
                t_rows_m: list[dict[str, Any]] = []
                raw_samples: list[dict[str, Any]] = []
                for ep, z_book, bank, toc, mem_ms in enc_cache[t_name]:
                    _reset_peak_vram(device)
                    gold_set = set(ep["gold_pages"])
                    rq_ids = torch.tensor(
                        _encode_text_or_mock(ep["router_query"], tokenizer, vocab_size), dtype=torch.long, device=device
                    ).unsqueeze(0)
                    q_str = _format_qa_prompt(ep["question"]) if use_chatml else f"\nQuestion: {ep['question'].strip()}\nAnswer:"
                    u_open = user_open_ids if use_chatml else None
                    p_prompt = torch.tensor(tokenizer.encode(q_str, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)

                    _sync_device(device)
                    t_q0 = time.perf_counter()
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        q_emb = embed_fn(rq_ids).mean(dim=1)
                        kd_use = min(lslc_k_deep, z_book.shape[1])
                        ks_use = min(lslc_k_skim, max(0, z_book.shape[1] - kd_use))
                        r_out, asm = sys_mod.route_and_assemble(
                            z_book=z_book,
                            toc_embeds=toc,
                            query_embed=q_emb,
                            exact_bank=bank,
                            embed_tokens_fn=embed_fn,
                            k_deep=kd_use,
                            k_skim=ks_use,
                            mode=asm_mode,
                            training_ste=False,
                            hierarchical_chapter_size=chapter_size,
                            query_input_ids=rq_ids,
                            salience_route_weight=2.0,
                            exact_bank_attenuation="none",
                        )
                    chosen = r_out.deep_idx[0].tolist()
                    router_hit = 1.0 if gold_set.issubset(set(chosen)) else (len(gold_set & set(chosen)) / max(1, len(gold_set)))
                    gen_txt = _greedy_generate_qa(
                        wrapper.llm,
                        asm.prefix_embeds.to(llm_dtype),
                        p_prompt,
                        tokenizer,
                        max_new_tokens=max_new_tokens,
                        user_open_ids=u_open,
                    )
                    _sync_device(device)
                    q_ms = (time.perf_counter() - t_q0) * 1000.0
                    qm = compute_community_qa_metrics(gen_txt, ep["answer"])
                    t_rows_m.append(
                        {
                            **qm,
                            "router_hit_rate": router_hit,
                            "active_prefix_tokens": int(asm.prefix_embeds.shape[1]),
                            "memory_update_ms": round(mem_ms, 3),
                            "per_query_ms": round(q_ms, 3),
                            "peak_vram_mb": _peak_vram_mb(device),
                            "max_toc_diff": 0.0,
                        }
                    )
                    raw_samples.append({"question": ep["question"], "gold": ep["answer"], "pred": gen_txt, "router_hit": router_hit, **qm})
                agg_t = _aggregate_eval_rows(t_rows_m)
                agg_t["samples"] = raw_samples
                by_task_m[t_name] = agg_t
                all_rows_m.extend(t_rows_m)
            modes_out[mode_label] = {"overall": _aggregate_eval_rows(all_rows_m), "by_task": by_task_m}

        stress = evaluate_router_stress_regression(
            wrapper=wrapper,
            tokenizer=tokenizer,
            val_pool=val_pool,
            device=device,
            is_mock=is_mock,
            seed=seed + 700,
            lslc_override=sys_mod,
            special_token_ids=special_ids,
            salience_span_len=16,
            use_contextual_keys=False,
        )
        print(
            f"[RULER {label_tag}] bank_7d={ov_audit['all_7_digits_contiguous_rate']:.3f} "
            f"dig_rec={ov_audit['digit_token_recall_mean']:.3f} key={ov_audit['key_in_bank_rate']:.3f} | "
            f"plain_EM={modes_out['plain_stage1_full']['overall']['extracted_em']:.3f} "
            f"exact_EM={modes_out['chatml_exact_only']['overall']['extracted_em']:.3f} | "
            f"stress_p64={stress['router_stress_p64']['hybrid_w2_top1_recall']:.3f} "
            f"stress_p128={stress['router_stress_p128']['hybrid_w2_top1_recall']:.3f}",
            flush=True,
        )
        return {
            "exact_bank_audit": {"overall": ov_audit, "by_task": audit_by_task},
            "ruler_modes": modes_out,
            "router_stress": stress,
        }

    # --- PART 1: RULER 4K span_len=16, k_exact=48 (0-training -> Conditional Jalur A) ---
    zero_train_res = _eval_ruler_span16(g2_sys, "span16_k48_zero_train")
    jalur_a_triggered = zero_train_res["exact_bank_audit"]["overall"]["all_7_digits_contiguous_rate"] < 0.75
    jalur_a_train_stats: dict[str, Any] | None = None
    jalur_a_res: dict[str, Any] | None = None
    active_sys = g2_sys
    saved_sal_ckpt: str | None = None

    if jalur_a_triggered:
        ft_sys = copy.deepcopy(g2_sys).to(device)
        jalur_a_train_stats = _finetune_salience_gate_span16(
            ft_sys,
            embed_fn,
            tokenizer,
            train_pool,
            device,
            steps=150,
            batch_size=8,
            lr=1e-3,
            span_len=16,
            seed=seed,
            is_mock=is_mock,
        )
        print(f"[JALUR A SalienceGate FT] {jalur_a_train_stats}", flush=True)
        jalur_a_res = _eval_ruler_span16(ft_sys, "span16_k48_jalur_a_ft")
        stress_ok = True if is_mock else (
            jalur_a_res["router_stress"]["router_stress_p128"]["hybrid_w2_top1_recall"] >= 0.90
        )
        if (
            stress_ok
            and jalur_a_res["exact_bank_audit"]["overall"]["all_7_digits_contiguous_rate"]
            >= zero_train_res["exact_bank_audit"]["overall"]["all_7_digits_contiguous_rate"]
            and jalur_a_res["exact_bank_audit"]["overall"]["digit_token_recall_mean"]
            >= zero_train_res["exact_bank_audit"]["overall"]["digit_token_recall_mean"]
        ):
            active_sys = ft_sys
            if save_salience_checkpoint:
                _save_lslc_checkpoint(active_sys, save_salience_checkpoint)
                saved_sal_ckpt = save_salience_checkpoint

    # --- PART 2: Plug-in Semantic Bi-Encoder TOC (BAAI/bge-small-en-v1.5, 33M, 0 training, 0 KV tokens) ---
    bge = _PlugInBiEncoderTOC(bge_model_name, device=device, is_mock=is_mock)

    # 2a) ZeroOverlapParaphraseRoutingBenchmark (P=16 & P=64)
    p4_eval_n = min(pilar4a_eval_n, 4) if is_mock else pilar4a_eval_n
    bge_zero_overlap: dict[str, Any] = {}
    for p_cnt in (16, 64):
        ds_eval = ZeroOverlapParaphraseRoutingBenchmark(
            num_episodes=p4_eval_n,
            num_pages=p_cnt,
            page_size=active_sys.cfg.page_size,
            split="eval",
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            seed=seed + p_cnt * 13,
            background_token_pool=val_pool,
        )
        for ep in ds_eval:
            assert content_words(ep.metadata["query_text"]) & content_words(ep.metadata["target_page_text"]) == set()

        stats_b: dict[str, dict[str, float]] = {
            "bm25_cpu": {"top1": 0.0, "top2": 0.0, "ms": 0.0},
            "bge_dense_1span_w0": {"top1": 0.0, "top2": 0.0, "ms": 0.0},
            "bge_dense_4span_w0": {"top1": 0.0, "top2": 0.0, "ms": 0.0},
            "lslc_bge_hybrid_w2": {"top1": 0.0, "top2": 0.0, "ms": 0.0},
        }
        _reset_peak_vram(device)

        for ep in ds_eval:
            tgt = int(ep.target_page_idx)
            q_txt = str(ep.metadata["query_text"])
            page_texts = list(ep.metadata["page_texts"])

            # BM25 CPU
            t0 = time.perf_counter()
            rr_bm25 = BM25PageRetriever(page_texts).route(q_txt, k_deep=2, k_skim=0, device=device)
            stats_b["bm25_cpu"]["ms"] += (time.perf_counter() - t0) * 1000.0
            if int(torch.argmax(rr_bm25.logits[0], dim=-1).item()) == tgt:
                stats_b["bm25_cpu"]["top1"] += 1.0
            if tgt in rr_bm25.deep_idx[0].tolist():
                stats_b["bm25_cpu"]["top2"] += 1.0

            p_t = ep.context_pages.unsqueeze(0).to(device)
            z_book, bank, toc = _encode_book_chunked(
                active_sys,
                p_t,
                embed_fn,
                vocab_size,
                device,
                amp_dtype,
                autocast_enabled,
                chunk_size=32,
                special_token_ids=special_ids,
                salience_span_len=16,
            )
            bge_p1 = bge.encode_book_pages(None, page_texts, tokenizer, num_spans=1, batch_size=64)
            bge_p4 = bge.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=64)
            rq_ids = ep.metadata["router_query_ids"].unsqueeze(0).to(device)

            # BGE 1-span dense (w=0.0)
            _sync_device(device)
            t1 = time.perf_counter()
            q_bge = bge.encode_texts([q_txt], is_query=True)
            s1 = bge.score_book(q_bge, bge_p1)
            r_b1, _ = active_sys.route_and_assemble(
                z_book, toc, embed_fn(rq_ids).mean(dim=1), bank, embed_fn,
                k_deep=2, k_skim=min(2, max(0, p_cnt - 2)), mode="toc_only", training_ste=False,
                query_input_ids=rq_ids, salience_route_weight=0.0, external_dense_logits=s1,
            )
            _sync_device(device)
            stats_b["bge_dense_1span_w0"]["ms"] += (time.perf_counter() - t1) * 1000.0
            if int(torch.argmax(r_b1.logits[0], dim=-1).item()) == tgt:
                stats_b["bge_dense_1span_w0"]["top1"] += 1.0
            if tgt in r_b1.deep_idx[0].tolist():
                stats_b["bge_dense_1span_w0"]["top2"] += 1.0

            # BGE 4-span dense (w=0.0)
            _sync_device(device)
            t4 = time.perf_counter()
            q_bge = bge.encode_texts([q_txt], is_query=True)
            s4 = bge.score_book(q_bge, bge_p4)
            r_b4, _ = active_sys.route_and_assemble(
                z_book, toc, embed_fn(rq_ids).mean(dim=1), bank, embed_fn,
                k_deep=2, k_skim=min(2, max(0, p_cnt - 2)), mode="toc_only", training_ste=False,
                query_input_ids=rq_ids, salience_route_weight=0.0, external_dense_logits=s4,
            )
            _sync_device(device)
            stats_b["bge_dense_4span_w0"]["ms"] += (time.perf_counter() - t4) * 1000.0
            if int(torch.argmax(r_b4.logits[0], dim=-1).item()) == tgt:
                stats_b["bge_dense_4span_w0"]["top1"] += 1.0
            if tgt in r_b4.deep_idx[0].tolist():
                stats_b["bge_dense_4span_w0"]["top2"] += 1.0

            # LSLC BGE 4-span + SparseExactBank Hybrid (w=2.0)
            _sync_device(device)
            th = time.perf_counter()
            q_bge = bge.encode_texts([q_txt], is_query=True)
            s4_h = bge.score_book(q_bge, bge_p4)
            r_bh, _ = active_sys.route_and_assemble(
                z_book, toc, embed_fn(rq_ids).mean(dim=1), bank, embed_fn,
                k_deep=2, k_skim=min(2, max(0, p_cnt - 2)), mode="toc_only", training_ste=False,
                query_input_ids=rq_ids, salience_route_weight=2.0, external_dense_logits=s4_h,
            )
            _sync_device(device)
            stats_b["lslc_bge_hybrid_w2"]["ms"] += (time.perf_counter() - th) * 1000.0
            if int(torch.argmax(r_bh.logits[0], dim=-1).item()) == tgt:
                stats_b["lslc_bge_hybrid_w2"]["top1"] += 1.0
            if tgt in r_bh.deep_idx[0].tolist():
                stats_b["lslc_bge_hybrid_w2"]["top2"] += 1.0

        d = float(max(1, len(ds_eval)))
        n_ep = len(ds_eval)
        bge_zero_overlap[f"P={p_cnt}"] = {
            "num_episodes": n_ep,
            "num_pages": p_cnt,
            "peak_vram_mb": _peak_vram_mb(device),
            **{
                arm: {
                    "top1_recall": round(vals["top1"] / d, 4),
                    "top1_ci95": _wilson_ci95(vals["top1"] / d, n_ep),
                    "top2_recall": round(vals["top2"] / d, 4),
                    "top2_ci95": _wilson_ci95(vals["top2"] / d, n_ep),
                    "per_query_ms": round(vals["ms"] / d, 3),
                }
                for arm, vals in stats_b.items()
            },
        }
        print(
            f"[BGE ZeroOverlap P={p_cnt}] bm25={bge_zero_overlap[f'P={p_cnt}']['bm25_cpu']['top1_recall']:.3f} | "
            f"bge1={bge_zero_overlap[f'P={p_cnt}']['bge_dense_1span_w0']['top1_recall']:.3f} | "
            f"bge4={bge_zero_overlap[f'P={p_cnt}']['bge_dense_4span_w0']['top1_recall']:.3f} | "
            f"hybrid_w2={bge_zero_overlap[f'P={p_cnt}']['lslc_bge_hybrid_w2']['top1_recall']:.3f}",
            flush=True,
        )

    # 2b) router_stress_p64 & router_stress_p128: bge_dense_w0 vs lslc_bge_hybrid_w2 (span_len=16, k_exact=48)
    bge_stress: dict[str, Any] = {}
    for label, num_p, n_entities in (
        ("router_stress_p64", 16 if is_mock else 64, 8 if is_mock else 12),
        ("router_stress_p128", 32 if is_mock else 128, 8 if is_mock else 20),
    ):
        rng_s = random.Random(seed + 700 + num_p)
        chosen_pages = rng_s.sample(range(num_p), n_entities)
        filler_vocab = val_pool if val_pool else list(range(2, max(16, vocab_size - 2)))
        pages = [_sample_background_page(rng_s, active_sys.cfg.page_size, val_pool, filler_vocab) for _ in range(num_p)]
        queries: list[tuple[int, str]] = []
        for idx_e, p_idx in enumerate(chosen_pages):
            proj = f"StressNode-{idx_e:02d}"
            val = f"stress-tier-{idx_e:02d}"
            fmt = "record" if (idx_e % 2 == 0) else ("eval_prose" if idx_e % 4 == 1 else "eval_code")
            stmt = AdversarialLivingBookDataset._format_fact_stmt("symbolic", fmt=fmt, proj=proj, val=val)
            toks = _encode_text_or_mock(stmt, tokenizer, vocab_size)
            offset = rng_s.randint(0, max(0, active_sys.cfg.page_size - len(toks)))
            n_w = min(len(toks), active_sys.cfg.page_size - offset)
            pages[p_idx][offset : offset + n_w] = toks[:n_w]
            queries.append((p_idx, f"What is the approved deployment state for {proj}?"))

        page_tensor = torch.tensor(pages, dtype=torch.long, device=device).unsqueeze(0)
        z_book, bank, toc = _encode_book_chunked(
            active_sys, page_tensor, embed_fn, vocab_size, device, amp_dtype, autocast_enabled,
            chunk_size=32, special_token_ids=special_ids, salience_span_len=16,
        )
        bge_p4 = bge.encode_book_pages(page_tensor, None, tokenizer, num_spans=4, batch_size=64)
        w0_t1, w0_t2, w2_t1, w2_t2 = 0, 0, 0, 0
        for p_idx, q_txt in queries:
            rq_ids = torch.tensor(_encode_text_or_mock(q_txt, tokenizer, vocab_size), dtype=torch.long, device=device).unsqueeze(0)
            q_bge = bge.encode_texts([q_txt], is_query=True)
            s4 = bge.score_book(q_bge, bge_p4)
            q_emb = embed_fn(rq_ids).mean(dim=1)
            r0, _ = active_sys.route_and_assemble(
                z_book, toc, q_emb, bank, embed_fn, mode="toc_only", training_ste=False,
                query_input_ids=rq_ids, salience_route_weight=0.0, external_dense_logits=s4,
            )
            r2, _ = active_sys.route_and_assemble(
                z_book, toc, q_emb, bank, embed_fn, mode="toc_only", training_ste=False,
                query_input_ids=rq_ids, salience_route_weight=2.0, external_dense_logits=s4,
            )
            if int(torch.argmax(r0.logits[0], dim=-1).item()) == p_idx:
                w0_t1 += 1
            if p_idx in r0.deep_idx[0].tolist():
                w0_t2 += 1
            if int(torch.argmax(r2.logits[0], dim=-1).item()) == p_idx:
                w2_t1 += 1
            if p_idx in r2.deep_idx[0].tolist():
                w2_t2 += 1
        d = max(1, len(queries))
        bge_stress[label] = {
            "num_pages": num_p,
            "num_entity_pages": n_entities,
            "bge_dense_w0_top1_recall": round(w0_t1 / d, 4),
            "bge_dense_w0_top2_recall": round(w0_t2 / d, 4),
            "lslc_bge_hybrid_w2_top1_recall": round(w2_t1 / d, 4),
            "lslc_bge_hybrid_w2_top2_recall": round(w2_t2 / d, 4),
        }
        print(f"[BGE Stress {label}] {bge_stress[label]}", flush=True)

    # --- PART 3: LongMemEval N=6 Sanity with span16_k48 + BGE-small hybrid router (3 prompt/assembly modes) ---
    lme_modes = (
        ("chatml_exact_only", "exact_only", True),
        ("chatml_full", "full", True),
        ("plain_stage1_full", "full", False),
    )
    lme_rows: dict[str, list[dict[str, Any]]] = {m[0]: [] for m in lme_modes}
    lme_rows_by_cat: dict[str, dict[str, list[dict[str, Any]]]] = {m[0]: {} for m in lme_modes}

    for item in lme_items[:6]:
        cat = str(item.get("question_type", "unknown"))
        question = str(item["question"])
        gold_answer = str(item["answer"])
        pack = _pack_longmemeval_sessions(item, tokenizer, active_sys.cfg.page_size, max_pages=max_pages)
        page_sids: list[str] = pack["page_session_ids"]
        ans_sids: set[str] = pack["answer_session_ids"]
        session_records: list[dict[str, Any]] = pack["session_records"]

        # Modular session encoding + BGE 4-span TOC + blind DeltaUpdater (once per conversation, shared across 3 QA modes)
        _sync_device(device)
        t_m0 = time.perf_counter()
        z_book: torch.Tensor | None = None
        toc_embeds: torch.Tensor | None = None
        exact_bank: SparseExactBank | None = None
        bge_pages: torch.Tensor | None = None
        max_toc_diff = 0.0
        ep_strip_gates: list[float] = []
        ep_bank_writes = 0

        for rec in session_records:
            s_tensor = torch.tensor(rec["chunks"], dtype=torch.long, device=device).unsqueeze(0)
            zc, bc, tc = _encode_book_chunked(
                active_sys, s_tensor, embed_fn, vocab_size, device, amp_dtype, autocast_enabled,
                chunk_size=16, special_token_ids=special_ids, salience_span_len=16,
            )
            bge_sc = bge.encode_book_pages(s_tensor, None, tokenizer, num_spans=4, batch_size=64)
            if z_book is None:
                z_book = zc
                toc_embeds = tc
                bge_pages = bge_sc
                num_p_cur = z_book.shape[1]
                exact_bank = SparseExactBank(
                    token_ids=bc.token_ids,
                    page_ids=torch.arange(num_p_cur, device=device, dtype=torch.int16).view(1, num_p_cur, 1).expand(1, num_p_cur, bc.token_ids.shape[-1]).contiguous(),
                    h_coords=bc.h_coords,
                    w_coords=bc.w_coords,
                )
            else:
                assert toc_embeds is not None and exact_bank is not None and bge_pages is not None
                user_txt = str(rec["user_text"]).strip()
                if user_txt and z_book.shape[1] >= 1:
                    edit_ids = torch.tensor(_encode_text_or_mock(user_txt, tokenizer, vocab_size)[:96], dtype=torch.long, device=device).unsqueeze(0)
                    q_bge_e = bge.encode_texts([user_txt], is_query=True)
                    s_upd = bge.score_book(q_bge_e, bge_pages)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        r_upd, _ = active_sys.route_and_assemble(
                            z_book=z_book, toc_embeds=toc_embeds, query_embed=embed_fn(edit_ids).mean(dim=1),
                            exact_bank=exact_bank, embed_tokens_fn=embed_fn, k_deep=1, k_skim=0, mode="toc_only",
                            training_ste=False, query_input_ids=edit_ids, salience_route_weight=2.0,
                            external_dense_logits=s_upd,
                        )
                        tgt_p_upd = r_upd.deep_idx[:, -1]
                        p_upd_i = int(tgt_p_upd[0].item())
                        toc_before = active_sys.projector.project_toc(z_book[:, p_upd_i : p_upd_i + 1]).clone()
                        override_e = (
                            extract_contextual_edit_embeds(wrapper.llm, edit_ids, layer_idx=active_sys.cfg.contextual_edit_layer)
                            if active_sys.cfg.use_contextual_edit_encoding
                            else None
                        )
                        prev_bt = exact_bank.token_ids[:, p_upd_i].clone()
                        z_book, g_upd, exact_bank = active_sys.apply_page_delta(
                            z_book, tgt_p_upd, edit_ids, embed_tokens_fn=embed_fn, exact_bank=exact_bank,
                            vocab_size=vocab_size, edit_embeds_override=override_e, in_place=True, bank_write_threshold=0.45,
                        )
                        ep_strip_gates.append(float(g_upd.float().mean().item()))
                        if not torch.equal(exact_bank.token_ids[:, p_upd_i], prev_bt):
                            ep_bank_writes += 1
                        toc_after = active_sys.projector.project_toc(z_book[:, p_upd_i : p_upd_i + 1])
                    diff = float((toc_after.float() - toc_before.float()).abs().max().item())
                    if diff > max_toc_diff:
                        max_toc_diff = diff

                z_book = torch.cat((z_book, zc), dim=1)
                toc_embeds = torch.cat((toc_embeds, tc), dim=1)
                bge_pages = torch.cat((bge_pages, bge_sc), dim=1)
                new_num_p = z_book.shape[1]
                k_cur = exact_bank.token_ids.shape[-1]
                exact_bank = SparseExactBank(
                    token_ids=torch.cat((exact_bank.token_ids, bc.token_ids), dim=1),
                    page_ids=torch.arange(new_num_p, device=device, dtype=torch.int16).view(1, new_num_p, 1).expand(1, new_num_p, k_cur).contiguous(),
                    h_coords=torch.cat((exact_bank.h_coords, bc.h_coords), dim=1),
                    w_coords=torch.cat((exact_bank.w_coords, bc.w_coords), dim=1),
                    base_token_ids=None if exact_bank.base_token_ids is None else torch.cat((exact_bank.base_token_ids, bc.token_ids), dim=1),
                )
        _sync_device(device)
        mem_ms = (time.perf_counter() - t_m0) * 1000.0
        assert z_book is not None and toc_embeds is not None and exact_bank is not None and bge_pages is not None

        rq_ids = torch.tensor(_encode_text_or_mock(question, tokenizer, vocab_size), dtype=torch.long, device=device).unsqueeze(0)
        q_bge_q = bge.encode_texts([question], is_query=True)
        s_q = bge.score_book(q_bge_q, bge_pages)

        for mode_label, asm_mode, use_chatml in lme_modes:
            _reset_peak_vram(device)
            q_str = _format_qa_prompt(question) if use_chatml else f"\nQuestion: {question.strip()}\nAnswer:"
            u_open = user_open_ids if use_chatml else None
            p_prompt = torch.tensor(tokenizer.encode(q_str, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)

            _sync_device(device)
            t_q0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                kd_use = min(lslc_k_deep, z_book.shape[1])
                ks_use = min(lslc_k_skim, max(0, z_book.shape[1] - kd_use))
                r_out, asm = active_sys.route_and_assemble(
                    z_book=z_book, toc_embeds=toc_embeds, query_embed=embed_fn(rq_ids).mean(dim=1),
                    exact_bank=exact_bank, embed_tokens_fn=embed_fn, k_deep=kd_use, k_skim=ks_use,
                    mode=asm_mode, training_ste=False, hierarchical_chapter_size=chapter_size,
                    query_input_ids=rq_ids, salience_route_weight=2.0, exact_bank_attenuation="none",
                    external_dense_logits=s_q,
                )
            chosen = r_out.deep_idx[0].tolist()
            chosen_sids = {page_sids[i] for i in chosen if 0 <= i < len(page_sids)}
            gold_hit = float(bool(ans_sids & chosen_sids)) if ans_sids else None
            gen_txt = _greedy_generate_qa(
                wrapper.llm, asm.prefix_embeds.to(llm_dtype), p_prompt, tokenizer,
                max_new_tokens=max_new_tokens, user_open_ids=u_open,
            )
            _sync_device(device)
            q_ms = (time.perf_counter() - t_q0) * 1000.0
            qm = compute_community_qa_metrics(gen_txt, gold_answer)
            row = {
                **qm,
                "question_id": str(item.get("question_id", "")),
                "question_type": cat,
                "question": question,
                "gold": gold_answer,
                "pred": gen_txt,
                "gold_session_hit_rate": gold_hit,
                "active_prefix_tokens": int(asm.prefix_embeds.shape[1]),
                "memory_update_ms": round(mem_ms, 3),
                "per_query_ms": round(q_ms, 3),
                "peak_vram_mb": _peak_vram_mb(device),
                "max_toc_diff": round(max_toc_diff, 6),
                "mean_strip_gate": round(sum(ep_strip_gates) / len(ep_strip_gates), 4) if ep_strip_gates else 0.0,
                "num_bank_writes": ep_bank_writes,
            }
            lme_rows[mode_label].append(row)
            lme_rows_by_cat[mode_label].setdefault(cat, []).append(row)

    lme_summary: dict[str, Any] = {}
    for mode_label, _, _ in lme_modes:
        s_rows = lme_rows[mode_label]
        ov = _aggregate_eval_rows(s_rows)
        hits = [float(r["gold_session_hit_rate"]) for r in s_rows if r.get("gold_session_hit_rate") is not None]
        if hits:
            ov["gold_session_hit_rate"] = round(sum(hits) / len(hits), 4)
        ov["mean_strip_gate"] = round(sum(float(r["mean_strip_gate"]) for r in s_rows) / max(1, len(s_rows)), 4)
        ov["num_bank_writes_mean"] = round(sum(float(r["num_bank_writes"]) for r in s_rows) / max(1, len(s_rows)), 3)
        lme_summary[mode_label] = {
            "overall": ov,
            "by_category": {c: _aggregate_eval_rows(rl) for c, rl in sorted(lme_rows_by_cat[mode_label].items())},
            "samples": [
                {
                    "question_id": r["question_id"],
                    "question_type": r["question_type"],
                    "question": r["question"],
                    "gold": r["gold"],
                    "pred": r["pred"],
                    "gold_session_hit": r["gold_session_hit_rate"],
                    "extracted_em": r["extracted_em"],
                    "token_f1": r["token_f1"],
                    "substring_match": r["substring_match"],
                }
                for r in s_rows
            ],
        }
        print(
            f"[LongMemEval span16+BGE {mode_label}] EM={ov['extracted_em']:.3f} F1={ov['token_f1']:.3f} "
            f"Sub={ov['substring_match']:.3f} hit={ov.get('gold_session_hit_rate', 0.0):.3f} "
            f"tok={ov['active_prefix_tokens']:.1f} toc_diff={ov['max_toc_diff']}",
            flush=True,
        )

    return {
        "context_length": ctx_len,
        "g2_checkpoint": g2_info,
        "bge_model": bge_model_name,
        "jalur_a_triggered": jalur_a_triggered,
        "jalur_a_train_stats": jalur_a_train_stats,
        "saved_salience_checkpoint": saved_sal_ckpt,
        "active_system_used_for_bge_and_lme": "jalur_a_finetuned" if active_sys is not g2_sys else "v3_g2_zero_train",
        "ruler_span16_k48_zero_train": zero_train_res,
        "ruler_span16_k48_jalur_a_ft": jalur_a_res,
        "bge_2x2_table": {
            "zero_overlap_eval": bge_zero_overlap,
            "router_stress": bge_stress,
        },
        "longmemeval_sanity_n6": {
            "dataset": lme_meta,
            "modes": lme_summary,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gerbang 4: Pure-Paraphrase Contrastive Routing + Community Benchmarks.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--mock-llm", action="store_true", help="Run 100% local CPU smoke test with TinyMockCausalLM.")
    parser.add_argument("--tasks", type=str, default="all", help="Comma-separated: pilar4a,bge_pilar4a,realign,ruler,longmemeval,pareto_rag_only,stage2_full,diag_probe,span_salience_probe,span16_bge_probe or 'all'.")
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--g2-checkpoint", type=str, default="")
    parser.add_argument("--g3-checkpoint", type=str, default="")
    parser.add_argument("--save-checkpoint", type=str, default="")
    parser.add_argument("--bge-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--use-bge-toc", action="store_true", default=False)
    parser.add_argument("--salience-route-weight", type=float, default=1.5)
    parser.add_argument("--ruler-exact-only-ablation", action="store_true", default=False)
    parser.add_argument("--ruler-use-raw-question-router", action="store_true", default=False)
    parser.add_argument("--realign-salience-steps", type=int, default=150)
    parser.add_argument("--realign-projector-steps", type=int, default=120)
    parser.add_argument("--realign-projector-lr", type=float, default=1e-4)
    parser.add_argument("--k-exact", type=int, default=24)
    parser.add_argument("--salience-span-len", type=int, default=1)
    parser.add_argument("--pilar4a-page-spans", type=int, default=1)
    parser.add_argument("--use-contextual-query-encoding", action="store_true", default=True)
    parser.add_argument("--no-contextual-query-encoding", dest="use_contextual_query_encoding", action="store_false")
    parser.add_argument("--pilar4a-steps", type=int, default=240)
    parser.add_argument("--pilar4a-batch-size", type=int, default=8)
    parser.add_argument("--pilar4a-lr", type=float, default=3e-4)
    parser.add_argument("--pilar4a-eval-n", type=int, default=24)
    parser.add_argument("--ruler-lengths", type=str, default="4096,16384,32768")
    parser.add_argument("--ruler-episodes-per-task", type=int, default=10)
    parser.add_argument("--ruler-mk-distractors", type=int, default=48)
    parser.add_argument("--ruler-vt-hops", type=int, default=3)
    parser.add_argument("--lslc-k-deep", type=int, default=4)
    parser.add_argument("--lslc-k-skim", type=int, default=2)
    parser.add_argument("--chapter-size", type=int, default=16)
    parser.add_argument("--longmemeval-repo", type=str, default="xiaowu0162/longmemeval-cleaned")
    parser.add_argument("--longmemeval-file", type=str, default="longmemeval_s_cleaned.json")
    parser.add_argument("--longmemeval-local-json", type=str, default="")
    parser.add_argument("--longmemeval-n", type=int, default=60)
    parser.add_argument("--longmemeval-max-pages", type=int, default=0)
    parser.add_argument("--longmemeval-compact-budget", type=int, default=256)
    parser.add_argument("--longmemeval-compact-window", type=int, default=2000)
    parser.add_argument("--longmemeval-compact-chunk", type=int, default=2000)
    parser.add_argument("--longmemeval-systems", type=str, default="", help="Optional comma-separated system override for LongMemEval.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-json", type=str, default="/content/lslc_results/v3_gerbang4_community.json")
    args, _ = parser.parse_known_args(argv)
    return args


def _build_replay_pools(tokenizer, vocab_size: int, is_mock: bool) -> tuple[list[int], list[int], str]:
    if is_mock:
        train_pool = [2 + (i % (vocab_size - 2)) for i in range(4096)]
        val_pool = [2 + ((i * 7) % (vocab_size - 2)) for i in range(2048)]
        return train_pool, val_pool, "mock-synthetic-pool"
    return _load_wikitext_pools(tokenizer)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    ZeroOverlapParaphraseRoutingBenchmark.assert_disjoint_splits()

    raw_tasks = {t.strip().lower() for t in args.tasks.split(",") if t.strip()}
    if "stage2_full" in raw_tasks:
        run_tasks = {"realign", "bge_pilar4a", "ruler", "longmemeval"}
        args.use_bge_toc = True
        args.ruler_exact_only_ablation = True
    elif "all" in raw_tasks:
        run_tasks = {"pilar4a", "ruler", "longmemeval"}
    else:
        run_tasks = raw_tasks

    use_ctx_q = args.use_contextual_query_encoding and (not args.use_bge_toc)

    if args.mock_llm:
        vocab_size = 2048
        cfg = LSLCConfig.tiny_mock(
            use_contextual_edit_encoding=True,
            contextual_edit_layer=1,
            use_contextual_query_encoding=use_ctx_q,
            contextual_query_layer=1 if use_ctx_q else None,
            contextual_page_spans=args.pilar4a_page_spans,
            salience_span_len=args.salience_span_len,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=min(args.k_exact, 48),
        )
        llm = TinyMockCausalLM(vocab_size=vocab_size, d_llm=cfg.d_llm, num_layers=cfg.num_llm_layers)
        tokenizer = _MockCodeTokenizer(vocab_size=vocab_size)
        device = torch.device("cpu")
    else:
        cfg = LSLCConfig.from_model_name(
            args.model,
            use_contextual_edit_encoding=True,
            use_contextual_query_encoding=use_ctx_q,
            contextual_page_spans=args.pilar4a_page_spans,
            salience_span_len=args.salience_span_len,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=args.k_exact,
        )
        cfg.contextual_edit_layer = cfg.num_llm_layers // 2
        cfg.contextual_query_layer = (cfg.num_llm_layers // 2) if use_ctx_q else None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        llm, tokenizer = load_frozen_qwen3(cfg, load_in_4bit=False, device=device)
        vocab_size = len(llm.get_input_embeddings().weight)

    wrapper = FrozenQwenLSLCWrapper(cfg, llm).to(device)
    init_ckpt_path = args.init_checkpoint or (args.g2_checkpoint if "realign" in run_tasks else "")
    ckpt_info = _load_lslc_checkpoint(wrapper.lslc, init_ckpt_path) if init_ckpt_path else None
    train_pool, val_pool, data_source = _build_replay_pools(tokenizer, vocab_size, args.mock_llm)

    bge_toc: _PlugInBiEncoderTOC | None = None
    if args.use_bge_toc or ("bge_pilar4a" in run_tasks):
        bge_toc = _PlugInBiEncoderTOC(args.bge_model, device=device, is_mock=args.mock_llm)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": "3.0.0-gerbang4-community",
        "model": "TinyMockCausalLM" if args.mock_llm else args.model,
        "mock_llm": args.mock_llm,
        "device": str(device),
        "data_source": data_source,
        "init_checkpoint": ckpt_info,
        "save_checkpoint": args.save_checkpoint or None,
        "config": {
            "page_size": cfg.page_size,
            "k_exact": cfg.k_exact,
            "salience_span_len": cfg.salience_span_len,
            "contextual_page_spans": cfg.contextual_page_spans,
            "router_mode": cfg.router_mode,
            "salience_mode": cfg.salience_mode,
            "delta_pooling_mode": cfg.delta_pooling_mode,
            "use_contextual_query_encoding": cfg.use_contextual_query_encoding,
            "contextual_query_layer": cfg.contextual_query_layer,
            "use_bge_toc": bool(bge_toc is not None),
            "bge_model": None if bge_toc is None else bge_toc.model_name,
            "salience_route_weight": float(args.salience_route_weight),
            "lslc_k_deep": args.lslc_k_deep,
            "lslc_k_skim": args.lslc_k_skim,
            "chapter_size": args.chapter_size,
        },
        "realign": None,
        "pilar4a": None,
        "bge_pilar4a": None,
        "ruler": None,
        "longmemeval": None,
        "diag_probe": None,
        "span_salience_probe": None,
        "span16_bge_probe": None,
    }

    if "realign" in run_tasks:
        sal_stats = _finetune_salience_gate_span16(
            wrapper.lslc,
            wrapper.llm.get_input_embeddings(),
            tokenizer,
            train_pool,
            device,
            steps=args.realign_salience_steps,
            batch_size=8,
            lr=1e-3,
            span_len=cfg.salience_span_len,
            mixed_natural_replay_ratio=0.30,
            seed=args.seed,
            is_mock=args.mock_llm,
        )
        print(f"[REALIGN 2a SalienceGate 70/30] {sal_stats}", flush=True)
        proj_stats = _align_projector_chatml_qa(
            wrapper,
            tokenizer,
            train_pool,
            device,
            bge_toc=bge_toc,
            steps=args.realign_projector_steps,
            lr=args.realign_projector_lr,
            lslc_k_deep=args.lslc_k_deep,
            lslc_k_skim=args.lslc_k_skim,
            chapter_size=args.chapter_size,
            salience_route_weight=args.salience_route_weight,
            seed=args.seed,
            is_mock=args.mock_llm,
        )
        print(f"[REALIGN 2b Projector ChatML QA 50/50] {proj_stats}", flush=True)
        if args.save_checkpoint:
            _save_lslc_checkpoint(wrapper.lslc, args.save_checkpoint)
        payload["realign"] = {
            "salience_gate": sal_stats,
            "projector_chatml_qa": proj_stats,
            "saved_checkpoint": args.save_checkpoint or None,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if "bge_pilar4a" in run_tasks:
        assert bge_toc is not None
        p4_eval_n = min(args.pilar4a_eval_n, 4) if args.mock_llm else args.pilar4a_eval_n
        payload["bge_pilar4a"] = evaluate_bge_pilar4a_final(
            wrapper=wrapper,
            bge_toc=bge_toc,
            tokenizer=tokenizer,
            val_pool=val_pool,
            device=device,
            eval_n=p4_eval_n,
            salience_route_weight=args.salience_route_weight,
            seed=args.seed,
            is_mock=args.mock_llm,
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if "pilar4a" in run_tasks:
        p4_steps = min(args.pilar4a_steps, 6) if args.mock_llm else args.pilar4a_steps
        p4_eval_n = min(args.pilar4a_eval_n, 4) if args.mock_llm else args.pilar4a_eval_n
        p4_eval_cache: dict[int, dict[str, Any]] = {}
        step0_zero_shot = evaluate_pilar4a_zero_overlap(
            wrapper=wrapper,
            tokenizer=tokenizer,
            val_pool=val_pool,
            device=device,
            eval_pages=(16, 64),
            eval_n=p4_eval_n,
            seed=args.seed,
            is_mock=args.mock_llm,
            eval_cache=p4_eval_cache,
        )
        train_stats = train_pilar4a_contrastive_router(
            wrapper=wrapper,
            tokenizer=tokenizer,
            train_pool=train_pool,
            device=device,
            steps=p4_steps,
            batch_size=min(args.pilar4a_batch_size, 4) if args.mock_llm else args.pilar4a_batch_size,
            lr=args.pilar4a_lr,
            seed=args.seed,
            is_mock=args.mock_llm,
        )
        zero_overlap_eval = evaluate_pilar4a_zero_overlap(
            wrapper=wrapper,
            tokenizer=tokenizer,
            val_pool=val_pool,
            device=device,
            eval_pages=(16, 64),
            eval_n=p4_eval_n,
            seed=args.seed,
            is_mock=args.mock_llm,
            eval_cache=p4_eval_cache,
        )
        p4_eval_cache.clear()
        stress_reg = evaluate_router_stress_regression(
            wrapper=wrapper,
            tokenizer=tokenizer,
            val_pool=val_pool,
            device=device,
            is_mock=args.mock_llm,
            seed=args.seed + 700,
        )
        payload["pilar4a"] = {
            "step0_zero_shot_eval": step0_zero_shot,
            "train": train_stats,
            "zero_overlap_eval": zero_overlap_eval,
            "router_stress_regression": stress_reg,
        }
        if args.save_checkpoint:
            _save_lslc_checkpoint(wrapper.lslc, args.save_checkpoint)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[PILAR 4A] train={train_stats} | stress={stress_reg}", flush=True)

    if "ruler" in run_tasks:
        r_lengths = (
            (1024, 2048)
            if args.mock_llm
            else tuple(int(x.strip()) for x in args.ruler_lengths.split(",") if x.strip())
        )
        r_eps = min(args.ruler_episodes_per_task, 2) if args.mock_llm else args.ruler_episodes_per_task
        payload["ruler"] = evaluate_ruler_benchmark(
            wrapper=wrapper,
            tokenizer=tokenizer,
            val_pool=val_pool,
            device=device,
            lengths=r_lengths,
            episodes_per_task=r_eps,
            mk_distractors=args.ruler_mk_distractors,
            vt_hops=args.ruler_vt_hops,
            lslc_k_deep=args.lslc_k_deep,
            lslc_k_skim=args.lslc_k_skim,
            chapter_size=args.chapter_size,
            max_new_tokens=min(args.max_new_tokens, 12) if args.mock_llm else args.max_new_tokens,
            seed=args.seed,
            is_mock=args.mock_llm,
            bge_toc=bge_toc,
            salience_route_weight=args.salience_route_weight,
            include_exact_only_ablation=args.ruler_exact_only_ablation,
            use_raw_question_router=args.ruler_use_raw_question_router,
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if "pareto_rag_only" in run_tasks:
        r_lengths = (
            (1024, 2048)
            if args.mock_llm
            else tuple(int(x.strip()) for x in args.ruler_lengths.split(",") if x.strip())
        )
        r_eps = min(args.ruler_episodes_per_task, 2) if args.mock_llm else args.ruler_episodes_per_task
        lme_items_p, lme_meta_p = load_longmemeval_stratified(
            repo_id=args.longmemeval_repo,
            filename=args.longmemeval_file,
            local_path=args.longmemeval_local_json,
            n_total=min(args.longmemeval_n, 6) if args.mock_llm else args.longmemeval_n,
            seed=args.seed,
            allow_fallback=args.mock_llm,
        )
        payload["pareto_rag_only"] = {
            "ruler_pareto_bm25": evaluate_ruler_benchmark(
                wrapper=wrapper,
                tokenizer=tokenizer,
                val_pool=val_pool,
                device=device,
                lengths=r_lengths,
                episodes_per_task=r_eps,
                mk_distractors=args.ruler_mk_distractors,
                vt_hops=args.ruler_vt_hops,
                max_new_tokens=min(args.max_new_tokens, 12) if args.mock_llm else args.max_new_tokens,
                seed=args.seed,
                is_mock=args.mock_llm,
                systems_override=("bm25_rag_k1", "bm25_rag_k2"),
                use_raw_question_router=args.ruler_use_raw_question_router,
            ),
            "longmemeval_pareto_bm25": evaluate_longmemeval_benchmark(
                wrapper=wrapper,
                tokenizer=tokenizer,
                items=lme_items_p,
                dataset_meta=lme_meta_p,
                device=device,
                max_pages=args.longmemeval_max_pages,
                max_new_tokens=min(args.max_new_tokens, 12) if args.mock_llm else args.max_new_tokens,
                systems_override=("bm25_rag_k1", "bm25_rag_k2"),
            ),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if "longmemeval" in run_tasks:
        lme_items, lme_meta = load_longmemeval_stratified(
            repo_id=args.longmemeval_repo,
            filename=args.longmemeval_file,
            local_path=args.longmemeval_local_json,
            n_total=min(args.longmemeval_n, 6) if args.mock_llm else args.longmemeval_n,
            seed=args.seed,
            allow_fallback=args.mock_llm,
        )
        lme_sys_ov = (
            tuple(s.strip() for s in args.longmemeval_systems.split(",") if s.strip())
            if args.longmemeval_systems.strip()
            else None
        )
        def _flush_lme(partial_snap: dict[str, Any]) -> None:
            payload["longmemeval"] = partial_snap
            tmp_p = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_p.replace(out_path)

        payload["longmemeval"] = evaluate_longmemeval_benchmark(
            wrapper=wrapper,
            tokenizer=tokenizer,
            items=lme_items,
            dataset_meta=lme_meta,
            device=device,
            lslc_k_deep=args.lslc_k_deep,
            lslc_k_skim=args.lslc_k_skim,
            chapter_size=args.chapter_size,
            compact_budget_tokens=min(args.longmemeval_compact_budget, 128)
            if args.mock_llm
            else args.longmemeval_compact_budget,
            compact_window_trigger=args.longmemeval_compact_window,
            compact_max_chunk_tokens=args.longmemeval_compact_chunk,
            max_pages=args.longmemeval_max_pages,
            max_new_tokens=min(args.max_new_tokens, 12) if args.mock_llm else args.max_new_tokens,
            bge_toc=bge_toc,
            salience_route_weight=args.salience_route_weight,
            systems_override=lme_sys_ov,
            flush_callback=_flush_lme,
            flush_every=10,
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if "diag_probe" in run_tasks:
        diag_items, diag_meta = load_longmemeval_stratified(
            repo_id=args.longmemeval_repo,
            filename=args.longmemeval_file,
            local_path=args.longmemeval_local_json,
            n_total=6,
            seed=args.seed,
            allow_fallback=args.mock_llm,
        )
        payload["diag_probe"] = run_diag_probe(
            wrapper=wrapper,
            tokenizer=tokenizer,
            val_pool=val_pool,
            lme_items=diag_items,
            lme_meta=diag_meta,
            device=device,
            g2_checkpoint=args.g2_checkpoint,
            g3_checkpoint=args.g3_checkpoint or args.init_checkpoint,
            k_exact=cfg.k_exact,
            episodes_per_task=min(args.ruler_episodes_per_task, 2) if args.mock_llm else args.ruler_episodes_per_task,
            mk_distractors=args.ruler_mk_distractors,
            lslc_k_deep=args.lslc_k_deep,
            lslc_k_skim=args.lslc_k_skim,
            chapter_size=args.chapter_size,
            compact_budget_tokens=min(args.longmemeval_compact_budget, 128)
            if args.mock_llm
            else args.longmemeval_compact_budget,
            max_pages=args.longmemeval_max_pages,
            max_new_tokens=min(args.max_new_tokens, 12) if args.mock_llm else args.max_new_tokens,
            seed=args.seed,
            is_mock=args.mock_llm,
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if "span_salience_probe" in run_tasks:
        payload["span_salience_probe"] = run_span_salience_probe(
            wrapper=wrapper,
            tokenizer=tokenizer,
            val_pool=val_pool,
            device=device,
            g2_checkpoint=args.g2_checkpoint,
            episodes_per_task=min(args.ruler_episodes_per_task, 2) if args.mock_llm else args.ruler_episodes_per_task,
            mk_distractors=args.ruler_mk_distractors,
            lslc_k_deep=args.lslc_k_deep,
            lslc_k_skim=args.lslc_k_skim,
            chapter_size=args.chapter_size,
            max_new_tokens=min(args.max_new_tokens, 12) if args.mock_llm else args.max_new_tokens,
            seed=args.seed,
            is_mock=args.mock_llm,
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if "span16_bge_probe" in run_tasks:
        lme_items_p, lme_meta_p = load_longmemeval_stratified(
            repo_id=args.longmemeval_repo,
            filename=args.longmemeval_file,
            local_path=args.longmemeval_local_json,
            n_total=6,
            seed=args.seed,
            allow_fallback=args.mock_llm,
        )
        payload["span16_bge_probe"] = run_span16_bge_probe(
            wrapper=wrapper,
            tokenizer=tokenizer,
            train_pool=train_pool,
            val_pool=val_pool,
            lme_items=lme_items_p,
            lme_meta=lme_meta_p,
            device=device,
            g2_checkpoint=args.g2_checkpoint,
            save_salience_checkpoint=args.save_checkpoint,
            bge_model_name=args.bge_model,
            pilar4a_eval_n=min(args.pilar4a_eval_n, 4) if args.mock_llm else args.pilar4a_eval_n,
            episodes_per_task=min(args.ruler_episodes_per_task, 2) if args.mock_llm else args.ruler_episodes_per_task,
            mk_distractors=args.ruler_mk_distractors,
            lslc_k_deep=args.lslc_k_deep,
            lslc_k_skim=args.lslc_k_skim,
            chapter_size=args.chapter_size,
            max_pages=args.longmemeval_max_pages if args.longmemeval_max_pages > 0 else 64,
            max_new_tokens=min(args.max_new_tokens, 12) if args.mock_llm else args.max_new_tokens,
            seed=args.seed,
            is_mock=args.mock_llm,
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[DONE] Wrote Gerbang 4 results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
