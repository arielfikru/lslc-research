#!/usr/bin/env python3
"""Colab LSLC v2.0.0 Complete Methodological & Empirical Hardening Suite.

Resolves every external peer-review critique on a single unified Qwen/Qwen3-1.7B checkpoint:
1. Strict Normalized EM (`strict_em`) reported alongside Substring Inclusion EM (`substr_em`),
   Teacher-Forced Content-Token Accuracy (`tf_content_tok_acc`), and Free-Running Character F1 (`char_f1`),
   with early newline/period stopping during greedy generation.
2. Pure End-to-End GPU-Resident LSLC Routing (`QueryToTOCRouter` + in-VRAM `SparseExactBank` late-interaction
   MaxSim, 0 B extra GPU storage, 0 B CPU RAM) vs. Hybrid Shared Okapi BM25 Routing (with exact CPU host-RAM
   byte accounting) at both 32K (P=128, N=180) and 128K (P=512, N=90).
3. 2-Level Chapter->Page Mean-Pooled Hierarchy (c=16) vs. Equal-Token-Count Truncated Flat TOC Control
   (isolating hierarchical chapter aggregation from simple TOC token-count reduction).
4. 5-Way Multi-Turn State-Override Benchmark (3 seeds) across:
     (a) Structured Key-Value Overrides (k=3 numeric N=108, k=5 symbolic N=90)
     (b) Paraphrased / Unstructured Natural-Language Revisions (k=3 discourse overrides N=90)
   comparing:
     - Raw-Text RAG (Immutable)
     - Raw-Text RAG (Append-Errata)
     - Raw-Text RAG (Direct Text-Patch + BM25 Re-Index — deterministic non-LLM mutable RAG)
     - Raw-Text RAG (In-Place LLM-Rewrite via frozen Qwen3-1.7B, + full 256-token page rewrite latency)
     - LSLC In-Place Convex DeltaUpdater (Pure End-to-End GPU Router & Shared BM25)
   with restored Static l=0 Policy Retention (`static_retention_em` & `static_strict_em`) via joint static replay
   and pre-gate channel-wise RMS alignment.
5. Controlled Iso-Param (40.42M) / Iso-Byte (12.125 KiB/page) / Iso-Step / Iso-ExactBank Ablation:
     - 1D-Flat + SparseExactBank (no layer split, no 2D spatial grid)
     - 1D-3Layer + SparseExactBank (3 volatility layers l in {0,1,2} with max|Delta_TOC|=0.0, but 1D flat slots)
     - 3D-Spatial LSLC [L=3, C=128, H=4, W=4] + SparseExactBank (full 2D spatial grid + FactorizedRoPE3D + Matryoshka)
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lslc import (
    ChunkedTextPageDataset,
    FrozenQwenLSLCWrapper,
    LSLCConfig,
    LSLCEpisode,
    LivingBookStateOverrideDataset,
    RouterOutput,
    SyntheticExactSemanticNIAHDataset,
    compute_analytical_memory_stats,
    load_frozen_qwen3,
    select_amp_dtype,
)
from lslc.encoder import SparseExactBank
from scripts.colab_run_m2 import (
    _cache_bytes,
    _load_wikitext_pools,
    evaluate_continuation_ppl,
    train_stage,
)
from scripts.colab_run_rebuttal_v2 import (
    BM25PageRetriever,
    _encode_book_chunked,
    _llm_rewrite_state_block,
    _mean_std_ci,
    _strip_leading_ws_answer,
    profile_full_page_llm_rewrite_latency,
    profile_memory_and_latency_scaling,
)


def _norm_answer(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"^[\s.,:;!?'\"`\-]+|[\s.,:;!?'\"`\-]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _char_f1_span(pred_str: str, gold_str: str) -> float:
    g_s = _norm_answer(gold_str)
    p_s = _norm_answer(pred_str)
    if not g_s:
        return 1.0 if not p_s else 0.0
    if not p_s:
        return 0.0
    p_cnt = Counter(p_s)
    g_cnt = Counter(g_s)
    common = sum((p_cnt & g_cnt).values())
    if common == 0:
        return 0.0
    return (2.0 * common) / float(len(p_s) + len(g_s))


def _extract_canonical_candidates(gen_text: str) -> set[str]:
    """Deterministically extract candidate short answers from the first non-empty generated sentence/line.

    Applies identically to Raw-Text RAG and LSLC so conversational preambles (e.g. 'The code is 3034-3582.'
    or '\\n[RECORD] API-KEY = 3034-3582') are evaluated on semantic/value correctness separately from format adherence.
    """
    trimmed = gen_text.lstrip()
    if not trimmed:
        return set()
    first_line = trimmed.splitlines()[0].strip()
    first_sent = re.split(r"[\n.;]", trimmed)[0].strip()
    cands: set[str] = set()
    for base in (first_line, first_sent):
        if not base:
            continue
        cands.add(_norm_answer(base))
        if "=" in base or ":" in base:
            rhs = re.split(r"[=:]", base)[-1].strip()
            rhs_clause = re.split(r"[.;,\n]", rhs)[0].strip()
            cands.add(_norm_answer(rhs_clause))
            if rhs_clause.split():
                cands.add(_norm_answer(rhs_clause.split()[0]))
        low = base.lower()
        for marker in (
            " updated to ",
            " stands at ",
            " moving the spending ceiling to ",
            " authorized cap to ",
            " prior figure with ",
            " requires ",
            " require ",
            " runs in ",
            " is ",
            " are ",
            " to ",
            " at ",
            " in ",
            " use ",
            " uses ",
            " on ",
        ):
            if marker in low:
                idx = low.rfind(marker) + len(marker)
                rhs = base[idx:].strip()
                rhs_clause = re.split(r"[.;,\n]", rhs)[0].strip()
                cands.add(_norm_answer(rhs_clause))
                words = rhs_clause.split()
                if words:
                    cands.add(_norm_answer(words[0]))
                    if len(words) >= 2:
                        cands.add(_norm_answer(" ".join(words[:2])))
                stripped_units = re.sub(
                    r"\b(?:mode|seconds|second|spikes|spike|replication|effective|immediately|for|this|cycle|across|all|ledgers)\b.*$",
                    "",
                    rhs_clause,
                    flags=re.IGNORECASE,
                ).strip()
                if stripped_units:
                    cands.add(_norm_answer(stripped_units))
        for m in re.findall(r"\b\d{4}-\d{4}\b|\b\d{2,3}M\b", base):
            cands.add(_norm_answer(m))
        for sym in SYMBOLIC_STATES:
            if sym in low:
                cands.add(_norm_answer(sym))
    cands.discard("")
    return cands


@torch.no_grad()
def greedy_generate_concise(
    llm,
    prefix_embeds: torch.Tensor,
    query_ids: torch.Tensor,
    tokenizer,
    max_new_tokens: int = 24,
) -> tuple[torch.Tensor, str]:
    """Fair greedy decoding from [prefix_embeds | query_ids] up to max_new_tokens.

    Never terminates on leading whitespace/newlines before content tokens are emitted; stops only on EOS
    or after a complete non-empty line has been generated, ensuring identical stopping rules for Raw-Text RAG and LSLC.
    """
    embed_fn = llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    prefix_cast = prefix_embeds.to(llm_dtype)
    q_embeds = embed_fn(query_ids).to(llm_dtype)
    inp_embeds = torch.cat((prefix_cast, q_embeds), dim=1)
    bsz, init_len, _ = inp_embeds.shape
    device = inp_embeds.device
    attn_mask = torch.ones((bsz, init_len), device=device, dtype=torch.long)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    prev_cache = getattr(llm.config, "use_cache", False) if hasattr(llm, "config") else False
    if hasattr(llm, "config"):
        llm.config.use_cache = True

    gen_ids: list[int] = []
    has_non_ws = False
    try:
        out = llm(inputs_embeds=inp_embeds, attention_mask=attn_mask, use_cache=True)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tid = int(next_tok.item())
        gen_ids.append(tid)
        pkv = getattr(out, "past_key_values", None)

        for _step_i in range(max_new_tokens - 1):
            if eos_id is not None and tid == eos_id:
                break
            piece = tokenizer.decode([tid], skip_special_tokens=False)
            if piece.strip():
                has_non_ws = True
            if has_non_ws and ("\n" in piece or "." in piece):
                break
            attn_mask = torch.cat((attn_mask, torch.ones((bsz, 1), device=device, dtype=torch.long)), dim=1)
            out = llm(input_ids=next_tok, attention_mask=attn_mask, past_key_values=pkv, use_cache=True)
            pkv = getattr(out, "past_key_values", None)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tid = int(next_tok.item())
            gen_ids.append(tid)
    finally:
        if hasattr(llm, "config"):
            llm.config.use_cache = prev_cache

    gen_tensor = torch.tensor([gen_ids], dtype=torch.long, device=device)
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return gen_tensor, gen_text


def compute_answer_metrics_v2(
    ans_logits: torch.Tensor,
    a_ids: torch.Tensor,
    gen_ids: torch.Tensor,
    gen_text: str,
    answer_str: str,
    tokenizer,
    alt_valid_strs: list[str] | None = None,
) -> tuple[float, float, float, float, float]:
    """Return (strict_em, extracted_em, substr_em, tf_content_tok_acc, char_f1)."""
    logits_c, a_c = _strip_leading_ws_answer(ans_logits, a_ids, tokenizer)
    tf_content_tok_acc = float((logits_c.argmax(dim=-1) == a_c).float().mean().item())

    gold_norm = _norm_answer(answer_str)
    valid_norms = {gold_norm}
    if alt_valid_strs:
        for s in alt_valid_strs:
            if s:
                valid_norms.add(_norm_answer(s))

    trimmed_gen = gen_text.lstrip()
    first_clause = _norm_answer(re.split(r"[\n.;]", trimmed_gen)[0]) if trimmed_gen else ""

    # Skip any leading whitespace/newline tokens in gen_ids before slicing target_tok_len
    raw_tok_list = gen_ids[0].tolist()
    non_ws_start = 0
    while non_ws_start < len(raw_tok_list):
        if tokenizer.decode([raw_tok_list[non_ws_start]], skip_special_tokens=True).strip():
            break
        non_ws_start += 1
    target_tok_len = max(1, a_c.shape[1])
    span_ids = raw_tok_list[non_ws_start : non_ws_start + target_tok_len]
    span_text = tokenizer.decode(span_ids, skip_special_tokens=True) if span_ids else ""
    span_norm = _norm_answer(span_text)

    strict_em = 1.0 if (first_clause in valid_norms or span_norm in valid_norms) else 0.0
    extracted_cands = _extract_canonical_candidates(gen_text)
    extracted_em = 1.0 if (strict_em == 1.0 or bool(valid_norms & extracted_cands)) else 0.0

    gen_low = _norm_answer(gen_text)
    substr_em = 1.0 if (extracted_em == 1.0 or any(v in gen_low for v in valid_norms if v)) else 0.0

    # Compute character F1 against the closest valid target representation
    eval_cands = ({first_clause} | extracted_cands) if (first_clause or extracted_cands) else {gen_low}
    best_f1 = max(
        (_char_f1_span(c, v) for c in eval_cands for v in valid_norms if v),
        default=0.0,
    )
    return strict_em, extracted_em, substr_em, tf_content_tok_acc, best_f1


def estimate_bm25_cpu_ram_bytes(bm25: BM25PageRetriever) -> int:
    """Measure serialized/in-memory CPU footprint of the BM25 lexical index (terms, counts, IDF, doc_lens)."""
    total_bytes = len(bm25.doc_lens) * 8
    for term, idf_v in bm25.idf.items():
        total_bytes += len(term.encode("utf-8")) + 16  # term string + float64 IDF + df int64
    for cnt in bm25.docs:
        for term in cnt.keys():
            total_bytes += len(term.encode("utf-8")) + 8  # term key + int64 tf count
    return total_bytes


SYMBOLIC_STATES = (
    "canary-staging",
    "shadow-mirror",
    "blue-green-prod",
    "quorum-degraded",
    "rollback-lockdown",
    "us-west-oregon",
    "eu-central-frankfurt",
    "ap-southeast-singapore",
)


def build_k5_symbolic_with_static_episodes(
    tokenizer,
    bg_pool: list[int],
    num_episodes: int = 30,
    num_pages: int = 4,
    page_size: int = 256,
    num_edits: int = 5,
    seed: int = 8001,
) -> list[LSLCEpisode]:
    """k=5 sequential symbolic state overrides with 1/3 static l=0 policy retention queries."""
    episodes: list[LSLCEpisode] = []
    static_toks = tokenizer.encode("\nSTATIC POLICY : security audit requires dual signoff .\n", add_special_tokens=False)

    for i in range(num_episodes):
        rng = random.Random(seed + i * 509)
        pages = []
        for p in range(num_pages):
            start = rng.randint(0, max(1, len(bg_pool) - page_size - 1))
            ptoks = list(bg_pool[start : start + page_size])
            if p == 0:
                ptoks[: len(static_toks)] = static_toks[:page_size]
            pages.append(ptoks)

        edit_page = 1
        init_toks = tokenizer.encode("\nDEPLOYMENT STATE INITIAL : cold-standby .\n", add_special_tokens=False)
        pages[edit_page][: len(init_toks)] = init_toks[:page_size]

        chosen_states = rng.sample(SYMBOLIC_STATES, k=min(num_edits, len(SYMBOLIC_STATES)))
        edits: list[tuple[int, torch.Tensor]] = []
        latest_state = "cold-standby"
        for latest_state in chosen_states:
            e_text = f"REVISION OVERRIDE : deployment state updated to {latest_state} ."
            e_ids = torch.tensor(tokenizer.encode(e_text, add_special_tokens=False), dtype=torch.long)
            edits.append((edit_page, e_ids))

        check_static = i % 3 == 0
        if check_static:
            q_text = "\nQuestion: What does the static security audit policy require? Answer:"
            a_text = " dual signoff"
            ans_str = "dual signoff"
            target_p = 0
        else:
            q_text = "\nQuestion: What is the current updated deployment state? Answer:"
            a_text = f" {latest_state}"
            ans_str = latest_state
            target_p = edit_page

        episodes.append(
            LSLCEpisode(
                context_pages=torch.tensor(pages, dtype=torch.long),
                query_ids=torch.tensor(tokenizer.encode(q_text, add_special_tokens=False), dtype=torch.long),
                answer_ids=torch.tensor(tokenizer.encode(a_text, add_special_tokens=False), dtype=torch.long),
                target_page_idx=target_p,
                metadata={"is_overridden_query": not check_static, "answer_str": ans_str, "num_edits": num_edits},
                edit_pages=edits,
            )
        )
    return episodes


PARAPHRASED_BUDGET_TEMPLATES = (
    "Board memo: leadership approved moving the spending ceiling to {val} for this cycle .",
    "Finance directive: adjust the authorized cap to {val} effective immediately .",
    "Treasury sign-off: the new quarterly limit now stands at {val} .",
    "Executive note: replace the prior figure with {val} across all ledgers .",
)


def build_paraphrased_nl_override_episodes(
    tokenizer,
    bg_pool: list[int],
    num_episodes: int = 30,
    num_pages: int = 4,
    page_size: int = 256,
    num_edits: int = 3,
    seed: int = 7001,
) -> list[LSLCEpisode]:
    """Paraphrased natural-language state overrides where rigid key-value regex patching cannot match."""
    episodes: list[LSLCEpisode] = []
    static_toks = tokenizer.encode("\nSTATIC POLICY : security audit requires dual signoff .\n", add_special_tokens=False)
    budget_pool = LivingBookStateOverrideDataset.BUDGET_POOL

    for i in range(num_episodes):
        rng = random.Random(seed + i * 613)
        pages = []
        for p in range(num_pages):
            start = rng.randint(0, max(1, len(bg_pool) - page_size - 1))
            ptoks = list(bg_pool[start : start + page_size])
            if p == 0:
                ptoks[: len(static_toks)] = static_toks[:page_size]
            pages.append(ptoks)

        edit_page = 1
        init_val = f"{rng.randint(10, 45)}M"
        init_toks = tokenizer.encode(f"\nSTATE BUDGET INITIAL : {init_val} .\n", add_special_tokens=False)
        pages[edit_page][: len(init_toks)] = init_toks[:page_size]

        chosen_vals = rng.sample(budget_pool, k=min(num_edits, len(budget_pool)))
        edits: list[tuple[int, torch.Tensor]] = []
        latest_val = init_val
        for step_idx, latest_val in enumerate(chosen_vals):
            tmpl = PARAPHRASED_BUDGET_TEMPLATES[(i + step_idx) % len(PARAPHRASED_BUDGET_TEMPLATES)]
            e_text = tmpl.format(val=latest_val)
            e_ids = torch.tensor(tokenizer.encode(e_text, add_special_tokens=False), dtype=torch.long)
            edits.append((edit_page, e_ids))

        check_static = i % 3 == 0
        if check_static:
            q_text = "\nQuestion: What does the static security audit policy require? Answer:"
            a_text = " dual signoff"
            ans_str = "dual signoff"
            target_p = 0
        else:
            q_text = "\nQuestion: What is the current updated budget allocation? Answer:"
            a_text = f" {latest_val}"
            ans_str = latest_val
            target_p = edit_page

        episodes.append(
            LSLCEpisode(
                context_pages=torch.tensor(pages, dtype=torch.long),
                query_ids=torch.tensor(tokenizer.encode(q_text, add_special_tokens=False), dtype=torch.long),
                answer_ids=torch.tensor(tokenizer.encode(a_text, add_special_tokens=False), dtype=torch.long),
                target_page_idx=target_p,
                metadata={
                    "is_overridden_query": not check_static,
                    "answer_str": ans_str,
                    "num_edits": num_edits,
                    "paraphrased_nl": True,
                },
                edit_pages=edits,
            )
        )
    return episodes


def build_concise_niah_episodes(
    tokenizer,
    bg_pool: list[int],
    num_episodes: int,
    num_pages: int,
    page_size: int,
    seed: int,
) -> list[LSLCEpisode]:
    """Wrap SyntheticExactSemanticNIAHDataset recording full_fact_str in metadata for strict clause-level & short-answer evaluation."""
    base_ds = SyntheticExactSemanticNIAHDataset(
        num_episodes=num_episodes,
        num_pages=num_pages,
        page_size=page_size,
        tokenizer=tokenizer,
        seed=seed,
        mode="mixed",
        background_token_pool=bg_pool,
    )
    out: list[LSLCEpisode] = []
    for ep in base_ds:
        if ep.metadata.get("needle_type") == "semantic":
            full_fact = tokenizer.decode(ep.answer_ids.tolist(), skip_special_tokens=True).strip()
            meta = dict(ep.metadata)
            meta["full_fact_str"] = full_fact
            out.append(
                LSLCEpisode(
                    context_pages=ep.context_pages,
                    query_ids=ep.query_ids,
                    answer_ids=ep.answer_ids,
                    target_page_idx=ep.target_page_idx,
                    metadata=meta,
                    edit_pages=ep.edit_pages,
                )
            )
        else:
            out.append(ep)
    return out


def train_unified_v2_checkpoint(
    wrapper: FrozenQwenLSLCWrapper,
    train_pool: list[int],
    tokenizer,
    device: torch.device,
) -> dict[str, Any]:
    """Train ONE unified Qwen3-1.7B adapter with static l=0 replay and multi-scale frozen-toc_proj router."""
    embed_fn = wrapper.llm.get_input_embeddings()
    amp_dtype, _ = select_amp_dtype(device)
    vocab_size = len(embed_fn.weight)

    train_niah_4p = build_concise_niah_episodes(tokenizer, train_pool, 100, 4, wrapper.cfg.page_size, seed=42)
    train_niah_16p = build_concise_niah_episodes(tokenizer, train_pool, 60, 16, wrapper.cfg.page_size, seed=142)
    # Include static l=0 policy episodes (with 0 edits) in V2_BASE so encoder & deep_l0_proj learn static rules!
    static_base_eps = [
        LSLCEpisode(
            context_pages=ep.context_pages,
            query_ids=ep.query_ids,
            answer_ids=ep.answer_ids,
            target_page_idx=0,
            metadata=ep.metadata,
            edit_pages=None,
        )
        for ep in LivingBookStateOverrideDataset(
            num_episodes=60,
            num_pages=4,
            page_size=wrapper.cfg.page_size,
            num_edits=1,
            tokenizer=tokenizer,
            seed=303,
            background_token_pool=train_pool,
        )
        if not ep.metadata["is_overridden_query"]
    ]
    base_stats = train_stage(
        "V2_BASE_WITH_STATIC_L0",
        wrapper,
        train_niah_4p + train_niah_16p + static_base_eps,
        steps=240,
        grad_accum=2,
        lr=3e-4,
        device=device,
        use_exact_bank=True,
        distill_salience=True,
        run_matryoshka=True,
    )

    # Stage 1B: Train DeltaUpdater with joint static l=0 replay across k=3, k=5, and paraphrased NL edits
    frozen_delta_stage = []
    for name_p, p in wrapper.lslc.named_parameters():
        if not (name_p.startswith("delta_updater.") or name_p.startswith("router.")) and p.requires_grad:
            p.requires_grad_(False)
            frozen_delta_stage.append(p)

    train_k3 = list(
        LivingBookStateOverrideDataset(
            num_episodes=72,
            num_pages=4,
            page_size=wrapper.cfg.page_size,
            num_edits=3,
            tokenizer=tokenizer,
            seed=42,
            background_token_pool=train_pool,
        )
    )
    train_nl = build_paraphrased_nl_override_episodes(
        tokenizer, train_pool, num_episodes=48, num_pages=4, num_edits=3, seed=7001
    )
    train_k5 = build_k5_symbolic_with_static_episodes(
        tokenizer, train_pool, num_episodes=60, num_pages=4, num_edits=5, seed=8001
    )

    k3_nl_stats = train_stage(
        "V2_DELTA_K3_AND_PARAPHRASED_WITH_STATIC_REPLAY",
        wrapper,
        train_k3 + train_nl,
        steps=130,
        grad_accum=2,
        lr=3e-4,
        device=device,
        use_exact_bank=True,
        distill_salience=False,
        run_matryoshka=False,
        use_edits=True,
    )
    k5_stats = train_stage(
        "V2_DELTA_K5_WITH_STATIC_REPLAY",
        wrapper,
        train_k5 + train_nl[:24],
        steps=120,
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

    # Stage 1C: Multi-scale router co-training across P in {4, 16, 128} with projector.toc_proj STRICTLY FROZEN
    # so TOC prefix embeddings remain 100% aligned with the frozen LLM input space.
    ds_r4 = build_concise_niah_episodes(tokenizer, train_pool, 30, 4, wrapper.cfg.page_size, seed=501)
    ds_r16 = build_concise_niah_episodes(tokenizer, train_pool, 30, 16, wrapper.cfg.page_size, seed=601)
    ds_r128 = build_concise_niah_episodes(tokenizer, train_pool, 30, 128, wrapper.cfg.page_size, seed=777)
    multi_scale_eps = ds_r4 + ds_r16 + ds_r128 + static_base_eps[:10]
    random.Random(42).shuffle(multi_scale_eps)

    opt_r = torch.optim.AdamW(wrapper.lslc.router.parameters(), lr=5e-4, weight_decay=0.01)
    wrapper.lslc.eval()
    wrapper.lslc.router.train()
    t_r0 = time.perf_counter()
    last_r_ce = 0.0
    for step_i, ep in enumerate(multi_scale_eps):
        pages = ep.context_pages.unsqueeze(0).to(device)
        q_ids = ep.query_ids.unsqueeze(0).to(device)
        target_p = torch.tensor([ep.target_page_idx], dtype=torch.long, device=device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            z_book, bank, toc_tokens = _encode_book_chunked(
                wrapper, pages, embed_fn, vocab_size, chunk_pages=32
            )
            q_tok_emb = embed_fn(q_ids)
            q_emb = q_tok_emb.mean(dim=1)
            bank_emb = embed_fn(bank.token_ids.clamp(min=0).to(device))

        opt_r.zero_grad(set_to_none=True)
        route_out = wrapper.lslc.router(
            q_emb.float(),
            toc_tokens.detach().float(),
            k_deep=2,
            k_skim=min(4, max(1, pages.shape[1] - 2)),
            query_token_embeds=q_tok_emb.detach().float(),
            exact_bank_embeds=bank_emb.detach().float(),
            salience_route_weight=8.0,
        )
        loss = F.cross_entropy(route_out.logits, target_p)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapper.lslc.router.parameters(), 1.0)
        opt_r.step()
        last_r_ce = float(loss.item())
        if (step_i + 1) in (1, 25, 50, 75, len(multi_scale_eps)):
            print(f"  [V2_ROUTER_MULTISCALE] step={step_i+1}/{len(multi_scale_eps)} route_ce={last_r_ce:.4f}", flush=True)

    wrapper.lslc.eval()
    return {
        "base_with_static_l0": base_stats,
        "delta_k3_and_paraphrased": k3_nl_stats,
        "delta_k5_with_static": k5_stats,
        "router_multiscale_frozen_toc_proj": {
            "steps": len(multi_scale_eps),
            "elapsed_sec": round(time.perf_counter() - t_r0, 2),
            "final_route_ce": round(last_r_ce, 4),
        },
        "trainable_params": wrapper.lslc.trainable_parameter_count(),
    }


@torch.no_grad()
def evaluate_long_context_niah_v2(
    wrapper: FrozenQwenLSLCWrapper,
    val_pool: list[int],
    tokenizer,
    device: torch.device,
    num_pages: int,
    seeds: list[int],
    episodes_per_seed: int,
    chapter_size: int = 16,
) -> dict[str, Any]:
    """Multi-seed NIAH at P=128 (32K) or P=512 (128K) comparing Pure GPU Router vs Hybrid BM25 + Hierarchy Ablation."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    amp_dtype, _ = select_amp_dtype(device)
    vocab_size = len(embed_fn.weight)
    wrapper.lslc.eval()

    systems = (
        "raw_rag_shared_bm25",
        "lslc_e2e_gpu_router_flat",
        "lslc_e2e_gpu_router_hier",
        "lslc_shared_bm25_flat",
        "lslc_shared_bm25_trunc_flat_ctrl",
        "lslc_shared_bm25_hier",
    )
    metrics_keys = (
        "exact_strict_em",
        "exact_extracted_em",
        "exact_substr_em",
        "exact_tf_content_tok_acc",
        "exact_char_f1",
        "semantic_strict_em",
        "semantic_extracted_em",
        "semantic_substr_em",
        "semantic_tf_content_tok_acc",
        "semantic_char_f1",
    )
    per_sys = {s: {k: [] for k in metrics_keys} for s in systems}
    active_tok_counts = {s: 0 for s in systems}

    capture_rates = []
    bm25_cpu_ram_bytes_list = []
    lr_t1_all, lr_t2_all, lr_t1_ex, lr_t2_ex, lr_t1_sem, lr_t2_sem = [], [], [], [], [], []
    bm_t1_all, bm_t2_all, bm_t1_ex, bm_t2_ex, bm_t1_sem, bm_t2_sem = [], [], [], [], [], []
    paired_diff_keys = (
        "exact_strict_em",
        "exact_extracted_em",
        "exact_substr_em",
        "semantic_strict_em",
        "semantic_extracted_em",
        "semantic_substr_em",
    )
    paired_seed_diffs: dict[str, list[float]] = {k: [] for k in paired_diff_keys}
    paired_ep_diffs: dict[str, list[float]] = {k: [] for k in paired_diff_keys}

    def _eval_pref(prefix_llm: torch.Tensor, q_ids: torch.Tensor, a_ids: torch.Tensor, turn_emb: torch.Tensor, ans_str: str, alt_strs: list[str]):
        inp = torch.cat((prefix_llm, turn_emb), dim=1)
        logits = wrapper.llm(inputs_embeds=inp).logits
        start = prefix_llm.shape[1] + q_ids.shape[1] - 1
        ans_logits = logits[:, start : start + a_ids.shape[1], :]
        gen_ids, gen_text = greedy_generate_concise(wrapper.llm, prefix_llm, q_ids, tokenizer, max_new_tokens=24)
        return compute_answer_metrics_v2(ans_logits, a_ids, gen_ids, gen_text, ans_str, tokenizer, alt_valid_strs=alt_strs)

    for seed in seeds:
        eps = build_concise_niah_episodes(
            tokenizer, val_pool, episodes_per_seed, num_pages, wrapper.cfg.page_size, seed=seed
        )
        seed_bucket = {s: {k: [] for k in metrics_keys} for s in systems}
        cap_hits, cap_tot = 0, 0
        lr_t1_e, lr_t2_e, lr_t1_s, lr_t2_s = [], [], [], []
        bm_t1_e, bm_t2_e, bm_t1_s, bm_t2_s = [], [], [], []

        for idx_ep, ep in enumerate(eps):
            pages = ep.context_pages.unsqueeze(0).to(device)
            q_ids = ep.query_ids.unsqueeze(0).to(device)
            a_ids = ep.answer_ids.unsqueeze(0).to(device)
            tp = ep.target_page_idx
            is_exact = ep.metadata["needle_type"] == "exact"
            ans_str = ep.metadata["answer_str"]
            alt_strs = [ep.metadata["full_fact_str"]] if "full_fact_str" in ep.metadata else []
            turn_emb = embed_fn(torch.cat((q_ids, a_ids), dim=1)).to(llm_dtype)

            page_texts = [tokenizer.decode(ep.context_pages[p].tolist(), skip_special_tokens=True) for p in range(num_pages)]
            query_text = tokenizer.decode(ep.query_ids.tolist(), skip_special_tokens=True)
            bm25 = BM25PageRetriever(page_texts)
            if idx_ep == 0:
                bm25_cpu_ram_bytes_list.append(float(estimate_bm25_cpu_ram_bytes(bm25)))
            bm_route = bm25.route(query_text, k_deep=2, k_skim=4, device=device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                z_book, bank, toc = _encode_book_chunked(
                    wrapper, pages, embed_fn, vocab_size, chunk_pages=32
                )
                q_emb = embed_fn(q_ids).mean(dim=1)

                lr_route, asm_lr_flat = wrapper.lslc.route_and_assemble(
                    z_book,
                    toc,
                    q_emb,
                    bank,
                    embed_fn,
                    k_deep=2,
                    k_skim=4,
                    mode="full",
                    training_ste=False,
                    query_input_ids=q_ids,
                )
                _, asm_lr_hier = wrapper.lslc.route_and_assemble(
                    z_book,
                    toc,
                    q_emb,
                    bank,
                    embed_fn,
                    k_deep=2,
                    k_skim=4,
                    mode="full",
                    training_ste=False,
                    router_out_override=lr_route,
                    hierarchical_chapter_size=chapter_size,
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
                    hierarchical_chapter_size=chapter_size,
                )
                hier_toc_budget = int(asm_bm_hier.slice_bounds["toc"][1] - asm_bm_hier.slice_bounds["toc"][0])
                _, asm_bm_trunc = wrapper.lslc.route_and_assemble(
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
                    flat_toc_max_tokens=hier_toc_budget,
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
                "raw_rag_shared_bm25": rag_prefix,
                "lslc_e2e_gpu_router_flat": asm_lr_flat.prefix_embeds.to(llm_dtype),
                "lslc_e2e_gpu_router_hier": asm_lr_hier.prefix_embeds.to(llm_dtype),
                "lslc_shared_bm25_flat": asm_bm_flat.prefix_embeds.to(llm_dtype),
                "lslc_shared_bm25_trunc_flat_ctrl": asm_bm_trunc.prefix_embeds.to(llm_dtype),
                "lslc_shared_bm25_hier": asm_bm_hier.prefix_embeds.to(llm_dtype),
            }
            ep_res: dict[str, tuple[float, float, float, float, float]] = {}
            for s_name, pref in prefixes.items():
                active_tok_counts[s_name] = int(pref.shape[1])
                st_em, ext_em, sub_em, ctok, cf1 = _eval_pref(pref, q_ids, a_ids, turn_emb, ans_str, alt_strs)
                ep_res[s_name] = (st_em, ext_em, sub_em, ctok, cf1)
                if is_exact:
                    seed_bucket[s_name]["exact_strict_em"].append(st_em)
                    seed_bucket[s_name]["exact_extracted_em"].append(ext_em)
                    seed_bucket[s_name]["exact_substr_em"].append(sub_em)
                    seed_bucket[s_name]["exact_tf_content_tok_acc"].append(ctok)
                    seed_bucket[s_name]["exact_char_f1"].append(cf1)
                else:
                    seed_bucket[s_name]["semantic_strict_em"].append(st_em)
                    seed_bucket[s_name]["semantic_extracted_em"].append(ext_em)
                    seed_bucket[s_name]["semantic_substr_em"].append(sub_em)
                    seed_bucket[s_name]["semantic_tf_content_tok_acc"].append(ctok)
                    seed_bucket[s_name]["semantic_char_f1"].append(cf1)

            h_st, h_ext, h_sub, _, _ = ep_res["lslc_shared_bm25_hier"]
            t_st, t_ext, t_sub, _, _ = ep_res["lslc_shared_bm25_trunc_flat_ctrl"]
            if is_exact:
                paired_ep_diffs["exact_strict_em"].append(h_st - t_st)
                paired_ep_diffs["exact_extracted_em"].append(h_ext - t_ext)
                paired_ep_diffs["exact_substr_em"].append(h_sub - t_sub)
            else:
                paired_ep_diffs["semantic_strict_em"].append(h_st - t_st)
                paired_ep_diffs["semantic_extracted_em"].append(h_ext - t_ext)
                paired_ep_diffs["semantic_substr_em"].append(h_sub - t_sub)

            if (idx_ep + 1) % 20 == 0:
                print(f"  [NIAH P={num_pages} seed={seed}] {idx_ep+1}/{episodes_per_seed} done", flush=True)

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
            for k in metrics_keys:
                vals = seed_bucket[s_name][k]
                per_sys[s_name][k].append(sum(vals) / max(1, len(vals)))
        for pk in paired_diff_keys:
            h_mean = sum(seed_bucket["lslc_shared_bm25_hier"][pk]) / max(1, len(seed_bucket["lslc_shared_bm25_hier"][pk]))
            t_mean = sum(seed_bucket["lslc_shared_bm25_trunc_flat_ctrl"][pk]) / max(1, len(seed_bucket["lslc_shared_bm25_trunc_flat_ctrl"][pk]))
            paired_seed_diffs[pk].append(h_mean - t_mean)

    return {
        "num_pages": num_pages,
        "context_tokens": num_pages * wrapper.cfg.page_size,
        "chapter_size": chapter_size,
        "seeds": seeds,
        "episodes_per_seed": episodes_per_seed,
        "total_episodes": len(seeds) * episodes_per_seed,
        "salience_capture_rate": _mean_std_ci(capture_rates),
        "bm25_cpu_host_ram_kib": round(sum(bm25_cpu_ram_bytes_list) / max(1, len(bm25_cpu_ram_bytes_list)) / 1024.0, 2),
        "paired_diff_hier_minus_trunc_flat": {
            "seed_level": {pk: _mean_std_ci(paired_seed_diffs[pk]) for pk in paired_diff_keys},
            "episode_level": {pk: _mean_std_ci(paired_ep_diffs[pk]) for pk in paired_diff_keys},
        },
        "retriever_recall": {
            "bm25_exact_top1": _mean_std_ci(bm_t1_ex),
            "bm25_exact_top2": _mean_std_ci(bm_t2_ex),
            "bm25_semantic_top1": _mean_std_ci(bm_t1_sem),
            "bm25_semantic_top2": _mean_std_ci(bm_t2_sem),
            "bm25_overall_top1": _mean_std_ci(bm_t1_all),
            "bm25_overall_top2": _mean_std_ci(bm_t2_all),
            "e2e_gpu_router_exact_top1": _mean_std_ci(lr_t1_ex),
            "e2e_gpu_router_exact_top2": _mean_std_ci(lr_t2_ex),
            "e2e_gpu_router_semantic_top1": _mean_std_ci(lr_t1_sem),
            "e2e_gpu_router_semantic_top2": _mean_std_ci(lr_t2_sem),
            "e2e_gpu_router_overall_top1": _mean_std_ci(lr_t1_all),
            "e2e_gpu_router_overall_top2": _mean_std_ci(lr_t2_all),
        },
        "systems": {
            s_name: {
                "active_prefix_tokens": active_tok_counts[s_name],
                **{k: _mean_std_ci(per_sys[s_name][k]) for k in metrics_keys},
            }
            for s_name in systems
        },
    }


_DIRECT_PATCH_RE = re.compile(
    r"REVISION\s+OVERRIDE\s*:\s*(?:budget|deployment\s+state)\s+updated\s+to\s+([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
_STATE_LINE_RE = re.compile(
    r"((?:STATE\s+BUDGET|DEPLOYMENT\s+STATE)\s+INITIAL\s*:\s*)([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)


def _apply_direct_text_patch(page_text: str, edit_text: str) -> tuple[str, bool]:
    """Deterministic non-LLM mutable RAG baseline: regex parse edit_text and patch key-value slot in page_text."""
    m = _DIRECT_PATCH_RE.search(edit_text)
    if not m:
        return page_text, False
    new_val = m.group(1)
    patched, n_subs = _STATE_LINE_RE.subn(rf"\g<1>{new_val}", page_text, count=1)
    return patched, (n_subs > 0)


@torch.no_grad()
def evaluate_state_override_v2(
    wrapper: FrozenQwenLSLCWrapper,
    val_pool: list[int],
    tokenizer,
    device: torch.device,
    seeds: list[int],
) -> dict[str, Any]:
    """5-way State-Override Benchmark across k=3 structured, k=5 symbolic, and k=3 paraphrased NL revisions."""
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
        gen_ids, gen_text = greedy_generate_concise(wrapper.llm, pref, q_ids, tokenizer, max_new_tokens=24)
        return compute_answer_metrics_v2(ans_logits, a_ids, gen_ids, gen_text, ans_str, tokenizer)

    def _run_suite(regime: str, num_edits: int, num_episodes_per_seed: int) -> dict[str, Any]:
        sys_names = (
            "raw_rag_immutable",
            "raw_rag_append_errata",
            "raw_rag_direct_text_patch",
            "raw_rag_in_place_llm_rewrite",
            "lslc_e2e_gpu_router_delta",
            "lslc_shared_bm25_delta",
        )
        m_keys = (
            "override_strict_em",
            "override_extracted_em",
            "override_substr_em",
            "override_tf_content_tok_acc",
            "override_char_f1",
            "static_retention_strict_em",
            "static_retention_extracted_em",
            "static_retention_substr_em",
            "static_retention_tf_tok_acc",
        )
        per_sys = {s: {k: [] for k in m_keys} for s in sys_names}
        edit_ms_lslc, edit_ms_rw, edit_ms_app, edit_ms_patch, rw_dec_toks, patch_hit_rates = [], [], [], [], [], []
        inv_diffs = []
        toc_only_diffs = []
        lr_recall_t2 = []
        rw_cache: dict[tuple[str, str], tuple[list[int], int, float]] = {}

        for seed in seeds:
            if regime == "k3_structured":
                ds = list(
                    LivingBookStateOverrideDataset(
                        num_episodes=num_episodes_per_seed,
                        num_pages=4,
                        page_size=wrapper.cfg.page_size,
                        num_edits=3,
                        tokenizer=tokenizer,
                        seed=seed,
                        background_token_pool=val_pool,
                    )
                )
            elif regime == "k5_symbolic":
                ds = build_k5_symbolic_with_static_episodes(
                    tokenizer, val_pool, num_episodes=num_episodes_per_seed, num_pages=4, num_edits=5, seed=seed
                )
            else:
                ds = build_paraphrased_nl_override_episodes(
                    tokenizer, val_pool, num_episodes=num_episodes_per_seed, num_pages=4, num_edits=3, seed=seed
                )

            seed_b = {s: {k: [] for k in m_keys} for s in sys_names}
            s_ms_lslc, s_ms_rw, s_ms_app, s_ms_patch, s_rw_toks, s_patch_hits = [], [], [], [], [], []
            s_lr_hits = []

            for ep in ds:
                pages = ep.context_pages.unsqueeze(0).to(device)
                q_ids = ep.query_ids.unsqueeze(0).to(device)
                a_ids = ep.answer_ids.unsqueeze(0).to(device)
                ans_str = ep.metadata["answer_str"]
                is_ov = bool(ep.metadata["is_overridden_query"])
                tp = ep.target_page_idx

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

                # 3. Direct Text-Patch + BM25 Re-Index RAG (deterministic non-LLM mutable RAG)
                patch_pages = [list(pt) for pt in base_pages]
                patch_texts = list(base_texts)
                bm_patch = BM25PageRetriever(list(base_texts))
                for p_idx, e_ids in (ep.edit_pages or []):
                    e_str = tokenizer.decode(e_ids.tolist(), skip_special_tokens=True)
                    t0 = time.perf_counter()
                    new_txt, matched = _apply_direct_text_patch(patch_texts[p_idx], e_str)
                    if matched:
                        patch_texts[p_idx] = new_txt
                        new_toks = tokenizer.encode(new_txt, add_special_tokens=False)[: wrapper.cfg.page_size]
                        if len(new_toks) < wrapper.cfg.page_size:
                            new_toks = new_toks + patch_pages[p_idx][len(new_toks) : wrapper.cfg.page_size]
                        patch_pages[p_idx] = new_toks
                        bm_patch.update_page(p_idx, new_txt)
                    s_ms_patch.append((time.perf_counter() - t0) * 1000.0)
                    s_patch_hits.append(1.0 if matched else 0.0)
                m_patch = _eval_raw_rag(patch_pages, bm_patch, q_ids, a_ids, ans_str)

                # 4. In-Place LLM-Rewrite RAG
                rw_pages = [list(pt) for pt in base_pages]
                header_texts = {
                    p: tokenizer.decode(rw_pages[p][:32], skip_special_tokens=True)
                    for p in range(len(rw_pages))
                }
                bm_rw = BM25PageRetriever(list(base_texts))
                for p_idx, e_ids in (ep.edit_pages or []):
                    e_str = tokenizer.decode(e_ids.tolist(), skip_special_tokens=True)
                    cache_key = (header_texts[p_idx], e_str)
                    if cache_key in rw_cache:
                        new_hdr_ids, dec_cnt, rw_ms = rw_cache[cache_key]
                    else:
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
                        torch.cuda.synchronize()
                        rw_ms = (time.perf_counter() - t0) * 1000.0
                        rw_cache[cache_key] = (list(new_hdr_ids), int(dec_cnt), rw_ms)
                    header_texts[p_idx] = tokenizer.decode(new_hdr_ids, skip_special_tokens=True)
                    if len(new_hdr_ids) < 32:
                        rw_pages[p_idx] = new_hdr_ids + rw_pages[p_idx][32:]
                    else:
                        rw_pages[p_idx] = new_hdr_ids[:32] + rw_pages[p_idx][32:]
                    bm_rw.update_page(p_idx, tokenizer.decode(rw_pages[p_idx], skip_special_tokens=True))
                    s_ms_rw.append(rw_ms)
                    s_rw_toks.append(float(dec_cnt))
                m_rw = _eval_raw_rag(rw_pages, bm_rw, q_ids, a_ids, ans_str)

                # 5 & 6. LSLC In-Place Convex DeltaUpdater (Pure E2E GPU Router & Shared BM25)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    z_book, bank, toc0 = wrapper.lslc.encode_book(
                        pages, embed_fn, use_exact_bank=True, vocab_size=vocab_size
                    )
                    q_emb = embed_fn(q_ids).mean(dim=1)
                    _, asm_before = wrapper.lslc.route_and_assemble(
                        z_book,
                        toc0,
                        q_emb,
                        bank,
                        embed_fn,
                        k_deep=2,
                        k_skim=2,
                        mode="full",
                        training_ste=False,
                        query_input_ids=q_ids,
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

                q_text = tokenizer.decode(q_ids[0].tolist(), skip_special_tokens=True)
                bm_route_lslc = bm_imm.route(q_text, k_deep=2, k_skim=2, device=device)

                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    toc_after = wrapper.lslc.projector.project_toc(z_book)
                    lr_route_after, asm_e2e_after = wrapper.lslc.route_and_assemble(
                        z_book,
                        toc_after,
                        q_emb,
                        bank,
                        embed_fn,
                        k_deep=2,
                        k_skim=2,
                        mode="full",
                        training_ste=False,
                        query_input_ids=q_ids,
                    )
                    _, asm_bm_after = wrapper.lslc.route_and_assemble(
                        z_book,
                        toc_after,
                        q_emb,
                        bank,
                        embed_fn,
                        k_deep=2,
                        k_skim=2,
                        mode="full",
                        training_ste=False,
                        router_out_override=bm_route_lslc,
                    )

                s_lr_hits.append(1.0 if tp in lr_route_after.deep_idx[0].tolist() else 0.0)
                imm_l = asm_before.immutable_prefix_len
                toc_only_diffs.append(float((toc_after - toc0).abs().max().item()))
                inv_diffs.append(
                    float((asm_e2e_after.prefix_embeds[:, :imm_l] - asm_before.prefix_embeds[:, :imm_l]).abs().max().item())
                )

                turn_emb = embed_fn(torch.cat((q_ids, a_ids), dim=1)).to(llm_dtype)

                def _eval_lslc_pref(pref_t: torch.Tensor):
                    p_cast = pref_t.to(llm_dtype)
                    inp = torch.cat((p_cast, turn_emb), dim=1)
                    logits = wrapper.llm(inputs_embeds=inp).logits
                    start = p_cast.shape[1] + q_ids.shape[1] - 1
                    ans_l = logits[:, start : start + a_ids.shape[1], :]
                    g_ids, g_txt = greedy_generate_concise(wrapper.llm, p_cast, q_ids, tokenizer, max_new_tokens=24)
                    return compute_answer_metrics_v2(ans_l, a_ids, g_ids, g_txt, ans_str, tokenizer)

                m_lslc_e2e = _eval_lslc_pref(asm_e2e_after.prefix_embeds)
                m_lslc_bm = _eval_lslc_pref(asm_bm_after.prefix_embeds)

                for s_name, (st_em, ext_em, sub_em, tok_v, f1_v) in (
                    ("raw_rag_immutable", m_imm),
                    ("raw_rag_append_errata", m_app),
                    ("raw_rag_direct_text_patch", m_patch),
                    ("raw_rag_in_place_llm_rewrite", m_rw),
                    ("lslc_e2e_gpu_router_delta", m_lslc_e2e),
                    ("lslc_shared_bm25_delta", m_lslc_bm),
                ):
                    if is_ov:
                        seed_b[s_name]["override_strict_em"].append(st_em)
                        seed_b[s_name]["override_extracted_em"].append(ext_em)
                        seed_b[s_name]["override_substr_em"].append(sub_em)
                        seed_b[s_name]["override_tf_content_tok_acc"].append(tok_v)
                        seed_b[s_name]["override_char_f1"].append(f1_v)
                    else:
                        seed_b[s_name]["static_retention_strict_em"].append(st_em)
                        seed_b[s_name]["static_retention_extracted_em"].append(ext_em)
                        seed_b[s_name]["static_retention_substr_em"].append(sub_em)
                        seed_b[s_name]["static_retention_tf_tok_acc"].append(tok_v)

            for s_name in sys_names:
                for k in m_keys:
                    vals = seed_b[s_name][k]
                    if vals:
                        per_sys[s_name][k].append(sum(vals) / len(vals))
            edit_ms_lslc.append(sum(s_ms_lslc) / max(1, len(s_ms_lslc)))
            edit_ms_rw.append(sum(s_ms_rw) / max(1, len(s_ms_rw)))
            edit_ms_app.append(sum(s_ms_app) / max(1, len(s_ms_app)))
            edit_ms_patch.append(sum(s_ms_patch) / max(1, len(s_ms_patch)))
            rw_dec_toks.append(sum(s_rw_toks) / max(1, len(s_rw_toks)))
            patch_hit_rates.append(sum(s_patch_hits) / max(1, len(s_patch_hits)))
            lr_recall_t2.append(sum(s_lr_hits) / max(1, len(s_lr_hits)))
            print(
                f"  [STATE {regime} seed={seed}] "
                f"LSLC_E2E={per_sys['lslc_e2e_gpu_router_delta']['override_strict_em'][-1]:.3f} "
                f"(Static={per_sys['lslc_e2e_gpu_router_delta']['static_retention_strict_em'][-1]:.3f}, {edit_ms_lslc[-1]:.2f}ms) | "
                f"DirectPatch={per_sys['raw_rag_direct_text_patch']['override_strict_em'][-1]:.3f} ({edit_ms_patch[-1]:.2f}ms) | "
                f"LLM-RW={per_sys['raw_rag_in_place_llm_rewrite']['override_strict_em'][-1]:.3f} ({edit_ms_rw[-1]:.1f}ms)",
                flush=True,
            )

        return {
            "regime": regime,
            "num_edits": num_edits,
            "seeds": seeds,
            "total_episodes": len(seeds) * num_episodes_per_seed,
            "lslc_max_toc_only_diff": max(toc_only_diffs) if toc_only_diffs else 0.0,
            "lslc_max_toc_prefix_diff": max(inv_diffs) if inv_diffs else 0.0,
            "lslc_e2e_gpu_router_top2_recall": _mean_std_ci(lr_recall_t2),
            "direct_patch_regex_match_rate": _mean_std_ci(patch_hit_rates),
            "edit_cost": {
                "lslc_ms_per_edit": _mean_std_ci(edit_ms_lslc),
                "rag_direct_text_patch_ms_per_edit": _mean_std_ci(edit_ms_patch),
                "rag_append_errata_ms_per_edit": _mean_std_ci(edit_ms_app),
                "rag_llm_rewrite_block_ms_per_edit": _mean_std_ci(edit_ms_rw),
                "rag_llm_rewrite_block_decode_tokens": _mean_std_ci(rw_dec_toks),
                "rag_immutable_ms_per_edit": {"mean": 0.0, "std": 0.0, "ci95": 0.0},
            },
            "systems": {
                s_name: {k: _mean_std_ci(per_sys[s_name][k]) for k in m_keys if per_sys[s_name][k]}
                for s_name in sys_names
            },
        }

    try:
        k3_struct = _run_suite("k3_structured", num_edits=3, num_episodes_per_seed=36)
        k5_sym = _run_suite("k5_symbolic", num_edits=5, num_episodes_per_seed=30)
        k3_nl = _run_suite("k3_paraphrased_nl", num_edits=3, num_episodes_per_seed=30)
        full_page_rw = profile_full_page_llm_rewrite_latency(wrapper.llm, tokenizer, val_pool, device, trials=5)
    finally:
        wrapper.llm.config.use_cache = prev_cache

    return {
        "k3_structured_numeric_overrides": k3_struct,
        "k5_structured_symbolic_overrides": k5_sym,
        "k3_paraphrased_natural_language_overrides": k3_nl,
        "full_page_256tok_llm_rewrite_latency": full_page_rw,
    }


def _configure_geometry_variant(wrapper: FrozenQwenLSLCWrapper, variant: str) -> None:
    """Configure LSLCSystem for iso-param (40.42M), iso-byte (12.125 KiB/p), iso-ExactBank geometry ablation."""
    if variant == "3d_spatial":
        return

    if variant in ("1d_flat", "1d_3layer"):
        # Disable 3D rotary geometry (replace with identity/1D flat order) and zero 2D depthwise Conv2d mixing
        for blk in list(wrapper.lslc.encoder.self_blocks) + [wrapper.lslc.delta_updater.self_block]:
            blk.rope_3d.forward = lambda x: x  # type: ignore[method-assign]
            with torch.no_grad():
                blk.dw_conv.weight.zero_()
                if blk.dw_conv.bias is not None:
                    blk.dw_conv.bias.zero_()
                blk.pw_conv.weight.zero_()
                if blk.pw_conv.bias is not None:
                    blk.pw_conv.bias.zero_()
            for p in list(blk.dw_conv.parameters()) + list(blk.pw_conv.parameters()):
                p.requires_grad_(False)

        # Remove 2D bottom-row sticky-note spatial prior (uniform 1D prior across all 16 slots)
        wrapper.lslc.delta_updater.sticky_spatial_prior.fill_(-1.4)

    if variant == "1d_flat":
        # Collapse layer disentanglement l in {0,1,2} so all 3 layers share a single flat 1D latent pool
        orig_enc_fwd = wrapper.lslc.encoder.forward

        def _flat_enc_fwd(page_embeds: torch.Tensor) -> torch.Tensor:
            z = orig_enc_fwd(page_embeds)
            z_mean = z.mean(dim=1, keepdim=True).expand_as(z).contiguous()
            return z_mean

        wrapper.lslc.encoder.forward = _flat_enc_fwd  # type: ignore[method-assign]

        orig_delta_fwd = wrapper.lslc.delta_updater.forward

        def _flat_delta_fwd(
            z_page: torch.Tensor, edit_embeds: torch.Tensor, allow_entity_update: bool | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            z_new, gates = orig_delta_fwd(z_page, edit_embeds, allow_entity_update=True)
            # In 1D-Flat (no layer separation), updating the latent slot mutates the shared representation across all strata
            z_shared = (0.34 * z_new[:, 0] + 0.33 * z_new[:, 1] + 0.33 * z_new[:, 2]).unsqueeze(1).expand_as(z_new).contiguous()
            return z_shared, gates

        wrapper.lslc.delta_updater.forward = _flat_delta_fwd  # type: ignore[method-assign]


def evaluate_iso_budget_1d_vs_3d_ablation(
    llm,
    tokenizer,
    cfg: LSLCConfig,
    train_pool: list[int],
    val_pool: list[int],
    device: torch.device,
) -> dict[str, Any]:
    """Controlled Iso-Param (40.42M) / Iso-Byte (12.125 KiB/p) / Iso-Step / Iso-ExactBank 1D-Flat vs 1D-3Layer vs 3D-Spatial."""
    embed_fn = llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    amp_dtype, _ = select_amp_dtype(device)
    vocab_size = len(embed_fn.weight)

    train_prose = list(ChunkedTextPageDataset(train_pool[:16384], num_pages=4, page_size=cfg.page_size, turn_len=64))[:80]
    val_prose = ChunkedTextPageDataset(val_pool[:8192], num_pages=4, page_size=cfg.page_size, turn_len=64)
    train_niah = build_concise_niah_episodes(tokenizer, train_pool, 80, 4, cfg.page_size, seed=42)
    val_niah = build_concise_niah_episodes(tokenizer, val_pool, 30, 4, cfg.page_size, seed=9999)
    train_k3 = list(
        LivingBookStateOverrideDataset(
            num_episodes=60, num_pages=4, page_size=cfg.page_size, num_edits=3, tokenizer=tokenizer, seed=42, background_token_pool=train_pool
        )
    )
    val_k3 = list(
        LivingBookStateOverrideDataset(
            num_episodes=24, num_pages=4, page_size=cfg.page_size, num_edits=3, tokenizer=tokenizer, seed=9999, background_token_pool=val_pool
        )
    )

    results: dict[str, Any] = {}
    for variant in ("1d_flat", "1d_3layer", "3d_spatial"):
        torch.manual_seed(42)
        w_var = FrozenQwenLSLCWrapper(cfg, llm)
        w_var.lslc.to(device)
        total_arch_params = sum(p.numel() for p in w_var.lslc.parameters())
        _configure_geometry_variant(w_var, variant)

        print(f"  [ABLATION {variant}] training 160 base + 80 delta steps (iso-byte 12.125 KiB/p + ExactBank)...", flush=True)
        _ = train_stage(
            f"ABL_{variant}_BASE",
            w_var,
            train_prose + train_niah + [ep for ep in train_k3 if not ep.metadata["is_overridden_query"]],
            steps=160,
            grad_accum=2,
            lr=3e-4,
            device=device,
            use_exact_bank=True,
            distill_salience=True,
            run_matryoshka=True,
        )
        frozen_abl = []
        for name_p, p in w_var.lslc.named_parameters():
            if not (name_p.startswith("delta_updater.") or name_p.startswith("router.")) and p.requires_grad:
                p.requires_grad_(False)
                frozen_abl.append(p)
        _ = train_stage(
            f"ABL_{variant}_DELTA",
            w_var,
            train_k3,
            steps=80,
            grad_accum=2,
            lr=3e-4,
            device=device,
            use_exact_bank=True,
            distill_salience=False,
            run_matryoshka=False,
            use_edits=True,
        )
        for p in frozen_abl:
            p.requires_grad_(True)

        w_var.lslc.eval()
        ppl_full = evaluate_continuation_ppl(w_var, val_prose, device=device, mode="lslc_full_exact")
        ppl_skim = evaluate_continuation_ppl(w_var, val_prose, device=device, mode="skim_only")

        # Evaluate NIAH (Exact & Semantic with Strict, Extracted & Substring EM)
        ex_st, ex_ext, ex_sub, ex_tok, sem_st, sem_ext, sem_sub, sem_skim_sub = [], [], [], [], [], [], [], []
        for ep in val_niah:
            pages = ep.context_pages.unsqueeze(0).to(device)
            q_ids = ep.query_ids.unsqueeze(0).to(device)
            a_ids = ep.answer_ids.unsqueeze(0).to(device)
            ans_str = ep.metadata["answer_str"]
            alt_strs = [ep.metadata["full_fact_str"]] if "full_fact_str" in ep.metadata else []
            turn_emb = embed_fn(torch.cat((q_ids, a_ids), dim=1)).to(llm_dtype)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                z_b, bank, toc = w_var.lslc.encode_book(pages, embed_fn, use_exact_bank=True, vocab_size=vocab_size)
                q_emb = embed_fn(q_ids).mean(dim=1)
                _, asm_f = w_var.lslc.route_and_assemble(
                    z_b, toc, q_emb, bank, embed_fn, mode="full", training_ste=False, query_input_ids=q_ids
                )
                _, asm_sk = w_var.lslc.route_and_assemble(
                    z_b, toc, q_emb, None, embed_fn, mode="skim_only", training_ste=False
                )

            pref_f = asm_f.prefix_embeds.to(llm_dtype)
            inp_f = torch.cat((pref_f, turn_emb), dim=1)
            logits_f = llm(inputs_embeds=inp_f).logits
            start_f = pref_f.shape[1] + q_ids.shape[1] - 1
            ans_l_f = logits_f[:, start_f : start_f + a_ids.shape[1], :]
            g_ids_f, g_txt_f = greedy_generate_concise(llm, pref_f, q_ids, tokenizer, max_new_tokens=24)
            st_v, ext_v, sub_v, tok_v, _ = compute_answer_metrics_v2(ans_l_f, a_ids, g_ids_f, g_txt_f, ans_str, tokenizer, alt_valid_strs=alt_strs)

            if ep.metadata["needle_type"] == "exact":
                ex_st.append(st_v)
                ex_ext.append(ext_v)
                ex_sub.append(sub_v)
                ex_tok.append(tok_v)
            else:
                sem_st.append(st_v)
                sem_ext.append(ext_v)
                sem_sub.append(sub_v)
                pref_sk = asm_sk.prefix_embeds.to(llm_dtype)
                _, g_txt_sk = greedy_generate_concise(llm, pref_sk, q_ids, tokenizer, max_new_tokens=24)
                sem_skim_sub.append(1.0 if ans_str.lower() in g_txt_sk.lower() else 0.0)

        # Evaluate k=3 State Override & Static Retention & TOC Prefix Invariance
        ov_st, ov_ext, ov_sub, st_ret_st, st_ret_ext, st_ret_sub, inv_diffs = [], [], [], [], [], [], []
        for ep in val_k3:
            pages = ep.context_pages.unsqueeze(0).to(device)
            q_ids = ep.query_ids.unsqueeze(0).to(device)
            a_ids = ep.answer_ids.unsqueeze(0).to(device)
            ans_str = ep.metadata["answer_str"]
            turn_emb = embed_fn(torch.cat((q_ids, a_ids), dim=1)).to(llm_dtype)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                z_b, bank, toc0 = w_var.lslc.encode_book(pages, embed_fn, use_exact_bank=True, vocab_size=vocab_size)
                q_emb = embed_fn(q_ids).mean(dim=1)
                _, asm_0 = w_var.lslc.route_and_assemble(
                    z_b, toc0, q_emb, bank, embed_fn, mode="full", training_ste=False, query_input_ids=q_ids
                )
                for p_idx, e_ids in (ep.edit_pages or []):
                    z_b, _, bank = w_var.lslc.apply_page_delta(
                        z_b, torch.tensor([p_idx], device=device), e_ids.unsqueeze(0).to(device), embed_fn, exact_bank=bank, vocab_size=vocab_size
                    )
                toc1 = w_var.lslc.projector.project_toc(z_b)
                _, asm_1 = w_var.lslc.route_and_assemble(
                    z_b, toc1, q_emb, bank, embed_fn, mode="full", training_ste=False, query_input_ids=q_ids
                )

            toc_diff = float((toc1 - toc0).abs().max().item())
            inv_diffs.append(toc_diff)

            pref_1 = asm_1.prefix_embeds.to(llm_dtype)
            inp_1 = torch.cat((pref_1, turn_emb), dim=1)
            logits_1 = llm(inputs_embeds=inp_1).logits
            start_1 = pref_1.shape[1] + q_ids.shape[1] - 1
            ans_l_1 = logits_1[:, start_1 : start_1 + a_ids.shape[1], :]
            g_ids_1, g_txt_1 = greedy_generate_concise(llm, pref_1, q_ids, tokenizer, max_new_tokens=24)
            st_v, ext_v, sub_v, _, _ = compute_answer_metrics_v2(ans_l_1, a_ids, g_ids_1, g_txt_1, ans_str, tokenizer)
            if ep.metadata["is_overridden_query"]:
                ov_st.append(st_v)
                ov_ext.append(ext_v)
                ov_sub.append(sub_v)
            else:
                st_ret_st.append(st_v)
                st_ret_ext.append(ext_v)
                st_ret_sub.append(sub_v)

        results[variant] = {
            "total_module_params": total_arch_params,
            "page_latent_bytes": 12288,
            "sparse_exact_bank_bytes_per_page": 128,
            "wikitext2_ppl_full": ppl_full,
            "wikitext2_ppl_skim_2x2": ppl_skim,
            "exact_niah_strict_em": round(sum(ex_st) / max(1, len(ex_st)), 4),
            "exact_niah_extracted_em": round(sum(ex_ext) / max(1, len(ex_ext)), 4),
            "exact_niah_substr_em": round(sum(ex_sub) / max(1, len(ex_sub)), 4),
            "exact_niah_tf_tok_acc": round(sum(ex_tok) / max(1, len(ex_tok)), 4),
            "semantic_niah_strict_em": round(sum(sem_st) / max(1, len(sem_st)), 4),
            "semantic_niah_extracted_em": round(sum(sem_ext) / max(1, len(sem_ext)), 4),
            "semantic_niah_substr_em": round(sum(sem_sub) / max(1, len(sem_sub)), 4),
            "semantic_niah_skim_only_em": round(sum(sem_skim_sub) / max(1, len(sem_skim_sub)), 4),
            "k3_override_strict_em": round(sum(ov_st) / max(1, len(ov_st)), 4),
            "k3_override_extracted_em": round(sum(ov_ext) / max(1, len(ov_ext)), 4),
            "k3_override_substr_em": round(sum(ov_sub) / max(1, len(ov_sub)), 4),
            "static_l0_retention_strict_em": round(sum(st_ret_st) / max(1, len(st_ret_st)), 4),
            "static_l0_retention_extracted_em": round(sum(st_ret_ext) / max(1, len(st_ret_ext)), 4),
            "static_l0_retention_substr_em": round(sum(st_ret_sub) / max(1, len(st_ret_sub)), 4),
            "max_toc_prefix_diff": round(max(inv_diffs) if inv_diffs else 0.0, 6),
        }
        print(f"  [ABLATION {variant} DONE] {results[variant]}", flush=True)
        del w_var
        torch.cuda.empty_cache()

    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--out-json", type=str, default="/content/lslc_results/v2_complete_summary.json")
    parser.add_argument("--load-checkpoint", type=str, default="")
    parser.add_argument("--prev-json", type=str, default="")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    if not torch.cuda.is_bf16_supported():
        torch.backends.cudnn.enabled = False
    props = torch.cuda.get_device_properties(0)
    print(f"=== LSLC V2.1.0 COMPLETE HARDENING SUITE on {props.name} ===", flush=True)

    cfg = LSLCConfig.from_model_name(args.model, lr=3e-4)
    llm, tokenizer = load_frozen_qwen3(cfg, load_in_4bit=False, device=device)
    wrapper = FrozenQwenLSLCWrapper(cfg, llm)
    wrapper.lslc.to(device)

    train_pool, val_pool, corpus_src = _load_wikitext_pools(tokenizer)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prev_data: dict[str, Any] = {}
    if args.prev_json and Path(args.prev_json).is_file():
        prev_data = json.loads(Path(args.prev_json).read_text(encoding="utf-8"))

    summary: dict[str, Any] = {
        "version": "2.1.0",
        "model": args.model,
        "gpu": props.name,
        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
        "corpus": corpus_src,
    }

    def _flush() -> None:
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Stage 1: Train or Load Unified Adapter Checkpoint
    from safetensors.torch import load_file, save_file

    if args.load_checkpoint and Path(args.load_checkpoint).is_file():
        print(f"--- Stage 1: Loading pre-trained unified checkpoint from {args.load_checkpoint} ---", flush=True)
        sd = load_file(args.load_checkpoint, device="cpu")
        wrapper.lslc.load_state_dict({k: v.to(device=device, dtype=torch.float32) for k, v in sd.items()}, strict=True)
        wrapper.lslc.eval()
        summary["training"] = prev_data.get(
            "training",
            {"loaded_checkpoint": args.load_checkpoint, "trainable_params": wrapper.lslc.trainable_parameter_count()},
        )
    else:
        print("--- Stage 1: Training Unified LSLC v2.1.0 Adapter Checkpoint ---", flush=True)
        summary["training"] = train_unified_v2_checkpoint(wrapper, train_pool, tokenizer, device)
        clean_sd = {k: v.detach().cpu().contiguous().to(torch.bfloat16) for k, v in wrapper.lslc.state_dict().items()}
        st_path = out_path.parent / "lslc_qwen3_1.7b_adapter.safetensors"
        save_file(
            clean_sd,
            str(st_path),
            metadata={
                "format": "pt",
                "architecture": "LayeredSpatialLatentCache",
                "version": "2.1.0",
                "base_model": args.model,
                "trainable_parameters": str(wrapper.lslc.trainable_parameter_count()),
                "suite": "lslc_v2_complete",
            },
        )
        print(f"Exported unified checkpoint -> {st_path}", flush=True)
    _flush()

    # Stage 2: 5-Way Multi-Turn State-Override Benchmark (Structured k=3, Structured k=5, Paraphrased NL k=3)
    if args.skip_stage2 and "state_override_v2" in prev_data:
        print("--- Stage 2: Reusing verified L4 State-Override v2.1.0 results from prev_json ---", flush=True)
        summary["state_override_v2"] = prev_data["state_override_v2"]
    else:
        print("--- Stage 2: 5-Way State Overrides (Structured k=3/k=5 + Paraphrased NL k=3 + Static l=0 Retention) ---", flush=True)
        summary["state_override_v2"] = evaluate_state_override_v2(
            wrapper, val_pool, tokenizer, device, seeds=[42, 1337, 9999]
        )
    _flush()

    # Stage 3: 32K (P=128, N=180) Multi-Seed NIAH + Pure E2E GPU Router vs Hybrid BM25 + Truncated Flat TOC Ablation
    print("--- Stage 3: 32K (P=128, N=180) NIAH (Pure E2E GPU Router vs BM25 + Hierarchy vs Truncated-TOC Ablation) ---", flush=True)
    summary["niah_32k_v2"] = evaluate_long_context_niah_v2(
        wrapper, val_pool, tokenizer, device, num_pages=128, seeds=[42, 1337, 9999], episodes_per_seed=60, chapter_size=16
    )
    _flush()

    # Stage 4: 128K (P=512, N=60 across 3 seeds) Long-Range Retrieval Quality Evaluation
    print("--- Stage 4: 128K (P=512, N=60) Long-Range NIAH Quality Benchmark ---", flush=True)
    summary["niah_128k_v2"] = evaluate_long_context_niah_v2(
        wrapper, val_pool, tokenizer, device, num_pages=512, seeds=[42, 1337, 9999], episodes_per_seed=20, chapter_size=16
    )
    _flush()

    # Stage 5: Live Memory & Latency Scaling (32K, 128K, 256K)
    if "memory_and_latency_scaling" in prev_data:
        print("--- Stage 5: Reusing verified L4 Memory & Latency Scaling from prev_json ---", flush=True)
        summary["memory_and_latency_scaling"] = prev_data["memory_and_latency_scaling"]
    else:
        print("--- Stage 5: Live Memory & Latency Scaling (32K, 128K, 256K) ---", flush=True)
        summary["memory_and_latency_scaling"] = profile_memory_and_latency_scaling(wrapper, val_pool, device)
    _flush()

    # Stage 6: Controlled Iso-Param / Iso-Byte / Iso-Step 1D-Flat vs 1D-3Layer vs 3D-Spatial Ablation
    del wrapper
    torch.cuda.empty_cache()
    if args.skip_ablation and "iso_budget_1d_vs_3d_ablation" in prev_data:
        print("--- Stage 6: Reusing verified L4 Iso-Budget 1D-vs-3D Ablation from prev_json ---", flush=True)
        summary["iso_budget_1d_vs_3d_ablation"] = prev_data["iso_budget_1d_vs_3d_ablation"]
    else:
        print("--- Stage 6: Controlled Iso-Budget 1D-Flat vs 1D-3Layer vs 3D-Spatial Ablation ---", flush=True)
        summary["iso_budget_1d_vs_3d_ablation"] = evaluate_iso_budget_1d_vs_3d_ablation(
            llm, tokenizer, cfg, train_pool, val_pool, device
        )
    _flush()

    print(f"=== SAVED V2.1.0 COMPLETE SUMMARY -> {out_path} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
