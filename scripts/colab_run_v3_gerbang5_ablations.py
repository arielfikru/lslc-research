#!/usr/bin/env python3
"""Gerbang 5: Mechanistic Ablations, True KV/Latent Baselines, Fair Gate-2 Formatting & Gate-3 Coding N=24.

Supports `--mock-llm` (0 CU local CPU verification) and modular `--tasks`:
  1. `ablation_2x2_edit`: 2x2 matrix `continuous_delta_z` (ON/OFF) x `discrete_exact_bank` (ON/OFF)
     on BOTH (a) verbatim post-edit controls and (b) implicit/inferential post-edit semantic queries
     where the gold answer NEVER appears verbatim in the edit sentence (N>=48, 95% CI, 100% disjoint).
  2. `ablation_1d_vs_3d`: Iso-token & iso-parameter 1D-3Layer flat compressor vs 3D-Spatial Pyramid
     (N>=48, 95% CI, prefix-cache invariance audit, scaling across P=8/16/32 pages).
  3. `true_kv_baselines`: True INT4 `past_key_values` quantization (NOT inputs_embeds), True
     attention-weight H2O/SnapKV KV eviction, and ICAE-style 1D latent autoencoder baseline on
     iso-token budget & equal training steps vs Full BF16 KV and LSLC v3.
  4. `fair_g2_and_coding_n24`:
     - Gate 2 Fair Formatting: identical 2-shot concise format instruction across ALL systems
       + neutral `strict_em`, `normalized_numeric_em` ("$29,000,000" == "29M"), and `f1`.
     - Gate 3 Coding Scale-Up (N=24): 24 cross-file scenarios on 65-file (~22.3K tok) repos,
       ZERO explicit path hints in query text, transparent initial repo scan counted in `files_read`,
       and dynamic margin-gated `k_deep`.
"""

from __future__ import annotations

import argparse
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
for p_str in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from colab_run_m2 import _load_wikitext_pools  # noqa: E402
from colab_run_m3 import (  # noqa: E402
    compute_nll_and_tok_acc_from_kv,
    compute_real_attention_scores_and_kv,
    evict_past_key_values_h2o_attention,
    greedy_generate_from_kv_cache,
    quantize_past_key_values_int4,
)
from colab_run_rebuttal_v2 import BM25PageRetriever  # noqa: E402
from colab_run_v3_gerbang3_coding import (  # noqa: E402
    _MockCodeTokenizer,
    _aggregate_system_rows,
    _build_edit_prompt,
    _build_homonym_prompt,
    _build_repo_pages,
    _extract_python_code_block,
    _greedy_generate_code_until_fence,
    _load_lslc_checkpoint,
    _lslc_route_and_assemble_excluding,
    _pad_or_chunk_tokens,
    _should_stop_code_gen,
    _should_stop_homonym_gen,
    _strip_think_blocks,
)
from colab_run_v3_gerbang4_community import (  # noqa: E402
    _USER_OPEN_STR,
    _PlugInBiEncoderTOC,
    _bootstrap_ci95,
    _encode_book_chunked,
    _encode_unpadded_rag_ids,
    _format_qa_prompt,
    _get_special_token_ids,
    _greedy_generate_qa,
    _normalize_qa_text,
    _peak_vram_mb,
    _reset_peak_vram,
    _sync_device,
    _wilson_ci95,
    compute_community_qa_metrics,
)
from lslc import (  # noqa: E402
    AdversarialLivingBookDataset,
    ChunkedTextPageDataset,
    CodingAgentHarness,
    FrozenQwenLSLCWrapper,
    LSLCConfig,
    LSLCSystem,
    MigrationTask,
    RouterOutput,
    SparseExactBank,
    SyntheticExactSemanticNIAHDataset,
    TinyMockCausalLM,
    estimate_int4_kv_bytes,
    extract_contextual_edit_embeds,
    load_frozen_qwen3,
    select_amp_dtype,
)
from lslc.coding_benchmark import _EVAL_SPECS, _MigrationSpec, _build_task_from_spec  # noqa: E402
from lslc.datasets import _encode_text_or_mock, _sample_background_page  # noqa: E402
from lslc.encoder import PerceiverCrossAttentionBlock, _FeedForward  # noqa: E402


# ---------------------------------------------------------------------------
# Shared Helpers & Neutral Numeric Canonicalization
# ---------------------------------------------------------------------------


def canonicalize_numeric_answer(text: str) -> str:
    """Neutral canonicalizer unifying equivalent numeric expressions (e.g. '$29,000,000' == '29M' == '29 million')."""
    s = _strip_think_blocks(text).strip().lower()
    s = s.replace("$", "").replace("usd", "").replace(",", "").strip()
    s = re.sub(r"^(?:the\s+answer\s+is|answer\s*[:=])\s*", "", s).strip()
    s = s.rstrip(" .,:;\"'`")

    # Match word multipliers: "29 million", "29.5 billion", "400 thousand"
    m_word = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(million|billion|thousand)", s)
    if m_word:
        num = float(m_word.group(1))
        unit = m_word.group(2)
        suffix = {"thousand": "k", "million": "m", "billion": "b"}[unit]
        num_str = str(int(num)) if num.is_integer() else f"{num:g}"
        return f"{num_str}{suffix}"

    # Match compact suffixes: "29m", "400k", "1.5b"
    m_short = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmb])", s)
    if m_short:
        num = float(m_short.group(1))
        num_str = str(int(num)) if num.is_integer() else f"{num:g}"
        return f"{num_str}{m_short.group(2)}"

    # Match raw expanded numbers: "29000000" -> "29m", "400000" -> "400k"
    m_raw = re.fullmatch(r"(\d+(?:\.\d+)?)", s)
    if m_raw:
        num = float(m_raw.group(1))
        if num >= 1_000_000_000 and (num / 1_000_000_000).is_integer():
            return f"{int(num // 1_000_000_000)}b"
        if num >= 1_000_000 and (num / 1_000_000).is_integer():
            return f"{int(num // 1_000_000)}m"
        if num >= 1_000 and (num / 1_000).is_integer():
            return f"{int(num // 1_000)}k"
        return str(int(num)) if num.is_integer() else f"{num:g}"

    return _normalize_qa_text(s)


def _summarize_metric_rows(rows: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, Any]:
    n = len(rows)
    out: dict[str, Any] = {"num_episodes": n, "ci95": {}}
    for k in keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        if not vals:
            continue
        mean_v = round(sum(vals) / len(vals), 4)
        out[k] = mean_v
        is_binary = all(v in (0.0, 1.0) for v in vals)
        out["ci95"][k] = _wilson_ci95(mean_v, len(vals)) if is_binary else _bootstrap_ci95(vals)
    return out


# ---------------------------------------------------------------------------
# Pillar 5.1: 2x2 Edit Ablation on Verbatim vs Implicit/Inferential Queries
# ---------------------------------------------------------------------------


def build_2x2_edit_episodes(
    tokenizer,
    val_pool: list[int],
    *,
    num_verbatim: int = 24,
    num_implicit: int = 24,
    num_pages: int = 8,
    page_size: int = 256,
    seed: int = 2026,
) -> dict[str, list[dict[str, Any]]]:
    """Build 100% disjoint verbatim control + implicit/inferential post-edit semantic episodes."""
    rng = random.Random(seed + 5101)
    vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else 2048
    filler_vocab = val_pool if val_pool else list(range(2, max(16, vocab_size - 2)))

    # (a) Verbatim post-edit control suite (answer appears literally in the edit turn)
    verbatim_specs = (
        ("HarborGate", "protocol", "TLS-1.2", "TLS-1.3"),
        ("CloudVault", "retention window", "14 days", "45 days"),
        ("MetricPulse", "sampling interval", "30 seconds", "5 seconds"),
        ("QueueBridge", "retry limit", "2 attempts", "6 attempts"),
        ("BeaconMesh", "failover zone", "nordic-west", "alpine-south"),
        ("TraceLoom", "compression codec", "gzip", "zstd"),
        ("QuotaDesk", "burst cap", "120 rpm", "360 rpm"),
        ("SignalForge", "batch size", "16 items", "64 items"),
    )
    verbatim_eps: list[dict[str, Any]] = []
    for idx in range(num_verbatim):
        svc, attr, old_v, new_v = verbatim_specs[idx % len(verbatim_specs)]
        svc_id = f"{svc}-{10 + idx}"
        pages = [_sample_background_page(rng, page_size, val_pool, filler_vocab) for _ in range(num_pages)]
        tgt_p = rng.randrange(num_pages)
        init_stmt = f"\nService {svc_id} currently uses {attr} set to {old_v}.\n"
        i_toks = _encode_text_or_mock(init_stmt, tokenizer, vocab_size)
        off = rng.randint(8, max(8, page_size - len(i_toks) - 8))
        pages[tgt_p][off : off + min(len(i_toks), page_size - off)] = i_toks[: page_size - off]

        edit_text = f"Approved revision: Service {svc_id} updated its {attr} from {old_v} to {new_v}."
        question = f"What is the updated {attr} for Service {svc_id}?"
        verbatim_eps.append(
            {
                "suite": "verbatim",
                "pages": torch.tensor(pages, dtype=torch.long),
                "target_page": tgt_p,
                "edit_text": edit_text,
                "question": question,
                "router_query": f"{svc_id} {attr}",
                "answer": new_v,
                "old_answer": old_v,
            }
        )

    # (b) Implicit / inferential post-edit semantic suite (gold answer NEVER appears verbatim in edit_text!)
    implicit_templates = (
        # Category 1: Region read-only vs read-write policy inheritance
        (
            "region_policy_no",
            "The {svc} service is hosted in the {reg} region.",
            "Policy bulletin: All services hosted in the {reg} region switch to read-only mode tomorrow.",
            "Can the {svc} service perform write operations tomorrow? Answer Yes or No.",
            "No",
        ),
        (
            "region_policy_yes",
            "The {svc} service is hosted in the {reg} region.",
            "Policy bulletin: All services hosted in the {reg} region are promoted to full read-write mode tomorrow.",
            "Can the {svc} service perform write operations tomorrow? Answer Yes or No.",
            "Yes",
        ),
        # Category 2: Badge clearance policy inheritance
        (
            "badge_access_no",
            "Contractor {svc} holds a Tier-B facility badge.",
            "Security update: Entering the Archive Room strictly requires Tier-A clearance.",
            "Is contractor {svc} permitted to enter the Archive Room? Answer Yes or No.",
            "No",
        ),
        (
            "badge_access_yes",
            "Contractor {svc} holds a Tier-A facility badge.",
            "Security update: Entering the Archive Room requires Tier-A clearance.",
            "Is contractor {svc} permitted to enter the Archive Room? Answer Yes or No.",
            "Yes",
        ),
        # Category 3: Dependency outage propagation
        (
            "dep_outage_no",
            "The {svc} gateway depends directly on {reg}DB to process requests.",
            "Operations notice: {reg}DB enters complete offline maintenance tonight.",
            "Will the {svc} gateway remain fully operational tonight? Answer Yes or No.",
            "No",
        ),
        (
            "dep_outage_yes",
            "The {svc} gateway depends directly on {reg}DB to process requests.",
            "Operations notice: {reg}DB maintenance was cancelled and stays fully online tonight.",
            "Will the {svc} gateway remain fully operational tonight? Answer Yes or No.",
            "Yes",
        ),
        # Category 4: Numeric quota comparison
        (
            "quota_fit_no",
            "The {svc} cluster enforces a strict maximum cap of 40 GPUs.",
            "Scheduling update: The {svc} workload request was raised to 64 GPUs.",
            "Does the updated {svc} workload fit within the 40 GPU cluster cap? Answer Yes or No.",
            "No",
        ),
        (
            "quota_fit_yes",
            "The {svc} cluster enforces a strict maximum cap of 40 GPUs.",
            "Scheduling update: The {svc} workload request was reduced to 24 GPUs.",
            "Does the updated {svc} workload fit within the 40 GPU cluster cap? Answer Yes or No.",
            "Yes",
        ),
    )
    svc_names = ("TokyoBilling", "SeoulLedger", "OsakaPay", "ZurichAuth", "MilanSync", "LisbonTax", "OsloCart", "ViennaMail")
    reg_names = ("Asia", "Nordic", "Iberia", "Alpine", "Pacific", "Baltic")
    implicit_eps: list[dict[str, Any]] = []
    for idx in range(num_implicit):
        kind, base_t, edit_t, q_t, gold = implicit_templates[idx % len(implicit_templates)]
        svc = f"{svc_names[idx % len(svc_names)]}-{idx + 1}"
        reg = reg_names[idx % len(reg_names)]
        base_stmt = "\n" + base_t.format(svc=svc, reg=reg) + "\n"
        edit_text = edit_t.format(svc=svc, reg=reg)
        question = q_t.format(svc=svc, reg=reg)

        # Hard guarantee: gold answer does NOT appear verbatim as a word in edit_text!
        edit_words = set(re.findall(r"[a-z0-9]+", edit_text.lower()))
        assert gold.lower() not in edit_words, (
            f"Implicit post-edit episode leaked verbatim gold={gold!r} into edit_text={edit_text!r}"
        )

        pages = [_sample_background_page(rng, page_size, val_pool, filler_vocab) for _ in range(num_pages)]
        tgt_p = rng.randrange(num_pages)
        b_toks = _encode_text_or_mock(base_stmt, tokenizer, vocab_size)
        off = rng.randint(8, max(8, page_size - len(b_toks) - 8))
        pages[tgt_p][off : off + min(len(b_toks), page_size - off)] = b_toks[: page_size - off]

        implicit_eps.append(
            {
                "suite": "implicit",
                "category": kind,
                "pages": torch.tensor(pages, dtype=torch.long),
                "target_page": tgt_p,
                "edit_text": edit_text,
                "question": question,
                "router_query": f"{svc} {reg}",
                "answer": gold,
            }
        )

    return {"verbatim": verbatim_eps, "implicit": implicit_eps}


@torch.no_grad()
def run_ablation_2x2_edit(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    bge_toc: _PlugInBiEncoderTOC | None = None,
    num_verbatim: int = 24,
    num_implicit: int = 24,
    salience_route_weight: float = 1.5,
    max_new_tokens: int = 24,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Run 2x2 Edit Ablation: continuous_delta_z (ON/OFF) x discrete_exact_bank (ON/OFF) on verbatim & implicit queries."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    eff_span_len = int(getattr(wrapper.cfg, "salience_span_len", 16))
    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)
    wrapper.eval()

    nv = min(num_verbatim, 8) if is_mock else num_verbatim
    ni = min(num_implicit, 8) if is_mock else num_implicit
    suites = build_2x2_edit_episodes(
        tokenizer, val_pool, num_verbatim=nv, num_implicit=ni, num_pages=8, page_size=wrapper.cfg.page_size, seed=seed
    )

    arms = (
        ("deltaZ_ON_bank_ON", True, True),
        ("deltaZ_ON_bank_OFF", True, False),
        ("deltaZ_OFF_bank_ON", False, True),
        ("deltaZ_OFF_bank_OFF", False, False),
    )
    results: dict[str, Any] = {}

    for arm_name, use_delta_z, use_exact_bank in arms:
        suite_out: dict[str, Any] = {}
        all_rows: list[dict[str, float]] = []
        max_toc_diff_arm = 0.0
        for s_name, eps in suites.items():
            rows: list[dict[str, float]] = []
            samples: list[dict[str, Any]] = []
            for ep in eps:
                p_t = ep["pages"].unsqueeze(0).to(device)
                tgt_p = int(ep["target_page"])
                z_book, bank, toc0 = _encode_book_chunked(
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
                z_orig = z_book.clone()
                bank_orig = SparseExactBank(
                    token_ids=bank.token_ids.clone(),
                    page_ids=bank.page_ids.clone(),
                    h_coords=bank.h_coords.clone(),
                    w_coords=bank.w_coords.clone(),
                )
                edit_ids = torch.tensor(
                    _encode_text_or_mock(ep["edit_text"], tokenizer, vocab_size), dtype=torch.long, device=device
                ).unsqueeze(0)
                p_idx_t = torch.tensor([tgt_p], dtype=torch.long, device=device)

                _sync_device(device)
                t_e0 = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    ov = (
                        extract_contextual_edit_embeds(wrapper.llm, edit_ids, layer_idx=wrapper.cfg.contextual_edit_layer)
                        if wrapper.cfg.use_contextual_edit_encoding
                        else None
                    )
                    z_upd, g_p, bank_upd = wrapper.lslc.apply_page_delta(
                        z_book.clone(),
                        p_idx_t,
                        edit_ids,
                        embed_tokens_fn=embed_fn,
                        exact_bank=bank,
                        vocab_size=vocab_size,
                        edit_embeds_override=ov,
                        in_place=False,
                        bank_write_threshold=0.45,
                    )
                _sync_device(device)
                edit_ms = (time.perf_counter() - t_e0) * 1000.0
                strip_gate_val = float(g_p[:, -1, :].float().mean().item())
                bank_wrote = float(not torch.equal(bank_upd.token_ids[:, tgt_p], bank_orig.token_ids[:, tgt_p]))

                # Iso-token 2x2 edit isolation: preserve bank_orig across all 4 cells (never None)
                z_use = z_upd if use_delta_z else z_orig
                bank_use = bank_upd if use_exact_bank else bank_orig
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    toc_after = wrapper.lslc.projector.project_toc(z_use)
                toc_diff = float((toc_after.float() - toc0.float()).abs().max().item())
                max_toc_diff_arm = max(max_toc_diff_arm, toc_diff)

                rq_ids = torch.tensor(
                    _encode_text_or_mock(ep["router_query"], tokenizer, vocab_size), dtype=torch.long, device=device
                ).unsqueeze(0)
                ext_s = None
                if bge_toc is not None:
                    bge_p4 = bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=64)
                    ext_s = bge_toc.score_book(bge_toc.encode_texts([ep["router_query"]], is_query=True), bge_p4)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    # Route using original bank across all 4 cells so page selection is strictly controlled
                    r_out, _ = wrapper.lslc.route_and_assemble(
                        z_book=z_use,
                        toc_embeds=toc_after,
                        query_embed=embed_fn(rq_ids).mean(dim=1),
                        exact_bank=bank_orig,
                        embed_tokens_fn=embed_fn,
                        k_deep=4,
                        k_skim=2,
                        mode="toc_only",
                        training_ste=False,
                        query_input_ids=rq_ids,
                        salience_route_weight=salience_route_weight,
                        external_dense_logits=ext_s,
                    )
                    _, asm = wrapper.lslc.route_and_assemble(
                        z_book=z_use,
                        toc_embeds=toc_after,
                        query_embed=embed_fn(rq_ids).mean(dim=1),
                        exact_bank=bank_use,
                        embed_tokens_fn=embed_fn,
                        k_deep=4,
                        k_skim=2,
                        mode="full",
                        training_ste=False,
                        router_out_override=r_out,
                        exact_bank_attenuation="none",
                    )

                p_prompt = torch.tensor(
                    tokenizer.encode(_format_qa_prompt(ep["question"]), add_special_tokens=False),
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                gen_txt = _greedy_generate_qa(
                    wrapper.llm, asm.prefix_embeds.to(llm_dtype), p_prompt, tokenizer,
                    max_new_tokens=max_new_tokens, user_open_ids=user_open_ids,
                )
                qm = compute_community_qa_metrics(gen_txt, ep["answer"])
                # Semantic accuracy for implicit yes/no & verbatim substring accuracy
                norm_g = _normalize_qa_text(ep["answer"])
                norm_p = _normalize_qa_text(gen_txt)
                if s_name == "implicit" and norm_g in ("yes", "no"):
                    first_words = norm_p.split()[:4]
                    sem_acc = 1.0 if (norm_g in first_words and (("no" if norm_g == "yes" else "yes") not in first_words)) else 0.0
                else:
                    sem_acc = max(qm["extracted_em"], qm["substring_match"])

                rec = {
                    "em": qm["extracted_em"],
                    "f1": qm["token_f1"],
                    "semantic_acc": float(sem_acc),
                    "edit_latency_ms": round(edit_ms, 3),
                    "mean_strip_gate": round(strip_gate_val, 4),
                    "bank_write_rate": bank_wrote,
                    "active_prefix_tokens": float(asm.prefix_embeds.shape[1]),
                    "max_toc_diff": toc_diff,
                }
                rows.append(rec)
                all_rows.append(rec)
                if len(samples) < 3:
                    samples.append({"question": ep["question"], "edit": ep["edit_text"], "gold": ep["answer"], "pred": gen_txt, **rec})
            agg_s = _summarize_metric_rows(
                rows, ("em", "f1", "semantic_acc", "edit_latency_ms", "mean_strip_gate", "bank_write_rate", "active_prefix_tokens")
            )
            agg_s["max_toc_diff"] = round(max((r["max_toc_diff"] for r in rows), default=0.0), 6)
            agg_s["samples"] = samples
            suite_out[s_name] = agg_s

        ov_arm = _summarize_metric_rows(
            all_rows, ("em", "f1", "semantic_acc", "edit_latency_ms", "mean_strip_gate", "bank_write_rate", "active_prefix_tokens")
        )
        ov_arm["max_toc_diff"] = round(max_toc_diff_arm, 6)
        results[arm_name] = {
            "continuous_delta_z": use_delta_z,
            "discrete_exact_bank": use_exact_bank,
            "overall": ov_arm,
            "by_suite": suite_out,
        }
        print(
            f"[G5 2x2 {arm_name}] verb_EM={suite_out['verbatim']['em']:.3f} verb_sem={suite_out['verbatim']['semantic_acc']:.3f} "
            f"(gate={suite_out['verbatim']['mean_strip_gate']:.3f}, bw={suite_out['verbatim']['bank_write_rate']:.2f}) | "
            f"impl_EM={suite_out['implicit']['em']:.3f} impl_sem={suite_out['implicit']['semantic_acc']:.3f} "
            f"(gate={suite_out['implicit']['mean_strip_gate']:.3f}, bw={suite_out['implicit']['bank_write_rate']:.2f}) | "
            f"tok={ov_arm['active_prefix_tokens']:.0f} toc_diff={max_toc_diff_arm}",
            flush=True,
        )

    tok_counts = {arm: results[arm]["overall"]["active_prefix_tokens"] for arm, _, _ in arms}
    assert len(set(tok_counts.values())) == 1, f"Iso-token violated across 2x2 edit arms: {tok_counts}"
    return results


# ---------------------------------------------------------------------------
# Pillar 5.2: Iso-Token & Iso-Parameter 1D-3Layer vs 3D-Spatial Pyramid
# ---------------------------------------------------------------------------


class Flat1D3LayerEncoder(nn.Module):
    """Iso-token & iso-parameter 1D-3Layer flat encoder: NO 2D/3D spatial grid, NO FactorizedRoPE3D, NO 2D conv."""

    def __init__(self, cfg: LSLCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_llm, cfg.d_enc)
        self.pos_1d = nn.Embedding(cfg.page_size, cfg.d_enc)
        self.slot_queries = nn.Parameter(torch.randn(1, cfg.num_book_slots, cfg.d_enc) * 0.02)
        self.cross_blocks = nn.ModuleList(
            [PerceiverCrossAttentionBlock(cfg.d_enc, cfg.enc_heads, cfg.enc_dropout) for _ in range(2)]
        )
        # Standard 1D Transformer self-attention blocks (no FactorizedRoPE3D, no 2D conv)
        # Slightly wider FFN (mult=3) to match SpatialLayerSelfAttentionBlock's conv parameter budget closely
        self.self_blocks = nn.ModuleList([self._make_1d_block(cfg) for _ in range(cfg.enc_layers)])
        self.out_norm = nn.LayerNorm(cfg.d_enc)
        self.out_proj = nn.Linear(cfg.d_enc, cfg.latent_channels)

    @staticmethod
    def _make_1d_block(cfg: LSLCConfig) -> nn.Module:
        class _Flat1DBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                d = cfg.d_enc
                self.h = cfg.enc_heads
                self.hd = d // cfg.enc_heads
                self.n1 = nn.LayerNorm(d)
                self.qkv = nn.Linear(d, d * 3)
                self.o = nn.Linear(d, d)
                self.n2 = nn.LayerNorm(d)
                self.ff = _FeedForward(d, mult=3, dropout=cfg.enc_dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, s, d = x.shape
                qkv = self.qkv(self.n1(x)).view(b, s, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
                a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
                x = x + self.o(a.transpose(1, 2).contiguous().view(b, s, d))
                return x + self.ff(self.n2(x))

        return _Flat1DBlock()

    def forward(self, page_embeds: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = page_embeds.shape
        if not torch.is_autocast_enabled():
            page_embeds = page_embeds.to(self.in_proj.weight.dtype)
        pos = self.pos_1d(torch.arange(seq_len, device=page_embeds.device)).unsqueeze(0)
        kv = self.in_proj(page_embeds) + pos
        q = self.slot_queries.expand(bsz, -1, -1)
        for blk in self.cross_blocks:
            q = blk(q, kv)
        for blk in self.self_blocks:
            q = blk(q)
        z_flat = self.out_proj(self.out_norm(q))
        # Reshape to [B, L=3, C, H=4, W=4] purely for iso-token projector interface (no spatial prior inside encoder!)
        return (
            z_flat.view(bsz, self.cfg.num_layers, self.cfg.grid_h, self.cfg.grid_w, self.cfg.latent_channels)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )

    def encode_pages(self, pages_embeds: torch.Tensor) -> torch.Tensor:
        bsz, num_pages, seq_len, d_llm = pages_embeds.shape
        z = self.forward(pages_embeds.reshape(bsz * num_pages, seq_len, d_llm))
        return z.view(bsz, num_pages, self.cfg.num_layers, self.cfg.latent_channels, self.cfg.grid_h, self.cfg.grid_w)


def _make_oracle_router_output(
    num_pages: int,
    target_page: int,
    k_deep: int,
    k_skim: int,
    device: torch.device,
) -> RouterOutput:
    """Deterministic oracle RouterOutput placing target_page consistently in deep_idx to isolate encoder topology."""
    tgt = max(0, min(int(target_page), num_pages - 1))
    kd = max(1, min(int(k_deep), num_pages))
    ks = max(0, min(int(k_skim), num_pages - kd))
    others = [p for p in range(num_pages) if p != tgt]
    deep_list = others[: kd - 1] + [tgt]
    deep_set = set(deep_list)
    rem = [p for p in range(num_pages) if p not in deep_set]
    skim_list = rem[:ks]
    toc_list = rem[ks:]
    logits = torch.zeros((1, num_pages), device=device, dtype=torch.float32)
    for p in deep_list:
        logits[0, p] = 1.0
    logits[0, tgt] = 5.0
    probs = F.softmax(logits, dim=-1)
    return RouterOutput(
        logits=logits,
        probs=probs,
        deep_idx=torch.tensor([deep_list], dtype=torch.long, device=device),
        skim_idx=torch.tensor([skim_list], dtype=torch.long, device=device),
        toc_idx=torch.tensor([toc_list], dtype=torch.long, device=device),
    )


def run_ablation_1d_vs_3d(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    train_pool: list[int],
    val_pool: list[int],
    device: torch.device,
    *,
    train_steps: int = 240,
    eval_n: int = 48,
    lr: float = 3e-4,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Iso-token & iso-parameter head-to-head: 1D-3Layer flat compressor vs 3D-Spatial Pyramid."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    eff_steps = min(train_steps, 12) if is_mock else train_steps
    eff_eval_n = min(eval_n, 12) if is_mock else eval_n

    train_niah = list(
        SyntheticExactSemanticNIAHDataset(
            num_episodes=16 if is_mock else 96,
            num_pages=8,
            page_size=wrapper.cfg.page_size,
            vocab_size=vocab_size,
            mode="mixed",
            tokenizer=tokenizer,
            seed=seed + 301,
            background_token_pool=train_pool,
        )
    )
    eval_niah = list(
        SyntheticExactSemanticNIAHDataset(
            num_episodes=eff_eval_n,
            num_pages=8,
            page_size=wrapper.cfg.page_size,
            vocab_size=vocab_size,
            mode="mixed",
            tokenizer=tokenizer,
            seed=seed + 9301,
            background_token_pool=val_pool,
        )
    )

    out: dict[str, Any] = {}
    active_tokens_by_variant: dict[str, int] = {}

    for variant in ("3d_spatial_pyramid", "1d_3layer_flat"):
        torch.manual_seed(seed + 42)
        cfg_v = copy.deepcopy(wrapper.cfg)
        sys_v = LSLCSystem(cfg_v).to(device)
        if variant == "1d_3layer_flat":
            sys_v.encoder = Flat1D3LayerEncoder(cfg_v).to(device)

        enc_params = sum(p.numel() for p in sys_v.encoder.parameters())
        tot_params = sum(p.numel() for p in sys_v.parameters())

        # Identical training budget + oracle router override (eliminates random router noise; purely tests encoder topology!)
        opt = torch.optim.AdamW(sys_v.parameters(), lr=lr, weight_decay=0.01)
        rng = random.Random(seed + 77)
        sys_v.train()
        t0 = time.perf_counter()
        losses: list[float] = []
        with torch.enable_grad():
            for _ in range(eff_steps):
                ep = train_niah[rng.randrange(len(train_niah))]
                p_t = ep.context_pages.unsqueeze(0).to(device)
                q_ids = ep.query_ids.unsqueeze(0).to(device)
                a_ids = ep.answer_ids.unsqueeze(0).to(device)
                oracle_r = _make_oracle_router_output(
                    p_t.shape[1], int(ep.target_page_idx), k_deep=2, k_skim=2, device=device
                )
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    z_b, _, toc = sys_v.encode_book(p_t, embed_fn, use_exact_bank=False, vocab_size=vocab_size)
                    q_emb = embed_fn(q_ids).mean(dim=1)
                    _, asm = sys_v.route_and_assemble(
                        z_b,
                        toc,
                        q_emb,
                        None,
                        embed_fn,
                        k_deep=2,
                        k_skim=2,
                        mode="full",
                        training_ste=False,
                        router_out_override=oracle_r,
                    )
                    turn_ids = torch.cat((q_ids, a_ids), dim=1)
                    turn_labels = torch.cat((torch.full_like(q_ids, -100), a_ids), dim=1)
                    loss = wrapper._run_llm_with_prefix(asm.prefix_embeds, turn_ids, turn_labels=turn_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sys_v.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach().item()))
        train_sec = round(time.perf_counter() - t0, 2)

        # Evaluate on N>=48 + prefix-cache invariance + multi-scale page latency (P=8, 16, 32)
        sys_v.eval()
        rows: list[dict[str, float]] = []
        with torch.no_grad():
            for ep in eval_niah:
                p_t = ep.context_pages.unsqueeze(0).to(device)
                q_ids = ep.query_ids.unsqueeze(0).to(device)
                a_ids = ep.answer_ids.unsqueeze(0).to(device)
                oracle_r = _make_oracle_router_output(
                    p_t.shape[1], int(ep.target_page_idx), k_deep=2, k_skim=2, device=device
                )
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    z_b, _, toc = sys_v.encode_book(p_t, embed_fn, use_exact_bank=False, vocab_size=vocab_size)
                    q_emb = embed_fn(q_ids).mean(dim=1)
                    _, asm = sys_v.route_and_assemble(
                        z_b,
                        toc,
                        q_emb,
                        None,
                        embed_fn,
                        k_deep=2,
                        k_skim=2,
                        mode="full",
                        training_ste=False,
                        router_out_override=oracle_r,
                    )
                active_tokens_by_variant[variant] = int(asm.prefix_embeds.shape[1])
                turn_ids = torch.cat((q_ids, a_ids), dim=1)
                inp = torch.cat((asm.prefix_embeds.to(llm_dtype), embed_fn(turn_ids).to(llm_dtype)), dim=1)
                logits = wrapper.llm(inputs_embeds=inp, attention_mask=torch.ones((1, inp.shape[1]), device=device, dtype=torch.long)).logits
                st = asm.prefix_embeds.shape[1] + q_ids.shape[1] - 1
                ans_l = logits[:, st : st + a_ids.shape[1], :].float()
                nll = float(F.cross_entropy(ans_l.reshape(-1, ans_l.shape[-1]), a_ids.reshape(-1)).item())
                tok_acc = float((ans_l.argmax(dim=-1) == a_ids).float().mean().item())
                gen_txt = _greedy_generate_qa(wrapper.llm, asm.prefix_embeds.to(llm_dtype), q_ids, tokenizer, max_new_tokens=20)
                qm = compute_community_qa_metrics(gen_txt, str(ep.metadata["answer_str"]))
                rows.append({"em": qm["extracted_em"], "f1": qm["token_f1"], "tok_acc": tok_acc, "nll": nll})

            # Prefix-cache invariance check under DeltaUpdater
            sample_p = eval_niah[0].context_pages.unsqueeze(0).to(device)
            e_ids = eval_niah[0].query_ids.unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                z0, _, toc0 = sys_v.encode_book(sample_p, embed_fn, use_exact_bank=False, vocab_size=vocab_size)
                z1, _ = sys_v.apply_page_delta(z0.clone(), torch.tensor([0], device=device), e_ids, embed_tokens_fn=embed_fn)
                toc1 = sys_v.projector.project_toc(z1)
            toc_diff = float((toc1.float() - toc0.float()).abs().max().item())
            l0_l1_diff = float((z1[:, :, :2].float() - z0[:, :, :2].float()).abs().max().item())

            # Page scaling audit across P=8, 16, 32
            scale_stats: dict[str, Any] = {}
            for p_scale in ((4, 8, 16) if is_mock else (8, 16, 32)):
                p_rand = torch.randint(2, min(vocab_size, 500), (1, p_scale, wrapper.cfg.page_size), device=device)
                _sync_device(device)
                ts0 = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    z_s, _, toc_s = sys_v.encode_book(p_rand, embed_fn, use_exact_bank=False, vocab_size=vocab_size)
                    _, asm_s = sys_v.route_and_assemble(
                        z_s, toc_s, embed_fn(e_ids).mean(dim=1), None, embed_fn,
                        k_deep=2, k_skim=2, mode="full", hierarchical_chapter_size=16,
                    )
                _sync_device(device)
                scale_stats[f"P={p_scale}"] = {
                    "encode_and_route_ms": round((time.perf_counter() - ts0) * 1000.0, 2),
                    "active_prefix_tokens": int(asm_s.prefix_embeds.shape[1]),
                }

        agg = _summarize_metric_rows(rows, ("em", "f1", "tok_acc", "nll"))
        agg["ppl"] = round(math.exp(min(20.0, agg.get("nll", 0.0))), 4)
        out[variant] = {
            "encoder_params": enc_params,
            "total_params": tot_params,
            "active_prefix_tokens": active_tokens_by_variant[variant],
            "train_steps": eff_steps,
            "train_sec": train_sec,
            "final_train_loss": round(losses[-1], 4) if losses else 0.0,
            "metrics": agg,
            "prefix_cache_invariance": {"max_toc_diff": round(toc_diff, 6), "max_l0_l1_diff": round(l0_l1_diff, 6)},
            "page_scaling": scale_stats,
        }
        print(
            f"[G5 1D-vs-3D {variant}] enc_params={enc_params} tok={active_tokens_by_variant[variant]} "
            f"EM={agg['em']:.3f} F1={agg['f1']:.3f} tok_acc={agg['tok_acc']:.3f} toc_diff={toc_diff}",
            flush=True,
        )

    assert active_tokens_by_variant["3d_spatial_pyramid"] == active_tokens_by_variant["1d_3layer_flat"], (
        f"Iso-token violated in 1D vs 3D ablation: {active_tokens_by_variant}"
    )
    out["iso_token_verified"] = True
    return out


# ---------------------------------------------------------------------------
# Pillar 5.3: True KV Baselines (True INT4 KV + True Attn H2O/SnapKV + ICAE)
# ---------------------------------------------------------------------------


class ICAELatentBaseline(nn.Module):
    """ICAE-style 1D memory-slot autoencoder baseline matched to LSLC encoder depth (frozen LLM, NO 3D pyramid, NO ExactBank)."""

    def __init__(self, cfg: LSLCConfig, mem_slots_per_page: int = 24) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_llm = cfg.d_llm
        self.mem_slots_per_page = mem_slots_per_page
        self.in_proj = nn.Linear(cfg.d_llm, cfg.d_enc)
        self.pos_1d = nn.Embedding(cfg.page_size, cfg.d_enc)
        self.mem_queries = nn.Parameter(torch.randn(1, mem_slots_per_page, cfg.d_enc) * 0.02)
        self.cross_blocks = nn.ModuleList(
            [PerceiverCrossAttentionBlock(cfg.d_enc, cfg.enc_heads, cfg.enc_dropout) for _ in range(2)]
        )
        self.self_blocks = nn.ModuleList([Flat1D3LayerEncoder._make_1d_block(cfg) for _ in range(cfg.enc_layers)])
        self.out_proj = nn.Sequential(nn.LayerNorm(cfg.d_enc), nn.Linear(cfg.d_enc, cfg.d_llm))

    def encode_book_slots(self, pages_embeds: torch.Tensor) -> torch.Tensor:
        bsz, num_p, tp, d = pages_embeds.shape
        x = pages_embeds.reshape(bsz * num_p, tp, d)
        if not torch.is_autocast_enabled():
            x = x.to(self.in_proj.weight.dtype)
        pos = self.pos_1d(torch.arange(tp, device=x.device)).unsqueeze(0)
        kv = self.in_proj(x) + pos
        q = self.mem_queries.expand(bsz * num_p, -1, -1)
        for blk in self.cross_blocks:
            q = blk(q, kv)
        for blk in self.self_blocks:
            q = blk(q)
        return self.out_proj(q).view(bsz, num_p * self.mem_slots_per_page, self.d_llm)


def run_true_kv_baselines(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    train_pool: list[int],
    val_pool: list[int],
    device: torch.device,
    *,
    bge_toc: _PlugInBiEncoderTOC | None = None,
    eval_n: int = 48,
    icae_steps: int = 240,
    salience_route_weight: float = 1.5,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Compare Full BF16 KV, True INT4 past_key_values, True Attention H2O/SnapKV, ICAE-style 1D AE, and LSLC v3."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    eff_span_len = int(getattr(wrapper.cfg, "salience_span_len", 16))
    eff_eval_n = min(eval_n, 12) if is_mock else eval_n
    num_p = 4 if is_mock else 8

    eval_niah = list(
        SyntheticExactSemanticNIAHDataset(
            num_episodes=eff_eval_n,
            num_pages=num_p,
            page_size=wrapper.cfg.page_size,
            vocab_size=vocab_size,
            mode="mixed",
            tokenizer=tokenizer,
            seed=seed + 9501,
            background_token_pool=val_pool,
        )
    )
    eval_ppl = list(ChunkedTextPageDataset(val_pool, num_pages=num_p, page_size=wrapper.cfg.page_size, turn_len=32))[: (4 if is_mock else 16)]

    # Measure LSLC active prefix token budget to set exact iso-token target for H2O/SnapKV & ICAE
    wrapper.eval()
    with torch.no_grad():
        sample_p = eval_niah[0].context_pages.unsqueeze(0).to(device)
        z0, b0, t0 = _encode_book_chunked(
            wrapper.lslc, sample_p, embed_fn, vocab_size, device, amp_dtype, autocast_enabled,
            chunk_size=16, special_token_ids=special_ids, salience_span_len=eff_span_len,
        )
        rq0 = eval_niah[0].query_ids.unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            _, asm0 = wrapper.lslc.route_and_assemble(
                z0, t0, embed_fn(rq0).mean(dim=1), b0, embed_fn,
                k_deep=min(4, num_p), k_skim=min(2, max(0, num_p - 4)), mode="full", exact_bank_attenuation="none",
            )
        iso_token_budget = int(asm0.prefix_embeds.shape[1])

    # Train ICAE-style 1D latent autoencoder baseline on equal depth, steps & iso-token budget (frozen LLM!)
    slots_per_page = max(4, iso_token_budget // num_p)
    icae = ICAELatentBaseline(wrapper.cfg, mem_slots_per_page=slots_per_page).to(device)
    icae_params = sum(p.numel() for p in icae.parameters())
    train_niah = list(
        SyntheticExactSemanticNIAHDataset(
            num_episodes=16 if is_mock else 96,
            num_pages=num_p,
            page_size=wrapper.cfg.page_size,
            vocab_size=vocab_size,
            mode="mixed",
            tokenizer=tokenizer,
            seed=seed + 501,
            background_token_pool=train_pool,
        )
    )
    opt_icae = torch.optim.AdamW(icae.parameters(), lr=3e-4, weight_decay=0.01)
    rng_i = random.Random(seed + 111)
    eff_icae_steps = min(icae_steps, 12) if is_mock else icae_steps
    icae.train()
    with torch.enable_grad():
        for _ in range(eff_icae_steps):
            ep = train_niah[rng_i.randrange(len(train_niah))]
            p_t = ep.context_pages.unsqueeze(0).to(device)
            q_ids = ep.query_ids.unsqueeze(0).to(device)
            a_ids = ep.answer_ids.unsqueeze(0).to(device)
            opt_icae.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                pref_i = icae.encode_book_slots(embed_fn(p_t))
                turn_ids = torch.cat((q_ids, a_ids), dim=1)
                turn_labels = torch.cat((torch.full_like(q_ids, -100), a_ids), dim=1)
                loss_i = wrapper._run_llm_with_prefix(pref_i, turn_ids, turn_labels=turn_labels)
            loss_i.backward()
            torch.nn.utils.clip_grad_norm_(icae.parameters(), 1.0)
            opt_icae.step()
    icae.eval()

    systems = ("full_kv_bf16", "true_int4_kv_cache", "true_attn_h2o_snapkv", "icae_1d_latent_baseline", "lslc_v3_universal")
    out: dict[str, Any] = {
        "iso_token_budget": iso_token_budget,
        "num_pages": num_p,
        "icae_params": icae_params,
        "icae_train_steps": eff_icae_steps,
        "systems": {},
    }

    with torch.no_grad():
        for sys_name in systems:
            rows_all: list[dict[str, float]] = []
            rows_exact: list[dict[str, float]] = []
            rows_sem: list[dict[str, float]] = []
            samples_arm: list[dict[str, Any]] = []
            ppl_nlls: list[float] = []
            active_toks_seen: list[int] = []
            _reset_peak_vram(device)

            # Evaluate continuation PPL
            for ep in eval_ppl:
                p_t = ep.context_pages.unsqueeze(0).to(device)
                q_ids = ep.query_ids.unsqueeze(0).to(device)
                a_ids = ep.answer_ids.unsqueeze(0).to(device)
                flat_ids = p_t.reshape(1, -1)
                ctx_emb = embed_fn(flat_ids).to(llm_dtype)
                orig_ctx_len = int(ctx_emb.shape[1])
                if sys_name == "full_kv_bf16":
                    out_pre = wrapper.llm(inputs_embeds=ctx_emb, attention_mask=torch.ones((1, orig_ctx_len), device=device, dtype=torch.long), use_cache=True)
                    nll, _ = compute_nll_and_tok_acc_from_kv(wrapper.llm, out_pre.past_key_values, orig_ctx_len, q_ids, a_ids, orig_seq_len=orig_ctx_len)
                elif sys_name == "true_int4_kv_cache":
                    out_pre = wrapper.llm(inputs_embeds=ctx_emb, attention_mask=torch.ones((1, orig_ctx_len), device=device, dtype=torch.long), use_cache=True)
                    q_kv = quantize_past_key_values_int4(out_pre.past_key_values, group_size=64)
                    nll, _ = compute_nll_and_tok_acc_from_kv(wrapper.llm, q_kv, orig_ctx_len, q_ids, a_ids, orig_seq_len=orig_ctx_len)
                elif sys_name == "true_attn_h2o_snapkv":
                    pkv, attn_s = compute_real_attention_scores_and_kv(wrapper.llm, ctx_emb, obs_window=32)
                    ev_kv, keep_idx = evict_past_key_values_h2o_attention(pkv, attn_s, keep_budget=iso_token_budget)
                    nll, _ = compute_nll_and_tok_acc_from_kv(wrapper.llm, ev_kv, int(keep_idx.numel()), q_ids, a_ids, orig_seq_len=orig_ctx_len)
                elif sys_name == "icae_1d_latent_baseline":
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        pref = icae.encode_book_slots(embed_fn(p_t)).to(llm_dtype)
                    out_pre = wrapper.llm(inputs_embeds=pref, attention_mask=torch.ones((1, pref.shape[1]), device=device, dtype=torch.long), use_cache=True)
                    nll, _ = compute_nll_and_tok_acc_from_kv(wrapper.llm, out_pre.past_key_values, int(pref.shape[1]), q_ids, a_ids)
                else:
                    zb, bk, tc = _encode_book_chunked(
                        wrapper.lslc, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled,
                        chunk_size=16, special_token_ids=special_ids, salience_span_len=eff_span_len,
                    )
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        _, asm = wrapper.lslc.route_and_assemble(
                            zb, tc, embed_fn(q_ids).mean(dim=1), bk, embed_fn,
                            k_deep=min(4, num_p), k_skim=min(2, max(0, num_p - 4)), mode="full",
                            query_input_ids=q_ids, salience_route_weight=salience_route_weight, exact_bank_attenuation="none",
                        )
                    pref = asm.prefix_embeds.to(llm_dtype)
                    out_pre = wrapper.llm(inputs_embeds=pref, attention_mask=torch.ones((1, pref.shape[1]), device=device, dtype=torch.long), use_cache=True)
                    nll, _ = compute_nll_and_tok_acc_from_kv(wrapper.llm, out_pre.past_key_values, int(pref.shape[1]), q_ids, a_ids)
                ppl_nlls.append(nll)

            # Evaluate NIAH (exact + semantic)
            for ep in eval_niah:
                p_t = ep.context_pages.unsqueeze(0).to(device)
                q_ids = ep.query_ids.unsqueeze(0).to(device)
                a_ids = ep.answer_ids.unsqueeze(0).to(device)
                flat_ids = p_t.reshape(1, -1)
                ctx_emb = embed_fn(flat_ids).to(llm_dtype)
                orig_ctx_len = int(ctx_emb.shape[1])
                _sync_device(device)
                tq0 = time.perf_counter()
                if sys_name == "full_kv_bf16":
                    out_pre = wrapper.llm(inputs_embeds=ctx_emb, attention_mask=torch.ones((1, orig_ctx_len), device=device, dtype=torch.long), use_cache=True)
                    kv_c, kv_len, orig_len = out_pre.past_key_values, orig_ctx_len, orig_ctx_len
                elif sys_name == "true_int4_kv_cache":
                    out_pre = wrapper.llm(inputs_embeds=ctx_emb, attention_mask=torch.ones((1, orig_ctx_len), device=device, dtype=torch.long), use_cache=True)
                    kv_c, kv_len, orig_len = quantize_past_key_values_int4(out_pre.past_key_values, group_size=64), orig_ctx_len, orig_ctx_len
                elif sys_name == "true_attn_h2o_snapkv":
                    pkv, attn_s = compute_real_attention_scores_and_kv(wrapper.llm, ctx_emb, obs_window=32)
                    kv_c, keep_idx = evict_past_key_values_h2o_attention(pkv, attn_s, keep_budget=iso_token_budget)
                    kv_len, orig_len = int(keep_idx.numel()), orig_ctx_len
                elif sys_name == "icae_1d_latent_baseline":
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        pref = icae.encode_book_slots(embed_fn(p_t)).to(llm_dtype)
                    out_pre = wrapper.llm(inputs_embeds=pref, attention_mask=torch.ones((1, pref.shape[1]), device=device, dtype=torch.long), use_cache=True)
                    kv_c, kv_len, orig_len = out_pre.past_key_values, int(pref.shape[1]), int(pref.shape[1])
                else:
                    zb, bk, tc = _encode_book_chunked(
                        wrapper.lslc, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled,
                        chunk_size=16, special_token_ids=special_ids, salience_span_len=eff_span_len,
                    )
                    ext_s = None
                    if bge_toc is not None:
                        q_txt = tokenizer.decode(q_ids[0].tolist(), skip_special_tokens=True)
                        bge_p4 = bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=64)
                        ext_s = bge_toc.score_book(bge_toc.encode_texts([q_txt], is_query=True), bge_p4)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        _, asm = wrapper.lslc.route_and_assemble(
                            zb, tc, embed_fn(q_ids).mean(dim=1), bk, embed_fn,
                            k_deep=min(4, num_p), k_skim=min(2, max(0, num_p - 4)), mode="full",
                            query_input_ids=q_ids, salience_route_weight=salience_route_weight,
                            exact_bank_attenuation="none", external_dense_logits=ext_s,
                        )
                    pref = asm.prefix_embeds.to(llm_dtype)
                    out_pre = wrapper.llm(inputs_embeds=pref, attention_mask=torch.ones((1, pref.shape[1]), device=device, dtype=torch.long), use_cache=True)
                    kv_c, kv_len, orig_len = out_pre.past_key_values, int(pref.shape[1]), int(pref.shape[1])

                _, tok_acc = compute_nll_and_tok_acc_from_kv(wrapper.llm, kv_c, kv_len, q_ids, a_ids, orig_seq_len=orig_len)
                gen_ids = greedy_generate_from_kv_cache(
                    wrapper.llm,
                    kv_c,
                    kv_len,
                    q_ids,
                    max_new_tokens=24,
                    eos_token_id=tokenizer.eos_token_id,
                    orig_seq_len=orig_len,
                    tokenizer=tokenizer,
                    stop_on_newline=True,
                )
                _sync_device(device)
                q_ms = (time.perf_counter() - tq0) * 1000.0
                active_toks_seen.append(kv_len)
                raw_gen_txt = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                # Identical first-line extraction & fair substring_match across all 5 arms
                first_line = next((ln.strip() for ln in raw_gen_txt.splitlines() if ln.strip()), raw_gen_txt.strip())
                qm = compute_community_qa_metrics(first_line, str(ep.metadata["answer_str"]))
                rec = {
                    "em": qm["extracted_em"],
                    "substring_match": qm["substring_match"],
                    "f1": qm["token_f1"],
                    "tok_acc": tok_acc,
                    "per_query_ms": round(q_ms, 3),
                }
                rows_all.append(rec)
                if ep.metadata.get("needle_type") == "exact":
                    rows_exact.append(rec)
                else:
                    rows_sem.append(rec)
                if len(samples_arm) < 4:
                    samples_arm.append(
                        {
                            "needle_type": ep.metadata.get("needle_type"),
                            "gold": str(ep.metadata["answer_str"]),
                            "pred_first_line": first_line,
                            "pred_raw": raw_gen_txt[:120],
                            **rec,
                        }
                    )

            agg = _summarize_metric_rows(rows_all, ("em", "substring_match", "f1", "tok_acc", "per_query_ms"))
            agg["ppl"] = round(math.exp(min(20.0, sum(ppl_nlls) / max(1, len(ppl_nlls)))), 4)
            agg["active_tokens_mean"] = round(sum(active_toks_seen) / max(1, len(active_toks_seen)), 1)
            agg["exact_niah"] = _summarize_metric_rows(rows_exact, ("em", "substring_match", "f1", "tok_acc"))
            agg["semantic_niah"] = _summarize_metric_rows(rows_sem, ("em", "substring_match", "f1", "tok_acc"))
            agg["peak_vram_mb"] = _peak_vram_mb(device)
            agg["samples"] = samples_arm
            if sys_name == "true_int4_kv_cache":
                agg["analytical_kv_4k_mib"] = round(estimate_int4_kv_bytes(wrapper.cfg, 4096) / (1024.0 ** 2), 2)
            out["systems"][sys_name] = agg
            print(
                f"[G5 TRUE KV {sys_name}] PPL={agg['ppl']:.2f} EM={agg['em']:.3f} Sub={agg['substring_match']:.3f} "
                f"F1={agg['f1']:.3f} tok_acc={agg['tok_acc']:.3f} tok={agg['active_tokens_mean']:.0f}",
                flush=True,
            )

    return out


# ---------------------------------------------------------------------------
# Pillar 5.4: Fair Gate-2 Formatting + Gate-3 Coding Scale-Up (N=24)
# ---------------------------------------------------------------------------

_FAIR_G2_FEWSHOT_PREFIX = (
    "Format instructions: Reply with ONLY the concise exact value (matching unit format).\n"
    "Example 1 -> Question: What is the approved budget for DemoA? Answer: 40M\n"
    "Example 2 -> Question: What is the approved deployment state for DemoB? Answer: blue-green-prod\n\n"
)


@torch.no_grad()
def run_fair_gate2_eval(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    bge_toc: _PlugInBiEncoderTOC | None = None,
    eval_n: int = 48,
    salience_route_weight: float = 1.5,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Gate 2 Fair Formatting: identical 2-shot format prefix + neutral strict_em, normalized_numeric_em, and f1."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    eff_span_len = int(getattr(wrapper.cfg, "salience_span_len", 16))
    wrapper.eval()

    n_each = (6 if is_mock else max(24, eval_n // 2))
    num_p = 8 if is_mock else 16
    eps_budget = list(
        AdversarialLivingBookDataset(
            num_episodes=n_each, num_pages=num_p, page_size=wrapper.cfg.page_size, num_edits=3,
            mode="budget", split="eval", tokenizer=tokenizer, vocab_size=vocab_size,
            seed=seed + 40, background_token_pool=val_pool,
        )
    )
    eps_sym = list(
        AdversarialLivingBookDataset(
            num_episodes=n_each, num_pages=num_p, page_size=wrapper.cfg.page_size, num_edits=5,
            mode="symbolic", split="eval", tokenizer=tokenizer, vocab_size=vocab_size,
            seed=seed + 41, background_token_pool=val_pool,
        )
    )
    all_eps = eps_budget + eps_sym

    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR + _FAIR_G2_FEWSHOT_PREFIX, add_special_tokens=False),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    systems = ("bm25_rag_immutable", "bm25_rag_append_errata", "bm25_rag_oracle_verified", "lslc_v3_e2e_gpu")
    results: dict[str, Any] = {}

    for sys_name in systems:
        rows: list[dict[str, float]] = []
        rows_by_mode: dict[str, list[dict[str, float]]] = {"budget": [], "symbolic": []}
        for ep in all_eps:
            clean_q = tokenizer.decode(ep.metadata["router_query_ids"].tolist(), skip_special_tokens=True).strip()
            gold = str(ep.metadata["answer_str"]).strip()
            p_prompt = torch.tensor(
                tokenizer.encode(_format_qa_prompt(clean_q), add_special_tokens=False), dtype=torch.long, device=device
            ).unsqueeze(0)

            if sys_name.startswith("bm25_rag_"):
                if sys_name == "bm25_rag_oracle_verified":
                    src_pages = ep.metadata["verified_final_pages"]
                    docs = [tokenizer.decode(src_pages[i], skip_special_tokens=True) for i in range(src_pages.shape[0])]
                else:
                    src_pages = ep.context_pages
                    docs = [tokenizer.decode(src_pages[i], skip_special_tokens=True) for i in range(src_pages.shape[0])]
                    if sys_name == "bm25_rag_append_errata":
                        for p_i, e_ids in ep.edit_pages:
                            docs[int(p_i)] = docs[int(p_i)] + "\n" + tokenizer.decode(e_ids, skip_special_tokens=True)
                bm25 = BM25PageRetriever(docs)
                rr = bm25.route(clean_q, k_deep=min(4, len(docs)), k_skim=0, device=device)
                chosen = rr.deep_idx[0].tolist()
                rag_txt = "\n\n".join(docs[i].strip() for i in chosen) + "\n"
                r_ids = _encode_unpadded_rag_ids(rag_txt, tokenizer, vocab_size, device)
                pref = embed_fn(r_ids).to(llm_dtype)
            else:
                p_t = ep.context_pages.unsqueeze(0).to(device)
                z_b, bk, tc = _encode_book_chunked(
                    wrapper.lslc, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled,
                    chunk_size=16, special_token_ids=special_ids, salience_span_len=eff_span_len,
                )
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    for p_i, e_ids_cpu in ep.edit_pages:
                        e_ids = e_ids_cpu.unsqueeze(0).to(device)
                        ov = (
                            extract_contextual_edit_embeds(wrapper.llm, e_ids, layer_idx=wrapper.cfg.contextual_edit_layer)
                            if wrapper.cfg.use_contextual_edit_encoding
                            else None
                        )
                        z_b, _, bk = wrapper.lslc.apply_page_delta(
                            z_b, torch.tensor([int(p_i)], device=device), e_ids, embed_tokens_fn=embed_fn,
                            exact_bank=bk, vocab_size=vocab_size, edit_embeds_override=ov, in_place=True, bank_write_threshold=0.45,
                        )
                    rq_ids = ep.metadata["router_query_ids"].unsqueeze(0).to(device)
                    ext_s = None
                    if bge_toc is not None:
                        bge_p4 = bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=64)
                        ext_s = bge_toc.score_book(bge_toc.encode_texts([clean_q], is_query=True), bge_p4)
                    _, asm = wrapper.lslc.route_and_assemble(
                        z_b, tc, embed_fn(rq_ids).mean(dim=1), bk, embed_fn,
                        k_deep=4, k_skim=2, mode="full", query_input_ids=rq_ids,
                        salience_route_weight=salience_route_weight, exact_bank_attenuation="none", external_dense_logits=ext_s,
                    )
                pref = asm.prefix_embeds.to(llm_dtype)

            gen_txt = _greedy_generate_qa(wrapper.llm, pref, p_prompt, tokenizer, max_new_tokens=20, user_open_ids=user_open_ids)
            qm = compute_community_qa_metrics(gen_txt, gold)
            canon_pred = canonicalize_numeric_answer(gen_txt.splitlines()[0] if gen_txt.strip() else "")
            canon_gold = canonicalize_numeric_answer(gold)
            norm_num_em = 1.0 if (canon_gold and canon_pred == canon_gold) else qm["extracted_em"]
            rec = {
                "strict_em": qm["extracted_em"],
                "normalized_numeric_em": float(norm_num_em),
                "f1": qm["token_f1"],
                "active_tokens": float(pref.shape[1]),
            }
            rows.append(rec)
            ep_mode = str(ep.metadata.get("mode", "budget"))
            if ep_mode in rows_by_mode:
                rows_by_mode[ep_mode].append(rec)
        agg = _summarize_metric_rows(rows, ("strict_em", "normalized_numeric_em", "f1", "active_tokens"))
        agg["by_mode"] = {
            m: _summarize_metric_rows(m_rows, ("strict_em", "normalized_numeric_em", "f1", "active_tokens"))
            for m, m_rows in rows_by_mode.items()
        }
        results[sys_name] = agg
        print(
            f"[G5 FAIR-G2 {sys_name}] strict_EM={agg['strict_em']:.3f} "
            f"norm_num_EM={agg['normalized_numeric_em']:.3f} F1={agg['f1']:.3f} "
            f"(budget={agg['by_mode']['budget']['normalized_numeric_em']:.3f}, "
            f"symbolic={agg['by_mode']['symbolic']['normalized_numeric_em']:.3f})",
            flush=True,
        )

    return results


# 18 brand-new disjoint eval specs expanding Gate 3 Coding Benchmark from N=6 to N=24
_EVAL_SPECS_EXT_18: tuple[_MigrationSpec, ...] = (
    _MigrationSpec("eval_07_verify_digest", "hashguard", "verify_digest", "blob", '"payload.bin"', "hash_algo", '"sha256"', '"md5"', "legacy-digest"),
    _MigrationSpec("eval_08_publish_event", "buscraft", "publish_event", "topic", '"orders"', "broker", '"kafka"', '"nats"', "legacy-event"),
    _MigrationSpec("eval_09_allocate_slot", "poolmgr", "allocate_slot", "worker_id", '"w-01"', "rack", '"az1"', '"az3"', "legacy-slot"),
    _MigrationSpec("eval_10_sign_payload", "cryptobox", "sign_payload", "message", '"hello"', "keyring", '"k-prod"', '"k-test"', "legacy-sign"),
    _MigrationSpec("eval_11_query_index", "searchkit", "query_index", "term", '"invoice"', "corpus", '"c-main"', '"c-old"', "legacy-index"),
    _MigrationSpec("eval_12_rotate_secret", "vaultops", "rotate_secret", "secret_name", '"db_pass"', "vault_rule", '"strict"', '"relaxed"', "legacy-secret"),
    _MigrationSpec("eval_13_schedule_backup", "snapflow", "schedule_backup", "volume", '"vol-a"', "retention", '"cold"', '"hot"', "legacy-backup"),
    _MigrationSpec("eval_14_compile_rule", "ruleforge", "compile_rule", "expr", '"x > 0"', "dialect", '"v3"', '"v1"', "legacy-rule"),
    _MigrationSpec("eval_15_check_quota", "limitbox", "check_quota", "org_id", '"org-7"', "cadence", '"hourly"', '"daily"', "legacy-quota"),
    _MigrationSpec("eval_16_fetch_artifact", "packhub", "fetch_artifact", "pkg_ref", '"lib-2"', "mirror", '"primary"', '"backup"', "legacy-artifact"),
    _MigrationSpec("eval_17_drain_queue", "pipeworks", "drain_queue", "queue_id", '"q-in"', "drain_kind", '"graceful"', '"force"', "legacy-queue"),
    _MigrationSpec("eval_18_audit_record", "ledgerkit", "audit_record", "entry_id", '"rec-4"', "verbosity", '"full"', '"brief"', "legacy-audit"),
    _MigrationSpec("eval_19_sync_replica", "repliflow", "sync_replica", "node_id", '"n-east"', "transport", '"tls"', '"plain"', "legacy-replica"),
    _MigrationSpec("eval_20_parse_filter", "filterbox", "parse_filter", "raw_expr", '"status=ok"', "grammar", '"ast"', '"regex"', "legacy-filter"),
    _MigrationSpec("eval_21_bind_socket", "netcraft", "bind_socket", "host_addr", '"127.0.0.1"', "proto", '"tcp"', '"udp"', "legacy-socket"),
    _MigrationSpec("eval_22_flush_buffer", "streamkit", "flush_buffer", "buf_id", '"b-0"', "durability", '"fsync"', '"async"', "legacy-buffer"),
    _MigrationSpec("eval_23_resolve_alias", "namegrid", "resolve_alias", "alias", '"svc.local"', "realm", '"prod"', '"canary"', "legacy-alias"),
    _MigrationSpec("eval_24_export_report", "repocraft", "export_report", "report_id", '"r-12"', "codec", '"parquet"', '"csv"', "legacy-report"),
)


def build_coding_n24_tasks(num_distractor_files: int = 58) -> list[MigrationTask]:
    """Build 24 cross-file API migration scenarios (6 original + 18 disjoint extended specs) on 65-file repos."""
    all_specs = _EVAL_SPECS + _EVAL_SPECS_EXT_18
    assert len({s.pkg_name for s in all_specs}) == 24
    assert len({s.symbol_name for s in all_specs}) == 24
    return [
        _build_task_from_spec(spec, split="eval", task_idx=idx, num_distractor_files=num_distractor_files)
        for idx, spec in enumerate(all_specs)
    ]


def build_pathless_coding_query(task: MigrationTask) -> str:
    """Path-hint-free query referencing ONLY the API symbol & architectural contract (ZERO .py/.md paths!)."""
    q = f"API contract signature and approved default policy for {task.pkg_name}.{task.symbol_name}"
    assert ".py" not in q and ".md" not in q and "/" not in q, f"Path hint leaked into query: {q!r}"
    return q


def select_dynamic_margin_k_deep(
    logits_1d: torch.Tensor,
    *,
    min_k: int = 2,
    mid_k: int = 4,
    max_k: int = 6,
    high_margin_thresh: float = 1.5,
    mid_margin_thresh: float = 0.6,
) -> int:
    """Dynamic margin-gated k_deep selection based on router logit separation (no hardcoded k_deep=2)."""
    n = int(logits_1d.numel())
    if n <= min_k:
        return max(1, n)
    top_vals = torch.topk(logits_1d.float(), k=min(n, max_k + 1)).values
    # Compare rank-2 score (needed because migration requires BOTH core.py AND session_notes) against rank-3
    m_23 = float((top_vals[min(1, len(top_vals) - 1)] - top_vals[min(2, len(top_vals) - 1)]).item())
    if m_23 >= high_margin_thresh:
        return min(min_k, n)
    if m_23 >= mid_margin_thresh:
        return min(mid_k, n)
    return min(max_k, n)


@torch.no_grad()
def run_coding_n24_eval(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    device: torch.device,
    *,
    bge_toc: _PlugInBiEncoderTOC | None = None,
    num_distractors: int = 58,
    max_tasks: int = 24,
    salience_route_weight: float = 1.5,
    lslc_k_skim: int = 2,
    max_new_tokens: int = 256,
    systems_override: tuple[str, ...] | None = None,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Evaluate Gate 3 Coding N=24 (65-file repos, zero path hints, transparent scan files_read, dynamic margin k_deep)."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    eff_span_len = int(getattr(wrapper.cfg, "salience_span_len", 16))
    wrapper.eval()

    n_dist = 8 if is_mock else num_distractors
    tasks = build_coding_n24_tasks(num_distractor_files=n_dist)[: (4 if is_mock else max_tasks)]
    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)

    systems = (
        tuple(systems_override)
        if systems_override is not None
        else ("bm25_rag_k2", "bm25_rag_k4", "bm25_rag_k8", "lslc_v3_k2", "lslc_v3_k4", "lslc_v3_dynamic_k")
    )
    out_systems: dict[str, Any] = {}

    for sys_name in systems:
        rows: list[dict[str, object]] = []
        dyn_ks: list[int] = []
        init_scans: list[int] = []
        for task in tasks:
            harness = CodingAgentHarness(task, use_initial_files=True)
            pack = _build_repo_pages(task, tokenizer, wrapper.cfg.page_size)
            page_paths: list[str] = list(pack["page_paths"])
            core_page_idx = int(pack["core_page_indices"][0])
            session_page_idx = int(pack["session_page_idx"])
            pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
            core_fmt_new = f"# FILE: {task.core_file}\n{task.core_new_content}\n"

            # Transparently count initial repo scan across ALL systems
            initial_scan_files = len(task.initial_files)
            init_scans.append(initial_scan_files)
            harness.apply_file_edit(task.core_file, task.core_new_content)

            q_txt = build_pathless_coding_query(task)
            bm25 = None
            bm25_docs: list[str] = []
            z_book = exact_bank = toc_embeds = bge_pages = None
            mem_ms = 0.0

            if sys_name.startswith("bm25_rag_k"):
                bm25_docs = list(pack["page_texts"])
                t0 = time.perf_counter()
                bm25 = BM25PageRetriever(bm25_docs)
                bm25_docs[core_page_idx] = core_fmt_new
                bm25.update_page(core_page_idx, core_fmt_new)
                sess_txt = bm25_docs[session_page_idx]
                for turn in task.session_turns:
                    sess_txt = sess_txt.rstrip() + "\n" + str(turn["text"]) + "\n"
                bm25_docs[session_page_idx] = sess_txt
                bm25.update_page(session_page_idx, sess_txt)
                mem_ms = (time.perf_counter() - t0) * 1000.0
            else:
                page_tensor = pack["page_tensor"].to(device)
                z_book, exact_bank, toc_embeds = _encode_book_chunked(
                    wrapper.lslc, page_tensor, embed_fn, vocab_size, device, amp_dtype, autocast_enabled,
                    chunk_size=32, special_token_ids=special_ids, salience_span_len=eff_span_len,
                )
                if bge_toc is not None:
                    bge_pages = bge_toc.encode_book_pages(page_tensor, None, tokenizer, num_spans=4, batch_size=64)
                core_toks_new = _pad_or_chunk_tokens(tokenizer.encode(core_fmt_new, add_special_tokens=False), wrapper.cfg.page_size, pad_id)[0]
                core_page_t = torch.tensor(core_toks_new, dtype=torch.long, device=device).unsqueeze(0)
                _sync_device(device)
                t0 = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    z_book, exact_bank, toc_embeds = wrapper.lslc.reencode_page(
                        z_book, toc_embeds, exact_bank, page_idx=core_page_idx, new_page_token_ids=core_page_t,
                        embed_tokens_fn=embed_fn, vocab_size=vocab_size, in_place=True,
                        special_token_ids=special_ids, salience_span_len=eff_span_len,
                    )
                    if bge_toc is not None and bge_pages is not None:
                        bge_pages[:, core_page_idx : core_page_idx + 1] = bge_toc.encode_book_pages(
                            core_page_t.unsqueeze(1), None, tokenizer, num_spans=4, batch_size=16
                        )
                    sess_p_t = torch.tensor([session_page_idx], dtype=torch.long, device=device)
                    for turn in task.session_turns:
                        e_toks = torch.tensor(tokenizer.encode(str(turn["text"]), add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
                        ov = (
                            extract_contextual_edit_embeds(wrapper.llm, e_toks, layer_idx=wrapper.cfg.contextual_edit_layer)
                            if wrapper.cfg.use_contextual_edit_encoding
                            else None
                        )
                        z_book, _, exact_bank = wrapper.lslc.apply_page_delta(
                            z_book, sess_p_t, e_toks, embed_tokens_fn=embed_fn, exact_bank=exact_bank,
                            vocab_size=vocab_size, edit_embeds_override=ov, in_place=True, bank_write_threshold=0.45,
                        )
                _sync_device(device)
                mem_ms = (time.perf_counter() - t0) * 1000.0

            prefix_toks: list[int] = []
            core_hits: list[float] = []
            sess_hits: list[float] = []
            both_hits: list[float] = []
            t_wall0 = time.perf_counter()

            for caller_path in task.caller_files:
                cur_code = harness.read_file(caller_path)
                excluded = {i for i, p in enumerate(page_paths) if p == caller_path}
                if sys_name.startswith("bm25_rag_k"):
                    assert bm25 is not None
                    k_val = int(sys_name.rsplit("_k", 1)[-1])
                    harness.search_calls += 1
                    rr = bm25.route(q_txt, k_deep=len(bm25_docs), k_skim=0, device=device)
                    ordered = [int(i) for i in torch.argsort(rr.logits[0], descending=True).tolist() if int(i) not in excluded]
                    chosen = ordered[:k_val]
                    for idx_c in chosen:
                        p_n = page_paths[idx_c]
                        if p_n != task.session_notes_path and p_n in harness.files:
                            _ = harness.read_file(p_n)
                    chosen_set = set(chosen)
                    rag_txt = "\n\n".join(bm25_docs[i].rstrip() for i in chosen) + "\n"
                    r_ids = _encode_unpadded_rag_ids(rag_txt, tokenizer, vocab_size, device)
                    pref = embed_fn(r_ids).to(llm_dtype)
                else:
                    rq_ids = torch.tensor(tokenizer.encode(q_txt, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
                    ext_s = (
                        bge_toc.score_book(bge_toc.encode_texts([q_txt], is_query=True), bge_pages)
                        if (bge_toc is not None and bge_pages is not None)
                        else None
                    )
                    # Fixed k_deep=2 / k_deep=4 vs dynamic margin-gated k_deep (2..6, high_margin=0.6)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        if sys_name == "lslc_v3_k2":
                            kd_use = 2
                        elif sys_name == "lslc_v3_k4":
                            kd_use = 4
                        else:
                            r_probe, _ = _lslc_route_and_assemble_excluding(
                                wrapper=wrapper, z_book=z_book, toc_embeds=toc_embeds, exact_bank=exact_bank,
                                embed_fn=embed_fn, router_query_ids=rq_ids, k_deep=2, k_skim=0,
                                salience_route_weight=salience_route_weight, excluded_indices=excluded,
                                mode="toc_only", exact_bank_attenuation="none", external_dense_logits=ext_s,
                            )
                            kd_use = select_dynamic_margin_k_deep(
                                r_probe.logits[0],
                                min_k=2,
                                mid_k=4,
                                max_k=6,
                                high_margin_thresh=0.6,
                                mid_margin_thresh=0.25,
                            )
                            dyn_ks.append(kd_use)
                        r_out, pref = _lslc_route_and_assemble_excluding(
                            wrapper=wrapper, z_book=z_book, toc_embeds=toc_embeds, exact_bank=exact_bank,
                            embed_fn=embed_fn, router_query_ids=rq_ids, k_deep=kd_use, k_skim=lslc_k_skim,
                            salience_route_weight=salience_route_weight, excluded_indices=excluded,
                            mode="full", exact_bank_attenuation="none", flat_toc_max_tokens=16, external_dense_logits=ext_s,
                        )
                    chosen_set = set(int(x) for x in r_out.deep_idx[0].tolist())
                    pref = pref.to(llm_dtype)

                prefix_toks.append(int(pref.shape[1]))
                core_hits.append(float(core_page_idx in chosen_set))
                sess_hits.append(float(session_page_idx in chosen_set))
                both_hits.append(float((core_page_idx in chosen_set) and (session_page_idx in chosen_set)))

                prompt_txt = _build_edit_prompt(task, caller_path, cur_code)
                p_ids = torch.tensor(tokenizer.encode(prompt_txt, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
                raw_out = _greedy_generate_code_until_fence(
                    wrapper.llm, pref, p_ids, tokenizer, max_new_tokens=(24 if is_mock else max_new_tokens),
                    stop_fn=lambda d: _should_stop_code_gen(d, expect_opening_fence=False), user_open_ids=user_open_ids,
                )
                patched = _extract_python_code_block("```python\n" + raw_out, cur_code)
                harness.apply_file_edit(caller_path, patched)

            # Genuine homonym preservation check on task.homonym_file (identically across all systems)
            raw_homonym = harness.read_file(task.homonym_file)
            excl_h = {i for i, p in enumerate(page_paths) if p == task.homonym_file}
            if sys_name.startswith("bm25_rag_k"):
                assert bm25 is not None
                k_val = int(sys_name.rsplit("_k", 1)[-1])
                harness.search_calls += 1
                rr_h = bm25.route(q_txt, k_deep=len(bm25_docs), k_skim=0, device=device)
                ord_h = [int(i) for i in torch.argsort(rr_h.logits[0], descending=True).tolist() if int(i) not in excl_h]
                ch_h = ord_h[:k_val]
                rag_h = "\n\n".join(bm25_docs[i].rstrip() for i in ch_h) + "\n"
                pref_h = embed_fn(_encode_unpadded_rag_ids(rag_h, tokenizer, vocab_size, device)).to(llm_dtype)
            else:
                rq_ids = torch.tensor(tokenizer.encode(q_txt, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
                ext_s = (
                    bge_toc.score_book(bge_toc.encode_texts([q_txt], is_query=True), bge_pages)
                    if (bge_toc is not None and bge_pages is not None)
                    else None
                )
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    if sys_name == "lslc_v3_k2":
                        kd_h = 2
                    elif sys_name == "lslc_v3_k4":
                        kd_h = 4
                    else:
                        r_pr_h, _ = _lslc_route_and_assemble_excluding(
                            wrapper=wrapper, z_book=z_book, toc_embeds=toc_embeds, exact_bank=exact_bank,
                            embed_fn=embed_fn, router_query_ids=rq_ids, k_deep=2, k_skim=0,
                            salience_route_weight=salience_route_weight, excluded_indices=excl_h,
                            mode="toc_only", exact_bank_attenuation="none", external_dense_logits=ext_s,
                        )
                        kd_h = select_dynamic_margin_k_deep(
                            r_pr_h.logits[0],
                            min_k=2,
                            mid_k=4,
                            max_k=6,
                            high_margin_thresh=0.6,
                            mid_margin_thresh=0.25,
                        )
                    _, pref_h = _lslc_route_and_assemble_excluding(
                        wrapper=wrapper, z_book=z_book, toc_embeds=toc_embeds, exact_bank=exact_bank,
                        embed_fn=embed_fn, router_query_ids=rq_ids, k_deep=kd_h, k_skim=lslc_k_skim,
                        salience_route_weight=salience_route_weight, excluded_indices=excl_h,
                        mode="full", exact_bank_attenuation="none", flat_toc_max_tokens=16, external_dense_logits=ext_s,
                    )
                pref_h = pref_h.to(llm_dtype)
            h_prompt = _build_homonym_prompt(task, raw_homonym)
            h_ids = torch.tensor(tokenizer.encode(h_prompt, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
            raw_h = _greedy_generate_code_until_fence(
                wrapper.llm, pref_h, h_ids, tokenizer, max_new_tokens=(16 if is_mock else 96),
                stop_fn=_should_stop_homonym_gen, user_open_ids=user_open_ids,
            )
            if "KEEP_UNCHANGED" not in _strip_think_blocks(raw_h).upper():
                cand_h = _extract_python_code_block(raw_h, raw_homonym)
                if cand_h.strip() and cand_h.strip() != raw_homonym.strip():
                    harness.apply_file_edit(task.homonym_file, cand_h)

            harness.memory_update_ms = mem_ms
            harness.active_prefix_tokens = int(round(sum(prefix_toks) / max(1, len(prefix_toks))))
            harness.peak_vram_mb = _peak_vram_mb(device)
            ev = harness.evaluate(run_subprocess=True)
            rows.append(
                {
                    "task_id": ev.task_id,
                    "patch_correct": ev.patch_correct,
                    "hidden_test_passed": ev.hidden_test_passed,
                    "stale_signature_used": ev.stale_signature_used,
                    "decoy_value_used": ev.decoy_value_used,
                    "homonym_preserved": ev.homonym_preserved,
                    "files_read_total": int(initial_scan_files + ev.files_read_total),
                    "initial_scan_files": initial_scan_files,
                    "incremental_files_read": int(ev.files_read_total),
                    "repeated_file_reads": int(ev.repeated_file_reads),
                    "search_calls": int(ev.search_calls),
                    "active_prefix_tokens": int(ev.active_prefix_tokens),
                    "memory_update_ms": round(mem_ms, 3),
                    "modular_reencode_ms": round(mem_ms, 3),
                    "delta_update_ms": 0.0,
                    "generation_ms": 0.0,
                    "wall_clock_ms": round(ev.wall_clock_ms, 3),
                    "peak_vram_mb": ev.peak_vram_mb,
                    "router_core_hit_rate": round(sum(core_hits) / max(1, len(core_hits)), 4),
                    "router_session_hit_rate": round(sum(sess_hits) / max(1, len(sess_hits)), 4),
                    "router_both_hit_rate": round(sum(both_hits) / max(1, len(both_hits)), 4),
                }
            )
        agg = _aggregate_system_rows(rows)
        agg["pass_at_1_ci95"] = _wilson_ci95(float(agg["pass_at_1"]), len(rows))
        agg["initial_scan_files_mean"] = round(sum(init_scans) / max(1, len(init_scans)), 2)
        agg["path_hints_in_query"] = False
        if dyn_ks:
            agg["dynamic_k_deep_mean"] = round(sum(dyn_ks) / len(dyn_ks), 2)
        out_systems[sys_name] = agg
        print(
            f"[G5 CODING N={len(rows)} {sys_name}] pass@1={agg['pass_at_1']:.3f} both_hit={agg['router_both_hit_rate']:.3f} "
            f"files_read_total={agg['files_read_mean']:.1f} tok={agg['active_prefix_tokens_mean']:.0f}",
            flush=True,
        )

    return {
        "num_tasks": len(tasks),
        "num_distractors": n_dist,
        "path_hints_removed": True,
        "systems": out_systems,
    }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gerbang 5: Mechanistic Ablations, True KV Baselines, Fair Gate-2 & Coding N=24.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--mock-llm", action="store_true", help="Run 0-CU CPU smoke test with TinyMockCausalLM.")
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated: ablation_2x2_edit,ablation_1d_vs_3d,true_kv_baselines,fair_g2_and_coding_n24 or 'all'.",
    )
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--bge-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--use-bge-toc", action="store_true", default=True)
    parser.add_argument("--no-bge-toc", dest="use_bge_toc", action="store_false")
    parser.add_argument("--k-exact", type=int, default=48)
    parser.add_argument("--salience-span-len", type=int, default=16)
    parser.add_argument("--salience-route-weight", type=float, default=1.5)
    parser.add_argument("--eval-n", type=int, default=48)
    parser.add_argument("--train-steps", type=int, default=240)
    parser.add_argument("--coding-distractors", type=int, default=58)
    parser.add_argument("--coding-max-tasks", type=int, default=24)
    parser.add_argument("--coding-systems", type=str, default="", help="Optional comma-separated coding systems override.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-json", type=str, default="/content/lslc_results/v3_gerbang5_ablations.json")
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    raw_tasks = {t.strip().lower() for t in args.tasks.split(",") if t.strip()}
    all_set = {"ablation_2x2_edit", "ablation_1d_vs_3d", "true_kv_baselines", "fair_g2_and_coding_n24"}
    run_tasks = all_set if "all" in raw_tasks else raw_tasks

    if args.mock_llm:
        vocab_size = 2048
        cfg = LSLCConfig.tiny_mock(
            use_contextual_edit_encoding=True,
            contextual_edit_layer=1,
            use_contextual_query_encoding=False,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=min(args.k_exact, 48),
            salience_span_len=args.salience_span_len,
        )
        llm = TinyMockCausalLM(vocab_size=vocab_size, d_llm=cfg.d_llm, num_layers=cfg.num_llm_layers)
        tokenizer = _MockCodeTokenizer(vocab_size=vocab_size)
        device = torch.device("cpu")
        train_pool = [2 + (i % (vocab_size - 2)) for i in range(4096)]
        val_pool = [2 + ((i * 7) % (vocab_size - 2)) for i in range(2048)]
        data_source = "mock-synthetic-pool"
    else:
        cfg = LSLCConfig.from_model_name(
            args.model,
            use_contextual_edit_encoding=True,
            use_contextual_query_encoding=False,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=args.k_exact,
            salience_span_len=args.salience_span_len,
        )
        cfg.contextual_edit_layer = cfg.num_llm_layers // 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        llm, tokenizer = load_frozen_qwen3(cfg, load_in_4bit=False, device=device)
        train_pool, val_pool, data_source = _load_wikitext_pools(tokenizer)

    wrapper = FrozenQwenLSLCWrapper(cfg, llm).to(device)
    ckpt_info = _load_lslc_checkpoint(wrapper.lslc, args.init_checkpoint) if args.init_checkpoint else None
    bge_toc = _PlugInBiEncoderTOC(args.bge_model, device=device, is_mock=args.mock_llm) if args.use_bge_toc else None

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": "3.0.0-gerbang5-ablations",
        "model": "TinyMockCausalLM" if args.mock_llm else args.model,
        "mock_llm": args.mock_llm,
        "device": str(device),
        "data_source": data_source,
        "init_checkpoint": ckpt_info,
        "config": {
            "page_size": cfg.page_size,
            "k_exact": cfg.k_exact,
            "salience_span_len": cfg.salience_span_len,
            "salience_route_weight": args.salience_route_weight,
            "bge_model": None if bge_toc is None else bge_toc.model_name,
            "eval_n": args.eval_n,
        },
        "ablation_2x2_edit": None,
        "ablation_1d_vs_3d": None,
        "true_kv_baselines": None,
        "fair_g2_and_coding_n24": None,
    }

    if "ablation_2x2_edit" in run_tasks:
        payload["ablation_2x2_edit"] = run_ablation_2x2_edit(
            wrapper, tokenizer, val_pool, device, bge_toc=bge_toc,
            num_verbatim=max(24, args.eval_n // 2), num_implicit=max(24, args.eval_n // 2),
            salience_route_weight=args.salience_route_weight, seed=args.seed, is_mock=args.mock_llm,
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if "ablation_1d_vs_3d" in run_tasks:
        payload["ablation_1d_vs_3d"] = run_ablation_1d_vs_3d(
            wrapper, tokenizer, train_pool, val_pool, device,
            train_steps=args.train_steps, eval_n=args.eval_n, seed=args.seed, is_mock=args.mock_llm,
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if "true_kv_baselines" in run_tasks:
        payload["true_kv_baselines"] = run_true_kv_baselines(
            wrapper, tokenizer, train_pool, val_pool, device, bge_toc=bge_toc,
            eval_n=args.eval_n, icae_steps=args.train_steps,
            salience_route_weight=args.salience_route_weight, seed=args.seed, is_mock=args.mock_llm,
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    coding_sys_ov = (
        tuple(s.strip() for s in args.coding_systems.split(",") if s.strip())
        if args.coding_systems.strip()
        else None
    )
    if "fair_g2_and_coding_n24" in run_tasks:
        payload["fair_g2_and_coding_n24"] = {
            "gate2_fair_formatting": run_fair_gate2_eval(
                wrapper, tokenizer, val_pool, device, bge_toc=bge_toc,
                eval_n=args.eval_n, salience_route_weight=args.salience_route_weight, seed=args.seed, is_mock=args.mock_llm,
            ),
            "gate3_coding_n24": run_coding_n24_eval(
                wrapper, tokenizer, device, bge_toc=bge_toc,
                num_distractors=args.coding_distractors, max_tasks=args.coding_max_tasks,
                salience_route_weight=args.salience_route_weight, systems_override=coding_sys_ov, is_mock=args.mock_llm,
            ),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if "coding_n24_only" in run_tasks:
        payload["gate3_coding_n24"] = run_coding_n24_eval(
            wrapper, tokenizer, device, bge_toc=bge_toc,
            num_distractors=args.coding_distractors, max_tasks=args.coding_max_tasks,
            salience_route_weight=args.salience_route_weight, systems_override=coding_sys_ov, is_mock=args.mock_llm,
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[DONE] Wrote Gerbang 5 results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
