#!/usr/bin/env python3
"""Path A vs. Path B Tie-Breaker Suite + 5 Code-Audit Hardening Pillars (with Reviewer #3 Addendum).

Supports `--mock-llm` (0 CU local CPU verification) and modular `--tasks`:
  1. `gate2_tiebreaker_n200`:
     - N=200 hold-out episodes (100 budget + 100 symbolic k=5), identical 2-shot format prefix.
     - 5 systems in 1 unified run: `bm25_rag_append_errata`, `gated_bm25_rag`,
       `standalone_discrete_span_bank` (ZERO `LayeredLatentEncoder`, ZERO `Z_book`, BGE BF16 3 KiB/p + Bank 384 B/p),
       `lslc_exact_only`, and `lslc_full`.
     - Multi-seed support (`--seeds 2026,2027,2028`), per-episode predictions & binary scores,
       exact two-sided McNemar p-values, paired 95% bootstrap CIs, gate FAR/FRR, and real independent timing.
  2. `coding_tiebreaker_n24`:
     - All 24 65-file cross-file scenarios (0 path hints) in 1 single run & 1 single JSON:
       `bm25_rag_k2/k4/k8`, `bge_4span_rag_k2/k4`, `lslc_exact_only_k2/k4`, `lslc_full_k2/k4`
       with strictly identical routed pages for `lslc_full` vs `lslc_exact_only`, per-task patches,
       exact McNemar p-values, paired bootstrap CIs, and automatic Path A criterion check (`full - exact_only > +20 pp`).
  3. `ruler_bge_and_unified_profile`:
     - RULER 16K & 32K (N=60/len) + Raw-Q 16K (N=60) with `full_context_bf16`, `bm25_rag_k1/k2/k4`,
       `bge_4span_rag_k1/k2/k4`, independent `standalone_discrete_span_bank`, and `lslc_full`.
     - Unified single-GPU Systems Profile (`k_exact=48`, BGE BF16 3 KiB/p + 33M params) sweeping `k_deep in {2,4,8}`.
     - Direct real Prefix KV-Cache Reuse benchmark post-edit verifying `bank1_pkv_max_abs_diff == 0.0`
       and measuring real wall-clock prefill speedup with vs. without Bank-1 KV reuse.
  4. `ood_kv_baselines`:
     - 100% OOD RULER-style (N=48 at 4K) + WikiText-2 continuation PPL (N=16) comparing:
       `full_kv_bf16`, `sim_int4_kv_qdq` (honestly labeled QDQ), `query_agnostic_h2o` (eager eviction vs clean SDPA),
       `query_aware_snapkv` (query-token observation window attention eviction + valid RoPE `position_ids` + clean SDPA),
       `standalone_discrete_span_bank`, and `lslc_full` (NO ICAE strawman).
  5. `longmemeval_discrete_probe`:
     - Fast N=9 stratified probe (explicit `packing_mode: "gold_prioritized_64p"`) testing
       `bm25_rag_k1/k2`, `bge_4span_rag_k1/k2`, `standalone_discrete_k64_d4`, `standalone_discrete_k48_d6`,
       and `lslc_full_k48_d4` with `answer_substr_in_exact_bank_rate` diagnostics before spending CU on N=60.
  6. `ablation_edit_masked_old`:
     - 5-cell iso-token Edit Ablation (N=48: 24 verbatim + 24 implicit) adding Reviewer #3's control cell
       `deltaZ_ON_bank_masked_old` (masking stale old-value tokens in `bank_orig` so continuous Delta-Z is tested
       without competing against raw old tokens) alongside the 2x2 grid + per-episode binary scores & exact McNemar.
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
    _extract_kv_pairs,
    _llm_accepts_position_ids,
    _rebuild_kv_cache,
    compute_memory_safe_obs_attention_and_kv,
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
    _answer_in_chosen_pages,
    _bootstrap_ci95,
    _encode_book_chunked,
    _encode_unpadded_rag_ids,
    _format_qa_prompt,
    _get_special_token_ids,
    _greedy_generate_qa,
    _normalize_qa_text,
    _pack_longmemeval_sessions,
    _peak_vram_mb,
    _reset_peak_vram,
    _sync_device,
    _wilson_ci95,
    build_ruler_episodes,
    compute_community_qa_metrics,
    load_longmemeval_stratified,
)
from colab_run_v3_gerbang5_ablations import (  # noqa: E402
    _FAIR_G2_FEWSHOT_PREFIX,
    build_2x2_edit_episodes,
    build_coding_n24_tasks,
    build_pathless_coding_query,
    canonicalize_numeric_answer,
)
from lslc import (  # noqa: E402
    AdversarialLivingBookDataset,
    ChunkedTextPageDataset,
    CodingAgentHarness,
    ContextualEditGateHead,
    FrozenQwenLSLCWrapper,
    LSLCConfig,
    LSLCSystem,
    PathBDiscreteMemory,
    RouterOutput,
    SparseExactBank,
    TinyMockCausalLM,
    canonicalize_w2n_answer,
    compute_prefix_lcs_len,
    compute_value_em,
    estimate_int4_kv_bytes,
    extract_contextual_edit_embeds,
    get_git_provenance,
    get_newline_sep_token_id,
    heuristic_span16_tfidf_bank,
    load_frozen_qwen3,
    load_pathb_checkpoint,
    save_pathb_checkpoint,
    select_amp_dtype,
)
from lslc.datasets import _encode_text_or_mock  # noqa: E402


# ---------------------------------------------------------------------------
# Statistical Rigor Helpers (Exact Two-Sided McNemar + Paired Bootstrap CI95)
# ---------------------------------------------------------------------------


def exact_mcnemar_test(scores_a: list[int | float], scores_b: list[int | float]) -> dict[str, Any]:
    """Compute exact two-sided McNemar test on paired binary outcomes (no external scipy required)."""
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Paired lengths differ: {len(scores_a)} vs {len(scores_b)}")
    n = len(scores_a)
    n11 = n10 = n01 = n00 = 0
    for xa, xb in zip(scores_a, scores_b):
        a_i = 1 if float(xa) >= 0.5 else 0
        b_i = 1 if float(xb) >= 0.5 else 0
        if a_i == 1 and b_i == 1:
            n11 += 1
        elif a_i == 1 and b_i == 0:
            n10 += 1
        elif a_i == 0 and b_i == 1:
            n01 += 1
        else:
            n00 += 1

    discordant = n10 + n01
    if discordant == 0:
        p_val = 1.0
    else:
        k = min(n10, n01)
        # Exact binomial tail sum computed safely in log-space for arbitrary discordant counts
        log_half_n = -discordant * math.log(2.0)
        log_terms = [math.lgamma(discordant + 1) - math.lgamma(i + 1) - math.lgamma(discordant - i + 1) + log_half_n for i in range(k + 1)]
        max_lt = max(log_terms)
        tail = math.exp(max_lt) * sum(math.exp(lt - max_lt) for lt in log_terms)
        p_val = min(1.0, 2.0 * tail)

    return {
        "n": n,
        "n11_both_correct": n11,
        "n10_a_only": n10,
        "n01_b_only": n01,
        "n00_both_wrong": n00,
        "discordant": discordant,
        "diff_a_minus_b": round((n10 - n01) / max(1, n), 4),
        "p_value_exact": round(p_val, 6),
    }


def paired_bootstrap_ci95(
    scores_a: list[float],
    scores_b: list[float],
    *,
    n_boot: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Compute deterministic paired 95% percentile bootstrap CI for mean(A - B) + exact McNemar if binary."""
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Paired lengths differ: {len(scores_a)} vs {len(scores_b)}")
    n = len(scores_a)
    if n == 0:
        return {"n": 0, "mean_a": 0.0, "mean_b": 0.0, "mean_diff": 0.0, "paired_diff_ci95": [0.0, 0.0]}

    fa = [float(x) for x in scores_a]
    fb = [float(x) for x in scores_b]
    diffs = [a - b for a, b in zip(fa, fb)]
    mean_a = sum(fa) / n
    mean_b = sum(fb) / n
    mean_diff = sum(diffs) / n

    if n == 1:
        ci = [round(mean_diff, 4), round(mean_diff, 4)]
    else:
        rng = random.Random(seed + n * 31)
        boot_means: list[float] = []
        for _ in range(n_boot):
            s = 0.0
            for _j in range(n):
                s += diffs[rng.randrange(n)]
            boot_means.append(s / n)
        boot_means.sort()
        lo = boot_means[max(0, int(0.025 * (n_boot - 1)))]
        hi = boot_means[min(n_boot - 1, int(0.975 * (n_boot - 1)))]
        ci = [round(lo, 4), round(hi, 4)]

    out: dict[str, Any] = {
        "n": n,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "mean_diff": round(mean_diff, 4),
        "paired_diff_ci95": ci,
    }
    if all(x in (0.0, 1.0) for x in fa) and all(x in (0.0, 1.0) for x in fb):
        out["mcnemar"] = exact_mcnemar_test(fa, fb)
    return out


def compute_pairwise_suite(
    scores_by_sys: dict[str, list[float]],
    pairs: list[tuple[str, str]],
    *,
    seed: int = 2026,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for a_name, b_name in pairs:
        if a_name in scores_by_sys and b_name in scores_by_sys and len(scores_by_sys[a_name]) == len(scores_by_sys[b_name]):
            out[f"{a_name}__vs__{b_name}"] = paired_bootstrap_ci95(
                scores_by_sys[a_name], scores_by_sys[b_name], seed=seed
            )
    return out


def _clone_kv_cache(past_key_values):
    """Deep-clone DynamicCache or legacy tuple KV cache so forward mutation never corrupts cached Bank-1."""
    pairs = [(k.clone(), v.clone()) for k, v in _extract_kv_pairs(past_key_values)]
    return _rebuild_kv_cache(past_key_values, pairs)


def _summarize_scalar_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    n = len(rows)
    out: dict[str, Any] = {"num_episodes": n, "ci95": {}}
    for k in keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        if not vals:
            continue
        mean_v = sum(vals) / len(vals)
        out[k] = round(mean_v, 4)
        is_binary = all(v in (0.0, 1.0) for v in vals)
        out["ci95"][k] = _wilson_ci95(mean_v, len(vals)) if is_binary else _bootstrap_ci95(vals)
    return out


# ---------------------------------------------------------------------------
# Standalone Discrete Span-Bank (True Two-Bank Path B: ZERO Encoder, ZERO Z_book!)
# ---------------------------------------------------------------------------


class StandaloneDiscreteSpanBank:
    """100% standalone Path (B) discrete memory: SalienceGate + BGE-4span (BF16) + lean edit gate -> SparseExactBank.

    - Strictly NEVER invokes `LayeredLatentEncoder`, `SpatialLatentProjector`, or `Z_book`.
    - Supports lean `ContextualEditGateHead(hidden_dim=64)` (preferred Path B, ZERO 7.24M DeltaUpdater) OR legacy
      `delta_updater` fallback if explicitly passed without `edit_gate`.
    - Compact `SparseExactBank` (`token_ids` int32 only = 192 B/page before edit; +192 B/page `base_token_ids` once edited).
    - True Two-Bank prefix ordering: unedited Bank-1 pages deterministically FIRST, mutated Bank-2 pages LAST.
    - Span-aligned selective overwrite (Opsi B) with clean newline separator (no `'39Budget'` splicing).
    """

    def __init__(
        self,
        *,
        cfg: LSLCConfig,
        salience_gate: nn.Module,
        hw_coord_embed: nn.Embedding | None = None,
        delta_updater: nn.Module | None = None,
        edit_gate: ContextualEditGateHead | None = None,
        bge_toc: _PlugInBiEncoderTOC | None = None,
        device: torch.device,
        k_exact: int | None = None,
        salience_span_len: int | None = None,
        bank_write_threshold: float = 0.45,
        span_aligned_overwrite: bool = True,
    ) -> None:
        self.cfg = cfg
        self.salience_gate = salience_gate
        self.hw_coord_embed = None  # Zero untrained coordinate weights in Path B
        self.edit_gate = edit_gate
        self.delta_updater = delta_updater if edit_gate is None else None
        self.bge_toc = bge_toc
        self.device = device
        self.k_exact = int(cfg.k_exact if k_exact is None else k_exact)
        self.salience_span_len = int(cfg.salience_span_len if salience_span_len is None else salience_span_len)
        self.bank_write_threshold = float(bank_write_threshold)
        self.span_aligned_overwrite = bool(span_aligned_overwrite)
        self._z_zero_stub = (
            torch.zeros(
                (1, cfg.num_layers, cfg.latent_channels, cfg.grid_h, cfg.grid_w),
                device=device,
                dtype=torch.float32,
            )
            if self.delta_updater is not None
            else None
        )

    @classmethod
    def from_pathb(
        cls,
        pathb: PathBDiscreteMemory,
        device: torch.device,
        *,
        k_exact: int | None = None,
        salience_span_len: int | None = None,
    ) -> "StandaloneDiscreteSpanBank":
        return cls(
            cfg=pathb.cfg,
            salience_gate=pathb.salience_gate,
            hw_coord_embed=None,
            delta_updater=None,
            edit_gate=pathb.edit_gate,
            bge_toc=pathb.bge_toc,
            device=device,
            k_exact=pathb.k_exact if k_exact is None else k_exact,
            salience_span_len=pathb.salience_span_len if salience_span_len is None else salience_span_len,
            bank_write_threshold=pathb.bank_write_threshold,
            span_aligned_overwrite=pathb.span_aligned_overwrite,
        )

    def bge_model_param_count(self) -> int:
        if self.bge_toc is None or getattr(self.bge_toc, "model", None) is None:
            return 33_360_000
        return int(sum(p.numel() for p in self.bge_toc.model.parameters()))

    def trainable_param_breakdown(self, frozen_llm: nn.Module | None = None) -> dict[str, int]:
        sal_p = int(sum(p.numel() for p in self.salience_gate.parameters()))
        eg_p = int(sum(p.numel() for p in self.edit_gate.parameters())) if self.edit_gate is not None else 0
        du_p = int(sum(p.numel() for p in self.delta_updater.parameters())) if self.delta_updater is not None else 0
        llm_p = int(sum(p.numel() for p in frozen_llm.parameters())) if frozen_llm is not None else 0
        return {
            "salience_gate_params": sal_p,
            "edit_gate_params": eg_p,
            "legacy_delta_updater_params": du_p,
            "hw_coord_embed_params": 0,
            "layered_latent_encoder_params": 0,
            "spatial_projector_params": 0,
            "total_pathb_trainable_params": sal_p + eg_p + du_p,
            "bge_router_frozen_params": self.bge_model_param_count(),
            "frozen_llm_params": llm_p,
        }

    def storage_breakdown_bytes_per_page(self, num_spans: int = 4) -> dict[str, int]:
        bank_before = int(self.k_exact * 4)  # token_ids:int32 ONLY = 192 B/page at k=48
        base_extra = int(self.k_exact * 4)  # base_token_ids:int32 = +192 B/page once edited
        bge_dim = int(getattr(self.bge_toc, "dim", 384)) if self.bge_toc is not None else 384
        bge_bf16_bytes = int(num_spans * bge_dim * 2)  # 4 * 384 * 2 = 3,072 B/page (3 KiB)
        return {
            "exact_bank_bytes_per_page_before_edit": bank_before,
            "base_token_ids_extra_bytes_per_page_after_edit": base_extra,
            "exact_bank_bytes_per_page_after_edit": bank_before + base_extra,
            "bge_4span_bf16_bytes_per_page": bge_bf16_bytes,
            "stored_bytes_per_page_before_edit": bank_before + bge_bf16_bytes,
            "stored_bytes_per_page_after_edit": bank_before + base_extra + bge_bf16_bytes,
            "total_stored_bytes_per_page": bank_before + base_extra + bge_bf16_bytes,
        }

    @staticmethod
    def order_two_bank_pages(deep_pages: list[int], edited_pages: set[int] | None = None) -> tuple[list[int], list[int]]:
        return PathBDiscreteMemory.order_two_bank_pages(deep_pages, edited_pages)

    @torch.no_grad()
    def encode_book(
        self,
        page_tensor: torch.Tensor,
        embed_fn,
        tokenizer,
        vocab_size: int,
        *,
        amp_dtype: torch.dtype,
        autocast_enabled: bool,
        special_token_ids: tuple[int, ...] | None = None,
        num_spans: int = 4,
        chunk_size: int = 32,
    ) -> dict[str, Any]:
        """Independently build compact SparseExactBank (token_ids only) + BF16 BGE-4span keys WITHOUT LayeredLatentEncoder."""
        _sync_device(self.device)
        t0 = time.perf_counter()

        bsz, num_p, seq_len = page_tensor.shape
        tok_chunks: list[torch.Tensor] = []
        prev_k = self.salience_gate.cfg.k_exact
        self.salience_gate.cfg.k_exact = min(self.k_exact, seq_len)
        try:
            with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=autocast_enabled):
                for s in range(0, num_p, chunk_size):
                    e = min(num_p, s + chunk_size)
                    p_chunk = page_tensor[:, s:e]
                    p_emb = embed_fn(p_chunk)
                    bc = SparseExactBank.from_pages(
                        p_chunk,
                        self.salience_gate,
                        p_emb,
                        vocab_size=vocab_size,
                        special_token_ids=special_token_ids,
                        salience_span_len=self.salience_span_len,
                        compact_pathb=True,
                    )
                    tok_chunks.append(bc.token_ids)
        finally:
            self.salience_gate.cfg.k_exact = prev_k

        bank = SparseExactBank(
            token_ids=torch.cat(tok_chunks, dim=1),
            page_ids=None,
            h_coords=None,
            w_coords=None,
            edited_pages=set(),
        )
        bge_keys_bf16: torch.Tensor | None = None
        if self.bge_toc is not None:
            bge_raw = self.bge_toc.encode_book_pages(
                page_tensor, None, tokenizer, num_spans=num_spans, batch_size=128
            )
            bge_keys_bf16 = bge_raw.to(torch.bfloat16)

        _sync_device(self.device)
        encode_ms = (time.perf_counter() - t0) * 1000.0
        bge_actual_bytes = int(bge_keys_bf16.numel() * bge_keys_bf16.element_size()) if bge_keys_bf16 is not None else 0
        b_bd = bank.storage_bytes_breakdown()
        bge_bpp = round(bge_actual_bytes / max(1, num_p), 2)
        before_bpp = round(b_bd["bytes_per_page_before_edit"] + bge_bpp, 2)
        after_bpp = round(b_bd["bytes_per_page_after_edit_with_base"] + bge_bpp, 2)
        return {
            "bank": bank,
            "bge_keys_bf16": bge_keys_bf16,
            "encode_ms": encode_ms,
            "stored_bytes_total": bank.storage_bytes() + bge_actual_bytes,
            "stored_bytes_per_page_before_edit": before_bpp,
            "stored_bytes_per_page_after_edit": after_bpp,
            "stored_bytes_per_page": before_bpp,
            "exact_bank_bytes_per_page_before_edit": b_bd["bytes_per_page_before_edit"],
            "exact_bank_bytes_per_page_after_edit": b_bd["bytes_per_page_after_edit_with_base"],
            "exact_bank_bytes_per_page": b_bd["bytes_per_page_before_edit"],
            "bge_keys_bytes_per_page": bge_bpp,
        }

    @torch.no_grad()
    def reencode_single_page(
        self,
        bank: SparseExactBank,
        bge_keys_bf16: torch.Tensor | None,
        page_idx: int,
        new_page_token_ids: torch.Tensor,
        embed_fn,
        tokenizer,
        vocab_size: int,
        *,
        amp_dtype: torch.dtype,
        autocast_enabled: bool,
        special_token_ids: tuple[int, ...] | None = None,
    ) -> tuple[SparseExactBank, torch.Tensor | None]:
        p_3d = new_page_token_ids.unsqueeze(1) if new_page_token_ids.dim() == 2 else new_page_token_ids
        prev_k = self.salience_gate.cfg.k_exact
        self.salience_gate.cfg.k_exact = min(self.k_exact, int(p_3d.shape[-1]))
        try:
            with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=autocast_enabled):
                bc = SparseExactBank.from_pages(
                    p_3d,
                    self.salience_gate,
                    embed_fn(p_3d),
                    vocab_size=vocab_size,
                    special_token_ids=special_token_ids,
                    salience_span_len=self.salience_span_len,
                    compact_pathb=True,
                )
        finally:
            self.salience_gate.cfg.k_exact = prev_k
        bank.token_ids[:, page_idx : page_idx + 1] = bc.token_ids.to(bank.token_ids.dtype)
        if bank.base_token_ids is not None:
            bank.base_token_ids[:, page_idx : page_idx + 1] = bc.token_ids.to(bank.base_token_ids.dtype)
        bank.edited_pages.add(int(page_idx))
        if self.bge_toc is not None and bge_keys_bf16 is not None:
            bge_keys_bf16[:, page_idx : page_idx + 1] = self.bge_toc.encode_book_pages(
                p_3d, None, tokenizer, num_spans=4, batch_size=16
            ).to(torch.bfloat16)
        return bank, bge_keys_bf16

    @torch.no_grad()
    def score_edit_gate(
        self,
        edit_ids: torch.Tensor,
        llm: nn.Module,
        embed_fn,
        *,
        amp_dtype: torch.dtype,
        autocast_enabled: bool,
    ) -> tuple[bool, float, torch.Tensor | None, torch.Tensor]:
        """Evaluate accept/reject edit gate strictly WITHOUT Z_book / encoder (uses lean ContextualEditGateHead when set)."""
        layer_idx = (
            self.cfg.contextual_edit_layer
            if self.cfg.contextual_edit_layer is not None
            else max(1, self.cfg.num_llm_layers // 2)
        )
        with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=autocast_enabled):
            if self.edit_gate is not None:
                ctx_e = extract_contextual_edit_embeds(llm, edit_ids, layer_idx=layer_idx)
                logit, tok_w = self.edit_gate(ctx_e)
                prob = float(torch.sigmoid(logit).mean().item())
                gates = torch.full((1, self.cfg.grid_h, self.cfg.grid_w), prob, device=self.device, dtype=torch.float32)
                return (prob > self.bank_write_threshold), prob, tok_w, gates

            raw_edit_emb = embed_fn(edit_ids)
            if self.cfg.use_contextual_edit_encoding:
                delta_in = extract_contextual_edit_embeds(llm, edit_ids, layer_idx=layer_idx)
            else:
                delta_in = raw_edit_emb
            assert self.delta_updater is not None and self._z_zero_stub is not None
            _, gates = self.delta_updater(self._z_zero_stub, delta_in)
            edit_attn = getattr(self.delta_updater, "_last_edit_attn", None)
        strip_gate = float(gates[:, -1, :].float().mean().item())
        return (strip_gate > self.bank_write_threshold), strip_gate, edit_attn, gates

    @torch.no_grad()
    def apply_edit(
        self,
        bank: SparseExactBank,
        page_idx: int,
        edit_ids: torch.Tensor,
        llm: nn.Module,
        embed_fn,
        vocab_size: int,
        *,
        amp_dtype: torch.dtype,
        autocast_enabled: bool,
        in_place: bool = True,
        span_aligned_overwrite: bool | None = None,
        sep_token_id: int | None = None,
    ) -> tuple[SparseExactBank, bool, float, float]:
        """Run standalone accept/reject gate + span-aligned SparseExactBank overwrite (ZERO encoder/Z_book)."""
        _sync_device(self.device)
        t0 = time.perf_counter()
        accepted, strip_gate, edit_attn, gates = self.score_edit_gate(
            edit_ids, llm, embed_fn, amp_dtype=amp_dtype, autocast_enabled=autocast_enabled
        )
        use_sa = self.span_aligned_overwrite if span_aligned_overwrite is None else bool(span_aligned_overwrite)
        with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=autocast_enabled):
            raw_edit_emb = embed_fn(edit_ids)
            new_bank = bank.update_page_from_delta(
                page_indices=torch.tensor([int(page_idx)], dtype=torch.long, device=self.device),
                edit_token_ids=edit_ids,
                edit_embeds=raw_edit_emb,
                salience_gate=self.salience_gate,
                delta_gates=gates,
                grid_h=self.cfg.grid_h,
                grid_w=self.cfg.grid_w,
                vocab_size=vocab_size,
                in_place=in_place,
                bank_write_threshold=self.bank_write_threshold,
                edit_token_weights=None if (self.edit_gate is not None or use_sa) else edit_attn,
                use_tail_bonus=False if (self.edit_gate is not None or use_sa) else None,
                span_aligned_overwrite=use_sa,
                salience_span_len=self.salience_span_len,
                sep_token_id=sep_token_id,
                embed_tokens_fn=embed_fn,
            )
        _sync_device(self.device)
        edit_ms = (time.perf_counter() - t0) * 1000.0
        return new_bank, accepted, strip_gate, edit_ms

    @torch.no_grad()
    def compute_routing_logits(
        self,
        bank: SparseExactBank,
        bge_keys_bf16: torch.Tensor | None,
        rq_ids: torch.Tensor,
        q_txt: str,
        embed_fn,
        *,
        salience_route_weight: float = 1.5,
        excluded_indices: set[int] | None = None,
    ) -> torch.Tensor:
        bsz, num_pages, _ = bank.token_ids.shape
        device = bank.token_ids.device

        if self.bge_toc is not None and bge_keys_bf16 is not None:
            q_vec = self.bge_toc.encode_texts([q_txt], is_query=True)
            dense_logits = self.bge_toc.score_book(q_vec, bge_keys_bf16).to(device=device, dtype=torch.float32)
        else:
            dense_logits = torch.zeros((bsz, num_pages), device=device, dtype=torch.float32)

        logits = dense_logits
        if salience_route_weight > 0.0 and num_pages > 1:
            q_tok = embed_fn(rq_ids).float()
            q_norm = F.normalize(q_tok, dim=-1)
            b_emb = embed_fn(bank.routing_token_ids().clamp(min=0).to(device)).float()
            b_norm = F.normalize(b_emb, dim=-1)
            sim = torch.einsum("bmd,bpkd->bpmk", q_norm, b_norm)
            hit_b = torch.sigmoid((sim - 0.85) * 20.0)
            hit1 = hit_b.amax(dim=-1)
            bi = (
                (hit_b[:, :, :-1, :-1] * hit_b[:, :, 1:, 1:]).amax(dim=-1)
                if (q_tok.shape[1] >= 2 and b_emb.shape[2] >= 2)
                else None
            )
            tri = (
                (hit_b[:, :, :-2, :-2] * hit_b[:, :, 1:-1, 1:-1] * hit_b[:, :, 2:, 2:]).amax(dim=-1)
                if (q_tok.shape[1] >= 3 and b_emb.shape[2] >= 3)
                else None
            )
            if bank.base_token_ids is not None:
                d_emb = embed_fn(bank.token_ids.clamp(min=0).to(device)).float()
                d_norm = F.normalize(d_emb, dim=-1)
                sim_d = torch.einsum("bmd,bpkd->bpmk", q_norm, d_norm)
                hit_d = torch.sigmoid((sim_d - 0.85) * 20.0)
                hit1 = torch.maximum(hit1, hit_d.amax(dim=-1))
                if bi is not None and d_emb.shape[2] >= 2:
                    bi = torch.maximum(bi, (hit_d[:, :, :-1, :-1] * hit_d[:, :, 1:, 1:]).amax(dim=-1))
                if tri is not None and d_emb.shape[2] >= 3:
                    tri = torch.maximum(
                        tri, (hit_d[:, :, :-2, :-2] * hit_d[:, :, 1:-1, 1:-1] * hit_d[:, :, 2:, 2:]).amax(dim=-1)
                    )
            df1 = hit1.sum(dim=1, keepdim=True)
            w1 = (df1 > 0.3).to(hit1.dtype) / (df1 + 0.5).pow(2)
            sal_score = (hit1 * w1).sum(dim=-1) / (w1.sum(dim=-1) + 1e-6)
            if bi is not None:
                df2 = bi.sum(dim=1, keepdim=True)
                w2 = (df2 > 0.3).to(bi.dtype) / (df2 + 0.5).pow(2)
                sal_score = sal_score + 2.0 * ((bi * w2).sum(dim=-1) / (w2.sum(dim=-1) + 1e-6))
            if tri is not None:
                df3 = tri.sum(dim=1, keepdim=True)
                w3 = (df3 > 0.3).to(tri.dtype) / (df3 + 0.5).pow(2)
                sal_score = sal_score + 3.0 * ((tri * w3).sum(dim=-1) / (w3.sum(dim=-1) + 1e-6))

            sal_std = sal_score.std(dim=-1, keepdim=True)
            sal_z = (sal_score - sal_score.mean(dim=-1, keepdim=True)) / sal_std.clamp(min=0.35)
            sal_conf = (sal_score.amax(dim=-1, keepdim=True) / 1.5).clamp(0.0, 1.0)
            dense_std = dense_logits.std(dim=-1, keepdim=True)
            dense_z = torch.where(
                dense_std > 1e-6,
                (dense_logits - dense_logits.mean(dim=-1, keepdim=True)) / (dense_std + 1e-6),
                torch.zeros_like(dense_logits),
            )
            logits = dense_z + (salience_route_weight * sal_conf) * sal_z

        if excluded_indices:
            masked = logits.clone()
            for idx_ex in excluded_indices:
                if 0 <= idx_ex < num_pages:
                    masked[:, idx_ex] = -1e9
            logits = masked
        return logits

    @torch.no_grad()
    def route_edit_target_page(
        self,
        bank: SparseExactBank,
        bge_keys_bf16: torch.Tensor | None,
        edit_ids: torch.Tensor,
        edit_text: str,
        embed_fn,
        *,
        salience_route_weight: float = 1.5,
    ) -> int:
        logits = self.compute_routing_logits(
            bank, bge_keys_bf16, edit_ids, edit_text, embed_fn, salience_route_weight=salience_route_weight
        )
        return int(logits[0].argmax().item())

    @torch.no_grad()
    def assemble_two_bank_prefix(
        self,
        bank: SparseExactBank,
        deep_pages: list[int],
        embed_fn,
        *,
        edited_pages: set[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
        """True Two-Bank prefix assembly: unedited Bank-1 FIRST, edited Bank-2 LAST."""
        ed_set = set(bank.edited_pages) if edited_pages is None else set(edited_pages)
        bank1_pages, bank2_pages = self.order_two_bank_pages(deep_pages, ed_set)
        dev = bank.token_ids.device
        b1_idx = torch.tensor([bank1_pages], dtype=torch.long, device=dev) if bank1_pages else torch.empty((1, 0), dtype=torch.long, device=dev)
        b2_idx = torch.tensor([bank2_pages], dtype=torch.long, device=dev) if bank2_pages else torch.empty((1, 0), dtype=torch.long, device=dev)
        b1_emb = bank.gather_embeds(b1_idx, embed_tokens_fn=embed_fn, hw_coord_embed=None, training_ste=False)
        b2_emb = bank.gather_embeds(b2_idx, embed_tokens_fn=embed_fn, hw_coord_embed=None, training_ste=False)
        return b1_emb, b2_emb, bank1_pages, bank2_pages

    @torch.no_grad()
    def route_and_assemble(
        self,
        bank: SparseExactBank,
        bge_keys_bf16: torch.Tensor | None,
        rq_ids: torch.Tensor,
        q_txt: str,
        embed_fn,
        *,
        k_deep: int = 4,
        salience_route_weight: float = 1.5,
        excluded_indices: set[int] | None = None,
        use_two_bank_layout: bool = True,
    ) -> tuple[RouterOutput, torch.Tensor, float]:
        """Pure-GPU BGE-BF16 + SparseExactBank adaptive hybrid routing + True Two-Bank gather (ZERO Z_book/TOC)."""
        _sync_device(self.device)
        t0 = time.perf_counter()
        num_pages = int(bank.token_ids.shape[1])
        device = bank.token_ids.device

        logits = self.compute_routing_logits(
            bank, bge_keys_bf16, rq_ids, q_txt, embed_fn,
            salience_route_weight=salience_route_weight, excluded_indices=excluded_indices,
        )
        probs = F.softmax(logits, dim=-1)
        kd = min(num_pages - len(excluded_indices or ()), max(1, int(k_deep)))
        sorted_idx = torch.argsort(logits, dim=-1, descending=True)
        top_pages = [int(x) for x in sorted_idx[0, :kd].tolist()]

        if use_two_bank_layout:
            b1_emb, b2_emb, b1_pages, b2_pages = self.assemble_two_bank_prefix(bank, top_pages, embed_fn)
            ordered = b1_pages + b2_pages
            prefix = torch.cat((b1_emb, b2_emb), dim=1)
            deep_idx = torch.tensor([ordered], dtype=torch.long, device=device)
        else:
            deep_idx = sorted_idx[:, :kd].flip(dims=[-1])
            prefix = bank.gather_embeds(deep_idx, embed_tokens_fn=embed_fn, hw_coord_embed=None, training_ste=False)

        r_out = RouterOutput(
            logits=logits,
            probs=probs,
            deep_idx=deep_idx,
            skim_idx=sorted_idx[:, :0],
            toc_idx=torch.sort(sorted_idx[:, kd:], dim=-1).values,
        )
        _sync_device(self.device)
        return r_out, prefix, (time.perf_counter() - t0) * 1000.0


def train_lean_pathb_qwen(
    llm: nn.Module,
    pathb: PathBDiscreteMemory,
    tokenizer,
    train_pool: list[int],
    device: torch.device,
    *,
    copy_salience_from: nn.Module | None = None,
    edit_gate_steps: int = 200,
    edit_gate_lr: float = 5e-4,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Train lean Layer-14 ContextualEditGateHead(hidden_dim=64) on 100% eval-disjoint data (ZERO 7.24M DeltaUpdater!)."""
    t0 = time.perf_counter()
    embed_fn = llm.get_input_embeddings()
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32

    if copy_salience_from is not None:
        pathb.salience_gate.load_state_dict(copy_salience_from.state_dict(), strict=True)
    pathb.salience_gate.eval()

    # Exact O(1)-RAM checksum before training to prove 0.0 frozen LLM mutation
    llm_before: dict[str, tuple[float, float]] = {}
    with torch.no_grad():
        for name, p in llm.named_parameters():
            pf = p.detach().float()
            llm_before[name] = (float(pf.sum().item()), float(pf.abs().max().item()))

    n_g2 = 10 if is_mock else 48
    edit_pairs: list[tuple[torch.Tensor, float]] = []
    for mode_m, sd_off in (("budget", 1101), ("symbolic", 1203)):
        ds_m = AdversarialLivingBookDataset(
            num_episodes=n_g2, num_pages=6, page_size=pathb.cfg.page_size, num_edits=4,
            mode=mode_m, split="train", tokenizer=tokenizer, vocab_size=vocab_size,
            seed=seed + sd_off, background_token_pool=train_pool, style="mixed_natural",
        )
        for ep in ds_m:
            for (_, e_ids_cpu), is_acc in zip(ep.edit_pages, ep.metadata["edit_accepted_flags"]):
                edit_pairs.append((e_ids_cpu, 1.0 if is_acc else 0.0))

    layer_idx = (
        pathb.cfg.contextual_edit_layer
        if pathb.cfg.contextual_edit_layer is not None
        else max(1, pathb.cfg.num_llm_layers // 2)
    )
    cached: list[tuple[torch.Tensor, float]] = []
    with torch.no_grad():
        for e_ids_cpu, y_val in edit_pairs:
            e_t = e_ids_cpu.unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                ctx_e = extract_contextual_edit_embeds(llm, e_t, layer_idx=layer_idx)
            cached.append((ctx_e.detach().float(), y_val))

    rng = random.Random(seed + 91)
    eff_steps = min(edit_gate_steps, 20) if is_mock else edit_gate_steps
    bs = 4 if is_mock else 16
    opt = torch.optim.AdamW(pathb.edit_gate.parameters(), lr=edit_gate_lr, weight_decay=0.01)
    pathb.edit_gate.train()
    losses: list[float] = []
    for _ in range(eff_steps):
        opt.zero_grad(set_to_none=True)
        step_l = []
        for _j in range(bs):
            ctx_e, y_val = cached[rng.randrange(len(cached))]
            logit, _ = pathb.edit_gate(ctx_e)
            step_l.append(F.binary_cross_entropy_with_logits(logit, torch.tensor([y_val], device=device)))
        loss = torch.stack(step_l).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pathb.edit_gate.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().item()))
    pathb.edit_gate.eval()

    correct = 0
    with torch.no_grad():
        for ctx_e, y_val in cached:
            logit, _ = pathb.edit_gate(ctx_e)
            pred = 1.0 if float(torch.sigmoid(logit).item()) > pathb.bank_write_threshold else 0.0
            correct += int(pred == y_val)

    max_diff = 0.0
    with torch.no_grad():
        for name, p in llm.named_parameters():
            pf = p.detach().float()
            s0, m0 = llm_before[name]
            max_diff = max(max_diff, abs(float(pf.sum().item()) - s0), abs(float(pf.abs().max().item()) - m0))
    assert max_diff == 0.0, f"Frozen LLM mutated during lean Path B edit-gate training: {max_diff}"

    return {
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "edit_gate_layer": layer_idx,
        "edit_gate_steps": eff_steps,
        "num_edit_turns": len(cached),
        "initial_loss": round(losses[0], 4),
        "final_loss": round(losses[-1], 4),
        "train_acc": round(correct / max(1, len(cached)), 4),
        "max_frozen_param_diff": max_diff,
        "trainable_param_breakdown": pathb.trainable_param_breakdown(llm),
    }


@torch.no_grad()
def measure_post_rerouting_cache_hit_telemetry(
    pathb_mem: StandaloneDiscreteSpanBank | PathBDiscreteMemory,
    llm: nn.Module,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    num_gate2_eps: int = 24,
    num_coding_tasks: int = 8,
    salience_route_weight: float = 1.5,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Measure realistic post-rerouting Bank-1 KV cache-hit rate across multi-turn Gate-2 & Coding episodes.

    Strictly separated from fixed working-set KV reuse: tracks how often consecutive turns keep an identical
    Bank-1 page sequence (`exact_bank1_hit_rate`) and the RadixAttention-style longest-common-prefix fraction
    (`prefix_lcs_reuse_fraction`) after real BGE+Salience rerouting.
    """
    embed_fn = llm.get_input_embeddings()
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    sep_id = get_newline_sep_token_id(tokenizer)

    def _call_encode(p_t: torch.Tensor):
        if isinstance(pathb_mem, PathBDiscreteMemory):
            return pathb_mem.encode_book(
                p_t, embed_fn, tokenizer, vocab_size, device=device,
                amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids,
            )
        return pathb_mem.encode_book(
            p_t, embed_fn, tokenizer, vocab_size,
            amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids,
        )

    def _call_edit(bk: SparseExactBank, p_i: int, e_t: torch.Tensor):
        if isinstance(pathb_mem, PathBDiscreteMemory):
            return pathb_mem.apply_edit(
                bk, p_i, e_t, llm, embed_fn, vocab_size, device=device,
                amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, in_place=True, sep_token_id=sep_id,
            )
        return pathb_mem.apply_edit(
            bk, p_i, e_t, llm, embed_fn, vocab_size,
            amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, in_place=True, sep_token_id=sep_id,
        )

    def _call_route(bk: SparseExactBank, bge_k, rq_t: torch.Tensor, q_s: str, excluded=None):
        if isinstance(pathb_mem, PathBDiscreteMemory):
            r_out, _, _ = pathb_mem.route_and_assemble(
                bk, bge_k, rq_t, q_s, embed_fn, device=device, k_deep=4,
                salience_route_weight=salience_route_weight, excluded_indices=excluded, use_two_bank_layout=True,
            )
        else:
            r_out, _, _ = pathb_mem.route_and_assemble(
                bk, bge_k, rq_t, q_s, embed_fn, k_deep=4,
                salience_route_weight=salience_route_weight, excluded_indices=excluded, use_two_bank_layout=True,
            )
        return [int(x) for x in r_out.deep_idx[0].tolist()]

    # 1. Multi-turn Gate-2 rerouting telemetry (query after EVERY edit turn 1..num_edits)
    g2_n = 4 if is_mock else num_gate2_eps
    g2_eps = list(
        AdversarialLivingBookDataset(
            num_episodes=g2_n, num_pages=(8 if is_mock else 16), page_size=pathb_mem.cfg.page_size, num_edits=4,
            mode="budget", split="eval", tokenizer=tokenizer, vocab_size=vocab_size,
            seed=2099, background_token_pool=val_pool,
        )
    )
    g2_exact_hits: list[float] = []
    g2_lcs_fracs: list[float] = []
    g2_imm_fracs: list[float] = []
    for ep in g2_eps:
        p_t = ep.context_pages.unsqueeze(0).to(device)
        enc = _call_encode(p_t)
        bk, bge_k = enc["bank"], enc["bge_keys_bf16"]
        clean_q = tokenizer.decode(ep.metadata["router_query_ids"].tolist(), skip_special_tokens=True).strip()
        rq_t = ep.metadata["router_query_ids"].unsqueeze(0).to(device)
        prev_b1: list[int] | None = None
        for p_i, e_ids_cpu in ep.edit_pages:
            bk, _, _, _ = _call_edit(bk, int(p_i), e_ids_cpu.unsqueeze(0).to(device))
            deep_p = _call_route(bk, bge_k, rq_t, clean_q)
            b1_p, _ = PathBDiscreteMemory.order_two_bank_pages(deep_p, bk.edited_pages)
            g2_imm_fracs.append(len(b1_p) / max(1, len(deep_p)))
            if prev_b1 is not None:
                g2_exact_hits.append(1.0 if (b1_p == prev_b1 and len(b1_p) > 0) else 0.0)
                lcs = compute_prefix_lcs_len(prev_b1, b1_p)
                g2_lcs_fracs.append(lcs / max(1, len(deep_p)))
            prev_b1 = b1_p

    # 2. Multi-turn Coding caller-update rerouting telemetry (consecutive caller turns within each repo task)
    code_n = 2 if is_mock else num_coding_tasks
    code_tasks = build_coding_n24_tasks(num_distractor_files=(8 if is_mock else 58))[:code_n]
    code_exact_hits: list[float] = []
    code_lcs_fracs: list[float] = []
    code_imm_fracs: list[float] = []
    for task in code_tasks:
        pack = _build_repo_pages(task, tokenizer, pathb_mem.cfg.page_size)
        page_paths = list(pack["page_paths"])
        core_idx = int(pack["core_page_indices"][0])
        sess_idx = int(pack["session_page_idx"])
        enc = _call_encode(pack["page_tensor"].to(device))
        bk, bge_k = enc["bank"], enc["bge_keys_bf16"]
        bk.edited_pages.add(core_idx)
        for turn in task.session_turns:
            e_t = torch.tensor(tokenizer.encode(str(turn["text"]), add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
            bk, _, _, _ = _call_edit(bk, sess_idx, e_t)
        q_txt = build_pathless_coding_query(task)
        rq_t = torch.tensor(tokenizer.encode(q_txt, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
        prev_b1 = None
        for caller_p in list(task.caller_files) + [task.homonym_file]:
            excl = {i for i, p in enumerate(page_paths) if p == caller_p}
            deep_p = _call_route(bk, bge_k, rq_t, q_txt, excluded=excl)
            b1_p, _ = PathBDiscreteMemory.order_two_bank_pages(deep_p, bk.edited_pages)
            code_imm_fracs.append(len(b1_p) / max(1, len(deep_p)))
            if prev_b1 is not None:
                code_exact_hits.append(1.0 if (b1_p == prev_b1 and len(b1_p) > 0) else 0.0)
                lcs = compute_prefix_lcs_len(prev_b1, b1_p)
                code_lcs_fracs.append(lcs / max(1, len(deep_p)))
            prev_b1 = b1_p

    return {
        "gate2_multiturn_rerouting": {
            "num_transitions": len(g2_exact_hits),
            "exact_bank1_hit_rate": round(sum(g2_exact_hits) / max(1, len(g2_exact_hits)), 4),
            "prefix_lcs_reuse_fraction": round(sum(g2_lcs_fracs) / max(1, len(g2_lcs_fracs)), 4),
            "mean_bank1_page_fraction": round(sum(g2_imm_fracs) / max(1, len(g2_imm_fracs)), 4),
        },
        "coding_multiturn_rerouting": {
            "num_transitions": len(code_exact_hits),
            "exact_bank1_hit_rate": round(sum(code_exact_hits) / max(1, len(code_exact_hits)), 4),
            "prefix_lcs_reuse_fraction": round(sum(code_lcs_fracs) / max(1, len(code_lcs_fracs)), 4),
            "mean_bank1_page_fraction": round(sum(code_imm_fracs) / max(1, len(code_imm_fracs)), 4),
        },
    }


# ---------------------------------------------------------------------------
# Pillar 1: Gate-2 Tie-Breaker N=200 + 3 Seeds + Exact McNemar & Gate FAR/FRR
# ---------------------------------------------------------------------------


def _build_append_192tok_bank(
    base_bank: SparseExactBank,
    appended_by_page: dict[int, list[int]],
    *,
    k_exact: int,
    span_len: int,
    sep_token_id: int | None,
) -> SparseExactBank:
    """Causal control: preserve initial span-16 fact block (NO selective overwrite) and APPEND accepted edit tokens,
    capping strictly at `k_exact` tokens/page so active prefix stays bit-identical (192 tok) to Path B overwrite."""
    new_tok = base_bank.token_ids.clone()
    edited_set = set(base_bank.edited_pages)
    keep_head = min(int(span_len), k_exact // 2)
    for p_i, app_toks in appended_by_page.items():
        if not app_toks:
            continue
        edited_set.add(int(p_i))
        orig_row = [int(t) for t in base_bank.token_ids[0, p_i].tolist()]
        sep_list = [int(sep_token_id)] if (sep_token_id is not None and int(sep_token_id) >= 0) else []
        tail_cap = max(0, k_exact - keep_head - len(sep_list))
        # Append chronologically (never overwriting orig_row[:keep_head]!) and retain latest rolling tail within k_exact
        rolling_tail = app_toks[-tail_cap:] if tail_cap > 0 else []
        merged = orig_row[:keep_head] + sep_list + rolling_tail
        if len(merged) < k_exact:
            merged = merged + orig_row[keep_head : keep_head + (k_exact - len(merged))]
        new_tok[0, p_i] = torch.tensor(merged[:k_exact], dtype=torch.int32, device=new_tok.device)
    return SparseExactBank(
        token_ids=new_tok,
        page_ids=None,
        h_coords=None,
        w_coords=None,
        base_token_ids=base_bank.token_ids.clone() if edited_set else None,
        edited_pages=edited_set,
    )


@torch.no_grad()
def run_gate2_tiebreaker_n200(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    bge_toc: _PlugInBiEncoderTOC | None = None,
    pathb: PathBDiscreteMemory | None = None,
    eval_n: int = 200,
    seeds: tuple[int, ...] = (2026, 2027, 2028),
    salience_route_weight: float = 1.5,
    blind_edit_slice_n: int = 60,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Gate-2 N=200 across seeds comparing ungated/gated BM25, raw_text_same_pages_k4, standalone_append_192tok,
    standalone_discrete_span_bank (span-aligned overwrite), blind_edit_routing slice, exact_only, and lslc_full."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    sep_id = get_newline_sep_token_id(tokenizer)
    eff_span_len = int(getattr(wrapper.cfg, "salience_span_len", 16))
    wrapper.eval()

    if pathb is not None:
        standalone = StandaloneDiscreteSpanBank.from_pathb(pathb, device=device)
    else:
        standalone = StandaloneDiscreteSpanBank(
            cfg=wrapper.cfg,
            salience_gate=wrapper.lslc.salience_gate,
            hw_coord_embed=None,
            delta_updater=wrapper.lslc.delta_updater,
            bge_toc=bge_toc,
            device=device,
            k_exact=wrapper.cfg.k_exact,
            salience_span_len=eff_span_len,
            bank_write_threshold=0.45,
            span_aligned_overwrite=True,
        )

    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR + _FAIR_G2_FEWSHOT_PREFIX, add_special_tokens=False),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    n_each = 4 if is_mock else max(1, eval_n // 2)
    num_p = 8 if is_mock else 16
    all_systems = (
        "bm25_rag_append_errata",
        "gated_bm25_rag",
        "raw_text_same_pages_k4",
        "standalone_append_192tok",
        "standalone_discrete_span_bank",
        "lslc_exact_only",
        "lslc_full",
    )
    eff_seeds = seeds[:2] if is_mock else seeds
    q_types_list = ("accepted_override", "rejected_proposal", "neighbor_preservation", "randomized_static_retention")

    by_seed: dict[str, Any] = {}
    pooled_scores: dict[str, list[float]] = {s: [] for s in all_systems}
    pooled_strict: dict[str, list[float]] = {s: [] for s in all_systems}
    pooled_w2n: dict[str, list[float]] = {s: [] for s in all_systems}

    for seed_val in eff_seeds:
        eps_budget = list(
            AdversarialLivingBookDataset(
                num_episodes=n_each, num_pages=num_p, page_size=wrapper.cfg.page_size, num_edits=3,
                mode="budget", split="eval", tokenizer=tokenizer, vocab_size=vocab_size,
                seed=seed_val + 40, background_token_pool=val_pool,
            )
        )
        eps_sym = list(
            AdversarialLivingBookDataset(
                num_episodes=n_each, num_pages=num_p, page_size=wrapper.cfg.page_size, num_edits=5,
                mode="symbolic", split="eval", tokenizer=tokenizer, vocab_size=vocab_size,
                seed=seed_val + 41, background_token_pool=val_pool,
            )
        )
        all_eps = eps_budget + eps_sym

        seed_sys_out: dict[str, Any] = {}
        seed_norm_scores: dict[str, list[float]] = {}
        seed_w2n_scores: dict[str, list[float]] = {}
        # Record chosen pages from standalone_discrete_span_bank so raw_text_same_pages_k4 uses 100% identical pages
        standalone_chosen_pages: dict[int, list[int]] = {}

        for sys_name in all_systems:
            rows: list[dict[str, Any]] = []
            rows_by_mode: dict[str, list[dict[str, Any]]] = {"budget": [], "symbolic": []}
            rows_by_qtype: dict[str, list[dict[str, Any]]] = {qt: [] for qt in q_types_list}
            gate_tp = gate_tn = gate_fp = gate_fn = 0
            acc_gates: list[float] = []
            rej_gates: list[float] = []

            for ep_idx, ep in enumerate(all_eps):
                clean_q = tokenizer.decode(ep.metadata["router_query_ids"].tolist(), skip_special_tokens=True).strip()
                gold = str(ep.metadata["answer_str"]).strip()
                ep_mode = str(ep.metadata.get("mode", "budget"))
                q_type = str(ep.metadata.get("query_type", ""))
                gold_flags = list(ep.metadata.get("edit_accepted_flags", []))
                p_prompt = torch.tensor(
                    tokenizer.encode(_format_qa_prompt(clean_q), add_special_tokens=False),
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)

                encode_ms = 0.0
                edit_ms = 0.0
                first_query_ms = 0.0
                stored_bpp_before = 0.0
                stored_bpp_after = 0.0

                if sys_name in ("bm25_rag_append_errata", "gated_bm25_rag"):
                    t_enc0 = time.perf_counter()
                    docs = [tokenizer.decode(ep.context_pages[i], skip_special_tokens=True) for i in range(ep.context_pages.shape[0])]
                    stored_bpp_before = round(sum(len(d.encode("utf-8")) for d in docs) / max(1, len(docs)), 2)
                    encode_ms = (time.perf_counter() - t_enc0) * 1000.0

                    _sync_device(device)
                    t_ed0 = time.perf_counter()
                    for (p_i, e_ids_cpu), true_acc in zip(ep.edit_pages, gold_flags):
                        if sys_name == "gated_bm25_rag":
                            e_ids = e_ids_cpu.unsqueeze(0).to(device)
                            pred_acc, g_val, _, _ = standalone.score_edit_gate(
                                e_ids, wrapper.llm, embed_fn, amp_dtype=amp_dtype, autocast_enabled=autocast_enabled
                            )
                            if true_acc:
                                acc_gates.append(g_val)
                                gate_tp += int(pred_acc)
                                gate_fn += int(not pred_acc)
                            else:
                                rej_gates.append(g_val)
                                gate_fp += int(pred_acc)
                                gate_tn += int(not pred_acc)
                            if pred_acc:
                                docs[int(p_i)] = docs[int(p_i)] + "\n" + tokenizer.decode(e_ids_cpu, skip_special_tokens=True)
                        else:
                            docs[int(p_i)] = docs[int(p_i)] + "\n" + tokenizer.decode(e_ids_cpu, skip_special_tokens=True)
                    _sync_device(device)
                    edit_ms = (time.perf_counter() - t_ed0) * 1000.0
                    stored_bpp_after = round(sum(len(d.encode("utf-8")) for d in docs) / max(1, len(docs)), 2)

                    _sync_device(device)
                    t_q0 = time.perf_counter()
                    bm25 = BM25PageRetriever(docs)
                    rr = bm25.route(clean_q, k_deep=min(4, len(docs)), k_skim=0, device=device)
                    chosen = rr.deep_idx[0].tolist()
                    rag_txt = "\n\n".join(docs[i].strip() for i in chosen) + "\n"
                    r_ids = _encode_unpadded_rag_ids(rag_txt, tokenizer, vocab_size, device)
                    pref = embed_fn(r_ids).to(llm_dtype)
                    gen_txt = _greedy_generate_qa(
                        wrapper.llm, pref, p_prompt, tokenizer, max_new_tokens=20, user_open_ids=user_open_ids
                    )
                    _sync_device(device)
                    first_query_ms = (time.perf_counter() - t_q0) * 1000.0

                elif sys_name == "raw_text_same_pages_k4":
                    # Causal control: identical BGE+Salience hybrid router + identical edit gate, FULL raw text!
                    p_t = ep.context_pages.unsqueeze(0).to(device)
                    enc_pack = standalone.encode_book(
                        p_t, embed_fn, tokenizer, vocab_size, amp_dtype=amp_dtype,
                        autocast_enabled=autocast_enabled, special_token_ids=special_ids, num_spans=4,
                    )
                    bk, bge_bf16 = enc_pack["bank"], enc_pack["bge_keys_bf16"]
                    docs = [tokenizer.decode(ep.context_pages[i], skip_special_tokens=True) for i in range(ep.context_pages.shape[0])]
                    encode_ms = float(enc_pack["encode_ms"])
                    raw_bpp_b = sum(len(d.encode("utf-8")) for d in docs) / max(1, len(docs))
                    stored_bpp_before = round(raw_bpp_b + float(enc_pack["bge_keys_bytes_per_page"]), 2)

                    _sync_device(device)
                    t_ed0 = time.perf_counter()
                    for (p_i, e_ids_cpu), true_acc in zip(ep.edit_pages, gold_flags):
                        e_ids = e_ids_cpu.unsqueeze(0).to(device)
                        pred_acc, g_val, _, _ = standalone.score_edit_gate(
                            e_ids, wrapper.llm, embed_fn, amp_dtype=amp_dtype, autocast_enabled=autocast_enabled
                        )
                        if true_acc:
                            acc_gates.append(g_val)
                            gate_tp += int(pred_acc)
                            gate_fn += int(not pred_acc)
                        else:
                            rej_gates.append(g_val)
                            gate_fp += int(pred_acc)
                            gate_tn += int(not pred_acc)
                        if pred_acc:
                            docs[int(p_i)] = docs[int(p_i)] + "\n" + tokenizer.decode(e_ids_cpu, skip_special_tokens=True)
                    _sync_device(device)
                    edit_ms = (time.perf_counter() - t_ed0) * 1000.0
                    raw_bpp_a = sum(len(d.encode("utf-8")) for d in docs) / max(1, len(docs))
                    stored_bpp_after = round(raw_bpp_a + float(enc_pack["bge_keys_bytes_per_page"]), 2)

                    rq_ids = ep.metadata["router_query_ids"].unsqueeze(0).to(device)
                    _sync_device(device)
                    t_q0 = time.perf_counter()
                    r_out, _, _ = standalone.route_and_assemble(
                        bk, bge_bf16, rq_ids, clean_q, embed_fn, k_deep=4, salience_route_weight=salience_route_weight
                    )
                    chosen = [int(x) for x in r_out.deep_idx[0].tolist()]
                    rag_txt = "\n\n".join(docs[i].strip() for i in chosen) + "\n"
                    pref = embed_fn(_encode_unpadded_rag_ids(rag_txt, tokenizer, vocab_size, device)).to(llm_dtype)
                    gen_txt = _greedy_generate_qa(
                        wrapper.llm, pref, p_prompt, tokenizer, max_new_tokens=20, user_open_ids=user_open_ids
                    )
                    _sync_device(device)
                    first_query_ms = (time.perf_counter() - t_q0) * 1000.0

                elif sys_name == "standalone_append_192tok":
                    # Causal control: identical pages, identical router, identical gate, identical 192 tok, APPEND (no overwrite!)
                    p_t = ep.context_pages.unsqueeze(0).to(device)
                    enc_pack = standalone.encode_book(
                        p_t, embed_fn, tokenizer, vocab_size, amp_dtype=amp_dtype,
                        autocast_enabled=autocast_enabled, special_token_ids=special_ids, num_spans=4,
                    )
                    base_bk, bge_bf16 = enc_pack["bank"], enc_pack["bge_keys_bf16"]
                    encode_ms = float(enc_pack["encode_ms"])
                    stored_bpp_before = float(enc_pack["stored_bytes_per_page_before_edit"])

                    appended_by_page: dict[int, list[int]] = {}
                    _sync_device(device)
                    t_ed0 = time.perf_counter()
                    for (p_i, e_ids_cpu), true_acc in zip(ep.edit_pages, gold_flags):
                        e_ids = e_ids_cpu.unsqueeze(0).to(device)
                        pred_acc, g_val, _, _ = standalone.score_edit_gate(
                            e_ids, wrapper.llm, embed_fn, amp_dtype=amp_dtype, autocast_enabled=autocast_enabled
                        )
                        if true_acc:
                            acc_gates.append(g_val)
                            gate_tp += int(pred_acc)
                            gate_fn += int(not pred_acc)
                        else:
                            rej_gates.append(g_val)
                            gate_fp += int(pred_acc)
                            gate_tn += int(not pred_acc)
                        if pred_acc:
                            buf = appended_by_page.setdefault(int(p_i), [])
                            if buf and sep_id is not None:
                                buf.append(int(sep_id))
                            buf.extend(int(t) for t in e_ids_cpu.tolist())
                    bk = _build_append_192tok_bank(
                        base_bk, appended_by_page, k_exact=standalone.k_exact, span_len=eff_span_len, sep_token_id=sep_id
                    )
                    _sync_device(device)
                    edit_ms = (time.perf_counter() - t_ed0) * 1000.0
                    stored_bpp_after = round(
                        (bk.storage_bytes() / max(1, p_t.shape[1])) + float(enc_pack["bge_keys_bytes_per_page"]), 2
                    )

                    rq_ids = ep.metadata["router_query_ids"].unsqueeze(0).to(device)
                    _sync_device(device)
                    t_q0 = time.perf_counter()
                    _, pref, _ = standalone.route_and_assemble(
                        bk, bge_bf16, rq_ids, clean_q, embed_fn, k_deep=4, salience_route_weight=salience_route_weight
                    )
                    pref = pref.to(llm_dtype)
                    gen_txt = _greedy_generate_qa(
                        wrapper.llm, pref, p_prompt, tokenizer, max_new_tokens=20, user_open_ids=user_open_ids
                    )
                    _sync_device(device)
                    first_query_ms = (time.perf_counter() - t_q0) * 1000.0

                elif sys_name == "standalone_discrete_span_bank":
                    p_t = ep.context_pages.unsqueeze(0).to(device)
                    enc_pack = standalone.encode_book(
                        p_t, embed_fn, tokenizer, vocab_size, amp_dtype=amp_dtype,
                        autocast_enabled=autocast_enabled, special_token_ids=special_ids, num_spans=4,
                    )
                    bk, bge_bf16 = enc_pack["bank"], enc_pack["bge_keys_bf16"]
                    encode_ms = float(enc_pack["encode_ms"])
                    stored_bpp_before = float(enc_pack["stored_bytes_per_page_before_edit"])

                    total_ed_ms = 0.0
                    for (p_i, e_ids_cpu), true_acc in zip(ep.edit_pages, gold_flags):
                        e_ids = e_ids_cpu.unsqueeze(0).to(device)
                        bk, pred_acc, g_val, dt_ed = standalone.apply_edit(
                            bk, int(p_i), e_ids, wrapper.llm, embed_fn, vocab_size,
                            amp_dtype=amp_dtype, autocast_enabled=autocast_enabled,
                            in_place=True, span_aligned_overwrite=True, sep_token_id=sep_id,
                        )
                        total_ed_ms += dt_ed
                        if true_acc:
                            acc_gates.append(g_val)
                            gate_tp += int(pred_acc)
                            gate_fn += int(not pred_acc)
                        else:
                            rej_gates.append(g_val)
                            gate_fp += int(pred_acc)
                            gate_tn += int(not pred_acc)
                    edit_ms = total_ed_ms
                    stored_bpp_after = round(
                        (bk.storage_bytes() / max(1, p_t.shape[1])) + float(enc_pack["bge_keys_bytes_per_page"]), 2
                    )

                    rq_ids = ep.metadata["router_query_ids"].unsqueeze(0).to(device)
                    _sync_device(device)
                    t_q0 = time.perf_counter()
                    r_out, pref, _ = standalone.route_and_assemble(
                        bk, bge_bf16, rq_ids, clean_q, embed_fn, k_deep=4, salience_route_weight=salience_route_weight
                    )
                    standalone_chosen_pages[ep_idx] = [int(x) for x in r_out.deep_idx[0].tolist()]
                    pref = pref.to(llm_dtype)
                    gen_txt = _greedy_generate_qa(
                        wrapper.llm, pref, p_prompt, tokenizer, max_new_tokens=20, user_open_ids=user_open_ids
                    )
                    _sync_device(device)
                    first_query_ms = (time.perf_counter() - t_q0) * 1000.0

                else:  # lslc_exact_only or lslc_full
                    is_ex = sys_name == "lslc_exact_only"
                    p_t = ep.context_pages.unsqueeze(0).to(device)
                    _sync_device(device)
                    t_enc0 = time.perf_counter()
                    z_b, bk, tc = _encode_book_chunked(
                        wrapper.lslc, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled,
                        chunk_size=16, special_token_ids=special_ids, salience_span_len=eff_span_len,
                    )
                    bge_p4 = (
                        bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=64).to(torch.bfloat16)
                        if bge_toc is not None
                        else None
                    )
                    _sync_device(device)
                    encode_ms = (time.perf_counter() - t_enc0) * 1000.0
                    z_bpp = (z_b.numel() * 2) / max(1, p_t.shape[1])
                    bge_bpp = (bge_p4.numel() * bge_p4.element_size()) / max(1, p_t.shape[1]) if bge_p4 is not None else 0.0
                    stored_bpp_before = round(z_bpp + (bk.storage_bytes() / max(1, p_t.shape[1])) + bge_bpp, 2)

                    _sync_device(device)
                    t_ed0 = time.perf_counter()
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        for (p_i, e_ids_cpu), true_acc in zip(ep.edit_pages, gold_flags):
                            e_ids = e_ids_cpu.unsqueeze(0).to(device)
                            ov = (
                                extract_contextual_edit_embeds(
                                    wrapper.llm, e_ids, layer_idx=wrapper.cfg.contextual_edit_layer
                                )
                                if wrapper.cfg.use_contextual_edit_encoding
                                else None
                            )
                            z_b, g_p, bk = wrapper.lslc.apply_page_delta(
                                z_b, torch.tensor([int(p_i)], device=device), e_ids,
                                embed_tokens_fn=embed_fn, exact_bank=bk, vocab_size=vocab_size,
                                edit_embeds_override=ov, in_place=True, bank_write_threshold=0.45,
                            )
                            g_val = float(g_p[:, -1, :].float().mean().item())
                            pred_acc = g_val > 0.45
                            if true_acc:
                                acc_gates.append(g_val)
                                gate_tp += int(pred_acc)
                                gate_fn += int(not pred_acc)
                            else:
                                rej_gates.append(g_val)
                                gate_fp += int(pred_acc)
                                gate_tn += int(not pred_acc)
                    _sync_device(device)
                    edit_ms = (time.perf_counter() - t_ed0) * 1000.0
                    stored_bpp_after = round(z_bpp + (bk.storage_bytes() / max(1, p_t.shape[1])) + bge_bpp, 2)

                    rq_ids = ep.metadata["router_query_ids"].unsqueeze(0).to(device)
                    _sync_device(device)
                    t_q0 = time.perf_counter()
                    ext_s = (
                        bge_toc.score_book(bge_toc.encode_texts([clean_q], is_query=True), bge_p4)
                        if (bge_toc is not None and bge_p4 is not None)
                        else None
                    )
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        _, asm = wrapper.lslc.route_and_assemble(
                            z_b, tc, embed_fn(rq_ids).mean(dim=1), bk, embed_fn,
                            k_deep=4, k_skim=2, mode="exact_only" if is_ex else "full",
                            query_input_ids=rq_ids, salience_route_weight=salience_route_weight,
                            exact_bank_attenuation="none", external_dense_logits=ext_s,
                        )
                    pref = asm.prefix_embeds.to(llm_dtype)
                    gen_txt = _greedy_generate_qa(
                        wrapper.llm, pref, p_prompt, tokenizer, max_new_tokens=20, user_open_ids=user_open_ids
                    )
                    _sync_device(device)
                    first_query_ms = (time.perf_counter() - t_q0) * 1000.0

                qm = compute_community_qa_metrics(gen_txt, gold)
                first_line = gen_txt.splitlines()[0] if gen_txt.strip() else ""
                canon_pred = canonicalize_numeric_answer(first_line)
                canon_gold = canonicalize_numeric_answer(gold)
                norm_num_em = 1.0 if (canon_gold and canon_pred == canon_gold) else float(qm["extracted_em"])
                w2n_pred = canonicalize_w2n_answer(first_line)
                w2n_gold = canonicalize_w2n_answer(gold)
                norm_w2n_em = 1.0 if (w2n_gold and w2n_pred == w2n_gold) else norm_num_em

                rec = {
                    "episode_idx": ep_idx,
                    "mode": ep_mode,
                    "query_type": q_type,
                    "gold": gold,
                    "pred": gen_txt,
                    "strict_em": float(qm["extracted_em"]),
                    "normalized_numeric_em": float(norm_num_em),
                    "normalized_w2n_em": float(norm_w2n_em),
                    "f1": float(qm["token_f1"]),
                    "active_tokens": float(pref.shape[1]),
                    "encode_ms": round(encode_ms, 3),
                    "edit_ms": round(edit_ms, 3),
                    "first_query_ms": round(first_query_ms, 3),
                    "stored_bytes_per_page_before_edit": stored_bpp_before,
                    "stored_bytes_per_page_after_edit": stored_bpp_after,
                    "stored_bytes_per_page": stored_bpp_after,
                }
                rows.append(rec)
                if ep_mode in rows_by_mode:
                    rows_by_mode[ep_mode].append(rec)
                if q_type in rows_by_qtype:
                    rows_by_qtype[q_type].append(rec)

            metric_keys = (
                "strict_em",
                "normalized_numeric_em",
                "normalized_w2n_em",
                "f1",
                "active_tokens",
                "encode_ms",
                "edit_ms",
                "first_query_ms",
                "stored_bytes_per_page_before_edit",
                "stored_bytes_per_page_after_edit",
                "stored_bytes_per_page",
            )
            agg = _summarize_scalar_rows(rows, metric_keys)
            agg["by_mode"] = {m: _summarize_scalar_rows(mr, metric_keys) for m, mr in rows_by_mode.items()}
            agg["by_query_type"] = {qt: _summarize_scalar_rows(qr, ("strict_em", "normalized_numeric_em", "normalized_w2n_em")) for qt, qr in rows_by_qtype.items()}
            total_acc = gate_tp + gate_fn
            total_rej = gate_tn + gate_fp
            if total_acc > 0 or total_rej > 0:
                agg["gate_telemetry"] = {
                    "tp": gate_tp, "tn": gate_tn, "fp": gate_fp, "fn": gate_fn,
                    "false_accept_rate": round(gate_fp / max(1, total_rej), 4),
                    "false_reject_rate": round(gate_fn / max(1, total_acc), 4),
                    "gate_accuracy": round((gate_tp + gate_tn) / max(1, total_acc + total_rej), 4),
                    "mean_gate_accepted": round(sum(acc_gates) / max(1, len(acc_gates)), 4),
                    "mean_gate_rejected": round(sum(rej_gates) / max(1, len(rej_gates)), 4),
                }
            agg["episodes"] = rows
            seed_sys_out[sys_name] = agg
            seed_norm_scores[sys_name] = [r["normalized_numeric_em"] for r in rows]
            seed_w2n_scores[sys_name] = [r["normalized_w2n_em"] for r in rows]
            pooled_scores[sys_name].extend(seed_norm_scores[sys_name])
            pooled_strict[sys_name].extend(r["strict_em"] for r in rows)
            pooled_w2n[sys_name].extend(seed_w2n_scores[sys_name])

            g_info = ""
            if "gate_telemetry" in agg:
                gt = agg["gate_telemetry"]
                g_info = f" FAR={gt['false_accept_rate']:.3f} FRR={gt['false_reject_rate']:.3f}"
            print(
                f"[TIE-G2 seed={seed_val} {sys_name}] norm_EM={agg['normalized_numeric_em']:.3f} "
                f"w2n={agg['normalized_w2n_em']:.3f} (bud={agg['by_mode']['budget']['normalized_numeric_em']:.3f}, "
                f"sym={agg['by_mode']['symbolic']['normalized_numeric_em']:.3f}) tok={agg['active_tokens']:.0f}{g_info}",
                flush=True,
            )

        # Causal Control 4a(3): Blind Edit Routing slice (ZERO oracle p_i — route edit via BGE+lexical!)
        blind_half = 2 if is_mock else max(1, min(blind_edit_slice_n // 2, len(eps_budget), len(eps_sym)))
        blind_indices = list(range(blind_half)) + [len(eps_budget) + j for j in range(blind_half)]
        blind_rows: list[dict[str, Any]] = []
        edit_route_hits: list[float] = []
        for ep_idx in blind_indices:
            ep = all_eps[ep_idx]
            clean_q = tokenizer.decode(ep.metadata["router_query_ids"].tolist(), skip_special_tokens=True).strip()
            gold = str(ep.metadata["answer_str"]).strip()
            p_prompt = torch.tensor(
                tokenizer.encode(_format_qa_prompt(clean_q), add_special_tokens=False), dtype=torch.long, device=device
            ).unsqueeze(0)
            p_t = ep.context_pages.unsqueeze(0).to(device)
            enc_pack = standalone.encode_book(
                p_t, embed_fn, tokenizer, vocab_size, amp_dtype=amp_dtype,
                autocast_enabled=autocast_enabled, special_token_ids=special_ids, num_spans=4,
            )
            bk, bge_bf16 = enc_pack["bank"], enc_pack["bge_keys_bf16"]
            ep_hits: list[float] = []
            for p_true, e_ids_cpu in ep.edit_pages:
                e_ids = e_ids_cpu.unsqueeze(0).to(device)
                e_txt = tokenizer.decode(e_ids_cpu.tolist(), skip_special_tokens=True)
                p_pred = standalone.route_edit_target_page(
                    bk, bge_bf16, e_ids, e_txt, embed_fn, salience_route_weight=salience_route_weight
                )
                is_hit = 1.0 if int(p_pred) == int(p_true) else 0.0
                edit_route_hits.append(is_hit)
                ep_hits.append(is_hit)
                bk, _, _, _ = standalone.apply_edit(
                    bk, int(p_pred), e_ids, wrapper.llm, embed_fn, vocab_size,
                    amp_dtype=amp_dtype, autocast_enabled=autocast_enabled,
                    in_place=True, span_aligned_overwrite=True, sep_token_id=sep_id,
                )
            rq_ids = ep.metadata["router_query_ids"].unsqueeze(0).to(device)
            _, pref, _ = standalone.route_and_assemble(
                bk, bge_bf16, rq_ids, clean_q, embed_fn, k_deep=4, salience_route_weight=salience_route_weight
            )
            gen_txt = _greedy_generate_qa(
                wrapper.llm, pref.to(llm_dtype), p_prompt, tokenizer, max_new_tokens=20, user_open_ids=user_open_ids
            )
            qm = compute_community_qa_metrics(gen_txt, gold)
            fl = gen_txt.splitlines()[0] if gen_txt.strip() else ""
            cp, cg = canonicalize_numeric_answer(fl), canonicalize_numeric_answer(gold)
            nem = 1.0 if (cg and cp == cg) else float(qm["extracted_em"])
            wp, wg = canonicalize_w2n_answer(fl), canonicalize_w2n_answer(gold)
            wem = 1.0 if (wg and wp == wg) else nem
            blind_rows.append(
                {
                    "episode_idx": ep_idx,
                    "mode": str(ep.metadata.get("mode", "budget")),
                    "query_type": str(ep.metadata.get("query_type", "")),
                    "gold": gold,
                    "pred": gen_txt,
                    "edit_page_routing_accuracy": round(sum(ep_hits) / max(1, len(ep_hits)), 4),
                    "strict_em": float(qm["extracted_em"]),
                    "normalized_numeric_em": float(nem),
                    "normalized_w2n_em": float(wem),
                }
            )
        oracle_slice_norm = [seed_sys_out["standalone_discrete_span_bank"]["episodes"][i]["normalized_numeric_em"] for i in blind_indices]
        blind_slice_norm = [r["normalized_numeric_em"] for r in blind_rows]
        blind_summary = _summarize_scalar_rows(
            blind_rows, ("edit_page_routing_accuracy", "strict_em", "normalized_numeric_em", "normalized_w2n_em")
        )
        blind_summary["edit_turn_routing_accuracy"] = round(sum(edit_route_hits) / max(1, len(edit_route_hits)), 4)
        blind_summary["num_edit_turns"] = len(edit_route_hits)
        blind_summary["vs_oracle_edit_page_same_slice"] = paired_bootstrap_ci95(blind_slice_norm, oracle_slice_norm, seed=seed_val)
        blind_summary["episodes"] = blind_rows

        pairs_g2 = [
            ("standalone_discrete_span_bank", "standalone_append_192tok"),
            ("standalone_discrete_span_bank", "raw_text_same_pages_k4"),
            ("standalone_discrete_span_bank", "gated_bm25_rag"),
            ("standalone_discrete_span_bank", "bm25_rag_append_errata"),
            ("standalone_discrete_span_bank", "lslc_exact_only"),
            ("lslc_full", "standalone_discrete_span_bank"),
            ("raw_text_same_pages_k4", "gated_bm25_rag"),
            ("gated_bm25_rag", "bm25_rag_append_errata"),
        ]
        by_seed[str(seed_val)] = {
            "num_episodes": len(all_eps),
            "systems": seed_sys_out,
            "blind_edit_routing_slice": blind_summary,
            "pairwise_norm_em": compute_pairwise_suite(seed_norm_scores, pairs_g2, seed=seed_val),
            "pairwise_w2n_em": compute_pairwise_suite(seed_w2n_scores, pairs_g2, seed=seed_val),
        }

    multi_seed_summary: dict[str, Any] = {}
    for sys_name in all_systems:
        per_s_means = [by_seed[str(s)]["systems"][sys_name]["normalized_numeric_em"] for s in eff_seeds]
        mean_s = sum(per_s_means) / len(per_s_means)
        std_s = math.sqrt(sum((x - mean_s) ** 2 for x in per_s_means) / max(1, len(per_s_means) - 1)) if len(per_s_means) > 1 else 0.0
        pooled_vals = pooled_scores[sys_name]
        pooled_mean = sum(pooled_vals) / max(1, len(pooled_vals))
        pooled_w2n_mean = sum(pooled_w2n[sys_name]) / max(1, len(pooled_w2n[sys_name]))
        multi_seed_summary[sys_name] = {
            "per_seed_norm_em": {str(s): by_seed[str(s)]["systems"][sys_name]["normalized_numeric_em"] for s in eff_seeds},
            "per_seed_w2n_em": {str(s): by_seed[str(s)]["systems"][sys_name]["normalized_w2n_em"] for s in eff_seeds},
            "mean_across_seeds": round(mean_s, 4),
            "std_across_seeds": round(std_s, 4),
            "pooled_n": len(pooled_vals),
            "pooled_norm_em": round(pooled_mean, 4),
            "pooled_norm_em_ci95": _wilson_ci95(pooled_mean, len(pooled_vals)),
            "pooled_w2n_em": round(pooled_w2n_mean, 4),
            "pooled_w2n_em_ci95": _wilson_ci95(pooled_w2n_mean, len(pooled_vals)),
            "pooled_strict_em": round(sum(pooled_strict[sys_name]) / max(1, len(pooled_strict[sys_name])), 4),
        }

    pairs_pooled = [
        ("standalone_discrete_span_bank", "standalone_append_192tok"),
        ("standalone_discrete_span_bank", "raw_text_same_pages_k4"),
        ("standalone_discrete_span_bank", "gated_bm25_rag"),
        ("standalone_discrete_span_bank", "bm25_rag_append_errata"),
        ("standalone_discrete_span_bank", "lslc_exact_only"),
        ("lslc_full", "standalone_discrete_span_bank"),
        ("raw_text_same_pages_k4", "gated_bm25_rag"),
        ("gated_bm25_rag", "bm25_rag_append_errata"),
    ]
    return {
        "eval_n_per_seed": n_each * 2,
        "seeds": list(eff_seeds),
        "span_aligned_overwrite": True,
        "standalone_storage_bytes": standalone.storage_breakdown_bytes_per_page(num_spans=4),
        "trainable_param_breakdown": standalone.trainable_param_breakdown(wrapper.llm),
        "bge_model_params": standalone.bge_model_param_count(),
        "multi_seed_summary": multi_seed_summary,
        "pooled_pairwise_norm_em": compute_pairwise_suite(pooled_scores, pairs_pooled, seed=eff_seeds[0]),
        "pooled_pairwise_w2n_em": compute_pairwise_suite(pooled_w2n, pairs_pooled, seed=eff_seeds[0]),
        "by_seed": by_seed,
    }


# ---------------------------------------------------------------------------
# Pillar 2: Coding N=24 Tie-Breaker (1 Unified Run, 1 JSON, Exact Same Pages)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_coding_tiebreaker_n24(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    device: torch.device,
    *,
    bge_toc: _PlugInBiEncoderTOC | None = None,
    pathb: PathBDiscreteMemory | None = None,
    num_distractors: int = 58,
    max_tasks: int = 24,
    salience_route_weight: float = 1.5,
    lslc_k_skim: int = 2,
    max_new_tokens: int = 256,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Single-run Coding N=24 comparing BM25, BGE-4span, pure Path B `standalone_discrete_k4`,
    matched-page/matched-gate `raw_text_same_pages_k4`, and identical-page `lslc_exact_only` vs `lslc_full`."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    sep_id = get_newline_sep_token_id(tokenizer)
    eff_span_len = int(getattr(wrapper.cfg, "salience_span_len", 16))
    wrapper.eval()

    if pathb is not None:
        standalone = StandaloneDiscreteSpanBank.from_pathb(pathb, device=device)
    else:
        standalone = StandaloneDiscreteSpanBank(
            cfg=wrapper.cfg,
            salience_gate=wrapper.lslc.salience_gate,
            hw_coord_embed=None,
            delta_updater=wrapper.lslc.delta_updater,
            bge_toc=bge_toc,
            device=device,
            k_exact=wrapper.cfg.k_exact,
            salience_span_len=eff_span_len,
            bank_write_threshold=0.45,
            span_aligned_overwrite=True,
        )

    n_dist = 8 if is_mock else num_distractors
    tasks = build_coding_n24_tasks(num_distractor_files=n_dist)[: (4 if is_mock else max_tasks)]
    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)

    systems = (
        "bm25_rag_k2",
        "bm25_rag_k4",
        "bm25_rag_k8",
        "bge_4span_rag_k2",
        "bge_4span_rag_k4",
        "standalone_discrete_k4",
        "raw_text_same_pages_k4",
        "lslc_exact_only_k2",
        "lslc_full_k2",
        "lslc_exact_only_k4",
        "lslc_full_k4",
    )

    out_systems: dict[str, Any] = {}
    pass_scores_by_sys: dict[str, list[float]] = {s: [] for s in systems}
    task_lslc_cache: dict[str, dict[str, Any]] = {}
    task_bge_docs_cache: dict[str, dict[str, Any]] = {}
    task_pathb_cache: dict[str, dict[str, Any]] = {}

    for sys_name in systems:
        rows: list[dict[str, Any]] = []
        per_task_records: list[dict[str, Any]] = []

        for task in tasks:
            harness = CodingAgentHarness(task, use_initial_files=True)
            pack = _build_repo_pages(task, tokenizer, wrapper.cfg.page_size)
            page_paths: list[str] = list(pack["page_paths"])
            core_page_idx = int(pack["core_page_indices"][0])
            session_page_idx = int(pack["session_page_idx"])
            pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
            core_fmt_new = f"# FILE: {task.core_file}\n{task.core_new_content}\n"
            initial_scan_files = len(task.initial_files)
            harness.apply_file_edit(task.core_file, task.core_new_content)
            q_txt = build_pathless_coding_query(task)

            bm25 = None
            docs_live: list[str] = []
            bge_pages_bf16 = None
            z_book = exact_bank = toc_embeds = None
            pb_bank = pb_bge = None
            mem_ms = 0.0

            if sys_name.startswith("bm25_rag_k"):
                docs_live = list(pack["page_texts"])
                t0 = time.perf_counter()
                bm25 = BM25PageRetriever(docs_live)
                docs_live[core_page_idx] = core_fmt_new
                bm25.update_page(core_page_idx, core_fmt_new)
                sess_txt = docs_live[session_page_idx]
                for turn in task.session_turns:
                    sess_txt = sess_txt.rstrip() + "\n" + str(turn["text"]) + "\n"
                docs_live[session_page_idx] = sess_txt
                bm25.update_page(session_page_idx, sess_txt)
                mem_ms = (time.perf_counter() - t0) * 1000.0

            elif sys_name.startswith("bge_4span_rag_k"):
                if task.task_id in task_bge_docs_cache:
                    docs_live = task_bge_docs_cache[task.task_id]["docs"]
                    bge_pages_bf16 = task_bge_docs_cache[task.task_id]["bge_pages_bf16"]
                    mem_ms = task_bge_docs_cache[task.task_id]["mem_ms"]
                else:
                    docs_live = list(pack["page_texts"])
                    docs_live[core_page_idx] = core_fmt_new
                    sess_txt = docs_live[session_page_idx]
                    for turn in task.session_turns:
                        sess_txt = sess_txt.rstrip() + "\n" + str(turn["text"]) + "\n"
                    docs_live[session_page_idx] = sess_txt
                    _sync_device(device)
                    t0 = time.perf_counter()
                    bge_pages_bf16 = (
                        bge_toc.encode_book_pages(None, docs_live, tokenizer, num_spans=4, batch_size=64).to(torch.bfloat16)
                        if bge_toc is not None
                        else None
                    )
                    _sync_device(device)
                    mem_ms = (time.perf_counter() - t0) * 1000.0
                    task_bge_docs_cache[task.task_id] = {
                        "docs": docs_live, "bge_pages_bf16": bge_pages_bf16, "mem_ms": mem_ms,
                    }

            elif sys_name in ("standalone_discrete_k4", "raw_text_same_pages_k4"):
                if task.task_id in task_pathb_cache:
                    pc = task_pathb_cache[task.task_id]
                    pb_bank, pb_bge, docs_live, mem_ms = pc["bank"], pc["bge"], pc["gated_docs"], pc["mem_ms"]
                else:
                    page_tensor = pack["page_tensor"].to(device)
                    enc_p = standalone.encode_book(
                        page_tensor, embed_fn, tokenizer, vocab_size,
                        amp_dtype=amp_dtype, autocast_enabled=autocast_enabled,
                        special_token_ids=special_ids, num_spans=4,
                    )
                    pb_bank, pb_bge = enc_p["bank"], enc_p["bge_keys_bf16"]
                    docs_live = list(pack["page_texts"])
                    docs_live[core_page_idx] = core_fmt_new
                    core_toks_new = _pad_or_chunk_tokens(
                        tokenizer.encode(core_fmt_new, add_special_tokens=False), wrapper.cfg.page_size, pad_id
                    )[0]
                    core_page_t = torch.tensor(core_toks_new, dtype=torch.long, device=device).unsqueeze(0)
                    _sync_device(device)
                    t0 = time.perf_counter()
                    pb_bank, pb_bge = standalone.reencode_single_page(
                        pb_bank, pb_bge, core_page_idx, core_page_t, embed_fn, tokenizer, vocab_size,
                        amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids,
                    )
                    sess_gated = docs_live[session_page_idx]
                    for turn in task.session_turns:
                        e_toks = torch.tensor(
                            tokenizer.encode(str(turn["text"]), add_special_tokens=False), dtype=torch.long, device=device
                        ).unsqueeze(0)
                        pb_bank, pred_acc, _, _ = standalone.apply_edit(
                            pb_bank, session_page_idx, e_toks, wrapper.llm, embed_fn, vocab_size,
                            amp_dtype=amp_dtype, autocast_enabled=autocast_enabled,
                            in_place=True, span_aligned_overwrite=True, sep_token_id=sep_id,
                        )
                        if pred_acc:
                            sess_gated = sess_gated.rstrip() + "\n" + str(turn["text"]) + "\n"
                    docs_live[session_page_idx] = sess_gated
                    _sync_device(device)
                    mem_ms = (time.perf_counter() - t0) * 1000.0
                    task_pathb_cache[task.task_id] = {
                        "bank": pb_bank, "bge": pb_bge, "gated_docs": docs_live, "mem_ms": mem_ms, "routes": {},
                    }

            else:  # lslc_exact_only_k2/k4 or lslc_full_k2/k4
                if task.task_id in task_lslc_cache:
                    c_entry = task_lslc_cache[task.task_id]
                    z_book = c_entry["z_book"]
                    exact_bank = c_entry["exact_bank"]
                    toc_embeds = c_entry["toc_embeds"]
                    bge_pages_bf16 = c_entry["bge_pages_bf16"]
                    mem_ms = c_entry["mem_ms"]
                else:
                    page_tensor = pack["page_tensor"].to(device)
                    z_book, exact_bank, toc_embeds = _encode_book_chunked(
                        wrapper.lslc, page_tensor, embed_fn, vocab_size, device, amp_dtype, autocast_enabled,
                        chunk_size=32, special_token_ids=special_ids, salience_span_len=eff_span_len,
                    )
                    if bge_toc is not None:
                        bge_pages_bf16 = bge_toc.encode_book_pages(
                            page_tensor, None, tokenizer, num_spans=4, batch_size=64
                        ).to(torch.bfloat16)
                    core_toks_new = _pad_or_chunk_tokens(
                        tokenizer.encode(core_fmt_new, add_special_tokens=False), wrapper.cfg.page_size, pad_id
                    )[0]
                    core_page_t = torch.tensor(core_toks_new, dtype=torch.long, device=device).unsqueeze(0)
                    _sync_device(device)
                    t0 = time.perf_counter()
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        z_book, exact_bank, toc_embeds = wrapper.lslc.reencode_page(
                            z_book, toc_embeds, exact_bank, page_idx=core_page_idx, new_page_token_ids=core_page_t,
                            embed_tokens_fn=embed_fn, vocab_size=vocab_size, in_place=True,
                            special_token_ids=special_ids, salience_span_len=eff_span_len,
                        )
                        if bge_toc is not None and bge_pages_bf16 is not None:
                            bge_pages_bf16[:, core_page_idx : core_page_idx + 1] = bge_toc.encode_book_pages(
                                core_page_t.unsqueeze(1), None, tokenizer, num_spans=4, batch_size=16
                            ).to(torch.bfloat16)
                        sess_p_t = torch.tensor([session_page_idx], dtype=torch.long, device=device)
                        for turn in task.session_turns:
                            e_toks = torch.tensor(
                                tokenizer.encode(str(turn["text"]), add_special_tokens=False), dtype=torch.long, device=device
                            ).unsqueeze(0)
                            ov = (
                                extract_contextual_edit_embeds(
                                    wrapper.llm, e_toks, layer_idx=wrapper.cfg.contextual_edit_layer
                                )
                                if wrapper.cfg.use_contextual_edit_encoding
                                else None
                            )
                            z_book, _, exact_bank = wrapper.lslc.apply_page_delta(
                                z_book, sess_p_t, e_toks, embed_tokens_fn=embed_fn, exact_bank=exact_bank,
                                vocab_size=vocab_size, edit_embeds_override=ov, in_place=True, bank_write_threshold=0.45,
                            )
                    _sync_device(device)
                    mem_ms = (time.perf_counter() - t0) * 1000.0
                    task_lslc_cache[task.task_id] = {
                        "z_book": z_book, "exact_bank": exact_bank, "toc_embeds": toc_embeds,
                        "bge_pages_bf16": bge_pages_bf16, "mem_ms": mem_ms, "routes": {},
                    }

            prefix_toks: list[int] = []
            core_hits: list[float] = []
            sess_hits: list[float] = []
            both_hits: list[float] = []
            chosen_by_caller: dict[str, list[int]] = {}
            patches_by_caller: dict[str, str] = {}

            for caller_path in task.caller_files:
                cur_code = harness.read_file(caller_path)
                excluded = {i for i, p in enumerate(page_paths) if p == caller_path}

                if sys_name.startswith("bm25_rag_k"):
                    assert bm25 is not None
                    k_val = int(sys_name.rsplit("_k", 1)[-1])
                    harness.search_calls += 1
                    rr = bm25.route(q_txt, k_deep=len(docs_live), k_skim=0, device=device)
                    ordered = [int(i) for i in torch.argsort(rr.logits[0], descending=True).tolist() if int(i) not in excluded]
                    chosen = ordered[:k_val]
                    for idx_c in chosen:
                        p_n = page_paths[idx_c]
                        if p_n != task.session_notes_path and p_n in harness.files:
                            _ = harness.read_file(p_n)
                    rag_txt = "\n\n".join(docs_live[i].rstrip() for i in chosen) + "\n"
                    pref = embed_fn(_encode_unpadded_rag_ids(rag_txt, tokenizer, vocab_size, device)).to(llm_dtype)

                elif sys_name.startswith("bge_4span_rag_k"):
                    k_val = int(sys_name.rsplit("_k", 1)[-1])
                    harness.search_calls += 1
                    if bge_toc is not None and bge_pages_bf16 is not None:
                        scores_bge = bge_toc.score_book(bge_toc.encode_texts([q_txt], is_query=True), bge_pages_bf16)[0]
                    else:
                        scores_bge = torch.zeros((len(docs_live),), device=device)
                    ordered = [int(i) for i in torch.argsort(scores_bge, descending=True).tolist() if int(i) not in excluded]
                    chosen = ordered[:k_val]
                    for idx_c in chosen:
                        p_n = page_paths[idx_c]
                        if p_n != task.session_notes_path and p_n in harness.files:
                            _ = harness.read_file(p_n)
                    rag_txt = "\n\n".join(docs_live[i].rstrip() for i in chosen) + "\n"
                    pref = embed_fn(_encode_unpadded_rag_ids(rag_txt, tokenizer, vocab_size, device)).to(llm_dtype)

                elif sys_name in ("standalone_discrete_k4", "raw_text_same_pages_k4"):
                    pb_routes = task_pathb_cache[task.task_id]["routes"]
                    if caller_path not in pb_routes:
                        rq_ids = torch.tensor(
                            tokenizer.encode(q_txt, add_special_tokens=False), dtype=torch.long, device=device
                        ).unsqueeze(0)
                        r_pb, pref_pb, _ = standalone.route_and_assemble(
                            pb_bank, pb_bge, rq_ids, q_txt, embed_fn,
                            k_deep=4, salience_route_weight=salience_route_weight, excluded_indices=excluded,
                        )
                        pb_routes[caller_path] = ([int(x) for x in r_pb.deep_idx[0].tolist()], pref_pb)
                    chosen, pref_pb = pb_routes[caller_path]
                    if sys_name == "standalone_discrete_k4":
                        pref = pref_pb.to(llm_dtype)
                    else:
                        harness.search_calls += 1
                        for idx_c in chosen:
                            p_n = page_paths[idx_c]
                            if p_n != task.session_notes_path and p_n in harness.files:
                                _ = harness.read_file(p_n)
                        rag_txt = "\n\n".join(docs_live[i].rstrip() for i in chosen) + "\n"
                        pref = embed_fn(_encode_unpadded_rag_ids(rag_txt, tokenizer, vocab_size, device)).to(llm_dtype)

                else:  # lslc_exact_only_k2/k4 or lslc_full_k2/k4
                    k_val = int(sys_name.rsplit("_k", 1)[-1])
                    is_ex = sys_name.startswith("lslc_exact_only_")
                    route_key = (caller_path, k_val)
                    routes_dict = task_lslc_cache[task.task_id]["routes"]
                    rq_ids = torch.tensor(
                        tokenizer.encode(q_txt, add_special_tokens=False), dtype=torch.long, device=device
                    ).unsqueeze(0)
                    ext_s = (
                        bge_toc.score_book(bge_toc.encode_texts([q_txt], is_query=True), bge_pages_bf16)
                        if (bge_toc is not None and bge_pages_bf16 is not None)
                        else None
                    )
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        if route_key not in routes_dict:
                            r_out, _ = _lslc_route_and_assemble_excluding(
                                wrapper=wrapper, z_book=z_book, toc_embeds=toc_embeds, exact_bank=exact_bank,
                                embed_fn=embed_fn, router_query_ids=rq_ids, k_deep=k_val, k_skim=lslc_k_skim,
                                salience_route_weight=salience_route_weight, excluded_indices=excluded,
                                mode="toc_only", exact_bank_attenuation="none", external_dense_logits=ext_s,
                            )
                            routes_dict[route_key] = r_out
                        r_fixed = routes_dict[route_key]
                        _, asm = wrapper.lslc.route_and_assemble(
                            z_book=z_book, toc_embeds=toc_embeds, query_embed=embed_fn(rq_ids).mean(dim=1),
                            exact_bank=exact_bank, embed_tokens_fn=embed_fn, k_deep=k_val, k_skim=lslc_k_skim,
                            mode="exact_only" if is_ex else "full", training_ste=False,
                            router_out_override=r_fixed, flat_toc_max_tokens=16, exact_bank_attenuation="none",
                        )
                    chosen = [int(x) for x in r_fixed.deep_idx[0].tolist()]
                    pref = asm.prefix_embeds.to(llm_dtype)

                chosen_set = set(chosen)
                chosen_by_caller[caller_path] = chosen
                prefix_toks.append(int(pref.shape[1]))
                core_hits.append(float(core_page_idx in chosen_set))
                sess_hits.append(float(session_page_idx in chosen_set))
                both_hits.append(float((core_page_idx in chosen_set) and (session_page_idx in chosen_set)))

                prompt_txt = _build_edit_prompt(task, caller_path, cur_code)
                p_ids = torch.tensor(
                    tokenizer.encode(prompt_txt, add_special_tokens=False), dtype=torch.long, device=device
                ).unsqueeze(0)
                raw_out = _greedy_generate_code_until_fence(
                    wrapper.llm, pref, p_ids, tokenizer, max_new_tokens=(24 if is_mock else max_new_tokens),
                    stop_fn=lambda d: _should_stop_code_gen(d, expect_opening_fence=False), user_open_ids=user_open_ids,
                )
                patched = _extract_python_code_block("```python\n" + raw_out, cur_code)
                patches_by_caller[caller_path] = patched
                harness.apply_file_edit(caller_path, patched)

            # Homonym check
            raw_homonym = harness.read_file(task.homonym_file)
            excl_h = {i for i, p in enumerate(page_paths) if p == task.homonym_file}
            if sys_name.startswith("bm25_rag_k"):
                k_val = int(sys_name.rsplit("_k", 1)[-1])
                rr_h = bm25.route(q_txt, k_deep=len(docs_live), k_skim=0, device=device)
                ch_h = [int(i) for i in torch.argsort(rr_h.logits[0], descending=True).tolist() if int(i) not in excl_h][:k_val]
                pref_h = embed_fn(
                    _encode_unpadded_rag_ids("\n\n".join(docs_live[i].rstrip() for i in ch_h) + "\n", tokenizer, vocab_size, device)
                ).to(llm_dtype)
            elif sys_name.startswith("bge_4span_rag_k"):
                k_val = int(sys_name.rsplit("_k", 1)[-1])
                scores_h = (
                    bge_toc.score_book(bge_toc.encode_texts([q_txt], is_query=True), bge_pages_bf16)[0]
                    if (bge_toc is not None and bge_pages_bf16 is not None)
                    else torch.zeros((len(docs_live),), device=device)
                )
                ch_h = [int(i) for i in torch.argsort(scores_h, descending=True).tolist() if int(i) not in excl_h][:k_val]
                pref_h = embed_fn(
                    _encode_unpadded_rag_ids("\n\n".join(docs_live[i].rstrip() for i in ch_h) + "\n", tokenizer, vocab_size, device)
                ).to(llm_dtype)
            elif sys_name in ("standalone_discrete_k4", "raw_text_same_pages_k4"):
                pb_routes = task_pathb_cache[task.task_id]["routes"]
                if "__homonym__" not in pb_routes:
                    rq_ids = torch.tensor(
                        tokenizer.encode(q_txt, add_special_tokens=False), dtype=torch.long, device=device
                    ).unsqueeze(0)
                    r_h, pref_hp, _ = standalone.route_and_assemble(
                        pb_bank, pb_bge, rq_ids, q_txt, embed_fn,
                        k_deep=4, salience_route_weight=salience_route_weight, excluded_indices=excl_h,
                    )
                    pb_routes["__homonym__"] = ([int(x) for x in r_h.deep_idx[0].tolist()], pref_hp)
                ch_h, pref_hp = pb_routes["__homonym__"]
                if sys_name == "standalone_discrete_k4":
                    pref_h = pref_hp.to(llm_dtype)
                else:
                    pref_h = embed_fn(
                        _encode_unpadded_rag_ids("\n\n".join(docs_live[i].rstrip() for i in ch_h) + "\n", tokenizer, vocab_size, device)
                    ).to(llm_dtype)
            else:
                k_val = int(sys_name.rsplit("_k", 1)[-1])
                is_ex = sys_name.startswith("lslc_exact_only_")
                h_key = ("__homonym__", k_val)
                routes_dict = task_lslc_cache[task.task_id]["routes"]
                rq_ids = torch.tensor(
                    tokenizer.encode(q_txt, add_special_tokens=False), dtype=torch.long, device=device
                ).unsqueeze(0)
                ext_s = (
                    bge_toc.score_book(bge_toc.encode_texts([q_txt], is_query=True), bge_pages_bf16)
                    if (bge_toc is not None and bge_pages_bf16 is not None)
                    else None
                )
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    if h_key not in routes_dict:
                        r_h, _ = _lslc_route_and_assemble_excluding(
                            wrapper=wrapper, z_book=z_book, toc_embeds=toc_embeds, exact_bank=exact_bank,
                            embed_fn=embed_fn, router_query_ids=rq_ids, k_deep=k_val, k_skim=lslc_k_skim,
                            salience_route_weight=salience_route_weight, excluded_indices=excl_h,
                            mode="toc_only", exact_bank_attenuation="none", external_dense_logits=ext_s,
                        )
                        routes_dict[h_key] = r_h
                    _, asm_h = wrapper.lslc.route_and_assemble(
                        z_book=z_book, toc_embeds=toc_embeds, query_embed=embed_fn(rq_ids).mean(dim=1),
                        exact_bank=exact_bank, embed_tokens_fn=embed_fn, k_deep=k_val, k_skim=lslc_k_skim,
                        mode="exact_only" if is_ex else "full", training_ste=False,
                        router_out_override=routes_dict[h_key], flat_toc_max_tokens=16, exact_bank_attenuation="none",
                    )
                pref_h = asm_h.prefix_embeds.to(llm_dtype)

            h_prompt = _build_homonym_prompt(task, raw_homonym)
            h_ids = torch.tensor(
                tokenizer.encode(h_prompt, add_special_tokens=False), dtype=torch.long, device=device
            ).unsqueeze(0)
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
            passed = 1.0 if (ev.patch_correct and ev.hidden_test_passed) else 0.0
            pass_scores_by_sys[sys_name].append(passed)

            row_agg = {
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
            rows.append(row_agg)
            per_task_records.append(
                {
                    **row_agg,
                    "pass_at_1": passed,
                    "chosen_pages_by_caller": chosen_by_caller,
                    "generated_patches": patches_by_caller,
                }
            )

        agg = _aggregate_system_rows(rows)
        agg["pass_at_1_ci95"] = _wilson_ci95(float(agg["pass_at_1"]), len(rows))
        agg["episodes"] = per_task_records
        out_systems[sys_name] = agg
        print(
            f"[TIE-CODE N={len(rows)} {sys_name}] pass@1={agg['pass_at_1']:.3f} both_hit={agg['router_both_hit_rate']:.3f} "
            f"decoy={agg['decoy_value_used_rate']:.3f} tok={agg['active_prefix_tokens_mean']:.0f}",
            flush=True,
        )

    # Verify strictly identical selected pages between lslc_full vs exact_only AND standalone_discrete_k4 vs raw_text_same_pages_k4
    for k_tag in ("k2", "k4"):
        f_eps = out_systems[f"lslc_full_{k_tag}"]["episodes"]
        e_eps = out_systems[f"lslc_exact_only_{k_tag}"]["episodes"]
        for fe, ee in zip(f_eps, e_eps):
            assert fe["chosen_pages_by_caller"] == ee["chosen_pages_by_caller"], (
                f"Selected pages mismatch on {fe['task_id']} ({k_tag})"
            )
    for se, re_ep in zip(out_systems["standalone_discrete_k4"]["episodes"], out_systems["raw_text_same_pages_k4"]["episodes"]):
        assert se["chosen_pages_by_caller"] == re_ep["chosen_pages_by_caller"], (
            f"Path B vs raw_text_same_pages_k4 page mismatch on {se['task_id']}"
        )

    pairs_c = [
        ("standalone_discrete_k4", "raw_text_same_pages_k4"),
        ("standalone_discrete_k4", "bge_4span_rag_k4"),
        ("standalone_discrete_k4", "bm25_rag_k8"),
        ("lslc_full_k2", "lslc_exact_only_k2"),
        ("lslc_full_k4", "lslc_exact_only_k4"),
        ("lslc_exact_only_k4", "bge_4span_rag_k4"),
        ("bge_4span_rag_k4", "bm25_rag_k4"),
    ]
    pairwise = compute_pairwise_suite(pass_scores_by_sys, pairs_c, seed=2026)
    diff_k2 = pairwise["lslc_full_k2__vs__lslc_exact_only_k2"]["mean_diff"]
    diff_k4 = pairwise["lslc_full_k4__vs__lslc_exact_only_k4"]["mean_diff"]
    return {
        "num_tasks": len(tasks),
        "num_distractors": n_dist,
        "single_run_all_baselines": True,
        "identical_pages_verified": True,
        "systems": out_systems,
        "pairwise_pass_at_1": pairwise,
        "path_decision": {
            "diff_k2_full_minus_exact_only": diff_k2,
            "diff_k4_full_minus_exact_only": diff_k4,
            "path_a_criterion_gt_20pp_met": bool((diff_k2 > 0.20) or (diff_k4 > 0.20)),
            "recommended_path": "Path_A" if ((diff_k2 > 0.20) or (diff_k4 > 0.20)) else "Path_B",
        },
    }


# ---------------------------------------------------------------------------
# Pillar 3: RULER + BGE-4span RAG + Independent Standalone + Unified Profile + KV Reuse
# ---------------------------------------------------------------------------


@torch.no_grad()
def benchmark_prefix_kv_cache_reuse(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    bge_toc: _PlugInBiEncoderTOC | None = None,
    pathb: PathBDiscreteMemory | None = None,
    page_counts: tuple[int, ...] = (16, 64, 128),
    num_trials: int = 12,
    is_mock: bool = False,
) -> dict[str, Any]:
    """True Two-Bank Path (B) KV Reuse directly on StandaloneDiscreteSpanBank / PathBDiscreteMemory (ZERO Z_book!).

    - Tests `first_p0` (page 0), `middle_p1` (page 1), and `last_p3` (last page) — NO `deep_pages[-1]` hardcoding,
      NO blind `[:-32]` slice! Verifies `bank1_embed_max_abs_diff == 0.0` & `bank1_pkv_max_abs_diff == 0.0` on ALL edit positions.
    - Separates `fixed_working_set_reuse` from realistic `post_rerouting_cache_hit_telemetry` (Gate-2 & Coding).
    """
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    sep_id = get_newline_sep_token_id(tokenizer)
    eff_span_len = int(getattr(wrapper.cfg, "salience_span_len", 16))
    wrapper.eval()

    if pathb is not None:
        standalone = StandaloneDiscreteSpanBank.from_pathb(pathb, device=device)
    else:
        standalone = StandaloneDiscreteSpanBank(
            cfg=wrapper.cfg,
            salience_gate=wrapper.lslc.salience_gate,
            hw_coord_embed=None,
            delta_updater=wrapper.lslc.delta_updater,
            bge_toc=bge_toc,
            device=device,
            k_exact=wrapper.cfg.k_exact,
            salience_span_len=eff_span_len,
            span_aligned_overwrite=True,
        )

    eff_pages = (4, 8) if is_mock else page_counts
    eff_trials = 3 if is_mock else num_trials
    use_pos_ids = _llm_accepts_position_ids(wrapper.llm)
    rng = random.Random(2026)
    filler_vocab = val_pool if val_pool else list(range(2, max(16, vocab_size - 2)))
    u_emb = embed_fn(
        torch.tensor(tokenizer.encode(_USER_OPEN_STR, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
    ).to(llm_dtype)
    q_ids = torch.tensor(
        tokenizer.encode(_format_qa_prompt("What is the updated protocol for Service HarborGate?"), add_special_tokens=False),
        dtype=torch.long, device=device,
    ).unsqueeze(0)
    q_emb_llm = embed_fn(q_ids).to(llm_dtype)
    edit1_ids = torch.tensor(
        _encode_text_or_mock("Approved revision: Service HarborGate updated protocol to TLS-1.2.", tokenizer, vocab_size),
        dtype=torch.long, device=device,
    ).unsqueeze(0)
    edit2_ids = torch.tensor(
        _encode_text_or_mock("Approved revision: Service HarborGate updated protocol to TLS-1.3.", tokenizer, vocab_size),
        dtype=torch.long, device=device,
    ).unsqueeze(0)

    out: dict[str, Any] = {"architecture": "PathB_TrueTwoBank_ZeroZBook"}
    for num_p in eff_pages:
        pages = [[filler_vocab[rng.randrange(len(filler_vocab))] for _ in range(wrapper.cfg.page_size)] for _ in range(num_p)]
        p_t = torch.tensor(pages, dtype=torch.long, device=device).unsqueeze(0)
        enc_p = standalone.encode_book(
            p_t, embed_fn, tokenizer, vocab_size,
            amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids,
        )
        kd = min(4, num_p)
        deep_pages = list(range(kd))
        edit_positions = [("first_p0", deep_pages[0]), ("middle_p1", deep_pages[min(1, kd - 1)]), ("last_p_end", deep_pages[-1])]

        by_edit_pos: dict[str, Any] = {}
        for pos_label, edited_page in edit_positions:
            # Turn 1 edit places `edited_page` into Bank-2 (tail) regardless of whether it is p0, p1, or last!
            delta_open = torch.ones((1, wrapper.cfg.grid_h, wrapper.cfg.grid_w), device=device, dtype=torch.float32)
            bank_t1 = enc_p["bank"].update_page_from_delta(
                page_indices=torch.tensor([edited_page], dtype=torch.long, device=device),
                edit_token_ids=edit1_ids,
                edit_embeds=embed_fn(edit1_ids),
                salience_gate=standalone.salience_gate,
                delta_gates=delta_open,
                vocab_size=vocab_size,
                in_place=False,
                bank_write_threshold=0.45,
                span_aligned_overwrite=True,
                salience_span_len=eff_span_len,
                sep_token_id=sep_id,
                embed_tokens_fn=embed_fn,
            )
            b1_before_raw, _, b1_pages_before, b2_pages_before = standalone.assemble_two_bank_prefix(
                bank_t1, deep_pages, embed_fn
            )
            # Turn 2 edit mutates the SAME page again — Bank-1 must remain 100% invariant (0.0 diff)!
            bank_t2 = bank_t1.update_page_from_delta(
                page_indices=torch.tensor([edited_page], dtype=torch.long, device=device),
                edit_token_ids=edit2_ids,
                edit_embeds=embed_fn(edit2_ids),
                salience_gate=standalone.salience_gate,
                delta_gates=delta_open,
                vocab_size=vocab_size,
                in_place=False,
                bank_write_threshold=0.45,
                span_aligned_overwrite=True,
                salience_span_len=eff_span_len,
                sep_token_id=sep_id,
                embed_tokens_fn=embed_fn,
            )
            b1_after_raw, b2_after_raw, b1_pages_after, b2_pages_after = standalone.assemble_two_bank_prefix(
                bank_t2, deep_pages, embed_fn
            )
            assert b1_pages_before == b1_pages_after and b2_pages_after == [edited_page]
            assert edited_page not in b1_pages_after

            bank1_before = torch.cat((u_emb, b1_before_raw.to(llm_dtype)), dim=1)
            bank1_after = torch.cat((u_emb, b1_after_raw.to(llm_dtype)), dim=1)
            tail_embeds = torch.cat((b2_after_raw.to(llm_dtype), q_emb_llm), dim=1)
            imm_len = int(bank1_before.shape[1])
            tail_len = int(tail_embeds.shape[1])
            total_len = imm_len + tail_len

            bank1_embed_diff = float((bank1_after.float() - bank1_before.float()).abs().max().item())
            attn_b1 = torch.ones((1, imm_len), device=device, dtype=torch.long)
            pkv_b1_before = wrapper.llm(inputs_embeds=bank1_before, attention_mask=attn_b1, use_cache=True).past_key_values
            pkv_b1_after = wrapper.llm(inputs_embeds=bank1_after, attention_mask=attn_b1, use_cache=True).past_key_values

            max_kv_diff = 0.0
            for (k0, v0), (k1, v1) in zip(_extract_kv_pairs(pkv_b1_before), _extract_kv_pairs(pkv_b1_after)):
                max_kv_diff = max(
                    max_kv_diff,
                    float((k1.float() - k0.float()).abs().max().item()),
                    float((v1.float() - v0.float()).abs().max().item()),
                )

            full_embeds = torch.cat((bank1_after, tail_embeds), dim=1)
            attn_full = torch.ones((1, total_len), device=device, dtype=torch.long)
            logits_full = wrapper.llm(inputs_embeds=full_embeds, attention_mask=attn_full, use_cache=True).logits[:, -1, :]

            reuse_kwargs: dict[str, Any] = {
                "inputs_embeds": tail_embeds,
                "attention_mask": attn_full,
                "past_key_values": _clone_kv_cache(pkv_b1_before),
                "use_cache": True,
            }
            if use_pos_ids:
                reuse_kwargs["position_ids"] = torch.arange(imm_len, total_len, device=device, dtype=torch.long).unsqueeze(0)
            logits_reused = wrapper.llm(**reuse_kwargs).logits[:, -1, :]
            logits_diff = float((logits_reused.float() - logits_full.float()).abs().max().item())

            for _ in range(2):
                _ = wrapper.llm(inputs_embeds=full_embeds, attention_mask=attn_full, use_cache=True)
                _ = wrapper.llm(**dict(reuse_kwargs, past_key_values=_clone_kv_cache(pkv_b1_before)))

            full_times, reuse_tot_times, reuse_cmp_times = [], [], []
            for _ in range(eff_trials):
                _sync_device(device)
                t0 = time.perf_counter()
                _ = wrapper.llm(inputs_embeds=full_embeds, attention_mask=attn_full, use_cache=True)
                _sync_device(device)
                full_times.append((time.perf_counter() - t0) * 1000.0)

                _sync_device(device)
                t1 = time.perf_counter()
                pw = _clone_kv_cache(pkv_b1_before)
                _sync_device(device)
                t1c = time.perf_counter()
                _ = wrapper.llm(**dict(reuse_kwargs, past_key_values=pw))
                _sync_device(device)
                t2 = time.perf_counter()
                reuse_tot_times.append((t2 - t1) * 1000.0)
                reuse_cmp_times.append((t2 - t1c) * 1000.0)

            mean_full = sum(full_times) / len(full_times)
            mean_rtot = sum(reuse_tot_times) / len(reuse_tot_times)
            mean_rcmp = sum(reuse_cmp_times) / len(reuse_cmp_times)
            by_edit_pos[pos_label] = {
                "edited_page_idx": edited_page,
                "bank1_pages": b1_pages_after,
                "bank2_pages": b2_pages_after,
                "bank1_immutable_tokens": imm_len,
                "bank2_plus_query_tokens": tail_len,
                "total_tokens": total_len,
                "immutable_fraction": round(imm_len / max(1, total_len), 4),
                "bank1_embed_max_abs_diff": round(bank1_embed_diff, 8),
                "bank1_pkv_max_abs_diff": round(max_kv_diff, 8),
                "final_logits_max_abs_diff": round(logits_diff, 5),
                "full_reprefill_ms": round(mean_full, 3),
                "reuse_with_clone_ms": round(mean_rtot, 3),
                "reuse_compute_only_ms": round(mean_rcmp, 3),
                "speedup_with_clone_x": round(mean_full / max(1e-6, mean_rtot), 3),
                "speedup_compute_only_x": round(mean_full / max(1e-6, mean_rcmp), 3),
            }

        # Top-level summary fields use `first_p0` (strictest test: editing page 0!) + full `by_edit_position`
        p0_rec = by_edit_pos["first_p0"]
        out[f"P={num_p}"] = {
            "num_pages": num_p,
            "fixed_working_set_reuse": by_edit_pos,
            "bank1_immutable_tokens": p0_rec["bank1_immutable_tokens"],
            "bank2_plus_query_tokens": p0_rec["bank2_plus_query_tokens"],
            "total_tokens": p0_rec["total_tokens"],
            "immutable_fraction": p0_rec["immutable_fraction"],
            "bank1_embed_max_abs_diff": max(v["bank1_embed_max_abs_diff"] for v in by_edit_pos.values()),
            "bank1_pkv_max_abs_diff": max(v["bank1_pkv_max_abs_diff"] for v in by_edit_pos.values()),
            "final_logits_max_abs_diff": max(v["final_logits_max_abs_diff"] for v in by_edit_pos.values()),
            "full_reprefill_ms": p0_rec["full_reprefill_ms"],
            "reuse_with_clone_ms": p0_rec["reuse_with_clone_ms"],
            "reuse_compute_only_ms": p0_rec["reuse_compute_only_ms"],
            "speedup_with_clone_x": p0_rec["speedup_with_clone_x"],
            "speedup_compute_only_x": p0_rec["speedup_compute_only_x"],
        }
        print(
            f"[KV-REUSE-PATHB P={num_p}] p0/p1/pEnd max_pkv_diff={out[f'P={num_p}']['bank1_pkv_max_abs_diff']:.1e} "
            f"Bank1={p0_rec['bank1_immutable_tokens']}/{p0_rec['total_tokens']} "
            f"full={p0_rec['full_reprefill_ms']:.2f}ms reuse={p0_rec['reuse_with_clone_ms']:.2f}ms ({p0_rec['speedup_with_clone_x']:.2f}x)",
            flush=True,
        )

    out["post_rerouting_cache_hit_telemetry"] = measure_post_rerouting_cache_hit_telemetry(
        standalone, wrapper.llm, tokenizer, val_pool, device, is_mock=is_mock
    )
    return out


@torch.no_grad()
def run_unified_systems_profile(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    bge_toc: _PlugInBiEncoderTOC | None = None,
    pathb: PathBDiscreteMemory | None = None,
    context_lengths: tuple[int, ...] = (16384, 32768),
    k_deep_values: tuple[int, ...] = (2, 4, 8),
    decode_tokens: int = 32,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Unified single-GPU systems profile (k_exact=48, BGE BF16 3 KiB/p + 33M params) sweeping k_deep in {2,4,8}."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    eff_span_len = int(getattr(wrapper.cfg, "salience_span_len", 16))
    wrapper.eval()

    if pathb is not None:
        standalone = StandaloneDiscreteSpanBank.from_pathb(pathb, device=device)
    else:
        standalone = StandaloneDiscreteSpanBank(
            cfg=wrapper.cfg,
            salience_gate=wrapper.lslc.salience_gate,
            hw_coord_embed=None,
            delta_updater=wrapper.lslc.delta_updater,
            bge_toc=bge_toc,
            device=device,
            k_exact=wrapper.cfg.k_exact,
            salience_span_len=eff_span_len,
        )
    bge_params = standalone.bge_model_param_count()
    bge_model_mib = round((bge_params * 2) / (1024.0 ** 2), 2)
    lslc_params = wrapper.lslc.trainable_parameter_count()
    lslc_model_mib = round((lslc_params * 2) / (1024.0 ** 2), 2)
    pathb_bd = standalone.trainable_param_breakdown(wrapper.llm)
    pathb_storage = standalone.storage_breakdown_bytes_per_page(num_spans=4)

    eff_lengths = (1024, 2048) if is_mock else context_lengths
    eff_k_vals = (2, 4) if is_mock else k_deep_values
    eff_decode = 8 if is_mock else decode_tokens
    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)
    q_txt = "What is the special magic number for amber-node-12?"
    rq_ids = torch.tensor(_encode_text_or_mock(q_txt, tokenizer, vocab_size), dtype=torch.long, device=device).unsqueeze(0)
    p_prompt = torch.tensor(
        tokenizer.encode(_format_qa_prompt(q_txt), add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)

    def _measure_prefill_and_decode(prefix_embeds: torch.Tensor) -> tuple[float, float, float]:
        _reset_peak_vram(device)
        u_emb = embed_fn(user_open_ids).to(llm_dtype)
        p_emb = embed_fn(p_prompt).to(llm_dtype)
        inp = torch.cat((u_emb, prefix_embeds.to(llm_dtype), p_emb), dim=1)
        attn = torch.ones((1, inp.shape[1]), device=device, dtype=torch.long)
        _sync_device(device)
        t0 = time.perf_counter()
        _ = wrapper.llm(inputs_embeds=inp, attention_mask=attn, use_cache=True)
        _sync_device(device)
        prefill_ms = (time.perf_counter() - t0) * 1000.0

        _sync_device(device)
        t1 = time.perf_counter()
        _ = _greedy_generate_qa(
            wrapper.llm, prefix_embeds.to(llm_dtype), p_prompt, tokenizer,
            max_new_tokens=eff_decode, user_open_ids=user_open_ids, stop_fn=lambda _: False,
        )
        _sync_device(device)
        full_decode_ms = (time.perf_counter() - t1) * 1000.0
        return prefill_ms, full_decode_ms, _peak_vram_mb(device)

    rng = random.Random(2026)
    filler_vocab = val_pool if val_pool else list(range(2, max(16, vocab_size - 2)))
    by_length: dict[str, Any] = {}

    for ctx_len in eff_lengths:
        num_p = max(4, ctx_len // wrapper.cfg.page_size)
        len_key = f"{ctx_len // 1024}K" if ctx_len >= 1024 else str(ctx_len)
        pages = [[filler_vocab[rng.randrange(len(filler_vocab))] for _ in range(wrapper.cfg.page_size)] for _ in range(num_p)]
        p_t = torch.tensor(pages, dtype=torch.long, device=device).unsqueeze(0)
        page_texts = [tokenizer.decode(p, skip_special_tokens=True) for p in pages]

        # Measure independent encoding costs ONCE per system at this context length
        _sync_device(device)
        t_bge0 = time.perf_counter()
        bge_keys_raw_bf16 = (
            bge_toc.encode_book_pages(None, page_texts, tokenizer, num_spans=4, batch_size=128).to(torch.bfloat16)
            if bge_toc is not None
            else None
        )
        _sync_device(device)
        bge_rag_encode_ms = (time.perf_counter() - t_bge0) * 1000.0

        disc_pack = standalone.encode_book(
            p_t, embed_fn, tokenizer, vocab_size,
            amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids, num_spans=4,
        )

        _sync_device(device)
        t_lslc0 = time.perf_counter()
        z_book, bank_lslc, toc_lslc = _encode_book_chunked(
            wrapper.lslc, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled,
            chunk_size=32, special_token_ids=special_ids, salience_span_len=eff_span_len,
        )
        bge_keys_lslc_bf16 = (
            bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=128).to(torch.bfloat16)
            if bge_toc is not None
            else None
        )
        _sync_device(device)
        lslc_encode_ms = (time.perf_counter() - t_lslc0) * 1000.0

        sweep: dict[str, Any] = {}
        for kd in eff_k_vals:
            kd_eff = min(kd, num_p)
            # 1) bge_4span_rag
            if bge_toc is not None and bge_keys_raw_bf16 is not None:
                sc_b = bge_toc.score_book(bge_toc.encode_texts([q_txt], is_query=True), bge_keys_raw_bf16)[0]
                chosen_b = torch.argsort(sc_b, descending=True)[:kd_eff].tolist()
            else:
                chosen_b = list(range(kd_eff))
            rag_txt = "\n\n".join(page_texts[i].strip() for i in chosen_b) + "\n"
            pref_b = embed_fn(_encode_unpadded_rag_ids(rag_txt, tokenizer, vocab_size, device)).to(llm_dtype)
            pf_b, fd_b, vram_b = _measure_prefill_and_decode(pref_b)

            # 2) standalone_discrete_span_bank
            _, pref_d, rt_d = standalone.route_and_assemble(
                disc_pack["bank"], disc_pack["bge_keys_bf16"], rq_ids, q_txt, embed_fn, k_deep=kd_eff, salience_route_weight=1.5
            )
            pf_d, fd_d, vram_d = _measure_prefill_and_decode(pref_d.to(llm_dtype))

            # 3) lslc_full
            ext_l = (
                bge_toc.score_book(bge_toc.encode_texts([q_txt], is_query=True), bge_keys_lslc_bf16)
                if (bge_toc is not None and bge_keys_lslc_bf16 is not None)
                else None
            )
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                _, asm_l = wrapper.lslc.route_and_assemble(
                    z_book, toc_lslc, embed_fn(rq_ids).mean(dim=1), bank_lslc, embed_fn,
                    k_deep=kd_eff, k_skim=min(2, max(0, num_p - kd_eff)), mode="full",
                    training_ste=False, hierarchical_chapter_size=16, query_input_ids=rq_ids,
                    salience_route_weight=1.5, exact_bank_attenuation="none", external_dense_logits=ext_l,
                )
            pf_l, fd_l, vram_l = _measure_prefill_and_decode(asm_l.prefix_embeds.to(llm_dtype))

            bge_keys_bpp = 4 * int(getattr(bge_toc, "dim", 384)) * 2
            bank_bpp = wrapper.cfg.k_exact * 8
            z_bpp = wrapper.cfg.num_layers * wrapper.cfg.latent_channels * wrapper.cfg.grid_h * wrapper.cfg.grid_w * 2
            raw_txt_bpp = round(sum(len(t.encode("utf-8")) for t in page_texts) / max(1, num_p), 2)

            sweep[f"k_deep={kd_eff}"] = {
                "bge_4span_rag": {
                    "active_prefix_tokens": int(pref_b.shape[1]),
                    "encode_ms": round(bge_rag_encode_ms, 2),
                    "prefill_ms": round(pf_b, 2),
                    "full_decode_ms": round(fd_b, 2),
                    "peak_vram_mb": vram_b,
                    "stored_bytes_per_page": round(bge_keys_bpp + raw_txt_bpp, 2),
                    "bge_keys_bf16_bytes_per_page": bge_keys_bpp,
                    "auxiliary_model_mib": bge_model_mib,
                },
                "standalone_discrete_span_bank": {
                    "active_prefix_tokens": int(pref_d.shape[1]),
                    "encode_ms": round(disc_pack["encode_ms"], 2),
                    "route_and_assemble_ms": round(rt_d, 3),
                    "prefill_ms": round(pf_d, 2),
                    "full_decode_ms": round(fd_d, 2),
                    "peak_vram_mb": vram_d,
                    "stored_bytes_per_page_before_edit": pathb_storage["stored_bytes_per_page_before_edit"],
                    "stored_bytes_per_page_after_edit": pathb_storage["stored_bytes_per_page_after_edit"],
                    "stored_bytes_per_page": pathb_storage["stored_bytes_per_page_before_edit"],
                    "bge_keys_bf16_bytes_per_page": bge_keys_bpp,
                    "exact_bank_bytes_per_page_before_edit": pathb_storage["exact_bank_bytes_per_page_before_edit"],
                    "exact_bank_bytes_per_page_after_edit": pathb_storage["exact_bank_bytes_per_page_after_edit"],
                    "auxiliary_model_mib": round(bge_model_mib + (pathb_bd["total_pathb_trainable_params"] * 2) / (1024.0 ** 2), 2),
                },
                "lslc_full": {
                    "active_prefix_tokens": int(asm_l.prefix_embeds.shape[1]),
                    "encode_ms": round(lslc_encode_ms, 2),
                    "prefill_ms": round(pf_l, 2),
                    "full_decode_ms": round(fd_l, 2),
                    "peak_vram_mb": vram_l,
                    "stored_bytes_per_page": bge_keys_bpp + bank_bpp + z_bpp,
                    "bge_keys_bf16_bytes_per_page": bge_keys_bpp,
                    "exact_bank_bytes_per_page": bank_bpp,
                    "z_book_bf16_bytes_per_page": z_bpp,
                    "auxiliary_model_mib": round(bge_model_mib + lslc_model_mib, 2),
                },
            }
            print(
                f"[PROFILE {len_key} k={kd_eff}] BGE-RAG={pf_b:.1f}/{fd_b:.1f}ms({pref_b.shape[1]}t) | "
                f"Discrete={pf_d:.1f}/{fd_d:.1f}ms({pref_d.shape[1]}t,enc={disc_pack['encode_ms']:.1f}ms) | "
                f"LSLC-Full={pf_l:.1f}/{fd_l:.1f}ms({asm_l.prefix_embeds.shape[1]}t,enc={lslc_encode_ms:.1f}ms)",
                flush=True,
            )

        by_length[len_key] = {
            "context_length": ctx_len,
            "num_pages": num_p,
            "full_kv_bf16_bytes_per_page": int(wrapper.cfg.page_size * wrapper.cfg.kv_bytes_per_token_fp16),
            "sweep": sweep,
        }

    return {
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "k_exact": wrapper.cfg.k_exact,
        "salience_span_len": eff_span_len,
        "decode_tokens": eff_decode,
        "bge_model_params": bge_params,
        "bge_model_bf16_mib": bge_model_mib,
        "lslc_trainable_params": lslc_params,
        "lslc_model_bf16_mib": lslc_model_mib,
        "by_length": by_length,
    }


@torch.no_grad()
def run_ruler_bge_and_discrete(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    bge_toc: _PlugInBiEncoderTOC | None = None,
    pathb: PathBDiscreteMemory | None = None,
    lengths: tuple[int, ...] = (16384, 32768),
    episodes_per_task: int = 20,
    use_raw_question_router: bool = False,
    salience_route_weight: float = 1.5,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Single-run RULER evaluation adding BGE-4span raw-text RAG k1/k2/k4, zero-training `heuristic_span16_tfidf`,
    independent `standalone_discrete_span_bank`, and `lslc_full`."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    eff_span_len = int(getattr(wrapper.cfg, "salience_span_len", 16))
    wrapper.eval()

    if pathb is not None:
        standalone = StandaloneDiscreteSpanBank.from_pathb(pathb, device=device)
    else:
        standalone = StandaloneDiscreteSpanBank(
            cfg=wrapper.cfg,
            salience_gate=wrapper.lslc.salience_gate,
            hw_coord_embed=None,
            delta_updater=wrapper.lslc.delta_updater,
            bge_toc=bge_toc,
            device=device,
            k_exact=wrapper.cfg.k_exact,
            salience_span_len=eff_span_len,
        )
    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)

    eff_lengths = (1024,) if is_mock else lengths
    eff_eps_per_task = 2 if is_mock else episodes_per_task
    tasks = ("s_niah", "mk_niah", "vt")
    systems = (
        "full_context_bf16",
        "bm25_rag_k1",
        "bm25_rag_k2",
        "bm25_rag_k4",
        "bge_4span_rag_k1",
        "bge_4span_rag_k2",
        "bge_4span_rag_k4",
        "heuristic_span16_tfidf",
        "standalone_discrete_span_bank",
        "lslc_full",
    )

    by_length: dict[str, Any] = {}
    for ctx_len in eff_lengths:
        len_key = f"{ctx_len // 1024}K" if ctx_len >= 1024 else str(ctx_len)
        all_eps_len: list[dict[str, Any]] = []
        for t_name in tasks:
            all_eps_len.extend(
                build_ruler_episodes(
                    task_name=t_name,
                    context_length=ctx_len,
                    num_episodes=eff_eps_per_task,
                    tokenizer=tokenizer,
                    val_pool=val_pool,
                    page_size=wrapper.cfg.page_size,
                    seed=seed,
                    mk_distractors=48,
                    vt_hops=3,
                    is_mock=is_mock,
                )
            )

        sys_out: dict[str, Any] = {}
        em_scores_by_sys: dict[str, list[float]] = {}
        # Cache BGE page encodings across bge_4span_rag_k1/k2/k4 (same router!) while keeping
        # standalone_discrete_span_bank and lslc_full 100% independently timed!
        bge_rag_cache: list[tuple[torch.Tensor | None, float]] = []

        for sys_name in systems:
            rows: list[dict[str, Any]] = []
            rows_by_task: dict[str, list[dict[str, Any]]] = {t: [] for t in tasks}
            for ep_idx, ep in enumerate(all_eps_len):
                p_ids_cpu: torch.Tensor = ep["pages"]
                page_texts: list[str] = ep["page_texts"]
                gold_set = set(ep["gold_pages"])
                route_q = str(ep["question"] if use_raw_question_router else ep["router_query"])
                p_prompt = torch.tensor(
                    tokenizer.encode(_format_qa_prompt(ep["question"]), add_special_tokens=False),
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)

                encode_ms = 0.0
                router_hit: float | None = None
                stored_bpp = 0.0

                if sys_name == "full_context_bf16":
                    _sync_device(device)
                    t_q0 = time.perf_counter()
                    flat_ids = p_ids_cpu.reshape(1, -1).to(device)
                    pref = embed_fn(flat_ids).to(llm_dtype)
                    gen_txt = _greedy_generate_qa(
                        wrapper.llm, pref, p_prompt, tokenizer, max_new_tokens=24, user_open_ids=user_open_ids
                    )
                    _sync_device(device)
                    q_ms = (time.perf_counter() - t_q0) * 1000.0
                    pref_toks = int(pref.shape[1])
                    stored_bpp = float(wrapper.cfg.page_size * wrapper.cfg.kv_bytes_per_token_fp16)
                    del flat_ids, pref
                    if device.type == "cuda" and ctx_len >= 16384:
                        torch.cuda.empty_cache()

                elif sys_name.startswith("bm25_rag_k"):
                    k_val = int(sys_name.rsplit("_k", 1)[-1])
                    t_e0 = time.perf_counter()
                    bm25 = BM25PageRetriever(page_texts)
                    encode_ms = (time.perf_counter() - t_e0) * 1000.0
                    _sync_device(device)
                    t_q0 = time.perf_counter()
                    rr = bm25.route(route_q, k_deep=min(k_val, len(page_texts)), k_skim=0, device=device)
                    chosen = rr.deep_idx[0].tolist()
                    router_hit = 1.0 if gold_set.issubset(set(chosen)) else (len(gold_set & set(chosen)) / max(1, len(gold_set)))
                    rag_txt = "\n\n".join(page_texts[i].strip() for i in chosen) + "\n"
                    pref = embed_fn(_encode_unpadded_rag_ids(rag_txt, tokenizer, vocab_size, device)).to(llm_dtype)
                    gen_txt = _greedy_generate_qa(
                        wrapper.llm, pref, p_prompt, tokenizer, max_new_tokens=24, user_open_ids=user_open_ids
                    )
                    _sync_device(device)
                    q_ms = (time.perf_counter() - t_q0) * 1000.0
                    pref_toks = int(pref.shape[1])
                    stored_bpp = round(sum(len(t.encode("utf-8")) for t in page_texts) / max(1, len(page_texts)), 2)

                elif sys_name.startswith("bge_4span_rag_k"):
                    k_val = int(sys_name.rsplit("_k", 1)[-1])
                    if ep_idx < len(bge_rag_cache):
                        bge_k_bf16, encode_ms = bge_rag_cache[ep_idx]
                    else:
                        _sync_device(device)
                        t_e0 = time.perf_counter()
                        bge_k_bf16 = (
                            bge_toc.encode_book_pages(
                                p_ids_cpu.unsqueeze(0).to(device), None, tokenizer, num_spans=4, batch_size=128
                            ).to(torch.bfloat16)
                            if bge_toc is not None
                            else None
                        )
                        _sync_device(device)
                        encode_ms = (time.perf_counter() - t_e0) * 1000.0
                        bge_rag_cache.append((bge_k_bf16, encode_ms))
                    _sync_device(device)
                    t_q0 = time.perf_counter()
                    if bge_toc is not None and bge_k_bf16 is not None:
                        sc = bge_toc.score_book(bge_toc.encode_texts([route_q], is_query=True), bge_k_bf16)[0]
                        chosen = torch.argsort(sc, descending=True)[: min(k_val, len(page_texts))].tolist()
                    else:
                        chosen = list(range(min(k_val, len(page_texts))))
                    router_hit = 1.0 if gold_set.issubset(set(chosen)) else (len(gold_set & set(chosen)) / max(1, len(gold_set)))
                    rag_txt = "\n\n".join(page_texts[i].strip() for i in chosen) + "\n"
                    pref = embed_fn(_encode_unpadded_rag_ids(rag_txt, tokenizer, vocab_size, device)).to(llm_dtype)
                    gen_txt = _greedy_generate_qa(
                        wrapper.llm, pref, p_prompt, tokenizer, max_new_tokens=24, user_open_ids=user_open_ids
                    )
                    _sync_device(device)
                    q_ms = (time.perf_counter() - t_q0) * 1000.0
                    pref_toks = int(pref.shape[1])
                    bge_bpp = (bge_k_bf16.numel() * bge_k_bf16.element_size()) / max(1, len(page_texts)) if bge_k_bf16 is not None else 0.0
                    stored_bpp = round(bge_bpp + sum(len(t.encode("utf-8")) for t in page_texts) / max(1, len(page_texts)), 2)

                elif sys_name == "heuristic_span16_tfidf":
                    # Causal Control 4c: 0 trained params (TF-IDF token-rarity span-16 extraction + identical BGE+lexical router)
                    p_t = p_ids_cpu.unsqueeze(0).to(device)
                    _sync_device(device)
                    t_e0 = time.perf_counter()
                    h_bank = heuristic_span16_tfidf_bank(
                        p_t, k_exact=standalone.k_exact, span_len=eff_span_len, special_token_ids=special_ids
                    )
                    bge_h = (
                        bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=128).to(torch.bfloat16)
                        if bge_toc is not None
                        else None
                    )
                    _sync_device(device)
                    encode_ms = (time.perf_counter() - t_e0) * 1000.0
                    bge_bpp = (bge_h.numel() * bge_h.element_size()) / max(1, p_t.shape[1]) if bge_h is not None else 0.0
                    stored_bpp = round((h_bank.storage_bytes() / max(1, p_t.shape[1])) + bge_bpp, 2)
                    rq_ids = torch.tensor(
                        _encode_text_or_mock(route_q, tokenizer, vocab_size), dtype=torch.long, device=device
                    ).unsqueeze(0)
                    _sync_device(device)
                    t_q0 = time.perf_counter()
                    r_out, pref, _ = standalone.route_and_assemble(
                        h_bank, bge_h, rq_ids, route_q, embed_fn,
                        k_deep=min(4, p_t.shape[1]), salience_route_weight=salience_route_weight,
                    )
                    chosen = r_out.deep_idx[0].tolist()
                    router_hit = 1.0 if gold_set.issubset(set(chosen)) else (len(gold_set & set(chosen)) / max(1, len(gold_set)))
                    gen_txt = _greedy_generate_qa(
                        wrapper.llm, pref.to(llm_dtype), p_prompt, tokenizer, max_new_tokens=24, user_open_ids=user_open_ids
                    )
                    _sync_device(device)
                    q_ms = (time.perf_counter() - t_q0) * 1000.0
                    pref_toks = int(pref.shape[1])

                elif sys_name == "standalone_discrete_span_bank":
                    # 100% independent encoding & timing (ZERO LayeredLatentEncoder, ZERO Z_book!)
                    p_t = p_ids_cpu.unsqueeze(0).to(device)
                    enc_pack = standalone.encode_book(
                        p_t, embed_fn, tokenizer, vocab_size,
                        amp_dtype=amp_dtype, autocast_enabled=autocast_enabled,
                        special_token_ids=special_ids, num_spans=4,
                    )
                    encode_ms = float(enc_pack["encode_ms"])
                    stored_bpp = float(enc_pack["stored_bytes_per_page"])
                    rq_ids = torch.tensor(
                        _encode_text_or_mock(route_q, tokenizer, vocab_size), dtype=torch.long, device=device
                    ).unsqueeze(0)
                    _sync_device(device)
                    t_q0 = time.perf_counter()
                    r_out, pref, _ = standalone.route_and_assemble(
                        enc_pack["bank"], enc_pack["bge_keys_bf16"], rq_ids, route_q, embed_fn,
                        k_deep=min(4, p_t.shape[1]), salience_route_weight=salience_route_weight,
                    )
                    chosen = r_out.deep_idx[0].tolist()
                    router_hit = 1.0 if gold_set.issubset(set(chosen)) else (len(gold_set & set(chosen)) / max(1, len(gold_set)))
                    gen_txt = _greedy_generate_qa(
                        wrapper.llm, pref.to(llm_dtype), p_prompt, tokenizer, max_new_tokens=24, user_open_ids=user_open_ids
                    )
                    _sync_device(device)
                    q_ms = (time.perf_counter() - t_q0) * 1000.0
                    pref_toks = int(pref.shape[1])

                else:  # lslc_full
                    p_t = p_ids_cpu.unsqueeze(0).to(device)
                    _sync_device(device)
                    t_e0 = time.perf_counter()
                    z_book, bank, toc = _encode_book_chunked(
                        wrapper.lslc, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled,
                        chunk_size=32, special_token_ids=special_ids, salience_span_len=eff_span_len,
                    )
                    bge_p4 = (
                        bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=128).to(torch.bfloat16)
                        if bge_toc is not None
                        else None
                    )
                    _sync_device(device)
                    encode_ms = (time.perf_counter() - t_e0) * 1000.0
                    z_bpp = (z_book.numel() * 2) / max(1, p_t.shape[1])
                    bge_bpp = (bge_p4.numel() * bge_p4.element_size()) / max(1, p_t.shape[1]) if bge_p4 is not None else 0.0
                    stored_bpp = round(z_bpp + (bank.storage_bytes() / max(1, p_t.shape[1])) + bge_bpp, 2)

                    rq_ids = torch.tensor(
                        _encode_text_or_mock(route_q, tokenizer, vocab_size), dtype=torch.long, device=device
                    ).unsqueeze(0)
                    _sync_device(device)
                    t_q0 = time.perf_counter()
                    ext_l = (
                        bge_toc.score_book(bge_toc.encode_texts([route_q], is_query=True), bge_p4)
                        if (bge_toc is not None and bge_p4 is not None)
                        else None
                    )
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        kd_u = min(4, z_book.shape[1])
                        ks_u = min(2, max(0, z_book.shape[1] - kd_u))
                        r_out, asm = wrapper.lslc.route_and_assemble(
                            z_book=z_book, toc_embeds=toc, query_embed=embed_fn(rq_ids).mean(dim=1),
                            exact_bank=bank, embed_tokens_fn=embed_fn, k_deep=kd_u, k_skim=ks_u,
                            mode="full", training_ste=False, hierarchical_chapter_size=16,
                            query_input_ids=rq_ids, salience_route_weight=salience_route_weight,
                            exact_bank_attenuation="none", external_dense_logits=ext_l,
                        )
                    chosen = r_out.deep_idx[0].tolist()
                    router_hit = 1.0 if gold_set.issubset(set(chosen)) else (len(gold_set & set(chosen)) / max(1, len(gold_set)))
                    gen_txt = _greedy_generate_qa(
                        wrapper.llm, asm.prefix_embeds.to(llm_dtype), p_prompt, tokenizer,
                        max_new_tokens=24, user_open_ids=user_open_ids,
                    )
                    _sync_device(device)
                    q_ms = (time.perf_counter() - t_q0) * 1000.0
                    pref_toks = int(asm.prefix_embeds.shape[1])

                qm = compute_community_qa_metrics(gen_txt, ep["answer"])
                rec = {
                    "episode_idx": ep_idx,
                    "task": ep["task"],
                    "gold": ep["answer"],
                    "pred": gen_txt,
                    "extracted_em": float(qm["extracted_em"]),
                    "token_f1": float(qm["token_f1"]),
                    "substring_match": float(qm["substring_match"]),
                    "router_hit_rate": router_hit,
                    "active_prefix_tokens": float(pref_toks),
                    "encode_ms": round(encode_ms, 3),
                    "first_query_ms": round(q_ms, 3),
                    "stored_bytes_per_page": stored_bpp,
                }
                rows.append(rec)
                rows_by_task[ep["task"]].append(rec)

            m_keys = (
                "extracted_em", "token_f1", "substring_match", "router_hit_rate",
                "active_prefix_tokens", "encode_ms", "first_query_ms", "stored_bytes_per_page",
            )
            agg = _summarize_scalar_rows(rows, m_keys)
            agg["by_task"] = {t: _summarize_scalar_rows(rows_by_task[t], m_keys) for t in tasks}
            agg["episodes"] = rows
            sys_out[sys_name] = agg
            em_scores_by_sys[sys_name] = [r["extracted_em"] for r in rows]
            print(
                f"[TIE-RULER {len_key} {'rawQ' if use_raw_question_router else 'shortQ'} {sys_name}] "
                f"EM={agg['extracted_em']:.3f} hit={agg.get('router_hit_rate', 0.0):.3f} "
                f"tok={agg['active_prefix_tokens']:.0f} enc={agg['encode_ms']:.1f}ms q={agg['first_query_ms']:.1f}ms",
                flush=True,
            )

        pairs_r = [
            ("standalone_discrete_span_bank", "heuristic_span16_tfidf"),
            ("lslc_full", "standalone_discrete_span_bank"),
            ("standalone_discrete_span_bank", "bge_4span_rag_k1"),
            ("standalone_discrete_span_bank", "bge_4span_rag_k2"),
            ("standalone_discrete_span_bank", "bge_4span_rag_k4"),
            ("standalone_discrete_span_bank", "bm25_rag_k2"),
            ("lslc_full", "bge_4span_rag_k2"),
            ("lslc_full", "full_context_bf16"),
            ("standalone_discrete_span_bank", "full_context_bf16"),
        ]
        by_length[len_key] = {
            "context_length": ctx_len,
            "num_episodes": len(all_eps_len),
            "systems": sys_out,
            "pairwise_extracted_em": compute_pairwise_suite(em_scores_by_sys, pairs_r, seed=seed),
        }

    return {
        "router_query_mode": "raw_question" if use_raw_question_router else "short_query",
        "by_length": by_length,
    }


# ---------------------------------------------------------------------------
# Pillar 4: Clean 100% OOD KV Baselines (RULER 4K N=48 + WikiText PPL N=16)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_query_aware_snapkv_scores_and_kv(
    llm: nn.Module,
    context_embeds: torch.Tensor,
    query_embeds: torch.Tensor,
    *,
    per_head: bool = False,
) -> tuple[object, torch.Tensor, float]:
    """Query-Aware SnapKV: prefill [context, query] and extract attention from QUERY observation window onto context.

    - If `per_head=True`: returns exact per-layer, per-KV-head (GQA-grouped) scores `[L, num_kv_heads, ctx_len]`
      using memory-safe layer-by-layer extraction on HF models.
    - If `per_head=False`: returns summed 1D scores `[ctx_len]` (backward-compatible legacy mode).
    """
    return compute_memory_safe_obs_attention_and_kv(
        llm, context_embeds, query_embeds=query_embeds, per_head=per_head
    )


@torch.no_grad()
def run_ood_kv_baselines(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    bge_toc: _PlugInBiEncoderTOC | None = None,
    pathb: PathBDiscreteMemory | None = None,
    context_length: int = 4096,
    eval_n: int = 48,
    ppl_n: int = 16,
    salience_route_weight: float = 1.5,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """100% OOD KV Baselines on RULER 4K (N=48) + WikiText-2 PPL (N=16) with:
    - True per-layer, per-KV-head (GQA) 1D-pooled SnapKV (`pool_kernel_size=5, recent_ratio=0.10`).
    - Unified timing split (`encode_or_evict_ms`, `prefill_ms`, `decode_ms`, `decode_ms_per_tok`, `total_query_ms`).
    - Identical query input `ep["question"]` across all primary arms (plus `standalone_discrete_router_query` transparency arm).
    - Identical BGE router across `standalone_discrete_span_bank` and `lslc_full` on WikiText PPL.
    """
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    eff_span_len = int(getattr(wrapper.cfg, "salience_span_len", 16))
    wrapper.eval()

    if pathb is not None:
        standalone = StandaloneDiscreteSpanBank.from_pathb(pathb, device=device)
    else:
        standalone = StandaloneDiscreteSpanBank(
            cfg=wrapper.cfg,
            salience_gate=wrapper.lslc.salience_gate,
            hw_coord_embed=None,
            delta_updater=wrapper.lslc.delta_updater,
            bge_toc=bge_toc,
            device=device,
            k_exact=wrapper.cfg.k_exact,
            salience_span_len=eff_span_len,
        )

    eff_ctx = 1024 if is_mock else context_length
    num_p = max(4, eff_ctx // wrapper.cfg.page_size)
    per_task = 2 if is_mock else max(1, eval_n // 3)
    ruler_eps: list[dict[str, Any]] = []
    for t_name in ("s_niah", "mk_niah", "vt"):
        ruler_eps.extend(
            build_ruler_episodes(
                task_name=t_name, context_length=eff_ctx, num_episodes=per_task,
                tokenizer=tokenizer, val_pool=val_pool, page_size=wrapper.cfg.page_size,
                seed=seed + 700, mk_distractors=48, vt_hops=3, is_mock=is_mock,
            )
        )
    ppl_eps = list(ChunkedTextPageDataset(val_pool, num_pages=num_p, page_size=wrapper.cfg.page_size, turn_len=32))[
        : (3 if is_mock else ppl_n)
    ]

    # Exact iso-token budget = Path B 4 * k_exact = 192 tokens!
    iso_token_budget = min(4, num_p) * int(standalone.k_exact)

    systems = (
        "full_kv_bf16",
        "sim_int4_kv_qdq",
        "query_agnostic_h2o",
        "query_aware_snapkv",
        "standalone_discrete_span_bank",
        "standalone_discrete_router_query",
        "lslc_full",
    )
    out_sys: dict[str, Any] = {}
    em_by_sys: dict[str, list[float]] = {}
    sub_by_sys: dict[str, list[float]] = {}

    for sys_name in systems:
        ppl_nlls: list[float] = []
        for ep in ppl_eps:
            p_t = ep.context_pages.unsqueeze(0).to(device)
            q_ids = ep.query_ids.unsqueeze(0).to(device)
            a_ids = ep.answer_ids.unsqueeze(0).to(device)
            ctx_emb = embed_fn(p_t.reshape(1, -1)).to(llm_dtype)
            orig_len = int(ctx_emb.shape[1])

            if sys_name == "full_kv_bf16":
                pre = wrapper.llm(inputs_embeds=ctx_emb, attention_mask=torch.ones((1, orig_len), device=device, dtype=torch.long), use_cache=True)
                nll, _ = compute_nll_and_tok_acc_from_kv(wrapper.llm, pre.past_key_values, orig_len, q_ids, a_ids, orig_seq_len=orig_len)
            elif sys_name == "sim_int4_kv_qdq":
                pre = wrapper.llm(inputs_embeds=ctx_emb, attention_mask=torch.ones((1, orig_len), device=device, dtype=torch.long), use_cache=True)
                q_kv = quantize_past_key_values_int4(pre.past_key_values, group_size=64)
                nll, _ = compute_nll_and_tok_acc_from_kv(wrapper.llm, q_kv, orig_len, q_ids, a_ids, orig_seq_len=orig_len)
            elif sys_name == "query_agnostic_h2o":
                pkv, attn_s, _ = compute_memory_safe_obs_attention_and_kv(wrapper.llm, ctx_emb, query_embeds=None, obs_window=32, per_head=False)
                ev_kv, k_idx = evict_past_key_values_h2o_attention(pkv, attn_s, keep_budget=iso_token_budget, sink_tokens=4, recent_ratio=0.10)
                nll, _ = compute_nll_and_tok_acc_from_kv(wrapper.llm, ev_kv, int(k_idx.shape[-1]), q_ids, a_ids, orig_seq_len=orig_len)
            elif sys_name == "query_aware_snapkv":
                q_emb_p = embed_fn(q_ids).to(llm_dtype)
                ctx_pkv, attn_lh, _ = compute_memory_safe_obs_attention_and_kv(wrapper.llm, ctx_emb, query_embeds=q_emb_p, per_head=True)
                ev_kv, k_idx = evict_past_key_values_h2o_attention(
                    ctx_pkv, attn_lh, keep_budget=iso_token_budget, sink_tokens=4, recent_ratio=0.10, pool_kernel_size=5, pool_mode="avg"
                )
                nll, _ = compute_nll_and_tok_acc_from_kv(wrapper.llm, ev_kv, int(k_idx.shape[-1]), q_ids, a_ids, orig_seq_len=orig_len)
            elif sys_name in ("standalone_discrete_span_bank", "standalone_discrete_router_query"):
                q_txt_p = tokenizer.decode(q_ids[0].tolist(), skip_special_tokens=True)
                enc_p = standalone.encode_book(
                    p_t, embed_fn, tokenizer, vocab_size, amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids
                )
                _, pref, _ = standalone.route_and_assemble(
                    enc_p["bank"], enc_p["bge_keys_bf16"], q_ids, q_txt_p, embed_fn,
                    k_deep=min(4, num_p), salience_route_weight=salience_route_weight,
                )
                pref = pref.to(llm_dtype)
                pre = wrapper.llm(inputs_embeds=pref, attention_mask=torch.ones((1, pref.shape[1]), device=device, dtype=torch.long), use_cache=True)
                nll, _ = compute_nll_and_tok_acc_from_kv(wrapper.llm, pre.past_key_values, int(pref.shape[1]), q_ids, a_ids)
            else:  # lslc_full — MUST use identical BGE router on WikiText PPL!
                q_txt_p = tokenizer.decode(q_ids[0].tolist(), skip_special_tokens=True)
                zb, bk, tc = _encode_book_chunked(
                    wrapper.lslc, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled,
                    chunk_size=32, special_token_ids=special_ids, salience_span_len=eff_span_len,
                )
                bge_p4 = (
                    bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=64).to(torch.bfloat16)
                    if bge_toc is not None
                    else None
                )
                ext_p = (
                    bge_toc.score_book(bge_toc.encode_texts([q_txt_p], is_query=True), bge_p4)
                    if (bge_toc is not None and bge_p4 is not None)
                    else None
                )
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    _, asm = wrapper.lslc.route_and_assemble(
                        zb, tc, embed_fn(q_ids).mean(dim=1), bk, embed_fn,
                        k_deep=min(4, num_p), k_skim=min(2, max(0, num_p - 4)), mode="full",
                        hierarchical_chapter_size=16, query_input_ids=q_ids,
                        salience_route_weight=salience_route_weight, exact_bank_attenuation="none",
                        external_dense_logits=ext_p,
                    )
                pref = asm.prefix_embeds.to(llm_dtype)
                pre = wrapper.llm(inputs_embeds=pref, attention_mask=torch.ones((1, pref.shape[1]), device=device, dtype=torch.long), use_cache=True)
                nll, _ = compute_nll_and_tok_acc_from_kv(wrapper.llm, pre.past_key_values, int(pref.shape[1]), q_ids, a_ids)
            ppl_nlls.append(nll)

        rows: list[dict[str, Any]] = []
        rows_by_task: dict[str, list[dict[str, Any]]] = {"s_niah": [], "mk_niah": [], "vt": []}
        eviction_vram_peak = 0.0
        sdpa_infer_vram_peak = 0.0

        for ep_idx, ep in enumerate(ruler_eps):
            p_t = ep["pages"].unsqueeze(0).to(device)
            ctx_emb = embed_fn(p_t.reshape(1, -1)).to(llm_dtype)
            orig_len = int(ctx_emb.shape[1])
            raw_q_str = str(ep["question"])
            short_q_str = str(ep["router_query"])
            q_prompt_str = _format_qa_prompt(raw_q_str)
            q_ids = torch.tensor(tokenizer.encode(q_prompt_str, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
            # Primary arms use IDENTICAL raw query `ep["question"]`; `standalone_discrete_router_query` logs short-query column
            route_q_use = short_q_str if sys_name == "standalone_discrete_router_query" else raw_q_str
            rq_ids = torch.tensor(
                _encode_text_or_mock(route_q_use, tokenizer, vocab_size), dtype=torch.long, device=device
            ).unsqueeze(0)

            encode_or_evict_ms = 0.0
            prefix_prefill_ms = 0.0
            route_ms = 0.0
            if sys_name == "full_kv_bf16":
                _reset_peak_vram(device)
                _sync_device(device)
                t_p0 = time.perf_counter()
                pre = wrapper.llm(inputs_embeds=ctx_emb, attention_mask=torch.ones((1, orig_len), device=device, dtype=torch.long), use_cache=True)
                _sync_device(device)
                prefix_prefill_ms = (time.perf_counter() - t_p0) * 1000.0
                kv_c, kv_len, pos_orig = pre.past_key_values, orig_len, orig_len
            elif sys_name == "sim_int4_kv_qdq":
                _reset_peak_vram(device)
                _sync_device(device)
                t_p0 = time.perf_counter()
                pre = wrapper.llm(inputs_embeds=ctx_emb, attention_mask=torch.ones((1, orig_len), device=device, dtype=torch.long), use_cache=True)
                kv_c = quantize_past_key_values_int4(pre.past_key_values, group_size=64)
                _sync_device(device)
                prefix_prefill_ms = (time.perf_counter() - t_p0) * 1000.0
                kv_len, pos_orig = orig_len, orig_len
            elif sys_name == "query_agnostic_h2o":
                _reset_peak_vram(device)
                _sync_device(device)
                t_ev0 = time.perf_counter()
                pkv, attn_s, _ = compute_memory_safe_obs_attention_and_kv(wrapper.llm, ctx_emb, query_embeds=None, obs_window=32, per_head=False)
                kv_c, k_idx = evict_past_key_values_h2o_attention(pkv, attn_s, keep_budget=iso_token_budget, sink_tokens=4, recent_ratio=0.10)
                _sync_device(device)
                encode_or_evict_ms = (time.perf_counter() - t_ev0) * 1000.0
                eviction_vram_peak = max(eviction_vram_peak, _peak_vram_mb(device))
                kv_len, pos_orig = int(k_idx.shape[-1]), orig_len
                _reset_peak_vram(device)
            elif sys_name == "query_aware_snapkv":
                _reset_peak_vram(device)
                q_emb_eval = embed_fn(q_ids).to(llm_dtype)
                _sync_device(device)
                t_ev0 = time.perf_counter()
                ctx_pkv, attn_lh, _ = compute_memory_safe_obs_attention_and_kv(wrapper.llm, ctx_emb, query_embeds=q_emb_eval, per_head=True)
                kv_c, k_idx = evict_past_key_values_h2o_attention(
                    ctx_pkv, attn_lh, keep_budget=iso_token_budget, sink_tokens=4, recent_ratio=0.10,
                    pool_kernel_size=5, pool_mode="avg",
                )
                _sync_device(device)
                encode_or_evict_ms = (time.perf_counter() - t_ev0) * 1000.0
                eviction_vram_peak = max(eviction_vram_peak, _peak_vram_mb(device))
                kv_len, pos_orig = int(k_idx.shape[-1]), orig_len
                _reset_peak_vram(device)
            elif sys_name in ("standalone_discrete_span_bank", "standalone_discrete_router_query"):
                _reset_peak_vram(device)
                enc_p = standalone.encode_book(
                    p_t, embed_fn, tokenizer, vocab_size, amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids
                )
                encode_or_evict_ms = float(enc_p["encode_ms"])
                _, pref, route_ms = standalone.route_and_assemble(
                    enc_p["bank"], enc_p["bge_keys_bf16"], rq_ids, route_q_use, embed_fn,
                    k_deep=min(4, num_p), salience_route_weight=salience_route_weight,
                )
                pref = pref.to(llm_dtype)
                _sync_device(device)
                t_p0 = time.perf_counter()
                pre = wrapper.llm(inputs_embeds=pref, attention_mask=torch.ones((1, pref.shape[1]), device=device, dtype=torch.long), use_cache=True)
                _sync_device(device)
                prefix_prefill_ms = (time.perf_counter() - t_p0) * 1000.0
                kv_c, kv_len, pos_orig = pre.past_key_values, int(pref.shape[1]), int(pref.shape[1])
            else:  # lslc_full
                _reset_peak_vram(device)
                _sync_device(device)
                t_e0 = time.perf_counter()
                zb, bk, tc = _encode_book_chunked(
                    wrapper.lslc, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled,
                    chunk_size=32, special_token_ids=special_ids, salience_span_len=eff_span_len,
                )
                bge_p4 = (
                    bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=64).to(torch.bfloat16)
                    if bge_toc is not None
                    else None
                )
                _sync_device(device)
                encode_or_evict_ms = (time.perf_counter() - t_e0) * 1000.0
                _sync_device(device)
                t_r0 = time.perf_counter()
                ext_s = (
                    bge_toc.score_book(bge_toc.encode_texts([route_q_use], is_query=True), bge_p4)
                    if (bge_toc is not None and bge_p4 is not None)
                    else None
                )
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    _, asm = wrapper.lslc.route_and_assemble(
                        zb, tc, embed_fn(rq_ids).mean(dim=1), bk, embed_fn,
                        k_deep=min(4, num_p), k_skim=min(2, max(0, num_p - 4)), mode="full",
                        hierarchical_chapter_size=16, query_input_ids=rq_ids,
                        salience_route_weight=salience_route_weight, exact_bank_attenuation="none",
                        external_dense_logits=ext_s,
                    )
                _sync_device(device)
                route_ms = (time.perf_counter() - t_r0) * 1000.0
                pref = asm.prefix_embeds.to(llm_dtype)
                _sync_device(device)
                t_p0 = time.perf_counter()
                pre = wrapper.llm(inputs_embeds=pref, attention_mask=torch.ones((1, pref.shape[1]), device=device, dtype=torch.long), use_cache=True)
                _sync_device(device)
                prefix_prefill_ms = (time.perf_counter() - t_p0) * 1000.0
                kv_c, kv_len, pos_orig = pre.past_key_values, int(pref.shape[1]), int(pref.shape[1])

            gen_ids, timing = greedy_generate_from_kv_cache(
                wrapper.llm, kv_c, kv_len, q_ids, max_new_tokens=24,
                eos_token_id=tokenizer.eos_token_id, orig_seq_len=pos_orig,
                tokenizer=tokenizer, stop_on_newline=True, return_timing=True,
            )
            sdpa_infer_vram_peak = max(sdpa_infer_vram_peak, _peak_vram_mb(device))

            prefill_ms = prefix_prefill_ms + float(timing["prefill_ms"])
            decode_ms = float(timing["decode_ms"])
            total_query_ms = route_ms + prefill_ms + decode_ms

            raw_gen = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            qm = compute_community_qa_metrics(raw_gen, ep["answer"])
            rec = {
                "episode_idx": ep_idx,
                "task": ep["task"],
                "gold": ep["answer"],
                "pred": raw_gen,
                "extracted_em": float(qm["extracted_em"]),
                "substring_match": float(qm["substring_match"]),
                "token_f1": float(qm["token_f1"]),
                "active_kv_tokens": float(kv_len),
                "encode_or_evict_ms": round(encode_or_evict_ms, 3),
                "route_ms": round(route_ms, 3),
                "prefill_ms": round(prefill_ms, 3),
                "decode_ms": round(decode_ms, 3),
                "decode_ms_per_tok": float(timing["decode_ms_per_tok"]),
                "total_query_ms": round(total_query_ms, 3),
                "eviction_eager_ms": round(encode_or_evict_ms, 3),
                "sdpa_inference_ms": round(total_query_ms, 3),
            }
            rows.append(rec)
            rows_by_task[ep["task"]].append(rec)

        m_keys = (
            "extracted_em", "substring_match", "token_f1", "active_kv_tokens",
            "encode_or_evict_ms", "route_ms", "prefill_ms", "decode_ms", "decode_ms_per_tok", "total_query_ms",
            "eviction_eager_ms", "sdpa_inference_ms",
        )
        agg = _summarize_scalar_rows(rows, m_keys)
        agg["query_input_mode"] = "router_query" if sys_name == "standalone_discrete_router_query" else "raw_question"
        agg["wikitext_ppl"] = round(math.exp(min(20.0, sum(ppl_nlls) / max(1, len(ppl_nlls)))), 4)
        agg["sdpa_inference_peak_vram_mb"] = sdpa_infer_vram_peak
        if sys_name in ("query_agnostic_h2o", "query_aware_snapkv"):
            agg["eviction_peak_vram_mb"] = eviction_vram_peak
            agg["snapkv_per_head_1d_pooled"] = bool(sys_name == "query_aware_snapkv")
        if sys_name == "sim_int4_kv_qdq":
            agg["honest_quantization_note"] = "Quantize-dequantize simulated INT4 (BF16 tensor storage; VRAM is NOT packed INT4)"
            agg["analytical_packed_int4_kv_mib"] = round(estimate_int4_kv_bytes(wrapper.cfg, eff_ctx) / (1024.0 ** 2), 2)
        agg["by_task"] = {t: _summarize_scalar_rows(rows_by_task[t], m_keys) for t in ("s_niah", "mk_niah", "vt")}
        agg["episodes"] = rows
        out_sys[sys_name] = agg
        em_by_sys[sys_name] = [r["extracted_em"] for r in rows]
        sub_by_sys[sys_name] = [r["substring_match"] for r in rows]
        print(
            f"[OOD-KV {sys_name}] PPL={agg['wikitext_ppl']:.2f} EM={agg['extracted_em']:.3f} "
            f"kv={agg['active_kv_tokens']:.0f} enc/ev={agg['encode_or_evict_ms']:.1f}ms "
            f"pf={agg['prefill_ms']:.1f}ms dec={agg['decode_ms']:.1f}ms ({agg['decode_ms_per_tok']:.2f}ms/t)",
            flush=True,
        )

    pairs_kv = [
        ("standalone_discrete_span_bank", "query_aware_snapkv"),
        ("standalone_discrete_router_query", "query_aware_snapkv"),
        ("standalone_discrete_span_bank", "standalone_discrete_router_query"),
        ("standalone_discrete_span_bank", "query_agnostic_h2o"),
        ("query_aware_snapkv", "query_agnostic_h2o"),
        ("standalone_discrete_span_bank", "full_kv_bf16"),
        ("lslc_full", "standalone_discrete_span_bank"),
    ]
    return {
        "context_length": eff_ctx,
        "num_ruler_episodes": len(ruler_eps),
        "num_ppl_episodes": len(ppl_eps),
        "iso_token_budget": iso_token_budget,
        "snapkv_config": {"per_layer_per_kv_head_gqa": True, "pool_kernel_size": 5, "pool_mode": "avg", "recent_ratio": 0.10, "sink_tokens": 4},
        "icae_strawman_removed": True,
        "systems": out_sys,
        "pairwise_extracted_em": compute_pairwise_suite(em_by_sys, pairs_kv, seed=seed),
        "pairwise_substring_match": compute_pairwise_suite(sub_by_sys, pairs_kv, seed=seed),
    }


# ---------------------------------------------------------------------------
# Pillar 5: LongMemEval Discrete Probe N=9 (Before N=60)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_longmemeval_discrete_probe(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    device: torch.device,
    *,
    bge_toc: _PlugInBiEncoderTOC | None = None,
    n_total: int = 9,
    max_pages: int = 64,
    salience_route_weight: float = 1.5,
    systems_override: tuple[str, ...] | None = None,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Fast LongMemEval discrete probe (N=9 full 7-arm or fast N=30 4-arm Limitations proof without /compact)."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    eff_span_len = int(getattr(wrapper.cfg, "salience_span_len", 16))
    wrapper.eval()

    eff_n = 6 if is_mock else n_total
    items, ds_meta = load_longmemeval_stratified(
        n_total=eff_n,
        categories=("knowledge-update", "single-session-user", "multi-session"),
        seed=seed,
        allow_fallback=is_mock,
    )
    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)

    if systems_override is not None:
        systems = tuple(systems_override)
    elif int(n_total) >= 15:
        # Fast N=30 Limitations mode (no /compact): adds matched-depth standalone_discrete_k48_d4 + d6!
        systems = (
            "bm25_rag_k2",
            "bge_4span_rag_k2",
            "standalone_discrete_k48_d4",
            "standalone_discrete_k48_d6",
            "lslc_full_k48_d4",
        )
    else:
        systems = (
            "bm25_rag_k1",
            "bm25_rag_k2",
            "bge_4span_rag_k1",
            "bge_4span_rag_k2",
            "standalone_discrete_k64_d4",
            "standalone_discrete_k48_d4",
            "standalone_discrete_k48_d6",
            "lslc_full_k48_d4",
        )

    disc_k64 = StandaloneDiscreteSpanBank(
        cfg=wrapper.cfg, salience_gate=wrapper.lslc.salience_gate, hw_coord_embed=None,
        delta_updater=wrapper.lslc.delta_updater, bge_toc=bge_toc, device=device, k_exact=64, salience_span_len=eff_span_len,
    )
    disc_k48 = StandaloneDiscreteSpanBank(
        cfg=wrapper.cfg, salience_gate=wrapper.lslc.salience_gate, hw_coord_embed=None,
        delta_updater=wrapper.lslc.delta_updater, bge_toc=bge_toc, device=device, k_exact=48, salience_span_len=eff_span_len,
    )

    rows_by_sys: dict[str, list[dict[str, Any]]] = {s: [] for s in systems}
    f1_by_sys: dict[str, list[float]] = {s: [] for s in systems}
    sub_by_sys: dict[str, list[float]] = {s: [] for s in systems}
    val_em_by_sys: dict[str, list[float]] = {s: [] for s in systems}

    for item_idx, item in enumerate(items):
        qid = str(item.get("question_id", f"q_{item_idx}"))
        qtype = str(item.get("question_type", "unknown"))
        question = str(item["question"])
        gold = str(item["answer"])
        pack = _pack_longmemeval_sessions(item, tokenizer, wrapper.cfg.page_size, max_pages=max_pages)
        p_t = pack["page_tensor"].to(device)
        page_texts: list[str] = pack["page_texts"]
        page_sids: list[str] = pack["page_session_ids"]
        ans_sids: set[str] = pack["answer_session_ids"]
        p_prompt = torch.tensor(
            tokenizer.encode(_format_qa_prompt(question), add_special_tokens=False), dtype=torch.long, device=device
        ).unsqueeze(0)
        rq_ids = torch.tensor(
            _encode_text_or_mock(question, tokenizer, vocab_size), dtype=torch.long, device=device
        ).unsqueeze(0)

        # Shared BGE keys for bge_4span_rag_k1/k2
        bge_k_bf16 = (
            bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=128).to(torch.bfloat16)
            if bge_toc is not None
            else None
        )

        for sys_name in systems:
            ans_in_bank: float | None = None
            if sys_name.startswith("bm25_rag_k"):
                k_val = int(sys_name.rsplit("_k", 1)[-1])
                bm25 = BM25PageRetriever(page_texts)
                rr = bm25.route(question, k_deep=min(k_val, len(page_texts)), k_skim=0, device=device)
                chosen = rr.deep_idx[0].tolist()
                chosen_texts = [page_texts[i] for i in chosen]
                pref = embed_fn(
                    _encode_unpadded_rag_ids("\n\n".join(t.strip() for t in chosen_texts) + "\n", tokenizer, vocab_size, device)
                ).to(llm_dtype)

            elif sys_name.startswith("bge_4span_rag_k"):
                k_val = int(sys_name.rsplit("_k", 1)[-1])
                if bge_toc is not None and bge_k_bf16 is not None:
                    sc = bge_toc.score_book(bge_toc.encode_texts([question], is_query=True), bge_k_bf16)[0]
                    chosen = torch.argsort(sc, descending=True)[: min(k_val, len(page_texts))].tolist()
                else:
                    chosen = list(range(min(k_val, len(page_texts))))
                chosen_texts = [page_texts[i] for i in chosen]
                pref = embed_fn(
                    _encode_unpadded_rag_ids("\n\n".join(t.strip() for t in chosen_texts) + "\n", tokenizer, vocab_size, device)
                ).to(llm_dtype)

            elif sys_name in ("standalone_discrete_k64_d4", "standalone_discrete_k48_d4", "standalone_discrete_k48_d6"):
                sd_mod = disc_k64 if sys_name == "standalone_discrete_k64_d4" else disc_k48
                kd_use = 6 if sys_name == "standalone_discrete_k48_d6" else 4
                enc_p = sd_mod.encode_book(
                    p_t, embed_fn, tokenizer, vocab_size, amp_dtype=amp_dtype, autocast_enabled=autocast_enabled,
                    special_token_ids=special_ids, num_spans=4,
                )
                r_out, pref, _ = sd_mod.route_and_assemble(
                    enc_p["bank"], enc_p["bge_keys_bf16"], rq_ids, question, embed_fn,
                    k_deep=min(kd_use, p_t.shape[1]), salience_route_weight=salience_route_weight,
                )
                pref = pref.to(llm_dtype)
                chosen = r_out.deep_idx[0].tolist()
                chosen_texts = [page_texts[i] for i in chosen]
                bank_decoded = [
                    tokenizer.decode([int(t) for t in enc_p["bank"].token_ids[0, i].tolist() if int(t) >= 0], skip_special_tokens=True)
                    for i in chosen
                ]
                ans_in_bank = _answer_in_chosen_pages(gold, bank_decoded)

            else:  # lslc_full_k48_d4
                zb, bk, tc = _encode_book_chunked(
                    wrapper.lslc, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled,
                    chunk_size=32, special_token_ids=special_ids, salience_span_len=eff_span_len,
                )
                ext_q = (
                    bge_toc.score_book(bge_toc.encode_texts([question], is_query=True), bge_k_bf16)
                    if (bge_toc is not None and bge_k_bf16 is not None)
                    else None
                )
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    kd_u = min(4, zb.shape[1])
                    ks_u = min(2, max(0, zb.shape[1] - kd_u))
                    r_out, asm = wrapper.lslc.route_and_assemble(
                        zb, tc, embed_fn(rq_ids).mean(dim=1), bk, embed_fn,
                        k_deep=kd_u, k_skim=ks_u, mode="full", training_ste=False,
                        hierarchical_chapter_size=16, query_input_ids=rq_ids,
                        salience_route_weight=salience_route_weight, exact_bank_attenuation="none",
                        external_dense_logits=ext_q,
                    )
                pref = asm.prefix_embeds.to(llm_dtype)
                chosen = r_out.deep_idx[0].tolist()
                chosen_texts = [page_texts[i] for i in chosen]
                bank_decoded = [
                    tokenizer.decode([int(t) for t in bk.token_ids[0, i].tolist() if int(t) >= 0], skip_special_tokens=True)
                    for i in chosen
                ]
                ans_in_bank = _answer_in_chosen_pages(gold, bank_decoded)

            chosen_sids = {page_sids[i] for i in chosen if 0 <= i < len(page_sids)}
            sess_hit = float(bool(ans_sids & chosen_sids)) if ans_sids else None
            ans_in_pages = _answer_in_chosen_pages(gold, chosen_texts)
            gen_txt = _greedy_generate_qa(
                wrapper.llm, pref, p_prompt, tokenizer, max_new_tokens=32, user_open_ids=user_open_ids
            )
            qm = compute_community_qa_metrics(gen_txt, gold)
            val_em = compute_value_em(gen_txt, gold)
            rec = {
                "question_id": qid,
                "question_type": qtype,
                "question": question,
                "gold": gold,
                "pred": gen_txt,
                "extracted_em": float(qm["extracted_em"]),
                "value_em": float(val_em),
                "token_f1": float(qm["token_f1"]),
                "substring_match": float(qm["substring_match"]),
                "session_hit_at_k": sess_hit,
                "answer_substr_in_chosen_pages_rate": ans_in_pages,
                "answer_substr_in_exact_bank_rate": ans_in_bank,
                "active_prefix_tokens": float(pref.shape[1]),
            }
            rows_by_sys[sys_name].append(rec)
            f1_by_sys[sys_name].append(float(qm["token_f1"]))
            sub_by_sys[sys_name].append(float(qm["substring_match"]))
            val_em_by_sys[sys_name].append(float(val_em))

    sys_summary: dict[str, Any] = {}
    m_keys = (
        "extracted_em", "value_em", "token_f1", "substring_match", "session_hit_at_k",
        "answer_substr_in_chosen_pages_rate", "answer_substr_in_exact_bank_rate", "active_prefix_tokens",
    )
    for sys_name in systems:
        agg = _summarize_scalar_rows(rows_by_sys[sys_name], m_keys)
        agg["episodes"] = rows_by_sys[sys_name]
        sys_summary[sys_name] = agg
        print(
            f"[LME-PROBE N={len(items)} {sys_name}] F1={agg['token_f1']:.3f} ValEM={agg['value_em']:.3f} Sub={agg['substring_match']:.3f} "
            f"sess_hit={agg.get('session_hit_at_k', 0.0):.3f} ans_in_pg={agg.get('answer_substr_in_chosen_pages_rate', 0.0):.3f} "
            f"ans_in_bk={agg.get('answer_substr_in_exact_bank_rate', -1.0):.3f} tok={agg['active_prefix_tokens']:.0f}",
            flush=True,
        )

    pairs_lme = [
        ("standalone_discrete_k64_d4", "lslc_full_k48_d4"),
        ("standalone_discrete_k48_d4", "lslc_full_k48_d4"),
        ("standalone_discrete_k48_d6", "lslc_full_k48_d4"),
        ("standalone_discrete_k48_d6", "standalone_discrete_k48_d4"),
        ("bge_4span_rag_k2", "standalone_discrete_k48_d4"),
        ("bge_4span_rag_k2", "standalone_discrete_k64_d4"),
        ("bge_4span_rag_k2", "standalone_discrete_k48_d6"),
        ("bge_4span_rag_k2", "bm25_rag_k2"),
    ]
    return {
        "packing_mode": "gold_prioritized_64p",
        "dataset": ds_meta,
        "max_pages": max_pages,
        "systems": sys_summary,
        "pairwise_token_f1": compute_pairwise_suite(f1_by_sys, pairs_lme, seed=seed),
        "pairwise_value_em": compute_pairwise_suite(val_em_by_sys, pairs_lme, seed=seed),
        "pairwise_substring_match": compute_pairwise_suite(sub_by_sys, pairs_lme, seed=seed),
    }


# ---------------------------------------------------------------------------
# Pillar 6 (Reviewer #3 Addendum #4): 5-Cell Edit Ablation with Masked-Old Control
# ---------------------------------------------------------------------------


def mask_old_value_in_exact_bank(
    bank: SparseExactBank,
    target_page: int,
    old_value_str: str,
    tokenizer,
    vocab_size: int,
) -> tuple[SparseExactBank, int]:
    """Return a cloned SparseExactBank with old-value tokens on target_page masked to a neutral filler token (iso-token!)."""
    new_tok = bank.token_ids.clone()
    row = new_tok[0, target_page].tolist()
    if not old_value_str or not old_value_str.strip():
        return (
            SparseExactBank(
                token_ids=new_tok,
                page_ids=bank.page_ids.clone(),
                h_coords=bank.h_coords.clone(),
                w_coords=bank.w_coords.clone(),
                base_token_ids=None if bank.base_token_ids is None else bank.base_token_ids.clone(),
            ),
            0,
        )

    # Determine a neutral non-special filler token ID strictly disjoint from any masked old token IDs
    masked_positions: set[int] = set()

    # Pass 1: exact token-subsequence matching for `old_value_str` and `" " + old_value_str`
    for probe in (f" {old_value_str}", old_value_str):
        sub_ids = [int(t) for t in _encode_text_or_mock(probe, tokenizer, vocab_size)]
        m = len(sub_ids)
        if m == 0 or m > len(row):
            continue
        for j in range(0, len(row) - m + 1):
            if [int(x) for x in row[j : j + m]] == sub_ids:
                for pos in range(j, j + m):
                    masked_positions.add(pos)

    # Pass 2: piece-level fallback for partially captured multi-token values (e.g. "TLS-1.2" -> "TLS", "-", "1", ".", "2")
    if not masked_positions:
        old_parts = {p.lower() for p in re.findall(r"[a-z0-9]+", old_value_str.lower()) if len(p) >= 2 or p.isdigit()}
        for pos, tid in enumerate(row):
            if int(tid) < 0:
                continue
            piece = tokenizer.decode([int(tid)], skip_special_tokens=True).strip().lower()
            piece_clean = re.sub(r"[^a-z0-9]", "", piece)
            if piece_clean and piece_clean in old_parts:
                masked_positions.add(pos)

    masked_tok_ids = {int(row[pos]) for pos in masked_positions}
    special_set = set(_get_special_token_ids(tokenizer))
    filler_id: int | None = None
    for cand_str in (" ", ",", ".", "the"):
        c_ids = [int(t) for t in tokenizer.encode(cand_str, add_special_tokens=False)] if tokenizer is not None else []
        for tid in c_ids:
            if tid not in masked_tok_ids and tid not in special_set and tid >= 0:
                filler_id = tid
                break
        if filler_id is not None:
            break
    if filler_id is None:
        filler_id = next((t for t in range(3, max(16, vocab_size)) if t not in masked_tok_ids and t not in special_set), 3)

    for pos in masked_positions:
        new_tok[0, target_page, pos] = filler_id

    masked_bank = SparseExactBank(
        token_ids=new_tok,
        page_ids=bank.page_ids.clone(),
        h_coords=bank.h_coords.clone(),
        w_coords=bank.w_coords.clone(),
        base_token_ids=bank.token_ids.clone() if bank.base_token_ids is None else bank.base_token_ids.clone(),
    )
    return masked_bank, len(masked_positions)


@torch.no_grad()
def run_ablation_edit_masked_old(
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
    """5-cell iso-token Edit Ablation adding `deltaZ_ON_bank_masked_old` control + full per-episode McNemar."""
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

    nv = min(num_verbatim, 6) if is_mock else num_verbatim
    ni = min(num_implicit, 6) if is_mock else num_implicit
    suites = build_2x2_edit_episodes(
        tokenizer, val_pool, num_verbatim=nv, num_implicit=ni, num_pages=8, page_size=wrapper.cfg.page_size, seed=seed
    )

    arms = (
        ("deltaZ_ON_bank_ON", True, "upd"),
        ("deltaZ_OFF_bank_ON", False, "upd"),
        ("deltaZ_ON_bank_masked_old", True, "masked_old"),
        ("deltaZ_ON_bank_OFF", True, "orig"),
        ("deltaZ_OFF_bank_OFF", False, "orig"),
    )

    results: dict[str, Any] = {}
    sem_scores_overall: dict[str, list[float]] = {a[0]: [] for a in arms}
    sem_scores_verbatim: dict[str, list[float]] = {a[0]: [] for a in arms}
    sem_scores_implicit: dict[str, list[float]] = {a[0]: [] for a in arms}

    for arm_name, use_delta_z, bank_mode in arms:
        suite_out: dict[str, Any] = {}
        all_rows: list[dict[str, Any]] = []
        for s_name, eps in suites.items():
            rows: list[dict[str, Any]] = []
            for ep_idx, ep in enumerate(eps):
                p_t = ep["pages"].unsqueeze(0).to(device)
                tgt_p = int(ep["target_page"])
                z_book, bank, toc0 = _encode_book_chunked(
                    wrapper.lslc, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled,
                    chunk_size=16, special_token_ids=special_ids, salience_span_len=eff_span_len,
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
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    ov = (
                        extract_contextual_edit_embeds(wrapper.llm, edit_ids, layer_idx=wrapper.cfg.contextual_edit_layer)
                        if wrapper.cfg.use_contextual_edit_encoding
                        else None
                    )
                    z_upd, g_p, bank_upd = wrapper.lslc.apply_page_delta(
                        z_book.clone(),
                        torch.tensor([tgt_p], dtype=torch.long, device=device),
                        edit_ids,
                        embed_tokens_fn=embed_fn,
                        exact_bank=bank,
                        vocab_size=vocab_size,
                        edit_embeds_override=ov,
                        in_place=False,
                        bank_write_threshold=0.45,
                    )

                num_masked_old = 0
                if bank_mode == "upd":
                    bank_use = bank_upd
                elif bank_mode == "masked_old":
                    bank_use, num_masked_old = mask_old_value_in_exact_bank(
                        bank_orig, tgt_p, str(ep.get("old_answer", "")), tokenizer, vocab_size
                    )
                else:
                    bank_use = bank_orig

                z_use = z_upd if use_delta_z else z_orig
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    toc_after = wrapper.lslc.projector.project_toc(z_use)

                rq_ids = torch.tensor(
                    _encode_text_or_mock(ep["router_query"], tokenizer, vocab_size), dtype=torch.long, device=device
                ).unsqueeze(0)
                ext_s = None
                if bge_toc is not None:
                    bge_p4 = bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=64).to(torch.bfloat16)
                    ext_s = bge_toc.score_book(bge_toc.encode_texts([ep["router_query"]], is_query=True), bge_p4)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    r_out, _ = wrapper.lslc.route_and_assemble(
                        z_book=z_use, toc_embeds=toc_after, query_embed=embed_fn(rq_ids).mean(dim=1),
                        exact_bank=bank_orig, embed_tokens_fn=embed_fn, k_deep=4, k_skim=2,
                        mode="toc_only", training_ste=False, query_input_ids=rq_ids,
                        salience_route_weight=salience_route_weight, external_dense_logits=ext_s,
                    )
                    _, asm = wrapper.lslc.route_and_assemble(
                        z_book=z_use, toc_embeds=toc_after, query_embed=embed_fn(rq_ids).mean(dim=1),
                        exact_bank=bank_use, embed_tokens_fn=embed_fn, k_deep=4, k_skim=2,
                        mode="full", training_ste=False, router_out_override=r_out, exact_bank_attenuation="none",
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
                norm_g = _normalize_qa_text(ep["answer"])
                norm_p = _normalize_qa_text(gen_txt)
                if s_name == "implicit" and norm_g in ("yes", "no"):
                    first_words = norm_p.split()[:4]
                    sem_acc = 1.0 if (norm_g in first_words and (("no" if norm_g == "yes" else "yes") not in first_words)) else 0.0
                else:
                    sem_acc = max(qm["extracted_em"], qm["substring_match"])

                rec = {
                    "episode_idx": ep_idx,
                    "suite": s_name,
                    "question": ep["question"],
                    "edit": ep["edit_text"],
                    "gold": ep["answer"],
                    "old_answer": ep.get("old_answer", ""),
                    "pred": gen_txt,
                    "em": float(qm["extracted_em"]),
                    "f1": float(qm["token_f1"]),
                    "semantic_acc": float(sem_acc),
                    "num_masked_old_tokens": float(num_masked_old),
                    "mean_strip_gate": round(float(g_p[:, -1, :].float().mean().item()), 4),
                    "active_prefix_tokens": float(asm.prefix_embeds.shape[1]),
                }
                rows.append(rec)
                all_rows.append(rec)
                sem_scores_overall[arm_name].append(float(sem_acc))
                if s_name == "verbatim":
                    sem_scores_verbatim[arm_name].append(float(sem_acc))
                else:
                    sem_scores_implicit[arm_name].append(float(sem_acc))

            m_keys = ("em", "f1", "semantic_acc", "num_masked_old_tokens", "mean_strip_gate", "active_prefix_tokens")
            agg_s = _summarize_scalar_rows(rows, m_keys)
            agg_s["episodes"] = rows
            suite_out[s_name] = agg_s

        ov_arm = _summarize_scalar_rows(all_rows, ("em", "f1", "semantic_acc", "num_masked_old_tokens", "active_prefix_tokens"))
        results[arm_name] = {
            "continuous_delta_z": use_delta_z,
            "bank_mode": bank_mode,
            "overall": ov_arm,
            "by_suite": suite_out,
        }
        print(
            f"[EDIT-5CELL {arm_name}] verb_sem={suite_out['verbatim']['semantic_acc']:.3f} "
            f"(masked_tok={suite_out['verbatim']['num_masked_old_tokens']:.1f}) | "
            f"impl_sem={suite_out['implicit']['semantic_acc']:.3f} | tok={ov_arm['active_prefix_tokens']:.0f}",
            flush=True,
        )

    tok_counts = {a[0]: results[a[0]]["overall"]["active_prefix_tokens"] for a in arms}
    assert len(set(tok_counts.values())) == 1, f"Iso-token violated across 5-cell edit arms: {tok_counts}"

    pairs_e = [
        ("deltaZ_ON_bank_masked_old", "deltaZ_ON_bank_OFF"),
        ("deltaZ_ON_bank_masked_old", "deltaZ_OFF_bank_OFF"),
        ("deltaZ_ON_bank_ON", "deltaZ_OFF_bank_ON"),
        ("deltaZ_ON_bank_ON", "deltaZ_ON_bank_masked_old"),
        ("deltaZ_OFF_bank_ON", "deltaZ_ON_bank_OFF"),
    ]
    return {
        "iso_token_verified": True,
        "arms": results,
        "pairwise_overall_semantic_acc": compute_pairwise_suite(sem_scores_overall, pairs_e, seed=seed),
        "pairwise_verbatim_semantic_acc": compute_pairwise_suite(sem_scores_verbatim, pairs_e, seed=seed),
        "pairwise_implicit_semantic_acc": compute_pairwise_suite(sem_scores_implicit, pairs_e, seed=seed),
    }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Path A vs Path B Tie-Breaker Suite & 5 Code-Audit Hardening Pillars.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--mock-llm", action="store_true", help="Run 0-CU CPU smoke test with TinyMockCausalLM.")
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help=(
            "Comma-separated: train_pathb_qwen,gate2_tiebreaker_n200,coding_tiebreaker_n24,ruler_bge_and_unified_profile,"
            "ood_kv_baselines,longmemeval_discrete_probe,ablation_edit_masked_old or 'all'."
        ),
    )
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--pathb-checkpoint", type=str, default="", help="Optional lean Path B .safetensors checkpoint.")
    parser.add_argument("--save-pathb-checkpoint", type=str, default="", help="Optional output path to save trained lean Path B Qwen checkpoint.")
    parser.add_argument("--bge-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--use-bge-toc", action="store_true", default=True)
    parser.add_argument("--no-bge-toc", dest="use_bge_toc", action="store_false")
    parser.add_argument("--k-exact", type=int, default=48)
    parser.add_argument("--salience-span-len", type=int, default=16)
    parser.add_argument("--salience-route-weight", type=float, default=1.5)
    parser.add_argument("--gate2-n", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="2026,2027,2028")
    parser.add_argument("--ruler-lengths", type=str, default="16384,32768")
    parser.add_argument("--ruler-episodes-per-task", type=int, default=20)
    parser.add_argument("--ood-kv-length", type=int, default=4096)
    parser.add_argument("--ood-kv-n", type=int, default=48)
    parser.add_argument("--lme-probe-n", "--longmemeval-n", dest="lme_probe_n", type=int, default=9)
    parser.add_argument("--lme-probe-systems", type=str, default="", help="Optional comma-separated systems override for longmemeval_discrete_probe.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-json", type=str, default="/content/lslc_results/v3_path_tiebreaker.json")
    args, _ = parser.parse_known_args(argv)
    return args


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    raw_tasks = {t.strip().lower() for t in args.tasks.split(",") if t.strip()}
    all_set = {
        "train_pathb_qwen",
        "gate2_tiebreaker_n200",
        "coding_tiebreaker_n24",
        "ruler_bge_and_unified_profile",
        "ood_kv_baselines",
        "longmemeval_discrete_probe",
        "ablation_edit_masked_old",
    }
    run_tasks = all_set if "all" in raw_tasks else raw_tasks
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    ruler_lengths = tuple(int(s.strip()) for s in args.ruler_lengths.split(",") if s.strip())

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
        val_pool = [2 + ((i * 7) % (vocab_size - 2)) for i in range(4096)]
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

    # Instantiate lean PathBDiscreteMemory (ContextualEditGateHead hidden_dim=64, 0 DeltaUpdater, 0 hw_coord_embed)
    pathb = PathBDiscreteMemory(
        cfg, bge_toc=bge_toc, bank_write_threshold=0.45, edit_gate_hidden_dim=64,
        span_aligned_overwrite=True, k_exact=cfg.k_exact, salience_span_len=cfg.salience_span_len,
    ).to(device)
    pathb_ckpt_info = None
    if args.pathb_checkpoint:
        pathb_ckpt_info = load_pathb_checkpoint(pathb, args.pathb_checkpoint)
    elif ckpt_info is not None:
        pathb.salience_gate.load_state_dict(wrapper.lslc.salience_gate.state_dict(), strict=True)

    use_lean_pathb = bool(args.pathb_checkpoint or ("train_pathb_qwen" in run_tasks))
    git_prov = get_git_provenance(REPO_ROOT)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": "3.1.0-path-tiebreaker-audit-hardened",
        "git_sha": git_prov["git_sha"],
        "git_dirty": git_prov["git_dirty"],
        "checkpoint_sha256": None if ckpt_info is None else ckpt_info.get("sha256"),
        "pathb_checkpoint_sha256": None if pathb_ckpt_info is None else pathb_ckpt_info.get("sha256"),
        "trainable_param_breakdown": pathb.trainable_param_breakdown(wrapper.llm),
        "model": "TinyMockCausalLM" if args.mock_llm else args.model,
        "mock_llm": args.mock_llm,
        "device": str(device),
        "data_source": data_source,
        "init_checkpoint": ckpt_info,
        "pathb_checkpoint": pathb_ckpt_info,
        "saved_pathb_checkpoint": None,
        "config": {
            "page_size": cfg.page_size,
            "k_exact": cfg.k_exact,
            "salience_span_len": cfg.salience_span_len,
            "salience_route_weight": args.salience_route_weight,
            "bge_model": None if bge_toc is None else bge_toc.model_name,
            "seeds": list(seeds),
            "span_aligned_overwrite": True,
            "true_two_bank_layout": True,
            "edit_gate_hidden_dim": 64,
        },
        "train_pathb_qwen": None,
        "gate2_tiebreaker_n200": None,
        "coding_tiebreaker_n24": None,
        "ruler_bge_and_unified_profile": None,
        "ood_kv_baselines": None,
        "longmemeval_discrete_probe": None,
        "ablation_edit_masked_old": None,
    }

    if "train_pathb_qwen" in run_tasks:
        payload["train_pathb_qwen"] = train_lean_pathb_qwen(
            wrapper.llm, pathb, tokenizer, train_pool, device,
            copy_salience_from=wrapper.lslc.salience_gate if ckpt_info is not None else None,
            seed=args.seed, is_mock=args.mock_llm,
        )
        if args.save_pathb_checkpoint:
            saved_pb = save_pathb_checkpoint(pathb, args.save_pathb_checkpoint)
            payload["saved_pathb_checkpoint"] = saved_pb
            payload["pathb_checkpoint_sha256"] = saved_pb["sha256"]
        use_lean_pathb = True
        _atomic_write_json(out_path, payload)

    active_pathb = pathb if use_lean_pathb else None

    if "gate2_tiebreaker_n200" in run_tasks:
        payload["gate2_tiebreaker_n200"] = run_gate2_tiebreaker_n200(
            wrapper, tokenizer, val_pool, device, bge_toc=bge_toc, pathb=active_pathb,
            eval_n=args.gate2_n, seeds=seeds, salience_route_weight=args.salience_route_weight, is_mock=args.mock_llm,
        )
        _atomic_write_json(out_path, payload)

    if "coding_tiebreaker_n24" in run_tasks:
        payload["coding_tiebreaker_n24"] = run_coding_tiebreaker_n24(
            wrapper, tokenizer, device, bge_toc=bge_toc, pathb=active_pathb,
            salience_route_weight=args.salience_route_weight, is_mock=args.mock_llm,
        )
        _atomic_write_json(out_path, payload)

    if "ruler_bge_and_unified_profile" in run_tasks:
        payload["ruler_bge_and_unified_profile"] = {
            "ruler_short_query": run_ruler_bge_and_discrete(
                wrapper, tokenizer, val_pool, device, bge_toc=bge_toc, pathb=active_pathb,
                lengths=ruler_lengths, episodes_per_task=args.ruler_episodes_per_task,
                use_raw_question_router=False, salience_route_weight=args.salience_route_weight,
                seed=args.seed, is_mock=args.mock_llm,
            ),
            "ruler_raw_question_16k": run_ruler_bge_and_discrete(
                wrapper, tokenizer, val_pool, device, bge_toc=bge_toc, pathb=active_pathb,
                lengths=(16384,), episodes_per_task=args.ruler_episodes_per_task,
                use_raw_question_router=True, salience_route_weight=args.salience_route_weight,
                seed=args.seed, is_mock=args.mock_llm,
            ),
            "unified_systems_profile": run_unified_systems_profile(
                wrapper, tokenizer, val_pool, device, bge_toc=bge_toc, pathb=active_pathb,
                context_lengths=ruler_lengths, k_deep_values=(2, 4, 8), is_mock=args.mock_llm,
            ),
            "prefix_kv_cache_reuse": benchmark_prefix_kv_cache_reuse(
                wrapper, tokenizer, val_pool, device, bge_toc=bge_toc, pathb=active_pathb,
                page_counts=(16, 64, 128), is_mock=args.mock_llm,
            ),
        }
        _atomic_write_json(out_path, payload)

    if "ood_kv_baselines" in run_tasks:
        payload["ood_kv_baselines"] = run_ood_kv_baselines(
            wrapper, tokenizer, val_pool, device, bge_toc=bge_toc, pathb=active_pathb,
            context_length=args.ood_kv_length, eval_n=args.ood_kv_n,
            salience_route_weight=args.salience_route_weight, seed=args.seed, is_mock=args.mock_llm,
        )
        _atomic_write_json(out_path, payload)

    if "longmemeval_discrete_probe" in run_tasks:
        lme_sys_ov = (
            tuple(s.strip() for s in args.lme_probe_systems.split(",") if s.strip())
            if args.lme_probe_systems.strip()
            else None
        )
        payload["longmemeval_discrete_probe"] = run_longmemeval_discrete_probe(
            wrapper, tokenizer, device, bge_toc=bge_toc,
            n_total=args.lme_probe_n, max_pages=64,
            salience_route_weight=args.salience_route_weight,
            systems_override=lme_sys_ov, seed=args.seed, is_mock=args.mock_llm,
        )
        _atomic_write_json(out_path, payload)

    if "ablation_edit_masked_old" in run_tasks:
        payload["ablation_edit_masked_old"] = run_ablation_edit_masked_old(
            wrapper, tokenizer, val_pool, device, bge_toc=bge_toc,
            salience_route_weight=args.salience_route_weight, seed=args.seed, is_mock=args.mock_llm,
        )
        _atomic_write_json(out_path, payload)

    print(f"[DONE] Wrote Path Tie-Breaker results -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
