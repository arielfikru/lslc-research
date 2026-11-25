"""Milestone 3 Colab L4 Orchestrator: Full 5-Baseline Suite (Tracks 0A-0E) + Tracks 1, 2, 3 (4K/32K Book), and 4 (DeltaUpdater)."""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
import random
import sys
import time
import unittest
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
    LivingBookStateOverrideDataset,
    SyntheticExactSemanticNIAHDataset,
    compute_analytical_memory_stats,
    estimate_int4_kv_bytes,
    greedy_generate_from_prefix,
    h2o_evict_keep_indices,
    load_frozen_qwen3,
    select_amp_dtype,
    streamingllm_keep_indices,
)
from scripts.colab_run_m2 import (
    _load_wikitext_pools,
    evaluate_continuation_ppl,
    evaluate_delta_suite,
    evaluate_niah_suite,
    profile_live_4k_32k,
    train_stage,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Milestone 3 full benchmark suite on Colab L4.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--steps-track2", type=int, default=300)
    parser.add_argument("--steps-track4", type=int, default=200)
    parser.add_argument("--steps-flat1d", type=int, default=200)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed-m2-json", type=str, default="/content/lslc_results/m2_v2_summary.json")
    parser.add_argument("--out-json", type=str, default="/content/lslc_results/m3_summary.json")
    args, _ = parser.parse_known_args(argv)
    return args


def _quantize_int4_groupwise(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """Group-wise symmetric INT4 quantization (-8..+7) applied to KV cache tensors."""
    orig_shape = x.shape
    orig_dtype = x.dtype
    flat = x.float().reshape(-1, group_size) if x.numel() % group_size == 0 else x.float().reshape(1, -1)
    scale = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7.0
    q = torch.clamp(torch.round(flat / scale), -8.0, 7.0)
    return (q * scale).reshape(orig_shape).to(orig_dtype)


def _extract_kv_pairs(past_key_values) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract per-layer (K, V) tensors from HF DynamicCache (v4/v5) or legacy tuple caches."""
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return [(k, v) for k, v in zip(past_key_values.key_cache, past_key_values.value_cache)]
    if hasattr(past_key_values, "layers"):
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_cache in past_key_values.layers:
            k = getattr(layer_cache, "keys", None)
            v = getattr(layer_cache, "values", None)
            if k is not None and v is not None:
                pairs.append((k, v))
        if pairs:
            return pairs
    return [(k, v) for (k, v) in past_key_values]


def _rebuild_kv_cache(template_cache, pairs: list[tuple[torch.Tensor, torch.Tensor]]):
    """Rebuild a KV cache matching template_cache's container class (DynamicCache or tuple)."""
    if hasattr(template_cache, "key_cache") and hasattr(template_cache, "value_cache"):
        new_cache = template_cache.__class__()
        for layer_idx, (k, v) in enumerate(pairs):
            new_cache.update(k.contiguous(), v.contiguous(), layer_idx)
        return new_cache
    if hasattr(template_cache, "layers"):
        try:
            new_cache = template_cache.__class__()
            for layer_idx, (k, v) in enumerate(pairs):
                new_cache.update(k.contiguous(), v.contiguous(), layer_idx)
            return new_cache
        except Exception:
            pass
    return tuple((k.contiguous(), v.contiguous()) for k, v in pairs)


def quantize_past_key_values_int4(past_key_values, group_size: int = 64):
    """True INT4 KV Cache baseline: quantize/dequantize directly on past_key_values (NEVER inputs_embeds)."""
    pairs = _extract_kv_pairs(past_key_values)
    q_pairs = [
        (_quantize_int4_groupwise(k.clone(), group_size=group_size), _quantize_int4_groupwise(v.clone(), group_size=group_size))
        for k, v in pairs
    ]
    return _rebuild_kv_cache(past_key_values, q_pairs)


@torch.no_grad()
def compute_real_attention_scores_and_kv(
    llm: nn.Module,
    context_embeds: torch.Tensor,
    *,
    obs_window: int = 32,
) -> tuple[object, torch.Tensor]:
    """Run real LLM prefill with output_attentions=True and accumulate actual attention scores over observation window."""
    bsz, seq_len, _ = context_embeds.shape
    device = context_embeds.device
    attn_mask = torch.ones((bsz, seq_len), device=device, dtype=torch.long)

    cfg = getattr(llm, "config", None)
    prev_attn_impl = getattr(cfg, "_attn_implementation", None) if cfg is not None else None
    prev_use_cache = getattr(cfg, "use_cache", False) if cfg is not None else False
    switched_eager = False
    try:
        if cfg is not None:
            cfg.use_cache = True
            if prev_attn_impl is not None and prev_attn_impl != "eager":
                try:
                    if hasattr(llm, "set_attn_implementation"):
                        llm.set_attn_implementation("eager")
                    else:
                        cfg._attn_implementation = "eager"
                    switched_eager = True
                except Exception:
                    pass
        out = llm(
            inputs_embeds=context_embeds,
            attention_mask=attn_mask,
            use_cache=True,
            output_attentions=True,
        )
    finally:
        if cfg is not None:
            cfg.use_cache = prev_use_cache
            if switched_eager and prev_attn_impl is not None:
                try:
                    if hasattr(llm, "set_attn_implementation"):
                        llm.set_attn_implementation(prev_attn_impl)
                    else:
                        cfg._attn_implementation = prev_attn_impl
                except Exception:
                    pass

    pkv = getattr(out, "past_key_values", None)
    attns = getattr(out, "attentions", None)
    if pkv is None:
        raise RuntimeError("LLM forward with use_cache=True did not return past_key_values.")
    if not attns or attns[0] is None:
        raise RuntimeError("LLM forward with output_attentions=True did not return real attention weights.")

    # Accumulate actual post-softmax attention mass over the observation window [B, H, Q_obs, K] -> [K]
    obs = min(max(1, int(obs_window)), seq_len)
    acc = torch.zeros((seq_len,), device=device, dtype=torch.float32)
    for layer_attn in attns:
        if layer_attn is None:
            continue
        # layer_attn: [B, H, Q, K]
        w_obs = layer_attn[0, :, -obs:, :seq_len].float()
        acc = acc + w_obs.sum(dim=(0, 1))
    del out, attns
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pkv, acc


def _apply_rotary_half_m3(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Standard Llama/Qwen3 rotate_half RoPE application on [B, H, S, D]."""
    half = x.shape[-1] // 2
    x_rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return (x * cos) + (x_rot * sin)


@torch.no_grad()
def compute_memory_safe_obs_attention_and_kv(
    llm: nn.Module,
    context_embeds: torch.Tensor,
    *,
    query_embeds: torch.Tensor | None = None,
    obs_window: int = 32,
    per_head: bool = True,
) -> tuple[object, torch.Tensor, float]:
    """Memory-safe exact observation-window attention over context keys [:ctx_len] (works on 1.7B & 8B BF16 without OOM).

    - If `query_embeds is not None`: Query-Aware SnapKV (observation window = query rows `[ctx_len : ctx_len+q_len]`).
    - If `query_embeds is None`: Query-Agnostic H2O (observation window = last `obs_window` context rows).
    - If `per_head=True`: returns exact per-layer, per-KV-head (GQA-grouped) scores `[L, num_kv_heads, ctx_len]`.
    - If `per_head=False`: returns legacy summed 1D scores `[ctx_len]`.
    """
    bsz, ctx_len, _ = context_embeds.shape
    device = context_embeds.device

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    backbone = getattr(llm, "model", None)
    layers = getattr(backbone, "layers", None) if backbone is not None else None
    rotary_emb = getattr(backbone, "rotary_emb", None) if backbone is not None else None
    if rotary_emb is None and layers is not None and len(layers) > 0:
        rotary_emb = getattr(getattr(layers[0], "self_attn", None), "rotary_emb", None)

    can_use_layerwise = (
        layers is not None
        and rotary_emb is not None
        and len(layers) > 0
        and hasattr(layers[0], "self_attn")
        and hasattr(layers[0].self_attn, "q_proj")
        and hasattr(layers[0], "input_layernorm")
    )

    _sync()
    t0 = time.perf_counter()
    if query_embeds is not None:
        q_len = int(query_embeds.shape[1])
        full_embeds = torch.cat((context_embeds, query_embeds), dim=1)
        obs_start, obs_end = ctx_len, ctx_len + q_len
    else:
        full_embeds = context_embeds
        obs_len = min(max(1, int(obs_window)), ctx_len)
        obs_start, obs_end = ctx_len - obs_len, ctx_len

    total_len = int(full_embeds.shape[1])
    attn_mask = torch.ones((bsz, total_len), device=device, dtype=torch.long)

    if not can_use_layerwise:
        cfg = getattr(llm, "config", None)
        prev_attn_impl = getattr(cfg, "_attn_implementation", None) if cfg is not None else None
        prev_use_cache = getattr(cfg, "use_cache", False) if cfg is not None else False
        switched_eager = False
        try:
            if cfg is not None:
                cfg.use_cache = True
                if prev_attn_impl is not None and prev_attn_impl != "eager":
                    try:
                        if hasattr(llm, "set_attn_implementation"):
                            llm.set_attn_implementation("eager")
                        else:
                            cfg._attn_implementation = "eager"
                        switched_eager = True
                    except Exception:
                        pass
            out = llm(
                inputs_embeds=full_embeds,
                attention_mask=attn_mask,
                use_cache=True,
                output_attentions=True,
            )
        finally:
            if cfg is not None:
                cfg.use_cache = prev_use_cache
                if switched_eager and prev_attn_impl is not None:
                    try:
                        if hasattr(llm, "set_attn_implementation"):
                            llm.set_attn_implementation(prev_attn_impl)
                        else:
                            cfg._attn_implementation = prev_attn_impl
                    except Exception:
                        pass
        pkv = getattr(out, "past_key_values", None)
        attns = getattr(out, "attentions", None)
        if pkv is None or not attns or attns[0] is None:
            raise RuntimeError("Attention extraction requires past_key_values and output_attentions=True.")
        kv_pairs = _extract_kv_pairs(pkv)
        num_kv_heads = int(kv_pairs[0][0].shape[1])

        per_layer_scores: list[torch.Tensor] = []
        acc_1d = torch.zeros((ctx_len,), device=device, dtype=torch.float32)
        for layer_attn in attns:
            if layer_attn is None:
                continue
            obs_w = layer_attn[0, :, obs_start:obs_end, :ctx_len].float()  # [H_q, obs, ctx_len]
            if per_head:
                q_sum = obs_w.sum(dim=1)  # [H_q, ctx_len]
                h_q = int(q_sum.shape[0])
                gqa_r = max(1, h_q // max(1, num_kv_heads))
                if h_q == num_kv_heads * gqa_r and gqa_r > 1:
                    kv_s = q_sum.view(num_kv_heads, gqa_r, ctx_len).mean(dim=1)
                elif h_q == num_kv_heads:
                    kv_s = q_sum
                else:
                    kv_s = q_sum.mean(dim=0, keepdim=True).expand(num_kv_heads, ctx_len).contiguous()
                per_layer_scores.append(kv_s)
            else:
                acc_1d = acc_1d + obs_w.sum(dim=(0, 1))

        if query_embeds is not None:
            ctx_pairs = [(k[:, :, :ctx_len, :].clone(), v[:, :, :ctx_len, :].clone()) for k, v in kv_pairs]
            ctx_pkv = _rebuild_kv_cache(pkv, ctx_pairs)
        else:
            ctx_pkv = pkv
        del out, attns
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _sync()
        scores_out = torch.stack(per_layer_scores, dim=0) if per_head else acc_1d
        return ctx_pkv, scores_out, (time.perf_counter() - t0) * 1000.0

    # Memory-safe exact SDPA prefill + layer-by-layer Q_obs @ K^T (18 GB vs 50.4 GB OOM on 8B!)
    prev_use_cache = getattr(llm.config, "use_cache", False) if hasattr(llm, "config") else False
    if hasattr(llm, "config"):
        llm.config.use_cache = True
    try:
        out = llm(
            inputs_embeds=full_embeds,
            attention_mask=attn_mask,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=True,
        )
    finally:
        if hasattr(llm, "config"):
            llm.config.use_cache = prev_use_cache

    pkv = out.past_key_values
    hs = out.hidden_states
    kv_pairs = _extract_kv_pairs(pkv)
    num_q_heads = int(getattr(llm.config, "num_attention_heads", 32))
    num_kv_heads = int(getattr(llm.config, "num_key_value_heads", kv_pairs[0][0].shape[1]))
    head_dim = int(getattr(llm.config, "head_dim", int(llm.config.hidden_size) // num_q_heads))
    gqa_rep = max(1, num_q_heads // max(1, num_kv_heads))

    obs_pos_ids = torch.arange(obs_start, obs_end, device=device, dtype=torch.long).unsqueeze(0)
    k_pos_ids = torch.arange(total_len, device=device, dtype=torch.long).unsqueeze(0)
    causal_mask = k_pos_ids > obs_pos_ids.transpose(0, 1)  # [obs_len, total_len]

    scale = head_dim ** -0.5
    per_layer_lh: list[torch.Tensor] = []
    acc_1d = torch.zeros((ctx_len,), device=device, dtype=torch.float32)

    for l_idx, layer in enumerate(layers):
        h_in = hs[l_idx][:, obs_start:obs_end, :]
        h_norm = layer.input_layernorm(h_in)
        q_raw = layer.self_attn.q_proj(h_norm).view(bsz, obs_end - obs_start, num_q_heads, head_dim).transpose(1, 2)
        if hasattr(layer.self_attn, "q_norm") and layer.self_attn.q_norm is not None:
            q_raw = layer.self_attn.q_norm(q_raw)
        cos, sin = rotary_emb(q_raw, obs_pos_ids)
        q_rot = _apply_rotary_half_m3(q_raw, cos.to(q_raw.dtype), sin.to(q_raw.dtype))

        k_full = kv_pairs[l_idx][0]  # [1, num_kv_heads, total_len, head_dim]
        k_exp = k_full.repeat_interleave(gqa_rep, dim=1) if gqa_rep > 1 else k_full

        scores = torch.matmul(q_rot.float(), k_exp.float().transpose(-2, -1)) * scale
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_w = F.softmax(scores, dim=-1)  # [1, H_q, obs_len, total_len]
        ctx_w = attn_w[0, :, :, :ctx_len]  # [H_q, obs_len, ctx_len]
        if per_head:
            q_sum = ctx_w.sum(dim=1)  # [H_q, ctx_len]
            if gqa_rep > 1:
                kv_s = q_sum.view(num_kv_heads, gqa_rep, ctx_len).mean(dim=1)
            else:
                kv_s = q_sum
            per_layer_lh.append(kv_s)
        else:
            acc_1d = acc_1d + ctx_w.sum(dim=(0, 1))

    if query_embeds is not None:
        ctx_pairs = [(k[:, :, :ctx_len, :].clone(), v[:, :, :ctx_len, :].clone()) for k, v in kv_pairs]
        ctx_pkv = _rebuild_kv_cache(pkv, ctx_pairs)
    else:
        ctx_pkv = pkv

    del out, hs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _sync()
    scores_out = torch.stack(per_layer_lh, dim=0) if per_head else acc_1d
    return ctx_pkv, scores_out, (time.perf_counter() - t0) * 1000.0


def evict_past_key_values_h2o_attention(
    past_key_values,
    attn_scores: torch.Tensor,
    *,
    keep_budget: int | None = None,
    budget_ratio: float = 0.20,
    sink_tokens: int = 4,
    recent_ratio: float = 0.10,
    pool_kernel_size: int = 0,
    pool_mode: str = "avg",
) -> tuple[object, torch.Tensor]:
    """True attention-based H2O / per-head 1D-pooled SnapKV eviction directly on `past_key_values`.

    - Respects `recent_ratio` faithfully (NO `max(recent_ratio, 0.25)` override!).
    - Accepts EITHER 1D `attn_scores [ctx_len]` (shared H2O indices) OR 3D `attn_scores [L, num_kv_heads, ctx_len]`
      (true per-layer, per-KV-head GQA SnapKV selection!).
    - When `pool_kernel_size > 1` (e.g. `5` for official SnapKV), applies 1D avg/max pooling along `ctx_len`
      before per-head `topk` so multi-token numeric/entity clusters stay contiguous.
    """
    pairs = _extract_kv_pairs(past_key_values)
    num_layers = len(pairs)
    bsz, num_kv_heads, seq_len, head_dim = pairs[0][0].shape
    device = pairs[0][0].device

    if keep_budget is None:
        budget = max(1, int(round(seq_len * budget_ratio)))
    else:
        budget = max(1, min(seq_len, int(keep_budget)))

    if budget >= seq_len:
        if attn_scores.dim() == 3:
            keep_all = (
                torch.arange(seq_len, device=device, dtype=torch.long)
                .view(1, 1, seq_len)
                .expand(num_layers, num_kv_heads, seq_len)
                .contiguous()
            )
            return past_key_values, keep_all
        return past_key_values, torch.arange(seq_len, device=device, dtype=torch.long)

    sinks = min(max(0, int(sink_tokens)), max(0, budget // 4))
    # Honor caller's exact recent_ratio (no hardcoded 0.25 floor!)
    recent = min(max(0, int(round(budget * float(recent_ratio)))), max(0, budget - sinks))
    hh_budget = max(0, budget - sinks - recent)
    recent_start = max(sinks, seq_len - recent)

    def _pool_1d(x: torch.Tensor) -> torch.Tensor:
        if pool_kernel_size is None or int(pool_kernel_size) <= 1:
            return x.float()
        ks = int(pool_kernel_size)
        pad_l = ks // 2
        pad_r = ks - 1 - pad_l
        xf = x.float()
        orig_shape = xf.shape
        flat = xf.reshape(-1, 1, orig_shape[-1])
        flat_pad = F.pad(flat, (pad_l, pad_r), mode="replicate")
        if pool_mode == "max":
            pooled = F.max_pool1d(flat_pad, kernel_size=ks, stride=1)
        else:
            pooled = F.avg_pool1d(flat_pad, kernel_size=ks, stride=1)
        return pooled.reshape(orig_shape)

    # Path A: True per-layer, per-KV-head (GQA) selection [L, H_kv, ctx_len]
    if attn_scores.dim() == 3:
        pooled_lh = _pool_1d(attn_scores.to(device))  # [L, H_kv, seq_len]
        sink_idx = torch.arange(sinks, device=device, dtype=torch.long)
        rec_idx = torch.arange(recent_start, seq_len, device=device, dtype=torch.long)

        evicted_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        keep_lh = torch.empty((num_layers, num_kv_heads, budget), device=device, dtype=torch.long)

        for l_idx, (k_t, v_t) in enumerate(pairs):
            layer_idx_list: list[torch.Tensor] = []
            for h_idx in range(num_kv_heads):
                parts: list[torch.Tensor] = []
                if sinks > 0:
                    parts.append(sink_idx)
                if hh_budget > 0 and recent_start > sinks:
                    mid = pooled_lh[l_idx, h_idx, sinks:recent_start]
                    k_hh = min(hh_budget, int(mid.numel()))
                    top_rel = torch.topk(mid, k=k_hh, dim=0, largest=True).indices + sinks
                    parts.append(top_rel)
                if recent > 0:
                    parts.append(rec_idx)
                comb = torch.cat(parts, dim=0)
                if comb.numel() > budget:
                    comb = comb[:budget]
                sorted_idx, _ = torch.sort(comb, dim=0)
                layer_idx_list.append(sorted_idx)
            l_keep = torch.stack(layer_idx_list, dim=0)  # [H_kv, budget]
            keep_lh[l_idx] = l_keep
            gather_idx = l_keep.view(1, num_kv_heads, budget, 1).expand(bsz, num_kv_heads, budget, head_dim)
            k_ev = torch.gather(k_t, dim=2, index=gather_idx).clone()
            v_ev = torch.gather(v_t, dim=2, index=gather_idx).clone()
            evicted_pairs.append((k_ev, v_ev))

        return _rebuild_kv_cache(past_key_values, evicted_pairs), keep_lh

    # Path B: Legacy 1D shared eviction scores [ctx_len]
    pooled_1d = _pool_1d(attn_scores.to(device))
    keep_set: set[int] = set(range(sinks))
    keep_set.update(range(recent_start, seq_len))

    if hh_budget > 0 and recent_start > sinks:
        mid_scores = pooled_1d[sinks:recent_start].float()
        k_hh = min(hh_budget, int(mid_scores.numel()))
        top_rel = torch.topk(mid_scores, k=k_hh, dim=0, largest=True).indices
        for idx_r in top_rel.tolist():
            keep_set.add(sinks + int(idx_r))

    keep_idx = torch.tensor(sorted(keep_set)[:budget], device=device, dtype=torch.long)
    evicted_pairs = [
        (k.index_select(2, keep_idx).clone(), v.index_select(2, keep_idx).clone())
        for k, v in pairs
    ]
    return _rebuild_kv_cache(past_key_values, evicted_pairs), keep_idx


def _llm_accepts_position_ids(llm: nn.Module) -> bool:
    try:
        sig = inspect.signature(llm.forward)
    except (TypeError, ValueError):
        return False
    if "position_ids" in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


@torch.no_grad()
def compute_nll_and_tok_acc_from_kv(
    llm: nn.Module,
    past_key_values,
    kv_seq_len: int,
    q_ids: torch.Tensor,
    a_ids: torch.Tensor,
    orig_seq_len: int | None = None,
) -> tuple[float, float]:
    """Score answer NLL and teacher-forced token accuracy directly from a prepared past_key_values cache."""
    # Deep-copy KV cache so forward mutation (DynamicCache.update) never corrupts caller's cache
    pairs_copy = [(k.clone(), v.clone()) for k, v in _extract_kv_pairs(past_key_values)]
    kv_work = _rebuild_kv_cache(past_key_values, pairs_copy)
    turn_ids = torch.cat((q_ids, a_ids), dim=1)
    bsz, turn_len = turn_ids.shape
    attn_mask = torch.ones((bsz, kv_seq_len + turn_len), device=turn_ids.device, dtype=torch.long)
    start_pos = int(kv_seq_len if orig_seq_len is None else orig_seq_len)
    fwd_kwargs: dict[str, object] = {
        "input_ids": turn_ids,
        "attention_mask": attn_mask,
        "past_key_values": kv_work,
        "use_cache": True,
    }
    if _llm_accepts_position_ids(llm):
        fwd_kwargs["position_ids"] = (
            torch.arange(start_pos, start_pos + turn_len, device=turn_ids.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(bsz, -1)
        )
    out = llm(**fwd_kwargs)
    start = q_ids.shape[1] - 1
    ans_logits = out.logits[:, start : start + a_ids.shape[1], :].float()
    nll = float(F.cross_entropy(ans_logits.reshape(-1, ans_logits.shape[-1]), a_ids.reshape(-1)).item())
    tok_acc = float((ans_logits.argmax(dim=-1) == a_ids).float().mean().item())
    return nll, tok_acc


@torch.no_grad()
def greedy_generate_from_kv_cache(
    llm: nn.Module,
    past_key_values,
    kv_seq_len: int,
    q_ids: torch.Tensor,
    max_new_tokens: int = 48,
    eos_token_id: int | set[int] | list[int] | tuple[int, ...] | None = None,
    orig_seq_len: int | None = None,
    tokenizer=None,
    stop_on_newline: bool = False,
    return_timing: bool = False,
):
    """Autoregressive greedy decode starting directly from a quantized or H2O/SnapKV-evicted past_key_values cache."""
    if eos_token_id is None:
        eos_set: set[int] = set()
    elif isinstance(eos_token_id, (set, list, tuple)):
        eos_set = {int(x) for x in eos_token_id if x is not None}
    else:
        eos_set = {int(eos_token_id)}
    pairs_copy = [(k.clone(), v.clone()) for k, v in _extract_kv_pairs(past_key_values)]
    kv_work = _rebuild_kv_cache(past_key_values, pairs_copy)
    bsz, q_len = q_ids.shape
    device = q_ids.device

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    start_pos = int(kv_seq_len if orig_seq_len is None else orig_seq_len)
    use_pos_ids = _llm_accepts_position_ids(llm)
    attn_mask = torch.ones((bsz, kv_seq_len + q_len), device=device, dtype=torch.long)
    prefill_kwargs: dict[str, object] = {
        "input_ids": q_ids,
        "attention_mask": attn_mask,
        "past_key_values": kv_work,
        "use_cache": True,
    }
    if use_pos_ids:
        prefill_kwargs["position_ids"] = (
            torch.arange(start_pos, start_pos + q_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        )
    _sync()
    t_pf0 = time.perf_counter()
    out = llm(**prefill_kwargs)
    _sync()
    prefill_ms = (time.perf_counter() - t_pf0) * 1000.0

    pkv = getattr(out, "past_key_values", None)
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    gen: list[torch.Tensor] = []
    gen_ids_py: list[int] = []
    decode_steps = 0
    _sync()
    t_dec0 = time.perf_counter()
    for step_i in range(max_new_tokens):
        gen.append(next_tok)
        tok_i = int(next_tok[0, 0].item())
        gen_ids_py.append(tok_i)
        if eos_set and tok_i in eos_set:
            break
        if stop_on_newline and tokenizer is not None:
            dec = tokenizer.decode(gen_ids_py, skip_special_tokens=True)
            if "\n" in dec.lstrip():
                break
        attn_mask = torch.cat((attn_mask, torch.ones((bsz, 1), device=device, dtype=torch.long)), dim=1)
        dec_kwargs: dict[str, object] = {
            "input_ids": next_tok,
            "attention_mask": attn_mask,
            "past_key_values": pkv,
            "use_cache": True,
        }
        if use_pos_ids:
            dec_kwargs["position_ids"] = torch.full((bsz, 1), start_pos + q_len + step_i, device=device, dtype=torch.long)
        out = llm(**dec_kwargs)
        decode_steps += 1
        pkv = getattr(out, "past_key_values", None)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    _sync()
    decode_ms = (time.perf_counter() - t_dec0) * 1000.0
    out_tensor = torch.cat(gen, dim=1) if gen else torch.empty((bsz, 0), dtype=torch.long, device=device)
    if return_timing:
        n_tok = max(1, decode_steps if decode_steps > 0 else len(gen))
        return out_tensor, {
            "prefill_ms": round(prefill_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "num_gen_tokens": len(gen),
            "decode_ms_per_tok": round(decode_ms / n_tok, 3),
        }
    return out_tensor


@torch.no_grad()
def evaluate_baseline_0b_0d(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    ppl_eps,
    niah_ds: SyntheticExactSemanticNIAHDataset,
    device: torch.device,
    mode: str,  # "int4_kv" or "h2o_20pct"
) -> dict[str, float]:
    """Fixed Track 0B (true INT4 past_key_values) and Track 0D (true attention-based H2O/SnapKV)."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype

    def _prepare_kv(flat_ids: torch.Tensor) -> tuple[object, int, int]:
        ctx_emb = embed_fn(flat_ids).to(llm_dtype)
        seq_len = int(ctx_emb.shape[1])
        if mode == "int4_kv":
            attn_m = torch.ones((1, seq_len), device=device, dtype=torch.long)
            out_pre = wrapper.llm(inputs_embeds=ctx_emb, attention_mask=attn_m, use_cache=True)
            q_kv = quantize_past_key_values_int4(out_pre.past_key_values, group_size=64)
            return q_kv, seq_len, seq_len
        if mode == "h2o_20pct":
            pkv, attn_scores = compute_real_attention_scores_and_kv(wrapper.llm, ctx_emb, obs_window=32)
            ev_kv, keep_idx = evict_past_key_values_h2o_attention(
                pkv, attn_scores, budget_ratio=0.20, sink_tokens=4, recent_ratio=0.25
            )
            return ev_kv, int(keep_idx.numel()), seq_len
        raise ValueError(mode)

    nlls = []
    for ep in ppl_eps:
        pages = ep.context_pages.unsqueeze(0).to(device)
        q_ids = ep.query_ids.unsqueeze(0).to(device)
        a_ids = ep.answer_ids.unsqueeze(0).to(device)
        kv_cache, kv_len, orig_len = _prepare_kv(pages.reshape(1, -1))
        nll, _ = compute_nll_and_tok_acc_from_kv(wrapper.llm, kv_cache, kv_len, q_ids, a_ids, orig_seq_len=orig_len)
        nlls.append(nll)
    ppl = round(math.exp(min(20.0, sum(nlls) / max(1, len(nlls)))), 4)

    exact_em, exact_tok, sem_em, sem_tok = [], [], [], []
    for ep in niah_ds:
        pages = ep.context_pages.unsqueeze(0).to(device)
        q_ids = ep.query_ids.unsqueeze(0).to(device)
        a_ids = ep.answer_ids.unsqueeze(0).to(device)
        kv_cache, kv_len, orig_len = _prepare_kv(pages.reshape(1, -1))
        _, tok_acc = compute_nll_and_tok_acc_from_kv(wrapper.llm, kv_cache, kv_len, q_ids, a_ids, orig_seq_len=orig_len)
        gen_ids = greedy_generate_from_kv_cache(
            wrapper.llm, kv_cache, kv_len, q_ids, max_new_tokens=48, eos_token_id=tokenizer.eos_token_id, orig_seq_len=orig_len
        )
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        em = 1.0 if ep.metadata["answer_str"].lower() in gen_text.lower() else 0.0
        if ep.metadata["needle_type"] == "exact":
            exact_em.append(em)
            exact_tok.append(tok_acc)
        else:
            sem_em.append(em)
            sem_tok.append(tok_acc)

    mib = 1024.0 * 1024.0
    kv_4k = estimate_int4_kv_bytes(wrapper.cfg, 4096) / mib if mode == "int4_kv" else (4096 * 0.20 * wrapper.cfg.kv_bytes_per_token_fp16) / mib
    kv_32k = estimate_int4_kv_bytes(wrapper.cfg, 32768) / mib if mode == "int4_kv" else (32768 * 0.20 * wrapper.cfg.kv_bytes_per_token_fp16) / mib
    return {
        "ppl": ppl,
        "exact_niah_em": round(sum(exact_em) / max(1, len(exact_em)), 4),
        "exact_niah_tok_acc": round(sum(exact_tok) / max(1, len(exact_tok)), 4),
        "semantic_niah_em": round(sum(sem_em) / max(1, len(sem_em)), 4),
        "semantic_niah_tok_acc": round(sum(sem_tok) / max(1, len(sem_tok)), 4),
        "vram_4k_mib": round(kv_4k, 2),
        "vram_32k_mib": round(kv_32k, 2),
    }


class Flat1DLatentBaseline(nn.Module):
    """Track 0E Ablation: 1D Flat Latent Cache (24 tokens/page, 12 KiB/page, NO [L,H,W] layers, NO SparseExactBank)."""

    def __init__(self, cfg: LSLCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.slots_per_page = 24  # Match LSLC Deep Read 24 dense tokens/page
        self.c_flat = 256  # 24 * 256 * 2B = 12,288 B/page (identical 12 KiB/page byte budget!)
        self.down = nn.Sequential(nn.LayerNorm(cfg.d_llm), nn.Linear(cfg.d_llm, self.c_flat))
        self.up = nn.Sequential(nn.LayerNorm(self.c_flat), nn.Linear(self.c_flat, cfg.d_llm), nn.GELU(), nn.Linear(cfg.d_llm, cfg.d_llm))
        self.edit_mlp = nn.Sequential(nn.Linear(self.c_flat * 2, self.c_flat), nn.Tanh(), nn.Linear(self.c_flat, self.c_flat))

    def encode_flat(self, pages_embeds: torch.Tensor) -> torch.Tensor:
        bsz, p, tp, d = pages_embeds.shape
        h = self.down(pages_embeds.reshape(bsz * p, tp, d)).transpose(1, 2)
        pooled = F.adaptive_avg_pool1d(h, output_size=self.slots_per_page).transpose(1, 2)
        return pooled.view(bsz, p, self.slots_per_page, self.c_flat)

    def project_flat(self, z_flat: torch.Tensor) -> torch.Tensor:
        bsz, p, s, c = z_flat.shape
        return self.up(z_flat).view(bsz, p * s, self.cfg.d_llm)

    def apply_flat_edit(self, z_flat: torch.Tensor, page_idx: int, edit_embeds: torch.Tensor) -> torch.Tensor:
        e_lat = F.adaptive_avg_pool1d(self.down(edit_embeds).transpose(1, 2), output_size=self.slots_per_page).transpose(1, 2)
        z_new = z_flat.clone()
        z_new[:, page_idx] = z_flat[:, page_idx] + 0.5 * self.edit_mlp(torch.cat((z_flat[:, page_idx], e_lat), dim=-1))
        return z_new


def run_track0e_flat1d(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    train_eps: list,
    ppl_eps,
    niah_ds: SyntheticExactSemanticNIAHDataset,
    steps: int,
    lr: float,
    device: torch.device,
) -> dict[str, float]:
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    amp_dtype, _ = select_amp_dtype(device)
    flat_mod = Flat1DLatentBaseline(wrapper.cfg).to(device)
    opt = torch.optim.AdamW(flat_mod.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(77)

    flat_mod.train()
    for _ in range(steps):
        ep = train_eps[rng.randrange(len(train_eps))]
        pages = ep.context_pages.unsqueeze(0).to(device)
        q_ids = ep.query_ids.unsqueeze(0).to(device)
        a_ids = ep.answer_ids.unsqueeze(0).to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            z_f = flat_mod.encode_flat(embed_fn(pages).float())
            prefix = flat_mod.project_flat(z_f)
            turn_ids = torch.cat((q_ids, a_ids), dim=1)
            q_ign = torch.full_like(q_ids, -100)
            turn_labels = torch.cat((q_ign, a_ids), dim=1)
            loss = wrapper._run_llm_with_prefix(prefix, turn_ids, turn_labels=turn_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flat_mod.parameters(), 1.0)
        opt.step()

    flat_mod.eval()
    nlls, exact_em, exact_tok, sem_em, sem_tok = [], [], [], [], []
    with torch.no_grad():
        for ep in ppl_eps:
            pages = ep.context_pages.unsqueeze(0).to(device)
            q_ids = ep.query_ids.unsqueeze(0).to(device)
            a_ids = ep.answer_ids.unsqueeze(0).to(device)
            z_f = flat_mod.encode_flat(embed_fn(pages).float())
            prefix_llm = flat_mod.project_flat(z_f).to(llm_dtype)
            turn_ids = torch.cat((q_ids, a_ids), dim=1)
            inp = torch.cat((prefix_llm, embed_fn(turn_ids).to(llm_dtype)), dim=1)
            attn = torch.ones((1, inp.shape[1]), device=device, dtype=torch.long)
            logits = wrapper.llm(inputs_embeds=inp, attention_mask=attn).logits
            start = prefix_llm.shape[1] + q_ids.shape[1] - 1
            ans_logits = logits[:, start : start + a_ids.shape[1], :]
            nlls.append(float(F.cross_entropy(ans_logits.reshape(-1, ans_logits.shape[-1]), a_ids.reshape(-1)).item()))

        for ep in niah_ds:
            pages = ep.context_pages.unsqueeze(0).to(device)
            q_ids = ep.query_ids.unsqueeze(0).to(device)
            a_ids = ep.answer_ids.unsqueeze(0).to(device)
            z_f = flat_mod.encode_flat(embed_fn(pages).float())
            prefix_llm = flat_mod.project_flat(z_f).to(llm_dtype)
            turn_ids = torch.cat((q_ids, a_ids), dim=1)
            inp = torch.cat((prefix_llm, embed_fn(turn_ids).to(llm_dtype)), dim=1)
            attn = torch.ones((1, inp.shape[1]), device=device, dtype=torch.long)
            logits = wrapper.llm(inputs_embeds=inp, attention_mask=attn).logits
            start = prefix_llm.shape[1] + q_ids.shape[1] - 1
            ans_logits = logits[:, start : start + a_ids.shape[1], :]
            tok_acc = float((ans_logits.argmax(dim=-1) == a_ids).float().mean().item())
            gen_ids = greedy_generate_from_prefix(
                wrapper.llm, prefix_llm, q_ids, max_new_tokens=48, eos_token_id=tokenizer.eos_token_id
            )
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            em = 1.0 if ep.metadata["answer_str"].lower() in gen_text.lower() else 0.0
            if ep.metadata["needle_type"] == "exact":
                exact_em.append(em)
                exact_tok.append(tok_acc)
            else:
                sem_em.append(em)
                sem_tok.append(tok_acc)

        # Measure prefix invariance violation on 1D flat latent when page 0 is edited
        sample_pages = niah_ds[0].context_pages.unsqueeze(0).to(device)
        z_before = flat_mod.encode_flat(embed_fn(sample_pages).float())
        pref_before = flat_mod.project_flat(z_before)
        z_after = flat_mod.apply_flat_edit(z_before, 0, embed_fn(niah_ds[0].query_ids.unsqueeze(0).to(device)).float())
        pref_after = flat_mod.project_flat(z_after)
        flat_prefix_diff = float((pref_after[:, :24] - pref_before[:, :24]).abs().max().item())

    return {
        "ppl": round(math.exp(min(20.0, sum(nlls) / len(nlls))), 4),
        "exact_niah_em": round(sum(exact_em) / max(1, len(exact_em)), 4),
        "exact_niah_tok_acc": round(sum(exact_tok) / max(1, len(exact_tok)), 4),
        "semantic_niah_em": round(sum(sem_em) / max(1, len(sem_em)), 4),
        "semantic_niah_tok_acc": round(sum(sem_tok) / max(1, len(sem_tok)), 4),
        "edited_page_prefix_invariance_diff": round(flat_prefix_diff, 5),
    }


@torch.no_grad()
def evaluate_multipage_4k_book(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
) -> dict[str, float]:
    """Track 3: Evaluate 16-page (4,096-token) multi-page book routing & Exact/Semantic NIAH."""
    ds_4k = SyntheticExactSemanticNIAHDataset(
        num_episodes=20,
        num_pages=16,
        page_size=wrapper.cfg.page_size,
        tokenizer=tokenizer,
        seed=8888,
        background_token_pool=val_pool,
    )
    niah_res = evaluate_niah_suite(wrapper, tokenizer, ds_4k, device, "lslc_full_exact")

    # Also measure Router top-k_deep recall across P=16 pages
    embed_fn = wrapper.llm.get_input_embeddings()
    amp_dtype, _ = select_amp_dtype(device)
    top1_hits, topk_hits = 0, 0
    for ep in ds_4k:
        pages = ep.context_pages.unsqueeze(0).to(device)
        q_ids = ep.query_ids.unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            z_book, _, toc = wrapper.lslc.encode_book(pages, embed_fn, use_exact_bank=False)
            q_emb = embed_fn(q_ids).mean(dim=1)
            r_out = wrapper.lslc.router(q_emb, toc, k_deep=2, k_skim=2)
            top1 = int(r_out.logits.argmax(dim=-1).item())
            deep_set = set(r_out.deep_idx[0].tolist())
            if top1 == ep.target_page_idx:
                top1_hits += 1
            if ep.target_page_idx in deep_set:
                topk_hits += 1

    return {
        "num_pages": 16,
        "context_tokens": 4096,
        "router_top1_acc": round(top1_hits / len(ds_4k), 4),
        "router_top2_deep_recall": round(topk_hits / len(ds_4k), 4),
        **niah_res,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"=== LSLC M3 FULL SUITE START (model={args.model}) ===", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for colab_run_m3.py")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM: {props.total_memory / (1024**3):.2f} GiB | BF16={torch.cuda.is_bf16_supported()}", flush=True)

    # 1. Unit Test Gate
    suite = unittest.defaultTestLoader.discover(str(REPO_ROOT / "tests"))
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    if not res.wasSuccessful():
        raise RuntimeError("Unit tests failed on Colab!")

    # 2. Load Frozen Qwen3 + WikiText-2
    device = torch.device("cuda")
    cfg = LSLCConfig.from_model_name(args.model, lr=args.lr)
    llm, tokenizer = load_frozen_qwen3(cfg, load_in_4bit=False, device=device)
    wrapper = FrozenQwenLSLCWrapper(cfg, llm)
    wrapper.lslc.to(device)

    train_pool, val_pool, corpus_src = _load_wikitext_pools(tokenizer)
    train_ppl = ChunkedTextPageDataset(train_pool, num_pages=4, page_size=cfg.page_size, turn_len=64)
    eval_ppl = ChunkedTextPageDataset(val_pool, num_pages=4, page_size=cfg.page_size, turn_len=64)[:24]
    train_niah = SyntheticExactSemanticNIAHDataset(
        num_episodes=120, num_pages=4, page_size=cfg.page_size, tokenizer=tokenizer, seed=42, background_token_pool=train_pool
    )
    eval_niah = SyntheticExactSemanticNIAHDataset(
        num_episodes=30, num_pages=4, page_size=cfg.page_size, tokenizer=tokenizer, seed=9999, background_token_pool=val_pool
    )
    train_niah_16p = SyntheticExactSemanticNIAHDataset(
        num_episodes=60, num_pages=16, page_size=cfg.page_size, tokenizer=tokenizer, seed=4242, background_token_pool=train_pool
    )
    train_delta = LivingBookStateOverrideDataset(
        num_episodes=96, num_pages=4, page_size=cfg.page_size, num_edits=3, tokenizer=tokenizer, seed=42, background_token_pool=train_pool
    )
    eval_delta = LivingBookStateOverrideDataset(
        num_episodes=24, num_pages=4, page_size=cfg.page_size, num_edits=3, tokenizer=tokenizer, seed=9999, background_token_pool=val_pool
    )
    m_train_blend = list(train_ppl)[:60] + list(train_niah) + list(train_niah_16p) * 2

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    for cand_path in (out_path, Path(args.seed_m2_json)):
        if cand_path.exists():
            try:
                loaded = json.loads(cand_path.read_text(encoding="utf-8"))
                summary.update(loaded)
                print(f"[SEED] Reused keys from {cand_path.name}: {list(loaded.keys())}", flush=True)
                break
            except Exception:
                pass

    summary.update({
        "gpu": props.name,
        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
        "model": cfg.model_name,
        "trainable_params": wrapper.lslc.trainable_parameter_count(),
        "corpus": corpus_src,
    })

    def _flush() -> None:
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 3. Baselines Track 0A, 0B, 0C, 0D, 0E
    if "track0a_full_kv" not in summary or "track0c_streaming_128" not in summary:
        print("=== Evaluating Track 0A (Full KV) & Track 0C (StreamingLLM) ===", flush=True)
        summary["track0a_full_kv"] = {
            "ppl": evaluate_continuation_ppl(wrapper, eval_ppl, device, "full_kv"),
            **evaluate_niah_suite(wrapper, tokenizer, eval_niah, device, "full_kv"),
            "delta_oracle": evaluate_delta_suite(wrapper, tokenizer, eval_delta, device, mode="full_kv"),
        }
        summary["track0c_streaming_128"] = {
            "ppl": evaluate_continuation_ppl(wrapper, eval_ppl, device, "streaming_128"),
            **evaluate_niah_suite(wrapper, tokenizer, eval_niah, device, "streaming_128"),
        }
        _flush()

    if "track0b_int4_kv" not in summary or "track0d_h2o_20pct" not in summary:
        print("=== Evaluating Track 0B (INT4 Quantized KV) & Track 0D (H2O 20% Heavy-Hitter) ===", flush=True)
        summary["track0b_int4_kv"] = evaluate_baseline_0b_0d(wrapper, tokenizer, eval_ppl, eval_niah, device, "int4_kv")
        summary["track0d_h2o_20pct"] = evaluate_baseline_0b_0d(wrapper, tokenizer, eval_ppl, eval_niah, device, "h2o_20pct")
        _flush()
        print(f"Track 0B INT4 KV: {summary['track0b_int4_kv']}", flush=True)
        print(f"Track 0D H2O 20%: {summary['track0d_h2o_20pct']}", flush=True)

    if "track0e_flat_1d" not in summary:
        print("=== Evaluating Track 0E (1D Flat Latent Cache Ablation) ===", flush=True)
        summary["track0e_flat_1d"] = run_track0e_flat1d(
            wrapper, tokenizer, list(train_ppl)[:80] + list(train_niah), eval_ppl, eval_niah, steps=args.steps_flat1d, lr=args.lr, device=device
        )
        _flush()
        print(f"Track 0E Flat 1D: {summary['track0e_flat_1d']}", flush=True)

    # 4. Track 2 + Track 3: Full LSLC 3D + SparseExactBank (with 100% right_closer capture & score-ordered deep_idx)
    print("=== Track 2 & Track 3: Training Full LSLC (4-page & 16-page Multi-Page Living Book) ===", flush=True)
    wrapper_t2 = FrozenQwenLSLCWrapper(cfg, llm)
    wrapper_t2.lslc.to(device)
    t2_train = train_stage(
        "TRACK2_M3", wrapper_t2, m_train_blend, steps=args.steps_track2, grad_accum=args.grad_accum, lr=args.lr,
        device=device, use_exact_bank=True, distill_salience=True, run_matryoshka=True
    )
    t2_ppl = evaluate_continuation_ppl(wrapper_t2, eval_ppl, device, "lslc_full_exact")
    t2_niah = evaluate_niah_suite(wrapper_t2, tokenizer, eval_niah, device, "lslc_full_exact")
    matryoshka_zoom = {}
    for zmode in ("toc_only", "skim_only", "deep_only"):
        matryoshka_zoom[zmode] = {
            "ppl": evaluate_continuation_ppl(wrapper_t2, eval_ppl, device, zmode),
            **evaluate_niah_suite(wrapper_t2, tokenizer, eval_niah, device, zmode),
        }
    summary["track2_sparse_exact"] = {"train": t2_train, "ppl": t2_ppl, **t2_niah, "matryoshka_zoom": matryoshka_zoom}
    _flush()
    print(f"Track 2 Eval: PPL={t2_ppl} | {t2_niah}", flush=True)

    print("=== Track 3: Evaluating 16-Page (4,096-Token) Multi-Page Book Routing & NIAH ===", flush=True)
    summary["track3_multipage_4k"] = evaluate_multipage_4k_book(wrapper_t2, tokenizer, val_pool, device)
    _flush()
    print(f"Track 3 (16-Page 4K Book): {summary['track3_multipage_4k']}", flush=True)

    # 5. Track 4: Sticky-Note DeltaUpdater (Freeze projector so deep_l2_proj stays general & never overfits)
    print("=== Track 4: Training DeltaUpdater on Living-Book State-Override ===", flush=True)
    frozen_in_t4 = []
    for name_p, p in wrapper_t2.lslc.named_parameters():
        is_dynamic_t4 = name_p.startswith("delta_updater.") or name_p.startswith("router.")
        if not is_dynamic_t4 and p.requires_grad:
            p.requires_grad_(False)
            frozen_in_t4.append(p)

    t4_train = train_stage(
        "TRACK4_M3", wrapper_t2, list(train_delta), steps=args.steps_track4, grad_accum=args.grad_accum, lr=args.lr,
        device=device, use_exact_bank=True, distill_salience=False, run_matryoshka=False, use_edits=True
    )
    for p in frozen_in_t4:
        p.requires_grad_(True)
    t4_delta_eval = evaluate_delta_suite(wrapper_t2, tokenizer, eval_delta, device, mode="lslc")
    summary["track4_delta_probe"] = {"train": t4_train, **t4_delta_eval}
    _flush()
    print(f"Track 4 Delta Eval: {t4_delta_eval}", flush=True)

    # 6. Live CUDA 4K & 32K Profiler
    if "live_cuda_profile" not in summary:
        print("=== Live CUDA 4K & 32K VRAM / TTFT Profiler ===", flush=True)
        summary["live_cuda_profile"] = profile_live_4k_32k(wrapper_t2, device)
        _flush()

    # Mirror updated metrics back to seed_m2_json if present
    seed_path = Path(args.seed_m2_json)
    if seed_path.parent.exists():
        seed_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved full M3 summary -> {out_path}", flush=True)
    print("=== M3 RUN COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
