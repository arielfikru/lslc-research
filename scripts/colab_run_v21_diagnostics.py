#!/usr/bin/env python3
"""LSLC v2.1.0 Diagnostics: Oracle A/B/C k=1..5 Symbolic Curve & P={4,128,1024} Edit Scaling.

Runs two fast targeted probes (~60-90s total on NVIDIA L4 BF16) using an existing unified checkpoint:
1. Probe 1 (Oracle-Page A/B/C Diagnostic on k=1..5 Symbolic Overrides):
   Isolates routing vs. compression/readout vs. DeltaUpdater cumulative drift at edit steps t=1..5
   by supplying the oracle target page (edit_page=1) to all three conditions:
     - Condition A: Verified up-to-date raw text pages [page 0 | patched page 1]
     - Condition B: Same verified up-to-date pages re-encoded from scratch by LSLC encoder
     - Condition C: Original book updated sequentially via DeltaUpdater (t=1..5)
2. Probe 2 (update_book_page Latency & Peak CUDA Allocation at P in {4, 128, 1024}):
   Quantifies the effect of z_book.clone() in autograd-safe update_book_page vs. zero-copy
   in-place slice assignment under inference mode across 1K, 32K, and 256K books.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safetensors.torch import load_file

from lslc import (
    FrozenQwenLSLCWrapper,
    LSLCConfig,
    RouterOutput,
    load_frozen_qwen3,
    select_amp_dtype,
)
from scripts.colab_run_m2 import _load_wikitext_pools
from scripts.colab_run_v2_complete import (
    SYMBOLIC_STATES,
    compute_answer_metrics_v2,
    greedy_generate_concise,
)


def _oracle_router_out(num_pages: int, target_page: int, other_page: int, device: torch.device) -> RouterOutput:
    """Construct oracle RouterOutput putting target_page last in deep_idx with max confidence."""
    logits = torch.full((1, num_pages), -10.0, dtype=torch.float32, device=device)
    logits[0, other_page] = 1.0
    logits[0, target_page] = 10.0
    probs = torch.zeros((1, num_pages), dtype=torch.float32, device=device)
    probs[0, other_page] = 0.01
    probs[0, target_page] = 0.99
    deep_idx = torch.tensor([[other_page, target_page]], dtype=torch.long, device=device)
    rem = [p for p in range(num_pages) if p not in (other_page, target_page)]
    skim_idx = torch.tensor([rem[:2]], dtype=torch.long, device=device)
    toc_idx = torch.tensor([rem[2:]], dtype=torch.long, device=device)
    return RouterOutput(logits=logits, probs=probs, deep_idx=deep_idx, skim_idx=skim_idx, toc_idx=toc_idx)


@torch.no_grad()
def run_oracle_abc_symbolic_k1_to_k5(
    wrapper: FrozenQwenLSLCWrapper,
    val_pool: list[int],
    tokenizer,
    device: torch.device,
    num_episodes: int = 30,
    num_pages: int = 4,
    page_size: int = 256,
    num_edits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare Condition A (Oracle Raw Text), B (Oracle Re-Encoded LSLC), and C (Sequential DeltaUpdater) at t=1..5."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    amp_dtype, _ = select_amp_dtype(device)
    vocab_size = len(embed_fn.weight)
    wrapper.lslc.eval()

    prev_cache = getattr(wrapper.llm.config, "use_cache", False)
    wrapper.llm.config.use_cache = True

    static_toks = tokenizer.encode("\nSTATIC POLICY : security audit requires dual signoff .\n", add_special_tokens=False)
    init_toks = tokenizer.encode("\nDEPLOYMENT STATE INITIAL : cold-standby .\n", add_special_tokens=False)
    q_text = "\nQuestion: What is the current updated deployment state? Answer:"
    q_ids = torch.tensor(tokenizer.encode(q_text, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
    oracle_route = _oracle_router_out(num_pages=num_pages, target_page=1, other_page=0, device=device)

    metrics = ("extracted_em", "strict_em", "substr_em", "tf_content_tok_acc")
    conds = ("cond_a_oracle_raw_text", "cond_b_oracle_reencoded_lslc", "cond_c_oracle_sequential_delta")
    per_step: dict[str, dict[str, dict[str, list[float]]]] = {
        str(t): {c: {m: [] for m in metrics} for c in conds} for t in range(1, num_edits + 1)
    }
    gates_per_step: dict[str, list[float]] = {str(t): [] for t in range(1, num_edits + 1)}

    def _score_prefix(prefix_llm: torch.Tensor, a_ids: torch.Tensor, ans_str: str) -> tuple[float, float, float, float]:
        turn_emb = embed_fn(torch.cat((q_ids, a_ids), dim=1)).to(llm_dtype)
        inp = torch.cat((prefix_llm, turn_emb), dim=1)
        logits = wrapper.llm(inputs_embeds=inp).logits
        start = prefix_llm.shape[1] + q_ids.shape[1] - 1
        ans_logits = logits[:, start : start + a_ids.shape[1], :]
        gen_ids, gen_text = greedy_generate_concise(wrapper.llm, prefix_llm, q_ids, tokenizer, max_new_tokens=24)
        st_em, ext_em, sub_em, ctok, _ = compute_answer_metrics_v2(
            ans_logits, a_ids, gen_ids, gen_text, ans_str, tokenizer
        )
        return ext_em, st_em, sub_em, ctok

    try:
        for ep_i in range(num_episodes):
            rng = random.Random(seed + ep_i * 509)
            base_pages: list[list[int]] = []
            for p in range(num_pages):
                start = rng.randint(0, max(1, len(val_pool) - page_size - 1))
                ptoks = list(val_pool[start : start + page_size])
                if p == 0:
                    ptoks[: len(static_toks)] = static_toks[:page_size]
                elif p == 1:
                    ptoks[: len(init_toks)] = init_toks[:page_size]
                base_pages.append(ptoks)

            chosen_states = rng.sample(SYMBOLIC_STATES, k=min(num_edits, len(SYMBOLIC_STATES)))
            orig_tensor = torch.tensor(base_pages, dtype=torch.long, device=device).unsqueeze(0)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                z_delta, bank_delta, _ = wrapper.lslc.encode_book(
                    orig_tensor, embed_fn, use_exact_bank=True, vocab_size=vocab_size
                )

            for step_idx, latest_state in enumerate(chosen_states, start=1):
                a_ids = torch.tensor(
                    tokenizer.encode(f" {latest_state}", add_special_tokens=False), dtype=torch.long, device=device
                ).unsqueeze(0)

                # Condition A: Oracle up-to-date raw text page 1 (patched cleanly in place)
                verified_pages = [list(pt) for pt in base_pages]
                patched_toks = tokenizer.encode(
                    f"\nDEPLOYMENT STATE INITIAL : {latest_state} .\n", add_special_tokens=False
                )
                verified_pages[1][: len(patched_toks)] = patched_toks[:page_size]
                raw_sel = torch.tensor([verified_pages[0] + verified_pages[1]], dtype=torch.long, device=device)
                pref_a = embed_fn(raw_sel).to(llm_dtype)
                res_a = _score_prefix(pref_a, a_ids, latest_state)

                # Condition B: Exact same verified up-to-date book re-encoded from scratch by LSLC encoder
                ver_tensor = torch.tensor(verified_pages, dtype=torch.long, device=device).unsqueeze(0)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    z_re, bank_re, toc_re = wrapper.lslc.encode_book(
                        ver_tensor, embed_fn, use_exact_bank=True, vocab_size=vocab_size
                    )
                    q_emb = embed_fn(q_ids).mean(dim=1)
                    _, asm_b = wrapper.lslc.route_and_assemble(
                        z_re,
                        toc_re,
                        q_emb,
                        bank_re,
                        embed_fn,
                        k_deep=2,
                        k_skim=2,
                        mode="full",
                        training_ste=False,
                        router_out_override=oracle_route,
                    )
                res_b = _score_prefix(asm_b.prefix_embeds.to(llm_dtype), a_ids, latest_state)

                # Condition C: Sequential DeltaUpdater on originally encoded book
                e_text = f"REVISION OVERRIDE : deployment state updated to {latest_state} ."
                e_ids = torch.tensor(tokenizer.encode(e_text, add_special_tokens=False), dtype=torch.long, device=device)
                p_t = torch.tensor([1], dtype=torch.long, device=device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    z_delta, gate_t, bank_delta = wrapper.lslc.apply_page_delta(
                        z_delta, p_t, e_ids.unsqueeze(0), embed_fn, exact_bank=bank_delta, vocab_size=vocab_size
                    )
                    toc_c = wrapper.lslc.projector.project_toc(z_delta)
                    _, asm_c = wrapper.lslc.route_and_assemble(
                        z_delta,
                        toc_c,
                        q_emb,
                        bank_delta,
                        embed_fn,
                        k_deep=2,
                        k_skim=2,
                        mode="full",
                        training_ste=False,
                        router_out_override=oracle_route,
                    )
                res_c = _score_prefix(asm_c.prefix_embeds.to(llm_dtype), a_ids, latest_state)

                t_key = str(step_idx)
                gates_per_step[t_key].append(float(gate_t.mean().item()))
                for cname, res_tuple in zip(conds, (res_a, res_b, res_c)):
                    for mname, val in zip(metrics, res_tuple):
                        per_step[t_key][cname][mname].append(val)

            if (ep_i + 1) % 10 == 0:
                print(f"  [DIAG ABC] {ep_i+1}/{num_episodes} episodes done", flush=True)
    finally:
        wrapper.llm.config.use_cache = prev_cache

    summary_steps: dict[str, Any] = {}
    for t_key in sorted(per_step.keys(), key=int):
        summary_steps[f"k={t_key}"] = {
            "mean_delta_gate": round(sum(gates_per_step[t_key]) / max(1, len(gates_per_step[t_key])), 4),
            **{
                cname: {
                    mname: round(sum(vals) / max(1, len(vals)), 4)
                    for mname, vals in per_step[t_key][cname].items()
                }
                for cname in conds
            },
        }
    return {
        "num_episodes": num_episodes,
        "seed": seed,
        "num_pages": num_pages,
        "steps": summary_steps,
    }


@torch.no_grad()
def run_edit_scaling_probe(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    device: torch.device,
    page_counts: tuple[int, ...] = (4, 128, 1024),
    trials: int = 10,
) -> dict[str, Any]:
    """Measure synchronized latency and peak CUDA allocation of update_book_page (clone) vs zero-copy slice write."""
    embed_fn = wrapper.llm.get_input_embeddings()
    amp_dtype, _ = select_amp_dtype(device)
    vocab_size = len(embed_fn.weight)
    wrapper.lslc.eval()

    cfg = wrapper.cfg
    e_ids = torch.tensor(
        tokenizer.encode("REVISION OVERRIDE : budget updated to 85M .", add_special_tokens=False),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    p_t = torch.tensor([1], dtype=torch.long, device=device)
    dummy_book_4 = torch.randint(100, 1000, (1, 4, cfg.page_size), device=device)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        z_4, bank_4, _ = wrapper.lslc.encode_book(dummy_book_4, embed_fn, use_exact_bank=True, vocab_size=vocab_size)
    page_template = z_4[:, :1].clone()

    out: dict[str, Any] = {}
    for p_cnt in page_counts:
        z_book_base = page_template.expand(1, p_cnt, -1, -1, -1, -1).contiguous()
        bank_base_ids = bank_4.token_ids[:, :1].expand(1, p_cnt, -1).contiguous()
        bank_base_h = bank_4.h_coords[:, :1].expand(1, p_cnt, -1).contiguous()
        bank_base_w = bank_4.w_coords[:, :1].expand(1, p_cnt, -1).contiguous()
        k_ex = bank_base_ids.shape[-1]
        bank_base_p = (
            torch.arange(p_cnt, device=device, dtype=torch.int16)
            .view(1, p_cnt, 1)
            .expand(1, p_cnt, k_ex)
            .contiguous()
        )

        from lslc.encoder import SparseExactBank

        bank_base = SparseExactBank(
            token_ids=bank_base_ids,
            page_ids=bank_base_p,
            h_coords=bank_base_h,
            w_coords=bank_base_w,
        )
        z_bytes = int(z_book_base.numel() * z_book_base.element_size())

        # Warmup
        for _ in range(3):
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _z, _g, _b = wrapper.lslc.apply_page_delta(
                    z_book_base, p_t, e_ids, embed_fn, exact_bank=bank_base, vocab_size=vocab_size
                )
        torch.cuda.synchronize()

        # 1. Current autograd-safe update_book_page (calls z_book.clone())
        clone_ms: list[float] = []
        clone_peak_deltas: list[int] = []
        for _ in range(trials):
            torch.cuda.reset_peak_memory_stats(device)
            base_alloc = torch.cuda.memory_allocated(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                z_out, _g, _b = wrapper.lslc.apply_page_delta(
                    z_book_base, p_t, e_ids, embed_fn, exact_bank=bank_base, vocab_size=vocab_size
                )
            torch.cuda.synchronize()
            clone_ms.append((time.perf_counter() - t0) * 1000.0)
            clone_peak_deltas.append(int(torch.cuda.max_memory_allocated(device) - base_alloc))
            del z_out

        # 2. Zero-copy inference slice write (no full-book z_book.clone())
        z_inplace = z_book_base.clone()
        inplace_ms: list[float] = []
        inplace_peak_deltas: list[int] = []
        for _ in range(trials):
            torch.cuda.reset_peak_memory_stats(device)
            base_alloc = torch.cuda.memory_allocated(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                edit_emb = embed_fn(e_ids)
                new_page, _g = wrapper.lslc.delta_updater(z_inplace[:, 1], edit_emb)
                z_inplace[:, 1] = new_page.to(z_inplace.dtype)
            torch.cuda.synchronize()
            inplace_ms.append((time.perf_counter() - t0) * 1000.0)
            inplace_peak_deltas.append(int(torch.cuda.max_memory_allocated(device) - base_alloc))

        out[f"P={p_cnt}"] = {
            "num_pages": p_cnt,
            "context_tokens": p_cnt * cfg.page_size,
            "z_book_mib": round(z_bytes / (1024 ** 2), 3),
            "autograd_safe_clone_ms": round(sum(clone_ms) / len(clone_ms), 3),
            "autograd_safe_clone_peak_delta_mib": round(max(clone_peak_deltas) / (1024 ** 2), 3),
            "inference_zero_copy_inplace_ms": round(sum(inplace_ms) / len(inplace_ms), 3),
            "inference_zero_copy_inplace_peak_delta_mib": round(max(inplace_peak_deltas) / (1024 ** 2), 3),
        }
        del z_inplace, z_book_base, bank_base
        torch.cuda.empty_cache()
        print(f"  [DIAG SCALE P={p_cnt}] {out[f'P={p_cnt}']}", flush=True)

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--load-checkpoint", type=str, required=True)
    parser.add_argument("--out-json", type=str, default="/content/lslc_results/v21_diagnostics.json")
    args = parser.parse_args()

    device = torch.device("cuda")
    cfg = LSLCConfig.from_model_name(args.model, lr=3e-4)
    llm, tokenizer = load_frozen_qwen3(cfg, load_in_4bit=False, device=device)
    wrapper = FrozenQwenLSLCWrapper(cfg, llm)
    wrapper.lslc.to(device)

    sd = load_file(args.load_checkpoint, device="cpu")
    wrapper.lslc.load_state_dict({k: v.to(device=device, dtype=torch.float32) for k, v in sd.items()}, strict=True)
    wrapper.lslc.eval()

    _, val_pool, _ = _load_wikitext_pools(tokenizer)

    print("=== LSLC V2.1.0 DIAGNOSTICS: EDIT SCALING (P=4,128,1024) + ORACLE A/B/C (k=1..5) ===", flush=True)
    scale_res = run_edit_scaling_probe(wrapper, tokenizer, device)
    abc_res = run_oracle_abc_symbolic_k1_to_k5(wrapper, val_pool, tokenizer, device)

    payload = {
        "version": "2.1.0-diagnostics",
        "model": args.model,
        "checkpoint": args.load_checkpoint,
        "gpu": torch.cuda.get_device_properties(0).name,
        "oracle_abc_k5_symbolic": abc_res,
        "edit_scaling_probe": scale_res,
    }
    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"=== SAVED DIAGNOSTICS -> {out_p} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
