"""Milestone 4 Colab L4 Orchestrator: Scaling Validation on Qwen/Qwen3-4B (144 KiB/tok FP16 KV -> 143.2x compression)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest
import torch

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
    load_frozen_qwen3,
)
from scripts.colab_run_m2 import (
    _load_wikitext_pools,
    evaluate_continuation_ppl,
    evaluate_delta_suite,
    evaluate_niah_suite,
    profile_live_4k_32k,
    train_stage,
)
from scripts.colab_run_m3 import evaluate_baseline_0b_0d, evaluate_multipage_4k_book


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Milestone 4 Qwen/Qwen3-4B scaling validation on Colab L4.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--steps-track1", type=int, default=200)
    parser.add_argument("--steps-track2", type=int, default=250)
    parser.add_argument("--steps-track4", type=int, default=150)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out-json", type=str, default="/content/lslc_results/m4_qwen3_4b_summary.json")
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"=== LSLC M4 SCALING VALIDATION START (model={args.model}) ===", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for colab_run_m4.py")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM: {props.total_memory / (1024**3):.2f} GiB | BF16={torch.cuda.is_bf16_supported()}", flush=True)

    # 1. Unit Test Gate
    suite = unittest.defaultTestLoader.discover(str(REPO_ROOT / "tests"))
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    if not res.wasSuccessful():
        raise RuntimeError("Unit tests failed on Colab!")

    # 2. Load Frozen Qwen/Qwen3-4B (BF16 on L4 22.5 GiB) + WikiText-2
    device = torch.device("cuda")
    cfg = LSLCConfig.from_model_name(args.model, lr=args.lr)
    llm, tokenizer = load_frozen_qwen3(cfg, load_in_4bit=False, device=device)
    wrapper = FrozenQwenLSLCWrapper(cfg, llm)
    wrapper.lslc.to(device)
    print(
        f"Loaded {cfg.model_name} (d_llm={cfg.d_llm}, layers={cfg.num_llm_layers}, "
        f"KV={cfg.kv_bytes_per_token_fp16 // 1024} KiB/tok, trainable LSLC params={wrapper.lslc.trainable_parameter_count():,})",
        flush=True,
    )

    train_pool, val_pool, corpus_src = _load_wikitext_pools(tokenizer)
    train_ppl = ChunkedTextPageDataset(train_pool, num_pages=4, page_size=cfg.page_size, turn_len=64)
    eval_ppl = ChunkedTextPageDataset(val_pool, num_pages=4, page_size=cfg.page_size, turn_len=64)[:24]
    train_niah = SyntheticExactSemanticNIAHDataset(
        num_episodes=120, num_pages=4, page_size=cfg.page_size, tokenizer=tokenizer, seed=42, background_token_pool=train_pool
    )
    eval_niah = SyntheticExactSemanticNIAHDataset(
        num_episodes=30, num_pages=4, page_size=cfg.page_size, tokenizer=tokenizer, seed=9999, background_token_pool=val_pool
    )
    train_delta = LivingBookStateOverrideDataset(
        num_episodes=96, num_pages=4, page_size=cfg.page_size, num_edits=3, tokenizer=tokenizer, seed=42, background_token_pool=train_pool
    )
    eval_delta = LivingBookStateOverrideDataset(
        num_episodes=24, num_pages=4, page_size=cfg.page_size, num_edits=3, tokenizer=tokenizer, seed=9999, background_token_pool=val_pool
    )
    m_train_blend = list(train_ppl)[:60] + list(train_niah)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "gpu": props.name,
        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
        "model": cfg.model_name,
        "d_llm": cfg.d_llm,
        "num_llm_layers": cfg.num_llm_layers,
        "kv_kib_per_token": cfg.kv_bytes_per_token_fp16 / 1024.0,
        "trainable_params": wrapper.lslc.trainable_parameter_count(),
        "corpus": corpus_src,
    }

    def _flush() -> None:
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 3. Baselines Track 0A, 0B, 0C, 0D on Qwen3-4B
    print("=== Evaluating Qwen3-4B Baselines (Tracks 0A, 0B, 0C, 0D) ===", flush=True)
    summary["track0a_full_kv"] = {
        "ppl": evaluate_continuation_ppl(wrapper, eval_ppl, device, "full_kv"),
        **evaluate_niah_suite(wrapper, tokenizer, eval_niah, device, "full_kv"),
        "delta_oracle": evaluate_delta_suite(wrapper, tokenizer, eval_delta, device, mode="full_kv"),
    }
    summary["track0b_int4_kv"] = evaluate_baseline_0b_0d(wrapper, tokenizer, eval_ppl, eval_niah, device, "int4_kv")
    summary["track0c_streaming_128"] = {
        "ppl": evaluate_continuation_ppl(wrapper, eval_ppl, device, "streaming_128"),
        **evaluate_niah_suite(wrapper, tokenizer, eval_niah, device, "streaming_128"),
    }
    summary["track0d_h2o_20pct"] = evaluate_baseline_0b_0d(wrapper, tokenizer, eval_ppl, eval_niah, device, "h2o_20pct")
    _flush()
    print(f"Qwen3-4B Track 0A Full KV: {summary['track0a_full_kv']}", flush=True)
    print(f"Qwen3-4B Track 0B INT4 KV: {summary['track0b_int4_kv']}", flush=True)
    print(f"Qwen3-4B Track 0C Stream128: {summary['track0c_streaming_128']}", flush=True)
    print(f"Qwen3-4B Track 0D H2O 20%: {summary['track0d_h2o_20pct']}", flush=True)

    # 4. Track 1: Dense Only on Qwen3-4B
    print("=== Qwen3-4B Track 1: Training LSLC Dense Only ===", flush=True)
    wrapper_t1 = FrozenQwenLSLCWrapper(cfg, llm)
    wrapper_t1.lslc.to(device)
    t1_train = train_stage(
        "M4_TRACK1", wrapper_t1, m_train_blend, steps=args.steps_track1, grad_accum=args.grad_accum, lr=args.lr,
        device=device, use_exact_bank=False, distill_salience=False, run_matryoshka=True
    )
    t1_ppl = evaluate_continuation_ppl(wrapper_t1, eval_ppl, device, "lslc_dense")
    t1_niah = evaluate_niah_suite(wrapper_t1, tokenizer, eval_niah, device, "lslc_dense")
    summary["track1_dense_only"] = {"train": t1_train, "ppl": t1_ppl, **t1_niah}
    _flush()
    print(f"Qwen3-4B Track 1 Eval: PPL={t1_ppl} | {t1_niah}", flush=True)
    del wrapper_t1
    torch.cuda.empty_cache()

    # 5. Track 2 & Track 3: Full LSLC + SparseExactBank on Qwen3-4B
    print("=== Qwen3-4B Track 2 & Track 3: Training Full LSLC + SparseExactBank ===", flush=True)
    wrapper_t2 = FrozenQwenLSLCWrapper(cfg, llm)
    wrapper_t2.lslc.to(device)
    t2_train = train_stage(
        "M4_TRACK2", wrapper_t2, m_train_blend, steps=args.steps_track2, grad_accum=args.grad_accum, lr=args.lr,
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
    print(f"Qwen3-4B Track 2 Eval: PPL={t2_ppl} | {t2_niah}", flush=True)

    print("=== Qwen3-4B Track 3: 16-Page (4K) Multi-Page Book ===", flush=True)
    summary["track3_multipage_4k"] = evaluate_multipage_4k_book(wrapper_t2, tokenizer, val_pool, device)
    _flush()
    print(f"Qwen3-4B Track 3 (16-Page 4K): {summary['track3_multipage_4k']}", flush=True)

    # 6. Track 4: Sticky-Note DeltaUpdater on Qwen3-4B
    print("=== Qwen3-4B Track 4: Sticky-Note DeltaUpdater ===", flush=True)
    frozen_in_t4 = []
    for name_p, p in wrapper_t2.lslc.named_parameters():
        is_dynamic_t4 = name_p.startswith("delta_updater.") or name_p.startswith("router.")
        if not is_dynamic_t4 and p.requires_grad:
            p.requires_grad_(False)
            frozen_in_t4.append(p)

    t4_train = train_stage(
        "M4_TRACK4", wrapper_t2, list(train_delta), steps=args.steps_track4, grad_accum=args.grad_accum, lr=args.lr,
        device=device, use_exact_bank=True, distill_salience=False, run_matryoshka=False, use_edits=True
    )
    for p in frozen_in_t4:
        p.requires_grad_(True)
    t4_delta_eval = evaluate_delta_suite(wrapper_t2, tokenizer, eval_delta, device, mode="lslc")
    summary["track4_delta_probe"] = {"train": t4_train, **t4_delta_eval}
    _flush()
    print(f"Qwen3-4B Track 4 Eval: {t4_delta_eval}", flush=True)

    # 7. Live CUDA 4K & 32K Profiler on Qwen3-4B
    print("=== Qwen3-4B Live CUDA 4K & 32K VRAM / TTFT Profiler ===", flush=True)
    summary["live_cuda_profile"] = profile_live_4k_32k(wrapper_t2, device)
    _flush()
    print(f"Saved full M4 Qwen3-4B summary -> {out_path}", flush=True)
    print("=== M4 RUN COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
