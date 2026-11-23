"""Analytical & live CUDA VRAM/TTFT/decode profiler for LSLC vs. 1D KV baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lslc.config import LSLCConfig, compute_analytical_memory_stats
from lslc.trainer import LSLCSystem, load_frozen_qwen3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile LSLC vs KV-cache VRAM and latency.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[4096, 32768])
    parser.add_argument("--analytical-only", action="store_true", help="Print verified byte tables without loading GPU weights.")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output-json", type=str, default="", help="Optional JSON report output path.")
    args, _ = parser.parse_known_args(argv)
    return args


def _routing_counts_for_context(num_tokens: int) -> tuple[int, int]:
    # Matches RFC benchmarks: 4K -> (k_deep=2, k_skim=2); 32K -> (k_deep=2, k_skim=4)
    return (2, 4) if num_tokens >= 16384 else (2, 2)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    presets = [LSLCConfig.qwen3_1_7b(), LSLCConfig.qwen3_4b()]
    rows = []

    print("=== Analytical End-to-End Memory Breakdown (Book + Exact Bank + Active FP16 KV) ===")
    for cfg in presets:
        for ctx_len in args.context_lengths:
            kd, ks = _routing_counts_for_context(ctx_len)
            stats = compute_analytical_memory_stats(cfg, num_tokens=ctx_len, k_deep=kd, k_skim=ks, use_exact_bank=True)
            stats["model"] = cfg.model_name
            rows.append(stats)
            print(
                f"{cfg.model_name:<16} N={ctx_len:>5} (P={stats['pages']:>3}, act={stats['active_total_tokens']:>3}) | "
                f"FP16={stats['full_fp16_kv_mib']:>7.1f} MiB | INT4={stats['int4_kv_mib']:>6.1f} MiB | "
                f"H2O-20%={stats['h2o_20pct_mib']:>6.1f} MiB | LSLC Net={stats['lslc_net_mib']:>5.2f} MiB "
                f"({stats['ratio_vs_fp16']:.1f}x vs FP16, {stats['ratio_vs_int4']:.1f}x vs INT4)"
            )

    live_results = []
    if not args.analytical_only and torch.cuda.is_available():
        device = torch.device("cuda")
        cfg = LSLCConfig.from_model_name(args.model)
        llm, _ = load_frozen_qwen3(cfg, load_in_4bit=args.load_in_4bit, device=device)
        llm.eval()
        system = LSLCSystem(cfg).to(device)
        system.eval()
        embed_fn = llm.get_input_embeddings()

        for ctx_len in args.context_lengths:
            kd, ks = _routing_counts_for_context(ctx_len)
            pages = ctx_len // cfg.page_size
            dummy_pages = torch.randint(10, 1000, (1, pages, cfg.page_size), device=device)
            dummy_query = torch.randint(10, 1000, (1, 16), device=device)

            torch.cuda.reset_peak_memory_stats()
            t_enc0 = time.perf_counter()
            with torch.no_grad():
                z_book, bank, toc = system.encode_book(dummy_pages, embed_tokens_fn=embed_fn)
                q_emb = embed_fn(dummy_query).mean(dim=1)
                _, assembly = system.route_and_assemble(
                    z_book, toc, q_emb, bank, embed_tokens_fn=embed_fn, k_deep=kd, k_skim=ks, training_ste=False
                )
                turn_emb = embed_fn(dummy_query).to(assembly.prefix_embeds.dtype)
                inp = torch.cat((assembly.prefix_embeds, turn_emb), dim=1)
                out = llm(inputs_embeds=inp, use_cache=True)
            torch.cuda.synchronize()
            ttft_ms = (time.perf_counter() - t_enc0) * 1000.0
            peak_mib = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            live_results.append(
                {
                    "context_len": ctx_len,
                    "active_prefix_tokens": int(assembly.prefix_embeds.shape[1]),
                    "ttft_ms": round(ttft_ms, 2),
                    "peak_cuda_mib": round(peak_mib, 2),
                    "logits_shape": list(out.logits.shape),
                }
            )
            print(f"[LIVE CUDA] N={ctx_len}: prefix={assembly.prefix_embeds.shape[1]} tok, TTFT={ttft_ms:.2f} ms, Peak={peak_mib:.1f} MiB")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps({"analytical": rows, "live_cuda": live_results}, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
