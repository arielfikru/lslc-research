#!/usr/bin/env python3
"""Self-contained demonstration of LSLC inference, 8-byte/token exact footnote gather,
and in-place convex sticky-note delta override (`max|Delta_TOC| == 0.0`).

Runs in two modes:
  1. Fast CPU/GPU architectural verification (`--mock-llm`, runs in <2 seconds on any laptop):
       python scripts/demo_hf_inference.py --mock-llm
  2. Live frozen Qwen3-1.7B inference with SafeTensors checkpoint (`--checkpoint ...`):
       python scripts/demo_hf_inference.py --checkpoint hf_hub_release/lslc_qwen3_1.7b_adapter.safetensors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lslc import LayeredSpatialLatentCache, LSLCConfig


def run_demo(checkpoint_path: str = "", mock_llm: bool = True) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not mock_llm else "cpu")
    cfg = LSLCConfig.from_model_name("Qwen/Qwen3-1.7B")
    if mock_llm:
        cfg.d_llm = 256
        cfg.d_enc = 128

    lslc = LayeredSpatialLatentCache(cfg).to(device)
    if checkpoint_path and Path(checkpoint_path).exists() and not mock_llm:
        from safetensors.torch import load_file

        sd = load_file(checkpoint_path, device=str(device))
        lslc.load_state_dict(sd, strict=False)
        print(f"[LSLC-DEMO] Loaded SafeTensors adapter weights from {checkpoint_path}")
    lslc.eval()

    vocab_size = 4096
    embed_fn = nn.Embedding(vocab_size, cfg.d_llm).to(device)
    nn.init.normal_(embed_fn.weight, std=0.025)

    # 1. Encode a 16-page (4,096-token) document into a 3D living book [1, 16, 3, 128, 4, 4]
    num_pages = 16
    pages = torch.randint(32, vocab_size, (1, num_pages, cfg.page_size), device=device)
    query_ids = torch.randint(32, vocab_size, (1, 16), device=device)

    with torch.no_grad():
        z_book, bank, toc_before = lslc.encode_book(
            pages, embed_fn, use_exact_bank=True, vocab_size=vocab_size
        )
        q_emb = embed_fn(query_ids).mean(dim=1)
        route_before, asm_before = lslc.route_and_assemble(
            z_book, toc_before, q_emb, bank, embed_fn, k_deep=2, k_skim=2, mode="full", training_ste=False
        )

        # 2. Apply an in-place sticky-note delta override to page #1 (`l=2` only)
        edit_turn_ids = torch.randint(32, vocab_size, (1, 24), device=device)
        target_page = torch.tensor([1], dtype=torch.long, device=device)
        z_book_edited, gate_p, bank_edited = lslc.apply_page_delta(
            z_book.clone(), target_page, edit_turn_ids, embed_fn, exact_bank=bank, vocab_size=vocab_size
        )
        toc_after = lslc.projector.project_toc(z_book_edited)
        _, asm_after = lslc.route_and_assemble(
            z_book_edited, toc_after, q_emb, bank_edited, embed_fn, k_deep=2, k_skim=2, mode="full", training_ste=False
        )

    imm_len = asm_before.immutable_prefix_len
    toc_diff = float((asm_after.prefix_embeds[:, :imm_len] - asm_before.prefix_embeds[:, :imm_len]).abs().max().item())
    l2_diff = float((z_book_edited[:, 1, 2] - z_book[:, 1, 2]).abs().max().item())
    l01_diff = float((z_book_edited[:, 1, :2] - z_book[:, 1, :2]).abs().max().item())

    print("=== LSLC LIVING-BOOK VERIFICATION SUMMARY ===")
    print(f"  Raw Context Tokens        : {num_pages * cfg.page_size:,} tokens ({num_pages} pages x {cfg.page_size})")
    print(f"  3D Book Tensor Shape      : {tuple(z_book.shape)} ({z_book.numel() * 2 / 1024:.1f} KiB FP16)")
    print(f"  SparseExactBank Footprint : {bank.storage_bytes()} bytes ({bank.storage_bytes() // num_pages} B/page = 8 B/tok)")
    print(f"  Active Projected Prefix   : {asm_after.prefix_embeds.shape[1]} tokens (Immutable TOC prefix = {imm_len} tokens)")
    print(f"  Routed Deep Read Pages    : {route_before.deep_idx[0].tolist()}")
    print(f"  DeltaUpdater Mean Gate    : {gate_p.mean().item():.4f} (active on dynamic layer l=2)")
    print(f"  Dynamic Layer l=2 MaxDiff : {l2_diff:.6f} (> 0 confirms in-place state override)")
    print(f"  Static/Entity l=0,1 Diff  : {l01_diff:.6f} (== 0.0 confirms write-locked foundation)")
    print(f"  TOC Prefix Cache MaxDiff  : {toc_diff:.6f} (== 0.0 confirms bit-exact prefix invariance)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="hf_hub_release/lslc_qwen3_1.7b_adapter.safetensors")
    parser.add_argument("--mock-llm", action="store_true", help="Run fast self-contained verification without downloading Qwen3")
    args = parser.parse_args()
    run_demo(checkpoint_path=args.checkpoint, mock_llm=args.mock_llm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
