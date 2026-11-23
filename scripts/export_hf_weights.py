#!/usr/bin/env python3
"""Export, verify, and optionally upload LSLC adapter weights in Hugging Face SafeTensors format.

Usage:
  1. Export trained or initialized LSLC adapter weights locally / on Colab:
       python scripts/export_hf_weights.py --out-dir hf_hub_release
  2. Upload directly to Hugging Face Hub (requires `huggingface-cli login` or `HF_TOKEN`):
       python scripts/export_hf_weights.py --out-dir hf_hub_release --repo-id <username>/LSLC-Qwen3-1.7B-Adapter --push
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

try:
    from safetensors.torch import load_file, save_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lslc import LayeredSpatialLatentCache, LSLCConfig


def export_adapter_checkpoint(
    model_name: str = "Qwen/Qwen3-1.7B",
    out_dir: Path = REPO_ROOT / "hf_hub_release",
    state_dict: dict[str, torch.Tensor] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = LSLCConfig.from_model_name(model_name)
    lslc = LayeredSpatialLatentCache(cfg).to(dtype=dtype)
    if state_dict is not None:
        lslc.load_state_dict(state_dict, strict=False)

    clean_sd = {k: v.detach().cpu().contiguous().to(dtype) for k, v in lslc.state_dict().items()}
    metadata = {
        "format": "pt",
        "architecture": "LayeredSpatialLatentCache",
        "base_model": model_name,
        "page_size": str(cfg.page_size),
        "latent_shape": f"[{cfg.num_layers}, {cfg.latent_channels}, {cfg.grid_h}, {cfg.grid_w}]",
        "exact_tokens_per_page": str(cfg.k_exact),
        "trainable_parameters": str(lslc.trainable_parameter_count()),
    }
    if HAS_SAFETENSORS:
        st_path = out_dir / "lslc_qwen3_1.7b_adapter.safetensors"
        save_file(clean_sd, str(st_path), metadata=metadata)
        reloaded = load_file(str(st_path))
    else:
        st_path = out_dir / "lslc_qwen3_1.7b_adapter.pt"
        torch.save({"state_dict": clean_sd, "metadata": metadata}, st_path)
        reloaded = torch.load(st_path, map_location="cpu", weights_only=True)["state_dict"]

    assert len(reloaded) == len(clean_sd), "Checkpoint key count mismatch!"
    total_bytes = sum(t.numel() * t.element_size() for t in reloaded.values())
    print(
        f"[HF-EXPORT] Saved {len(reloaded)} tensors ({lslc.trainable_parameter_count():,} params, "
        f"{total_bytes / (1024**2):.2f} MiB) -> {st_path}",
        flush=True,
    )
    return st_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export and optionally push LSLC Hugging Face weights.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "hf_hub_release"))
    parser.add_argument("--from-pt", type=str, default="", help="Optional path to an existing .pt state_dict")
    parser.add_argument("--repo-id", type=str, default="", help="Hugging Face repo ID (e.g. username/LSLC-Qwen3-1.7B)")
    parser.add_argument("--push", action="store_true", help="Upload out-dir to Hugging Face Hub")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    sd = None
    if args.from_pt:
        sd = torch.load(args.from_pt, map_location="cpu", weights_only=True)
    st_path = export_adapter_checkpoint(args.model_name, out_dir, state_dict=sd)

    if args.push:
        if not args.repo_id:
            raise ValueError("--repo-id is required when --push is set.")
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Upload LSLC 3D spatial-layer book adapter weights + model card",
        )
        print(f"[HF-UPLOAD] Uploaded {out_dir} -> https://huggingface.co/{args.repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
