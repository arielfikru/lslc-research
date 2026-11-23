---
license: mit
library_name: pytorch
base_model: Qwen/Qwen3-1.7B
tags:
  - kv-cache
  - context-compression
  - latent-memory
  - editable-memory
  - prefix-caching
  - qwen3
  - gqa
pipeline_tag: text-generation
---

# LSLC (`Layered Spatial Latent Caching`) — Adapter Weights for Frozen `Qwen3-1.7B`

**Layered Spatial Latent Caching (LSLC)** is an **in-place editable, multi-resolution compressed context memory** designed to beat Raw-Text Chunk RAG on active KV footprint and state editability (with Full BF16 KV as uncompressed context). Each 256-token page is compiled into a 3D living-book sheet $\mathbf{Z} \in \mathbb{R}^{B \times P \times 3 \times 128 \times 4 \times 4}$ (`12 KiB/page` in FP16/BF16, or `48 bytes/raw token`) paired with a **`128 byte/page` Sparse Exact-Recall Footnote Bank** (`8 bytes/token`: `token_id: int32, page_id: uint16, h: uint8, w: uint8`). Repository: [`NekoFi/LSLC-Qwen3-1.7B-Adapter`](https://huggingface.co/NekoFi/LSLC-Qwen3-1.7B-Adapter).

All base LLM weights (`Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`) remain **100% frozen**. This repository contains the trained LSLC adapter modules (`40.42M` parameters for `Qwen/Qwen3-1.7B` in `bfloat16`):
- `LayeredLatentEncoder` (`FactorizedRoPE3D` over `[L=3, H=4, W=4]`)
- `SalienceGate` & `SparseExactBank` (`8 B/token` discrete high-entropy identifier + right-of-header policy bypass)
- `SpatialLatentProjector` (Matryoshka `1×1 TOC` + **2-level Chapter$\rightarrow$Page Hierarchical TOC**, `2×2 Skim`, and layer-ordered `Deep Read`)
- `QueryToTOCRouter` (1D page-index rotary routing + parameter-free in-VRAM `SparseExactBank` late-interaction `MaxSim` requiring `0 B` CPU RAM, quadratic runner-up confidence attenuation $(p_{\text{deep}}/p_{\max})^2$, and pluggable external retriever `router_out_override`)
- `DeltaUpdater` (In-place convex sticky-note overrides on $\ell=2$ with pre-gate channel-wise RMS alignment: $\mathbf{Z}_{p,2} \leftarrow (1-\mathbf{G}_p)\odot\mathbf{Z}_{p,2} + \mathbf{G}_p\odot\Delta\mathbf{Z}_{p,2}$ and bit-exact TOC prefix invariance $\max|\Delta_{\text{TOC}}| = 0.0$)

---

## Verified `v2.0.0` Benchmark Summary vs. Raw-Text RAG & Full KV Context (Live CUDA)

| Regime / Metric | **Constant-$k=2$ Raw RAG** (+Patch / LLM-Rewrite) | **LSLC `v2.0.0` (E2E GPU Router / Shared BM25)** | Full 1D KV Context (BF16) |
| :--- | :---: | :---: | :---: |
| **32K (`P=128`) Net VRAM (`1.7B`)** | `57.89 MiB` GPU + `0.26 MiB` CPU BM25 | **`16.17–17.81 MiB` (2-Lvl, `3.25×–3.58×` < RAG, `0 B` CPU) \| `27.11 MiB` (1-Lvl)** | `3,585.8 MiB` |
| **128K / 256K Net VRAM (`1.7B`)** | `58.31` / `58.88 MiB` GPU + `0.84 MiB` CPU | **`24.98–25.09 MiB` (`2.33×` < RAG) / `34.55 MiB` (`1.70×` < RAG, 2-Lvl)** | `14.0` / `28.0 GiB` (OOM) |
| **32K NIAH (`N=180`, Strict / Substr)** | `13.3% / 27.8%` Ex (`94.4%` T4) \| `1.1% / 94.4%` Sem | **E2E GPU (`0 B` CPU): `31.1% / 61.1%` Ex, `74.4% / 97.8%` Sem \| BM25: `45.6% / 82.2%` Ex (`97.0%` tok), `74.4% / 98.9%` Sem** | — |
| **128K NIAH (`N=60`, Strict / Substr)** | `16.7% / 30.0%` Ex \| `0.0% / 100.0%` Sem | **E2E GPU (`0 B` CPU): `33.3% / 56.7%` Ex, `46.7% / 76.7%` Sem \| BM25: `43.3% / 73.3%` Ex (`94.1%` tok), `23.3% / 70.0%` Sem** | — |
| **$k=3$ Numeric Overrides (`N=108`)** | `0%` (Imm) / `52.8%` (App) / `33.3%` (Patch) / `47.2%` (LLM-RW, `498 ms`) | **`94.4%` Strict / `95.8%` Substr (`100.0%` tok) & `97.2%` Static $\ell=0$ @ `8.62 ms/edit`** | `50.0%` |
| **$k=5$ Symbolic Overrides (`N=90`)** | `0%` (Imm) / `41.7%` (App) / `3.3% / 86.7%` (Patch) / `3.3% / 90.0%` (LLM-RW, `516 ms`) | **`38.3%–43.3%` Strict / `40.0%–46.7%` Substr (`82.0%–86.6%` tok) & `86.7%–90.0%` Static $\ell=0$ @ `8.70 ms/edit`** | — |
| **$k=3$ Paraphrased NL Edits (`N=90`)** | `0.0%` (Imm & Direct Regex Patch) / `40.0%` (App) / `30.0%` (LLM-RW, `496 ms`) | **`8.3%–10.0%` Strict / `35.0%–40.0%` Substr (`55.6%–67.8%` tok) & `93.3%–96.7%` Static $\ell=0$ @ `8.64 ms/edit`** | — |

> Full multi-seed JSON traces—including `1D-Flat` vs. `1D-3Layer` vs. `3D-Spatial` iso-budget ablation, equal-token Truncated Flat TOC controls (`133`/`158` tokens), and 5-way State-Override metrics—are archived in [`evals/m2_colab_l4/v2_complete_summary.json`](https://github.com/arielfikru/lslc-research/blob/main/evals/m2_colab_l4/v2_complete_summary.json). This `v2.0.0` unified checkpoint co-trains `QueryToTOCRouter` across $P \in \{4, 16, 128\}$ with `projector.toc_proj` strictly frozen and in-VRAM `SparseExactBank` `MaxSim`, achieving `100.0%` top-2 recall at $P=4$ (`k=3` overrides) with `0 B` CPU RAM.

---

## Quickstart Usage

```python
import torch
from safetensors.torch import load_file
from lslc import LSLCConfig, FrozenQwenLSLCWrapper, load_frozen_qwen3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = LSLCConfig.from_model_name("Qwen/Qwen3-1.7B")
llm, tokenizer = load_frozen_qwen3(cfg, device=device)
wrapper = FrozenQwenLSLCWrapper(cfg, llm)
wrapper.lslc.to(device)

# Load trained LSLC adapter weights (40.42M params)
state_dict = load_file("hf_hub_release/lslc_qwen3_1.7b_adapter.safetensors", device=str(device))
wrapper.lslc.load_state_dict(state_dict, strict=False)
wrapper.lslc.eval()
```

Or run the self-contained inference & in-place delta-override demo directly:
```bash
python scripts/demo_hf_inference.py --checkpoint hf_hub_release/lslc_qwen3_1.7b_adapter.safetensors
```

---

## Citation

```bibtex
@article{fikru2026lslc,
  title={Layered Spatial Latent Caching (LSLC): In-Place Editable Compressed Context Memory vs. Raw-Text RAG with Volatility-Ordered Prefix Caching},
  author={Fikru, Ariel},
  year={2026},
  month={November},
  doi={10.5281/zenodo.24473281},
  url={https://doi.org/10.5281/zenodo.24473281}
}
```
