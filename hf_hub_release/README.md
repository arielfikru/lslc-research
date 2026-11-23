---
license: mit
library_name: pytorch
base_model:
  - Qwen/Qwen3-1.7B
  - Qwen/Qwen3-4B
  - Qwen/Qwen3-8B
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

**Layered Spatial Latent Caching (LSLC)** transforms a flat, append-only 1D LLM Key-Value (KV) cache into an **in-place editable, multi-resolution 3D living book** $\mathbf{Z} \in \mathbb{R}^{B \times P \times 3 \times 128 \times 4 \times 4}$ (`12 KiB/page` in FP16/BF16, or `48 bytes/raw token`) paired with a **`128 byte/page` Sparse Exact-Recall Footnote Bank** (`8 bytes/token`: `token_id: int32, page_id: uint16, h: uint8, w: uint8`).

All base LLM weights (`Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`) remain **100% frozen**. This repository contains the trained LSLC adapter modules (`40.42M` parameters for `Qwen/Qwen3-1.7B` in `bfloat16`):
- `LayeredLatentEncoder` (`FactorizedRoPE3D` over `[L=3, H=4, W=4]`)
- `SalienceGate` & `SparseExactBank` (`8 B/token` discrete high-entropy identifier bypass)
- `SpatialLatentProjector` (Matryoshka `1×1 TOC`, `2×2 Skim`, and layer-ordered `Deep Read`)
- `QueryToTOCRouter` (1D page-index rotary routing with quadratic runner-up confidence attenuation $(p_{\text{deep}}/p_{\max})^2$)
- `DeltaUpdater` (In-place convex sticky-note overrides on $\ell=2$: $\mathbf{Z}_{p,2} \leftarrow (1-\mathbf{G}_p)\odot\mathbf{Z}_{p,2} + \mathbf{G}_p\odot\Delta\mathbf{Z}_{p,2}$ with bit-exact TOC prefix invariance $\max|\Delta_{\text{TOC}}| = 0.0$)

---

## Verified Benchmark Summary (Single NVIDIA L4 22.03 GiB, `bfloat16`)

| Regime / Metric | Full 1D KV (BF16) | StreamingLLM / H2O-20% | Raw-Text Chunk RAG (+Append) | **LSLC 3D + Exact Bank + Delta** |
| :--- | :---: | :---: | :---: | :---: |
| **32K (`P=128`) Net VRAM (`1.7B` / `4B`)** | `3,585.8` / `4,610.3 MiB` | `14.0` / `716.8 MiB` | `185.0` / `237.8 MiB` | **`27.11 MiB` (`132.3×`) / `34.42 MiB` (`133.9×`)** |
| **32K (`P=128`) Query TTFT (Cold / Warm)** | `5,235.4` / `12,725.8 ms` | `54.8` / `1,050.0 ms` | `~145` / `~210 ms` | **`92.9 / 58.0 ms` (`1.7B`) \| `111.8 / 84.3 ms` (`4B`)** |
| **256K (`P=1,024`) Net VRAM (`1.7B` / `8B`)** | `28.0` / `36.0 GiB` (OOM) | — | `284.4` / `365.5 MiB` | **`135.72 MiB` (`211.3×`) / `171.03 MiB` (`215.5×`)** |
| **Semantic NIAH (`1K–4K` EM)** | `93.3%–100.0%` | `0.0%–53.3%` | `100.0%` | **`100.0%`** |
| **Exact NIAH (`1K` 9-tok Code EM / Tok)** | `93.3% / 90.0%` | `0.0% / 16.0%` | `93.3% / 90.0%` | **`66.7% / 90.7%`** (`0.0%` without `SparseExactBank`) |
| **EC-1: 8-Page Python Codebase (EM / Tok)** | — | `0.0%` | — | **`90.0% / 96.0%` (`1.7B`) \| `80.0% / 96.5%` (`8B`)** |
| **EC-2: 4-Key Adversarial Decoys (EM / Tok)** | — | `0.0%` | — | **`85.0% / 96.0%` (`1.7B`) \| `80.0% / 95.0%` (`8B`)** |
| **$k=3$ State Overrides (3 Seeds `{42,1337,9999}`)** | `50.0%` | N/A | `0.0%` (Imm.) / `52.1% ± 15.7%` (App.) | **`100.0% ± 0.0%` EM (`100.0%` Tok, $\max|\Delta_{\text{TOC}}|=0.0$)** |
| **$k=5$ Symbolic Overrides (3 Seeds `{42,1337,9999}`)** | — | N/A | `0.0%` (Imm.) / `38.3% ± 7.6%` (App.) | **`91.7% ± 2.9%` EM (`97.3% ± 1.9%` Tok on `1.7B`)** |
| **EC-5: Multi-Step Context Reasoning** | `88.9%` (`1.7B`) / `100%` (`8B`) | — | — | **`66.7%` (`1.7B`) \| `100.0%` (`8B`, `100%` retention)** |

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
  title={Layered Spatial Latent Caching (LSLC): In-Place Editable Compressed Context Memory with Volatility-Ordered Prefix Caching and 128-Byte/Page Exact Footnotes},
  author={Fikru, Ariel},
  year={2026},
  month={November}
}
```
