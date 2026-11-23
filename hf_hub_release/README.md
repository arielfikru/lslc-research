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
- `SalienceGate` & `SparseExactBank` (`8 B/token` discrete high-entropy identifier bypass)
- `SpatialLatentProjector` (Matryoshka `1×1 TOC` + **2-level Chapter$\rightarrow$Page Hierarchical TOC**, `2×2 Skim`, and layer-ordered `Deep Read`)
- `QueryToTOCRouter` (1D page-index rotary routing with quadratic runner-up confidence attenuation $(p_{\text{deep}}/p_{\max})^2$, plus pluggable external retriever `router_out_override`)
- `DeltaUpdater` (In-place convex sticky-note overrides on $\ell=2$: $\mathbf{Z}_{p,2} \leftarrow (1-\mathbf{G}_p)\odot\mathbf{Z}_{p,2} + \mathbf{G}_p\odot\Delta\mathbf{Z}_{p,2}$ with bit-exact TOC prefix invariance $\max|\Delta_{\text{TOC}}| = 0.0$)

---

## Verified Benchmark Summary vs. Raw-Text RAG & Full KV Context (Live CUDA)

| Regime / Metric | **Constant-$k=2$ Raw RAG** (+In-Place Rewrite) | **LSLC 3D + Exact Bank + Delta** | Full 1D KV Context (BF16) |
| :--- | :---: | :---: | :---: |
| **32K (`P=128`) Net VRAM (`1.7B`)** | `57.89 MiB` (const $k=2$) / `71.7–183.7 MiB` (+TOC) | **`17.81 MiB` (2-Lvl Hier., `3.25×` < RAG) \| `27.11 MiB` (1-Lvl)** | `3,585.8 MiB` |
| **128K / 256K Net VRAM (`1.7B`)** | `58.31 MiB` / `58.88 MiB` (const $k=2$) | **`24.98 MiB` (`2.33×` < RAG) / `34.55 MiB` (`1.70×` < RAG, 2-Lvl)** | `14.0` / `28.0 GiB` (OOM) |
| **32K Shared-BM25 NIAH (`N=180`)** | `94.4%` Exact / `94.4%` Semantic EM | **`71.1%` Exact (`93.0%` content-tok) / `91.1%` Semantic EM** | — |
| **EC-1 & EC-2 Codebase / Decoys** | — | **`90.0%` & `85.0%` EM (`1.7B`) \| `80.0%` EM (`8B`)** | — |
| **$k=3$ Numeric Overrides (`N=108`)** | `0.0%` (Imm) / `55.6%` (App) / `45.8%` (LLM-RW, `791 ms`) | **`100.0% ± 0.0%` EM (`8.59 ms/edit`, $\max|\Delta_{\text{TOC}}|=0.0$)** | `50.0%` |
| **$k=5$ Symbolic Overrides (`N=90`)** | `0.0%` (Imm) / `46.7%` (App) / `88.9%` (LLM-RW, `809 ms`) | **`70.0%` (`93.4%` tok, T4) / `80.0%–91.7%` (L4) @ `8.42 ms/edit`** | — |
| **EC-5: Multi-Step Context Reasoning** | — | **`66.7%` (`1.7B`) \| `100.0%` (`8B`, `100%` retention)** | `88.9%` (`1.7B`) / `100%` (`8B`) |

> Full multi-seed JSON traces, including targeted (`791–809 ms/edit`) vs. full 256-token page (`14,631 ms/edit`) LLM-rewrite RAG latencies and shared Okapi BM25 recall (`97.22%`), are archived in [`evals/m2_colab_l4/rebuttal_v2_summary.json`](https://github.com/arielfikru/lslc-research/blob/main/evals/m2_colab_l4/rebuttal_v2_summary.json). Note that this unified checkpoint fine-tuned `QueryToTOCRouter` at `P=128`, so `P=4` state-override evaluation pairs LSLC with BM25 (`router_out_override`) cleanly at `100%` top-2 recall.

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
