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

**Layered Spatial Latent Caching (LSLC)** is an **in-place editable, multi-resolution compressed context memory** designed to beat Raw-Text Chunk RAG on active KV footprint and state editability (with Full BF16 KV as uncompressed context). Each 256-token page is compiled into a 3D living-book sheet $\mathbf{Z} \in \mathbb{R}^{B \times P \times 3 \times 128 \times 4 \times 4}$ (`12 KiB/page` in FP16/BF16, or `48 bytes/raw token`) paired with a **`128 byte/page` Sparse Exact-Recall Footnote Bank** (`8 bytes/token`: `token_id: int32, page_id: uint16, h: uint8, w: uint8`). Repository: [`NekoFi/LSLC-Qwen3-1.7B-Adapter`](https://huggingface.co/NekoFi/LSLC-Qwen3-1.7B-Adapter).

All base LLM weights (`Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`) remain **100% frozen**. This repository contains the trained LSLC adapter modules (`40.42M` parameters for `Qwen/Qwen3-1.7B` in `bfloat16`):
- `LayeredLatentEncoder` (`FactorizedRoPE3D` over `[L=3, H=4, W=4]`)
- `SalienceGate` & `SparseExactBank` (`8 B/token` discrete high-entropy identifier bypass)
- `SpatialLatentProjector` (Matryoshka `1×1 TOC` + **2-level Chapter$\rightarrow$Page Hierarchical TOC**, `2×2 Skim`, and layer-ordered `Deep Read`)
- `QueryToTOCRouter` (1D page-index rotary routing with quadratic runner-up confidence attenuation $(p_{\text{deep}}/p_{\max})^2$, plus pluggable external retriever `router_out_override`)
- `DeltaUpdater` (In-place convex sticky-note overrides on $\ell=2$: $\mathbf{Z}_{p,2} \leftarrow (1-\mathbf{G}_p)\odot\mathbf{Z}_{p,2} + \mathbf{G}_p\odot\Delta\mathbf{Z}_{p,2}$ with bit-exact TOC prefix invariance $\max|\Delta_{\text{TOC}}| = 0.0$)

---

## Verified Benchmark Summary vs. Raw-Text RAG & Full KV Context (NVIDIA L4, `bfloat16`)

| Regime / Metric | **Constant-$k=2$ Raw RAG** / (+Skim-4) | StreamingLLM / H2O-20% | **LSLC 3D + Exact Bank + Delta** | Full 1D KV Context (BF16) |
| :--- | :---: | :---: | :---: | :---: |
| **32K (`P=128`) Net VRAM (`1.7B`)** | `56.1 MiB` (const $k=2$) / `71.7–183.7 MiB` (+TOC/Skim) | `14.0` / `716.8 MiB` | **`27.11 MiB` (`2.1×–6.8×` < RAG; `132.3×` < Full)** | `3,585.8 MiB` |
| **32K (`P=128`) Query TTFT (`1.7B`)** | `62.6` (`d2`) / `159.4 ms` (`d2s4`) | `54.8` / `1,050.0 ms` | **`57.5 ms` Cold \| `54.6–58.0 ms` Warm** | `5,235.4 ms` |
| **256K (`P=1,024`) Net VRAM (`1.7B`)** | `57.1 MiB` (const $k=2$, analytical) / `284.4 MiB` (+TOC/Skim) | — | **`135.72 MiB` (1-lvl Flat) / `~32.9 MiB` (2-lvl Hier.)** | `28,672.0 MiB` (OOM) |
| **Semantic NIAH (`1K–4K` EM)** | `100.0%` | `0.0%–53.3%` | **`100.0%`** | `93.3%–100.0%` |
| **Exact NIAH (`1K` 9-tok Code EM)** | `93.3%` | `0.0%` | **`66.7%`** (`0.0%` without `SparseExactBank`) | `93.3%` |
| **EC-1: 8-Page Python Codebase (EM / Tok)** | — | `0.0%` | **`90.0% / 96.0%` (`1.7B`) \| `80.0% / 96.5%` (`8B`)** | — |
| **EC-2: 4-Key Adversarial Decoys (EM / Tok)** | — | `0.0%` | **`85.0% / 96.0%` (`1.7B`) \| `80.0% / 95.0%` (`8B`)** | — |
| **$k=3$ State Overrides (3 Seeds `{42,1337,9999}`)** | `0.0%` (Imm.) / `52.1% ± 15.7%` (Append) | N/A | **`100.0% ± 0.0%` EM ($\max|\Delta_{\text{TOC}}|=0.0$)** | `50.0%` |
| **$k=5$ Symbolic Overrides (3 Seeds `{42,1337,9999}`)** | `0.0%` (Imm.) / `38.3% ± 7.6%` (Append) | N/A | **`91.7% ± 2.9%` (`k=3`-warmed) \| `80.0%` (cold `1.7B`)** | — |
| **EC-5: Multi-Step Context Reasoning** | — | — | **`66.7%` (`1.7B`) \| `100.0%` (`8B`, `100%` retention)** | `88.9%` (`1.7B`) / `100%` (`8B`) |

> **Methodological Notes (Forensic Audit Transparency):**
> 1. In v1 `32K` comparison ([`reviewer_rebuttal_summary.json`](../evals/m2_colab_l4/reviewer_rebuttal_summary.json), $N=10$ Exact + $N=10$ Semantic per seed), `Raw-Text Chunk RAG` received the same `route.deep_idx` indices produced by LSLC's `QueryToTOCRouter` (`100%` top-2 recall on Exact NIAH; `40%` on Semantic NIAH).
> 2. Teacher-forced `token_acc = 90.0% ± 0.0%` reflects a 1-token leading-whitespace offset (`f" {code_val}"` at position 0), so autoregressive **Exact Match (EM)** is the primary metric.
> 3. Above $P^* \approx 384\text{--}392$ pages ($\approx 98.4\text{K--}100.4\text{K}$ tokens), 1-level Flat LSLC exceeds Constant-$k=2$ Raw-Text RAG (`56.6–57.1 MiB`); the 2-level Chapter$\rightarrow$Page hierarchy in `src/lslc/projector.py` restores strict memory dominance across `128K–256K`.
> 4. State-override comparisons above cover Immutable and Append-Errata RAG; live In-Place LLM-Rewrite RAG accuracy/latency (`ms/edit`) and Shared Okapi BM25 retrieval (`N=180`) are implemented in `scripts/colab_run_rebuttal_v2.py`.

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
