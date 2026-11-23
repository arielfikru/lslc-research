# LSLC: Layered Spatial Latent Caching

**Goal:** Turn flat, append-only 1D LLM KV-caches ("a $10\,\text{km}$ stone scroll") into an **editable, multi-resolution living book** (`Z in R^{B x P x L x C x H x W}`) paired with an **8-byte/token Sparse Exact-Recall Footnote Bank** — achieving **`40x–143x` net end-to-end VRAM reduction** on frozen dense GQA backbones (`Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B`) while preserving verbatim exact recall and supporting in-place sticky-note delta-updates.

---

## Why LSLC Exists (The "10 km Stone Scroll vs. Book" Thesis)

1. **Flat 1D KV-Caches Don't Scale & Can't Be Edited Locally:** Every generated token forces an $O(N)$ HBM read across a monolithic KV stream (`112 KiB/tok` on `Qwen3-1.7B`; `144 KiB/tok` on `Qwen3-4B` in FP16), and revising an earlier user constraint forces expensive suffix re-prefill.
2. **Optical Text-as-Image (DeepSeek-OCR / Glyph) Wastes Capacity on Pixels:** Rendering text into RGB images burns encoder/VLM capacity on font glyphs and whitespace, drops verbatim accuracy above `10x–20x` compression, and remains strictly read-only.
3. **2026 Hierarchical Latents (LCLMs / SeKV) Are Read-Only:** Recent skim-and-expand models still treat history as an immutable printed novel rather than a living, layered workbook, and lack dedicated discrete protection for high-entropy IDs/numbers.
4. **LSLC Synthesis:** Directly compile `256`-token pages into `[L=3, C=128, H=4, W=4]` semantic latent sheets (`12 KiB/page`) with **Table-of-Contents (`1x1`) / Skim (`2x2`) / Deep Read (`24 tok/page`) zoom**, **sticky-note delta patches on `l=2`**, and **O(1) verbatim footnote lookup**.

---

## The 4 Architectural Pillars

| Pillar | What We Engineer | Why It Matters |
| :--- | :--- | :--- |
| **1. Page-Modular 2D Book & 3D Factorized RoPE** | `LayeredLatentEncoder` maps `256`-token pages onto `[3, 128, 4, 4]` latents (`48 B/raw tok`) using factorized $(\ell, h, w)$ RoPE + 1D page offsets. | Pages append modularly without re-encoding prior context, preserving 2D local span structure under pooling. |
| **2. Multi-Resolution Zoom Pyramid** | `SpatialLatentProjector` + `QueryToTOCRouter` trained via Matryoshka dropout emit `1x1` TOC (`1 tok`), `2x2` Skim (`4 tok`), or Deep Read (`24 tok`). | LLM keeps global awareness across all $P$ pages via TOC while spending attention FLOPs only on relevant pages. |
| **3. Anti-Collapse Layers & Delta-Updates** | Asymmetric layers (`l=0` Static, `l=1` Entity, `l=2` Dynamic) ordered `[TOC \| Skim \| l=0 \| l=1 \| l=2 \| Exact]` + sparse `DeltaUpdater` (`G_p * dZ_p`). | Revising `l=2` updates context in-place while preserving 100% of the LLM's prefix KV-cache for TOC and `l=0,1`. |
| **4. Sparse Exact-Recall Footnote Bank** | `SalienceGate` stores top-`6.25%` (`16 tok/page`) verbatim IDs/numbers/syntax as `8 B/tok` `(token_id, page_id, h, w)` records (`128 B/page`). | Resolves rate-distortion limits on UUIDs/code/numbers with $O(1)$ lookup from frozen `embed_tokens`. |

---

## Verified Analytical & Live CUDA Net VRAM / TTFT Results (NVIDIA L4, `bfloat16`)

| Context ($N$) | Backbone | Full BF16 KV (Measured) | Ideal INT4 KV | H2O-20% | **LSLC Net Total (Analytical / Live CUDA)** | **Live CUDA TTFT (Full $\rightarrow$ LSLC Cold / Warm)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`4,096` (`P=16`)** | `Qwen3-1.7B` | `449.75 MiB` | `112.0 MiB` | `89.6 MiB` | **`11.13 MiB` / `12.88 MiB` (`34.93×`)** | `372.34 ms` $\rightarrow$ **`74.03 ms` (`5.0×`) / `55.62 ms` (`6.7×`)** |
| **`4,096` (`P=16`)** | `Qwen3-4B` | `578.25 MiB` | `144.0 MiB` | `115.2 MiB` | **`14.25 MiB` / `16.50 MiB` (`35.04×`)** | `972.38 ms` $\rightarrow$ **`79.67 ms` (`12.2×`) / `66.71 ms` (`14.6×`)** |
| **`32,768` (`P=128`)** | `Qwen3-1.7B` | `3,585.75 MiB` | `896.0 MiB` | `716.8 MiB` | **`25.36 MiB` / `27.11 MiB` (`132.27×`)** | `5,235.43 ms` $\rightarrow$ **`92.93 ms` (`56.3×`) / `58.03 ms` (`90.2×`)** |
| **`32,768` (`P=128`)** | `Qwen3-4B` | `4,610.25 MiB` | `1,152.0 MiB` | `921.6 MiB` | **`32.17 MiB` / `34.42 MiB` (`133.93×`)** | `12,725.77 ms` $\rightarrow$ **`111.83 ms` (`113.8×`) / `84.30 ms` (`151.0×`)** |

---

## Empirical Quality & Ablation Summary ([m3_summary.json](evals/m2_colab_l4/m3_summary.json) & [m4_qwen3_4b_summary.json](evals/m2_colab_l4/m4_qwen3_4b_summary.json))

- **3D Spatial-Layer `[L, H, W]` vs. 1D Flat Latent (`Track 0E` vs. `Track 2` on `Qwen3-1.7B`):** At an identical `12 KiB/page` (`24 tok/page`) byte budget, LSLC 3D lowers WikiText-2 continuation PPL from **`24.0471` (`1D Flat`) $\rightarrow$ `15.6630` (`LSLC 3D`)**, raises Semantic NIAH EM from **`46.67%` $\rightarrow$ `100.0%`**, and preserves bit-exact prefix invariance under edits (**`max_prefix_invariance_diff = 0.0`** vs. `1.0965` on `1D Flat`).
- **8-Byte/Token `SparseExactBank` Rescue (`Track 1` vs. `Track 2`):** `SalienceGate` achieves **`100.0%` (`1.0`) needle token capture rate** on both `1.7B` and `4B`, lifting teacher-forced Exact NIAH token accuracy from **`24.67%` $\rightarrow$ `90.67%`** on `Qwen3-1.7B` (matching Full KV `90.00%`) and lifting autoregressive Exact String Match from **`0.0%` $\rightarrow$ `66.67%` (`+66.67 pts`)** on both `Qwen3-1.7B` and `Qwen3-4B`.
- **Positive Scaling (`Qwen3-1.7B` $\rightarrow$ `Qwen3-4B`):** WikiText-2 PPL improves from **`16.5866` $\rightarrow$ `12.8757`** (`Track 1`) and **`15.6630` $\rightarrow$ `14.2428`** (`Track 2`, with `deep_only` zoom reaching **`13.5112`**), outperforming both `StreamingLLM-128` (`13.6437`) and `H2O-20%` (`14.2660`).
- **Sticky-Note `DeltaUpdater` (`Track 4`, $k=3$ sequential overrides):** Achieves **`100.0%` Delta-Edit Exact Match** on both `Qwen3-1.7B` (`mean_gate = 0.2591`) and `Qwen3-4B` (`mean_gate = 0.2033`) with **`max_prefix_invariance_diff = 0.0`** (`100.0%` Static $\ell=0$ policy retention in pure latent mode).
- **Brutal Stress-Test & `Qwen/Qwen3-8B` Scaling ([hard_stress_summary.json](evals/m2_colab_l4/hard_stress_summary.json)):**
  - **EC-1 Agentic Multi-File Python Codebase (`8` pages / `2,048` tokens):** **`90.0%` EM (`96.0%` tok_acc)** on `Qwen3-1.7B` and **`80.0%` EM (`96.5%` tok_acc)** on `Qwen3-8B`.
  - **EC-2 Adversarial Near-Miss Decoy Keys (`4` co-existing keys/book):** **`85.0%` EM (`96.0%` tok_acc, `98.9%` capture)** on `Qwen3-1.7B` and **`80.0%` EM (`95.0%` tok_acc, `98.9%` capture)** on `Qwen3-8B`.
  - **EC-3 Deep $k=5$ Sequential Non-Numeric Symbolic Overrides:** **`80.0%` EM (`93.75%` tok_acc, `max_prefix_invariance_diff = 0.0`)** on `Qwen3-1.7B` and **`55.0%` EM (`90.89%` tok_acc, `max_prefix_invariance_diff = 0.0`)** on `Qwen3-8B`.
  - **EC-4 Ultra-Long `128K` (`512p`) & `256K` (`1,024p`) Live CUDA Book Profiling:** At `256K` tokens, Full KV OOMs (`28.0–36.0 GiB`), whereas LSLC Net VRAM is **`135.72 MiB` (`211.26×`, `97.74 ms` Warm TTFT)** on `Qwen3-1.7B` and **`171.03 MiB` (`215.54×`, `453.23 ms` Warm TTFT)** on `Qwen3-8B`.
  - **EC-5 Context-Grounded Multi-Step Reasoning & Intrinsic General IQ Retention:** Multi-step context reasoning reaches **`100.0%` (`100%` retention vs. Full KV)** on `Qwen3-8B` (`66.7%` / `75%` retention on `1.7B`), while intrinsic general IQ retention is **`125.0%`** on `1.7B` and **`83.3%`** on `8B`.

---

## Colab Hardware Tiers (~25M–40M Trainable Params, Frozen LLM)

- **Colab T4 (`15 GB` VRAM):** Runs `Qwen/Qwen3-1.7B` in FP16 (`~7–9 GB` peak) or `Qwen/Qwen3-4B` in 4-bit NF4 frozen + gradient checkpointing (`~8–10 GB` peak).
- **Colab L4 (`24 GB` VRAM, `~1.94 CU/hr`):** Primary lossless BF16 training & evaluation workhorse across both `1.7B` and `4B`.
- **Colab A100 (`40 GB` VRAM, `~6.73 CU/hr`):** Optional accelerator for rapid hyperparameter sweeps and 32K systems profiling.

---

## Documentation, Reproducibility & Hugging Face Release Map

- [REPRODUCE.md](REPRODUCE.md) — **1-Command Reproduction Guide:** local unit tests (`10/10`), instant Living-Book demo (`scripts/demo_hf_inference.py`), analytical VRAM calculator, Colab L4 benchmark scripts (`M2`, `M3`, `M4`, `Hard Stress`, `3-Seed Rebuttal`), and Hugging Face weight export.
- [hf_hub_release/README.md](hf_hub_release/README.md) & [hf_hub_release/config.json](hf_hub_release/config.json) — **Hugging Face Hub Model Card & Adapter Package (`40.42M` params, `77.10 MiB` `bfloat16`)** with SafeTensors/PyTorch export via [scripts/export_hf_weights.py](scripts/export_hf_weights.py) and live inference demo via [scripts/demo_hf_inference.py](scripts/demo_hf_inference.py).
- [docs/paper_draft_lslc.tex](docs/paper_draft_lslc.tex) — **Finalized Two-Column Academic Paper:** complete mathematical specification, 17 verified citations up to November 2026 (including `Memorilla`, `Models Take Notes at Prefill`, `GroupKV`, `LeanSparse`, and `GLM-5.4 Loop Transformer`), and full empirical tables across `Qwen3-1.7B`, `Qwen3-4B`, and `Qwen3-8B`.
- [docs/blog_lslc_netra_style.html](docs/blog_lslc_netra_style.html) & [docs/article_engineering_deep_dive.md](docs/article_engineering_deep_dive.md) — **Netra Runtime-Style Engineering Blog Post** with KaTeX underbraced derivations, custom inline SVG charts, recipe, and FAQ.
- [ARCHITECTURE.md](ARCHITECTURE.md) — Formal technical architecture specification, tensor shapes, and design rationale.
- [evals/m2_colab_l4/](evals/m2_colab_l4/) — Archived empirical JSON benchmark logs across `Qwen3-1.7B`, `Qwen3-4B`, and `Qwen3-8B`.
