# LSLC: Layered Spatial Latent Caching

**Goal:** Turn flat, append-only 1D LLM KV-caches and read-only RAG text chunks into an **in-place editable, multi-resolution living book** (`Z in R^{B x P x L x C x H x W}`) paired with an **8-byte/token Sparse Exact-Recall Footnote Bank** — beating Raw-Text Chunk RAG on 32K active KV VRAM (**`27.11 MiB`** vs. **`71.72–183.72 MiB`**, or **`132×–134×`** below Full BF16 KV context) on frozen dense GQA backbones (`Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, and `Qwen/Qwen3-8B`), supporting **2-Level Chapter$\rightarrow$Page Hierarchical TOC** above `~100K` tokens, and enabling in-place sticky-note delta-updates without conflicting errata.

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

## Verified Benchmark Summary vs. Raw-Text RAG & Full KV Context (Live CUDA)

| Context ($N$) | Backbone | **Constant-$k=2$ Raw RAG** / (+TOC/Skim) | **LSLC Net Total (Live 1-Lvl Flat / Live 2-Lvl Hier. `c=16`)** | Full BF16 KV Context (Reference) | **Live Query TTFT (RAG / Full $\rightarrow$ LSLC Warm)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`4,096` (`P=16`)** | `Qwen3-1.7B` | `56.0 MiB` (const $k=2$) | **`12.88 MiB`** (`4.3×` < RAG; `34.9×` < Full, L4) | `449.75 MiB` | `372.3 ms` (Full) $\rightarrow$ **`55.62 ms`** (L4) |
| **`32,768` (`P=128`)** | `Qwen3-1.7B` | `57.89 MiB` (const live) / `71.7–183.7 MiB` | **`27.11 MiB` (1-Lvl, `2.14×` < RAG) / `17.81 MiB` (2-Lvl, `3.25×` < RAG)** | `3,585.75 MiB` | `54.62–58.03 ms` (L4) \| `181.2 ms` vs. `859.4 ms` (T4) |
| **`32,768` (`P=128`)** | `Qwen3-4B` | `72.1 MiB` (const) / `92.2–237.8 MiB` | **`34.42 MiB`** (`2.1×–6.9×` < RAG; `133.9×` < Full, L4) | `4,610.25 MiB` | `12,725.8 ms` (Full) $\rightarrow$ **`84.30 ms`** (L4) |
| **`131,072` (`P=512`)** | `Qwen3-1.7B` | `58.31 MiB` (const live) | `73.66 MiB` (1-Lvl Flat, `0.79×`) / **`24.98 MiB` (2-Lvl, `2.33×` < RAG; `573.8×` < Full)** | `14,336 MiB` (OOM) | `57.96 ms` (L4 Flat) \| **`182.4 ms`** vs. `860.7 ms` (T4 Hier.) |
| **`262,144` (`P=1,024`)** | `Qwen3-1.7B` | `58.88 MiB` (const live) | `135.72 MiB` (1-Lvl Flat, `0.43×`) / **`34.55 MiB` (2-Lvl, `1.70×` < RAG; `829.9×` < Full)** | `28,672 MiB` (OOM) | `97.74 ms` (L4 Flat) \| **`184.5 ms`** vs. `863.1 ms` (T4 Hier.) |

> **Constant-$k=2$ RAG Crossover & 2-Level Hierarchy:** Above **$P^* \approx 384.3$ pages ($\approx 98.4\text{K}$ tokens)** on `Qwen3-1.7B` ($392.1$ pages / $\approx 100.4\text{K}$ on `4B/8B`), 1-level Flat LSLC crosses above Constant-$k=2$ Raw RAG (`58.31–58.88 MiB`). Enabling the 2-level Chapter$\rightarrow$Page TOC (`hierarchical_chapter_size=16` in [src/lslc/projector.py](src/lslc/projector.py), verified $\max|\Delta_{\text{TOC}}|=0.0$) restores strict VRAM dominance through `256K` (`17.81 / 24.98 / 34.55 MiB` live CUDA).

---

## Empirical Quality & Ablation Summary ([m3_summary.json](evals/m2_colab_l4/m3_summary.json), [hard_stress_summary.json](evals/m2_colab_l4/hard_stress_summary.json) & [rebuttal_v2_summary.json](evals/m2_colab_l4/rebuttal_v2_summary.json))

- **3D Spatial-Layer `[L, H, W]` vs. 1D Flat Latent (`Track 0E` vs. `Track 2` on `Qwen3-1.7B`):** At an identical `12 KiB/page` (`24 tok/page`) byte budget, LSLC 3D lowers WikiText-2 continuation PPL from **`24.0471` (`1D Flat`) $\rightarrow$ `15.6630` (`LSLC 3D`)**, raises Semantic NIAH EM from **`46.67%` $\rightarrow$ `100.0%`**, and preserves bit-exact prefix invariance under edits (**`max_prefix_invariance_diff = 0.0`** vs. `1.0965` on `1D Flat`).
- **8-Byte/Token `SparseExactBank` Rescue (`Track 1` vs. `Track 2`):** `SalienceGate` achieves **`100.0%` (`1.0`) needle token capture rate** (`99.26%` at `32K`), lifting Exact NIAH EM from **`0.0%` $\rightarrow$ `66.67%`** on 4-page books (`90.67%` tok acc on `1.7B`) and reaching **`90.0%` EM** on 8-page agentic Python codebases (`EC-1`) and **`85.0%` EM** on 4-key adversarial decoys (`EC-2`).
- **Positive Scaling (`Qwen3-1.7B` $\rightarrow$ `Qwen3-4B` / `Qwen3-8B`):** WikiText-2 PPL improves from **`16.5866` $\rightarrow$ `12.8757`** (`Track 1`), and multi-step context reasoning (`EC-5`) improves from `66.7%` (`1.7B`) to **`100.0%` (`100%` retention vs. Full KV)** on `Qwen3-8B` (4-bit NF4).
- **Unified `32K` (`P=128`, `N=180`, 3 Seeds) Under Shared Non-Oracle Okapi BM25 ([rebuttal_v2_summary.json](evals/m2_colab_l4/rebuttal_v2_summary.json)):** Sharing identical BM25 retrieval (`97.22%` top-2 recall; `100%` Exact, `94.44%` Semantic) with stripped `content_tok_acc`:
  - **2-Level Hierarchical LSLC (`133` active tokens):** Eliminates 122-token flat TOC clutter, lifting Semantic NIAH from `2.22%` (Flat BM25) to **`91.11% ± 5.09%` EM** (near Raw RAG's `94.44%` at **`3.25×` smaller VRAM**) and **`71.11% ± 6.94%` Exact EM (`92.96%` content-token acc)**.
- **4-Way Editable RAG vs. LSLC State Overrides & Per-Edit Latency (`k=3` `N=108`, `k=5` `N=90`, Shared BM25):**
  - **$k=3$ numeric:** Immutable RAG `0.0%` (`0 ms`), Append-Errata RAG `55.56%` (`0.58 ms`), **In-Place LLM-Rewrite RAG** `45.83%` (**`790.89 ms/edit`** targeted 11-tok / **`14,630.9 ms/edit`** full 256-tok page), **`LSLC DeltaUpdater`** **`100.0% ± 0.0%` EM (`100.0%` content-tok)** at **`8.59 ms/edit`** (**`92×–1,703×` faster**, $\max|\Delta_{\text{TOC}}|=0.0$).
  - **$k=5$ symbolic:** Immutable RAG `0.0%`, Append-Errata RAG `46.67%` (`0.59 ms`), **In-Place LLM-Rewrite RAG** `88.89%` (**`808.64 ms/edit`** targeted / **`14,630.9 ms/edit`** full page), **`LSLC DeltaUpdater`** **`70.00% ± 15.28%` EM (`93.35%` content-tok)** on unified T4 `fp16` (`80.0%–91.67%` `bf16` L4) at **`8.42 ms/edit`** (**`96×–1,738×` faster**, $\max|\Delta_{\text{TOC}}|=0.0$). *(Note: specializing the learned router strictly at $P=128$ degrades $P=4$ recall to `0.0%`, confirming the value of BM25 or multi-scale router co-training.)*

---

## Colab Hardware Tiers (~25M–40M Trainable Params, Frozen LLM)

- **Colab T4 (`15 GB` VRAM):** Runs `Qwen/Qwen3-1.7B` in FP16 (`~7–9 GB` peak) or `Qwen/Qwen3-4B` in 4-bit NF4 frozen + gradient checkpointing (`~8–10 GB` peak).
- **Colab L4 (`24 GB` VRAM, `~1.94 CU/hr`):** Primary lossless BF16 training & evaluation workhorse across both `1.7B` and `4B`.
- **Colab A100 (`40 GB` VRAM, `~6.73 CU/hr`):** Optional accelerator for rapid hyperparameter sweeps and 32K systems profiling.

---

## Documentation, Reproducibility & Hugging Face Release Map

- [REPRODUCE.md](REPRODUCE.md) — **1-Command Reproduction Guide:** local unit tests (`12/12`), instant Living-Book demo (`scripts/demo_hf_inference.py`), analytical VRAM calculator, Colab GPU benchmark scripts (Core Ablations, `4B` Scaling, Adversarial Stress Suite, and Unified Multi-Seed Evaluation), and Hugging Face weight export.
- [`NekoFi/LSLC-Qwen3-1.7B-Adapter`](https://huggingface.co/NekoFi/LSLC-Qwen3-1.7B-Adapter) ([hf_hub_release/README.md](hf_hub_release/README.md) & [hf_hub_release/config.json](hf_hub_release/config.json)) — **Hugging Face Hub Model Card & Adapter Package (`40.42M` params, `77.10 MiB` `bfloat16`)** with SafeTensors/PyTorch export via [scripts/export_hf_weights.py](scripts/export_hf_weights.py) and live inference demo via [scripts/demo_hf_inference.py](scripts/demo_hf_inference.py).
- [docs/paper_draft_lslc.pdf](docs/paper_draft_lslc.pdf) ([LaTeX source](docs/paper_draft_lslc.tex)) — **Academic Paper (`Qwen3-1.7B`, `4B`, `8B`):** RAG-centered evaluation & Constant-$k$ crossover math, 17 verified citations through Nov 2026 (`Memorilla`, `Models Take Notes at Prefill`, `GroupKV`, `LeanSparse`, `GLM-5.4 Loop Transformer`), and explicit methodological/audit disclosures.
- [docs/blog_lslc_netra_style.html](docs/blog_lslc_netra_style.html) & [docs/article_engineering_deep_dive.md](docs/article_engineering_deep_dive.md) — **Netra Runtime-Style Engineering Blog Post** with KaTeX underbraced derivations, custom inline SVG charts, Constant-$k$ RAG crossover analysis, and FAQ.
- [ARCHITECTURE.md](ARCHITECTURE.md) — Formal technical architecture specification, tensor shapes, and design rationale.
- [evals/m2_colab_l4/](evals/m2_colab_l4/) — Archived empirical JSON benchmark logs across `Qwen3-1.7B`, `Qwen3-4B`, and `Qwen3-8B`.
