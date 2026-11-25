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

## Empirical Quality & Ablation Summary ([m3_summary.json](evals/m2_colab_l4/m3_summary.json), [hard_stress_summary.json](evals/m2_colab_l4/hard_stress_summary.json), [rebuttal_v2_summary.json](evals/m2_colab_l4/rebuttal_v2_summary.json) & [v2_complete_summary.json](evals/m2_colab_l4/v2_complete_summary.json))

- **3D Spatial-Layer `[L, H, W]` vs. 1D Flat Latent (`Track 0E`, `Track 2` & Iso-Budget Ablation on `Qwen3-1.7B`):** At an identical `12 KiB/page` (`24 tok/page`) byte budget, LSLC 3D lowers WikiText-2 continuation PPL from **`24.0471` (`1D Flat`) $\rightarrow$ `15.6630` (`LSLC 3D`)**, raises Semantic NIAH EM from **`46.67%` $\rightarrow$ `100.0%`**, and preserves bit-exact prefix invariance under edits (**`max_prefix_invariance_diff = 0.0`** vs. `1.0965` on `1D Flat`). In our strictly controlled iso-parameter (`40.42M`), iso-byte (`12.125 KiB/p`), iso-step, and iso-`SparseExactBank` ablation ([v2_complete_summary.json](evals/m2_colab_l4/v2_complete_summary.json)), `3D-Spatial` improves WikiText-2 PPL to **`18.94`** (vs. `22.41` on `1D-Flat` and `24.86` on `1D-3Layer`), achieves **`100.0%` Semantic NIAH** (vs. `66.7%` on `1D-Flat` and `93.3%` on `1D-3Layer`), and maintains **`100.0%` static $\ell=0$ retention** with $\max|\Delta_{\text{TOC}}|=0.0$.
- **8-Byte/Token `SparseExactBank` Rescue (`Track 1` vs. `Track 2`):** `SalienceGate` achieves **`100.0%` (`1.0`) needle token capture rate** (`99.51%` at `32K`, `99.63%` at `128K`), lifting Exact NIAH EM from **`0.0%` $\rightarrow$ `66.67%`** on 4-page books (`90.67%` tok acc on `1.7B`) and reaching **`90.0%` EM** on 8-page agentic Python codebases (`EC-1`) and **`85.0%` EM** on 4-key adversarial decoys (`EC-2`).
- **Positive Scaling (`Qwen3-1.7B` $\rightarrow$ `Qwen3-4B` / `Qwen3-8B`):** WikiText-2 PPL improves from **`16.5866` $\rightarrow$ `12.8757`** (`Track 1`), and multi-step context reasoning (`EC-5`) improves from `66.7%` (`1.7B`) to **`100.0%` (`100%` retention vs. Full KV)** on `Qwen3-8B` (4-bit NF4).
- **Unified `32K` (`P=128`, `N=180`) & `128K` (`P=512`, `N=60`) Multi-Seed NIAH (`v2.1.0` Fair Canonical Extraction, [v2_complete_summary.json](evals/m2_colab_l4/v2_complete_summary.json)):**
  - **Headline Metric (`extracted_em`, deterministic first-sentence extraction under identical 24-token greedy decoding):** Fixing `v2.0.0`'s step-0 `\n` early-stop bug restores Raw-Text Chunk RAG (`512` tokens, `270–857 KiB` CPU BM25) to **`93.3%` Exact / `42.2%` (`94.4%` Substr) Semantic Extracted EM at `32K`** and **`90.0%` Exact / `50.0%` (`100.0%` Substr) Semantic Extracted EM at `128K`**.
  - **100% GPU-Resident Routing (`QueryToTOCRouter` + in-VRAM `SparseExactBank` `MaxSim`, `0 B` additional CPU lexical index):** 2-Level Hierarchical LSLC (`131` active tokens at `32K`, `158` at `128K`) achieves **`61.1%` Exact / `77.8%` (`98.9%` Substr) Semantic Extracted EM** at `32K` and **`56.7%` Exact / `43.3%` (`73.3%` Substr) Semantic Extracted EM** at `128K`.
  - **Hybrid Shared-BM25 & Equal-Token Truncated Flat Control:** 2-Level Hierarchical LSLC reaches **`80.0%` Exact (`97.0%` content-tok) / `77.8%` (`100.0%` Substr) Semantic Extracted EM** at `32K` and **`70.0%` Exact (`94.1%` content-tok) / `33.3%` (`70.0%` Substr) Semantic Extracted EM** at `128K`—trading a bounded fraction of long-context retrieval ceiling against Raw-Text RAG for `1.70×–3.58×` smaller live VRAM. At equal token budget, pruning active TOC tokens drives the primary jump from Flat TOC, with 2-level pooling matching Truncated Flat (`+0.0%` Exact; `+2.2% ± 2.2%` `32K` Semantic Extracted EM) while retaining chapter summaries across `100%` of the book.
- **5-Way Editable RAG vs. LSLC State Overrides & Restored Static $\ell=0$ Retention (`N=288`, 3 Seeds, `v2.1.0`, [v2_complete_summary.json](evals/m2_colab_l4/v2_complete_summary.json)):**
  - **$k=3$ structured numeric (`N=108`):** Immutable RAG `0.0%`, best mutable RAG (**Append-Errata**) `52.8%` Extracted EM (`0.48 ms`), `Direct Text-Patch` `33.3%` (`1.20 ms`), `In-Place LLM-Rewrite` `47.2%` (**`487.0 ms/edit`** 11-tok block / **`11,297.0 ms/edit`** 256-tok page on L4), **`LSLC DeltaUpdater (E2E GPU / Shared BM25)`** **`94.4%–95.8%` Extracted EM (`97.7%–100.0%` content-tok)** and **`97.2%` Static $\ell=0$ Retention EM** at **`8.85 ms/edit`** (`4.47–4.51 ms` zero-copy inference slice write; $\max|\Delta_{\text{TOC}}|=0.0$).
  - **$k=5$ structured symbolic (`N=90`):** Best RAG (`Direct Text-Patch` `1.15 ms` & `LLM-Rewrite` `504.7 ms`) reaches **`90.0%` Extracted EM**, whereas `LSLC DeltaUpdater` scores **`40.0%–48.3%` Extracted EM (`82.0%–87.4%` content-tok)** with `86.7%–90.0%` static $\ell=0$ retention at `8.25 ms/edit`. Oracle A/B/C diagnostics ([v21_diagnostics.json](evals/m2_colab_l4/v21_diagnostics.json)) trace this gap to base encoder/projector unseen-schema generalization (`Cond B = 0.0%`) and multi-step `DeltaUpdater` drift (`Cond C = 10%–40%`).
  - **$k=3$ paraphrased natural-language revisions (`N=90`):** Deterministic `Direct Text-Patch` collapses to `0.0%` (`0%` regex match), while simplest **`Append-Errata RAG`** reaches **`40.0% ± 8.7%` Extracted EM** vs. `LSLC DeltaUpdater`'s **`30.0%–35.0%` Extracted EM (`38.3%` Substr)** and `LLM-Rewrite` `30.0%` (`489.2 ms/edit`), with **`93.3%–96.7%` Static $\ell=0$ Retention EM** at `8.35 ms/edit`. *(See [`docs/release_notes_v2.1.0.md`](docs/release_notes_v2.1.0.md) for the kernel-style post-mortem & `P=4/128/1024` edit-scaling probe.)*

---

## Colab Hardware Tiers (~25M–40M Trainable Params, Frozen LLM)

- **Colab T4 (`15 GB` VRAM):** Runs `Qwen/Qwen3-1.7B` in FP16 (`~7–9 GB` peak) or `Qwen/Qwen3-4B` in 4-bit NF4 frozen + gradient checkpointing (`~8–10 GB` peak).
- **Colab L4 (`24 GB` VRAM, `~1.94 CU/hr`):** Primary lossless BF16 training & evaluation workhorse across both `1.7B` and `4B`.
- **Colab A100 (`40 GB` VRAM, `~6.73 CU/hr`):** Optional accelerator for rapid hyperparameter sweeps and 32K systems profiling.

---

## Documentation, Reproducibility & Hugging Face Release Map

- [**`v2.1.0` Release Notes & Engineering Post-Mortem**](docs/release_notes_v2.1.0.md) — **Kernel-Style Engineering Mini-Blog:** Why `v2.1.0` fixed the step-0 decoding asymmetry, separated `extracted_em` from `clause_prefix_em`, ran Oracle A/B/C ($k=1 \dots 5$) & `P ∈ {4, 128, 1024}` edit-scaling diagnostics, and reports unvarnished `v2.0.0 -> v2.1.0` deltas.
- [REPRODUCE.md](REPRODUCE.md) — **1-Command Reproduction Guide:** local unit tests (`14/14`), instant Living-Book demo (`scripts/demo_hf_inference.py`), analytical VRAM calculator, Colab GPU benchmark scripts (including `v2.1.0` `--load-checkpoint` & `colab_run_v21_diagnostics.py`), and Hugging Face weight export.
- [`NekoFi/LSLC-Qwen3-1.7B-Adapter`](https://huggingface.co/NekoFi/LSLC-Qwen3-1.7B-Adapter) ([hf_hub_release/README.md](hf_hub_release/README.md) & [hf_hub_release/config.json](hf_hub_release/config.json)) — **Hugging Face Hub Model Card & Adapter Package (`40.42M` params, `77.12 MiB` `bfloat16`, SHA-256 `5bfc03fe...`)** with SafeTensors/PyTorch export via [scripts/export_hf_weights.py](scripts/export_hf_weights.py) and live inference demo via [scripts/demo_hf_inference.py](scripts/demo_hf_inference.py).
- [docs/LSLC_2026_Fikru.pdf](docs/LSLC_2026_Fikru.pdf) ([LaTeX source](docs/paper_draft_lslc.tex) | DOI: [`10.5281/zenodo.24473281`](https://doi.org/10.5281/zenodo.24473281)) — **Academic Paper (`v2.1.0`, Ariel Fikru; `Qwen3-1.7B`, `4B`, `8B`):** Split Tables 4–6, fair canonical extraction, Oracle A/B/C & edit-scaling diagnostics, and explicit methodological/audit disclosures.
- [docs/blog_lslc_netra_style.html](docs/blog_lslc_netra_style.html) & [docs/article_engineering_deep_dive.md](docs/article_engineering_deep_dive.md) — **Netra Runtime-Style Engineering Blog Post** with KaTeX underbraced derivations, custom inline SVG charts, Constant-$k$ RAG crossover analysis, and FAQ.
- [ARCHITECTURE.md](ARCHITECTURE.md) — Formal technical architecture specification, tensor shapes, and design rationale.
- [evals/m2_colab_l4/](evals/m2_colab_l4/) — Archived empirical JSON benchmark logs across `Qwen3-1.7B`, `Qwen3-4B`, and `Qwen3-8B`.
