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

## Verified Benchmark Summary vs. Raw-Text RAG & Full KV Context (NVIDIA L4, `bfloat16`)

| Context ($N$) | Backbone | **Constant-$k=2$ Raw RAG** / (+TOC/Skim) | **LSLC Net Total (Live CUDA 1-Lvl / Analytical 2-Lvl Hier.)** | Full BF16 KV Context (Reference) | **Live CUDA Query TTFT (RAG / Full KV $\rightarrow$ LSLC Warm)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`4,096` (`P=16`)** | `Qwen3-1.7B` | `56.0 MiB` (const $k=2$) | **`12.88 MiB`** (`4.3×` < RAG; `34.9×` < Full) | `449.75 MiB` | `372.3 ms` (Full) $\rightarrow$ **`55.62 ms`** |
| **`32,768` (`P=128`)** | `Qwen3-1.7B` | `56.1 MiB` (const) / `71.7–183.7 MiB` (+TOC/Skim) | **`27.11 MiB` (`2.1×–6.8×` < RAG) / `≤16.2 MiB` (2-Lvl)** | `3,585.75 MiB` | `62.6–159.4 ms` (RAG) $\rightarrow$ **`54.6–58.0 ms`** |
| **`32,768` (`P=128`)** | `Qwen3-4B` | `72.1 MiB` (const) / `92.2–237.8 MiB` (+TOC/Skim) | **`34.42 MiB` (`2.1×–6.9×` < RAG; `133.9×` < Full)** | `4,610.25 MiB` | `12,725.8 ms` (Full) $\rightarrow$ **`84.30 ms`** |
| **`131,072` (`P=512`)** | `Qwen3-1.7B` | `56.6 MiB` (const $k=2$) | `73.66 MiB` (1-Lvl Flat) / **`≤23.3 MiB` (2-Lvl Hier.)** | `14,336 MiB` (OOM) | **`57.96 ms`** |
| **`262,144` (`P=1,024`)** | `Qwen3-1.7B` | `57.1 MiB` (const $k=2$) | `135.72 MiB` (1-Lvl Flat) / **`≤32.9 MiB` (2-Lvl Hier.)** | `28,672 MiB` (OOM) | **`97.74 ms`** |

> **Constant-$k=2$ RAG Crossover & 2-Level Hierarchy:** Above **$P^* \approx 384\text{--}392$ pages ($\approx 98.4\text{K--}100.4\text{K}$ tokens)**, 1-level Flat LSLC exceeds Constant-$k=2$ Raw RAG (`56.6–57.1 MiB` analytical). Enabling the 2-level Chapter$\rightarrow$Page TOC (`hierarchical_chapter_size=16` in `src/lslc/projector.py`, $\max|\Delta_{\text{TOC}}|=0.0$) restores strict memory superiority through `256K`.

---

## Empirical Quality & Ablation Summary ([m3_summary.json](evals/m2_colab_l4/m3_summary.json) & [m4_qwen3_4b_summary.json](evals/m2_colab_l4/m4_qwen3_4b_summary.json))

- **3D Spatial-Layer `[L, H, W]` vs. 1D Flat Latent (`Track 0E` vs. `Track 2` on `Qwen3-1.7B`):** At an identical `12 KiB/page` (`24 tok/page`) byte budget, LSLC 3D lowers WikiText-2 continuation PPL from **`24.0471` (`1D Flat`) $\rightarrow$ `15.6630` (`LSLC 3D`)**, raises Semantic NIAH EM from **`46.67%` $\rightarrow$ `100.0%`**, and preserves bit-exact prefix invariance under edits (**`max_prefix_invariance_diff = 0.0`** vs. `1.0965` on `1D Flat`).
- **8-Byte/Token `SparseExactBank` Rescue (`Track 1` vs. `Track 2`):** `SalienceGate` achieves **`100.0%` (`1.0`) needle token capture rate** on both `1.7B` and `4B`, lifting teacher-forced Exact NIAH token accuracy from **`24.67%` $\rightarrow$ `90.67%`** on `Qwen3-1.7B` (matching Full KV `90.00%`) and lifting autoregressive Exact String Match from **`0.0%` $\rightarrow$ `66.67%` (`+66.67 pts`)** on both `Qwen3-1.7B` and `Qwen3-4B`.
- **Positive Scaling (`Qwen3-1.7B` $\rightarrow$ `Qwen3-4B`):** WikiText-2 PPL improves from **`16.5866` $\rightarrow$ `12.8757`** (`Track 1`) and **`15.6630` $\rightarrow$ `14.2428`** (`Track 2`, with `deep_only` zoom reaching **`13.5112`**), outperforming both `StreamingLLM-128` (`13.6437`) and `H2O-20%` (`14.2660`).
- **Sticky-Note `DeltaUpdater` ($k=3$ & $k=5$ State Overrides):** Achieves **`100.0% ± 0.0%` EM** across 3 seeds on $k=3$ numeric overrides (`reviewer_rebuttal_summary.json`, `max_prefix_invariance_diff = 0.0`) vs. `0.0%` Immutable RAG and `52.1%` Append-Errata RAG; on $k=5$ symbolic overrides, a cold `DeltaUpdater` scores `80.0%` (`hard_stress_summary.json`, $N=20$) while a `k=3`-curriculum-warmed `DeltaUpdater` reaches **`91.67% ± 2.89%`** across 3 seeds (`reviewer_rebuttal_summary.json`, vs. `38.3%` Append-Errata RAG). *(Live In-Place LLM-Rewrite RAG latency/accuracy and shared Okapi BM25 32K evaluation at `N=180` are implemented in [scripts/colab_run_rebuttal_v2.py](scripts/colab_run_rebuttal_v2.py).)*
- **Brutal Stress-Test & `Qwen/Qwen3-8B` Scaling ([hard_stress_summary.json](evals/m2_colab_l4/hard_stress_summary.json)):**
  - **EC-1 Agentic Multi-File Python Codebase (`8` pages / `2,048` tokens):** **`90.0%` EM (`96.0%` tok_acc)** on `Qwen3-1.7B` and **`80.0%` EM (`96.5%` tok_acc)** on `Qwen3-8B`.
  - **EC-2 Adversarial Near-Miss Decoy Keys (`4` co-existing keys/book):** **`85.0%` EM (`96.0%` tok_acc, `98.9%` capture)** on `Qwen3-1.7B` and **`80.0%` EM (`95.0%` tok_acc, `98.9%` capture)** on `Qwen3-8B`.
  - **EC-3 Deep $k=5$ Sequential Non-Numeric Symbolic Overrides:** **`80.0%` EM (`93.75%` tok_acc, `max_prefix_invariance_diff = 0.0`)** on `Qwen3-1.7B` and **`55.0%` EM (`90.89%` tok_acc, `max_prefix_invariance_diff = 0.0`)** on `Qwen3-8B`.
  - **EC-4 Ultra-Long `128K` (`512p`) & `256K` (`1,024p`) Live CUDA Profiling:** At `256K` tokens, Full KV OOMs (`28.0–36.0 GiB`), whereas 1-level Flat LSLC Net VRAM is `135.72 MiB` (`211.26×` below Full KV, `97.74 ms` Warm TTFT) on `Qwen3-1.7B` and `171.03 MiB` (`215.54×`, `453.23 ms` Warm TTFT) on `Qwen3-8B` — above Constant-$k=2$ Raw RAG (`57.1 MiB`), which the 2-level Chapter$\rightarrow$Page TOC (`≤ 32.9 MiB` analytical) resolves.
  - **EC-5 Context-Grounded Multi-Step Reasoning & Intrinsic General IQ Retention:** Multi-step context reasoning reaches **`100.0%` (`100%` retention vs. Full KV)** on `Qwen3-8B` (`66.7%` / `75%` retention on `1.7B`), while intrinsic general IQ retention is **`125.0%`** on `1.7B` and **`83.3%`** on `8B`.
  - **32K Methodological Transparency ([reviewer_rebuttal_summary.json](evals/m2_colab_l4/reviewer_rebuttal_summary.json)):** In v1 32K (`P=128`, 3 seeds, $N=10$ Exact + $10$ Semantic/seed), Raw-Text RAG received the exact same `route.deep_idx` from `QueryToTOCRouter` (`100%` top-2 recall on Exact NIAH, `40%` on Semantic NIAH = `70%` overall); teacher-forced `token_acc = 90.00% ± 0.00%` reflects a 1-token leading-whitespace offset (`f" {code_val}"` at position 0), so autoregressive **Exact Match (EM)** is the primary metric (`content_tok_acc`, `gen_char_f1`, `N=180`, and shared non-oracle Okapi BM25 are implemented in [scripts/colab_run_rebuttal_v2.py](scripts/colab_run_rebuttal_v2.py)).

---

## Colab Hardware Tiers (~25M–40M Trainable Params, Frozen LLM)

- **Colab T4 (`15 GB` VRAM):** Runs `Qwen/Qwen3-1.7B` in FP16 (`~7–9 GB` peak) or `Qwen/Qwen3-4B` in 4-bit NF4 frozen + gradient checkpointing (`~8–10 GB` peak).
- **Colab L4 (`24 GB` VRAM, `~1.94 CU/hr`):** Primary lossless BF16 training & evaluation workhorse across both `1.7B` and `4B`.
- **Colab A100 (`40 GB` VRAM, `~6.73 CU/hr`):** Optional accelerator for rapid hyperparameter sweeps and 32K systems profiling.

---

## Documentation, Reproducibility & Hugging Face Release Map

- [REPRODUCE.md](REPRODUCE.md) — **1-Command Reproduction Guide:** local unit tests (`12/12`), instant Living-Book demo (`scripts/demo_hf_inference.py`), analytical VRAM calculator, Colab L4 benchmark scripts (`M2`, `M3`, `M4`, `Hard Stress`, `Rebuttal v1/v2`), and Hugging Face weight export.
- [`NekoFi/LSLC-Qwen3-1.7B-Adapter`](https://huggingface.co/NekoFi/LSLC-Qwen3-1.7B-Adapter) ([hf_hub_release/README.md](hf_hub_release/README.md) & [hf_hub_release/config.json](hf_hub_release/config.json)) — **Hugging Face Hub Model Card & Adapter Package (`40.42M` params, `77.10 MiB` `bfloat16`)** with SafeTensors/PyTorch export via [scripts/export_hf_weights.py](scripts/export_hf_weights.py) and live inference demo via [scripts/demo_hf_inference.py](scripts/demo_hf_inference.py).
- [docs/paper_draft_lslc.tex](docs/paper_draft_lslc.tex) — **Academic Paper (`Qwen3-1.7B`, `4B`, `8B`):** RAG-centered evaluation & Constant-$k$ crossover math, 17 verified citations through Nov 2026 (`Memorilla`, `Models Take Notes at Prefill`, `GroupKV`, `LeanSparse`, `GLM-5.4 Loop Transformer`), and explicit methodological/audit disclosures.
- [docs/blog_lslc_netra_style.html](docs/blog_lslc_netra_style.html) & [docs/article_engineering_deep_dive.md](docs/article_engineering_deep_dive.md) — **Netra Runtime-Style Engineering Blog Post** with KaTeX underbraced derivations, custom inline SVG charts, Constant-$k$ RAG crossover analysis, and FAQ.
- [ARCHITECTURE.md](ARCHITECTURE.md) — Formal technical architecture specification, tensor shapes, and design rationale.
- [evals/m2_colab_l4/](evals/m2_colab_l4/) — Archived empirical JSON benchmark logs across `Qwen3-1.7B`, `Qwen3-4B`, and `Qwen3-8B`.
