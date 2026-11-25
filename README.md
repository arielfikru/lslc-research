# In-Place Span Overwrite vs. Append for Editable Compressed Context in Frozen LLMs (`LSLC v3.0.0`)

**Author:** Ariel Fikru (Independent Researcher)  
**Preprint (v3.0.0):** [docs/tmlr_submission/main_tmlr_preprint.pdf](docs/tmlr_submission/main_tmlr_preprint.pdf) · **DOI:** [`10.5281/zenodo.24473281`](https://doi.org/10.5281/zenodo.24473281)  
**CPU Metric Audit:** `python3 docs/reconcile_paper_metrics.py` (**110/110 checks passed**, pure Python stdlib, 0 GPU)

> **Supersede Notice (`v3.0.0`):** This release and manuscript supersede earlier `v1`/`v2` LSLC preprints under the same Zenodo concept DOI after controlled causal ablations established that verbatim state-edit recovery was carried by the **discrete span-bank (`PathBDiscreteMemory`, denoted `OW-192`)** rather than the co-trained continuous latent delta (`LSLC-Full`). Earlier `v2.1.0` engineering notes and checkpoints remain archived in the repository for full provenance.

---

## Overview & Three Scoped Findings

We conduct a controlled study across two strictly frozen BF16 Transformer backbones (**`Qwen/Qwen3-1.7B`** and **`unsloth/Meta-Llama-3.1-8B-Instruct`**, `r = 0`, `max_frozen_param_diff = 0.0`) comparing:
1. **`PathBDiscreteMemory` (`OW-192`)** — a lean gated discrete span-bank (`326,407` trained parameters on `Qwen3-1.7B`, `594,695` on `Llama-3.1-8B-Instruct`; `0` latent encoder, `0` continuous $\mathbf{Z}_{\text{book}}$) using frozen `BAAI/bge-small-en-v1.5` 4-span routing (`3,072 B/page`), contiguous $3\times 16$-token `SalienceGate` extraction (`192 B/page`), mid-layer `ContextualEditGateHead` (`h = 64`), span-aligned selective overwrite, and partitioned two-bank ordering (`edited_last`).
2. **Controlled baselines** — fixed-budget rolling-tail append (`App-192`), gated raw-text append (`Raw-App`), and schema-aware rule-based raw-text overwrite (`Raw-OW`).
3. **Short-budget co-trained continuous latent prefixes (`LSLC-Full`)** — `400+250+400 = 1,050` post-adapter steps atop the released base adapter on `Qwen3-1.7B` (`41.51M` adapter params) and a `120`-step cross-family probe on `Llama-3.1-8B-Instruct`.

### 1. Controlled State Revision (`Gate-2` $2\times 4$ Locked-Page Factorial, $N = 200$/Backbone)

Conditional on shared oracle edit page addresses $p_i$, identical edit-gate decisions (`0.0%` FAR/FRR), and `4` pages locked by `OW-192`'s post-edit router (`192` active memory tokens excluding prompt/query):

| Write Policy (Locked 4 Pages) | Qwen3 `ed_last` Norm (W2N) | Qwen3 `id_asc` Norm (W2N) | Llama-8B `ed_last` Norm (W2N) | Llama-8B `id_asc` Norm (W2N) | Mem. Tok |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Raw-App` (`raw_text_append`) | `48.0%` (`54.0%`) | `45.5%` (`52.0%`) | `70.5%` (`76.5%`) | `65.5%` (`72.0%`) | `1,059–1,062` |
| `Raw-OW` (schema-aware rule-based) | `71.0%` (`77.0%`) | `60.5%` (`66.5%`) | **`89.0%`** (**`95.0%`**) | **`85.5%`** (**`92.0%`**) | `1,029–1,030` |
| `App-192` (`31`-tok rolling tail) | `59.5%` (`64.5%`) | `55.5%` (`60.5%`) | `76.0%` (`80.5%`) | `70.0%` (`74.5%`) | `192` |
| **`OW-192`** (`PathBDiscreteMemory`) | **`74.5%`** (**`79.5%`**) | **`66.5%`** (**`71.5%`**) | `85.5%` (`90.0%`) | `79.0%` (`83.5%`) | **`192`** |

- **Overwrite vs. Rolling-Tail Append (`OW-192` vs. `App-192`):** `+15.0 / +11.0 pp` on `Qwen3-1.7B` (`p_Holm <= 6.78e-4`, cluster sign-flip `p_sf,H <= 0.015`) and `+9.5 / +9.0 pp` on `Llama-3.1-8B-Instruct` (`p_Holm <= 3.73e-3`; cluster sign-flip `p_sf,H = 0.126–0.130` on single-seed Llama). Two disjoint fresh `Qwen3-1.7B` seeds (`2030+2031`, `N = 400`) replicate the exact `+15.0 pp` margin (`77.8%` vs. `62.8%`, `66–6`, `p = 7.26e-14`, cluster 95% CI `[+9.3%, +20.8%]`).
- **Update-Aware Append Prompt Control (`v3.3`, `N = 200`/backbone, `0` locked-page mismatches):** Adding an explicit recency rule (*"later updates supersede earlier records; always answer using the latest approved update"*) across `Raw-App`, `App-192`, and `OW-192` lifts append by `+3.5–5.5 pp` on `Qwen3-1.7B` — leaving `OW-192` decisively ahead (`+13.5 / +7.5 pp` vs. `App-192`, `p_Holm <= 0.012`; `+23.0 / +17.0 pp` vs. `Raw-App`, `p_Holm <= 9.73e-8`). On instruction-tuned `Llama-3.1-8B-Instruct`, explicit prompting recovers `+7.0 / +13.0 pp` on `Raw-App` and `+5.0 / +8.5 pp` on `App-192`, narrowing `OW-192` vs. `App-192` to `+4.5 / +3.5 pp` (non-significant after Holm correction) while `OW-192` retains `+8.0 pp` over update-aware `Raw-App` under Edited-Last (`p_Holm = 0.0150`) and `Raw-OW` still leads update-aware `Raw-App` by `+7.0–11.5 pp`.
- **Backbone-Divergent Outcome vs. Schema-Aware `Raw-OW`:** On `Qwen3-1.7B`, `OW-192` scores `+3.5` to `+6.0 pp` higher (`p_Holm >= 0.091`, n.s.). On `Llama-3.1-8B-Instruct`, **`Raw-OW` outperforms `OW-192`** by `-3.5 / -6.5 pp` Norm-EM (CI `[-13.5%, -0.9%]` `id_asc`), `-5.0 / -8.5 pp` W2N-EM (`p_Holm = 0.0116`, CI `[-16.0%, -2.6%]`), and `90% vs. 99%` (`78% vs. 91%` `id_asc`) on mutable queries.
- **Blind Edit-Page Routing (`0` oracle $p_i$):** `91.81%` turn routing / `70.6%` E2E (`76.7%` W2N) on `Qwen3-1.7B` (`N = 180`), and `94.17%` turn routing / `90.0%` E2E (`93.3%` W2N) on `Llama-3.1-8B-Instruct` (`N = 60`).
- **Write Latency Amortization:** Mid-layer edit gating adds `~172–186 ms` per `3–5`-turn episode, yielding an estimated break-even of **`>= 4` queries/episode** on `Llama-3.1-8B-Instruct` (`3.17` queries) and no amortization on `Qwen3-1.7B` at `4K` sequence length.

### 2. Task-Dependent Diagnostics of Short-Budget Latent Prefixes (`LSLC-Full`)

| Diagnostic Probe | Bank-Only / Exact-Only | With Latent Prefix (`ΔZ` / `LSLC-Full`) | Takeaway |
| :--- | :---: | :---: | :--- |
| **Masked-Old Verbatim (`N = 24`, Qwen3)** | `24/24` (`100.0%`) | **`0/24`** (`0.0%`) | Frozen decoder outputs `0/24` verbatim edits without bank updates |
| **Masked-Old Verbatim (`N = 24`, Llama-8B Value-EM)** | `24/24` (`100.0%`) | **`0/24`** (`0.0%`) | Replicated across model families |
| **Locked-Page `65`-File Coding (`N = 24`, Qwen3)** | `20/24` (`83.3%`) | **`1/24`** (`4.2%`, `p = 3.81e-6`) | Severe syntax/API collapse when adding $\mathbf{Z}_{\text{book}}$ |
| **`4K` `WikiText-2` Continuation PPL (Qwen3 / Llama-8B)** | `25.91` / `14.08` | **`39.86`** / **`17.89`** | Consistent continuation PPL degradation |
| **Legacy Suffix-Overwrite `Gate-2` (`N = 600`, Qwen3)** | `351/600` (`58.5%`) | **`383/600`** (`63.8%`, `p = 2.56e-4`) | Modest positive assist under legacy suffix corruption (still `12.8 pp` below `OW-192`'s `76.7%`) |

### 3. Empirical & Systems Boundaries

- **`RULER-style` `16K/32K` (`N = 60`/length):** `OW-192` (`192` memory tokens, `171×` compression vs. `32K` full context) scores `68.3% / 65.0%` on `Qwen3-1.7B` and `71.7% / 68.3%` on `Llama-3.1-8B-Instruct`, matching `bm25_rag_k1` (`66.7%`) while speeding up `32K` Llama-8B inference by `10.4×` cold (`392.3` vs. `4,085.8 ms`) and `17.9×` warm (`227.7 ms`).
- **`4K` KV-Eviction Pareto Frontier (`N = 48` QA + `WikiText-2` PPL):** Per-head `SnapKV-style` eviction remains stronger in PPL (`19.41 / 8.11` vs. `25.91 / 14.08`) and `8B` QA (`81.3%` vs. `72.9%–77.1%`) when per-query prefill is available, whereas `OW-192` encodes offline once in `~29 ms` and saves `1.70–1.71 GiB` inference VRAM vs. `full_kv_bf16`.
- **Cross-File Python Coding (`65`-File Repo, `N = 24`):** `OW-192` preserves `22/24` (`91.7%` Qwen3) and `21/24` (`87.5%` Llama-8B) at `192` tokens (`1.76×–1.85×` fewer tokens than matched raw text `24/24` / `22/24`), avoiding the `1/24` latent collapse.
- **Partitioned Two-Bank KV Reuse (`t1 -> t2`):** Verifies bit-exact `0.0` Bank-1 KV drift across `65.1%–66.5%` of prefix tokens across all `18` tested configs (`P in {16, 64, 128}`), but runs at **`0.82×–0.95×` speed (`5%–22%` slower)** with BF16 logit drift (`0.156–1.000`) under eager HuggingFace `DynamicCache`.
- **`LongMemEval` `knowledge-update` (`N = 72`, Static Read-Only Mode, `0` Edit Calls, `gold_prioritized_64p`):** Dropping `81.25%` of tokens per page (`48/256`) lowers the bag-of-words lexical-coverage proxy from `68%–79%` to `25%–39%` and trails raw-text RAG on `Qwen3-1.7B` (`9.7%–15.3%` vs. `26.4%` `extracted_em`), while scoring `23.6%–31.9%` vs. `31.9%–37.5%` on `Llama-3.1-8B-Instruct` (`d6` ties `bge_4span_rag_k2` at `23/72 = 31.9%`, `11–11` discordant, CI `[-12.5, +12.5] pp`).

---

## Exact Parameter & Storage Accounting (`PathBDiscreteMemory` / `OW-192`)

| Quantity | `Qwen/Qwen3-1.7B` | `unsloth/Meta-Llama-3.1-8B-Instruct` |
| :--- | :---: | :---: |
| Frozen LLM parameters (`max_frozen_param_diff = 0.0`) | `1,720,574,976` | `8,030,261,248` |
| Frozen `BAAI/bge-small-en-v1.5` router parameters | `33,360,000` | `33,360,000` |
| Trained `SalienceGate` + `ContextualEditGateHead` (`h = 64`) | **`326,407`** (`0.019%`) | **`594,695`** (`0.0074%`) |
| Continuous latent encoder / `DeltaUpdater` / `hw_coord` | `0` | `0` |
| Checkpoint size (`.safetensors`) | `1.25 MiB` | `2.27 MiB` |
| Storage per `256`-token page (pre-edit $\rightarrow$ post-edit `base` clone) | `3,264 -> 3,456 B/p` | `3,264 -> 3,456 B/p` |
| Reference: Bare UTF-8 / BGE-4span RAG / FP16 KV | `1,083` / `4,155 B/p` / `28.0 MiB/p` | `1,138` / `4,210 B/p` / `32.0 MiB/p` |

---

## Reproducibility & Repository Map

### Instant Local CPU Verification (0 GPU)
Verify all tables, figures, exact unrounded Holm $p$-values, cluster-bootstrap CIs, exact cluster sign-flip sensitivity checks, and legacy W2N rescoring directly against the archived JSON logs:
```bash
python3 docs/reconcile_paper_metrics.py
```

### Key Artifacts & Scripts
- **Official Preprint (`v3.0.0`, `tmlr.sty`):** [docs/tmlr_submission/main_tmlr_preprint.pdf](docs/tmlr_submission/main_tmlr_preprint.pdf) · [LaTeX Source](docs/tmlr_submission/main_tmlr_preprint.tex) · [BibTeX](docs/tmlr_submission/references.bib)
- **Metric Reconciliation Script:** [docs/reconcile_paper_metrics.py](docs/reconcile_paper_metrics.py)
- **Core Implementation:** [src/lslc/](src/lslc/) (`PathBDiscreteMemory`, `SalienceGate`, `ContextualEditGateHead`, `SparseExactBank`, and diagnostic `LSLC-Full` modules)
- **Primary `v3.1` / `v3.2` Experiment Runners (GPU):**
  - [scripts/colab_run_v33_update_aware_append.py](scripts/colab_run_v33_update_aware_append.py) — Matched `2×3` update-aware append prompt control (`N=200`/backbone, `0` locked-page mismatches vs. `v3.2`)
  - [scripts/colab_run_v32_tmlr_controls.py](scripts/colab_run_v32_tmlr_controls.py) — Locked-page $2\times 4$ factorial (`LivingBook-Edit`), `LongMemEval` `N=72`, Two-Bank KV reuse, and `32K` unified systems profile
  - [scripts/colab_run_v3_gerbang6_llama8b_pathb.py](scripts/colab_run_v3_gerbang6_llama8b_pathb.py) — Dual-backbone `PathBDiscreteMemory` training, `Gate-2` E2E, `4K` KV-eviction Pareto, `RULER-style` `16K/32K`, `65`-file coding, and Llama-8B $\mathbf{Z}_{\text{book}}$ probe
  - [scripts/colab_run_v3_path_tiebreaker.py](scripts/colab_run_v3_path_tiebreaker.py) — `5`-cell iso-token masked-old edit probe & legacy `Gate-2` comparisons
- **Archived Empirical JSON Logs:** [evals/m2_colab_l4/](evals/m2_colab_l4/) (`17` primary `v3.0` paper artifacts + earlier `v2.x` provenance logs)
- **Unit Test Suite (`tests/`):** `99` unit tests covering tensor shapes, Path B discrete overwrite, two-bank ordering, and TMLR control harnesses
- **Hugging Face Base Adapter Archive:** [`NekoFi/LSLC-Qwen3-1.7B-Adapter`](https://huggingface.co/NekoFi/LSLC-Qwen3-1.7B-Adapter) ([hf_hub_release/README.md](hf_hub_release/README.md)) — archived `40.42M`-parameter `Qwen3-1.7B` base adapter used to initialize the `LSLC-Full` diagnostic baseline
- **Historical `v2.x` Engineering Archive:** [docs/release_notes_v2.1.0.md](docs/release_notes_v2.1.0.md), [ARCHITECTURE.md](ARCHITECTURE.md), [REPRODUCE.md](REPRODUCE.md)

---

## Citation

```bibtex
@article{fikru2026inplacespan,
  title={In-Place Span Overwrite vs. Append for Editable Compressed Context in Frozen LLMs, with Task-Dependent Diagnostics of Short-Budget Latent Prefixes},
  author={Fikru, Ariel},
  year={2026},
  month={November},
  note={Zenodo Preprint v3.0.0},
  doi={10.5281/zenodo.24473281},
  url={https://doi.org/10.5281/zenodo.24473281}
}
```
