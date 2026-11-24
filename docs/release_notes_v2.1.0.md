# LSLC `v2.1.0` Release Notes — Fair Decoding, Root-Cause A/B/C Diagnostics, and Honest Trade-Offs

**Release:** `v2.1.0` (`2026-11-24`)  
**Authors:** Ariel Fikru, Muhammadalif Ramadhan  
**Target Checkpoint:** [`NekoFi/LSLC-Qwen3-1.7B-Adapter`](https://huggingface.co/NekoFi/LSLC-Qwen3-1.7B-Adapter) (`40,420,550` trainable params, `80,861,684` bytes, `bfloat16`)  
**Checkpoint SHA-256:** `5bfc03fe9e6dcb6950176d8d11f13373c00722e74088681c1b0c0dc3ca3ed095`  
**Paper & Raw Artifacts:** [LSLC_2026_Fikru.pdf](LSLC_2026_Fikru.pdf) | [v2_complete_summary.json](../evals/m2_colab_l4/v2_complete_summary.json) | [v21_diagnostics.json](../evals/m2_colab_l4/v21_diagnostics.json)

---

> *Why this release exists (the short version):*  
> In `v2.0.0`, Raw-Text Chunk RAG showed `100.0%` teacher-forced content-token accuracy on `32K` Exact NIAH yet only `13.3%` Strict EM (`27.8%` Substring EM). During audit, we found an asymmetric early-stop bug in `greedy_generate_concise()` that penalized Raw-Text RAG whenever the frozen LLM emitted a leading newline before the answer. Instead of tuning around the artifact or hiding the gap, `v2.1.0` fixes the decoder symmetrically on the **exact same frozen adapter checkpoint**, separates canonical answer extraction from output formatting, runs an Oracle A/B/C diagnostic on $k=1 \dots 5$ symbolic edits plus a $P \in \{4, 128, 1024\}$ edit-scaling probe, and publishes the unvarnished numbers alongside a cleaner 3-table manuscript layout.

---

## 1. What Changed and Why (Engineering Rationale)

### 1.1 Fixing the Step-0 Newline Trap in `greedy_generate_concise()`
**Symptom in `v2.0.0`:** Given a 512-token raw Wikipedia chunk followed by `Answer:`, frozen `Qwen3-1.7B` frequently emitted `\n` as token 0 (`step_i == 0`) before writing the actual code (`3034-3582`) or a brief preamble (`"The exact code for API-KEY is 3034-3582."`). Because `greedy_generate_concise()` checked `if "\n" in piece: break` from step 0 without checking whether any non-whitespace content had been emitted, Raw-Text RAG terminated with an empty string—while LSLC (trained to emit `" 3034..."` without a leading newline) kept decoding.

**The `v2.1.0` fix ([`scripts/colab_run_v2_complete.py`](../scripts/colab_run_v2_complete.py#L174-L229)):**
1. Track `has_non_ws = False`; only set `has_non_ws = True` once `piece.strip()` is non-empty, and stop uniformly across all baselines and LSLC on `eos_token_id` or `if has_non_ws and ("\n" in piece or "." in piece): break` (up to `max_new_tokens = 24`).
2. Split evaluation into three distinct, deterministic metrics computed identically for every system (`compute_answer_metrics_v2()`):
   - **`extracted_em` (Primary Cross-System Metric):** Deterministic canonical answer extracted from the first non-empty sentence/line (`_extract_canonical_candidates()`), isolating semantic answer correctness from preamble phrasing.
   - **`strict_em` (Clause/Prefix EM):** Exact normalized match on either the first clause or the leading target-length token span (note: an output whose opening clause matches can still satisfy Clause/Prefix EM even if trailing tokens diverge, which is why `extracted_em` is the primary comparison metric).
   - **`substr_em` (Substring EM):** Normalized target containment in the generated response.

**Immediate Impact (`v2.0.0` $\rightarrow$ `v2.1.0`):**
- Raw-Text Chunk RAG Exact `extracted_em` jumped from `27.8%` (`v2.0.0` Substr) to **`93.3%`** at `32K` and **`90.0%`** at `128K` (finally matching its `100.0%` TF content-token accuracy!), while its Clause/Prefix EM remained `13.3%`–`16.7%` strictly due to preamble formatting.
- LSLC's own `32K` Exact `extracted_em` rose from `31.1%` $\rightarrow$ **`61.1%`** (Pure GPU Router) and `45.6%` $\rightarrow$ **`80.0%`** (Shared BM25), and `k3_paraphrased_nl` LSLC Strict/Extracted EM jumped from `8.3%–10.0%` $\rightarrow$ **`30.0%–35.0%`** (`38.3%` Substring EM) once premature period-less/newline halts were resolved.

### 1.2 Checkpoint Isolation (`--load-checkpoint`)
Rather than re-training Stage 1 and introducing run-to-run weight variance, all `v2.1.0` evaluations loaded the **exact `v2.0.0` adapter weights** (`lslc_qwen3_1.7b_adapter.safetensors`, SHA-256 `5bfc03fe9e6dcb6950176d8d11f13373c00722e74088681c1b0c0dc3ca3ed095`, `strict=True`) via `--load-checkpoint`. Every delta between `v2.0.0` and `v2.1.0` therefore comes purely from symmetric decoding and metric isolation.

### 1.3 Separating the `32K` Quality Threshold from the `~100K` VRAM Crossover
In `v2.0.0`, our text paired the 2-level Chapter$\rightarrow$Page hierarchy mainly with contexts above `~100K` tokens. The `32K` ablation proves why that conflated two separate phenomena:
- **VRAM Crossover (`P* = 384.3` pages / `~98.4K` tokens on `Qwen3-1.7B`):** Where 1-level Flat LSLC's `O(P)` TOC KV-cache overtakes Constant-`k=2` Raw-Text RAG (`58.3–58.9 MiB`).
- **Attention-Dilution Quality Threshold (`>= 32K` tokens):** At `32K` (`P=128`), pruning active TOC tokens from `218` down to `133` eliminates attention dilution over Deep Read tokens and lifts LSLC Semantic `extracted_em` from `5.6%` $\rightarrow$ **`77.8%`** (`35.6%` $\rightarrow$ **`100.0%`** Substring EM).
- **Attribution (`paired_diff_hier_minus_trunc_flat`):** At equal token budgets (`133` at `32K`, `158` at `128K`), `2-Level Hier.` matches `Truncated Flat Control` on Exact NIAH (`+0.0% ± 0.0%`) and `128K` Semantic NIAH (`+0.0% ± 0.0%`), and adds a modest `+2.2% ± 2.2%` (seed 95% CI; `±4.4%` episode CI) Semantic Extracted EM edge at `32K` while retaining coarse chapter coverage across `100%` of the book.

### 1.4 Oracle A/B/C Root-Cause Diagnostic on $k=1 \dots 5$ Symbolic Overrides (`v21_diagnostics.json`)
On $k=5$ symbolic overrides, Raw-Text `Direct Text-Patch` & `LLM-Rewrite` reach `90.0%` `extracted_em` while LSLC achieves `40.0%` (E2E GPU) / `48.3%` (Shared BM25). To isolate whether the gap comes from routing, compression, or `DeltaUpdater` drift, we evaluated `N=30` episodes (`seed=42`) supplying the **oracle target page (`page 1`)** across steps $t=1 \dots 5$:

| Edit Step | Cond. A: Oracle Raw Text (`Extr. EM` / `Tok Acc`) | Cond. B: Oracle Re-Encoded LSLC (`Extr. EM` / `Tok Acc`) | Cond. C: Oracle Sequential `DeltaUpdater` (`Extr. EM` / `Tok Acc`) | Mean Gate $\bar{g}$ |
| :---: | :---: | :---: | :---: | :---: |
| **$k=1$** | `90.0%` (`74.7%`) | `0.0%` (`25.9%`) | `10.0%` (`70.6%`) | `0.2007` |
| **$k=2$** | `86.7%` (`76.3%`) | `0.0%` (`30.6%`) | `30.0%` (`84.4%`) | `0.2501` |
| **$k=3$** | `93.3%` (`76.5%`) | `0.0%` (`32.1%`) | `26.7%` (`80.6%`) | `0.2500` |
| **$k=4$** | `93.3%` (`75.9%`) | `0.0%` (`34.0%`) | `40.0%` (`82.8%`) | `0.2499` |
| **$k=5$** | `90.0%` (`75.8%`) | `0.0%` (`27.8%`) | `23.3%` (`76.0%`) | `0.2499` |

**Engineering Takeaway:** Router error accounts for only `8.3 pts` (`40.0%` $\rightarrow$ `48.3%`). Strikingly, **Condition B (re-encoding the verified page from scratch) scores `0.0%`** (`25.9%–34.0%` TF tok acc) because base Stage 1 trained `LayeredLatentEncoder` only on NIAH codes/facts + static policy, never on `DEPLOYMENT STATE INITIAL : <symbolic>` templates; meanwhile `DeltaUpdater` (Condition C) carries the entire `10.0%–40.0%` readout via delta training + `SparseExactBank` injection, yet fluctuates across multi-step updates. This pinpoints the exact path for future hardening: train the base encoder/projector on diverse symbolic/NL state schemas first, then distill `DeltaUpdater` against Condition B.

### 1.5 Book-Size Edit Scaling (`P ∈ {4, 128, 1024}`) & `z_book.clone()` Disclosure (`v21_diagnostics.json`)
`DeltaUpdater.update_book_page()` calls `z_book.clone()` so training remains autograd-safe. Measuring synchronized L4 latency and peak CUDA allocation delta across $P \in \{4, 128, 1024\}$ (`1K` to `256K` tokens, known target page address):

| Book Size | `z_book` Tensor | Autograd-Safe `.clone()` Latency / Peak $\Delta$ | Zero-Copy Inference Slice Latency / Peak $\Delta$ |
| :---: | :---: | :---: | :---: |
| `P=4` (`1K`) | `0.047 MiB` | `7.572 ms` / `11.874 MiB` | **`4.484 ms`** / `11.631 MiB` |
| `P=128` (`32K`) | `1.500 MiB` | `7.561 ms` / `13.327 MiB` | **`4.469 ms`** / `12.459 MiB` |
| `P=1024` (`256K`) | `12.000 MiB` | `7.722 ms` / `23.827 MiB` (`+12 MiB` clone) | **`4.506 ms`** / **`11.584 MiB` (`O(1)` in $P$)** |

### 1.6 Explicit Honest Trade-Offs & Audit Disclosures
1. **Long-Context Retrieval Ceiling:** On `128K` NIAH, Raw-Text Chunk RAG (`512` active tokens) achieves `90.0%` Exact Extracted EM and `100.0%` Semantic Substring EM vs. LSLC + BM25 (`70.0%` Exact Extracted EM / `70.0%` Semantic Substring EM) at `2.32×`–`3.58×` larger active+store VRAM (`58.31 MiB` vs. `25.09 MiB` / `16.17–17.81 MiB`).
2. **Natural-Language Editing vs. Simplest Mutable RAG:** On paraphrased NL edits (`N=90`), while rigid regex patching collapses (`0.0%`), `Append-Errata + BM25` reaches `40.0% ± 8.7%` Extracted EM vs. LSLC's `30.0%–35.0%` (a 5-point difference = `3/60` override queries). LSLC's clear editability advantage currently sits on **structured numeric overrides** (`94.4%–95.8%` vs. `52.8%` best RAG at `8.85 ms/edit` with `97.2%` static $\ell=0$ retention and $\max|\Delta_{\text{TOC}}|=0.0$).
3. **Static Retention & 1D Ablation Caveats:** In Paper 1's override benchmark, static facts reside on fixed page 0 (`"dual signoff"`), edits target page 1, and 4 paraphrase templates are shared across train/eval; likewise, our `1D-Flat` and `1D-3Layer` controls disable `FactorizedRoPE3D`, `Conv2d`, and the spatial prior simultaneously (testing the full 3D package rather than `FactorizedRoPE3D` alone).

---

## 2. Before / After Delta Table (`v2.0.0` $\rightarrow$ `v2.1.0`, Same Checkpoint on NVIDIA L4 `BF16`)

| Benchmark & System | `v2.0.0` (Clause/Substr) | **`v2.1.0` (Extracted / Clause / Substr)** | Why It Changed |
| :--- | :---: | :---: | :--- |
| **32K Exact — Raw-Text RAG (`BM25`)** | `13.3% / 27.8%` (`100%` tok) | **`93.3%`** `[13.3% / 93.3%]` (`100%` tok) | Fixed step-0 `\n` early-stop + first-sentence extraction |
| **32K Exact — LSLC E2E GPU (`2-Lvl`)** | `31.1% / 61.1%` | **`61.1%`** `[61.1% / 61.1%]` | Symmetric non-whitespace stop rule |
| **32K Exact — LSLC + BM25 (`2-Lvl`)** | `45.6% / 82.2%` | **`80.0%`** `[80.0% / 81.1%]` | Symmetric non-whitespace stop rule |
| **32K Sem. — LSLC E2E GPU (`2-Lvl`)** | `74.4% / 97.8%` | **`77.8%`** `[74.4% / 98.9%]` | Symmetric non-whitespace stop + canonical extraction |
| **128K Exact — Raw-Text RAG (`BM25`)** | `16.7% / 30.0%` (`100%` tok) | **`90.0%`** `[16.7% / 90.0%]` (`100%` tok) | Fixed step-0 `\n` early-stop + first-sentence extraction |
| **128K Exact — LSLC + BM25 (`2-Lvl`)** | `43.3% / 73.3%` | **`70.0%`** `[70.0% / 70.0%]` | Symmetric non-whitespace stop rule |
| **$k=3$ Num. — LSLC (`E2E` / `BM25`)** | `94.4% / 95.8%` | **`94.4% / 95.8%`** (vs. `52.8%` App-Errata) | Preserved; RAG extracted scores clarified |
| **$k=5$ Sym. — LSLC (`BM25`) vs. RAG** | `43.3% / 46.7%` vs. `90.0%` | **`48.3%`** vs. **`90.0%`** (`DirectPatch` & `LLM-RW`) | Fair comparison + A/B/C diagnostic added |
| **$k=3$ Paraphrased NL — LSLC (`BM25`)** | `10.0% / 40.0%` | **`35.0%`** `[33.3% / 38.3%]` (vs. `40.0%` App-Errata) | Fixed premature stop; honest `Append-Errata` baseline highlighted |

---

## 3. Reproducing `v2.1.0` & Verification Ledger

```bash
# Local architectural, tensor-shape, gradient, and prefix-invariance verification (0 cloud cost)
/Users/ariel/workspace/l2d-extractor/.venv/bin/python -m unittest -v tests/test_lslc_shapes.py  # 14/14 PASS

# Full v2.1.0 multi-seed benchmark on Colab L4 using the exact published checkpoint
python scripts/colab_run_v2_complete.py \
  --model Qwen/Qwen3-1.7B \
  --load-checkpoint hf_hub_release/lslc_qwen3_1.7b_adapter.safetensors \
  --out-json evals/m2_colab_l4/v2_complete_summary.json

# Oracle A/B/C k=1..5 symbolic diagnostic + P={4,128,1024} edit-scaling probe
python scripts/colab_run_v21_diagnostics.py \
  --model Qwen/Qwen3-1.7B \
  --load-checkpoint hf_hub_release/lslc_qwen3_1.7b_adapter.safetensors \
  --out-json evals/m2_colab_l4/v21_diagnostics.json
```
