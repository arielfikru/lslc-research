# Reproducing `LSLC` (Layered Spatial Latent Caching)

This guide provides step-by-step instructions to reproduce every unit test, memory accounting table, baseline comparison, multi-seed state-override benchmark, and Hugging Face adapter weight export reported in the paper ([docs/paper_draft_lslc.tex](docs/paper_draft_lslc.tex)).

---

## 1. Environment Setup (< 2 Minutes)

```bash
git clone https://github.com/arielfikru/lslc-research.git
cd lslc-research
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[quant,dev]"
```

---

## 2. Instant Local Verification (CPU or Laptop GPU, 0 Cloud Cost)

1. **Run the 14-test architectural & tensor-shape suite:**
   ```bash
   python -m unittest -v tests/test_lslc_shapes.py
   ```
   Verifies:
   - `[B, P, 3, 128, 4, 4]` book tensor geometry (`12 KiB/page` FP16)
   - `FactorizedRoPE3D` rotary invariance and `PageRoPE1D` modularity
   - `SparseExactBank` strict `8 bytes/token` discrete storage (`128 B/page`) and right-of-header `SalienceGate` capture
   - `DeltaUpdater` pre-gate channel-wise RMS alignment and convex override on $\ell=2$ with strict $\max|\Delta_{\text{TOC}}| = 0.0$ across $\ell \in \{0, 1\}$
   - 2-level Chapter$\rightarrow$Page hierarchical TOC active-token cap, equal-token `flat_toc_max_tokens` control, in-VRAM `SparseExactBank` `MaxSim` routing (`0 B` CPU RAM), and external retriever `router_out_override`
   - Non-zero gradient flow across all 5 trainable LSLC modules under a frozen LLM

2. **Run the self-contained Living-Book inference & delta-override demo:**
   ```bash
   python scripts/demo_hf_inference.py --mock-llm
   ```

3. **Run the analytical VRAM & active-KV calculator across `4K`, `32K`, `128K`, `256K`, and `1M` tokens:**
   ```bash
   python scripts/profile_vram.py --context-lengths 4096 32768 131072 262144
   ```

---

## 3. Reproducing the GPU Benchmarks (Single NVIDIA L4 22.03 GiB or Tesla T4 14.56 GiB)

All empirical JSON outputs from our NVIDIA L4 (`bfloat16`) and Tesla T4 (`float16`) runs are pre-archived in [`evals/m2_colab_l4/`](evals/m2_colab_l4/):

| Script | What It Reproduces | Archived Verified Output |
| :--- | :--- | :--- |
| `python scripts/colab_run_m3.py` | **Primary `Qwen/Qwen3-1.7B` Benchmark (L4):** Tracks `0A` (Full BF16 KV), `0B` (INT4 KV), `0C` (StreamingLLM-128), `0D` (H2O-20%), `0E` (1D Flat Latent `12 KiB/p`), `1` (LSLC Dense Only), `2` (LSLC + `SparseExactBank`), `3` (16-Page `4K` Book), `4` (`DeltaUpdater` $k=3$), and live CUDA `4K/32K` VRAM & TTFT profiling. | [`evals/m2_colab_l4/m3_summary.json`](evals/m2_colab_l4/m3_summary.json) |
| `python scripts/colab_run_m4.py` | **Equal-Precision `Qwen/Qwen3-4B` Scaling Validation (L4):** Tracks `0A–0D`, `1`, `2`, `4`, and live CUDA `4K/32K` VRAM & TTFT profiling (`144 KiB/tok` FP16 KV). | [`evals/m2_colab_l4/m4_qwen3_4b_summary.json`](evals/m2_colab_l4/m4_qwen3_4b_summary.json) |
| `python scripts/colab_run_hard_stress.py` | **5-Regime Adversarial Stress Suite (`Qwen3-1.7B` BF16 & `Qwen3-8B` 4-bit NF4, L4):** (EC-1) 8-page multi-file Python codebases; (EC-2) 4-key adversarial decoys; (EC-3) deep $k=5$ symbolic state overrides; (EC-4) live CUDA `128K` (`P=512`) & `256K` (`P=1,024`) profiling; (EC-5) multi-step context reasoning. | [`evals/m2_colab_l4/hard_stress_summary.json`](evals/m2_colab_l4/hard_stress_summary.json) |
| `python scripts/colab_run_reviewer_rebuttal.py` | **3-Seed L4 BF16 Override & Batch-Encode Suite (`seeds={42,1337,9999}`):** Head-to-head `LSLC In-Place DeltaUpdater` vs. `Raw-Text Chunk RAG (Immutable & Append-Errata)` on $k=3$ numeric and $k=5$ symbolic overrides + `32K` NIAH and `128`-page encode latency. | [`evals/m2_colab_l4/reviewer_rebuttal_summary.json`](evals/m2_colab_l4/reviewer_rebuttal_summary.json) |
| `python scripts/colab_run_rebuttal_v2.py` | **Unified 3-Seed 2-Level Hierarchy & Editable-RAG Benchmark (T4 FP16):** Live CUDA `32K/128K/256K` 2-level hierarchy vs. Constant-$k=2$ RAG, $N=180$ shared-BM25 `32K` NIAH, and 4-way $k=3/k=5$ overrides with targeted & full-page In-Place LLM-Rewrite RAG latency. | [`evals/m2_colab_l4/rebuttal_v2_summary.json`](evals/m2_colab_l4/rebuttal_v2_summary.json) |
| `python scripts/colab_run_v2_complete.py --load-checkpoint hf_hub_release/lslc_qwen3_1.7b_adapter.safetensors` | **Complete `v2.1.0` Fair Multi-Seed Benchmark (L4 BF16, 3 Seeds `{42,1337,9999}`):** End-to-End Pure GPU Router (`0 B` additional CPU index) vs. Shared BM25 (`270–857 KiB` CPU) at `32K` ($N=180$) & `128K` ($N=60$) NIAH with equal-token Truncated Flat TOC control (paired 95% CIs); fair `has_non_ws` greedy decoding + 3-tier EM (`extracted_em`, `strict_em`, `substr_em`); 5-way State Overrides ($N=288$) across $k=3$ numeric (`94.4%–95.8%`), $k=5$ symbolic (`40.0%–48.3%`), and $k=3$ paraphrased NL (`30.0%–35.0%` vs. `40.0%` Append-Errata); plus controlled stored-param/iso-byte `1D-Flat` vs. `1D-3Layer` vs. `3D-Spatial` ablation. (Also supports `--skip-stage2`, `--skip-ablation`, `--prev-json`.) | [`evals/m2_colab_l4/v2_complete_summary.json`](evals/m2_colab_l4/v2_complete_summary.json) |
| `python scripts/colab_run_v21_diagnostics.py --load-checkpoint hf_hub_release/lslc_qwen3_1.7b_adapter.safetensors` | **`v2.1.0` Oracle A/B/C (`k=1..5` Symbolic) & `P={4,128,1024}` Edit-Scaling Diagnostics (L4 BF16):** Isolates routing vs. encoder compression (`Cond B`) vs. sequential `DeltaUpdater` drift (`Cond C`), and measures autograd-safe `.clone()` vs. zero-copy inference slice write latency (`4.47–4.51 ms`) and peak CUDA delta across `1K`–`256K` books. | [`evals/m2_colab_l4/v21_diagnostics.json`](evals/m2_colab_l4/v21_diagnostics.json) |

---

## 4. Exporting & Uploading Weights to Hugging Face Hub

To export the LSLC adapter weights (`40.42M` parameters, `77.09 MiB` in `bfloat16` SafeTensors format) into [`hf_hub_release/`](hf_hub_release/) and optionally push to Hugging Face Hub:

```bash
# 1. Export SafeTensors + verify round-trip load
python scripts/export_hf_weights.py --out-dir hf_hub_release

# 2. (Optional) Push directly to your Hugging Face Hub account
huggingface-cli login
python scripts/export_hf_weights.py --out-dir hf_hub_release --repo-id <your-hf-username>/LSLC-Qwen3-1.7B-Adapter --push
```
