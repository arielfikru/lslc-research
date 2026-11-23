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

1. **Run the 10-test architectural & tensor-shape suite:**
   ```bash
   python -m unittest -v tests/test_lslc_shapes.py
   ```
   Verifies:
   - `[B, P, 3, 128, 4, 4]` book tensor geometry (`12 KiB/page` FP16)
   - `FactorizedRoPE3D` rotary invariance and `PageRoPE1D` modularity
   - `SparseExactBank` strict `8 bytes/token` discrete storage (`128 B/page`)
   - `DeltaUpdater` convex override on $\ell=2$ with strict $\max|\Delta_{\text{TOC}}| = 0.0$ across $\ell \in \{0, 1\}$
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

## 3. Reproducing the GPU Benchmarks (Single NVIDIA L4 22.03 GiB or A100)

All empirical JSON outputs from our NVIDIA L4 runs are pre-archived in [`evals/m2_colab_l4/`](evals/m2_colab_l4/):

| Script | What It Reproduces | Archived Verified Output |
| :--- | :--- | :--- |
| `python scripts/colab_run_m3.py` | **Primary `Qwen/Qwen3-1.7B` Benchmark:** Tracks `0A` (Full BF16 KV), `0B` (INT4 KV), `0C` (StreamingLLM-128), `0D` (H2O-20%), `0E` (1D Flat Latent `12 KiB/p`), `1` (LSLC Dense Only), `2` (LSLC + `SparseExactBank`), `3` (16-Page `4K` Book), `4` (`DeltaUpdater` $k=3$), and live CUDA `4K/32K` VRAM & TTFT profiling. | [`evals/m2_colab_l4/m3_summary.json`](evals/m2_colab_l4/m3_summary.json) |
| `python scripts/colab_run_m4.py` | **Equal-Precision `Qwen/Qwen3-4B` Scaling Validation:** Tracks `0A–0D`, `1`, `2`, `4`, and live CUDA `4K/32K` VRAM & TTFT profiling (`144 KiB/tok` FP16 KV). | [`evals/m2_colab_l4/m4_qwen3_4b_summary.json`](evals/m2_colab_l4/m4_qwen3_4b_summary.json) |
| `python scripts/colab_run_hard_stress.py` | **5-Regime Adversarial Stress Suite (`Qwen3-1.7B` BF16 & `Qwen3-8B` 4-bit NF4):** (EC-1) 8-page multi-file Python codebases; (EC-2) 4-key adversarial decoys; (EC-3) deep $k=5$ symbolic state overrides; (EC-4) live CUDA `128K` (`P=512`) & `256K` (`P=1,024`) profiling; (EC-5) multi-step context reasoning. | [`evals/m2_colab_l4/hard_stress_summary.json`](evals/m2_colab_l4/hard_stress_summary.json) |
| `python scripts/colab_run_reviewer_rebuttal.py` | **Multi-Seed (`seeds={42,1337,9999}`) Rebuttal Suite:** Head-to-head `LSLC In-Place DeltaUpdater` vs. `Raw-Text Chunk RAG (Immutable & Append-Errata)` on $k=3$ numeric and $k=5$ symbolic overrides + `32K` (`P=128`) NIAH, router recall, and 128-page batch encoding latency. | [`evals/m2_colab_l4/reviewer_rebuttal_summary.json`](evals/m2_colab_l4/reviewer_rebuttal_summary.json) |

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
