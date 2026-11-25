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

# LSLC (`Qwen3-1.7B`) — Base Adapter Checkpoint (`LSLC-Full` Diagnostic Baseline)

> **Important `v3.0.0` Scientific Update (Superseding `v2.0.0` Claims):**  
> This repository archives the **`40.42M`-parameter (`77.12 MiB` `bfloat16`) base adapter checkpoint** (`lslc_qwen3_1.7b_adapter.safetensors`) for strictly frozen `Qwen/Qwen3-1.7B` (`r = 0`). In our controlled **`v3.0.0` study** ([Zenodo Preprint v3](https://doi.org/10.5281/zenodo.24473281) · [GitHub](https://github.com/arielfikru/lslc-research)), this checkpoint served as the initialization base for the `1,050`-step (`400+250+400`) `LSLC-Full` continuous latent prefix baseline.
>
> Controlled causal ablations established that **verbatim state-edit recovery is carried by the discrete span-bank, not the co-trained continuous latent delta ($\Delta\mathbf{Z}$)**:
> - **Masked-old verbatim probe (`N = 24`):** `0/24` edited values decoded without discrete bank updates on both `Qwen3-1.7B` and `Llama-3.1-8B-Instruct` (vs. `24/24` with bank updates).
> - **Cross-task penalties:** Adding the continuous latent book $\mathbf{Z}_{\text{book}}$ collapses locked-page `65`-file Python coding (`20/24 -> 1/24`, `p = 3.81e-6`) and degrades `4K` `WikiText-2` continuation PPL (`25.91 -> 39.86` on `Qwen3-1.7B`; `14.08 -> 17.89` on `Llama-3.1-8B-Instruct`).
> - **Recommended editable memory architecture:** For mutable compressed context atop frozen LLMs, use the lean discrete-only **`PathBDiscreteMemory` (`OW-192`)** (`326,407` trained params on `Qwen3-1.7B`, `594,695` on `Llama-3.1-8B-Instruct`, `0` latent encoder, `0` $\mathbf{Z}_{\text{book}}$) in [`arielfikru/lslc-research`](https://github.com/arielfikru/lslc-research).
>
> This checkpoint is preserved here for **full reproducibility of the `LSLC-Full` diagnostic baseline and earlier `v2.x` experiments**.

---

## Checkpoint Architecture (`lslc_qwen3_1.7b_adapter.safetensors`)

All base `Qwen/Qwen3-1.7B` weights remain **100% frozen** (`max_frozen_param_diff = 0.0`). This `bfloat16` SafeTensors package contains the `40.42M`-parameter `LayeredSpatialLatentCache` adapter suite:
- `LayeredLatentEncoder` (`FactorizedRoPE3D` mapping `256`-token pages to $\mathbf{Z}_p \in \mathbb{R}^{3 \times 128 \times 4 \times 4}$, `12 KiB/page` FP16)
- `SalienceGate` & `SparseExactBank` (discrete exact-recall token bank)
- `SpatialLatentProjector` (Matryoshka `1×1` TOC / `2×2` Skim / Deep Read projection)
- `QueryToTOCRouter` (1D page-index rotary routing + `SparseExactBank` `MaxSim`)
- `DeltaUpdater` (in-place gated convex updates on $\ell = 2$: $\mathbf{Z}_{p,2} \leftarrow (1 - \mathbf{G}_p)\odot\mathbf{Z}_{p,2} + \mathbf{G}_p\odot\Delta\mathbf{Z}_{p,2}$)

---

## `v3.0.0` Controlled Diagnostic Summary (`Qwen/Qwen3-1.7B`, BF16)

All numbers below reconcile directly against the archived JSON evaluation logs via `python3 docs/reconcile_paper_metrics.py` (`91/91 PASSED`) in [`arielfikru/lslc-research`](https://github.com/arielfikru/lslc-research):

| Evaluation | Bank-Only / Lean Discrete (`OW-192` or `exact_only`) | Co-Trained Latent Prefix (`LSLC-Full`, `1,050` post-adapter steps) | Reference Baseline |
| :--- | :---: | :---: | :---: |
| **Masked-Old Verbatim Edit Probe (`N = 24`)** | `24/24` (`100.0%`, `bank_ON`) | **`0/24`** (`0.0%`, `deltaZ_ON_bank_masked_old`) | `0/24` (`bank_OFF`) |
| **Locked-Page `65`-File Coding (`N = 24`, `pass@1`)** | `22/24` (`OW-192`) / `20/24` (`exact_only`) | **`1/24`** (`4.2%`, `p = 3.81e-6`) | `24/24` (`raw_text_same_pages_k4`) |
| **`4K` `WikiText-2` Continuation PPL (`N = 16`)** | `25.91` (`OW-192`, `192` mem tok) | **`39.86`** (`306` mem tok) | `13.79` (`full_kv_bf16`) / `19.41` (`SnapKV-style`) |
| **`4K` `RULER-style` QA (`N = 48`)** | `56.3%` raw-Q / `58.3%` short-Q (`OW-192`) | `50.0%` (`p = 0.45`, n.s.) | `72.9%` (`full_kv_bf16`) / `41.7%` (`SnapKV-style`) |
| **End-to-End `Gate-2` (`N = 600`, seeds `2029–2031`)** | **`76.7%`** (`OW-192`) / `58.5%` (legacy `exact_only`) | **`63.8%`** (`+5.33 pp` vs. legacy suffix `exact_only`, `p = 2.56e-4`) | `61.7%` (`App-192`) / `54.0%` (`bm25_rag_append`) |

> **Note on `v2.0.0` Metrics:** Earlier `v2.0.0` numbers previously displayed on this card reflected a pre-`v2.1.0` step-0 `\n` decoding asymmetry on Raw-Text RAG and have been replaced with the audited `v3.0.0` controlled results above.

---

## Quickstart Usage (Reproducing the Base Adapter)

```python
import torch
from safetensors.torch import load_file
from lslc import LSLCConfig, FrozenQwenLSLCWrapper, load_frozen_qwen3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = LSLCConfig.from_model_name("Qwen/Qwen3-1.7B")
llm, tokenizer = load_frozen_qwen3(cfg, device=device)
wrapper = FrozenQwenLSLCWrapper(cfg, llm)
wrapper.lslc.to(device)

# Load archived base LSLC adapter weights (40.42M params)
state_dict = load_file("hf_hub_release/lslc_qwen3_1.7b_adapter.safetensors", device=str(device))
wrapper.lslc.load_state_dict(state_dict, strict=False)
wrapper.lslc.eval()
```

Or run the self-contained inference demo directly from the repository:
```bash
python scripts/demo_hf_inference.py --checkpoint hf_hub_release/lslc_qwen3_1.7b_adapter.safetensors
```

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
