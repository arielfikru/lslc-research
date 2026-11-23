# ARCHITECTURE.md — Formal RFC & Systems Design of `LSLC`

This document defines the complete Phase 1 mathematical formulation, exact tensor geometry, anti-collapse layer mechanics, exact-recall bypass, delta-update protocol, and end-to-end VRAM accounting for **Layered Spatial Latent Caching (LSLC)**.

> **Core Analogy:** Standard 1D KV-caches act like a **$10\,\text{km}$ stone scroll** (linear $O(N)$ scan per decode token, immutable to local revisions). **LSLC** transforms LLM context memory into an **editable, multi-resolution living book** stored as `Z in R^{B x P x L x C x H x W}` alongside a compact **Sparse Exact-Recall Footnote Bank**.

---

## 1. Frozen Backbone Dissection & the "GQA Trap"

Before designing a compression scheme, we must audit the actual byte cost of modern Grouped-Query Attention (GQA) KV-caches. Older LLMs with Multi-Head Attention inflated KV-cache sizes artificially; modern open-weight models already compress KV heads aggressively. LSLC targets pure dense GQA backbones so every baseline comparison (**FP16 KV**, **INT4 KV**, **StreamingLLM**, **H2O**) remains 100% apples-to-apples:

| Model | Role | Layers ($N_{\text{lay}}$) | `hidden_size` | Q / KV Heads | `head_dim` | Weights (FP16 / NF4) | FP16 KV / Tok | Ideal INT4 KV / Tok |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen/Qwen3-1.7B`** | **Primary MVE Backbone** | `28` | `2048` | `16` / **`8`** | `128` | `~3.4 GB` / `~1.2 GB` | **`112 KiB`** (`114,688 B`) | **`28 KiB`** |
| **`Qwen/Qwen3-4B`** | **Scaling Validation** | `36` | `2560` | `32` / **`8`** | `128` | `~8.0 GB` / `~2.8 GB` | **`144 KiB`** (`147,456 B`) | **`36 KiB`** |

### Why Newer Hybrid or Sliding-Window Models Are Excluded for Primary Benchmarks
- **`Qwen3.5` (2B/4B/9B):** Uses a **3:1 Gated DeltaNet (linear attention) + Full Attention** hybrid layout. Because 75% of layers lack a standard KV-cache, H2O, StreamingLLM, and standard INT4 KV baselines are confounded.
- **`Gemma 3 / Gemma 4`:** Combines alternating sliding-window attention with heavy multimodal towers and gated Hub access in Colab.
- **Implication of the GQA Trap:** At `112–144 KiB/tok` FP16 (`28–36 KiB/tok` INT4), any compression method using soft-prompt injection (`Mode A`) **must** keep active projected tokens well below `25%` of raw sequence length—otherwise active LLM KV-cache alone erases INT4 savings! LSLC's hierarchical Book Pyramid achieves `<2.5%` active tokens at 4K and `<0.7%` at 32K.

---

## 2. End-to-End Pipeline & Tensor Shapes

```text
Raw Context Stream X [B, N]
   │  (Partitioned into P = ceil(N / 256) pages, Tp = 256 tokens/page)
   ├──────────────────────────────────────────────────────────────────────┐
   ▼                                                                      ▼
Frozen LLM embed_tokens                                          SalienceGate (Entropy + Learned)
   E_p [B, 256, d_llm]                                             Selects top-rho (rho=6.25%, 16 tok/page)
   │                                                                      │
   ▼                                                                      ▼
LayeredLatentEncoder (Perceiver X-Attn + 3D Factorized RoPE)     Sparse Exact Footnote Bank
   Z_p [B, L=3, C=128, H=4, W=4]  <─────── 12 KiB/page FP16        (token_id:int32, page_id:uint16,
   │                                                                h:uint8, w:uint8) -> 8 B/tok (128 B/page)
   ├──────────────────────────┬──────────────────────────┐                │
   ▼                          ▼                          ▼                │
Level 0: TOC (1x1)     Level 1: Skim (2x2)     Level 2: Deep Read         │
  1 tok/page             4 tok/page              l=0(2x2) + l=1(2x2)      │
  (All P pages)          (Next k_skim pages)     + l=2(4x4) = 24 tok/page │
   │                          │                          │                │
   └──────────────────────────┴──────────┬───────────────┴────────────────┘
                                         ▼
              QueryToTOCRouter (Dot-Product Routing on TOC + Query)
                                         │
                                         ▼
           Layer-Ordered Soft-Prompt Prefix Injection (Mode A, Frozen LLM)
   [All TOC | Skim | Deep l=0 (Static) | Deep l=1 (Entity) | Deep l=2 (Delta) | Exact Index | Turn]
```

### Exact Module Tensor Contract (`Tp = 256`, `L = 3`, `C = 128`, `H = 4`, `W = 4`)
1. **Input Embeddings per Page:** `E_p: [B, 256, d_llm]` where `d_llm = 2048` (`Qwen3-1.7B`) or `2560` (`Qwen3-4B`).
   - Why `(H=4, W=4)`? The `256` tokens split into `16` ordered `16`-token discourse sub-spans laid out in row-major order across `(4, 4)` so 2D average pooling aggregates contiguous narrative blocks cleanly.
2. **`LayeredLatentEncoder`:**
   - Learnable spatial-layer queries `Q_book: [1, L*H*W, d_enc]` (`48` queries, `d_enc = 512`).
   - 2–4 Perceiver cross-attention blocks attending over `E_p` (`d_llm -> d_enc`), followed by 2 self-attention refinement blocks using **3D Factorized Spatial-Layer RoPE**, then bottleneck projection `d_enc -> C = 128`.
   - Output Page Book Tensor: **`Z: [B, P, 3, 128, 4, 4]`** in `fp16/bf16`.
   - Storage per page: `3 * 128 * 4 * 4 * 2 bytes = 12,288 bytes = 12 KiB/page` (**`48 bytes/raw token`**).
3. **`SpatialLatentProjector` (Multi-Resolution Pyramid):**
   - **Level 0 — TOC (`1x1`):** Pools strictly the consolidated structural layers `Z[:, :, :2]` (`l=0` Static + `l=1` Entity, `2*128 = 256` ch) via `AvgPool2d(4) -> MLP(256, d_llm)` -> **`[B, P, 1, d_llm]`** (`1 tok/page`). Excluding `l=2` guarantees bit-identical TOC invariance (`max_abs_diff == 0.0`) under sticky-note delta-updates.
   - **Level 1 — Skim (`2x2`):** Pools `Z[:, :, :2]` (`256` ch) via `AvgPool2d(2) -> [B, P, 4, 256] -> MLP -> [B, P, 4, d_llm]` (`4 tok/page`), likewise invariant to `l=2` edits.
   - **Level 2 — Deep Read (Layer-Ordered):**
     - `l = 0` (Static): `AvgPool2d(2)` -> `4 tok/page`
     - `l = 1` (Entity): `AvgPool2d(2)` -> `4 tok/page`
     - `l = 2` (Dynamic/Delta): full `4x4` -> `16 tok/page`
     - Total per Deep Read page = **`24 dense tokens/page`** (+ `16` sparse exact tokens when populated).
4. **`QueryToTOCRouter`:**
   - Computes scaled dot-product score between projected user query `W_q(q_embed)` and page TOC vectors `W_k(TOC_p)` with 1D page-index rotary bias `p`.
   - Assigns top-$k_{\text{deep}}$ pages to Level 2, next-$k_{\text{skim}}$ pages to Level 1, and remaining `P - k_deep - k_skim` pages to Level 0.
5. **Active Token Budgets Used in Memory Benchmarks:**
   - **At $4{,}096$ tokens ($P = 16$):** $12 \text{ TOC} \times 1 + 2 \text{ Skim} \times 4 + 2 \text{ Deep} \times 24 + 32 \text{ Exact} = \mathbf{100\text{ active tokens}}$.
   - **At $32{,}768$ tokens ($P = 128$):** $122 \text{ TOC} \times 1 + 4 \text{ Skim} \times 4 + 2 \text{ Deep} \times 24 + 32 \text{ Exact} = \mathbf{218\text{ active tokens}}$.

---

## 3. 3D Factorized Spatial-Layer RoPE

Inside `LayeredLatentEncoder` and `DeltaUpdater` self-attention, rotary embeddings encode each latent token's 3-tuple coordinate $(\ell, h, w)$ where $\ell \in \{0, 1, 2\}$, $h \in \{0, 1, 2, 3\}$, and $w \in \{0, 1, 2, 3\}$. Splitting rotary head dimension $d_{\text{rot}}$ into three orthogonal sub-bands $(d_\ell, d_h, d_w)$:
$$\mathbf{R}(\ell, h, w) = \mathrm{diag}\big(\mathbf{R}_{d_\ell}(\ell),\; \mathbf{R}_{d_h}(h),\; \mathbf{R}_{d_w}(w)\big).$$
- **Why factorize $(\ell, h, w)$ instead of flattening to $0..47$?** Factorized 3D RoPE keeps relative vertical/horizontal distances between semantic spans $(h, w)$ invariant across layers $\ell$, while keeping cross-layer interaction distance identical at every spatial patch $(h, w)$.
- **Why keep Page Index $p$ outside `(H, W)`?** Encoding $p$ only in `QueryToTOCRouter` (1D RoPE over $p$) ensures every page $[3, 128, 4, 4]$ is positionally self-contained: new pages append at $P \to P+1$ and individual pages can be updated in-place without re-rotating older pages.

---

## 4. Preventing Layer Collapse & Enabling In-Place Delta-Updates

A naive `[B, L, C, H, W]` tensor trained purely with next-token prediction will collapse into an unspecialized 3-channel mix. LSLC enforces **structural asymmetry** across layers:

| Layer Index | Semantic Role | Write / Update Rule | Deep Read Resolution | Prefix KV-Cache Impact |
| :--- | :--- | :--- | :---: | :--- |
| **`l = 0`** | **Static Rules & Core Facts** | Initial write at page creation; write-locked / conservative decay thereafter | `2x2` (`4 tok/page`) | Always cached ahead of `l=1,2` |
| **`l = 1`** | **Entity & Relation State** | Key-addressed gated overwrite for mutable entities/variables | `2x2` (`4 tok/page`) | Invalidated only if `l=1` changes |
| **`l = 2`** | **Dynamic Working Memory & Sticky-Notes** | Fast sparse-gated residual delta-patch `G_p * dZ_p` | `4x4` (`16 tok/page`) | Placed last among dense layers |

### In-Place Sticky-Note Delta-Update Equation
When a new user turn revises context relevant to page $p$ (e.g., *"Actually, change the server port from 8080 to 9443"*):
1. `DeltaUpdater` cross-attends the revision turn $X_\Delta$ against $\vect{Z}_p$ and outputs a candidate patch $\Delta\vect{Z}_p \in \mathbb{R}^{128 \times 4 \times 4}$ and spatial gate $\vect{G}_p = \sigma(\cdot) \in [0, 1]^{4 \times 4}$.
2. Page $p$'s dynamic layer updates strictly in-place:
   $$\vect{Z}_{p, 2}^{\text{new}} = \vect{Z}_{p, 2}^{\text{old}} + \vect{G}_p \odot \Delta\vect{Z}_p, \qquad \mathcal{L}_{\text{sparse}} = \lambda_{\text{sp}} \|\vect{G}_p\|_1.$$
3. **Layer-Ordered Prefix KV-Cache Synergy (Mode A):** Because LLM `inputs_embeds` follows strict volatility order:
   `[All TOC | Skim | Deep Static (l=0) | Deep Entity (l=1) | Deep Dynamic (l=2) | Sparse Exact Index | Turn]`
   patching $\ell=2$ on the active Deep Read page preserves 100% of the LLM's cached K/V states for TOC, Skim, $\ell=0$, and $\ell=1$!

---

## 5. Solving the Exact-Recall Bottleneck

Information theory dictates that compressing $256$ tokens into $4{,}096\text{--}12{,}288$ bytes of continuous latents cannot reliably preserve high-entropy verbatim strings (UUIDs, port numbers, passwords, function identifiers). Rather than overloading dense layers:
1. **`SalienceGate`:** Combines token embedding salience and token surprisal $s_i = \sigma(\vect{w}_s^\top \vect{e}_i + \gamma H_{\text{surprisal}}(x_i))$ to pick the top-$\rho$ ($\rho \approx 6.25\%$, i.e., **16 tokens/page**) verbatim tokens.
2. **Ultra-Compact Record Format (`8 bytes/token`):**
   - `token_id`: `int32` (`4 B`)
   - `page_id`: `uint16` (`2 B`)
   - `(h, w)`: `uint8 + uint8` (`2 B`)
   - **Total storage:** `16 * 8 = 128 bytes/page` (**`2.0 KiB`** for 4K tokens; **`16.0 KiB`** for 32K tokens).
3. **Zero-Copy $O(1)$ Reconstruction:** Only for pages selected into Deep Read, `embed_tokens[token_id]` is gathered directly from the frozen LLM's input table, added to a learned `(h, w)` coordinate embedding, and appended right after dense Deep Read tokens.

---

## 6. Architecture Decision Record: Mode A (Soft-Prompt) vs. Mode B (Cross-Attention)

| Criterion | Mode A — Layer-Ordered Soft-Prompt (MVE Default) | Mode B — Gated Cross-Attention Adapters (v2 Roadmap) |
| :--- | :--- | :--- |
| **Frozen LLM Modifications** | **Zero architectural edits** (`inputs_embeds` only) | Inserts trainable Flamingo-style X-Attn layers inside LLM |
| **MVE Parameter & Data Cost** | **~25M–40M params**; converges on toy/medium runs | Higher param count; requires larger training corpus |
| **Active LLM KV Cost** | Pays KV-cache for `100–218` active tokens | Zero KV-cache consumption for memory tokens |
| **Edit Independence** | Editing an older page $p < P$ invalidates later causal prefix KV | **100% patch-independent** across any page $p$ |
| **Decision** | **Lock Mode A for Phase 2–3 MVE & baselines** | **Document Mode B as v2 scaling upgrade** |

---

## 7. Exact End-to-End Memory Comparison & Colab Execution Matrix

All numbers below count **every single byte** resident in VRAM (Persistent Book + Sparse Index + Active LLM FP16 KV-Cache):

### End-to-End Memory Breakdown
- **4,096 Tokens (`P = 16` pages; Book `192 KiB` + Exact Bank `2 KiB` = `0.19 MiB`; `100` active LLM tokens):**
  - `Qwen3-1.7B`: Full FP16 KV = **`448.0 MiB`** | Ideal INT4 KV = **`112.0 MiB`** | H2O-20% (819 tok) = **`89.6 MiB`** | **LSLC Net Total = `11.13 MiB`** (`10.94 MiB` active KV + `0.19 MiB` book/index) $\rightarrow$ **`40.2×`** vs FP16, **`10.1×`** vs INT4.
  - `Qwen3-4B`: Full FP16 KV = **`576.0 MiB`** | Ideal INT4 KV = **`144.0 MiB`** | H2O-20% (819 tok) = **`115.2 MiB`** | **LSLC Net Total = `14.25 MiB`** (`14.06 MiB` active KV + `0.19 MiB` book/index) $\rightarrow$ **`40.4×`** vs FP16, **`10.1×`** vs INT4.
- **32,768 Tokens (`P = 128` pages; Book `1.50 MiB` + Exact Bank `16 KiB` (`0.02 MiB`); `218` active LLM tokens):**
  - `Qwen3-1.7B`: Full FP16 KV = **`3,584.0 MiB` (`3.50 GiB`)** | Ideal INT4 KV = **`896.0 MiB`** | **LSLC Net Total = `25.36 MiB`** (`23.84 MiB` active KV + `1.50 MiB` book + `0.02 MiB` index) $\rightarrow$ **`141.3×`** vs FP16, **`35.3×`** vs INT4.
  - `Qwen3-4B`: Full FP16 KV = **`4,608.0 MiB` (`4.50 GiB`)** | Ideal INT4 KV = **`1,152.0 MiB` (`1.125 GiB`)** | **LSLC Net Total = `32.17 MiB`** (`30.66 MiB` active KV + `1.50 MiB` book + `0.02 MiB` index) $\rightarrow$ **`143.2×`** vs FP16, **`35.8×`** vs INT4.

### Google Colab Execution Matrix (~25M–40M Trainable Params, ~600 MB AdamW + Gradients)

| Colab GPU | VRAM | Supported Configuration | Estimated Peak VRAM | Recommended Usage |
| :--- | :---: | :--- | :---: | :--- |
| **T4 (Free/Standard)** | `15 GB` | `Qwen3-1.7B` (FP16 + GradScaler) or `Qwen3-4B` (4-bit NF4 frozen + grad checkpointing) | `~7–10 GB` | Accessible MVE prototyping & unit ablations |
| **L4 (`~1.94 CU/hr`)** | `24 GB` | `Qwen3-1.7B` & `Qwen3-4B` native **BF16** (no quantization) | `~6–13 GB` | **Primary quality-first training & benchmark tier** |
| **A100 (`~6.73 CU/hr`)** | `40 GB` | `Qwen3-1.7B` & `Qwen3-4B` BF16 high batch / 32K stress tests | `~10–18 GB` | Optional fast scaling sweeps & 32K profiling |
