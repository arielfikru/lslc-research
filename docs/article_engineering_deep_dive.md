# Why Your LLM’s KV Cache Needs to Be an Editable Book, Not a 10 km Stone Scroll

* **Category:** Guides / Engineering
* **Published:** Nov 23, 2026
* **Written by:** Ariel Fikru

---

This is the wall every long-context deployment hits right after solving model weight memory: you quantize an 8B or 4B model so it loads comfortably on a single GPU, you open the context window to 32K or 128K tokens for multi-turn coding agents, and within twenty turns your latency climbs past ten seconds while your VRAM fills with stale Key-Value (KV) states. Worse, the moment a user says *"wait—change the budget cap from 10M to 60M and move the staging endpoint to Frankfurt"*, you cannot edit turn 3 in place. In a standard causal KV cache, everything downstream of an edit goes stale: you either re-prefill the history or append conflicting errata that confuse the model.

So this guide derives what it takes to turn a flat, append-only 1D KV cache into an **editable, multi-resolution living book**—an architecture we call **Layered Spatial Latent Caching (LSLC)**. There are only three things in a compressed book's memory budget at inference time, and two of them cost pennies. The third—the active KV cache of the projected tokens you actually hand to the LLM—is the one that decides whether your compression ratio is real or marketing fiction. Every number below is derived from first principles and verified on a single NVIDIA L4 GPU (`22.03 GiB`) across frozen `Qwen3-1.7B`, `Qwen3-4B`, and `Qwen3-8B` backbones.

---

## TL;DR

* Even with Grouped-Query Attention (GQA), a standard FP16/BF16 KV cache costs **112 KiB per token** on `Qwen3-1.7B` and **144 KiB per token** on `Qwen3-4B` and `Qwen3-8B`. At 32K tokens that is **3.50 to 4.50 GiB** per stream; at 256K tokens it is **28 to 36 GiB**, which crashes a 24 GB card before you serve a second user.
* Most latent context compressors (`LCLM`, `Memorilla`) are **read-only** and lose exact arbitrary strings (UUIDs, ports, `snake_case` code symbols), while recent editable KV methods (*Models Take Notes at Prefill*) operate on **uncompressed** KV caches.
* **LSLC** compiles each 256-token page into a `[L=3, C=128, H=4, W=4]` 2D spatial-layer latent sheet (**12 KiB per page**, or **48 bytes per raw token**) plus a 16-slot **Sparse Exact-Recall Footnote Bank** (**8 bytes per token**, or **128 bytes per page**).
* Counting *both* the persistent book and the active LLM KV cache of the 218 routed tokens (`122 TOC + 16 Skim + 48 Deep + 32 Exact`), a 32K context drops from **3,585.8 MiB to 27.11 MiB (132.3×)** on `Qwen3-1.7B` and from **4,610.3 MiB to 34.42 MiB (133.9×)** on `Qwen3-4B`, cutting query Time-to-First-Token (TTFT) from **5.24–12.73 seconds** to **58–84 ms**.
* The real prize is **in-place editability without breaking prefix cache**: ordering prefix tokens by volatility (`[TOC | Skim | l=0 Static | l=1 Entity | l=2 Sticky-Note | Exact]`) lets an in-place convex `DeltaUpdater` overwrite $\ell=2$ across $k=3$ sequential turn revisions at **97.9%–100.0% Exact Match** (versus **0.0%** on immutable Raw-Text RAG and **52.1%** on append-errata RAG) with zero TOC prefix drift ($\max|\Delta_{\text{TOC}}| = 0.0$).

---

## The cache budget

When you replace a flat 1D KV cache with a routed latent book, everything resident on the GPU on top of the frozen model weights falls into three buckets:

$$
M_{\text{context}} = \underbrace{P \cdot (L \cdot C \cdot H \cdot W \cdot b_{\text{lat}})}_{\text{3D latent book}} + \underbrace{P \cdot (K_{\text{exact}} \cdot 8\text{ B})}_{\text{exact footnote bank}} + \underbrace{N_{\text{active}}(P) \cdot M_{\text{KV/tok}}}_{\text{active LLM KV cache}}
$$

Here $P = \lceil N / 256 \rceil$ is the number of 256-token pages, $[L, C, H, W] = [3, 128, 4, 4]$ is the 3D spatial-layer tensor per page stored at $b_{\text{lat}} = 2$ bytes (FP16/BF16), $K_{\text{exact}} = 16$ is the number of verbatim high-salience tokens kept per page as 8-byte discrete records, and $N_{\text{active}}(P)$ is the number of projected soft-prompt and footnote tokens actually entering the frozen LLM's attention layers on the current query.

Many compression papers report only the first term ($12\text{ KiB/page}$) and claim $2{,}000\times$ compression. That is an accounting trick. The moment you project compressed memory slots back into a frozen Transformer so it can attend to them, those projected tokens populate a standard KV cache of size $N_{\text{active}}(P) \cdot M_{\text{KV/tok}}$. In real serving, **the third term is 90% of your context memory**, and honesty requires counting all three.

---

## The GQA baseline: why flat KV still bites

For any autoregressive Transformer with $N_{\text{lay}}$ layers, $n_{\text{kv}}$ key/value heads, head dimension $d_{\text{head}}$, and $b_{\text{kv}} = 2$ bytes per element (FP16/BF16), each cached token costs:

$$
M_{\text{KV/tok}} = 2 \cdot N_{\text{lay}} \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot b_{\text{kv}}
$$

Take the three dense GQA backbones we benchmark in our study—all using $n_{\text{kv}} = 8$ KV heads and $d_{\text{head}} = 128$:

| Backbone | Layers ($N_{\text{lay}}$) | Hidden size ($d_{\text{llm}}$) | KV cost per token (BF16) | 32K KV cache | 256K KV cache |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Qwen3-1.7B` | 28 | 2,048 | **112 KiB** | 3,584.0 MiB (3.50 GiB) | 28,672.0 MiB (28.0 GiB) |
| `Qwen3-4B` | 36 | 2,560 | **144 KiB** | 4,608.0 MiB (4.50 GiB) | 36,864.0 MiB (36.0 GiB) |
| `Qwen3-8B` | 36 | 4,096 | **144 KiB** | 4,608.0 MiB (4.50 GiB) | 36,864.0 MiB (36.0 GiB) |

Notice two things in that table. First, `Qwen3-4B` and `Qwen3-8B` have the *exact same* KV cache size per token (`144 KiB`) because both have 36 layers, 8 KV heads, and head dimension 128. Second, even with GQA's $4\times$ to $8\times$ savings over classic Multi-Head Attention, a 32K conversation on `Qwen3-4B` costs **4.50 GiB** of KV cache—more than the 4-bit weights of the model itself—and takes **12.73 seconds** to prefill cold on an NVIDIA L4.

That creates a brutal engineering constraint that we call **the GQA trap**: if your soft-prompt compressor emits 25% as many tokens as the raw text, you have not even beaten plain 4-bit KV quantization (`36 KiB/token`). To win by two orders of magnitude, you must shrink $N_{\text{active}}(P)$ without losing global awareness of the other pages.

---

## Turning the stone scroll into a multi-resolution book

Human knowledge systems solved long-horizon memory centuries ago not by gluing longer scrolls together, but by inventing **the book**: modular pages ($P$), a 1-line Table of Contents (TOC) per page, coarse skimming, deep reading on the two pages that matter right now, margin sticky-notes for revisions, and a back-of-book footnote index for exact numbers and identifiers.

LSLC compiles that exact structure into GPU memory:

1. **The 3D Page Sheet (`12 KiB/page`):** Each 256-token page is divided into 16 ordered 16-token discourse spans on a $4\times 4$ spatial grid and encoded via Perceiver cross-attention + **3D Factorized Spatial-Layer RoPE** ($\text{diag}(\mathbf{R}_\ell(\ell), \mathbf{R}_h(h), \mathbf{R}_w(w))$) into $\mathbf{Z}_p \in \mathbb{R}^{3 \times 128 \times 4 \times 4}$ (`6,144` FP16 scalars = `12 KiB`).
2. **The 8-Byte Footnote Bank (`128 B/page`):** Continuous bottleneck vectors project into distributed $d_{\text{llm}}$-dimensional semantics rather than indexing discrete vocabulary rows, which causes pure soft-prompt compressors to fail on arbitrary hex codes or `snake_case` identifiers. `SalienceGate` diverts the top $6.25\%$ ($16$ tokens per page) high-surprisal tokens into compact 8-byte records `(`$\texttt{token\_id: int32}, \texttt{page\_id: uint16}, \texttt{h: uint8}, \texttt{w: uint8}$`)`.
3. **The Matryoshka Zoom Pyramid (`218 active tokens at 32K`):** A lightweight `QueryToTOCRouter` scores the user query against all $P=128$ page TOC vectors (using integer `PageRoPE1D`) and allocates attention non-uniformly:

$$
N_{\text{active}}(128) = \underbrace{122 \times 1}_{\text{TOC } (1\times 1)} + \underbrace{4 \times 4}_{\text{Skim } (2\times 2)} + \underbrace{2 \times (4 + 4 + 16)}_{\text{Deep } (\ell=0,1,2)} + \underbrace{2 \times 16}_{\text{Exact footnotes}} = 218\text{ tokens}
$$

Now plug $P = 128$ pages (`32,768` tokens) on `Qwen3-4B` (`144 KiB/token` active KV) into our three-bucket formula:

$$
M_{\text{LSLC}}(32\text{K}) = \underbrace{1.50\text{ MiB}}_{\text{128 pages } \times 12\text{ KiB}} + \underbrace{0.016\text{ MiB}}_{\text{128 pages } \times 128\text{ B}} + \underbrace{30.66\text{ MiB}}_{218 \times 144\text{ KiB active KV}} = \mathbf{32.17\text{ MiB analytical}} \; (\mathbf{34.42\text{ MiB live CUDA}})
$$

Here is how the numbers compare across all four context scales on a single 22.03 GiB NVIDIA L4 GPU:

| Context ($N$) | Backbone | Full BF16 KV | Ideal INT4 KV | Raw-Text RAG ($k=2$ Deep + $4$ Skim) | LSLC Net Total (Live CUDA) | Measured Ratio / Warm TTFT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **4K** ($P=16$) | `Qwen3-1.7B` | 449.8 MiB | 112.0 MiB | 177.8 MiB | **12.88 MiB** | **34.9×** / **55.6 ms** |
| **4K** ($P=16$) | `Qwen3-4B` | 578.3 MiB | 144.0 MiB | 228.5 MiB | **16.50 MiB** | **35.0×** / **66.7 ms** |
| **32K** ($P=128$) | `Qwen3-1.7B` | 3,585.8 MiB | 896.0 MiB | 185.0 MiB | **27.11 MiB** | **132.3×** / **58.0 ms** |
| **32K** ($P=128$) | `Qwen3-4B` | 4,610.3 MiB | 1,152.0 MiB | 237.8 MiB | **34.42 MiB** | **133.9×** / **84.3 ms** |
| **128K** ($P=512$) | `Qwen3-1.7B` | 14,336.0 MiB (OOM) | 3,584.0 MiB | 227.6 MiB | **73.66 MiB** | **194.6×** / **58.0 ms** |
| **256K** ($P=1,024$) | `Qwen3-1.7B` | 28,672.0 MiB (OOM) | 7,168.0 MiB | 284.4 MiB | **135.72 MiB** | **211.3×** / **97.7 ms** |
| **256K** ($P=1,024$) | `Qwen3-8B` (NF4) | 36,864.0 MiB (OOM) | 9,216.0 MiB | 365.5 MiB | **171.03 MiB** | **215.5×** / **453.2 ms** |

---

## Why editability—not compression alone—is the real prize

If your only goal were throwing away 99% of the KV cache on static text, `StreamingLLM` already does that with zero training by keeping 4 attention sinks and a 124-token sliding window—at the cost of going completely blind (`0.0%` Exact NIAH) to anything that scrolled out of the window. And if your only goal were retrieving two static pages verbatim, **Raw-Text Chunk RAG** (storing 1 KiB of raw `int32` token IDs per page plus a 1-token TOC embedding, then re-prefilling the top-2 raw pages on demand) can retrieve unedited text at the cost of $7\times$ higher active KV footprint (`1,658` active tokens vs. `218`).

Where every existing baseline breaks down is **live state revision**. Suppose page 2 records `budget = 25M` (or `deployment_state = canary-staging`), and across turns 4, 7, and 11 the user overrides that constraint three times in a row (`55M -> 70M -> 90M`).

* **Immutable Raw-Text RAG (`0.0%` EM):** Re-reads the original page (`25M`) and misses 100% of revisions.
* **Append-Errata RAG / Full KV Append (`50.0%–52.1%` EM):** Appends all three revision turns onto the retrieved chunk. Now the prompt contains four conflicting numbers (`25M`, `55M`, `70M`, `90M`), causing a 1.7B model to guess an outdated value nearly half the time while invalidating downstream prefix caches on every edit.
* **LSLC In-Place Convex `DeltaUpdater` (`97.9%–100.0%` EM):** Separates each page into three semantic strata—Static rules ($\ell=0$), Entity state ($\ell=1$), and Dynamic sticky-note working memory ($\ell=2$)—and orders the soft-prompt prefix monotonically by volatility:

$$
\mathcal{S}_{\text{LLM}} = \big[\underbrace{\text{TOC}_{1..P}}_{\text{100\% invariant}} \;\big|\; \text{Skim} \;\big|\; \text{Deep}_{\ell=0} \;\big|\; \text{Deep}_{\ell=1} \;\big|\; \underbrace{\text{Deep}_{\ell=2} \;\big|\; \mathcal{I}_{\text{deep}}}_{\text{patched in-place}} \;\big|\; X_{\text{turn}}\big]
$$

When a revision turn arrives, `DeltaUpdater` predicts a sparse spatial gate $\mathbf{G}_p \in [0, 1]^{4\times 4}$ ($\bar{g} \approx 0.20\text{--}0.25$) and overwrites $\ell=2$ in place via convex interpolation:

$$
\mathbf{Z}_{p, 2}^{\text{new}} = (1 - \mathbf{G}_p) \odot \mathbf{Z}_{p, 2}^{\text{old}} + \mathbf{G}_p \odot \Delta\mathbf{Z}_{p, 2}
$$

Because $\text{TOC}_{1..P}$ pools exclusively over $\ell \in \{0, 1\}$, the entire global TOC prefix remains bit-identical ($\max|\Delta_{\text{TOC}}| = 0.0$ by construction), and stale values are overwritten in latent space rather than left around to distract attention.

| Benchmark / Stress Regime | Full 1D KV (BF16) | StreamingLLM / H2O-20% | 1D Flat Latent (12 KiB/p) | Raw-Text RAG (+Append Errata) | **LSLC 3D + Exact Bank + Delta** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **WikiText-2 PPL** (`Qwen3-1.7B` $\downarrow$) | 15.24 | 15.81 / 16.02 | 24.05 | N/A | **15.66** |
| **Semantic NIAH** (`1K–4K` EM $\uparrow$) | 93.3%–100.0% | 0.0%–53.3% | 46.7% | 100.0% | **100.0%** |
| **Exact NIAH** (`1K` 9-tok Code EM / Tok $\uparrow$) | 93.3% / 90.0% | 0.0% / 16.0% | 0.0% / 29.3% | 93.3% / 90.0% | **66.7% / 90.7%** (`0.0%` w/o Bank) |
| **EC-1: Agentic Python Codebase** (`8p` EM / Tok $\uparrow$) | — | 0.0% | — | — | **90.0% / 96.0%** (`1.7B`) \| **80.0% / 96.5%** (`8B`) |
| **EC-2: 4-Key Adversarial Decoys** (EM / Tok $\uparrow$) | — | 0.0% | — | — | **85.0% / 96.0%** (`1.7B`) \| **80.0% / 95.0%** (`8B`) |
| **Track 4: $k=3$ State Overrides** (3-Seed EM $\uparrow$) | 50.0% (`1.7B`) | N/A | 0.0% ($\Delta=1.097$) | 0.0% (Imm.) / 52.1% ± 15.7% (App.) | **97.9% ± 3.6%** (**100.0%** seed 9999, $\Delta_{\text{TOC}}=0.0$) |
| **EC-3: Deep $k=5$ Symbolic Overrides** (EM / Tok $\uparrow$) | — | N/A | — | 0.0% (Imm.) / 38.3% (App.) | **80.0% / 93.8%** (`1.7B`) \| **55.0% / 90.9%** (`8B` NF4) |
| **EC-5: Multi-Step Context Reasoning** (Acc $\uparrow$) | 88.9% (`1.7B`) / 100% (`8B`) | — | — | — | **66.7%** (`1.7B`) \| **100.0%** (`8B`, 100% ret.) |

---

## The 231× asymptote and how to reach 1M tokens

Look closely at the compression ratios in our live CUDA table: `34.9×` at 4K, `132.3×` at 32K, `194.6×` at 128K, and `211.3×` at 256K. Why does the curve flatten near `215×` instead of climbing forever?

Because a flat 1-level TOC emits **one active KV token per page** ($M_{\text{KV/tok}} = 112\text{ KiB}$ on `1.7B` or $144\text{ KiB}$ on `4B/8B`) on top of the $12.125\text{ KiB/page}$ book-plus-footnote storage. Taking the limit as $P \to \infty$ gives the exact closed-form ceiling of a 1-level TOC:

$$
\lim_{P \to \infty} \frac{256 \cdot M_{\text{KV/tok}}}{12.125\text{ KiB} + M_{\text{KV/tok}}} = \frac{28{,}672\text{ KiB}}{124.125\text{ KiB}} = \mathbf{231.0\times} \; (\texttt{Qwen3-1.7B}) \quad \text{and} \quad \mathbf{236.2\times} \; (\texttt{Qwen3-4B / 8B})
$$

To break through that asymptote at **512K (`P = 2,048` pages)** and **1M tokens (`P = 4,096` pages)**—and to keep the router from diluting softmax probability across 4,096 flat candidates—you add a **2-Level Chapter-to-Page Hierarchy** grouping every $M = 32$ pages (`8,192` tokens) into one Chapter:

* At **1M tokens (`4,096` pages)**, there are only **128 Chapters**—the exact same routing width as a 32K book.
* The router picks top-$k_{\text{chap}} = 2$ Chapters (`126` remaining Chapter-TOC tokens), opens Page-TOC only inside those 2 chapters (`58` sub-TOC tokens), and routes `4` Skim pages (`16` tokens) plus `2` Deep pages (`48` latent + `32` exact tokens), locking active LLM context at **280 tokens total**.
* Total net VRAM at **1,048,576 tokens** (`48.50 MiB` book+bank + `280` active KV tokens) lands at **79.13 MiB (`1,449.5×`)** on `Qwen3-1.7B` and **87.88 MiB (`1,677.3×`)** on `Qwen3-4B` and `Qwen3-8B`—under **90 MiB of VRAM for a million tokens**.

---

## A recipe you can follow

* **Count active soft-prompt KV bytes first:** Before designing a latent compressor, multiply its projected token count $N_{\text{active}}$ by $2 \cdot L \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot 2$. If $N_{\text{active}} > 0.20 \cdot N$, plain INT4 KV quantization will beat you on memory with zero training.
* **Order your prefix by volatility:** Put immutable global summaries (`TOC`) first, structural facts ($\ell \in \{0, 1\}$) in the middle, and mutable working state ($\ell=2$ + footnotes) right before the user turn. That gives you vLLM/HuggingFace prefix caching across turns by construction.
* **Never force continuous latents to memorize random IDs:** Add an $8\text{ byte/token}$ discrete bypass (`token_id: int32, page_id: uint16, h: uint8, w: uint8`) for the top $6.25\%$ high-salience tokens and gather their embeddings directly from the frozen LLM's `embed_tokens` table on Deep Read.
* **Attenuate runner-up page footnotes quadratically:** When injecting exact footnotes from top-$k_{\text{deep}}$ pages, weight secondary pages by $(p_{\text{deep}} / p_{\max})^2$ so random numbers from a runner-up page do not interfere with single-page needle copy (`66.7% -> 85.0%–90.0%` Exact Match).
* **Reconsolidate after 4 in-place overwrites:** Convex $\ell=2$ delta updates ($\mathbf{Z}_{p,2} \leftarrow (1-\mathbf{G}_p)\mathbf{Z}_{p,2} + \mathbf{G}_p\Delta\mathbf{Z}_{p,2}$) stay sharp through $k=3$ to $k=4$ revisions (`97.9%–100%` EM), after which cumulative 1–2 character drift begins to appear on 4-bit backbones (`55.0%` EM / `90.9%` token accuracy at $k=5$ on `Qwen3-8B` NF4). Re-encoding just that single 256-token page every 4 edits costs only `~1.8 ms` ($O(1)$ in total book length) and resets drift to zero.

---

## Frequently asked questions

### Isn't LSLC just RAG with extra steps?
No. In standard Raw-Text Chunk RAG, the LLM is completely blind to unretrieved pages, must re-prefill 512 to 1,536 raw tokens on every query ($7\times$ larger active KV cache), and cannot edit state in place—when $k=3$ sequential revisions arrive, immutable RAG scores `0.0%` and append-errata RAG drops to `52.1%` because stale and updated values co-exist in the raw text. LSLC keeps all $P$ pages simultaneously visible in self-attention via a differentiable Matryoshka pyramid (`1×1 TOC`, `2×2 Skim`, `4×4 Deep`) and overwrites $\ell=2$ in place at `97.9%–100.0%` EM.

### Why does a 12 KiB continuous page fail on exact IDs when 256 token IDs only cost 1 KiB?
Because 256 `int32` token IDs (`1 KiB`) are discrete lookup pointers into a vocabulary table, whereas a `[3, 128, 4, 4]` FP16 latent (`12 KiB`) must project into continuous $d_{\text{llm}}$-dimensional hidden states (`2,048` to `4,096` floats per soft token). Expecting a continuous soft-prompt projection trained in 300 steps to act as a bit-exact hash table for arbitrary 9-token codes is an optimization and representation-alignment bottleneck, not a Shannon bit-capacity limit. Adding `128 bytes/page` of discrete footnote pointers solves the discrete copy task while letting the 3D latent book handle discourse and code completion.

### Why did Episode Exact Match (90.0%) exceed BPE sub-token capture (81.8%) on the Python codebase benchmark?
Because `81.82%` (`54/66`) measures the fraction of *individual BPE sub-tokens* stored in the 16-slot footnote bank, whereas `90.0%` (`18/20`) measures full-episode answer accuracy. On multi-token `snake_case` identifiers like `half_open_probe` (`3` BPE sub-tokens), capturing the leading high-surprisal sub-tokens (`half`, `_open`) in `SparseExactBank` gives the frozen LLM enough anchor context to complete the trailing sub-token (`_probe`) from the 24 continuous 3D latent tokens of that page.

### Does carrying a compressed LSLC book prefix hurt the LLM's multi-step reasoning?
On `Qwen3-8B`, context-grounded multi-step reasoning across the compressed book (2-step arithmetic, Python `if/else` branch tracing, and 3-entity transitive deduction) reaches **100.0% accuracy**, matching Full BF16 KV (`100.0%` retention). On smaller `Qwen3-1.7B`, multi-step reasoning retains `75.0%` of Full KV accuracy (`66.7%` vs. `88.9%`).

### What is the difference between Cold TTFT, Warm TTFT, and page encoding time?
Encoding a 256-token page into $\mathbf{Z}_p$ takes roughly **0.22 to 1.8 ms per page** (`~220 ms` chunked for a 128-page 32K book, amortized as pages stream in). Once the book is in VRAM, **Cold Query TTFT** (`92.9 ms` on `1.7B`, `111.8 ms` on `4B` at 32K) measures router scoring, projector assembly, and LLM prefill over all 218 active tokens; **Warm Query TTFT** (`58.0 ms` on `1.7B`, `84.3 ms` on `4B`) reuses the cached KV states of the invariant 122-token `TOC` prefix block.

---

## References

* [Yang et al., Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388), for the dense GQA backbone configurations (`Qwen3-1.7B`, `Qwen3-4B`, `Qwen3-8B`).
* [Li, Models Take Notes at Prefill: KV Cache Can Be Editable and Composable (arXiv:2606.17107)](https://arxiv.org/abs/2606.17107), on editable uncompressed KV caches and vLLM prefix-cache hit rates.
* [Fallahpour et al., Memorilla: Latent Semantic Memory for LLMs (arXiv:2610.21375)](https://arxiv.org/abs/2610.21375), on persistent 1D chunk latent memory for frozen LLMs.
* [Li et al., End-to-End Context Compression at Scale / LCLMs (arXiv:2606.09659)](https://arxiv.org/abs/2606.09659) and [Abaskohi et al., SeKV (arXiv:2606.31145)](https://arxiv.org/abs/2606.31145), on hierarchical latent and SVD-offloaded KV compression.
* [Wang et al., GroupKV (arXiv:2609.17573)](https://arxiv.org/abs/2609.17573) and [Pan et al., LeanSparse (arXiv:2611.04671)](https://arxiv.org/abs/2611.04671), on late-2026 coarse-to-fine KV indexing.
* [LSLC Technical Paper (`docs/paper_draft_lslc.tex`)](paper_draft_lslc.tex) and [Architecture RFC (`ARCHITECTURE.md`)](../ARCHITECTURE.md).
