#!/usr/bin/env python3
"""Gerbang 6: Cross-Family Path (B) Replication + Mini Z_book Negative Probe on Frozen Llama-3.1-8B-Instruct BF16 (r=0).

Implements the locked Paper 1 Path (B) architecture via the shared `PathBDiscreteMemory` core:
  - ZERO `LayeredLatentEncoder`, ZERO `SpatialLatentProjector`, ZERO `Z_book`, ZERO untrained `hw_coord_embed`.
  - Lean Layer-16 `ContextualEditGateHead(hidden_dim=64)` + `SalienceGate` (span-16 `k_exact=48`, 192 B/p before edit,
    384 B/p after edit incl. `base_token_ids`) + `BGE-small 4-span` BF16 (3,072 B/p, 33.4M frozen params).
  - True Two-Bank prefix ordering (unedited Bank-1 pages deterministically FIRST, mutated Bank-2 pages LAST).
  - Span-aligned selective overwrite (Opsi B) with clean newline separator (no `'39Budget'` splicing).
  - True per-layer, per-KV-head (GQA) 1D-pooled SnapKV (`pool_kernel_size=5, recent_ratio=0.10`) without 8B OOM.

Modular `--tasks`:
  1. `train_pathb`: Lightweight Path (B)-only training (`SalienceGate` + Layer-16 `ContextualEditGateHead(hidden_dim=64)`).
  2. `llama_gate2_n200`: N=200 (100 budget + 100 symbolic k=5) with `bm25_rag_append_errata`, `gated_bm25_rag`,
     `raw_text_same_pages_k4`, `standalone_append_192tok`, `standalone_discrete_span_bank`, and `blind_edit_routing_slice`
     (reporting `strict_em`, `normalized_numeric_em`, `normalized_w2n_em`, `by_query_type`, and exact McNemar).
  3. `llama_coding_n24`: 24 65-file cross-file scenarios (0 path hints, 1 run): `bm25_rag_k8`, `bge_4span_rag_k4`,
     `standalone_discrete_k4`, and matched-page/matched-gate `raw_text_same_pages_k4`.
  4. `llama_ood_kv_4k`: OOD RULER 4K (N=48) + WikiText PPL (N=16) with per-layer/per-KV-head 1D-pooled SnapKV,
     unified timing (`encode_or_evict_ms`, `prefill_ms`, `decode_ms`, `decode_ms_per_tok`, `total_query_ms`),
     and identical query input `ep["question"]` (+ `standalone_discrete_router_query` transparency arm).
  5. `llama_ruler_16k_32k_and_profile`: RULER 16K & 32K (N=60/len) including zero-training `heuristic_span16_tfidf`
     + Unified `k_deep in {2,4,8}` Systems Profile + True Two-Bank KV Reuse (`p0`, `p1`, `pEnd` `bank1_pkv_max_abs_diff == 0.0`
     plus realistic `post_rerouting_cache_hit_telemetry`).
  6. `llama_zbook_probe`: Cross-family mini Layer-0 `Z_book` negative probe on frozen Llama-3.1-8B (5-cell masked-old
     edit ablation N=48 + WikiText-2 continuation PPL N=16 proving BOTH sides of the Path B thesis across model families).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from colab_run_m2 import _load_wikitext_pools  # noqa: E402
from colab_run_m3 import (  # noqa: E402
    _extract_kv_pairs,
    _llm_accepts_position_ids,
    compute_memory_safe_obs_attention_and_kv,
    compute_nll_and_tok_acc_from_kv,
    evict_past_key_values_h2o_attention,
    greedy_generate_from_kv_cache,
)
from colab_run_rebuttal_v2 import BM25PageRetriever  # noqa: E402
from colab_run_v3_gerbang3_coding import (  # noqa: E402
    _MockCodeTokenizer,
    _aggregate_system_rows,
    _build_repo_pages,
    _extract_python_code_block,
    _pad_or_chunk_tokens,
    _should_stop_code_gen,
    _should_stop_homonym_gen,
)
from colab_run_v3_gerbang4_community import (  # noqa: E402
    _PlugInBiEncoderTOC,
    _encode_book_chunked,
    _encode_unpadded_rag_ids,
    _get_special_token_ids,
    _normalize_qa_text,
    _peak_vram_mb,
    _reset_peak_vram,
    _should_stop_qa_gen,
    _sync_device,
    _wilson_ci95,
    build_ruler_episodes,
    compute_community_qa_metrics,
)
from colab_run_v3_gerbang5_ablations import (  # noqa: E402
    _FAIR_G2_FEWSHOT_PREFIX,
    build_2x2_edit_episodes,
    build_coding_n24_tasks,
    build_pathless_coding_query,
    canonicalize_numeric_answer,
)
from colab_run_v3_path_tiebreaker import (  # noqa: E402
    _build_append_192tok_bank,
    _clone_kv_cache,
    _summarize_scalar_rows,
    compute_pairwise_suite,
    mask_old_value_in_exact_bank,
    measure_post_rerouting_cache_hit_telemetry,
    paired_bootstrap_ci95,
)
from colab_run_v3_universal_ckpt import (  # noqa: E402
    _build_disjoint_span16_salience_pages,
    _build_natural_python_qa_episodes,
    assert_universal_train_disjoint,
    load_ungated_hf_python_snippets,
)
from lslc import (  # noqa: E402
    AdversarialLivingBookDataset,
    ChunkedTextPageDataset,
    CodingAgentHarness,
    ContextualEditGateHead,
    LSLCConfig,
    LSLCSystem,
    MigrationTask,
    PathBDiscreteMemory,
    SparseExactBank,
    TinyMockCausalLM,
    canonicalize_w2n_answer,
    extract_contextual_edit_embeds,
    get_git_provenance,
    get_newline_sep_token_id,
    heuristic_span16_tfidf_bank,
    load_pathb_checkpoint,
    save_pathb_checkpoint,
    select_amp_dtype,
)
from lslc.datasets import _encode_text_or_mock  # noqa: E402


# ---------------------------------------------------------------------------
# Official Llama-3.1-Instruct Prompt Templates & Multi-EOS Stop Handling
# (STRICTLY ZERO Qwen3 `<|im_start|>` / `<|im_end|>` / `<think>` tags!)
# ---------------------------------------------------------------------------

LLAMA3_USER_OPEN = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n# Reference Memory:\n"
LLAMA3_ASSIST_OPEN = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA3_FALLBACK_MIRRORS: tuple[str, ...] = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
)


def format_llama3_qa_prompt(question: str) -> str:
    """Official Llama-3.1-Instruct QA turn suffix (ZERO Qwen3 tags)."""
    return (
        f"\n\n# Question:\n{question.strip()}\n"
        f"Answer concisely with only the exact answer."
        f"{LLAMA3_ASSIST_OPEN}"
    )


def format_llama3_edit_prompt(task: MigrationTask, target_path: str, raw_target_content: str) -> str:
    """Official Llama-3.1-Instruct coding edit turn suffix (identical task spec, Llama-3.1 tags)."""
    if target_path == "tests/test_service.py":
        role_hint = (
            f"For `{target_path}`, update `fake_{task.symbol_name}` (its signature, `seen.append(...)`, "
            f"and returned dict) and update the test assertions to verify the new keyword-only parameter "
            f"and approved caller default value from Reference Memory."
        )
    else:
        role_hint = (
            f"For `{target_path}`, update the caller function signature (adding the new parameter with the "
            f"approved default value from Reference Memory) and pass that keyword argument to `{task.symbol_name}(...)`."
        )
    return (
        f"\n\n# Task:\n{task.prompt}\n{role_hint}\n\n"
        f"# Current `{target_path}`:\n```python\n{raw_target_content.rstrip()}\n```\n\n"
        f"Output ONLY the complete updated Python source code for `{target_path}` inside a ```python block."
        f"{LLAMA3_ASSIST_OPEN}```python\n"
    )


def format_llama3_homonym_prompt(task: MigrationTask, raw_homonym_content: str) -> str:
    """Official Llama-3.1-Instruct homonym preservation turn suffix."""
    return (
        f"\n\n# Task:\n{task.prompt}\n\n"
        f"# Current `{task.homonym_file}`:\n```python\n{raw_homonym_content.rstrip()}\n```\n\n"
        f"Should `{task.homonym_file}` be modified? Reply ONLY `KEEP_UNCHANGED` if it must remain unchanged, "
        f"or output the full updated file inside ```python ... ```."
        f"{LLAMA3_ASSIST_OPEN}"
    )


def get_llama3_stop_token_ids(tokenizer) -> set[int]:
    """Collect all valid EOS / EOT stop token IDs (handles int or list[int] eos_token_id + `<|eot_id|>`)."""
    stops: set[int] = set()
    eos_raw = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_raw, (list, tuple, set)):
        for x in eos_raw:
            if x is not None and int(x) >= 0:
                stops.add(int(x))
    elif eos_raw is not None and int(eos_raw) >= 0:
        stops.add(int(eos_raw))

    conv = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(conv):
        for tok_str in ("<|eot_id|>", "<|end_of_text|>"):
            try:
                tid = conv(tok_str)
                if isinstance(tid, int) and tid >= 0:
                    stops.add(int(tid))
            except Exception:
                pass
    return stops


def _clean_llama3_decoded(text: str) -> str:
    for tok_s in ("<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>"):
        text = text.replace(tok_s, "")
    return text.strip()


def _should_stop_llama3_qa(decoded: str) -> bool:
    if "<|eot_id|>" in decoded or "<|end_of_text|>" in decoded:
        return True
    return _should_stop_qa_gen(decoded)


@torch.no_grad()
def greedy_generate_llama3(
    llm: nn.Module,
    prefix_embeds: torch.Tensor,
    prompt_ids: torch.Tensor,
    tokenizer,
    *,
    max_new_tokens: int = 32,
    user_open_ids: torch.Tensor | None = None,
    stop_fn=None,
) -> str:
    """KV-cached greedy generation tailored for Llama-3.1-Instruct multi-EOS (`<|eot_id|>` + `<|end_of_text|>`)."""
    embed_fn = llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    prefix_cast = prefix_embeds.to(llm_dtype)
    p_embeds = embed_fn(prompt_ids).to(llm_dtype)
    if user_open_ids is not None:
        u_embeds = embed_fn(user_open_ids).to(llm_dtype)
        inp_embeds = torch.cat((u_embeds, prefix_cast, p_embeds), dim=1)
    else:
        inp_embeds = torch.cat((prefix_cast, p_embeds), dim=1)

    bsz, init_len, _ = inp_embeds.shape
    device = inp_embeds.device
    attn_mask = torch.ones((bsz, init_len), device=device, dtype=torch.long)
    stop_ids = get_llama3_stop_token_ids(tokenizer)
    active_stop_fn = _should_stop_llama3_qa if stop_fn is None else stop_fn

    prev_use_cache = getattr(llm.config, "use_cache", False) if hasattr(llm, "config") else False
    if hasattr(llm, "config"):
        llm.config.use_cache = True

    gen_tokens: list[int] = []
    try:
        out = llm(inputs_embeds=inp_embeds, attention_mask=attn_mask, use_cache=True)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        pkv = getattr(out, "past_key_values", None)
        for step_idx in range(max_new_tokens):
            tid = int(next_tok[0, 0].item())
            if tid in stop_ids:
                break
            gen_tokens.append(tid)
            if len(gen_tokens) % 4 == 0 or step_idx >= max_new_tokens - 1:
                dec = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                if active_stop_fn(dec):
                    break
            if step_idx >= max_new_tokens - 1:
                break
            attn_mask = torch.cat((attn_mask, torch.ones((bsz, 1), device=device, dtype=torch.long)), dim=1)
            if pkv is not None:
                out = llm(input_ids=next_tok, attention_mask=attn_mask, past_key_values=pkv, use_cache=True)
                pkv = getattr(out, "past_key_values", None)
            else:
                inp_embeds = torch.cat((inp_embeds, embed_fn(next_tok).to(llm_dtype)), dim=1)
                out = llm(inputs_embeds=inp_embeds, attention_mask=attn_mask)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    finally:
        if hasattr(llm, "config"):
            llm.config.use_cache = prev_use_cache

    return _clean_llama3_decoded(tokenizer.decode(gen_tokens, skip_special_tokens=True))


# ---------------------------------------------------------------------------
# Backbone Loader with Auto-Fallback to Ungated Mirrors (Strictly BF16 Frozen)
# ---------------------------------------------------------------------------


def load_frozen_llama8b_with_fallback(
    cfg: LSLCConfig,
    requested_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    device: torch.device | str = "cuda",
) -> tuple[nn.Module, Any, str, list[dict[str, str]]]:
    """Load frozen BF16 Llama-3.1-8B-Instruct with automatic ungated mirror fallback."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    candidates: list[str] = []
    for m in (requested_model,) + LLAMA3_FALLBACK_MIRRORS:
        if m and m not in candidates:
            candidates.append(m)

    dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else (torch.float16 if torch.device(device).type == "cuda" else torch.float32)
    )
    attempts: list[dict[str, str]] = []
    last_err: Exception | None = None

    for model_id in candidates:
        try:
            print(f"[LLAMA8B LOAD] Trying {model_id} (dtype={dtype}, load_in_4bit=False)...", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "<|end_of_text|>"
            llm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
            llm.to(device)
            for p in llm.parameters():
                p.requires_grad_(False)
            llm.eval()
            if hasattr(llm, "config"):
                llm.config.use_cache = False
            cfg.model_name = model_id
            cfg.sync_from_hf_config(llm.config)
            cfg.contextual_edit_layer = max(1, cfg.num_llm_layers // 2)
            attempts.append({"model_id": model_id, "status": "loaded"})
            print(
                f"[LLAMA8B LOADED] resolved={model_id} d_llm={cfg.d_llm} layers={cfg.num_llm_layers} "
                f"kv_heads={cfg.num_kv_heads} edit_layer={cfg.contextual_edit_layer}",
                flush=True,
            )
            return llm, tokenizer, model_id, attempts
        except Exception as exc:
            attempts.append({"model_id": model_id, "status": f"failed: {type(exc).__name__}: {str(exc)[:180]}"})
            print(f"[LLAMA8B FALLBACK] {model_id} failed ({exc}); trying next mirror.", flush=True)
            last_err = exc

    raise RuntimeError(f"All Llama-3.1-8B candidates failed: {attempts}") from last_err


# ---------------------------------------------------------------------------
# Lightweight Path (B)-Only Training (SalienceGate + Layer-16 Edit Gate h=64)
# ---------------------------------------------------------------------------


def _frozen_llm_param_checksums(llm: nn.Module) -> dict[str, tuple[float, float]]:
    """O(1)-RAM exact checksum (sum & max-abs in float64) across EVERY parameter of frozen 8B LLM."""
    checks: dict[str, tuple[float, float]] = {}
    with torch.no_grad():
        for name, p in llm.named_parameters():
            pf = p.detach().float()
            checks[name] = (float(pf.sum().item()), float(pf.abs().max().item()))
    return checks


def _verify_frozen_llm_unchanged(llm: nn.Module, before: dict[str, tuple[float, float]]) -> float:
    max_diff = 0.0
    with torch.no_grad():
        for name, p in llm.named_parameters():
            pf = p.detach().float()
            s_now, m_now = float(pf.sum().item()), float(pf.abs().max().item())
            s_old, m_old = before[name]
            max_diff = max(max_diff, abs(s_now - s_old), abs(m_now - m_old))
    return max_diff


def train_pathb_llama8b(
    llm: nn.Module,
    pathb: PathBDiscreteMemory,
    tokenizer,
    train_pool: list[int],
    device: torch.device,
    *,
    salience_steps: int = 250,
    edit_gate_steps: int = 200,
    salience_lr: float = 1e-3,
    edit_gate_lr: float = 5e-4,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Train ONLY lean Path (B) modules (`SalienceGate` span-16 + Layer-16 `ContextualEditGateHead(hidden_dim=64)`), Frozen r=0."""
    t0 = time.perf_counter()
    embed_fn = llm.get_input_embeddings()
    vocab_size = len(embed_fn.weight)
    page_size = pathb.cfg.page_size
    span_len = int(pathb.salience_span_len)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    special_set = set(special_ids)
    llm_before = _frozen_llm_param_checksums(llm)

    # 1. Build 100% eval-disjoint training datasets across general clauses, state revisions, and natural Python
    n_g2 = 10 if is_mock else 48
    sal_samples: list[tuple[list[int], list[float]]] = []
    edit_turn_pairs: list[tuple[torch.Tensor, float]] = []
    texts_by_domain: dict[str, list[str]] = {
        "prose_and_state_revision": [],
        "span16_factual_clauses": [],
        "natural_python_code": [],
    }

    for mode_m, sd_off in (("budget", 1101), ("symbolic", 1203)):
        ds_m = AdversarialLivingBookDataset(
            num_episodes=n_g2, num_pages=6, page_size=page_size, num_edits=4,
            mode=mode_m, split="train", tokenizer=tokenizer, vocab_size=vocab_size,
            seed=seed + sd_off, background_token_pool=train_pool, style="mixed_natural",
        )
        for ep in ds_m:
            tgt_p = int(ep.target_page_idx)
            p_ids = ep.context_pages[tgt_p].tolist()
            m_row = ep.metadata["context_salience_span_mask"][tgt_p].tolist()
            if sum(m_row) > 0:
                sal_samples.append((p_ids, m_row))
            ep_txt = tokenizer.decode(p_ids, skip_special_tokens=True)
            for (_, e_ids_cpu), is_acc in zip(ep.edit_pages, ep.metadata["edit_accepted_flags"]):
                edit_turn_pairs.append((e_ids_cpu, 1.0 if is_acc else 0.0))
                ep_txt += " " + tokenizer.decode(e_ids_cpu.tolist(), skip_special_tokens=True)
            texts_by_domain["prose_and_state_revision"].append(ep_txt)

    n_clauses = 24 if is_mock else 160
    clause_pages, clause_texts = _build_disjoint_span16_salience_pages(
        train_pool, tokenizer, vocab_size, num_pages=n_clauses, page_size=page_size, seed=seed
    )
    sal_samples.extend(clause_pages)
    texts_by_domain["span16_factual_clauses"].extend(clause_texts)

    py_snips, py_src = load_ungated_hf_python_snippets(max_snippets=128, is_mock=is_mock, seed=seed)
    py_eps = _build_natural_python_qa_episodes(
        py_snips, train_pool, tokenizer, vocab_size,
        num_episodes=(12 if is_mock else 80), num_pages=4, page_size=page_size, seed=seed,
    )
    for pep in py_eps:
        sal_samples.append((pep["sal_page_ids"], pep["sal_page_mask"]))
        texts_by_domain["natural_python_code"].append(pep["raw_text"])

    disjoint_audit = assert_universal_train_disjoint(texts_by_domain)

    # 2. Train SalienceGate (pure token embedding lookup, 0 LLM forward passes!)
    rng = random.Random(seed + 77)
    eff_sal_steps = min(salience_steps, 20) if is_mock else salience_steps
    opt_sal = torch.optim.AdamW(pathb.salience_gate.parameters(), lr=salience_lr, weight_decay=0.01)
    pathb.salience_gate.train()
    sal_losses: list[float] = []
    batch_sz = 4 if is_mock else 16

    for _ in range(eff_sal_steps):
        batch = [sal_samples[rng.randrange(len(sal_samples))] for _ in range(batch_sz)]
        p_t = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
        m_t = torch.tensor([x[1] for x in batch], dtype=torch.float32, device=device)
        valid_t = torch.ones_like(p_t, dtype=torch.bool)
        for sid in special_ids:
            valid_t &= p_t != int(sid)
        with torch.no_grad():
            p_emb = embed_fn(p_t)
        opt_sal.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            logits = pathb.salience_gate.score_logits(p_emb, surprisal=None, token_ids=p_t, vocab_size=vocab_size).float()
            tok_loss = F.binary_cross_entropy_with_logits(
                logits, (m_t > 0.25).float(), pos_weight=torch.tensor(6.0, device=device), reduction="none"
            )
            tok_loss = (tok_loss * valid_t.float()).sum() / valid_t.float().sum().clamp(min=1.0)
            s_l = min(span_len, page_size)
            win_logits = F.avg_pool1d(logits.unsqueeze(1), kernel_size=s_l, stride=1).squeeze(1)
            win_tgt = (F.avg_pool1d((m_t > 0.25).float().unsqueeze(1), kernel_size=s_l, stride=1).squeeze(1) >= 0.35).float()
            win_loss = F.binary_cross_entropy_with_logits(win_logits, win_tgt, pos_weight=torch.tensor(4.0, device=device))
            loss_s = tok_loss + win_loss
        loss_s.backward()
        torch.nn.utils.clip_grad_norm_(pathb.salience_gate.parameters(), 1.0)
        opt_sal.step()
        sal_losses.append(float(loss_s.detach().item()))
    pathb.salience_gate.eval()

    # Verify zero special tokens land in SparseExactBank
    special_in_bank = 0
    with torch.no_grad():
        check_p = torch.tensor([x[0] for x in sal_samples[:8]], dtype=torch.long, device=device).unsqueeze(0)
        bk_chk = SparseExactBank.from_pages(
            check_p, pathb.salience_gate, embed_fn(check_p), vocab_size=vocab_size,
            special_token_ids=special_ids, salience_span_len=span_len, compact_pathb=True,
        )
        for tid in bk_chk.token_ids.reshape(-1).tolist():
            if int(tid) in special_set:
                special_in_bank += 1
    assert special_in_bank == 0, f"Special tokens leaked into SparseExactBank: {special_in_bank}"

    # 3. Pre-extract frozen Layer-16 contextual edit embeddings once & train lean ContextualEditGateHead(hidden_dim=64)
    layer_idx = pathb.cfg.contextual_edit_layer if pathb.cfg.contextual_edit_layer is not None else max(1, pathb.cfg.num_llm_layers // 2)
    cached_edits: list[tuple[torch.Tensor, float]] = []
    with torch.no_grad():
        for e_ids_cpu, y_val in edit_turn_pairs:
            e_t = e_ids_cpu.unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                ctx_e = extract_contextual_edit_embeds(llm, e_t, layer_idx=layer_idx)
            cached_edits.append((ctx_e.detach().float(), y_val))

    eff_eg_steps = min(edit_gate_steps, 20) if is_mock else edit_gate_steps
    opt_eg = torch.optim.AdamW(pathb.edit_gate.parameters(), lr=edit_gate_lr, weight_decay=0.01)
    pathb.edit_gate.train()
    eg_losses: list[float] = []
    eg_bs = 4 if is_mock else 16

    for _ in range(eff_eg_steps):
        opt_eg.zero_grad(set_to_none=True)
        step_losses = []
        for _j in range(eg_bs):
            ctx_e, y_val = cached_edits[rng.randrange(len(cached_edits))]
            logit, _ = pathb.edit_gate(ctx_e)
            step_losses.append(F.binary_cross_entropy_with_logits(logit, torch.tensor([y_val], device=device)))
        loss_e = torch.stack(step_losses).mean()
        loss_e.backward()
        torch.nn.utils.clip_grad_norm_(pathb.edit_gate.parameters(), 1.0)
        opt_eg.step()
        eg_losses.append(float(loss_e.detach().item()))
    pathb.edit_gate.eval()

    train_correct = 0
    with torch.no_grad():
        for ctx_e, y_val in cached_edits:
            logit, _ = pathb.edit_gate(ctx_e)
            pred = 1.0 if float(torch.sigmoid(logit).item()) > pathb.bank_write_threshold else 0.0
            train_correct += int(pred == y_val)
    max_frozen_diff = _verify_frozen_llm_unchanged(llm, llm_before)
    assert max_frozen_diff == 0.0, f"Frozen LLM mutated during Path B training: {max_frozen_diff}"

    p_bd = pathb.trainable_param_breakdown(llm)
    return {
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "python_corpus_source": py_src,
        "num_salience_pages": len(sal_samples),
        "num_edit_turns": len(cached_edits),
        "salience_steps": eff_sal_steps,
        "salience_initial_loss": round(sal_losses[0], 4),
        "salience_final_loss": round(sal_losses[-1], 4),
        "edit_gate_layer": layer_idx,
        "edit_gate_hidden_dim": pathb.edit_gate.hidden_dim,
        "edit_gate_steps": eff_eg_steps,
        "edit_gate_initial_loss": round(eg_losses[0], 4),
        "edit_gate_final_loss": round(eg_losses[-1], 4),
        "edit_gate_train_acc": round(train_correct / max(1, len(cached_edits)), 4),
        "trainable_pathb_params": p_bd["total_pathb_trainable_params"],
        "trainable_param_breakdown": p_bd,
        "special_tokens_in_bank": special_in_bank,
        "max_frozen_param_diff": max_frozen_diff,
        "disjointness_audit": disjoint_audit,
    }


# ---------------------------------------------------------------------------
# Benchmark 1: llama_gate2_n200 (N=200 + 3 Causal Controls + w2n + Blind Routing)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_llama_gate2_n200(
    llm: nn.Module,
    pathb: PathBDiscreteMemory,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    eval_n: int = 200,
    salience_route_weight: float = 1.5,
    blind_edit_slice_n: int = 60,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Gate-2 N=200 on Llama-3.1-8B with bm25_rag_append_errata, gated_bm25_rag, raw_text_same_pages_k4,
    standalone_append_192tok, standalone_discrete_span_bank (span-aligned overwrite), and blind_edit_routing slice."""
    embed_fn = llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    sep_id = get_newline_sep_token_id(tokenizer)
    pathb.eval()

    n_each = 4 if is_mock else max(1, eval_n // 2)
    num_p = 8 if is_mock else 16
    eps_budget = list(
        AdversarialLivingBookDataset(
            num_episodes=n_each, num_pages=num_p, page_size=pathb.cfg.page_size, num_edits=3,
            mode="budget", split="eval", tokenizer=tokenizer, vocab_size=vocab_size,
            seed=seed + 40, background_token_pool=val_pool,
        )
    )
    eps_sym = list(
        AdversarialLivingBookDataset(
            num_episodes=n_each, num_pages=num_p, page_size=pathb.cfg.page_size, num_edits=5,
            mode="symbolic", split="eval", tokenizer=tokenizer, vocab_size=vocab_size,
            seed=seed + 41, background_token_pool=val_pool,
        )
    )
    all_eps = eps_budget + eps_sym

    user_open_ids = torch.tensor(
        tokenizer.encode(LLAMA3_USER_OPEN + _FAIR_G2_FEWSHOT_PREFIX, add_special_tokens=False),
        dtype=torch.long, device=device,
    ).unsqueeze(0)

    systems = (
        "bm25_rag_append_errata",
        "gated_bm25_rag",
        "raw_text_same_pages_k4",
        "standalone_append_192tok",
        "standalone_discrete_span_bank",
    )
    q_types_list = ("accepted_override", "rejected_proposal", "neighbor_preservation", "randomized_static_retention")
    sys_out: dict[str, Any] = {}
    scores_norm_by_sys: dict[str, list[float]] = {}
    scores_w2n_by_sys: dict[str, list[float]] = {}

    for sys_name in systems:
        rows: list[dict[str, Any]] = []
        rows_by_mode: dict[str, list[dict[str, Any]]] = {"budget": [], "symbolic": []}
        rows_by_qtype: dict[str, list[dict[str, Any]]] = {qt: [] for qt in q_types_list}
        gate_tp = gate_tn = gate_fp = gate_fn = 0
        acc_gates: list[float] = []
        rej_gates: list[float] = []

        for ep_idx, ep in enumerate(all_eps):
            clean_q = tokenizer.decode(ep.metadata["router_query_ids"].tolist(), skip_special_tokens=True).strip()
            gold = str(ep.metadata["answer_str"]).strip()
            ep_mode = str(ep.metadata.get("mode", "budget"))
            q_type = str(ep.metadata.get("query_type", ""))
            gold_flags = list(ep.metadata.get("edit_accepted_flags", []))
            p_prompt = torch.tensor(
                tokenizer.encode(format_llama3_qa_prompt(clean_q), add_special_tokens=False),
                dtype=torch.long, device=device,
            ).unsqueeze(0)

            stored_bpp_before = 0.0
            stored_bpp_after = 0.0

            if sys_name in ("bm25_rag_append_errata", "gated_bm25_rag"):
                t_e0 = time.perf_counter()
                docs = [tokenizer.decode(ep.context_pages[i], skip_special_tokens=True) for i in range(ep.context_pages.shape[0])]
                stored_bpp_before = round(sum(len(d.encode("utf-8")) for d in docs) / max(1, len(docs)), 2)
                encode_ms = (time.perf_counter() - t_e0) * 1000.0

                _sync_device(device)
                t_ed0 = time.perf_counter()
                for (p_i, e_ids_cpu), true_acc in zip(ep.edit_pages, gold_flags):
                    if sys_name == "gated_bm25_rag":
                        pred_acc, prob, _ = pathb.score_edit_gate(
                            e_ids_cpu.unsqueeze(0).to(device), llm,
                            device=device, amp_dtype=amp_dtype, autocast_enabled=autocast_enabled,
                        )
                        if true_acc:
                            acc_gates.append(prob)
                            gate_tp += int(pred_acc)
                            gate_fn += int(not pred_acc)
                        else:
                            rej_gates.append(prob)
                            gate_fp += int(pred_acc)
                            gate_tn += int(not pred_acc)
                        if pred_acc:
                            docs[int(p_i)] = docs[int(p_i)] + "\n" + tokenizer.decode(e_ids_cpu, skip_special_tokens=True)
                    else:
                        docs[int(p_i)] = docs[int(p_i)] + "\n" + tokenizer.decode(e_ids_cpu, skip_special_tokens=True)
                _sync_device(device)
                edit_ms = (time.perf_counter() - t_ed0) * 1000.0
                stored_bpp_after = round(sum(len(d.encode("utf-8")) for d in docs) / max(1, len(docs)), 2)

                _sync_device(device)
                t_q0 = time.perf_counter()
                bm25 = BM25PageRetriever(docs)
                rr = bm25.route(clean_q, k_deep=min(4, len(docs)), k_skim=0, device=device)
                chosen = rr.deep_idx[0].tolist()
                rag_txt = "\n\n".join(docs[i].strip() for i in chosen) + "\n"
                pref = embed_fn(_encode_unpadded_rag_ids(rag_txt, tokenizer, vocab_size, device)).to(llm_dtype)
                gen_txt = greedy_generate_llama3(llm, pref, p_prompt, tokenizer, max_new_tokens=20, user_open_ids=user_open_ids)
                _sync_device(device)
                first_query_ms = (time.perf_counter() - t_q0) * 1000.0

            elif sys_name == "raw_text_same_pages_k4":
                p_t = ep.context_pages.unsqueeze(0).to(device)
                enc_p = pathb.encode_book(
                    p_t, embed_fn, tokenizer, vocab_size, device=device,
                    amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids, num_spans=4,
                )
                bk, bge_bf16 = enc_p["bank"], enc_p["bge_keys_bf16"]
                docs = [tokenizer.decode(ep.context_pages[i], skip_special_tokens=True) for i in range(ep.context_pages.shape[0])]
                encode_ms = float(enc_p["encode_ms"])
                stored_bpp_before = round((sum(len(d.encode("utf-8")) for d in docs) / max(1, len(docs))) + float(enc_p["bge_keys_bytes_per_page"]), 2)

                _sync_device(device)
                t_ed0 = time.perf_counter()
                for (p_i, e_ids_cpu), true_acc in zip(ep.edit_pages, gold_flags):
                    pred_acc, prob, _ = pathb.score_edit_gate(
                        e_ids_cpu.unsqueeze(0).to(device), llm,
                        device=device, amp_dtype=amp_dtype, autocast_enabled=autocast_enabled,
                    )
                    if true_acc:
                        acc_gates.append(prob)
                        gate_tp += int(pred_acc)
                        gate_fn += int(not pred_acc)
                    else:
                        rej_gates.append(prob)
                        gate_fp += int(pred_acc)
                        gate_tn += int(not pred_acc)
                    if pred_acc:
                        docs[int(p_i)] = docs[int(p_i)] + "\n" + tokenizer.decode(e_ids_cpu, skip_special_tokens=True)
                _sync_device(device)
                edit_ms = (time.perf_counter() - t_ed0) * 1000.0
                stored_bpp_after = round((sum(len(d.encode("utf-8")) for d in docs) / max(1, len(docs))) + float(enc_p["bge_keys_bytes_per_page"]), 2)

                rq_ids = ep.metadata["router_query_ids"].unsqueeze(0).to(device)
                _sync_device(device)
                t_q0 = time.perf_counter()
                r_out, _, _ = pathb.route_and_assemble(
                    bk, bge_bf16, rq_ids, clean_q, embed_fn,
                    device=device, k_deep=4, salience_route_weight=salience_route_weight,
                )
                chosen = [int(x) for x in r_out.deep_idx[0].tolist()]
                rag_txt = "\n\n".join(docs[i].strip() for i in chosen) + "\n"
                pref = embed_fn(_encode_unpadded_rag_ids(rag_txt, tokenizer, vocab_size, device)).to(llm_dtype)
                gen_txt = greedy_generate_llama3(llm, pref, p_prompt, tokenizer, max_new_tokens=20, user_open_ids=user_open_ids)
                _sync_device(device)
                first_query_ms = (time.perf_counter() - t_q0) * 1000.0

            elif sys_name == "standalone_append_192tok":
                p_t = ep.context_pages.unsqueeze(0).to(device)
                enc_p = pathb.encode_book(
                    p_t, embed_fn, tokenizer, vocab_size, device=device,
                    amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids, num_spans=4,
                )
                base_bk, bge_bf16 = enc_p["bank"], enc_p["bge_keys_bf16"]
                encode_ms = float(enc_p["encode_ms"])
                stored_bpp_before = float(enc_p["stored_bytes_per_page_before_edit"])

                appended_by_page: dict[int, list[int]] = {}
                _sync_device(device)
                t_ed0 = time.perf_counter()
                for (p_i, e_ids_cpu), true_acc in zip(ep.edit_pages, gold_flags):
                    pred_acc, prob, _ = pathb.score_edit_gate(
                        e_ids_cpu.unsqueeze(0).to(device), llm,
                        device=device, amp_dtype=amp_dtype, autocast_enabled=autocast_enabled,
                    )
                    if true_acc:
                        acc_gates.append(prob)
                        gate_tp += int(pred_acc)
                        gate_fn += int(not pred_acc)
                    else:
                        rej_gates.append(prob)
                        gate_fp += int(pred_acc)
                        gate_tn += int(not pred_acc)
                    if pred_acc:
                        buf = appended_by_page.setdefault(int(p_i), [])
                        if buf and sep_id is not None:
                            buf.append(int(sep_id))
                        buf.extend(int(t) for t in e_ids_cpu.tolist())
                bk = _build_append_192tok_bank(
                    base_bk, appended_by_page, k_exact=pathb.k_exact, span_len=pathb.salience_span_len, sep_token_id=sep_id
                )
                _sync_device(device)
                edit_ms = (time.perf_counter() - t_ed0) * 1000.0
                stored_bpp_after = round((bk.storage_bytes() / max(1, p_t.shape[1])) + float(enc_p["bge_keys_bytes_per_page"]), 2)

                rq_ids = ep.metadata["router_query_ids"].unsqueeze(0).to(device)
                _sync_device(device)
                t_q0 = time.perf_counter()
                _, pref, _ = pathb.route_and_assemble(
                    bk, bge_bf16, rq_ids, clean_q, embed_fn,
                    device=device, k_deep=4, salience_route_weight=salience_route_weight,
                )
                pref = pref.to(llm_dtype)
                gen_txt = greedy_generate_llama3(llm, pref, p_prompt, tokenizer, max_new_tokens=20, user_open_ids=user_open_ids)
                _sync_device(device)
                first_query_ms = (time.perf_counter() - t_q0) * 1000.0

            else:  # standalone_discrete_span_bank (span-aligned overwrite)
                p_t = ep.context_pages.unsqueeze(0).to(device)
                enc_p = pathb.encode_book(
                    p_t, embed_fn, tokenizer, vocab_size, device=device,
                    amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids, num_spans=4,
                )
                bk, bge_bf16 = enc_p["bank"], enc_p["bge_keys_bf16"]
                encode_ms = float(enc_p["encode_ms"])
                stored_bpp_before = float(enc_p["stored_bytes_per_page_before_edit"])

                edit_ms = 0.0
                for (p_i, e_ids_cpu), true_acc in zip(ep.edit_pages, gold_flags):
                    bk, pred_acc, prob, dt_ed = pathb.apply_edit(
                        bk, int(p_i), e_ids_cpu.unsqueeze(0).to(device), llm, embed_fn, vocab_size,
                        device=device, amp_dtype=amp_dtype, autocast_enabled=autocast_enabled,
                        in_place=True, span_aligned_overwrite=True, sep_token_id=sep_id,
                    )
                    edit_ms += dt_ed
                    if true_acc:
                        acc_gates.append(prob)
                        gate_tp += int(pred_acc)
                        gate_fn += int(not pred_acc)
                    else:
                        rej_gates.append(prob)
                        gate_fp += int(pred_acc)
                        gate_tn += int(not pred_acc)
                stored_bpp_after = round((bk.storage_bytes() / max(1, p_t.shape[1])) + float(enc_p["bge_keys_bytes_per_page"]), 2)

                rq_ids = ep.metadata["router_query_ids"].unsqueeze(0).to(device)
                _sync_device(device)
                t_q0 = time.perf_counter()
                _, pref, _ = pathb.route_and_assemble(
                    bk, bge_bf16, rq_ids, clean_q, embed_fn,
                    device=device, k_deep=4, salience_route_weight=salience_route_weight,
                )
                pref = pref.to(llm_dtype)
                gen_txt = greedy_generate_llama3(llm, pref, p_prompt, tokenizer, max_new_tokens=20, user_open_ids=user_open_ids)
                _sync_device(device)
                first_query_ms = (time.perf_counter() - t_q0) * 1000.0

            qm = compute_community_qa_metrics(gen_txt, gold)
            fl = gen_txt.splitlines()[0] if gen_txt.strip() else ""
            canon_pred, canon_gold = canonicalize_numeric_answer(fl), canonicalize_numeric_answer(gold)
            norm_num_em = 1.0 if (canon_gold and canon_pred == canon_gold) else float(qm["extracted_em"])
            w2n_pred, w2n_gold = canonicalize_w2n_answer(fl), canonicalize_w2n_answer(gold)
            norm_w2n_em = 1.0 if (w2n_gold and w2n_pred == w2n_gold) else norm_num_em

            rec = {
                "episode_idx": ep_idx,
                "mode": ep_mode,
                "query_type": q_type,
                "gold": gold,
                "pred": gen_txt,
                "strict_em": float(qm["extracted_em"]),
                "normalized_numeric_em": float(norm_num_em),
                "normalized_w2n_em": float(norm_w2n_em),
                "f1": float(qm["token_f1"]),
                "active_tokens": float(pref.shape[1]),
                "encode_ms": round(encode_ms, 3),
                "edit_ms": round(edit_ms, 3),
                "first_query_ms": round(first_query_ms, 3),
                "stored_bytes_per_page_before_edit": stored_bpp_before,
                "stored_bytes_per_page_after_edit": stored_bpp_after,
                "stored_bytes_per_page": stored_bpp_after,
            }
            rows.append(rec)
            if ep_mode in rows_by_mode:
                rows_by_mode[ep_mode].append(rec)
            if q_type in rows_by_qtype:
                rows_by_qtype[q_type].append(rec)

        m_keys = (
            "strict_em", "normalized_numeric_em", "normalized_w2n_em", "f1", "active_tokens",
            "encode_ms", "edit_ms", "first_query_ms",
            "stored_bytes_per_page_before_edit", "stored_bytes_per_page_after_edit", "stored_bytes_per_page",
        )
        agg = _summarize_scalar_rows(rows, m_keys)
        agg["by_mode"] = {m: _summarize_scalar_rows(mr, m_keys) for m, mr in rows_by_mode.items()}
        agg["by_query_type"] = {
            qt: _summarize_scalar_rows(qr, ("strict_em", "normalized_numeric_em", "normalized_w2n_em"))
            for qt, qr in rows_by_qtype.items()
        }
        if gate_tp + gate_fn + gate_tn + gate_fp > 0:
            agg["gate_telemetry"] = {
                "tp": gate_tp, "tn": gate_tn, "fp": gate_fp, "fn": gate_fn,
                "false_accept_rate": round(gate_fp / max(1, gate_tn + gate_fp), 4),
                "false_reject_rate": round(gate_fn / max(1, gate_tp + gate_fn), 4),
                "gate_accuracy": round((gate_tp + gate_tn) / max(1, gate_tp + gate_tn + gate_fp + gate_fn), 4),
                "mean_gate_accepted": round(sum(acc_gates) / max(1, len(acc_gates)), 4),
                "mean_gate_rejected": round(sum(rej_gates) / max(1, len(rej_gates)), 4),
            }
        agg["episodes"] = rows
        sys_out[sys_name] = agg
        scores_norm_by_sys[sys_name] = [r["normalized_numeric_em"] for r in rows]
        scores_w2n_by_sys[sys_name] = [r["normalized_w2n_em"] for r in rows]
        print(
            f"[G6 LLAMA-GATE2 N={len(rows)} {sys_name}] norm_EM={agg['normalized_numeric_em']:.3f} "
            f"w2n={agg['normalized_w2n_em']:.3f} (bud={agg['by_mode']['budget']['normalized_numeric_em']:.3f}, "
            f"sym={agg['by_mode']['symbolic']['normalized_numeric_em']:.3f}) tok={agg['active_tokens']:.0f}",
            flush=True,
        )

    # Blind Edit Routing Slice (ZERO oracle p_i — route edit target page via BGE + lexical!)
    blind_half = 2 if is_mock else max(1, min(blind_edit_slice_n // 2, len(eps_budget), len(eps_sym)))
    blind_indices = list(range(blind_half)) + [len(eps_budget) + j for j in range(blind_half)]
    blind_rows: list[dict[str, Any]] = []
    edit_route_hits: list[float] = []
    for ep_idx in blind_indices:
        ep = all_eps[ep_idx]
        clean_q = tokenizer.decode(ep.metadata["router_query_ids"].tolist(), skip_special_tokens=True).strip()
        gold = str(ep.metadata["answer_str"]).strip()
        p_prompt = torch.tensor(
            tokenizer.encode(format_llama3_qa_prompt(clean_q), add_special_tokens=False), dtype=torch.long, device=device
        ).unsqueeze(0)
        p_t = ep.context_pages.unsqueeze(0).to(device)
        enc_p = pathb.encode_book(
            p_t, embed_fn, tokenizer, vocab_size, device=device,
            amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids, num_spans=4,
        )
        bk, bge_bf16 = enc_p["bank"], enc_p["bge_keys_bf16"]
        ep_hits: list[float] = []
        for p_true, e_ids_cpu in ep.edit_pages:
            e_ids = e_ids_cpu.unsqueeze(0).to(device)
            e_txt = tokenizer.decode(e_ids_cpu.tolist(), skip_special_tokens=True)
            p_pred = pathb.route_edit_target_page(
                bk, bge_bf16, e_ids, e_txt, embed_fn, device=device, salience_route_weight=salience_route_weight
            )
            hit = 1.0 if int(p_pred) == int(p_true) else 0.0
            edit_route_hits.append(hit)
            ep_hits.append(hit)
            bk, _, _, _ = pathb.apply_edit(
                bk, int(p_pred), e_ids, llm, embed_fn, vocab_size, device=device,
                amp_dtype=amp_dtype, autocast_enabled=autocast_enabled,
                in_place=True, span_aligned_overwrite=True, sep_token_id=sep_id,
            )
        rq_ids = ep.metadata["router_query_ids"].unsqueeze(0).to(device)
        _, pref, _ = pathb.route_and_assemble(
            bk, bge_bf16, rq_ids, clean_q, embed_fn, device=device, k_deep=4, salience_route_weight=salience_route_weight
        )
        gen_txt = greedy_generate_llama3(llm, pref.to(llm_dtype), p_prompt, tokenizer, max_new_tokens=20, user_open_ids=user_open_ids)
        qm = compute_community_qa_metrics(gen_txt, gold)
        fl = gen_txt.splitlines()[0] if gen_txt.strip() else ""
        cp, cg = canonicalize_numeric_answer(fl), canonicalize_numeric_answer(gold)
        nem = 1.0 if (cg and cp == cg) else float(qm["extracted_em"])
        wp, wg = canonicalize_w2n_answer(fl), canonicalize_w2n_answer(gold)
        wem = 1.0 if (wg and wp == wg) else nem
        blind_rows.append(
            {
                "episode_idx": ep_idx,
                "mode": str(ep.metadata.get("mode", "budget")),
                "query_type": str(ep.metadata.get("query_type", "")),
                "gold": gold,
                "pred": gen_txt,
                "edit_page_routing_accuracy": round(sum(ep_hits) / max(1, len(ep_hits)), 4),
                "strict_em": float(qm["extracted_em"]),
                "normalized_numeric_em": float(nem),
                "normalized_w2n_em": float(wem),
            }
        )
    oracle_slice = [sys_out["standalone_discrete_span_bank"]["episodes"][i]["normalized_numeric_em"] for i in blind_indices]
    blind_slice = [r["normalized_numeric_em"] for r in blind_rows]
    blind_summary = _summarize_scalar_rows(
        blind_rows, ("edit_page_routing_accuracy", "strict_em", "normalized_numeric_em", "normalized_w2n_em")
    )
    blind_summary["edit_turn_routing_accuracy"] = round(sum(edit_route_hits) / max(1, len(edit_route_hits)), 4)
    blind_summary["num_edit_turns"] = len(edit_route_hits)
    blind_summary["vs_oracle_edit_page_same_slice"] = paired_bootstrap_ci95(blind_slice, oracle_slice, seed=seed)
    blind_summary["episodes"] = blind_rows

    pairs = [
        ("standalone_discrete_span_bank", "standalone_append_192tok"),
        ("standalone_discrete_span_bank", "raw_text_same_pages_k4"),
        ("standalone_discrete_span_bank", "gated_bm25_rag"),
        ("standalone_discrete_span_bank", "bm25_rag_append_errata"),
        ("raw_text_same_pages_k4", "gated_bm25_rag"),
        ("gated_bm25_rag", "bm25_rag_append_errata"),
    ]
    return {
        "num_episodes": len(all_eps),
        "span_aligned_overwrite": True,
        "storage_bytes": pathb.storage_breakdown_bytes_per_page(num_spans=4),
        "systems": sys_out,
        "blind_edit_routing_slice": blind_summary,
        "pairwise_norm_em": compute_pairwise_suite(scores_norm_by_sys, pairs, seed=seed),
        "pairwise_w2n_em": compute_pairwise_suite(scores_w2n_by_sys, pairs, seed=seed),
    }


# ---------------------------------------------------------------------------
# Benchmark 2: llama_coding_n24 (65-file Repo + raw_text_same_pages_k4 Control)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_llama_coding_n24(
    llm: nn.Module,
    pathb: PathBDiscreteMemory,
    tokenizer,
    device: torch.device,
    *,
    num_distractors: int = 58,
    max_tasks: int = 24,
    salience_route_weight: float = 1.5,
    max_new_tokens: int = 256,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Coding N=24 on Llama-3.1-8B (1 unified run): bm25_rag_k8, bge_4span_rag_k4, standalone_discrete_k4, raw_text_same_pages_k4."""
    embed_fn = llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    sep_id = get_newline_sep_token_id(tokenizer)
    pathb.eval()

    n_dist = 8 if is_mock else num_distractors
    tasks = build_coding_n24_tasks(num_distractor_files=n_dist)[: (4 if is_mock else max_tasks)]
    user_open_ids = torch.tensor(
        tokenizer.encode(LLAMA3_USER_OPEN, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)

    systems = ("bm25_rag_k8", "bge_4span_rag_k4", "standalone_discrete_k4", "raw_text_same_pages_k4")
    out_systems: dict[str, Any] = {}
    pass_by_sys: dict[str, list[float]] = {s: [] for s in systems}
    # Cache Path B bank, gated docs, & routed pages so standalone_discrete_k4 and raw_text_same_pages_k4 are 100% matched!
    task_pathb_cache: dict[str, dict[str, Any]] = {}

    for sys_name in systems:
        rows: list[dict[str, Any]] = []
        ep_records: list[dict[str, Any]] = []
        gate_acc_probs: list[float] = []
        gate_rej_probs: list[float] = []
        gate_tp = gate_tn = gate_fp = gate_fn = 0

        for task in tasks:
            harness = CodingAgentHarness(task, use_initial_files=True)
            pack = _build_repo_pages(task, tokenizer, pathb.cfg.page_size)
            page_paths: list[str] = list(pack["page_paths"])
            core_page_idx = int(pack["core_page_indices"][0])
            session_page_idx = int(pack["session_page_idx"])
            pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
            core_fmt_new = f"# FILE: {task.core_file}\n{task.core_new_content}\n"
            init_scan = len(task.initial_files)
            harness.apply_file_edit(task.core_file, task.core_new_content)
            q_txt = build_pathless_coding_query(task)

            bm25 = None
            docs_live: list[str] = []
            bank = None
            bge_pages_bf16 = None
            mem_ms = 0.0

            if sys_name == "bm25_rag_k8":
                docs_live = list(pack["page_texts"])
                t0 = time.perf_counter()
                bm25 = BM25PageRetriever(docs_live)
                docs_live[core_page_idx] = core_fmt_new
                bm25.update_page(core_page_idx, core_fmt_new)
                sess_txt = docs_live[session_page_idx]
                for turn in task.session_turns:
                    sess_txt = sess_txt.rstrip() + "\n" + str(turn["text"]) + "\n"
                docs_live[session_page_idx] = sess_txt
                bm25.update_page(session_page_idx, sess_txt)
                mem_ms = (time.perf_counter() - t0) * 1000.0
            elif sys_name == "bge_4span_rag_k4":
                docs_live = list(pack["page_texts"])
                docs_live[core_page_idx] = core_fmt_new
                sess_txt = docs_live[session_page_idx]
                for turn in task.session_turns:
                    sess_txt = sess_txt.rstrip() + "\n" + str(turn["text"]) + "\n"
                docs_live[session_page_idx] = sess_txt
                _sync_device(device)
                t0 = time.perf_counter()
                bge_pages_bf16 = (
                    pathb.bge_toc.encode_book_pages(None, docs_live, tokenizer, num_spans=4, batch_size=64).to(torch.bfloat16)
                    if pathb.bge_toc is not None
                    else None
                )
                _sync_device(device)
                mem_ms = (time.perf_counter() - t0) * 1000.0
            else:  # standalone_discrete_k4 or raw_text_same_pages_k4
                if task.task_id in task_pathb_cache:
                    pc = task_pathb_cache[task.task_id]
                    bank, bge_pages_bf16, docs_live, mem_ms = pc["bank"], pc["bge"], pc["gated_docs"], pc["mem_ms"]
                else:
                    page_tensor = pack["page_tensor"].to(device)
                    enc_p = pathb.encode_book(
                        page_tensor, embed_fn, tokenizer, vocab_size, device=device,
                        amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids, num_spans=4,
                    )
                    bank, bge_pages_bf16 = enc_p["bank"], enc_p["bge_keys_bf16"]
                    docs_live = list(pack["page_texts"])
                    docs_live[core_page_idx] = core_fmt_new
                    core_toks_new = _pad_or_chunk_tokens(
                        tokenizer.encode(core_fmt_new, add_special_tokens=False), pathb.cfg.page_size, pad_id
                    )[0]
                    core_page_t = torch.tensor(core_toks_new, dtype=torch.long, device=device).unsqueeze(0)
                    _sync_device(device)
                    t0 = time.perf_counter()
                    bank, bge_pages_bf16 = pathb.reencode_single_page(
                        bank, bge_pages_bf16, core_page_idx, core_page_t, embed_fn, tokenizer, vocab_size,
                        device=device, amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids,
                    )
                    sess_gated = docs_live[session_page_idx]
                    for turn in task.session_turns:
                        e_toks = torch.tensor(
                            tokenizer.encode(str(turn["text"]), add_special_tokens=False), dtype=torch.long, device=device
                        ).unsqueeze(0)
                        bank, pred_acc, prob, _ = pathb.apply_edit(
                            bank, session_page_idx, e_toks, llm, embed_fn, vocab_size, device=device,
                            amp_dtype=amp_dtype, autocast_enabled=autocast_enabled,
                            in_place=True, span_aligned_overwrite=True, sep_token_id=sep_id,
                        )
                        if bool(turn.get("accepted", False)):
                            gate_acc_probs.append(prob)
                            gate_tp += int(pred_acc)
                            gate_fn += int(not pred_acc)
                        else:
                            gate_rej_probs.append(prob)
                            gate_fp += int(pred_acc)
                            gate_tn += int(not pred_acc)
                        if pred_acc:
                            sess_gated = sess_gated.rstrip() + "\n" + str(turn["text"]) + "\n"
                    docs_live[session_page_idx] = sess_gated
                    _sync_device(device)
                    mem_ms = (time.perf_counter() - t0) * 1000.0
                    task_pathb_cache[task.task_id] = {
                        "bank": bank, "bge": bge_pages_bf16, "gated_docs": docs_live, "mem_ms": mem_ms, "routes": {},
                    }

            prefix_toks: list[int] = []
            core_hits, sess_hits, both_hits = [], [], []
            chosen_by_caller: dict[str, list[int]] = {}
            patches_by_caller: dict[str, str] = {}

            for caller_path in task.caller_files:
                cur_code = harness.read_file(caller_path)
                excluded = {i for i, p in enumerate(page_paths) if p == caller_path}
                if sys_name == "bm25_rag_k8":
                    harness.search_calls += 1
                    rr = bm25.route(q_txt, k_deep=len(docs_live), k_skim=0, device=device)
                    chosen = [int(i) for i in torch.argsort(rr.logits[0], descending=True).tolist() if int(i) not in excluded][:8]
                    for idx_c in chosen:
                        p_n = page_paths[idx_c]
                        if p_n != task.session_notes_path and p_n in harness.files:
                            _ = harness.read_file(p_n)
                    pref = embed_fn(
                        _encode_unpadded_rag_ids("\n\n".join(docs_live[i].rstrip() for i in chosen) + "\n", tokenizer, vocab_size, device)
                    ).to(llm_dtype)
                elif sys_name == "bge_4span_rag_k4":
                    harness.search_calls += 1
                    sc = (
                        pathb.bge_toc.score_book(pathb.bge_toc.encode_texts([q_txt], is_query=True), bge_pages_bf16)[0]
                        if (pathb.bge_toc is not None and bge_pages_bf16 is not None)
                        else torch.zeros((len(docs_live),), device=device)
                    )
                    chosen = [int(i) for i in torch.argsort(sc, descending=True).tolist() if int(i) not in excluded][:4]
                    for idx_c in chosen:
                        p_n = page_paths[idx_c]
                        if p_n != task.session_notes_path and p_n in harness.files:
                            _ = harness.read_file(p_n)
                    pref = embed_fn(
                        _encode_unpadded_rag_ids("\n\n".join(docs_live[i].rstrip() for i in chosen) + "\n", tokenizer, vocab_size, device)
                    ).to(llm_dtype)
                else:  # standalone_discrete_k4 or raw_text_same_pages_k4
                    pb_routes = task_pathb_cache[task.task_id]["routes"]
                    if caller_path not in pb_routes:
                        rq_ids = torch.tensor(
                            tokenizer.encode(q_txt, add_special_tokens=False), dtype=torch.long, device=device
                        ).unsqueeze(0)
                        r_out, pref_pb, _ = pathb.route_and_assemble(
                            bank, bge_pages_bf16, rq_ids, q_txt, embed_fn,
                            device=device, k_deep=4, salience_route_weight=salience_route_weight, excluded_indices=excluded,
                        )
                        pb_routes[caller_path] = ([int(x) for x in r_out.deep_idx[0].tolist()], pref_pb)
                    chosen, pref_pb = pb_routes[caller_path]
                    if sys_name == "standalone_discrete_k4":
                        pref = pref_pb.to(llm_dtype)
                    else:
                        harness.search_calls += 1
                        for idx_c in chosen:
                            p_n = page_paths[idx_c]
                            if p_n != task.session_notes_path and p_n in harness.files:
                                _ = harness.read_file(p_n)
                        pref = embed_fn(
                            _encode_unpadded_rag_ids("\n\n".join(docs_live[i].rstrip() for i in chosen) + "\n", tokenizer, vocab_size, device)
                        ).to(llm_dtype)

                chosen_set = set(chosen)
                chosen_by_caller[caller_path] = chosen
                prefix_toks.append(int(pref.shape[1]))
                core_hits.append(float(core_page_idx in chosen_set))
                sess_hits.append(float(session_page_idx in chosen_set))
                both_hits.append(float((core_page_idx in chosen_set) and (session_page_idx in chosen_set)))

                p_ids = torch.tensor(
                    tokenizer.encode(format_llama3_edit_prompt(task, caller_path, cur_code), add_special_tokens=False),
                    dtype=torch.long, device=device,
                ).unsqueeze(0)
                raw_out = greedy_generate_llama3(
                    llm, pref, p_ids, tokenizer,
                    max_new_tokens=(24 if is_mock else max_new_tokens),
                    user_open_ids=user_open_ids,
                    stop_fn=lambda d: _should_stop_code_gen(d, expect_opening_fence=False),
                )
                patched = _extract_python_code_block("```python\n" + raw_out, cur_code)
                patches_by_caller[caller_path] = patched
                harness.apply_file_edit(caller_path, patched)

            # Homonym check
            raw_homonym = harness.read_file(task.homonym_file)
            excl_h = {i for i, p in enumerate(page_paths) if p == task.homonym_file}
            if sys_name == "bm25_rag_k8":
                rr_h = bm25.route(q_txt, k_deep=len(docs_live), k_skim=0, device=device)
                ch_h = [int(i) for i in torch.argsort(rr_h.logits[0], descending=True).tolist() if int(i) not in excl_h][:8]
                pref_h = embed_fn(
                    _encode_unpadded_rag_ids("\n\n".join(docs_live[i].rstrip() for i in ch_h) + "\n", tokenizer, vocab_size, device)
                ).to(llm_dtype)
            elif sys_name == "bge_4span_rag_k4":
                sc_h = (
                    pathb.bge_toc.score_book(pathb.bge_toc.encode_texts([q_txt], is_query=True), bge_pages_bf16)[0]
                    if (pathb.bge_toc is not None and bge_pages_bf16 is not None)
                    else torch.zeros((len(docs_live),), device=device)
                )
                ch_h = [int(i) for i in torch.argsort(sc_h, descending=True).tolist() if int(i) not in excl_h][:4]
                pref_h = embed_fn(
                    _encode_unpadded_rag_ids("\n\n".join(docs_live[i].rstrip() for i in ch_h) + "\n", tokenizer, vocab_size, device)
                ).to(llm_dtype)
            else:
                pb_routes = task_pathb_cache[task.task_id]["routes"]
                if "__homonym__" not in pb_routes:
                    rq_ids = torch.tensor(
                        tokenizer.encode(q_txt, add_special_tokens=False), dtype=torch.long, device=device
                    ).unsqueeze(0)
                    r_h, pref_hp, _ = pathb.route_and_assemble(
                        bank, bge_pages_bf16, rq_ids, q_txt, embed_fn,
                        device=device, k_deep=4, salience_route_weight=salience_route_weight, excluded_indices=excl_h,
                    )
                    pb_routes["__homonym__"] = ([int(x) for x in r_h.deep_idx[0].tolist()], pref_hp)
                ch_h, pref_hp = pb_routes["__homonym__"]
                if sys_name == "standalone_discrete_k4":
                    pref_h = pref_hp.to(llm_dtype)
                else:
                    pref_h = embed_fn(
                        _encode_unpadded_rag_ids("\n\n".join(docs_live[i].rstrip() for i in ch_h) + "\n", tokenizer, vocab_size, device)
                    ).to(llm_dtype)

            h_ids = torch.tensor(
                tokenizer.encode(format_llama3_homonym_prompt(task, raw_homonym), add_special_tokens=False),
                dtype=torch.long, device=device,
            ).unsqueeze(0)
            raw_h = greedy_generate_llama3(
                llm, pref_h, h_ids, tokenizer, max_new_tokens=(16 if is_mock else 96),
                user_open_ids=user_open_ids, stop_fn=_should_stop_homonym_gen,
            )
            if "KEEP_UNCHANGED" not in raw_h.upper():
                cand_h = _extract_python_code_block(raw_h, raw_homonym)
                if cand_h.strip() and cand_h.strip() != raw_homonym.strip():
                    harness.apply_file_edit(task.homonym_file, cand_h)

            harness.memory_update_ms = mem_ms
            harness.active_prefix_tokens = int(round(sum(prefix_toks) / max(1, len(prefix_toks))))
            harness.peak_vram_mb = _peak_vram_mb(device)
            ev = harness.evaluate(run_subprocess=True)
            passed = 1.0 if (ev.patch_correct and ev.hidden_test_passed) else 0.0
            pass_by_sys[sys_name].append(passed)

            row_agg = {
                "task_id": ev.task_id,
                "patch_correct": ev.patch_correct,
                "hidden_test_passed": ev.hidden_test_passed,
                "stale_signature_used": ev.stale_signature_used,
                "decoy_value_used": ev.decoy_value_used,
                "homonym_preserved": ev.homonym_preserved,
                "files_read_total": int(init_scan + ev.files_read_total),
                "initial_scan_files": init_scan,
                "incremental_files_read": int(ev.files_read_total),
                "repeated_file_reads": int(ev.repeated_file_reads),
                "search_calls": int(ev.search_calls),
                "active_prefix_tokens": int(ev.active_prefix_tokens),
                "memory_update_ms": round(mem_ms, 3),
                "modular_reencode_ms": round(mem_ms, 3),
                "delta_update_ms": 0.0,
                "generation_ms": 0.0,
                "wall_clock_ms": round(ev.wall_clock_ms, 3),
                "peak_vram_mb": ev.peak_vram_mb,
                "router_core_hit_rate": round(sum(core_hits) / max(1, len(core_hits)), 4),
                "router_session_hit_rate": round(sum(sess_hits) / max(1, len(sess_hits)), 4),
                "router_both_hit_rate": round(sum(both_hits) / max(1, len(both_hits)), 4),
            }
            rows.append(row_agg)
            ep_records.append(
                {**row_agg, "pass_at_1": passed, "chosen_pages_by_caller": chosen_by_caller, "generated_patches": patches_by_caller}
            )

        agg = _aggregate_system_rows(rows)
        agg["pass_at_1_ci95"] = _wilson_ci95(float(agg["pass_at_1"]), len(rows))
        if gate_tp + gate_tn + gate_fp + gate_fn > 0:
            agg["gate_telemetry"] = {
                "tp": gate_tp, "tn": gate_tn, "fp": gate_fp, "fn": gate_fn,
                "false_accept_rate": round(gate_fp / max(1, gate_tn + gate_fp), 4),
                "false_reject_rate": round(gate_fn / max(1, gate_tp + gate_fn), 4),
                "mean_gate_accepted": round(sum(gate_acc_probs) / max(1, len(gate_acc_probs)), 4),
                "mean_gate_rejected": round(sum(gate_rej_probs) / max(1, len(gate_rej_probs)), 4),
            }
        agg["episodes"] = ep_records
        out_systems[sys_name] = agg
        print(
            f"[G6 LLAMA-CODE N={len(rows)} {sys_name}] pass@1={agg['pass_at_1']:.3f} both_hit={agg['router_both_hit_rate']:.3f} "
            f"decoy={agg['decoy_value_used_rate']:.3f} tok={agg['active_prefix_tokens_mean']:.0f}",
            flush=True,
        )

    for se, re_ep in zip(out_systems["standalone_discrete_k4"]["episodes"], out_systems["raw_text_same_pages_k4"]["episodes"]):
        assert se["chosen_pages_by_caller"] == re_ep["chosen_pages_by_caller"], (
            f"Llama Path B vs raw_text_same_pages_k4 page mismatch on {se['task_id']}"
        )

    pairs = [
        ("standalone_discrete_k4", "raw_text_same_pages_k4"),
        ("standalone_discrete_k4", "bge_4span_rag_k4"),
        ("standalone_discrete_k4", "bm25_rag_k8"),
        ("raw_text_same_pages_k4", "bge_4span_rag_k4"),
        ("bge_4span_rag_k4", "bm25_rag_k8"),
    ]
    return {
        "num_tasks": len(tasks),
        "single_run_all_baselines": True,
        "identical_pages_verified": True,
        "systems": out_systems,
        "pairwise_pass_at_1": compute_pairwise_suite(pass_by_sys, pairs, seed=2026),
    }


# ---------------------------------------------------------------------------
# Benchmark 3: llama_ood_kv_4k (Per-Head 1D-Pooled SnapKV + Unified Timing)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_llama_ood_kv_4k(
    llm: nn.Module,
    pathb: PathBDiscreteMemory,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    context_length: int = 4096,
    eval_n: int = 48,
    ppl_n: int = 16,
    salience_route_weight: float = 1.5,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """OOD RULER 4K (N=48) + WikiText PPL (N=16) on Llama-3.1-8B with:
    - True per-layer, per-KV-head (GQA) 1D-pooled SnapKV (`pool_kernel_size=5, recent_ratio=0.10`, ~18 GB VRAM).
    - Unified timing split (`encode_or_evict_ms`, `prefill_ms`, `decode_ms`, `decode_ms_per_tok`, `total_query_ms`).
    - Identical raw query `ep["question"]` across all primary arms (+ `standalone_discrete_router_query` transparency arm).
    """
    embed_fn = llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    pathb.eval()

    eff_ctx = 1024 if is_mock else context_length
    num_p = max(4, eff_ctx // pathb.cfg.page_size)
    per_task = 2 if is_mock else max(1, eval_n // 3)
    ruler_eps: list[dict[str, Any]] = []
    for t_name in ("s_niah", "mk_niah", "vt"):
        ruler_eps.extend(
            build_ruler_episodes(
                task_name=t_name, context_length=eff_ctx, num_episodes=per_task,
                tokenizer=tokenizer, val_pool=val_pool, page_size=pathb.cfg.page_size,
                seed=seed + 700, mk_distractors=48, vt_hops=3, is_mock=is_mock,
            )
        )
    ppl_eps = list(ChunkedTextPageDataset(val_pool, num_pages=num_p, page_size=pathb.cfg.page_size, turn_len=32))[
        : (3 if is_mock else ppl_n)
    ]

    iso_token_budget = min(4, num_p) * int(pathb.k_exact)
    stop_ids = get_llama3_stop_token_ids(tokenizer)

    systems = (
        "full_kv_bf16",
        "query_agnostic_h2o",
        "query_aware_snapkv",
        "standalone_discrete_span_bank",
        "standalone_discrete_router_query",
    )
    out_sys: dict[str, Any] = {}
    em_by_sys: dict[str, list[float]] = {}

    for sys_name in systems:
        ppl_nlls: list[float] = []
        for ep in ppl_eps:
            p_t = ep.context_pages.unsqueeze(0).to(device)
            q_ids = ep.query_ids.unsqueeze(0).to(device)
            a_ids = ep.answer_ids.unsqueeze(0).to(device)
            ctx_emb = embed_fn(p_t.reshape(1, -1)).to(llm_dtype)
            orig_len = int(ctx_emb.shape[1])

            if sys_name == "full_kv_bf16":
                pre = llm(inputs_embeds=ctx_emb, attention_mask=torch.ones((1, orig_len), device=device, dtype=torch.long), use_cache=True)
                nll, _ = compute_nll_and_tok_acc_from_kv(llm, pre.past_key_values, orig_len, q_ids, a_ids, orig_seq_len=orig_len)
            elif sys_name == "query_agnostic_h2o":
                pkv, attn_s, _ = compute_memory_safe_obs_attention_and_kv(llm, ctx_emb, query_embeds=None, obs_window=32, per_head=False)
                ev_kv, k_idx = evict_past_key_values_h2o_attention(pkv, attn_s, keep_budget=iso_token_budget, sink_tokens=4, recent_ratio=0.10)
                nll, _ = compute_nll_and_tok_acc_from_kv(llm, ev_kv, int(k_idx.shape[-1]), q_ids, a_ids, orig_seq_len=orig_len)
            elif sys_name == "query_aware_snapkv":
                ctx_pkv, attn_lh, _ = compute_memory_safe_obs_attention_and_kv(llm, ctx_emb, query_embeds=embed_fn(q_ids).to(llm_dtype), per_head=True)
                ev_kv, k_idx = evict_past_key_values_h2o_attention(
                    ctx_pkv, attn_lh, keep_budget=iso_token_budget, sink_tokens=4, recent_ratio=0.10, pool_kernel_size=5, pool_mode="avg"
                )
                nll, _ = compute_nll_and_tok_acc_from_kv(llm, ev_kv, int(k_idx.shape[-1]), q_ids, a_ids, orig_seq_len=orig_len)
            else:  # standalone_discrete_span_bank or standalone_discrete_router_query
                q_txt_p = tokenizer.decode(q_ids[0].tolist(), skip_special_tokens=True)
                enc_p = pathb.encode_book(
                    p_t, embed_fn, tokenizer, vocab_size, device=device,
                    amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids,
                )
                _, pref, _ = pathb.route_and_assemble(
                    enc_p["bank"], enc_p["bge_keys_bf16"], q_ids, q_txt_p, embed_fn,
                    device=device, k_deep=min(4, num_p), salience_route_weight=salience_route_weight,
                )
                pref = pref.to(llm_dtype)
                pre = llm(inputs_embeds=pref, attention_mask=torch.ones((1, pref.shape[1]), device=device, dtype=torch.long), use_cache=True)
                nll, _ = compute_nll_and_tok_acc_from_kv(llm, pre.past_key_values, int(pref.shape[1]), q_ids, a_ids)
            ppl_nlls.append(nll)

        rows: list[dict[str, Any]] = []
        rows_by_task: dict[str, list[dict[str, Any]]] = {"s_niah": [], "mk_niah": [], "vt": []}
        evict_vram_peak = 0.0
        sdpa_vram_peak = 0.0

        for ep_idx, ep in enumerate(ruler_eps):
            p_t = ep["pages"].unsqueeze(0).to(device)
            ctx_emb = embed_fn(p_t.reshape(1, -1)).to(llm_dtype)
            orig_len = int(ctx_emb.shape[1])
            raw_q_str = str(ep["question"])
            short_q_str = str(ep["router_query"])
            q_ids = torch.tensor(
                tokenizer.encode(format_llama3_qa_prompt(raw_q_str), add_special_tokens=False),
                dtype=torch.long, device=device,
            ).unsqueeze(0)
            route_q_use = short_q_str if sys_name == "standalone_discrete_router_query" else raw_q_str
            rq_ids = torch.tensor(
                _encode_text_or_mock(route_q_use, tokenizer, vocab_size), dtype=torch.long, device=device
            ).unsqueeze(0)

            encode_or_evict_ms = 0.0
            prefix_prefill_ms = 0.0
            route_ms = 0.0
            if sys_name == "full_kv_bf16":
                _reset_peak_vram(device)
                _sync_device(device)
                t_p0 = time.perf_counter()
                pre = llm(inputs_embeds=ctx_emb, attention_mask=torch.ones((1, orig_len), device=device, dtype=torch.long), use_cache=True)
                _sync_device(device)
                prefix_prefill_ms = (time.perf_counter() - t_p0) * 1000.0
                kv_c, kv_len, pos_orig = pre.past_key_values, orig_len, orig_len
            elif sys_name == "query_agnostic_h2o":
                _reset_peak_vram(device)
                _sync_device(device)
                t_ev0 = time.perf_counter()
                pkv, attn_s, _ = compute_memory_safe_obs_attention_and_kv(llm, ctx_emb, query_embeds=None, obs_window=32, per_head=False)
                kv_c, k_idx = evict_past_key_values_h2o_attention(pkv, attn_s, keep_budget=iso_token_budget, sink_tokens=4, recent_ratio=0.10)
                _sync_device(device)
                encode_or_evict_ms = (time.perf_counter() - t_ev0) * 1000.0
                evict_vram_peak = max(evict_vram_peak, _peak_vram_mb(device))
                kv_len, pos_orig = int(k_idx.shape[-1]), orig_len
                _reset_peak_vram(device)
            elif sys_name == "query_aware_snapkv":
                _reset_peak_vram(device)
                _sync_device(device)
                t_ev0 = time.perf_counter()
                ctx_pkv, attn_lh, _ = compute_memory_safe_obs_attention_and_kv(
                    llm, ctx_emb, query_embeds=embed_fn(q_ids).to(llm_dtype), per_head=True
                )
                kv_c, k_idx = evict_past_key_values_h2o_attention(
                    ctx_pkv, attn_lh, keep_budget=iso_token_budget, sink_tokens=4, recent_ratio=0.10,
                    pool_kernel_size=5, pool_mode="avg",
                )
                _sync_device(device)
                encode_or_evict_ms = (time.perf_counter() - t_ev0) * 1000.0
                evict_vram_peak = max(evict_vram_peak, _peak_vram_mb(device))
                kv_len, pos_orig = int(k_idx.shape[-1]), orig_len
                _reset_peak_vram(device)
            else:  # standalone_discrete_span_bank or standalone_discrete_router_query
                _reset_peak_vram(device)
                enc_p = pathb.encode_book(
                    p_t, embed_fn, tokenizer, vocab_size, device=device,
                    amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids,
                )
                encode_or_evict_ms = float(enc_p["encode_ms"])
                _, pref, route_ms = pathb.route_and_assemble(
                    enc_p["bank"], enc_p["bge_keys_bf16"], rq_ids, route_q_use, embed_fn,
                    device=device, k_deep=min(4, num_p), salience_route_weight=salience_route_weight,
                )
                pref = pref.to(llm_dtype)
                _sync_device(device)
                t_p0 = time.perf_counter()
                pre = llm(inputs_embeds=pref, attention_mask=torch.ones((1, pref.shape[1]), device=device, dtype=torch.long), use_cache=True)
                _sync_device(device)
                prefix_prefill_ms = (time.perf_counter() - t_p0) * 1000.0
                kv_c, kv_len, pos_orig = pre.past_key_values, int(pref.shape[1]), int(pref.shape[1])

            gen_ids, timing = greedy_generate_from_kv_cache(
                llm, kv_c, kv_len, q_ids, max_new_tokens=24,
                eos_token_id=stop_ids, orig_seq_len=pos_orig, tokenizer=tokenizer,
                stop_on_newline=True, return_timing=True,
            )
            sdpa_vram_peak = max(sdpa_vram_peak, _peak_vram_mb(device))
            prefill_ms = prefix_prefill_ms + float(timing["prefill_ms"])
            decode_ms = float(timing["decode_ms"])
            total_query_ms = route_ms + prefill_ms + decode_ms

            raw_gen = _clean_llama3_decoded(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
            qm = compute_community_qa_metrics(raw_gen, ep["answer"])
            rec = {
                "episode_idx": ep_idx,
                "task": ep["task"],
                "gold": ep["answer"],
                "pred": raw_gen,
                "extracted_em": float(qm["extracted_em"]),
                "substring_match": float(qm["substring_match"]),
                "token_f1": float(qm["token_f1"]),
                "active_kv_tokens": float(kv_len),
                "encode_or_evict_ms": round(encode_or_evict_ms, 3),
                "route_ms": round(route_ms, 3),
                "prefill_ms": round(prefill_ms, 3),
                "decode_ms": round(decode_ms, 3),
                "decode_ms_per_tok": float(timing["decode_ms_per_tok"]),
                "total_query_ms": round(total_query_ms, 3),
                "eviction_ms": round(encode_or_evict_ms, 3),
                "sdpa_inference_ms": round(total_query_ms, 3),
            }
            rows.append(rec)
            rows_by_task[ep["task"]].append(rec)

        m_keys = (
            "extracted_em", "substring_match", "token_f1", "active_kv_tokens",
            "encode_or_evict_ms", "route_ms", "prefill_ms", "decode_ms", "decode_ms_per_tok", "total_query_ms",
            "eviction_ms", "sdpa_inference_ms",
        )
        agg = _summarize_scalar_rows(rows, m_keys)
        agg["query_input_mode"] = "router_query" if sys_name == "standalone_discrete_router_query" else "raw_question"
        agg["wikitext_ppl"] = round(math.exp(min(20.0, sum(ppl_nlls) / max(1, len(ppl_nlls)))), 4)
        agg["sdpa_inference_peak_vram_mb"] = sdpa_vram_peak
        if sys_name in ("query_agnostic_h2o", "query_aware_snapkv"):
            agg["eviction_peak_vram_mb"] = evict_vram_peak
            agg["snapkv_per_head_1d_pooled"] = bool(sys_name == "query_aware_snapkv")
        agg["by_task"] = {t: _summarize_scalar_rows(rows_by_task[t], m_keys) for t in ("s_niah", "mk_niah", "vt")}
        agg["episodes"] = rows
        out_sys[sys_name] = agg
        em_by_sys[sys_name] = [r["extracted_em"] for r in rows]
        print(
            f"[G6 LLAMA-OOD-KV {sys_name}] PPL={agg['wikitext_ppl']:.2f} EM={agg['extracted_em']:.3f} "
            f"kv={agg['active_kv_tokens']:.0f} enc/ev={agg['encode_or_evict_ms']:.1f}ms "
            f"pf={agg['prefill_ms']:.1f}ms dec={agg['decode_ms']:.1f}ms ({agg['decode_ms_per_tok']:.2f}ms/t)",
            flush=True,
        )

    pairs = [
        ("standalone_discrete_span_bank", "query_aware_snapkv"),
        ("standalone_discrete_router_query", "query_aware_snapkv"),
        ("standalone_discrete_span_bank", "standalone_discrete_router_query"),
        ("standalone_discrete_span_bank", "query_agnostic_h2o"),
        ("standalone_discrete_span_bank", "full_kv_bf16"),
        ("query_aware_snapkv", "query_agnostic_h2o"),
    ]
    return {
        "context_length": eff_ctx,
        "iso_token_budget": iso_token_budget,
        "snapkv_config": {"per_layer_per_kv_head_gqa": True, "pool_kernel_size": 5, "pool_mode": "avg", "recent_ratio": 0.10, "sink_tokens": 4},
        "systems": out_sys,
        "pairwise_extracted_em": compute_pairwise_suite(em_by_sys, pairs, seed=seed),
    }


# ---------------------------------------------------------------------------
# Benchmark 4: llama_ruler_16k_32k_and_profile + True Two-Bank KV Reuse
# ---------------------------------------------------------------------------


@torch.no_grad()
def benchmark_two_bank_discrete_kv_reuse(
    llm: nn.Module,
    pathb: PathBDiscreteMemory,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    page_counts: tuple[int, ...] = (16, 64, 128),
    num_trials: int = 12,
    is_mock: bool = False,
) -> dict[str, Any]:
    """True Two-Bank Path (B) KV Reuse on Llama-3.1-8B (0 Z_book) testing first (p0), middle (p1), and last (pEnd)
    edits — NO deep_pages[-1] hardcoding, NO blind `[:-32]` slice — plus realistic post-rerouting cache-hit telemetry."""
    embed_fn = llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    sep_id = get_newline_sep_token_id(tokenizer)
    use_pos_ids = _llm_accepts_position_ids(llm)
    pathb.eval()

    eff_pages = (4, 8) if is_mock else page_counts
    eff_trials = 3 if is_mock else num_trials
    rng = random.Random(2026)
    filler_vocab = val_pool if val_pool else list(range(2, max(16, vocab_size - 2)))
    q_ids = torch.tensor(
        tokenizer.encode(format_llama3_qa_prompt("What is the updated protocol for Service HarborGate?"), add_special_tokens=False),
        dtype=torch.long, device=device,
    ).unsqueeze(0)
    edit1_ids = torch.tensor(
        _encode_text_or_mock("Approved revision: Service HarborGate updated protocol to TLS-1.2.", tokenizer, vocab_size),
        dtype=torch.long, device=device,
    ).unsqueeze(0)
    edit2_ids = torch.tensor(
        _encode_text_or_mock("Approved revision: Service HarborGate updated protocol to TLS-1.3.", tokenizer, vocab_size),
        dtype=torch.long, device=device,
    ).unsqueeze(0)
    u_emb = embed_fn(
        torch.tensor(tokenizer.encode(LLAMA3_USER_OPEN, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
    ).to(llm_dtype)
    q_emb = embed_fn(q_ids).to(llm_dtype)

    out: dict[str, Any] = {"architecture": "PathB_TrueTwoBank_ZeroZBook"}
    for num_p in eff_pages:
        pages = [[filler_vocab[rng.randrange(len(filler_vocab))] for _ in range(pathb.cfg.page_size)] for _ in range(num_p)]
        p_t = torch.tensor(pages, dtype=torch.long, device=device).unsqueeze(0)
        enc_p = pathb.encode_book(
            p_t, embed_fn, tokenizer, vocab_size, device=device,
            amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids,
        )
        kd = min(4, num_p)
        deep_pages = list(range(kd))
        edit_positions = [("first_p0", deep_pages[0]), ("middle_p1", deep_pages[min(1, kd - 1)]), ("last_p_end", deep_pages[-1])]

        by_edit_pos: dict[str, Any] = {}
        for pos_label, edited_page in edit_positions:
            delta_open = torch.ones((1, pathb.cfg.grid_h, pathb.cfg.grid_w), device=device, dtype=torch.float32)
            bank_t1 = enc_p["bank"].update_page_from_delta(
                page_indices=torch.tensor([edited_page], dtype=torch.long, device=device),
                edit_token_ids=edit1_ids,
                edit_embeds=embed_fn(edit1_ids),
                salience_gate=pathb.salience_gate,
                delta_gates=delta_open,
                vocab_size=vocab_size,
                in_place=False,
                bank_write_threshold=0.45,
                span_aligned_overwrite=True,
                salience_span_len=pathb.salience_span_len,
                sep_token_id=sep_id,
                embed_tokens_fn=embed_fn,
            )
            b1_before_raw, _, b1_pages_b, _ = pathb.assemble_two_bank_prefix(bank_t1, deep_pages, embed_fn)

            bank_t2 = bank_t1.update_page_from_delta(
                page_indices=torch.tensor([edited_page], dtype=torch.long, device=device),
                edit_token_ids=edit2_ids,
                edit_embeds=embed_fn(edit2_ids),
                salience_gate=pathb.salience_gate,
                delta_gates=delta_open,
                vocab_size=vocab_size,
                in_place=False,
                bank_write_threshold=0.45,
                span_aligned_overwrite=True,
                salience_span_len=pathb.salience_span_len,
                sep_token_id=sep_id,
                embed_tokens_fn=embed_fn,
            )
            b1_after_raw, b2_after_raw, b1_pages_a, b2_pages_a = pathb.assemble_two_bank_prefix(bank_t2, deep_pages, embed_fn)
            assert b1_pages_b == b1_pages_a and b2_pages_a == [edited_page]
            assert edited_page not in b1_pages_a

            bank1_before = torch.cat((u_emb, b1_before_raw.to(llm_dtype)), dim=1)
            bank1_after = torch.cat((u_emb, b1_after_raw.to(llm_dtype)), dim=1)
            tail_after = torch.cat((b2_after_raw.to(llm_dtype), q_emb), dim=1)
            imm_len = int(bank1_before.shape[1])
            tail_len = int(tail_after.shape[1])
            total_len = imm_len + tail_len

            embed_diff = float((bank1_after.float() - bank1_before.float()).abs().max().item())
            attn_b1 = torch.ones((1, imm_len), device=device, dtype=torch.long)
            pkv_b1_before = llm(inputs_embeds=bank1_before, attention_mask=attn_b1, use_cache=True).past_key_values
            pkv_b1_after = llm(inputs_embeds=bank1_after, attention_mask=attn_b1, use_cache=True).past_key_values

            max_kv_diff = 0.0
            for (k0, v0), (k1, v1) in zip(_extract_kv_pairs(pkv_b1_before), _extract_kv_pairs(pkv_b1_after)):
                max_kv_diff = max(
                    max_kv_diff,
                    float((k1.float() - k0.float()).abs().max().item()),
                    float((v1.float() - v0.float()).abs().max().item()),
                )

            full_embeds = torch.cat((bank1_after, tail_after), dim=1)
            attn_full = torch.ones((1, total_len), device=device, dtype=torch.long)
            logits_full = llm(inputs_embeds=full_embeds, attention_mask=attn_full, use_cache=True).logits[:, -1, :]

            reuse_kwargs: dict[str, Any] = {
                "inputs_embeds": tail_after,
                "attention_mask": attn_full,
                "past_key_values": _clone_kv_cache(pkv_b1_before),
                "use_cache": True,
            }
            if use_pos_ids:
                reuse_kwargs["position_ids"] = torch.arange(imm_len, total_len, device=device, dtype=torch.long).unsqueeze(0)
            logits_reused = llm(**reuse_kwargs).logits[:, -1, :]
            logits_diff = float((logits_reused.float() - logits_full.float()).abs().max().item())

            for _ in range(2):
                _ = llm(inputs_embeds=full_embeds, attention_mask=attn_full, use_cache=True)
                _ = llm(**dict(reuse_kwargs, past_key_values=_clone_kv_cache(pkv_b1_before)))

            full_ms_l, reuse_tot_l, reuse_cmp_l = [], [], []
            for _ in range(eff_trials):
                _sync_device(device)
                t0 = time.perf_counter()
                _ = llm(inputs_embeds=full_embeds, attention_mask=attn_full, use_cache=True)
                _sync_device(device)
                full_ms_l.append((time.perf_counter() - t0) * 1000.0)

                _sync_device(device)
                t1 = time.perf_counter()
                pw = _clone_kv_cache(pkv_b1_before)
                _sync_device(device)
                t1c = time.perf_counter()
                _ = llm(**dict(reuse_kwargs, past_key_values=pw))
                _sync_device(device)
                t2 = time.perf_counter()
                reuse_tot_l.append((t2 - t1) * 1000.0)
                reuse_cmp_l.append((t2 - t1c) * 1000.0)

            m_full = sum(full_ms_l) / len(full_ms_l)
            m_rtot = sum(reuse_tot_l) / len(reuse_tot_l)
            m_rcmp = sum(reuse_cmp_l) / len(reuse_cmp_l)
            by_edit_pos[pos_label] = {
                "edited_page_idx": edited_page,
                "bank1_pages": b1_pages_a,
                "bank2_pages": b2_pages_a,
                "bank1_immutable_tokens": imm_len,
                "bank2_plus_query_tokens": tail_len,
                "total_tokens": total_len,
                "immutable_fraction": round(imm_len / max(1, total_len), 4),
                "bank1_embed_max_abs_diff": round(embed_diff, 8),
                "bank1_pkv_max_abs_diff": round(max_kv_diff, 8),
                "final_logits_max_abs_diff": round(logits_diff, 5),
                "full_reprefill_ms": round(m_full, 3),
                "reuse_with_clone_ms": round(m_rtot, 3),
                "reuse_compute_only_ms": round(m_rcmp, 3),
                "speedup_with_clone_x": round(m_full / max(1e-6, m_rtot), 3),
                "speedup_compute_only_x": round(m_full / max(1e-6, m_rcmp), 3),
            }

        p0 = by_edit_pos["first_p0"]
        out[f"P={num_p}"] = {
            "num_pages": num_p,
            "fixed_working_set_reuse": by_edit_pos,
            "bank1_immutable_tokens": p0["bank1_immutable_tokens"],
            "bank2_plus_query_tokens": p0["bank2_plus_query_tokens"],
            "total_tokens": p0["total_tokens"],
            "immutable_fraction": p0["immutable_fraction"],
            "bank1_embed_max_abs_diff": max(v["bank1_embed_max_abs_diff"] for v in by_edit_pos.values()),
            "bank1_pkv_max_abs_diff": max(v["bank1_pkv_max_abs_diff"] for v in by_edit_pos.values()),
            "final_logits_max_abs_diff": max(v["final_logits_max_abs_diff"] for v in by_edit_pos.values()),
            "full_reprefill_ms": p0["full_reprefill_ms"],
            "reuse_with_clone_ms": p0["reuse_with_clone_ms"],
            "reuse_compute_only_ms": p0["reuse_compute_only_ms"],
            "speedup_with_clone_x": p0["speedup_with_clone_x"],
            "speedup_compute_only_x": p0["speedup_compute_only_x"],
        }
        print(
            f"[G6 KV-REUSE P={num_p}] p0/p1/pEnd max_pkv_diff={out[f'P={num_p}']['bank1_pkv_max_abs_diff']:.1e} "
            f"Bank1={p0['bank1_immutable_tokens']}/{p0['total_tokens']} full={p0['full_reprefill_ms']:.2f}ms "
            f"reuse={p0['reuse_with_clone_ms']:.2f}ms ({p0['speedup_with_clone_x']:.2f}x)",
            flush=True,
        )

    out["post_rerouting_cache_hit_telemetry"] = measure_post_rerouting_cache_hit_telemetry(
        pathb, llm, tokenizer, val_pool, device, is_mock=is_mock
    )
    return out


@torch.no_grad()
def run_llama_ruler_16k_32k_and_profile(
    llm: nn.Module,
    pathb: PathBDiscreteMemory,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    lengths: tuple[int, ...] = (16384, 32768),
    episodes_per_task: int = 20,
    salience_route_weight: float = 1.5,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """RULER 16K & 32K (N=60/len) including zero-training `heuristic_span16_tfidf` + Unified k_deep Profile + True Two-Bank KV Reuse."""
    embed_fn = llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    pathb.eval()

    user_open_ids = torch.tensor(
        tokenizer.encode(LLAMA3_USER_OPEN, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)

    eff_lengths = (1024,) if is_mock else lengths
    eff_eps = 2 if is_mock else episodes_per_task
    tasks = ("s_niah", "mk_niah", "vt")
    systems = (
        "full_context_bf16",
        "bm25_rag_k1",
        "bm25_rag_k4",
        "bge_4span_rag_k1",
        "bge_4span_rag_k4",
        "heuristic_span16_tfidf",
        "standalone_discrete_span_bank",
    )

    by_length: dict[str, Any] = {}
    for ctx_len in eff_lengths:
        len_key = f"{ctx_len // 1024}K" if ctx_len >= 1024 else str(ctx_len)
        all_eps_len: list[dict[str, Any]] = []
        for t_name in tasks:
            all_eps_len.extend(
                build_ruler_episodes(
                    task_name=t_name, context_length=ctx_len, num_episodes=eff_eps,
                    tokenizer=tokenizer, val_pool=val_pool, page_size=pathb.cfg.page_size,
                    seed=seed, mk_distractors=48, vt_hops=3, is_mock=is_mock,
                )
            )

        sys_out: dict[str, Any] = {}
        em_by_sys: dict[str, list[float]] = {}
        bge_cache: list[tuple[torch.Tensor | None, float]] = []

        for sys_name in systems:
            rows: list[dict[str, Any]] = []
            rows_by_task: dict[str, list[dict[str, Any]]] = {t: [] for t in tasks}
            hit_oom = False

            for ep_idx, ep in enumerate(all_eps_len):
                p_ids_cpu: torch.Tensor = ep["pages"]
                page_texts: list[str] = ep["page_texts"]
                gold_set = set(ep["gold_pages"])
                route_q = str(ep["router_query"])
                p_prompt = torch.tensor(
                    tokenizer.encode(format_llama3_qa_prompt(ep["question"]), add_special_tokens=False),
                    dtype=torch.long, device=device,
                ).unsqueeze(0)

                try:
                    encode_ms = 0.0
                    router_hit: float | None = None
                    stored_bpp = 0.0

                    if sys_name == "full_context_bf16":
                        _sync_device(device)
                        t_q0 = time.perf_counter()
                        flat_ids = p_ids_cpu.reshape(1, -1).to(device)
                        pref = embed_fn(flat_ids).to(llm_dtype)
                        gen_txt = greedy_generate_llama3(llm, pref, p_prompt, tokenizer, max_new_tokens=24, user_open_ids=user_open_ids)
                        _sync_device(device)
                        q_ms = (time.perf_counter() - t_q0) * 1000.0
                        pref_toks = int(pref.shape[1])
                        stored_bpp = float(pathb.cfg.page_size * pathb.cfg.kv_bytes_per_token_fp16)
                        del flat_ids, pref
                        if device.type == "cuda" and ctx_len >= 16384:
                            torch.cuda.empty_cache()

                    elif sys_name.startswith("bm25_rag_k"):
                        k_val = int(sys_name.rsplit("_k", 1)[-1])
                        t_e0 = time.perf_counter()
                        bm25 = BM25PageRetriever(page_texts)
                        encode_ms = (time.perf_counter() - t_e0) * 1000.0
                        _sync_device(device)
                        t_q0 = time.perf_counter()
                        rr = bm25.route(route_q, k_deep=min(k_val, len(page_texts)), k_skim=0, device=device)
                        chosen = rr.deep_idx[0].tolist()
                        router_hit = 1.0 if gold_set.issubset(set(chosen)) else (len(gold_set & set(chosen)) / max(1, len(gold_set)))
                        pref = embed_fn(
                            _encode_unpadded_rag_ids("\n\n".join(page_texts[i].strip() for i in chosen) + "\n", tokenizer, vocab_size, device)
                        ).to(llm_dtype)
                        gen_txt = greedy_generate_llama3(llm, pref, p_prompt, tokenizer, max_new_tokens=24, user_open_ids=user_open_ids)
                        _sync_device(device)
                        q_ms = (time.perf_counter() - t_q0) * 1000.0
                        pref_toks = int(pref.shape[1])
                        stored_bpp = round(sum(len(t.encode("utf-8")) for t in page_texts) / max(1, len(page_texts)), 2)

                    elif sys_name.startswith("bge_4span_rag_k"):
                        k_val = int(sys_name.rsplit("_k", 1)[-1])
                        if ep_idx < len(bge_cache):
                            bge_bf16, encode_ms = bge_cache[ep_idx]
                        else:
                            _sync_device(device)
                            t_e0 = time.perf_counter()
                            bge_bf16 = (
                                pathb.bge_toc.encode_book_pages(
                                    p_ids_cpu.unsqueeze(0).to(device), None, tokenizer, num_spans=4, batch_size=128
                                ).to(torch.bfloat16)
                                if pathb.bge_toc is not None
                                else None
                            )
                            _sync_device(device)
                            encode_ms = (time.perf_counter() - t_e0) * 1000.0
                            bge_cache.append((bge_bf16, encode_ms))
                        _sync_device(device)
                        t_q0 = time.perf_counter()
                        if pathb.bge_toc is not None and bge_bf16 is not None:
                            sc = pathb.bge_toc.score_book(pathb.bge_toc.encode_texts([route_q], is_query=True), bge_bf16)[0]
                            chosen = torch.argsort(sc, descending=True)[: min(k_val, len(page_texts))].tolist()
                        else:
                            chosen = list(range(min(k_val, len(page_texts))))
                        router_hit = 1.0 if gold_set.issubset(set(chosen)) else (len(gold_set & set(chosen)) / max(1, len(gold_set)))
                        pref = embed_fn(
                            _encode_unpadded_rag_ids("\n\n".join(page_texts[i].strip() for i in chosen) + "\n", tokenizer, vocab_size, device)
                        ).to(llm_dtype)
                        gen_txt = greedy_generate_llama3(llm, pref, p_prompt, tokenizer, max_new_tokens=24, user_open_ids=user_open_ids)
                        _sync_device(device)
                        q_ms = (time.perf_counter() - t_q0) * 1000.0
                        pref_toks = int(pref.shape[1])
                        bge_bpp = (bge_bf16.numel() * bge_bf16.element_size()) / max(1, len(page_texts)) if bge_bf16 is not None else 0.0
                        stored_bpp = round(bge_bpp + sum(len(t.encode("utf-8")) for t in page_texts) / max(1, len(page_texts)), 2)

                    elif sys_name == "heuristic_span16_tfidf":
                        # Zero-training TF-IDF span-16 extraction + identical BGE+lexical router
                        p_t = p_ids_cpu.unsqueeze(0).to(device)
                        _sync_device(device)
                        t_e0 = time.perf_counter()
                        h_bank = heuristic_span16_tfidf_bank(
                            p_t, k_exact=pathb.k_exact, span_len=pathb.salience_span_len, special_token_ids=special_ids
                        )
                        bge_h = (
                            pathb.bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=128).to(torch.bfloat16)
                            if pathb.bge_toc is not None
                            else None
                        )
                        _sync_device(device)
                        encode_ms = (time.perf_counter() - t_e0) * 1000.0
                        bge_bpp = (bge_h.numel() * bge_h.element_size()) / max(1, p_t.shape[1]) if bge_h is not None else 0.0
                        stored_bpp = round((h_bank.storage_bytes() / max(1, p_t.shape[1])) + bge_bpp, 2)
                        rq_ids = torch.tensor(
                            _encode_text_or_mock(route_q, tokenizer, vocab_size), dtype=torch.long, device=device
                        ).unsqueeze(0)
                        _sync_device(device)
                        t_q0 = time.perf_counter()
                        r_out, pref, _ = pathb.route_and_assemble(
                            h_bank, bge_h, rq_ids, route_q, embed_fn,
                            device=device, k_deep=min(4, p_t.shape[1]), salience_route_weight=salience_route_weight,
                        )
                        chosen = r_out.deep_idx[0].tolist()
                        router_hit = 1.0 if gold_set.issubset(set(chosen)) else (len(gold_set & set(chosen)) / max(1, len(gold_set)))
                        gen_txt = greedy_generate_llama3(
                            llm, pref.to(llm_dtype), p_prompt, tokenizer, max_new_tokens=24, user_open_ids=user_open_ids
                        )
                        _sync_device(device)
                        q_ms = (time.perf_counter() - t_q0) * 1000.0
                        pref_toks = int(pref.shape[1])

                    else:  # standalone_discrete_span_bank
                        p_t = p_ids_cpu.unsqueeze(0).to(device)
                        enc_p = pathb.encode_book(
                            p_t, embed_fn, tokenizer, vocab_size, device=device,
                            amp_dtype=amp_dtype, autocast_enabled=autocast_enabled,
                            special_token_ids=special_ids, num_spans=4,
                        )
                        encode_ms = float(enc_p["encode_ms"])
                        stored_bpp = float(enc_p["stored_bytes_per_page_before_edit"])
                        rq_ids = torch.tensor(
                            _encode_text_or_mock(route_q, tokenizer, vocab_size), dtype=torch.long, device=device
                        ).unsqueeze(0)
                        _sync_device(device)
                        t_q0 = time.perf_counter()
                        r_out, pref, _ = pathb.route_and_assemble(
                            enc_p["bank"], enc_p["bge_keys_bf16"], rq_ids, route_q, embed_fn,
                            device=device, k_deep=min(4, p_t.shape[1]), salience_route_weight=salience_route_weight,
                        )
                        chosen = r_out.deep_idx[0].tolist()
                        router_hit = 1.0 if gold_set.issubset(set(chosen)) else (len(gold_set & set(chosen)) / max(1, len(gold_set)))
                        gen_txt = greedy_generate_llama3(
                            llm, pref.to(llm_dtype), p_prompt, tokenizer, max_new_tokens=24, user_open_ids=user_open_ids
                        )
                        _sync_device(device)
                        q_ms = (time.perf_counter() - t_q0) * 1000.0
                        pref_toks = int(pref.shape[1])

                    qm = compute_community_qa_metrics(gen_txt, ep["answer"])
                    rec = {
                        "episode_idx": ep_idx,
                        "task": ep["task"],
                        "gold": ep["answer"],
                        "pred": gen_txt,
                        "extracted_em": float(qm["extracted_em"]),
                        "token_f1": float(qm["token_f1"]),
                        "substring_match": float(qm["substring_match"]),
                        "router_hit_rate": router_hit,
                        "active_prefix_tokens": float(pref_toks),
                        "encode_ms": round(encode_ms, 3),
                        "first_query_ms": round(q_ms, 3),
                        "stored_bytes_per_page": stored_bpp,
                    }
                    rows.append(rec)
                    rows_by_task[ep["task"]].append(rec)
                except torch.cuda.OutOfMemoryError:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    hit_oom = True
                    break

            if hit_oom:
                sys_out[sys_name] = {"status": "oom"}
            else:
                m_keys = (
                    "extracted_em", "token_f1", "substring_match", "router_hit_rate",
                    "active_prefix_tokens", "encode_ms", "first_query_ms", "stored_bytes_per_page",
                )
                agg = _summarize_scalar_rows(rows, m_keys)
                agg["by_task"] = {t: _summarize_scalar_rows(rows_by_task[t], m_keys) for t in tasks}
                agg["episodes"] = rows
                sys_out[sys_name] = agg
                em_by_sys[sys_name] = [r["extracted_em"] for r in rows]
                print(
                    f"[G6 LLAMA-RULER {len_key} {sys_name}] EM={agg['extracted_em']:.3f} "
                    f"hit={agg.get('router_hit_rate', 0.0):.3f} tok={agg['active_prefix_tokens']:.0f} "
                    f"enc={agg['encode_ms']:.1f}ms q={agg['first_query_ms']:.1f}ms",
                    flush=True,
                )

        pairs_r = [
            ("standalone_discrete_span_bank", "heuristic_span16_tfidf"),
            ("standalone_discrete_span_bank", "bge_4span_rag_k1"),
            ("standalone_discrete_span_bank", "bge_4span_rag_k4"),
            ("standalone_discrete_span_bank", "bm25_rag_k1"),
            ("standalone_discrete_span_bank", "bm25_rag_k4"),
            ("standalone_discrete_span_bank", "full_context_bf16"),
        ]
        by_length[len_key] = {
            "context_length": ctx_len,
            "num_episodes": len(all_eps_len),
            "systems": sys_out,
            "pairwise_extracted_em": compute_pairwise_suite(em_by_sys, pairs_r, seed=seed),
        }

    # Unified k_deep in {2, 4, 8} Systems Profile
    eff_k_vals = (2, 4) if is_mock else (2, 4, 8)
    eff_decode = 8 if is_mock else 32
    rng = random.Random(seed + 19)
    filler_vocab = val_pool if val_pool else list(range(2, max(16, vocab_size - 2)))
    q_txt_p = "What is the special magic number for amber-node-12?"
    rq_p = torch.tensor(_encode_text_or_mock(q_txt_p, tokenizer, vocab_size), dtype=torch.long, device=device).unsqueeze(0)
    pp_p = torch.tensor(
        tokenizer.encode(format_llama3_qa_prompt(q_txt_p), add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)

    def _prof_prefill_decode(pref_e: torch.Tensor) -> tuple[float, float, float]:
        _reset_peak_vram(device)
        inp = torch.cat((embed_fn(user_open_ids).to(llm_dtype), pref_e.to(llm_dtype), embed_fn(pp_p).to(llm_dtype)), dim=1)
        attn = torch.ones((1, inp.shape[1]), device=device, dtype=torch.long)
        _sync_device(device)
        t0 = time.perf_counter()
        _ = llm(inputs_embeds=inp, attention_mask=attn, use_cache=True)
        _sync_device(device)
        pf_ms = (time.perf_counter() - t0) * 1000.0
        _sync_device(device)
        t1 = time.perf_counter()
        _ = greedy_generate_llama3(
            llm, pref_e.to(llm_dtype), pp_p, tokenizer, max_new_tokens=eff_decode,
            user_open_ids=user_open_ids, stop_fn=lambda _: False,
        )
        _sync_device(device)
        fd_ms = (time.perf_counter() - t1) * 1000.0
        return pf_ms, fd_ms, _peak_vram_mb(device)

    profile_by_len: dict[str, Any] = {}
    for ctx_len in eff_lengths:
        num_p = max(4, ctx_len // pathb.cfg.page_size)
        len_key = f"{ctx_len // 1024}K" if ctx_len >= 1024 else str(ctx_len)
        pages = [[filler_vocab[rng.randrange(len(filler_vocab))] for _ in range(pathb.cfg.page_size)] for _ in range(num_p)]
        p_t = torch.tensor(pages, dtype=torch.long, device=device).unsqueeze(0)
        page_texts = [tokenizer.decode(p, skip_special_tokens=True) for p in pages]

        _sync_device(device)
        tb0 = time.perf_counter()
        bge_k_raw = (
            pathb.bge_toc.encode_book_pages(None, page_texts, tokenizer, num_spans=4, batch_size=128).to(torch.bfloat16)
            if pathb.bge_toc is not None
            else None
        )
        _sync_device(device)
        bge_enc_ms = (time.perf_counter() - tb0) * 1000.0

        disc_enc = pathb.encode_book(
            p_t, embed_fn, tokenizer, vocab_size, device=device,
            amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids, num_spans=4,
        )
        sweep: dict[str, Any] = {}
        for kd in eff_k_vals:
            kd_e = min(kd, num_p)
            chosen_b = list(range(kd_e))
            if pathb.bge_toc is not None and bge_k_raw is not None:
                sc_b = pathb.bge_toc.score_book(pathb.bge_toc.encode_texts([q_txt_p], is_query=True), bge_k_raw)[0]
                chosen_b = torch.argsort(sc_b, descending=True)[:kd_e].tolist()
            pref_b = embed_fn(
                _encode_unpadded_rag_ids("\n\n".join(page_texts[i].strip() for i in chosen_b) + "\n", tokenizer, vocab_size, device)
            ).to(llm_dtype)
            pf_b, fd_b, vr_b = _prof_prefill_decode(pref_b)

            _, pref_d, rt_d = pathb.route_and_assemble(
                disc_enc["bank"], disc_enc["bge_keys_bf16"], rq_p, q_txt_p, embed_fn,
                device=device, k_deep=kd_e, salience_route_weight=salience_route_weight,
            )
            pf_d, fd_d, vr_d = _prof_prefill_decode(pref_d)
            sweep[f"k_deep={kd_e}"] = {
                "bge_4span_rag": {
                    "active_prefix_tokens": int(pref_b.shape[1]),
                    "encode_ms": round(bge_enc_ms, 2),
                    "prefill_ms": round(pf_b, 2),
                    "full_decode_ms": round(fd_b, 2),
                    "peak_vram_mb": vr_b,
                },
                "standalone_discrete_span_bank": {
                    "active_prefix_tokens": int(pref_d.shape[1]),
                    "encode_ms": round(disc_enc["encode_ms"], 2),
                    "route_and_assemble_ms": round(rt_d, 3),
                    "prefill_ms": round(pf_d, 2),
                    "full_decode_ms": round(fd_d, 2),
                    "peak_vram_mb": vr_d,
                    "stored_bytes_per_page_before_edit": disc_enc["stored_bytes_per_page_before_edit"],
                    "stored_bytes_per_page_after_edit": disc_enc["stored_bytes_per_page_after_edit"],
                    "stored_bytes_per_page": disc_enc["stored_bytes_per_page_before_edit"],
                },
            }
        profile_by_len[len_key] = {
            "num_pages": num_p,
            "full_kv_bf16_bytes_per_page": int(pathb.cfg.page_size * pathb.cfg.kv_bytes_per_token_fp16),
            "sweep": sweep,
        }

    kv_reuse = benchmark_two_bank_discrete_kv_reuse(
        llm, pathb, tokenizer, val_pool, device, page_counts=(16, 64, 128), is_mock=is_mock
    )
    return {
        "ruler_by_length": by_length,
        "unified_systems_profile": {
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "bge_model_params": pathb.bge_model_param_count(),
            "trainable_param_breakdown": pathb.trainable_param_breakdown(llm),
            "storage_bytes": pathb.storage_breakdown_bytes_per_page(num_spans=4),
            "by_length": profile_by_len,
        },
        "prefix_kv_cache_reuse": kv_reuse,
    }


# ---------------------------------------------------------------------------
# Benchmark 5 (Causal Control 4e): Mini Negative Probe Z_book on Llama-3.1-8B
# ---------------------------------------------------------------------------


def run_llama_zbook_probe(
    llm: nn.Module,
    pathb: PathBDiscreteMemory,
    tokenizer,
    train_pool: list[int],
    val_pool: list[int],
    device: torch.device,
    *,
    zbook_train_steps: int = 120,
    num_verbatim: int = 24,
    num_implicit: int = 24,
    ppl_n: int = 16,
    salience_route_weight: float = 1.5,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Cross-family Layer-0 Z_book negative probe on frozen Llama-3.1-8B-Instruct BF16 (r=0).

    1. Trains a compact Layer-0 `LSLCSystem` (`LayeredLatentEncoder` + `SpatialLatentProjector` + `DeltaUpdater`)
       on 100% eval-disjoint prose + state-revision data while verifying `max_frozen_param_diff == 0.0`.
    2. Runs the 5-Cell Masked-Old Edit Ablation (N=48: 24 verbatim + 24 implicit, iso-token) on Llama-3.1-8B.
    3. Runs WikiText-2 Continuation PPL (N=16 at 4K) comparing `full_kv_bf16`, `pathb_discrete_192tok` (0 Z_book),
       and `llama_lslc_zbook_full` (with Layer-0 Z_book soft tokens).
    """
    t0 = time.perf_counter()
    embed_fn = llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    llm_before = _frozen_llm_param_checksums(llm)

    zbook_sys = LSLCSystem(pathb.cfg).to(device)
    # Seed salience_gate from trained Path B so the ONLY difference tested is Layer-0 continuous Z_book + Delta-Z!
    zbook_sys.salience_gate.load_state_dict(pathb.salience_gate.state_dict(), strict=True)

    # 1. Brief Layer-0 Z_book + DeltaUpdater training on 100% eval-disjoint data
    eff_steps = 8 if is_mock else int(zbook_train_steps)
    n_tr_g2 = 6 if is_mock else 24
    tr_g2 = list(
        AdversarialLivingBookDataset(
            num_episodes=n_tr_g2, num_pages=6, page_size=pathb.cfg.page_size, num_edits=2,
            mode="budget", split="train", tokenizer=tokenizer, vocab_size=vocab_size,
            seed=seed + 3301, background_token_pool=train_pool, style="mixed_natural",
        )
    )
    tr_prose = list(ChunkedTextPageDataset(train_pool, num_pages=4, page_size=pathb.cfg.page_size, turn_len=32))[
        : (6 if is_mock else 32)
    ]

    opt = torch.optim.AdamW(zbook_sys.parameters(), lr=3e-4, weight_decay=0.01)
    rng = random.Random(seed + 3307)
    zbook_sys.train()
    tr_losses: list[float] = []
    layer_idx = pathb.cfg.contextual_edit_layer if pathb.cfg.contextual_edit_layer is not None else max(1, pathb.cfg.num_llm_layers // 2)

    for step_i in range(eff_steps):
        opt.zero_grad(set_to_none=True)
        if step_i % 2 == 0:
            ep = tr_prose[rng.randrange(len(tr_prose))]
            p_t = ep.context_pages.unsqueeze(0).to(device)
            q_ids = ep.query_ids.unsqueeze(0).to(device)
            a_ids = ep.answer_ids.unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                zb, bk, tc = zbook_sys.encode_book(
                    p_t, embed_tokens_fn=embed_fn, vocab_size=vocab_size,
                    special_token_ids=special_ids, salience_span_len=pathb.salience_span_len,
                )
                _, asm = zbook_sys.route_and_assemble(
                    zb, tc, embed_fn(q_ids).mean(dim=1), bk, embed_fn,
                    k_deep=min(2, p_t.shape[1]), k_skim=1, mode="full", training_ste=True,
                )
                qa_ids = torch.cat((q_ids, a_ids), dim=1)
                inp = torch.cat((asm.prefix_embeds.to(llm_dtype), embed_fn(qa_ids).to(llm_dtype)), dim=1)
                attn = torch.ones((1, inp.shape[1]), device=device, dtype=torch.long)
                logits = llm(inputs_embeds=inp, attention_mask=attn).logits
                p_len = int(asm.prefix_embeds.shape[1])
                ans_start = p_len + int(q_ids.shape[1]) - 1
                ans_logits = logits[:, ans_start : ans_start + int(a_ids.shape[1]), :].float()
                loss = F.cross_entropy(ans_logits.reshape(-1, ans_logits.shape[-1]), a_ids.reshape(-1))
        else:
            ep = tr_g2[rng.randrange(len(tr_g2))]
            p_t = ep.context_pages.unsqueeze(0).to(device)
            q_ids = ep.query_ids.unsqueeze(0).to(device)
            a_ids = ep.answer_ids.unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                zb, bk, tc = zbook_sys.encode_book(
                    p_t, embed_tokens_fn=embed_fn, vocab_size=vocab_size,
                    special_token_ids=special_ids, salience_span_len=pathb.salience_span_len,
                )
                gate_losses = []
                for (p_i, e_ids_cpu), is_acc in zip(ep.edit_pages, ep.metadata["edit_accepted_flags"]):
                    e_t = e_ids_cpu.unsqueeze(0).to(device)
                    with torch.no_grad():
                        ov = extract_contextual_edit_embeds(llm, e_t, layer_idx=layer_idx)
                    zb, g_p, bk = zbook_sys.apply_page_delta(
                        zb, torch.tensor([int(p_i)], device=device), e_t,
                        embed_tokens_fn=embed_fn, exact_bank=bk, vocab_size=vocab_size,
                        edit_embeds_override=ov, in_place=False, bank_write_threshold=0.45,
                    )
                    with torch.autocast(device_type=device.type, enabled=False):
                        tgt_g = torch.full_like(g_p[:, -1, :].float(), 1.0 if is_acc else 0.0)
                        gate_losses.append(
                            F.binary_cross_entropy(g_p[:, -1, :].float().clamp(1e-5, 1.0 - 1e-5), tgt_g)
                        )
                tc_after = zbook_sys.projector.project_toc(zb)
                _, asm = zbook_sys.route_and_assemble(
                    zb, tc_after, embed_fn(q_ids).mean(dim=1), bk, embed_fn,
                    k_deep=min(2, p_t.shape[1]), k_skim=1, mode="full", training_ste=True,
                )
                qa_ids = torch.cat((q_ids, a_ids), dim=1)
                inp = torch.cat((asm.prefix_embeds.to(llm_dtype), embed_fn(qa_ids).to(llm_dtype)), dim=1)
                attn = torch.ones((1, inp.shape[1]), device=device, dtype=torch.long)
                logits = llm(inputs_embeds=inp, attention_mask=attn).logits
                p_len = int(asm.prefix_embeds.shape[1])
                ans_start = p_len + int(q_ids.shape[1]) - 1
                ans_logits = logits[:, ans_start : ans_start + int(a_ids.shape[1]), :].float()
                lm_loss = F.cross_entropy(ans_logits.reshape(-1, ans_logits.shape[-1]), a_ids.reshape(-1))
                loss = lm_loss + torch.stack(gate_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(zbook_sys.parameters(), 1.0)
        opt.step()
        tr_losses.append(float(loss.detach().item()))
    zbook_sys.eval()

    max_frozen_diff = _verify_frozen_llm_unchanged(llm, llm_before)
    assert max_frozen_diff == 0.0, f"Frozen Llama-3.1-8B mutated during Z_book probe: {max_frozen_diff}"

    # 2. 5-Cell Iso-Token Masked-Old Edit Ablation on Llama-3.1-8B (N=48: 24 verbatim + 24 implicit)
    suites = build_2x2_edit_episodes(
        tokenizer, val_pool,
        num_verbatim=(4 if is_mock else num_verbatim),
        num_implicit=(4 if is_mock else num_implicit),
        num_pages=8, page_size=pathb.cfg.page_size, seed=seed,
    )
    user_open_ids = torch.tensor(
        tokenizer.encode(LLAMA3_USER_OPEN, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)
    arms = (
        ("deltaZ_ON_bank_ON", True, "upd"),
        ("deltaZ_OFF_bank_ON", False, "upd"),
        ("deltaZ_ON_bank_masked_old", True, "masked_old"),
        ("deltaZ_ON_bank_OFF", True, "orig"),
        ("deltaZ_OFF_bank_OFF", False, "orig"),
    )
    edit_results: dict[str, Any] = {}
    sem_overall: dict[str, list[float]] = {a[0]: [] for a in arms}
    sem_verb: dict[str, list[float]] = {a[0]: [] for a in arms}
    sem_impl: dict[str, list[float]] = {a[0]: [] for a in arms}

    with torch.no_grad():
        for arm_name, use_dz, bank_mode in arms:
            suite_out: dict[str, Any] = {}
            all_rows: list[dict[str, Any]] = []
            for s_name, eps in suites.items():
                rows: list[dict[str, Any]] = []
                for ep_idx, ep in enumerate(eps):
                    p_t = ep["pages"].unsqueeze(0).to(device)
                    tgt_p = int(ep["target_page"])
                    z_book, bank, toc0 = _encode_book_chunked(
                        zbook_sys, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled,
                        chunk_size=16, special_token_ids=special_ids, salience_span_len=pathb.salience_span_len,
                    )
                    z_orig = z_book.clone()
                    bank_orig = SparseExactBank(
                        token_ids=bank.token_ids.clone(),
                        page_ids=None if bank.page_ids is None else bank.page_ids.clone(),
                        h_coords=None if bank.h_coords is None else bank.h_coords.clone(),
                        w_coords=None if bank.w_coords is None else bank.w_coords.clone(),
                    )
                    edit_ids = torch.tensor(
                        _encode_text_or_mock(ep["edit_text"], tokenizer, vocab_size), dtype=torch.long, device=device
                    ).unsqueeze(0)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        ov = extract_contextual_edit_embeds(llm, edit_ids, layer_idx=layer_idx)
                        z_upd, g_p, bank_upd = zbook_sys.apply_page_delta(
                            z_book.clone(), torch.tensor([tgt_p], dtype=torch.long, device=device), edit_ids,
                            embed_tokens_fn=embed_fn, exact_bank=bank, vocab_size=vocab_size,
                            edit_embeds_override=ov, in_place=False, bank_write_threshold=0.45,
                        )
                    num_masked_old = 0
                    if bank_mode == "upd":
                        bank_use = bank_upd
                    elif bank_mode == "masked_old":
                        bank_use, num_masked_old = mask_old_value_in_exact_bank(
                            bank_orig, tgt_p, str(ep.get("old_answer", "")), tokenizer, vocab_size
                        )
                    else:
                        bank_use = bank_orig

                    z_use = z_upd if use_dz else z_orig
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        toc_after = zbook_sys.projector.project_toc(z_use)
                        rq_ids = torch.tensor(
                            _encode_text_or_mock(ep["router_query"], tokenizer, vocab_size), dtype=torch.long, device=device
                        ).unsqueeze(0)
                        ext_s = (
                            pathb.bge_toc.score_book(
                                pathb.bge_toc.encode_texts([ep["router_query"]], is_query=True),
                                pathb.bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=64).to(torch.bfloat16),
                            )
                            if pathb.bge_toc is not None
                            else None
                        )
                        r_out, _ = zbook_sys.route_and_assemble(
                            z_book=z_use, toc_embeds=toc_after, query_embed=embed_fn(rq_ids).mean(dim=1),
                            exact_bank=bank_orig, embed_tokens_fn=embed_fn, k_deep=4, k_skim=2,
                            mode="toc_only", training_ste=False, query_input_ids=rq_ids,
                            salience_route_weight=salience_route_weight, external_dense_logits=ext_s,
                        )
                        _, asm = zbook_sys.route_and_assemble(
                            z_book=z_use, toc_embeds=toc_after, query_embed=embed_fn(rq_ids).mean(dim=1),
                            exact_bank=bank_use, embed_tokens_fn=embed_fn, k_deep=4, k_skim=2,
                            mode="full", training_ste=False, router_out_override=r_out, exact_bank_attenuation="none",
                        )
                    p_prompt = torch.tensor(
                        tokenizer.encode(format_llama3_qa_prompt(ep["question"]), add_special_tokens=False),
                        dtype=torch.long, device=device,
                    ).unsqueeze(0)
                    gen_txt = greedy_generate_llama3(
                        llm, asm.prefix_embeds.to(llm_dtype), p_prompt, tokenizer,
                        max_new_tokens=20, user_open_ids=user_open_ids,
                    )
                    qm = compute_community_qa_metrics(gen_txt, ep["answer"])
                    norm_g, norm_p = _normalize_qa_text(ep["answer"]), _normalize_qa_text(gen_txt)
                    if s_name == "implicit" and norm_g in ("yes", "no"):
                        fw = norm_p.split()[:4]
                        sem_acc = 1.0 if (norm_g in fw and (("no" if norm_g == "yes" else "yes") not in fw)) else 0.0
                    else:
                        sem_acc = max(float(qm["extracted_em"]), float(qm["substring_match"]))
                    rec = {
                        "episode_idx": ep_idx,
                        "suite": s_name,
                        "gold": ep["answer"],
                        "old_answer": ep.get("old_answer", ""),
                        "pred": gen_txt,
                        "em": float(qm["extracted_em"]),
                        "semantic_acc": float(sem_acc),
                        "num_masked_old_tokens": float(num_masked_old),
                        "active_prefix_tokens": float(asm.prefix_embeds.shape[1]),
                    }
                    rows.append(rec)
                    all_rows.append(rec)
                    sem_overall[arm_name].append(float(sem_acc))
                    (sem_verb if s_name == "verbatim" else sem_impl)[arm_name].append(float(sem_acc))
                m_k = ("em", "semantic_acc", "num_masked_old_tokens", "active_prefix_tokens")
                agg_s = _summarize_scalar_rows(rows, m_k)
                agg_s["episodes"] = rows
                suite_out[s_name] = agg_s

            ov_arm = _summarize_scalar_rows(all_rows, ("em", "semantic_acc", "num_masked_old_tokens", "active_prefix_tokens"))
            edit_results[arm_name] = {"continuous_delta_z": use_dz, "bank_mode": bank_mode, "overall": ov_arm, "by_suite": suite_out}
            print(
                f"[G6 LLAMA-ZBOOK-5CELL {arm_name}] verb={suite_out['verbatim']['semantic_acc']:.3f} "
                f"impl={suite_out['implicit']['semantic_acc']:.3f} tok={ov_arm['active_prefix_tokens']:.0f}",
                flush=True,
            )

        tok_counts = {a[0]: edit_results[a[0]]["overall"]["active_prefix_tokens"] for a in arms}
        assert len(set(tok_counts.values())) == 1, f"Iso-token violated in Llama Z_book probe: {tok_counts}"

        # 3. WikiText-2 Continuation PPL (N=16 at 4K) comparing full_kv_bf16 vs pathb_discrete_192tok vs llama_lslc_zbook_full
        eff_ctx = 1024 if is_mock else 4096
        num_p = max(4, eff_ctx // pathb.cfg.page_size)
        ppl_eps = list(ChunkedTextPageDataset(val_pool, num_pages=num_p, page_size=pathb.cfg.page_size, turn_len=32))[
            : (3 if is_mock else ppl_n)
        ]
        ppl_summary: dict[str, Any] = {}
        for p_arm in ("full_kv_bf16", "pathb_discrete_192tok", "llama_lslc_zbook_full"):
            nlls: list[float] = []
            for ep in ppl_eps:
                p_t = ep.context_pages.unsqueeze(0).to(device)
                q_ids = ep.query_ids.unsqueeze(0).to(device)
                a_ids = ep.answer_ids.unsqueeze(0).to(device)
                if p_arm == "full_kv_bf16":
                    ctx_e = embed_fn(p_t.reshape(1, -1)).to(llm_dtype)
                    pre = llm(inputs_embeds=ctx_e, attention_mask=torch.ones((1, ctx_e.shape[1]), device=device, dtype=torch.long), use_cache=True)
                    nll, _ = compute_nll_and_tok_acc_from_kv(llm, pre.past_key_values, int(ctx_e.shape[1]), q_ids, a_ids, orig_seq_len=int(ctx_e.shape[1]))
                elif p_arm == "pathb_discrete_192tok":
                    q_txt_p = tokenizer.decode(q_ids[0].tolist(), skip_special_tokens=True)
                    enc_p = pathb.encode_book(
                        p_t, embed_fn, tokenizer, vocab_size, device=device,
                        amp_dtype=amp_dtype, autocast_enabled=autocast_enabled, special_token_ids=special_ids,
                    )
                    _, pref, _ = pathb.route_and_assemble(
                        enc_p["bank"], enc_p["bge_keys_bf16"], q_ids, q_txt_p, embed_fn,
                        device=device, k_deep=min(4, num_p), salience_route_weight=salience_route_weight,
                    )
                    pref = pref.to(llm_dtype)
                    pre = llm(inputs_embeds=pref, attention_mask=torch.ones((1, pref.shape[1]), device=device, dtype=torch.long), use_cache=True)
                    nll, _ = compute_nll_and_tok_acc_from_kv(llm, pre.past_key_values, int(pref.shape[1]), q_ids, a_ids)
                else:
                    q_txt_p = tokenizer.decode(q_ids[0].tolist(), skip_special_tokens=True)
                    zb, bk, tc = _encode_book_chunked(
                        zbook_sys, p_t, embed_fn, vocab_size, device, amp_dtype, autocast_enabled,
                        chunk_size=32, special_token_ids=special_ids, salience_span_len=pathb.salience_span_len,
                    )
                    ext_p = (
                        pathb.bge_toc.score_book(
                            pathb.bge_toc.encode_texts([q_txt_p], is_query=True),
                            pathb.bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=64).to(torch.bfloat16),
                        )
                        if pathb.bge_toc is not None
                        else None
                    )
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        _, asm = zbook_sys.route_and_assemble(
                            zb, tc, embed_fn(q_ids).mean(dim=1), bk, embed_fn,
                            k_deep=min(4, num_p), k_skim=min(2, max(0, num_p - 4)), mode="full",
                            hierarchical_chapter_size=16, query_input_ids=q_ids,
                            salience_route_weight=salience_route_weight, exact_bank_attenuation="none",
                            external_dense_logits=ext_p,
                        )
                    pref = asm.prefix_embeds.to(llm_dtype)
                    pre = llm(inputs_embeds=pref, attention_mask=torch.ones((1, pref.shape[1]), device=device, dtype=torch.long), use_cache=True)
                    nll, _ = compute_nll_and_tok_acc_from_kv(llm, pre.past_key_values, int(pref.shape[1]), q_ids, a_ids)
                nlls.append(nll)
            ppl_val = round(math.exp(min(20.0, sum(nlls) / max(1, len(nlls)))), 4)
            ppl_summary[p_arm] = {"wikitext_ppl": ppl_val, "mean_nll": round(sum(nlls) / max(1, len(nlls)), 4), "num_episodes": len(nlls)}
            print(f"[G6 LLAMA-ZBOOK-PPL {p_arm}] PPL={ppl_val:.2f}", flush=True)

    zbook_params = zbook_sys.trainable_parameter_count()
    del zbook_sys
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pairs_e = [
        ("deltaZ_ON_bank_masked_old", "deltaZ_ON_bank_OFF"),
        ("deltaZ_ON_bank_masked_old", "deltaZ_OFF_bank_OFF"),
        ("deltaZ_ON_bank_ON", "deltaZ_OFF_bank_ON"),
        ("deltaZ_ON_bank_ON", "deltaZ_ON_bank_masked_old"),
    ]
    return {
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "zbook_adapter_trainable_params": zbook_params,
        "zbook_train_steps": eff_steps,
        "zbook_train_initial_loss": round(tr_losses[0], 4),
        "zbook_train_final_loss": round(tr_losses[-1], 4),
        "max_frozen_param_diff": max_frozen_diff,
        "edit_ablation_5cell": {
            "iso_token_verified": True,
            "arms": edit_results,
            "pairwise_overall_semantic_acc": compute_pairwise_suite(sem_overall, pairs_e, seed=seed),
            "pairwise_verbatim_semantic_acc": compute_pairwise_suite(sem_verb, pairs_e, seed=seed),
            "pairwise_implicit_semantic_acc": compute_pairwise_suite(sem_impl, pairs_e, seed=seed),
        },
        "wikitext_ppl_comparison_4k": ppl_summary,
    }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gerbang 6: Cross-Family Path (B) Replication on Llama-3.1-8B-Instruct BF16.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--mock-llm", action="store_true", help="Run 0-CU CPU smoke test with TinyMockCausalLM.")
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help=(
            "Comma-separated: train_pathb,llama_gate2_n200,llama_coding_n24,"
            "llama_ood_kv_4k,llama_ruler_16k_32k_and_profile,llama_zbook_probe or 'all'."
        ),
    )
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--save-checkpoint", type=str, default="/content/lslc_results/lslc_pathb_llama31_8b.safetensors")
    parser.add_argument("--bge-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--k-exact", type=int, default=48)
    parser.add_argument("--salience-span-len", type=int, default=16)
    parser.add_argument("--salience-route-weight", type=float, default=1.5)
    parser.add_argument("--salience-steps", type=int, default=250)
    parser.add_argument("--edit-gate-steps", type=int, default=200)
    parser.add_argument("--gate2-n", type=int, default=200)
    parser.add_argument("--ood-kv-n", type=int, default=48)
    parser.add_argument("--ruler-lengths", type=str, default="16384,32768")
    parser.add_argument("--ruler-episodes-per-task", type=int, default=20)
    parser.add_argument("--zbook-train-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-json", type=str, default="/content/lslc_results/v3_gerbang6_llama8b_pathb.json")
    args, _ = parser.parse_known_args(argv)
    return args


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    raw_tasks = {t.strip().lower() for t in args.tasks.split(",") if t.strip()}
    all_set = {
        "train_pathb",
        "llama_gate2_n200",
        "llama_coding_n24",
        "llama_ood_kv_4k",
        "llama_ruler_16k_32k_and_profile",
        "llama_zbook_probe",
    }
    run_tasks = all_set if "all" in raw_tasks else raw_tasks
    ruler_lengths = tuple(int(s.strip()) for s in args.ruler_lengths.split(",") if s.strip())

    if args.mock_llm:
        vocab_size = 2048
        cfg = LSLCConfig.tiny_mock(
            model_name="mock-llama-3.1-8b",
            use_contextual_edit_encoding=True,
            contextual_edit_layer=1,
            use_contextual_query_encoding=False,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=min(args.k_exact, 48),
            salience_span_len=args.salience_span_len,
        )
        llm = TinyMockCausalLM(vocab_size=vocab_size, d_llm=cfg.d_llm, num_layers=cfg.num_llm_layers)
        for p in llm.parameters():
            p.requires_grad_(False)
        llm.eval()
        tokenizer = _MockCodeTokenizer(vocab_size=vocab_size)
        device = torch.device("cpu")
        resolved_model_id = "mock-llama-3.1-8b"
        load_attempts = [{"model_id": resolved_model_id, "status": "mock"}]
        train_pool = [2 + (i % (vocab_size - 2)) for i in range(4096)]
        val_pool = [2 + ((i * 7) % (vocab_size - 2)) for i in range(4096)]
        data_source = "mock-synthetic-pool"
    else:
        cfg = LSLCConfig.from_model_name(
            args.model,
            use_contextual_edit_encoding=True,
            use_contextual_query_encoding=False,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=args.k_exact,
            salience_span_len=args.salience_span_len,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        llm, tokenizer, resolved_model_id, load_attempts = load_frozen_llama8b_with_fallback(
            cfg, requested_model=args.model, device=device
        )
        train_pool, val_pool, data_source = _load_wikitext_pools(tokenizer)

    bge_toc = _PlugInBiEncoderTOC(args.bge_model, device=device, is_mock=args.mock_llm)
    pathb = PathBDiscreteMemory(
        cfg, bge_toc=bge_toc, bank_write_threshold=0.45, edit_gate_hidden_dim=64,
        span_aligned_overwrite=True, k_exact=cfg.k_exact, salience_span_len=cfg.salience_span_len,
    ).to(device)
    init_ckpt_info = load_pathb_checkpoint(pathb, args.init_checkpoint) if args.init_checkpoint else None
    git_prov = get_git_provenance(REPO_ROOT)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": "3.1.0-gerbang6-llama8b-pathb-hardened",
        "git_sha": git_prov["git_sha"],
        "git_dirty": git_prov["git_dirty"],
        "checkpoint_sha256": None if init_ckpt_info is None else init_ckpt_info.get("sha256"),
        "trainable_param_breakdown": pathb.trainable_param_breakdown(llm),
        "storage_bytes": pathb.storage_breakdown_bytes_per_page(num_spans=4),
        "requested_model": args.model,
        "resolved_model_id": resolved_model_id,
        "model_load_attempts": load_attempts,
        "mock_llm": args.mock_llm,
        "device": str(device),
        "data_source": data_source,
        "init_checkpoint": init_ckpt_info,
        "saved_checkpoint": None,
        "config": {
            "d_llm": cfg.d_llm,
            "num_llm_layers": cfg.num_llm_layers,
            "num_kv_heads": cfg.num_kv_heads,
            "llm_head_dim": cfg.llm_head_dim,
            "contextual_edit_layer": cfg.contextual_edit_layer,
            "page_size": cfg.page_size,
            "k_exact": cfg.k_exact,
            "salience_span_len": cfg.salience_span_len,
            "salience_route_weight": args.salience_route_weight,
            "edit_gate_hidden_dim": 64,
            "span_aligned_overwrite": True,
            "true_two_bank_layout": True,
            "bge_model": bge_toc.model_name,
            "has_layered_latent_encoder": False,
            "has_z_book": False,
            "has_hw_coord_embed": False,
            "prompt_format": "llama-3.1-instruct",
        },
        "train_pathb": None,
        "llama_gate2_n200": None,
        "llama_coding_n24": None,
        "llama_ood_kv_4k": None,
        "llama_ruler_16k_32k_and_profile": None,
        "llama_zbook_probe": None,
    }

    if "train_pathb" in run_tasks:
        payload["train_pathb"] = train_pathb_llama8b(
            llm, pathb, tokenizer, train_pool, device,
            salience_steps=args.salience_steps, edit_gate_steps=args.edit_gate_steps,
            seed=args.seed, is_mock=args.mock_llm,
        )
        if args.save_checkpoint:
            saved_info = save_pathb_checkpoint(pathb, args.save_checkpoint)
            payload["saved_checkpoint"] = saved_info
            payload["checkpoint_sha256"] = saved_info["sha256"]
        _atomic_write_json(out_path, payload)

    if "llama_gate2_n200" in run_tasks:
        payload["llama_gate2_n200"] = run_llama_gate2_n200(
            llm, pathb, tokenizer, val_pool, device,
            eval_n=args.gate2_n, salience_route_weight=args.salience_route_weight,
            seed=args.seed, is_mock=args.mock_llm,
        )
        _atomic_write_json(out_path, payload)

    if "llama_coding_n24" in run_tasks:
        payload["llama_coding_n24"] = run_llama_coding_n24(
            llm, pathb, tokenizer, device,
            salience_route_weight=args.salience_route_weight, is_mock=args.mock_llm,
        )
        _atomic_write_json(out_path, payload)

    if "llama_ood_kv_4k" in run_tasks:
        payload["llama_ood_kv_4k"] = run_llama_ood_kv_4k(
            llm, pathb, tokenizer, val_pool, device,
            context_length=4096, eval_n=args.ood_kv_n,
            salience_route_weight=args.salience_route_weight, seed=args.seed, is_mock=args.mock_llm,
        )
        _atomic_write_json(out_path, payload)

    if "llama_ruler_16k_32k_and_profile" in run_tasks:
        payload["llama_ruler_16k_32k_and_profile"] = run_llama_ruler_16k_32k_and_profile(
            llm, pathb, tokenizer, val_pool, device,
            lengths=ruler_lengths, episodes_per_task=args.ruler_episodes_per_task,
            salience_route_weight=args.salience_route_weight, seed=args.seed, is_mock=args.mock_llm,
        )
        _atomic_write_json(out_path, payload)

    if "llama_zbook_probe" in run_tasks:
        payload["llama_zbook_probe"] = run_llama_zbook_probe(
            llm, pathb, tokenizer, train_pool, val_pool, device,
            zbook_train_steps=args.zbook_train_steps,
            salience_route_weight=args.salience_route_weight, seed=args.seed, is_mock=args.mock_llm,
        )
        _atomic_write_json(out_path, payload)

    print(f"[DONE] Wrote Gerbang 6 Llama-3.1-8B Path B results -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
