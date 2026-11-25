#!/usr/bin/env python3
"""Gerbang 3 Evaluation: CrossFileAPIMigrationBenchmark (LSLC v3 vs BM25 RAG vs /compact).

Fair Hybrid Working Memory evaluation on 38-64 file (8K-16K token) Python repositories:
  - Target caller file being edited is provided raw & lossless in working memory.
  - Remaining 37-63 repo files + session review history come ONLY from each system's memory:
      1. oracle_raw_context   (sanity upper bound: updated core.py + approved session decision)
      2. bm25_rag_k2          (iso-token k=2 Okapi BM25 over updated repo + append-errata session log)
      3. bm25_rag_k8          (generous k=8 Okapi BM25 over updated repo + append-errata session log)
      4. llm_auto_compact     (Qwen3-1.7B self-compaction summary <=256 tok + recent turn <=512 tok total)
      5. lslc_v3_hybrid_k*    (Frozen Mode A Book + modular core.py re-encode + DeltaUpdater + GPU Hybrid Router)
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from colab_run_m2 import _load_wikitext_pools  # noqa: E402
from colab_run_rebuttal_v2 import BM25PageRetriever  # noqa: E402
from colab_run_v3_phase1 import (  # noqa: E402
    _build_stage1_episode_view,
    _save_lslc_checkpoint,
    trace_identifier_pipeline,
)
from lslc import (  # noqa: E402
    AdversarialLivingBookDataset,
    CodingAgentHarness,
    CrossFileAPIMigrationBenchmark,
    FrozenQwenLSLCWrapper,
    LSLCConfig,
    MigrationEvalResult,
    MigrationTask,
    RouterOutput,
    SyntheticExactSemanticNIAHDataset,
    TinyMockCausalLM,
    extract_contextual_edit_embeds,
    load_frozen_qwen3,
    select_amp_dtype,
)


class _MockCodeTokenizer:
    """Deterministic lightweight tokenizer for --mock-llm Gerbang 3 verification."""

    def __init__(self, vocab_size: int = 2048) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = None
        self._tok_to_id: dict[str, int] = {"<pad>": 0, "<eos>": 1}
        self._id_to_tok: dict[int, str] = {0: "<pad>", 1: "<eos>"}
        self._next_id = 2

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        tokens = re.findall(r"[A-Za-z0-9_.\-$]+|[^\w\s]", text)
        ids: list[int] = []
        for tok in tokens:
            if tok not in self._tok_to_id:
                tid = 2 + ((self._next_id - 2) % (self.vocab_size - 2))
                self._tok_to_id[tok] = tid
                self._id_to_tok[tid] = tok
                self._next_id += 1
            ids.append(self._tok_to_id[tok])
        return ids or [2]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        words = []
        for tid in token_ids:
            tid_i = int(tid)
            if skip_special_tokens and tid_i in (self.pad_token_id, self.eos_token_id):
                continue
            words.append(self._id_to_tok.get(tid_i, f"t{tid_i}"))
        return " ".join(words)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_lslc_checkpoint(lslc_mod: torch.nn.Module, ckpt_path: str) -> dict[str, object]:
    p = Path(ckpt_path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(p), device="cpu")
    else:
        sd = torch.load(str(p), map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "lslc" in sd:
            sd = sd["lslc"]
    incompat = lslc_mod.load_state_dict(sd, strict=False)
    print(
        f"[CKPT] Loaded {p.name}: missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}",
        flush=True,
    )
    return {
        "path": str(p),
        "missing_keys": list(incompat.missing_keys),
        "unexpected_keys": list(incompat.unexpected_keys),
    }


def _pad_or_chunk_tokens(tokens: list[int], page_size: int, pad_token_id: int) -> list[list[int]]:
    if not tokens:
        return [[pad_token_id] * page_size]
    chunks: list[list[int]] = []
    for start in range(0, len(tokens), page_size):
        seg = tokens[start : start + page_size]
        if len(seg) < page_size:
            seg = seg + [pad_token_id] * (page_size - len(seg))
        chunks.append(seg)
    return chunks


def _build_repo_pages(
    task: MigrationTask,
    tokenizer,
    page_size: int,
) -> dict[str, object]:
    """Pack initial repo files + session notes into fixed page_size token blocks."""
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    page_ids_list: list[list[int]] = []
    page_texts: list[str] = []
    page_paths: list[str] = []
    total_content_tokens = 0

    ordered_paths = sorted(task.initial_files.keys()) + [task.session_notes_path]
    for path in ordered_paths:
        content = (
            task.session_notes_initial
            if path == task.session_notes_path
            else task.initial_files[path]
        )
        formatted = f"# FILE: {path}\n{content}\n"
        toks = tokenizer.encode(formatted, add_special_tokens=False)
        total_content_tokens += len(toks)
        for chunk in _pad_or_chunk_tokens(toks, page_size, pad_id):
            page_ids_list.append(chunk)
            page_texts.append(formatted if len(toks) <= page_size else tokenizer.decode(chunk, skip_special_tokens=True))
            page_paths.append(path)

    page_tensor = torch.tensor(page_ids_list, dtype=torch.long).unsqueeze(0)  # [1, P, Tp]
    core_page_indices = [i for i, p in enumerate(page_paths) if p == task.core_file]
    session_page_idx = [i for i, p in enumerate(page_paths) if p == task.session_notes_path][-1]
    return {
        "page_tensor": page_tensor,
        "page_texts": page_texts,
        "page_paths": page_paths,
        "core_page_indices": core_page_indices,
        "session_page_idx": session_page_idx,
        "num_files": len(ordered_paths),
        "num_pages": len(page_ids_list),
        "total_content_tokens": total_content_tokens,
        "total_padded_tokens": len(page_ids_list) * page_size,
    }


def _code_value_variants(tokenizer, text: str) -> list[list[int]]:
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    special = {
        getattr(tokenizer, "pad_token_id", -1),
        getattr(tokenizer, "eos_token_id", -1),
        getattr(tokenizer, "bos_token_id", -1),
    } | (set(int(x) for x in nl_ids) if len(nl_ids) == 1 else set())

    raw_candidates = [
        text,
        f" {text}",
        f"={text}",
        f".{text}",
    ]
    unquoted = text.strip("\"'")
    if unquoted != text:
        raw_candidates.extend([unquoted, f" {unquoted}", f'"{unquoted}"', f"'{unquoted}'"])

    variants: list[list[int]] = []
    for cand in raw_candidates:
        toks = [int(t) for t in tokenizer.encode(cand, add_special_tokens=False) if int(t) not in special]
        if toks and toks not in variants:
            variants.append(toks)
    return variants


def _any_variant_in_bank(bank_set: set[int], variants: list[list[int]]) -> bool:
    return any(bool(v) and all(t in bank_set for t in v) for v in variants)


def measure_code_exact_bank_capture(
    exact_bank,
    core_page_idx: int,
    session_page_idx: int,
    task: MigrationTask,
    tokenizer,
) -> dict[str, float]:
    """Check whether key code identifiers landed verbatim in SparseExactBank slots."""
    core_set = set(int(x) for x in exact_bank.token_ids[0, core_page_idx].tolist())
    sess_set = set(int(x) for x in exact_bank.token_ids[0, session_page_idx].tolist())
    combined_set = core_set | sess_set

    sym_vars = _code_value_variants(tokenizer, task.symbol_name)
    kw_vars = _code_value_variants(tokenizer, task.new_kwarg)
    val_vars = _code_value_variants(tokenizer, task.default_value)
    decoy_vars = _code_value_variants(tokenizer, task.decoy_value)

    return {
        "in_exact_bank_symbol": float(_any_variant_in_bank(core_set, sym_vars) or _any_variant_in_bank(sess_set, sym_vars)),
        "in_exact_bank_kwarg": float(_any_variant_in_bank(core_set, kw_vars) or _any_variant_in_bank(sess_set, kw_vars)),
        "in_exact_bank_default": float(_any_variant_in_bank(sess_set, val_vars) or _any_variant_in_bank(combined_set, val_vars)),
        "decoy_in_exact_bank": float(_any_variant_in_bank(sess_set, decoy_vars)),
    }


def _tokenize_with_char_offsets(text: str, tokenizer) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize text while preserving aligned (start_char, end_char) offsets."""
    if not isinstance(tokenizer, _MockCodeTokenizer) and callable(tokenizer):
        try:
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            ids = [int(x) for x in enc["input_ids"]]
            offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
            if len(ids) == len(offsets):
                return ids, offsets
        except Exception:
            pass

    if isinstance(tokenizer, _MockCodeTokenizer):
        ids = tokenizer.encode(text, add_special_tokens=False)
        matches = list(re.finditer(r"[A-Za-z0-9_.\-$]+|[^\w\s]", text))
        if len(matches) == len(ids):
            return ids, [(m.start(), m.end()) for m in matches]
        return ids, [(0, 0)] * len(ids)

    ids = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
    offsets: list[tuple[int, int]] = []
    cursor = 0
    prev_len = 0
    for i in range(len(ids)):
        pref_dec = tokenizer.decode(ids[: i + 1], skip_special_tokens=False)
        piece_len = max(0, len(pref_dec) - prev_len)
        start = min(cursor, len(text))
        end = min(len(text), start + piece_len)
        offsets.append((start, end))
        cursor = end
        prev_len = len(pref_dec)
    return ids, offsets


def _mark_spans_on_mask(
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int]],
    page_size: int,
) -> torch.Tensor:
    mask = torch.zeros((page_size,), dtype=torch.float32)
    limit = min(len(offsets), page_size)
    for idx in range(limit):
        t_s, t_e = offsets[idx]
        if t_e <= t_s:
            continue
        for s_s, s_e in spans:
            if s_e > s_s and t_e > s_s and t_s < s_e:
                mask[idx] = 1.0
                break
    return mask


def _compute_python_code_salience_masks(
    formatted_text: str,
    page_size: int,
    pad_id: int,
    tokenizer,
    *,
    k_exact: int = 24,
) -> dict[str, torch.Tensor]:
    """Derive generic AST/line-based token masks for `# FILE: <path>\\n<code>` pages."""
    raw_ids, offsets = _tokenize_with_char_offsets(formatted_text, tokenizer)
    content_len = min(len(raw_ids), page_size)
    padded_ids = _pad_or_chunk_tokens(raw_ids, page_size, pad_id)[0]

    first_nl = formatted_text.find("\n")
    if first_nl < 0:
        first_nl = len(formatted_text)
    if formatted_text.startswith("# FILE: "):
        file_prefix_spans = [(0, min(8, first_nl))]
        path_spans = [(8, first_nl)] if first_nl > 8 else []
    else:
        file_prefix_spans = []
        path_spans = [(0, first_nl)] if first_nl > 0 else []

    body_offset = min(len(formatted_text), first_nl + 1)
    source_code = formatted_text[body_offset:]
    source_lines = source_code.splitlines(keepends=True)
    line_starts = [0]
    for ln in source_lines:
        line_starts.append(line_starts[-1] + len(ln))

    def _lc_to_char(lineno: int, col: int) -> int:
        idx = min(max(0, lineno - 1), len(line_starts) - 1)
        return min(len(formatted_text), body_offset + line_starts[idx] + max(0, col))

    doc_spans: list[tuple[int, int]] = []
    raw_sig_spans: list[tuple[int, int]] = []
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                first_stmt = node.body[0]
                if (
                    isinstance(first_stmt, ast.Expr)
                    and isinstance(first_stmt.value, ast.Constant)
                    and isinstance(first_stmt.value.value, str)
                    and first_stmt.end_lineno is not None
                ):
                    doc_spans.append(
                        (
                            _lc_to_char(first_stmt.lineno, first_stmt.col_offset),
                            _lc_to_char(first_stmt.end_lineno, first_stmt.end_col_offset),
                        )
                    )
        fn_nodes = [
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        fn_nodes.sort(key=lambda n: (n.lineno, n.col_offset))
        for fn in fn_nodes:
            def_start = _lc_to_char(fn.lineno, fn.col_offset)
            if fn.body:
                body_start = _lc_to_char(fn.body[0].lineno, fn.body[0].col_offset)
                colon_pos = formatted_text.rfind(":", def_start, max(def_start, body_start))
                if colon_pos >= def_start:
                    raw_sig_spans.append((def_start, colon_pos + 1))
                    continue
            ln_idx = fn.lineno - 1
            if 0 <= ln_idx < len(source_lines):
                raw_sig_spans.append((def_start, body_offset + line_starts[ln_idx + 1]))
    except SyntaxError:
        pass

    file_prefix_mask = _mark_spans_on_mask(offsets, file_prefix_spans, page_size)
    path_mask = _mark_spans_on_mask(offsets, path_spans, page_size)
    doc_mask = _mark_spans_on_mask(offsets, doc_spans, page_size)

    sig_mask = torch.zeros((page_size,), dtype=torch.float32)
    ambiguous_mask = torch.zeros((page_size,), dtype=torch.float32)
    used_budget = int(path_mask.sum().item()) + int(file_prefix_mask.sum().item())
    soft_cap = max(8, k_exact - 1)
    for fn_idx, span in enumerate(raw_sig_spans):
        fn_m = _mark_spans_on_mask(offsets, [span], page_size)
        fn_tok_cnt = int(fn_m.sum().item())
        if fn_idx == 0 or (used_budget + fn_tok_cnt <= soft_cap):
            sig_mask = torch.maximum(sig_mask, fn_m)
            used_budget += fn_tok_cnt
        else:
            ambiguous_mask = torch.maximum(ambiguous_mask, fn_m)

    doc_mask = doc_mask * (1.0 - sig_mask) * (1.0 - path_mask) * (1.0 - ambiguous_mask)
    pad_mask = torch.zeros((page_size,), dtype=torch.float32)
    if content_len < page_size:
        pad_mask[content_len:] = 1.0

    return {
        "token_ids": torch.tensor(padded_ids, dtype=torch.long),
        "sig_mask": sig_mask,
        "path_mask": path_mask,
        "file_prefix_mask": file_prefix_mask,
        "doc_mask": doc_mask,
        "pad_mask": pad_mask,
        "ambiguous_mask": ambiguous_mask,
    }


def _compute_session_notes_masks(
    formatted_text: str,
    decision_substrings: list[str],
    page_size: int,
    pad_id: int,
    tokenizer,
) -> dict[str, torch.Tensor]:
    """Derive generic masks for SESSION_NOTES.md pages boosting header path and ratified turn."""
    raw_ids, offsets = _tokenize_with_char_offsets(formatted_text, tokenizer)
    content_len = min(len(raw_ids), page_size)
    padded_ids = _pad_or_chunk_tokens(raw_ids, page_size, pad_id)[0]

    first_nl = formatted_text.find("\n")
    if first_nl < 0:
        first_nl = len(formatted_text)
    file_prefix_spans = [(0, min(8, first_nl))] if formatted_text.startswith("# FILE: ") else []
    path_spans = [(8, first_nl)] if (formatted_text.startswith("# FILE: ") and first_nl > 8) else []

    sig_spans: list[tuple[int, int]] = []
    for sub in decision_substrings:
        if not sub:
            continue
        pos = formatted_text.find(sub)
        if pos >= 0:
            sig_spans.append((pos, pos + len(sub)))

    sig_mask = _mark_spans_on_mask(offsets, sig_spans, page_size)
    path_mask = _mark_spans_on_mask(offsets, path_spans, page_size)
    file_prefix_mask = _mark_spans_on_mask(offsets, file_prefix_spans, page_size)
    pad_mask = torch.zeros((page_size,), dtype=torch.float32)
    if content_len < page_size:
        pad_mask[content_len:] = 1.0

    zeros = torch.zeros((page_size,), dtype=torch.float32)
    return {
        "token_ids": torch.tensor(padded_ids, dtype=torch.long),
        "sig_mask": sig_mask,
        "path_mask": path_mask,
        "file_prefix_mask": file_prefix_mask,
        "doc_mask": zeros,
        "pad_mask": pad_mask,
        "ambiguous_mask": zeros.clone(),
    }


def _set_only_module_trainable(wrapper: FrozenQwenLSLCWrapper, target_module_name: str | None) -> list[torch.nn.Parameter]:
    """Freeze entire LLM and all LSLC submodules except `target_module_name`."""
    wrapper.freeze_llm_()
    target_mod = getattr(wrapper.lslc, target_module_name) if target_module_name else None
    target_ids = {id(p) for p in target_mod.parameters()} if target_mod is not None else set()
    trainable: list[torch.nn.Parameter] = []
    for p in wrapper.lslc.parameters():
        req = id(p) in target_ids
        p.requires_grad_(req)
        if req:
            trainable.append(p)
    return trainable


def _run_stage_a_salience_alignment(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    train_pool: list[int],
    num_distractors: int,
    steps: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    """Tahap A: Align ONLY SalienceGate on 50% TRAIN_SPECS AST code+session + 50% Stage-1 prose replay."""
    if steps <= 0:
        return {"steps": 0, "skipped": True}

    bench = CrossFileAPIMigrationBenchmark(split="train", num_distractor_files=min(16, num_distractors))
    embed_fn = wrapper.llm.get_input_embeddings()
    vocab_size = len(embed_fn.weight)
    page_size = wrapper.cfg.page_size
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32

    core_items: list[dict[str, torch.Tensor]] = []
    distractor_items: list[dict[str, torch.Tensor]] = []
    session_items: list[dict[str, torch.Tensor]] = []

    for task in bench:
        core_items.append(
            _compute_python_code_salience_masks(
                f"# FILE: {task.core_file}\n{task.core_new_content}\n",
                page_size,
                pad_id,
                tokenizer,
                k_exact=wrapper.cfg.k_exact,
            )
        )
        core_items.append(
            _compute_python_code_salience_masks(
                f"# FILE: {task.core_file}\n{task.core_old_content}\n",
                page_size,
                pad_id,
                tokenizer,
                k_exact=wrapper.cfg.k_exact,
            )
        )
        for path, src in sorted(task.initial_files.items()):
            if path == task.core_file:
                continue
            distractor_items.append(
                _compute_python_code_salience_masks(
                    f"# FILE: {path}\n{src}\n",
                    page_size,
                    pad_id,
                    tokenizer,
                    k_exact=wrapper.cfg.k_exact,
                )
            )
        acc_text = next(str(t["text"]) for t in task.session_turns if bool(t["accepted"]))
        session_items.append(
            _compute_session_notes_masks(
                f"# FILE: {task.session_notes_path}\n{task.session_notes_initial}\n",
                [f"# Architecture Review Notes: {task.pkg_name}", f"{task.core_file}:{task.symbol_name}"],
                page_size,
                pad_id,
                tokenizer,
            )
        )
        session_items.append(
            _compute_session_notes_masks(
                f"# FILE: {task.session_notes_path}\n{task.session_notes_initial}\n{acc_text}\n",
                [acc_text],
                page_size,
                pad_id,
                tokenizer,
            )
        )

    n_prose = max(16, min(64, steps))
    prose_pages: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    for ds_cls, mode_name, s_off in (
        (AdversarialLivingBookDataset, "budget", 11),
        (AdversarialLivingBookDataset, "symbolic", 12),
    ):
        ds = ds_cls(
            num_episodes=n_prose // 2,
            num_pages=6 if isinstance(wrapper.llm, TinyMockCausalLM) else 16,
            page_size=page_size,
            num_edits=3,
            mode=mode_name,
            split="train",
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            seed=seed + s_off,
            background_token_pool=train_pool,
            style="mixed_natural",
        )
        for ep in ds:
            ep_v = _build_stage1_episode_view(ep)
            sal_m = ep_v.metadata.get("verified_salience_span_mask")
            t_idx = ep_v.target_page_idx
            prose_pages.append(
                (ep_v.context_pages[t_idx], None if sal_m is None else sal_m[t_idx])
            )
    niah_ds = SyntheticExactSemanticNIAHDataset(
        num_episodes=max(8, n_prose // 2),
        num_pages=6 if isinstance(wrapper.llm, TinyMockCausalLM) else 16,
        page_size=page_size,
        vocab_size=vocab_size,
        mode="mixed",
        tokenizer=tokenizer,
        seed=seed + 13,
        background_token_pool=train_pool,
    )
    for ep in niah_ds:
        t_idx = ep.target_page_idx
        sal_m = ep.metadata.get("verified_salience_span_mask")
        prose_pages.append((ep.context_pages[t_idx], None if sal_m is None else sal_m[t_idx]))

    sal_params = _set_only_module_trainable(wrapper, "salience_gate")
    non_sal_grad = [n for n, p in wrapper.named_parameters() if p.requires_grad and not n.startswith("lslc.salience_gate.")]
    assert not non_sal_grad, f"Unexpected trainable params in Tahap A: {non_sal_grad}"

    optimizer = torch.optim.AdamW(sal_params, lr=lr, weight_decay=wrapper.cfg.weight_decay, betas=wrapper.cfg.betas)
    warmup = max(1, int(steps * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: float(s + 1) / warmup
        if s < warmup
        else max(0.1, 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))),
    )
    wrapper.lslc.eval()
    wrapper.lslc.salience_gate.train()

    rng = random.Random(seed)
    t0 = time.perf_counter()
    last_code_loss = 0.0
    last_prose_loss = 0.0
    pos_w = torch.tensor(8.0, device=device, dtype=torch.float32)

    for step in range(1, steps + 1):
        is_code_step = (step % 2 == 1)
        if is_code_step:
            r = rng.random()
            if r < 0.50:
                item = core_items[rng.randrange(len(core_items))]
            elif r < 0.75:
                item = session_items[rng.randrange(len(session_items))]
            else:
                item = distractor_items[rng.randrange(len(distractor_items))]

            sal_ids_t = item["token_ids"].unsqueeze(0).to(device)
            sig_m = item["sig_mask"].unsqueeze(0).to(device)
            path_m = item["path_mask"].unsqueeze(0).to(device)
            fpref_m = item["file_prefix_mask"].unsqueeze(0).to(device)
            doc_m = item["doc_mask"].unsqueeze(0).to(device)
            pad_m = item["pad_mask"].unsqueeze(0).to(device)
            amb_m = item["ambiguous_mask"].unsqueeze(0).to(device)

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    logits_llm = wrapper.llm(input_ids=sal_ids_t).logits
                shift_l = logits_llm[:, :-1, :].contiguous().float()
                shift_t = sal_ids_t[:, 1:].contiguous()
                nll = F.cross_entropy(
                    shift_l.view(-1, shift_l.shape[-1]), shift_t.view(-1), reduction="none"
                ).view(1, -1)
                surp = F.pad(nll, (1, 0), value=0.0)
                surp_smooth = 0.7 * surp + 0.3 * F.avg_pool1d(
                    surp.unsqueeze(1), kernel_size=3, stride=1, padding=1
                ).squeeze(1)
                teacher_score = (
                    surp_smooth
                    + 20.0 * sig_m
                    + 14.0 * path_m
                    + 3.0 * fpref_m
                    - 12.0 * doc_m
                    - 1e4 * pad_m
                )
                k_target = min(wrapper.cfg.k_exact, sal_ids_t.shape[1])
                top_idx = torch.topk(teacher_score, k=k_target, dim=-1).indices
                target_mask = torch.zeros_like(surp)
                target_mask.scatter_(1, top_idx, 1.0)
                sal_embeds = embed_fn(sal_ids_t)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                g_logits = wrapper.lslc.salience_gate.score_logits(
                    sal_embeds, surprisal=None, token_ids=sal_ids_t, vocab_size=vocab_size
                )
                raw_bce = F.binary_cross_entropy_with_logits(
                    g_logits.float(), target_mask.float(), pos_weight=pos_w, reduction="none"
                )
                tok_w = (1.0 + 1.0 * sig_m + 1.0 * doc_m) * (1.0 - amb_m)
                loss = (raw_bce * tok_w).sum() / tok_w.sum().clamp(min=1.0)
        else:
            p_ids, span_m = prose_pages[rng.randrange(len(prose_pages))]
            sal_ids_t = p_ids.unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    logits_llm = wrapper.llm(input_ids=sal_ids_t).logits
                shift_l = logits_llm[:, :-1, :].contiguous().float()
                shift_t = sal_ids_t[:, 1:].contiguous()
                nll = F.cross_entropy(
                    shift_l.view(-1, shift_l.shape[-1]), shift_t.view(-1), reduction="none"
                ).view(1, -1)
                surp = F.pad(nll, (1, 0), value=0.0)
                surp_smooth = 0.7 * surp + 0.3 * F.avg_pool1d(
                    surp.unsqueeze(1), kernel_size=3, stride=1, padding=1
                ).squeeze(1)
                if span_m is not None:
                    surp_smooth = surp_smooth + 10.0 * span_m.unsqueeze(0).to(device=device, dtype=surp_smooth.dtype)
                k_target = min(wrapper.cfg.k_exact, sal_ids_t.shape[1])
                top_idx = torch.topk(surp_smooth, k=k_target, dim=-1).indices
                target_mask = torch.zeros_like(surp)
                target_mask.scatter_(1, top_idx, 1.0)
                sal_embeds = embed_fn(sal_ids_t)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                g_logits = wrapper.lslc.salience_gate.score_logits(
                    sal_embeds, surprisal=None, token_ids=sal_ids_t, vocab_size=vocab_size
                )
                loss = F.binary_cross_entropy_with_logits(
                    g_logits.float(), target_mask.float(), pos_weight=pos_w
                )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(sal_params, 1.0).item())
        if not math.isfinite(float(loss.item())) or not math.isfinite(gnorm):
            raise RuntimeError(f"[Stage-1.5A] Non-finite loss/grad at step {step}: loss={loss.item()} gnorm={gnorm}")
        optimizer.step()
        scheduler.step()

        l_val = float(loss.detach().item())
        if is_code_step:
            last_code_loss = l_val
        else:
            last_prose_loss = l_val
        if step == 1 or step % 25 == 0 or step == steps:
            print(
                f"[Stage-1.5A] step={step}/{steps} code_loss={last_code_loss:.4f} prose_loss={last_prose_loss:.4f} gnorm={gnorm:.3f}",
                flush=True,
            )

    wrapper.lslc.eval()
    recalls: list[float] = []
    with torch.no_grad():
        for task in bench:
            item = _compute_python_code_salience_masks(
                f"# FILE: {task.core_file}\n{task.core_new_content}\n",
                page_size,
                pad_id,
                tokenizer,
                k_exact=wrapper.cfg.k_exact,
            )
            sig_pos = set(torch.nonzero(item["sig_mask"] > 0.5, as_tuple=False).view(-1).tolist())
            t_in = item["token_ids"].unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                sel_pos, _, _, _ = wrapper.lslc.salience_gate(
                    embed_fn(t_in), surprisal=None, token_ids=t_in, vocab_size=vocab_size
                )
            sel_set = set(int(x) for x in sel_pos[0].tolist())
            if sig_pos:
                recalls.append(len(sig_pos & sel_set) / len(sig_pos))

    return {
        "steps": steps,
        "lr": lr,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "final_code_loss": round(last_code_loss, 4),
        "final_prose_loss": round(last_prose_loss, 4),
        "train_core_sig_token_recall": round(sum(recalls) / max(1, len(recalls)), 4),
    }


def _compute_code_target_weights(
    orig_src: str,
    gold_src: str,
    target_str: str,
    tokenizer,
    add_eos_id: int | None,
) -> tuple[list[int], list[float]]:
    """Upweight modified signature/call/assert lines generically via line-diff."""
    ids, offsets = _tokenize_with_char_offsets(target_str, tokenizer)
    orig_lines = {ln.strip() for ln in orig_src.splitlines() if ln.strip()}
    gold_lines = gold_src.rstrip().splitlines(keepends=True)

    changed_spans: list[tuple[int, int]] = []
    cursor = 0
    for ln in gold_lines:
        ln_end = cursor + len(ln)
        if ln.strip() and ln.strip() not in orig_lines:
            changed_spans.append((cursor, ln_end))
        cursor = ln_end
    if cursor < len(target_str):
        # Closing fence delimiter
        changed_spans.append((cursor, len(target_str)))

    weights: list[float] = []
    for t_s, t_e in offsets:
        in_changed = any(t_e > c_s and t_s < c_e for c_s, c_e in changed_spans)
        weights.append(2.5 if in_changed else 1.0)
    if add_eos_id is not None:
        ids.append(int(add_eos_id))
        weights.append(1.5)
    return ids, weights


def _encode_book_no_grad(
    wrapper: FrozenQwenLSLCWrapper,
    page_tensor: torch.Tensor,
    embed_fn,
    vocab_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    autocast_enabled: bool,
):
    from lslc.encoder import SparseExactBank

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            z_chunks, tok_chunks, h_chunks, w_chunks = [], [], [], []
            for s in range(0, page_tensor.shape[1], 32):
                e = min(page_tensor.shape[1], s + 32)
                zc, bc, _ = wrapper.lslc.encode_book(
                    page_tensor[:, s:e], embed_fn, use_exact_bank=True, vocab_size=vocab_size
                )
                z_chunks.append(zc)
                tok_chunks.append(bc.token_ids)
                h_chunks.append(bc.h_coords)
                w_chunks.append(bc.w_coords)
            z_book = torch.cat(z_chunks, dim=1)
            num_p = page_tensor.shape[1]
            p_ids = (
                torch.arange(num_p, device=device, dtype=torch.int16)
                .view(1, num_p, 1)
                .expand(1, num_p, wrapper.cfg.k_exact)
                .contiguous()
            )
            exact_bank = SparseExactBank(
                token_ids=torch.cat(tok_chunks, dim=1),
                page_ids=p_ids,
                h_coords=torch.cat(h_chunks, dim=1),
                w_coords=torch.cat(w_chunks, dim=1),
            )
            toc_embeds = wrapper.lslc.projector.project_toc(z_book)
    return z_book, exact_bank, toc_embeds


def _run_stage_b_projector_alignment(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    train_pool: list[int],
    num_distractors: int,
    steps: int,
    lr: float,
    lslc_k_skim: int,
    salience_route_weight: float,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    """Tahap B: Align ONLY SpatialLatentProjector on 50% ChatML/Code + 50% Stage-1 prose replay."""
    if steps <= 0:
        return {"steps": 0, "skipped": True}

    bench = CrossFileAPIMigrationBenchmark(split="train", num_distractor_files=num_distractors)
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    page_size = wrapper.cfg.page_size
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(wrapper.llm, TinyMockCausalLM):
        eos_id_use: int | None = int(eos_id) if eos_id is not None else None
    else:
        eos_id_use = int(eos_id) if eos_id is not None else None
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32

    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)
    user_open_embeds = embed_fn(user_open_ids).to(llm_dtype).detach()

    # Pre-cache encoded 38-file train repos with modular core.py re-encode + blind DeltaUpdater
    wrapper.lslc.eval()
    code_cache: list[dict[str, object]] = []
    train_acc_gates: list[float] = []
    train_rej_gates: list[float] = []

    for task in bench:
        pack = _build_repo_pages(task, tokenizer, page_size)
        page_paths: list[str] = list(pack["page_paths"])
        core_page_idx = int(pack["core_page_indices"][0])
        session_page_idx = int(pack["session_page_idx"])
        z_book, exact_bank, toc_embeds = _encode_book_no_grad(
            wrapper, pack["page_tensor"].to(device), embed_fn, vocab_size, device, amp_dtype, autocast_enabled
        )

        core_fmt_new = f"# FILE: {task.core_file}\n{task.core_new_content}\n"
        core_toks_new = _pad_or_chunk_tokens(
            tokenizer.encode(core_fmt_new, add_special_tokens=False), page_size, pad_id
        )[0]
        core_page_t = torch.tensor(core_toks_new, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                z_book, exact_bank, toc_embeds = wrapper.lslc.reencode_page(
                    z_book,
                    toc_embeds,
                    exact_bank,
                    page_idx=core_page_idx,
                    new_page_token_ids=core_page_t,
                    embed_tokens_fn=embed_fn,
                    vocab_size=vocab_size,
                    in_place=True,
                )
            sess_p_t = torch.tensor([session_page_idx], dtype=torch.long, device=device)
            for turn in task.session_turns:
                edit_toks = torch.tensor(
                    tokenizer.encode(str(turn["text"]), add_special_tokens=False),
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    override = (
                        extract_contextual_edit_embeds(
                            wrapper.llm, edit_toks, layer_idx=wrapper.cfg.contextual_edit_layer
                        )
                        if wrapper.cfg.use_contextual_edit_encoding
                        else None
                    )
                    z_book, gates, exact_bank = wrapper.lslc.apply_page_delta(
                        z_book,
                        sess_p_t,
                        edit_toks,
                        embed_tokens_fn=embed_fn,
                        exact_bank=exact_bank,
                        vocab_size=vocab_size,
                        edit_embeds_override=override,
                        in_place=True,
                        bank_write_threshold=0.45,
                    )
                g_val = float(gates[:, -1, :].mean().item())
                (train_acc_gates if bool(turn["accepted"]) else train_rej_gates).append(g_val)

        target_items: list[dict[str, object]] = []
        for t_path in task.caller_files:
            orig_src = task.initial_files[t_path]
            gold_src = task.gold_files[t_path]
            prompt_str = _build_edit_prompt(task, t_path, orig_src)
            target_str = gold_src.rstrip() + "\n```"
            tgt_ids_list, tgt_w_list = _compute_code_target_weights(
                orig_src, gold_src, target_str, tokenizer, eos_id_use
            )
            excluded = {i for i, p in enumerate(page_paths) if p == t_path}
            rq_str = _build_router_query(task, t_path)
            target_items.append(
                {
                    "path": t_path,
                    "weight_group": 0.30 if "service_" in t_path else 0.25,
                    "prompt_ids": torch.tensor(
                        tokenizer.encode(prompt_str, add_special_tokens=False), dtype=torch.long, device=device
                    ).unsqueeze(0),
                    "target_ids": torch.tensor(tgt_ids_list, dtype=torch.long, device=device).unsqueeze(0),
                    "target_weights": torch.tensor(tgt_w_list, dtype=torch.float32, device=device),
                    "router_query_ids": torch.tensor(
                        tokenizer.encode(rq_str, add_special_tokens=False), dtype=torch.long, device=device
                    ).unsqueeze(0),
                    "excluded": excluded,
                }
            )

        # Homonym preservation target (`KEEP_UNCHANGED`)
        orig_h = task.initial_files[task.homonym_file]
        h_prompt = _build_homonym_prompt(task, orig_h)
        h_tgt_ids = tokenizer.encode("KEEP_UNCHANGED", add_special_tokens=False)
        h_weights = [1.0] * len(h_tgt_ids)
        if eos_id_use is not None:
            h_tgt_ids.append(eos_id_use)
            h_weights.append(1.0)
        target_items.append(
            {
                "path": task.homonym_file,
                "weight_group": 0.15,
                "prompt_ids": torch.tensor(
                    tokenizer.encode(h_prompt, add_special_tokens=False), dtype=torch.long, device=device
                ).unsqueeze(0),
                "target_ids": torch.tensor(h_tgt_ids, dtype=torch.long, device=device).unsqueeze(0),
                "target_weights": torch.tensor(h_weights, dtype=torch.float32, device=device),
                "router_query_ids": torch.tensor(
                    tokenizer.encode(_build_router_query(task, task.homonym_file), add_special_tokens=False),
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0),
                "excluded": {i for i, p in enumerate(page_paths) if p == task.homonym_file},
            }
        )
        code_cache.append(
            {
                "z_book": z_book.detach(),
                "exact_bank": exact_bank,
                "target_items": target_items,
            }
        )

    # Pre-cache Stage-1 + Stage-2 non-ChatML mixed_natural prose replay episodes
    prose_cache: list[dict[str, object]] = []
    num_prose_pages = 6 if isinstance(wrapper.llm, TinyMockCausalLM) else 16
    n_s1_each = 8 if isinstance(wrapper.llm, TinyMockCausalLM) else 24
    n_s2_each = 4 if isinstance(wrapper.llm, TinyMockCausalLM) else 12
    prose_eps: list[tuple[object, bool]] = []
    for ds_mode, s_off in (("budget", 21), ("symbolic", 22)):
        ds_s1 = AdversarialLivingBookDataset(
            num_episodes=n_s1_each,
            num_pages=num_prose_pages,
            page_size=page_size,
            num_edits=3,
            mode=ds_mode,
            split="train",
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            seed=seed + s_off,
            background_token_pool=train_pool,
            style="mixed_natural",
        )
        for ep in ds_s1:
            prose_eps.append((_build_stage1_episode_view(ep), False))
        ds_s2 = AdversarialLivingBookDataset(
            num_episodes=n_s2_each,
            num_pages=num_prose_pages,
            page_size=page_size,
            num_edits=3 if ds_mode == "budget" else 4,
            mode=ds_mode,
            split="train",
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            seed=seed + s_off + 100,
            background_token_pool=train_pool,
            style="mixed_natural",
        )
        for ep in ds_s2:
            prose_eps.append((ep, True))

    for ep, is_delta in prose_eps:
        pages_p = ep.context_pages.unsqueeze(0).to(device)
        z_p, bank_p, _ = _encode_book_no_grad(
            wrapper, pages_p, embed_fn, vocab_size, device, amp_dtype, autocast_enabled
        )
        if is_delta and ep.edit_pages:
            with torch.no_grad():
                for e_p, e_tok in ep.edit_pages:
                    p_t = torch.tensor([e_p], dtype=torch.long, device=device)
                    e_t = e_tok.unsqueeze(0).to(device)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        override = (
                            extract_contextual_edit_embeds(
                                wrapper.llm, e_t, layer_idx=wrapper.cfg.contextual_edit_layer
                            )
                            if wrapper.cfg.use_contextual_edit_encoding
                            else None
                        )
                        z_p, _, bank_p = wrapper.lslc.apply_page_delta(
                            z_p,
                            p_t,
                            e_t,
                            embed_fn,
                            exact_bank=bank_p,
                            vocab_size=vocab_size,
                            edit_embeds_override=override,
                            in_place=True,
                        )
        rq_raw = ep.metadata.get("router_query_ids", ep.query_ids)
        prose_cache.append(
            {
                "z_book": z_p.detach(),
                "exact_bank": bank_p,
                "q_ids": ep.query_ids.unsqueeze(0).to(device),
                "r_q_ids": rq_raw.unsqueeze(0).to(device),
                "a_ids": ep.answer_ids.unsqueeze(0).to(device),
                "t_page": torch.tensor([ep.target_page_idx], dtype=torch.long, device=device),
                "is_delta": is_delta,
            }
        )

    proj_params = _set_only_module_trainable(wrapper, "projector")
    non_proj_grad = [n for n, p in wrapper.named_parameters() if p.requires_grad and not n.startswith("lslc.projector.")]
    assert not non_proj_grad, f"Unexpected trainable params in Tahap B: {non_proj_grad}"
    wrapper.llm.eval()
    if hasattr(wrapper.llm, "config"):
        wrapper.llm.config.use_cache = False
    wrapper.lslc.eval()
    wrapper.lslc.projector.train()

    optimizer = torch.optim.AdamW(proj_params, lr=lr, weight_decay=wrapper.cfg.weight_decay, betas=wrapper.cfg.betas)
    warmup = max(1, int(steps * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: float(s + 1) / warmup
        if s < warmup
        else max(0.1, 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))),
    )

    rng = random.Random(seed + 7)
    t0 = time.perf_counter()
    last_code_ntp = 0.0
    last_prose_ntp = 0.0
    last_route_loss = 0.0

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)

        # Micro-batch 1: TRAIN_SPECS ChatML + Code NTP (grad-accumulated 0.5x)
        repo_entry = code_cache[rng.randrange(len(code_cache))]
        items = repo_entry["target_items"]
        chosen_item = rng.choices(items, weights=[float(it["weight_group"]) for it in items], k=1)[0]
        kd_step = 4 if (step % 4 == 0) else 2
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            toc_c = wrapper.lslc.projector.project_toc(repo_entry["z_book"])
            _, pref_c = _lslc_route_and_assemble_excluding(
                wrapper=wrapper,
                z_book=repo_entry["z_book"],
                toc_embeds=toc_c,
                exact_bank=repo_entry["exact_bank"],
                embed_fn=embed_fn,
                router_query_ids=chosen_item["router_query_ids"],
                k_deep=kd_step,
                k_skim=lslc_k_skim,
                salience_route_weight=salience_route_weight,
                excluded_indices=chosen_item["excluded"],
                mode="full",
                exact_bank_attenuation="none",
                flat_toc_max_tokens=16,
            )
            q_ids_c = chosen_item["prompt_ids"]
            t_ids_c = chosen_item["target_ids"]
            t_w_c = chosen_item["target_weights"]
            inp_c = torch.cat(
                (
                    user_open_embeds,
                    pref_c.to(llm_dtype),
                    embed_fn(q_ids_c).to(llm_dtype),
                    embed_fn(t_ids_c).to(llm_dtype),
                ),
                dim=1,
            )
            attn_c = torch.ones((1, inp_c.shape[1]), device=device, dtype=torch.long)
            logits_c = wrapper.llm(inputs_embeds=inp_c, attention_mask=attn_c).logits
            s_start = user_open_embeds.shape[1] + pref_c.shape[1] + q_ids_c.shape[1] - 1
            pred_c = logits_c[:, s_start : s_start + t_ids_c.shape[1], :].float()
            tok_ce = F.cross_entropy(
                pred_c.reshape(-1, pred_c.shape[-1]),
                t_ids_c.reshape(-1),
                reduction="none",
            )
            loss_code = (tok_ce * t_w_c).sum() / t_w_c.sum().clamp(min=1e-6)
        (0.5 * loss_code).backward()

        # Micro-batch 2: Stage-1/2 non-ChatML prose replay + dense-TOC router preservation (0.5x)
        pr = prose_cache[rng.randrange(len(prose_cache))]
        sal_w_p = 2.0 if bool(pr["is_delta"]) else 0.0
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            toc_p = wrapper.lslc.projector.project_toc(pr["z_book"])
            q_emb_p = embed_fn(pr["r_q_ids"]).mean(dim=1)
            r_out_p, asm_p = wrapper.lslc.route_and_assemble(
                z_book=pr["z_book"],
                toc_embeds=toc_p,
                query_embed=q_emb_p,
                exact_bank=pr["exact_bank"],
                embed_tokens_fn=embed_fn,
                k_deep=wrapper.cfg.k_deep,
                k_skim=wrapper.cfg.k_skim,
                mode="full",
                training_ste=False,
                query_input_ids=pr["r_q_ids"],
                salience_route_weight=sal_w_p,
                flat_toc_max_tokens=16,
                exact_bank_attenuation="none",
            )
            turn_ids = torch.cat((pr["q_ids"], pr["a_ids"]), dim=1)
            turn_labels = torch.cat((torch.full_like(pr["q_ids"], -100), pr["a_ids"]), dim=1)
            loss_prose_ntp = wrapper._run_llm_with_prefix(asm_p.prefix_embeds, turn_ids, turn_labels=turn_labels)

            if sal_w_p == 0.0:
                dense_logits = r_out_p.logits
            else:
                q_tok_p = embed_fn(pr["r_q_ids"])
                bank_p_emb = embed_fn(pr["exact_bank"].routing_token_ids().clamp(min=0).to(device))
                dense_logits = wrapper.lslc.router(
                    q_emb_p,
                    toc_p,
                    k_deep=wrapper.cfg.k_deep,
                    k_skim=wrapper.cfg.k_skim,
                    query_token_embeds=q_tok_p,
                    exact_bank_embeds=bank_p_emb,
                    salience_route_weight=0.0,
                ).logits
            loss_route_p = F.cross_entropy(dense_logits, pr["t_page"])
            loss_prose = loss_prose_ntp + 0.5 * loss_route_p
        (0.5 * loss_prose).backward()

        gnorm = float(torch.nn.utils.clip_grad_norm_(proj_params, 1.0).item())
        code_val = float(loss_code.detach().item())
        prose_val = float(loss_prose_ntp.detach().item())
        route_val = float(loss_route_p.detach().item())
        if not (math.isfinite(code_val) and math.isfinite(prose_val) and math.isfinite(gnorm)):
            raise RuntimeError(
                f"[Stage-1.5B] Non-finite loss/grad at step {step}: code={code_val} prose={prose_val} gnorm={gnorm}"
            )
        optimizer.step()
        scheduler.step()

        last_code_ntp = code_val
        last_prose_ntp = prose_val
        last_route_loss = route_val
        if step == 1 or step % 20 == 0 or step == steps:
            cur_lr = float(optimizer.param_groups[0]["lr"])
            print(
                f"[Stage-1.5B] step={step}/{steps} code_ntp={code_val:.4f} prose_ntp={prose_val:.4f} "
                f"route={route_val:.4f} lr={cur_lr:.2e} gnorm={gnorm:.3f}",
                flush=True,
            )

    wrapper.lslc.eval()
    return {
        "steps": steps,
        "lr": lr,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "final_code_ntp": round(last_code_ntp, 4),
        "final_prose_ntp": round(last_prose_ntp, 4),
        "final_route_loss": round(last_route_loss, 4),
        "train_delta_accepted_gate_mean": round(sum(train_acc_gates) / max(1, len(train_acc_gates)), 4),
        "train_delta_rejected_gate_mean": round(sum(train_rej_gates) / max(1, len(train_rej_gates)), 4),
    }


def _finetune_salience_on_train_code(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    num_distractors: int,
    steps: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Optional Stage-1b SalienceGate adaptation strictly on TRAIN_SPECS code tasks."""
    if steps <= 0:
        return {"steps": 0}

    bench = CrossFileAPIMigrationBenchmark(split="train", num_distractor_files=min(8, num_distractors))
    embed_fn = wrapper.llm.get_input_embeddings()
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32

    saved_reqs: list[tuple[torch.nn.Parameter, bool]] = []
    sal_ids = {id(p) for p in wrapper.lslc.salience_gate.parameters()}
    for p in wrapper.lslc.parameters():
        saved_reqs.append((p, p.requires_grad))
        p.requires_grad_(id(p) in sal_ids)

    sal_params = [p for p in wrapper.lslc.salience_gate.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(sal_params, lr=wrapper.cfg.lr, weight_decay=wrapper.cfg.weight_decay)
    wrapper.lslc.salience_gate.train()
    rng = random.Random(seed)
    t0 = time.perf_counter()
    final_loss = 0.0

    # Pre-encode train tasks with updated core.py + accepted session decision
    train_items: list[tuple[torch.Tensor, list[int], list[list[int]]]] = []
    for task in bench:
        pack = _build_repo_pages(task, tokenizer, wrapper.cfg.page_size)
        page_tensor = pack["page_tensor"].clone()
        pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        core_fmt = f"# FILE: {task.core_file}\n{task.core_new_content}\n"
        core_toks = tokenizer.encode(core_fmt, add_special_tokens=False)
        core_p = pack["core_page_indices"][0]
        page_tensor[0, core_p] = torch.tensor(
            _pad_or_chunk_tokens(core_toks, wrapper.cfg.page_size, pad_id)[0], dtype=torch.long
        )
        sess_p = int(pack["session_page_idx"])
        acc_text = next(str(t["text"]) for t in task.session_turns if bool(t["accepted"]))
        sess_fmt = f"# FILE: {task.session_notes_path}\n{task.session_notes_initial}\n{acc_text}\n"
        sess_toks = tokenizer.encode(sess_fmt, add_special_tokens=False)
        page_tensor[0, sess_p] = torch.tensor(
            _pad_or_chunk_tokens(sess_toks, wrapper.cfg.page_size, pad_id)[0], dtype=torch.long
        )
        key_seqs = (
            _code_value_variants(tokenizer, task.symbol_name)[:2]
            + _code_value_variants(tokenizer, task.new_kwarg)[:2]
            + _code_value_variants(tokenizer, task.default_value)[:2]
        )
        train_items.append((page_tensor, [core_p, sess_p], key_seqs))

    try:
        for step in range(1, steps + 1):
            page_tensor, target_pages, key_seqs = train_items[rng.randrange(len(train_items))]
            p_idx = target_pages[rng.randrange(len(target_pages))]
            sal_ids_t = page_tensor[:, p_idx, :].to(device)
            seq_list = [int(x) for x in sal_ids_t[0].tolist()]

            span_mask = torch.zeros_like(sal_ids_t, dtype=torch.float32)
            for seq in key_seqs:
                m_len = len(seq)
                for pos in range(0, len(seq_list) - m_len + 1):
                    if seq_list[pos : pos + m_len] == seq:
                        span_mask[0, pos : pos + m_len] = 1.0

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    logits_llm = wrapper.llm(input_ids=sal_ids_t).logits
                shift_l = logits_llm[:, :-1, :].contiguous().float()
                shift_t = sal_ids_t[:, 1:].contiguous()
                nll = F.cross_entropy(shift_l.view(-1, shift_l.shape[-1]), shift_t.view(-1), reduction="none").view(1, -1)
                surp = F.pad(nll, (1, 0), value=0.0)
                surp_smooth = 0.7 * surp + 0.3 * F.avg_pool1d(surp.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
                surp_smooth = surp_smooth + 12.0 * span_mask
                k_target = min(wrapper.cfg.k_exact, sal_ids_t.shape[1])
                top_idx = torch.topk(surp_smooth, k=k_target, dim=-1).indices
                target_mask = torch.zeros_like(surp)
                target_mask.scatter_(1, top_idx, 1.0)
                sal_embeds = embed_fn(sal_ids_t)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                g_logits = wrapper.lslc.salience_gate.score_logits(
                    sal_embeds, surprisal=None, token_ids=sal_ids_t, vocab_size=vocab_size
                )
                pos_w = torch.tensor(8.0, device=device, dtype=torch.float32)
                loss = F.binary_cross_entropy_with_logits(
                    g_logits.float(), target_mask.float(), pos_weight=pos_w
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sal_params, wrapper.cfg.grad_clip)
            optimizer.step()
            final_loss = float(loss.detach().item())
    finally:
        for p, prev_req in saved_reqs:
            p.requires_grad_(prev_req)
        wrapper.lslc.eval()

    return {
        "steps": steps,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "final_salience_loss": round(final_loss, 4),
    }


def _build_router_query(task: MigrationTask, target_path: str) -> str:
    return (
        f"{task.pkg_name} {task.symbol_name} {task.core_file} {task.session_notes_path} "
        f"architecture review session caller default contract migration {target_path}"
    )


def _bm25_route_excluding(
    bm25: BM25PageRetriever,
    query_text: str,
    k_deep: int,
    excluded_indices: set[int],
) -> list[int]:
    scores = bm25.score_pages(query_text)
    for idx in excluded_indices:
        if 0 <= idx < len(scores):
            scores[idx] = -1e9
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    # Put highest-scoring page last right before target working memory
    return list(reversed(order[: min(k_deep, len(order))]))


def _lslc_route_and_assemble_excluding(
    wrapper: FrozenQwenLSLCWrapper,
    z_book: torch.Tensor,
    toc_embeds: torch.Tensor,
    exact_bank,
    embed_fn,
    router_query_ids: torch.Tensor,
    k_deep: int,
    k_skim: int,
    salience_route_weight: float,
    excluded_indices: set[int],
    *,
    mode: str = "full",
    exact_bank_attenuation: str = "prob_sq",
    flat_toc_max_tokens: int | None = None,
    external_dense_logits: torch.Tensor | None = None,
) -> tuple[RouterOutput, torch.Tensor]:
    q_tok_emb = embed_fn(router_query_ids)
    q_emb = q_tok_emb.mean(dim=1)
    bank_emb = None
    delta_bank_emb = None
    if exact_bank is not None and (salience_route_weight > 0.0 or wrapper.cfg.router_mode == "content_attn"):
        bank_emb = embed_fn(exact_bank.routing_token_ids().clamp(min=0).to(z_book.device))
        if salience_route_weight > 0.0 and exact_bank.base_token_ids is not None:
            delta_bank_emb = embed_fn(exact_bank.token_ids.clamp(min=0).to(z_book.device))

    raw_router = wrapper.lslc.router(
        q_emb,
        toc_embeds,
        k_deep=k_deep,
        k_skim=k_skim,
        query_token_embeds=q_tok_emb,
        exact_bank_embeds=bank_emb,
        salience_route_weight=salience_route_weight,
        delta_bank_embeds=delta_bank_emb,
        external_dense_logits=external_dense_logits,
    )
    logits = raw_router.logits.clone()
    num_pages = logits.shape[1]
    for idx in excluded_indices:
        if 0 <= idx < num_pages:
            logits[0, idx] = -1e9

    kd = min(num_pages, max(0, k_deep))
    ks = min(max(0, num_pages - kd), max(0, k_skim))
    sorted_idx = torch.argsort(logits, dim=-1, descending=True)
    deep_idx = sorted_idx[:, :kd].flip(dims=[-1])
    skim_idx = torch.sort(sorted_idx[:, kd : kd + ks], dim=-1).values
    toc_idx = torch.sort(sorted_idx[:, kd + ks :], dim=-1).values
    masked_router = RouterOutput(
        logits=logits,
        probs=F.softmax(logits, dim=-1),
        deep_idx=deep_idx,
        skim_idx=skim_idx,
        toc_idx=toc_idx,
    )
    _, assembly = wrapper.lslc.route_and_assemble(
        z_book=z_book,
        toc_embeds=toc_embeds,
        query_embed=q_emb,
        exact_bank=exact_bank,
        embed_tokens_fn=embed_fn,
        k_deep=kd,
        k_skim=ks,
        mode=mode,
        training_ste=False,
        router_out_override=masked_router,
        flat_toc_max_tokens=flat_toc_max_tokens,
        exact_bank_attenuation=exact_bank_attenuation,
    )
    return masked_router, assembly.prefix_embeds


def _inspect_exact_bank_page(token_row: torch.Tensor, tokenizer) -> dict[str, object]:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    special = {int(x) for x in (pad_id, eos_id) if x is not None}
    ids = [int(x) for x in token_row.tolist()]
    non_special = [tid for tid in ids if tid not in special]
    return {
        "token_ids": ids,
        "num_total": len(ids),
        "num_special_or_pad": len(ids) - len(non_special),
        "token_pieces": [tokenizer.decode([tid], skip_special_tokens=False) for tid in ids],
        "decoded_clean": tokenizer.decode(non_special, skip_special_tokens=True) if non_special else "",
    }


def _parse_lslc_arm_config(system_name: str, default_k_skim: int) -> tuple[int, int, str, str, int | None]:
    k_val = int(system_name.rsplit("_k", 1)[-1])
    if system_name.startswith("lslc_exact_only_nosquash_k"):
        return k_val, 0, "exact_only", "none", None
    if system_name.startswith("lslc_full_nosquash_compact_toc_k"):
        return k_val, default_k_skim, "full", "none", 16
    return k_val, default_k_skim, "full", "prob_sq", None


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks, handling unclosed trailing <think> cleanly."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in cleaned:
        before = cleaned.split("<think>", 1)[0].strip()
        return before
    return cleaned.strip()


def _should_stop_code_gen(decoded: str, *, expect_opening_fence: bool = False) -> bool:
    """Never stop inside an unclosed <think>; wait for closing ``` fence when an opening fence is emitted."""
    if "<think>" in decoded and "</think>" not in decoded:
        return False
    post = decoded.split("</think>", 1)[1] if "</think>" in decoded else decoded
    post_s = post.lstrip()
    if not post_s:
        return False
    if expect_opening_fence or post_s.startswith("```") or "```python" in post_s:
        return post_s.count("```") >= 2
    return "```" in post_s


def _should_stop_homonym_gen(decoded: str) -> bool:
    """Stop homonym check only after </think> on KEEP_UNCHANGED or a complete fenced block."""
    if "<think>" in decoded and "</think>" not in decoded:
        return False
    post = (decoded.split("</think>", 1)[1] if "</think>" in decoded else decoded).lstrip()
    if not post:
        return False
    if "KEEP_UNCHANGED" in post:
        return True
    if "```" in post:
        return post.count("```") >= 2
    return False


_USER_OPEN_STR = "<|im_start|>user\n# Reference Memory:\n"


@torch.no_grad()
def _greedy_generate_code_until_fence(
    llm: torch.nn.Module,
    prefix_embeds: torch.Tensor,
    prompt_ids: torch.Tensor,
    tokenizer,
    max_new_tokens: int,
    stop_fn=None,
    user_open_ids: torch.Tensor | None = None,
) -> str:
    """KV-cached autoregressive greedy generation with <think>-aware early stopping."""
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

    eos_id = getattr(tokenizer, "eos_token_id", None)
    prev_use_cache = getattr(llm.config, "use_cache", False) if hasattr(llm, "config") else False
    if hasattr(llm, "config"):
        llm.config.use_cache = True

    active_stop_fn = _should_stop_code_gen if stop_fn is None else stop_fn
    gen_tokens: list[int] = []
    try:
        out = llm(inputs_embeds=inp_embeds, attention_mask=attn_mask, use_cache=True)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        pkv = getattr(out, "past_key_values", None)
        for step_idx in range(max_new_tokens):
            tid = int(next_tok[0, 0].item())
            if eos_id is not None and tid == int(eos_id):
                break
            gen_tokens.append(tid)
            if len(gen_tokens) % 4 == 0 or step_idx >= max_new_tokens - 1:
                decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                if active_stop_fn(decoded):
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

    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


def _extract_python_code_block(raw_text: str, orig_code: str) -> str:
    cleaned = _strip_think_blocks(raw_text)
    fenced = re.search(r"```(?:python)?\s*\n(.*?)```", cleaned, flags=re.DOTALL)
    if fenced:
        body = fenced.group(1).strip()
    elif re.match(r"^```(?:python)?\s*\n", cleaned):
        body = re.sub(r"^```(?:python)?\s*\n", "", cleaned, count=1).strip()
    elif "```" in cleaned:
        body = cleaned.split("```", 1)[0].strip()
    else:
        body = cleaned
    if not body:
        return orig_code
    return body + "\n"


def _build_edit_prompt(task: MigrationTask, target_path: str, raw_target_content: str) -> str:
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
        f"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n```python\n"
    )


def _build_homonym_prompt(task: MigrationTask, raw_homonym_content: str) -> str:
    return (
        f"\n\n# Task:\n{task.prompt}\n\n"
        f"# Current `{task.homonym_file}`:\n```python\n{raw_homonym_content.rstrip()}\n```\n\n"
        f"Should `{task.homonym_file}` be modified? Reply ONLY `KEEP_UNCHANGED` if it must remain unchanged, "
        f"or output the full updated file inside ```python ... ```."
        f"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _build_llm_auto_compact_memory(
    task: MigrationTask,
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    device: torch.device,
    *,
    summary_budget_tokens: int = 256,
    total_budget_tokens: int = 512,
) -> tuple[str, float]:
    """Generate a real LLM auto-compaction summary (<=256 tok) + recent sliding window (<=512 tok)."""
    sorted_files = sorted(task.initial_files.keys())
    sample_paths = ", ".join(sorted_files[:10])
    turns_block = "\n".join(f"- Turn {t['turn_id']}: {t['text']}" for t in task.session_turns)
    recent_turn = str(task.session_turns[-1]["text"])

    _sync_device(device)
    t0 = time.perf_counter()
    if isinstance(wrapper.llm, TinyMockCausalLM):
        neutral_summary = (
            f"Repo `{task.pkg_name}` has {len(sorted_files)} files (e.g. {sample_paths}). "
            f"Updated `{task.core_file}`:\n{task.core_new_content.strip()}\n"
            f"Notes:\n{task.session_notes_initial.strip()}\n{turns_block}"
        )
        sum_ids = tokenizer.encode(neutral_summary, add_special_tokens=False)[:summary_budget_tokens]
        summary_text = tokenizer.decode(sum_ids, skip_special_tokens=True)
    else:
        compact_prompt = (
            f"<|im_start|>user\n"
            f"# Context Compaction Request\n"
            f"Summarize the active API signatures, ratified caller defaults, rejected proposals, and homonym rules "
            f"from the following files and session notes concisely in bullet points.\n\n"
            f"Files ({len(sorted_files)} total): {sample_paths}, ...\n\n"
            f"# {task.core_file}\n{task.core_new_content.strip()}\n\n"
            f"# {task.session_notes_path}\n{task.session_notes_initial.strip()}\n"
            f"{turns_block}"
            f"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n- "
        )
        embed_fn = wrapper.llm.get_input_embeddings()
        d_llm = int(embed_fn.weight.shape[1])
        empty_pref = torch.empty((1, 0, d_llm), device=device, dtype=embed_fn.weight.dtype)
        c_ids = torch.tensor(tokenizer.encode(compact_prompt, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
        raw_summary = _greedy_generate_code_until_fence(
            wrapper.llm,
            empty_pref,
            c_ids,
            tokenizer,
            max_new_tokens=320,
            stop_fn=lambda s: "<think>" not in s and len(tokenizer.encode(_strip_think_blocks(s), add_special_tokens=False)) >= summary_budget_tokens,
        )
        cleaned_summary = _strip_think_blocks(raw_summary)
        sum_ids = tokenizer.encode(cleaned_summary, add_special_tokens=False)[:summary_budget_tokens]
        summary_text = tokenizer.decode(sum_ids, skip_special_tokens=True).strip()
        if not summary_text:
            summary_text = f"{task.core_file} updated; see recent session turn below."

    combined = (
        f"# /compact Summary (LLM Auto-Compaction):\n{summary_text}\n\n"
        f"# Recent Session Turn (Sliding Window):\n- {recent_turn}\n"
    )
    tot_ids = tokenizer.encode(combined, add_special_tokens=False)
    if len(tot_ids) > total_budget_tokens:
        combined = tokenizer.decode(tot_ids[:total_budget_tokens], skip_special_tokens=True)
    _sync_device(device)
    compact_ms = (time.perf_counter() - t0) * 1000.0
    return combined, compact_ms


def _evaluate_single_system_on_task(
    system_name: str,
    task: MigrationTask,
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    device: torch.device,
    *,
    page_size: int,
    salience_route_weight: float,
    lslc_k_skim: int,
    max_new_tokens: int,
    run_subprocess: bool,
    bge_toc: Any | None = None,
) -> dict[str, object]:
    """Run one system end-to-end on one MigrationTask under identical Hybrid Working Memory."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = tuple(
        sorted(
            {
                int(x)
                for x in (
                    getattr(tokenizer, "pad_token_id", None),
                    getattr(tokenizer, "eos_token_id", None),
                    getattr(tokenizer, "bos_token_id", None),
                    *(getattr(tokenizer, "all_special_ids", None) or ()),
                )
                if x is not None
            }
        )
    )
    eff_span_len = int(getattr(wrapper.cfg, "salience_span_len", 1))

    harness = CodingAgentHarness(task, use_initial_files=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    memory_update_ms = 0.0
    reencode_ms = 0.0
    delta_update_ms = 0.0
    max_other_page_diff = 0.0
    max_toc_diff_after_delta = 0.0
    acc_gates: list[float] = []
    rej_gates: list[float] = []
    lslc_bank_capture: dict[str, float] | None = None
    core_hits: list[float] = []
    session_hits: list[float] = []
    both_hits: list[float] = []
    prefix_token_counts: list[int] = []
    gen_ms_total = 0.0

    # Prepare repo memory representation prior to target edits
    pack = _build_repo_pages(task, tokenizer, page_size)
    page_paths: list[str] = list(pack["page_paths"])
    core_page_idx = int(pack["core_page_indices"][0])
    session_page_idx = int(pack["session_page_idx"])
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    core_fmt_new = f"# FILE: {task.core_file}\n{task.core_new_content}\n"

    # Build system-specific memory state (applying updated core.py + all session_turns without oracle filtering!)
    bm25: BM25PageRetriever | None = None
    bm25_docs: list[str] = []
    z_book = None
    exact_bank = None
    toc_embeds = None
    bge_pages = None
    fixed_prefix_embeds: torch.Tensor | None = None
    exact_bank_inspection: dict[str, object] | None = None

    if system_name == "oracle_raw_context":
        _ = harness.read_file(task.core_file)
        acc_text = next(str(t["text"]) for t in task.session_turns if bool(t["accepted"]))
        oracle_txt = (
            f"{core_fmt_new}\n"
            f"# Approved Architecture Session Decision:\n{acc_text}\n"
        )
        o_ids = torch.tensor(tokenizer.encode(oracle_txt, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
        fixed_prefix_embeds = embed_fn(o_ids).to(llm_dtype)
        # Apply core.py migration on disk inside harness as well
        harness.apply_file_edit(task.core_file, task.core_new_content)
    elif system_name.startswith("bm25_rag_k"):
        harness.apply_file_edit(task.core_file, task.core_new_content)
        bm25_docs = list(pack["page_texts"])

        t_up0 = time.perf_counter()
        bm25 = BM25PageRetriever(bm25_docs)
        bm25_docs[core_page_idx] = core_fmt_new
        bm25.update_page(core_page_idx, core_fmt_new)

        sess_txt = bm25_docs[session_page_idx]
        for turn in task.session_turns:
            turn_txt = str(turn["text"])
            sess_txt = sess_txt.rstrip() + "\n" + turn_txt + "\n"
        bm25_docs[session_page_idx] = sess_txt
        bm25.update_page(session_page_idx, sess_txt)
        memory_update_ms = (time.perf_counter() - t_up0) * 1000.0
    elif system_name == "llm_auto_compact":
        harness.apply_file_edit(task.core_file, task.core_new_content)
        compact_txt, compact_ms = _build_llm_auto_compact_memory(
            task,
            wrapper,
            tokenizer,
            device,
            summary_budget_tokens=page_size,
            total_budget_tokens=page_size * 2,
        )
        c_ids = torch.tensor(tokenizer.encode(compact_txt, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
        fixed_prefix_embeds = embed_fn(c_ids).to(llm_dtype)
        memory_update_ms = compact_ms
    elif system_name.startswith("lslc_"):
        harness.apply_file_edit(task.core_file, task.core_new_content)
        wrapper.lslc.eval()
        page_tensor = pack["page_tensor"].to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                z_chunks, toc_chunks, tok_chunks, h_chunks, w_chunks = [], [], [], [], []
                for s in range(0, page_tensor.shape[1], 32):
                    e = min(page_tensor.shape[1], s + 32)
                    zc, bc, tc = wrapper.lslc.encode_book(
                        page_tensor[:, s:e],
                        embed_fn,
                        use_exact_bank=True,
                        vocab_size=vocab_size,
                        special_token_ids=special_ids if eff_span_len > 1 else None,
                        salience_span_len=eff_span_len,
                    )
                    z_chunks.append(zc)
                    toc_chunks.append(tc)
                    tok_chunks.append(bc.token_ids)
                    h_chunks.append(bc.h_coords)
                    w_chunks.append(bc.w_coords)
                from lslc.encoder import SparseExactBank

                z_book = torch.cat(z_chunks, dim=1)
                toc_embeds = torch.cat(toc_chunks, dim=1)
                if bge_toc is not None:
                    bge_pages = bge_toc.encode_book_pages(page_tensor, None, tokenizer, num_spans=4, batch_size=64)
                num_p = page_tensor.shape[1]
                p_ids = (
                    torch.arange(num_p, device=device, dtype=torch.int16)
                    .view(1, num_p, 1)
                    .expand(1, num_p, wrapper.cfg.k_exact)
                    .contiguous()
                )
                exact_bank = SparseExactBank(
                    token_ids=torch.cat(tok_chunks, dim=1),
                    page_ids=p_ids,
                    h_coords=torch.cat(h_chunks, dim=1),
                    w_coords=torch.cat(w_chunks, dim=1),
                )

            # 1. Modular single-page re-encode on updated core.py
            z_before_reenc = z_book.clone()
            core_toks_new = _pad_or_chunk_tokens(
                tokenizer.encode(core_fmt_new, add_special_tokens=False), page_size, pad_id
            )[0]
            core_page_t = torch.tensor(core_toks_new, dtype=torch.long, device=device).unsqueeze(0)
            _sync_device(device)
            t_re0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                z_book, exact_bank, toc_embeds = wrapper.lslc.reencode_page(
                    z_book,
                    toc_embeds,
                    exact_bank,
                    page_idx=core_page_idx,
                    new_page_token_ids=core_page_t,
                    embed_tokens_fn=embed_fn,
                    vocab_size=vocab_size,
                    in_place=True,
                    special_token_ids=special_ids if eff_span_len > 1 else None,
                    salience_span_len=eff_span_len,
                )
                if bge_toc is not None and bge_pages is not None:
                    bge_core = bge_toc.encode_book_pages(core_page_t.unsqueeze(1), None, tokenizer, num_spans=4, batch_size=16)
                    bge_pages[:, core_page_idx : core_page_idx + 1] = bge_core
            _sync_device(device)
            reencode_ms = (time.perf_counter() - t_re0) * 1000.0
            other_mask = [p for p in range(z_book.shape[1]) if p != core_page_idx]
            if other_mask:
                max_other_page_diff = float(
                    (z_book[:, other_mask].float() - z_before_reenc[:, other_mask].float()).abs().max().item()
                )

            # 2. Sequential DeltaUpdater on session_notes page (ZERO oracle accepted flag!)
            toc_before_delta = toc_embeds.clone()
            sess_p_tensor = torch.tensor([session_page_idx], dtype=torch.long, device=device)
            _sync_device(device)
            t_d0 = time.perf_counter()
            for turn in task.session_turns:
                edit_toks = torch.tensor(
                    tokenizer.encode(str(turn["text"]), add_special_tokens=False),
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    override = (
                        extract_contextual_edit_embeds(
                            wrapper.llm, edit_toks, layer_idx=wrapper.cfg.contextual_edit_layer
                        )
                        if wrapper.cfg.use_contextual_edit_encoding
                        else None
                    )
                    z_book, gates, exact_bank = wrapper.lslc.apply_page_delta(
                        z_book,
                        sess_p_tensor,
                        edit_toks,
                        embed_tokens_fn=embed_fn,
                        exact_bank=exact_bank,
                        vocab_size=vocab_size,
                        edit_embeds_override=override,
                        in_place=True,
                        bank_write_threshold=0.45,
                    )
                strip_gate = float(gates[:, -1, :].mean().item())
                if bool(turn["accepted"]):
                    acc_gates.append(strip_gate)
                else:
                    rej_gates.append(strip_gate)
            _sync_device(device)
            delta_update_ms = (time.perf_counter() - t_d0) * 1000.0
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                toc_after_delta = wrapper.lslc.projector.project_toc(z_book)
            max_toc_diff_after_delta = float((toc_after_delta.float() - toc_before_delta.float()).abs().max().item())
            toc_embeds = toc_after_delta
            memory_update_ms = reencode_ms + delta_update_ms
            lslc_bank_capture = measure_code_exact_bank_capture(
                exact_bank, core_page_idx, session_page_idx, task, tokenizer
            )
            if task.task_id == "eval_01_load_config":
                exact_bank_inspection = {
                    "core_page_idx": core_page_idx,
                    "session_page_idx": session_page_idx,
                    "core_page_post_reencode": _inspect_exact_bank_page(exact_bank.token_ids[0, core_page_idx], tokenizer),
                    "session_page_post_delta_live": _inspect_exact_bank_page(
                        exact_bank.token_ids[0, session_page_idx], tokenizer
                    ),
                    "session_page_base_routing": (
                        None
                        if exact_bank.base_token_ids is None
                        else _inspect_exact_bank_page(exact_bank.base_token_ids[0, session_page_idx], tokenizer)
                    ),
                }
    else:
        raise ValueError(f"Unknown Gerbang 3 system_name: {system_name!r}")

    def _get_memory_prefix_for_target(target_path: str) -> torch.Tensor:
        excluded = {i for i, p in enumerate(page_paths) if p == target_path}
        q_txt = _build_router_query(task, target_path)
        if system_name in ("oracle_raw_context", "llm_auto_compact"):
            assert fixed_prefix_embeds is not None
            return fixed_prefix_embeds
        if system_name.startswith("bm25_rag_k"):
            assert bm25 is not None
            k_val = int(system_name.rsplit("_k", 1)[-1])
            harness.search_calls += 1
            chosen = _bm25_route_excluding(bm25, q_txt, k_deep=k_val, excluded_indices=excluded)
            read_paths_seen_turn: set[str] = set()
            for idx in chosen:
                p_name = page_paths[idx]
                if p_name != task.session_notes_path and p_name not in read_paths_seen_turn and p_name in harness.files:
                    _ = harness.read_file(p_name)
                    read_paths_seen_turn.add(p_name)
            chosen_set = set(chosen)
            core_hits.append(float(core_page_idx in chosen_set))
            session_hits.append(float(session_page_idx in chosen_set))
            both_hits.append(float((core_page_idx in chosen_set) and (session_page_idx in chosen_set)))
            rag_text = "\n\n".join(bm25_docs[i].rstrip() for i in chosen) + "\n"
            toks = torch.tensor(tokenizer.encode(rag_text, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
            return embed_fn(toks).to(llm_dtype)

        # LSLC v3 Hybrid Router (0 BM25 CPU)
        k_val, ks_val, asm_mode, attn_mode, flat_toc_cap = _parse_lslc_arm_config(system_name, lslc_k_skim)
        rq_ids = torch.tensor(tokenizer.encode(q_txt, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
        ext_logits = (
            bge_toc.score_book(bge_toc.encode_texts([q_txt], is_query=True), bge_pages)
            if (bge_toc is not None and bge_pages is not None)
            else None
        )
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                r_out, pref = _lslc_route_and_assemble_excluding(
                    wrapper=wrapper,
                    z_book=z_book,
                    toc_embeds=toc_embeds,
                    exact_bank=exact_bank,
                    embed_fn=embed_fn,
                    router_query_ids=rq_ids,
                    k_deep=k_val,
                    k_skim=ks_val,
                    salience_route_weight=salience_route_weight,
                    excluded_indices=excluded,
                    mode=asm_mode,
                    exact_bank_attenuation=attn_mode,
                    flat_toc_max_tokens=flat_toc_cap,
                    external_dense_logits=ext_logits,
                )
        chosen_set = set(int(x) for x in r_out.deep_idx[0].tolist())
        core_hits.append(float(core_page_idx in chosen_set))
        session_hits.append(float(session_page_idx in chosen_set))
        both_hits.append(float((core_page_idx in chosen_set) and (session_page_idx in chosen_set)))
        return pref.to(llm_dtype)

    # Execute edits over target caller files + homonym preservation check
    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)
    for target_path in task.caller_files:
        raw_target = harness.read_file(target_path)
        mem_pref = _get_memory_prefix_for_target(target_path)
        prefix_token_counts.append(int(mem_pref.shape[1]))
        prompt_str = _build_edit_prompt(task, target_path, raw_target)
        p_ids = torch.tensor(tokenizer.encode(prompt_str, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
        _sync_device(device)
        tg0 = time.perf_counter()
        raw_gen = _greedy_generate_code_until_fence(
            wrapper.llm,
            mem_pref,
            p_ids,
            tokenizer,
            max_new_tokens=max_new_tokens,
            user_open_ids=user_open_ids,
        )
        _sync_device(device)
        gen_ms_total += (time.perf_counter() - tg0) * 1000.0
        updated_src = _extract_python_code_block(raw_gen, raw_target)
        harness.apply_file_edit(target_path, updated_src)

    # Homonym check on legacy_compat.py
    raw_homonym = harness.read_file(task.homonym_file)
    mem_pref_h = _get_memory_prefix_for_target(task.homonym_file)
    prefix_token_counts.append(int(mem_pref_h.shape[1]))
    h_prompt = _build_homonym_prompt(task, raw_homonym)
    h_ids = torch.tensor(tokenizer.encode(h_prompt, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
    _sync_device(device)
    tg0 = time.perf_counter()
    raw_h_gen = _greedy_generate_code_until_fence(
        wrapper.llm,
        mem_pref_h,
        h_ids,
        tokenizer,
        max_new_tokens=min(256, max_new_tokens),
        stop_fn=_should_stop_homonym_gen,
        user_open_ids=user_open_ids,
    )
    _sync_device(device)
    gen_ms_total += (time.perf_counter() - tg0) * 1000.0
    cleaned_h_gen = _strip_think_blocks(raw_h_gen)
    if "KEEP_UNCHANGED" not in cleaned_h_gen and "```" in cleaned_h_gen:
        cand_h = _extract_python_code_block(cleaned_h_gen, raw_homonym)
        if f"def {task.symbol_name}" in cand_h:
            harness.apply_file_edit(task.homonym_file, cand_h)

    harness.memory_update_ms += memory_update_ms
    harness.active_prefix_tokens = round(sum(prefix_token_counts) / max(1, len(prefix_token_counts)))
    harness.peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2), 2)
        if device.type == "cuda"
        else 0.0
    )

    eval_res: MigrationEvalResult = harness.evaluate(run_subprocess=run_subprocess)
    failure_debug = (
        None
        if eval_res.patch_correct
        else {p: harness.files[p][:1200] for p in task.caller_files if p in harness.files}
    )
    return {
        "task_id": eval_res.task_id,
        "patch_correct": bool(eval_res.patch_correct),
        "hidden_test_passed": bool(eval_res.hidden_test_passed),
        "stale_signature_used": bool(eval_res.stale_signature_used),
        "decoy_value_used": bool(eval_res.decoy_value_used),
        "homonym_preserved": bool(eval_res.homonym_preserved),
        "failure_reason": eval_res.failure_reason,
        "failure_debug": failure_debug,
        "files_read_total": eval_res.files_read_total,
        "repeated_file_reads": eval_res.repeated_file_reads,
        "chars_read_total": eval_res.chars_read_total,
        "search_calls": eval_res.search_calls,
        "active_prefix_tokens": eval_res.active_prefix_tokens,
        "memory_update_ms": round(eval_res.memory_update_ms, 3),
        "modular_reencode_ms": round(reencode_ms, 3),
        "delta_update_ms": round(delta_update_ms, 3),
        "generation_ms": round(gen_ms_total, 3),
        "wall_clock_ms": round(eval_res.wall_clock_ms, 3),
        "peak_vram_mb": eval_res.peak_vram_mb,
        "router_core_hit_rate": round(sum(core_hits) / max(1, len(core_hits)), 4) if core_hits else None,
        "router_session_hit_rate": round(sum(session_hits) / max(1, len(session_hits)), 4) if session_hits else None,
        "router_both_hit_rate": round(sum(both_hits) / max(1, len(both_hits)), 4) if both_hits else None,
        "delta_accepted_gate_mean": round(sum(acc_gates) / len(acc_gates), 4) if acc_gates else None,
        "delta_rejected_gate_mean": round(sum(rej_gates) / len(rej_gates), 4) if rej_gates else None,
        "max_other_page_diff_after_reencode": round(max_other_page_diff, 6),
        "max_toc_diff_after_delta": round(max_toc_diff_after_delta, 6),
        "lslc_bank_capture": lslc_bank_capture,
        "exact_bank_inspection": exact_bank_inspection,
    }


def _aggregate_system_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    n = max(1, len(rows))

    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def _optional_mean(key: str) -> float | None:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    out: dict[str, object] = {
        "num_tasks": len(rows),
        "pass_at_1": round(sum(1 for r in rows if r["patch_correct"]) / n, 4),
        "hidden_test_pass_rate": round(sum(1 for r in rows if r["hidden_test_passed"]) / n, 4),
        "stale_signature_used_rate": round(sum(1 for r in rows if r["stale_signature_used"]) / n, 4),
        "decoy_value_used_rate": round(sum(1 for r in rows if r["decoy_value_used"]) / n, 4),
        "homonym_preserved_rate": round(sum(1 for r in rows if r["homonym_preserved"]) / n, 4),
        "files_read_mean": _mean("files_read_total"),
        "repeated_file_reads_mean": _mean("repeated_file_reads"),
        "search_calls_mean": _mean("search_calls"),
        "active_prefix_tokens_mean": _mean("active_prefix_tokens"),
        "memory_update_ms_mean": _mean("memory_update_ms"),
        "modular_reencode_ms_mean": _mean("modular_reencode_ms"),
        "delta_update_ms_mean": _mean("delta_update_ms"),
        "generation_ms_mean": _mean("generation_ms"),
        "wall_clock_ms_mean": _mean("wall_clock_ms"),
        "peak_vram_mb_max": round(max((float(r["peak_vram_mb"]) for r in rows), default=0.0), 2),
    }
    for opt_key in (
        "router_core_hit_rate",
        "router_session_hit_rate",
        "router_both_hit_rate",
        "delta_accepted_gate_mean",
        "delta_rejected_gate_mean",
    ):
        val = _optional_mean(opt_key)
        if val is not None:
            out[opt_key] = val

    if any(r.get("lslc_bank_capture") is not None for r in rows):
        caps = [r["lslc_bank_capture"] for r in rows if isinstance(r.get("lslc_bank_capture"), dict)]
        out["in_exact_bank_symbol_rate"] = round(sum(float(c["in_exact_bank_symbol"]) for c in caps) / len(caps), 4)
        out["in_exact_bank_kwarg_rate"] = round(sum(float(c["in_exact_bank_kwarg"]) for c in caps) / len(caps), 4)
        out["in_exact_bank_default_rate"] = round(sum(float(c["in_exact_bank_default"]) for c in caps) / len(caps), 4)
        out["decoy_in_exact_bank_rate"] = round(sum(float(c["decoy_in_exact_bank"]) for c in caps) / len(caps), 4)
        out["max_other_page_diff_after_reencode"] = round(
            max(float(r["max_other_page_diff_after_reencode"]) for r in rows), 6
        )
        out["max_toc_diff_after_delta"] = round(
            max(float(r["max_toc_diff_after_delta"]) for r in rows), 6
        )

    out["tasks"] = rows
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gerbang 3 CrossFileAPIMigrationBenchmark Runner.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--mock-llm", action="store_true", help="Run CPU smoke test with TinyMockCausalLM.")
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--save-checkpoint", type=str, default="")
    parser.add_argument("--num-distractors", type=int, default=32, help="32->38 files (~8-10K tok), 58->64 files (~14-16K tok).")
    parser.add_argument("--k-exact", type=int, default=24)
    parser.add_argument("--salience-span-len", type=int, default=1)
    parser.add_argument("--use-bge-toc", action="store_true", default=False)
    parser.add_argument("--bge-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--lslc-k-deep", type=str, default="2,4")
    parser.add_argument("--lslc-k-skim", type=int, default=2)
    parser.add_argument("--salience-route-weight", type=float, default=2.0)
    parser.add_argument("--finetune-code-salience-steps", type=int, default=0)
    parser.add_argument("--align-stage15", action="store_true", help="Run Stage-1.5 SalienceGate (A) + Projector ChatML/Code (B) alignment.")
    parser.add_argument("--stage-a-steps", type=int, default=100)
    parser.add_argument("--stage-a-lr", type=float, default=2e-4)
    parser.add_argument("--stage-b-steps", type=int, default=180)
    parser.add_argument("--stage-b-lr", type=float, default=1e-4)
    parser.add_argument("--run-g12-regression", action="store_true", help="Run quick Gerbang 1 & 2 regression trace on eval_symbolic.")
    parser.add_argument("--g12-max-episodes", type=int, default=24)
    parser.add_argument("--oracle-sanity-only", action="store_true")
    parser.add_argument("--diag-probe", action="store_true", help="Run 6-arm diagnostic probe (fixed BM25 + LSLC nosquash ablations).")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-json", type=str, default="/content/lslc_results/v3_gerbang3_coding.json")
    args, _ = parser.parse_known_args(argv)
    return args


def _build_replay_pools(tokenizer, vocab_size: int, is_mock: bool) -> tuple[list[int], list[int], str]:
    if is_mock:
        train_pool = [2 + (i % (vocab_size - 2)) for i in range(4096)]
        val_pool = [2 + ((i * 7) % (vocab_size - 2)) for i in range(2048)]
        return train_pool, val_pool, "mock-synthetic-pool"
    return _load_wikitext_pools(tokenizer)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    CrossFileAPIMigrationBenchmark.assert_disjoint_splits()

    if args.mock_llm:
        vocab_size = 2048
        cfg = LSLCConfig.tiny_mock(
            use_contextual_edit_encoding=True,
            contextual_edit_layer=1,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=min(args.k_exact, 48),
            salience_span_len=args.salience_span_len,
        )
        llm = TinyMockCausalLM(vocab_size=vocab_size, d_llm=cfg.d_llm, num_layers=cfg.num_llm_layers)
        tokenizer = _MockCodeTokenizer(vocab_size=vocab_size)
        device = torch.device("cpu")
    else:
        cfg = LSLCConfig.from_model_name(
            args.model,
            use_contextual_edit_encoding=True,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=args.k_exact,
            salience_span_len=args.salience_span_len,
        )
        cfg.contextual_edit_layer = cfg.num_llm_layers // 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        llm, tokenizer = load_frozen_qwen3(cfg, load_in_4bit=False, device=device)
        vocab_size = len(llm.get_input_embeddings().weight)

    wrapper = FrozenQwenLSLCWrapper(cfg, llm).to(device)
    ckpt_info = _load_lslc_checkpoint(wrapper.lslc, args.init_checkpoint) if args.init_checkpoint else None
    bge_toc = None
    if args.use_bge_toc:
        from colab_run_v3_gerbang4_community import _PlugInBiEncoderTOC

        bge_toc = _PlugInBiEncoderTOC(args.bge_model, device=device, is_mock=args.mock_llm)

    train_pool: list[int] | None = None
    val_pool: list[int] | None = None
    data_source: str | None = None
    if args.align_stage15 or args.run_g12_regression:
        train_pool, val_pool, data_source = _build_replay_pools(tokenizer, vocab_size, args.mock_llm)

    stage15_stats: dict[str, object] | None = None
    if args.align_stage15:
        assert train_pool is not None
        stage_a_stats = _run_stage_a_salience_alignment(
            wrapper=wrapper,
            tokenizer=tokenizer,
            train_pool=train_pool,
            num_distractors=args.num_distractors,
            steps=args.stage_a_steps,
            lr=args.stage_a_lr,
            device=device,
            seed=args.seed + 17,
        )
        stage_b_stats = _run_stage_b_projector_alignment(
            wrapper=wrapper,
            tokenizer=tokenizer,
            train_pool=train_pool,
            num_distractors=args.num_distractors,
            steps=args.stage_b_steps,
            lr=args.stage_b_lr,
            lslc_k_skim=args.lslc_k_skim,
            salience_route_weight=args.salience_route_weight,
            device=device,
            seed=args.seed + 29,
        )
        stage15_stats = {
            "data_source": data_source,
            "stage_a_salience": stage_a_stats,
            "stage_b_projector": stage_b_stats,
        }
        finetune_stats: dict[str, object] = {"steps": 0, "note": "superseded_by_stage15"}
    else:
        finetune_stats = _finetune_salience_on_train_code(
            wrapper=wrapper,
            tokenizer=tokenizer,
            num_distractors=args.num_distractors,
            steps=args.finetune_code_salience_steps,
            device=device,
            seed=args.seed + 9,
        )

    if args.save_checkpoint:
        _save_lslc_checkpoint(wrapper.lslc, args.save_checkpoint)

    eval_bench = CrossFileAPIMigrationBenchmark(split="eval", num_distractor_files=args.num_distractors)
    repo_packs = [_build_repo_pages(t, tokenizer, cfg.page_size) for t in eval_bench]
    repo_stats = {
        "num_eval_tasks": len(eval_bench),
        "num_distractors": args.num_distractors,
        "num_files_per_repo": int(repo_packs[0]["num_files"]),
        "num_pages_mean": round(sum(int(p["num_pages"]) for p in repo_packs) / len(repo_packs), 2),
        "content_tokens_mean": round(sum(int(p["total_content_tokens"]) for p in repo_packs) / len(repo_packs), 1),
        "padded_tokens_mean": round(sum(int(p["total_padded_tokens"]) for p in repo_packs) / len(repo_packs), 1),
    }
    print(f"[REPO STATS] {repo_stats}", flush=True)

    k_deep_list = [max(1, int(x.strip())) for x in args.lslc_k_deep.split(",") if x.strip()]
    if args.oracle_sanity_only:
        systems = ["oracle_raw_context"]
    elif args.diag_probe:
        systems = [
            "bm25_rag_k2",
            "bm25_rag_k8",
            "lslc_exact_only_nosquash_k2",
            "lslc_exact_only_nosquash_k4",
            "lslc_full_nosquash_compact_toc_k2",
            "lslc_full_nosquash_compact_toc_k4",
        ]
    elif args.align_stage15:
        systems = [
            "lslc_exact_only_nosquash_k2",
            "lslc_full_nosquash_compact_toc_k2",
            "lslc_full_nosquash_compact_toc_k4",
            "oracle_raw_context",
            "bm25_rag_k2",
            "bm25_rag_k8",
            "llm_auto_compact",
        ]
    else:
        systems = [
            "oracle_raw_context",
            "bm25_rag_k2",
            "bm25_rag_k8",
            "llm_auto_compact",
        ] + [f"lslc_v3_hybrid_k{kd}" for kd in k_deep_list]

    results: dict[str, object] = {}
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": "3.0.0-gerbang3-coding",
        "model": "TinyMockCausalLM" if args.mock_llm else args.model,
        "mock_llm": args.mock_llm,
        "device": str(device),
        "oracle_sanity_only": args.oracle_sanity_only,
        "diag_probe": args.diag_probe,
        "align_stage15": args.align_stage15,
        "init_checkpoint": ckpt_info,
        "save_checkpoint": args.save_checkpoint or None,
        "stage15_alignment": stage15_stats,
        "finetune_code_salience": finetune_stats,
        "config": {
            "page_size": cfg.page_size,
            "k_exact": cfg.k_exact,
            "salience_span_len": cfg.salience_span_len,
            "use_bge_toc": bool(bge_toc is not None),
            "bge_model": None if bge_toc is None else bge_toc.model_name,
            "lslc_k_deep": k_deep_list,
            "lslc_k_skim": args.lslc_k_skim,
            "salience_route_weight": args.salience_route_weight,
            "contextual_edit_layer": cfg.contextual_edit_layer,
            "router_mode": cfg.router_mode,
            "salience_mode": cfg.salience_mode,
            "delta_pooling_mode": cfg.delta_pooling_mode,
            "max_new_tokens": args.max_new_tokens,
        },
        "repo_stats": repo_stats,
        "eval_01_exact_bank_inspection": None,
        "gerbang12_regression": None,
        "systems": results,
    }

    for sys_name in systems:
        rows: list[dict[str, object]] = []
        for task in eval_bench:
            row = _evaluate_single_system_on_task(
                system_name=sys_name,
                task=task,
                wrapper=wrapper,
                tokenizer=tokenizer,
                device=device,
                page_size=cfg.page_size,
                salience_route_weight=args.salience_route_weight,
                lslc_k_skim=args.lslc_k_skim,
                max_new_tokens=args.max_new_tokens,
                run_subprocess=True,
                bge_toc=bge_toc,
            )
            if payload["eval_01_exact_bank_inspection"] is None and row.get("exact_bank_inspection") is not None:
                payload["eval_01_exact_bank_inspection"] = row["exact_bank_inspection"]
                insp = row["exact_bank_inspection"]
                print(
                    f"[EVAL_01 EXACT BANK] core_p={insp['core_page_idx']} pieces={insp['core_page_post_reencode']['token_pieces']} | "
                    f"sess_p={insp['session_page_idx']} pieces={insp['session_page_post_delta_live']['token_pieces']}",
                    flush=True,
                )
            row.pop("exact_bank_inspection", None)
            rows.append(row)
            if args.oracle_sanity_only:
                print(
                    f"[ORACLE TASK] {row['task_id']} pass={row['patch_correct']} reason={row['failure_reason']!r}",
                    flush=True,
                )
        summary = _aggregate_system_rows(rows)
        results[sys_name] = summary
        compact_log = {k: v for k, v in summary.items() if k != "tasks"}
        print(f"[{sys_name}] {compact_log}", flush=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.run_g12_regression:
        assert val_pool is not None
        g12_num_pages = 8 if args.mock_llm else 16
        g12_eps_n = min(args.g12_max_episodes, 8) if args.mock_llm else args.g12_max_episodes
        eval_symbolic_g12 = list(
            AdversarialLivingBookDataset(
                num_episodes=max(g12_eps_n, 1),
                num_pages=g12_num_pages,
                page_size=cfg.page_size,
                num_edits=5,
                mode="symbolic",
                split="eval",
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                seed=args.seed + 41,
                background_token_pool=val_pool,
            )
        )
        g12_trace = trace_identifier_pipeline(
            wrapper,
            tokenizer,
            eval_symbolic_g12,
            device,
            max_episodes=g12_eps_n,
            salience_route_weight=0.0,
            modes=(
                "cond_b_natural_router",
                "cond_b_hybrid_router",
                "cond_c_oracle",
                "cond_c_hybrid_router",
            ),
        )
        payload["gerbang12_regression"] = g12_trace
        print(
            "[G12 REGRESSION]",
            {
                k: {
                    m: v[m]
                    for m in ("in_exact_bank_rate", "router_top1_recall", "router_hit_rate", "in_generated_answer_rate")
                }
                for k, v in g12_trace.items()
            },
            flush=True,
        )

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote Gerbang 3 results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
