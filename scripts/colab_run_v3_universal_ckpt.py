#!/usr/bin/env python3
"""Single Universal Checkpoint Trainer for LSLC v3 (Frozen Mode A r=0).

Trains ONE universal checkpoint (`lslc_universal_qwen3_1.7b.safetensors`) with uniform config
(`k_exact=48`, `salience_span_len=16`, `router_mode="content_attn"`, `delta_pooling_mode="cross_attn"`,
`exact_bank_attenuation="none"`, Frozen LLM `r=0`) in a single mixed multi-task training loop over
4 strictly disjoint general-domain data mixtures:
  1. 30% General Prose & State-Revision Replay (WikiText-2/103 + plain/Stage-1-2 state revisions;
     0 Gate 2 hold-out entities/templates/values).
  2. 25% General Contiguous Span-16 Factual/Numeric Clauses for SalienceGate
     (0 RULER wording/templates/entities).
  3. 25% General WikiText Self-Supervised ChatML QA & General Multi-Turn Dialogue
     (0 LongMemEval templates/sessions/questions).
  4. 20% Natural Python Code from ungated HuggingFace public corpora (`Nan-Do/code-search-net-python`,
     fallback `bigcode/the-stack-smol` `data/python`, fallback `mbpp`; 0 Gate 3 eval repo identifiers/files).

Zero per-benchmark micro-alignment, zero new dynamic gating between continuous & discrete branches.
"""

from __future__ import annotations

import argparse
import hashlib
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
from colab_run_v3_gerbang3_coding import (  # noqa: E402
    _MockCodeTokenizer,
    _load_lslc_checkpoint,
    _pad_or_chunk_tokens,
)
from colab_run_v3_gerbang4_community import (  # noqa: E402
    _USER_OPEN_STR,
    _PlugInBiEncoderTOC,
    _build_wikitext_selfsup_qa_episodes,
    _encode_book_chunked,
    _format_qa_prompt,
    _get_special_token_ids,
)
from colab_run_v3_phase1 import _save_lslc_checkpoint  # noqa: E402
from lslc import (  # noqa: E402
    AdversarialLivingBookDataset,
    CrossFileAPIMigrationBenchmark,
    FrozenQwenLSLCWrapper,
    LSLCConfig,
    RouterOutput,
    SparseExactBank,
    TinyMockCausalLM,
    extract_contextual_edit_embeds,
    load_frozen_qwen3,
    select_amp_dtype,
)
from lslc.coding_benchmark import _EVAL_SPECS as _GATE3_EVAL_SPECS  # noqa: E402
from lslc.datasets import _encode_text_or_mock, _sample_background_page  # noqa: E402


# ---------------------------------------------------------------------------
# Hard Disjointness Guards Across ALL 5 Evaluation Benchmarks
# ---------------------------------------------------------------------------


def collect_eval_forbidden_terms() -> dict[str, set[str]]:
    """Collect exact forbidden eval tokens/substrings across Gate 2, Gate 3, RULER, and LongMemEval."""
    gate2_terms: set[str] = set()
    for p in AdversarialLivingBookDataset.EVAL_PROJECTS:
        gate2_terms.add(p.lower())
    for b in AdversarialLivingBookDataset.EVAL_BUDGET_POOL:
        gate2_terms.add(b.lower())
    for s in AdversarialLivingBookDataset.EVAL_SYMBOLIC_STATES:
        gate2_terms.add(s.lower())
    for _, _, ans in AdversarialLivingBookDataset.EVAL_STATIC_POLICIES:
        gate2_terms.add(ans.lower())

    gate3_terms: set[str] = set()
    for spec in _GATE3_EVAL_SPECS:
        gate3_terms.add(spec.pkg_name.lower())
        gate3_terms.add(spec.symbol_name.lower())
        gate3_terms.add(spec.task_id.lower())
    # Also dynamically pull eval repo filenames/paths (original 6 + extended 18 Gate 5 N=24 specs)
    eval_g3 = CrossFileAPIMigrationBenchmark(split="eval", num_distractor_files=0)
    for t in eval_g3:
        gate3_terms.add(t.pkg_name.lower())
        gate3_terms.add(t.symbol_name.lower())
        if t.core_file:
            gate3_terms.add(t.core_file.lower())
    try:
        from colab_run_v3_gerbang5_ablations import _EVAL_SPECS_EXT_18

        for spec in _EVAL_SPECS_EXT_18:
            gate3_terms.add(spec.pkg_name.lower())
            gate3_terms.add(spec.symbol_name.lower())
            gate3_terms.add(spec.task_id.lower())
    except Exception:
        pass

    ruler_terms: set[str] = {
        "special magic number",
        "special magic numbers",
        "assign var_",
        "ember-node-",
        "agate-node-",
        "cinder-node-",
        "delta-node-",
        "stressnode-",
    }

    longmemeval_terms: set[str] = {
        "longmemeval",
        "answer_session_ids",
        "haystack_sessions",
        "comandante c40",
        "wells fargo",
        "university of melbourne",
        "great ocean road",
    }

    return {
        "gate2_holdout": gate2_terms,
        "gate3_eval_repo": gate3_terms,
        "ruler_eval": ruler_terms,
        "longmemeval_eval": longmemeval_terms,
    }


def _text_hits_forbidden(text: str, forbidden: dict[str, set[str]], *, check_gate2_short: bool = True) -> tuple[str, str] | None:
    low = text.lower()
    words = set(re.findall(r"[a-z0-9_-]+", low))
    for group, terms in forbidden.items():
        for term in terms:
            if not term:
                continue
            if group == "gate2_holdout" and not check_gate2_short and len(term) <= 4:
                continue
            if " " in term or "-" in term or "/" in term or "_" in term:
                if term in low:
                    return group, term
            else:
                if term in words:
                    return group, term
    return None


def assert_universal_train_disjoint(
    texts_by_domain: dict[str, list[str]],
    forbidden: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Hard-assert zero eval leakage across every sample in all 4 universal training domains."""
    forb = collect_eval_forbidden_terms() if forbidden is None else forbidden
    checked_counts: dict[str, int] = {}
    for domain, texts in texts_by_domain.items():
        checked_counts[domain] = len(texts)
        check_short_g2 = domain != "natural_python_code"
        for idx, txt in enumerate(texts):
            hit = _text_hits_forbidden(txt, forb, check_gate2_short=check_short_g2)
            assert hit is None, (
                f"Universal training disjointness violation in domain={domain!r} item={idx}: "
                f"matched forbidden group={hit[0]!r} term={hit[1]!r} in text={txt[:200]!r}"
            )
    return {
        "verified": True,
        "checked_items_by_domain": checked_counts,
        "forbidden_term_counts": {k: len(v) for k, v in forb.items()},
    }


# ---------------------------------------------------------------------------
# 4-Domain General-Purpose Disjoint Data Builders
# ---------------------------------------------------------------------------

_MOCK_PYTHON_SNIPPETS: tuple[str, ...] = (
    (
        "def compute_moving_mean(values: list[float], window: int) -> list[float]:\n"
        "    if window <= 0:\n"
        "        raise ValueError('window must be positive')\n"
        "    out = []\n"
        "    running = 0.0\n"
        "    for idx, val in enumerate(values):\n"
        "        running += val\n"
        "        if idx >= window:\n"
        "            running -= values[idx - window]\n"
        "        if idx + 1 >= window:\n"
        "            out.append(running / float(window))\n"
        "    return out\n"
    ),
    (
        "def parse_key_pairs(raw_lines: list[str], delimiter: str = '=') -> dict[str, str]:\n"
        "    parsed: dict[str, str] = {}\n"
        "    for line in raw_lines:\n"
        "        cleaned = line.strip()\n"
        "        if not cleaned or cleaned.startswith('#'):\n"
        "            continue\n"
        "        if delimiter in cleaned:\n"
        "            left, right = cleaned.split(delimiter, 1)\n"
        "            parsed[left.strip()] = right.strip()\n"
        "    return parsed\n"
    ),
    (
        "def merge_sorted_intervals(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:\n"
        "    if not spans:\n"
        "        return []\n"
        "    ordered = sorted(spans)\n"
        "    merged = [ordered[0]]\n"
        "    for start, end in ordered[1:]:\n"
        "        prev_start, prev_end = merged[-1]\n"
        "        if start <= prev_end:\n"
        "            merged[-1] = (prev_start, max(prev_end, end))\n"
        "        else:\n"
        "            merged.append((start, end))\n"
        "    return merged\n"
    ),
    (
        "def normalize_histogram(counts: dict[str, int]) -> dict[str, float]:\n"
        "    total = sum(counts.values())\n"
        "    if total <= 0:\n"
        "        return {k: 0.0 for k in counts}\n"
        "    return {k: round(v / float(total), 6) for k, v in counts.items()}\n"
    ),
    (
        "def flatten_nested_lists(items: list[object]) -> list[object]:\n"
        "    result: list[object] = []\n"
        "    stack = list(reversed(items))\n"
        "    while stack:\n"
        "        cur = stack.pop()\n"
        "        if isinstance(cur, list):\n"
        "            stack.extend(reversed(cur))\n"
        "        else:\n"
        "            result.append(cur)\n"
        "    return result\n"
    ),
    (
        "def clamp_matrix_values(rows: list[list[float]], low: float, high: float) -> list[list[float]]:\n"
        "    return [[min(high, max(low, cell)) for cell in row] for row in rows]\n"
    ),
)


def load_ungated_hf_python_snippets(
    max_snippets: int = 128,
    *,
    is_mock: bool = False,
    seed: int = 2026,
    forbidden: dict[str, set[str]] | None = None,
) -> tuple[list[str], str]:
    """Load natural Python functions from ungated HuggingFace public corpora with hard eval-disjoint filtering."""
    forb = collect_eval_forbidden_terms() if forbidden is None else forbidden
    rng = random.Random(seed + 31337)

    def _clean_and_accept(code: str) -> str | None:
        if not isinstance(code, str):
            return None
        s = code.strip()
        if len(s) < 90 or len(s) > 2200:
            return None
        if "def " not in s or "return" not in s:
            return None
        if _text_hits_forbidden(s, forb, check_gate2_short=False) is not None:
            return None
        return s

    if is_mock:
        base = list(_MOCK_PYTHON_SNIPPETS)
        out = [base[i % len(base)] for i in range(min(max_snippets, 16))]
        return out, "mock-python-stdlib-snippets"

    candidates = (
        ("Nan-Do/code-search-net-python", None, "train", ("code", "func_code_string")),
        ("bigcode/the-stack-smol", "data/python", "train", ("content",)),
        ("mbpp", None, "train", ("code",)),
    )
    from datasets import load_dataset

    for repo_id, data_dir, split_name, fields in candidates:
        try:
            kwargs: dict[str, Any] = {"split": split_name, "streaming": True}
            if data_dir is not None:
                kwargs["data_dir"] = data_dir
            ds = load_dataset(repo_id, **kwargs)
            collected: list[str] = []
            seen_hashes: set[str] = set()
            for idx, row in enumerate(ds):
                if idx >= max_snippets * 12 or len(collected) >= max_snippets:
                    break
                raw_code = ""
                for f_name in fields:
                    val = row.get(f_name)
                    if isinstance(val, str) and val.strip():
                        raw_code = val
                        break
                ok = _clean_and_accept(raw_code)
                if ok is None:
                    continue
                h = hashlib.md5(ok.encode("utf-8", errors="ignore")).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                collected.append(ok)
            if len(collected) >= min(16, max_snippets):
                rng.shuffle(collected)
                src_label = f"hf_ungated:{repo_id}" + (f"/{data_dir}" if data_dir else "")
                return collected, src_label
        except Exception as exc:
            print(f"[UNIVERSAL PY LOAD] Skipping {repo_id} ({exc}); trying next ungated corpus.", flush=True)

    # Last-resort deterministic fallback if HF network unavailable
    base = list(_MOCK_PYTHON_SNIPPETS)
    return [base[i % len(base)] for i in range(min(max_snippets, 24))], "offline-python-stdlib-fallback"


def _build_general_multiturn_dialogue_episodes(
    train_pool: list[int],
    tokenizer,
    vocab_size: int,
    *,
    num_episodes: int = 32,
    num_pages: int = 8,
    page_size: int = 256,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """General multi-turn user/assistant dialogues (100% disjoint from LongMemEval topics/entities)."""
    rng = random.Random(seed + 4401)
    filler_vocab = train_pool if train_pool else list(range(2, max(16, vocab_size - 2)))
    topics = (
        ("bicycle", "Which frame material did the speaker pick for their commuter bicycle?", ("aluminum", "chromoly", "titanium", "carbon"),
         "User: I finally ordered my commuter bicycle with a {val} frame today.\nAssistant: Great choice for daily riding."),
        ("lighting", "What color temperature bulb was installed in the studio desk lamp?", ("3200K", "4000K", "5000K", "2700K"),
         "User: I swapped the studio desk lamp to a {val} bulb for cleaner video calls.\nAssistant: That should reduce eye strain."),
        ("garden", "Which herb is planted along the southern balcony planter?", ("rosemary", "thyme", "basil", "oregano"),
         "User: Along the southern balcony planter I put fresh {val} seedlings this weekend.\nAssistant: They love full afternoon sun."),
        ("monitor", "What refresh rate does the speaker's new office display support?", ("120Hz", "144Hz", "75Hz", "165Hz"),
         "User: My new office display arrived and runs smoothly at {val} over DisplayPort.\nAssistant: Nice upgrade for scrolling code."),
        ("train", "Which platform does the Tuesday express commuter train depart from?", ("Platform 4", "Platform 7", "Platform 2", "Platform 9"),
         "User: Remember that the Tuesday express commuter train leaves from {val}.\nAssistant: Noted, I will keep that schedule handy."),
        ("kitchen", "What oven temperature is noted for baking the weekend focaccia?", ("220C", "230C", "200C", "215C"),
         "User: For the weekend focaccia recipe, preheat the oven to {val} for twenty minutes.\nAssistant: Got it, high heat gives a crisp crust."),
        ("hiking", "How many liters is the daypack chosen for the autumn trail?", ("24 liters", "30 liters", "18 liters", "28 liters"),
         "User: I packed my {val} daypack with rain gear for the autumn ridge walk.\nAssistant: Plenty of room for water and layers."),
        ("guitar", "Which string gauge set did the musician put on the acoustic guitar?", ("11-52", "12-53", "10-47", "13-56"),
         "User: I restrung the acoustic guitar with {val} phosphor bronze strings yesterday.\nAssistant: Those usually sound warm once settled."),
    )
    episodes: list[dict[str, Any]] = []
    for idx in range(num_episodes):
        _, question, vals, tmpl = topics[idx % len(topics)]
        val = vals[(idx // len(topics) + rng.randrange(len(vals))) % len(vals)]
        pages = [_sample_background_page(rng, page_size, train_pool, filler_vocab) for _ in range(num_pages)]
        tgt_p = rng.randrange(num_pages)
        turn_txt = f"\n[Dialogue Note #{idx + 1}]\n" + tmpl.format(val=val) + "\n"
        t_toks = _encode_text_or_mock(turn_txt, tokenizer, vocab_size)
        off = rng.randint(8, max(8, page_size - len(t_toks) - 8))
        n_w = min(len(t_toks), page_size - off)
        pages[tgt_p][off : off + n_w] = t_toks[:n_w]
        episodes.append(
            {
                "pages": torch.tensor(pages, dtype=torch.long),
                "target_page": tgt_p,
                "question": question,
                "router_query": question,
                "answer": val,
                "raw_text": turn_txt + " " + question + " " + val,
                "source": "general_multiturn_dialogue",
            }
        )
    return episodes


def _build_natural_python_qa_episodes(
    py_snippets: list[str],
    train_pool: list[int],
    tokenizer,
    vocab_size: int,
    *,
    num_episodes: int = 48,
    num_pages: int = 8,
    page_size: int = 256,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Build natural Python code-reading QA episodes + span masks from ungated HF Python snippets."""
    rng = random.Random(seed + 5507)
    filler_vocab = train_pool if train_pool else list(range(2, max(16, vocab_size - 2)))
    def_re = re.compile(r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)", flags=re.MULTILINE)
    episodes: list[dict[str, Any]] = []

    for idx in range(num_episodes):
        code = py_snippets[idx % len(py_snippets)]
        m = def_re.search(code)
        if m:
            fn_name = m.group(1).strip()
            params_raw = m.group(2).strip()
            param_names = [
                p.split(":")[0].split("=")[0].strip().lstrip("*")
                for p in params_raw.split(",")
                if p.strip()
            ]
            param_names = [p for p in param_names if p and p != "self" and p != "cls"]
        else:
            fn_name = "helper_fn"
            param_names = []

        ret_m = re.search(r"^\s*return\s+([^\n#]+)", code, flags=re.MULTILINE)
        is_code_fence = (idx % 2 == 1)
        clean_params = [p for p in param_names if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", p)][:3]
        if is_code_fence:
            sig_args = ", ".join(clean_params)
            call_args = ", ".join(f"{p}={p}" for p in clean_params)
            question = (
                f"Write a short Python wrapper function `call_{fn_name}` that forwards the parameters "
                f"to `{fn_name}(...)` from the Python snippet in Reference Memory."
            )
            raw_prompt_suffix = (
                f"\n\n# Task:\nUsing the Python function `{fn_name}` from Reference Memory, "
                f"write a short Python wrapper `call_{fn_name}` that forwards the parameters to `{fn_name}(...)`.\n\n"
                f"Output ONLY valid Python code inside a ```python block."
                f"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n```python\n"
            )
            answer = f"def call_{fn_name}({sig_args}):\n    return {fn_name}({call_args})\n```\n"
            py_mode = "code_fence"
        else:
            raw_prompt_suffix = None
            py_mode = "qa"
            if param_names and (idx // 2) % 2 == 0:
                question = f"In the Python code above, what is the first parameter name of function `{fn_name}`?"
                answer = param_names[0]
            elif ret_m:
                ret_expr = ret_m.group(1).strip()[:48]
                question = f"In the Python function `{fn_name}`, what expression is returned?"
                answer = ret_expr
            else:
                question = "What Python function name is defined in the snippet above?"
                answer = fn_name

        pages = [_sample_background_page(rng, page_size, train_pool, filler_vocab) for _ in range(num_pages)]
        tgt_p = rng.randrange(num_pages)
        code_block = f"\n# python snippet\n{code}\n"
        c_toks = _encode_text_or_mock(code_block, tokenizer, vocab_size)[: page_size - 8]
        off = rng.randint(0, max(0, page_size - len(c_toks)))
        n_w = min(len(c_toks), page_size - off)
        pages[tgt_p][off : off + n_w] = c_toks[:n_w]

        # Mark def signature + return line for SalienceGate span supervision
        sal_mask = [0.0] * page_size
        sig_line = code.splitlines()[0] if code.splitlines() else code[:64]
        sig_toks = _encode_text_or_mock(sig_line, tokenizer, vocab_size)
        sig_end = min(page_size, off + max(len(sig_toks) + 4, 16))
        for j in range(off, sig_end):
            sal_mask[j] = 1.0

        episodes.append(
            {
                "pages": torch.tensor(pages, dtype=torch.long),
                "target_page": tgt_p,
                "question": question,
                "raw_prompt_suffix": raw_prompt_suffix,
                "py_mode": py_mode,
                "router_query": f"def {fn_name}",
                "answer": answer,
                "sal_page_ids": list(pages[tgt_p]),
                "sal_page_mask": sal_mask,
                "raw_text": code_block + " " + question + " " + answer,
                "source": "natural_python_code",
            }
        )
    return episodes


def _build_disjoint_span16_salience_pages(
    train_pool: list[int],
    tokenizer,
    vocab_size: int,
    *,
    num_pages: int = 128,
    page_size: int = 256,
    seed: int = 2026,
) -> tuple[list[tuple[list[int], list[float]]], list[str]]:
    """Build 100% disjoint general factual/numeric span-16 pages for SalienceGate (0 RULER wording)."""
    rng = random.Random(seed + 6161)
    filler_vocab = train_pool if train_pool else list(range(2, max(16, vocab_size - 2)))
    months = ("March", "April", "July", "October", "November", "January")
    cities = ("Rotterdam", "Osaka", "Zurich", "Halifax", "Bergen", "Lyon", "Bilbao", "Graz")
    prefixes = ("VX", "K7", "RT", "M4", "sector-b", "lot", "ref", "unit-c", "bay", "dock")

    out_pages: list[tuple[list[int], list[float]]] = []
    raw_texts: list[str] = []
    for _ in range(num_pages):
        p_ids = _sample_background_page(rng, page_size, train_pool, filler_vocab)
        mask = [0.0] * page_size
        occupied: list[tuple[int, int]] = []
        clauses_this: list[str] = []
        for _k in range(rng.randint(1, 3)):
            kind = rng.randrange(7)
            tag = f"{rng.choice(prefixes)}-{rng.randint(10, 999)}"
            n7 = str(rng.randint(1000000, 9999999))
            n5 = str(rng.randint(10000, 99999))
            n3 = str(rng.randint(12, 980))
            if kind == 0:
                clause = f" Invoice #{tag} recorded settlement ${n5} on {rng.choice(months)} {rng.randint(1, 28)}. "
            elif kind == 1:
                clause = f" Calibration gauge {tag} registered {n7} units during shift {rng.randint(1, 4)}. "
            elif kind == 2:
                clause = f" Parcel {tag} cleared customs at {rng.choice(cities)} weighing {n3}.{rng.randint(1, 9)} kg. "
            elif kind == 3:
                clause = f" Account {tag} raised reserve threshold from {n3} to {n5} credits. "
            elif kind == 4:
                clause = f" Station {tag} logged serial {n7} at pressure {n3} hPa. "
            elif kind == 5:
                clause = f" Dispatch {tag} assigned {rng.randint(2, 48)} technicians to depot {rng.choice(cities)}. "
            else:
                clause = f" Registry record {tag} lists code {n7} valid until {rng.randint(2027, 2036)}. "
            c_toks = _encode_text_or_mock(clause, tokenizer, vocab_size)
            if len(c_toks) >= page_size - 16:
                continue
            for _try in range(8):
                off = rng.randint(8, page_size - len(c_toks) - 8)
                if all(off + len(c_toks) + 4 <= s or off >= e + 4 for s, e in occupied):
                    p_ids[off : off + len(c_toks)] = c_toks
                    for j in range(off, off + len(c_toks)):
                        mask[j] = 1.0
                    occupied.append((off, off + len(c_toks)))
                    clauses_this.append(clause)
                    break
        out_pages.append((p_ids, mask))
        raw_texts.append(" ".join(clauses_this))
    return out_pages, raw_texts


# ---------------------------------------------------------------------------
# Unified Single-Loop Multi-Task Trainer (4 Domains: 30% / 25% / 25% / 20%)
# ---------------------------------------------------------------------------


def _expand_pages_with_wikitext_distractors(
    pages_cpu: torch.Tensor,
    target_num_pages: int,
    train_pool: list[int],
    vocab_size: int,
    page_size: int,
    rng: random.Random,
) -> torch.Tensor:
    """Append random WikiText-2 train_pool distractor pages so hierarchical chapter TOC (P > 16) is trained in-distribution."""
    cur_p = int(pages_cpu.shape[0])
    if target_num_pages <= cur_p:
        return pages_cpu
    filler_vocab = train_pool if train_pool else list(range(2, max(16, vocab_size - 2)))
    extra = [
        _sample_background_page(rng, page_size, train_pool, filler_vocab)
        for _ in range(target_num_pages - cur_p)
    ]
    extra_t = torch.tensor(extra, dtype=pages_cpu.dtype)
    return torch.cat((pages_cpu, extra_t), dim=0)


def train_universal_checkpoint(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    train_pool: list[int],
    device: torch.device,
    *,
    bge_toc: _PlugInBiEncoderTOC | None = None,
    steps: int = 400,
    lr: float = 8e-5,
    encoder_lr: float = 0.0,
    salience_lr: float = 8e-4,
    lslc_k_deep: int = 4,
    lslc_k_skim: int = 2,
    chapter_size: int = 16,
    salience_route_weight: float = 1.5,
    multiscale_pages: tuple[int, ...] = (8, 24, 48),
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Train ONE universal LSLC checkpoint in a single multi-task loop (30/25/25/20 mix) with Frozen LLM r=0."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    page_size = wrapper.cfg.page_size
    span_len = int(wrapper.cfg.salience_span_len)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    stop_tail = im_end_ids if im_end_ids else ([int(eos_id)] if eos_id is not None else [])

    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR, add_special_tokens=False), dtype=torch.long, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        user_open_embeds = embed_fn(user_open_ids).to(llm_dtype).detach()

    # 1. Build ALL 4 strictly disjoint training pools
    n_s1_each = 8 if is_mock else 40
    d1_prose_rev: list[dict[str, Any]] = []
    d1_texts: list[str] = []
    d2_sal_replay: list[tuple[list[int], list[float]]] = []
    for mode_m, sd_off in (("budget", 1101), ("symbolic", 1203)):
        ds_m = AdversarialLivingBookDataset(
            num_episodes=n_s1_each,
            num_pages=6,
            page_size=page_size,
            num_edits=3,
            mode=mode_m,
            split="train",
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            seed=seed + sd_off,
            background_token_pool=train_pool,
            style="mixed_natural",
        )
        for ep in ds_m:
            clean_q = tokenizer.decode(ep.metadata["router_query_ids"].tolist(), skip_special_tokens=True).strip()
            ans_s = str(ep.metadata["answer_str"]).strip()
            d1_texts.append(f"{clean_q} {ans_s}")
            d1_prose_rev.append(
                {
                    "ep": ep,
                    "clean_q": clean_q,
                    "answer": ans_s,
                    "use_chatml": (len(d1_prose_rev) % 2 == 1),
                }
            )
            sal_m = ep.metadata["context_salience_span_mask"]
            for p_i in range(ep.context_pages.shape[0]):
                if float(sal_m[p_i].sum().item()) > 0.5:
                    d2_sal_replay.append(
                        (ep.context_pages[p_i].tolist(), (sal_m[p_i] > 0.0).float().tolist())
                    )

    n_sal_pages = 16 if is_mock else 160
    d2_sal_clauses, d2_texts = _build_disjoint_span16_salience_pages(
        train_pool, tokenizer, vocab_size, num_pages=n_sal_pages, page_size=page_size, seed=seed
    )

    n_d3_each = 8 if is_mock else 40
    d3_wiki = _build_wikitext_selfsup_qa_episodes(
        train_pool, tokenizer, vocab_size, num_episodes=n_d3_each, num_pages=8, page_size=page_size, seed=seed
    )
    d3_dial = _build_general_multiturn_dialogue_episodes(
        train_pool, tokenizer, vocab_size, num_episodes=n_d3_each, num_pages=8, page_size=page_size, seed=seed
    )
    d3_all = d3_wiki + d3_dial
    d3_texts = [f"{e['question']} {e['answer']}" for e in d3_all]

    py_snippets, py_source = load_ungated_hf_python_snippets(
        max_snippets=16 if is_mock else 128, is_mock=is_mock, seed=seed
    )
    d4_py = _build_natural_python_qa_episodes(
        py_snippets,
        train_pool,
        tokenizer,
        vocab_size,
        num_episodes=12 if is_mock else 64,
        num_pages=8,
        page_size=page_size,
        seed=seed,
    )
    d4_texts = [e["raw_text"] for e in d4_py]

    # 2. Hard Disjointness Assertion across ALL 4 domains
    disjoint_report = assert_universal_train_disjoint(
        {
            "prose_and_state_revision": d1_texts,
            "span16_factual_clauses": d2_texts,
            "wikitext_qa_and_dialogue": d3_texts,
            "natural_python_code": d4_texts,
        }
    )

    # 3. Verify LLM is strictly frozen (r=0) and optionally freeze/slow encoder for multi-scale P calibration
    for p in wrapper.llm.parameters():
        p.requires_grad_(False)
    for p in wrapper.lslc.parameters():
        p.requires_grad_(True)
    freeze_encoder = float(encoder_lr) <= 0.0
    if freeze_encoder:
        for p in wrapper.lslc.encoder.parameters():
            p.requires_grad_(False)
    llm_snap = {n: p.detach().cpu().clone() for n, p in wrapper.llm.named_parameters()}

    prev_use_cache = getattr(wrapper.llm.config, "use_cache", False) if hasattr(wrapper.llm, "config") else False
    if hasattr(wrapper.llm, "config"):
        wrapper.llm.config.use_cache = False
    wrapper.llm.eval()
    wrapper.lslc.train()
    if freeze_encoder:
        wrapper.lslc.encoder.eval()

    sal_ids = {id(p) for p in wrapper.lslc.salience_gate.parameters()}
    enc_ids = {id(p) for p in wrapper.lslc.encoder.parameters()}
    param_groups = [
        {
            "params": [
                p for p in wrapper.lslc.parameters() if p.requires_grad and id(p) not in sal_ids and id(p) not in enc_ids
            ],
            "lr": lr,
        },
        {"params": [p for p in wrapper.lslc.salience_gate.parameters() if p.requires_grad], "lr": salience_lr},
    ]
    if not freeze_encoder:
        enc_params = [p for p in wrapper.lslc.encoder.parameters() if p.requires_grad]
        if enc_params:
            param_groups.append({"params": enc_params, "lr": float(encoder_lr)})
    eff_steps = min(steps, 16) if is_mock else steps
    optimizer = torch.optim.AdamW(param_groups, weight_decay=wrapper.cfg.weight_decay, betas=wrapper.cfg.betas)
    warmup = max(1, int(eff_steps * 0.08))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: float(s + 1) / warmup
        if s < warmup
        else max(0.1, 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(1, eff_steps - warmup)))),
    )

    rng = random.Random(seed + 777)
    pos_weight = torch.tensor([4.0], device=device, dtype=torch.float32)
    domain_weights = (0.30, 0.25, 0.25, 0.20)
    domain_names = ("prose_revision", "span16_salience", "chatml_qa_dialogue", "natural_python")
    domain_counts = {k: 0 for k in domain_names}
    domain_losses: dict[str, list[float]] = {k: [] for k in domain_names}
    gate_bce_losses: list[float] = []
    eff_multiscale = (6, 10, 14) if is_mock else tuple(int(x) for x in multiscale_pages if int(x) >= 6)
    if not eff_multiscale:
        eff_multiscale = (8, 24, 48)
    eff_ms_weights = (0.35, 0.35, 0.30) if len(eff_multiscale) == 3 else tuple(1.0 for _ in eff_multiscale)
    multiscale_counts: dict[str, int] = {str(p_c): 0 for p_c in eff_multiscale}
    python_mode_counts: dict[str, int] = {"qa": 0, "code_fence": 0}
    max_toc_diff_seen = 0.0

    def _encode_pages_train(p_t: torch.Tensor, grad_pages: int | None = None):
        if freeze_encoder:
            with torch.no_grad():
                z_b, bk, _ = _encode_book_chunked(
                    wrapper.lslc,
                    p_t,
                    embed_fn,
                    vocab_size,
                    device,
                    amp_dtype,
                    autocast_enabled,
                    chunk_size=16,
                    special_token_ids=special_ids,
                    salience_span_len=span_len,
                )
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                toc_g = wrapper.lslc.projector.project_toc(z_b)
            return z_b, bk, toc_g

        num_p_tot = int(p_t.shape[1])
        if grad_pages is not None and 0 < int(grad_pages) < num_p_tot:
            gp = int(grad_pages)
            z_main, bk_main, _ = wrapper.lslc.encode_book(
                p_t[:, :gp],
                embed_fn,
                use_exact_bank=True,
                vocab_size=vocab_size,
                special_token_ids=special_ids,
                salience_span_len=span_len,
            )
            with torch.no_grad():
                z_dist, bk_dist, _ = _encode_book_chunked(
                    wrapper.lslc,
                    p_t[:, gp:],
                    embed_fn,
                    vocab_size,
                    device,
                    amp_dtype,
                    autocast_enabled,
                    chunk_size=16,
                    special_token_ids=special_ids,
                    salience_span_len=span_len,
                )
            z_b = torch.cat((z_main, z_dist.to(z_main.dtype)), dim=1)
            k_cur = int(bk_main.token_ids.shape[-1])
            bk = SparseExactBank(
                token_ids=torch.cat((bk_main.token_ids, bk_dist.token_ids), dim=1),
                page_ids=torch.arange(num_p_tot, device=device, dtype=torch.int16)
                .view(1, num_p_tot, 1)
                .expand(1, num_p_tot, k_cur)
                .contiguous(),
                h_coords=torch.cat((bk_main.h_coords, bk_dist.h_coords), dim=1),
                w_coords=torch.cat((bk_main.w_coords, bk_dist.w_coords), dim=1),
            )
            toc_g = wrapper.lslc.projector.project_toc(z_b)
            return z_b, bk, toc_g

        return wrapper.lslc.encode_book(
            p_t,
            embed_fn,
            use_exact_bank=True,
            vocab_size=vocab_size,
            special_token_ids=special_ids,
            salience_span_len=span_len,
        )

    def _salience_span_loss(batch_ids_list: list[list[int]], batch_masks_list: list[list[float]]) -> torch.Tensor:
        ids_t = torch.tensor(batch_ids_list, dtype=torch.long, device=device)
        y_t = torch.tensor(batch_masks_list, dtype=torch.float32, device=device)
        with torch.no_grad():
            emb_t = embed_fn(ids_t).detach()
        logits = wrapper.lslc.salience_gate.score_logits(emb_t, token_ids=ids_t, vocab_size=vocab_size).float()
        loss_bce = F.binary_cross_entropy_with_logits(logits, y_t, pos_weight=pos_weight)
        s_w = min(span_len, page_size)
        win_mean = F.avg_pool1d(logits.unsqueeze(1), kernel_size=s_w, stride=1).squeeze(1)
        win_max = F.max_pool1d(logits.unsqueeze(1), kernel_size=s_w, stride=1).squeeze(1)
        win_score = 0.7 * win_mean + 0.3 * win_max
        win_y = F.avg_pool1d(y_t.unsqueeze(1), kernel_size=s_w, stride=1).squeeze(1)
        pos_w = win_y >= 0.45
        neg_w = win_y <= 0.01
        loss_rank = torch.zeros((), device=device, dtype=torch.float32)
        n_r = 0
        for bi in range(ids_t.shape[0]):
            if bool(pos_w[bi].any()) and bool(neg_w[bi].any()):
                loss_rank = loss_rank + F.relu(2.5 - (win_score[bi][pos_w[bi]].mean() - win_score[bi][neg_w[bi]].amax()))
                n_r += 1
        if n_r > 0:
            loss_rank = loss_rank / float(n_r)
        return loss_bce + 1.0 * loss_rank

    def _chatml_qa_ce(
        pages_cpu: torch.Tensor,
        tgt_p: int,
        question: str,
        router_q: str,
        answer: str,
        *,
        use_chatml: bool = True,
        raw_prompt_suffix: str | None = None,
    ) -> torch.Tensor:
        target_p_cnt = rng.choices(eff_multiscale, weights=eff_ms_weights, k=1)[0]
        multiscale_counts[str(target_p_cnt)] = multiscale_counts.get(str(target_p_cnt), 0) + 1
        orig_p_cnt = int(pages_cpu.shape[0])
        pages_ms = _expand_pages_with_wikitext_distractors(
            pages_cpu, target_p_cnt, train_pool, vocab_size, page_size, rng
        )
        p_t = pages_ms.unsqueeze(0).to(device)
        num_p = p_t.shape[1]
        kd_use = min(lslc_k_deep, num_p)
        ks_use = min(lslc_k_skim, max(0, num_p - kd_use))
        rq_ids = torch.tensor(_encode_text_or_mock(router_q, tokenizer, vocab_size), dtype=torch.long, device=device).unsqueeze(0)
        ext_s = None
        if bge_toc is not None and num_p <= 16:
            with torch.no_grad():
                bge_p4 = bge_toc.encode_book_pages(p_t, None, tokenizer, num_spans=4, batch_size=64)
                ext_s = bge_toc.score_book(bge_toc.encode_texts([router_q], is_query=True), bge_p4)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            z_book, bank, toc = _encode_pages_train(p_t, grad_pages=orig_p_cnt)
            q_emb = embed_fn(rq_ids).mean(dim=1)
            r_out, asm = wrapper.lslc.route_and_assemble(
                z_book=z_book,
                toc_embeds=toc,
                query_embed=q_emb,
                exact_bank=bank,
                embed_tokens_fn=embed_fn,
                k_deep=kd_use,
                k_skim=ks_use,
                mode="full",
                training_ste=False,
                hierarchical_chapter_size=chapter_size,
                query_input_ids=rq_ids,
                salience_route_weight=salience_route_weight,
                exact_bank_attenuation="none",
                external_dense_logits=ext_s,
            )
            if tgt_p not in r_out.deep_idx[0].tolist():
                deep_list = [p for p in r_out.deep_idx[0].tolist() if p != tgt_p][-(kd_use - 1) :] + [tgt_p]
                rem = [p for p in range(num_p) if p not in set(deep_list)]
                r_ov = RouterOutput(
                    logits=r_out.logits,
                    probs=r_out.probs,
                    deep_idx=torch.tensor([deep_list], dtype=torch.long, device=device),
                    skim_idx=torch.tensor([rem[:ks_use]], dtype=torch.long, device=device),
                    toc_idx=torch.tensor([rem[ks_use:]], dtype=torch.long, device=device),
                )
                _, asm = wrapper.lslc.route_and_assemble(
                    z_book=z_book,
                    toc_embeds=toc,
                    query_embed=q_emb,
                    exact_bank=bank,
                    embed_tokens_fn=embed_fn,
                    k_deep=kd_use,
                    k_skim=ks_use,
                    mode="full",
                    training_ste=False,
                    hierarchical_chapter_size=chapter_size,
                    router_out_override=r_ov,
                    exact_bank_attenuation="none",
                )
            if use_chatml:
                q_str = raw_prompt_suffix if raw_prompt_suffix is not None else _format_qa_prompt(question)
                q_ids = torch.tensor(tokenizer.encode(q_str, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
                t_toks = _encode_text_or_mock(answer, tokenizer, vocab_size) + list(stop_tail)
                t_ids = torch.tensor(t_toks or [2], dtype=torch.long, device=device).unsqueeze(0)
                inp_e = torch.cat(
                    (user_open_embeds, asm.prefix_embeds.to(llm_dtype), embed_fn(q_ids).to(llm_dtype), embed_fn(t_ids).to(llm_dtype)),
                    dim=1,
                )
                s_start = user_open_embeds.shape[1] + asm.prefix_embeds.shape[1] + q_ids.shape[1] - 1
            else:
                q_str = f"\nQuestion: {question.strip()}\nAnswer:"
                q_ids = torch.tensor(tokenizer.encode(q_str, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
                t_toks = _encode_text_or_mock(f" {answer}\n", tokenizer, vocab_size)
                t_ids = torch.tensor(t_toks or [2], dtype=torch.long, device=device).unsqueeze(0)
                inp_e = torch.cat(
                    (asm.prefix_embeds.to(llm_dtype), embed_fn(q_ids).to(llm_dtype), embed_fn(t_ids).to(llm_dtype)),
                    dim=1,
                )
                s_start = asm.prefix_embeds.shape[1] + q_ids.shape[1] - 1

            attn_m = torch.ones((1, inp_e.shape[1]), device=device, dtype=torch.long)
            logits = wrapper.llm(inputs_embeds=inp_e, attention_mask=attn_m).logits
            pred = logits[:, s_start : s_start + t_ids.shape[1], :].float()
            return F.cross_entropy(pred.reshape(-1, pred.shape[-1]), t_ids.reshape(-1))

    t0 = time.perf_counter()
    try:
        for step in range(1, eff_steps + 1):
            # Guarantee exact 4-domain coverage even on tiny mock runs, then sample by (0.30, 0.25, 0.25, 0.20)
            if step <= 4:
                dom = domain_names[step - 1]
            else:
                dom = rng.choices(domain_names, weights=domain_weights, k=1)[0]
            domain_counts[dom] += 1
            optimizer.zero_grad(set_to_none=True)

            if dom == "prose_revision":
                item = d1_prose_rev[rng.randrange(len(d1_prose_rev))]
                ep = item["ep"]
                # Alternate between Stage-1 verified prose reading and Stage-2 DeltaUpdater state revision
                if (step == 1 or step % 2 == 0) and ep.edit_pages:
                    target_p_cnt = rng.choices(eff_multiscale, weights=eff_ms_weights, k=1)[0]
                    multiscale_counts[str(target_p_cnt)] = multiscale_counts.get(str(target_p_cnt), 0) + 1
                    orig_p_cnt = int(ep.context_pages.shape[0])
                    pages_ms = _expand_pages_with_wikitext_distractors(
                        ep.context_pages, target_p_cnt, train_pool, vocab_size, page_size, rng
                    )
                    p_t = pages_ms.unsqueeze(0).to(device)
                    edit_flags = ep.metadata.get("edit_accepted_flags")
                    if not edit_flags or len(edit_flags) != len(ep.edit_pages):
                        edit_flags = [True] * len(ep.edit_pages)
                    gate_step_losses: list[torch.Tensor] = []
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                        z_book, bank, toc0 = _encode_pages_train(p_t, grad_pages=orig_p_cnt)
                        z_cur = z_book
                        bank_cur = bank
                        for (edit_p, e_ids_cpu), is_accepted in zip(ep.edit_pages, edit_flags):
                            e_ids = e_ids_cpu.unsqueeze(0).to(device)
                            ep_t = torch.tensor([int(edit_p)], dtype=torch.long, device=device)
                            ov = (
                                extract_contextual_edit_embeds(wrapper.llm, e_ids, layer_idx=wrapper.cfg.contextual_edit_layer)
                                if wrapper.cfg.use_contextual_edit_encoding
                                else None
                            )
                            z_cur, g_p, bank_cur = wrapper.lslc.apply_page_delta(
                                z_cur,
                                ep_t,
                                e_ids,
                                embed_tokens_fn=embed_fn,
                                exact_bank=bank_cur,
                                vocab_size=vocab_size,
                                edit_embeds_override=ov,
                                in_place=False,
                                bank_write_threshold=0.45,
                            )
                            g_logits = getattr(wrapper.lslc.delta_updater, "_last_gate_logits", None)
                            if hasattr(wrapper.lslc.delta_updater, "_last_gate_logits"):
                                wrapper.lslc.delta_updater._last_gate_logits = None
                            if g_logits is not None:
                                strip_logits = g_logits[:, -1, :].float()
                                upper_logits = g_logits[:, :-1, :].float()
                            else:
                                strip_logits = torch.logit(g_p[:, -1, :].float().clamp(1e-5, 1.0 - 1e-5))
                                upper_logits = torch.logit(g_p[:, :-1, :].float().clamp(1e-5, 1.0 - 1e-5))
                            target_strip = torch.full_like(strip_logits, 0.90 if bool(is_accepted) else 0.05)
                            target_upper = torch.full_like(upper_logits, 0.02)
                            loss_strip = F.binary_cross_entropy_with_logits(strip_logits, target_strip)
                            loss_upper = F.binary_cross_entropy_with_logits(upper_logits, target_upper)
                            gate_step_losses.append(1.0 * loss_strip + 2.0 * loss_upper)

                        toc1 = wrapper.lslc.projector.project_toc(z_cur)
                        t_diff = float((toc1.detach().float() - toc0.detach().float()).abs().max().item())
                        if t_diff > max_toc_diff_seen:
                            max_toc_diff_seen = t_diff

                        rq_ids = ep.metadata["router_query_ids"].unsqueeze(0).to(device)
                        num_p = z_cur.shape[1]
                        kd_use = min(lslc_k_deep, num_p)
                        ks_use = min(lslc_k_skim, max(0, num_p - kd_use))
                        tgt_p_ep = int(ep.target_page_idx)
                        r_out, asm = wrapper.lslc.route_and_assemble(
                            z_book=z_cur,
                            toc_embeds=toc1,
                            query_embed=embed_fn(rq_ids).mean(dim=1),
                            exact_bank=bank_cur,
                            embed_tokens_fn=embed_fn,
                            k_deep=kd_use,
                            k_skim=ks_use,
                            mode="full",
                            training_ste=False,
                            hierarchical_chapter_size=chapter_size,
                            query_input_ids=rq_ids,
                            salience_route_weight=salience_route_weight,
                            exact_bank_attenuation="none",
                        )
                        if tgt_p_ep not in r_out.deep_idx[0].tolist():
                            deep_list = [p for p in r_out.deep_idx[0].tolist() if p != tgt_p_ep][-(kd_use - 1) :] + [tgt_p_ep]
                            rem = [p for p in range(num_p) if p not in set(deep_list)]
                            r_ov = RouterOutput(
                                logits=r_out.logits,
                                probs=r_out.probs,
                                deep_idx=torch.tensor([deep_list], dtype=torch.long, device=device),
                                skim_idx=torch.tensor([rem[:ks_use]], dtype=torch.long, device=device),
                                toc_idx=torch.tensor([rem[ks_use:]], dtype=torch.long, device=device),
                            )
                            _, asm = wrapper.lslc.route_and_assemble(
                                z_book=z_cur,
                                toc_embeds=toc1,
                                query_embed=embed_fn(rq_ids).mean(dim=1),
                                exact_bank=bank_cur,
                                embed_tokens_fn=embed_fn,
                                k_deep=kd_use,
                                k_skim=ks_use,
                                mode="full",
                                training_ste=False,
                                hierarchical_chapter_size=chapter_size,
                                router_out_override=r_ov,
                                exact_bank_attenuation="none",
                            )
                        if bool(item["use_chatml"]):
                            q_str = _format_qa_prompt(item["clean_q"])
                            q_ids = torch.tensor(
                                tokenizer.encode(q_str, add_special_tokens=False), dtype=torch.long, device=device
                            ).unsqueeze(0)
                            t_toks = _encode_text_or_mock(item["answer"], tokenizer, vocab_size) + list(stop_tail)
                            t_ids = torch.tensor(t_toks or [2], dtype=torch.long, device=device).unsqueeze(0)
                            inp_e = torch.cat(
                                (
                                    user_open_embeds,
                                    asm.prefix_embeds.to(llm_dtype),
                                    embed_fn(q_ids).to(llm_dtype),
                                    embed_fn(t_ids).to(llm_dtype),
                                ),
                                dim=1,
                            )
                            s_start = user_open_embeds.shape[1] + asm.prefix_embeds.shape[1] + q_ids.shape[1] - 1
                        else:
                            q_str = f"\nQuestion: {item['clean_q'].strip()}\nAnswer:"
                            q_ids = torch.tensor(
                                tokenizer.encode(q_str, add_special_tokens=False), dtype=torch.long, device=device
                            ).unsqueeze(0)
                            t_toks = _encode_text_or_mock(f" {item['answer']}\n", tokenizer, vocab_size)
                            t_ids = torch.tensor(t_toks or [2], dtype=torch.long, device=device).unsqueeze(0)
                            inp_e = torch.cat(
                                (asm.prefix_embeds.to(llm_dtype), embed_fn(q_ids).to(llm_dtype), embed_fn(t_ids).to(llm_dtype)),
                                dim=1,
                            )
                            s_start = asm.prefix_embeds.shape[1] + q_ids.shape[1] - 1

                        attn_m = torch.ones((1, inp_e.shape[1]), device=device, dtype=torch.long)
                        logits = wrapper.llm(inputs_embeds=inp_e, attention_mask=attn_m).logits
                        pred = logits[:, s_start : s_start + t_ids.shape[1], :].float()
                        loss_ce = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), t_ids.reshape(-1))
                        loss_gate = (
                            torch.stack(gate_step_losses).mean()
                            if gate_step_losses
                            else torch.zeros((), device=device, dtype=torch.float32)
                        )
                        gate_bce_losses.append(float(loss_gate.detach().item()))
                        loss = loss_ce + 0.5 * loss_gate
                else:
                    loss = _chatml_qa_ce(
                        ep.metadata["verified_final_pages"],
                        int(ep.target_page_idx),
                        item["clean_q"],
                        item["clean_q"],
                        item["answer"],
                        use_chatml=bool(item["use_chatml"]),
                    )
            elif dom == "span16_salience":
                bs_s = 4 if is_mock else 8
                b_ids, b_masks = [], []
                for _ in range(bs_s):
                    if d2_sal_replay and rng.random() < 0.30:
                        pi, pm = d2_sal_replay[rng.randrange(len(d2_sal_replay))]
                    else:
                        pi, pm = d2_sal_clauses[rng.randrange(len(d2_sal_clauses))]
                    b_ids.append(pi)
                    b_masks.append(pm)
                loss = _salience_span_loss(b_ids, b_masks)
            elif dom == "chatml_qa_dialogue":
                ep_d3 = d3_all[rng.randrange(len(d3_all))]
                loss = _chatml_qa_ce(
                    ep_d3["pages"],
                    int(ep_d3["target_page"]),
                    str(ep_d3["question"]),
                    str(ep_d3["router_query"]),
                    str(ep_d3["answer"]),
                    use_chatml=True,
                )
            else:  # natural_python (20%: 50/50 signature QA + natural fenced-code generation)
                ep_d4 = d4_py[rng.randrange(len(d4_py))]
                pm = str(ep_d4.get("py_mode", "qa"))
                python_mode_counts[pm] = python_mode_counts.get(pm, 0) + 1
                loss_qa = _chatml_qa_ce(
                    ep_d4["pages"],
                    int(ep_d4["target_page"]),
                    str(ep_d4["question"]),
                    str(ep_d4["router_query"]),
                    str(ep_d4["answer"]),
                    use_chatml=True,
                    raw_prompt_suffix=ep_d4.get("raw_prompt_suffix"),
                )
                loss_sal_py = _salience_span_loss([ep_d4["sal_page_ids"]], [ep_d4["sal_page_mask"]])
                loss = loss_qa + 0.25 * loss_sal_py

            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.lslc.parameters(), wrapper.cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            domain_losses[dom].append(float(loss.detach().item()))
    finally:
        if hasattr(wrapper.llm, "config"):
            wrapper.llm.config.use_cache = prev_use_cache
        wrapper.eval()

    # 4. Verify strict Freeze Contract (LLM r=0) + zero special tokens in bank + max_toc_diff == 0.0
    max_frozen_diff = 0.0
    for n, p in wrapper.llm.named_parameters():
        diff = float((p.detach().cpu() - llm_snap[n]).abs().max().item())
        if diff > max_frozen_diff:
            max_frozen_diff = diff
    assert max_frozen_diff == 0.0, f"Frozen LLM parameters mutated: {max_frozen_diff}"
    assert max_toc_diff_seen == 0.0, f"Prefix-cache TOC invariance violated: {max_toc_diff_seen}"

    with torch.no_grad():
        sample_p = d4_py[0]["pages"][:2].unsqueeze(0).to(device)
        _, sample_bank, _ = _encode_book_chunked(
            wrapper.lslc,
            sample_p,
            embed_fn,
            vocab_size,
            device,
            amp_dtype,
            autocast_enabled,
            chunk_size=4,
            special_token_ids=special_ids,
            salience_span_len=span_len,
        )
        sp_set = set(special_ids)
        special_in_bank = sum(int(t) in sp_set for t in sample_bank.token_ids.reshape(-1).tolist())
        assert special_in_bank == 0, f"Special tokens leaked into SparseExactBank: {special_in_bank}"

    loss_summary = {
        k: {
            "steps": domain_counts[k],
            "initial_loss": round(v[0], 4) if v else None,
            "final_loss": round(v[-1], 4) if v else None,
            "mean_loss": round(sum(v) / len(v), 4) if v else None,
        }
        for k, v in domain_losses.items()
    }
    return {
        "steps": eff_steps,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "python_corpus_source": py_source,
        "num_python_snippets": len(py_snippets),
        "domain_weights": {"prose_revision": 0.30, "span16_salience": 0.25, "chatml_qa_dialogue": 0.25, "natural_python": 0.20},
        "domain_stats": loss_summary,
        "encoder_frozen": freeze_encoder,
        "encoder_lr": float(encoder_lr),
        "multiscale_page_counts": multiscale_counts,
        "python_mode_counts": python_mode_counts,
        "gate_bce_steps": len(gate_bce_losses),
        "mean_gate_bce_loss": round(sum(gate_bce_losses) / len(gate_bce_losses), 4) if gate_bce_losses else None,
        "disjointness_audit": disjoint_report,
        "max_frozen_param_diff": max_frozen_diff,
        "max_toc_diff": max_toc_diff_seen,
        "special_tokens_in_bank": special_in_bank,
        "trainable_lslc_params": wrapper.lslc.trainable_parameter_count(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Single Universal LSLC v3 Checkpoint (Frozen Mode A r=0).")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--mock-llm", action="store_true", help="Run 0-CU local CPU smoke test with TinyMockCausalLM.")
    parser.add_argument("--train-universal", action="store_true", default=True)
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--save-checkpoint", type=str, default="/content/lslc_results/lslc_universal_qwen3_1.7b.safetensors")
    parser.add_argument("--bge-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--use-bge-toc", action="store_true", default=True)
    parser.add_argument("--no-bge-toc", dest="use_bge_toc", action="store_false")
    parser.add_argument("--k-exact", type=int, default=48)
    parser.add_argument("--salience-span-len", type=int, default=16)
    parser.add_argument("--salience-route-weight", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--encoder-lr", type=float, default=0.0, help="0.0 freezes encoder; >0 trains encoder at given LR.")
    parser.add_argument("--salience-lr", type=float, default=8e-4)
    parser.add_argument("--multiscale-pages", type=str, default="8,24,48")
    parser.add_argument("--lslc-k-deep", type=int, default=4)
    parser.add_argument("--lslc-k-skim", type=int, default=2)
    parser.add_argument("--chapter-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-json", type=str, default="/content/lslc_results/v3_universal_ckpt_train.json")
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.mock_llm:
        vocab_size = 2048
        cfg = LSLCConfig.tiny_mock(
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
        tokenizer = _MockCodeTokenizer(vocab_size=vocab_size)
        device = torch.device("cpu")
        train_pool = [2 + (i % (vocab_size - 2)) for i in range(4096)]
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
        cfg.contextual_edit_layer = cfg.num_llm_layers // 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        llm, tokenizer = load_frozen_qwen3(cfg, load_in_4bit=False, device=device)
        vocab_size = len(llm.get_input_embeddings().weight)
        train_pool, _, data_source = _load_wikitext_pools(tokenizer)

    wrapper = FrozenQwenLSLCWrapper(cfg, llm).to(device)
    init_info = _load_lslc_checkpoint(wrapper.lslc, args.init_checkpoint) if args.init_checkpoint else None
    bge_toc = _PlugInBiEncoderTOC(args.bge_model, device=device, is_mock=args.mock_llm) if args.use_bge_toc else None

    ms_pages = tuple(int(x.strip()) for x in args.multiscale_pages.split(",") if x.strip())
    train_stats = train_universal_checkpoint(
        wrapper=wrapper,
        tokenizer=tokenizer,
        train_pool=train_pool,
        device=device,
        bge_toc=bge_toc,
        steps=args.steps,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        salience_lr=args.salience_lr,
        lslc_k_deep=args.lslc_k_deep,
        lslc_k_skim=args.lslc_k_skim,
        chapter_size=args.chapter_size,
        salience_route_weight=args.salience_route_weight,
        multiscale_pages=ms_pages,
        seed=args.seed,
        is_mock=args.mock_llm,
    )

    ckpt_sha256: str | None = None
    if args.save_checkpoint:
        _save_lslc_checkpoint(wrapper.lslc, args.save_checkpoint)
        ckpt_sha256 = hashlib.sha256(Path(args.save_checkpoint).read_bytes()).hexdigest()

    payload = {
        "version": "3.0.0-universal-checkpoint",
        "model": "TinyMockCausalLM" if args.mock_llm else args.model,
        "mock_llm": args.mock_llm,
        "device": str(device),
        "data_source": data_source,
        "init_checkpoint": init_info,
        "saved_checkpoint": args.save_checkpoint or None,
        "checkpoint_sha256": ckpt_sha256,
        "config": {
            "page_size": cfg.page_size,
            "k_exact": cfg.k_exact,
            "salience_span_len": cfg.salience_span_len,
            "router_mode": cfg.router_mode,
            "salience_mode": cfg.salience_mode,
            "delta_pooling_mode": cfg.delta_pooling_mode,
            "contextual_edit_layer": cfg.contextual_edit_layer,
            "exact_bank_attenuation": "none",
            "frozen_mode_a_lora_r": 0,
            "bge_model": None if bge_toc is None else bge_toc.model_name,
            "salience_route_weight": args.salience_route_weight,
        },
        "train": train_stats,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[UNIVERSAL CKPT DONE] sha256={ckpt_sha256} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
