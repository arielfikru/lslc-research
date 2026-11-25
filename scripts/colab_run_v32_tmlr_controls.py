#!/usr/bin/env python3
"""v3.2 TMLR Micro-Controls Evaluation Suite (Zero Architecture Changes, Zero Retraining).

Loads locked final Path (B) checkpoints (`lslc_pathb_qwen3_1.7b.safetensors` or
`lslc_pathb_llama31_8b.safetensors`) onto strictly frozen BF16 backbones (`r=0`)
and closes the final 3 audit gaps:

  1. Control #1 — Gate-2 Locked-Pages & Locked-Order 2x4 Factorial + Strong `raw_text_overwrite`:
     - N=200 hold-out episodes (100 budget k=3 + 100 symbolic k=5; seed=2029 Qwen3, seed=2026 Llama-8B).
     - Runs Path B hybrid router ONCE on the post-edit bank to lock 4 page IDs (`locked_pages`),
       asserting 100% identical 4 pages across all 8 arms.
     - 2 Locked Page Orders (`edited_last` vs `page_id_asc`) x 4 Write Policies:
         (a) `raw_text_append`
         (b) `raw_text_overwrite` (in-place entity line/sentence replacement on raw text; removes stale value
             while preserving neighbor & static facts; gate-matched)
         (c) `standalone_append_192tok` (rolling tail `app_toks[-tail_cap:]`; 192 tok; gate-matched)
         (d) `standalone_overwrite_192tok` (Path B Span-Aligned Selective Overwrite; 192 tok; gate-matched)
     - Reports `strict_em`, `normalized_numeric_em`, `normalized_w2n_em`, per-mode & per-query-type
       breakdowns, exact two-sided McNemar, paired bootstrap CI95, **cluster-bootstrap CI95**
       on `(mode, query_type, gold)` clusters (B=2000), and Holm-Bonferroni adjusted p-values.

  2. Control #2 — Full Real-Text Mutable Benchmark: LongMemEval `knowledge-update` Subset:
     - Loads ALL non-abstention `question_type == "knowledge-update"` items (~78) from
       `xiaowu0162/longmemeval-cleaned` (`longmemeval_s_cleaned.json`), mixing ZERO static categories.
     - Evaluates 4 arms: `bm25_rag_k2`, `bge_4span_rag_k2`, `standalone_discrete_k48_d4` (192 tok),
       and `standalone_discrete_k48_d6` (288 tok).
     - Explicitly logs `packing_mode` (`gold_prioritized_64p` or `full_untruncated`), `num_episodes`,
       `extracted_em`, `substring_match`, `token_f1`, `local_value_em`, `session_hit_at_k`,
       `answer_substr_in_chosen_pages_rate`, `answer_substr_in_exact_bank_rate`, `active_prefix_tokens`,
       exact McNemar, paired bootstrap CI95, and Holm-Bonferroni adjusted p-values.

  3. Clean Provenance:
     - Explicit `--git-sha` CLI flag (defaults to `git rev-parse HEAD`), plus automatic `checkpoint_sha256`
       computed directly from the loaded `.safetensors` file.
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
import urllib.request
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from colab_run_m2 import _load_wikitext_pools  # noqa: E402
from colab_run_rebuttal_v2 import BM25PageRetriever  # noqa: E402
from colab_run_v3_gerbang3_coding import _MockCodeTokenizer  # noqa: E402
from colab_run_v3_gerbang4_community import (  # noqa: E402
    _USER_OPEN_STR,
    _PlugInBiEncoderTOC,
    _answer_in_chosen_pages,
    _build_verified_local_longmemeval_fallback,
    _encode_unpadded_rag_ids,
    _format_qa_prompt,
    _get_special_token_ids,
    _greedy_generate_qa,
    _pack_longmemeval_sessions,
    _sync_device,
    compute_community_qa_metrics,
)
from colab_run_v3_gerbang5_ablations import (  # noqa: E402
    _FAIR_G2_FEWSHOT_PREFIX,
    canonicalize_numeric_answer,
)
from colab_run_v3_gerbang6_llama8b_pathb import (  # noqa: E402
    LLAMA3_USER_OPEN,
    format_llama3_qa_prompt,
    greedy_generate_llama3,
    load_frozen_llama8b_with_fallback,
)
from colab_run_v3_path_tiebreaker import (  # noqa: E402
    _build_append_192tok_bank,
    _summarize_scalar_rows,
    exact_mcnemar_test,
    paired_bootstrap_ci95,
)
from lslc import (  # noqa: E402
    AdversarialLivingBookDataset,
    LSLCConfig,
    PathBDiscreteMemory,
    SparseExactBank,
    TinyMockCausalLM,
    canonicalize_w2n_answer,
    compute_value_em,
    get_git_provenance,
    get_newline_sep_token_id,
    load_frozen_qwen3,
    load_pathb_checkpoint,
    select_amp_dtype,
)
from lslc.datasets import _encode_text_or_mock  # noqa: E402


# ---------------------------------------------------------------------------
# Provenance & Statistical Helpers (Cluster-Bootstrap CI95 + Holm-Bonferroni)
# ---------------------------------------------------------------------------


def resolve_git_provenance(explicit_git_sha: str = "", repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Resolve real git SHA provenance (CLI override takes priority over local `git rev-parse HEAD`)."""
    local_prov = get_git_provenance(repo_root)
    cli_sha = explicit_git_sha.strip()
    if cli_sha:
        return {
            "git_sha": cli_sha,
            "git_dirty": local_prov.get("git_dirty"),
            "git_sha_source": "cli_arg",
        }
    return {
        "git_sha": local_prov.get("git_sha", "unknown"),
        "git_dirty": local_prov.get("git_dirty"),
        "git_sha_source": "git_rev_parse",
    }


def holm_bonferroni_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Step-down Holm-Bonferroni multiple-comparison adjustment on a family of p-values."""
    if not p_values:
        return {}
    m = len(p_values)
    ranked = sorted(p_values.items(), key=lambda kv: (float(kv[1]), kv[0]))
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank_idx, (key, raw_p) in enumerate(ranked):
        mult = m - rank_idx
        adj = min(1.0, max(running_max, float(raw_p) * mult))
        running_max = adj
        adjusted[key] = round(adj, 6)
    return adjusted


def cluster_bootstrap_ci95(
    scores_a: list[float],
    scores_b: list[float],
    cluster_ids: list[str],
    *,
    n_boot: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Deterministic cluster-level 95% percentile bootstrap CI for mean(A - B).

    Resamples whole clusters WITH replacement so episodes sharing `(mode, query_type, gold)`
    are never treated as independent draws.
    """
    n = len(scores_a)
    if len(scores_b) != n or len(cluster_ids) != n:
        raise ValueError(f"Mismatched lengths: A={n}, B={len(scores_b)}, clusters={len(cluster_ids)}")
    if n == 0:
        return {
            "n": 0,
            "num_clusters": 0,
            "mean_diff": 0.0,
            "cluster_diff_ci95": [0.0, 0.0],
            "n_boot": int(n_boot),
        }

    diffs = [float(a) - float(b) for a, b in zip(scores_a, scores_b)]
    mean_diff = sum(diffs) / n

    cluster_map: dict[str, list[float]] = {}
    for cid, d in zip(cluster_ids, diffs):
        cluster_map.setdefault(str(cid), []).append(d)
    keys = sorted(cluster_map.keys())
    num_c = len(keys)
    # Precompute (sum_of_diffs, count) per cluster for fast O(B * C) resampling
    c_stats = [(sum(cluster_map[k]), len(cluster_map[k])) for k in keys]

    if num_c <= 1:
        ci = [round(mean_diff, 4), round(mean_diff, 4)]
    else:
        rng = random.Random(int(seed) + n * 37 + num_c * 17)
        boot_means: list[float] = []
        for _ in range(int(n_boot)):
            tot_d = 0.0
            tot_cnt = 0
            for _j in range(num_c):
                s_c, cnt_c = c_stats[rng.randrange(num_c)]
                tot_d += s_c
                tot_cnt += cnt_c
            boot_means.append(tot_d / max(1, tot_cnt))
        boot_means.sort()
        lo = boot_means[max(0, int(0.025 * (n_boot - 1)))]
        hi = boot_means[min(n_boot - 1, int(0.975 * (n_boot - 1)))]
        ci = [round(lo, 4), round(hi, 4)]

    return {
        "n": n,
        "num_clusters": num_c,
        "mean_diff": round(mean_diff, 4),
        "cluster_diff_ci95": ci,
        "n_boot": int(n_boot),
    }


def compute_pairwise_with_cluster_and_holm(
    scores_by_sys: dict[str, list[float]],
    pairs: list[tuple[str, str]],
    *,
    cluster_ids: list[str] | None = None,
    n_boot: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Compute paired bootstrap CI95 + exact McNemar + optional cluster-bootstrap CI95 + Holm-Bonferroni p-values."""
    out: dict[str, Any] = {}
    raw_p_map: dict[str, float] = {}
    for a_name, b_name in pairs:
        if a_name not in scores_by_sys or b_name not in scores_by_sys:
            continue
        sa = scores_by_sys[a_name]
        sb = scores_by_sys[b_name]
        if len(sa) != len(sb):
            continue
        key = f"{a_name}__vs__{b_name}"
        entry = paired_bootstrap_ci95(sa, sb, n_boot=n_boot, seed=seed)
        if cluster_ids is not None:
            entry["cluster_bootstrap"] = cluster_bootstrap_ci95(sa, sb, cluster_ids, n_boot=n_boot, seed=seed)
        if "mcnemar" in entry:
            raw_p_map[key] = float(entry["mcnemar"]["p_value_exact"])
        out[key] = entry

    if raw_p_map:
        adj_map = holm_bonferroni_adjust(raw_p_map)
        for key, adj_p in adj_map.items():
            out[key]["mcnemar"]["p_value_holm"] = adj_p
    return out


# ---------------------------------------------------------------------------
# Strong Non-Oracle Raw-Text In-Place Overwrite (`raw_text_overwrite`)
# ---------------------------------------------------------------------------

_GENERIC_EDIT_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "before", "board", "budget", "by", "cap",
        "ceiling", "changes", "cluster", "committee", "complete", "confirmed", "controller",
        "current", "cutover", "cycle", "declined", "denied", "deployment", "directive",
        "discarded", "down", "draft", "effective", "elevates", "enacted", "executive",
        "existing", "failed", "figure", "finalized", "finance", "for", "from", "funding",
        "gate", "granted", "guard", "health", "holds", "immediately", "in", "inactive",
        "increase", "infrastructure", "intact", "into", "is", "it", "leadership", "lead",
        "leaving", "ledger", "limit", "live", "memo", "mode", "motion", "move", "moves",
        "moving", "new", "no", "notice", "now", "of", "off", "official", "on", "operating",
        "operations", "orchestrator", "override", "pass", "places", "planned", "policy",
        "prior", "project", "promoted", "promoting", "proposal", "proposed", "puts", "quarterly",
        "raise", "raising", "ratified", "record", "refused", "rejected", "release", "remains",
        "replacing", "request", "review", "revised", "revision", "rollback", "rollout", "runtime",
        "seeking", "serves", "set", "shift", "shifted", "shifting", "signed", "sign-off", "sre",
        "staging", "stands", "state", "suggested", "switched", "the", "this", "to", "topology",
        "toward", "transitioned", "transitioning", "treasury", "turned", "unapproved", "under",
        "unratified", "up", "update", "updated", "vetoed", "voted", "was", "without", "allowance",
        "allocation", "amendment", "addendum", "approval", "approved", "aborted", "actively",
        "attempt", "audit", "authorized", "blocked", "bump", "called", "canary", "cancelled",
        "check", "did", "not",
    }
)

_DOMAIN_STATE_KEYWORDS: frozenset[str] = frozenset(
    {
        "budget", "funding", "cap", "ceiling", "allowance", "allocation", "limit", "spending",
        "deployment", "topology", "state", "mode", "runtime", "cluster", "failover", "serves",
        "record", "project_budget", "deploy_state", "cluster_mode", "runtime_allocations",
    }
)


def _extract_salient_entities(text: str) -> tuple[set[str], set[str]]:
    """Extract candidate entity tokens (case-sensitive proper nouns + lowercased identifiers) with ZERO oracle metadata."""
    proper_entities: set[str] = set()
    id_tokens: set[str] = set()
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", text):
        low = tok.lower()
        if low in _GENERIC_EDIT_WORDS:
            continue
        if tok[0].isupper():
            proper_entities.add(tok)
        if "-" in tok or "_" in tok or any(ch.isdigit() for ch in tok):
            # Exclude pure value quantities like "90M", "35M", "100GB" from entity keys
            if not re.fullmatch(r"\d+(?:\.\d+)?(?:[mMkKgGbB]|ms|gb|mb|kb|rps)?", tok):
                id_tokens.add(low)
    return proper_entities, id_tokens


def _is_static_policy_segment(seg: str) -> bool:
    s_up = seg.strip().upper()
    return (
        s_up.startswith("POLICY:")
        or "STATIC_POLICY_RULE" in s_up
        or "SYSTEM_STANDARDS" in s_up
        or s_up.startswith("PER CURRENT ENGINEERING STANDARDS")
        or s_up.startswith("ENGINEERING COMPLIANCE BASELINE")
    )


def _score_segment_for_overwrite(
    seg: str,
    edit_proper: set[str],
    edit_ids: set[str],
    edit_low_words: set[str],
) -> float:
    """Score how strongly `seg` is the prior record/edit for the entity updated by `edit_text`."""
    s = seg.strip()
    if not s or _is_static_policy_segment(s):
        return 0.0

    seg_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", s)
    seg_tok_set = set(seg_tokens)
    seg_low_set = {t.lower() for t in seg_tokens}

    # 1. Exact proper-noun entity overlap (e.g., "Lyra", "Orion", "Kestrel-Core")
    matched_proper = edit_proper & seg_tok_set
    matched_proper_ci = {e.lower() for e in edit_proper} & seg_low_set
    matched_ids = edit_ids & seg_low_set

    if not matched_proper and not matched_proper_ci and not matched_ids:
        return 0.0

    # Guard against touching a different project's RECORD line that doesn't contain our entity
    score = 5.0 * len(matched_proper) + 3.0 * len(matched_proper_ci - {e.lower() for e in matched_proper}) + 3.0 * len(matched_ids)

    # 2. Record / prior-edit structural indicators
    if re.search(r"\bRECORD\s*:", s) or any(
        tag in s for tag in ("PROJECT_BUDGET", "DEPLOY_STATE", "CLUSTER_MODE", "RUNTIME_ALLOCATIONS")
    ):
        score += 2.5
    if seg_low_set & _DOMAIN_STATE_KEYWORDS:
        score += 1.5
    # Slight bonus for shared domain context words (excluding pure stopwords)
    domain_overlap = (seg_low_set & edit_low_words) & _DOMAIN_STATE_KEYWORDS
    score += 0.5 * len(domain_overlap)
    return score


def apply_raw_text_in_place_overwrite(doc_text: str, edit_text: str) -> tuple[str, dict[str, Any]]:
    """Strong non-oracle raw-text overwrite baseline: replace the prior record/edit line (or sentence)
    for the same entity in-place with `edit_text`, leaving neighbor records and static policies intact.
    """
    clean_edit = edit_text.strip()
    if not clean_edit:
        return doc_text, {"replaced": False, "mode": "empty_edit", "num_segments_replaced": 0}

    edit_proper, edit_ids = _extract_salient_entities(clean_edit)
    edit_low_words = {w.lower() for w in re.findall(r"[A-Za-z0-9_-]+", clean_edit)}

    lines = doc_text.splitlines()
    line_scores: list[float] = []
    line_sents_cache: list[list[str]] = []
    for ln in lines:
        sents = [s for s in re.split(r"(?<=[.!?])\s+", ln.strip()) if s.strip()]
        line_sents_cache.append(sents)
        if len(sents) >= 2:
            sc_s = [_score_segment_for_overwrite(st, edit_proper, edit_ids, edit_low_words) for st in sents]
            line_scores.append(max(sc_s) if sc_s else 0.0)
        else:
            line_scores.append(_score_segment_for_overwrite(ln, edit_proper, edit_ids, edit_low_words))

    best_line_idx = max(range(len(lines)), key=lambda i: line_scores[i]) if lines else -1
    best_line_score = line_scores[best_line_idx] if best_line_idx >= 0 else 0.0

    if best_line_score >= 5.0:
        sents = line_sents_cache[best_line_idx]
        if len(sents) >= 2:
            sent_scores = [_score_segment_for_overwrite(st, edit_proper, edit_ids, edit_low_words) for st in sents]
            best_s_idx = max(range(len(sents)), key=lambda i: sent_scores[i])
            other_nonmatching = any(
                (i != best_s_idx) and (sent_scores[i] < 5.0)
                for i in range(len(sents))
            )
            if sent_scores[best_s_idx] >= 5.0 and other_nonmatching:
                new_sents: list[str] = []
                replaced_cnt = 0
                for s_i, st in enumerate(sents):
                    if s_i == best_s_idx:
                        new_sents.append(clean_edit)
                        replaced_cnt += 1
                    elif sent_scores[s_i] >= 6.5:
                        # Collapse any stale duplicate record sentence for the exact same entity
                        replaced_cnt += 1
                    else:
                        new_sents.append(st)
                lines[best_line_idx] = " ".join(new_sents)
                out_txt = "\n".join(lines)
                if doc_text.endswith("\n"):
                    out_txt += "\n"
                return out_txt, {"replaced": True, "mode": "sentence_in_place", "num_segments_replaced": replaced_cnt}

        # Standard line-level in-place replacement (also collapses any duplicate prior record/edit lines for same entity)
        new_lines: list[str] = []
        replaced_cnt = 0
        for i, ln in enumerate(lines):
            if i == best_line_idx:
                new_lines.append(clean_edit)
                replaced_cnt += 1
            elif line_scores[i] >= 6.5:
                replaced_cnt += 1
            else:
                new_lines.append(ln)
        out_txt = "\n".join(new_lines)
        if doc_text.endswith("\n"):
            out_txt += "\n"
        return out_txt, {"replaced": True, "mode": "line_in_place", "num_segments_replaced": replaced_cnt}

    # Fallback: no matching entity segment found on this page -> append cleanly
    sep = "" if (not doc_text or doc_text.endswith("\n")) else "\n"
    return f"{doc_text}{sep}{clean_edit}\n", {"replaced": False, "mode": "append_fallback", "num_segments_replaced": 0}


# ---------------------------------------------------------------------------
# Locked Page Ordering Helper (2 Exact Orders on Identical `locked_pages`)
# ---------------------------------------------------------------------------


def order_locked_pages(locked_pages: list[int], edited_pages: set[int], order_mode: str) -> list[int]:
    """Produce exact deterministic page order over `locked_pages` without altering page membership.

    - `edited_last`: unedited pages ascending first, edited pages ascending last (True Two-Bank order).
    - `page_id_asc`: strictly ascending page IDs (ZERO edited-page recency bias).
    """
    canonical = sorted(int(p) for p in locked_pages)
    if order_mode == "edited_last":
        b1, b2 = PathBDiscreteMemory.order_two_bank_pages(canonical, edited_pages)
        ordered = b1 + b2
    elif order_mode == "page_id_asc":
        ordered = list(canonical)
    else:
        raise ValueError(f"Unknown order_mode: {order_mode!r}")
    assert sorted(ordered) == canonical, f"Order corrupted locked_pages: {ordered} vs {canonical}"
    return ordered


# ---------------------------------------------------------------------------
# Model Family Helpers (Qwen3 ChatML vs Official Llama-3.1-Instruct)
# ---------------------------------------------------------------------------


def is_llama_model_name(model_name: str) -> bool:
    return "llama" in str(model_name).lower()


def load_frozen_backbone_for_v32(
    requested_model: str,
    cfg: LSLCConfig,
    device: torch.device,
) -> tuple[nn.Module, Any, str, list[dict[str, str]], str]:
    """Load strictly frozen BF16 backbone (Qwen3 or Llama-3.1-8B with ungated fallback)."""
    if is_llama_model_name(requested_model):
        llm, tokenizer, resolved_id, attempts = load_frozen_llama8b_with_fallback(
            cfg, requested_model=requested_model, device=device
        )
        prompt_family = "llama-3.1-instruct"
    else:
        llm, tokenizer = load_frozen_qwen3(cfg, device=device)
        resolved_id = requested_model
        attempts = [{"model_id": requested_model, "status": "loaded"}]
        prompt_family = "qwen3-chatml"

    for p in llm.parameters():
        p.requires_grad_(False)
    llm.eval()
    if cfg.contextual_edit_layer is None:
        cfg.contextual_edit_layer = max(1, cfg.num_llm_layers // 2)
    return llm, tokenizer, resolved_id, attempts, prompt_family


def build_prompt_and_generate(
    llm: nn.Module,
    prefix_embeds: torch.Tensor,
    question: str,
    tokenizer,
    device: torch.device,
    *,
    prompt_family: str,
    fewshot_prefix: str = "",
    max_new_tokens: int = 24,
) -> str:
    """Format prompt & run KV-cached greedy generation using the backbone's official template."""
    if prompt_family == "llama-3.1-instruct":
        user_open_ids = torch.tensor(
            tokenizer.encode(LLAMA3_USER_OPEN + fewshot_prefix, add_special_tokens=False),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        p_prompt = torch.tensor(
            tokenizer.encode(format_llama3_qa_prompt(question), add_special_tokens=False),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        return greedy_generate_llama3(
            llm,
            prefix_embeds,
            p_prompt,
            tokenizer,
            max_new_tokens=max_new_tokens,
            user_open_ids=user_open_ids,
        )

    user_open_ids = torch.tensor(
        tokenizer.encode(_USER_OPEN_STR + fewshot_prefix, add_special_tokens=False),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    p_prompt = torch.tensor(
        tokenizer.encode(_format_qa_prompt(question), add_special_tokens=False),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    return _greedy_generate_qa(
        llm,
        prefix_embeds,
        p_prompt,
        tokenizer,
        max_new_tokens=max_new_tokens,
        user_open_ids=user_open_ids,
    )


# ---------------------------------------------------------------------------
# Control #1: Gate-2 Locked-Pages & Locked-Order 2x4 Factorial
# ---------------------------------------------------------------------------

GATE2_ORDER_MODES: tuple[str, ...] = ("edited_last", "page_id_asc")
GATE2_WRITE_POLICIES: tuple[str, ...] = (
    "raw_text_append",
    "raw_text_overwrite",
    "standalone_append_192tok",
    "standalone_overwrite_192tok",
)


@torch.no_grad()
def run_gate2_factorial_2x4(
    llm: nn.Module,
    pathb: PathBDiscreteMemory,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    prompt_family: str,
    eval_n: int = 200,
    seed: int = 2029,
    salience_route_weight: float = 1.5,
    n_boot: int = 2000,
    is_mock: bool = False,
) -> dict[str, Any]:
    """2x4 Factorial Gate-2 control with 100% locked 4 pages & locked page orders across all 8 arms."""
    t_start = time.perf_counter()
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
            num_episodes=n_each,
            num_pages=num_p,
            page_size=pathb.cfg.page_size,
            num_edits=3,
            mode="budget",
            split="eval",
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            seed=seed + 40,
            background_token_pool=val_pool,
        )
    )
    eps_sym = list(
        AdversarialLivingBookDataset(
            num_episodes=n_each,
            num_pages=num_p,
            page_size=pathb.cfg.page_size,
            num_edits=5,
            mode="symbolic",
            split="eval",
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            seed=seed + 41,
            background_token_pool=val_pool,
        )
    )
    all_eps = eps_budget + eps_sym

    arm_names = [f"{ord_m}__{wp}" for ord_m in GATE2_ORDER_MODES for wp in GATE2_WRITE_POLICIES]
    q_types_list = ("accepted_override", "rejected_proposal", "neighbor_preservation", "randomized_static_retention")

    rows_by_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in arm_names}
    rows_by_arm_mode: dict[str, dict[str, list[dict[str, Any]]]] = {a: {"budget": [], "symbolic": []} for a in arm_names}
    rows_by_arm_qtype: dict[str, dict[str, list[dict[str, Any]]]] = {
        a: {qt: [] for qt in q_types_list} for a in arm_names
    }
    cluster_ids: list[str] = []

    gate_tp = gate_tn = gate_fp = gate_fn = 0
    raw_overwrite_replaced_turns = 0
    raw_overwrite_total_accepted_turns = 0

    for ep_idx, ep in enumerate(all_eps):
        clean_q = tokenizer.decode(ep.metadata["router_query_ids"].tolist(), skip_special_tokens=True).strip()
        gold = str(ep.metadata["answer_str"]).strip()
        ep_mode = str(ep.metadata.get("mode", "budget"))
        q_type = str(ep.metadata.get("query_type", ""))
        gold_flags = list(ep.metadata.get("edit_accepted_flags", []))
        cluster_ids.append(f"{ep_mode}|{q_type}|{gold}")

        p_t = ep.context_pages.unsqueeze(0).to(device)
        page_texts = [tokenizer.decode(ep.context_pages[i].tolist(), skip_special_tokens=True) for i in range(num_p)]

        # 1. Encode Path B bank ONCE per episode
        enc_pack = pathb.encode_book(
            p_t,
            embed_fn,
            tokenizer,
            vocab_size,
            device=device,
            amp_dtype=amp_dtype,
            autocast_enabled=autocast_enabled,
            special_token_ids=special_ids,
            num_spans=4,
        )
        base_bank: SparseExactBank = enc_pack["bank"]
        bge_bf16 = enc_pack["bge_keys_bf16"]

        # Clone base bank for overwrite arm so base_bank remains pristine for append arm
        bank_overwrite = SparseExactBank(
            token_ids=base_bank.token_ids.clone(),
            page_ids=None,
            h_coords=None,
            w_coords=None,
            base_token_ids=None,
            edited_pages=set(),
        )
        appended_toks_by_page: dict[int, list[int]] = {}
        docs_append = list(page_texts)
        docs_overwrite = list(page_texts)
        edited_pages_set: set[int] = set()

        # 2. Apply identical Path B mid-layer blind edit gate decisions across all 4 write policies
        for (p_i, e_ids_cpu), is_acc_gold in zip(ep.edit_pages, gold_flags):
            e_ids = e_ids_cpu.unsqueeze(0).to(device)
            e_txt = tokenizer.decode(e_ids_cpu.tolist(), skip_special_tokens=True).strip()

            # Score gate & update Path B overwrite bank
            bank_overwrite, pred_acc, _prob, _e_ms = pathb.apply_edit(
                bank_overwrite,
                int(p_i),
                e_ids,
                llm,
                embed_fn,
                vocab_size,
                device=device,
                amp_dtype=amp_dtype,
                autocast_enabled=autocast_enabled,
                in_place=True,
                span_aligned_overwrite=True,
                sep_token_id=sep_id,
            )
            if is_acc_gold and pred_acc:
                gate_tp += 1
            elif is_acc_gold and not pred_acc:
                gate_fn += 1
            elif (not is_acc_gold) and pred_acc:
                gate_fp += 1
            else:
                gate_tn += 1

            if pred_acc:
                edited_pages_set.add(int(p_i))
                # Rolling-tail 192-tok append control (consistent with v3.1 separator between consecutive edits)
                buf = appended_toks_by_page.setdefault(int(p_i), [])
                if buf and sep_id is not None and int(sep_id) >= 0:
                    buf.append(int(sep_id))
                buf.extend([int(t) for t in e_ids_cpu.tolist()])
                # Raw-text append control
                docs_append[int(p_i)] = docs_append[int(p_i)].rstrip() + f"\nUpdate: {e_txt}\n"
                # Strong raw-text in-place overwrite control
                updated_doc, ow_info = apply_raw_text_in_place_overwrite(docs_overwrite[int(p_i)], e_txt)
                docs_overwrite[int(p_i)] = updated_doc
                raw_overwrite_total_accepted_turns += 1
                if ow_info["replaced"]:
                    raw_overwrite_replaced_turns += 1

        bank_append = _build_append_192tok_bank(
            base_bank,
            appended_toks_by_page,
            k_exact=pathb.k_exact,
            span_len=pathb.salience_span_len,
            sep_token_id=sep_id,
        )

        # 3. Run Path B hybrid router ONCE on the post-edit bank to lock EXACT 4 page IDs across all 8 arms
        rq_ids = ep.metadata["router_query_ids"].unsqueeze(0).to(device)
        logits = pathb.compute_routing_logits(
            bank_overwrite,
            bge_bf16,
            rq_ids,
            clean_q,
            embed_fn,
            device=device,
            salience_route_weight=salience_route_weight,
        )
        kd = min(4, num_p)
        top_pages = [int(x) for x in torch.argsort(logits, dim=-1, descending=True)[0, :kd].tolist()]
        locked_pages = sorted(top_pages)
        assert len(locked_pages) == kd and len(set(locked_pages)) == kd, f"Invalid locked_pages: {locked_pages}"

        # 4. Evaluate all 2 orders x 4 write policies on the EXACT SAME locked_pages
        for ord_mode in GATE2_ORDER_MODES:
            ordered_pages = order_locked_pages(locked_pages, edited_pages_set, ord_mode)
            assert sorted(ordered_pages) == locked_pages

            ord_idx_t = torch.tensor([ordered_pages], dtype=torch.long, device=device)
            for wp in GATE2_WRITE_POLICIES:
                arm = f"{ord_mode}__{wp}"
                _sync_device(device)
                t_q0 = time.perf_counter()

                if wp == "raw_text_append":
                    ctx_str = "\n\n".join(docs_append[i].strip() for i in ordered_pages) + "\n"
                    pref = embed_fn(_encode_unpadded_rag_ids(ctx_str, tokenizer, vocab_size, device)).to(llm_dtype)
                elif wp == "raw_text_overwrite":
                    ctx_str = "\n\n".join(docs_overwrite[i].strip() for i in ordered_pages) + "\n"
                    pref = embed_fn(_encode_unpadded_rag_ids(ctx_str, tokenizer, vocab_size, device)).to(llm_dtype)
                elif wp == "standalone_append_192tok":
                    pref = bank_append.gather_embeds(
                        ord_idx_t, embed_tokens_fn=embed_fn, hw_coord_embed=None, training_ste=False
                    ).to(llm_dtype)
                else:  # standalone_overwrite_192tok
                    pref = bank_overwrite.gather_embeds(
                        ord_idx_t, embed_tokens_fn=embed_fn, hw_coord_embed=None, training_ste=False
                    ).to(llm_dtype)

                gen_txt = build_prompt_and_generate(
                    llm,
                    pref,
                    clean_q,
                    tokenizer,
                    device,
                    prompt_family=prompt_family,
                    fewshot_prefix=_FAIR_G2_FEWSHOT_PREFIX,
                    max_new_tokens=20,
                )
                _sync_device(device)
                q_ms = (time.perf_counter() - t_q0) * 1000.0

                qm = compute_community_qa_metrics(gen_txt, gold)
                first_line = gen_txt.splitlines()[0] if gen_txt.strip() else ""
                canon_pred = canonicalize_numeric_answer(first_line)
                canon_gold = canonicalize_numeric_answer(gold)
                norm_num_em = 1.0 if (canon_gold and canon_pred == canon_gold) else float(qm["extracted_em"])
                w2n_pred = canonicalize_w2n_answer(first_line)
                w2n_gold = canonicalize_w2n_answer(gold)
                norm_w2n_em = 1.0 if (w2n_gold and w2n_pred == w2n_gold) else norm_num_em

                rec = {
                    "episode_idx": ep_idx,
                    "mode": ep_mode,
                    "query_type": q_type,
                    "gold": gold,
                    "pred": gen_txt,
                    "locked_pages": list(locked_pages),
                    "ordered_pages": list(ordered_pages),
                    "edited_pages": sorted(edited_pages_set),
                    "strict_em": float(qm["extracted_em"]),
                    "normalized_numeric_em": float(norm_num_em),
                    "normalized_w2n_em": float(norm_w2n_em),
                    "f1": float(qm["token_f1"]),
                    "active_tokens": float(pref.shape[1]),
                    "first_query_ms": round(q_ms, 3),
                }
                rows_by_arm[arm].append(rec)
                if ep_mode in rows_by_arm_mode[arm]:
                    rows_by_arm_mode[arm][ep_mode].append(rec)
                if q_type in rows_by_arm_qtype[arm]:
                    rows_by_arm_qtype[arm][q_type].append(rec)

        # Hard invariance check across all 8 arms for this episode
        ref_lp = rows_by_arm[arm_names[0]][ep_idx]["locked_pages"]
        for arm in arm_names[1:]:
            cur_lp = rows_by_arm[arm][ep_idx]["locked_pages"]
            assert cur_lp == ref_lp, f"Locked pages mismatch on ep {ep_idx} arm {arm}: {cur_lp} vs {ref_lp}"

    metric_keys = ("strict_em", "normalized_numeric_em", "normalized_w2n_em", "f1", "active_tokens", "first_query_ms")
    systems_out: dict[str, Any] = {}
    strict_scores: dict[str, list[float]] = {}
    norm_scores: dict[str, list[float]] = {}
    w2n_scores: dict[str, list[float]] = {}

    for arm in arm_names:
        rows = rows_by_arm[arm]
        agg = _summarize_scalar_rows(rows, metric_keys)
        agg["by_mode"] = {m: _summarize_scalar_rows(mr, metric_keys) for m, mr in rows_by_arm_mode[arm].items()}
        agg["by_query_type"] = {
            qt: _summarize_scalar_rows(qr, ("strict_em", "normalized_numeric_em", "normalized_w2n_em"))
            for qt, qr in rows_by_arm_qtype[arm].items()
        }
        agg["episodes"] = rows
        systems_out[arm] = agg
        strict_scores[arm] = [r["strict_em"] for r in rows]
        norm_scores[arm] = [r["normalized_numeric_em"] for r in rows]
        w2n_scores[arm] = [r["normalized_w2n_em"] for r in rows]
        print(
            f"[V32-G2-2x4 {arm:42s}] norm={agg['normalized_numeric_em']:.3f} "
            f"w2n={agg['normalized_w2n_em']:.3f} strict={agg['strict_em']:.3f} "
            f"tok={agg['active_tokens']:.0f} "
            f"(ov={agg['by_query_type']['accepted_override']['normalized_numeric_em']:.2f}, "
            f"rej={agg['by_query_type']['rejected_proposal']['normalized_numeric_em']:.2f}, "
            f"nb={agg['by_query_type']['neighbor_preservation']['normalized_numeric_em']:.2f}, "
            f"st={agg['by_query_type']['randomized_static_retention']['normalized_numeric_em']:.2f})",
            flush=True,
        )

    # 2x4 Marginal & Interaction summary
    marginals_by_policy = {
        wp: round(
            sum(systems_out[f"{om}__{wp}"]["normalized_numeric_em"] for om in GATE2_ORDER_MODES) / len(GATE2_ORDER_MODES),
            4,
        )
        for wp in GATE2_WRITE_POLICIES
    }
    marginals_by_order = {
        om: round(
            sum(systems_out[f"{om}__{wp}"]["normalized_numeric_em"] for wp in GATE2_WRITE_POLICIES) / len(GATE2_WRITE_POLICIES),
            4,
        )
        for om in GATE2_ORDER_MODES
    }

    key_pairs = [
        # Overwrite vs Append (within each locked order, for BOTH 192-tok Path B and raw text)
        ("edited_last__standalone_overwrite_192tok", "edited_last__standalone_append_192tok"),
        ("page_id_asc__standalone_overwrite_192tok", "page_id_asc__standalone_append_192tok"),
        ("edited_last__raw_text_overwrite", "edited_last__raw_text_append"),
        ("page_id_asc__raw_text_overwrite", "page_id_asc__raw_text_append"),
        # Path B 192-tok Overwrite vs Strong Raw-Text Overwrite & Append (within each locked order)
        ("edited_last__standalone_overwrite_192tok", "edited_last__raw_text_overwrite"),
        ("page_id_asc__standalone_overwrite_192tok", "page_id_asc__raw_text_overwrite"),
        ("edited_last__standalone_overwrite_192tok", "edited_last__raw_text_append"),
        ("page_id_asc__standalone_overwrite_192tok", "page_id_asc__raw_text_append"),
        # Order effect (`edited_last` vs `page_id_asc`) for each write policy
        ("edited_last__standalone_overwrite_192tok", "page_id_asc__standalone_overwrite_192tok"),
        ("edited_last__standalone_append_192tok", "page_id_asc__standalone_append_192tok"),
        ("edited_last__raw_text_overwrite", "page_id_asc__raw_text_overwrite"),
        ("edited_last__raw_text_append", "page_id_asc__raw_text_append"),
    ]

    total_acc = gate_tp + gate_fn
    total_rej = gate_tn + gate_fp
    return {
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "num_episodes": len(all_eps),
        "seed": int(seed),
        "locked_pages_invariant_verified": True,
        "orders": list(GATE2_ORDER_MODES),
        "write_policies": list(GATE2_WRITE_POLICIES),
        "num_unique_clusters": len(set(cluster_ids)),
        "cluster_definition": "(mode, query_type, gold)",
        "shared_gate_telemetry": {
            "tp": gate_tp,
            "tn": gate_tn,
            "fp": gate_fp,
            "fn": gate_fn,
            "false_accept_rate": round(gate_fp / max(1, total_rej), 4),
            "false_reject_rate": round(gate_fn / max(1, total_acc), 4),
            "gate_accuracy": round((gate_tp + gate_tn) / max(1, total_acc + total_rej), 4),
        },
        "raw_text_overwrite_telemetry": {
            "accepted_turns": raw_overwrite_total_accepted_turns,
            "in_place_replaced_turns": raw_overwrite_replaced_turns,
            "in_place_replace_rate": round(
                raw_overwrite_replaced_turns / max(1, raw_overwrite_total_accepted_turns), 4
            ),
        },
        "factorial_marginals_norm_em": {
            "by_write_policy": marginals_by_policy,
            "by_order": marginals_by_order,
        },
        "systems": systems_out,
        "pairwise_norm_em": compute_pairwise_with_cluster_and_holm(
            norm_scores, key_pairs, cluster_ids=cluster_ids, n_boot=n_boot, seed=seed
        ),
        "pairwise_w2n_em": compute_pairwise_with_cluster_and_holm(
            w2n_scores, key_pairs, cluster_ids=cluster_ids, n_boot=n_boot, seed=seed
        ),
        "pairwise_strict_em": compute_pairwise_with_cluster_and_holm(
            strict_scores, key_pairs, cluster_ids=cluster_ids, n_boot=n_boot, seed=seed
        ),
    }


# ---------------------------------------------------------------------------
# Control #2: Full LongMemEval `knowledge-update` Subset (~78 questions)
# ---------------------------------------------------------------------------


def load_longmemeval_knowledge_update(
    repo_id: str = "xiaowu0162/longmemeval-cleaned",
    filename: str = "longmemeval_s_cleaned.json",
    local_path: str = "",
    *,
    max_items: int = 0,
    seed: int = 2026,
    allow_fallback: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load strictly `question_type == 'knowledge-update'` items (ZERO static categories mixed in)."""
    raw_items: list[dict[str, Any]] | None = None
    meta: dict[str, Any] = {
        "repo_id": repo_id,
        "filename": filename,
        "filter_question_type": "knowledge-update",
        "exclude_abstention_suffix": "_abs",
    }

    if local_path and Path(local_path).is_file():
        raw_items = json.loads(Path(local_path).read_text(encoding="utf-8"))
        meta["dataset_source"] = "local_json"
        meta["resolved_path"] = str(Path(local_path).resolve())
    elif allow_fallback:
        raw_items = _build_verified_local_longmemeval_fallback(seed=seed)
        meta["dataset_source"] = "verified_local_fallback"
        meta["fallback_reason"] = "explicit_allow_fallback_mock"
    else:
        load_err: str | None = None
        try:
            from huggingface_hub import hf_hub_download

            dl_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
            raw_items = json.loads(Path(dl_path).read_text(encoding="utf-8"))
            meta["dataset_source"] = f"hf_hub:{repo_id}/{filename}"
            meta["resolved_path"] = str(dl_path)
        except Exception as exc_hf:
            load_err = f"hf_hub_download failed: {exc_hf}"
            cache_dir = Path("/tmp/longmemeval_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached_file = cache_dir / filename
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
            try:
                if not cached_file.is_file() or cached_file.stat().st_size == 0:
                    urllib.request.urlretrieve(url, str(cached_file))
                raw_items = json.loads(cached_file.read_text(encoding="utf-8"))
                meta["dataset_source"] = f"hf_https:{url}"
                meta["resolved_path"] = str(cached_file)
            except Exception as exc_url:
                raise RuntimeError(
                    f"Failed to download LongMemEval ({load_err}; urlretrieve failed: {exc_url}). "
                    "Pass --longmemeval-local-json or run with --mock-llm."
                ) from exc_url

    assert raw_items is not None
    meta["raw_total_items"] = len(raw_items)

    ku_items = [
        x
        for x in raw_items
        if str(x.get("question_type", "")) == "knowledge-update"
        and not str(x.get("question_id", "")).endswith("_abs")
    ]
    ku_items.sort(key=lambda x: str(x.get("question_id", "")))
    if max_items > 0:
        ku_items = ku_items[: int(max_items)]

    assert all(str(x.get("question_type", "")) == "knowledge-update" for x in ku_items), (
        "Non-knowledge-update item leaked into knowledge-update loader!"
    )
    meta["num_episodes"] = len(ku_items)
    meta["question_types_present"] = sorted({str(x.get("question_type", "")) for x in ku_items})
    return ku_items, meta


@torch.no_grad()
def run_longmemeval_knowledge_update(
    llm: nn.Module,
    pathb: PathBDiscreteMemory,
    tokenizer,
    device: torch.device,
    *,
    prompt_family: str,
    repo_id: str = "xiaowu0162/longmemeval-cleaned",
    filename: str = "longmemeval_s_cleaned.json",
    local_path: str = "",
    max_pages: int = 64,
    max_items: int = 0,
    salience_route_weight: float = 1.5,
    n_boot: int = 2000,
    seed: int = 2026,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Evaluate all 4 arms on the full LongMemEval `knowledge-update` subset (~78 items)."""
    t_start = time.perf_counter()
    embed_fn = llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    special_ids = _get_special_token_ids(tokenizer)
    pathb.eval()

    eff_max_items = 4 if is_mock else int(max_items)
    eff_max_pages = 16 if is_mock else int(max_pages)
    items, ds_meta = load_longmemeval_knowledge_update(
        repo_id=repo_id,
        filename=filename,
        local_path=local_path,
        max_items=eff_max_items,
        seed=seed,
        allow_fallback=is_mock,
    )
    packing_mode = f"gold_prioritized_{eff_max_pages}p" if eff_max_pages > 0 else "full_untruncated"

    systems = (
        "bm25_rag_k2",
        "bge_4span_rag_k2",
        "standalone_discrete_k48_d4",
        "standalone_discrete_k48_d6",
    )
    rows_by_sys: dict[str, list[dict[str, Any]]] = {s: [] for s in systems}
    em_by_sys: dict[str, list[float]] = {s: [] for s in systems}
    sub_by_sys: dict[str, list[float]] = {s: [] for s in systems}
    f1_by_sys: dict[str, list[float]] = {s: [] for s in systems}
    val_em_by_sys: dict[str, list[float]] = {s: [] for s in systems}

    for item_idx, item in enumerate(items):
        qid = str(item.get("question_id", f"ku_{item_idx}"))
        qtype = str(item.get("question_type", "knowledge-update"))
        assert qtype == "knowledge-update", f"Unexpected question_type: {qtype!r}"
        question = str(item["question"])
        gold = str(item["answer"])

        pack = _pack_longmemeval_sessions(item, tokenizer, pathb.cfg.page_size, max_pages=eff_max_pages)
        p_t = pack["page_tensor"].to(device)
        page_texts: list[str] = pack["page_texts"]
        page_sids: list[str] = pack["page_session_ids"]
        ans_sids: set[str] = pack["answer_session_ids"]
        rq_ids = torch.tensor(
            _encode_text_or_mock(question, tokenizer, vocab_size), dtype=torch.long, device=device
        ).unsqueeze(0)

        # Encode Path B bank & shared BF16 BGE keys ONCE per conversation
        enc_p = pathb.encode_book(
            p_t,
            embed_fn,
            tokenizer,
            vocab_size,
            device=device,
            amp_dtype=amp_dtype,
            autocast_enabled=autocast_enabled,
            special_token_ids=special_ids,
            num_spans=4,
        )
        bge_k_bf16 = enc_p["bge_keys_bf16"]
        pb_bank = enc_p["bank"]

        for sys_name in systems:
            ans_in_bank: float | None = None
            if sys_name == "bm25_rag_k2":
                bm25 = BM25PageRetriever(page_texts)
                rr = bm25.route(question, k_deep=min(2, len(page_texts)), k_skim=0, device=device)
                chosen = rr.deep_idx[0].tolist()
                chosen_texts = [page_texts[i] for i in chosen]
                pref = embed_fn(
                    _encode_unpadded_rag_ids("\n\n".join(t.strip() for t in chosen_texts) + "\n", tokenizer, vocab_size, device)
                ).to(llm_dtype)

            elif sys_name == "bge_4span_rag_k2":
                if pathb.bge_toc is not None and bge_k_bf16 is not None:
                    sc = pathb.bge_toc.score_book(pathb.bge_toc.encode_texts([question], is_query=True), bge_k_bf16)[0]
                    chosen = torch.argsort(sc, descending=True)[: min(2, len(page_texts))].tolist()
                else:
                    chosen = list(range(min(2, len(page_texts))))
                chosen_texts = [page_texts[i] for i in chosen]
                pref = embed_fn(
                    _encode_unpadded_rag_ids("\n\n".join(t.strip() for t in chosen_texts) + "\n", tokenizer, vocab_size, device)
                ).to(llm_dtype)

            else:  # standalone_discrete_k48_d4 or standalone_discrete_k48_d6
                kd_use = 6 if sys_name == "standalone_discrete_k48_d6" else 4
                r_out, pref, _ = pathb.route_and_assemble(
                    pb_bank,
                    bge_k_bf16,
                    rq_ids,
                    question,
                    embed_fn,
                    device=device,
                    k_deep=min(kd_use, int(p_t.shape[1])),
                    salience_route_weight=salience_route_weight,
                )
                pref = pref.to(llm_dtype)
                chosen = r_out.deep_idx[0].tolist()
                chosen_texts = [page_texts[i] for i in chosen]
                bank_decoded = [
                    tokenizer.decode([int(t) for t in pb_bank.token_ids[0, i].tolist() if int(t) >= 0], skip_special_tokens=True)
                    for i in chosen
                ]
                ans_in_bank = _answer_in_chosen_pages(gold, bank_decoded)

            chosen_sids = {page_sids[i] for i in chosen if 0 <= i < len(page_sids)}
            sess_hit = float(bool(ans_sids & chosen_sids)) if ans_sids else None
            ans_in_pages = _answer_in_chosen_pages(gold, chosen_texts)
            gen_txt = build_prompt_and_generate(
                llm,
                pref,
                question,
                tokenizer,
                device,
                prompt_family=prompt_family,
                fewshot_prefix="",
                max_new_tokens=32,
            )
            qm = compute_community_qa_metrics(gen_txt, gold)
            loc_val_em = float(compute_value_em(gen_txt, gold))

            rec = {
                "question_id": qid,
                "question_type": qtype,
                "question": question,
                "gold": gold,
                "pred": gen_txt,
                "chosen_pages": [int(i) for i in chosen],
                "extracted_em": float(qm["extracted_em"]),
                "substring_match": float(qm["substring_match"]),
                "token_f1": float(qm["token_f1"]),
                "local_value_em": loc_val_em,
                "value_em": loc_val_em,
                "session_hit_at_k": sess_hit,
                "answer_substr_in_chosen_pages_rate": ans_in_pages,
                "answer_substr_in_exact_bank_rate": ans_in_bank,
                "active_prefix_tokens": float(pref.shape[1]),
            }
            rows_by_sys[sys_name].append(rec)
            em_by_sys[sys_name].append(float(qm["extracted_em"]))
            sub_by_sys[sys_name].append(float(qm["substring_match"]))
            f1_by_sys[sys_name].append(float(qm["token_f1"]))
            val_em_by_sys[sys_name].append(loc_val_em)

    m_keys = (
        "extracted_em",
        "substring_match",
        "token_f1",
        "local_value_em",
        "session_hit_at_k",
        "answer_substr_in_chosen_pages_rate",
        "answer_substr_in_exact_bank_rate",
        "active_prefix_tokens",
    )
    sys_summary: dict[str, Any] = {}
    for sys_name in systems:
        agg = _summarize_scalar_rows(rows_by_sys[sys_name], m_keys)
        agg["episodes"] = rows_by_sys[sys_name]
        sys_summary[sys_name] = agg
        print(
            f"[V32-LME-KU N={len(items)} {sys_name:28s}] EM={agg['extracted_em']:.3f} "
            f"Sub={agg['substring_match']:.3f} F1={agg['token_f1']:.3f} ValEM={agg['local_value_em']:.3f} "
            f"hit={agg.get('session_hit_at_k', 0.0):.3f} pg={agg.get('answer_substr_in_chosen_pages_rate', 0.0):.3f} "
            f"bk={agg.get('answer_substr_in_exact_bank_rate', -1.0):.3f} tok={agg['active_prefix_tokens']:.0f}",
            flush=True,
        )

    pairs_lme = [
        ("bge_4span_rag_k2", "standalone_discrete_k48_d4"),
        ("bge_4span_rag_k2", "standalone_discrete_k48_d6"),
        ("bm25_rag_k2", "standalone_discrete_k48_d4"),
        ("bm25_rag_k2", "standalone_discrete_k48_d6"),
        ("standalone_discrete_k48_d6", "standalone_discrete_k48_d4"),
        ("bge_4span_rag_k2", "bm25_rag_k2"),
    ]
    return {
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "packing_mode": packing_mode,
        "uses_gold_prioritized_64p": packing_mode == "gold_prioritized_64p",
        "max_pages": eff_max_pages,
        "num_episodes": len(items),
        "dataset": ds_meta,
        "systems": sys_summary,
        "pairwise_extracted_em": compute_pairwise_with_cluster_and_holm(em_by_sys, pairs_lme, n_boot=n_boot, seed=seed),
        "pairwise_substring_match": compute_pairwise_with_cluster_and_holm(sub_by_sys, pairs_lme, n_boot=n_boot, seed=seed),
        "pairwise_local_value_em": compute_pairwise_with_cluster_and_holm(val_em_by_sys, pairs_lme, n_boot=n_boot, seed=seed),
        "pairwise_token_f1": compute_pairwise_with_cluster_and_holm(f1_by_sys, pairs_lme, n_boot=n_boot, seed=seed),
    }


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="v3.2 TMLR Micro-Controls: Gate-2 2x4 Locked Factorial + Full LongMemEval knowledge-update."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--pathb-checkpoint", type=str, default="")
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated: gate2_factorial_2x4,longmemeval_knowledge_update or 'all'.",
    )
    parser.add_argument("--gate2-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--bge-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--k-exact", type=int, default=48)
    parser.add_argument("--salience-span-len", type=int, default=16)
    parser.add_argument("--salience-route-weight", type=float, default=1.5)
    parser.add_argument("--longmemeval-repo", type=str, default="xiaowu0162/longmemeval-cleaned")
    parser.add_argument("--longmemeval-file", type=str, default="longmemeval_s_cleaned.json")
    parser.add_argument("--longmemeval-local-json", type=str, default="")
    parser.add_argument("--longmemeval-max-pages", type=int, default=64)
    parser.add_argument("--longmemeval-max-items", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--git-sha", type=str, default="")
    parser.add_argument("--out-json", type=str, default="/content/lslc_results/v32_tmlr_controls.json")
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
    all_set = {"gate2_factorial_2x4", "longmemeval_knowledge_update"}
    run_tasks = all_set if "all" in raw_tasks else raw_tasks

    if args.mock_llm:
        vocab_size = 2048
        cfg = LSLCConfig.tiny_mock(
            model_name=f"mock-{args.model}",
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
        resolved_model_id = cfg.model_name
        load_attempts = [{"model_id": resolved_model_id, "status": "mock"}]
        prompt_family = "llama-3.1-instruct" if is_llama_model_name(args.model) else "qwen3-chatml"
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
        llm, tokenizer, resolved_model_id, load_attempts, prompt_family = load_frozen_backbone_for_v32(
            args.model, cfg, device
        )
        _, val_pool, data_source = _load_wikitext_pools(tokenizer)

    bge_toc = _PlugInBiEncoderTOC(args.bge_model, device=device, is_mock=args.mock_llm)
    pathb = PathBDiscreteMemory(
        cfg,
        bge_toc=bge_toc,
        bank_write_threshold=0.45,
        edit_gate_hidden_dim=64,
        span_aligned_overwrite=True,
        k_exact=cfg.k_exact,
        salience_span_len=cfg.salience_span_len,
    ).to(device)

    ckpt_info: dict[str, Any] | None = None
    checkpoint_sha256: str | None = None
    if args.pathb_checkpoint:
        ckpt_info = load_pathb_checkpoint(pathb, args.pathb_checkpoint)
        checkpoint_sha256 = str(ckpt_info["sha256"])
        assert not ckpt_info["missing_keys"] and not ckpt_info["unexpected_keys"], f"Bad Path B ckpt: {ckpt_info}"

    git_prov = resolve_git_provenance(args.git_sha, REPO_ROOT)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": "3.2.0-tmlr-micro-controls",
        "git_sha": git_prov["git_sha"],
        "git_dirty": git_prov["git_dirty"],
        "git_sha_source": git_prov["git_sha_source"],
        "checkpoint_sha256": checkpoint_sha256,
        "pathb_checkpoint": ckpt_info,
        "trainable_param_breakdown": pathb.trainable_param_breakdown(llm),
        "storage_bytes": pathb.storage_breakdown_bytes_per_page(num_spans=4),
        "requested_model": args.model,
        "resolved_model_id": resolved_model_id,
        "model_load_attempts": load_attempts,
        "prompt_family": prompt_family,
        "mock_llm": args.mock_llm,
        "device": str(device),
        "data_source": data_source,
        "config": {
            "d_llm": cfg.d_llm,
            "num_llm_layers": cfg.num_llm_layers,
            "contextual_edit_layer": cfg.contextual_edit_layer,
            "page_size": cfg.page_size,
            "k_exact": cfg.k_exact,
            "salience_span_len": cfg.salience_span_len,
            "salience_route_weight": args.salience_route_weight,
            "bge_model": bge_toc.model_name,
            "n_boot": int(args.n_boot),
            "seed": int(args.seed),
        },
        "gate2_factorial_2x4": None,
        "longmemeval_knowledge_update": None,
    }

    if "gate2_factorial_2x4" in run_tasks:
        payload["gate2_factorial_2x4"] = run_gate2_factorial_2x4(
            llm,
            pathb,
            tokenizer,
            val_pool,
            device,
            prompt_family=prompt_family,
            eval_n=args.gate2_n,
            seed=args.seed,
            salience_route_weight=args.salience_route_weight,
            n_boot=args.n_boot,
            is_mock=args.mock_llm,
        )
        _atomic_write_json(out_path, payload)

    if "longmemeval_knowledge_update" in run_tasks:
        payload["longmemeval_knowledge_update"] = run_longmemeval_knowledge_update(
            llm,
            pathb,
            tokenizer,
            device,
            prompt_family=prompt_family,
            repo_id=args.longmemeval_repo,
            filename=args.longmemeval_file,
            local_path=args.longmemeval_local_json,
            max_pages=args.longmemeval_max_pages,
            max_items=args.longmemeval_max_items,
            salience_route_weight=args.salience_route_weight,
            n_boot=args.n_boot,
            seed=args.seed,
            is_mock=args.mock_llm,
        )
        _atomic_write_json(out_path, payload)

    print(f"[DONE] Wrote v3.2 TMLR micro-control results -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
