#!/usr/bin/env python3
"""v3.3 TMLR Control: Update-Aware Prompt Ablation for Append vs. Overwrite (Zero Retraining).

Evaluates whether giving append-only baselines an explicit recency-priority instruction
("If multiple updates or conflicting values appear for the same project, later updates
supersede earlier records; always answer using the latest approved update.") closes the
overwrite-vs-append gap on LivingBook-Edit (`gate2_factorial_2x4`).

Protocol (100% locked to v3.2 factorial episodes, seeds, checkpoints, and page selections):
  - N=200 hold-out episodes (100 budget k=3 + 100 symbolic k=5; seed=2029 Qwen3, seed=2026 Llama-8B).
  - Loads the exact locked Path B checkpoint (`lslc_pathb_qwen3_1.7b.safetensors` or
    `lslc_pathb_llama31_8b.safetensors`) onto strictly frozen BF16 backbones (`r=0`).
  - Runs Path B hybrid router ONCE on the post-edit bank to lock 4 page IDs (`locked_pages`),
    and optionally hard-verifies 100% exact parity against `--ref-v32-json` episode by episode.
  - Evaluates 2 locked page orders (`edited_last`, `page_id_asc`) x 3 write policies
    (`raw_text_append`, `standalone_append_192tok`, `standalone_overwrite_192tok`) under the
    EXACT SAME update-aware prompt (`_UPDATE_AWARE_G2_FEWSHOT_PREFIX`), eliminating any
    prompt asymmetry between append and overwrite.
  - Computes exact two-sided McNemar p-values (unrounded + Holm-Bonferroni adjusted),
    cluster-bootstrap 95% CIs (`(mode, query_type, gold)`, B=2000, seed=2026), exact DP
    cluster-level sign-flip p-values, and paired deltas vs. baseline v3.2 prompt arms.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from colab_run_m2 import _load_wikitext_pools  # noqa: E402
from colab_run_v3_gerbang3_coding import _MockCodeTokenizer  # noqa: E402
from colab_run_v3_gerbang4_community import (  # noqa: E402
    _PlugInBiEncoderTOC,
    _encode_unpadded_rag_ids,
    _get_special_token_ids,
    _sync_device,
    compute_community_qa_metrics,
)
from colab_run_v3_gerbang5_ablations import canonicalize_numeric_answer  # noqa: E402
from colab_run_v3_path_tiebreaker import (  # noqa: E402
    _build_append_192tok_bank,
    _summarize_scalar_rows,
)
from colab_run_v32_tmlr_controls import (  # noqa: E402
    GATE2_ORDER_MODES,
    _atomic_write_json,
    build_prompt_and_generate,
    compute_pairwise_with_cluster_and_holm,
    is_llama_model_name,
    load_frozen_backbone_for_v32,
    order_locked_pages,
    resolve_git_provenance,
)
from lslc import (  # noqa: E402
    AdversarialLivingBookDataset,
    LSLCConfig,
    PathBDiscreteMemory,
    SparseExactBank,
    TinyMockCausalLM,
    canonicalize_w2n_answer,
    get_newline_sep_token_id,
    load_pathb_checkpoint,
    select_amp_dtype,
)


UPDATE_AWARE_G2_FEWSHOT_PREFIX: str = (
    "Format instructions: Reply with ONLY the concise exact value (matching unit format). "
    "If multiple updates or conflicting values appear for the same project, later updates "
    "supersede earlier records; always answer using the latest approved update.\n"
    "Example 1 -> Question: What is the approved budget for DemoA? Answer: 40M\n"
    "Example 2 -> Question: What is the approved deployment state for DemoB? Answer: blue-green-prod\n\n"
)

V33_WRITE_POLICIES: tuple[str, ...] = (
    "raw_text_append",
    "standalone_append_192tok",
    "standalone_overwrite_192tok",
)


def exact_mcnemar_unrounded(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def exact_cluster_signflip_p(scores_a: list[float], scores_b: list[float], cluster_ids: list[str]) -> float:
    """Exact two-sided DP cluster-sum sign-flip p-value."""
    clusters: dict[str, int] = {}
    for a, b, cid in zip(scores_a, scores_b, cluster_ids):
        clusters[cid] = clusters.get(cid, 0) + (int(round(a)) - int(round(b)))
    nz = [v for v in clusters.values() if v != 0]
    if not nz:
        return 1.0
    obs = abs(sum(nz))
    dp: dict[int, int] = {0: 1}
    for v in nz:
        ndp: dict[int, int] = {}
        for s, cnt in dp.items():
            ndp[s + v] = ndp.get(s + v, 0) + cnt
            ndp[s - v] = ndp.get(s - v, 0) + cnt
        dp = ndp
    tot = 2 ** len(nz)
    return sum(cnt for s, cnt in dp.items() if abs(s) >= obs) / tot


def holm_unrounded(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda x: (x[1], x[0]))
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for idx, (k, p) in enumerate(items):
        running = max(running, min(1.0, p * (m - idx)))
        out[k] = running
    return out


def enrich_pairwise_exact_stats(
    pw: dict[str, Any],
    scores_by_sys: dict[str, list[float]],
    pairs: list[tuple[str, str]],
    cluster_ids: list[str],
) -> dict[str, Any]:
    """Attach unrounded exact McNemar p, unrounded Holm p, and exact cluster sign-flip p (raw + Holm)."""
    raw_ep: dict[str, float] = {}
    raw_sf: dict[str, float] = {}
    for a_name, b_name in pairs:
        key = f"{a_name}__vs__{b_name}"
        if key not in pw:
            continue
        sa, sb = scores_by_sys[a_name], scores_by_sys[b_name]
        na = sum(1 for x, y in zip(sa, sb) if x > 0.5 and y <= 0.5)
        nb = sum(1 for x, y in zip(sa, sb) if y > 0.5 and x <= 0.5)
        raw_ep[key] = exact_mcnemar_unrounded(na, nb)
        raw_sf[key] = exact_cluster_signflip_p(sa, sb, cluster_ids)
    h_ep = holm_unrounded(raw_ep)
    h_sf = holm_unrounded(raw_sf)
    for key in raw_ep:
        pw[key]["exact_unrounded"] = {
            "p_mcnemar_raw": raw_ep[key],
            "p_mcnemar_holm": h_ep[key],
            "p_cluster_signflip_raw": raw_sf[key],
            "p_cluster_signflip_holm": h_sf[key],
        }
    return pw


@torch.no_grad()
def run_update_aware_append_control(
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
    ref_v32_factorial: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run 2x3 update-aware prompt control with 100% locked pages matching v3.2 factorial."""
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

    arm_names = [f"{ord_m}__{wp}" for ord_m in GATE2_ORDER_MODES for wp in V33_WRITE_POLICIES]
    q_types_list = ("accepted_override", "rejected_proposal", "neighbor_preservation", "randomized_static_retention")

    rows_by_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in arm_names}
    rows_by_arm_mode: dict[str, dict[str, list[dict[str, Any]]]] = {a: {"budget": [], "symbolic": []} for a in arm_names}
    rows_by_arm_qtype: dict[str, dict[str, list[dict[str, Any]]]] = {
        a: {qt: [] for qt in q_types_list} for a in arm_names
    }
    cluster_ids: list[str] = []

    gate_tp = gate_tn = gate_fp = gate_fn = 0
    ref_locked_mismatches = 0
    ref_ordered_mismatches = 0

    for ep_idx, ep in enumerate(all_eps):
        clean_q = tokenizer.decode(ep.metadata["router_query_ids"].tolist(), skip_special_tokens=True).strip()
        gold = str(ep.metadata["answer_str"]).strip()
        ep_mode = str(ep.metadata.get("mode", "budget"))
        q_type = str(ep.metadata.get("query_type", ""))
        gold_flags = list(ep.metadata.get("edit_accepted_flags", []))
        cluster_ids.append(f"{ep_mode}|{q_type}|{gold}")

        p_t = ep.context_pages.unsqueeze(0).to(device)
        page_texts = [tokenizer.decode(ep.context_pages[i].tolist(), skip_special_tokens=True) for i in range(num_p)]

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
        edited_pages_set: set[int] = set()

        for (p_i, e_ids_cpu), is_acc_gold in zip(ep.edit_pages, gold_flags):
            e_ids = e_ids_cpu.unsqueeze(0).to(device)
            e_txt = tokenizer.decode(e_ids_cpu.tolist(), skip_special_tokens=True).strip()

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
                buf = appended_toks_by_page.setdefault(int(p_i), [])
                if buf and sep_id is not None and int(sep_id) >= 0:
                    buf.append(int(sep_id))
                buf.extend([int(t) for t in e_ids_cpu.tolist()])
                docs_append[int(p_i)] = docs_append[int(p_i)].rstrip() + f"\nUpdate: {e_txt}\n"

        bank_append = _build_append_192tok_bank(
            base_bank,
            appended_toks_by_page,
            k_exact=pathb.k_exact,
            span_len=pathb.salience_span_len,
            sep_token_id=sep_id,
        )

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

        # If a reference v3.2 factorial JSON is provided, verify exact locked_pages match
        if ref_v32_factorial is not None and not is_mock:
            ref_ep = ref_v32_factorial["systems"]["edited_last__standalone_overwrite_192tok"]["episodes"][ep_idx]
            if list(ref_ep["locked_pages"]) != locked_pages:
                ref_locked_mismatches += 1
                # Lock strictly to reference pages if any floating-point tie broke differently
                locked_pages = list(ref_ep["locked_pages"])

        for ord_mode in GATE2_ORDER_MODES:
            ordered_pages = order_locked_pages(locked_pages, edited_pages_set, ord_mode)
            assert sorted(ordered_pages) == locked_pages
            if ref_v32_factorial is not None and not is_mock:
                ref_arm = f"{ord_mode}__standalone_overwrite_192tok"
                ref_ord = list(ref_v32_factorial["systems"][ref_arm]["episodes"][ep_idx]["ordered_pages"])
                if ref_ord != ordered_pages:
                    ref_ordered_mismatches += 1

            ord_idx_t = torch.tensor([ordered_pages], dtype=torch.long, device=device)
            for wp in V33_WRITE_POLICIES:
                arm = f"{ord_mode}__{wp}"
                _sync_device(device)
                t_q0 = time.perf_counter()

                if wp == "raw_text_append":
                    ctx_str = "\n\n".join(docs_append[i].strip() for i in ordered_pages) + "\n"
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
                    fewshot_prefix=UPDATE_AWARE_G2_FEWSHOT_PREFIX,
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

        if (ep_idx + 1) % 50 == 0 or (ep_idx + 1) == len(all_eps):
            print(f"[v3.3] Progress {ep_idx + 1}/{len(all_eps)} ({time.perf_counter() - t_start:.1f}s)", flush=True)

    metric_keys = ("strict_em", "normalized_numeric_em", "normalized_w2n_em", "f1", "active_tokens", "first_query_ms")
    systems_out: dict[str, Any] = {}
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
        norm_scores[arm] = [r["normalized_numeric_em"] for r in rows]
        w2n_scores[arm] = [r["normalized_w2n_em"] for r in rows]
        print(
            f"[V33-UPDATE-AWARE {arm:42s}] norm={agg['normalized_numeric_em']:.3f} "
            f"w2n={agg['normalized_w2n_em']:.3f} tok={agg['active_tokens']:.1f} q_ms={agg['first_query_ms']:.1f}",
            flush=True,
        )

    # Pre-specified within-prompt contrasts (m=6 per metric)
    within_pairs = [
        ("edited_last__standalone_overwrite_192tok", "edited_last__standalone_append_192tok"),
        ("page_id_asc__standalone_overwrite_192tok", "page_id_asc__standalone_append_192tok"),
        ("edited_last__standalone_overwrite_192tok", "edited_last__raw_text_append"),
        ("page_id_asc__standalone_overwrite_192tok", "page_id_asc__raw_text_append"),
        ("edited_last__standalone_append_192tok", "edited_last__raw_text_append"),
        ("page_id_asc__standalone_append_192tok", "page_id_asc__raw_text_append"),
    ]
    pw_norm = compute_pairwise_with_cluster_and_holm(
        norm_scores, within_pairs, cluster_ids=cluster_ids, n_boot=n_boot, seed=2026
    )
    enrich_pairwise_exact_stats(pw_norm, norm_scores, within_pairs, cluster_ids)
    pw_w2n = compute_pairwise_with_cluster_and_holm(
        w2n_scores, within_pairs, cluster_ids=cluster_ids, n_boot=n_boot, seed=2026
    )
    enrich_pairwise_exact_stats(pw_w2n, w2n_scores, within_pairs, cluster_ids)

    # Cross-prompt comparison against baseline v3.2 factorial (if supplied)
    vs_baseline_norm: dict[str, Any] = {}
    vs_baseline_w2n: dict[str, Any] = {}
    if ref_v32_factorial is not None and not is_mock:
        ref_sys = ref_v32_factorial["systems"]
        combined_norm = {f"ua__{k}": v for k, v in norm_scores.items()}
        combined_w2n = {f"ua__{k}": v for k, v in w2n_scores.items()}
        for arm in arm_names:
            combined_norm[f"base__{arm}"] = [float(r["normalized_numeric_em"]) for r in ref_sys[arm]["episodes"]]
            combined_w2n[f"base__{arm}"] = [float(r["normalized_w2n_em"]) for r in ref_sys[arm]["episodes"]]
        for ord_m in GATE2_ORDER_MODES:
            rw = f"{ord_m}__raw_text_overwrite"
            combined_norm[f"base__{rw}"] = [float(r["normalized_numeric_em"]) for r in ref_sys[rw]["episodes"]]
            combined_w2n[f"base__{rw}"] = [float(r["normalized_w2n_em"]) for r in ref_sys[rw]["episodes"]]

        cross_pairs = [
            # Lift from adding update-aware prompt to each policy
            (f"ua__{arm}", f"base__{arm}") for arm in arm_names
        ] + [
            # Baseline OW-192 vs Update-Aware Append policies
            ("base__edited_last__standalone_overwrite_192tok", "ua__edited_last__standalone_append_192tok"),
            ("base__page_id_asc__standalone_overwrite_192tok", "ua__page_id_asc__standalone_append_192tok"),
            ("base__edited_last__standalone_overwrite_192tok", "ua__edited_last__raw_text_append"),
            ("base__page_id_asc__standalone_overwrite_192tok", "ua__page_id_asc__raw_text_append"),
            # Baseline Raw-OW vs Update-Aware Raw-App
            ("base__edited_last__raw_text_overwrite", "ua__edited_last__raw_text_append"),
            ("base__page_id_asc__raw_text_overwrite", "ua__page_id_asc__raw_text_append"),
        ]
        vs_baseline_norm = compute_pairwise_with_cluster_and_holm(
            combined_norm, cross_pairs, cluster_ids=cluster_ids, n_boot=n_boot, seed=2026
        )
        enrich_pairwise_exact_stats(vs_baseline_norm, combined_norm, cross_pairs, cluster_ids)
        vs_baseline_w2n = compute_pairwise_with_cluster_and_holm(
            combined_w2n, cross_pairs, cluster_ids=cluster_ids, n_boot=n_boot, seed=2026
        )
        enrich_pairwise_exact_stats(vs_baseline_w2n, combined_w2n, cross_pairs, cluster_ids)

    return {
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "num_episodes": len(all_eps),
        "seed": int(seed),
        "fewshot_prefix": UPDATE_AWARE_G2_FEWSHOT_PREFIX,
        "orders": list(GATE2_ORDER_MODES),
        "write_policies": list(V33_WRITE_POLICIES),
        "num_unique_clusters": len(set(cluster_ids)),
        "cluster_definition": "(mode, query_type, gold)",
        "ref_v32_locked_pages_mismatches": int(ref_locked_mismatches),
        "ref_v32_ordered_pages_mismatches": int(ref_ordered_mismatches),
        "shared_gate_telemetry": {
            "tp": gate_tp,
            "tn": gate_tn,
            "fp": gate_fp,
            "fn": gate_fn,
            "far": round(gate_fp / max(1, gate_fp + gate_tn), 6),
            "frr": round(gate_fn / max(1, gate_fn + gate_tp), 6),
        },
        "systems": systems_out,
        "pairwise_norm_em": pw_norm,
        "pairwise_w2n_em": pw_w2n,
        "vs_baseline_v32_norm_em": vs_baseline_norm,
        "vs_baseline_v32_w2n_em": vs_baseline_w2n,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v3.3 TMLR Control: Update-Aware Append Prompt Ablation")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--pathb-checkpoint", type=str, default="")
    p.add_argument("--ref-v32-json", type=str, default="")
    p.add_argument("--bge-model", type=str, default="BAAI/bge-small-en-v1.5")
    p.add_argument("--gate2-n", type=int, default=200)
    p.add_argument("--seed", type=int, default=2029)
    p.add_argument("--k-exact", type=int, default=48)
    p.add_argument("--salience-span-len", type=int, default=16)
    p.add_argument("--salience-route-weight", type=float, default=1.5)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--git-sha", type=str, default="")
    p.add_argument("--out-json", type=str, default="evals/m2_colab_l4/v33_update_aware_qwen3_1.7b.json")
    p.add_argument("--mock-llm", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

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

    ref_v32_fac: dict[str, Any] | None = None
    if args.ref_v32_json and Path(args.ref_v32_json).exists():
        ref_v32_fac = json.loads(Path(args.ref_v32_json).read_text())["gate2_factorial_2x4"]

    git_prov = resolve_git_provenance(args.git_sha, REPO_ROOT)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "version": "3.3.0-tmlr-update-aware-append",
        "git_sha": git_prov["git_sha"],
        "git_dirty": git_prov["git_dirty"],
        "git_sha_source": git_prov["git_sha_source"],
        "checkpoint_sha256": checkpoint_sha256,
        "pathb_checkpoint": ckpt_info,
        "requested_model": args.model,
        "resolved_model_id": resolved_model_id,
        "model_load_attempts": load_attempts,
        "prompt_family": prompt_family,
        "mock_llm": args.mock_llm,
        "device": str(device),
        "data_source": data_source,
        "seed": int(args.seed),
        "update_aware_control_2x3": run_update_aware_append_control(
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
            ref_v32_factorial=ref_v32_fac,
        ),
    }
    _atomic_write_json(out_path, payload)
    print(f"[DONE] Wrote v3.3 update-aware append control -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
