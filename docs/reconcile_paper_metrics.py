#!/usr/bin/env python3
"""Read-only CPU reconciliation script for LSLC Paper 1 (TMLR Rev 7).

Recomputes exact unrounded McNemar p-values, Holm-Bonferroni adjustments (m=12 Gate-2,
m=6 LongMemEval), cluster-bootstrap 95% CIs (seed=2026, B=2000), exact cluster-level
sign-flip sensitivity p-values, fresh Qwen3 N=400 replication (seeds 2030+2031),
Table 4 unrounded cycle sums & Llama break-even, legacy W2N rescoring, and verifies
every reported metric, citation, cross-reference, and double-blind anonymity check
against docs/paper_draft_lslc.tex and docs/paper_draft_lslc_tmlr_anon.tex.

Zero GPU required; pure Python standard library (~1 second runtime).
"""

from __future__ import annotations

import json
import pathlib
import random
import re
import subprocess
import sys
from math import comb

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evals" / "m2_colab_l4"
TEX_PATH = ROOT / "docs" / "paper_draft_lslc.tex"
ANON_TEX_PATH = ROOT / "docs" / "paper_draft_lslc_tmlr_anon.tex"
ANON_PDF_PATH = ROOT / "docs" / "paper_draft_lslc_tmlr_anon.pdf"

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_W2N_MULT = {"thousand": "k", "million": "m", "billion": "b"}


def _replace_number_word_phrases(text: str) -> str:
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?|[^a-z\d]+", text.lower())
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _TENS:
            val = _TENS[tok]
            if i + 2 < len(tokens) and tokens[i + 1] in (" ", "-") and tokens[i + 2] in _UNITS and _UNITS[tokens[i + 2]] < 10:
                val += _UNITS[tokens[i + 2]]
                i += 3
            else:
                i += 1
            out.append(str(val))
            continue
        if tok in _UNITS:
            out.append(str(_UNITS[tok]))
            i += 1
            continue
        out.append(tok)
        i += 1
    return "".join(out)


def canonicalize_w2n_answer(text: str) -> str:
    """Pure-Python mirror of src/lslc/pathb.py:canonicalize_w2n_answer."""
    s = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL | re.IGNORECASE).strip().lower()
    if "\n" in s:
        s = s.splitlines()[0].strip()
    s = s.replace("$", "").replace("usd", "").replace(",", "").strip()
    s = re.sub(r"^(?:the\s+answer\s+is|answer\s*[:=])\s*", "", s).strip()
    s = s.rstrip(" .,:;\"'`")
    s_w2n = _replace_number_word_phrases(s).strip()
    m_word = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(thousand|million|billion)", s_w2n)
    if m_word:
        num = float(m_word.group(1))
        num_str = str(int(num)) if num.is_integer() else f"{num:g}"
        return f"{num_str}{_W2N_MULT[m_word.group(2)]}"
    m_short = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmb])", s_w2n)
    if m_short:
        num = float(m_short.group(1))
        num_str = str(int(num)) if num.is_integer() else f"{num:g}"
        return f"{num_str}{m_short.group(2)}"
    m_raw = re.fullmatch(r"(\d+(?:\.\d+)?)", s_w2n)
    if m_raw:
        num = float(m_raw.group(1))
        if num >= 1_000_000_000 and (num / 1_000_000_000).is_integer():
            return f"{int(num // 1_000_000_000)}b"
        if num >= 1_000_000 and (num / 1_000_000).is_integer():
            return f"{int(num // 1_000_000)}m"
        if num >= 1_000 and (num / 1_000).is_integer():
            return f"{int(num // 1_000)}k"
        return str(int(num)) if num.is_integer() else f"{num:g}"
    return re.sub(r"\s+", " ", s_w2n).strip()


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def holm_adjust(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda x: x[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for idx, (k, p) in enumerate(items):
        running = max(running, min(1.0, p * (m - idx)))
        out[k] = running
    return out


def cluster_signflip_p(eps_a: list[dict], eps_b: list[dict], key: str) -> float:
    clusters: dict[tuple[str, str, str], int] = {}
    for ea, eb in zip(eps_a, eps_b):
        cid = (ea["mode"], ea["query_type"], str(ea["gold"]))
        clusters[cid] = clusters.get(cid, 0) + (int(ea[key]) - int(eb[key]))
    nz = [v for v in clusters.values() if v != 0]
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


def cluster_boot_ci(eps_a: list[dict], eps_b: list[dict], key: str, seed: int = 2026, n_boot: int = 2000) -> tuple[float, float, int]:
    clusters: dict[tuple[str, str, str], list[int]] = {}
    for ea, eb in zip(eps_a, eps_b):
        cid = (ea["mode"], ea["query_type"], str(ea["gold"]))
        clusters.setdefault(cid, []).append(int(ea[key]) - int(eb[key]))
    cids = sorted(clusters.keys())
    c_stats = [(sum(clusters[c]), len(clusters[c])) for c in cids]
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_boot):
        sv, nv = 0, 0
        for _j in range(len(cids)):
            s_c, n_c = c_stats[rng.randrange(len(cids))]
            sv += s_c
            nv += n_c
        diffs.append(100.0 * sv / nv)
    diffs.sort()
    return diffs[int(0.025 * (n_boot - 1))], diffs[int(0.975 * (n_boot - 1))], len(cids)


def main() -> int:
    tex_raw = TEX_PATH.read_text()
    tex = tex_raw.replace("\\allowbreak ", "").replace("\\allowbreak", "")

    v32_q = json.loads((EVAL_DIR / "v32_tmlr_qwen3_1.7b.json").read_text())
    v32_l = json.loads((EVAL_DIR / "v32_tmlr_llama31_8b.json").read_text())
    q_g2 = json.loads((EVAL_DIR / "v31_qwen_5_gate2_s2029_2031.json").read_text())["gate2_tiebreaker_n200"]
    l_g2 = json.loads((EVAL_DIR / "v31_llama_2_gate2_n200.json").read_text())["llama_gate2_n200"]["systems"]
    leg = json.loads((EVAL_DIR / "v3_tb_3_gate2_n200.json").read_text())

    checks: list[tuple[str, bool, str]] = []

    def chk(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))

    # 1. Exact parameters, checksums, resolved model ID (7)
    chk("qwen_params_326407", "326{,}407" in tex)
    chk("llama_params_594695", "594{,}695" in tex)
    chk("qwen_sha256", v32_q["checkpoint_sha256"] in tex)
    chk("llama_sha256", v32_l["checkpoint_sha256"] in tex)
    chk("legacy_sha256", "7048dfc167de9729a8507163b330a3c1ae46a5d8e517bdc72ffe2eb3323278e7" in tex)
    chk("git_sha_0e8f2d8", "0e8f2d8" in tex)
    chk("unsloth_mirror", "unsloth/Meta-Llama-3.1-8B-Instruct" in tex)

    # 2. v3.2 factorial exact Norm-EM & W2N-EM + unrounded Holm + cluster sign-flip (38)
    for tag, d in [("qwen", v32_q), ("llama", v32_l)]:
        fac = d["gate2_factorial_2x4"]
        arms = fac["systems"]
        for order in ["edited_last", "page_id_asc"]:
            for pol in ["raw_text_append", "raw_text_overwrite", "standalone_append_192tok", "standalone_overwrite_192tok"]:
                entry = arms[f"{order}__{pol}"]
                em = f"{entry['normalized_numeric_em'] * 100:.1f}\\%"
                w2n = f"{entry['normalized_w2n_em'] * 100:.1f}\\%"
                chk(f"v32_{tag}_{order}_{pol}_em", em in tex, em)
                chk(f"v32_{tag}_{order}_{pol}_w2n", w2n in tex, w2n)

        # Recompute unrounded Holm & cluster sign-flip for Norm-EM
        raw_p, sf_p = {}, {}
        for cname in fac["pairwise_norm_em"]:
            a, b = cname.split("__vs__")
            ea, eb = arms[a]["episodes"], arms[b]["episodes"]
            na = sum(1 for x, y in zip(ea, eb) if x["normalized_numeric_em"] and not y["normalized_numeric_em"])
            nb = sum(1 for x, y in zip(ea, eb) if y["normalized_numeric_em"] and not x["normalized_numeric_em"])
            raw_p[cname] = exact_mcnemar(na, nb)
            sf_p[cname] = cluster_signflip_p(ea, eb, "normalized_numeric_em")
        h_ep = holm_adjust(raw_p)
        h_sf = holm_adjust(sf_p)
        if tag == "qwen":
            chk("unrounded_holm_qwen_ed", abs(h_ep["edited_last__standalone_overwrite_192tok__vs__edited_last__standalone_append_192tok"] - 6.245e-7) < 1e-8)
            chk("unrounded_holm_qwen_raw_id", abs(h_ep["page_id_asc__raw_text_overwrite__vs__page_id_asc__raw_text_append"] - 1.106e-5) < 1e-7)
            chk("signflip_qwen_id", abs(h_sf["page_id_asc__standalone_overwrite_192tok__vs__page_id_asc__standalone_append_192tok"] - 0.01465) < 5e-4)
        else:
            chk("unrounded_holm_llama_id", abs(h_ep["page_id_asc__standalone_overwrite_192tok__vs__page_id_asc__standalone_append_192tok"] - 3.735e-3) < 1e-5)
            chk("unrounded_holm_llama_raw_app", abs(h_ep["page_id_asc__standalone_overwrite_192tok__vs__page_id_asc__raw_text_append"] - 6.685e-5) < 1e-6)
            chk("signflip_llama_ed_id", abs(h_sf["edited_last__standalone_overwrite_192tok__vs__edited_last__standalone_append_192tok"] - 0.1265) < 5e-4 and abs(h_sf["page_id_asc__standalone_overwrite_192tok__vs__page_id_asc__standalone_append_192tok"] - 0.1304) < 5e-4)

    # 3. v3.2 LongMemEval N=72 (18)
    for tag, d in [("qwen", v32_q), ("llama", v32_l)]:
        lme = d["longmemeval_knowledge_update"]
        arms = lme["systems"]
        chk(f"v32_{tag}_lme_n72", lme["num_episodes"] == 72 and "N=72" in tex)
        for arm in ["bm25_rag_k2", "bge_4span_rag_k2", "standalone_discrete_k48_d4", "standalone_discrete_k48_d6"]:
            entry = arms[arm]
            val = f"{entry['local_value_em'] * 100:.1f}\\%"
            em = f"{entry['extracted_em'] * 100:.1f}\\%"
            chk(f"v32_{tag}_lme_{arm}_val", val in tex, val)
            chk(f"v32_{tag}_lme_{arm}_em", em in tex, em)

    # 4. Disjoint Qwen N=400 replication (seeds 2030+2031) & Table 4 unrounded sums + Llama break-even (4)
    sb_400 = q_g2["by_seed"]["2030"]["systems"]["standalone_discrete_span_bank"]["episodes"] + q_g2["by_seed"]["2031"]["systems"]["standalone_discrete_span_bank"]["episodes"]
    ap_400 = q_g2["by_seed"]["2030"]["systems"]["standalone_append_192tok"]["episodes"] + q_g2["by_seed"]["2031"]["systems"]["standalone_append_192tok"]["episodes"]
    sa400 = sum(int(x["normalized_numeric_em"]) for x in sb_400)
    sb400 = sum(int(x["normalized_numeric_em"]) for x in ap_400)
    na400 = sum(1 for a, b in zip(sb_400, ap_400) if a["normalized_numeric_em"] and not b["normalized_numeric_em"])
    nb400 = sum(1 for a, b in zip(sb_400, ap_400) if b["normalized_numeric_em"] and not a["normalized_numeric_em"])
    lo400, hi400, _ = cluster_boot_ci(sb_400, ap_400, "normalized_numeric_em", seed=2026)
    chk("fresh_qwen_400", sa400 == 311 and sb400 == 251 and na400 == 66 and nb400 == 6 and "311/400" in tex and "251/400" in tex, f"{sa400}/{sb400} [{lo400:.2f},{hi400:.2f}]")

    llama_extra_fixed = (l_g2["standalone_discrete_span_bank"]["encode_ms"] + l_g2["standalone_discrete_span_bank"]["edit_ms"]) - (l_g2["bm25_rag_append_errata"]["encode_ms"] + l_g2["bm25_rag_append_errata"]["edit_ms"])
    llama_q_save = l_g2["bm25_rag_append_errata"]["first_query_ms"] - l_g2["standalone_discrete_span_bank"]["first_query_ms"]
    chk("llama_breakeven_4q", abs(llama_extra_fixed / llama_q_save - 3.17) < 0.05 and "\\ge 4" in tex, f"{llama_extra_fixed/llama_q_save:.2f}")

    # 5. Legacy W2N exact rescoring via pure-Python canonicalize_w2n_answer (4)
    def collect_eps(obj, arm):
        out = []
        if isinstance(obj, dict):
            if arm in obj and isinstance(obj[arm], dict) and "episodes" in obj[arm]:
                out.extend(obj[arm]["episodes"])
            for v in obj.values():
                out.extend(collect_eps(v, arm))
        return out

    for arm, exp_hits, exp_str in [
        ("bm25_rag_append_errata", 342, "57.00\\%"),
        ("gated_bm25_rag", 318, "53.00\\%"),
        ("standalone_discrete_span_bank", 377, "62.83\\%"),
        ("lslc_full", 422, "70.33\\%"),
    ]:
        eps = collect_eps(leg, arm)
        if not eps and arm == "standalone_discrete_span_bank":
            eps = collect_eps(leg, "standalone_discrete")
        hits = sum(1 for e in eps if canonicalize_w2n_answer(e.get("pred", e.get("prediction", ""))) == canonicalize_w2n_answer(e.get("gold", e.get("answer", ""))))
        chk(f"legacy_w2n_{arm}", hits == exp_hits and exp_str in tex, f"{hits}/{len(eps)}")

    # 6. Structural, citation, figure, overclaim & anonymity checks (19)
    cited = {k.strip() for grp in re.findall(r"\\cite\{([^}]+)\}", tex) for k in grp.split(",")}
    defined = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    fig_labels = {l for l in labels if l.startswith("fig:")}
    chk("bib_29_all_defined_cited", len(defined) == 29 and not (cited - defined) and not (defined - cited))
    chk("refs_zero_undefined", not (refs - labels))
    chk("figs_5_zero_orphan", len(fig_labels) == 5 and not (fig_labels - refs))
    chk("perceiver_and_cpack_authors", "A.~Zisserman, O.~Vinyals" in tex and "D.~Lian, and J.-Y.~Nie" in tex)
    chk("fig2_no_fake_teal", "Path B (Span-Aligned" not in tex)
    chk("table8_twobank_rows", "Two-Bank KV Reuse" in tex)
    chk("full_gen_no_double_count", "906.3$ vs.\\ $889.8" in tex and "946" not in tex)
    chk("lme_static_and_proxy", "0$ calls to \\texttt{apply\\_edit}" in tex and "lexical-coverage proxy" in tex.lower())
    chk("19_json_manifest", "19$ saved JSON" in tex and "v3\\_universal\\_v2\\_train.json" in tex and "v33\\_update\\_aware\\_qwen3\\_1.7b.json" in tex)

    # 5b. v3.3 Update-Aware Append Control (N=200 per backbone, 0 locked-page mismatches) (16)
    for tag, fn in [("qwen", "v33_update_aware_qwen3_1.7b.json"), ("llama", "v33_update_aware_llama31_8b.json")]:
        v33_d = json.loads((EVAL_DIR / fn).read_text())["update_aware_control_2x3"]
        chk(f"v33_{tag}_zero_locked_mismatch", v33_d["ref_v32_locked_pages_mismatches"] == 0 and v33_d["ref_v32_ordered_pages_mismatches"] == 0)
        for order in ["edited_last", "page_id_asc"]:
            for pol in ["raw_text_append", "standalone_append_192tok", "standalone_overwrite_192tok"]:
                entry = v33_d["systems"][f"{order}__{pol}"]
                em = f"{entry['normalized_numeric_em'] * 100:.1f}\\%"
                w2n = f"{entry['normalized_w2n_em'] * 100:.1f}\\%"
                chk(f"v33_{tag}_{order}_{pol}_em_w2n", em in tex and w2n in tex, f"{em}/{w2n}")
        pw = v33_d["pairwise_norm_em"]
        if tag == "qwen":
            chk("v33_qwen_holm_ed", abs(pw["edited_last__standalone_overwrite_192tok__vs__edited_last__standalone_append_192tok"]["exact_unrounded"]["p_mcnemar_holm"] - 4.47e-7) < 1e-8)
        else:
            chk("v33_llama_holm_raw_ed", abs(pw["edited_last__standalone_overwrite_192tok__vs__edited_last__raw_text_append"]["exact_unrounded"]["p_mcnemar_holm"] - 0.01496) < 2e-4)

    lower = tex.lower()
    for w in ["statistical parity", "100\\% ood", "100% ood", "steelmanned", "note honestly", "honest caveats", "falsified", "orthogonal", "earlier released adapter", "prior adapter release", "genuine assist"]:
        chk(f"no_forbidden_{w.replace(' ', '_')}", w not in lower, w)

    # Double-blind PDF & LaTeX source anonymity check (including public commit hash, SHA-256 hashes & ckpt filenames)
    leak_terms = [
        "fikru", "ramadhan", "arielfikru", "zenodo", "24473281", "0e8f2d8",
        "94cc5270b441a55a", "17c7918a6d9da31c", "7048dfc167de9729",
        "lslc_pathb_qwen3_1.7b.safetensors", "lslc_pathb_llama31_8b.safetensors",
        "lslc_universal_v2_qwen3_1.7b.safetensors",
    ]
    if ANON_PDF_PATH.exists():
        pdf_txt = subprocess.check_output(["pdftotext", str(ANON_PDF_PATH), "-"], text=True).lower()
        leaks = [w for w in leak_terms if w in pdf_txt]
        chk("anon_pdf_zero_leaks", len(leaks) == 0, str(leaks))
    else:
        chk("anon_pdf_exists", False, "missing")
    anon_tex_txt = ANON_TEX_PATH.read_text()
    tex_leaks = [w for w in leak_terms if w in anon_tex_txt.lower()]
    chk("anon_tex_zero_leaks", len(tex_leaks) == 0, str(tex_leaks))
    chk("fig3_asymmetric_table", "x error minus=xm, x error plus=xp" in tex)
    chk("qwen_1050_steps", "1{,}050" in tex and "400+250+400" in tex)
    tmlr_anon_pdf = ROOT / "docs" / "tmlr_submission" / "main_tmlr_anon.pdf"
    if tmlr_anon_pdf.exists():
        tmlr_txt = subprocess.check_output(["pdftotext", str(tmlr_anon_pdf), "-"], text=True)
        tmlr_leaks = [w for w in leak_terms if w in tmlr_txt.lower()]
        chk("official_tmlr_anon_pdf_zero_leaks", len(tmlr_leaks) == 0 and "??" not in tmlr_txt, str(tmlr_leaks))
    else:
        chk("official_tmlr_anon_pdf_exists", False, "missing")

    passed = sum(1 for _, ok, _ in checks if ok)
    failed = [(n, d) for n, ok, d in checks if not ok]
    print(f"LSLC PAPER 1 CPU RECONCILIATION: {passed}/{len(checks)} PASSED (bib={len(defined)}, figs={len(fig_labels)})")
    if failed:
        print("FAILED CHECKS:", failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
