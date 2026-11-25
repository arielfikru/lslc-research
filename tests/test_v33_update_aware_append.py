"""Unit tests for scripts/colab_run_v33_update_aware_append.py (CPU mock-LLM)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import colab_run_v33_update_aware_append as v33  # noqa: E402


def test_update_aware_prompt_explicit_recency() -> None:
    p = v33.UPDATE_AWARE_G2_FEWSHOT_PREFIX
    assert "later updates supersede earlier records" in p
    assert "always answer using the latest approved update" in p
    assert "Example 1 -> Question:" in p
    assert "Example 2 -> Question:" in p


def test_exact_stats_helpers() -> None:
    p_mc = v33.exact_mcnemar_unrounded(10, 0)
    assert abs(p_mc - (2.0 / 1024.0)) < 1e-12
    # 4 clusters all +1 -> exact two-sided sign-flip p = 2/16 = 0.125
    sa = [1.0, 1.0, 1.0, 1.0]
    sb = [0.0, 0.0, 0.0, 0.0]
    cids = ["c1", "c2", "c3", "c4"]
    assert abs(v33.exact_cluster_signflip_p(sa, sb, cids) - 0.125) < 1e-12


def test_v33_mock_cli_end_to_end(tmp_path: Path) -> None:
    out_json = tmp_path / "v33_mock.json"
    v33.main(
        [
            "--mock-llm",
            "--model",
            "Qwen/Qwen3-1.7B",
            "--gate2-n",
            "8",
            "--n-boot",
            "50",
            "--seed",
            "2029",
            "--out-json",
            str(out_json),
        ]
    )
    assert out_json.exists()
    data = json.loads(out_json.read_text())
    ctrl = data["update_aware_control_2x3"]
    assert ctrl["num_episodes"] == 8
    assert len(ctrl["systems"]) == 6
    assert len(ctrl["pairwise_norm_em"]) == 6
    for arm, entry in ctrl["systems"].items():
        assert len(entry["episodes"]) == 8
    # Verify exact locked_pages across all 6 arms
    arms = list(ctrl["systems"].keys())
    for i in range(8):
        ref_lp = ctrl["systems"][arms[0]]["episodes"][i]["locked_pages"]
        for a in arms[1:]:
            assert ctrl["systems"][a]["episodes"][i]["locked_pages"] == ref_lp
    # Verify unrounded exact stats present
    k0 = "edited_last__standalone_overwrite_192tok__vs__edited_last__standalone_append_192tok"
    assert "exact_unrounded" in ctrl["pairwise_norm_em"][k0]
    assert "p_cluster_signflip_holm" in ctrl["pairwise_norm_em"][k0]["exact_unrounded"]
