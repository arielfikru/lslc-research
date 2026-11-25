"""Local 0-CU unit tests for v3.2 TMLR micro-controls (`scripts/colab_run_v32_tmlr_controls.py`)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import colab_run_v32_tmlr_controls as v32  # noqa: E402
from colab_run_v3_gerbang3_coding import _MockCodeTokenizer  # noqa: E402
from lslc import (  # noqa: E402
    AdversarialLivingBookDataset,
    LSLCConfig,
    PathBDiscreteMemory,
    save_pathb_checkpoint,
)


class TestV32TMLRControls(unittest.TestCase):
    def test_raw_text_overwrite_replaces_old_preserves_static_and_neighbor(self) -> None:
        doc = (
            "Background context about engineering ops.\n"
            "POLICY: primary compliance region is eu-frankfurt .\n"
            "RECORD: Project Lyra approved budget = 35M .\n"
            "RECORD: Project Orion approved budget = 29M .\n"
            "Closing notes for Q3 review.\n"
        )
        # 1. Accepted edit replaces Lyra's old record in-place without touching static policy or Orion neighbor
        updated, info = v32.apply_raw_text_in_place_overwrite(
            doc, "Finalized audit amendment places Lyra budget at 105M ."
        )
        self.assertTrue(info["replaced"])
        self.assertEqual(info["mode"], "line_in_place")
        self.assertNotIn("35M", updated)
        self.assertIn("105M", updated)
        self.assertIn("Project Orion approved budget = 29M", updated)
        self.assertIn("primary compliance region is eu-frankfurt", updated)

        # 2. Multi-turn sequence: Edit 1 accepted (35M -> 85M), Edit 2 rejected (skipped), Edit 3 accepted (-> 115M)
        step1, info1 = v32.apply_raw_text_in_place_overwrite(
            doc, "Official ledger update: Lyra authorized funding shifted from 35M to 85M ."
        )
        self.assertTrue(info1["replaced"])
        self.assertIn("85M", step1)
        # Gate rejects proposal for 999M -> raw_text_overwrite is NOT invoked when gate rejects
        gate_accept_edit2 = False
        step2 = step1
        if gate_accept_edit2:
            step2, _ = v32.apply_raw_text_in_place_overwrite(step1, "Controller memo: set Lyra ceiling to 999M .")
        self.assertEqual(step2, step1)
        self.assertNotIn("999M", step2)

        step3, info3 = v32.apply_raw_text_in_place_overwrite(
            step2, "Controller memo: set Lyra ceiling to 115M for this cycle ."
        )
        self.assertTrue(info3["replaced"])
        self.assertNotIn("35M", step3)
        self.assertNotIn("85M", step3)
        self.assertNotIn("999M", step3)
        self.assertIn("115M", step3)
        self.assertIn("Project Orion approved budget = 29M", step3)
        self.assertIn("primary compliance region is eu-frankfurt", step3)

    def test_raw_text_overwrite_single_line_multi_sentence_isolation(self) -> None:
        single_line = (
            "POLICY: signing keys rotate every ninety days. "
            "RECORD: Project Polaris approved deployment state = passive-failover. "
            "RECORD: Project Vega approved deployment state = read-only-replica."
        )
        updated, info = v32.apply_raw_text_in_place_overwrite(
            single_line, "Cluster health update puts Polaris actively in eu-central-frankfurt."
        )
        self.assertTrue(info["replaced"])
        self.assertEqual(info["mode"], "sentence_in_place")
        self.assertNotIn("passive-failover", updated)
        self.assertIn("eu-central-frankfurt", updated)
        self.assertIn("signing keys rotate every ninety days", updated)
        self.assertIn("Project Vega approved deployment state = read-only-replica", updated)

    def test_raw_text_overwrite_on_real_gate2_eval_episodes(self) -> None:
        tok = _MockCodeTokenizer(vocab_size=2048)
        for mode in ("budget", "symbolic"):
            ds = AdversarialLivingBookDataset(
                num_episodes=6,
                num_pages=8,
                page_size=256,
                num_edits=3,
                mode=mode,
                split="eval",
                tokenizer=tok,
                vocab_size=2048,
                seed=2029,
            )
            for ep in ds:
                p_a = int(ep.metadata["proj_a_page"])
                p_b = int(ep.metadata["proj_b_page"])
                p_s = int(ep.metadata["static_page"])
                docs = [tok.decode(ep.context_pages[i].tolist(), skip_special_tokens=True) for i in range(8)]
                orig_b = docs[p_b]
                orig_s = docs[p_s]
                for (p_i, e_ids), is_acc in zip(ep.edit_pages, ep.metadata["edit_accepted_flags"]):
                    if is_acc:
                        e_txt = tok.decode(e_ids.tolist(), skip_special_tokens=True).strip()
                        docs[int(p_i)], info = v32.apply_raw_text_in_place_overwrite(docs[int(p_i)], e_txt)
                        self.assertTrue(info["replaced"], f"Failed in-place replace on mode={mode}: {e_txt!r}")
                # Neighbor & static pages remain 100% bit-identical
                self.assertEqual(docs[p_b], orig_b)
                self.assertEqual(docs[p_s], orig_s)
                # Final accepted answer string is present on proj_a_page
                if ep.metadata["query_type"] in ("accepted_override", "rejected_proposal"):
                    self.assertIn(str(ep.metadata["answer_str"]), docs[p_a])

    def test_order_locked_pages_exact_edited_last_vs_page_id_asc(self) -> None:
        locked = [1, 4, 7, 11]
        # Edited first page (p=1) -> edited_last moves 1 to tail, page_id_asc keeps [1, 4, 7, 11]
        self.assertEqual(v32.order_locked_pages(locked, {1}, "edited_last"), [4, 7, 11, 1])
        self.assertEqual(v32.order_locked_pages(locked, {1}, "page_id_asc"), [1, 4, 7, 11])

        # Edited middle pages (p=4, p=7) -> edited_last puts [1, 11, 4, 7], page_id_asc keeps [1, 4, 7, 11]
        self.assertEqual(v32.order_locked_pages(locked, {4, 7}, "edited_last"), [1, 11, 4, 7])
        self.assertEqual(v32.order_locked_pages(locked, {4, 7}, "page_id_asc"), [1, 4, 7, 11])

        # No edited page in locked set -> both orders coincide as ascending
        self.assertEqual(v32.order_locked_pages(locked, {9}, "edited_last"), [1, 4, 7, 11])
        self.assertEqual(v32.order_locked_pages(locked, {9}, "page_id_asc"), [1, 4, 7, 11])

    def test_cluster_bootstrap_deterministic_and_holm_bonferroni(self) -> None:
        a = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
        b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        clusters = ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4"]

        r1 = v32.cluster_bootstrap_ci95(a, b, clusters, n_boot=500, seed=2029)
        r2 = v32.cluster_bootstrap_ci95(a, b, clusters, n_boot=500, seed=2029)
        self.assertEqual(r1, r2)
        self.assertEqual(r1["num_clusters"], 4)
        self.assertAlmostEqual(r1["mean_diff"], 0.375, places=4)
        self.assertLessEqual(r1["cluster_diff_ci95"][0], r1["mean_diff"])
        self.assertGreaterEqual(r1["cluster_diff_ci95"][1], r1["mean_diff"])

        # Identical vectors -> [0.0, 0.0]
        r_eq = v32.cluster_bootstrap_ci95(a, a, clusters, n_boot=200, seed=7)
        self.assertEqual(r_eq["cluster_diff_ci95"], [0.0, 0.0])

        # Step-down Holm-Bonferroni check
        adj = v32.holm_bonferroni_adjust({"p1": 0.005, "p2": 0.02, "p3": 0.03, "p4": 0.40})
        self.assertAlmostEqual(adj["p1"], 0.02, places=6)   # 0.005 * 4
        self.assertAlmostEqual(adj["p2"], 0.06, places=6)   # 0.02 * 3
        self.assertAlmostEqual(adj["p3"], 0.06, places=6)   # max(0.06, 0.03 * 2)
        self.assertAlmostEqual(adj["p4"], 0.40, places=6)   # max(0.06, 0.40 * 1)

    def test_knowledge_update_loader_strict_filter(self) -> None:
        fixture = [
            {"question_id": "ku_001", "question_type": "knowledge-update", "question": "Q1", "answer": "A1", "haystack_sessions": []},
            {"question_id": "ku_002_abs", "question_type": "knowledge-update", "question": "Q2", "answer": "A2", "haystack_sessions": []},
            {"question_id": "ss_001", "question_type": "single-session-user", "question": "Q3", "answer": "A3", "haystack_sessions": []},
            {"question_id": "ms_001", "question_type": "multi-session", "question": "Q4", "answer": "A4", "haystack_sessions": []},
            {"question_id": "ku_003", "question_type": "knowledge-update", "question": "Q5", "answer": "A5", "haystack_sessions": []},
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lme_fixture.json"
            p.write_text(json.dumps(fixture), encoding="utf-8")
            items, meta = v32.load_longmemeval_knowledge_update(local_path=str(p))
        self.assertEqual([x["question_id"] for x in items], ["ku_001", "ku_003"])
        self.assertEqual(meta["num_episodes"], 2)
        self.assertEqual(meta["question_types_present"], ["knowledge-update"])

    def test_git_provenance_resolution(self) -> None:
        prov_local = v32.resolve_git_provenance("", REPO_ROOT)
        self.assertNotEqual(prov_local["git_sha"], "unknown")
        self.assertEqual(prov_local["git_sha_source"], "git_rev_parse")

        prov_cli = v32.resolve_git_provenance("deadbeef1234", REPO_ROOT)
        self.assertEqual(prov_cli["git_sha"], "deadbeef1234")
        self.assertEqual(prov_cli["git_sha_source"], "cli_arg")

    def test_full_cli_mock_smoke_qwen_and_llama_with_checkpoint_sha(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            # Create a real temporary Path B safetensors checkpoint for mock dimensions
            cfg = LSLCConfig.tiny_mock(
                model_name="mock-qwen",
                use_contextual_edit_encoding=True,
                contextual_edit_layer=1,
                use_contextual_query_encoding=False,
                salience_mode="learned_surprisal",
                delta_pooling_mode="cross_attn",
                router_mode="content_attn",
                k_exact=48,
                salience_span_len=16,
            )
            pb = PathBDiscreteMemory(cfg, bge_toc=None, edit_gate_hidden_dim=64, k_exact=48, salience_span_len=16)
            ckpt_path = Path(td) / "mock_pathb.safetensors"
            saved_info = save_pathb_checkpoint(pb, ckpt_path)

            for model_name, expected_family in (
                ("Qwen/Qwen3-1.7B", "qwen3-chatml"),
                ("meta-llama/Llama-3.1-8B-Instruct", "llama-3.1-instruct"),
            ):
                out_json = Path(td) / f"out_{expected_family}.json"
                v32.main(
                    [
                        "--mock-llm",
                        "--model",
                        model_name,
                        "--pathb-checkpoint",
                        str(ckpt_path),
                        "--git-sha",
                        "abc1234567890",
                        "--n-boot",
                        "100",
                        "--seed",
                        "2029",
                        "--out-json",
                        str(out_json),
                    ]
                )
                self.assertTrue(out_json.is_file())
                data = json.loads(out_json.read_text(encoding="utf-8"))
                self.assertEqual(data["version"], "3.2.0-tmlr-micro-controls")
                self.assertEqual(data["git_sha"], "abc1234567890")
                self.assertEqual(data["checkpoint_sha256"], saved_info["sha256"])
                self.assertEqual(data["prompt_family"], expected_family)

                # Verify Control #1: 2x4 Factorial + 100% locked pages invariant across all 8 arms
                g2 = data["gate2_factorial_2x4"]
                self.assertTrue(g2["locked_pages_invariant_verified"])
                self.assertEqual(len(g2["systems"]), 8)
                arm_names = list(g2["systems"].keys())
                n_eps = g2["num_episodes"]
                for ep_i in range(n_eps):
                    ref_lp = g2["systems"][arm_names[0]]["episodes"][ep_i]["locked_pages"]
                    for a_name in arm_names:
                        ep_rec = g2["systems"][a_name]["episodes"][ep_i]
                        self.assertEqual(ep_rec["locked_pages"], ref_lp)
                        self.assertEqual(sorted(ep_rec["ordered_pages"]), ref_lp)
                        if a_name.startswith("page_id_asc__"):
                            self.assertEqual(ep_rec["ordered_pages"], ref_lp)
                        else:
                            ed_s = set(ep_rec["edited_pages"])
                            exp_ord = sorted(p for p in ref_lp if p not in ed_s) + sorted(p for p in ref_lp if p in ed_s)
                            self.assertEqual(ep_rec["ordered_pages"], exp_ord)
                # Verify cluster bootstrap + Holm-Bonferroni present in pairwise suite
                pw_norm = g2["pairwise_norm_em"]
                self.assertIn("edited_last__standalone_overwrite_192tok__vs__edited_last__raw_text_overwrite", pw_norm)
                sample_pw = pw_norm["edited_last__standalone_overwrite_192tok__vs__edited_last__raw_text_overwrite"]
                self.assertIn("cluster_bootstrap", sample_pw)
                self.assertIn("p_value_holm", sample_pw["mcnemar"])

                # Verify Control #2: LongMemEval knowledge-update only
                lme = data["longmemeval_knowledge_update"]
                self.assertEqual(lme["dataset"]["question_types_present"], ["knowledge-update"])
                self.assertEqual(
                    set(lme["systems"].keys()),
                    {"bm25_rag_k2", "bge_4span_rag_k2", "standalone_discrete_k48_d4", "standalone_discrete_k48_d6"},
                )
                for s_name, s_val in lme["systems"].items():
                    for rec in s_val["episodes"]:
                        self.assertEqual(rec["question_type"], "knowledge-update")
                        self.assertIn("local_value_em", rec)
                sample_lme_pw = lme["pairwise_local_value_em"]["bge_4span_rag_k2__vs__standalone_discrete_k48_d4"]
                self.assertIn("p_value_holm", sample_lme_pw["mcnemar"])


if __name__ == "__main__":
    unittest.main()
