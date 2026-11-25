"""Unit tests for the v3 Cross-File API Migration Coding-Agent Benchmark."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lslc import (  # noqa: E402
    CodingAgentHarness,
    CrossFileAPIMigrationBenchmark,
    LSLCConfig,
    LSLCSystem,
    verify_migration_patch,
)


class TestV3CodingBenchmark(unittest.TestCase):
    def test_disjoint_train_and_eval_splits(self) -> None:
        CrossFileAPIMigrationBenchmark.assert_disjoint_splits()
        train_bench = CrossFileAPIMigrationBenchmark(split="train")
        eval_bench = CrossFileAPIMigrationBenchmark(split="eval")
        self.assertGreaterEqual(len(train_bench), 4)
        self.assertGreaterEqual(len(eval_bench), 6)
        self.assertEqual(eval_bench[0].symbol_name, "load_config")
        self.assertEqual(eval_bench[0].new_kwarg, "env")

    def test_unpatched_repo_fails_with_stale_signature(self) -> None:
        task = CrossFileAPIMigrationBenchmark(split="eval")[0]
        res = verify_migration_patch(task, task.files, run_subprocess=True)
        self.assertFalse(res.patch_correct)
        self.assertTrue(res.stale_signature_used)
        self.assertTrue(res.homonym_preserved)
        self.assertFalse(res.hidden_test_passed)

    def test_gold_patch_passes_all_eval_tasks(self) -> None:
        eval_bench = CrossFileAPIMigrationBenchmark(split="eval")
        for task in eval_bench:
            res = verify_migration_patch(task, task.gold_files, run_subprocess=True)
            self.assertTrue(res.patch_correct, msg=f"{task.task_id} failed: {res.failure_reason}")
            self.assertFalse(res.stale_signature_used)
            self.assertFalse(res.decoy_value_used)
            self.assertTrue(res.homonym_preserved)
            self.assertTrue(res.hidden_test_passed)

    def test_naive_regex_touching_homonym_fails_homonym_check(self) -> None:
        task = CrossFileAPIMigrationBenchmark(split="eval")[0]
        corrupted = dict(task.gold_files)
        corrupted[task.homonym_file] = corrupted[task.homonym_file].replace(
            "def load_config(path: str) -> str:",
            "def load_config(path: str, *, env: str = \"prod\") -> str:",
        )
        res = verify_migration_patch(task, corrupted, run_subprocess=True)
        self.assertFalse(res.patch_correct)
        self.assertFalse(res.homonym_preserved)

    def test_coding_agent_harness_telemetry(self) -> None:
        task = CrossFileAPIMigrationBenchmark(split="eval")[1]
        mem_events: list[str] = []
        harness = CodingAgentHarness(
            task,
            on_read=lambda path, _txt: (mem_events.append(f"read:{path}") or 1.25),
            on_edit=lambda path, _old, _new: (mem_events.append(f"edit:{path}") or 2.5),
        )
        hits = harness.search_code(task.symbol_name)
        self.assertGreaterEqual(len(hits), 4)

        _ = harness.read_file(f"{task.pkg_name}/core.py")
        _ = harness.read_file(f"{task.pkg_name}/service_a.py")
        _ = harness.read_file(f"{task.pkg_name}/service_a.py")  # repeated read
        for path in task.caller_files:
            harness.apply_file_edit(path, task.gold_files[path])

        res = harness.evaluate(run_subprocess=True)
        self.assertTrue(res.patch_correct)
        self.assertEqual(res.search_calls, 1)
        self.assertEqual(res.files_read_total, 3)
        self.assertEqual(res.repeated_file_reads, 1)
        self.assertGreater(res.chars_read_total, 0)
        self.assertAlmostEqual(res.memory_update_ms, 3 * 1.25 + 3 * 2.5, places=4)
        self.assertGreater(res.wall_clock_ms, 0.0)

    def test_scaled_repo_38_and_64_files_all_parseable(self) -> None:
        tok_re = re.compile(r"[A-Za-z0-9_.\-$]+|[^\w\s]")
        bench_38 = CrossFileAPIMigrationBenchmark(split="eval", num_distractor_files=32)
        bench_64 = CrossFileAPIMigrationBenchmark(split="eval", num_distractor_files=58)

        task_38 = bench_38[0]
        task_64 = bench_64[0]
        self.assertEqual(len(task_38.files), 38)
        self.assertEqual(len(task_64.files), 64)
        self.assertEqual(set(task_38.initial_files.keys()), set(task_38.files.keys()))
        self.assertNotEqual(task_38.initial_files[task_38.core_file], task_38.files[task_38.core_file])

        toks_38 = sum(len(tok_re.findall(src)) for src in task_38.files.values())
        toks_64 = sum(len(tok_re.findall(src)) for src in task_64.files.values())
        self.assertGreaterEqual(toks_38, 7000)
        self.assertGreaterEqual(toks_64, 13000)

        for file_map in (task_64.initial_files, task_64.files, task_64.gold_files):
            for path, src in file_map.items():
                try:
                    ast.parse(src, filename=path)
                except SyntaxError as exc:
                    self.fail(f"Generated file {path} failed ast.parse: {exc}")

    def test_zero_answer_leakage_in_prompt_and_stale_callers(self) -> None:
        for split in ("train", "eval"):
            bench = CrossFileAPIMigrationBenchmark(split=split, num_distractor_files=32)
            for task in bench:
                val_bare = task.default_value.strip("\"'")
                decoy_bare = task.decoy_value.strip("\"'")
                for secret in (task.new_signature, task.new_kwarg, task.default_value, val_bare, decoy_bare):
                    self.assertNotIn(secret, task.prompt, msg=f"Prompt leaked {secret!r} in {task.task_id}")

                for path, src in task.initial_files.items():
                    if path == task.core_file:
                        continue
                    for leak_tok in (task.new_kwarg, val_bare, decoy_bare):
                        words = set(re.findall(r"[A-Za-z0-9_-]+", src))
                        self.assertNotIn(
                            leak_tok,
                            words,
                            msg=f"File {path} leaked {leak_tok!r} in initial repo for {task.task_id}",
                        )
                self.assertNotIn("TODO", task.files[f"{task.pkg_name}/service_a.py"])

    def test_rejected_decoy_default_fails_hidden_tests_and_flagged(self) -> None:
        task = CrossFileAPIMigrationBenchmark(split="eval", num_distractor_files=32)[0]
        decoy_patched = dict(task.gold_files)
        for path in task.caller_files:
            decoy_patched[path] = decoy_patched[path].replace(task.default_value, task.decoy_value)
        res = verify_migration_patch(task, decoy_patched, run_subprocess=True)
        self.assertFalse(res.patch_correct)
        self.assertFalse(res.hidden_test_passed)
        self.assertTrue(res.decoy_value_used)

    def test_session_turns_disjoint_and_alternating_order(self) -> None:
        train_tasks = list(CrossFileAPIMigrationBenchmark(split="train", num_distractor_files=4))
        eval_tasks = list(CrossFileAPIMigrationBenchmark(split="eval", num_distractor_files=4))

        train_texts = " ".join(str(t["text"]) for task in train_tasks for t in task.session_turns)
        eval_texts = " ".join(str(t["text"]) for task in eval_tasks for t in task.session_turns)
        for phrase in ("Engineering lead ratified", "was vetoed during code review"):
            self.assertIn(phrase, train_texts)
            self.assertNotIn(phrase, eval_texts)
        for phrase in ("Architecture committee finalized", "failed approval and was discarded", "denied by review board"):
            self.assertIn(phrase, eval_texts)
            self.assertNotIn(phrase, train_texts)

        orders = [tuple(bool(t["accepted"]) for t in task.session_turns) for task in eval_tasks]
        self.assertIn((True, False), orders)
        self.assertIn((False, True), orders)

    def test_modular_single_page_reencode_invariance(self) -> None:
        torch.manual_seed(7)
        cfg = LSLCConfig.tiny_mock(k_exact=8)
        sys_mod = LSLCSystem(cfg)
        embed_fn = torch.nn.Embedding(128, cfg.d_llm)
        pages = torch.randint(2, 120, (1, 6, cfg.page_size), dtype=torch.long)

        with torch.no_grad():
            z0, bank0, toc0 = sys_mod.encode_book(pages, embed_fn, use_exact_bank=True, vocab_size=128)
            z_before = z0.clone()
            toc_before = toc0.clone()
            bank_before = bank0.token_ids.clone()
            new_page = torch.randint(2, 120, (1, cfg.page_size), dtype=torch.long)
            z1, bank1, toc1 = sys_mod.reencode_page(
                z0, toc0, bank0, page_idx=2, new_page_token_ids=new_page, embed_tokens_fn=embed_fn, vocab_size=128, in_place=True
            )

        untouched = [0, 1, 3, 4, 5]
        self.assertEqual(float((z1[:, untouched] - z_before[:, untouched]).abs().max().item()), 0.0)
        self.assertEqual(float((toc1[:, untouched] - toc_before[:, untouched]).abs().max().item()), 0.0)
        self.assertEqual(int((bank1.token_ids[:, untouched] - bank_before[:, untouched]).abs().max().item()), 0)
        self.assertGreater(float((z1[:, 2] - z_before[:, 2]).abs().max().item()), 0.0)

    def test_gerbang3_script_mock_llm_smoke(self) -> None:
        import colab_run_v3_gerbang3_coding as g3

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "g3_mock.json"
            g3.main(
                [
                    "--mock-llm",
                    "--num-distractors",
                    "32",
                    "--lslc-k-deep",
                    "2,4",
                    "--finetune-code-salience-steps",
                    "2",
                    "--max-new-tokens",
                    "12",
                    "--out-json",
                    str(out_json),
                ]
            )
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["repo_stats"]["num_files_per_repo"], 39)  # 38 repo files + SESSION_NOTES.md
            for sys_name in (
                "oracle_raw_context",
                "bm25_rag_k2",
                "bm25_rag_k8",
                "llm_auto_compact",
                "lslc_v3_hybrid_k2",
                "lslc_v3_hybrid_k4",
            ):
                self.assertIn(sys_name, payload["systems"])
            lslc_k2 = payload["systems"]["lslc_v3_hybrid_k2"]
            self.assertIn("in_exact_bank_symbol_rate", lslc_k2)
            self.assertIn("in_exact_bank_kwarg_rate", lslc_k2)
            self.assertIn("in_exact_bank_default_rate", lslc_k2)
            self.assertEqual(lslc_k2["max_other_page_diff_after_reencode"], 0.0)
            self.assertEqual(lslc_k2["max_toc_diff_after_delta"], 0.0)

    def test_think_and_fence_stop_and_extraction_helpers(self) -> None:
        import colab_run_v3_gerbang3_coding as g3

        self.assertFalse(g3._should_stop_code_gen("<think>draft ```python\nx=1\n```"))
        self.assertFalse(g3._should_stop_code_gen("<think>draft</think>\n```python\nx=1"))
        self.assertTrue(g3._should_stop_code_gen("<think>draft</think>\n```python\nx=1\n```"))
        self.assertTrue(g3._should_stop_code_gen("from .core import f\n```"))

        extracted = g3._extract_python_code_block(
            "<think>discarded:\n```python\nwrong=1\n```\n</think>\n```python\nright = 2\n```\n",
            "orig = 0\n",
        )
        self.assertEqual(extracted, "right = 2\n")

        self.assertFalse(g3._should_stop_homonym_gen("<think>Maybe KEEP_UNCHANGED or ```python"))
        self.assertTrue(g3._should_stop_homonym_gen("<think>checked</think>\nKEEP_UNCHANGED"))
        self.assertFalse(g3._should_stop_homonym_gen("<think>checked</think>\n```python\ndef f(): pass"))
        self.assertTrue(g3._should_stop_homonym_gen("<think>checked</think>\n```python\ndef f(): pass\n```"))

    def test_exact_bank_attenuation_none_preserves_multipage_and_exact_only_mode(self) -> None:
        from lslc import BookPrefixAssembler, RouterOutput, SalienceGate, SparseExactBank, SpatialLatentProjector

        torch.manual_seed(11)
        cfg = LSLCConfig.tiny_mock(k_exact=8)
        assembler = BookPrefixAssembler(cfg)
        projector = SpatialLatentProjector(cfg)
        sal = SalienceGate(cfg)
        embed_fn = torch.nn.Embedding(128, cfg.d_llm)
        hw = torch.nn.Embedding(cfg.num_spatial_slots, cfg.d_llm)

        z_book = torch.randn(1, 4, cfg.num_layers, cfg.latent_channels, cfg.grid_h, cfg.grid_w)
        pages = torch.randint(2, 120, (1, 4, cfg.page_size), dtype=torch.long)
        bank = SparseExactBank.from_pages(pages, sal, embed_fn(pages), vocab_size=128)
        logits = torch.tensor([[20.0, 0.0, -5.0, -5.0]], dtype=torch.float32)
        probs = torch.softmax(logits, dim=-1)
        router_out = RouterOutput(
            logits=logits,
            probs=probs,
            deep_idx=torch.tensor([[1, 0]], dtype=torch.long),  # page 1 runner-up, page 0 top-1
            skim_idx=torch.tensor([[2]], dtype=torch.long),
            toc_idx=torch.tensor([[3]], dtype=torch.long),
        )

        out_squash = assembler(
            z_book, projector, router_out, bank, embed_fn, hw, mode="exact_only", training_ste=False, exact_bank_attenuation="prob_sq"
        )
        out_clean = assembler(
            z_book, projector, router_out, bank, embed_fn, hw, mode="exact_only", training_ste=False, exact_bank_attenuation="none"
        )
        self.assertEqual(tuple(out_clean.prefix_embeds.shape), (1, 2 * cfg.k_exact, cfg.d_llm))
        # Runner-up page (first slice 0:k_exact) is squashed to near-zero under prob_sq, untouched under none
        self.assertLess(float(out_squash.prefix_embeds[:, : cfg.k_exact].abs().max().item()), 1e-6)
        raw_gather = bank.gather_embeds(router_out.deep_idx, embed_fn, hw, grid_w=cfg.grid_w, training_ste=False)
        self.assertEqual(float((out_clean.prefix_embeds - raw_gather).abs().max().item()), 0.0)

        out_full_compact = assembler(
            z_book,
            projector,
            router_out,
            bank,
            embed_fn,
            hw,
            mode="full",
            training_ste=False,
            flat_toc_max_tokens=1,
            exact_bank_attenuation="none",
        )
        ex_s, ex_e = out_full_compact.slice_bounds["exact"]
        self.assertEqual(float((out_full_compact.prefix_embeds[:, ex_s:ex_e] - raw_gather).abs().max().item()), 0.0)

    def test_gerbang3_diag_probe_mock_smoke(self) -> None:
        import colab_run_v3_gerbang3_coding as g3

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "g3_diag_probe_mock.json"
            g3.main(
                [
                    "--mock-llm",
                    "--diag-probe",
                    "--num-distractors",
                    "8",
                    "--max-new-tokens",
                    "10",
                    "--out-json",
                    str(out_json),
                ]
            )
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["diag_probe"])
            self.assertIsNotNone(payload["eval_01_exact_bank_inspection"])
            self.assertEqual(payload["eval_01_exact_bank_inspection"]["core_page_post_reencode"]["num_special_or_pad"], 0)
            for sys_name in (
                "bm25_rag_k2",
                "bm25_rag_k8",
                "lslc_exact_only_nosquash_k2",
                "lslc_exact_only_nosquash_k4",
                "lslc_full_nosquash_compact_toc_k2",
                "lslc_full_nosquash_compact_toc_k4",
            ):
                self.assertIn(sys_name, payload["systems"])
            # Unpadded BM25 RAG at k=2 should be well below 512 padded tokens when pages are shorter than 256 tokens
            self.assertLess(payload["systems"]["bm25_rag_k2"]["active_prefix_tokens_mean"], 400.0)

    def test_python_code_salience_ast_masks(self) -> None:
        import colab_run_v3_gerbang3_coding as g3

        tok = g3._MockCodeTokenizer(vocab_size=2048)
        task = CrossFileAPIMigrationBenchmark(split="eval", num_distractor_files=0)[0]
        formatted = f"# FILE: {task.core_file}\n{task.core_new_content}\n"
        masks = g3._compute_python_code_salience_masks(formatted, page_size=128, pad_id=0, tokenizer=tok, k_exact=24)

        ids = masks["token_ids"].tolist()
        sig_words = {tok.decode([ids[i]]) for i, v in enumerate(masks["sig_mask"].tolist()) if v > 0.5}
        path_words = {tok.decode([ids[i]]) for i, v in enumerate(masks["path_mask"].tolist()) if v > 0.5}
        doc_words = {tok.decode([ids[i]]) for i, v in enumerate(masks["doc_mask"].tolist()) if v > 0.5}

        self.assertIn("load_config", sig_words)
        self.assertIn("env", sig_words)
        self.assertIn("cfgflow", path_words)
        self.assertIn("core.py", path_words)
        self.assertTrue(any("Core" in w or "mandatory" in w for w in doc_words))
        self.assertGreater(float(masks["pad_mask"].sum().item()), 0.0)

    def test_stage15_alignment_mock_smoke_and_freeze_contract(self) -> None:
        from safetensors.torch import load_file
        import colab_run_v3_gerbang3_coding as g3
        from lslc import FrozenQwenLSLCWrapper, TinyMockCausalLM

        torch.manual_seed(17)
        cfg = LSLCConfig.tiny_mock(
            use_contextual_edit_encoding=True,
            contextual_edit_layer=1,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=24,
        )
        llm = TinyMockCausalLM(vocab_size=2048, d_llm=cfg.d_llm, num_layers=cfg.num_llm_layers)
        tok = g3._MockCodeTokenizer(vocab_size=2048)
        wrapper = FrozenQwenLSLCWrapper(cfg, llm)
        train_pool = [2 + (i % 2046) for i in range(4096)]

        def _snap(mod):
            return {k: v.detach().clone() for k, v in mod.state_dict().items()}

        def _max_diff(before, mod):
            after = mod.state_dict()
            return max(float((before[k] - after[k]).abs().max().item()) for k in before)

        llm_0 = _snap(wrapper.llm)
        enc_0 = _snap(wrapper.lslc.encoder)
        rtr_0 = _snap(wrapper.lslc.router)
        du_0 = _snap(wrapper.lslc.delta_updater)
        hw_0 = _snap(wrapper.lslc.hw_coord_embed)
        sal_0 = _snap(wrapper.lslc.salience_gate)
        proj_0 = _snap(wrapper.lslc.projector)

        stats_a = g3._run_stage_a_salience_alignment(
            wrapper=wrapper,
            tokenizer=tok,
            train_pool=train_pool,
            num_distractors=4,
            steps=4,
            lr=2e-4,
            device=torch.device("cpu"),
            seed=101,
        )
        self.assertEqual(stats_a["steps"], 4)
        self.assertGreater(_max_diff(sal_0, wrapper.lslc.salience_gate), 0.0)
        self.assertEqual(_max_diff(proj_0, wrapper.lslc.projector), 0.0)
        self.assertEqual(_max_diff(enc_0, wrapper.lslc.encoder), 0.0)
        self.assertEqual(_max_diff(rtr_0, wrapper.lslc.router), 0.0)
        self.assertEqual(_max_diff(du_0, wrapper.lslc.delta_updater), 0.0)
        self.assertEqual(_max_diff(llm_0, wrapper.llm), 0.0)

        sal_after_a = _snap(wrapper.lslc.salience_gate)
        stats_b = g3._run_stage_b_projector_alignment(
            wrapper=wrapper,
            tokenizer=tok,
            train_pool=train_pool,
            num_distractors=4,
            steps=4,
            lr=1e-4,
            lslc_k_skim=2,
            salience_route_weight=2.0,
            device=torch.device("cpu"),
            seed=202,
        )
        self.assertEqual(stats_b["steps"], 4)
        self.assertGreater(_max_diff(proj_0, wrapper.lslc.projector), 0.0)
        self.assertEqual(_max_diff(sal_after_a, wrapper.lslc.salience_gate), 0.0)
        self.assertEqual(_max_diff(enc_0, wrapper.lslc.encoder), 0.0)
        self.assertEqual(_max_diff(rtr_0, wrapper.lslc.router), 0.0)
        self.assertEqual(_max_diff(du_0, wrapper.lslc.delta_updater), 0.0)
        self.assertEqual(_max_diff(hw_0, wrapper.lslc.hw_coord_embed), 0.0)
        self.assertEqual(_max_diff(llm_0, wrapper.llm), 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "g3_stage15_mock.json"
            ckpt_path = Path(tmpdir) / "v3_g3_mock.safetensors"
            g3.main(
                [
                    "--mock-llm",
                    "--align-stage15",
                    "--stage-a-steps",
                    "4",
                    "--stage-b-steps",
                    "4",
                    "--num-distractors",
                    "8",
                    "--max-new-tokens",
                    "10",
                    "--run-g12-regression",
                    "--g12-max-episodes",
                    "4",
                    "--save-checkpoint",
                    str(ckpt_path),
                    "--out-json",
                    str(out_json),
                ]
            )
            self.assertTrue(ckpt_path.exists())
            sd = load_file(str(ckpt_path), device="cpu")
            incompat = LSLCSystem(cfg).load_state_dict(sd, strict=True)
            self.assertEqual(len(incompat.missing_keys), 0)
            self.assertEqual(len(incompat.unexpected_keys), 0)

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["align_stage15"])
            self.assertIn("stage15_alignment", payload)
            self.assertIsNotNone(payload["eval_01_exact_bank_inspection"])
            self.assertIsNotNone(payload["gerbang12_regression"])
            for gmode in (
                "cond_b_natural_router",
                "cond_b_hybrid_router",
                "cond_c_oracle",
                "cond_c_hybrid_router",
            ):
                self.assertIn(gmode, payload["gerbang12_regression"])
            for sys_name in (
                "lslc_exact_only_nosquash_k2",
                "lslc_full_nosquash_compact_toc_k2",
                "lslc_full_nosquash_compact_toc_k4",
                "oracle_raw_context",
                "bm25_rag_k2",
                "bm25_rag_k8",
                "llm_auto_compact",
            ):
                self.assertIn(sys_name, payload["systems"])


if __name__ == "__main__":
    unittest.main()
