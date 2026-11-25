"""Unit & --mock-llm smoke tests for Fase 1: Universal Checkpoint, Pareto BM25 k1/k2, Lossless Compaction, and Gerbang 5."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import colab_run_m3 as m3  # noqa: E402
import colab_run_v3_gerbang3_coding as g3  # noqa: E402
import colab_run_v3_gerbang4_community as g4  # noqa: E402
import colab_run_v3_gerbang5_ablations as g5  # noqa: E402
import colab_run_v3_phase1 as p1  # noqa: E402
import colab_run_v3_universal_ckpt as uckpt  # noqa: E402
from lslc import LSLCConfig, LSLCSystem, TinyMockCausalLM  # noqa: E402


class TestV3Gerbang5AndUniversal(unittest.TestCase):
    def test_universal_disjointness_guard_and_mock_train_smoke(self) -> None:
        # Disjointness guard must catch leaks from Gate 2, Gate 3, RULER, and LongMemEval
        with self.assertRaises(AssertionError):
            uckpt.assert_universal_train_disjoint({"prose": ["Project Orion budget is 90M."]})
        with self.assertRaises(AssertionError):
            uckpt.assert_universal_train_disjoint({"code": ["from cfgflow.core import load_config"]})
        with self.assertRaises(AssertionError):
            uckpt.assert_universal_train_disjoint({"sal": ["One of the special magic numbers for amber-88 is: 1234567."]})
        with self.assertRaises(AssertionError):
            uckpt.assert_universal_train_disjoint({"dial": ["My mortgage from Wells Fargo was approved."]})

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_p = Path(tmpdir) / "lslc_universal_mock.safetensors"
            out_json = Path(tmpdir) / "universal_mock.json"
            uckpt.main(
                [
                    "--mock-llm",
                    "--train-universal",
                    "--steps",
                    "16",
                    "--encoder-lr",
                    "1.5e-5",
                    "--k-exact",
                    "48",
                    "--salience-span-len",
                    "16",
                    "--save-checkpoint",
                    str(ckpt_p),
                    "--out-json",
                    str(out_json),
                ]
            )
            self.assertTrue(ckpt_p.is_file())
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            tr = payload["train"]
            self.assertEqual(tr["max_frozen_param_diff"], 0.0)
            self.assertEqual(tr["max_toc_diff"], 0.0)
            self.assertEqual(tr["special_tokens_in_bank"], 0)
            self.assertFalse(tr["encoder_frozen"])
            self.assertAlmostEqual(tr["encoder_lr"], 1.5e-5, places=8)
            self.assertGreater(sum(tr["python_mode_counts"].values()), 0)
            self.assertTrue(tr["disjointness_audit"]["verified"])
            self.assertGreater(tr["gate_bce_steps"], 0)
            self.assertIsNotNone(tr["mean_gate_bce_loss"])
            for dom in ("prose_revision", "span16_salience", "chatml_qa_dialogue", "natural_python"):
                self.assertGreater(tr["domain_stats"][dom]["steps"], 0)

            # Saved universal checkpoint must load strict=True cleanly
            from safetensors.torch import load_file

            cfg = LSLCConfig.tiny_mock(
                use_contextual_edit_encoding=True,
                contextual_edit_layer=1,
                use_contextual_query_encoding=False,
                salience_mode="learned_surprisal",
                delta_pooling_mode="cross_attn",
                router_mode="content_attn",
                k_exact=48,
                salience_span_len=16,
            )
            inc = LSLCSystem(cfg).load_state_dict(load_file(str(ckpt_p), device="cpu"), strict=True)
            self.assertEqual(len(inc.missing_keys), 0)
            self.assertEqual(len(inc.unexpected_keys), 0)

    def test_cross_script_salience_span16_and_bge_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            g3_json = Path(tmpdir) / "g3_span16_bge.json"
            g3.main(
                [
                    "--mock-llm",
                    "--salience-span-len",
                    "16",
                    "--k-exact",
                    "48",
                    "--use-bge-toc",
                    "--num-distractors",
                    "4",
                    "--lslc-k-deep",
                    "2",
                    "--max-new-tokens",
                    "12",
                    "--out-json",
                    str(g3_json),
                ]
            )
            p3 = json.loads(g3_json.read_text(encoding="utf-8"))
            self.assertEqual(p3["config"]["salience_span_len"], 16)
            self.assertTrue(p3["config"]["use_bge_toc"])

            p1_json = Path(tmpdir) / "p1_span16_bge.json"
            p1.main(
                [
                    "--mock-llm",
                    "--stage1-only",
                    "--skip-stage1",
                    "--salience-span-len",
                    "16",
                    "--k-exact",
                    "48",
                    "--use-bge-toc",
                    "--eval-episodes",
                    "4",
                    "--oracle-episodes",
                    "4",
                    "--out-json",
                    str(p1_json),
                ]
            )
            pp1 = json.loads(p1_json.read_text(encoding="utf-8"))
            self.assertEqual(pp1["config"]["salience_span_len"], 16)
            self.assertTrue(pp1["config"]["use_bge_toc"])

    def test_lossless_compaction_zero_dropped_tokens_and_pareto_rag_only(self) -> None:
        class _ExactWordTok:
            def __init__(self) -> None:
                self._m: dict[str, int] = {}
                self._inv: dict[int, str] = {}
                self._nxt = 10

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                del add_special_tokens
                out = []
                for w in text.split():
                    if w not in self._m:
                        self._m[w] = self._nxt
                        self._inv[self._nxt] = w
                        self._nxt += 1
                    out.append(self._m[w])
                return out

            def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
                del skip_special_tokens
                return " ".join(self._inv[i] for i in ids if i in self._inv)

        tok = _ExactWordTok()
        # Mix normal sessions + an oversized 350-token single session (> max_chunk_tokens=100)
        sessions = [
            {"text": " ".join(f"s0_w{i}" for i in range(60))},
            {"text": " ".join(f"s1_w{i}" for i in range(55))},
            {"text": " ".join(f"s2_huge_{i}" for i in range(350))},
            {"text": " ".join(f"s3_w{i}" for i in range(40))},
            {"text": " ".join(f"s4_w{i}" for i in range(75))},
        ]
        all_expected_toks: list[int] = []
        for rec in sessions:
            all_expected_toks.extend(tok.encode(rec["text"]))

        windows = g4.split_sessions_into_lossless_chunks(sessions, tok, max_chunk_tokens=100)
        self.assertGreaterEqual(len(windows), 6)
        reconstructed_toks: list[int] = []
        for win in windows:
            self.assertLessEqual(win["num_tokens"], 100)
            reconstructed_toks.extend(win["token_ids"])
        # Hard invariant: 0 session tokens dropped or truncated!
        self.assertEqual(reconstructed_toks, all_expected_toks)

        # Smoke test --tasks pareto_rag_only + --ruler-use-raw-question-router
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "pareto_mock.json"
            g4.main(
                [
                    "--mock-llm",
                    "--tasks",
                    "pareto_rag_only",
                    "--ruler-episodes-per-task",
                    "2",
                    "--longmemeval-n",
                    "6",
                    "--longmemeval-max-pages",
                    "16",
                    "--ruler-use-raw-question-router",
                    "--max-new-tokens",
                    "10",
                    "--out-json",
                    str(out_json),
                ]
            )
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            pr = payload["pareto_rag_only"]
            self.assertEqual(pr["ruler_pareto_bm25"]["router_query_mode"], "raw_question")
            for k_arm in ("bm25_rag_k1", "bm25_rag_k2"):
                r_row = pr["ruler_pareto_bm25"]["by_length"]["1K"]["summary"][k_arm]
                self.assertIn("active_tokens_mean", r_row)
                self.assertIn("update_latency_ms", r_row)
                self.assertIn("token_f1_bootstrap", r_row["ci95"])
                l_row = pr["longmemeval_pareto_bm25"]["systems"][k_arm]["overall"]
                self.assertIn("session_hit_at_k", l_row)
                self.assertIn("answer_substr_in_chosen_pages_rate", l_row)
                self.assertIn("active_tokens_mean", l_row)

    def test_true_int4_kv_mutates_past_key_values_not_inputs_embeds(self) -> None:
        torch.manual_seed(17)
        llm = TinyMockCausalLM(vocab_size=256, d_llm=64, num_layers=2, num_heads=4, num_kv_heads=4).eval()
        ctx_ids = torch.randint(5, 200, (1, 32))
        q_ids = torch.randint(5, 200, (1, 6))
        a_ids = torch.randint(5, 200, (1, 4))
        with torch.no_grad():
            ctx_emb = llm.get_input_embeddings()(ctx_ids)
            ctx_emb_before = ctx_emb.clone()
            out_pre = llm(inputs_embeds=ctx_emb, use_cache=True)
            orig_kv = out_pre.past_key_values
            q_kv = m3.quantize_past_key_values_int4(orig_kv, group_size=16)

            # inputs_embeds MUST remain 100% untouched
            self.assertEqual(float((ctx_emb - ctx_emb_before).abs().max().item()), 0.0)
            # past_key_values K and V tensors MUST actually be modified by INT4 quantization
            orig_pairs = m3._extract_kv_pairs(orig_kv)
            q_pairs = m3._extract_kv_pairs(q_kv)
            max_kv_diff = max(
                max(float((ko - kq).abs().max().item()), float((vo - vq).abs().max().item()))
                for (ko, vo), (kq, vq) in zip(orig_pairs, q_pairs)
            )
            self.assertGreater(max_kv_diff, 1e-4)

            nll_fp, _ = m3.compute_nll_and_tok_acc_from_kv(llm, orig_kv, 32, q_ids, a_ids)
            nll_int4, _ = m3.compute_nll_and_tok_acc_from_kv(llm, q_kv, 32, q_ids, a_ids)
            self.assertNotAlmostEqual(nll_fp, nll_int4, places=5)

    def test_true_attention_h2o_eviction_depends_on_real_attention_weights(self) -> None:
        torch.manual_seed(29)
        llm = TinyMockCausalLM(vocab_size=256, d_llm=64, num_layers=2, num_heads=4, num_kv_heads=4).eval()
        ctx_ids = torch.randint(5, 200, (1, 48))
        with torch.no_grad():
            ctx_emb = llm.get_input_embeddings()(ctx_ids)
            pkv, real_attn = m3.compute_real_attention_scores_and_kv(llm, ctx_emb, obs_window=8)
            self.assertEqual(tuple(real_attn.shape), (48,))
            self.assertGreater(float(real_attn.sum().item()), 0.0)

            ev_kv, keep_idx = m3.evict_past_key_values_h2o_attention(pkv, real_attn, keep_budget=20, sink_tokens=4)
            self.assertEqual(int(keep_idx.numel()), 20)
            ev_pairs = m3._extract_kv_pairs(ev_kv)
            self.assertEqual(int(ev_pairs[0][0].shape[2]), 20)

            # Verify changing attention weights directly changes the selected evicted KV indices
            scores_a = torch.zeros(48)
            scores_a[10:15] = 100.0
            scores_b = torch.zeros(48)
            scores_b[28:33] = 100.0
            _, keep_a = m3.evict_past_key_values_h2o_attention(pkv, scores_a, keep_budget=20, sink_tokens=4, recent_ratio=0.25)
            _, keep_b = m3.evict_past_key_values_h2o_attention(pkv, scores_b, keep_budget=20, sink_tokens=4, recent_ratio=0.25)
            self.assertNotEqual(keep_a.tolist(), keep_b.tolist())
            for i in range(10, 15):
                self.assertIn(i, keep_a.tolist())
            for i in range(28, 33):
                self.assertIn(i, keep_b.tolist())

    def test_kv_eval_and_generate_forward_orig_seq_len_position_ids(self) -> None:
        torch.manual_seed(37)
        base_llm = TinyMockCausalLM(vocab_size=256, d_llm=64, num_layers=2, num_heads=4, num_kv_heads=4).eval()
        captured_pos_ids: list[list[int]] = []

        class _SpyPosLLM(torch.nn.Module):
            def __init__(self, inner: TinyMockCausalLM) -> None:
                super().__init__()
                self.inner = inner
                self.config = inner.config

            def get_input_embeddings(self):
                return self.inner.get_input_embeddings()

            def forward(self, *args, position_ids=None, **kwargs):
                if position_ids is not None:
                    captured_pos_ids.append(position_ids[0].tolist())
                return self.inner(*args, position_ids=position_ids, **kwargs)

        spy = _SpyPosLLM(base_llm)
        ctx_ids = torch.randint(5, 200, (1, 48))
        q_ids = torch.randint(5, 200, (1, 5))
        a_ids = torch.randint(5, 200, (1, 3))
        with torch.no_grad():
            ctx_emb = spy.get_input_embeddings()(ctx_ids)
            pkv, attn = m3.compute_real_attention_scores_and_kv(base_llm, ctx_emb, obs_window=8)
            ev_kv, keep_idx = m3.evict_past_key_values_h2o_attention(pkv, attn, keep_budget=20, sink_tokens=4)
            self.assertEqual(int(keep_idx.numel()), 20)

            captured_pos_ids.clear()
            m3.compute_nll_and_tok_acc_from_kv(spy, ev_kv, 20, q_ids, a_ids, orig_seq_len=48)
            self.assertEqual(len(captured_pos_ids), 1)
            self.assertEqual(captured_pos_ids[0], list(range(48, 56)))

            captured_pos_ids.clear()
            m3.greedy_generate_from_kv_cache(spy, ev_kv, 20, q_ids, max_new_tokens=3, eos_token_id=None, orig_seq_len=48)
            # Prefill (5 tokens at 48..52) + 3 decode steps (53, 54, 55)
            self.assertEqual(captured_pos_ids[0], [48, 49, 50, 51, 52])
            self.assertEqual(captured_pos_ids[1], [53])
            self.assertEqual(captured_pos_ids[2], [54])
            self.assertEqual(captured_pos_ids[3], [55])

    def test_2x2_implicit_canonical_numeric_and_coding_n24_helpers(self) -> None:
        self.assertEqual(g5.canonicalize_numeric_answer("$29,000,000"), "29m")
        self.assertEqual(g5.canonicalize_numeric_answer("29M"), "29m")
        self.assertEqual(g5.canonicalize_numeric_answer("29 million"), "29m")
        self.assertEqual(g5.canonicalize_numeric_answer("$400,000"), "400k")

        tok = g3._MockCodeTokenizer(vocab_size=2048)
        suites = g5.build_2x2_edit_episodes(tok, [2 + i for i in range(512)], num_verbatim=8, num_implicit=8)
        self.assertEqual(len(suites["implicit"]), 8)
        answers = [e["answer"] for e in suites["implicit"]]
        self.assertEqual(answers.count("Yes"), 4)
        self.assertEqual(answers.count("No"), 4)

        tasks_24 = g5.build_coding_n24_tasks(num_distractor_files=58)
        self.assertEqual(len(tasks_24), 24)
        for t in tasks_24:
            q = g5.build_pathless_coding_query(t)
            self.assertNotIn(".py", q)
            self.assertNotIn(".md", q)
            self.assertNotIn("/", q)

        # Dynamic margin-gated k_deep adapts across high, medium, and low margins
        self.assertEqual(g5.select_dynamic_margin_k_deep(torch.tensor([10.0, 9.5, 1.0, 0.5, 0.2, 0.1, 0.0])), 2)
        self.assertEqual(g5.select_dynamic_margin_k_deep(torch.tensor([5.0, 4.5, 3.6, 1.0, 0.5, 0.1, 0.0])), 4)
        self.assertEqual(g5.select_dynamic_margin_k_deep(torch.tensor([5.0, 4.8, 4.6, 4.5, 4.4, 4.3, 4.2])), 6)

    def test_gerbang5_all_4_tasks_mock_llm_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "g5_mock_all.json"
            g5.main(
                [
                    "--mock-llm",
                    "--tasks",
                    "all",
                    "--k-exact",
                    "48",
                    "--salience-span-len",
                    "16",
                    "--eval-n",
                    "12",
                    "--train-steps",
                    "6",
                    "--coding-distractors",
                    "4",
                    "--coding-max-tasks",
                    "4",
                    "--out-json",
                    str(out_json),
                ]
            )
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            # 1) 2x2 Edit Ablation (iso-token bank_orig + gate telemetry)
            ab2 = payload["ablation_2x2_edit"]
            arm_toks = set()
            for arm in ("deltaZ_ON_bank_ON", "deltaZ_ON_bank_OFF", "deltaZ_OFF_bank_ON", "deltaZ_OFF_bank_OFF"):
                self.assertIn(arm, ab2)
                self.assertEqual(ab2[arm]["overall"]["max_toc_diff"], 0.0)
                self.assertIn("active_prefix_tokens", ab2[arm]["overall"])
                arm_toks.add(ab2[arm]["overall"]["active_prefix_tokens"])
                for s_k in ("verbatim", "implicit"):
                    self.assertIn(s_k, ab2[arm]["by_suite"])
                    self.assertIn("mean_strip_gate", ab2[arm]["by_suite"][s_k])
                    self.assertIn("bank_write_rate", ab2[arm]["by_suite"][s_k])
            self.assertEqual(len(arm_toks), 1)

            # 2) Iso-Token 1D vs 3D
            ab13 = payload["ablation_1d_vs_3d"]
            self.assertTrue(ab13["iso_token_verified"])
            self.assertEqual(
                ab13["3d_spatial_pyramid"]["active_prefix_tokens"],
                ab13["1d_3layer_flat"]["active_prefix_tokens"],
            )
            self.assertEqual(ab13["3d_spatial_pyramid"]["prefix_cache_invariance"]["max_toc_diff"], 0.0)

            # 3) True KV Baselines + ICAE
            self.assertIn("icae_params", payload["true_kv_baselines"])
            tkv = payload["true_kv_baselines"]["systems"]
            for s in ("full_kv_bf16", "true_int4_kv_cache", "true_attn_h2o_snapkv", "icae_1d_latent_baseline", "lslc_v3_universal"):
                self.assertIn(s, tkv)
                self.assertIn("ppl", tkv[s])
                self.assertIn("active_tokens_mean", tkv[s])

            # 4) Fair Gate-2 + Coding N=24
            fg = payload["fair_g2_and_coding_n24"]
            self.assertIn("lslc_v3_e2e_gpu", fg["gate2_fair_formatting"])
            self.assertIn("normalized_numeric_em", fg["gate2_fair_formatting"]["lslc_v3_e2e_gpu"])
            self.assertIn("by_mode", fg["gate2_fair_formatting"]["lslc_v3_e2e_gpu"])
            self.assertIn("budget", fg["gate2_fair_formatting"]["lslc_v3_e2e_gpu"]["by_mode"])
            self.assertIn("symbolic", fg["gate2_fair_formatting"]["lslc_v3_e2e_gpu"]["by_mode"])
            c24 = fg["gate3_coding_n24"]
            self.assertTrue(c24["path_hints_removed"])
            self.assertIn("bm25_rag_k8", c24["systems"])
            self.assertIn("lslc_v3_k2", c24["systems"])
            self.assertIn("lslc_v3_k4", c24["systems"])
            self.assertIn("lslc_v3_dynamic_k", c24["systems"])
            self.assertIn("dynamic_k_deep_mean", c24["systems"]["lslc_v3_dynamic_k"])
            self.assertGreater(c24["systems"]["lslc_v3_dynamic_k"]["initial_scan_files_mean"], 0)

    def test_longmemeval_incremental_flush_callback_every_k_items(self) -> None:
        torch.manual_seed(11)
        cfg = LSLCConfig.tiny_mock(
            use_contextual_edit_encoding=True,
            contextual_edit_layer=1,
            use_contextual_query_encoding=False,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=48,
            salience_span_len=16,
        )
        llm = TinyMockCausalLM(vocab_size=2048, d_llm=cfg.d_llm, num_layers=cfg.num_llm_layers)
        tok = g3._MockCodeTokenizer(vocab_size=2048)
        from lslc import FrozenQwenLSLCWrapper

        wrapper = FrozenQwenLSLCWrapper(cfg, llm)
        items, meta = g4.load_longmemeval_stratified(n_total=6, seed=2026, allow_fallback=True)
        flushed: list[int] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            out_p = Path(tmpdir) / "lme_flush.json"

            def _cb(snap: dict) -> None:
                flushed.append(int(snap["dataset"]["completed_items"]))
                out_p.write_text(json.dumps(snap), encoding="utf-8")

            res = g4.evaluate_longmemeval_benchmark(
                wrapper,
                tok,
                items,
                meta,
                torch.device("cpu"),
                max_pages=16,
                max_new_tokens=6,
                systems_override=("bm25_rag_k2",),
                flush_callback=_cb,
                flush_every=2,
            )
            self.assertEqual(flushed, [2, 4, 6])
            self.assertTrue(out_p.is_file())
            loaded = json.loads(out_p.read_text(encoding="utf-8"))
            self.assertEqual(loaded["dataset"]["completed_items"], 6)
            self.assertEqual(res["dataset"]["completed_items"], 6)


if __name__ == "__main__":
    unittest.main()
