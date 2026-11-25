"""Unit tests for Path A vs. Path B Tie-Breaker Suite & 5 Code-Audit Hardening Pillars."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from colab_run_v3_gerbang3_coding import _MockCodeTokenizer
from colab_run_v3_gerbang4_community import _PlugInBiEncoderTOC
from colab_run_v3_path_tiebreaker import (
    StandaloneDiscreteSpanBank,
    _build_append_192tok_bank,
    benchmark_prefix_kv_cache_reuse,
    compute_query_aware_snapkv_scores_and_kv,
    exact_mcnemar_test,
    main as tiebreaker_main,
    mask_old_value_in_exact_bank,
    paired_bootstrap_ci95,
    run_ablation_edit_masked_old,
    run_gate2_tiebreaker_n200,
)
from lslc import FrozenQwenLSLCWrapper, LSLCConfig, SparseExactBank, TinyMockCausalLM


def _make_mock_wrapper(k_exact: int = 48):
    vocab_size = 2048
    cfg = LSLCConfig.tiny_mock(
        use_contextual_edit_encoding=True,
        contextual_edit_layer=1,
        use_contextual_query_encoding=False,
        salience_mode="learned_surprisal",
        delta_pooling_mode="cross_attn",
        router_mode="content_attn",
        k_exact=k_exact,
        salience_span_len=16,
    )
    llm = TinyMockCausalLM(vocab_size=vocab_size, d_llm=cfg.d_llm, num_layers=cfg.num_llm_layers)
    tok = _MockCodeTokenizer(vocab_size=vocab_size)
    wrapper = FrozenQwenLSLCWrapper(cfg, llm)
    val_pool = [2 + ((i * 7) % (vocab_size - 2)) for i in range(4096)]
    bge = _PlugInBiEncoderTOC("BAAI/bge-small-en-v1.5", device=torch.device("cpu"), is_mock=True)
    return wrapper, tok, val_pool, bge


class TestV3PathTiebreaker(unittest.TestCase):
    def test_exact_mcnemar_and_paired_bootstrap_ci95(self) -> None:
        # 10 discordant pairs all favoring A -> exact two-sided p = 2 * (0.5^10) = 0.001953125
        a = [1] * 10 + [1] * 5 + [0] * 5
        b = [0] * 10 + [1] * 5 + [0] * 5
        mc = exact_mcnemar_test(a, b)
        self.assertEqual(mc["n10_a_only"], 10)
        self.assertEqual(mc["n01_b_only"], 0)
        self.assertAlmostEqual(mc["p_value_exact"], 2.0 / 1024.0, places=5)
        self.assertAlmostEqual(mc["diff_a_minus_b"], 0.5, places=4)

        # Tied discordant pairs -> p == 1.0
        mc_tie = exact_mcnemar_test([1, 1, 0, 0], [0, 0, 1, 1])
        self.assertAlmostEqual(mc_tie["p_value_exact"], 1.0, places=6)

        # Paired bootstrap CI strictly positive when A dominates B
        pb = paired_bootstrap_ci95(a, b, n_boot=500, seed=2026)
        self.assertGreater(pb["paired_diff_ci95"][0], 0.0)
        self.assertIn("mcnemar", pb)
        self.assertAlmostEqual(pb["mcnemar"]["p_value_exact"], mc["p_value_exact"], places=6)

    def test_standalone_discrete_never_calls_encoder_and_bf16_storage(self) -> None:
        wrapper, tok, _, bge = _make_mock_wrapper(k_exact=48)
        device = torch.device("cpu")
        embed_fn = wrapper.llm.get_input_embeddings()

        def _forbid_encoder(*args, **kwargs):
            raise AssertionError("StandaloneDiscreteSpanBank MUST NEVER call LayeredLatentEncoder!")

        wrapper.lslc.encoder.forward = _forbid_encoder  # type: ignore[method-assign]
        wrapper.lslc.encoder.encode_pages = _forbid_encoder  # type: ignore[method-assign]

        sd = StandaloneDiscreteSpanBank(
            cfg=wrapper.cfg,
            salience_gate=wrapper.lslc.salience_gate,
            hw_coord_embed=wrapper.lslc.hw_coord_embed,
            delta_updater=wrapper.lslc.delta_updater,
            bge_toc=bge,
            device=device,
            k_exact=48,
            salience_span_len=16,
        )
        pages = torch.randint(2, 500, (1, 8, wrapper.cfg.page_size), dtype=torch.long)
        enc = sd.encode_book(
            pages, embed_fn, tok, len(embed_fn.weight), amp_dtype=torch.float32, autocast_enabled=False
        )
        self.assertEqual(enc["bge_keys_bf16"].dtype, torch.bfloat16)
        # Honest Path B storage: before edit = 48*4 = 192 B/p bank + 3072 B/p BGE = 3264 B/p;
        # after edit (+192 B/p base_token_ids) = 384 B/p bank + 3072 B/p BGE = 3456 B/p!
        self.assertEqual(enc["bge_keys_bytes_per_page"], 3072.0)
        self.assertEqual(enc["exact_bank_bytes_per_page_before_edit"], 192.0)
        self.assertEqual(enc["exact_bank_bytes_per_page_after_edit"], 384.0)
        self.assertEqual(enc["stored_bytes_per_page_before_edit"], 3264.0)
        self.assertEqual(enc["stored_bytes_per_page_after_edit"], 3456.0)
        self.assertIsNone(enc["bank"].page_ids)
        self.assertIsNone(enc["bank"].h_coords)
        self.assertIsNone(enc["bank"].w_coords)

        edit_ids = torch.randint(2, 500, (1, 16), dtype=torch.long)
        bank2, _, _, edit_ms = sd.apply_edit(
            enc["bank"], 2, edit_ids, wrapper.llm, embed_fn, len(embed_fn.weight),
            amp_dtype=torch.float32, autocast_enabled=False,
        )
        self.assertGreaterEqual(edit_ms, 0.0)
        # Verify honest post-edit bank storage includes +192 B/p base_token_ids when mutated
        if bank2.base_token_ids is not None:
            self.assertEqual(bank2.storage_bytes(), 8 * 48 * 8)
        rq_ids = torch.randint(2, 500, (1, 8), dtype=torch.long)
        r_out, pref, rt_ms = sd.route_and_assemble(
            bank2, enc["bge_keys_bf16"], rq_ids, "sample query", embed_fn, k_deep=4, salience_route_weight=1.5
        )
        self.assertEqual(tuple(pref.shape), (1, 4 * 48, wrapper.cfg.d_llm))
        self.assertEqual(r_out.deep_idx.shape, (1, 4))
        self.assertGreaterEqual(rt_ms, 0.0)

    def test_gated_bm25_and_standalone_share_identical_gate(self) -> None:
        wrapper, tok, val_pool, bge = _make_mock_wrapper(k_exact=48)
        res = run_gate2_tiebreaker_n200(
            wrapper, tok, val_pool, torch.device("cpu"), bge_toc=bge,
            eval_n=8, seeds=(2026,), is_mock=True,
        )
        s0 = res["by_seed"]["2026"]["systems"]
        g_bm25 = s0["gated_bm25_rag"]["gate_telemetry"]
        g_disc = s0["standalone_discrete_span_bank"]["gate_telemetry"]
        # Exact same accept/reject gate -> bit-for-bit identical TP/TN/FP/FN, FAR, FRR, mean gates!
        self.assertEqual(g_bm25, g_disc)
        self.assertEqual(g_bm25, s0["raw_text_same_pages_k4"]["gate_telemetry"])
        self.assertEqual(g_bm25, s0["standalone_append_192tok"]["gate_telemetry"])
        # Verify _build_append_192tok_bank retains latest rolling tail (never truncating latest edit)
        base_b = SparseExactBank(token_ids=torch.arange(48, dtype=torch.int32).view(1, 1, 48))
        app_b = _build_append_192tok_bank(
            base_b, {0: list(range(100, 160))}, k_exact=48, span_len=16, sep_token_id=999
        )
        self.assertEqual(app_b.token_ids[0, 0, :16].tolist(), list(range(16)))
        self.assertEqual(int(app_b.token_ids[0, 0, 16].item()), 999)
        self.assertEqual(app_b.token_ids[0, 0, 17:].tolist(), list(range(129, 160)))
        self.assertEqual(len(s0["lslc_full"]["episodes"]), 8)
        self.assertIn("normalized_w2n_em", s0["standalone_discrete_span_bank"])
        self.assertIn("randomized_static_retention", s0["standalone_discrete_span_bank"]["by_query_type"])
        self.assertIn("blind_edit_routing_slice", res["by_seed"]["2026"])
        self.assertIn("edit_turn_routing_accuracy", res["by_seed"]["2026"]["blind_edit_routing_slice"])
        self.assertIn("mcnemar", res["pooled_pairwise_norm_em"]["standalone_discrete_span_bank__vs__standalone_append_192tok"])

    def test_query_aware_snapkv_uses_query_observation_window(self) -> None:
        class _SpyAttnLLM(nn.Module):
            """Returns controlled attention where query rows attend ONLY to k=17 while context rows attend ONLY to k=5."""

            def __init__(self, ctx_len: int, target_k: int = 17, decoy_k: int = 5) -> None:
                super().__init__()
                self.ctx_len = ctx_len
                self.target_k = target_k
                self.decoy_k = decoy_k
                self.config = SimpleNamespace(_attn_implementation="sdpa", use_cache=False)
                self.seen_impls: list[str] = []

            def set_attn_implementation(self, impl: str) -> None:
                self.config._attn_implementation = impl
                self.seen_impls.append(impl)

            def forward(self, inputs_embeds=None, attention_mask=None, use_cache=True, output_attentions=False):
                bsz, total_len, d = inputs_embeds.shape
                k = torch.zeros((bsz, 2, total_len, 8), dtype=inputs_embeds.dtype)
                v = torch.zeros((bsz, 2, total_len, 8), dtype=inputs_embeds.dtype)
                attn = torch.zeros((bsz, 2, total_len, total_len), dtype=torch.float32)
                attn[:, :, : self.ctx_len, self.decoy_k] = 1.0  # Context-only rows point to decoy_k=5
                attn[:, :, self.ctx_len :, self.target_k] = 1.0  # Query rows point to target_k=17
                return SimpleNamespace(past_key_values=((k, v),), attentions=(attn,) if output_attentions else None)

        ctx_len, q_len = 32, 6
        spy = _SpyAttnLLM(ctx_len=ctx_len, target_k=17, decoy_k=5)
        ctx_emb = torch.randn(1, ctx_len, 16)
        q_emb = torch.randn(1, q_len, 16)
        ctx_pkv, attn_scores, _ = compute_query_aware_snapkv_scores_and_kv(spy, ctx_emb, q_emb)

        self.assertEqual(int(ctx_pkv[0][0].shape[2]), ctx_len)
        self.assertEqual(int(attn_scores.argmax().item()), 17)
        self.assertEqual(float(attn_scores[5].item()), 0.0)
        # Must switch to eager during extraction and cleanly restore sdpa afterward!
        self.assertEqual(spy.seen_impls, ["eager", "sdpa"])
        self.assertEqual(spy.config._attn_implementation, "sdpa")

    def test_prefix_kv_cache_reuse_exact_zero_diff(self) -> None:
        wrapper, tok, val_pool, bge = _make_mock_wrapper(k_exact=48)
        res = benchmark_prefix_kv_cache_reuse(
            wrapper, tok, val_pool, torch.device("cpu"), bge_toc=bge, page_counts=(4, 8), num_trials=2, is_mock=True
        )
        self.assertEqual(res["architecture"], "PathB_TrueTwoBank_ZeroZBook")
        for p_key in ("P=4", "P=8"):
            self.assertEqual(res[p_key]["bank1_embed_max_abs_diff"], 0.0)
            self.assertEqual(res[p_key]["bank1_pkv_max_abs_diff"], 0.0)
            self.assertLess(res[p_key]["final_logits_max_abs_diff"], 1e-4)
            # Explicitly verify first (p0), middle (p1), and last (pEnd) ALL achieve exact 0.0 Bank-1 KV diff!
            for pos_k in ("first_p0", "middle_p1", "last_p_end"):
                pos_rec = res[p_key]["fixed_working_set_reuse"][pos_k]
                self.assertEqual(pos_rec["bank1_embed_max_abs_diff"], 0.0)
                self.assertEqual(pos_rec["bank1_pkv_max_abs_diff"], 0.0)
                self.assertNotIn(pos_rec["edited_page_idx"], pos_rec["bank1_pages"])
                self.assertEqual(pos_rec["bank2_pages"], [pos_rec["edited_page_idx"]])
        self.assertIn("gate2_multiturn_rerouting", res["post_rerouting_cache_hit_telemetry"])
        self.assertIn("coding_multiturn_rerouting", res["post_rerouting_cache_hit_telemetry"])

    def test_mask_old_value_and_5cell_edit_ablation(self) -> None:
        wrapper, tok, val_pool, bge = _make_mock_wrapper(k_exact=24)
        old_toks = tok.encode("TLS-1.2", add_special_tokens=False)
        row = torch.full((1, 2, 24), 5, dtype=torch.int32)
        row[0, 1, 4 : 4 + len(old_toks)] = torch.tensor(old_toks, dtype=torch.int32)
        bank = SparseExactBank(
            token_ids=row,
            page_ids=torch.zeros((1, 2, 24), dtype=torch.int16),
            h_coords=torch.zeros((1, 2, 24), dtype=torch.uint8),
            w_coords=torch.zeros((1, 2, 24), dtype=torch.uint8),
        )
        masked_bank, n_masked = mask_old_value_in_exact_bank(bank, 1, "TLS-1.2", tok, len(tok))
        self.assertGreaterEqual(n_masked, len(old_toks))
        self.assertEqual(masked_bank.token_ids.shape, bank.token_ids.shape)
        self.assertNotIn("TLS-1.2", tok.decode(masked_bank.token_ids[0, 1].tolist(), skip_special_tokens=True))

        ab = run_ablation_edit_masked_old(
            wrapper, tok, val_pool, torch.device("cpu"), bge_toc=bge,
            num_verbatim=4, num_implicit=4, is_mock=True,
        )
        self.assertTrue(ab["iso_token_verified"])
        self.assertIn("deltaZ_ON_bank_masked_old", ab["arms"])
        self.assertIn("deltaZ_ON_bank_masked_old__vs__deltaZ_ON_bank_OFF", ab["pairwise_verbatim_semantic_acc"])

    def test_full_mock_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpd:
            out_json = Path(tmpd) / "tiebreaker_mock.json"
            tiebreaker_main(
                [
                    "--mock-llm",
                    "--tasks",
                    "all",
                    "--seeds",
                    "2026,2027",
                    "--out-json",
                    str(out_json),
                ]
            )
            data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertIn("git_sha", data)
            self.assertIn("checkpoint_sha256", data)
            self.assertIn("trainable_param_breakdown", data)
            self.assertEqual(data["trainable_param_breakdown"]["hw_coord_embed_params"], 0)
            self.assertIsNotNone(data["train_pathb_qwen"])
            self.assertEqual(data["train_pathb_qwen"]["max_frozen_param_diff"], 0.0)
            self.assertTrue(data["coding_tiebreaker_n24"]["single_run_all_baselines"])
            self.assertTrue(data["coding_tiebreaker_n24"]["identical_pages_verified"])
            self.assertIn("standalone_discrete_k4", data["coding_tiebreaker_n24"]["systems"])
            self.assertIn("raw_text_same_pages_k4", data["coding_tiebreaker_n24"]["systems"])
            self.assertIn("generated_patches", data["coding_tiebreaker_n24"]["systems"]["lslc_full_k2"]["episodes"][0])
            self.assertIn("heuristic_span16_tfidf", data["ruler_bge_and_unified_profile"]["ruler_short_query"]["by_length"]["1K"]["systems"])
            self.assertEqual(data["longmemeval_discrete_probe"]["packing_mode"], "gold_prioritized_64p")
            self.assertIn("standalone_discrete_k48_d4", data["longmemeval_discrete_probe"]["systems"])
            self.assertIn("value_em", data["longmemeval_discrete_probe"]["systems"]["standalone_discrete_k48_d4"])
            self.assertTrue(data["ood_kv_baselines"]["icae_strawman_removed"])
            self.assertTrue(data["ood_kv_baselines"]["snapkv_config"]["per_layer_per_kv_head_gqa"])
            self.assertIn("encode_or_evict_ms", data["ood_kv_baselines"]["systems"]["query_aware_snapkv"])
            self.assertIn("decode_ms_per_tok", data["ood_kv_baselines"]["systems"]["query_aware_snapkv"])
            self.assertEqual(
                data["ruler_bge_and_unified_profile"]["prefix_kv_cache_reuse"]["P=4"]["bank1_pkv_max_abs_diff"],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
