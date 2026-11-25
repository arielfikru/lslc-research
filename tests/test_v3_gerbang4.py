"""Unit tests for Gerbang 4: Pure-Paraphrase Contrastive Dense Routing + Community Benchmarks (RULER & LongMemEval)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import colab_run_v3_gerbang4_community as g4  # noqa: E402
from colab_run_rebuttal_v2 import BM25PageRetriever  # noqa: E402
from colab_run_v3_gerbang3_coding import _MockCodeTokenizer  # noqa: E402
from lslc import (  # noqa: E402
    FrozenQwenLSLCWrapper,
    LSLCConfig,
    LSLCSystem,
    SparseExactBank,
    TinyMockCausalLM,
    ZeroOverlapParaphraseRoutingBenchmark,
    compute_router_infonce_loss,
    content_words,
    extract_contextual_edit_embeds,
    extract_contextual_page_keys,
)


class TestV3Gerbang4(unittest.TestCase):
    def test_v2_and_v3_g3_checkpoint_backwards_compatibility(self) -> None:
        # 1. Verify v2 HF checkpoint compatibility
        ckpt_path = REPO_ROOT / "hf_hub_release" / "lslc_qwen3_1.7b_adapter.safetensors"
        if ckpt_path.is_file():
            with open(ckpt_path, "rb") as f:
                hdr_len = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hdr_len))
            hdr.pop("__metadata__", None)
            sys_v2 = LSLCSystem(LSLCConfig.qwen3_1_7b())
            sd_v2 = sys_v2.state_dict()
            self.assertEqual(set(sd_v2.keys()), set(hdr.keys()))
            for k, tensor in sd_v2.items():
                self.assertEqual(list(tensor.shape), hdr[k]["shape"])

        # 2. Simulate v3_g3_aligned_ckpt (use_contextual_query_encoding=False by default)
        cfg_g3 = LSLCConfig.tiny_mock(
            use_contextual_edit_encoding=True,
            contextual_edit_layer=1,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=24,
        )
        self.assertFalse(cfg_g3.use_contextual_query_encoding)
        sys_g3 = LSLCSystem(cfg_g3)
        sd_g3 = {k: v.clone() for k, v in sys_g3.state_dict().items()}

        # Default config loads G3 checkpoint with strict=True
        incompat_strict = LSLCSystem(cfg_g3).load_state_dict(sd_g3, strict=True)
        self.assertEqual(len(incompat_strict.missing_keys), 0)
        self.assertEqual(len(incompat_strict.unexpected_keys), 0)

        # G4 contextual query config loads G3 checkpoint cleanly with strict=False (0 unexpected, only new ctx keys missing)
        cfg_g4 = LSLCConfig.tiny_mock(
            use_contextual_edit_encoding=True,
            contextual_edit_layer=1,
            use_contextual_query_encoding=True,
            contextual_query_layer=1,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=24,
        )
        incompat_g4 = LSLCSystem(cfg_g4).load_state_dict(sd_g3, strict=False)
        self.assertEqual(len(incompat_g4.unexpected_keys), 0)
        self.assertGreater(len(incompat_g4.missing_keys), 0)
        for mk in incompat_g4.missing_keys:
            self.assertTrue(
                mk.startswith("router.q_ctx_")
                or mk.startswith("router.k_ctx_")
                or mk.startswith("router.k_toc_")
                or mk.startswith("router.k_bank_"),
                msg=f"Unexpected missing key when upgrading G3 -> G4: {mk}",
            )

        # 3. Verify real local v3 checkpoints load cleanly when present (read-only)
        ckpt_dir = REPO_ROOT.parent / "research_notes" / "lslc_internal" / "checkpoints"
        for name in ("v3_g2_cond_b_ckpt.safetensors", "v3_g3_aligned_ckpt.safetensors"):
            real_ckpt = ckpt_dir / name
            if real_ckpt.is_file():
                from safetensors.torch import load_file

                sd_real = load_file(str(real_ckpt), device="cpu")
                cfg_real_def = LSLCConfig.qwen3_1_7b(
                    use_contextual_edit_encoding=True,
                    contextual_edit_layer=14,
                    salience_mode="learned_surprisal",
                    delta_pooling_mode="cross_attn",
                    router_mode="content_attn",
                    k_exact=24,
                )
                inc_def = LSLCSystem(cfg_real_def).load_state_dict(sd_real, strict=True)
                self.assertEqual(len(inc_def.missing_keys), 0)
                self.assertEqual(len(inc_def.unexpected_keys), 0)

                cfg_real_g4 = LSLCConfig.qwen3_1_7b(
                    use_contextual_edit_encoding=True,
                    contextual_edit_layer=14,
                    use_contextual_query_encoding=True,
                    contextual_query_layer=14,
                    salience_mode="learned_surprisal",
                    delta_pooling_mode="cross_attn",
                    router_mode="content_attn",
                    k_exact=24,
                )
                inc_g4_real = LSLCSystem(cfg_real_g4).load_state_dict(sd_real, strict=False)
                self.assertEqual(len(inc_g4_real.unexpected_keys), 0)
                for mk in inc_g4_real.missing_keys:
                    self.assertTrue(
                        mk.startswith("router.q_ctx_")
                        or mk.startswith("router.k_ctx_")
                        or mk.startswith("router.k_toc_")
                        or mk.startswith("router.k_bank_"),
                        msg=f"Unexpected missing key in {name}: {mk}",
                    )

    def test_contextual_query_router_fp32_permutation_invariance_grads_and_hybrid_stress(self) -> None:
        torch.manual_seed(42)
        cfg = LSLCConfig.tiny_mock(
            use_contextual_edit_encoding=True,
            contextual_edit_layer=1,
            use_contextual_query_encoding=True,
            contextual_query_layer=1,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=12,
        )
        llm = TinyMockCausalLM(vocab_size=512, d_llm=cfg.d_llm, num_layers=cfg.num_llm_layers)
        with torch.no_grad():
            llm.embed_tokens.weight.normal_(mean=0.0, std=1.0)
        wrapper = FrozenQwenLSLCWrapper(cfg, llm)
        embed_fn = llm.get_input_embeddings()

        pages = torch.randint(10, 300, (1, 8, cfg.page_size))
        rq_ids = torch.randint(10, 300, (1, 7))
        perm = torch.tensor([5, 1, 7, 0, 4, 2, 6, 3], dtype=torch.long)

        wrapper.eval()
        with torch.no_grad():
            z1, b1, t1 = wrapper.lslc.encode_book(pages, embed_fn, use_exact_bank=True, vocab_size=512)
            z2, b2, t2 = wrapper.lslc.encode_book(pages[:, perm], embed_fn, use_exact_bank=True, vocab_size=512)
            q_ctx = extract_contextual_edit_embeds(llm, rq_ids, layer_idx=1)
            k_ctx1 = extract_contextual_page_keys(llm, pages, layer_idx=1, pad_token_id=0, chunk_size=4)
            k_ctx2 = extract_contextual_page_keys(llm, pages[:, perm], layer_idx=1, pad_token_id=0, chunk_size=4)
            self.assertEqual(tuple(k_ctx1.shape), (1, 8, cfg.d_llm))
            self.assertLess(float((k_ctx2[0] - k_ctx1[0, perm]).abs().max().item()), 1e-4)

            # Verify .model backbone branch on HF-style wrapper models skips lm_head cleanly
            class _WrapWithModelAttr(torch.nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner
                def get_input_embeddings(self):
                    return self.inner.get_input_embeddings()
                def model(self, input_ids=None, attention_mask=None, output_hidden_states=True, use_cache=False):
                    return self.inner(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states, use_cache=use_cache)

            k_ctx_wrapped = extract_contextual_page_keys(_WrapWithModelAttr(llm), pages, layer_idx=1, pad_token_id=0, chunk_size=4)
            self.assertLess(float((k_ctx_wrapped - k_ctx1).abs().max().item()), 1e-5)

            q_tok_raw = embed_fn(rq_ids)
            q_mean_raw = q_tok_raw.mean(dim=1)
            b1_emb = embed_fn(b1.routing_token_ids())
            b2_emb = embed_fn(b2.routing_token_ids())

            # Router directly keeps logits strictly FP32 even under bfloat16 inputs + contextual_page_keys
            r1_bf16 = wrapper.lslc.router(
                q_mean_raw.bfloat16(),
                t1.bfloat16(),
                k_deep=2,
                k_skim=2,
                query_token_embeds=q_tok_raw.bfloat16(),
                exact_bank_embeds=b1_emb.bfloat16(),
                salience_route_weight=0.0,
                contextual_query_embeds=q_ctx.bfloat16(),
                contextual_page_keys=k_ctx1.bfloat16(),
            )
            self.assertEqual(r1_bf16.logits.dtype, torch.float32)

            r1, asm1 = wrapper.lslc.route_and_assemble(
                z1,
                t1,
                q_mean_raw,
                b1,
                embed_fn,
                query_input_ids=rq_ids,
                salience_route_weight=0.0,
                contextual_query_embeds=q_ctx,
                contextual_page_keys=k_ctx1,
            )
            r2, _ = wrapper.lslc.route_and_assemble(
                z2,
                t2,
                q_mean_raw,
                b2,
                embed_fn,
                query_input_ids=rq_ids,
                salience_route_weight=0.0,
                contextual_query_embeds=q_ctx,
                contextual_page_keys=k_ctx2,
            )
            self.assertEqual(r1.logits.dtype, torch.float32)
            self.assertLess(float((r2.logits[0] - r1.logits[0, perm]).abs().max().item()), 1e-4)
            # Zero active KV tokens added by contextual_page_keys
            r_no_ctx, asm_no_ctx = wrapper.lslc.route_and_assemble(
                z1, t1, q_mean_raw, b1, embed_fn, query_input_ids=rq_ids, salience_route_weight=0.0, contextual_query_embeds=q_ctx
            )
            self.assertEqual(asm1.prefix_embeds.shape[1], asm_no_ctx.prefix_embeds.shape[1])

        # Verify gradient flow into contextual router projections (including k_ctx_mlp) with frozen LLM
        wrapper.lslc.router.train()
        wrapper.lslc.zero_grad(set_to_none=True)
        z_tr, b_tr, t_tr = wrapper.lslc.encode_book(pages, embed_fn, use_exact_bank=True, vocab_size=512)
        q_ctx_tr = extract_contextual_edit_embeds(llm, rq_ids, layer_idx=1)
        k_ctx_tr = extract_contextual_page_keys(llm, pages, layer_idx=1, pad_token_id=0, chunk_size=4)
        ro_tr, _ = wrapper.lslc.route_and_assemble(
            z_tr,
            t_tr,
            embed_fn(rq_ids).mean(dim=1),
            b_tr,
            embed_fn,
            query_input_ids=rq_ids,
            salience_route_weight=0.0,
            contextual_query_embeds=q_ctx_tr,
            contextual_page_keys=k_ctx_tr,
        )
        torch.nn.functional.cross_entropy(ro_tr.logits, torch.tensor([3], dtype=torch.long)).backward()
        for pname, p in wrapper.llm.named_parameters():
            self.assertIsNone(p.grad, f"Frozen LLM param {pname} leaked grad")
        for rname in ("q_ctx_gate", "q_ctx_mlp", "k_ctx_mlp", "k_toc_mlp", "k_bank_mlp"):
            sub = getattr(wrapper.lslc.router, rname)
            gnorm = sum(float(p.grad.abs().sum().item()) for p in sub.parameters() if p.grad is not None)
            self.assertGreater(gnorm, 0.0, f"Missing grad in router.{rname}")

        # Verify hybrid w=2.0 StressNode P=64 stays 100% even under max logit_scale=50 and contextual_page_keys
        num_p = 64
        targets = [i * 5 + 1 for i in range(12)]
        shared_prefix = [310, 311, 312, 313, 314]
        bank_toks = torch.randint(10, 290, (1, num_p, cfg.k_exact))
        for idx_t, p_idx in enumerate(targets):
            bank_toks[0, p_idx, :6] = torch.tensor(shared_prefix + [400 + idx_t])
        bank_p64 = SparseExactBank(
            token_ids=bank_toks,
            page_ids=torch.arange(num_p).view(1, num_p, 1).expand(1, num_p, cfg.k_exact),
            h_coords=torch.zeros((1, num_p, cfg.k_exact), dtype=torch.long),
            w_coords=torch.zeros((1, num_p, cfg.k_exact), dtype=torch.long),
        )
        toc_p64 = torch.randn((1, num_p, 1, cfg.d_llm))
        k_ctx_p64 = torch.randn((1, num_p, cfg.d_llm))
        z_p64 = torch.zeros((1, num_p, 3, cfg.latent_channels, cfg.grid_h, cfg.grid_w))
        with torch.no_grad():
            wrapper.lslc.router.logit_scale.fill_(math.log(50.0))
        wrapper.eval()
        hits = 0
        with torch.no_grad():
            for idx_t, p_idx in enumerate(targets):
                q_s = torch.tensor([shared_prefix + [400 + idx_t]])
                q_s_ctx = extract_contextual_edit_embeds(llm, q_s, layer_idx=1)
                ro_h, _ = wrapper.lslc.route_and_assemble(
                    z_p64,
                    toc_p64,
                    embed_fn(q_s).mean(dim=1),
                    bank_p64,
                    embed_fn,
                    query_input_ids=q_s,
                    salience_route_weight=2.0,
                    contextual_query_embeds=q_s_ctx,
                    contextual_page_keys=k_ctx_p64,
                )
                if int(torch.argmax(ro_h.logits[0], dim=-1).item()) == p_idx:
                    hits += 1
        self.assertEqual(hits, len(targets))

    def test_infonce_contrastive_loss_and_freeze_contract(self) -> None:
        torch.manual_seed(17)
        cfg = LSLCConfig.tiny_mock(
            use_contextual_edit_encoding=True,
            contextual_edit_layer=1,
            use_contextual_query_encoding=True,
            contextual_query_layer=1,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=16,
        )
        llm = TinyMockCausalLM(vocab_size=2048, d_llm=cfg.d_llm, num_layers=cfg.num_llm_layers)
        tok = _MockCodeTokenizer(vocab_size=2048)
        wrapper = FrozenQwenLSLCWrapper(cfg, llm)
        train_pool = [2 + (i % 2046) for i in range(4096)]

        # Unit check compute_router_infonce_loss directly with topic_ids masking & B=1 fallback
        q_vecs = torch.nn.functional.normalize(torch.randn(4, cfg.d_route), dim=-1)
        k_vecs = torch.nn.functional.normalize(torch.randn(4, 8, cfg.d_route), dim=-1)
        tgt = torch.tensor([1, 3, 5, 2], dtype=torch.long)
        t_ids = torch.tensor([10, 10, 20, 30], dtype=torch.long)
        loss_b4, m_b4 = compute_router_infonce_loss(wrapper.lslc.router, q_vecs, k_vecs, tgt, topic_ids=t_ids)
        self.assertTrue(torch.isfinite(loss_b4).item())
        self.assertGreater(m_b4["loss_inbatch"], 0.0)
        loss_b1, m_b1 = compute_router_infonce_loss(wrapper.lslc.router, q_vecs[:1], k_vecs[:1], tgt[:1])
        self.assertTrue(torch.isfinite(loss_b1).item())
        self.assertEqual(m_b1["loss_inbatch"], 0.0)

        # Full Pilar 4A training loop verifies loss drops & non-router modules stay 0.0 diff
        stats = g4.train_pilar4a_contrastive_router(
            wrapper=wrapper,
            tokenizer=tok,
            train_pool=train_pool,
            device=torch.device("cpu"),
            steps=32,
            batch_size=4,
            lr=2e-3,
            seed=77,
            is_mock=True,
        )
        self.assertEqual(stats["max_frozen_param_diff"], 0.0)
        self.assertLess(stats["final_loss_mean"], stats["initial_loss_mean"])

    def test_zero_overlap_paraphrase_benchmark_disjoint_and_hard_assertion(self) -> None:
        ZeroOverlapParaphraseRoutingBenchmark.assert_disjoint_splits()
        self.assertGreaterEqual(len(ZeroOverlapParaphraseRoutingBenchmark.TRAIN_TOPICS), 64)
        self.assertGreaterEqual(len(ZeroOverlapParaphraseRoutingBenchmark.EVAL_TOPICS), 24)

        tok = _MockCodeTokenizer(vocab_size=2048)
        val_pool = [2 + ((i * 7) % 2046) for i in range(2048)]
        for p_cnt in (16, 64):
            ds_ev = ZeroOverlapParaphraseRoutingBenchmark(
                num_episodes=12,
                num_pages=p_cnt,
                page_size=256,
                split="eval",
                tokenizer=tok,
                vocab_size=2048,
                seed=101,
                background_token_pool=val_pool,
            )
            bm25_hits = 0
            for ep in ds_ev:
                q_txt = ep.metadata["query_text"]
                p_txt = ep.metadata["target_page_text"]
                self.assertEqual(content_words(q_txt) & content_words(p_txt), set())
                decoded = tok.decode(ep.context_pages[ep.target_page_idx], skip_special_tokens=True)
                self.assertEqual(content_words(q_txt) & content_words(decoded), set())
                bm25 = BM25PageRetriever(ep.metadata["page_texts"])
                rr = bm25.route(q_txt, k_deep=2, k_skim=0, device=torch.device("cpu"))
                if int(torch.argmax(rr.logits[0], dim=-1).item()) == ep.target_page_idx:
                    bm25_hits += 1
            if p_cnt == 16:
                self.assertLessEqual(bm25_hits / len(ds_ev), 0.25)

    def test_ruler_builders_and_fair_qa_metrics(self) -> None:
        tok = _MockCodeTokenizer(vocab_size=2048)
        val_pool = [2 + ((i * 5) % 2046) for i in range(4096)]

        # Check determinism and mk_niah 48-distractor requirement when is_mock=False
        eps_s1 = g4.build_ruler_episodes("s_niah", 4096, 3, tok, val_pool, page_size=256, seed=11, is_mock=False)
        eps_s2 = g4.build_ruler_episodes("s_niah", 4096, 3, tok, val_pool, page_size=256, seed=11, is_mock=False)
        self.assertTrue(torch.equal(eps_s1[0]["pages"], eps_s2[0]["pages"]))
        self.assertEqual(tuple(eps_s1[0]["pages"].shape), (16, 256))

        eps_mk = g4.build_ruler_episodes("mk_niah", 4096, 2, tok, val_pool, page_size=256, seed=11, mk_distractors=48, is_mock=False)
        self.assertEqual(eps_mk[0]["metadata"]["num_distractor_keys"], 48)
        joined_mk = "\n".join(eps_mk[0]["page_texts"])
        self.assertEqual(joined_mk.count("special magic numbers for"), 49)

        eps_vt = g4.build_ruler_episodes("vt", 4096, 2, tok, val_pool, page_size=256, seed=11, vt_hops=3, is_mock=False)
        joined_vt = "\n".join(eps_vt[0]["page_texts"])
        self.assertIn("ASSIGN VAR_", joined_vt)
        self.assertGreaterEqual(joined_vt.count("ASSIGN "), 12)  # 3 chains x 4 statements
        chain_vars = eps_vt[0]["metadata"]["chain_vars"]
        final_var = chain_vars[-1]
        self.assertIn(final_var, eps_vt[0]["router_query"])
        for hidden_hop in chain_vars[:-1]:
            self.assertNotIn(hidden_hop, eps_vt[0]["router_query"])

        # Unpadded BM25 RAG strips pad_token_id and eos_token_id completely
        class _PadTok:
            pad_token_id = 0
            eos_token_id = 1
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                del add_special_tokens
                return [0, 10, 11, 0, 12, 1]
        rag_ids = g4._encode_unpadded_rag_ids("hello world", _PadTok(), 64, torch.device("cpu"))
        self.assertEqual(rag_ids.tolist(), [[10, 11, 12]])

        # Fair QA metrics + <think> stripping
        m1 = g4.compute_community_qa_metrics("<think>let me check\nwrong 999\n</think>\nThe answer is 4182.", "4182")
        self.assertEqual(m1["extracted_em"], 1.0)
        self.assertEqual(m1["substring_match"], 1.0)
        m2 = g4.compute_community_qa_metrics("It is Comandante C40 grinder.", "Comandante C40")
        self.assertEqual(m2["substring_match"], 1.0)
        self.assertGreater(m2["token_f1"], 0.5)
        self.assertEqual(m2["token_recall"], 1.0)

    def test_longmemeval_stratified_loader_and_schema(self) -> None:
        items_a, meta_a = g4.load_longmemeval_stratified(n_total=6, seed=42, allow_fallback=True)
        items_b, _ = g4.load_longmemeval_stratified(n_total=6, seed=42, allow_fallback=True)
        self.assertEqual([x["question_id"] for x in items_a], [x["question_id"] for x in items_b])
        self.assertEqual(meta_a["dataset_source"], "verified_local_fallback")
        self.assertEqual(len(items_a), 6)
        self.assertEqual(
            meta_a["counts_by_category"],
            {"knowledge-update": 2, "single-session-user": 2, "multi-session": 2},
        )
        for item in items_a:
            self.assertFalse(str(item["question_id"]).endswith("_abs"))
            for k in ("question_id", "question_type", "question", "answer", "answer_session_ids", "haystack_dates", "haystack_session_ids", "haystack_sessions"):
                self.assertIn(k, item)

        # Verify _compose_longmemeval_compact_prompt preserves trailing ChatML assistant turn even on oversized chunks
        class _WordTok:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                del add_special_tokens
                return [abs(hash(w)) % 1000 for w in text.split()] if text.strip() else []

            def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
                del skip_special_tokens
                return " ".join(f"w{i}" for i in ids)

        big_chunk = " ".join(f"tok{i}" for i in range(5000))
        prompt_c = g4._compose_longmemeval_compact_prompt("old summary", big_chunk, _WordTok(), max_summary_tokens=16, max_chunk_tokens=120)
        self.assertTrue(prompt_c.endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n- "))

        # Verify evenly spaced distractor packing across chronology (not front-loaded 0,1,2...)
        tok = _MockCodeTokenizer(vocab_size=2048)
        synth_item = {
            "question_id": "pack_test",
            "question_type": "knowledge-update",
            "question": "What happened?",
            "answer": "gold",
            "answer_session_ids": ["s29"],
            "haystack_dates": [f"2025-01-{i+1:02d}" for i in range(30)],
            "haystack_session_ids": [f"s{i}" for i in range(30)],
            "haystack_sessions": [[{"role": "user", "content": f"session marker {i} info"}] for i in range(30)],
        }
        pack = g4._pack_longmemeval_sessions(synth_item, tok, page_size=256, max_pages=6)
        orig_idxs = [r["orig_session_idx"] for r in pack["session_records"]]
        self.assertIn(29, orig_idxs)
        self.assertLessEqual(pack["num_pages"], 6)
        # Should include sessions past index 15 rather than front-loading [0, 1, 2, 3, 4, 29]
        self.assertTrue(any(10 < idx < 29 for idx in orig_idxs), msg=f"Front-loaded orig_idxs={orig_idxs}")

    def test_gerbang4_full_mock_llm_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "g4_mock.json"
            ckpt_path = Path(tmpdir) / "g4_router_ckpt.safetensors"
            g4.main(
                [
                    "--mock-llm",
                    "--tasks",
                    "all",
                    "--pilar4a-steps",
                    "6",
                    "--pilar4a-eval-n",
                    "4",
                    "--ruler-episodes-per-task",
                    "2",
                    "--longmemeval-n",
                    "6",
                    "--max-new-tokens",
                    "10",
                    "--save-checkpoint",
                    str(ckpt_path),
                    "--out-json",
                    str(out_json),
                ]
            )
            self.assertTrue(ckpt_path.is_file())
            payload = json.loads(out_json.read_text(encoding="utf-8"))

            # Verify Pilar 4A payload
            p4 = payload["pilar4a"]
            self.assertIsNotNone(p4)
            self.assertEqual(p4["train"]["max_frozen_param_diff"], 0.0)
            self.assertIn("num_paraphrase_items", p4["train"])
            for p_key in ("P=16", "P=64"):
                self.assertIn(p_key, p4["zero_overlap_eval"])
                for arm in ("bm25_cpu", "lslc_dense_w0", "lslc_hybrid_w2"):
                    self.assertIn(arm, p4["zero_overlap_eval"][p_key])
                    self.assertIn("top1_ci95", p4["zero_overlap_eval"][p_key][arm])
                    self.assertIn("top2_ci95", p4["zero_overlap_eval"][p_key][arm])
            self.assertIn("router_stress_p64", p4["router_stress_regression"])
            self.assertIn("router_stress_p128", p4["router_stress_regression"])

            # Verify RULER payload
            ruler = payload["ruler"]
            self.assertIsNotNone(ruler)
            for len_key in ("1K", "2K"):
                self.assertIn(len_key, ruler["by_length"])
                for task_name in ("s_niah", "mk_niah", "vt"):
                    self.assertIn(task_name, ruler["by_length"][len_key]["by_task"])
                    for arm in ("full_context_bf16", "bm25_rag_k2", "bm25_rag_k4", "lslc_v3_e2e_gpu"):
                        self.assertIn(arm, ruler["by_length"][len_key]["by_task"][task_name])
                        row = ruler["by_length"][len_key]["by_task"][task_name][arm]
                        self.assertIn("ci95", row)
                        self.assertIn("samples", row)
                        self.assertGreater(len(row["samples"]), 0)
                        for mkey in ("active_prefix_tokens", "memory_update_ms", "per_query_ms", "peak_vram_mb", "max_toc_diff"):
                            self.assertIn(mkey, row)

            # Verify LongMemEval payload
            lme = payload["longmemeval"]
            self.assertIsNotNone(lme)
            for dk in ("mean_num_sessions", "mean_num_pages", "mean_content_tokens"):
                self.assertIn(dk, lme["dataset"])
            for arm in ("bm25_rag_k4", "llm_auto_compact", "lslc_v3_e2e_gpu"):
                self.assertIn(arm, lme["systems"])
                ov = lme["systems"][arm]["overall"]
                self.assertIn("ci95", ov)
                self.assertIn("samples", lme["systems"][arm])
                self.assertGreater(len(lme["systems"][arm]["samples"]), 0)
                for mkey in ("extracted_em", "token_f1", "substring_match", "active_prefix_tokens", "memory_update_ms", "per_query_ms", "peak_vram_mb", "max_toc_diff"):
                    self.assertIn(mkey, ov)
                for cat in ("knowledge-update", "single-session-user", "multi-session"):
                    self.assertIn(cat, lme["systems"][arm]["by_category"])
            lslc_ov = lme["systems"]["lslc_v3_e2e_gpu"]["overall"]
            self.assertEqual(lslc_ov["max_toc_diff"], 0.0)
            self.assertIn("mean_strip_gate", lslc_ov)
            self.assertIn("num_bank_writes_mean", lslc_ov)

    def test_gerbang4_sym_and_diag_probe_mock_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "g4_diag_sym_mock.json"
            g4.main(
                [
                    "--mock-llm",
                    "--tasks",
                    "pilar4a,diag_probe",
                    "--pilar4a-steps",
                    "6",
                    "--pilar4a-eval-n",
                    "4",
                    "--ruler-episodes-per-task",
                    "2",
                    "--longmemeval-max-pages",
                    "16",
                    "--max-new-tokens",
                    "10",
                    "--out-json",
                    str(out_json),
                ]
            )
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertIn("step0_zero_shot_eval", payload["pilar4a"])
            self.assertIn("zero_overlap_eval", payload["pilar4a"])
            diag = payload["diag_probe"]
            self.assertIsNotNone(diag)
            for sys_label in ("v3_g2_cond_b", "v3_g3_aligned"):
                self.assertIn(sys_label, diag["ruler_exact_bank_audit"])
                ov = diag["ruler_exact_bank_audit"][sys_label]["overall"]
                for k in ("all_7_digits_contiguous_rate", "all_7_digits_multiset_rate", "digit_token_recall_mean", "key_in_bank_rate"):
                    self.assertIn(k, ov)
            for mode_name in ("chatml_full", "plain_stage1_full", "chatml_exact_only"):
                self.assertIn(mode_name, diag["ruler_g2_prompt_ablation"])
            self.assertEqual(len(diag["longmemeval_compact_probe"]), 2)
            for rec in diag["longmemeval_compact_probe"]:
                self.assertIn("rolling_summary_first_250", rec)
                self.assertIn("num_windows", rec)

    def test_4span_page_keys_maxsim_router_and_infonce(self) -> None:
        torch.manual_seed(123)
        cfg = LSLCConfig.tiny_mock(
            use_contextual_edit_encoding=True,
            contextual_edit_layer=1,
            use_contextual_query_encoding=True,
            contextual_query_layer=1,
            contextual_page_spans=4,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=16,
        )
        llm = TinyMockCausalLM(vocab_size=512, d_llm=cfg.d_llm, num_layers=cfg.num_llm_layers)
        wrapper = FrozenQwenLSLCWrapper(cfg, llm)
        embed_fn = llm.get_input_embeddings()

        pages = torch.randint(10, 300, (1, 8, cfg.page_size))
        # Make trailing half of page 3 pure pad tokens to test all-pad span fallback (zero NaNs)
        pages[0, 3, cfg.page_size // 2 :] = 0
        rq_ids = torch.randint(10, 300, (1, 6))
        perm = torch.tensor([4, 0, 7, 2, 5, 1, 6, 3], dtype=torch.long)

        wrapper.eval()
        with torch.no_grad():
            z1, b1, t1 = wrapper.lslc.encode_book(pages, embed_fn, use_exact_bank=True, vocab_size=512)
            z2, b2, t2 = wrapper.lslc.encode_book(pages[:, perm], embed_fn, use_exact_bank=True, vocab_size=512)
            k4_1 = extract_contextual_page_keys(llm, pages, layer_idx=1, pad_token_id=0, chunk_size=4, num_page_spans=4)
            k4_2 = extract_contextual_page_keys(llm, pages[:, perm], layer_idx=1, pad_token_id=0, chunk_size=4, num_page_spans=4)
            self.assertEqual(tuple(k4_1.shape), (1, 8, 4, cfg.d_llm))
            self.assertTrue(torch.isfinite(k4_1).all().item())
            self.assertLess(float((k4_2[0] - k4_1[0, perm]).abs().max().item()), 1e-4)

            q_ctx = extract_contextual_edit_embeds(llm, rq_ids, layer_idx=1)
            q_raw = embed_fn(rq_ids)
            r1, asm1 = wrapper.lslc.route_and_assemble(
                z1, t1, q_raw.mean(dim=1), b1, embed_fn,
                query_input_ids=rq_ids, salience_route_weight=0.0,
                contextual_query_embeds=q_ctx, contextual_page_keys=k4_1,
            )
            r2, _ = wrapper.lslc.route_and_assemble(
                z2, t2, q_raw.mean(dim=1), b2, embed_fn,
                query_input_ids=rq_ids, salience_route_weight=0.0,
                contextual_query_embeds=q_ctx, contextual_page_keys=k4_2,
            )
            r_no_ctx, asm_no_ctx = wrapper.lslc.route_and_assemble(
                z1, t1, q_raw.mean(dim=1), b1, embed_fn,
                query_input_ids=rq_ids, salience_route_weight=0.0,
                contextual_query_embeds=q_ctx,
            )
            self.assertEqual(r1.logits.dtype, torch.float32)
            self.assertLess(float((r2.logits[0] - r1.logits[0, perm]).abs().max().item()), 1e-4)
            self.assertEqual(asm1.prefix_embeds.shape[1], asm_no_ctx.prefix_embeds.shape[1])

        # Verify 4D InfoNCE loss + gradients into k_ctx_mlp & q_ctx_mlp with frozen LLM
        wrapper.lslc.router.train()
        wrapper.lslc.zero_grad(set_to_none=True)
        q_vecs = torch.nn.functional.normalize(
            torch.cat(
                [wrapper.lslc.router.encode_query(q_raw.mean(dim=1), query_token_embeds=q_raw, contextual_query_embeds=q_ctx) for _ in range(4)],
                dim=0,
            ),
            dim=-1,
        )
        k_vecs = wrapper.lslc.router.encode_keys(
            t1.expand(4, -1, -1, -1),
            exact_bank_embeds=embed_fn(b1.routing_token_ids()).expand(4, -1, -1, -1),
            contextual_page_keys=k4_1.expand(4, -1, -1, -1),
        )
        self.assertEqual(tuple(k_vecs.shape), (4, 8, 4, cfg.d_route))
        loss4, m4 = compute_router_infonce_loss(
            wrapper.lslc.router, q_vecs, k_vecs, torch.tensor([1, 2, 5, 6], dtype=torch.long)
        )
        self.assertTrue(torch.isfinite(loss4).item())
        self.assertGreater(m4["loss_inbatch"], 0.0)
        loss4.backward()
        for pname, p in wrapper.llm.named_parameters():
            self.assertIsNone(p.grad, f"Frozen LLM param {pname} leaked grad")
        for rname in ("q_ctx_mlp", "k_ctx_mlp"):
            gnorm = sum(float(p.grad.abs().sum().item()) for p in getattr(wrapper.lslc.router, rname).parameters() if p.grad is not None)
            self.assertGreater(gnorm, 0.0, f"Missing grad in router.{rname}")

    def test_salience_gate_contiguous_span_selection_and_zero_pad(self) -> None:
        torch.manual_seed(99)
        for k_ex in (24, 48):
            cfg = LSLCConfig.tiny_mock(
                salience_mode="learned_surprisal",
                delta_pooling_mode="cross_attn",
                router_mode="content_attn",
                k_exact=k_ex,
                salience_span_len=8,
            )
            sys_lslc = LSLCSystem(cfg).eval()
            embed_fn = torch.nn.Embedding(512, cfg.d_llm)
            # Construct a page with pad=0 & eos=1 regions and a strong high-logit 8-token burst at positions 70..77
            page_ids = torch.randint(10, 400, (1, 1, cfg.page_size))
            page_ids[0, 0, 200:] = 0  # pad tail
            page_ids[0, 0, 199] = 1   # eos
            page_ids[0, 0, 70:78] = torch.tensor([450, 451, 452, 453, 454, 455, 456, 457])
            with torch.no_grad():
                # Boost scorer weights along burst embeddings so burst positions 70..77 have clear peak salience
                w_dir = sys_lslc.salience_gate.scorer.weight.data[0]
                w_unit = w_dir / w_dir.norm().clamp(min=1e-6)
                embed_fn.weight.data.normal_(0.0, 0.05)
                for tid in range(450, 458):
                    embed_fn.weight.data[tid] = 5.0 * w_unit

                _, bank, _ = sys_lslc.encode_book(
                    page_ids,
                    embed_fn,
                    use_exact_bank=True,
                    vocab_size=512,
                    special_token_ids=(0, 1),
                    salience_span_len=8,
                )
            picked = bank.token_ids[0, 0].tolist()
            self.assertEqual(len(picked), k_ex)
            self.assertNotIn(0, picked, "pad token leaked into SparseExactBank")
            self.assertNotIn(1, picked, "eos token leaked into SparseExactBank")
            # Entire 8-token burst 450..457 must be captured intact in exact source order
            for i in range(len(picked) - 7):
                if picked[i : i + 8] == list(range(450, 458)):
                    break
            else:
                self.fail(f"8-token contiguous burst not captured intact in k_exact={k_ex}: {picked}")

        # Verify real v3_g2_cond_b_ckpt loads strict=True with k_exact=48, salience_span_len=8
        g2_real = REPO_ROOT.parent / "research_notes" / "lslc_internal" / "checkpoints" / "v3_g2_cond_b_ckpt.safetensors"
        if g2_real.is_file():
            from safetensors.torch import load_file

            sd_g2 = load_file(str(g2_real), device="cpu")
            cfg_k48 = LSLCConfig.qwen3_1_7b(
                use_contextual_edit_encoding=True,
                contextual_edit_layer=14,
                salience_mode="learned_surprisal",
                delta_pooling_mode="cross_attn",
                router_mode="content_attn",
                k_exact=48,
                salience_span_len=8,
            )
            inc = LSLCSystem(cfg_k48).load_state_dict(sd_g2, strict=True)
            self.assertEqual(len(inc.missing_keys), 0)
            self.assertEqual(len(inc.unexpected_keys), 0)

    def test_gerbang4_4span_and_span_salience_probe_mock_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "g4_4span_mock.json"
            ckpt_out = Path(tmpdir) / "g4_4span_ckpt.safetensors"
            g4.main(
                [
                    "--mock-llm",
                    "--tasks",
                    "pilar4a,span_salience_probe",
                    "--pilar4a-page-spans",
                    "4",
                    "--pilar4a-steps",
                    "6",
                    "--pilar4a-eval-n",
                    "4",
                    "--ruler-episodes-per-task",
                    "2",
                    "--max-new-tokens",
                    "10",
                    "--save-checkpoint",
                    str(ckpt_out),
                    "--out-json",
                    str(out_json),
                ]
            )
            self.assertTrue(ckpt_out.is_file())
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["contextual_page_spans"], 4)
            self.assertIn("step0_zero_shot_eval", payload["pilar4a"])
            self.assertIn("zero_overlap_eval", payload["pilar4a"])
            sp = payload["span_salience_probe"]
            self.assertIsNotNone(sp)
            for cfg_label in ("span8_k24", "span8_k48"):
                self.assertIn(cfg_label, sp["configs"])
                entry = sp["configs"][cfg_label]
                self.assertEqual(entry["exact_bank_audit"]["overall"]["special_tokens_in_bank_total"], 0)
                self.assertIn("plain_stage1_full", entry["ruler_modes"])
                self.assertIn("chatml_exact_only", entry["ruler_modes"])
                self.assertIn("router_stress_p128", entry["router_stress_g2"])
                self.assertIn("router_stress_p128", entry["router_stress_active_router"])

    def test_external_dense_logits_and_span16_bge_probe_mock_smoke(self) -> None:
        torch.manual_seed(77)
        cfg = LSLCConfig.tiny_mock(
            use_contextual_edit_encoding=True,
            contextual_edit_layer=1,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=32,
            salience_span_len=16,
        )
        llm = TinyMockCausalLM(vocab_size=512, d_llm=cfg.d_llm, num_layers=cfg.num_llm_layers)
        wrapper = FrozenQwenLSLCWrapper(cfg, llm).eval()
        embed_fn = llm.get_input_embeddings()
        pages = torch.randint(10, 300, (1, 8, cfg.page_size))
        rq_ids = torch.randint(10, 300, (1, 6))
        with torch.no_grad():
            z_book, bank, toc = wrapper.lslc.encode_book(pages, embed_fn, use_exact_bank=True, vocab_size=512)
            q_emb = embed_fn(rq_ids).mean(dim=1)
            ext = torch.tensor([[0.1, 0.2, 0.9, 0.3, -0.1, 0.0, 0.4, 0.05]], dtype=torch.float32)
            r_ext0, asm_ext0 = wrapper.lslc.route_and_assemble(
                z_book, toc, q_emb, bank, embed_fn,
                k_deep=2, k_skim=2, mode="full", training_ste=False,
                query_input_ids=rq_ids, salience_route_weight=0.0, external_dense_logits=ext,
            )
            r_ext2, asm_ext2 = wrapper.lslc.route_and_assemble(
                z_book, toc, q_emb, bank, embed_fn,
                k_deep=2, k_skim=2, mode="full", training_ste=False,
                query_input_ids=rq_ids, salience_route_weight=2.0, external_dense_logits=ext,
            )
            r_base, asm_base = wrapper.lslc.route_and_assemble(
                z_book, toc, q_emb, bank, embed_fn,
                k_deep=2, k_skim=2, mode="full", training_ste=False,
                query_input_ids=rq_ids, salience_route_weight=0.0,
            )
            self.assertEqual(r_ext0.logits.dtype, torch.float32)
            self.assertEqual(r_ext2.logits.dtype, torch.float32)
            self.assertEqual(int(torch.argmax(r_ext0.logits[0], dim=-1).item()), 2)
            # Zero extra KV tokens added by plug-in external dense scores
            self.assertEqual(asm_ext0.prefix_embeds.shape[1], asm_base.prefix_embeds.shape[1])
            self.assertEqual(asm_ext2.prefix_embeds.shape[1], asm_base.prefix_embeds.shape[1])

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "g4_span16_bge_mock.json"
            ckpt_out = Path(tmpdir) / "g4_span16_sal_ckpt.safetensors"
            g4.main(
                [
                    "--mock-llm",
                    "--tasks",
                    "span16_bge_probe",
                    "--k-exact",
                    "48",
                    "--salience-span-len",
                    "16",
                    "--pilar4a-eval-n",
                    "4",
                    "--ruler-episodes-per-task",
                    "2",
                    "--longmemeval-max-pages",
                    "16",
                    "--max-new-tokens",
                    "10",
                    "--save-checkpoint",
                    str(ckpt_out),
                    "--out-json",
                    str(out_json),
                ]
            )
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            pr = payload["span16_bge_probe"]
            self.assertIsNotNone(pr)
            self.assertIn("ruler_span16_k48_zero_train", pr)
            self.assertEqual(pr["ruler_span16_k48_zero_train"]["exact_bank_audit"]["overall"]["special_tokens_in_bank_total"], 0)
            if pr["jalur_a_triggered"]:
                self.assertEqual(pr["jalur_a_train_stats"]["max_frozen_param_diff"], 0.0)
                self.assertIsNotNone(pr["ruler_span16_k48_jalur_a_ft"])
            bge_t = pr["bge_2x2_table"]
            for pk in ("P=16", "P=64"):
                self.assertIn(pk, bge_t["zero_overlap_eval"])
                for arm in ("bm25_cpu", "bge_dense_1span_w0", "bge_dense_4span_w0", "lslc_bge_hybrid_w2"):
                    self.assertIn(arm, bge_t["zero_overlap_eval"][pk])
            for sk in ("router_stress_p64", "router_stress_p128"):
                self.assertIn(sk, bge_t["router_stress"])
                self.assertIn("bge_dense_w0_top1_recall", bge_t["router_stress"][sk])
                self.assertIn("lslc_bge_hybrid_w2_top1_recall", bge_t["router_stress"][sk])
            lme_m = pr["longmemeval_sanity_n6"]["modes"]
            for mode_name in ("chatml_exact_only", "chatml_full", "plain_stage1_full"):
                self.assertIn(mode_name, lme_m)
                self.assertEqual(lme_m[mode_name]["overall"]["max_toc_diff"], 0.0)
                self.assertEqual(len(lme_m[mode_name]["samples"]), 6)

    def test_router_noise_floor_and_confidence_gate(self) -> None:
        torch.manual_seed(2026)
        cfg = LSLCConfig.tiny_mock(
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=24,
            salience_span_len=16,
        )
        llm = TinyMockCausalLM(vocab_size=512, d_llm=cfg.d_llm, num_layers=cfg.num_llm_layers)
        with torch.no_grad():
            llm.embed_tokens.weight.normal_(mean=0.0, std=1.0)
        wrapper = FrozenQwenLSLCWrapper(cfg, llm).eval()
        embed_fn = llm.get_input_embeddings()

        num_p = 64
        bank_toks = torch.randint(10, 200, (1, num_p, cfg.k_exact))
        # (a) 0-overlap / noise query (token IDs 300..307 do NOT appear in any bank except 1 accidental unigram on page 5)
        bank_toks[0, 5, 0] = 300
        # (b) Exact multi-token entity StressNode on page 42 (tokens 420..424)
        entity_toks = [420, 421, 422, 423, 424]
        bank_toks[0, 42, : len(entity_toks)] = torch.tensor(entity_toks)
        bank = SparseExactBank(
            token_ids=bank_toks,
            page_ids=torch.arange(num_p).view(1, num_p, 1).expand(1, num_p, cfg.k_exact),
            h_coords=torch.zeros((1, num_p, cfg.k_exact), dtype=torch.long),
            w_coords=torch.zeros((1, num_p, cfg.k_exact), dtype=torch.long),
        )
        toc = torch.zeros((1, num_p, 1, cfg.d_llm))
        z_dummy = torch.zeros((1, num_p, 3, cfg.latent_channels, cfg.grid_h, cfg.grid_w))

        with torch.no_grad():
            # Case A: Strong external dense peak on page 17 + noisy/0-overlap query -> must preserve page 17!
            ext_a = torch.randn((1, num_p), dtype=torch.float32) * 0.05
            ext_a[0, 17] = 0.45  # clear BGE peak (~+6 sigma)
            q_zero = torch.tensor([[300, 301, 302, 303, 304, 305]], dtype=torch.long)
            r_a, _ = wrapper.lslc.route_and_assemble(
                z_dummy, toc, embed_fn(q_zero).mean(dim=1), bank, embed_fn,
                mode="toc_only", training_ste=False,
                query_input_ids=q_zero, salience_route_weight=1.5, external_dense_logits=ext_a,
            )
            self.assertEqual(int(torch.argmax(r_a.logits[0], dim=-1).item()), 17)

            # Case B: Exact multi-token entity on page 42 vs misleading dense peak on page 9 -> must select page 42!
            ext_b = torch.randn((1, num_p), dtype=torch.float32) * 0.05
            ext_b[0, 9] = 0.25  # misleading dense peak
            q_ent = torch.tensor([[310, 311] + entity_toks], dtype=torch.long)
            r_b, _ = wrapper.lslc.route_and_assemble(
                z_dummy, toc, embed_fn(q_ent).mean(dim=1), bank, embed_fn,
                mode="toc_only", training_ste=False,
                query_input_ids=q_ent, salience_route_weight=1.5, external_dense_logits=ext_b,
            )
            self.assertEqual(int(torch.argmax(r_b.logits[0], dim=-1).item()), 42)

    def test_gerbang4_stage2_full_mock_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "g4_stage2_full_mock.json"
            ckpt_out = Path(tmpdir) / "v3_g4_final_ckpt.safetensors"
            g4.main(
                [
                    "--mock-llm",
                    "--tasks",
                    "stage2_full",
                    "--k-exact",
                    "48",
                    "--salience-span-len",
                    "16",
                    "--salience-route-weight",
                    "1.5",
                    "--realign-salience-steps",
                    "8",
                    "--realign-projector-steps",
                    "6",
                    "--pilar4a-eval-n",
                    "4",
                    "--ruler-episodes-per-task",
                    "2",
                    "--longmemeval-n",
                    "6",
                    "--longmemeval-max-pages",
                    "16",
                    "--longmemeval-compact-window",
                    "2800",
                    "--longmemeval-compact-chunk",
                    "2600",
                    "--max-new-tokens",
                    "10",
                    "--save-checkpoint",
                    str(ckpt_out),
                    "--out-json",
                    str(out_json),
                ]
            )
            self.assertTrue(ckpt_out.is_file())
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["realign"]["salience_gate"]["max_frozen_param_diff"], 0.0)
            self.assertEqual(payload["realign"]["salience_gate"]["mixed_natural_replay_ratio"], 0.3)
            self.assertEqual(payload["realign"]["projector_chatml_qa"]["max_frozen_param_diff"], 0.0)
            bge_p4 = payload["bge_pilar4a"]
            self.assertIsNotNone(bge_p4)
            for pk in ("P=16", "P=64"):
                self.assertIn("lslc_bge_hybrid_w1.5", bge_p4["zero_overlap_eval"][pk])
            self.assertIn("router_stress_p128", bge_p4["router_stress"])
            self.assertIn("lslc_exact_only", payload["ruler"]["by_length"]["1K"]["summary"])
            self.assertIn("lslc_v3_e2e_gpu", payload["longmemeval"]["systems"])
            self.assertEqual(payload["longmemeval"]["systems"]["lslc_v3_e2e_gpu"]["overall"]["max_toc_diff"], 0.0)
            self.assertEqual(len(payload["longmemeval"]["systems"]["lslc_v3_e2e_gpu"]["samples"]), 6)


if __name__ == "__main__":
    unittest.main()
