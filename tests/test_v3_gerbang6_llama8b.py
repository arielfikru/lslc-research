"""Unit tests for Gerbang 6: Cross-Family Path (B) Replication on Llama-3.1-8B-Instruct."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from colab_run_m3 import compute_memory_safe_obs_attention_and_kv, evict_past_key_values_h2o_attention
from colab_run_v3_gerbang3_coding import _MockCodeTokenizer
from colab_run_v3_gerbang4_community import _PlugInBiEncoderTOC
from colab_run_v3_gerbang5_ablations import build_coding_n24_tasks
from colab_run_v3_gerbang6_llama8b_pathb import (
    LLAMA3_ASSIST_OPEN,
    LLAMA3_USER_OPEN,
    benchmark_two_bank_discrete_kv_reuse,
    format_llama3_edit_prompt,
    format_llama3_homonym_prompt,
    format_llama3_qa_prompt,
    get_llama3_stop_token_ids,
    main as gerbang6_main,
    run_llama_gate2_n200,
    train_pathb_llama8b,
)
from colab_run_v3_path_tiebreaker import run_longmemeval_discrete_probe
from lslc import (
    FrozenQwenLSLCWrapper,
    LSLCConfig,
    PathBDiscreteMemory,
    SalienceGate,
    SparseExactBank,
    TinyMockCausalLM,
    canonicalize_w2n_answer,
    compute_value_em,
    heuristic_span16_tfidf_bank,
    load_pathb_checkpoint,
    save_pathb_checkpoint,
)


class TestV3Gerbang6Llama8BPathB(unittest.TestCase):
    def test_llama31_prompt_templates_zero_qwen_tags_and_stop_ids(self) -> None:
        task = build_coding_n24_tasks(num_distractor_files=2)[0]
        prompts = [
            LLAMA3_USER_OPEN,
            format_llama3_qa_prompt("What is the budget?"),
            format_llama3_edit_prompt(task, task.caller_files[0], "def f(): pass\n"),
            format_llama3_homonym_prompt(task, "def g(): pass\n"),
        ]
        for p in prompts:
            self.assertNotIn("<|im_start|>", p)
            self.assertNotIn("<|im_end|>", p)
            self.assertNotIn("<think>", p)
        self.assertIn("<|begin_of_text|><|start_header_id|>user<|end_header_id|>", LLAMA3_USER_OPEN)
        self.assertIn(LLAMA3_ASSIST_OPEN, prompts[1])
        self.assertTrue(prompts[2].endswith(f"{LLAMA3_ASSIST_OPEN}```python\n"))

        class _FakeLlamaTok:
            eos_token_id = [128001, 128009]

            def convert_tokens_to_ids(self, tok: str) -> int:
                return {"<|eot_id|>": 128009, "<|end_of_text|>": 128001}.get(tok, -1)

        stops = get_llama3_stop_token_ids(_FakeLlamaTok())
        self.assertEqual(stops, {128001, 128009})

    def test_dynamic_llama8b_config_d4096_layer16(self) -> None:
        cfg = LSLCConfig.from_model_name(
            "meta-llama/Llama-3.1-8B-Instruct",
            use_contextual_edit_encoding=True,
            salience_mode="learned_surprisal",
            k_exact=48,
            salience_span_len=16,
        )
        self.assertEqual(cfg.d_llm, 4096)
        self.assertEqual(cfg.num_llm_layers, 32)
        self.assertEqual(cfg.num_kv_heads, 8)
        self.assertEqual(cfg.llm_head_dim, 128)

        # Verify dynamic sync_from_hf_config sets d_llm=4096, layers=32, contextual_edit_layer=16
        cfg2 = LSLCConfig.from_model_name("custom-model", use_contextual_edit_encoding=True)
        fake_hf = SimpleNamespace(hidden_size=4096, num_hidden_layers=32, num_key_value_heads=8, num_attention_heads=32, head_dim=128)
        cfg2.sync_from_hf_config(fake_hf)
        self.assertEqual(cfg2.d_llm, 4096)
        self.assertEqual(cfg2.num_llm_layers, 32)
        self.assertEqual(cfg2.contextual_edit_layer, 16)

    def test_pathb_no_encoder_frozen_llm_diff_zero_and_checkpoint_roundtrip(self) -> None:
        vocab_size = 2048
        cfg = LSLCConfig.tiny_mock(
            model_name="mock-llama-3.1-8b",
            use_contextual_edit_encoding=True,
            contextual_edit_layer=1,
            salience_mode="learned_surprisal",
            k_exact=48,
            salience_span_len=16,
        )
        llm = TinyMockCausalLM(vocab_size=vocab_size, d_llm=cfg.d_llm, num_layers=cfg.num_llm_layers)
        for p in llm.parameters():
            p.requires_grad_(False)
        llm.eval()
        tok = _MockCodeTokenizer(vocab_size=vocab_size)
        bge = _PlugInBiEncoderTOC("BAAI/bge-small-en-v1.5", device=torch.device("cpu"), is_mock=True)
        pathb = PathBDiscreteMemory(cfg, bge_toc=bge, edit_gate_hidden_dim=64)
        self.assertIsNone(pathb.encoder)
        self.assertIsNone(pathb.hw_coord_embed)
        st = pathb.storage_breakdown_bytes_per_page()
        self.assertEqual(st["stored_bytes_per_page_before_edit"], 3264)
        self.assertEqual(st["stored_bytes_per_page_after_edit"], 3456)
        self.assertEqual(pathb.trainable_param_breakdown(llm)["hw_coord_embed_params"], 0)
        self.assertEqual(pathb.trainable_param_breakdown(llm)["delta_updater_params"], 0)

        # Verify real 1.7B / 8B lean Path B parameter counts (hidden_dim=64, ZERO hw_coord_embed)
        pb_qwen = PathBDiscreteMemory(LSLCConfig.qwen3_1_7b(k_exact=48, salience_span_len=16), edit_gate_hidden_dim=64)
        pb_llama = PathBDiscreteMemory(LSLCConfig.llama31_8b(k_exact=48, salience_span_len=16), edit_gate_hidden_dim=64)
        self.assertLess(pb_qwen.trainable_parameter_count(), 350_000)
        self.assertLess(pb_llama.trainable_parameter_count(), 620_000)

        train_pool = [2 + (i % (vocab_size - 2)) for i in range(2048)]
        tr = train_pathb_llama8b(
            llm, pathb, tok, train_pool, torch.device("cpu"),
            salience_steps=10, edit_gate_steps=10, is_mock=True,
        )
        self.assertEqual(tr["max_frozen_param_diff"], 0.0)
        self.assertEqual(tr["special_tokens_in_bank"], 0)
        self.assertTrue(tr["disjointness_audit"]["verified"])

        with tempfile.TemporaryDirectory() as tmpd:
            ck = Path(tmpd) / "lslc_pathb_llama31_8b.safetensors"
            saved = save_pathb_checkpoint(pathb, ck)
            pathb2 = PathBDiscreteMemory(cfg, bge_toc=bge)
            loaded = load_pathb_checkpoint(pathb2, ck)
            self.assertEqual(saved["sha256"], loaded["sha256"])
            self.assertEqual(loaded["missing_keys"], [])
            self.assertEqual(loaded["unexpected_keys"], [])

    def test_memory_safe_layerwise_obs_attention_matches_exact_attention(self) -> None:
        """Verify memory-safe layerwise Q_obs @ K^T matches naive full output_attentions=True within 1e-5."""
        torch.manual_seed(0)
        d_model, n_q, n_kv, hd = 32, 4, 2, 8

        class _ToyRotary(nn.Module):
            def forward(self, x: torch.Tensor, pos_ids: torch.Tensor):
                ang = pos_ids.float().unsqueeze(-1) * torch.linspace(0.1, 1.0, hd // 2).unsqueeze(0).unsqueeze(0)
                emb = torch.cat((ang, ang), dim=-1)
                return emb.cos(), emb.sin()

        class _ToyAttn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(d_model, n_q * hd, bias=False)
                self.k_proj = nn.Linear(d_model, n_kv * hd, bias=False)
                self.v_proj = nn.Linear(d_model, n_kv * hd, bias=False)
                self.o_proj = nn.Linear(n_q * hd, d_model, bias=False)

        class _ToyLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_layernorm = nn.LayerNorm(d_model)
                self.self_attn = _ToyAttn()

        class _ToyLlama(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = SimpleNamespace(
                    hidden_size=d_model, num_attention_heads=n_q, num_key_value_heads=n_kv, head_dim=hd, use_cache=False
                )
                self.model = SimpleNamespace(layers=nn.ModuleList([_ToyLayer(), _ToyLayer()]), rotary_emb=_ToyRotary())

            def forward(self, inputs_embeds=None, attention_mask=None, use_cache=True, output_attentions=False, output_hidden_states=False):
                bsz, seq_len, _ = inputs_embeds.shape
                pos = torch.arange(seq_len).unsqueeze(0)
                h = inputs_embeds
                hs = [h]
                pkv = []
                attns = []
                rep = n_q // n_kv
                for layer in self.model.layers:
                    hn = layer.input_layernorm(h)
                    q = layer.self_attn.q_proj(hn).view(bsz, seq_len, n_q, hd).transpose(1, 2)
                    k = layer.self_attn.k_proj(hn).view(bsz, seq_len, n_kv, hd).transpose(1, 2)
                    v = layer.self_attn.v_proj(hn).view(bsz, seq_len, n_kv, hd).transpose(1, 2)
                    cos, sin = self.model.rotary_emb(q, pos)
                    c1, s1 = cos.unsqueeze(1), sin.unsqueeze(1)
                    q = q * c1 + torch.cat((-q[..., hd // 2 :], q[..., : hd // 2]), dim=-1) * s1
                    k = k * c1 + torch.cat((-k[..., hd // 2 :], k[..., : hd // 2]), dim=-1) * s1
                    pkv.append((k, v))
                    ke = k.repeat_interleave(rep, dim=1)
                    ve = v.repeat_interleave(rep, dim=1)
                    sc = torch.matmul(q, ke.transpose(-2, -1)) / (hd ** 0.5)
                    cm = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
                    sc = sc.masked_fill(cm.unsqueeze(0).unsqueeze(0), float("-inf"))
                    aw = F.softmax(sc.float(), dim=-1)
                    attns.append(aw)
                    out_a = torch.matmul(aw, ve).transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
                    h = h + layer.self_attn.o_proj(out_a)
                    hs.append(h)
                return SimpleNamespace(
                    past_key_values=tuple(pkv),
                    hidden_states=tuple(hs) if output_hidden_states else None,
                    attentions=tuple(attns) if output_attentions else None,
                )

        toy = _ToyLlama()
        ctx_e = torch.randn(1, 20, d_model)
        q_e = torch.randn(1, 5, d_model)
        ctx_pkv, fast_scores_1d, _ = compute_memory_safe_obs_attention_and_kv(toy, ctx_e, query_embeds=q_e, per_head=False)
        ref_out = toy(inputs_embeds=torch.cat((ctx_e, q_e), dim=1), output_attentions=True)
        ref_scores = sum(a[0, :, 20:25, :20].sum(dim=(0, 1)) for a in ref_out.attentions)
        self.assertEqual(int(ctx_pkv[0][0].shape[2]), 20)
        self.assertLess(float((fast_scores_1d - ref_scores).abs().max().item()), 1e-5)

        # Verify per_head=True returns exact GQA-averaged [L=2, H_kv=2, ctx_len=20] matching naive attention!
        _, fast_scores_lh, _ = compute_memory_safe_obs_attention_and_kv(toy, ctx_e, query_embeds=q_e, per_head=True)
        self.assertEqual(tuple(fast_scores_lh.shape), (2, n_kv, 20))
        ref_lh = torch.stack([a[0, :, 20:25, :20].sum(dim=1).view(n_kv, n_q // n_kv, 20).mean(dim=1) for a in ref_out.attentions], dim=0)
        self.assertLess(float((fast_scores_lh - ref_lh).abs().max().item()), 1e-5)

    def test_true_per_head_1d_pooled_snapkv_and_exact_recent_ratio(self) -> None:
        """Audit Blocker #3: verify (a) per-layer/per-KV-head diversity, (b) 1D pooling keeps multi-token neighbors,
        and (c) recent_ratio=0.10 is strictly honored (NOT overridden to 0.25!)."""
        num_layers, num_kv_heads, seq_len, hd = 2, 2, 40, 8
        # Encode original token positions directly into K values so we can verify exact gathered positions!
        pos_vals = torch.arange(seq_len, dtype=torch.float32).view(1, 1, seq_len, 1).expand(1, num_kv_heads, seq_len, hd).clone()
        pkv = tuple((pos_vals.clone(), pos_vals.clone()) for _ in range(num_layers))

        # (a) Distinct spikes per (layer, kv_head): (0,0)->pos 8, (0,1)->pos 24, (1,0)->pos 14, (1,1)->pos 30
        scores_lh = torch.zeros((num_layers, num_kv_heads, seq_len), dtype=torch.float32)
        scores_lh[0, 0, 8] = 50.0
        scores_lh[0, 1, 24] = 50.0
        scores_lh[1, 0, 14] = 50.0
        scores_lh[1, 1, 30] = 50.0

        ev_kv, keep_lh = evict_past_key_values_h2o_attention(
            pkv, scores_lh, keep_budget=10, sink_tokens=2, recent_ratio=0.10, pool_kernel_size=5, pool_mode="avg"
        )
        self.assertEqual(tuple(keep_lh.shape), (2, 2, 10))
        # Verify per-layer / per-KV-head indices genuinely differ!
        self.assertFalse(torch.equal(keep_lh[0, 0], keep_lh[0, 1]))
        self.assertFalse(torch.equal(keep_lh[0, 0], keep_lh[1, 0]))
        self.assertFalse(torch.equal(keep_lh[1, 0], keep_lh[1, 1]))
        # Verify gathered K tensor matches per-head keep_lh exactly!
        self.assertTrue(torch.equal(ev_kv[0][0][0, 1, :, 0].long(), keep_lh[0, 1]))

        # (b) 1D pooling preserves immediate neighbors of a multi-token number spike at pos=15 (keeps 14,15,16!)
        # Background positions 25..34 have noise=0.5, spike at 15 has 10.0 (so 14 & 16 have 0.0 unpooled, 2.0 pooled!)
        sc_pool = torch.zeros((1, 1, seq_len), dtype=torch.float32)
        sc_pool[0, 0, 25:35] = 0.5
        sc_pool[0, 0, 15] = 10.0
        pkv_1 = ((pos_vals[:, :1].clone(), pos_vals[:, :1].clone()),)
        _, k_unpooled = evict_past_key_values_h2o_attention(pkv_1, sc_pool, keep_budget=10, sink_tokens=2, recent_ratio=0.10, pool_kernel_size=0)
        _, k_pooled = evict_past_key_values_h2o_attention(pkv_1, sc_pool, keep_budget=10, sink_tokens=2, recent_ratio=0.10, pool_kernel_size=5, pool_mode="avg")
        self.assertNotIn(14, k_unpooled[0, 0].tolist())
        self.assertNotIn(16, k_unpooled[0, 0].tolist())
        self.assertIn(14, k_pooled[0, 0].tolist())
        self.assertIn(15, k_pooled[0, 0].tolist())
        self.assertIn(16, k_pooled[0, 0].tolist())

        # (c) Verify recent_ratio=0.10 with budget=20, sinks=4 keeps EXACTLY 2 recent tokens (38, 39), NOT 5 (35..39)!
        sc_mid = torch.zeros((seq_len,), dtype=torch.float32)
        sc_mid[10:26] = 10.0  # 16 strong middle tokens; with sinks=4, recent=2, hh=14 -> 35,36,37 must NOT be forced!
        _, k_rec = evict_past_key_values_h2o_attention(pkv_1, sc_mid, keep_budget=20, sink_tokens=4, recent_ratio=0.10, pool_kernel_size=0)
        k_list = k_rec.tolist()
        self.assertEqual(len(k_list), 20)
        self.assertIn(38, k_list)
        self.assertIn(39, k_list)
        self.assertNotIn(35, k_list)
        self.assertNotIn(36, k_list)
        self.assertNotIn(37, k_list)

    def test_span_aligned_overwrite_erases_old_block_no_splicing_and_w2n_value_em(self) -> None:
        """Audit Blocker #2 & #4c/d: span-aligned selective overwrite erases 0:16 stale fact, never splices mid-span,
        falls back to last block on 0 overlap, plus strict w2n, value_em, and zero-training TF-IDF span-16."""
        vocab_size = 2048
        cfg = LSLCConfig.tiny_mock(k_exact=48, salience_span_len=16, salience_mode="learned_surprisal")
        tok = _MockCodeTokenizer(vocab_size=vocab_size)
        sal = SalienceGate(cfg)
        emb = nn.Embedding(vocab_size, cfg.d_llm)
        sep_id = tok.encode("\n", add_special_tokens=False)[-1]

        # Put old fact in block 0:16, unrelated filler in 16:32 and 32:48
        old_fact_ids = tok.encode("Project Atlas approved budget 39M", add_special_tokens=False)
        row = torch.full((1, 2, 48), 900, dtype=torch.int32)
        row[0, 0, : len(old_fact_ids)] = torch.tensor(old_fact_ids, dtype=torch.int32)
        row[0, 0, 16:32] = torch.arange(700, 716, dtype=torch.int32)
        row[0, 0, 32:48] = torch.arange(800, 816, dtype=torch.int32)
        bank = SparseExactBank(token_ids=row, page_ids=None, h_coords=None, w_coords=None)

        # Edit matching entity "Project Atlas" (<=15 tok) MUST overwrite block 0:16 and erase "39M"!
        edit_ids = torch.tensor(tok.encode("Project Atlas budget 82M", add_special_tokens=False), dtype=torch.long).unsqueeze(0)
        gate_open = torch.ones((1, cfg.grid_h, cfg.grid_w), dtype=torch.float32)
        bank_upd = bank.update_page_from_delta(
            page_indices=torch.tensor([0], dtype=torch.long),
            edit_token_ids=edit_ids,
            edit_embeds=emb(edit_ids),
            salience_gate=sal,
            delta_gates=gate_open,
            vocab_size=vocab_size,
            in_place=False,
            span_aligned_overwrite=True,
            salience_span_len=16,
            sep_token_id=sep_id,
            embed_tokens_fn=emb,
        )
        # Slot 0 starts with clean sep_id; old "39M" in 0:16 is completely gone; untouched blocks 16:32 and 32:48 intact!
        self.assertEqual(int(bank_upd.token_ids[0, 0, 0].item()), sep_id)
        decoded_b0 = tok.decode(bank_upd.token_ids[0, 0, :16].tolist(), skip_special_tokens=True)
        self.assertIn("82M", decoded_b0)
        self.assertNotIn("39M", decoded_b0)
        self.assertTrue(torch.equal(bank_upd.token_ids[0, 0, 16:32], row[0, 0, 16:32]))
        self.assertTrue(torch.equal(bank_upd.token_ids[0, 0, 32:48], row[0, 0, 32:48]))
        self.assertIn(0, bank_upd.edited_pages)

        # Zero-overlap edit falls back strictly to last block 32:48 (leaving 0:16 untouched!)
        disjoint_edit = torch.tensor([[1501, 1502, 1503, 1504]], dtype=torch.long)
        bank_fb = bank.update_page_from_delta(
            page_indices=torch.tensor([0], dtype=torch.long),
            edit_token_ids=disjoint_edit,
            edit_embeds=emb(disjoint_edit),
            salience_gate=sal,
            delta_gates=gate_open,
            vocab_size=vocab_size,
            in_place=False,
            span_aligned_overwrite=True,
            salience_span_len=16,
            sep_token_id=sep_id,
            embed_tokens_fn=None,
        )
        self.assertTrue(torch.equal(bank_fb.token_ids[0, 0, :32], row[0, 0, :32]))
        self.assertEqual(int(bank_fb.token_ids[0, 0, 32].item()), sep_id)
        self.assertEqual(bank_fb.token_ids[0, 0, 33:37].tolist(), [1501, 1502, 1503, 1504])

        # Verify strict w2n canonicalizer & value_em
        self.assertEqual(canonicalize_w2n_answer("fifteen"), "15")
        self.assertEqual(canonicalize_w2n_answer("ninety"), "90")
        self.assertEqual(canonicalize_w2n_answer("twenty-five million"), "25m")
        self.assertEqual(compute_value_em("45 minutes", "30 minutes"), 0.0)
        self.assertEqual(compute_value_em("thirty minutes", "30 minutes"), 1.0)

        # Verify zero-training TF-IDF span-16 selects rare needle span cleanly
        pages_h = torch.full((1, 8, 64), 10, dtype=torch.long)
        needle = torch.arange(500, 516, dtype=torch.long)
        pages_h[0, 3, 20:36] = needle
        h_bank = heuristic_span16_tfidf_bank(pages_h, k_exact=48, span_len=16)
        self.assertEqual(tuple(h_bank.token_ids.shape), (1, 8, 48))
        for tok_n in needle.tolist():
            self.assertIn(tok_n, h_bank.token_ids[0, 3].tolist())

    def test_two_bank_discrete_kv_reuse_exact_zero_diff_and_gate_parity(self) -> None:
        vocab_size = 2048
        cfg = LSLCConfig.tiny_mock(
            use_contextual_edit_encoding=True, contextual_edit_layer=1,
            salience_mode="learned_surprisal", k_exact=24, salience_span_len=16,
        )
        llm = TinyMockCausalLM(vocab_size=vocab_size, d_llm=cfg.d_llm, num_layers=cfg.num_llm_layers)
        tok = _MockCodeTokenizer(vocab_size=vocab_size)
        bge = _PlugInBiEncoderTOC("BAAI/bge-small-en-v1.5", device=torch.device("cpu"), is_mock=True)
        pathb = PathBDiscreteMemory(cfg, bge_toc=bge)
        val_pool = [2 + ((i * 7) % (vocab_size - 2)) for i in range(2048)]

        kv_res = benchmark_two_bank_discrete_kv_reuse(
            llm, pathb, tok, val_pool, torch.device("cpu"), page_counts=(4, 8), num_trials=2, is_mock=True
        )
        for p_k in ("P=4", "P=8"):
            self.assertEqual(kv_res[p_k]["bank1_embed_max_abs_diff"], 0.0)
            self.assertEqual(kv_res[p_k]["bank1_pkv_max_abs_diff"], 0.0)
            self.assertLess(kv_res[p_k]["final_logits_max_abs_diff"], 1e-4)
            # Verify first (p0), middle (p1), and last (pEnd) ALL preserve Bank-1 KV with 0.0 diff!
            for pos_k in ("first_p0", "middle_p1", "last_p_end"):
                pos_rec = kv_res[p_k]["fixed_working_set_reuse"][pos_k]
                self.assertEqual(pos_rec["bank1_pkv_max_abs_diff"], 0.0)
                self.assertNotIn(pos_rec["edited_page_idx"], pos_rec["bank1_pages"])
                self.assertEqual(pos_rec["bank2_pages"], [pos_rec["edited_page_idx"]])
        self.assertIn("gate2_multiturn_rerouting", kv_res["post_rerouting_cache_hit_telemetry"])
        self.assertIn("coding_multiturn_rerouting", kv_res["post_rerouting_cache_hit_telemetry"])

        g2 = run_llama_gate2_n200(llm, pathb, tok, val_pool, torch.device("cpu"), eval_n=8, is_mock=True)
        self.assertEqual(
            g2["systems"]["gated_bm25_rag"]["gate_telemetry"],
            g2["systems"]["standalone_discrete_span_bank"]["gate_telemetry"],
        )
        self.assertEqual(
            g2["systems"]["gated_bm25_rag"]["gate_telemetry"],
            g2["systems"]["raw_text_same_pages_k4"]["gate_telemetry"],
        )
        self.assertEqual(
            g2["systems"]["gated_bm25_rag"]["gate_telemetry"],
            g2["systems"]["standalone_append_192tok"]["gate_telemetry"],
        )
        self.assertIn("blind_edit_routing_slice", g2)
        self.assertIn("normalized_w2n_em", g2["systems"]["standalone_discrete_span_bank"])

    def test_gerbang6_full_mock_cli_and_fast_lme_n30_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpd:
            out_json = Path(tmpd) / "g6_mock.json"
            ckpt_out = Path(tmpd) / "g6_mock.safetensors"
            gerbang6_main(
                [
                    "--mock-llm",
                    "--tasks",
                    "all",
                    "--save-checkpoint",
                    str(ckpt_out),
                    "--out-json",
                    str(out_json),
                ]
            )
            data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertIn("git_sha", data)
            self.assertIsNotNone(data["checkpoint_sha256"])
            self.assertEqual(data["trainable_param_breakdown"]["hw_coord_embed_params"], 0)
            self.assertEqual(data["config"]["prompt_format"], "llama-3.1-instruct")
            self.assertFalse(data["config"]["has_layered_latent_encoder"])
            self.assertFalse(data["config"]["has_z_book"])
            self.assertFalse(data["config"]["has_hw_coord_embed"])
            self.assertEqual(data["train_pathb"]["max_frozen_param_diff"], 0.0)
            self.assertTrue(ckpt_out.is_file())
            self.assertTrue(data["llama_coding_n24"]["single_run_all_baselines"])
            self.assertTrue(data["llama_coding_n24"]["identical_pages_verified"])
            self.assertIn("raw_text_same_pages_k4", data["llama_coding_n24"]["systems"])
            self.assertIn("generated_patches", data["llama_coding_n24"]["systems"]["standalone_discrete_k4"]["episodes"][0])
            self.assertIn("query_aware_snapkv", data["llama_ood_kv_4k"]["systems"])
            self.assertTrue(data["llama_ood_kv_4k"]["snapkv_config"]["per_layer_per_kv_head_gqa"])
            self.assertIn("encode_or_evict_ms", data["llama_ood_kv_4k"]["systems"]["query_aware_snapkv"])
            self.assertIn("heuristic_span16_tfidf", data["llama_ruler_16k_32k_and_profile"]["ruler_by_length"]["1K"]["systems"])
            self.assertEqual(
                data["llama_ruler_16k_32k_and_profile"]["prefix_kv_cache_reuse"]["P=4"]["bank1_pkv_max_abs_diff"],
                0.0,
            )
            self.assertIsNotNone(data["llama_zbook_probe"])
            self.assertEqual(data["llama_zbook_probe"]["max_frozen_param_diff"], 0.0)
            self.assertTrue(data["llama_zbook_probe"]["edit_ablation_5cell"]["iso_token_verified"])
            self.assertIn("llama_lslc_zbook_full", data["llama_zbook_probe"]["wikitext_ppl_comparison_4k"])

        # Verify fast N=30 LongMemEval 5-arm Limitations mode (incl. matched-depth d4 + value_em)
        vocab_size = 2048
        cfg_q = LSLCConfig.tiny_mock(
            use_contextual_edit_encoding=True, contextual_edit_layer=1,
            salience_mode="learned_surprisal", delta_pooling_mode="cross_attn",
            router_mode="content_attn", k_exact=48, salience_span_len=16,
        )
        llm_q = TinyMockCausalLM(vocab_size=vocab_size, d_llm=cfg_q.d_llm, num_layers=cfg_q.num_llm_layers)
        wrap_q = FrozenQwenLSLCWrapper(cfg_q, llm_q)
        tok_q = _MockCodeTokenizer(vocab_size=vocab_size)
        bge_q = _PlugInBiEncoderTOC("BAAI/bge-small-en-v1.5", device=torch.device("cpu"), is_mock=True)
        lme30 = run_longmemeval_discrete_probe(
            wrap_q, tok_q, torch.device("cpu"), bge_toc=bge_q, n_total=30, is_mock=True
        )
        self.assertEqual(
            set(lme30["systems"].keys()),
            {"bm25_rag_k2", "bge_4span_rag_k2", "standalone_discrete_k48_d4", "standalone_discrete_k48_d6", "lslc_full_k48_d4"},
        )
        self.assertIn("value_em", lme30["systems"]["standalone_discrete_k48_d4"])
        self.assertEqual(lme30["packing_mode"], "gold_prioritized_64p")


if __name__ == "__main__":
    unittest.main()
