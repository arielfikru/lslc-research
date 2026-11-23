"""Milestone 1 unit & verification tests enforcing all shapes, byte accounting, invariance, and grad gates."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lslc import (
    ChunkedTextPageDataset,
    FactorizedRoPE3D,
    FrozenQwenLSLCWrapper,
    LSLCConfig,
    LSLCSystem,
    LivingBookStateOverrideDataset,
    PageRoPE1D,
    RouterOutput,
    SyntheticExactSemanticNIAHDataset,
    TinyMockCausalLM,
    compute_analytical_memory_stats,
    greedy_generate_from_prefix,
    h2o_evict_keep_indices,
    streamingllm_keep_indices,
)


class TestLSLCMilestone1Gates(unittest.TestCase):
    """Enforces every Milestone 1 Pass/Fail Gate locally without requiring GPUs or HF downloads."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.cfg = LSLCConfig.tiny_mock()
        self.llm = TinyMockCausalLM(
            vocab_size=512,
            d_llm=self.cfg.d_llm,
            num_layers=self.cfg.num_llm_layers,
            num_heads=4,
            num_kv_heads=self.cfg.num_kv_heads,
        )
        self.wrapper = FrozenQwenLSLCWrapper(self.cfg, self.llm)

    def test_exact_shapes_and_active_token_budgets(self) -> None:
        embed_fn = self.llm.get_input_embeddings()

        # 4,096 tokens -> P=16 pages -> k_deep=2, k_skim=2 => 100 active tokens
        pages_16 = torch.randint(0, 512, (1, 16, self.cfg.page_size))
        q_ids = torch.randint(0, 512, (1, 8))
        z_16, bank_16, toc_16 = self.wrapper.lslc.encode_book(pages_16, embed_fn)
        self.assertEqual(tuple(z_16.shape), (1, 16, 3, 128, 4, 4))
        self.assertEqual(tuple(toc_16.shape), (1, 16, 1, self.cfg.d_llm))
        skim_16 = self.wrapper.lslc.projector.project_skim(z_16)
        deep_16 = self.wrapper.lslc.projector.project_deep(z_16)
        self.assertEqual(tuple(skim_16.shape), (1, 16, 4, self.cfg.d_llm))
        self.assertEqual(tuple(deep_16["l0"].shape), (1, 16, 4, self.cfg.d_llm))
        self.assertEqual(tuple(deep_16["l1"].shape), (1, 16, 4, self.cfg.d_llm))
        self.assertEqual(tuple(deep_16["l2"].shape), (1, 16, 16, self.cfg.d_llm))

        q_emb = embed_fn(q_ids).mean(dim=1)
        _, asm_16 = self.wrapper.lslc.route_and_assemble(
            z_16, toc_16, q_emb, bank_16, embed_fn, k_deep=2, k_skim=2, mode="full"
        )
        self.assertEqual(asm_16.prefix_embeds.shape[1], 100)

        # 32,768 tokens -> P=128 pages -> k_deep=2, k_skim=4 => 218 active tokens
        pages_128 = torch.randint(0, 512, (1, 128, self.cfg.page_size))
        z_128, bank_128, toc_128 = self.wrapper.lslc.encode_book(pages_128, embed_fn)
        _, asm_128 = self.wrapper.lslc.route_and_assemble(
            z_128, toc_128, q_emb, bank_128, embed_fn, k_deep=2, k_skim=4, mode="full"
        )
        self.assertEqual(asm_128.prefix_embeds.shape[1], 218)

    def test_exact_storage_bytes_book_and_sparse_bank(self) -> None:
        embed_fn = self.llm.get_input_embeddings()
        pages_16 = torch.randint(0, 512, (1, 16, self.cfg.page_size))
        z_16, bank_16, _ = self.wrapper.lslc.encode_book(pages_16, embed_fn)

        # Book FP16: 12 KiB/page = 12,288 B/page -> 196,608 B for 16 pages
        z_fp16 = z_16.to(torch.float16)
        self.assertEqual(z_fp16.numel() * z_fp16.element_size(), 16 * 12_288)
        # SparseExactBank: strictly 8 B/token -> 128 B/page -> 2,048 B for 16 pages
        self.assertEqual(bank_16.storage_bytes(), 16 * 128)

        pages_128 = torch.randint(0, 512, (1, 128, self.cfg.page_size))
        _, bank_128, _ = self.wrapper.lslc.encode_book(pages_128, embed_fn)
        self.assertEqual(bank_128.storage_bytes(), 128 * 128)

    def test_factorized_rope3d_and_page_rope1d_norm_preservation(self) -> None:
        rope3d = FactorizedRoPE3D(head_dim=64, num_layers=3, grid_h=4, grid_w=4)
        x = torch.randn(2, 4, 48, 64)
        y = rope3d(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5))

        rope1d = PageRoPE1D(dim=64)
        p_vec = torch.randn(2, 16, 64)
        p_rot = rope1d(p_vec, torch.arange(16))
        self.assertTrue(torch.allclose(p_rot.norm(dim=-1), p_vec.norm(dim=-1), atol=1e-5))

    def test_prefix_invariance_under_delta_update(self) -> None:
        """Mutating dynamic layer l=2 MUST keep TOC, Skim, Deep l=0, and Deep l=1 strictly bit-identical."""
        embed_fn = self.llm.get_input_embeddings()
        pages = torch.randint(0, 512, (1, 4, self.cfg.page_size))
        q_ids = torch.randint(0, 512, (1, 8))
        edit_ids = torch.randint(0, 512, (1, 12))
        edit_page = torch.tensor([1], dtype=torch.long)

        # Force non-zero delta_head weights so DeltaUpdater meaningfully alters Z[:, :, 2]
        with torch.no_grad():
            nn.init.normal_(self.wrapper.lslc.delta_updater.delta_head.weight, std=0.05)
            nn.init.normal_(self.wrapper.lslc.delta_updater.delta_head.bias, std=0.05)

        z_old, bank, toc_old = self.wrapper.lslc.encode_book(pages, embed_fn)
        q_emb = embed_fn(q_ids).mean(dim=1)
        _, asm_old = self.wrapper.lslc.route_and_assemble(
            z_old, toc_old, q_emb, bank, embed_fn, k_deep=2, k_skim=1, mode="full", training_ste=False
        )

        z_new, gates = self.wrapper.lslc.apply_page_delta(z_old, edit_page, edit_ids, embed_fn)
        toc_new = self.wrapper.lslc.projector.project_toc(z_new)
        _, asm_new = self.wrapper.lslc.route_and_assemble(
            z_new, toc_new, q_emb, bank, embed_fn, k_deep=2, k_skim=1, mode="full", training_ste=False
        )

        # Verify Static (l=0) and Entity (l=1) unchanged while Dynamic (l=2) mutated
        self.assertEqual((z_new[:, :, 0] - z_old[:, :, 0]).abs().max().item(), 0.0)
        self.assertEqual((z_new[:, :, 1] - z_old[:, :, 1]).abs().max().item(), 0.0)
        self.assertGreater((z_new[:, 1, 2] - z_old[:, 1, 2]).abs().max().item(), 1e-4)
        self.assertTrue(0.05 <= gates.mean().item() <= 0.45)

        # Verify immutable causal prefix [TOC | Skim | Deep l=0 | Deep l=1] has max_abs_diff == 0.0
        imm = asm_old.immutable_prefix_len
        self.assertEqual(imm, asm_new.immutable_prefix_len)
        max_prefix_diff = (asm_new.prefix_embeds[:, :imm] - asm_old.prefix_embeds[:, :imm]).abs().max().item()
        self.assertEqual(max_prefix_diff, 0.0)

    def test_full_backward_and_frozen_llm_gate(self) -> None:
        """All 5 LSLC modules + hw_coord_embed must receive finite non-zero grads while LLM stays frozen."""
        pages = torch.randint(0, 512, (1, 4, self.cfg.page_size))
        q_ids = torch.randint(0, 512, (1, 8))
        a_ids = torch.randint(0, 512, (1, 6))
        t_page = torch.tensor([1], dtype=torch.long)
        e_page = torch.tensor([1], dtype=torch.long)
        e_ids = torch.randint(0, 512, (1, 10))

        out = self.wrapper.forward_step(
            page_token_ids=pages,
            query_ids=q_ids,
            answer_ids=a_ids,
            target_page_idx=t_page,
            edit_page_idx=e_page,
            edit_token_ids=e_ids,
            use_exact_bank=True,
            matryoshka_mode="skim_only",
            run_reconstruction=True,
            run_matryoshka=True,
        )
        self.assertTrue(torch.isfinite(out.loss_total).item())
        out.loss_total.backward()

        # Frozen LLM check
        for name, param in self.wrapper.llm.named_parameters():
            self.assertFalse(param.requires_grad, f"LLM param {name} should be frozen")
            self.assertIsNone(param.grad, f"LLM param {name} should not accumulate grads")

        # All 5 trainable modules + delta gate_head & delta_head + hw_coord_embed must receive finite non-zero grads
        submodules = {
            "encoder": self.wrapper.lslc.encoder,
            "salience_gate": self.wrapper.lslc.salience_gate,
            "projector": self.wrapper.lslc.projector,
            "router": self.wrapper.lslc.router,
            "delta_updater": self.wrapper.lslc.delta_updater,
            "delta_gate_head": self.wrapper.lslc.delta_updater.gate_head,
            "delta_head": self.wrapper.lslc.delta_updater.delta_head,
            "hw_coord_embed": self.wrapper.lslc.hw_coord_embed,
        }
        self.assertTrue(torch.isfinite(out.loss_salience).item())
        self.assertIsNotNone(self.wrapper.lslc.ae_cue_embed.grad)
        self.assertGreater(float(self.wrapper.lslc.ae_cue_embed.grad.norm().item()), 0.0)
        for mod_name, mod in submodules.items():
            grads = [p.grad for p in mod.parameters() if p.grad is not None]
            self.assertTrue(len(grads) > 0, f"No gradients found in {mod_name}")
            total_norm = sum(float(g.norm().item()) for g in grads)
            self.assertTrue(total_norm > 0.0 and torch.isfinite(torch.tensor(total_norm)), f"Invalid grad norm in {mod_name}: {total_norm}")

    def test_greedy_generate_from_prefix(self) -> None:
        embed_fn = self.llm.get_input_embeddings()
        pages = torch.randint(0, 512, (1, 4, self.cfg.page_size))
        q_ids = torch.randint(0, 512, (1, 8))
        z_book, bank, toc = self.wrapper.lslc.encode_book(pages, embed_fn, vocab_size=512)
        _, asm = self.wrapper.lslc.route_and_assemble(
            z_book, toc, embed_fn(q_ids).mean(dim=1), bank, embed_fn, training_ste=False
        )
        gen = greedy_generate_from_prefix(self.llm, asm.prefix_embeds, q_ids, max_new_tokens=6)
        self.assertEqual(tuple(gen.shape), (1, 6))

    def test_mixed_bfloat16_llm_with_fp32_lslc(self) -> None:
        """Verify fp32 LSLCSystem + bfloat16 frozen LLM runs encode, greedy_generate, and backward cleanly."""
        bf16_llm = TinyMockCausalLM(
            vocab_size=512,
            d_llm=self.cfg.d_llm,
            num_layers=self.cfg.num_llm_layers,
            num_heads=4,
            num_kv_heads=self.cfg.num_kv_heads,
        ).to(torch.bfloat16)
        bf16_wrapper = FrozenQwenLSLCWrapper(self.cfg, bf16_llm)
        embed_fn = bf16_llm.get_input_embeddings()

        pages = torch.randint(0, 512, (1, 4, self.cfg.page_size))
        q_ids = torch.randint(0, 512, (1, 8))
        a_ids = torch.randint(0, 512, (1, 6))

        z_book, bank, toc = bf16_wrapper.lslc.encode_book(pages, embed_fn, vocab_size=512)
        _, asm = bf16_wrapper.lslc.route_and_assemble(
            z_book, toc, embed_fn(q_ids).mean(dim=1), bank, embed_fn, training_ste=False
        )
        # Prefix may be float32 even though bf16_llm is bfloat16; greedy_generate_from_prefix must handle it safely
        gen = greedy_generate_from_prefix(bf16_llm, asm.prefix_embeds, q_ids, max_new_tokens=6)
        self.assertEqual(tuple(gen.shape), (1, 6))

        out = bf16_wrapper.forward_step(
            page_token_ids=pages,
            query_ids=q_ids,
            answer_ids=a_ids,
            target_page_idx=torch.tensor([1]),
            use_exact_bank=True,
            distill_salience=True,
        )
        self.assertTrue(torch.isfinite(out.loss_total).item())
        out.loss_total.backward()

    def test_analytical_vram_table_parity(self) -> None:
        c17 = LSLCConfig.qwen3_1_7b()
        c4 = LSLCConfig.qwen3_4b()

        s17_4k = compute_analytical_memory_stats(c17, 4096, k_deep=2, k_skim=2)
        s17_32k = compute_analytical_memory_stats(c17, 32768, k_deep=2, k_skim=4)
        s4_4k = compute_analytical_memory_stats(c4, 4096, k_deep=2, k_skim=2)
        s4_32k = compute_analytical_memory_stats(c4, 32768, k_deep=2, k_skim=4)

        self.assertEqual(s17_4k["kv_kib_per_token"], 112.0)
        self.assertEqual(s4_4k["kv_kib_per_token"], 144.0)
        self.assertAlmostEqual(s17_4k["lslc_net_mib"], 11.13, places=2)
        self.assertAlmostEqual(s17_32k["lslc_net_mib"], 25.36, places=2)
        self.assertAlmostEqual(s4_4k["lslc_net_mib"], 14.25, places=2)
        self.assertAlmostEqual(s4_32k["lslc_net_mib"], 32.17, places=2)

    def test_datasets_and_baseline_helpers(self) -> None:
        niah_a = SyntheticExactSemanticNIAHDataset(num_episodes=6, num_pages=4, page_size=256, seed=7)
        niah_b = SyntheticExactSemanticNIAHDataset(num_episodes=6, num_pages=4, page_size=256, seed=7)
        self.assertTrue(torch.equal(niah_a[0].context_pages, niah_b[0].context_pages))

        delta_ds = LivingBookStateOverrideDataset(num_episodes=4, num_pages=4, page_size=256, num_edits=3)
        self.assertEqual(len(delta_ds[0].edit_pages), 3)

        chunk_ds = ChunkedTextPageDataset(list(range(1200)), num_pages=4, page_size=256, turn_len=64)
        self.assertGreaterEqual(len(chunk_ds), 1)
        self.assertEqual(tuple(chunk_ds[0].context_pages.shape), (4, 256))

        s_idx = streamingllm_keep_indices(1024, sink_tokens=4, window_tokens=128)
        self.assertEqual(len(s_idx), 132)
        h_idx = h2o_evict_keep_indices(torch.rand(100), budget_ratio=0.20, recent_ratio=0.10)
        self.assertEqual(len(h_idx), 20)

    def test_salience_bpe_digit_record_capture_and_non_dead_delta_gate(self) -> None:
        """Regression test for M2-v2 fixes: BPE digit record capture >= 85% and non-collapsing DeltaUpdater gate."""
        embed_fn = self.llm.get_input_embeddings()
        # Simulate a 256-token page where background prose has BPE word IDs in [256, 511]
        # and a 12-token exact record span '[RECORD] API-KEY = 3034-3582' has byte/digit IDs in [15, 60].
        page_ids = torch.randint(256, 512, (1, 1, self.cfg.page_size))
        needle_ids = torch.tensor([58, 320, 60, 340, 12, 350, 28, 18, 15, 18, 19, 12, 18, 20, 23, 17], dtype=torch.long)
        page_ids[0, 0, 80 : 80 + len(needle_ids)] = needle_ids
        pages_embeds = embed_fn(page_ids)

        bank = self.wrapper.lslc.encode_book(page_ids, embed_fn, use_exact_bank=True, vocab_size=151936)[1]
        captured = set(bank.token_ids[0, 0].tolist())
        digit_ids = [18, 15, 18, 19, 12, 18, 20, 23, 17]
        hits = sum(1 for tid in digit_ids if tid in captured)
        self.assertGreaterEqual(hits / len(digit_ids), 0.85, f"Expected >=85% digit capture, got {hits}/{len(digit_ids)}")

        # Verify multi-step DeltaUpdater gate stays inside [0.08, 0.35] across 10 gradient steps
        opt = torch.optim.AdamW(self.wrapper.lslc.delta_updater.parameters(), lr=1e-3)
        q_ids = torch.randint(0, 512, (1, 6))
        a_ids = torch.randint(0, 512, (1, 4))
        edit_steps = [
            (torch.tensor([0]), torch.randint(0, 512, (1, 8))),
            (torch.tensor([0]), torch.randint(0, 512, (1, 8))),
        ]
        last_gate = 0.0
        for _ in range(10):
            opt.zero_grad()
            out = self.wrapper.forward_step(
                page_token_ids=page_ids,
                query_ids=q_ids,
                answer_ids=a_ids,
                target_page_idx=torch.tensor([0]),
                edit_steps=edit_steps,
                use_exact_bank=False,
                run_matryoshka=False,
                distill_salience=False,
            )
            out.loss_total.backward()
            opt.step()
            last_gate = out.mean_gate
        self.assertTrue(0.08 <= last_gate <= 0.35, f"DeltaUpdater gate collapsed or saturated: {last_gate}")

    def test_hierarchical_toc_shrinks_active_tokens_and_preserves_invariance(self) -> None:
        """2-level Chapter->Page TOC must cut P=128 active tokens below 160 and preserve max_abs_diff == 0.0."""
        embed_fn = self.llm.get_input_embeddings()
        pages_128 = torch.randint(0, 512, (1, 128, self.cfg.page_size))
        q_ids = torch.randint(0, 512, (1, 8))
        edit_ids = torch.randint(0, 512, (1, 10))
        q_emb = embed_fn(q_ids).mean(dim=1)

        with torch.no_grad():
            nn.init.normal_(self.wrapper.lslc.delta_updater.delta_head.weight, std=0.05)
            nn.init.normal_(self.wrapper.lslc.delta_updater.delta_head.bias, std=0.05)

        z_old, bank, toc_old = self.wrapper.lslc.encode_book(pages_128, embed_fn, use_exact_bank=True, vocab_size=512)
        _, asm_flat = self.wrapper.lslc.route_and_assemble(
            z_old, toc_old, q_emb, bank, embed_fn, k_deep=2, k_skim=4, mode="full", training_ste=False
        )
        route_hier, asm_hier_before = self.wrapper.lslc.route_and_assemble(
            z_old,
            toc_old,
            q_emb,
            bank,
            embed_fn,
            k_deep=2,
            k_skim=4,
            mode="full",
            training_ste=False,
            hierarchical_chapter_size=16,
        )
        self.assertEqual(asm_flat.prefix_embeds.shape[1], 218)
        self.assertLess(asm_hier_before.prefix_embeds.shape[1], 160)
        self.assertIn("chapter_toc", asm_hier_before.slice_bounds)

        target_p = torch.tensor([int(route_hier.deep_idx[0, -1].item())], dtype=torch.long)
        z_new, _gates = self.wrapper.lslc.apply_page_delta(z_old, target_p, edit_ids, embed_fn)
        toc_new = self.wrapper.lslc.projector.project_toc(z_new)
        _, asm_hier_after = self.wrapper.lslc.route_and_assemble(
            z_new,
            toc_new,
            q_emb,
            bank,
            embed_fn,
            k_deep=2,
            k_skim=4,
            mode="full",
            training_ste=False,
            router_out_override=route_hier,
            hierarchical_chapter_size=16,
        )
        imm = asm_hier_before.immutable_prefix_len
        self.assertEqual(imm, asm_hier_after.immutable_prefix_len)
        max_diff = (asm_hier_after.prefix_embeds[:, :imm] - asm_hier_before.prefix_embeds[:, :imm]).abs().max().item()
        self.assertEqual(max_diff, 0.0)

    def test_router_out_override_integration(self) -> None:
        """External retriever RouterOutput override must select the exact specified deep/skim/toc pages."""
        embed_fn = self.llm.get_input_embeddings()
        pages = torch.randint(0, 512, (1, 8, self.cfg.page_size))
        q_ids = torch.randint(0, 512, (1, 8))
        q_emb = embed_fn(q_ids).mean(dim=1)
        z_book, bank, toc = self.wrapper.lslc.encode_book(pages, embed_fn, use_exact_bank=True, vocab_size=512)

        ext_route = RouterOutput(
            logits=torch.zeros((1, 8)),
            probs=torch.full((1, 8), 0.125),
            deep_idx=torch.tensor([[2, 5]], dtype=torch.long),
            skim_idx=torch.tensor([[0, 1]], dtype=torch.long),
            toc_idx=torch.tensor([[3, 4, 6, 7]], dtype=torch.long),
        )
        route_used, asm = self.wrapper.lslc.route_and_assemble(
            z_book, toc, q_emb, bank, embed_fn, mode="full", training_ste=False, router_out_override=ext_route
        )
        self.assertTrue(torch.equal(route_used.deep_idx, ext_route.deep_idx))
        self.assertEqual(asm.prefix_embeds.shape[1], 4 + 8 + 48 + 32)


if __name__ == "__main__":
    unittest.main()
