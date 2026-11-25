"""Unit tests for LSLC v3.0.0 Phase 1 core: zero-copy edits, contextual edit encoding, Cond-B distillation, and adversarial dataset."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import unittest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lslc import (
    AdversarialLivingBookDataset,
    FrozenQwenLSLCWrapper,
    LSLCConfig,
    LSLCSystem,
    TinyMockCausalLM,
    extract_contextual_edit_embeds,
    inject_lora,
    lora_parameters,
    merge_lora,
    set_lora_enabled,
    unmerge_lora,
)


class TestLSLCV3Core(unittest.TestCase):
    """Verifies LSLC v3.0.0 upgrades locally (0 cloud cost) and backwards compatibility with v2 weights."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.cfg = LSLCConfig.tiny_mock(
            use_contextual_edit_encoding=True,
            contextual_edit_layer=1,
            lambda_cond_b_distill=0.5,
        )
        self.llm = TinyMockCausalLM(
            vocab_size=512,
            d_llm=self.cfg.d_llm,
            num_layers=self.cfg.num_llm_layers,
            num_heads=4,
            num_kv_heads=self.cfg.num_kv_heads,
        )
        self.wrapper = FrozenQwenLSLCWrapper(self.cfg, self.llm)

    def test_zero_copy_in_place_delta_update_parity_and_invariance(self) -> None:
        embed_fn = self.llm.get_input_embeddings()
        pages = torch.randint(0, 512, (1, 6, self.cfg.page_size))
        edit_ids = torch.randint(0, 512, (1, 12))
        edit_page = torch.tensor([2], dtype=torch.long)

        with torch.no_grad():
            nn.init.normal_(self.wrapper.lslc.delta_updater.delta_head.weight, std=0.05)
            nn.init.normal_(self.wrapper.lslc.delta_updater.delta_head.bias, std=0.05)

        z_base, bank_base, _ = self.wrapper.lslc.encode_book(pages, embed_fn, use_exact_bank=True, vocab_size=512)

        # Autograd-enabled in_place=True must raise RuntimeError
        with torch.enable_grad():
            with self.assertRaises(RuntimeError):
                self.wrapper.lslc.apply_page_delta(
                    z_base.clone(), edit_page, edit_ids, embed_fn, exact_bank=bank_base, vocab_size=512, in_place=True
                )

        # Under torch.no_grad(), zero-copy in_place=True must mutate storage in place and match cloned output
        with torch.no_grad():
            z_clone, g_clone, bank_clone = self.wrapper.lslc.apply_page_delta(
                z_base.clone(), edit_page, edit_ids, embed_fn, exact_bank=bank_base, vocab_size=512, in_place=False
            )
            z_ip = z_base.clone()
            orig_ptr = z_ip.data_ptr()
            # Keep a fresh bank copy before mutating in place
            from lslc.encoder import SparseExactBank

            bank_ip = SparseExactBank(
                token_ids=bank_base.token_ids.clone(),
                page_ids=bank_base.page_ids.clone(),
                h_coords=bank_base.h_coords.clone(),
                w_coords=bank_base.w_coords.clone(),
                _ste_gates=None if bank_base._ste_gates is None else bank_base._ste_gates.clone(),
            )
            z_ip_out, g_ip, bank_ip_out = self.wrapper.lslc.apply_page_delta(
                z_ip, edit_page, edit_ids, embed_fn, exact_bank=bank_ip, vocab_size=512, in_place=True
            )
            self.assertEqual(z_ip_out.data_ptr(), orig_ptr)
            self.assertIs(bank_ip_out, bank_ip)
            self.assertEqual((z_ip_out - z_clone).abs().max().item(), 0.0)
            self.assertEqual((g_ip - g_clone).abs().max().item(), 0.0)
            self.assertTrue(torch.equal(bank_ip_out.token_ids, bank_clone.token_ids))
            # Untouched pages and structural layers l=0,1 must remain bit-identical
            self.assertEqual((z_ip_out[:, :, :2] - z_base[:, :, :2]).abs().max().item(), 0.0)
            self.assertEqual((z_ip_out[:, :2] - z_base[:, :2]).abs().max().item(), 0.0)

    def test_contextual_hidden_state_edit_encoding_and_backward(self) -> None:
        embed_fn = self.llm.get_input_embeddings()
        e_ids = torch.randint(0, 512, (1, 10))
        raw = embed_fn(e_ids)
        ctx = extract_contextual_edit_embeds(self.llm, e_ids, layer_idx=1)
        self.assertEqual(ctx.shape, raw.shape)
        self.assertFalse(ctx.requires_grad)
        raw_rms = float(raw.float().pow(2).mean(dim=-1).sqrt().mean().item())
        ctx_rms = float(ctx.float().pow(2).mean(dim=-1).sqrt().mean().item())
        self.assertAlmostEqual(ctx_rms, raw_rms, places=3)

        pages = torch.randint(0, 512, (1, 4, self.cfg.page_size))
        q_ids = torch.randint(0, 512, (1, 8))
        a_ids = torch.randint(0, 512, (1, 4))
        out = self.wrapper.forward_step(
            page_token_ids=pages,
            query_ids=q_ids,
            answer_ids=a_ids,
            target_page_idx=torch.tensor([1]),
            edit_page_idx=torch.tensor([1]),
            edit_token_ids=e_ids,
            use_exact_bank=True,
            run_matryoshka=False,
            distill_salience=False,
        )
        out.loss_total.backward()
        for name, p in self.wrapper.llm.named_parameters():
            self.assertIsNone(p.grad, f"Frozen LLM param {name} leaked gradients")
        delta_grad = sum(float(p.grad.norm().item()) for p in self.wrapper.lslc.delta_updater.parameters() if p.grad is not None)
        self.assertGreater(delta_grad, 0.0)

    def test_rejected_proposal_noop_gate_suppression(self) -> None:
        pages = torch.randint(0, 512, (1, 4, self.cfg.page_size))
        q_ids = torch.randint(0, 512, (1, 8))
        a_ids = torch.randint(0, 512, (1, 4))
        e_acc = torch.randint(0, 512, (1, 10))
        e_rej = torch.randint(0, 512, (1, 10))
        p_idx = torch.tensor([1])

        opt = torch.optim.AdamW(self.wrapper.lslc.delta_updater.parameters(), lr=5e-3)
        first_gate = None
        last_gate = None
        for _ in range(8):
            opt.zero_grad()
            out = self.wrapper.forward_step(
                page_token_ids=pages,
                query_ids=q_ids,
                answer_ids=a_ids,
                target_page_idx=p_idx,
                edit_steps=[(p_idx, e_rej)],
                edit_accepted_flags=[False],
                use_exact_bank=True,
                run_reconstruction=False,
                run_matryoshka=False,
                distill_salience=False,
            )
            if first_gate is None:
                first_gate = out.mean_gate
            last_gate = out.mean_gate
            out.loss_total.backward()
            opt.step()
        self.assertLess(last_gate, first_gate)

        # Mixed accepted + rejected sequence validates shape/flag execution cleanly
        out_mix = self.wrapper.forward_step(
            page_token_ids=pages,
            query_ids=q_ids,
            answer_ids=a_ids,
            target_page_idx=p_idx,
            edit_steps=[(p_idx, e_acc), (p_idx, e_rej)],
            edit_accepted_flags=[True, False],
            use_exact_bank=True,
            run_matryoshka=False,
            distill_salience=False,
        )
        self.assertTrue(torch.isfinite(out_mix.loss_sparse).item())

    def test_condition_b_distillation_loss_and_grads(self) -> None:
        pages = torch.randint(0, 512, (1, 4, self.cfg.page_size))
        verified = pages.clone()
        verified[0, 1, :16] = torch.randint(0, 512, (16,))
        q_ids = torch.randint(0, 512, (1, 8))
        a_ids = torch.randint(0, 512, (1, 4))
        e_ids = torch.randint(0, 512, (1, 10))

        out = self.wrapper.forward_step(
            page_token_ids=pages,
            query_ids=q_ids,
            answer_ids=a_ids,
            target_page_idx=torch.tensor([1]),
            edit_page_idx=torch.tensor([1]),
            edit_token_ids=e_ids,
            edit_accepted_flags=[True],
            verified_final_pages=verified,
            lambda_cond_b=0.5,
            use_exact_bank=True,
            run_matryoshka=False,
            distill_salience=False,
        )
        self.assertIsNotNone(out.loss_cond_b)
        self.assertTrue(torch.isfinite(out.loss_cond_b).item())
        self.assertGreater(float(out.loss_cond_b.item()), 0.0)
        out.loss_total.backward()
        for name, p in self.wrapper.llm.named_parameters():
            self.assertIsNone(p.grad, f"Frozen LLM param {name} leaked gradients during Cond-B distillation")

    def test_adversarial_living_book_dataset_no_shortcuts_and_disjoint_splits(self) -> None:
        # Verify strictly zero paraphrase template overlap between train and eval
        self.assertTrue(
            set(AdversarialLivingBookDataset.TRAIN_ACCEPT_BUDGET_TEMPLATES).isdisjoint(
                AdversarialLivingBookDataset.EVAL_ACCEPT_BUDGET_TEMPLATES
            )
        )
        self.assertTrue(
            set(AdversarialLivingBookDataset.TRAIN_REJECT_BUDGET_TEMPLATES).isdisjoint(
                AdversarialLivingBookDataset.EVAL_REJECT_BUDGET_TEMPLATES
            )
        )
        self.assertTrue(
            set(AdversarialLivingBookDataset.TRAIN_ACCEPT_SYMBOLIC_TEMPLATES).isdisjoint(
                AdversarialLivingBookDataset.EVAL_ACCEPT_SYMBOLIC_TEMPLATES
            )
        )
        self.assertTrue(
            set(AdversarialLivingBookDataset.TRAIN_REJECT_SYMBOLIC_TEMPLATES).isdisjoint(
                AdversarialLivingBookDataset.EVAL_REJECT_SYMBOLIC_TEMPLATES
            )
        )

        for mode in ("budget", "symbolic"):
            ds_eval = AdversarialLivingBookDataset(num_episodes=16, num_pages=6, num_edits=4, mode=mode, split="eval", seed=77)
            static_pages = {ep.metadata["static_page"] for ep in ds_eval}
            proj_a_pages = {ep.metadata["proj_a_page"] for ep in ds_eval}
            proj_b_pages = {ep.metadata["proj_b_page"] for ep in ds_eval}
            static_answers = {ep.metadata["answer_str"] for ep in ds_eval if ep.metadata["query_type"] == "randomized_static_retention"}

            # Randomized page positions and randomized static answers (no page-0 / dual-signoff shortcut)
            self.assertGreater(len(static_pages), 1)
            self.assertGreater(len(proj_a_pages), 1)
            self.assertGreater(len(proj_b_pages), 1)
            self.assertGreater(len(static_answers), 1)

            for ep in ds_eval:
                m = ep.metadata
                self.assertEqual(len({m["static_page"], m["proj_a_page"], m["proj_b_page"]}), 3)
                self.assertEqual(len(m["edit_accepted_flags"]), len(ep.edit_pages))
                self.assertEqual(tuple(m["verified_final_pages"].shape), tuple(ep.context_pages.shape))
                if m["query_type"] == "rejected_proposal":
                    self.assertFalse(m["edit_accepted_flags"][-1])
                    self.assertIsNotNone(m["decoy_str"])
                    self.assertNotEqual(m["decoy_str"], m["answer_str"])
                elif m["query_type"] == "neighbor_preservation":
                    self.assertEqual(ep.target_page_idx, m["proj_b_page"])
                elif m["query_type"] == "randomized_static_retention":
                    self.assertEqual(ep.target_page_idx, m["static_page"])

    def test_v2_checkpoint_backwards_compatibility(self) -> None:
        ckpt_path = REPO_ROOT / "hf_hub_release" / "lslc_qwen3_1.7b_adapter.safetensors"
        if not ckpt_path.is_file():
            self.skipTest("v2 checkpoint file not present locally")
        with open(ckpt_path, "rb") as f:
            hdr_len = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hdr_len))
        hdr.pop("__metadata__", None)
        sys_v2 = LSLCSystem(LSLCConfig.qwen3_1_7b())
        sd = sys_v2.state_dict()
        self.assertEqual(set(sd.keys()) - set(hdr.keys()), set())
        self.assertEqual(set(hdr.keys()) - set(sd.keys()), set())
        for k, tensor in sd.items():
            self.assertEqual(list(tensor.shape), hdr[k]["shape"])

    def test_rejected_proposal_preserves_old_exact_bank_and_blocks_rejected_tokens(self) -> None:
        torch.manual_seed(7)
        pages = torch.randint(10, 400, (1, 4, self.cfg.page_size))
        pages[0, 1, 0:4] = torch.tensor([411, 412, 413, 430])  # initial timeout = 30 (430)
        q_ids = torch.randint(10, 400, (1, 6))
        a_ids = torch.tensor([[445]])
        p_idx = torch.tensor([1])
        e_acc = torch.tensor([[411, 412, 445, 445]])  # accepted timeout = 45 (445)
        e_rej = torch.tensor([[415, 416, 490, 490]])  # rejected proposal timeout = 90 (490)

        opt = torch.optim.AdamW(self.wrapper.lslc.delta_updater.parameters(), lr=5e-3)
        self.wrapper.lslc.train()
        for step in range(40):
            opt.zero_grad()
            is_acc = (step % 2 == 0)
            out = self.wrapper.forward_step(
                page_token_ids=pages,
                query_ids=q_ids,
                answer_ids=a_ids,
                target_page_idx=p_idx,
                edit_steps=[(p_idx, e_acc if is_acc else e_rej)],
                edit_accepted_flags=[is_acc],
                use_exact_bank=True,
                run_reconstruction=False,
                run_matryoshka=False,
                distill_salience=False,
            )
            out.loss_total.backward()
            opt.step()

        self.wrapper.lslc.eval()
        embed_fn = self.wrapper.llm.get_input_embeddings()
        with torch.no_grad():
            z0, bank0, _ = self.wrapper.lslc.encode_book(pages, embed_fn, use_exact_bank=True, vocab_size=512)
            # 1) Accepted edit opens strip gate > 0.25 and writes 445 into exact bank
            z_acc, g_acc, bank_acc = self.wrapper.lslc.apply_page_delta(
                z0, p_idx, e_acc, embed_fn, exact_bank=bank0, vocab_size=512
            )
            self.assertGreater(float(g_acc[:, -1, :].mean().item()), 0.25)
            acc_tokens = bank_acc.token_ids[0, 1].tolist()
            self.assertIn(445, acc_tokens)

            # 2) Rejected edit inference (NO external flag passed): strip gate <= 0.25 and exact_bank stays bit-identical
            z_rej, g_rej, bank_rej = self.wrapper.lslc.apply_page_delta(
                z_acc, p_idx, e_rej, embed_fn, exact_bank=bank_acc, vocab_size=512, in_place=False
            )
            strip_gate = float(g_rej[:, -1, :].mean().item())
            self.assertLessEqual(strip_gate, 0.25)
            self.assertEqual(bank_rej.token_ids[0, 1].tolist(), acc_tokens)
            self.assertNotIn(490, bank_rej.token_ids[0, 1].tolist())
            self.assertEqual(float((z_rej[:, :, :2] - z_acc[:, :, :2]).abs().max().item()), 0.0)

    def test_distill_modes_none_cond_b_and_cond_a_verified_text(self) -> None:
        pages = torch.randint(0, 512, (1, 4, self.cfg.page_size))
        verified = pages.clone()
        verified[0, 1, :16] = torch.randint(0, 512, (16,))
        q_ids = torch.randint(0, 512, (1, 8))
        a_ids = torch.randint(0, 512, (1, 4))
        e_ids = torch.randint(0, 512, (1, 10))

        self.wrapper.lslc.train()
        losses: dict[str, float] = {}
        for d_mode in ("none", "cond_b", "cond_a_verified_text"):
            self.wrapper.lslc.zero_grad(set_to_none=True)
            out = self.wrapper.forward_step(
                page_token_ids=pages,
                query_ids=q_ids,
                answer_ids=a_ids,
                target_page_idx=torch.tensor([1]),
                edit_page_idx=torch.tensor([1]),
                edit_token_ids=e_ids,
                edit_accepted_flags=[True],
                verified_final_pages=verified,
                lambda_cond_b=0.5,
                distill_mode=d_mode,
                use_exact_bank=True,
                run_matryoshka=False,
                distill_salience=False,
            )
            self.assertIsNotNone(out.loss_cond_b)
            self.assertTrue(torch.isfinite(out.loss_cond_b).item())
            losses[d_mode] = float(out.loss_cond_b.item())
            out.loss_total.backward()
            has_grad = any(p.grad is not None and torch.isfinite(p.grad).all().item() for p in self.wrapper.lslc.parameters())
            self.assertTrue(has_grad, f"Missing finite grad for distill_mode={d_mode}")

        self.assertEqual(losses["none"], 0.0)
        self.assertGreater(losses["cond_b"], 0.0)
        self.assertGreater(losses["cond_a_verified_text"], 0.0)

    def test_v3_salience_gate_tokenizer_agnostic_and_unclamped(self) -> None:
        torch.manual_seed(11)
        cfg_v3 = LSLCConfig.tiny_mock(salience_mode="learned_surprisal")
        llm = TinyMockCausalLM(vocab_size=512, d_llm=cfg_v3.d_llm, num_layers=cfg_v3.num_llm_layers)
        wrapper_v3 = FrozenQwenLSLCWrapper(cfg_v3, llm)
        wrapper_v3.lslc.eval()

        # Inflate linear weights so unclamped magnitude strictly exceeds v2's [-0.5, 0.5] tanh bound
        with torch.no_grad():
            wrapper_v3.lslc.salience_gate.scorer.weight.mul_(5.0)
            wrapper_v3.lslc.salience_gate.ctx_out.weight.mul_(5.0)

        embeds = torch.randn(1, 32, cfg_v3.d_llm)
        toks_a = torch.randint(0, 512, (1, 32))
        toks_b = torch.randint(0, 150000, (1, 32))
        with torch.no_grad():
            logits_a = wrapper_v3.lslc.salience_gate.score_logits(embeds, token_ids=toks_a, vocab_size=512)
            logits_b = wrapper_v3.lslc.salience_gate.score_logits(embeds, token_ids=toks_b, vocab_size=151936)
        self.assertTrue(torch.allclose(logits_a, logits_b, atol=1e-6))
        self.assertGreater(float(logits_a.abs().max().item()), 0.5)

    def test_v3_delta_updater_cross_attn_position_agnostic(self) -> None:
        torch.manual_seed(17)
        cfg_v3 = LSLCConfig.tiny_mock(salience_mode="learned_surprisal", delta_pooling_mode="cross_attn")
        llm = TinyMockCausalLM(vocab_size=512, d_llm=cfg_v3.d_llm, num_layers=cfg_v3.num_llm_layers)
        wrapper_v3 = FrozenQwenLSLCWrapper(cfg_v3, llm)
        embed_fn = llm.get_input_embeddings()

        pages = torch.randint(10, 400, (1, 4, cfg_v3.page_size))
        e_toks = torch.tensor([[311, 312, 313, 314, 11, 12, 13, 14, 15, 16, 17, 18]])
        z_book, bank, _ = wrapper_v3.lslc.encode_book(pages, embed_fn, use_exact_bank=True, vocab_size=512)
        z_new, g_p, bank_new = wrapper_v3.lslc.apply_page_delta(
            z_book, torch.tensor([1]), e_toks, embed_fn, exact_bank=bank, vocab_size=512
        )
        self.assertIsNotNone(wrapper_v3.lslc.delta_updater._last_edit_attn)
        self.assertEqual(tuple(wrapper_v3.lslc.delta_updater._last_edit_attn.shape), (1, e_toks.shape[1]))
        z_new.sum().backward()
        has_attn_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all().item()
            for p in wrapper_v3.lslc.delta_updater.sticky_attn.parameters()
        )
        self.assertTrue(has_attn_grad)
        self.assertTrue(torch.isfinite(g_p).all().item())
        self.assertEqual(tuple(bank_new.token_ids.shape), tuple(bank.token_ids.shape))

        # Verify position-agnostic update selects early high-weight tokens rather than tail filler
        with torch.no_grad():
            early_weights = torch.tensor([[10.0, 9.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            bank_early = bank.update_page_from_delta(
                page_indices=torch.tensor([2]),
                edit_token_ids=e_toks,
                edit_embeds=embed_fn(e_toks),
                salience_gate=wrapper_v3.lslc.salience_gate,
                delta_gates=torch.ones_like(g_p),
                grid_h=cfg_v3.grid_h,
                grid_w=cfg_v3.grid_w,
                vocab_size=512,
                edit_token_weights=early_weights,
                use_tail_bonus=False,
            )
            k_sticky = min(max(cfg_v3.k_exact // 2, (cfg_v3.k_exact * 2) // 3), e_toks.shape[1])
            picked = set(bank_early.token_ids[0, 2, cfg_v3.k_exact - k_sticky :].tolist())
            self.assertTrue({311, 312, 313}.issubset(picked), f"Expected early tokens in {picked}")

    def test_v3_assembler_gather_first_exact_parity(self) -> None:
        torch.manual_seed(23)
        embed_fn = self.wrapper.llm.get_input_embeddings()
        pages = torch.randint(0, 512, (1, 16, self.cfg.page_size))
        q_ids = torch.randint(0, 512, (1, 6))
        self.wrapper.lslc.eval()
        with torch.no_grad():
            z_book, bank, toc = self.wrapper.lslc.encode_book(pages, embed_fn, use_exact_bank=True, vocab_size=512)
            q_emb = embed_fn(q_ids).mean(dim=1)
            r_out, asm = self.wrapper.lslc.route_and_assemble(
                z_book, toc, q_emb, bank, embed_fn, k_deep=2, k_skim=2, mode="full", training_ste=False
            )
            # Manual reference from full-book skim + deep projection
            skim_all = self.wrapper.lslc.projector.project_skim(z_book)
            deep_all = self.wrapper.lslc.projector.project_deep(z_book)
            d_idx = r_out.deep_idx[0]
            s_idx = r_out.skim_idx[0]
            ref_skim = skim_all[:, s_idx].reshape(1, -1, self.cfg.d_llm)
            ref_l0 = deep_all["l0"][:, d_idx].reshape(1, -1, self.cfg.d_llm)
            ref_l1 = deep_all["l1"][:, d_idx].reshape(1, -1, self.cfg.d_llm)
            ref_l2 = deep_all["l2"][:, d_idx].reshape(1, -1, self.cfg.d_llm)

            s0, s1 = asm.slice_bounds["skim"]
            self.assertLess(float((asm.prefix_embeds[:, s0:s1] - ref_skim).abs().max().item()), 1e-5)
            d0a, d0b = asm.slice_bounds["deep_l0"]
            self.assertLess(float((asm.prefix_embeds[:, d0a:d0b] - ref_l0).abs().max().item()), 1e-5)
            d1a, d1b = asm.slice_bounds["deep_l1"]
            self.assertLess(float((asm.prefix_embeds[:, d1a:d1b] - ref_l1).abs().max().item()), 1e-5)
            d2a, d2b = asm.slice_bounds["deep_l2"]
            self.assertLess(float((asm.prefix_embeds[:, d2a:d2b] - ref_l2).abs().max().item()), 1e-5)

    def test_adversarial_dataset_100pct_disjoint_values_projects_policies(self) -> None:
        self.assertTrue(set(AdversarialLivingBookDataset.TRAIN_STATIC_POLICIES).isdisjoint(AdversarialLivingBookDataset.EVAL_STATIC_POLICIES))
        self.assertTrue(set(AdversarialLivingBookDataset.TRAIN_PROJECTS).isdisjoint(AdversarialLivingBookDataset.EVAL_PROJECTS))
        self.assertTrue(set(AdversarialLivingBookDataset.TRAIN_BUDGET_POOL).isdisjoint(AdversarialLivingBookDataset.EVAL_BUDGET_POOL))
        self.assertTrue(set(AdversarialLivingBookDataset.TRAIN_SYMBOLIC_STATES).isdisjoint(AdversarialLivingBookDataset.EVAL_SYMBOLIC_STATES))

        for mode in ("budget", "symbolic"):
            ds_tr = AdversarialLivingBookDataset(num_episodes=16, num_pages=6, num_edits=3, mode=mode, split="train", seed=5)
            ds_ev = AdversarialLivingBookDataset(num_episodes=16, num_pages=6, num_edits=3, mode=mode, split="eval", seed=5)
            tr_projs = {ep.metadata["proj_a"] for ep in ds_tr} | {ep.metadata["proj_b"] for ep in ds_tr}
            ev_projs = {ep.metadata["proj_a"] for ep in ds_ev} | {ep.metadata["proj_b"] for ep in ds_ev}
            self.assertTrue(tr_projs.isdisjoint(ev_projs))
            tr_ans = {ep.metadata["answer_str"] for ep in ds_tr}
            ev_ans = {ep.metadata["answer_str"] for ep in ds_ev}
            self.assertTrue(tr_ans.isdisjoint(ev_ans), f"Answer overlap in mode={mode}: {tr_ans & ev_ans}")
            # Verify randomized intra-page offsets and clean un-diluted router_query_ids
            self.assertGreater(len({ep.metadata["proj_a_offset"] for ep in ds_ev}), 1)
            self.assertLess(len(ds_ev[0].metadata["router_query_ids"]), len(ds_ev[0].query_ids))

    def test_v3_procedural_train_open_vocab_and_salience_span_mask(self) -> None:
        cfg_v3 = LSLCConfig.tiny_mock(salience_mode="learned_surprisal", delta_pooling_mode="cross_attn")
        ds_sym = AdversarialLivingBookDataset(num_episodes=24, num_pages=6, page_size=cfg_v3.page_size, num_edits=3, mode="symbolic", split="train", seed=17)
        ds_bud = AdversarialLivingBookDataset(num_episodes=24, num_pages=6, page_size=cfg_v3.page_size, num_edits=3, mode="budget", split="train", seed=19)
        sym_ans = {ep.metadata["answer_str"] for ep in ds_sym}
        bud_ans = {ep.metadata["answer_str"] for ep in ds_bud}
        self.assertGreaterEqual(len(sym_ans), 10)
        self.assertGreaterEqual(len(bud_ans), 10)
        self.assertTrue(sym_ans.isdisjoint(AdversarialLivingBookDataset.EVAL_SYMBOLIC_STATES))
        self.assertTrue(bud_ans.isdisjoint(AdversarialLivingBookDataset.EVAL_BUDGET_POOL))

        ep = ds_sym[0]
        ver_mask = ep.metadata["verified_salience_span_mask"]
        ctx_mask = ep.metadata["context_salience_span_mask"]
        self.assertEqual(tuple(ver_mask.shape), (6, cfg_v3.page_size))
        self.assertEqual(tuple(ctx_mask.shape), (6, cfg_v3.page_size))
        self.assertGreater(float(ver_mask[ep.target_page_idx].sum().item()), 0.0)

        llm_v3 = TinyMockCausalLM(vocab_size=512, d_llm=cfg_v3.d_llm, num_layers=cfg_v3.num_llm_layers)
        wrapper_v3 = FrozenQwenLSLCWrapper(cfg_v3, llm_v3)
        wrapper_v3.lslc.zero_grad()
        out = wrapper_v3.forward_step(
            page_token_ids=ep.metadata["verified_final_pages"].unsqueeze(0),
            query_ids=ep.query_ids.unsqueeze(0),
            answer_ids=ep.answer_ids.unsqueeze(0),
            target_page_idx=torch.tensor([ep.target_page_idx], dtype=torch.long),
            run_reconstruction=True,
            run_matryoshka=True,
            distill_salience=True,
            lambda_sal=2.0,
            router_query_ids=ep.metadata["router_query_ids"].unsqueeze(0),
            salience_span_mask=ver_mask.unsqueeze(0),
        )
        out.loss_total.backward()
        self.assertTrue(torch.isfinite(out.loss_total))
        self.assertIsNotNone(wrapper_v3.lslc.salience_gate.ctx_conv2.weight.grad)
        self.assertGreater(float(wrapper_v3.lslc.salience_gate.ctx_conv2.weight.grad.abs().sum().item()), 0.0)

    def test_v3_lora_injection_step0_identity_and_co_adaptation_grads(self) -> None:
        torch.manual_seed(31)
        cfg_v3 = LSLCConfig.tiny_mock(salience_mode="learned_surprisal", delta_pooling_mode="cross_attn")
        llm_v3 = TinyMockCausalLM(vocab_size=512, d_llm=cfg_v3.d_llm, num_layers=cfg_v3.num_llm_layers)
        x = torch.randint(0, 512, (1, 16))
        with torch.no_grad():
            base_logits = llm_v3(input_ids=x).logits.detach()

        wrapper_v3 = FrozenQwenLSLCWrapper(cfg_v3, llm_v3)
        replaced = inject_lora(wrapper_v3.llm, r=4, alpha=8.0, target_modules=("qkv", "out"))
        self.assertEqual(len(replaced), cfg_v3.num_llm_layers * 2)
        for p_base_name, p_base in wrapper_v3.llm.named_parameters():
            if "lora_A" in p_base_name or "lora_B" in p_base_name:
                self.assertTrue(p_base.requires_grad)
            else:
                self.assertFalse(p_base.requires_grad)

        with torch.no_grad():
            step0_logits = wrapper_v3.llm(input_ids=x).logits
        self.assertLess(float((step0_logits - base_logits).abs().max().item()), 1e-5)

        first_lora_b = lora_parameters(wrapper_v3.llm)[1]
        with torch.no_grad():
            first_lora_b.fill_(1.0)
            perturbed_logits = wrapper_v3.llm(input_ids=x).logits
            self.assertGreater(float((perturbed_logits - base_logits).abs().max().item()), 1e-4)
            set_lora_enabled(wrapper_v3.llm, False)
            disabled_logits = wrapper_v3.llm(input_ids=x).logits
            self.assertLess(float((disabled_logits - base_logits).abs().max().item()), 1e-5)
            first_lora_b.zero_()
            set_lora_enabled(wrapper_v3.llm, True)

        ds = AdversarialLivingBookDataset(num_episodes=1, num_pages=4, page_size=cfg_v3.page_size, num_edits=2, mode="symbolic", split="train", seed=33)
        ep = ds[0]
        wrapper_v3.lslc.zero_grad()
        wrapper_v3.llm.zero_grad()
        out = wrapper_v3.forward_step(
            page_token_ids=ep.metadata["verified_final_pages"].unsqueeze(0),
            query_ids=ep.query_ids.unsqueeze(0),
            answer_ids=ep.answer_ids.unsqueeze(0),
            target_page_idx=torch.tensor([ep.target_page_idx], dtype=torch.long),
            run_reconstruction=True,
            run_matryoshka=True,
            distill_salience=True,
            router_query_ids=ep.metadata["router_query_ids"].unsqueeze(0),
            salience_span_mask=ep.metadata["verified_salience_span_mask"].unsqueeze(0),
        )
        out.loss_total.backward()
        self.assertTrue(torch.isfinite(out.loss_total))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in wrapper_v3.lslc.parameters() if p.requires_grad))
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all().item() and p.grad.abs().sum().item() > 0 for p in lora_parameters(wrapper_v3.llm)))

    def test_v3_k_exact_24_config_not_overwritten(self) -> None:
        cfg24 = LSLCConfig.qwen3_1_7b(k_exact=24)
        self.assertEqual(cfg24.k_exact, 24)
        self.assertAlmostEqual(cfg24.exact_ratio, 24.0 / 256.0, places=6)
        cfg_v2 = LSLCConfig.qwen3_1_7b()
        self.assertEqual(cfg_v2.k_exact, 16)
        self.assertEqual(cfg_v2.router_mode, "v2_legacy")

    def test_v3_lora_merge_unmerge_exact_parity(self) -> None:
        torch.manual_seed(41)
        cfg_m = LSLCConfig.tiny_mock()
        llm_m = TinyMockCausalLM(vocab_size=512, d_llm=cfg_m.d_llm, num_layers=cfg_m.num_llm_layers)
        replaced = inject_lora(llm_m, r=4, alpha=8.0, target_modules=("qkv", "out"))
        self.assertGreater(len(replaced), 0)
        for idx, p in enumerate(lora_parameters(llm_m)):
            with torch.no_grad():
                p.fill_(0.05 if idx % 2 == 1 else 0.1)
        x = torch.randint(0, 512, (1, 12))
        with torch.no_grad():
            unmerged_logits = llm_m(input_ids=x).logits
            n_merged = merge_lora(llm_m)
            self.assertEqual(n_merged, len(replaced))
            merged_logits = llm_m(input_ids=x).logits
            self.assertLess(float((merged_logits - unmerged_logits).abs().max().item()), 1e-4)
            n_unmerged = unmerge_lora(llm_m)
            self.assertEqual(n_unmerged, len(replaced))
            restored_logits = llm_m(input_ids=x).logits
            self.assertLess(float((restored_logits - unmerged_logits).abs().max().item()), 1e-5)

    def test_v3_content_attn_router_permutation_invariance_and_grads(self) -> None:
        torch.manual_seed(47)
        cfg_c = LSLCConfig.tiny_mock(
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
        )
        llm_c = TinyMockCausalLM(vocab_size=512, d_llm=cfg_c.d_llm, num_layers=cfg_c.num_llm_layers)
        wrapper_c = FrozenQwenLSLCWrapper(cfg_c, llm_c)
        embed_fn = llm_c.get_input_embeddings()
        pages = torch.randint(0, 512, (1, 6, cfg_c.page_size))
        perm = torch.tensor([3, 0, 5, 1, 4, 2], dtype=torch.long)
        rq_ids = torch.randint(0, 512, (1, 8))

        wrapper_c.lslc.eval()
        with torch.no_grad():
            z1, bank1, toc1 = wrapper_c.lslc.encode_book(pages, embed_fn, use_exact_bank=True, vocab_size=512)
            r1, _ = wrapper_c.lslc.route_and_assemble(
                z1, toc1, embed_fn(rq_ids).mean(dim=1), bank1, embed_fn, query_input_ids=rq_ids, salience_route_weight=0.0
            )
            z2, bank2, toc2 = wrapper_c.lslc.encode_book(pages[:, perm], embed_fn, use_exact_bank=True, vocab_size=512)
            r2, _ = wrapper_c.lslc.route_and_assemble(
                z2, toc2, embed_fn(rq_ids).mean(dim=1), bank2, embed_fn, query_input_ids=rq_ids, salience_route_weight=0.0
            )
            self.assertLess(float((r2.logits[0] - r1.logits[0, perm]).abs().max().item()), 1e-4)

        wrapper_c.lslc.train()
        wrapper_c.lslc.zero_grad()
        ds = AdversarialLivingBookDataset(num_episodes=1, num_pages=6, page_size=cfg_c.page_size, num_edits=2, mode="symbolic", split="train", style="mixed_natural", seed=51)
        ep = ds[0]
        out = wrapper_c.forward_step(
            page_token_ids=ep.metadata["verified_final_pages"].unsqueeze(0),
            query_ids=ep.query_ids.unsqueeze(0),
            answer_ids=ep.answer_ids.unsqueeze(0),
            target_page_idx=torch.tensor([ep.target_page_idx], dtype=torch.long),
            router_query_ids=ep.metadata["router_query_ids"].unsqueeze(0),
            salience_span_mask=ep.metadata["verified_salience_span_mask"].unsqueeze(0),
        )
        out.loss_total.backward()
        self.assertIsNotNone(wrapper_c.lslc.router.q_token_gate.weight.grad)
        self.assertIsNotNone(wrapper_c.lslc.projector.toc_slot_attn.weight.grad)
        self.assertGreater(float(wrapper_c.lslc.router.q_token_gate.weight.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(wrapper_c.lslc.projector.toc_slot_attn.weight.grad.abs().sum().item()), 0.0)

    def test_v3_stop_delimiter_diverse_values_and_headerless_split(self) -> None:
        class _TokWithEOS:
            eos_token_id = 500
            pad_token_id = 0
            bos_token_id = 1
            def __len__(self) -> int:
                return 512
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                del add_special_tokens
                return [2 + (sum(ord(ch) for ch in tok) % 490) for tok in text.split()] or [2]
            def decode(self, ids, skip_special_tokens: bool = True) -> str:
                del skip_special_tokens
                return " ".join(str(x) for x in ids)

        tok = _TokWithEOS()
        ds_tr = AdversarialLivingBookDataset(
            num_episodes=24, num_pages=6, page_size=64, num_edits=3, mode="symbolic", split="train", tokenizer=tok, style="mixed_natural", seed=61
        )
        ds_hd = AdversarialLivingBookDataset(
            num_episodes=16, num_pages=6, page_size=64, num_edits=3, mode="symbolic", split="eval", style="headerless_eval", seed=61
        )
        for ep in ds_tr:
            self.assertEqual(int(ep.answer_ids[-1].item()), tok.eos_token_id)
        tr_vals = [ep.metadata["answer_str"] for ep in ds_tr]
        self.assertTrue(any("_" in v or " " in v or any(c.isupper() for c in v) for v in tr_vals))
        self.assertTrue(set(tr_vals).isdisjoint(AdversarialLivingBookDataset.EVAL_SYMBOLIC_STATES))

        # Verify headerless eval has zero RECORD: or POLICY: strings anywhere
        for ep in ds_hd:
            self.assertEqual(ep.metadata["style"], "headerless_eval")
            self.assertGreater(float(ep.metadata["verified_salience_span_mask"][ep.target_page_idx].max().item()), 0.99)
        for kind in ("static", "budget", "symbolic"):
            for fmt in ("eval_prose", "eval_code"):
                s = AdversarialLivingBookDataset._format_fact_stmt(kind, fmt=fmt, proj="Polaris", val="90M", rule="r")
                self.assertNotIn("RECORD:", s)
                self.assertNotIn("POLICY:", s)

    def test_v3_content_attn_router_fp32_logits_and_loss_drop(self) -> None:
        torch.manual_seed(71)
        cfg_r = LSLCConfig.tiny_mock(
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
        )
        llm_r = TinyMockCausalLM(vocab_size=512, d_llm=cfg_r.d_llm, num_layers=cfg_r.num_llm_layers)
        wrapper_r = FrozenQwenLSLCWrapper(cfg_r, llm_r)
        embed_fn = llm_r.get_input_embeddings()

        pages = torch.randint(10, 400, (1, 8, cfg_r.page_size))
        for p in range(8):
            pages[0, p, 4:8] = 420 + p
        q_tok = torch.tensor([[20, 425, 21, 22]])
        with torch.no_grad():
            z_b, bank_b, toc_b = wrapper_r.lslc.encode_book(pages, embed_fn, use_exact_bank=True, vocab_size=512)
            bank_emb = embed_fn(bank_b.token_ids).bfloat16()
            r_out = wrapper_r.lslc.router(
                embed_fn(q_tok).mean(dim=1).bfloat16(),
                toc_b.bfloat16(),
                k_deep=2,
                k_skim=2,
                query_token_embeds=embed_fn(q_tok).bfloat16(),
                exact_bank_embeds=bank_emb,
                salience_route_weight=0.0,
            )
            self.assertEqual(r_out.logits.dtype, torch.float32)
            self.assertGreater(float(r_out.logits.std().item()), 1e-4)

        opt = torch.optim.AdamW(
            list(wrapper_r.lslc.encoder.parameters())
            + list(wrapper_r.lslc.projector.parameters())
            + list(wrapper_r.lslc.router.parameters()),
            lr=3e-3,
        )
        wrapper_r.lslc.train()
        init_loss = None
        final_loss = None
        for step in range(40):
            tgt = step % 8
            q_s = torch.tensor([[20, 420 + tgt, 21, 22]])
            opt.zero_grad()
            z_s, bank_s, toc_s = wrapper_r.lslc.encode_book(pages, embed_fn, use_exact_bank=True, vocab_size=512)
            ro, _ = wrapper_r.lslc.route_and_assemble(
                z_s, toc_s, embed_fn(q_s).mean(dim=1), bank_s, embed_fn, query_input_ids=q_s, salience_route_weight=0.0
            )
            loss = torch.nn.functional.cross_entropy(ro.logits, torch.tensor([tgt], dtype=torch.long))
            loss.backward()
            opt.step()
            if step == 0:
                init_loss = float(loss.item())
            final_loss = float(loss.item())
        self.assertIsNotNone(init_loss)
        self.assertIsNotNone(final_loss)
        self.assertLess(final_loss, init_loss * 0.5)

    def test_v3_soft_idf_disambiguates_shared_prefix_entities_p64(self) -> None:
        torch.manual_seed(79)
        cfg_p = LSLCConfig.tiny_mock(
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=12,
        )
        llm_p = TinyMockCausalLM(vocab_size=512, d_llm=cfg_p.d_llm, num_layers=cfg_p.num_llm_layers)
        with torch.no_grad():
            llm_p.embed_tokens.weight.normal_(mean=0.0, std=1.0)
        wrapper_p = FrozenQwenLSLCWrapper(cfg_p, llm_p)
        embed_fn = llm_p.get_input_embeddings()

        num_p = 64
        pages = torch.randint(10, 300, (1, num_p, cfg_p.page_size))
        targets = [i * 5 + 1 for i in range(12)]
        shared_prefix = [310, 311, 312, 313, 314]  # simulates shared "What approved deployment state StressNode -"
        for idx_t, p_idx in enumerate(targets):
            pages[0, p_idx, 8:14] = torch.tensor(shared_prefix + [400 + idx_t])

        from lslc import SparseExactBank

        bank_toks = torch.randint(10, 300, (1, num_p, cfg_p.k_exact))
        for idx_t, p_idx in enumerate(targets):
            bank_toks[0, p_idx, :6] = torch.tensor(shared_prefix + [400 + idx_t])
        bank = SparseExactBank(
            token_ids=bank_toks,
            page_ids=torch.arange(num_p).view(1, num_p, 1).expand(1, num_p, cfg_p.k_exact),
            h_coords=torch.zeros((1, num_p, cfg_p.k_exact), dtype=torch.long),
            w_coords=torch.zeros((1, num_p, cfg_p.k_exact), dtype=torch.long),
        )
        toc = torch.zeros((1, num_p, 1, cfg_p.d_llm))
        z_dummy = torch.zeros((1, num_p, 3, cfg_p.latent_channels, cfg_p.grid_h, cfg_p.grid_w))

        top1_hits = 0
        wrapper_p.lslc.eval()
        with torch.no_grad():
            for idx_t, p_idx in enumerate(targets):
                q_ids = torch.tensor([shared_prefix + [400 + idx_t]])
                ro, _ = wrapper_p.lslc.route_and_assemble(
                    z_dummy, toc, embed_fn(q_ids).mean(dim=1), bank, embed_fn, query_input_ids=q_ids, salience_route_weight=2.0
                )
                if int(torch.argmax(ro.logits[0], dim=-1).item()) == p_idx:
                    top1_hits += 1
        self.assertGreaterEqual(top1_hits / len(targets), 0.90)

    def test_v3_invdf_ngram_disambiguates_stopwords_record_bias_and_split_digits_p64(self) -> None:
        torch.manual_seed(83)
        cfg_p = LSLCConfig.tiny_mock(
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=12,
        )
        llm_p = TinyMockCausalLM(vocab_size=512, d_llm=cfg_p.d_llm, num_layers=cfg_p.num_llm_layers)
        with torch.no_grad():
            llm_p.embed_tokens.weight.normal_(mean=0.0, std=1.0)
        wrapper_p = FrozenQwenLSLCWrapper(cfg_p, llm_p)
        embed_fn = llm_p.get_input_embeddings()

        num_p = 64
        pages = torch.randint(10, 250, (1, num_p, cfg_p.page_size))
        targets = [i * 5 + 2 for i in range(12)]
        stopwords = [260, 261, 262]  # df=0: never inserted into any bank
        boilerplate = [270, 271, 272]  # RECORD: boilerplate on half the entities
        shared_stem = [310, 311, 312]
        digit_pairs = [
            [400, 401],
            [401, 400],
            [400, 400],
            [400, 402],
            [402, 400],
            [401, 402],
            [402, 401],
            [403, 400],
            [400, 403],
            [403, 401],
            [401, 403],
            [403, 403],
        ]

        from lslc import SparseExactBank

        bank_toks = torch.randint(10, 250, (1, num_p, cfg_p.k_exact))
        for idx_t, p_idx in enumerate(targets):
            seq = (boilerplate if idx_t % 2 == 0 else []) + shared_stem + digit_pairs[idx_t]
            bank_toks[0, p_idx, : len(seq)] = torch.tensor(seq)
        bank = SparseExactBank(
            token_ids=bank_toks,
            page_ids=torch.arange(num_p).view(1, num_p, 1).expand(1, num_p, cfg_p.k_exact),
            h_coords=torch.zeros((1, num_p, cfg_p.k_exact), dtype=torch.long),
            w_coords=torch.zeros((1, num_p, cfg_p.k_exact), dtype=torch.long),
        )
        toc = torch.zeros((1, num_p, 1, cfg_p.d_llm))
        z_dummy = torch.zeros((1, num_p, 3, cfg_p.latent_channels, cfg_p.grid_h, cfg_p.grid_w))

        top1_hits = 0
        wrapper_p.lslc.eval()
        with torch.no_grad():
            for idx_t, p_idx in enumerate(targets):
                q_ids = torch.tensor([stopwords + boilerplate + shared_stem + digit_pairs[idx_t]])
                ro, _ = wrapper_p.lslc.route_and_assemble(
                    z_dummy, toc, embed_fn(q_ids).mean(dim=1), bank, embed_fn, query_input_ids=q_ids, salience_route_weight=2.0
                )
                if int(torch.argmax(ro.logits[0], dim=-1).item()) == p_idx:
                    top1_hits += 1

            # Zero-match query must produce finite logits without NaNs
            q_zero = torch.tensor([[260, 261, 262, 263]])
            ro_zero, _ = wrapper_p.lslc.route_and_assemble(
                z_dummy, toc, embed_fn(q_zero).mean(dim=1), bank, embed_fn, query_input_ids=q_zero, salience_route_weight=2.0
            )
            self.assertFalse(torch.isnan(ro_zero.logits).any().item())
        self.assertGreaterEqual(top1_hits / len(targets), 0.90)

    def test_v3_stage2_multistep_gate_never_collapses_and_per_token_kl_bounded(self) -> None:
        torch.manual_seed(97)
        pages = torch.randint(10, 400, (1, 4, self.cfg.page_size))
        verified = pages.clone()
        verified[0, 1, :16] = torch.randint(10, 400, (16,))
        q_ids = torch.randint(10, 400, (1, 8))
        a_ids = torch.randint(10, 400, (1, 8))
        p_idx = torch.tensor([1])
        e_acc1 = torch.tensor([[411, 412, 445, 445, 446, 447]])
        e_rej1 = torch.tensor([[415, 416, 490, 490, 491, 492]])
        e_acc2 = torch.tensor([[411, 412, 455, 455, 456, 457]])
        e_rej2 = torch.tensor([[415, 416, 495, 495, 496, 497]])

        opt = torch.optim.AdamW(self.wrapper.lslc.delta_updater.parameters(), lr=5e-3)
        self.wrapper.lslc.train()
        last_gate = 0.0
        for step in range(12):
            mode = "cond_b" if step % 2 == 0 else "cond_a_verified_text"
            opt.zero_grad()
            out = self.wrapper.forward_step(
                page_token_ids=pages,
                query_ids=q_ids,
                answer_ids=a_ids,
                target_page_idx=p_idx,
                edit_steps=[(p_idx, e_acc1), (p_idx, e_rej1), (p_idx, e_acc2), (p_idx, e_rej2)],
                edit_accepted_flags=[True, False, True, False],
                verified_final_pages=verified,
                lambda_cond_b=0.5,
                distill_mode=mode,
                use_exact_bank=True,
                run_reconstruction=False,
                run_matryoshka=True,
                distill_salience=False,
            )
            self.assertTrue(torch.isfinite(out.loss_total).item())
            self.assertLess(float(out.loss_cond_b.item()), 25.0)
            last_gate = float(out.mean_gate)
            self.assertGreater(last_gate, 0.05)
            out.loss_total.backward()
            opt.step()

        self.wrapper.lslc.eval()
        embed_fn = self.wrapper.llm.get_input_embeddings()
        with torch.no_grad():
            z0, bank0, _ = self.wrapper.lslc.encode_book(pages, embed_fn, use_exact_bank=True, vocab_size=512)
            _, g_acc, _ = self.wrapper.lslc.apply_page_delta(z0, p_idx, e_acc2, embed_fn, exact_bank=bank0, vocab_size=512)
            _, g_rej, _ = self.wrapper.lslc.apply_page_delta(z0, p_idx, e_rej2, embed_fn, exact_bank=bank0, vocab_size=512)
            strip_acc = float(g_acc[:, -1, :].mean().item())
            strip_rej = float(g_rej[:, -1, :].mean().item())
            self.assertGreater(strip_acc, strip_rej)

    def test_v3_edited_page_retains_router_top1_errata_and_structural_invariance(self) -> None:
        torch.manual_seed(109)
        cfg_e = LSLCConfig.tiny_mock(
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode="content_attn",
            k_exact=24,
        )
        llm_e = TinyMockCausalLM(vocab_size=512, d_llm=cfg_e.d_llm, num_layers=cfg_e.num_llm_layers)
        with torch.no_grad():
            llm_e.embed_tokens.weight.normal_(mean=0.0, std=1.0)
        wrapper_e = FrozenQwenLSLCWrapper(cfg_e, llm_e)
        embed_fn = llm_e.get_input_embeddings()

        num_p = 16
        pages = torch.randint(10, 240, (1, num_p, cfg_e.page_size))
        boilerplate = [270, 271, 272]  # simulates "approved deployment state"
        proj_a_id = [350]
        proj_b_id = [351]
        from lslc import SparseExactBank

        bank_toks = torch.randint(10, 240, (1, num_p, cfg_e.k_exact), dtype=torch.int32)
        bank_toks[0, 3, :4] = torch.tensor(proj_a_id + boilerplate, dtype=torch.int32)
        bank_toks[0, 9, :4] = torch.tensor(proj_b_id + boilerplate, dtype=torch.int32)
        bank = SparseExactBank(
            token_ids=bank_toks,
            page_ids=torch.arange(num_p, dtype=torch.int16).view(1, num_p, 1).expand(1, num_p, cfg_e.k_exact),
            h_coords=torch.zeros((1, num_p, cfg_e.k_exact), dtype=torch.uint8),
            w_coords=torch.zeros((1, num_p, cfg_e.k_exact), dtype=torch.uint8),
        )
        wrapper_e.lslc.eval()
        with torch.no_grad():
            z_book = wrapper_e.lslc.encoder.encode_pages(embed_fn(pages))
            toc0 = wrapper_e.lslc.projector.project_toc(z_book)
            q_a = torch.tensor([[260, 261] + proj_a_id + boilerplate])
            r_pre_w0, _ = wrapper_e.lslc.route_and_assemble(
                z_book, toc0, embed_fn(q_a).mean(dim=1), bank, embed_fn, query_input_ids=q_a, salience_route_weight=0.0
            )

            z_cur = z_book.clone()
            bank_cur = bank
            p_edit = torch.tensor([3])
            for step_i in range(5):
                # 5 paraphrase edits on proj_a omitting boilerplate [270,271,272] and proj_a_id [350]
                e_tok = torch.tensor([[410 + step_i] * 6 + [440 + step_i] + [420 + step_i] * 10])
                z_cur, _, bank_cur = wrapper_e.lslc.apply_page_delta(
                    z_cur,
                    p_edit,
                    e_tok,
                    embed_fn,
                    exact_bank=bank_cur,
                    vocab_size=512,
                    in_place=True,
                )
            # Simulate active delta write onto k_sticky slots of proj_a without boilerplate/proj_a_id
            for step_i in range(5):
                e_tok = torch.tensor([[410 + step_i] * 6 + [440 + step_i] + [420 + step_i] * 10])
                bank_cur = bank_cur.update_page_from_delta(
                    page_indices=p_edit,
                    edit_token_ids=e_tok,
                    edit_embeds=embed_fn(e_tok),
                    salience_gate=wrapper_e.lslc.salience_gate,
                    delta_gates=torch.ones((1, cfg_e.grid_h, cfg_e.grid_w)),
                    grid_h=cfg_e.grid_h,
                    grid_w=cfg_e.grid_w,
                    vocab_size=512,
                    in_place=True,
                )
            self.assertIsNotNone(bank_cur.base_token_ids)
            toc_after = wrapper_e.lslc.projector.project_toc(z_cur)

            # Querying proj_a + boilerplate after 5 edits MUST still rank proj_a (#3) above bystander proj_b (#9)
            r_post_hyb, _ = wrapper_e.lslc.route_and_assemble(
                z_cur, toc_after, embed_fn(q_a).mean(dim=1), bank_cur, embed_fn, query_input_ids=q_a, salience_route_weight=2.0
            )
            self.assertEqual(int(torch.argmax(r_post_hyb.logits[0], dim=-1).item()), 3)

            # Querying newly-written edit token [444] must ALSO find proj_a (#3) via delta_bank_embeds
            q_new = torch.tensor([[260, 261, 444]])
            r_new_hyb, _ = wrapper_e.lslc.route_and_assemble(
                z_cur, toc_after, embed_fn(q_new).mean(dim=1), bank_cur, embed_fn, query_input_ids=q_new, salience_route_weight=2.0
            )
            self.assertEqual(int(torch.argmax(r_new_hyb.logits[0], dim=-1).item()), 3)

            # Dense w=0 routing uses immutable routing_token_ids() and TOC -> strictly bit-identical before/after edits
            r_post_w0, _ = wrapper_e.lslc.route_and_assemble(
                z_cur, toc_after, embed_fn(q_a).mean(dim=1), bank_cur, embed_fn, query_input_ids=q_a, salience_route_weight=0.0
            )
            self.assertEqual(float((r_post_w0.logits - r_pre_w0.logits).abs().max().item()), 0.0)

        # Verify rag_append_errata appends chronological edits cleanly (distinct from rag_immutable)
        if str(REPO_ROOT / "scripts") not in sys.path:
            sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from colab_run_v3_phase1 import _rag_prefix_for_episode

        class _SimpleTok:
            def decode(self, ids, skip_special_tokens=True):
                toks = ids.tolist() if hasattr(ids, "tolist") else list(ids)
                return " ".join(f"w{int(t)}" for t in toks)

        ds = AdversarialLivingBookDataset(num_episodes=1, num_pages=8, page_size=cfg_e.page_size, num_edits=3, mode="symbolic", split="eval", seed=123)
        ep = ds[0]
        s_tok = _SimpleTok()
        pref_imm = _rag_prefix_for_episode(ep, "rag_immutable", s_tok, embed_fn, torch.float32, torch.device("cpu"), k_top=2)
        pref_err = _rag_prefix_for_episode(ep, "rag_append_errata", s_tok, embed_fn, torch.float32, torch.device("cpu"), k_top=2)
        self.assertNotEqual(pref_imm.shape[1], pref_err.shape[1])


if __name__ == "__main__":
    unittest.main()
