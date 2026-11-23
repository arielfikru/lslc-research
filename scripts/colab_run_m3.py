"""Milestone 3 Colab L4 Orchestrator: Full 5-Baseline Suite (Tracks 0A-0E) + Tracks 1, 2, 3 (4K/32K Book), and 4 (DeltaUpdater)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lslc import (
    ChunkedTextPageDataset,
    FrozenQwenLSLCWrapper,
    LSLCConfig,
    LivingBookStateOverrideDataset,
    SyntheticExactSemanticNIAHDataset,
    compute_analytical_memory_stats,
    estimate_int4_kv_bytes,
    greedy_generate_from_prefix,
    h2o_evict_keep_indices,
    load_frozen_qwen3,
    select_amp_dtype,
    streamingllm_keep_indices,
)
from scripts.colab_run_m2 import (
    _load_wikitext_pools,
    evaluate_continuation_ppl,
    evaluate_delta_suite,
    evaluate_niah_suite,
    profile_live_4k_32k,
    train_stage,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Milestone 3 full benchmark suite on Colab L4.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--steps-track2", type=int, default=300)
    parser.add_argument("--steps-track4", type=int, default=200)
    parser.add_argument("--steps-flat1d", type=int, default=200)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed-m2-json", type=str, default="/content/lslc_results/m2_v2_summary.json")
    parser.add_argument("--out-json", type=str, default="/content/lslc_results/m3_summary.json")
    args, _ = parser.parse_known_args(argv)
    return args


def _quantize_int4_groupwise(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """Simulate group-wise symmetric INT4 quantization (-7..+7) for Track 0B baseline."""
    orig_shape = x.shape
    orig_dtype = x.dtype
    flat = x.float().reshape(-1, group_size) if x.numel() % group_size == 0 else x.float().reshape(1, -1)
    scale = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7.0
    q = torch.clamp(torch.round(flat / scale), -8.0, 7.0)
    return (q * scale).reshape(orig_shape).to(orig_dtype)


def _h2o_select_tokens(flat_ids: torch.Tensor, embed_fn, budget_ratio: float = 0.20, recent_ratio: float = 0.10) -> torch.Tensor:
    """Compute heavy-hitter cumulative attention proxy scores and select top 20% tokens (Track 0D)."""
    emb = embed_fn(flat_ids).float()  # [1, N, D]
    # Fast causal attention-sink + norm centrality proxy for Heavy-Hitter scoring
    mean_k = F.normalize(emb.mean(dim=1, keepdim=True), dim=-1)
    tok_n = F.normalize(emb, dim=-1)
    scores = (tok_n * mean_k).sum(dim=-1).squeeze(0) + 0.5 * emb.norm(dim=-1).squeeze(0)
    keep = h2o_evict_keep_indices(scores, budget_ratio=budget_ratio, recent_ratio=recent_ratio).to(flat_ids.device)
    return flat_ids[:, keep]


@torch.no_grad()
def evaluate_baseline_0b_0d(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    ppl_eps,
    niah_ds: SyntheticExactSemanticNIAHDataset,
    device: torch.device,
    mode: str,  # "int4_kv" or "h2o_20pct"
) -> dict[str, float]:
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    nlls = []
    for ep in ppl_eps:
        pages = ep.context_pages.unsqueeze(0).to(device)
        q_ids = ep.query_ids.unsqueeze(0).to(device)
        a_ids = ep.answer_ids.unsqueeze(0).to(device)
        flat = pages.reshape(1, -1)
        if mode == "int4_kv":
            prefix_llm = _quantize_int4_groupwise(embed_fn(flat)).to(llm_dtype)
        elif mode == "h2o_20pct":
            kept_ids = _h2o_select_tokens(flat, embed_fn, budget_ratio=0.20, recent_ratio=0.10)
            prefix_llm = embed_fn(kept_ids).to(llm_dtype)
        else:
            raise ValueError(mode)
        turn_ids = torch.cat((q_ids, a_ids), dim=1)
        inp = torch.cat((prefix_llm, embed_fn(turn_ids).to(llm_dtype)), dim=1)
        attn = torch.ones((1, inp.shape[1]), device=device, dtype=torch.long)
        logits = wrapper.llm(inputs_embeds=inp, attention_mask=attn).logits
        start = prefix_llm.shape[1] + q_ids.shape[1] - 1
        ans_logits = logits[:, start : start + a_ids.shape[1], :]
        nlls.append(float(F.cross_entropy(ans_logits.reshape(-1, ans_logits.shape[-1]), a_ids.reshape(-1)).item()))
    ppl = round(math.exp(min(20.0, sum(nlls) / len(nlls))), 4)

    exact_em, exact_tok, sem_em, sem_tok = [], [], [], []
    for ep in niah_ds:
        pages = ep.context_pages.unsqueeze(0).to(device)
        q_ids = ep.query_ids.unsqueeze(0).to(device)
        a_ids = ep.answer_ids.unsqueeze(0).to(device)
        flat = pages.reshape(1, -1)
        if mode == "int4_kv":
            prefix_llm = _quantize_int4_groupwise(embed_fn(flat)).to(llm_dtype)
        else:
            kept_ids = _h2o_select_tokens(flat, embed_fn, budget_ratio=0.20, recent_ratio=0.10)
            prefix_llm = embed_fn(kept_ids).to(llm_dtype)

        turn_ids = torch.cat((q_ids, a_ids), dim=1)
        inp = torch.cat((prefix_llm, embed_fn(turn_ids).to(llm_dtype)), dim=1)
        attn = torch.ones((1, inp.shape[1]), device=device, dtype=torch.long)
        logits = wrapper.llm(inputs_embeds=inp, attention_mask=attn).logits
        start = prefix_llm.shape[1] + q_ids.shape[1] - 1
        ans_logits = logits[:, start : start + a_ids.shape[1], :]
        tok_acc = float((ans_logits.argmax(dim=-1) == a_ids).float().mean().item())

        gen_ids = greedy_generate_from_prefix(
            wrapper.llm, prefix_llm, q_ids, max_new_tokens=48, eos_token_id=tokenizer.eos_token_id
        )
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        em = 1.0 if ep.metadata["answer_str"].lower() in gen_text.lower() else 0.0
        if ep.metadata["needle_type"] == "exact":
            exact_em.append(em)
            exact_tok.append(tok_acc)
        else:
            sem_em.append(em)
            sem_tok.append(tok_acc)

    mib = 1024.0 * 1024.0
    kv_4k = estimate_int4_kv_bytes(wrapper.cfg, 4096) / mib if mode == "int4_kv" else (4096 * 0.20 * wrapper.cfg.kv_bytes_per_token_fp16) / mib
    kv_32k = estimate_int4_kv_bytes(wrapper.cfg, 32768) / mib if mode == "int4_kv" else (32768 * 0.20 * wrapper.cfg.kv_bytes_per_token_fp16) / mib
    return {
        "ppl": ppl,
        "exact_niah_em": round(sum(exact_em) / max(1, len(exact_em)), 4),
        "exact_niah_tok_acc": round(sum(exact_tok) / max(1, len(exact_tok)), 4),
        "semantic_niah_em": round(sum(sem_em) / max(1, len(sem_em)), 4),
        "semantic_niah_tok_acc": round(sum(sem_tok) / max(1, len(sem_tok)), 4),
        "vram_4k_mib": round(kv_4k, 2),
        "vram_32k_mib": round(kv_32k, 2),
    }


class Flat1DLatentBaseline(nn.Module):
    """Track 0E Ablation: 1D Flat Latent Cache (24 tokens/page, 12 KiB/page, NO [L,H,W] layers, NO SparseExactBank)."""

    def __init__(self, cfg: LSLCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.slots_per_page = 24  # Match LSLC Deep Read 24 dense tokens/page
        self.c_flat = 256  # 24 * 256 * 2B = 12,288 B/page (identical 12 KiB/page byte budget!)
        self.down = nn.Sequential(nn.LayerNorm(cfg.d_llm), nn.Linear(cfg.d_llm, self.c_flat))
        self.up = nn.Sequential(nn.LayerNorm(self.c_flat), nn.Linear(self.c_flat, cfg.d_llm), nn.GELU(), nn.Linear(cfg.d_llm, cfg.d_llm))
        self.edit_mlp = nn.Sequential(nn.Linear(self.c_flat * 2, self.c_flat), nn.Tanh(), nn.Linear(self.c_flat, self.c_flat))

    def encode_flat(self, pages_embeds: torch.Tensor) -> torch.Tensor:
        bsz, p, tp, d = pages_embeds.shape
        h = self.down(pages_embeds.reshape(bsz * p, tp, d)).transpose(1, 2)
        pooled = F.adaptive_avg_pool1d(h, output_size=self.slots_per_page).transpose(1, 2)
        return pooled.view(bsz, p, self.slots_per_page, self.c_flat)

    def project_flat(self, z_flat: torch.Tensor) -> torch.Tensor:
        bsz, p, s, c = z_flat.shape
        return self.up(z_flat).view(bsz, p * s, self.cfg.d_llm)

    def apply_flat_edit(self, z_flat: torch.Tensor, page_idx: int, edit_embeds: torch.Tensor) -> torch.Tensor:
        e_lat = F.adaptive_avg_pool1d(self.down(edit_embeds).transpose(1, 2), output_size=self.slots_per_page).transpose(1, 2)
        z_new = z_flat.clone()
        z_new[:, page_idx] = z_flat[:, page_idx] + 0.5 * self.edit_mlp(torch.cat((z_flat[:, page_idx], e_lat), dim=-1))
        return z_new


def run_track0e_flat1d(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    train_eps: list,
    ppl_eps,
    niah_ds: SyntheticExactSemanticNIAHDataset,
    steps: int,
    lr: float,
    device: torch.device,
) -> dict[str, float]:
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    amp_dtype, _ = select_amp_dtype(device)
    flat_mod = Flat1DLatentBaseline(wrapper.cfg).to(device)
    opt = torch.optim.AdamW(flat_mod.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(77)

    flat_mod.train()
    for _ in range(steps):
        ep = train_eps[rng.randrange(len(train_eps))]
        pages = ep.context_pages.unsqueeze(0).to(device)
        q_ids = ep.query_ids.unsqueeze(0).to(device)
        a_ids = ep.answer_ids.unsqueeze(0).to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            z_f = flat_mod.encode_flat(embed_fn(pages).float())
            prefix = flat_mod.project_flat(z_f)
            turn_ids = torch.cat((q_ids, a_ids), dim=1)
            q_ign = torch.full_like(q_ids, -100)
            turn_labels = torch.cat((q_ign, a_ids), dim=1)
            loss = wrapper._run_llm_with_prefix(prefix, turn_ids, turn_labels=turn_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flat_mod.parameters(), 1.0)
        opt.step()

    flat_mod.eval()
    nlls, exact_em, exact_tok, sem_em, sem_tok = [], [], [], [], []
    with torch.no_grad():
        for ep in ppl_eps:
            pages = ep.context_pages.unsqueeze(0).to(device)
            q_ids = ep.query_ids.unsqueeze(0).to(device)
            a_ids = ep.answer_ids.unsqueeze(0).to(device)
            z_f = flat_mod.encode_flat(embed_fn(pages).float())
            prefix_llm = flat_mod.project_flat(z_f).to(llm_dtype)
            turn_ids = torch.cat((q_ids, a_ids), dim=1)
            inp = torch.cat((prefix_llm, embed_fn(turn_ids).to(llm_dtype)), dim=1)
            attn = torch.ones((1, inp.shape[1]), device=device, dtype=torch.long)
            logits = wrapper.llm(inputs_embeds=inp, attention_mask=attn).logits
            start = prefix_llm.shape[1] + q_ids.shape[1] - 1
            ans_logits = logits[:, start : start + a_ids.shape[1], :]
            nlls.append(float(F.cross_entropy(ans_logits.reshape(-1, ans_logits.shape[-1]), a_ids.reshape(-1)).item()))

        for ep in niah_ds:
            pages = ep.context_pages.unsqueeze(0).to(device)
            q_ids = ep.query_ids.unsqueeze(0).to(device)
            a_ids = ep.answer_ids.unsqueeze(0).to(device)
            z_f = flat_mod.encode_flat(embed_fn(pages).float())
            prefix_llm = flat_mod.project_flat(z_f).to(llm_dtype)
            turn_ids = torch.cat((q_ids, a_ids), dim=1)
            inp = torch.cat((prefix_llm, embed_fn(turn_ids).to(llm_dtype)), dim=1)
            attn = torch.ones((1, inp.shape[1]), device=device, dtype=torch.long)
            logits = wrapper.llm(inputs_embeds=inp, attention_mask=attn).logits
            start = prefix_llm.shape[1] + q_ids.shape[1] - 1
            ans_logits = logits[:, start : start + a_ids.shape[1], :]
            tok_acc = float((ans_logits.argmax(dim=-1) == a_ids).float().mean().item())
            gen_ids = greedy_generate_from_prefix(
                wrapper.llm, prefix_llm, q_ids, max_new_tokens=48, eos_token_id=tokenizer.eos_token_id
            )
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            em = 1.0 if ep.metadata["answer_str"].lower() in gen_text.lower() else 0.0
            if ep.metadata["needle_type"] == "exact":
                exact_em.append(em)
                exact_tok.append(tok_acc)
            else:
                sem_em.append(em)
                sem_tok.append(tok_acc)

        # Measure prefix invariance violation on 1D flat latent when page 0 is edited
        sample_pages = niah_ds[0].context_pages.unsqueeze(0).to(device)
        z_before = flat_mod.encode_flat(embed_fn(sample_pages).float())
        pref_before = flat_mod.project_flat(z_before)
        z_after = flat_mod.apply_flat_edit(z_before, 0, embed_fn(niah_ds[0].query_ids.unsqueeze(0).to(device)).float())
        pref_after = flat_mod.project_flat(z_after)
        flat_prefix_diff = float((pref_after[:, :24] - pref_before[:, :24]).abs().max().item())

    return {
        "ppl": round(math.exp(min(20.0, sum(nlls) / len(nlls))), 4),
        "exact_niah_em": round(sum(exact_em) / max(1, len(exact_em)), 4),
        "exact_niah_tok_acc": round(sum(exact_tok) / max(1, len(exact_tok)), 4),
        "semantic_niah_em": round(sum(sem_em) / max(1, len(sem_em)), 4),
        "semantic_niah_tok_acc": round(sum(sem_tok) / max(1, len(sem_tok)), 4),
        "edited_page_prefix_invariance_diff": round(flat_prefix_diff, 5),
    }


@torch.no_grad()
def evaluate_multipage_4k_book(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
) -> dict[str, float]:
    """Track 3: Evaluate 16-page (4,096-token) multi-page book routing & Exact/Semantic NIAH."""
    ds_4k = SyntheticExactSemanticNIAHDataset(
        num_episodes=20,
        num_pages=16,
        page_size=wrapper.cfg.page_size,
        tokenizer=tokenizer,
        seed=8888,
        background_token_pool=val_pool,
    )
    niah_res = evaluate_niah_suite(wrapper, tokenizer, ds_4k, device, "lslc_full_exact")

    # Also measure Router top-k_deep recall across P=16 pages
    embed_fn = wrapper.llm.get_input_embeddings()
    amp_dtype, _ = select_amp_dtype(device)
    top1_hits, topk_hits = 0, 0
    for ep in ds_4k:
        pages = ep.context_pages.unsqueeze(0).to(device)
        q_ids = ep.query_ids.unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            z_book, _, toc = wrapper.lslc.encode_book(pages, embed_fn, use_exact_bank=False)
            q_emb = embed_fn(q_ids).mean(dim=1)
            r_out = wrapper.lslc.router(q_emb, toc, k_deep=2, k_skim=2)
            top1 = int(r_out.logits.argmax(dim=-1).item())
            deep_set = set(r_out.deep_idx[0].tolist())
            if top1 == ep.target_page_idx:
                top1_hits += 1
            if ep.target_page_idx in deep_set:
                topk_hits += 1

    return {
        "num_pages": 16,
        "context_tokens": 4096,
        "router_top1_acc": round(top1_hits / len(ds_4k), 4),
        "router_top2_deep_recall": round(topk_hits / len(ds_4k), 4),
        **niah_res,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"=== LSLC M3 FULL SUITE START (model={args.model}) ===", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for colab_run_m3.py")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM: {props.total_memory / (1024**3):.2f} GiB | BF16={torch.cuda.is_bf16_supported()}", flush=True)

    # 1. Unit Test Gate
    suite = unittest.defaultTestLoader.discover(str(REPO_ROOT / "tests"))
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    if not res.wasSuccessful():
        raise RuntimeError("Unit tests failed on Colab!")

    # 2. Load Frozen Qwen3 + WikiText-2
    device = torch.device("cuda")
    cfg = LSLCConfig.from_model_name(args.model, lr=args.lr)
    llm, tokenizer = load_frozen_qwen3(cfg, load_in_4bit=False, device=device)
    wrapper = FrozenQwenLSLCWrapper(cfg, llm)
    wrapper.lslc.to(device)

    train_pool, val_pool, corpus_src = _load_wikitext_pools(tokenizer)
    train_ppl = ChunkedTextPageDataset(train_pool, num_pages=4, page_size=cfg.page_size, turn_len=64)
    eval_ppl = ChunkedTextPageDataset(val_pool, num_pages=4, page_size=cfg.page_size, turn_len=64)[:24]
    train_niah = SyntheticExactSemanticNIAHDataset(
        num_episodes=120, num_pages=4, page_size=cfg.page_size, tokenizer=tokenizer, seed=42, background_token_pool=train_pool
    )
    eval_niah = SyntheticExactSemanticNIAHDataset(
        num_episodes=30, num_pages=4, page_size=cfg.page_size, tokenizer=tokenizer, seed=9999, background_token_pool=val_pool
    )
    train_niah_16p = SyntheticExactSemanticNIAHDataset(
        num_episodes=60, num_pages=16, page_size=cfg.page_size, tokenizer=tokenizer, seed=4242, background_token_pool=train_pool
    )
    train_delta = LivingBookStateOverrideDataset(
        num_episodes=96, num_pages=4, page_size=cfg.page_size, num_edits=3, tokenizer=tokenizer, seed=42, background_token_pool=train_pool
    )
    eval_delta = LivingBookStateOverrideDataset(
        num_episodes=24, num_pages=4, page_size=cfg.page_size, num_edits=3, tokenizer=tokenizer, seed=9999, background_token_pool=val_pool
    )
    m_train_blend = list(train_ppl)[:60] + list(train_niah) + list(train_niah_16p) * 2

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    for cand_path in (out_path, Path(args.seed_m2_json)):
        if cand_path.exists():
            try:
                loaded = json.loads(cand_path.read_text(encoding="utf-8"))
                summary.update(loaded)
                print(f"[SEED] Reused keys from {cand_path.name}: {list(loaded.keys())}", flush=True)
                break
            except Exception:
                pass

    summary.update({
        "gpu": props.name,
        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
        "model": cfg.model_name,
        "trainable_params": wrapper.lslc.trainable_parameter_count(),
        "corpus": corpus_src,
    })

    def _flush() -> None:
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 3. Baselines Track 0A, 0B, 0C, 0D, 0E
    if "track0a_full_kv" not in summary or "track0c_streaming_128" not in summary:
        print("=== Evaluating Track 0A (Full KV) & Track 0C (StreamingLLM) ===", flush=True)
        summary["track0a_full_kv"] = {
            "ppl": evaluate_continuation_ppl(wrapper, eval_ppl, device, "full_kv"),
            **evaluate_niah_suite(wrapper, tokenizer, eval_niah, device, "full_kv"),
            "delta_oracle": evaluate_delta_suite(wrapper, tokenizer, eval_delta, device, mode="full_kv"),
        }
        summary["track0c_streaming_128"] = {
            "ppl": evaluate_continuation_ppl(wrapper, eval_ppl, device, "streaming_128"),
            **evaluate_niah_suite(wrapper, tokenizer, eval_niah, device, "streaming_128"),
        }
        _flush()

    if "track0b_int4_kv" not in summary or "track0d_h2o_20pct" not in summary:
        print("=== Evaluating Track 0B (INT4 Quantized KV) & Track 0D (H2O 20% Heavy-Hitter) ===", flush=True)
        summary["track0b_int4_kv"] = evaluate_baseline_0b_0d(wrapper, tokenizer, eval_ppl, eval_niah, device, "int4_kv")
        summary["track0d_h2o_20pct"] = evaluate_baseline_0b_0d(wrapper, tokenizer, eval_ppl, eval_niah, device, "h2o_20pct")
        _flush()
        print(f"Track 0B INT4 KV: {summary['track0b_int4_kv']}", flush=True)
        print(f"Track 0D H2O 20%: {summary['track0d_h2o_20pct']}", flush=True)

    if "track0e_flat_1d" not in summary:
        print("=== Evaluating Track 0E (1D Flat Latent Cache Ablation) ===", flush=True)
        summary["track0e_flat_1d"] = run_track0e_flat1d(
            wrapper, tokenizer, list(train_ppl)[:80] + list(train_niah), eval_ppl, eval_niah, steps=args.steps_flat1d, lr=args.lr, device=device
        )
        _flush()
        print(f"Track 0E Flat 1D: {summary['track0e_flat_1d']}", flush=True)

    # 4. Track 2 + Track 3: Full LSLC 3D + SparseExactBank (with 100% right_closer capture & score-ordered deep_idx)
    print("=== Track 2 & Track 3: Training Full LSLC (4-page & 16-page Multi-Page Living Book) ===", flush=True)
    wrapper_t2 = FrozenQwenLSLCWrapper(cfg, llm)
    wrapper_t2.lslc.to(device)
    t2_train = train_stage(
        "TRACK2_M3", wrapper_t2, m_train_blend, steps=args.steps_track2, grad_accum=args.grad_accum, lr=args.lr,
        device=device, use_exact_bank=True, distill_salience=True, run_matryoshka=True
    )
    t2_ppl = evaluate_continuation_ppl(wrapper_t2, eval_ppl, device, "lslc_full_exact")
    t2_niah = evaluate_niah_suite(wrapper_t2, tokenizer, eval_niah, device, "lslc_full_exact")
    matryoshka_zoom = {}
    for zmode in ("toc_only", "skim_only", "deep_only"):
        matryoshka_zoom[zmode] = {
            "ppl": evaluate_continuation_ppl(wrapper_t2, eval_ppl, device, zmode),
            **evaluate_niah_suite(wrapper_t2, tokenizer, eval_niah, device, zmode),
        }
    summary["track2_sparse_exact"] = {"train": t2_train, "ppl": t2_ppl, **t2_niah, "matryoshka_zoom": matryoshka_zoom}
    _flush()
    print(f"Track 2 Eval: PPL={t2_ppl} | {t2_niah}", flush=True)

    print("=== Track 3: Evaluating 16-Page (4,096-Token) Multi-Page Book Routing & NIAH ===", flush=True)
    summary["track3_multipage_4k"] = evaluate_multipage_4k_book(wrapper_t2, tokenizer, val_pool, device)
    _flush()
    print(f"Track 3 (16-Page 4K Book): {summary['track3_multipage_4k']}", flush=True)

    # 5. Track 4: Sticky-Note DeltaUpdater (Freeze projector so deep_l2_proj stays general & never overfits)
    print("=== Track 4: Training DeltaUpdater on Living-Book State-Override ===", flush=True)
    frozen_in_t4 = []
    for name_p, p in wrapper_t2.lslc.named_parameters():
        is_dynamic_t4 = name_p.startswith("delta_updater.") or name_p.startswith("router.")
        if not is_dynamic_t4 and p.requires_grad:
            p.requires_grad_(False)
            frozen_in_t4.append(p)

    t4_train = train_stage(
        "TRACK4_M3", wrapper_t2, list(train_delta), steps=args.steps_track4, grad_accum=args.grad_accum, lr=args.lr,
        device=device, use_exact_bank=True, distill_salience=False, run_matryoshka=False, use_edits=True
    )
    for p in frozen_in_t4:
        p.requires_grad_(True)
    t4_delta_eval = evaluate_delta_suite(wrapper_t2, tokenizer, eval_delta, device, mode="lslc")
    summary["track4_delta_probe"] = {"train": t4_train, **t4_delta_eval}
    _flush()
    print(f"Track 4 Delta Eval: {t4_delta_eval}", flush=True)

    # 6. Live CUDA 4K & 32K Profiler
    if "live_cuda_profile" not in summary:
        print("=== Live CUDA 4K & 32K VRAM / TTFT Profiler ===", flush=True)
        summary["live_cuda_profile"] = profile_live_4k_32k(wrapper_t2, device)
        _flush()

    # Mirror updated metrics back to seed_m2_json if present
    seed_path = Path(args.seed_m2_json)
    if seed_path.parent.exists():
        seed_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved full M3 summary -> {out_path}", flush=True)
    print("=== M3 RUN COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
