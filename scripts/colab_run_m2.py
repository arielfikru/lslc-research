"""Self-contained Colab L4 Milestone 2 + Track 4 Probe + Live CUDA 4K/32K Profiler orchestrator."""

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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lslc import (
    ChunkedTextPageDataset,
    FrozenQwenLSLCWrapper,
    LSLCConfig,
    LivingBookStateOverrideDataset,
    SyntheticExactSemanticNIAHDataset,
    compute_analytical_memory_stats,
    greedy_generate_from_prefix,
    load_frozen_qwen3,
    select_amp_dtype,
    streamingllm_keep_indices,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Milestone 2 empirical validation on Colab L4/T4.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--steps-track1", type=int, default=300)
    parser.add_argument("--steps-track2", type=int, default=300)
    parser.add_argument("--steps-track4", type=int, default=250)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out-json", type=str, default="/content/lslc_results/m2_v2_summary.json")
    parser.add_argument("--resume", action="store_true", help="Reuse completed Track 0A/0C/Smoke/Track1 from out-json")
    args, _ = parser.parse_known_args(argv)
    return args


def _load_wikitext_pools(tokenizer) -> tuple[list[int], list[int], str]:
    try:
        from datasets import load_dataset

        ds_train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        ds_val = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
        train_text = "\n".join(t for t in ds_train["text"] if len(t.strip()) > 30)
        val_text = "\n".join(t for t in ds_val["text"] if len(t.strip()) > 30)
        train_ids = tokenizer.encode(train_text[:350_000], add_special_tokens=False)
        val_ids = tokenizer.encode(val_text[:150_000], add_special_tokens=False)
        if len(train_ids) >= 20_000 and len(val_ids) >= 10_000:
            return train_ids, val_ids, "Salesforce/wikitext:wikitext-2-raw-v1"
    except Exception as exc:
        print(f"[DATA] WikiText-2 download fallback ({exc})", flush=True)

    # Fallback deterministic prose pool if HF dataset hub fails
    base_prose = (
        "Distributed systems coordinate state transitions across replicated consensus logs. "
        "Engineers benchmark memory bandwidth and cache locality before deploying cluster upgrades. "
        "Experimental telemetry verifies latency percentiles under sustained production throughput. "
    )
    train_ids = tokenizer.encode((base_prose * 250), add_special_tokens=False)
    val_ids = tokenizer.encode((base_prose * 120), add_special_tokens=False)
    return train_ids, val_ids, "deterministic-prose-fallback"


@torch.no_grad()
def evaluate_continuation_ppl(wrapper: FrozenQwenLSLCWrapper, ppl_eps, device: torch.device, mode: str) -> float:
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    amp_dtype, _ = select_amp_dtype(device)
    wrapper.lslc.eval()
    nlls = []
    for ep in ppl_eps:
        pages = ep.context_pages.unsqueeze(0).to(device)
        q_ids = ep.query_ids.unsqueeze(0).to(device)
        a_ids = ep.answer_ids.unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            if mode == "full_kv":
                prefix = embed_fn(pages.reshape(1, -1))
            elif mode == "streaming_128":
                flat = pages.reshape(1, -1)
                keep = streamingllm_keep_indices(flat.shape[1], sink_tokens=4, window_tokens=124).to(device)
                prefix = embed_fn(flat[:, keep])
            else:
                use_exact = mode == "lslc_full_exact"
                z_book, bank, toc = wrapper.lslc.encode_book(pages, embed_fn, use_exact_bank=use_exact, vocab_size=len(embed_fn.weight))
                q_emb = embed_fn(q_ids).mean(dim=1)
                asm_mode = "full" if mode in ("lslc_dense", "lslc_full_exact") else mode
                _, asm = wrapper.lslc.route_and_assemble(
                    z_book, toc, q_emb, bank if use_exact else None, embed_fn, mode=asm_mode, training_ste=False
                )
                prefix = asm.prefix_embeds

        prefix_llm = prefix.to(llm_dtype)
        turn_ids = torch.cat((q_ids, a_ids), dim=1)
        turn_emb = embed_fn(turn_ids).to(llm_dtype)
        inp = torch.cat((prefix_llm, turn_emb), dim=1)
        attn = torch.ones((1, inp.shape[1]), device=device, dtype=torch.long)
        logits = wrapper.llm(inputs_embeds=inp, attention_mask=attn).logits
        start = prefix_llm.shape[1] + q_ids.shape[1] - 1
        ans_logits = logits[:, start : start + a_ids.shape[1], :]
        loss = F.cross_entropy(ans_logits.reshape(-1, ans_logits.shape[-1]), a_ids.reshape(-1))
        nlls.append(float(loss.item()))
    return round(math.exp(min(20.0, sum(nlls) / len(nlls))), 4)


@torch.no_grad()
def evaluate_niah_suite(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    niah_ds: SyntheticExactSemanticNIAHDataset,
    device: torch.device,
    mode: str,
) -> dict[str, float]:
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    amp_dtype, _ = select_amp_dtype(device)
    wrapper.lslc.eval()
    vocab_size = len(embed_fn.weight)
    exact_em, exact_tok, sem_em, sem_tok, capture_hits, capture_total = [], [], [], [], 0, 0
    logged_samples = 0

    for ep in niah_ds:
        pages = ep.context_pages.unsqueeze(0).to(device)
        q_ids = ep.query_ids.unsqueeze(0).to(device)
        a_ids = ep.answer_ids.unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            if mode == "full_kv":
                prefix = embed_fn(pages.reshape(1, -1))
            elif mode == "streaming_128":
                flat = pages.reshape(1, -1)
                keep = streamingllm_keep_indices(flat.shape[1], sink_tokens=4, window_tokens=124).to(device)
                prefix = embed_fn(flat[:, keep])
            else:
                use_exact = mode == "lslc_full_exact"
                z_book, bank, toc = wrapper.lslc.encode_book(pages, embed_fn, use_exact_bank=use_exact, vocab_size=vocab_size)
                q_emb = embed_fn(q_ids).mean(dim=1)
                asm_mode = "full" if mode in ("lslc_dense", "lslc_full_exact") else mode
                _, asm = wrapper.lslc.route_and_assemble(
                    z_book, toc, q_emb, bank if use_exact else None, embed_fn, mode=asm_mode, training_ste=False
                )
                prefix = asm.prefix_embeds

                # Measure SalienceGate exact-needle capture rate on target page
                if use_exact and bank is not None and ep.metadata["needle_type"] == "exact":
                    tp = ep.target_page_idx
                    captured = set(bank.token_ids[0, tp].tolist())
                    ans_only_ids = tokenizer.encode(ep.metadata["answer_str"], add_special_tokens=False)
                    for tid in ans_only_ids:
                        capture_total += 1
                        if tid in captured:
                            capture_hits += 1

        prefix_llm = prefix.to(llm_dtype)
        # Teacher-forced answer token accuracy
        turn_ids = torch.cat((q_ids, a_ids), dim=1)
        turn_emb = embed_fn(turn_ids).to(llm_dtype)
        inp = torch.cat((prefix_llm, turn_emb), dim=1)
        attn = torch.ones((1, inp.shape[1]), device=device, dtype=torch.long)
        logits = wrapper.llm(inputs_embeds=inp, attention_mask=attn).logits
        start = prefix_llm.shape[1] + q_ids.shape[1] - 1
        ans_logits = logits[:, start : start + a_ids.shape[1], :]
        tok_acc = float((ans_logits.argmax(dim=-1) == a_ids).float().mean().item())

        # Real greedy generation Exact Match (substring match of target answer string)
        gen_ids = greedy_generate_from_prefix(
            wrapper.llm, prefix_llm, q_ids, max_new_tokens=48, eos_token_id=tokenizer.eos_token_id
        )
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        em = 1.0 if ep.metadata["answer_str"].lower() in gen_text.lower() else 0.0
        if logged_samples < 2 and ep.metadata["needle_type"] == "exact":
            print(f"  [SAMPLE {mode} exact] target={ep.metadata['answer_str']!r} gen={gen_text!r}", flush=True)
            logged_samples += 1

        if ep.metadata["needle_type"] == "exact":
            exact_em.append(em)
            exact_tok.append(tok_acc)
        else:
            sem_em.append(em)
            sem_tok.append(tok_acc)

    res = {
        "exact_niah_em": round(sum(exact_em) / max(1, len(exact_em)), 4),
        "exact_niah_tok_acc": round(sum(exact_tok) / max(1, len(exact_tok)), 4),
        "semantic_niah_em": round(sum(sem_em) / max(1, len(sem_em)), 4),
        "semantic_niah_tok_acc": round(sum(sem_tok) / max(1, len(sem_tok)), 4),
    }
    if capture_total > 0:
        res["salience_needle_capture_rate"] = round(capture_hits / capture_total, 4)
    return res


@torch.no_grad()
def evaluate_delta_suite(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    delta_ds: LivingBookStateOverrideDataset,
    device: torch.device,
    mode: str = "lslc",
) -> dict[str, float]:
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    amp_dtype, _ = select_amp_dtype(device)
    wrapper.lslc.eval()
    vocab_size = len(embed_fn.weight)
    ov_em, ov_tok, st_em, st_tok, gates, inv_diffs = [], [], [], [], [], []
    logged_ov = 0

    for ep in delta_ds:
        pages = ep.context_pages.unsqueeze(0).to(device)
        q_ids = ep.query_ids.unsqueeze(0).to(device)
        a_ids = ep.answer_ids.unsqueeze(0).to(device)
        if mode == "full_kv":
            parts = [pages.reshape(1, -1)]
            if ep.edit_pages:
                for _, e_ids in ep.edit_pages:
                    parts.append(e_ids.unsqueeze(0).to(device))
            prefix_llm = embed_fn(torch.cat(parts, dim=1)).to(llm_dtype)
        else:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                z_book, bank, toc0 = wrapper.lslc.encode_book(
                    pages, embed_fn, use_exact_bank=True, vocab_size=vocab_size
                )
                q_emb = embed_fn(q_ids).mean(dim=1)
                _, asm_before = wrapper.lslc.route_and_assemble(
                    z_book, toc0, q_emb, bank, embed_fn, mode="full", training_ste=False
                )

                if ep.edit_pages:
                    for p_idx, e_ids in ep.edit_pages:
                        p_t = torch.tensor([p_idx], dtype=torch.long, device=device)
                        z_book, g_p, bank = wrapper.lslc.apply_page_delta(
                            z_book, p_t, e_ids.unsqueeze(0).to(device), embed_fn, exact_bank=bank, vocab_size=vocab_size
                        )
                        gates.append(float(g_p.mean().item()))

                toc_after = wrapper.lslc.projector.project_toc(z_book)
                _, asm_after = wrapper.lslc.route_and_assemble(
                    z_book, toc_after, q_emb, bank, embed_fn, mode="full", training_ste=False
                )
                imm = asm_before.immutable_prefix_len
                diff = float((asm_after.prefix_embeds[:, :imm] - asm_before.prefix_embeds[:, :imm]).abs().max().item())
                inv_diffs.append(diff)
                prefix_llm = asm_after.prefix_embeds.to(llm_dtype)

        turn_ids = torch.cat((q_ids, a_ids), dim=1)
        turn_emb = embed_fn(turn_ids).to(llm_dtype)
        inp = torch.cat((prefix_llm, turn_emb), dim=1)
        attn = torch.ones((1, inp.shape[1]), device=device, dtype=torch.long)
        logits = wrapper.llm(inputs_embeds=inp, attention_mask=attn).logits
        start = prefix_llm.shape[1] + q_ids.shape[1] - 1
        ans_logits = logits[:, start : start + a_ids.shape[1], :]
        tok_acc = float((ans_logits.argmax(dim=-1) == a_ids).float().mean().item())

        gen_ids = greedy_generate_from_prefix(
            wrapper.llm, prefix_llm, q_ids, max_new_tokens=32, eos_token_id=tokenizer.eos_token_id
        )
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        em = 1.0 if ep.metadata["answer_str"].lower() in gen_text.lower() else 0.0
        if mode != "full_kv" and logged_ov < 3 and ep.metadata["is_overridden_query"]:
            print(f"  [SAMPLE delta override] target={ep.metadata['answer_str']!r} gen={gen_text!r}", flush=True)
            logged_ov += 1

        if ep.metadata["is_overridden_query"]:
            ov_em.append(em)
            ov_tok.append(tok_acc)
        else:
            st_em.append(em)
            st_tok.append(tok_acc)

    res = {
        "delta_edit_em": round(sum(ov_em) / max(1, len(ov_em)), 4),
        "delta_edit_tok_acc": round(sum(ov_tok) / max(1, len(ov_tok)), 4),
        "static_retention_em": round(sum(st_em) / max(1, len(st_em)), 4),
        "static_retention_tok_acc": round(sum(st_tok) / max(1, len(st_tok)), 4),
    }
    if gates:
        res["mean_gate"] = round(sum(gates) / len(gates), 5)
    if inv_diffs:
        res["max_prefix_invariance_diff"] = max(inv_diffs)
    return res


def train_stage(
    name: str,
    wrapper: FrozenQwenLSLCWrapper,
    episodes: list,
    steps: int,
    grad_accum: int,
    lr: float,
    device: torch.device,
    use_exact_bank: bool,
    distill_salience: bool,
    run_matryoshka: bool,
    use_edits: bool = False,
) -> dict[str, float]:
    amp_dtype, use_scaler = select_amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    params = [p for p in wrapper.lslc.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wrapper.cfg.weight_decay, betas=wrapper.cfg.betas)
    warmup = max(1, int(steps * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: float(s + 1) / warmup if s < warmup else max(0.1, 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))),
    )

    wrapper.lslc.train()
    optimizer.zero_grad(set_to_none=True)
    rng = random.Random(101)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    last_losses = {}

    for step in range(1, steps + 1):
        tot_val = 0.0
        for _ in range(grad_accum):
            ep = episodes[rng.randrange(len(episodes))]
            pages = ep.context_pages.unsqueeze(0).to(device)
            q_ids = ep.query_ids.unsqueeze(0).to(device)
            a_ids = ep.answer_ids.unsqueeze(0).to(device)
            t_page = torch.tensor([ep.target_page_idx], dtype=torch.long, device=device)

            edit_steps = None
            if use_edits and ep.edit_pages:
                edit_steps = [
                    (torch.tensor([p_idx], dtype=torch.long, device=device), toks.unsqueeze(0).to(device))
                    for p_idx, toks in ep.edit_pages
                ]

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = wrapper.forward_step(
                    page_token_ids=pages,
                    query_ids=q_ids,
                    answer_ids=a_ids,
                    target_page_idx=t_page,
                    edit_steps=edit_steps,
                    use_exact_bank=use_exact_bank,
                    run_matryoshka=run_matryoshka,
                    distill_salience=distill_salience,
                )
                loss_scaled = out.loss_total / grad_accum
            if use_scaler:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
            tot_val += float(out.loss_total.detach().item()) / grad_accum
            last_losses = {
                "ntp": float(out.loss_ntp.item()),
                "rec": float(out.loss_rec.item()),
                "mat": float(out.loss_mat.item()),
                "sal": float(out.loss_salience.item()),
                "route": float(out.loss_route.item()),
                "gate": out.mean_gate,
            }

        if use_scaler:
            scaler.unscale_(optimizer)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, wrapper.cfg.grad_clip).item())
        if not math.isfinite(tot_val) or not math.isfinite(gnorm):
            raise RuntimeError(f"[{name}] Non-finite loss/grad at step {step}: loss={tot_val}, grad={gnorm}")
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step == 20 or step % 50 == 0 or step == steps:
            print(
                f"[{name}] step={step}/{steps} total={tot_val:.4f} ntp={last_losses['ntp']:.4f} "
                f"rec={last_losses['rec']:.4f} sal={last_losses['sal']:.4f} route={last_losses['route']:.4f} "
                f"gate={last_losses['gate']:.4f} gnorm={gnorm:.3f}",
                flush=True,
            )

    peak_gb = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
    elapsed = time.perf_counter() - t0
    return {
        "steps": steps,
        "elapsed_sec": round(elapsed, 2),
        "peak_vram_gb": round(peak_gb, 3),
        "final_loss": round(tot_val, 4),
        **{f"final_{k}": round(v, 4) for k, v in last_losses.items()},
    }


def _cache_bytes(pkv) -> int:
    """Count exact KV-cache bytes across transformers v5 DynamicCache.layers, v4 key_cache, or legacy tuples."""
    if pkv is None:
        return 0
    if hasattr(pkv, "layers"):
        total = 0
        for layer in pkv.layers:
            for attr in ("keys", "values", "key", "value"):
                t = getattr(layer, attr, None)
                if torch.is_tensor(t):
                    total += t.numel() * t.element_size()
            if isinstance(layer, (tuple, list)):
                for t in layer:
                    if torch.is_tensor(t):
                        total += t.numel() * t.element_size()
        if total > 0:
            return total
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        return sum(t.numel() * t.element_size() for t in (*pkv.key_cache, *pkv.value_cache) if torch.is_tensor(t))
    total = 0
    for layer in pkv:
        if torch.is_tensor(layer):
            total += layer.numel() * layer.element_size()
        elif isinstance(layer, (tuple, list)):
            for t in layer:
                if torch.is_tensor(t):
                    total += t.numel() * t.element_size()
    return total


@torch.no_grad()
def profile_live_4k_32k(wrapper: FrozenQwenLSLCWrapper, device: torch.device) -> list[dict]:
    import inspect

    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    amp_dtype, _ = select_amp_dtype(device)
    wrapper.lslc.eval()
    prev_cache = getattr(wrapper.llm.config, "use_cache", False)
    wrapper.llm.config.use_cache = True
    fwd_params = inspect.signature(wrapper.llm.forward).parameters
    extra_kw = {"logits_to_keep": 1} if "logits_to_keep" in fwd_params else {}
    results = []

    try:
        for ctx_len, kd, ks in ((4096, 2, 2), (32768, 2, 4)):
            pages = ctx_len // wrapper.cfg.page_size
            ctx_ids = torch.randint(100, min(vocab_size, 50000), (1, pages, wrapper.cfg.page_size), device=device)
            q_ids = torch.randint(100, min(vocab_size, 50000), (1, 16), device=device)
            mib = 1024.0 * 1024.0

            # 1. Measure Full KV Prefill TTFT & KV cache bytes (gracefully catch OOM at 32K)
            flat_ids = torch.cat((ctx_ids.reshape(1, -1), q_ids), dim=1)
            full_kv_mib: float | str
            full_ttft_ms: float | str
            try:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                full_out = wrapper.llm(input_ids=flat_ids, use_cache=True, **extra_kw)
                torch.cuda.synchronize()
                full_ttft_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                pkv_full = full_out.past_key_values
                full_kv_bytes = _cache_bytes(pkv_full)
                full_kv_mib = round(full_kv_bytes / mib, 2)
                del full_out, pkv_full
            except torch.OutOfMemoryError:
                full_kv_bytes = 0
                full_kv_mib = "OOM"
                full_ttft_ms = "OOM"
            torch.cuda.empty_cache()

            # 2. Measure LSLC Encode + Route + Active Prefill TTFT & Net Memory
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                z_book, bank, toc = wrapper.lslc.encode_book(ctx_ids, embed_fn, use_exact_bank=True, vocab_size=vocab_size)
                q_emb = embed_fn(q_ids).mean(dim=1)
                _, asm = wrapper.lslc.route_and_assemble(
                    z_book, toc, q_emb, bank, embed_fn, k_deep=kd, k_skim=ks, mode="full", training_ste=False
                )
            turn_emb = embed_fn(q_ids).to(llm_dtype)
            inp = torch.cat((asm.prefix_embeds.to(llm_dtype), turn_emb), dim=1)
            lslc_out = wrapper.llm(inputs_embeds=inp, use_cache=True, **extra_kw)
            torch.cuda.synchronize()
            lslc_ttft_ms = (time.perf_counter() - t1) * 1000.0

            # Measure ONLY query TTFT from pre-built LSLC book (warm cache!)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                q_emb = embed_fn(q_ids).mean(dim=1)
                _, asm2 = wrapper.lslc.route_and_assemble(
                    z_book, toc, q_emb, bank, embed_fn, k_deep=kd, k_skim=ks, mode="full", training_ste=False
                )
            inp2 = torch.cat((asm2.prefix_embeds.to(llm_dtype), turn_emb), dim=1)
            _ = wrapper.llm(inputs_embeds=inp2, use_cache=True, **extra_kw)
            torch.cuda.synchronize()
            warm_ttft_ms = (time.perf_counter() - t2) * 1000.0

            pkv_lslc = lslc_out.past_key_values
            active_kv_bytes = _cache_bytes(pkv_lslc)
            book_fp16_bytes = z_book.to(torch.float16).numel() * 2
            bank_bytes = bank.storage_bytes()
            net_bytes = book_fp16_bytes + bank_bytes + active_kv_bytes

            row = {
                "context_len": ctx_len,
                "active_prefix_tokens": int(asm.prefix_embeds.shape[1]),
                "full_kv_mib": full_kv_mib,
                "lslc_book_mib": round(book_fp16_bytes / mib, 3),
                "lslc_bank_mib": round(bank_bytes / mib, 4),
                "lslc_active_kv_mib": round(active_kv_bytes / mib, 2),
                "lslc_net_mib": round(net_bytes / mib, 2),
                "measured_ratio_vs_fp16": round(full_kv_bytes / max(1, net_bytes), 2) if full_kv_bytes > 0 else "N/A",
                "full_ttft_ms": full_ttft_ms,
                "lslc_cold_ttft_ms": round(lslc_ttft_ms, 2),
                "lslc_warm_ttft_ms": round(warm_ttft_ms, 2),
                "analytical": compute_analytical_memory_stats(wrapper.cfg, ctx_len, k_deep=kd, k_skim=ks),
            }
            results.append(row)
            del lslc_out, pkv_lslc, z_book, bank
            torch.cuda.empty_cache()
            print(
                f"[PROFILE N={ctx_len}] FullKV={row['full_kv_mib']} MiB ({row['full_ttft_ms']}ms) | "
                f"LSLC Net={row['lslc_net_mib']} MiB ({row['measured_ratio_vs_fp16']}x) | "
                f"Cold={row['lslc_cold_ttft_ms']}ms | Warm={row['lslc_warm_ttft_ms']}ms",
                flush=True,
            )
    finally:
        wrapper.llm.config.use_cache = prev_cache
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"=== LSLC M2 RUN START (model={args.model}) ===", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for colab_run_m2.py")
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
    print(f"Loaded {cfg.model_name} (trainable LSLC params: {wrapper.lslc.trainable_parameter_count():,})", flush=True)

    train_pool, val_pool, corpus_src = _load_wikitext_pools(tokenizer)
    print(f"Corpus: {corpus_src} | train_toks={len(train_pool):,} | val_toks={len(val_pool):,}", flush=True)

    train_ppl = ChunkedTextPageDataset(train_pool, num_pages=4, page_size=cfg.page_size, turn_len=64)
    eval_ppl = ChunkedTextPageDataset(val_pool, num_pages=4, page_size=cfg.page_size, turn_len=64)[:24]

    train_niah = SyntheticExactSemanticNIAHDataset(
        num_episodes=120, num_pages=4, page_size=cfg.page_size, tokenizer=tokenizer, seed=42, background_token_pool=train_pool
    )
    eval_niah = SyntheticExactSemanticNIAHDataset(
        num_episodes=30, num_pages=4, page_size=cfg.page_size, tokenizer=tokenizer, seed=9999, background_token_pool=val_pool
    )
    train_delta = LivingBookStateOverrideDataset(
        num_episodes=96, num_pages=4, page_size=cfg.page_size, num_edits=3, tokenizer=tokenizer, seed=42, background_token_pool=train_pool
    )
    eval_delta = LivingBookStateOverrideDataset(
        num_episodes=24, num_pages=4, page_size=cfg.page_size, num_edits=3, tokenizer=tokenizer, seed=9999, background_token_pool=val_pool
    )

    m1_train_blend = list(train_ppl)[:80] + list(train_niah)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    if args.resume and out_path.exists():
        try:
            summary = json.loads(out_path.read_text(encoding="utf-8"))
            print(f"[RESUME] Loaded existing summary keys: {list(summary.keys())}", flush=True)
        except Exception:
            summary = {}
    summary.update({
        "gpu": props.name,
        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
        "model": cfg.model_name,
        "trainable_params": wrapper.lslc.trainable_parameter_count(),
        "corpus": corpus_src,
    })

    def _flush() -> None:
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 3. Baselines (Track 0A Full KV & Track 0C StreamingLLM)
    if "track0a_full_kv" not in summary or "track0c_streaming_128" not in summary:
        print("=== Evaluating Track 0A (Full BF16 KV) & Track 0C (StreamingLLM-128) ===", flush=True)
        t0a_ppl = evaluate_continuation_ppl(wrapper, eval_ppl, device, "full_kv")
        t0a_niah = evaluate_niah_suite(wrapper, tokenizer, eval_niah, device, "full_kv")
        t0a_delta = evaluate_delta_suite(wrapper, tokenizer, eval_delta, device, mode="full_kv")
        t0c_ppl = evaluate_continuation_ppl(wrapper, eval_ppl, device, "streaming_128")
        t0c_niah = evaluate_niah_suite(wrapper, tokenizer, eval_niah, device, "streaming_128")
        summary["track0a_full_kv"] = {"ppl": t0a_ppl, **t0a_niah, "delta_oracle": t0a_delta}
        summary["track0c_streaming_128"] = {"ppl": t0c_ppl, **t0c_niah}
        _flush()
        print(f"Track 0A Full KV: PPL={t0a_ppl} | {t0a_niah} | DeltaOracle={t0a_delta}", flush=True)
        print(f"Track 0C Stream128: PPL={t0c_ppl} | {t0c_niah}", flush=True)

    # 4. 20-step Smoke Test Gate
    if "smoke_20" not in summary:
        print("=== Running 20-Step Smoke Test Gate ===", flush=True)
        smoke_stats = train_stage(
            "SMOKE20", wrapper, m1_train_blend, steps=20, grad_accum=1, lr=args.lr, device=device,
            use_exact_bank=True, distill_salience=True, run_matryoshka=True
        )
        summary["smoke_20"] = smoke_stats
        _flush()
        if smoke_stats["peak_vram_gb"] > 18.0:
            raise RuntimeError(f"Smoke VRAM too high: {smoke_stats['peak_vram_gb']} GB")

    # 5. Track 1: Dense Only (Fresh LSLC init)
    if "track1_dense_only" not in summary:
        print("=== Track 1: Training LSLC Dense Only (No Exact Bank) ===", flush=True)
        wrapper_t1 = FrozenQwenLSLCWrapper(cfg, llm)
        wrapper_t1.lslc.to(device)
        t1_train = train_stage(
            "TRACK1", wrapper_t1, m1_train_blend, steps=args.steps_track1, grad_accum=args.grad_accum, lr=args.lr, device=device,
            use_exact_bank=False, distill_salience=False, run_matryoshka=True
        )
        t1_ppl = evaluate_continuation_ppl(wrapper_t1, eval_ppl, device, "lslc_dense")
        t1_niah = evaluate_niah_suite(wrapper_t1, tokenizer, eval_niah, device, "lslc_dense")
        summary["track1_dense_only"] = {"train": t1_train, "ppl": t1_ppl, **t1_niah}
        _flush()
        print(f"Track 1 Eval: PPL={t1_ppl} | {t1_niah}", flush=True)
        del wrapper_t1
        torch.cuda.empty_cache()

    # 6. Track 2: Full LSLC + Surprisal-Distilled SalienceGate + Sparse Exact Bank (Fresh LSLC init)
    print("=== Track 2: Training LSLC + Sparse Exact Bank + Surprisal Distillation ===", flush=True)
    wrapper_t2 = FrozenQwenLSLCWrapper(cfg, llm)
    wrapper_t2.lslc.to(device)
    t2_train = train_stage(
        "TRACK2", wrapper_t2, m1_train_blend, steps=args.steps_track2, grad_accum=args.grad_accum, lr=args.lr, device=device,
        use_exact_bank=True, distill_salience=True, run_matryoshka=True
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
    print(f"Matryoshka Zoom: {matryoshka_zoom}", flush=True)

    # 7. Track 4 Probe: Sticky-Note DeltaUpdater on Living-Book State-Override (warm-started from Track 2)
    print("=== Track 4 Probe: Training DeltaUpdater on Living-Book State-Override ===", flush=True)
    # Freeze structural encoder/router/l0/l1 modules so Static l=0 retention stays 100% locked
    # while DeltaUpdater and dynamic l=2 projector learn in-place state override.
    frozen_in_t4 = []
    for name_p, p in wrapper_t2.lslc.named_parameters():
        is_dynamic_t4 = (
            name_p.startswith("delta_updater.")
            or name_p.startswith("router.")
        )
        if not is_dynamic_t4 and p.requires_grad:
            p.requires_grad_(False)
            frozen_in_t4.append(p)

    t4_train = train_stage(
        "TRACK4", wrapper_t2, list(train_delta), steps=args.steps_track4, grad_accum=args.grad_accum, lr=args.lr,
        device=device, use_exact_bank=True, distill_salience=False, run_matryoshka=False, use_edits=True
    )
    for p in frozen_in_t4:
        p.requires_grad_(True)
    t4_delta_eval = evaluate_delta_suite(wrapper_t2, tokenizer, eval_delta, device, mode="lslc")
    summary["track4_delta_probe"] = {"train": t4_train, **t4_delta_eval}
    _flush()
    print(f"Track 4 Delta Eval: {t4_delta_eval}", flush=True)

    # 8. Live CUDA 4K & 32K Profiler
    print("=== Live CUDA 4K & 32K VRAM / TTFT Profiler ===", flush=True)
    cuda_profile = profile_live_4k_32k(wrapper_t2, device)
    summary["live_cuda_profile"] = cuda_profile
    _flush()
    print(f"Saved full M2 summary -> {out_path}", flush=True)
    print("=== M2 RUN COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
