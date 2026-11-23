"""Brutal Edge-Case & Agentic Coding Stress-Test Suite + Qwen3-8B (4-bit NF4) + 128K/256K Ultra-Long Book Profiler."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
import unittest
from typing import Any
import torch
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
    LSLCEpisode,
    SyntheticExactSemanticNIAHDataset,
    compute_analytical_memory_stats,
    greedy_generate_from_prefix,
    load_frozen_qwen3,
    select_amp_dtype,
)
from scripts.colab_run_m2 import (
    _cache_bytes,
    _load_wikitext_pools,
    evaluate_delta_suite,
    evaluate_niah_suite,
    train_stage,
)


# --- 1. Agentic Multi-File Codebase Haystack & Exact Bug/Constant Retrieval ---

AGENTIC_CODE_FILES = (
    """def verify_session_token(headers: dict, clock_skew_sec: int = 30) -> bool:
    raw_sig = headers.get("x_hmac_digest", "")
    if not raw_sig or len(raw_sig) < 16:
        raise PermissionError("missing_hmac_digest")
    return True
""",
    """class ConnectionPoolManager:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.active_sockets = 0
        self.idle_evict_sec = 180
    def acquire_lease(self) -> int:
        self.active_sockets += 1
        return self.active_sockets
""",
    """async def dispatch_webhook_batch(payloads: list[dict]) -> dict:
    delivered = 0
    for item in payloads:
        if item.get("retry_count", 0) <= 3:
            delivered += 1
    return {"status": "dispatched", "count": delivered}
""",
    """def reconcile_ledger_shard(shard_id: int, checksum: str) -> str:
    if not checksum.startswith("sha256:"):
        raise ValueError("invalid_shard_checksum")
    return f"verified:{shard_id}"
""",
)

AGENTIC_CODE_TARGETS = (
    ("MAX_RETRY_BACKOFF_MS", "4500-8200", "What is the exact value of MAX_RETRY_BACKOFF_MS in the codebase?"),
    ("AUTH_NONCE_HEADER", "x_auth_nonce", "What is the exact header key assigned to AUTH_NONCE_HEADER?"),
    ("FATAL_ERR_CODE", "7409-9912", "Output only the exact code assigned to FATAL_ERR_CODE."),
    ("FALLBACK_RPC_PORT", "8443-9443", "Output only the exact port range assigned to FALLBACK_RPC_PORT."),
    ("CIRCUIT_BREAKER_MODE", "half_open_probe", "What is the exact mode string assigned to CIRCUIT_BREAKER_MODE?"),
)


def build_agentic_coding_episodes(tokenizer, num_episodes: int = 30, num_pages: int = 8, page_size: int = 256, seed: int = 7001) -> list[LSLCEpisode]:
    code_blob = "\n".join(AGENTIC_CODE_FILES * 40)
    code_pool = tokenizer.encode(code_blob, add_special_tokens=False)
    episodes: list[LSLCEpisode] = []
    depths = (0.0, 0.25, 0.50, 0.75, 1.0)

    for i in range(num_episodes):
        rng = random.Random(seed + i * 313)
        depth = depths[i % len(depths)]
        target_page = min(num_pages - 1, int(round(depth * (num_pages - 1))))
        var_name, val_str, q_str = AGENTIC_CODE_TARGETS[i % len(AGENTIC_CODE_TARGETS)]
        if i % 2 == 0 and "-" in val_str and val_str[0].isdigit():
            val_str = f"{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
        needle_line = f"\n# [CONFIG] {var_name} = {val_str} .\n"
        needle_toks = tokenizer.encode(needle_line, add_special_tokens=False)

        pages = []
        for p in range(num_pages):
            start = rng.randint(0, max(1, len(code_pool) - page_size - 1))
            ptoks = list(code_pool[start : start + page_size])
            if p == target_page:
                ins = page_size // 3
                end = min(page_size, ins + len(needle_toks))
                ptoks[ins:end] = needle_toks[: end - ins]
            pages.append(ptoks)

        q_ids = tokenizer.encode(f"\nQuestion: {q_str}\nAnswer:", add_special_tokens=False)
        a_ids = tokenizer.encode(f" {val_str}", add_special_tokens=False)
        episodes.append(
            LSLCEpisode(
                context_pages=torch.tensor(pages, dtype=torch.long),
                query_ids=torch.tensor(q_ids, dtype=torch.long),
                answer_ids=torch.tensor(a_ids, dtype=torch.long),
                target_page_idx=target_page,
                metadata={"needle_type": "exact", "answer_str": val_str, "var_name": var_name, "depth": depth},
            )
        )
    return episodes


# --- 2. Deep k=5 Sequential Multi-Hop Symbolic (Non-Numeric) + Numeric State Override ---

SYMBOLIC_STATES = (
    "canary-staging",
    "shadow-mirror",
    "blue-green-prod",
    "quorum-degraded",
    "rollback-lockdown",
    "us-west-oregon",
    "eu-central-frankfurt",
    "ap-southeast-singapore",
)


def build_k5_symbolic_override_episodes(
    tokenizer,
    bg_pool: list[int],
    num_episodes: int = 30,
    num_pages: int = 4,
    page_size: int = 256,
    num_edits: int = 5,
    seed: int = 8001,
) -> list[LSLCEpisode]:
    episodes: list[LSLCEpisode] = []
    static_toks = tokenizer.encode("\nSTATIC POLICY : security audit requires dual signoff .\n", add_special_tokens=False)

    for i in range(num_episodes):
        rng = random.Random(seed + i * 509)
        pages = []
        for p in range(num_pages):
            start = rng.randint(0, max(1, len(bg_pool) - page_size - 1))
            ptoks = list(bg_pool[start : start + page_size])
            if p == 0:
                ptoks[: len(static_toks)] = static_toks[:page_size]
            pages.append(ptoks)

        edit_page = 1
        init_toks = tokenizer.encode("\nDEPLOYMENT STATE INITIAL : cold-standby .\n", add_special_tokens=False)
        pages[edit_page][: len(init_toks)] = init_toks[:page_size]

        chosen_states = rng.sample(SYMBOLIC_STATES, k=min(num_edits, len(SYMBOLIC_STATES)))
        edits: list[tuple[int, torch.Tensor]] = []
        latest_state = "cold-standby"
        for latest_state in chosen_states:
            e_text = f"REVISION OVERRIDE : deployment state updated to {latest_state} ."
            e_ids = torch.tensor(tokenizer.encode(e_text, add_special_tokens=False), dtype=torch.long)
            edits.append((edit_page, e_ids))

        q_text = "\nQuestion: What is the current updated deployment state? Answer:"
        a_text = f" {latest_state}"
        episodes.append(
            LSLCEpisode(
                context_pages=torch.tensor(pages, dtype=torch.long),
                query_ids=torch.tensor(tokenizer.encode(q_text, add_special_tokens=False), dtype=torch.long),
                answer_ids=torch.tensor(tokenizer.encode(a_text, add_special_tokens=False), dtype=torch.long),
                target_page_idx=edit_page,
                metadata={"is_overridden_query": True, "answer_str": latest_state, "num_edits": num_edits},
                edit_pages=edits,
            )
        )
    return episodes


# --- 3. Adversarial Decoy / Near-Miss Multi-Key Confusion Suite ---

DECOY_ENV_KEYS = ("PROD-KEY", "STAGING-KEY", "CANARY-KEY", "SANDBOX-KEY")


def build_adversarial_decoy_episodes(
    tokenizer,
    bg_pool: list[int],
    num_episodes: int = 20,
    num_pages: int = 4,
    page_size: int = 256,
    seed: int = 9001,
) -> list[LSLCEpisode]:
    """Every page p in {0..3} contains a DIFFERENT decoy key-value pair; query asks for one specific page's key."""
    episodes: list[LSLCEpisode] = []
    for i in range(num_episodes):
        rng = random.Random(seed + i * 701)
        target_page = i % num_pages
        pages = []
        target_code = ""
        target_key = DECOY_ENV_KEYS[target_page]

        for p in range(num_pages):
            start = rng.randint(0, max(1, len(bg_pool) - page_size - 1))
            ptoks = list(bg_pool[start : start + page_size])
            k_name = DECOY_ENV_KEYS[p]
            c_val = f"{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
            if p == target_page:
                target_code = c_val
            n_toks = tokenizer.encode(f"\n[RECORD] {k_name} = {c_val} .\n", add_special_tokens=False)
            ins = page_size // 3
            end = min(page_size, ins + len(n_toks))
            ptoks[ins:end] = n_toks[: end - ins]
            pages.append(ptoks)

        q_ids = tokenizer.encode(f"\nQuestion: Output only the exact code for {target_key}.\nAnswer:", add_special_tokens=False)
        a_ids = tokenizer.encode(f" {target_code}", add_special_tokens=False)
        episodes.append(
            LSLCEpisode(
                context_pages=torch.tensor(pages, dtype=torch.long),
                query_ids=torch.tensor(q_ids, dtype=torch.long),
                answer_ids=torch.tensor(a_ids, dtype=torch.long),
                target_page_idx=target_page,
                metadata={"needle_type": "exact", "answer_str": target_code, "target_key": target_key},
            )
        )
    return episodes


# --- 4. Edge Case 5: Context-Grounded Multi-Step Reasoning & Intrinsic IQ Retention ---

CONTEXT_REASONING_PROBLEMS = (
    (
        "\n[SPEC] Service Alpha uses a base timeout of 20 seconds plus an extra 15 seconds for TLS handshake .\n",
        "Based on the spec, what is the total timeout in seconds (20 + 15)? Output the number.",
        "35",
    ),
    (
        "\n[SPEC] A cluster runs 3 database shards, and each shard holds 20 gigabytes of index data .\n",
        "Based on the spec, how many total gigabytes of index data do all 3 shards hold together (3 * 20)? Output the number.",
        "60",
    ),
    (
        "\n[RULE] If error_rate exceeds 5 percent, trigger mode ROLLBACK; otherwise keep mode ACTIVE .\n",
        "If the measured error_rate is 9 percent, which mode must be triggered according to the rule?",
        "ROLLBACK",
    ),
    (
        "\n[RULE] Requests from tier_guest are routed to queue_slow, while tier_enterprise routes to queue_fast .\n",
        "According to the rule, which queue receives requests from tier_enterprise?",
        "queue_fast",
    ),
    (
        "\n[LOGIC] Module Auth depends on Module Crypto, and Module Crypto depends on Hardware HSM .\n",
        "By transitive dependency, which hardware component does Module Auth ultimately depend on?",
        "HSM",
    ),
    (
        "\n[SPEC] Initial worker pool starts with 50 threads, and autoscaling removes 15 threads during off-peak hours .\n",
        "How many threads remain during off-peak hours (50 - 15)? Output the number.",
        "35",
    ),
)

INTRINSIC_IQ_BENCHMARK = (
    ("Question: If a server processes 120 requests per minute, how many requests does it process in 3 minutes?\nAnswer:", "360"),
    ("Question: In Python, what is the result of evaluating `len([10, 20, 30, 40]) * 2`?\nAnswer:", "8"),
    ("Question: All containers in Pod-X are stateless. Service-Y runs inside Pod-X. Is Service-Y stateless? Answer Yes or No.\nAnswer:", "yes"),
    ("Question: A cache has 100 MB capacity. After storing three 25 MB tensors, how many MB of free space remain?\nAnswer:", "25"),
    ("Question: What is the binary value of decimal 7?\nAnswer:", "111"),
    ("Question: If `x = 12` and `y = 5`, what is `x % y` in Python?\nAnswer:", "2"),
)


def build_context_reasoning_episodes(
    tokenizer, bg_pool: list[int], num_episodes: int = 18, num_pages: int = 4, page_size: int = 256, seed: int = 6001
) -> list[LSLCEpisode]:
    episodes: list[LSLCEpisode] = []
    depths = (0.0, 0.33, 0.67, 1.0)
    for i in range(num_episodes):
        rng = random.Random(seed + i * 401)
        depth = depths[i % len(depths)]
        target_page = min(num_pages - 1, int(round(depth * (num_pages - 1))))
        fact_line, q_str, ans_str = CONTEXT_REASONING_PROBLEMS[i % len(CONTEXT_REASONING_PROBLEMS)]
        fact_toks = tokenizer.encode(fact_line, add_special_tokens=False)

        pages = []
        for p in range(num_pages):
            start = rng.randint(0, max(1, len(bg_pool) - page_size - 1))
            ptoks = list(bg_pool[start : start + page_size])
            if p == target_page:
                ins = page_size // 4
                end = min(page_size, ins + len(fact_toks))
                ptoks[ins:end] = fact_toks[: end - ins]
            pages.append(ptoks)

        q_ids = tokenizer.encode(f"\nQuestion: {q_str}\nAnswer:", add_special_tokens=False)
        a_ids = tokenizer.encode(f" {ans_str}", add_special_tokens=False)
        episodes.append(
            LSLCEpisode(
                context_pages=torch.tensor(pages, dtype=torch.long),
                query_ids=torch.tensor(q_ids, dtype=torch.long),
                answer_ids=torch.tensor(a_ids, dtype=torch.long),
                target_page_idx=target_page,
                metadata={"needle_type": "semantic", "answer_str": ans_str},
            )
        )
    return episodes


@torch.no_grad()
def evaluate_intelligence_retention(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    reasoning_eps: list[LSLCEpisode],
    device: torch.device,
) -> dict[str, float]:
    """Compare Full-KV vs. LSLC on context-grounded multi-step reasoning AND raw-LLM vs. LSLC-prefixed intrinsic IQ."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    amp_dtype, _ = select_amp_dtype(device)
    wrapper.lslc.eval()

    # A. Context-Grounded Reasoning (Arithmetic, If/Else Code Logic, Transitive Deduction over Compressed Book)
    full_kv_hits, lslc_hits = 0, 0
    for ep in reasoning_eps:
        pages = ep.context_pages.unsqueeze(0).to(device)
        q_ids = ep.query_ids.unsqueeze(0).to(device)
        target = ep.metadata["answer_str"].lower()

        # Full KV baseline
        pref_full = embed_fn(pages.reshape(1, -1)).to(llm_dtype)
        gen_full = greedy_generate_from_prefix(wrapper.llm, pref_full, q_ids, max_new_tokens=36, eos_token_id=tokenizer.eos_token_id)
        txt_full = tokenizer.decode(gen_full[0], skip_special_tokens=True).lower()
        if target in txt_full:
            full_kv_hits += 1

        # LSLC compressed book
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            z_b, bank, toc = wrapper.lslc.encode_book(pages, embed_fn, use_exact_bank=True, vocab_size=len(embed_fn.weight))
            _, asm = wrapper.lslc.route_and_assemble(
                z_b, toc, embed_fn(q_ids).mean(dim=1), bank, embed_fn, mode="full", training_ste=False
            )
        gen_lslc = greedy_generate_from_prefix(
            wrapper.llm, asm.prefix_embeds.to(llm_dtype), q_ids, max_new_tokens=36, eos_token_id=tokenizer.eos_token_id
        )
        txt_lslc = tokenizer.decode(gen_lslc[0], skip_special_tokens=True).lower()
        if target in txt_lslc:
            lslc_hits += 1

    # B. Intrinsic General IQ / Math / Coding Reasoning (Raw LLM vs. LLM holding a 4-page LSLC Book in Memory)
    raw_iq_hits, lslc_iq_hits = 0, 0
    bg_pages = reasoning_eps[0].context_pages.unsqueeze(0).to(device)
    for q_prompt, expected in INTRINSIC_IQ_BENCHMARK:
        q_ids = torch.tensor([tokenizer.encode(q_prompt, add_special_tokens=False)], dtype=torch.long, device=device)
        empty_pref = torch.zeros((1, 0, wrapper.cfg.d_llm), dtype=llm_dtype, device=device)
        gen_raw = greedy_generate_from_prefix(wrapper.llm, empty_pref, q_ids, max_new_tokens=32, eos_token_id=tokenizer.eos_token_id)
        if expected.lower() in tokenizer.decode(gen_raw[0], skip_special_tokens=True).lower():
            raw_iq_hits += 1

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            z_b, bank, toc = wrapper.lslc.encode_book(bg_pages, embed_fn, use_exact_bank=True, vocab_size=len(embed_fn.weight))
            _, asm = wrapper.lslc.route_and_assemble(
                z_b, toc, embed_fn(q_ids).mean(dim=1), bank, embed_fn, mode="full", training_ste=False
            )
        gen_pref = greedy_generate_from_prefix(
            wrapper.llm, asm.prefix_embeds.to(llm_dtype), q_ids, max_new_tokens=32, eos_token_id=tokenizer.eos_token_id
        )
        if expected.lower() in tokenizer.decode(gen_pref[0], skip_special_tokens=True).lower():
            lslc_iq_hits += 1

    ctx_full_acc = round(full_kv_hits / max(1, len(reasoning_eps)), 4)
    ctx_lslc_acc = round(lslc_hits / max(1, len(reasoning_eps)), 4)
    raw_iq_acc = round(raw_iq_hits / len(INTRINSIC_IQ_BENCHMARK), 4)
    lslc_iq_acc = round(lslc_iq_hits / len(INTRINSIC_IQ_BENCHMARK), 4)
    return {
        "context_reasoning_full_kv_acc": ctx_full_acc,
        "context_reasoning_lslc_acc": ctx_lslc_acc,
        "context_reasoning_retention_ratio": round(ctx_lslc_acc / max(1e-4, ctx_full_acc), 4),
        "intrinsic_iq_raw_llm_acc": raw_iq_acc,
        "intrinsic_iq_with_lslc_book_acc": lslc_iq_acc,
        "intrinsic_iq_retention_ratio": round(lslc_iq_acc / max(1e-4, raw_iq_acc), 4),
    }


# --- 5. Ultra-Long 128K (P=512) & 256K (P=1,024) Live CUDA Book Profiler ---

@torch.no_grad()
def profile_ultra_long_128k_256k(wrapper: FrozenQwenLSLCWrapper, device: torch.device) -> list[dict[str, Any]]:
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
    mib = 1024.0 * 1024.0
    results = []

    try:
        for ctx_len, kd, ks in ((131072, 2, 4), (262144, 2, 4)):
            num_pages = ctx_len // wrapper.cfg.page_size
            q_ids = torch.randint(100, min(vocab_size, 50000), (1, 16), device=device)

            # Encode book in page chunks of 64 pages so encoding peak memory is O(1)
            torch.cuda.synchronize()
            t_enc0 = time.perf_counter()
            z_chunks, tok_id_chunks, h_chunks, w_chunks, toc_chunks = [], [], [], [], []
            chunk_p = 64
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                for p_start in range(0, num_pages, chunk_p):
                    p_cur = min(chunk_p, num_pages - p_start)
                    c_ids = torch.randint(100, min(vocab_size, 50000), (1, p_cur, wrapper.cfg.page_size), device=device)
                    z_c, bank_c, toc_c = wrapper.lslc.encode_book(c_ids, embed_fn, use_exact_bank=True, vocab_size=vocab_size)
                    z_chunks.append(z_c.to(torch.float16))
                    tok_id_chunks.append(bank_c.token_ids)
                    h_chunks.append(bank_c.h_coords)
                    w_chunks.append(bank_c.w_coords)
                    toc_chunks.append(toc_c)

                z_book = torch.cat(z_chunks, dim=1)
                toc_all = torch.cat(toc_chunks, dim=1)
                from lslc.encoder import SparseExactBank

                page_idx = (
                    torch.arange(num_pages, device=device, dtype=torch.int16)
                    .view(1, num_pages, 1)
                    .expand(1, num_pages, wrapper.cfg.k_exact)
                    .contiguous()
                )
                bank = SparseExactBank(
                    token_ids=torch.cat(tok_id_chunks, dim=1),
                    page_ids=page_idx,
                    h_coords=torch.cat(h_chunks, dim=1),
                    w_coords=torch.cat(w_chunks, dim=1),
                )
            torch.cuda.synchronize()
            cold_encode_ms = (time.perf_counter() - t_enc0) * 1000.0

            # Measure Warm Query Route + Assemble + LLM Prefill TTFT across 128K / 256K book!
            torch.cuda.synchronize()
            t_warm0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                q_emb = embed_fn(q_ids).mean(dim=1)
                _, asm = wrapper.lslc.route_and_assemble(
                    z_book, toc_all, q_emb, bank, embed_fn, k_deep=kd, k_skim=ks, mode="full", training_ste=False
                )
            turn_emb = embed_fn(q_ids).to(llm_dtype)
            inp = torch.cat((asm.prefix_embeds.to(llm_dtype), turn_emb), dim=1)
            out = wrapper.llm(inputs_embeds=inp, use_cache=True, **extra_kw)
            torch.cuda.synchronize()
            warm_ttft_ms = (time.perf_counter() - t_warm0) * 1000.0

            active_kv_bytes = _cache_bytes(out.past_key_values)
            book_fp16_bytes = z_book.numel() * 2
            bank_bytes = bank.storage_bytes()
            net_bytes = book_fp16_bytes + bank_bytes + active_kv_bytes
            analytical = compute_analytical_memory_stats(wrapper.cfg, ctx_len, k_deep=kd, k_skim=ks)

            row = {
                "context_len": ctx_len,
                "pages": num_pages,
                "active_prefix_tokens": int(asm.prefix_embeds.shape[1]),
                "full_fp16_kv_analytical_mib": round(analytical["full_fp16_kv_mib"], 2),
                "full_fp16_status_on_single_l4": "OOM (>14-36 GiB KV)",
                "lslc_book_mib": round(book_fp16_bytes / mib, 3),
                "lslc_bank_mib": round(bank_bytes / mib, 4),
                "lslc_active_kv_mib": round(active_kv_bytes / mib, 2),
                "lslc_net_mib": round(net_bytes / mib, 2),
                "compression_ratio_vs_fp16": round(analytical["full_fp16_kv_mib"] / (net_bytes / mib), 2),
                "lslc_chunked_encode_ms": round(cold_encode_ms, 2),
                "lslc_warm_ttft_ms": round(warm_ttft_ms, 2),
            }
            results.append(row)
            print(
                f"[ULTRA-LONG N={ctx_len} ({num_pages}p)] FullKV={row['full_fp16_kv_analytical_mib']} MiB (OOM) | "
                f"LSLC Net={row['lslc_net_mib']} MiB ({row['compression_ratio_vs_fp16']}x) | "
                f"ActiveToks={row['active_prefix_tokens']} | WarmTTFT={row['lslc_warm_ttft_ms']}ms",
                flush=True,
            )
            del out, z_book, bank, toc_all
            torch.cuda.empty_cache()
    finally:
        wrapper.llm.config.use_cache = prev_cache
    return results


def run_stress_on_backbone(model_name: str, load_in_4bit: bool, steps: int, lr: float, device: torch.device) -> dict[str, Any]:
    print(f"\n================ STRESS-TESTING BACKBONE: {model_name} (4bit={load_in_4bit}) ================", flush=True)
    cfg = LSLCConfig.from_model_name(model_name, lr=lr)
    llm, tokenizer = load_frozen_qwen3(cfg, load_in_4bit=load_in_4bit, device=device)
    wrapper = FrozenQwenLSLCWrapper(cfg, llm)
    wrapper.lslc.to(device)
    print(
        f"Loaded {model_name} | d_llm={cfg.d_llm} | layers={cfg.num_llm_layers} | "
        f"KV={cfg.kv_bytes_per_token_fp16 // 1024} KiB/tok | trainable LSLC={wrapper.lslc.trainable_parameter_count():,}",
        flush=True,
    )

    train_pool, val_pool, _ = _load_wikitext_pools(tokenizer)

    # Build the 3 brutal edge-case datasets (train & held-out eval)
    train_code = build_agentic_coding_episodes(tokenizer, num_episodes=60, num_pages=8, seed=7001)
    eval_code = build_agentic_coding_episodes(tokenizer, num_episodes=20, num_pages=8, seed=7999)

    train_decoy = build_adversarial_decoy_episodes(tokenizer, train_pool, num_episodes=60, num_pages=4, seed=9001)
    eval_decoy = build_adversarial_decoy_episodes(tokenizer, val_pool, num_episodes=20, num_pages=4, seed=9999)

    train_k5_sym = build_k5_symbolic_override_episodes(tokenizer, train_pool, num_episodes=60, num_pages=4, num_edits=5, seed=8001)
    eval_k5_sym = build_k5_symbolic_override_episodes(tokenizer, val_pool, num_episodes=20, num_pages=4, num_edits=5, seed=8999)

    train_reason = build_context_reasoning_episodes(tokenizer, train_pool, num_episodes=36, num_pages=4, seed=6001)
    eval_reason = build_context_reasoning_episodes(tokenizer, val_pool, num_episodes=18, num_pages=4, seed=6999)

    # 1. Train LSLC on mixed Agentic Code (8-page) + Adversarial Decoy (4-page) + Context Reasoning (4-page)
    blend = train_code + train_decoy + train_reason
    t_stats = train_stage(
        f"STRESS_{model_name.split('/')[-1]}",
        wrapper,
        blend,
        steps=steps,
        grad_accum=2,
        lr=lr,
        device=device,
        use_exact_bank=True,
        distill_salience=True,
        run_matryoshka=True,
    )

    print("--- Evaluating Edge Case 1: Agentic Multi-File Codebase (8 Pages / 2,048 Tokens) ---", flush=True)
    code_res = evaluate_niah_suite(wrapper, tokenizer, eval_code, device, "lslc_full_exact")
    print(f"Agentic Codebase 8-Page Eval: {code_res}", flush=True)

    print("--- Evaluating Edge Case 2: Adversarial Near-Miss Decoy Keys (4 Co-Existing Keys/Book) ---", flush=True)
    decoy_res = evaluate_niah_suite(wrapper, tokenizer, eval_decoy, device, "lslc_full_exact")
    print(f"Adversarial Decoy Eval: {decoy_res}", flush=True)

    print("--- Evaluating Edge Case 5: Context-Grounded Multi-Step Reasoning & Intrinsic IQ Retention ---", flush=True)
    iq_res = evaluate_intelligence_retention(wrapper, tokenizer, eval_reason, device)
    print(f"Intelligence & Reasoning Retention Eval: {iq_res}", flush=True)

    # 2. Train & Evaluate Edge Case 3: Deep k=5 Sequential Symbolic State Overrides
    print("--- Evaluating Edge Case 3: Deep k=5 Sequential Symbolic State Overrides ---", flush=True)
    frozen_in_t4 = []
    for name_p, p in wrapper.lslc.named_parameters():
        is_dynamic_t4 = name_p.startswith("delta_updater.") or name_p.startswith("router.")
        if not is_dynamic_t4 and p.requires_grad:
            p.requires_grad_(False)
            frozen_in_t4.append(p)

    k5_train = train_stage(
        f"K5_SYM_{model_name.split('/')[-1]}",
        wrapper,
        train_k5_sym,
        steps=max(100, steps // 2),
        grad_accum=2,
        lr=lr,
        device=device,
        use_exact_bank=True,
        distill_salience=False,
        run_matryoshka=False,
        use_edits=True,
    )
    for p in frozen_in_t4:
        p.requires_grad_(True)
    k5_res = evaluate_delta_suite(wrapper, tokenizer, eval_k5_sym, device, mode="lslc")
    print(f"Deep k=5 Symbolic Override Eval: {k5_res}", flush=True)

    # 3. Edge Case 4: Ultra-Long 128K (512 pages) & 256K (1,024 pages) Book Profiler
    print("--- Evaluating Edge Case 4: Ultra-Long 128K & 256K Context Book Profiler ---", flush=True)
    ultra_profile = profile_ultra_long_128k_256k(wrapper, device)

    res = {
        "model": model_name,
        "load_in_4bit": load_in_4bit,
        "d_llm": cfg.d_llm,
        "num_llm_layers": cfg.num_llm_layers,
        "kv_kib_per_token_fp16": cfg.kv_bytes_per_token_fp16 / 1024.0,
        "trainable_lslc_params": wrapper.lslc.trainable_parameter_count(),
        "train_stats": t_stats,
        "edge_case_1_agentic_codebase_8p": code_res,
        "edge_case_2_adversarial_decoy_keys": decoy_res,
        "edge_case_3_k5_symbolic_override": {"train": k5_train, **k5_res},
        "edge_case_4_ultra_long_128k_256k": ultra_profile,
        "edge_case_5_intelligence_and_reasoning_retention": iq_res,
    }
    del wrapper, llm
    torch.cuda.empty_cache()
    return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", type=str, default="/content/lslc_results/hard_stress_summary.json")
    args, _ = parser.parse_known_args(argv)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"=== LSLC BRUTAL STRESS-TEST START on {props.name} ({props.total_memory / (1024**3):.2f} GiB) ===", flush=True)

    suite = unittest.defaultTestLoader.discover(str(REPO_ROOT / "tests"))
    if not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful():
        raise RuntimeError("Unit tests failed!")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "gpu": props.name,
        "vram_gb": round(props.total_memory / (1024 ** 3), 2),
        "runs": {},
    }

    # 1. Run on Qwen/Qwen3-1.7B (BF16)
    summary["runs"]["Qwen3-1.7B"] = run_stress_on_backbone("Qwen/Qwen3-1.7B", load_in_4bit=False, steps=220, lr=3e-4, device=device)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 2. Run on Larger Scale Qwen/Qwen3-8B (native BF16 on A100 40GB, or 4-bit NF4 on L4 22GB)
    is_a100 = (props.total_memory / (1024 ** 3)) >= 35.0
    try:
        mode_tag = "Qwen3-8B-BF16" if is_a100 else "Qwen3-8B-NF4"
        summary["runs"][mode_tag] = run_stress_on_backbone(
            "Qwen/Qwen3-8B", load_in_4bit=(not is_a100), steps=200, lr=3e-4, device=device
        )
    except Exception as exc:
        print(f"[WARN] Qwen3-8B fallback to Qwen3-4B BF16 ({exc})", flush=True)
        torch.cuda.empty_cache()
        summary["runs"]["Qwen3-4B-BF16"] = run_stress_on_backbone(
            "Qwen/Qwen3-4B", load_in_4bit=False, steps=200, lr=3e-4, device=device
        )
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved full hard stress summary -> {out_path}", flush=True)
    print("=== HARD STRESS SUITE COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
