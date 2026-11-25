"""LSLC v3.0.0 Phase 1 Training, Identifier Funnel Tracer, Hold-out Eval, Oracle A/B/C, and Break-Even Script.

Supports both:
  1) Local CPU smoke verification (`--mock-llm`) using TinyMockCausalLM (0 CU)
  2) Full or Stage-1-sanity Colab L4 BF16 runs (`--model Qwen/Qwen3-1.7B`)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
import sys
import time
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
for p_str in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from lslc import (  # noqa: E402
    AdversarialLivingBookDataset,
    FrozenQwenLSLCWrapper,
    LSLCConfig,
    QueryToTOCRouter,
    RouterOutput,
    SalienceGate,
    SparseExactBank,
    SyntheticExactSemanticNIAHDataset,
    TinyMockCausalLM,
    compute_analytical_memory_stats,
    extract_contextual_edit_embeds,
    greedy_generate_from_prefix,
    inject_lora,
    load_frozen_qwen3,
    lora_parameters,
    lora_state_dict,
    merge_lora,
    select_amp_dtype,
    set_lora_enabled,
    unmerge_lora,
)
from colab_run_m2 import _load_wikitext_pools  # noqa: E402
from colab_run_rebuttal_v2 import BM25PageRetriever  # noqa: E402
from colab_run_v2_complete import greedy_generate_concise  # noqa: E402


def _norm_ans_v3(text: str) -> str:
    s = re.sub(r"\s+", " ", text.strip().lower())
    return s.strip(" .,:;\"'`()[]\u201c\u201d\u2018\u2019")


def _extract_canonical_v3(gen_text: str) -> set[str]:
    """Generic first-clause & preamble-stripping extractor (zero closed-set value tables)."""
    cands: set[str] = set()
    lines = [ln.strip() for ln in gen_text.splitlines() if ln.strip()]
    if not lines:
        return cands
    first_line = lines[0]
    first_clause = re.split(r"[.;\n]", first_line)[0].strip()
    for seg in (first_line, first_clause):
        n = _norm_ans_v3(seg)
        if n:
            cands.add(n)

    low = first_clause.lower()
    for sep in ("the answer is ", "answer is ", "answer:", "it is ", " is ", " = ", ": "):
        if sep in low:
            rhs = first_clause[low.rfind(sep) + len(sep) :].strip()
            nr = _norm_ans_v3(rhs)
            if nr:
                cands.add(nr)
                words = nr.split()
                if words:
                    cands.add(words[0])
                if len(words) >= 2:
                    cands.add(" ".join(words[:2]))
    words0 = _norm_ans_v3(first_clause).split()
    if words0:
        cands.add(words0[0])
    if len(words0) >= 2:
        cands.add(" ".join(words0[:2]))
    cands.discard("")
    return cands


def compute_answer_metrics_v3(
    ans_logits: torch.Tensor,
    a_ids: torch.Tensor,
    gen_text: str,
    answer_str: str,
    tokenizer,
) -> tuple[float, float, float, float, float]:
    """Fair metrics without closed-set SYMBOLIC_STATES substring leaks."""
    gold_norm = _norm_ans_v3(answer_str)
    first_line = next((ln.strip() for ln in gen_text.splitlines() if ln.strip()), "")
    first_clause_norm = _norm_ans_v3(re.split(r"[.;\n]", first_line)[0])
    strict_clause_em = 1.0 if (gold_norm and first_clause_norm == gold_norm) else 0.0

    cands = _extract_canonical_v3(gen_text)
    extracted_em = 1.0 if (gold_norm and gold_norm in cands) else 0.0

    gen_norm = _norm_ans_v3(gen_text)
    substr_em = 1.0 if (gold_norm and gold_norm in gen_norm) else 0.0

    a_list = [int(t) for t in a_ids[0].tolist()]
    pred_list = [int(t) for t in ans_logits[0].argmax(dim=-1).tolist()]
    content_pairs = [
        (p, g)
        for p, g in zip(pred_list, a_list)
        if tokenizer.decode([g], skip_special_tokens=True).strip() != ""
    ]
    if content_pairs:
        tf_tok_acc = sum(1 for p, g in content_pairs if p == g) / len(content_pairs)
    else:
        tf_tok_acc = float((ans_logits.argmax(dim=-1) == a_ids).float().mean().item())

    best_f1 = 0.0
    if gold_norm:
        gold_chars = list(gold_norm)
        for c in cands or {""}:
            if not c:
                continue
            common = 0
            c_counts: dict[str, int] = defaultdict(int)
            for ch in c:
                c_counts[ch] += 1
            for ch in gold_chars:
                if c_counts[ch] > 0:
                    common += 1
                    c_counts[ch] -= 1
            if common > 0:
                prec = common / len(c)
                rec = common / len(gold_chars)
                best_f1 = max(best_f1, 2.0 * prec * rec / (prec + rec))

    return strict_clause_em, extracted_em, substr_em, float(tf_tok_acc), float(best_f1)


class _MockTokenizer:
    """Deterministic lightweight tokenizer for local --mock-llm runs."""

    def __init__(self, vocab_size: int = 512) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = None
        self._tok_to_id: dict[str, int] = {"<pad>": 0, "<eos>": 1}
        self._id_to_tok: dict[int, str] = {0: "<pad>", 1: "<eos>"}
        self._next_id = 2

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        tokens = re.findall(r"[A-Za-z0-9_.\-$]+|[^\w\s]", text)
        ids: list[int] = []
        for tok in tokens:
            if tok not in self._tok_to_id:
                tid = 2 + ((self._next_id - 2) % (self.vocab_size - 2))
                self._tok_to_id[tok] = tid
                self._id_to_tok[tid] = tok
                self._next_id += 1
            ids.append(self._tok_to_id[tok])
        return ids or [2]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        words = []
        for tid in token_ids:
            tid_i = int(tid)
            if skip_special_tokens and tid_i in (self.pad_token_id, self.eos_token_id):
                continue
            words.append(self._id_to_tok.get(tid_i, f"t{tid_i}"))
        return " ".join(words)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LSLC v3 Phase 1 training and diagnostics.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--mock-llm", action="store_true", help="Run CPU smoke test with TinyMockCausalLM.")
    parser.add_argument("--stage1-only", action="store_true", help="Run Stage 1 sanity gate + Oracle/Tracer and exit.")
    parser.add_argument("--contextual-edit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--distill-mode",
        type=str,
        choices=("none", "cond_b", "cond_a_verified_text"),
        default="cond_b",
    )
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--reset-salience-gate", action="store_true")
    parser.add_argument("--reset-router", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--save-checkpoint", type=str, default="")
    parser.add_argument("--stage1-steps", type=int, default=400)
    parser.add_argument("--stage2-steps", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=80)
    parser.add_argument("--oracle-episodes", type=int, default=48)
    parser.add_argument("--salience-route-weight", type=float, default=0.0)
    parser.add_argument("--timing-salience-route-weight", type=float, default=2.0)
    parser.add_argument("--k-exact", type=int, default=24)
    parser.add_argument("--salience-span-len", type=int, default=1)
    parser.add_argument("--use-bge-toc", action="store_true", default=False)
    parser.add_argument("--bge-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--router-mode", type=str, choices=("v2_legacy", "content_attn"), default="content_attn")
    parser.add_argument("--train-style", type=str, choices=("standard", "mixed_natural"), default="mixed_natural")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-cond-b", type=float, default=0.5)
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--lm-reg-prob", type=float, default=0.2)
    parser.add_argument("--save-lora-checkpoint", type=str, default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-json", type=str, default="/content/lslc_results/v3_phase1_summary.json")
    args, _ = parser.parse_known_args(argv)
    return args


@torch.no_grad()
def evaluate_heldout_lm_ppl(
    llm: torch.nn.Module,
    token_pool: list[int],
    device: torch.device,
    *,
    chunk_len: int = 256,
    max_chunks: int = 32,
) -> dict[str, float]:
    """Measure standard causal LM perplexity (without LSLC prefix) on held-out token chunks."""
    n_chunks = min(max_chunks, len(token_pool) // chunk_len)
    if n_chunks <= 0:
        return {"n_chunks": 0, "tokens": 0, "mean_nll": 0.0, "ppl": 0.0}
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    prev_training = llm.training
    llm.eval()
    losses: list[float] = []
    try:
        for idx in range(n_chunks):
            chunk = torch.tensor(
                token_pool[idx * chunk_len : (idx + 1) * chunk_len],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                out = llm(input_ids=chunk, labels=chunk)
            losses.append(float(out.loss.detach().item()))
    finally:
        llm.train(prev_training)
    mean_nll = sum(losses) / len(losses)
    return {
        "n_chunks": n_chunks,
        "tokens": n_chunks * chunk_len,
        "mean_nll": round(mean_nll, 4),
        "ppl": round(math.exp(min(mean_nll, 20.0)), 4),
    }


OOD_CODE_PASSAGES = (
    (
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class ConnectionPool:\n"
        "    max_idle: int\n"
        "    timeout_ms: float\n"
        "    def acquire(self, key: str) -> tuple[str, float]:\n"
        "        deadline = self.timeout_ms / 1000.0\n"
        "        return key.strip().lower(), max(0.01, deadline)\n"
    ),
    (
        "def merge_sorted_intervals(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:\n"
        "    ordered = sorted(spans, key=lambda pair: (pair[0], pair[1]))\n"
        "    merged: list[tuple[int, int]] = []\n"
        "    for start, end in ordered:\n"
        "        if not merged or start > merged[-1][1]:\n"
        "            merged.append((start, end))\n"
        "        else:\n"
        "            merged[-1] = (merged[-1][0], max(merged[-1][1], end))\n"
        "    return merged\n"
    ),
    (
        "class TokenBucketLimiter:\n"
        "    def __init__(self, rate_per_sec: float, burst: int) -> None:\n"
        "        self.rate = rate_per_sec\n"
        "        self.capacity = float(burst)\n"
        "        self.tokens = float(burst)\n"
        "    def consume(self, amount: int, dt: float) -> bool:\n"
        "        self.tokens = min(self.capacity, self.tokens + dt * self.rate)\n"
        "        if self.tokens >= amount:\n"
        "            self.tokens -= amount\n"
        "            return True\n"
        "        return False\n"
    ),
)

OOD_PROSE_PASSAGES = (
    (
        "Distributed storage engines separate write-ahead log durability from asynchronous compaction. "
        "By appending sequential frames before flushing sorted string tables, tail latency remains bounded "
        "even when background checksum verification scans older shards."
    ),
    (
        "Modern compiler register allocators model variable interference as a graph coloring problem. "
        "When live ranges exceed available hardware vector lanes, heuristics spill values with the lowest "
        "loop-weighted reload frequency into stack frames."
    ),
    (
        "Coastal tide gauges calibrate pressure transducers against barometric drift every six hours. "
        "High-frequency swell measurements are filtered through a windowed median pass before archiving "
        "regional sea-level anomalies."
    ),
)


@torch.no_grad()
def evaluate_ood_forgetting_ppl(
    llm: torch.nn.Module,
    tokenizer,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Evaluate causal LM perplexity on unseen non-WikiText Python code and technical prose."""
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    prev_training = llm.training
    llm.eval()

    def _score_texts(texts: tuple[str, ...]) -> tuple[list[float], int]:
        nlls: list[float] = []
        tok_total = 0
        for txt in texts:
            ids = tokenizer.encode(txt, add_special_tokens=False)
            if len(ids) < 4:
                continue
            inp = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                out = llm(input_ids=inp, labels=inp)
            nlls.append(float(out.loss.detach().item()))
            tok_total += len(ids)
        return nlls, tok_total

    try:
        code_nlls, code_toks = _score_texts(OOD_CODE_PASSAGES)
        prose_nlls, prose_toks = _score_texts(OOD_PROSE_PASSAGES)
    finally:
        llm.train(prev_training)

    all_nlls = code_nlls + prose_nlls
    c_mean = sum(code_nlls) / max(1, len(code_nlls))
    p_mean = sum(prose_nlls) / max(1, len(prose_nlls))
    a_mean = sum(all_nlls) / max(1, len(all_nlls))
    return {
        "code_ood": {"tokens": code_toks, "mean_nll": round(c_mean, 4), "ppl": round(math.exp(min(c_mean, 20.0)), 4)},
        "prose_ood": {"tokens": prose_toks, "mean_nll": round(p_mean, 4), "ppl": round(math.exp(min(p_mean, 20.0)), 4)},
        "overall": {"tokens": code_toks + prose_toks, "mean_nll": round(a_mean, 4), "ppl": round(math.exp(min(a_mean, 20.0)), 4)},
    }


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_oracle_router_out(num_pages: int, target_page: int, device: torch.device) -> RouterOutput:
    logits = torch.zeros((1, num_pages), dtype=torch.float32, device=device)
    logits[0, target_page] = 10.0
    probs = F.softmax(logits, dim=-1)
    deep_idx = torch.tensor([[target_page]], dtype=torch.long, device=device)
    skim_idx = torch.empty((1, 0), dtype=torch.long, device=device)
    toc_pages = [p for p in range(num_pages) if p != target_page]
    toc_idx = torch.tensor([toc_pages], dtype=torch.long, device=device)
    return RouterOutput(logits=logits, probs=probs, deep_idx=deep_idx, skim_idx=skim_idx, toc_idx=toc_idx)


def _load_lslc_checkpoint(lslc_mod: torch.nn.Module, ckpt_path: str, device: torch.device) -> dict[str, object]:
    p = Path(ckpt_path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(p), device="cpu")
    else:
        sd = torch.load(str(p), map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "lslc" in sd:
            sd = sd["lslc"]
    incompat = lslc_mod.load_state_dict(sd, strict=False)
    print(
        f"[CKPT] Loaded {p.name}: missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}",
        flush=True,
    )
    return {
        "path": str(p),
        "missing_keys": list(incompat.missing_keys),
        "unexpected_keys": list(incompat.unexpected_keys),
    }


def _save_lslc_checkpoint(lslc_mod: torch.nn.Module, ckpt_path: str) -> None:
    p = Path(ckpt_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file({k: v.detach().cpu().contiguous() for k, v in lslc_mod.state_dict().items()}, str(p))
    else:
        torch.save(lslc_mod.state_dict(), str(p))
    print(f"[CKPT] Saved updated v3 weights to {p}", flush=True)


def _build_stage1_episode_view(ep) -> object:
    """Use verified_final_pages directly for Stage 1 multi-schema base representation training."""
    if "verified_final_pages" not in ep.metadata:
        return ep
    return type(ep)(
        context_pages=ep.metadata["verified_final_pages"],
        query_ids=ep.query_ids,
        answer_ids=ep.answer_ids,
        target_page_idx=ep.target_page_idx,
        edit_pages=None,
        metadata=ep.metadata,
    )


def _train_v3_stage(
    name: str,
    wrapper: FrozenQwenLSLCWrapper,
    episodes: list,
    steps: int,
    grad_accum: int,
    lr: float,
    device: torch.device,
    *,
    use_edits: bool,
    lambda_cond_b: float,
    distill_mode: str = "cond_b",
    freeze_toc_proj: bool = False,
    lora_lr: float = 1e-4,
    lm_reg_pool: list[int] | None = None,
    lm_reg_prob: float = 0.0,
    seed: int = 101,
) -> dict[str, float]:
    amp_dtype, use_scaler = select_amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    saved_toc_flags: list[tuple[torch.nn.Parameter, bool]] = []
    if use_edits:
        delta_ids = {id(p) for p in wrapper.lslc.delta_updater.parameters()}
        for param in wrapper.lslc.parameters():
            if id(param) not in delta_ids:
                saved_toc_flags.append((param, param.requires_grad))
                param.requires_grad_(False)
    elif freeze_toc_proj:
        for param in wrapper.lslc.projector.toc_proj.parameters():
            saved_toc_flags.append((param, param.requires_grad))
            param.requires_grad_(False)

    lslc_params = [p for p in wrapper.lslc.parameters() if p.requires_grad]
    llm_lora_params = lora_parameters(wrapper.llm)
    param_groups = [{"params": lslc_params, "lr": lr}]
    if llm_lora_params:
        param_groups.append({"params": llm_lora_params, "lr": lora_lr})
    params = lslc_params + llm_lora_params
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=wrapper.cfg.weight_decay,
        betas=wrapper.cfg.betas,
    )
    warmup = max(1, int(steps * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: float(s + 1) / warmup
        if s < warmup
        else max(0.1, 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))),
    )

    wrapper.lslc.train()
    optimizer.zero_grad(set_to_none=True)
    rng = random.Random(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    last_losses: dict[str, float] = {}

    try:
        for step in range(1, steps + 1):
            tot_val = 0.0
            for _ in range(grad_accum):
                ep = episodes[rng.randrange(len(episodes))]
                pages = ep.context_pages.unsqueeze(0).to(device)
                q_ids = ep.query_ids.unsqueeze(0).to(device)
                rq_raw = ep.metadata.get("router_query_ids", ep.query_ids)
                r_q_ids = rq_raw.unsqueeze(0).to(device)
                a_ids = ep.answer_ids.unsqueeze(0).to(device)
                t_page = torch.tensor([ep.target_page_idx], dtype=torch.long, device=device)

                edit_steps = None
                edit_accepted_flags = None
                verified_final_pages = None
                mask_key = "context_salience_span_mask" if use_edits else "verified_salience_span_mask"
                sal_mask_raw = ep.metadata.get(mask_key)
                sal_mask = sal_mask_raw.unsqueeze(0).to(device) if sal_mask_raw is not None else None
                if use_edits and ep.edit_pages:
                    edit_steps = [
                        (torch.tensor([p_idx], dtype=torch.long, device=device), toks.unsqueeze(0).to(device))
                        for p_idx, toks in ep.edit_pages
                    ]
                    edit_accepted_flags = ep.metadata.get("edit_accepted_flags")
                    vfp = ep.metadata.get("verified_final_pages")
                    if vfp is not None:
                        verified_final_pages = vfp.unsqueeze(0).to(device)

                autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    out = wrapper.forward_step(
                        page_token_ids=pages,
                        query_ids=q_ids,
                        answer_ids=a_ids,
                        target_page_idx=t_page,
                        edit_steps=edit_steps,
                        use_exact_bank=True,
                        run_reconstruction=(not use_edits),
                        run_matryoshka=True,
                        distill_salience=(not use_edits),
                        lambda_sal=2.0,
                        edit_accepted_flags=edit_accepted_flags,
                        verified_final_pages=verified_final_pages,
                        lambda_cond_b=lambda_cond_b if use_edits else 0.0,
                        distill_mode=distill_mode if use_edits else "none",
                        router_query_ids=r_q_ids,
                        salience_span_mask=sal_mask,
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
                    "cond_b": float(out.loss_cond_b.item()) if out.loss_cond_b is not None else 0.0,
                    "gate": float(out.mean_gate),
                }

            lm_reg_val = 0.0
            lm_interval = (
                max(1, round(1.0 / lm_reg_prob))
                if (llm_lora_params and lm_reg_pool and len(lm_reg_pool) >= 256 and lm_reg_prob > 0.0)
                else 0
            )
            if lm_interval > 0 and (step % lm_interval == 0):
                start_lm = rng.randrange(0, len(lm_reg_pool) - 255)
                lm_ids = torch.tensor(lm_reg_pool[start_lm : start_lm + 256], dtype=torch.long, device=device).unsqueeze(0)
                autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    lm_out = wrapper.llm(input_ids=lm_ids, labels=lm_ids)
                    lm_reg_loss = 0.5 * lm_out.loss
                if use_scaler:
                    scaler.scale(lm_reg_loss).backward()
                else:
                    lm_reg_loss.backward()
                lm_reg_val = float(lm_out.loss.detach().item())
            if last_losses:
                last_losses["lm_reg"] = lm_reg_val

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

            if step == 1 or step % 50 == 0 or step == steps:
                print(
                    f"[{name}] step={step}/{steps} total={tot_val:.4f} ntp={last_losses['ntp']:.4f} "
                    f"cond_b={last_losses['cond_b']:.4f} route={last_losses['route']:.4f} "
                    f"sal={last_losses['sal']:.4f} lm_reg={last_losses.get('lm_reg', 0.0):.4f} "
                    f"gate={last_losses['gate']:.4f} gnorm={gnorm:.3f}",
                    flush=True,
                )
    finally:
        for param, prev_req in saved_toc_flags:
            param.requires_grad_(prev_req)

    peak_gb = (torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)) if device.type == "cuda" else 0.0
    elapsed = time.perf_counter() - t0
    return {
        "steps": steps,
        "elapsed_sec": round(elapsed, 2),
        "peak_vram_gb": round(peak_gb, 3),
        "final_loss": round(tot_val, 4),
        **{f"final_{k}": round(v, 4) for k, v in last_losses.items()},
    }


def _score_generation_with_text(
    llm: torch.nn.Module,
    embed_fn: torch.nn.Module,
    prefix_llm: torch.Tensor,
    q_ids: torch.Tensor,
    a_ids: torch.Tensor,
    answer_str: str,
    tokenizer,
    device: torch.device,
) -> tuple[float, float, float, float, float, str]:
    llm_dtype = embed_fn.weight.dtype
    prefix_llm = prefix_llm.to(llm_dtype)
    turn_ids = torch.cat((q_ids, a_ids), dim=1)
    turn_emb = embed_fn(turn_ids).to(llm_dtype)
    inp = torch.cat((prefix_llm, turn_emb), dim=1)
    attn = torch.ones((1, inp.shape[1]), device=device, dtype=torch.long)
    logits = llm(inputs_embeds=inp, attention_mask=attn).logits
    start = prefix_llm.shape[1] + q_ids.shape[1] - 1
    ans_logits = logits[:, start : start + a_ids.shape[1], :]
    if isinstance(llm, TinyMockCausalLM):
        gen_ids = greedy_generate_from_prefix(
            llm, prefix_llm, q_ids, max_new_tokens=24, eos_token_id=tokenizer.eos_token_id
        )
        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    else:
        gen_ids, gen_text = greedy_generate_concise(llm, prefix_llm, q_ids, tokenizer, max_new_tokens=24)
    clause_em, extracted_em, substr_em, tok_acc, f1 = compute_answer_metrics_v3(
        ans_logits, a_ids, gen_text, answer_str, tokenizer
    )
    return clause_em, extracted_em, substr_em, tok_acc, f1, gen_text


def _answer_token_variants(tokenizer, answer_str: str, fallback_ids: torch.Tensor) -> list[list[int]]:
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    special = {
        getattr(tokenizer, "pad_token_id", -1),
        getattr(tokenizer, "eos_token_id", -1),
        getattr(tokenizer, "bos_token_id", -1),
    } | (set(int(x) for x in nl_ids) if len(nl_ids) == 1 else set())
    variants: list[list[int]] = []
    # Prioritize clean value tokenizations first (without stop delimiters), then stripped fallback_ids
    for raw in (
        tokenizer.encode(f" {answer_str}", add_special_tokens=False),
        tokenizer.encode(answer_str, add_special_tokens=False),
        fallback_ids.tolist(),
    ):
        toks = [int(t) for t in raw if int(t) not in special]
        if toks and toks not in variants:
            variants.append(toks)
    return variants


def _any_variant_present(container: set[int], variants: list[list[int]]) -> bool:
    return any(bool(v) and all(t in container for t in v) for v in variants)


@torch.no_grad()
def trace_identifier_pipeline(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    episodes: list,
    device: torch.device,
    *,
    max_episodes: int = 24,
    salience_route_weight: float = 0.0,
    modes: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Trace where answer identifiers drop along source -> SparseExactBank -> prefix -> output."""
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    amp_dtype, _ = select_amp_dtype(device)
    vocab_size = len(embed_fn.weight)
    wrapper.lslc.eval()

    sample_eps = episodes[:max_episodes]
    active_modes = (
        (
            "cond_b_oracle",
            "cond_b_natural_router",
            "cond_b_hybrid_router",
            "cond_c_oracle",
            "cond_c_natural_router",
            "cond_c_hybrid_router",
        )
        if modes is None
        else tuple(modes)
    )
    out: dict[str, object] = {}

    for trace_mode in active_modes:
        mode_sal_w = 2.0 if trace_mode.endswith("_hybrid_router") else salience_route_weight
        n_total = 0
        n_src = 0
        n_bank = 0
        n_route_top1 = 0
        n_route_hit = 0
        n_pref = 0
        n_gen = 0
        qt_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        failed_samples: list[dict[str, object]] = []

        for ep in sample_eps:
            n_total += 1
            qtype = str(ep.metadata.get("query_type", "unknown"))
            qt_counts[qtype]["n"] += 1
            t_page = ep.target_page_idx
            ans_str = str(ep.metadata["answer_str"])
            ans_variants = _answer_token_variants(tokenizer, ans_str, ep.answer_ids)
            ans_toks = ans_variants[0] if ans_variants else []
            verified_pages = ep.metadata["verified_final_pages"]
            in_source = _any_variant_present(set(int(x) for x in verified_pages[t_page].tolist()), ans_variants)
            if in_source:
                n_src += 1
                qt_counts[qtype]["src"] += 1

            q_ids = ep.query_ids.unsqueeze(0).to(device)
            rq_raw = ep.metadata.get("router_query_ids", ep.query_ids)
            r_q_ids = rq_raw.unsqueeze(0).to(device)
            a_ids = ep.answer_ids.unsqueeze(0).to(device)
            autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                q_emb = embed_fn(r_q_ids).mean(dim=1)
                if trace_mode.startswith("cond_b_"):
                    pages_b = verified_pages.unsqueeze(0).to(device)
                    z_cur, bank_cur, toc_cur = wrapper.lslc.encode_book(
                        pages_b, embed_fn, use_exact_bank=True, vocab_size=vocab_size
                    )
                    r_override = (
                        _make_oracle_router_out(pages_b.shape[1], t_page, device)
                        if trace_mode == "cond_b_oracle"
                        else None
                    )
                else:
                    pages_c = ep.context_pages.unsqueeze(0).to(device)
                    z_cur, bank_cur, _ = wrapper.lslc.encode_book(
                        pages_c, embed_fn, use_exact_bank=True, vocab_size=vocab_size
                    )
                    for p_idx, e_ids in (ep.edit_pages or []):
                        p_t = torch.tensor([p_idx], dtype=torch.long, device=device)
                        e_t = e_ids.unsqueeze(0).to(device)
                        override = (
                            extract_contextual_edit_embeds(
                                wrapper.llm, e_t, layer_idx=wrapper.cfg.contextual_edit_layer
                            )
                            if wrapper.cfg.use_contextual_edit_encoding
                            else None
                        )
                        z_cur, _, bank_cur = wrapper.lslc.apply_page_delta(
                            z_cur,
                            p_t,
                            e_t,
                            embed_fn,
                            exact_bank=bank_cur,
                            vocab_size=vocab_size,
                            edit_embeds_override=override,
                            in_place=True,
                        )
                    toc_cur = wrapper.lslc.projector.project_toc(z_cur)
                    r_override = (
                        _make_oracle_router_out(pages_c.shape[1], t_page, device)
                        if trace_mode == "cond_c_oracle"
                        else None
                    )

                router_out, asm = wrapper.lslc.route_and_assemble(
                    z_cur,
                    toc_cur,
                    q_emb,
                    bank_cur,
                    embed_fn,
                    mode="full",
                    training_ste=False,
                    router_out_override=r_override,
                    query_input_ids=r_q_ids,
                    salience_route_weight=mode_sal_w,
                )

            deep_pages = [int(p) for p in router_out.deep_idx[0].tolist()]
            top1_page = int(torch.argmax(router_out.logits[0], dim=-1).item())
            router_top1 = top1_page == t_page
            router_hit = t_page in deep_pages
            bank_target_set = set(int(x) for x in bank_cur.token_ids[0, t_page].tolist()) if bank_cur is not None else set()
            in_bank = _any_variant_present(bank_target_set, ans_variants)
            in_prefix = router_hit and in_bank
            if router_top1:
                n_route_top1 += 1
                qt_counts[qtype]["route_top1"] += 1
            if router_hit:
                n_route_hit += 1
                qt_counts[qtype]["route_hit"] += 1
            if in_bank:
                n_bank += 1
                qt_counts[qtype]["bank"] += 1
            if in_prefix:
                n_pref += 1
                qt_counts[qtype]["pref"] += 1

            _, extracted_em, _, _, _, gen_text = _score_generation_with_text(
                wrapper.llm, embed_fn, asm.prefix_embeds.to(llm_dtype), q_ids, a_ids, ans_str, tokenizer, device
            )
            if extracted_em >= 0.99:
                n_gen += 1
                qt_counts[qtype]["gen"] += 1
            elif len(failed_samples) < 5:
                failed_samples.append(
                    {
                        "query_type": qtype,
                        "target_page": t_page,
                        "router_top1": top1_page,
                        "router_deep_idx": deep_pages,
                        "router_hit": router_hit,
                        "answer_str": ans_str,
                        "ans_tok_ids": ans_toks,
                        "in_source": in_source,
                        "in_bank": in_bank,
                        "in_prefix": in_prefix,
                        "gen_text": gen_text,
                    }
                )

        denom = max(1, n_total)
        by_qt_summary = {}
        for qt, cnts in sorted(qt_counts.items()):
            d_q = max(1, cnts["n"])
            by_qt_summary[qt] = {
                "num_episodes": cnts["n"],
                "in_source_page_rate": round(cnts["src"] / d_q, 4),
                "in_exact_bank_rate": round(cnts["bank"] / d_q, 4),
                "router_top1_recall": round(cnts["route_top1"] / d_q, 4),
                "router_top2_recall": round(cnts["route_hit"] / d_q, 4),
                "router_hit_rate": round(cnts["route_hit"] / d_q, 4),
                "in_assembled_prefix_rate": round(cnts["pref"] / d_q, 4),
                "in_generated_answer_rate": round(cnts["gen"] / d_q, 4),
            }
        out[trace_mode] = {
            "num_episodes": n_total,
            "in_source_page_rate": round(n_src / denom, 4),
            "in_exact_bank_rate": round(n_bank / denom, 4),
            "router_top1_recall": round(n_route_top1 / denom, 4),
            "router_top2_recall": round(n_route_hit / denom, 4),
            "router_hit_rate": round(n_route_hit / denom, 4),
            "in_assembled_prefix_rate": round(n_pref / denom, 4),
            "in_generated_answer_rate": round(n_gen / denom, 4),
            "by_query_type": by_qt_summary,
            "failed_samples": failed_samples,
        }
    return out


def _rag_prefix_for_episode(ep, mode: str, tokenizer, embed_fn, llm_dtype: torch.dtype, device: torch.device, k_top: int) -> torch.Tensor:
    num_pages = ep.context_pages.shape[0]
    page_texts = [tokenizer.decode(ep.context_pages[i], skip_special_tokens=True) for i in range(num_pages)]
    rq_raw = ep.metadata.get("router_query_ids", ep.query_ids)
    query_text = tokenizer.decode(rq_raw, skip_special_tokens=True)

    if mode == "rag_immutable":
        docs = list(page_texts)
        doc_tensors = [ep.context_pages[i] for i in range(num_pages)]
    elif mode == "rag_append_errata":
        docs = list(page_texts)
        doc_tensors = [ep.context_pages[i] for i in range(num_pages)]
        for p_idx, e_toks in (ep.edit_pages or []):
            edit_txt = tokenizer.decode(e_toks, skip_special_tokens=True)
            docs[p_idx] = docs[p_idx] + " " + edit_txt
            doc_tensors[p_idx] = torch.cat((doc_tensors[p_idx], e_toks.to(doc_tensors[p_idx].dtype)), dim=0)
    elif mode == "rag_direct_patch_oracle_verified":
        verified_pages = ep.metadata["verified_final_pages"]
        docs = [tokenizer.decode(verified_pages[i], skip_special_tokens=True) for i in range(num_pages)]
        doc_tensors = [verified_pages[i] for i in range(num_pages)]
    else:
        raise ValueError(f"Unknown RAG mode: {mode}")

    retriever = BM25PageRetriever(docs)
    k = min(k_top, len(docs))
    router_out = retriever.route(query_text, k_deep=k, k_skim=0, device=device)
    chosen = router_out.deep_idx[0].tolist()
    toks = torch.cat([doc_tensors[i].to(device) for i in chosen], dim=0).unsqueeze(0)
    return embed_fn(toks).to(llm_dtype)


@torch.no_grad()
def evaluate_five_way_adversarial(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    episodes: list,
    device: torch.device,
    *,
    salience_route_weight: float = 0.0,
    bge_toc=None,
) -> dict[str, dict[str, object]]:
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    amp_dtype, _ = select_amp_dtype(device)
    vocab_size = len(embed_fn.weight)
    wrapper.lslc.eval()

    systems = (
        "rag_immutable",
        "rag_append_errata",
        "rag_direct_patch_oracle_verified",
        "lslc_v3_e2e_gpu",
        "lslc_v3_shared_bm25",
    )
    results: dict[str, dict[str, object]] = {}

    for sys_name in systems:
        agg_overall: dict[str, list[float]] = defaultdict(list)
        by_qtype: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        inv_diffs: list[float] = []
        struct_inv_diffs: list[float] = []
        acc_gates: list[float] = []
        rej_gates: list[float] = []

        for ep in episodes:
            pages = ep.context_pages.unsqueeze(0).to(device)
            q_ids = ep.query_ids.unsqueeze(0).to(device)
            a_ids = ep.answer_ids.unsqueeze(0).to(device)
            qtype = str(ep.metadata.get("query_type", "unknown"))

            if sys_name.startswith("rag_"):
                prefix_llm = _rag_prefix_for_episode(
                    ep, sys_name, tokenizer, embed_fn, llm_dtype, device, k_top=wrapper.cfg.k_deep
                )
            else:
                autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    z_book, bank, toc0 = wrapper.lslc.encode_book(
                        pages, embed_fn, use_exact_bank=True, vocab_size=vocab_size
                    )
                    rq_raw = ep.metadata.get("router_query_ids", ep.query_ids)
                    r_q_ids = rq_raw.unsqueeze(0).to(device)
                    q_emb = embed_fn(r_q_ids).mean(dim=1)

                    page_texts: list[str] | None = None
                    bm25: BM25PageRetriever | None = None
                    router_before_override = None
                    if sys_name == "lslc_v3_shared_bm25":
                        page_texts = [
                            tokenizer.decode(ep.context_pages[i], skip_special_tokens=True)
                            for i in range(ep.context_pages.shape[0])
                        ]
                        bm25 = BM25PageRetriever(page_texts)
                        q_text = tokenizer.decode(rq_raw, skip_special_tokens=True)
                        router_before_override = bm25.route(
                            q_text, k_deep=wrapper.cfg.k_deep, k_skim=wrapper.cfg.k_skim, device=device
                        )

                    router_before, asm_before = wrapper.lslc.route_and_assemble(
                        z_book,
                        toc0,
                        q_emb,
                        bank,
                        embed_fn,
                        mode="full",
                        training_ste=False,
                        router_out_override=router_before_override,
                        query_input_ids=r_q_ids,
                        salience_route_weight=salience_route_weight,
                    )

                    accepted_flags = ep.metadata.get("edit_accepted_flags", [])
                    for step_idx, (p_idx, e_ids) in enumerate(ep.edit_pages or []):
                        p_t = torch.tensor([p_idx], dtype=torch.long, device=device)
                        e_t = e_ids.unsqueeze(0).to(device)
                        override = (
                            extract_contextual_edit_embeds(
                                wrapper.llm,
                                e_t,
                                layer_idx=wrapper.cfg.contextual_edit_layer,
                            )
                            if wrapper.cfg.use_contextual_edit_encoding
                            else None
                        )
                        z_book, g_p, bank = wrapper.lslc.apply_page_delta(
                            z_book,
                            p_t,
                            e_t,
                            embed_fn,
                            exact_bank=bank,
                            vocab_size=vocab_size,
                            edit_embeds_override=override,
                            in_place=True,
                        )
                        g_val = float(g_p.mean().item())
                        if step_idx < len(accepted_flags):
                            if accepted_flags[step_idx]:
                                acc_gates.append(g_val)
                            else:
                                rej_gates.append(g_val)
                        # Strictly unfiltered BM25 update (no eval oracle leakage from accepted_flags)
                        if bm25 is not None and page_texts is not None:
                            edit_txt = tokenizer.decode(e_ids, skip_special_tokens=True)
                            page_texts[p_idx] = page_texts[p_idx] + " " + edit_txt
                            bm25.update_page(p_idx, page_texts[p_idx])

                    toc_after = wrapper.lslc.projector.project_toc(z_book)
                    imm = asm_before.immutable_prefix_len
                    _, asm_after_same_route = wrapper.lslc.route_and_assemble(
                        z_book,
                        toc_after,
                        q_emb,
                        bank,
                        embed_fn,
                        mode="full",
                        training_ste=False,
                        router_out_override=router_before,
                    )
                    struct_inv_diffs.append(
                        float(
                            (asm_after_same_route.prefix_embeds[:, :imm] - asm_before.prefix_embeds[:, :imm])
                            .abs()
                            .max()
                            .item()
                        )
                    )
                    router_override = None
                    ext_logits = None
                    if bm25 is not None:
                        q_text = tokenizer.decode(rq_raw, skip_special_tokens=True)
                        router_override = bm25.route(
                            q_text, k_deep=wrapper.cfg.k_deep, k_skim=wrapper.cfg.k_skim, device=device
                        )
                    elif bge_toc is not None:
                        q_text = tokenizer.decode(rq_raw, skip_special_tokens=True)
                        bge_p4 = bge_toc.encode_book_pages(pages, None, tokenizer, num_spans=4, batch_size=64)
                        ext_logits = bge_toc.score_book(bge_toc.encode_texts([q_text], is_query=True), bge_p4)
                    _, asm_after = wrapper.lslc.route_and_assemble(
                        z_book,
                        toc_after,
                        q_emb,
                        bank,
                        embed_fn,
                        mode="full",
                        training_ste=False,
                        router_out_override=router_override,
                        query_input_ids=r_q_ids,
                        salience_route_weight=salience_route_weight,
                        external_dense_logits=ext_logits,
                    )

                    inv_diffs.append(
                        float((asm_after.prefix_embeds[:, :imm] - asm_before.prefix_embeds[:, :imm]).abs().max().item())
                    )
                    prefix_llm = asm_after.prefix_embeds.to(llm_dtype)

            clause_em, extracted_em, substr_em, tok_acc, f1, _ = _score_generation_with_text(
                wrapper.llm, embed_fn, prefix_llm, q_ids, a_ids, str(ep.metadata["answer_str"]), tokenizer, device
            )
            for target_dict in (agg_overall, by_qtype[qtype]):
                target_dict["extracted_em"].append(extracted_em)
                target_dict["clause_or_prefix_em"].append(clause_em)
                target_dict["substr_em"].append(substr_em)
                target_dict["tf_tok_acc"].append(tok_acc)
                target_dict["f1"].append(f1)

        summary: dict[str, object] = {
            k: round(sum(v) / max(1, len(v)), 4) for k, v in agg_overall.items()
        }
        summary["by_query_type"] = {
            qt: {m: round(sum(vals) / max(1, len(vals)), 4) for m, vals in m_dict.items()}
            for qt, m_dict in sorted(by_qtype.items())
        }
        if struct_inv_diffs:
            summary["max_structural_invariance_diff"] = max(struct_inv_diffs)
        if inv_diffs:
            summary["max_prefix_invariance_diff"] = max(inv_diffs)
        if acc_gates:
            summary["mean_accepted_gate"] = round(sum(acc_gates) / len(acc_gates), 5)
        if rej_gates:
            summary["mean_rejected_gate"] = round(sum(rej_gates) / len(rej_gates), 5)
        results[sys_name] = summary

    return results


def _clone_exact_bank(bank: SparseExactBank) -> SparseExactBank:
    return SparseExactBank(
        token_ids=bank.token_ids.clone(),
        page_ids=bank.page_ids.clone(),
        h_coords=bank.h_coords.clone(),
        w_coords=bank.w_coords.clone(),
        _ste_gates=None if bank._ste_gates is None else bank._ste_gates.clone(),
        base_token_ids=None if bank.base_token_ids is None else bank.base_token_ids.clone(),
    )


def _tile_exact_bank(bank: SparseExactBank, num_p: int) -> SparseExactBank:
    base_p = bank.token_ids.shape[1]
    reps = (num_p + base_p - 1) // base_p
    k_ex = bank.token_ids.shape[-1]
    toks = bank.token_ids.repeat(1, reps, 1)[:, :num_p, :].contiguous()
    h = bank.h_coords.repeat(1, reps, 1)[:, :num_p, :].contiguous()
    w = bank.w_coords.repeat(1, reps, 1)[:, :num_p, :].contiguous()
    p_ids = (
        torch.arange(num_p, device=toks.device, dtype=bank.page_ids.dtype)
        .view(1, num_p, 1)
        .expand(bank.token_ids.shape[0], num_p, k_ex)
        .contiguous()
    )
    ste = None if bank._ste_gates is None else bank._ste_gates.repeat(1, reps, 1)[:, :num_p, :].contiguous()
    return SparseExactBank(token_ids=toks, page_ids=p_ids, h_coords=h, w_coords=w, _ste_gates=ste)


@torch.no_grad()
def _eval_oracle_abc_subset(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    episodes: list,
    device: torch.device,
) -> dict[str, object]:
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    amp_dtype, _ = select_amp_dtype(device)
    vocab_size = len(embed_fn.weight)

    by_cond_k: dict[str, dict[int, list[float]]] = {
        "A_raw_text_base_frozen": defaultdict(list),
        "A_raw_text": defaultdict(list),
        "B_reencode_scratch": defaultdict(list),
        "C_delta_updater": defaultdict(list),
    }
    a_failed_samples: list[dict[str, object]] = []
    has_lora = bool(lora_parameters(wrapper.llm))

    for ep in episodes:
        t_page = ep.target_page_idx
        k_edits = len(ep.edit_pages or [])
        verified_pages = ep.metadata["verified_final_pages"].unsqueeze(0).to(device)
        init_pages = ep.context_pages.unsqueeze(0).to(device)
        num_p = init_pages.shape[1]
        q_ids = ep.query_ids.unsqueeze(0).to(device)
        a_ids = ep.answer_ids.unsqueeze(0).to(device)
        ans_str = str(ep.metadata["answer_str"])

        prefix_a = embed_fn(verified_pages[:, t_page, :]).to(llm_dtype)
        if has_lora:
            set_lora_enabled(wrapper.llm, False)
            _, ex_a_frozen, _, _, _, _ = _score_generation_with_text(
                wrapper.llm, embed_fn, prefix_a, q_ids, a_ids, ans_str, tokenizer, device
            )
            set_lora_enabled(wrapper.llm, True)
            _, ex_a, _, _, _, gen_a = _score_generation_with_text(
                wrapper.llm, embed_fn, prefix_a, q_ids, a_ids, ans_str, tokenizer, device
            )
        else:
            _, ex_a, _, _, _, gen_a = _score_generation_with_text(
                wrapper.llm, embed_fn, prefix_a, q_ids, a_ids, ans_str, tokenizer, device
            )
            ex_a_frozen = ex_a
        by_cond_k["A_raw_text_base_frozen"][k_edits].append(ex_a_frozen)
        by_cond_k["A_raw_text"][k_edits].append(ex_a)
        if ex_a_frozen < 1.0 and len(a_failed_samples) < 3:
            a_failed_samples.append({"answer": ans_str, "gen_text": gen_a, "k_edits": k_edits})

        oracle_router = _make_oracle_router_out(num_p, t_page, device)
        rq_raw = ep.metadata.get("router_query_ids", ep.query_ids)
        r_q_ids = rq_raw.unsqueeze(0).to(device)
        autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            q_emb = embed_fn(r_q_ids).mean(dim=1)
            z_b, bank_b, toc_b = wrapper.lslc.encode_book(verified_pages, embed_fn, use_exact_bank=True, vocab_size=vocab_size)
            _, asm_b = wrapper.lslc.route_and_assemble(
                z_b, toc_b, q_emb, bank_b, embed_fn, mode="full", training_ste=False, router_out_override=oracle_router
            )

            z_c, bank_c, _ = wrapper.lslc.encode_book(init_pages, embed_fn, use_exact_bank=True, vocab_size=vocab_size)
            for p_idx, e_ids in (ep.edit_pages or []):
                p_t = torch.tensor([p_idx], dtype=torch.long, device=device)
                e_t = e_ids.unsqueeze(0).to(device)
                override = (
                    extract_contextual_edit_embeds(
                        wrapper.llm, e_t, layer_idx=wrapper.cfg.contextual_edit_layer
                    )
                    if wrapper.cfg.use_contextual_edit_encoding
                    else None
                )
                z_c, _, bank_c = wrapper.lslc.apply_page_delta(
                    z_c,
                    p_t,
                    e_t,
                    embed_fn,
                    exact_bank=bank_c,
                    vocab_size=vocab_size,
                    edit_embeds_override=override,
                    in_place=True,
                )
            toc_c = wrapper.lslc.projector.project_toc(z_c)
            _, asm_c = wrapper.lslc.route_and_assemble(
                z_c, toc_c, q_emb, bank_c, embed_fn, mode="full", training_ste=False, router_out_override=oracle_router
            )

        _, ex_b, _, _, _, _ = _score_generation_with_text(
            wrapper.llm, embed_fn, asm_b.prefix_embeds.to(llm_dtype), q_ids, a_ids, ans_str, tokenizer, device
        )
        _, ex_c, _, _, _, _ = _score_generation_with_text(
            wrapper.llm, embed_fn, asm_c.prefix_embeds.to(llm_dtype), q_ids, a_ids, ans_str, tokenizer, device
        )
        by_cond_k["B_reencode_scratch"][k_edits].append(ex_b)
        by_cond_k["C_delta_updater"][k_edits].append(ex_c)

    out_oracle: dict[str, object] = {
        cond: {
            f"k={k}": round(sum(vals) / max(1, len(vals)), 4)
            for k, vals in sorted(k_map.items())
        }
        for cond, k_map in by_cond_k.items()
    }
    out_oracle["a_failed_samples"] = a_failed_samples
    return out_oracle


@torch.no_grad()
def evaluate_oracle_abc_and_q_breakeven(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    oracle_symbolic_eps: list,
    oracle_budget_eps: list,
    device: torch.device,
    *,
    is_mock: bool,
    salience_route_weight: float = 0.0,
    timing_salience_route_weight: float = 2.0,
) -> dict[str, object]:
    embed_fn = wrapper.llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    vocab_size = len(embed_fn.weight)
    wrapper.lslc.eval()

    oracle_symbolic = _eval_oracle_abc_subset(wrapper, tokenizer, oracle_symbolic_eps, device)
    oracle_budget = _eval_oracle_abc_subset(wrapper, tokenizer, oracle_budget_eps, device)

    # Identical-API (z_book + SparseExactBank) scaling benchmark for in_place=False vs in_place=True
    page_counts = (4, 16, 64) if is_mock else (4, 128, 1024)
    page_scaling: dict[str, dict[str, float]] = {}
    episodes = oracle_symbolic_eps or oracle_budget_eps
    t_page_size = episodes[0].context_pages.shape[1]
    edit_sample = (
        episodes[0].edit_pages[0][1].unsqueeze(0).to(device)
        if episodes[0].edit_pages
        else episodes[0].query_ids.unsqueeze(0).to(device)
    )
    base_p = 16 if not is_mock else 8
    base_pages = torch.randint(0, vocab_size, (1, base_p, t_page_size), device=device)
    z_tile_src, bank_tile_src, _ = wrapper.lslc.encode_book(
        base_pages, embed_fn, use_exact_bank=True, vocab_size=vocab_size
    )
    reps_needed = lambda np: (np + base_p - 1) // base_p  # noqa: E731

    for num_p in page_counts:
        reps = reps_needed(num_p)
        z_clone_in = z_tile_src.repeat(1, reps, 1, 1, 1, 1)[:, :num_p].clone()
        bank_clone_in = _tile_exact_bank(bank_tile_src, num_p)
        z_inplace_in = z_tile_src.repeat(1, reps, 1, 1, 1, 1)[:, :num_p].clone()
        bank_inplace_in = _tile_exact_bank(bank_tile_src, num_p)
        p_t = torch.tensor([0], dtype=torch.long, device=device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync_device(device)
        t0 = time.perf_counter()
        _ = wrapper.lslc.apply_page_delta(
            z_clone_in, p_t, edit_sample, embed_fn, exact_bank=bank_clone_in, vocab_size=vocab_size, in_place=False
        )
        _sync_device(device)
        clone_ms = (time.perf_counter() - t0) * 1000.0
        clone_mb = (torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)) if device.type == "cuda" else 0.0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync_device(device)
        t0 = time.perf_counter()
        _ = wrapper.lslc.apply_page_delta(
            z_inplace_in, p_t, edit_sample, embed_fn, exact_bank=bank_inplace_in, vocab_size=vocab_size, in_place=True
        )
        _sync_device(device)
        inplace_ms = (time.perf_counter() - t0) * 1000.0
        inplace_mb = (torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)) if device.type == "cuda" else 0.0

        page_scaling[f"P={num_p}"] = {
            "clone_ms": round(clone_ms, 3),
            "in_place_ms": round(inplace_ms, 3),
            "clone_peak_mb": round(clone_mb, 3),
            "in_place_peak_mb": round(inplace_mb, 3),
        }

    # Temporarily merge LoRA during end-to-end timing so LoRA inference pays 0 adapter overhead
    merged_n = merge_lora(wrapper.llm)
    try:
        ep0 = episodes[0]
        pages0 = ep0.context_pages.unsqueeze(0).to(device)
        q0 = ep0.query_ids.unsqueeze(0).to(device)
        e0 = ep0.edit_pages[0][1].unsqueeze(0).to(device) if ep0.edit_pages else q0
        z0, bank0, _ = wrapper.lslc.encode_book(pages0, embed_fn, use_exact_bank=True, vocab_size=vocab_size)
        raw_flat_emb = embed_fn(pages0.reshape(1, -1)).to(llm_dtype)

        _sync_device(device)
        t_ctx0 = time.perf_counter()
        override0 = (
            extract_contextual_edit_embeds(wrapper.llm, e0, layer_idx=wrapper.cfg.contextual_edit_layer)
            if wrapper.cfg.use_contextual_edit_encoding
            else None
        )
        _sync_device(device)
        contextual_encode_ms = (
            (time.perf_counter() - t_ctx0) * 1000.0 if wrapper.cfg.use_contextual_edit_encoding else 0.0
        )

        _sync_device(device)
        t_write0 = time.perf_counter()
        z0, _, bank0 = wrapper.lslc.apply_page_delta(
            z0,
            torch.tensor([0], device=device),
            e0,
            embed_fn,
            exact_bank=bank0,
            vocab_size=vocab_size,
            edit_embeds_override=override0,
            in_place=True,
        )
        toc0 = wrapper.lslc.projector.project_toc(z0)
        _sync_device(device)
        local_write_ms = (time.perf_counter() - t_write0) * 1000.0
        end_to_end_edit_ms = contextual_encode_ms + local_write_ms

        _sync_device(device)
        t_reenc0 = time.perf_counter()
        for _ in range(5):
            _ = wrapper.lslc.encode_book(pages0[:, :1, :], embed_fn, use_exact_bank=True, vocab_size=vocab_size)
        _sync_device(device)
        single_page_reencode_ms = ((time.perf_counter() - t_reenc0) * 1000.0) / 5.0

        num_p0 = ep0.context_pages.shape[0]
        page_texts0 = [tokenizer.decode(ep0.context_pages[i], skip_special_tokens=True) for i in range(num_p0)]
        page_tensors0 = [ep0.context_pages[i].to(device) for i in range(num_p0)]
        bm25_bench = BM25PageRetriever(page_texts0)
        k_rag = min(wrapper.cfg.k_deep, num_p0)

        t_rag_edit0 = time.perf_counter()
        edit_txt0 = tokenizer.decode(ep0.edit_pages[0][1] if ep0.edit_pages else ep0.query_ids, skip_special_tokens=True)
        page_texts0[0] = page_texts0[0] + " " + edit_txt0
        bm25_bench.update_page(0, page_texts0[0])
        rag_edit_ms = (time.perf_counter() - t_rag_edit0) * 1000.0

        rq0_raw = ep0.metadata.get("router_query_ids", ep0.query_ids)
        rq0 = rq0_raw.unsqueeze(0).to(device)
        q_text0 = tokenizer.decode(rq0_raw, skip_special_tokens=True)
        n_warmup = 1 if is_mock else 3
        n_bench = 3 if is_mock else 10

        def _time_lslc_query(k_d: int, k_s: int, sal_w: float) -> tuple[float, int]:
            pref_toks = 0
            for _ in range(n_warmup):
                q_e = embed_fn(rq0).mean(dim=1)
                _, a_w = wrapper.lslc.route_and_assemble(
                    z0, toc0, q_e, bank0, embed_fn, k_deep=k_d, k_skim=k_s, mode="full", training_ste=False,
                    query_input_ids=rq0, salience_route_weight=sal_w,
                )
                li = torch.cat((a_w.prefix_embeds.to(llm_dtype), embed_fn(q0).to(llm_dtype)), dim=1)
                _ = wrapper.llm(inputs_embeds=li, attention_mask=torch.ones((1, li.shape[1]), device=device, dtype=torch.long))
            _sync_device(device)
            t_q0 = time.perf_counter()
            for _ in range(n_bench):
                q_e = embed_fn(rq0).mean(dim=1)
                _, asm_i = wrapper.lslc.route_and_assemble(
                    z0, toc0, q_e, bank0, embed_fn, k_deep=k_d, k_skim=k_s, mode="full", training_ste=False,
                    query_input_ids=rq0, salience_route_weight=sal_w,
                )
                pref_toks = int(asm_i.prefix_embeds.shape[1])
                li = torch.cat((asm_i.prefix_embeds.to(llm_dtype), embed_fn(q0).to(llm_dtype)), dim=1)
                _ = wrapper.llm(inputs_embeds=li, attention_mask=torch.ones((1, li.shape[1]), device=device, dtype=torch.long))
            _sync_device(device)
            return ((time.perf_counter() - t_q0) * 1000.0) / n_bench, pref_toks

        def _time_rag_query(k_d: int) -> tuple[float, int]:
            pref_toks = 0
            for _ in range(n_warmup):
                rr = bm25_bench.route(q_text0, k_deep=k_d, k_skim=0, device=device)
                rt = torch.cat([page_tensors0[i] for i in rr.deep_idx[0].tolist()], dim=0).unsqueeze(0)
                ri = torch.cat((embed_fn(rt).to(llm_dtype), embed_fn(q0).to(llm_dtype)), dim=1)
                _ = wrapper.llm(inputs_embeds=ri, attention_mask=torch.ones((1, ri.shape[1]), device=device, dtype=torch.long))
            _sync_device(device)
            t_r0 = time.perf_counter()
            for _ in range(n_bench):
                rr = bm25_bench.route(q_text0, k_deep=k_d, k_skim=0, device=device)
                rt = torch.cat([page_tensors0[i] for i in rr.deep_idx[0].tolist()], dim=0).unsqueeze(0)
                pref_toks = int(rt.shape[1])
                ri = torch.cat((embed_fn(rt).to(llm_dtype), embed_fn(q0).to(llm_dtype)), dim=1)
                _ = wrapper.llm(inputs_embeds=ri, attention_mask=torch.ones((1, ri.shape[1]), device=device, dtype=torch.long))
            _sync_device(device)
            return ((time.perf_counter() - t_r0) * 1000.0) / n_bench, pref_toks

        lslc_query_ms, _ = _time_lslc_query(k_rag, wrapper.cfg.k_skim, timing_salience_route_weight)
        lslc_dense_ms, _ = _time_lslc_query(k_rag, wrapper.cfg.k_skim, 0.0)
        rag_query_ms, _ = _time_rag_query(k_rag)

        raw_inp = torch.cat((raw_flat_emb, embed_fn(q0).to(llm_dtype)), dim=1)
        for _ in range(n_warmup):
            _ = wrapper.llm(inputs_embeds=raw_inp, attention_mask=torch.ones((1, raw_inp.shape[1]), device=device, dtype=torch.long))
        _sync_device(device)
        t_q_raw0 = time.perf_counter()
        for _ in range(n_bench):
            _ = wrapper.llm(inputs_embeds=raw_inp, attention_mask=torch.ones((1, raw_inp.shape[1]), device=device, dtype=torch.long))
        _sync_device(device)
        raw_query_ms = ((time.perf_counter() - t_q_raw0) * 1000.0) / n_bench

        breakeven: dict[str, dict[str, float]] = {}
        for q_count in (1, 10, 100):
            lslc_tot = end_to_end_edit_ms + q_count * lslc_query_ms
            rag_tot = rag_edit_ms + q_count * rag_query_ms
            raw_tot = q_count * raw_query_ms
            breakeven[f"Q={q_count}"] = {
                "lslc_total_ms": round(lslc_tot, 3),
                "rag_bm25_total_ms": round(rag_tot, 3),
                "full_context_total_ms": round(raw_tot, 3),
                "speedup_vs_rag_x": round(rag_tot / max(1e-6, lslc_tot), 3),
                "speedup_vs_full_x": round(raw_tot / max(1e-6, lslc_tot), 3),
            }

        q_star_vs_rag = (
            0
            if (rag_query_ms > lslc_query_ms and end_to_end_edit_ms <= rag_edit_ms)
            else (
                math.ceil((end_to_end_edit_ms - rag_edit_ms) / (rag_query_ms - lslc_query_ms))
                if rag_query_ms > lslc_query_ms
                else None
            )
        )
        q_star_vs_full = (
            math.ceil(end_to_end_edit_ms / (raw_query_ms - lslc_query_ms))
            if raw_query_ms > lslc_query_ms
            else None
        )

        by_k_deep: dict[str, dict[str, object]] = {}
        for kd_req in (2, 8):
            k_d = min(kd_req, num_p0)
            k_s = min(wrapper.cfg.k_skim, max(0, num_p0 - k_d))
            lslc_k_ms, lslc_pref_toks = _time_lslc_query(k_d, k_s, timing_salience_route_weight)
            rag_k_ms, rag_pref_toks = _time_rag_query(k_d)

            q_star_k = (
                0
                if (rag_k_ms > lslc_k_ms and end_to_end_edit_ms <= rag_edit_ms)
                else (
                    math.ceil((end_to_end_edit_ms - rag_edit_ms) / (rag_k_ms - lslc_k_ms))
                    if rag_k_ms > lslc_k_ms
                    else None
                )
            )
            by_q_k = {
                f"Q={qc}": {
                    "lslc_total_ms": round(end_to_end_edit_ms + qc * lslc_k_ms, 3),
                    "rag_bm25_total_ms": round(rag_edit_ms + qc * rag_k_ms, 3),
                    "speedup_vs_rag_x": round((rag_edit_ms + qc * rag_k_ms) / max(1e-6, end_to_end_edit_ms + qc * lslc_k_ms), 3),
                }
                for qc in (1, 10, 100)
            }
            by_k_deep[f"k_deep={kd_req}"] = {
                "k_deep": k_d,
                "k_skim": k_s,
                "lslc_prefix_tokens": lslc_pref_toks,
                "rag_prefix_tokens": rag_pref_toks,
                "lslc_per_query_ms": round(lslc_k_ms, 3),
                "rag_per_query_ms": round(rag_k_ms, 3),
                "per_query_speedup_vs_rag_x": round(rag_k_ms / max(1e-6, lslc_k_ms), 3),
                "q_star_vs_rag_bm25": q_star_k,
                "by_q": by_q_k,
            }
    finally:
        if merged_n > 0:
            unmerge_lora(wrapper.llm)

    return {
        "oracle_abc_symbolic": oracle_symbolic,
        "oracle_abc_budget": oracle_budget,
        "page_scaling": page_scaling,
        "q_breakeven": {
            "lora_merged_for_timing": merged_n > 0,
            "timing_salience_route_weight": timing_salience_route_weight,
            "single_page_reencode_ms": round(single_page_reencode_ms, 3),
            "local_write_ms": round(local_write_ms, 3),
            "contextual_encode_ms": round(contextual_encode_ms, 3),
            "end_to_end_edit_ms": round(end_to_end_edit_ms, 3),
            "lslc_per_query_ms": round(lslc_query_ms, 3),
            "lslc_per_query_ms_dense_w0": round(lslc_dense_ms, 3),
            "rag_edit_ms": round(rag_edit_ms, 3),
            "rag_per_query_ms": round(rag_query_ms, 3),
            "full_context_per_query_ms": round(raw_query_ms, 3),
            "q_star_vs_rag_bm25": q_star_vs_rag,
            "q_star_vs_full_context": q_star_vs_full,
            "by_q": breakeven,
            "by_k_deep": by_k_deep,
        },
        "analytical_memory_4k": compute_analytical_memory_stats(wrapper.cfg, num_tokens=4096),
        "sample_page_size": t_page_size,
    }


@torch.no_grad()
def evaluate_router_stress_p64(
    wrapper: FrozenQwenLSLCWrapper,
    tokenizer,
    val_pool: list[int],
    device: torch.device,
    *,
    is_mock: bool,
    seed: int = 9091,
    num_pages_override: int | None = None,
    n_entities_override: int | None = None,
) -> dict[str, object]:
    """Stress-test router & SalienceGate at P=64/P=128 with 12-20 similar-entity pages (mix of headerless & record)."""
    from lslc.datasets import _encode_text_or_mock, _sample_background_page

    rng = random.Random(seed)
    embed_fn = wrapper.llm.get_input_embeddings()
    amp_dtype, _ = select_amp_dtype(device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    vocab_size = len(embed_fn.weight)
    num_p = num_pages_override if num_pages_override is not None else (16 if is_mock else 64)
    n_entities = n_entities_override if n_entities_override is not None else (8 if is_mock else 12)
    page_size = wrapper.cfg.page_size
    wrapper.lslc.eval()

    chosen_pages = rng.sample(range(num_p), n_entities)
    filler_vocab = val_pool if val_pool else list(range(2, max(16, vocab_size - 2)))
    pages: list[list[int]] = [
        _sample_background_page(rng, page_size, val_pool, filler_vocab)
        for _ in range(num_p)
    ]

    queries: list[tuple[int, str, str]] = []
    for idx_e, p_idx in enumerate(chosen_pages):
        proj = f"StressNode-{idx_e:02d}"
        val = f"stress-tier-{idx_e:02d}"
        fmt = "record" if (idx_e % 2 == 0) else ("eval_prose" if idx_e % 4 == 1 else "eval_code")
        stmt = AdversarialLivingBookDataset._format_fact_stmt("symbolic", fmt=fmt, proj=proj, val=val)
        toks = _encode_text_or_mock(stmt, tokenizer, vocab_size)
        offset = rng.randint(0, max(0, page_size - len(toks)))
        n_w = min(len(toks), page_size - offset)
        pages[p_idx][offset : offset + n_w] = toks[:n_w]
        queries.append((p_idx, f"What is the approved deployment state for {proj}?", val))

    page_tensor = torch.tensor(pages, dtype=torch.long, device=device).unsqueeze(0)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
        z_book, bank, toc = wrapper.lslc.encode_book(page_tensor, embed_fn, use_exact_bank=True, vocab_size=vocab_size)

    n_bank = 0
    w0_top1 = 0
    w0_top2 = 0
    w2_top1 = 0
    w2_top2 = 0
    dummy_fb = torch.empty((0,), dtype=torch.long)

    for p_idx, q_txt, ans_str in queries:
        rq_ids = torch.tensor(_encode_text_or_mock(q_txt, tokenizer, vocab_size), dtype=torch.long, device=device).unsqueeze(0)
        ans_variants = _answer_token_variants(tokenizer, ans_str, dummy_fb)
        bank_set = set(int(x) for x in bank.token_ids[0, p_idx].tolist()) if bank is not None else set()
        if _any_variant_present(bank_set, ans_variants):
            n_bank += 1

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            q_emb = embed_fn(rq_ids).mean(dim=1)
            r0, _ = wrapper.lslc.route_and_assemble(
                z_book, toc, q_emb, bank, embed_fn, mode="full", training_ste=False, query_input_ids=rq_ids, salience_route_weight=0.0
            )
            r2, _ = wrapper.lslc.route_and_assemble(
                z_book, toc, q_emb, bank, embed_fn, mode="full", training_ste=False, query_input_ids=rq_ids, salience_route_weight=2.0
            )
        if int(torch.argmax(r0.logits[0], dim=-1).item()) == p_idx:
            w0_top1 += 1
        if p_idx in r0.deep_idx[0].tolist():
            w0_top2 += 1
        if int(torch.argmax(r2.logits[0], dim=-1).item()) == p_idx:
            w2_top1 += 1
        if p_idx in r2.deep_idx[0].tolist():
            w2_top2 += 1

    d = max(1, len(queries))
    return {
        "num_pages": num_p,
        "num_entity_pages": n_entities,
        "in_exact_bank_rate": round(n_bank / d, 4),
        "dense_w0_top1_recall": round(w0_top1 / d, 4),
        "dense_w0_top2_recall": round(w0_top2 / d, 4),
        "hybrid_w2_top1_recall": round(w2_top1 / d, 4),
        "hybrid_w2_top2_recall": round(w2_top2 / d, 4),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.mock_llm:
        mock_vocab_size = 512
        cfg = LSLCConfig.tiny_mock(
            use_contextual_edit_encoding=args.contextual_edit,
            contextual_edit_layer=1,
            lambda_cond_b_distill=args.lambda_cond_b,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode=args.router_mode,
            k_exact=min(args.k_exact, 48),
            salience_span_len=args.salience_span_len,
        )
        llm = TinyMockCausalLM(vocab_size=mock_vocab_size, d_llm=cfg.d_llm, num_layers=cfg.num_llm_layers)
        tokenizer = _MockTokenizer(vocab_size=mock_vocab_size)
        device = torch.device("cpu")
        train_pool = [2 + (i % (mock_vocab_size - 2)) for i in range(4096)]
        val_pool = [2 + ((i * 7) % (mock_vocab_size - 2)) for i in range(2048)]
        data_source = "mock-synthetic-pool"
        num_pages = 8
    else:
        cfg = LSLCConfig.from_model_name(
            args.model,
            use_contextual_edit_encoding=args.contextual_edit,
            lambda_cond_b_distill=args.lambda_cond_b,
            salience_mode="learned_surprisal",
            delta_pooling_mode="cross_attn",
            router_mode=args.router_mode,
            k_exact=args.k_exact,
            salience_span_len=args.salience_span_len,
        )
        cfg.contextual_edit_layer = cfg.num_llm_layers // 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        llm, tokenizer = load_frozen_qwen3(cfg, load_in_4bit=False, device=device)
        train_pool, val_pool, data_source = _load_wikitext_pools(tokenizer)
        num_pages = 16
    if args.router_mode == "content_attn":
        cfg.lambda_route = 1.0

    bge_toc = None
    if args.use_bge_toc:
        from colab_run_v3_gerbang4_community import _PlugInBiEncoderTOC

        bge_toc = _PlugInBiEncoderTOC(args.bge_model, device=device, is_mock=args.mock_llm)

    wrapper = FrozenQwenLSLCWrapper(cfg, llm).to(device)
    ckpt_info = None
    if args.init_checkpoint:
        ckpt_info = _load_lslc_checkpoint(wrapper.lslc, args.init_checkpoint, device)
    if args.reset_salience_gate:
        wrapper.lslc.salience_gate = SalienceGate(cfg).to(device)
        print("[CKPT] Reset SalienceGate weights for clean learned_surprisal training", flush=True)
    if args.reset_router and not args.skip_stage1:
        wrapper.lslc.router = QueryToTOCRouter(cfg.d_llm, d_route=cfg.d_route, router_mode=cfg.router_mode).to(device)
        if wrapper.lslc.projector.toc_slot_attn is not None:
            wrapper.lslc.projector.toc_slot_attn.reset_parameters()
        print("[CKPT] Reset QueryToTOCRouter + toc_slot_attn for clean content_attn training", flush=True)

    if args.lora_r > 0:
        lora_targets = ("qkv", "out") if args.mock_llm else ("q_proj", "k_proj", "v_proj", "o_proj")
        replaced_lora = inject_lora(wrapper.llm, r=args.lora_r, alpha=args.lora_alpha, target_modules=lora_targets)
        wrapper.llm.config.use_cache = False
        set_lora_enabled(wrapper.llm, True)
        lora_trainable = sum(p.numel() for p in lora_parameters(wrapper.llm))
        lora_info: dict[str, object] = {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "lr": args.lora_lr,
            "lm_reg_prob": args.lm_reg_prob,
            "num_modules": len(replaced_lora),
            "trainable_params": lora_trainable,
        }
        print(f"[LoRA] Injected r={args.lora_r} alpha={args.lora_alpha} into {len(replaced_lora)} layers ({lora_trainable} params)", flush=True)
    else:
        lora_info = {"r": 0}

    vocab_size = len(llm.get_input_embeddings().weight)
    page_size = cfg.page_size
    grad_accum = 1 if args.mock_llm else args.grad_accum
    ppl_chunks = 8 if args.mock_llm else 32
    lm_ppl_pre = evaluate_heldout_lm_ppl(wrapper.llm, val_pool, device, max_chunks=ppl_chunks)
    ood_ppl_pre = evaluate_ood_forgetting_ppl(wrapper.llm, tokenizer, device)
    print(f"[PPL pre] wiki={lm_ppl_pre} ood={ood_ppl_pre['overall']}", flush=True)

    eval_budget = list(
        AdversarialLivingBookDataset(
            num_episodes=args.eval_episodes,
            num_pages=num_pages,
            page_size=page_size,
            num_edits=3,
            mode="budget",
            split="eval",
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            seed=args.seed + 40,
            background_token_pool=val_pool,
        )
    )
    eval_symbolic = list(
        AdversarialLivingBookDataset(
            num_episodes=args.eval_episodes,
            num_pages=num_pages,
            page_size=page_size,
            num_edits=5,
            mode="symbolic",
            split="eval",
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            seed=args.seed + 41,
            background_token_pool=val_pool,
        )
    )
    eval_headerless = list(
        AdversarialLivingBookDataset(
            num_episodes=min(24, args.eval_episodes),
            num_pages=num_pages,
            page_size=page_size,
            num_edits=5,
            mode="symbolic",
            split="eval",
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            seed=args.seed + 42,
            background_token_pool=val_pool,
            style="headerless_eval",
        )
    )
    oracle_symbolic_eps: list = []
    oracle_budget_eps: list = []
    for k_val in (1, 2, 3, 5):
        n_k = max(2, args.oracle_episodes // 4)
        oracle_symbolic_eps.extend(
            list(
                AdversarialLivingBookDataset(
                    num_episodes=n_k,
                    num_pages=num_pages,
                    page_size=page_size,
                    num_edits=k_val,
                    mode="symbolic",
                    split="eval",
                    tokenizer=tokenizer,
                    vocab_size=vocab_size,
                    seed=args.seed + 100 + k_val,
                    background_token_pool=val_pool,
                )
            )
        )
        oracle_budget_eps.extend(
            list(
                AdversarialLivingBookDataset(
                    num_episodes=n_k,
                    num_pages=num_pages,
                    page_size=page_size,
                    num_edits=k_val,
                    mode="budget",
                    split="eval",
                    tokenizer=tokenizer,
                    vocab_size=vocab_size,
                    seed=args.seed + 200 + k_val,
                    background_token_pool=val_pool,
                )
            )
        )

    trace_max = 8 if args.mock_llm else min(24, len(eval_symbolic))
    trace_pre_stage1 = trace_identifier_pipeline(
        wrapper, tokenizer, eval_symbolic, device, max_episodes=trace_max, salience_route_weight=args.salience_route_weight
    )
    print(
        "[TRACE pre_stage1]",
        {
            k: {m: v[m] for m in ("in_source_page_rate", "in_exact_bank_rate", "router_hit_rate", "in_assembled_prefix_rate", "in_generated_answer_rate")}
            for k, v in trace_pre_stage1.items()
        },
        flush=True,
    )

    if args.skip_stage1:
        stage1_stats = None
        trace_post_stage1 = None
        trace_headerless_post_stage1 = None
        router_stress_p64 = None
        router_stress_p128 = None
    else:
        n_train_s1 = 24 if args.mock_llm else 400
        s1_niah = list(
            SyntheticExactSemanticNIAHDataset(
                num_episodes=max(8, n_train_s1 // 2),
                num_pages=num_pages,
                page_size=page_size,
                vocab_size=vocab_size,
                mode="mixed",
                tokenizer=tokenizer,
                seed=args.seed + 1,
                background_token_pool=train_pool,
            )
        )
        s1_budget = [
            _build_stage1_episode_view(ep)
            for ep in AdversarialLivingBookDataset(
                num_episodes=n_train_s1,
                num_pages=num_pages,
                page_size=page_size,
                num_edits=3,
                mode="budget",
                split="train",
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                seed=args.seed + 2,
                background_token_pool=train_pool,
                style=args.train_style,
            )
        ]
        s1_symbolic = [
            _build_stage1_episode_view(ep)
            for ep in AdversarialLivingBookDataset(
                num_episodes=n_train_s1,
                num_pages=num_pages,
                page_size=page_size,
                num_edits=3,
                mode="symbolic",
                split="train",
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                seed=args.seed + 3,
                background_token_pool=train_pool,
                style=args.train_style,
            )
        ]
        stage1_stats = _train_v3_stage(
            "Stage1-MultiSchema",
            wrapper,
            s1_niah + s1_budget + s1_symbolic,
            steps=args.stage1_steps,
            grad_accum=grad_accum,
            lr=args.lr,
            device=device,
            use_edits=False,
            lambda_cond_b=0.0,
            distill_mode="none",
            freeze_toc_proj=False,
            lora_lr=args.lora_lr,
            lm_reg_pool=train_pool,
            lm_reg_prob=(args.lm_reg_prob if args.lora_r > 0 else 0.0),
            seed=args.seed + 10,
        )
        trace_post_stage1 = trace_identifier_pipeline(
            wrapper, tokenizer, eval_symbolic, device, max_episodes=trace_max, salience_route_weight=args.salience_route_weight
        )
        print(
            "[TRACE post_stage1]",
            {
                k: {m: v[m] for m in ("in_source_page_rate", "in_exact_bank_rate", "router_hit_rate", "in_assembled_prefix_rate", "in_generated_answer_rate")}
                for k, v in trace_post_stage1.items()
            },
            flush=True,
        )

        trace_headerless_post_stage1 = trace_identifier_pipeline(
            wrapper, tokenizer, eval_headerless, device, max_episodes=trace_max, salience_route_weight=args.salience_route_weight
        )
        router_stress_p64 = evaluate_router_stress_p64(
            wrapper, tokenizer, val_pool, device, is_mock=args.mock_llm, seed=args.seed + 77
        )
        print(f"[STRESS P={router_stress_p64['num_pages']}] {router_stress_p64}", flush=True)
        router_stress_p128 = evaluate_router_stress_p64(
            wrapper,
            tokenizer,
            val_pool,
            device,
            is_mock=args.mock_llm,
            seed=args.seed + 78,
            num_pages_override=24 if args.mock_llm else 128,
            n_entities_override=10 if args.mock_llm else 20,
        )
        print(f"[STRESS P={router_stress_p128['num_pages']}] {router_stress_p128}", flush=True)

    stage2_stats = None
    trace_post_stage2 = None
    stage3_budget = None
    stage3_symbolic = None

    if not args.stage1_only:
        n_train_s2 = 16 if args.mock_llm else 120
        s2_budget = list(
            AdversarialLivingBookDataset(
                num_episodes=n_train_s2,
                num_pages=num_pages,
                page_size=page_size,
                num_edits=3,
                mode="budget",
                split="train",
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                seed=args.seed + 20,
                background_token_pool=train_pool,
                style=args.train_style,
            )
        )
        s2_symbolic = list(
            AdversarialLivingBookDataset(
                num_episodes=n_train_s2,
                num_pages=num_pages,
                page_size=page_size,
                num_edits=4,
                mode="symbolic",
                split="train",
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                seed=args.seed + 21,
                background_token_pool=train_pool,
                style=args.train_style,
            )
        )
        stage2_stats = _train_v3_stage(
            "Stage2-DeltaDistill",
            wrapper,
            s2_budget + s2_symbolic,
            steps=args.stage2_steps,
            grad_accum=grad_accum,
            lr=args.lr,
            device=device,
            use_edits=True,
            lambda_cond_b=args.lambda_cond_b,
            distill_mode=args.distill_mode,
            freeze_toc_proj=True,
            lora_lr=args.lora_lr,
            lm_reg_pool=train_pool,
            lm_reg_prob=(args.lm_reg_prob if args.lora_r > 0 else 0.0),
            seed=args.seed + 30,
        )
        trace_post_stage2 = trace_identifier_pipeline(
            wrapper, tokenizer, eval_symbolic, device, max_episodes=trace_max, salience_route_weight=args.salience_route_weight
        )
        stage3_budget = evaluate_five_way_adversarial(
            wrapper, tokenizer, eval_budget, device, salience_route_weight=args.salience_route_weight, bge_toc=bge_toc
        )
        stage3_symbolic = evaluate_five_way_adversarial(
            wrapper, tokenizer, eval_symbolic, device, salience_route_weight=args.salience_route_weight, bge_toc=bge_toc
        )

    lm_ppl_post = evaluate_heldout_lm_ppl(wrapper.llm, val_pool, device, max_chunks=ppl_chunks)
    ood_ppl_post = evaluate_ood_forgetting_ppl(wrapper.llm, tokenizer, device)
    print(f"[PPL post] wiki={lm_ppl_post} ood={ood_ppl_post['overall']}", flush=True)

    if args.save_checkpoint:
        _save_lslc_checkpoint(wrapper.lslc, args.save_checkpoint)
    if args.save_lora_checkpoint and args.lora_r > 0:
        lora_path = Path(args.save_lora_checkpoint)
        lora_path.parent.mkdir(parents=True, exist_ok=True)
        sd_lora = lora_state_dict(wrapper.llm)
        if lora_path.suffix == ".safetensors":
            from safetensors.torch import save_file

            save_file(sd_lora, str(lora_path))
        else:
            torch.save(sd_lora, str(lora_path))
        print(f"[CKPT] Saved LoRA weights to {lora_path}", flush=True)

    stage4_diag = evaluate_oracle_abc_and_q_breakeven(
        wrapper,
        tokenizer,
        oracle_symbolic_eps,
        oracle_budget_eps,
        device,
        is_mock=args.mock_llm,
        salience_route_weight=args.salience_route_weight,
        timing_salience_route_weight=args.timing_salience_route_weight,
    )

    payload = {
        "version": "3.0.0-clean-phase1",
        "model": "TinyMockCausalLM" if args.mock_llm else args.model,
        "mock_llm": args.mock_llm,
        "stage1_only": args.stage1_only,
        "device": str(device),
        "data_source": data_source,
        "init_checkpoint": ckpt_info,
        "lora": lora_info,
        "lm_ppl_pre": lm_ppl_pre,
        "lm_ppl_post": lm_ppl_post,
        "ood_ppl_pre": ood_ppl_pre,
        "ood_ppl_post": ood_ppl_post,
        "config": {
            "use_contextual_edit_encoding": cfg.use_contextual_edit_encoding,
            "contextual_edit_layer": cfg.contextual_edit_layer,
            "distill_mode": args.distill_mode,
            "lambda_cond_b_distill": args.lambda_cond_b,
            "salience_mode": cfg.salience_mode,
            "delta_pooling_mode": cfg.delta_pooling_mode,
            "router_mode": cfg.router_mode,
            "k_exact": cfg.k_exact,
            "salience_span_len": cfg.salience_span_len,
            "use_bge_toc": bool(bge_toc is not None),
            "bge_model": None if bge_toc is None else bge_toc.model_name,
            "train_style": args.train_style,
            "salience_route_weight": args.salience_route_weight,
            "timing_salience_route_weight": args.timing_salience_route_weight,
            "num_pages": num_pages,
            "page_size": page_size,
        },
        "identifier_trace": {
            "pre_stage1": trace_pre_stage1,
            "post_stage1": trace_post_stage1,
            "post_stage1_headerless": trace_headerless_post_stage1,
            "post_stage2": trace_post_stage2,
        },
        "router_stress_p64": router_stress_p64,
        "router_stress_p128": router_stress_p128,
        "stage1_train": stage1_stats,
        "stage2_train": stage2_stats,
        "stage3_eval_holdout_budget": stage3_budget,
        "stage3_eval_holdout_symbolic": stage3_symbolic,
        "stage4_oracle_and_breakeven": stage4_diag,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote v3 Phase 1 results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
