"""Standardized evaluation suite for Tracks 0A-0E and Tracks 1-4 (PPL, Exact/Semantic NIAH, Delta-Edit)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lslc.config import LSLCConfig, compute_analytical_memory_stats
from lslc.datasets import LivingBookStateOverrideDataset, SyntheticExactSemanticNIAHDataset
from lslc.trainer import FrozenQwenLSLCWrapper, TinyMockCausalLM, load_frozen_qwen3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LSLC or baseline tracks.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--track", type=str, default="3", choices=["0a", "0b", "0c", "0d", "0e", "1", "2", "3", "4"])
    parser.add_argument("--checkpoint", type=str, default="", help="Optional path to lslc_weights.pt")
    parser.add_argument("--num-pages", type=int, default=4)
    parser.add_argument("--num-episodes", type=int, default=16)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--dry-run-mock", action="store_true")
    parser.add_argument("--output-dir", type=str, default="evals/latest")
    args, _ = parser.parse_known_args(argv)
    return args


@torch.no_grad()
def _score_episode_prefix(llm, prefix_embeds: torch.Tensor, query_ids: torch.Tensor, answer_ids: torch.Tensor):
    embed_fn = llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    turn_ids = torch.cat((query_ids, answer_ids), dim=1)
    turn_embeds = embed_fn(turn_ids).to(llm_dtype)
    prefix_cast = prefix_embeds.to(llm_dtype)
    inputs_embeds = torch.cat((prefix_cast, turn_embeds), dim=1)
    attn_mask = torch.ones((1, inputs_embeds.shape[1]), device=inputs_embeds.device, dtype=torch.long)

    logits = llm(inputs_embeds=inputs_embeds, attention_mask=attn_mask).logits
    q_end = prefix_embeds.shape[1] + query_ids.shape[1]
    ans_len = answer_ids.shape[1]
    ans_logits = logits[:, q_end - 1 : q_end - 1 + ans_len, :]
    loss = F.cross_entropy(ans_logits.reshape(-1, ans_logits.shape[-1]), answer_ids.reshape(-1))
    pred_ids = ans_logits.argmax(dim=-1)
    exact_match = float(torch.equal(pred_ids, answer_ids))
    tok_acc = float((pred_ids == answer_ids).float().mean().item())
    return float(loss.item()), exact_match, tok_acc


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.dry_run_mock:
        device = torch.device("cpu")
        cfg = LSLCConfig.tiny_mock()
        llm = TinyMockCausalLM(
            vocab_size=512,
            d_llm=cfg.d_llm,
            num_layers=cfg.num_llm_layers,
            num_heads=4,
            num_kv_heads=cfg.num_kv_heads,
        )
        tokenizer = None
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = LSLCConfig.from_model_name(args.model)
        llm, tokenizer = load_frozen_qwen3(cfg, load_in_4bit=args.load_in_4bit, device=device)

    wrapper = FrozenQwenLSLCWrapper(cfg, llm)
    if not args.dry_run_mock:
        wrapper.lslc.to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        wrapper.lslc.load_state_dict(ckpt["state_dict"], strict=True)
    wrapper.eval()

    niah_ds = SyntheticExactSemanticNIAHDataset(
        num_episodes=args.num_episodes,
        num_pages=args.num_pages,
        page_size=cfg.page_size,
        tokenizer=tokenizer,
    )
    delta_ds = LivingBookStateOverrideDataset(
        num_episodes=max(6, args.num_episodes // 2),
        num_pages=max(2, args.num_pages),
        page_size=cfg.page_size,
        tokenizer=tokenizer,
    )

    embed_fn = llm.get_input_embeddings()
    exact_hits, sem_hits, losses = [], [], []
    use_exact = args.track not in ("1",)
    is_baseline_full = args.track in ("0a", "0b")

    for ep in niah_ds:
        pages = ep.context_pages.unsqueeze(0).to(device)
        q_ids = ep.query_ids.unsqueeze(0).to(device)
        a_ids = ep.answer_ids.unsqueeze(0).to(device)

        if is_baseline_full:
            prefix = embed_fn(pages.reshape(1, -1))
        else:
            z_book, exact_bank, toc = wrapper.lslc.encode_book(pages, embed_tokens_fn=embed_fn, use_exact_bank=use_exact)
            q_emb = embed_fn(q_ids).mean(dim=1)
            _, assembly = wrapper.lslc.route_and_assemble(
                z_book=z_book,
                toc_embeds=toc,
                query_embed=q_emb,
                exact_bank=exact_bank,
                embed_tokens_fn=embed_fn,
                mode="full",
                training_ste=False,
            )
            prefix = assembly.prefix_embeds

        nll, em, tok_acc = _score_episode_prefix(llm, prefix, q_ids, a_ids)
        losses.append(nll)
        if ep.metadata["needle_type"] == "exact":
            exact_hits.append(tok_acc)
        else:
            sem_hits.append(tok_acc)

    delta_override_acc, static_ret_acc = [], []
    for ep in delta_ds:
        pages = ep.context_pages.unsqueeze(0).to(device)
        q_ids = ep.query_ids.unsqueeze(0).to(device)
        a_ids = ep.answer_ids.unsqueeze(0).to(device)
        z_book, exact_bank, toc = wrapper.lslc.encode_book(pages, embed_tokens_fn=embed_fn, use_exact_bank=True)
        if ep.edit_pages:
            for p_idx, e_toks in ep.edit_pages:
                p_t = torch.tensor([p_idx], dtype=torch.long, device=device)
                z_book, _ = wrapper.lslc.apply_page_delta(z_book, p_t, e_toks.unsqueeze(0).to(device), embed_tokens_fn=embed_fn)
        toc = wrapper.lslc.projector.project_toc(z_book)
        q_emb = embed_fn(q_ids).mean(dim=1)
        _, assembly = wrapper.lslc.route_and_assemble(
            z_book=z_book,
            toc_embeds=toc,
            query_embed=q_emb,
            exact_bank=exact_bank,
            embed_tokens_fn=embed_fn,
            mode="full",
            training_ste=False,
        )
        _, _, tok_acc = _score_episode_prefix(llm, assembly.prefix_embeds, q_ids, a_ids)
        if ep.metadata["is_overridden_query"]:
            delta_override_acc.append(tok_acc)
        else:
            static_ret_acc.append(tok_acc)

    raw_toks = args.num_pages * cfg.page_size
    mem = compute_analytical_memory_stats(cfg, num_tokens=raw_toks, use_exact_bank=use_exact)
    mean_nll = sum(losses) / max(1, len(losses))
    report = {
        "track": args.track,
        "model": cfg.model_name,
        "num_tokens": raw_toks,
        "ppl_proxy": round(math.exp(min(20.0, mean_nll)), 4),
        "exact_niah_tok_acc": round(sum(exact_hits) / max(1, len(exact_hits)), 4),
        "semantic_niah_tok_acc": round(sum(sem_hits) / max(1, len(sem_hits)), 4),
        "delta_edit_tok_acc": round(sum(delta_override_acc) / max(1, len(delta_override_acc)), 4),
        "static_retention_tok_acc": round(sum(static_ret_acc) / max(1, len(static_ret_acc)), 4),
        "analytical_memory": mem,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
