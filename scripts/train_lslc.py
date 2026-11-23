"""CLI training entrypoint for LSLC MVE (supports Colab T4/L4/A100 and local --dry-run-mock)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lslc.config import LSLCConfig, select_amp_dtype
from lslc.datasets import LivingBookStateOverrideDataset, SyntheticExactSemanticNIAHDataset
from lslc.trainer import FrozenQwenLSLCWrapper, TinyMockCausalLM, load_frozen_qwen3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSLC modules over a frozen Qwen3 backbone.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="HF model name or mock.")
    parser.add_argument("--track", type=int, default=2, choices=[1, 2, 3, 4], help="LSLC ablation track (1..4).")
    parser.add_argument("--smoke-test", action="store_true", help="Run mandatory 20-step smoke test.")
    parser.add_argument("--max-steps", type=int, default=200, help="Total optimizer steps.")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate.")
    parser.add_argument("--num-pages", type=int, default=4, help="Pages per episode during training.")
    parser.add_argument("--num-episodes", type=int, default=64, help="Synthetic/dataset episodes per epoch.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load frozen Qwen3-4B in NF4 on 15GB T4.")
    parser.add_argument("--dry-run-mock", action="store_true", help="Offline CPU/MPS check with TinyMockCausalLM.")
    parser.add_argument("--output-dir", type=str, default="checkpoints/lslc_run", help="Directory for weights & logs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args, _ = parser.parse_known_args(argv)
    if args.smoke_test:
        args.max_steps = min(args.max_steps, 20)
    return args


def _pad_or_trim_batch(episodes, device: torch.device):
    # Batch size 1 cleanly supports variable query/answer lengths without padding artifacts
    ep = episodes[0]
    page_ids = ep.context_pages.unsqueeze(0).to(device)
    query_ids = ep.query_ids.unsqueeze(0).to(device)
    answer_ids = ep.answer_ids.unsqueeze(0).to(device)
    target_page = torch.tensor([ep.target_page_idx], dtype=torch.long, device=device)

    edit_page = None
    edit_ids = None
    if ep.edit_pages:
        last_p, last_toks = ep.edit_pages[-1]
        edit_page = torch.tensor([last_p], dtype=torch.long, device=device)
        edit_ids = last_toks.unsqueeze(0).to(device)

    return page_ids, query_ids, answer_ids, target_page, edit_page, edit_ids


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    if args.dry_run_mock:
        device = torch.device("cpu")
        cfg = LSLCConfig.tiny_mock(lr=args.lr)
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
        cfg = LSLCConfig.from_model_name(args.model, lr=args.lr)
        llm, tokenizer = load_frozen_qwen3(cfg, load_in_4bit=args.load_in_4bit, device=device)

    wrapper = FrozenQwenLSLCWrapper(cfg, llm)
    if not args.dry_run_mock:
        wrapper.lslc.to(device)

    if args.track == 4:
        dataset = LivingBookStateOverrideDataset(
            num_episodes=args.num_episodes,
            num_pages=max(2, args.num_pages),
            page_size=cfg.page_size,
            tokenizer=tokenizer,
            seed=args.seed,
        )
    else:
        dataset = SyntheticExactSemanticNIAHDataset(
            num_episodes=args.num_episodes,
            num_pages=args.num_pages,
            page_size=cfg.page_size,
            tokenizer=tokenizer,
            seed=args.seed,
        )

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    amp_dtype, use_scaler = select_amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if device.type == "cuda" else None

    trainable_params = [p for p in wrapper.lslc.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    warmup_steps = max(1, int(args.max_steps * cfg.warmup_ratio))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, args.max_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_exact = args.track >= 2

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    logs: list[dict] = []
    wrapper.lslc.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()

    step_iter = iter(loader)
    for opt_step in range(1, args.max_steps + 1):
        accum_total = 0.0
        last_out = None
        for _ in range(args.grad_accum):
            try:
                batch = next(step_iter)
            except StopIteration:
                step_iter = iter(loader)
                batch = next(step_iter)

            page_ids, query_ids, answer_ids, target_page, edit_page, edit_ids = _pad_or_trim_batch(batch, device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                out = wrapper.forward_step(
                    page_token_ids=page_ids,
                    query_ids=query_ids,
                    answer_ids=answer_ids,
                    target_page_idx=target_page if args.track >= 3 else None,
                    edit_page_idx=edit_page if args.track == 4 else None,
                    edit_token_ids=edit_ids if args.track == 4 else None,
                    use_exact_bank=use_exact,
                    run_matryoshka=(args.track >= 2),
                )
                loss_scaled = out.loss_total / args.grad_accum

            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            accum_total += float(out.loss_total.detach().item()) / args.grad_accum
            last_out = out

        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip).item())

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        rec = {
            "step": opt_step,
            "loss_total": round(accum_total, 5),
            "loss_ntp": round(float(last_out.loss_ntp.item()), 5),
            "loss_rec": round(float(last_out.loss_rec.item()), 5),
            "loss_mat": round(float(last_out.loss_mat.item()), 5),
            "loss_sparse": round(float(last_out.loss_sparse.item()), 5),
            "loss_route": round(float(last_out.loss_route.item()), 5),
            "mean_gate": round(last_out.mean_gate, 5),
            "grad_norm": round(grad_norm, 5),
            "matryoshka_mode": last_out.matryoshka_mode,
        }
        logs.append(rec)
        if opt_step == 1 or opt_step % 5 == 0 or opt_step == args.max_steps:
            print(
                f"[Track {args.track}] step={opt_step}/{args.max_steps} "
                f"loss={rec['loss_total']:.4f} ntp={rec['loss_ntp']:.4f} "
                f"gate={rec['mean_gate']:.4f} grad={rec['grad_norm']:.4f}"
            )

    elapsed = time.perf_counter() - t0
    peak_vram_mib = (torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0

    torch.save({"config": cfg.__dict__, "state_dict": wrapper.lslc.state_dict()}, out_dir / "lslc_weights.pt")
    summary = {
        "model": cfg.model_name,
        "track": args.track,
        "steps": args.max_steps,
        "trainable_params": wrapper.lslc.trainable_parameter_count(),
        "elapsed_sec": round(elapsed, 3),
        "peak_vram_mib": round(peak_vram_mib, 2),
        "history": logs,
    }
    (out_dir / "train_log.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved checkpoint & train_log.json to {out_dir} (trainable={summary['trainable_params']:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
