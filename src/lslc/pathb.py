"""Shared Path (B) Architecture: Two-Bank Gated Mutable Discrete Span-Banks.

Single source of truth used identically across frozen Qwen3-1.7B and Llama-3.1-8B-Instruct:
  - ZERO `LayeredLatentEncoder`, ZERO `SpatialLatentProjector`, ZERO `Z_book`, ZERO untrained `hw_coord_embed`.
  - Lean `ContextualEditGateHead(hidden_dim=64)` + `SalienceGate` (span-16, k_exact=48) + BGE-small 4-span BF16.
  - Compact `SparseExactBank` storing ONLY `token_ids` (192 B/page before edit; +192 B/page `base_token_ids` once edited).
  - True Two-Bank prefix ordering: unedited Bank-1 pages deterministically FIRST, mutated Bank-2 pages LAST.
  - Span-aligned selective overwrite (Opsi B) with clean newline separator (no `'39Budget'` mid-span splicing).
  - Zero-training `heuristic_span16_tfidf_bank` baseline, strict word-to-number canonicalizer, LongMemEval `value_em`,
    and git/checkpoint provenance helpers.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LSLCConfig
from .encoder import SalienceGate, SparseExactBank
from .projector import RouterOutput
from .trainer import extract_contextual_edit_embeds


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Provenance Helpers
# ---------------------------------------------------------------------------


def get_git_provenance(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Return current git HEAD SHA and dirty flag for strict JSON provenance."""
    cwd = str(Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2])
    sha = "unknown"
    dirty: bool | None = None
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
        st = subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(st)
    except Exception:
        pass
    return {"git_sha": sha, "git_dirty": dirty}


def get_newline_sep_token_id(tokenizer) -> int | None:
    """Resolve a clean newline separator token ID for span-aligned discrete bank writes."""
    if tokenizer is None:
        return None
    try:
        ids = tokenizer.encode("\n", add_special_tokens=False)
        if ids:
            return int(ids[-1])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Strict Word-to-Number Canonicalization & Numeric Value EM
# ---------------------------------------------------------------------------

_W2N_UNITS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_W2N_TENS: dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_W2N_SCALES: dict[str, int] = {"hundred": 100}
_W2N_MULT_SUFFIX: dict[str, str] = {"thousand": "k", "million": "m", "billion": "b"}
_W2N_WORDS = set(_W2N_UNITS) | set(_W2N_TENS) | set(_W2N_SCALES)


def _parse_english_number_words(words: list[str]) -> int | None:
    """Strictly parse positive English integer word sequence (e.g. ['twenty', 'five'] -> 25, ['ninety'] -> 90)."""
    if not words:
        return None
    total = 0
    current = 0
    matched_any = False
    for w in words:
        if w == "and" and matched_any:
            continue
        if w in _W2N_UNITS:
            current += _W2N_UNITS[w]
            matched_any = True
        elif w in _W2N_TENS:
            current += _W2N_TENS[w]
            matched_any = True
        elif w == "hundred":
            current = (current if current > 0 else 1) * 100
            matched_any = True
        else:
            return None
    if not matched_any:
        return None
    total += current
    return total


def _replace_number_word_phrases(text: str) -> str:
    """Replace standalone English number-word phrases with digits (e.g., 'fifteen million' -> '15 million')."""
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?|[^a-z0-9\s]+|\s+", text.lower())
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok in _W2N_WORDS:
            phrase_words = [tok]
            j = i + 1
            while j < n:
                if tokens[j].isspace() or tokens[j] == "-":
                    if j + 1 < n and (tokens[j + 1] in _W2N_WORDS or tokens[j + 1] == "and"):
                        phrase_words.append(tokens[j + 1])
                        j += 2
                        continue
                    break
                break
            parsed = _parse_english_number_words(phrase_words)
            if parsed is not None:
                out.append(str(parsed))
                i = j
                continue
        out.append(tok)
        i += 1
    return "".join(out)


def canonicalize_w2n_answer(text: str) -> str:
    """Strict word-to-number canonicalizer (kept strictly separate from `strict_em` and `normalized_numeric_em`).

    Maps e.g. 'fifteen' -> '15', 'ninety' -> '90', 'twenty-five million' -> '25m', '$29,000,000' -> '29m'.
    """
    s = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL | re.IGNORECASE).strip().lower()
    if "\n" in s:
        s = s.splitlines()[0].strip()
    s = s.replace("$", "").replace("usd", "").replace(",", "").strip()
    s = re.sub(r"^(?:the\s+answer\s+is|answer\s*[:=])\s*", "", s).strip()
    s = s.rstrip(" .,:;\"'`")
    s_w2n = _replace_number_word_phrases(s).strip()

    m_word = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(thousand|million|billion)", s_w2n)
    if m_word:
        num = float(m_word.group(1))
        suffix = _W2N_MULT_SUFFIX[m_word.group(2)]
        num_str = str(int(num)) if num.is_integer() else f"{num:g}"
        return f"{num_str}{suffix}"

    m_short = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmb])", s_w2n)
    if m_short:
        num = float(m_short.group(1))
        num_str = str(int(num)) if num.is_integer() else f"{num:g}"
        return f"{num_str}{m_short.group(2)}"

    m_raw = re.fullmatch(r"(\d+(?:\.\d+)?)", s_w2n)
    if m_raw:
        num = float(m_raw.group(1))
        if num >= 1_000_000_000 and (num / 1_000_000_000).is_integer():
            return f"{int(num // 1_000_000_000)}b"
        if num >= 1_000_000 and (num / 1_000_000).is_integer():
            return f"{int(num // 1_000_000)}m"
        if num >= 1_000 and (num / 1_000).is_integer():
            return f"{int(num // 1_000)}k"
        return str(int(num)) if num.is_integer() else f"{num:g}"

    return re.sub(r"\s+", " ", s_w2n).strip()


def _extract_canonical_numbers(text: str) -> list[str]:
    """Extract canonical numeric values (including number words & K/M/B multipliers) from arbitrary QA text."""
    s = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL | re.IGNORECASE).lower()
    s = s.replace(",", "")
    s = _replace_number_word_phrases(s)
    found: list[str] = []
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(thousand|million|billion|[kmb]\b)?", s):
        num = float(m.group(1))
        unit = (m.group(2) or "").strip()
        if unit in _W2N_MULT_SUFFIX:
            unit = _W2N_MULT_SUFFIX[unit]
        if unit in ("k", "m", "b"):
            num_str = str(int(num)) if num.is_integer() else f"{num:g}"
            found.append(f"{num_str}{unit}")
        else:
            if num >= 1_000_000_000 and (num / 1_000_000_000).is_integer():
                found.append(f"{int(num // 1_000_000_000)}b")
            elif num >= 1_000_000 and (num / 1_000_000).is_integer():
                found.append(f"{int(num // 1_000_000)}m")
            elif num >= 1_000 and (num / 1_000).is_integer():
                found.append(f"{int(num // 1_000)}k")
            else:
                found.append(str(int(num)) if num.is_integer() else f"{num:g}")
    return found


def compute_value_em(pred: str, gold: str) -> float:
    """Numeric/entity exact-value EM preventing token-F1 inflation on wrong numbers (e.g. '45 min' vs '30 min')."""
    gold_nums = _extract_canonical_numbers(gold)
    if gold_nums:
        pred_nums = _extract_canonical_numbers(pred)
        if not pred_nums:
            return 0.0
        # Every canonical numeric value in gold must appear in pred and pred must not contradict with wrong sole number
        gold_set = set(gold_nums)
        pred_set = set(pred_nums)
        return 1.0 if gold_set == pred_set else 0.0

    # Non-numeric entity/phrase: strict normalized exact match or full entity match on concise extracted answer
    g_clean = re.sub(r"[^a-z0-9\s]", " ", canonicalize_w2n_answer(gold))
    g_clean = re.sub(r"\s+", " ", g_clean).strip()
    p_clean = re.sub(r"[^a-z0-9\s]", " ", canonicalize_w2n_answer(pred))
    p_clean = re.sub(r"\s+", " ", p_clean).strip()
    if not g_clean:
        return 0.0
    if p_clean == g_clean:
        return 1.0
    if len(g_clean) >= 3 and re.search(rf"(?<!\w){re.escape(g_clean)}(?!\w)", p_clean):
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Zero-Training Heuristic Span-16 TF-IDF Baseline (0 Trained Params)
# ---------------------------------------------------------------------------


@torch.no_grad()
def heuristic_span16_tfidf_bank(
    page_tensor: torch.Tensor,
    *,
    k_exact: int = 48,
    span_len: int = 16,
    special_token_ids: tuple[int, ...] | None = None,
) -> SparseExactBank:
    """Zero-training heuristic baseline: book-level TF-IDF token rarity + non-overlapping top span-16 extraction.

    Strictly 0 trained parameters. Returns compact Path (B) `SparseExactBank` (`token_ids` only).
    """
    bsz, num_p, seq_len = page_tensor.shape
    device = page_tensor.device
    k = min(int(k_exact), seq_len)
    s_len = min(int(span_len), k, seq_len)
    special_set = set(int(s) for s in (special_token_ids or ()))

    picked_all = torch.zeros((bsz, num_p, k), dtype=torch.int32, device=device)

    for b in range(bsz):
        # Compute book-level document frequency df[tok] across P pages
        df: dict[int, int] = {}
        pages_cpu = page_tensor[b].detach().cpu().tolist()
        for row in pages_cpu:
            for tid in set(row):
                if tid not in special_set and tid >= 0:
                    df[tid] = df.get(tid, 0) + 1

        # Build per-page token-rarity TF-IDF scores [num_p, seq_len]
        scores_mat = torch.full((num_p, seq_len), -1e4, dtype=torch.float32, device=device)
        valid_mask = torch.ones((num_p, seq_len), dtype=torch.bool, device=device)
        for p_i, row in enumerate(pages_cpu):
            tf_local: dict[int, int] = {}
            for tid in row:
                if tid not in special_set and tid >= 0:
                    tf_local[tid] = tf_local.get(tid, 0) + 1
            row_scores = []
            row_valid = []
            for tid in row:
                if tid in special_set or tid < 0:
                    row_scores.append(-1e4)
                    row_valid.append(False)
                else:
                    # Pure book-level IDF (0.0 for ubiquitous stopword/filler appearing on every page)
                    # plus mild within-page rarity bonus (1 / (1 + log(tf))) so repetitive filler never outranks needles.
                    idf = math.log((num_p + 1.0) / (df.get(tid, 1) + 1.0))
                    rarity_local = 0.25 / (1.0 + math.log(max(1, tf_local.get(tid, 1))))
                    row_scores.append(idf + rarity_local)
                    row_valid.append(True)
            scores_mat[p_i] = torch.tensor(row_scores, dtype=torch.float32, device=device)
            valid_mask[p_i] = torch.tensor(row_valid, dtype=torch.bool, device=device)

        if s_len <= 1:
            _, raw_idx = torch.topk(scores_mat.masked_fill(~valid_mask, -1e9), k=k, dim=-1, sorted=False)
            sel_pos, _ = torch.sort(raw_idx, dim=-1)
        else:
            win_mean = F.avg_pool1d(scores_mat.unsqueeze(1), kernel_size=s_len, stride=1).squeeze(1)
            win_max = F.max_pool1d(scores_mat.unsqueeze(1), kernel_size=s_len, stride=1).squeeze(1)
            win_valid = F.avg_pool1d(valid_mask.float().unsqueeze(1), kernel_size=s_len, stride=1).squeeze(1) > 0.999
            win_score = (0.7 * win_mean + 0.3 * win_max).masked_fill(~win_valid, -1e9)

            taken = (~valid_mask).clone()
            picked = torch.zeros((num_p, seq_len), dtype=torch.bool, device=device)
            offsets = torch.arange(s_len, device=device).unsqueeze(0)
            row_idx = torch.arange(num_p, device=device)

            for _ in range(k // s_len):
                has_taken = F.max_pool1d(taken.float().unsqueeze(1), kernel_size=s_len, stride=1).squeeze(1) > 0.5
                cand = win_score.masked_fill(has_taken, -1e9)
                best_vals, best_st = cand.max(dim=-1)
                ok = best_vals > -1e8
                if not bool(ok.any()):
                    break
                ok_rows = row_idx[ok]
                pos_mat = best_st[ok].unsqueeze(1) + offsets
                taken[ok_rows.unsqueeze(1), pos_mat] = True
                picked[ok_rows.unsqueeze(1), pos_mat] = True

            picked_counts = picked.sum(dim=-1)
            if bool((picked_counts < k).any()):
                rem_valid = scores_mat.masked_fill(taken, -1e9)
                for r in range(num_p):
                    need = int(k - picked_counts[r].item())
                    if need <= 0:
                        continue
                    avail_v = int((~taken[r]).sum().item())
                    take_v = min(need, avail_v)
                    if take_v > 0:
                        top_v = torch.topk(rem_valid[r], k=take_v, sorted=False).indices
                        picked[r, top_v] = True
                        taken[r, top_v] = True
                        need -= take_v
                    if need > 0:
                        top_a = torch.topk(scores_mat[r].masked_fill(picked[r], -1e9), k=need, sorted=False).indices
                        picked[r, top_a] = True

            sel_pos = torch.nonzero(picked, as_tuple=False)[:, 1].view(num_p, k)

        picked_all[b] = torch.gather(page_tensor[b], dim=-1, index=sel_pos).to(torch.int32)

    return SparseExactBank(
        token_ids=picked_all,
        page_ids=None,
        h_coords=None,
        w_coords=None,
        edited_pages=set(),
    )


# ---------------------------------------------------------------------------
# Lean Path (B) Modules: ContextualEditGateHead(hidden_dim=64) & PathBDiscreteMemory
# ---------------------------------------------------------------------------


class ContextualEditGateHead(nn.Module):
    """Lean mid-layer blind accept/reject edit classifier (`hidden_dim=64`, ZERO DeltaUpdater 7.24M, ZERO Z_book)."""

    def __init__(self, d_llm: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.d_llm = int(d_llm)
        self.hidden_dim = int(hidden_dim)
        self.in_norm = nn.LayerNorm(self.d_llm)
        self.in_proj = nn.Linear(self.d_llm, self.hidden_dim)
        self.tok_attn = nn.Linear(self.hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, contextual_edit_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map [B, T_e, d_llm] mid-layer edit states -> (accept_logit [B], token_attn_weights [B, T_e])."""
        x = contextual_edit_embeds.float()
        h = F.gelu(self.in_proj(self.in_norm(x)))  # [B, T_e, H]
        w = F.softmax(self.tok_attn(h).squeeze(-1), dim=-1)  # [B, T_e]
        pooled_attn = (h * w.unsqueeze(-1)).sum(dim=1)  # [B, H]
        pooled_mean = h.mean(dim=1)  # [B, H]
        feat = torch.cat((pooled_attn, pooled_mean), dim=-1)  # [B, 2H]
        logit = self.classifier(feat).squeeze(-1)  # [B]
        return logit, w


class PathBDiscreteMemory(nn.Module):
    """Locked Paper 1 Path (B): True Two-Bank Gated Mutable Discrete Span-Banks.

    - ZERO `LayeredLatentEncoder`, ZERO `SpatialLatentProjector`, ZERO `Z_book`.
    - ZERO `hw_coord_embed` (0 untrained random weights).
    - Lean `ContextualEditGateHead(hidden_dim=64)` + `SalienceGate(span_len=16, k_exact=48)`.
    - Compact `SparseExactBank` (`token_ids` int32 only = 192 B/page before edit; +192 B/page `base_token_ids` once edited).
    - True Two-Bank layout: unedited Bank-1 pages deterministically FIRST, mutated Bank-2 pages LAST.
    - Span-aligned selective overwrite (Opsi B) with clean newline separator.
    """

    def __init__(
        self,
        cfg: LSLCConfig,
        *,
        bge_toc: Any | None = None,
        bank_write_threshold: float = 0.45,
        edit_gate_hidden_dim: int = 64,
        span_aligned_overwrite: bool = True,
        k_exact: int | None = None,
        salience_span_len: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.k_exact = int(cfg.k_exact if k_exact is None else k_exact)
        self.salience_span_len = int(cfg.salience_span_len if salience_span_len is None else salience_span_len)
        self.salience_gate = SalienceGate(cfg)
        self.edit_gate = ContextualEditGateHead(cfg.d_llm, hidden_dim=int(edit_gate_hidden_dim))
        self.hw_coord_embed = None  # Explicitly ZERO untrained coordinate weights
        self.encoder = None  # Explicitly ZERO LayeredLatentEncoder / Z_book
        self.bge_toc = bge_toc
        self.bank_write_threshold = float(bank_write_threshold)
        self.span_aligned_overwrite = bool(span_aligned_overwrite)

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def bge_model_param_count(self) -> int:
        if self.bge_toc is None or getattr(self.bge_toc, "model", None) is None:
            return 33_360_000
        return int(sum(p.numel() for p in self.bge_toc.model.parameters()))

    def trainable_param_breakdown(self, frozen_llm: nn.Module | None = None) -> dict[str, int]:
        sal_p = int(sum(p.numel() for p in self.salience_gate.parameters()))
        eg_p = int(sum(p.numel() for p in self.edit_gate.parameters()))
        hw_p = int(sum(p.numel() for p in self.hw_coord_embed.parameters())) if self.hw_coord_embed is not None else 0
        llm_p = int(sum(p.numel() for p in frozen_llm.parameters())) if frozen_llm is not None else 0
        return {
            "salience_gate_params": sal_p,
            "edit_gate_params": eg_p,
            "hw_coord_embed_params": hw_p,
            "layered_latent_encoder_params": 0,
            "spatial_projector_params": 0,
            "delta_updater_params": 0,
            "total_pathb_trainable_params": sal_p + eg_p + hw_p,
            "bge_router_frozen_params": self.bge_model_param_count(),
            "frozen_llm_params": llm_p,
        }

    def storage_breakdown_bytes_per_page(self, num_spans: int = 4) -> dict[str, int]:
        bank_before = int(self.k_exact * 4)  # token_ids:int32 ONLY (no unused page_ids/h/w) = 192 B/page at k=48
        base_extra = int(self.k_exact * 4)  # base_token_ids:int32 (+192 B/page once mutated)
        bge_dim = int(getattr(self.bge_toc, "dim", 384)) if self.bge_toc is not None else 384
        bge_bf16_bytes = int(num_spans * bge_dim * 2)  # 4 * 384 * 2 = 3,072 B/page (3 KiB)
        return {
            "exact_bank_bytes_per_page_before_edit": bank_before,
            "base_token_ids_extra_bytes_per_page_after_edit": base_extra,
            "exact_bank_bytes_per_page_after_edit": bank_before + base_extra,
            "bge_4span_bf16_bytes_per_page": bge_bf16_bytes,
            "stored_bytes_per_page_before_edit": bank_before + bge_bf16_bytes,
            "stored_bytes_per_page_after_edit": bank_before + base_extra + bge_bf16_bytes,
            # Legacy alias kept explicit (= post-edit worst-case total)
            "total_stored_bytes_per_page": bank_before + base_extra + bge_bf16_bytes,
        }

    @staticmethod
    def order_two_bank_pages(
        deep_pages: list[int],
        edited_pages: set[int] | None = None,
    ) -> tuple[list[int], list[int]]:
        """True Two-Bank prefix ordering:
        - Bank-1 (Immutable Static Prefix): all unedited pages placed FIRST in deterministic ascending page-ID order.
        - Bank-2 (Mutable Edited Tail): all edited pages placed LAST in deterministic ascending page-ID order.
        """
        ed_set = set(edited_pages or ())
        bank1_pages = sorted(int(p) for p in deep_pages if int(p) not in ed_set)
        bank2_pages = sorted(int(p) for p in deep_pages if int(p) in ed_set)
        return bank1_pages, bank2_pages

    @torch.no_grad()
    def encode_book(
        self,
        page_tensor: torch.Tensor,
        embed_fn,
        tokenizer,
        vocab_size: int,
        *,
        device: torch.device | None = None,
        amp_dtype: torch.dtype = torch.float32,
        autocast_enabled: bool = False,
        special_token_ids: tuple[int, ...] | None = None,
        num_spans: int = 4,
        chunk_size: int = 32,
    ) -> dict[str, Any]:
        """Independently build compact SparseExactBank (token_ids only) + BF16 BGE-4span keys (0 encoder, 0 Z_book)."""
        dev = page_tensor.device if device is None else device
        _sync_if_cuda(dev)
        t0 = time.perf_counter()
        bsz, num_p, seq_len = page_tensor.shape
        tok_chunks: list[torch.Tensor] = []
        prev_k = self.salience_gate.cfg.k_exact
        self.salience_gate.cfg.k_exact = min(self.k_exact, seq_len)
        try:
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=autocast_enabled):
                for s in range(0, num_p, chunk_size):
                    e = min(num_p, s + chunk_size)
                    p_chunk = page_tensor[:, s:e]
                    p_emb = embed_fn(p_chunk)
                    bc = SparseExactBank.from_pages(
                        p_chunk,
                        self.salience_gate,
                        p_emb,
                        vocab_size=vocab_size,
                        special_token_ids=special_token_ids,
                        salience_span_len=self.salience_span_len,
                        compact_pathb=True,
                    )
                    tok_chunks.append(bc.token_ids)
        finally:
            self.salience_gate.cfg.k_exact = prev_k

        bank = SparseExactBank(
            token_ids=torch.cat(tok_chunks, dim=1),
            page_ids=None,
            h_coords=None,
            w_coords=None,
            edited_pages=set(),
        )
        bge_keys_bf16: torch.Tensor | None = None
        if self.bge_toc is not None:
            bge_keys_bf16 = self.bge_toc.encode_book_pages(
                page_tensor, None, tokenizer, num_spans=num_spans, batch_size=128
            ).to(torch.bfloat16)

        _sync_if_cuda(dev)
        encode_ms = (time.perf_counter() - t0) * 1000.0
        bge_bytes = int(bge_keys_bf16.numel() * bge_keys_bf16.element_size()) if bge_keys_bf16 is not None else 0
        b_bd = bank.storage_bytes_breakdown()
        bge_bpp = round(bge_bytes / max(1, num_p), 2)
        before_bpp = round(b_bd["bytes_per_page_before_edit"] + bge_bpp, 2)
        after_proj_bpp = round(b_bd["bytes_per_page_after_edit_with_base"] + bge_bpp, 2)
        return {
            "bank": bank,
            "bge_keys_bf16": bge_keys_bf16,
            "encode_ms": encode_ms,
            "stored_bytes_per_page_before_edit": before_bpp,
            "stored_bytes_per_page_after_edit": after_proj_bpp,
            "stored_bytes_per_page": before_bpp,
            "exact_bank_bytes_per_page_before_edit": b_bd["bytes_per_page_before_edit"],
            "exact_bank_bytes_per_page_after_edit": b_bd["bytes_per_page_after_edit_with_base"],
            "bge_keys_bytes_per_page": bge_bpp,
        }

    @torch.no_grad()
    def reencode_single_page(
        self,
        bank: SparseExactBank,
        bge_keys_bf16: torch.Tensor | None,
        page_idx: int,
        new_page_token_ids: torch.Tensor,
        embed_fn,
        tokenizer,
        vocab_size: int,
        *,
        device: torch.device | None = None,
        amp_dtype: torch.dtype = torch.float32,
        autocast_enabled: bool = False,
        special_token_ids: tuple[int, ...] | None = None,
    ) -> tuple[SparseExactBank, torch.Tensor | None]:
        """Modular O(1) single-page re-encoding for Path (B) discrete span bank + BF16 BGE key."""
        dev = bank.token_ids.device if device is None else device
        p_3d = new_page_token_ids.unsqueeze(1) if new_page_token_ids.dim() == 2 else new_page_token_ids
        prev_k = self.salience_gate.cfg.k_exact
        self.salience_gate.cfg.k_exact = min(self.k_exact, int(p_3d.shape[-1]))
        try:
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=autocast_enabled):
                bc = SparseExactBank.from_pages(
                    p_3d,
                    self.salience_gate,
                    embed_fn(p_3d),
                    vocab_size=vocab_size,
                    special_token_ids=special_token_ids,
                    salience_span_len=self.salience_span_len,
                    compact_pathb=True,
                )
        finally:
            self.salience_gate.cfg.k_exact = prev_k

        bank.token_ids[:, page_idx : page_idx + 1] = bc.token_ids.to(bank.token_ids.dtype)
        if bank.base_token_ids is not None:
            bank.base_token_ids[:, page_idx : page_idx + 1] = bc.token_ids.to(bank.base_token_ids.dtype)
        bank.edited_pages.add(int(page_idx))
        if self.bge_toc is not None and bge_keys_bf16 is not None:
            bge_keys_bf16[:, page_idx : page_idx + 1] = self.bge_toc.encode_book_pages(
                p_3d, None, tokenizer, num_spans=4, batch_size=16
            ).to(torch.bfloat16)
        return bank, bge_keys_bf16

    @torch.no_grad()
    def score_edit_gate(
        self,
        edit_ids: torch.Tensor,
        llm: nn.Module,
        *,
        device: torch.device | None = None,
        amp_dtype: torch.dtype = torch.float32,
        autocast_enabled: bool = False,
    ) -> tuple[bool, float, torch.Tensor]:
        """Evaluate lean mid-layer blind accept/reject edit gate (0 oracle labels, 0 DeltaUpdater, 0 Z_book)."""
        dev = edit_ids.device if device is None else device
        layer_idx = (
            self.cfg.contextual_edit_layer
            if self.cfg.contextual_edit_layer is not None
            else max(1, self.cfg.num_llm_layers // 2)
        )
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=autocast_enabled):
            ctx_e = extract_contextual_edit_embeds(llm, edit_ids, layer_idx=layer_idx)
        logit, tok_w = self.edit_gate(ctx_e)
        prob = float(torch.sigmoid(logit).mean().item())
        return (prob > self.bank_write_threshold), prob, tok_w

    @torch.no_grad()
    def apply_edit(
        self,
        bank: SparseExactBank,
        page_idx: int,
        edit_ids: torch.Tensor,
        llm: nn.Module,
        embed_fn,
        vocab_size: int,
        *,
        device: torch.device | None = None,
        amp_dtype: torch.dtype = torch.float32,
        autocast_enabled: bool = False,
        in_place: bool = True,
        span_aligned_overwrite: bool | None = None,
        sep_token_id: int | None = None,
    ) -> tuple[SparseExactBank, bool, float, float]:
        """Run lean mid-layer accept/reject gate + span-aligned selective SparseExactBank overwrite."""
        dev = edit_ids.device if device is None else device
        _sync_if_cuda(dev)
        t0 = time.perf_counter()
        accepted, prob, _ = self.score_edit_gate(
            edit_ids, llm, device=dev, amp_dtype=amp_dtype, autocast_enabled=autocast_enabled
        )
        delta_gates = torch.full(
            (1, self.cfg.grid_h, self.cfg.grid_w),
            fill_value=prob,
            device=dev,
            dtype=torch.float32,
        )
        use_span_align = self.span_aligned_overwrite if span_aligned_overwrite is None else bool(span_aligned_overwrite)
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=autocast_enabled):
            raw_e_emb = embed_fn(edit_ids)
            new_bank = bank.update_page_from_delta(
                page_indices=torch.tensor([int(page_idx)], dtype=torch.long, device=dev),
                edit_token_ids=edit_ids,
                edit_embeds=raw_e_emb,
                salience_gate=self.salience_gate,
                delta_gates=delta_gates,
                grid_h=self.cfg.grid_h,
                grid_w=self.cfg.grid_w,
                vocab_size=vocab_size,
                in_place=in_place,
                bank_write_threshold=self.bank_write_threshold,
                edit_token_weights=None,
                use_tail_bonus=False,
                span_aligned_overwrite=use_span_align,
                salience_span_len=self.salience_span_len,
                sep_token_id=sep_token_id,
                embed_tokens_fn=embed_fn,
            )
        _sync_if_cuda(dev)
        edit_ms = (time.perf_counter() - t0) * 1000.0
        return new_bank, accepted, prob, edit_ms

    @torch.no_grad()
    def compute_routing_logits(
        self,
        bank: SparseExactBank,
        bge_keys_bf16: torch.Tensor | None,
        rq_ids: torch.Tensor,
        q_txt: str,
        embed_fn,
        *,
        device: torch.device | None = None,
        salience_route_weight: float = 1.5,
        excluded_indices: set[int] | None = None,
    ) -> torch.Tensor:
        """Pure-GPU BGE-4span (BF16) + SparseExactBank adaptive hybrid routing logits [B, P]."""
        dev = bank.token_ids.device if device is None else device
        bsz, num_pages, _ = bank.token_ids.shape

        if self.bge_toc is not None and bge_keys_bf16 is not None:
            q_vec = self.bge_toc.encode_texts([q_txt], is_query=True)
            dense_logits = self.bge_toc.score_book(q_vec, bge_keys_bf16).to(device=dev, dtype=torch.float32)
        else:
            dense_logits = torch.zeros((bsz, num_pages), device=dev, dtype=torch.float32)

        logits = dense_logits
        if salience_route_weight > 0.0 and num_pages > 1:
            q_tok = embed_fn(rq_ids).float()
            q_norm = F.normalize(q_tok, dim=-1)
            b_emb = embed_fn(bank.routing_token_ids().clamp(min=0).to(dev)).float()
            b_norm = F.normalize(b_emb, dim=-1)
            sim = torch.einsum("bmd,bpkd->bpmk", q_norm, b_norm)
            hit_b = torch.sigmoid((sim - 0.85) * 20.0)
            hit1 = hit_b.amax(dim=-1)
            bi = (
                (hit_b[:, :, :-1, :-1] * hit_b[:, :, 1:, 1:]).amax(dim=-1)
                if (q_tok.shape[1] >= 2 and b_emb.shape[2] >= 2)
                else None
            )
            tri = (
                (hit_b[:, :, :-2, :-2] * hit_b[:, :, 1:-1, 1:-1] * hit_b[:, :, 2:, 2:]).amax(dim=-1)
                if (q_tok.shape[1] >= 3 and b_emb.shape[2] >= 3)
                else None
            )
            if bank.base_token_ids is not None:
                d_emb = embed_fn(bank.token_ids.clamp(min=0).to(dev)).float()
                d_norm = F.normalize(d_emb, dim=-1)
                sim_d = torch.einsum("bmd,bpkd->bpmk", q_norm, d_norm)
                hit_d = torch.sigmoid((sim_d - 0.85) * 20.0)
                hit1 = torch.maximum(hit1, hit_d.amax(dim=-1))
                if bi is not None and d_emb.shape[2] >= 2:
                    bi = torch.maximum(bi, (hit_d[:, :, :-1, :-1] * hit_d[:, :, 1:, 1:]).amax(dim=-1))
                if tri is not None and d_emb.shape[2] >= 3:
                    tri = torch.maximum(
                        tri, (hit_d[:, :, :-2, :-2] * hit_d[:, :, 1:-1, 1:-1] * hit_d[:, :, 2:, 2:]).amax(dim=-1)
                    )
            df1 = hit1.sum(dim=1, keepdim=True)
            w1 = (df1 > 0.3).to(hit1.dtype) / (df1 + 0.5).pow(2)
            sal_score = (hit1 * w1).sum(dim=-1) / (w1.sum(dim=-1) + 1e-6)
            if bi is not None:
                df2 = bi.sum(dim=1, keepdim=True)
                w2 = (df2 > 0.3).to(bi.dtype) / (df2 + 0.5).pow(2)
                sal_score = sal_score + 2.0 * ((bi * w2).sum(dim=-1) / (w2.sum(dim=-1) + 1e-6))
            if tri is not None:
                df3 = tri.sum(dim=1, keepdim=True)
                w3 = (df3 > 0.3).to(tri.dtype) / (df3 + 0.5).pow(2)
                sal_score = sal_score + 3.0 * ((tri * w3).sum(dim=-1) / (w3.sum(dim=-1) + 1e-6))

            sal_std = sal_score.std(dim=-1, keepdim=True)
            sal_z = (sal_score - sal_score.mean(dim=-1, keepdim=True)) / sal_std.clamp(min=0.35)
            sal_conf = (sal_score.amax(dim=-1, keepdim=True) / 1.5).clamp(0.0, 1.0)
            dense_std = dense_logits.std(dim=-1, keepdim=True)
            dense_z = torch.where(
                dense_std > 1e-6,
                (dense_logits - dense_logits.mean(dim=-1, keepdim=True)) / (dense_std + 1e-6),
                torch.zeros_like(dense_logits),
            )
            logits = dense_z + (salience_route_weight * sal_conf) * sal_z

        if excluded_indices:
            masked = logits.clone()
            for idx_ex in excluded_indices:
                if 0 <= idx_ex < num_pages:
                    masked[:, idx_ex] = -1e9
            logits = masked
        return logits

    @torch.no_grad()
    def route_edit_target_page(
        self,
        bank: SparseExactBank,
        bge_keys_bf16: torch.Tensor | None,
        edit_ids: torch.Tensor,
        edit_text: str,
        embed_fn,
        *,
        device: torch.device | None = None,
        salience_route_weight: float = 1.5,
    ) -> int:
        """Blind edit routing (ZERO oracle page index `p_i`): route edit text to the best page via BGE + lexical match."""
        logits = self.compute_routing_logits(
            bank, bge_keys_bf16, edit_ids, edit_text, embed_fn,
            device=device, salience_route_weight=salience_route_weight,
        )
        return int(logits[0].argmax().item())

    @torch.no_grad()
    def assemble_two_bank_prefix(
        self,
        bank: SparseExactBank,
        deep_pages: list[int],
        embed_fn,
        *,
        edited_pages: set[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
        """True Two-Bank prefix assembly (NO blind `[:-32]` slice, NO `deep_pages[-1]` hardcoding!).

        - Bank-1 (Immutable Static Prefix): all unedited pages in `deep_pages`, placed FIRST in deterministic order.
        - Bank-2 (Mutable Edited Tail): all edited pages in `deep_pages`, placed LAST in deterministic order.
        """
        ed_set = set(bank.edited_pages) if edited_pages is None else set(edited_pages)
        bank1_pages, bank2_pages = self.order_two_bank_pages(deep_pages, ed_set)
        dev = bank.token_ids.device

        b1_idx = torch.tensor([bank1_pages], dtype=torch.long, device=dev) if bank1_pages else torch.empty((1, 0), dtype=torch.long, device=dev)
        b2_idx = torch.tensor([bank2_pages], dtype=torch.long, device=dev) if bank2_pages else torch.empty((1, 0), dtype=torch.long, device=dev)

        bank1_embeds = bank.gather_embeds(b1_idx, embed_tokens_fn=embed_fn, hw_coord_embed=None, training_ste=False)
        bank2_embeds = bank.gather_embeds(b2_idx, embed_tokens_fn=embed_fn, hw_coord_embed=None, training_ste=False)
        return bank1_embeds, bank2_embeds, bank1_pages, bank2_pages

    @torch.no_grad()
    def route_and_assemble(
        self,
        bank: SparseExactBank,
        bge_keys_bf16: torch.Tensor | None,
        rq_ids: torch.Tensor,
        q_txt: str,
        embed_fn,
        *,
        device: torch.device | None = None,
        k_deep: int = 4,
        salience_route_weight: float = 1.5,
        excluded_indices: set[int] | None = None,
        use_two_bank_layout: bool = True,
    ) -> tuple[RouterOutput, torch.Tensor, float]:
        """Pure-GPU BGE-4span (BF16) + SparseExactBank hybrid routing + True Two-Bank prefix assembly."""
        dev = bank.token_ids.device if device is None else device
        _sync_if_cuda(dev)
        t0 = time.perf_counter()
        num_pages = int(bank.token_ids.shape[1])

        logits = self.compute_routing_logits(
            bank, bge_keys_bf16, rq_ids, q_txt, embed_fn,
            device=dev, salience_route_weight=salience_route_weight, excluded_indices=excluded_indices,
        )
        probs = F.softmax(logits, dim=-1)
        kd = min(num_pages - len(excluded_indices or ()), max(1, int(k_deep)))
        sorted_idx = torch.argsort(logits, dim=-1, descending=True)
        top_pages = [int(x) for x in sorted_idx[0, :kd].tolist()]

        if use_two_bank_layout:
            b1_emb, b2_emb, b1_pages, b2_pages = self.assemble_two_bank_prefix(bank, top_pages, embed_fn)
            ordered_pages = b1_pages + b2_pages
            prefix = torch.cat((b1_emb, b2_emb), dim=1)
            deep_idx = torch.tensor([ordered_pages], dtype=torch.long, device=dev)
        else:
            deep_idx = sorted_idx[:, :kd].flip(dims=[-1])
            prefix = bank.gather_embeds(deep_idx, embed_tokens_fn=embed_fn, hw_coord_embed=None, training_ste=False)

        r_out = RouterOutput(
            logits=logits,
            probs=probs,
            deep_idx=deep_idx,
            skim_idx=sorted_idx[:, :0],
            toc_idx=torch.sort(sorted_idx[:, kd:], dim=-1).values,
        )
        _sync_if_cuda(dev)
        return r_out, prefix, (time.perf_counter() - t0) * 1000.0


def save_pathb_checkpoint(pathb: PathBDiscreteMemory, save_path: str | Path) -> dict[str, Any]:
    """Save lean Path (B) weights (`salience_gate`, `edit_gate`) to `.safetensors` + SHA256."""
    from safetensors.torch import save_file

    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in pathb.state_dict().items()}
    save_file(sd, str(out_p))
    sha = hashlib.sha256(out_p.read_bytes()).hexdigest()
    return {"path": str(out_p), "bytes": out_p.stat().st_size, "sha256": sha, "num_tensors": len(sd)}


def load_pathb_checkpoint(pathb: PathBDiscreteMemory, ckpt_path: str | Path) -> dict[str, Any]:
    """Load lean Path (B) `.safetensors` checkpoint strictly + return SHA256 provenance."""
    from safetensors.torch import load_file

    p = Path(ckpt_path)
    sd = load_file(str(p))
    incompat = pathb.load_state_dict(sd, strict=True)
    return {
        "path": str(p),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "missing_keys": list(incompat.missing_keys),
        "unexpected_keys": list(incompat.unexpected_keys),
    }


def compute_prefix_lcs_len(seq_a: list[int], seq_b: list[int]) -> int:
    """Longest common prefix length between two page-ID sequences (for RadixAttention-style prefix hit telemetry)."""
    k = 0
    for a, b in zip(seq_a, seq_b):
        if a != b:
            break
        k += 1
    return k
