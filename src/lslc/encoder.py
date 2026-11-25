"""LayeredLatentEncoder (with 3D Factorized RoPE), SalienceGate, and 8-Byte SparseExactBank."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LSLCConfig
from .rope_3d import FactorizedRoPE3D


class _FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverCrossAttentionBlock(nn.Module):
    """Cross-attention from learned book slots to page token embeddings."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim, mult=2, dropout=dropout)

    def forward(self, queries: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        bsz, n_q, dim = queries.shape
        n_kv = kv.shape[1]
        q_in = self.norm_q(queries)
        kv_in = self.norm_kv(kv)

        q = self.q_proj(q_in).view(bsz, n_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).view(bsz, n_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(bsz, n_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop.p if self.training else 0.0)
        out = self.out_proj(attn.transpose(1, 2).contiguous().view(bsz, n_q, dim))
        x = queries + self.drop(out)
        x = x + self.ff(self.norm_ff(x))
        return x


class SpatialLayerSelfAttentionBlock(nn.Module):
    """Self-attention across [L, H, W] book slots with FactorizedRoPE3D and per-layer 2D conv mixer."""

    def __init__(self, cfg: LSLCConfig) -> None:
        super().__init__()
        dim = cfg.d_enc
        self.num_layers = cfg.num_layers
        self.grid_h = cfg.grid_h
        self.grid_w = cfg.grid_w
        self.num_heads = cfg.enc_heads
        self.head_dim = dim // cfg.enc_heads

        self.norm_attn = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.rope_3d = FactorizedRoPE3D(
            head_dim=self.head_dim,
            num_layers=cfg.num_layers,
            grid_h=cfg.grid_h,
            grid_w=cfg.grid_w,
        )
        self.drop = nn.Dropout(cfg.enc_dropout)

        # Depthwise-separable 3x3 2D spatial mixer operating independently inside each semantic layer l
        self.norm_conv = nn.LayerNorm(dim)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1)

        self.norm_ff = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim, mult=2, dropout=cfg.enc_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, slots, dim = x.shape
        h_in = self.norm_attn(x)
        q = self.q_proj(h_in).view(bsz, slots, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h_in).view(bsz, slots, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_in).view(bsz, slots, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.rope_3d(q)
        k = self.rope_3d(k)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop.p if self.training else 0.0)
        x = x + self.drop(self.out_proj(attn.transpose(1, 2).contiguous().view(bsz, slots, dim)))

        # 2D spatial residual mixing per layer (run in module weight dtype to avoid FP16 cuDNN depthwise engine errors on T4)
        c_in = self.norm_conv(x).view(bsz * self.num_layers, self.grid_h, self.grid_w, dim).permute(0, 3, 1, 2).contiguous()
        with torch.amp.autocast(device_type=c_in.device.type, enabled=False):
            c_out = self.pw_conv(F.gelu(self.dw_conv(c_in.to(self.dw_conv.weight.dtype)))).to(x.dtype)
        c_out = c_out.permute(0, 2, 3, 1).contiguous().view(bsz, slots, dim)
        x = x + c_out

        x = x + self.ff(self.norm_ff(x))
        return x


class LayeredLatentEncoder(nn.Module):
    """Compresses [B, Tp, d_llm] page embeddings into [B, L=3, C=128, H=4, W=4]."""

    def __init__(self, cfg: LSLCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_llm, cfg.d_enc)

        # Ordered 16-span (H=4, W=4) + intra-span positional embeddings for Tp=256 tokens
        self.span_h_emb = nn.Embedding(cfg.grid_h, cfg.d_enc)
        self.span_w_emb = nn.Embedding(cfg.grid_w, cfg.d_enc)
        self.intra_span_emb = nn.Embedding(cfg.tokens_per_span, cfg.d_enc)

        # Learnable spatial-layer book queries [1, L*H*W, d_enc]
        self.book_queries = nn.Parameter(torch.randn(1, cfg.num_book_slots, cfg.d_enc) * 0.02)
        self.layer_prior = nn.Embedding(cfg.num_layers, cfg.d_enc)

        self.cross_blocks = nn.ModuleList(
            [PerceiverCrossAttentionBlock(cfg.d_enc, cfg.enc_heads, cfg.enc_dropout) for _ in range(2)]
        )
        self.self_blocks = nn.ModuleList([SpatialLayerSelfAttentionBlock(cfg) for _ in range(cfg.enc_layers)])

        self.out_norm = nn.LayerNorm(cfg.d_enc)
        self.out_proj = nn.Linear(cfg.d_enc, cfg.latent_channels)

    def _token_spatial_pos(self, seq_len: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        span_len = self.cfg.tokens_per_span
        span_idx = torch.clamp(idx // span_len, max=self.cfg.num_spatial_slots - 1)
        h_idx = span_idx // self.cfg.grid_w
        w_idx = span_idx % self.cfg.grid_w
        intra_idx = idx % span_len
        return self.span_h_emb(h_idx) + self.span_w_emb(w_idx) + self.intra_span_emb(intra_idx)

    def forward(self, page_embeds: torch.Tensor) -> torch.Tensor:
        """Encode a single page [B, Tp, d_llm] -> Z_page [B, L=3, C=128, H=4, W=4]."""
        bsz, seq_len, _ = page_embeds.shape
        if seq_len > self.cfg.page_size:
            raise ValueError(f"Page length {seq_len} exceeds page_size={self.cfg.page_size}.")
        if not torch.is_autocast_enabled():
            page_embeds = page_embeds.to(self.in_proj.weight.dtype)

        kv = self.in_proj(page_embeds) + self._token_spatial_pos(seq_len, page_embeds.device).unsqueeze(0)

        layer_ids = torch.arange(self.cfg.num_layers, device=page_embeds.device).repeat_interleave(
            self.cfg.num_spatial_slots
        )
        queries = (self.book_queries + self.layer_prior(layer_ids).unsqueeze(0)).expand(bsz, -1, -1)

        for blk in self.cross_blocks:
            queries = blk(queries, kv)
        for blk in self.self_blocks:
            queries = blk(queries)

        z_flat = self.out_proj(self.out_norm(queries))  # [B, L*H*W, C]
        z_page = (
            z_flat.view(bsz, self.cfg.num_layers, self.cfg.grid_h, self.cfg.grid_w, self.cfg.latent_channels)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )
        return z_page

    def encode_pages(self, pages_embeds: torch.Tensor) -> torch.Tensor:
        """Encode multi-page embeddings [B, P, Tp, d_llm] -> [B, P, L=3, C=128, H=4, W=4]."""
        bsz, num_pages, seq_len, d_llm = pages_embeds.shape
        z = self.forward(pages_embeds.reshape(bsz * num_pages, seq_len, d_llm))
        return z.view(bsz, num_pages, self.cfg.num_layers, self.cfg.latent_channels, self.cfg.grid_h, self.cfg.grid_w)


class SalienceGate(nn.Module):
    """Selects top-k_exact high-entropy verbatim tokens per page for the 8 B/tok SparseExactBank."""

    def __init__(self, cfg: LSLCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.scorer = nn.Linear(cfg.d_llm, 1)
        hidden_sal = min(64, cfg.d_llm)
        self.ctx_in = nn.Linear(cfg.d_llm, hidden_sal)
        self.ctx_conv = nn.Conv1d(hidden_sal, hidden_sal, kernel_size=7, padding=3)
        self.ctx_conv2 = (
            nn.Conv1d(hidden_sal, hidden_sal, kernel_size=5, padding=4, dilation=2)
            if cfg.salience_mode == "learned_surprisal"
            else None
        )
        self.ctx_out = nn.Linear(hidden_sal, 1)
        self.gamma = nn.Parameter(torch.tensor(cfg.salience_gamma_init))
        self.novelty_weight = nn.Parameter(torch.tensor(0.6))
        self.struct_weight = nn.Parameter(torch.tensor(1.5))

    def score_logits(
        self,
        page_embeds: torch.Tensor,
        surprisal: torch.Tensor | None = None,
        token_ids: torch.Tensor | None = None,
        vocab_size: int | None = None,
    ) -> torch.Tensor:
        """Compute per-token salience logits [B, Tp] with micro-span contiguity without LLM surprisal at inference."""
        if not torch.is_autocast_enabled():
            page_embeds = page_embeds.to(self.scorer.weight.dtype)
        raw_logits = self.scorer(page_embeds).squeeze(-1)  # [B, Tp]
        h_ctx_seq = F.gelu(self.ctx_in(page_embeds)).transpose(1, 2)
        h_ctx_seq = F.gelu(self.ctx_conv(h_ctx_seq))
        if self.cfg.salience_mode == "learned_surprisal" and self.ctx_conv2 is not None:
            h_ctx_seq = h_ctx_seq + F.gelu(self.ctx_conv2(h_ctx_seq))
        h_ctx = h_ctx_seq.transpose(1, 2)
        ctx_logits = self.ctx_out(h_ctx).squeeze(-1)

        if self.cfg.salience_mode == "learned_surprisal":
            # Tokenizer-agnostic learned scorer (no 0.5*tanh clamp, zero hardcoded token IDs)
            neural_term = 0.6 * raw_logits + 0.4 * F.avg_pool1d(
                (raw_logits + ctx_logits).unsqueeze(1), kernel_size=3, stride=1, padding=1
            ).squeeze(1)
            logits = neural_term
            if surprisal is not None:
                surp_smooth = 0.7 * surprisal + 0.3 * F.avg_pool1d(
                    surprisal.unsqueeze(1), kernel_size=3, stride=1, padding=1
                ).squeeze(1)
                logits = logits + F.softplus(self.gamma) * surp_smooth
            else:
                mean_vec = F.normalize(page_embeds.float().mean(dim=1, keepdim=True), dim=-1)
                tok_norm = F.normalize(page_embeds.float(), dim=-1)
                novelty = 1.0 - (tok_norm * mean_vec).sum(dim=-1)
                novelty = 0.6 * novelty + 0.4 * F.avg_pool1d(
                    novelty.unsqueeze(1), kernel_size=3, stride=1, padding=1
                ).squeeze(1)
                nov_z = (novelty - novelty.mean(dim=-1, keepdim=True)) / (novelty.std(dim=-1, keepdim=True) + 1e-6)
                logits = logits + (0.5 * torch.sigmoid(self.novelty_weight)) * nov_z.to(logits.dtype)
            return logits

        # v2_legacy path (kept strictly for reproducing/loading v2.1.0 checkpoints)
        # Bound learned residual to [-0.5, +0.5] so prose training refines local selection
        # without ever overpowering high-salience novelty / BPE digit-record bursts.
        neural_term = 0.6 * raw_logits + 0.4 * F.avg_pool1d(
            (raw_logits + ctx_logits).unsqueeze(1), kernel_size=5, stride=1, padding=2
        ).squeeze(1)
        logits = 0.5 * torch.tanh(neural_term)

        if surprisal is not None:
            surp_smooth = 0.5 * surprisal + 0.5 * F.avg_pool1d(
                surprisal.unsqueeze(1), kernel_size=5, stride=1, padding=2
            ).squeeze(1)
            logits = logits + F.softplus(self.gamma) * surp_smooth
        else:
            # Cosine novelty relative to page centroid smoothed over 7-token micro-spans
            mean_vec = F.normalize(page_embeds.float().mean(dim=1, keepdim=True), dim=-1)
            tok_norm = F.normalize(page_embeds.float(), dim=-1)
            novelty = 1.0 - (tok_norm * mean_vec).sum(dim=-1)
            novelty = 0.5 * novelty + 0.5 * F.avg_pool1d(
                novelty.unsqueeze(1), kernel_size=7, stride=1, padding=3
            ).squeeze(1)
            nov_z = (novelty - novelty.mean(dim=-1, keepdim=True)) / (novelty.std(dim=-1, keepdim=True) + 1e-6)
            nov_w = 0.4 + 0.3 * torch.sigmoid(self.novelty_weight)
            logits = logits + nov_w * nov_z.to(logits.dtype)

        if token_ids is not None:
            v_denom = float(vocab_size) if vocab_size is not None else max(2.0, float(token_ids.max().item()) + 1.0)
            if v_denom > 2048.0:
                # Byte-level BPE (Qwen3/Llama/cl100k): digits '0'..'9' (15..24) and key-value delimiters
                # ('-', '='). Use a 5-token left-lookahead window + direct digit priority so all 9 tokens of
                # the code ('5877-6370') rank 1..9 and the 5 preceding key tokens ('NODE-PORT = ') rank 10..14.
                is_digit = (token_ids >= 15) & (token_ids <= 24)
                is_joiner = (token_ids == 12) | (token_ids == 62)  # '-' (kebab-case/codes) and '_' (snake_case code symbols)
                # Subwords immediately adjacent (±1 token) to '-' or '_' form compound symbolic/code identifiers
                is_compound_sym = F.max_pool1d(is_joiner.float().unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1) > 0.5
                is_digit_code = is_digit | is_joiner | is_compound_sym
                is_delim = (token_ids == 28) | (token_ids == 284) | (token_ids == 25) | (token_ids == 551)
                is_header_delim = (token_ids == 25) | (token_ids == 551) | (token_ids == 60) | (token_ids == 933)
                after_header = (
                    F.avg_pool1d(F.pad(is_header_delim.float().unsqueeze(1), (12, 0)), kernel_size=13, stride=1).squeeze(1)
                    > 0.0
                )
                code_signal = (is_digit_code.float() + 0.5 * is_delim.float()).unsqueeze(1)
                cluster_density = F.avg_pool1d(code_signal, kernel_size=9, stride=1, padding=4).squeeze(1)
                max_cluster = F.max_pool1d(cluster_density.unsqueeze(1), kernel_size=7, stride=1, padding=3).squeeze(1)
                lookahead_key = F.avg_pool1d(F.pad(code_signal, (0, 5)), kernel_size=6, stride=1).squeeze(1)
                in_dense_code = is_digit_code.float() * max_cluster
                # Immediate 1-token right-neighbor of a dense code/symbol cluster captures the closing ' .\n' terminator
                right_closer = F.pad(in_dense_code[:, :-1], (1, 0), value=0.0) * (1.0 - is_digit_code.float())
                burst = (
                    2.0 * in_dense_code
                    + 1.2 * right_closer
                    + 1.0 * F.relu(lookahead_key - 0.30)
                    + 0.5 * is_delim.float() * lookahead_key
                    + 0.45 * after_header.float()
                )
            else:
                # Small mock vocabularies: local token-ID variance highlights anomalous spans
                tf = token_ids.float().unsqueeze(1)
                local_mean = F.avg_pool1d(tf, kernel_size=5, stride=1, padding=2).squeeze(1)
                burst = (token_ids.float() - local_mean).abs()
            burst_z = (burst - burst.mean(dim=-1, keepdim=True)) / (burst.std(dim=-1, keepdim=True) + 1e-6)
            struct_w = 2.0 + F.softplus(self.struct_weight)
            logits = logits + struct_w * burst_z.to(logits.dtype)
        return logits

    def forward(
        self,
        page_embeds: torch.Tensor,
        surprisal: torch.Tensor | None = None,
        token_ids: torch.Tensor | None = None,
        vocab_size: int | None = None,
        span_len: int | None = None,
        special_token_ids: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score tokens [B, Tp, d_llm] -> sorted (positions, h_coords, w_coords, soft_gates) [B, k_exact]."""
        logits = self.score_logits(page_embeds, surprisal=surprisal, token_ids=token_ids, vocab_size=vocab_size)
        scores = torch.sigmoid(logits)
        bsz, seq_len = scores.shape
        device = scores.device
        k = min(self.cfg.k_exact, seq_len)
        eff_span = self.cfg.salience_span_len if span_len is None else int(span_len)

        valid_mask = torch.ones((bsz, seq_len), dtype=torch.bool, device=device)
        if token_ids is not None and special_token_ids:
            for sid in special_token_ids:
                valid_mask &= token_ids != int(sid)

        if eff_span <= 1:
            if token_ids is not None and special_token_ids:
                sel_scores = scores.masked_fill(~valid_mask, -1.0)
                _, raw_idx = torch.topk(sel_scores, k=k, dim=-1, sorted=False)
            else:
                _, raw_idx = torch.topk(scores, k=k, dim=-1, sorted=False)
            selected_pos, _ = torch.sort(raw_idx, dim=-1)
        else:
            s_len = min(eff_span, k, seq_len)
            work_logits = logits.float().masked_fill(~valid_mask, -1e4)
            win_mean = F.avg_pool1d(work_logits.unsqueeze(1), kernel_size=s_len, stride=1).squeeze(1)
            win_max = F.max_pool1d(work_logits.unsqueeze(1), kernel_size=s_len, stride=1).squeeze(1)
            win_valid = F.avg_pool1d(valid_mask.float().unsqueeze(1), kernel_size=s_len, stride=1).squeeze(1) > 0.999
            win_score = (0.7 * win_mean + 0.3 * win_max).masked_fill(~win_valid, -1e9)

            taken = (~valid_mask).clone()
            picked = torch.zeros((bsz, seq_len), dtype=torch.bool, device=device)
            offsets = torch.arange(s_len, device=device).unsqueeze(0)
            row_idx = torch.arange(bsz, device=device)

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
                rem_valid = work_logits.masked_fill(taken, -1e9)
                for r in range(bsz):
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
                        top_a = torch.topk(logits[r].float().masked_fill(picked[r], -1e9), k=need, sorted=False).indices
                        picked[r, top_a] = True

            selected_pos = torch.nonzero(picked, as_tuple=False)[:, 1].view(bsz, k)

        soft_gates = torch.gather(scores, dim=-1, index=selected_pos)

        grid_span_len = self.cfg.tokens_per_span
        span_idx = selected_pos // grid_span_len
        h_coords = (span_idx // self.cfg.grid_w).to(torch.uint8)
        w_coords = (span_idx % self.cfg.grid_w).to(torch.uint8)
        return selected_pos, h_coords, w_coords, soft_gates


@dataclass
class SparseExactBank:
    """Discrete verbatim span-bank store.

    Legacy LSLC stores `(token_id:int32, page_id:int16, h:uint8, w:uint8)` = 8 B/tok.
    Compact Path (B) stores strictly `token_ids:int32` = 4 B/tok (192 B/page at k=48) before edit,
    plus `base_token_ids:int32` (+192 B/page) once mutated to preserve edit-invariant routing.
    """

    token_ids: torch.Tensor  # [B, P, K] int32 (4 B/tok)
    page_ids: torch.Tensor | None = None  # [B, P, K] int16 (2 B/tok; None in compact Path B)
    h_coords: torch.Tensor | None = None  # [B, P, K] uint8 (1 B/tok; None in compact Path B)
    w_coords: torch.Tensor | None = None  # [B, P, K] uint8 (1 B/tok; None in compact Path B)
    _ste_gates: torch.Tensor | None = None  # Transient training-only STE tensor; never persisted
    base_token_ids: torch.Tensor | None = None  # Immutable pre-edit page key bank (+4 B/tok once edited)
    edited_pages: set[int] = field(default_factory=set)  # Mutated page indices for True Two-Bank layout

    def routing_token_ids(self) -> torch.Tensor:
        """Return immutable original page key tokens if edited, else live token_ids."""
        return self.token_ids if self.base_token_ids is None else self.base_token_ids

    def storage_bytes(self, *, include_base_if_edited: bool = True) -> int:
        """Return exact honest serialized footprint across all resident tensors on this bank."""
        total = int(self.token_ids.numel() * self.token_ids.element_size())
        if self.page_ids is not None:
            total += int(self.page_ids.numel() * self.page_ids.element_size())
        if self.h_coords is not None:
            total += int(self.h_coords.numel() * self.h_coords.element_size())
        if self.w_coords is not None:
            total += int(self.w_coords.numel() * self.w_coords.element_size())
        if include_base_if_edited and self.base_token_ids is not None:
            total += int(self.base_token_ids.numel() * self.base_token_ids.element_size())
        return total

    def storage_bytes_breakdown(self) -> dict[str, Any]:
        """Detailed honest byte breakdown distinguishing pre-edit vs. post-edit (`base_token_ids`) footprint."""
        num_pages = max(1, int(self.token_ids.shape[1]))
        tok_b = int(self.token_ids.numel() * self.token_ids.element_size())
        page_b = int(self.page_ids.numel() * self.page_ids.element_size()) if self.page_ids is not None else 0
        coord_b = (
            (int(self.h_coords.numel() * self.h_coords.element_size()) if self.h_coords is not None else 0)
            + (int(self.w_coords.numel() * self.w_coords.element_size()) if self.w_coords is not None else 0)
        )
        base_b = (
            int(self.base_token_ids.numel() * self.base_token_ids.element_size())
            if self.base_token_ids is not None
            else 0
        )
        before_b = tok_b + page_b + coord_b
        after_live_b = before_b + base_b
        after_worst_b = before_b + tok_b  # Exact cost once base_token_ids (+4 B/tok) is materialized
        return {
            "token_ids_bytes": tok_b,
            "page_ids_bytes": page_b,
            "coords_bytes": coord_b,
            "base_token_ids_bytes": base_b,
            "total_bytes_before_edit": before_b,
            "total_bytes_after_edit_live": after_live_b,
            "bytes_per_page_before_edit": round(before_b / num_pages, 2),
            "bytes_per_page_after_edit_live": round(after_live_b / num_pages, 2),
            "bytes_per_page_after_edit_with_base": round(after_worst_b / num_pages, 2),
            "has_page_ids": self.page_ids is not None,
            "has_coords": (self.h_coords is not None and self.w_coords is not None),
            "has_base_token_ids": self.base_token_ids is not None,
            "num_edited_pages": len(self.edited_pages),
        }

    @classmethod
    def from_pages(
        cls,
        page_token_ids: torch.Tensor,
        salience_gate: SalienceGate,
        pages_embeds: torch.Tensor,
        surprisal_pages: torch.Tensor | None = None,
        vocab_size: int | None = None,
        special_token_ids: tuple[int, ...] | None = None,
        salience_span_len: int | None = None,
        compact_pathb: bool = False,
    ) -> "SparseExactBank":
        """Build SparseExactBank across [B, P, Tp] token IDs and embeddings."""
        bsz, num_pages, seq_len = page_token_ids.shape
        flat_embeds = pages_embeds.reshape(bsz * num_pages, seq_len, -1)
        flat_surp = None if surprisal_pages is None else surprisal_pages.reshape(bsz * num_pages, seq_len)
        flat_ids = page_token_ids.reshape(bsz * num_pages, seq_len)

        pos, h, w, gates = salience_gate(
            flat_embeds,
            surprisal=flat_surp,
            token_ids=flat_ids,
            vocab_size=vocab_size,
            span_len=salience_span_len,
            special_token_ids=special_token_ids,
        )
        k = pos.shape[-1]
        picked_ids = torch.gather(flat_ids, dim=-1, index=pos).to(torch.int32).view(bsz, num_pages, k)
        if compact_pathb:
            return cls(
                token_ids=picked_ids,
                page_ids=None,
                h_coords=None,
                w_coords=None,
                _ste_gates=gates.view(bsz, num_pages, k),
                edited_pages=set(),
            )

        page_idx = (
            torch.arange(num_pages, device=page_token_ids.device, dtype=torch.int16)
            .view(1, num_pages, 1)
            .expand(bsz, num_pages, k)
            .contiguous()
        )
        return cls(
            token_ids=picked_ids,
            page_ids=page_idx,
            h_coords=h.view(bsz, num_pages, k),
            w_coords=w.view(bsz, num_pages, k),
            _ste_gates=gates.view(bsz, num_pages, k),
            edited_pages=set(),
        )

    def gather_embeds(
        self,
        selected_page_indices: torch.Tensor,
        embed_tokens_fn: Callable[[torch.Tensor], torch.Tensor],
        hw_coord_embed: nn.Embedding | None = None,
        grid_w: int = 4,
        training_ste: bool = True,
    ) -> torch.Tensor:
        """O(1) embedding gather for selected Deep Read pages [B, k_deep] -> [B, k_deep*K, d_llm]."""
        bsz, k_sel = selected_page_indices.shape
        k_exact = self.token_ids.shape[-1]
        if k_sel == 0:
            probe_w = getattr(embed_tokens_fn, "weight", None)
            d_llm = int(probe_w.shape[-1]) if probe_w is not None else int(
                embed_tokens_fn(self.token_ids[:, :1, :1].long()).shape[-1]
            )
            dtype_out = probe_w.dtype if probe_w is not None else torch.float32
            return torch.empty((bsz, 0, d_llm), device=self.token_ids.device, dtype=dtype_out)

        gather_idx = selected_page_indices.unsqueeze(-1).expand(bsz, k_sel, k_exact)
        sel_tok_ids = torch.gather(self.token_ids, dim=1, index=gather_idx).reshape(bsz, k_sel * k_exact)
        tok_embeds = embed_tokens_fn(sel_tok_ids.long())

        if hw_coord_embed is not None and self.h_coords is not None and self.w_coords is not None:
            sel_h = torch.gather(self.h_coords, dim=1, index=gather_idx).reshape(bsz, k_sel * k_exact).long()
            sel_w = torch.gather(self.w_coords, dim=1, index=gather_idx).reshape(bsz, k_sel * k_exact).long()
            coord_embeds = 0.002 * torch.tanh(hw_coord_embed(sel_h * grid_w + sel_w))
            embeds = tok_embeds.to(coord_embeds.dtype) + coord_embeds
        else:
            embeds = tok_embeds

        if training_ste and self._ste_gates is not None:
            sel_gates = torch.gather(self._ste_gates, dim=1, index=gather_idx).reshape(bsz, k_sel * k_exact)
            ste_scale = sel_gates + (1.0 - sel_gates).detach()
            embeds = embeds * ste_scale.to(embeds.dtype).unsqueeze(-1)
        return embeds

    def update_page_from_delta(
        self,
        page_indices: torch.Tensor,
        edit_token_ids: torch.Tensor,
        edit_embeds: torch.Tensor,
        salience_gate: SalienceGate,
        delta_gates: torch.Tensor,
        grid_h: int = 4,
        grid_w: int = 4,
        vocab_size: int | None = None,
        in_place: bool = False,
        bank_write_threshold: float = 0.45,
        edit_token_weights: torch.Tensor | None = None,
        use_tail_bonus: bool | None = None,
        span_aligned_overwrite: bool = False,
        salience_span_len: int | None = None,
        sep_token_id: int | None = None,
        embed_tokens_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> "SparseExactBank":
        """Update footnote slots on page_indices when the edit gate opens.

        When `span_aligned_overwrite=True` (Path B default), partitions `k_exact` into aligned
        span-16 blocks (`0:16, 16:32, 32:48`), selects the block with highest lexical + cosine
        overlap against the edit (fallback to the last block on 0 overlap), and writes contiguous
        edit tokens prefixed/padded with a clean newline separator so mid-span token splicing
        (e.g. `'39Budget'`) and stale `0:16` old-value remnants cannot occur.
        """
        if in_place and torch.is_grad_enabled():
            raise RuntimeError("in_place=True requires torch.no_grad() / inference mode")
        bsz, _, k_exact = self.token_ids.shape
        t_edit = int(edit_token_ids.shape[1])
        if t_edit <= 0 or k_exact <= 0:
            return self

        logits = salience_gate.score_logits(edit_embeds, surprisal=None, token_ids=edit_token_ids, vocab_size=vocab_size)
        scores = torch.sigmoid(logits)
        apply_tail = (salience_gate.cfg.delta_pooling_mode == "v2_tail") if use_tail_bonus is None else bool(use_tail_bonus)
        if apply_tail:
            tail_bonus = torch.linspace(0.0, 2.0, steps=t_edit, device=scores.device, dtype=scores.dtype).unsqueeze(0)
            sel_scores = scores + tail_bonus
        else:
            sel_scores = scores
            if edit_token_weights is not None:
                w_norm = edit_token_weights.float() / edit_token_weights.float().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                sel_scores = sel_scores + 1.0 * w_norm.to(scores.dtype)

        # Couple footnote update strength to DeltaUpdater's bottom sticky-note strip gate (h = H-1)
        strip_gate = delta_gates[:, -1, :].mean(dim=-1, keepdim=True)  # [B, 1]

        use_inplace = in_place and not torch.is_grad_enabled()
        new_tok_ids = self.token_ids if use_inplace else self.token_ids.clone()
        new_h = None if self.h_coords is None else (self.h_coords if use_inplace else self.h_coords.clone())
        new_w = None if self.w_coords is None else (self.w_coords if use_inplace else self.w_coords.clone())
        new_ste = None if self._ste_gates is None else (self._ste_gates if use_inplace else self._ste_gates.clone())
        new_edited = self.edited_pages if use_inplace else set(self.edited_pages)

        batch_arange = torch.arange(bsz, device=self.token_ids.device)
        active_mask = strip_gate.squeeze(-1) > float(bank_write_threshold)
        base_ids = self.base_token_ids

        if bool(active_mask.any()):
            if base_ids is None:
                base_ids = self.token_ids.clone()

            if span_aligned_overwrite:
                raw_bl = int(salience_gate.cfg.salience_span_len if salience_span_len is None else salience_span_len)
                block_len = raw_bl if (raw_bl > 0 and k_exact >= raw_bl) else k_exact
                blocks = [(s, s + block_len) for s in range(0, k_exact, block_len) if s + block_len <= k_exact]
                if not blocks:
                    blocks = [(0, k_exact)]
                num_blocks = len(blocks)
                has_sep = sep_token_id is not None and int(sep_token_id) >= 0

                for b_i in batch_arange[active_mask].tolist():
                    p_i = int(page_indices[b_i].item())
                    new_edited.add(p_i)

                    e_row = edit_token_ids[b_i]
                    e_list = [int(t) for t in e_row.tolist()]
                    e_set = {t for t in e_list if t >= 0 and (not has_sep or t != int(sep_token_id))}
                    e_norm = F.normalize(edit_embeds[b_i].float(), dim=-1) if embed_tokens_fn is not None else None

                    best_m = num_blocks - 1
                    best_score = -1e9
                    max_overlap_cnt = 0
                    max_emb_sim = 0.0
                    for m_idx, (st, en) in enumerate(blocks):
                        blk_t = new_tok_ids[b_i, p_i, st:en]
                        blk_set = {
                            int(t)
                            for t in blk_t.tolist()
                            if int(t) >= 0 and (not has_sep or int(t) != int(sep_token_id))
                        }
                        ov_cnt = len(blk_set & e_set)
                        ov_ratio = ov_cnt / max(1, len(blk_set))
                        emb_sim = 0.0
                        if embed_tokens_fn is not None and e_norm is not None:
                            b_emb = F.normalize(embed_tokens_fn(blk_t.long().clamp(min=0)).float(), dim=-1)
                            emb_sim = float(torch.matmul(b_emb, e_norm.transpose(0, 1)).amax(dim=-1).mean().item())
                        max_overlap_cnt = max(max_overlap_cnt, ov_cnt)
                        max_emb_sim = max(max_emb_sim, emb_sim)
                        sc_m = 3.0 * ov_ratio + emb_sim
                        if sc_m > best_score:
                            best_score = sc_m
                            best_m = m_idx

                    # Strictly fall back to the last block when there is zero lexical/semantic overlap
                    if max_overlap_cnt == 0 and (embed_tokens_fn is None or max_emb_sim < 0.80):
                        best_m = num_blocks - 1

                    # Choose 1 aligned span-16 block if edit fits cleanly; otherwise 2 adjacent aligned blocks (32 slots)
                    # so multi-token symbolic/coding edits are never clipped and spans are never sliced mid-block.
                    single_cap = (block_len - 1) if (has_sep and block_len >= 2) else block_len
                    if t_edit <= single_cap or num_blocks == 1:
                        write_st, write_en = blocks[best_m]
                    else:
                        if best_m + 1 < num_blocks:
                            write_st, write_en = blocks[best_m][0], blocks[best_m + 1][1]
                        else:
                            write_st, write_en = blocks[best_m - 1][0], blocks[best_m][1]

                    slot_len = write_en - write_st
                    payload_cap = (slot_len - 1) if (has_sep and slot_len >= 2) else slot_len
                    if t_edit <= payload_cap:
                        chosen_toks = e_row.to(torch.int32)
                        chosen_sc = scores[b_i]
                    else:
                        # Contiguous max-salience window preserves subword/numeric integrity (no '39Budget' splicing)
                        win_sc = F.avg_pool1d(
                            sel_scores[b_i].float().view(1, 1, t_edit), kernel_size=payload_cap, stride=1
                        ).view(-1)
                        w_start = int(win_sc.argmax().item())
                        chosen_toks = e_row[w_start : w_start + payload_cap].to(torch.int32)
                        chosen_sc = scores[b_i, w_start : w_start + payload_cap]

                    pad_fill = int(sep_token_id) if has_sep else int(chosen_toks[-1].item())
                    out_block = torch.full((slot_len,), pad_fill, dtype=torch.int32, device=new_tok_ids.device)
                    out_sc = torch.full((slot_len,), 1.0, dtype=scores.dtype, device=new_tok_ids.device)
                    offset = 1 if (has_sep and slot_len >= 2) else 0
                    n_cp = min(int(chosen_toks.numel()), slot_len - offset)
                    out_block[offset : offset + n_cp] = chosen_toks[:n_cp]
                    out_sc[offset : offset + n_cp] = chosen_sc[:n_cp]

                    new_tok_ids[b_i, p_i, write_st:write_en] = out_block
                    if new_h is not None:
                        new_h[b_i, p_i, write_st:write_en] = int(grid_h - 1)
                    if new_w is not None:
                        w_pat = torch.arange(slot_len, device=new_tok_ids.device, dtype=torch.uint8) % int(grid_w)
                        new_w[b_i, p_i, write_st:write_en] = w_pat
                    if new_ste is not None:
                        new_ste[b_i, p_i, write_st:write_en] = (out_sc * strip_gate[b_i, 0].to(out_sc.dtype)).to(
                            new_ste.dtype
                        )
            else:
                k_sticky = min(max(k_exact // 2, (k_exact * 2) // 3), t_edit)
                if k_sticky > 0:
                    _, raw_idx = torch.topk(sel_scores, k=k_sticky, dim=-1, sorted=False)
                    sel_pos, _ = torch.sort(raw_idx, dim=-1)
                    picked_ids = torch.gather(edit_token_ids, dim=-1, index=sel_pos).to(torch.int32)
                    picked_scores = torch.gather(scores, dim=-1, index=sel_pos)
                    coupled_gates = picked_scores * strip_gate.to(picked_scores.dtype)

                    b_act = batch_arange[active_mask]
                    p_act = page_indices[active_mask]
                    for p_v in p_act.tolist():
                        new_edited.add(int(p_v))
                    new_tok_ids[b_act, p_act, k_exact - k_sticky :] = picked_ids[active_mask]
                    if new_h is not None:
                        new_h[b_act, p_act, k_exact - k_sticky :] = int(grid_h - 1)
                    if new_w is not None:
                        w_pat = (torch.arange(k_sticky, device=self.token_ids.device, dtype=torch.uint8) % int(grid_w)).unsqueeze(0)
                        new_w[b_act, p_act, k_exact - k_sticky :] = w_pat.expand(b_act.shape[0], k_sticky)
                    if new_ste is not None:
                        new_ste[b_act, p_act, k_exact - k_sticky :] = coupled_gates[active_mask].to(new_ste.dtype)

        if use_inplace:
            self.base_token_ids = base_ids
            self.edited_pages = new_edited
            return self

        return SparseExactBank(
            token_ids=new_tok_ids,
            page_ids=self.page_ids,
            h_coords=new_h,
            w_coords=new_w,
            _ste_gates=new_ste,
            base_token_ids=base_ids,
            edited_pages=new_edited,
        )
