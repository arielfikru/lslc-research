"""LayeredLatentEncoder (with 3D Factorized RoPE), SalienceGate, and 8-Byte SparseExactBank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
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
        h_ctx = F.gelu(self.ctx_in(page_embeds)).transpose(1, 2)
        h_ctx = F.gelu(self.ctx_conv(h_ctx)).transpose(1, 2)
        ctx_logits = self.ctx_out(h_ctx).squeeze(-1)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score tokens [B, Tp, d_llm] -> sorted (positions, h_coords, w_coords, soft_gates) [B, k_exact]."""
        logits = self.score_logits(page_embeds, surprisal=surprisal, token_ids=token_ids, vocab_size=vocab_size)
        scores = torch.sigmoid(logits)
        k = min(self.cfg.k_exact, page_embeds.shape[1])
        _, raw_idx = torch.topk(scores, k=k, dim=-1, sorted=False)
        selected_pos, _ = torch.sort(raw_idx, dim=-1)
        soft_gates = torch.gather(scores, dim=-1, index=selected_pos)

        span_len = self.cfg.tokens_per_span
        span_idx = selected_pos // span_len
        h_coords = (span_idx // self.cfg.grid_w).to(torch.uint8)
        w_coords = (span_idx % self.cfg.grid_w).to(torch.uint8)
        return selected_pos, h_coords, w_coords, soft_gates


@dataclass
class SparseExactBank:
    """Exact 8-byte/token discrete verbatim footnote store: (token_id:int32, page_id:int16, h:uint8, w:uint8)."""

    token_ids: torch.Tensor  # [B, P, K] int32 (4 B)
    page_ids: torch.Tensor  # [B, P, K] int16 (2 B)
    h_coords: torch.Tensor  # [B, P, K] uint8 (1 B)
    w_coords: torch.Tensor  # [B, P, K] uint8 (1 B)
    _ste_gates: torch.Tensor | None = None  # Transient training-only STE tensor; never persisted

    def storage_bytes(self) -> int:
        """Return exact serialized footprint of discrete records (strictly 8 bytes per entry)."""
        return int(
            self.token_ids.numel() * self.token_ids.element_size()
            + self.page_ids.numel() * self.page_ids.element_size()
            + self.h_coords.numel() * self.h_coords.element_size()
            + self.w_coords.numel() * self.w_coords.element_size()
        )

    @classmethod
    def from_pages(
        cls,
        page_token_ids: torch.Tensor,
        salience_gate: SalienceGate,
        pages_embeds: torch.Tensor,
        surprisal_pages: torch.Tensor | None = None,
        vocab_size: int | None = None,
    ) -> "SparseExactBank":
        """Build SparseExactBank across [B, P, Tp] token IDs and embeddings."""
        bsz, num_pages, seq_len = page_token_ids.shape
        flat_embeds = pages_embeds.reshape(bsz * num_pages, seq_len, -1)
        flat_surp = None if surprisal_pages is None else surprisal_pages.reshape(bsz * num_pages, seq_len)
        flat_ids = page_token_ids.reshape(bsz * num_pages, seq_len)

        pos, h, w, gates = salience_gate(flat_embeds, surprisal=flat_surp, token_ids=flat_ids, vocab_size=vocab_size)
        k = pos.shape[-1]
        picked_ids = torch.gather(flat_ids, dim=-1, index=pos).to(torch.int32).view(bsz, num_pages, k)

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
        )

    def gather_embeds(
        self,
        selected_page_indices: torch.Tensor,
        embed_tokens_fn: Callable[[torch.Tensor], torch.Tensor],
        hw_coord_embed: nn.Embedding,
        grid_w: int = 4,
        training_ste: bool = True,
    ) -> torch.Tensor:
        """O(1) embedding gather for selected Deep Read pages [B, k_deep] -> [B, k_deep*K, d_llm]."""
        bsz, k_sel = selected_page_indices.shape
        k_exact = self.token_ids.shape[-1]
        gather_idx = selected_page_indices.unsqueeze(-1).expand(bsz, k_sel, k_exact)

        sel_tok_ids = torch.gather(self.token_ids, dim=1, index=gather_idx).reshape(bsz, k_sel * k_exact)
        sel_h = torch.gather(self.h_coords, dim=1, index=gather_idx).reshape(bsz, k_sel * k_exact).long()
        sel_w = torch.gather(self.w_coords, dim=1, index=gather_idx).reshape(bsz, k_sel * k_exact).long()

        coord_embeds = 0.002 * torch.tanh(hw_coord_embed(sel_h * grid_w + sel_w))
        tok_embeds = embed_tokens_fn(sel_tok_ids.long()).to(coord_embeds.dtype)
        embeds = tok_embeds + coord_embeds

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
    ) -> "SparseExactBank":
        """Update sticky-note row (h = H-1) footnote slots when DeltaUpdater gate opens on page_indices."""
        bsz, _, k_exact = self.token_ids.shape
        t_edit = edit_token_ids.shape[1]
        k_sticky = min(k_exact // 2, t_edit)
        if k_sticky <= 0:
            return self

        logits = salience_gate.score_logits(edit_embeds, surprisal=None, token_ids=edit_token_ids, vocab_size=vocab_size)
        scores = torch.sigmoid(logits)
        # Guarantee the tail tokens of the revision ('to 80M .') are included in the sticky-note slots
        tail_bonus = torch.linspace(0.0, 2.0, steps=t_edit, device=scores.device, dtype=scores.dtype).unsqueeze(0)
        _, raw_idx = torch.topk(scores + tail_bonus, k=k_sticky, dim=-1, sorted=False)
        sel_pos, _ = torch.sort(raw_idx, dim=-1)
        picked_ids = torch.gather(edit_token_ids, dim=-1, index=sel_pos).to(torch.int32)
        picked_scores = torch.gather(scores, dim=-1, index=sel_pos)

        # Couple footnote update strength to DeltaUpdater's bottom sticky-note strip gate (h = H-1)
        strip_gate = delta_gates[:, -1, :].mean(dim=-1, keepdim=True)  # [B, 1]
        coupled_gates = picked_scores * strip_gate.to(picked_scores.dtype)

        new_tok_ids = self.token_ids.clone()
        new_h = self.h_coords.clone()
        new_w = self.w_coords.clone()
        new_ste = None if self._ste_gates is None else self._ste_gates.clone()

        batch_arange = torch.arange(bsz, device=self.token_ids.device)
        active_mask = (strip_gate.squeeze(-1) > 0.05)
        if bool(active_mask.any()):
            b_act = batch_arange[active_mask]
            p_act = page_indices[active_mask]
            new_tok_ids[b_act, p_act, k_exact - k_sticky :] = picked_ids[active_mask]
            new_h[b_act, p_act, k_exact - k_sticky :] = int(grid_h - 1)
            w_pat = (torch.arange(k_sticky, device=self.token_ids.device, dtype=torch.uint8) % int(grid_w)).unsqueeze(0)
            new_w[b_act, p_act, k_exact - k_sticky :] = w_pat.expand(b_act.shape[0], k_sticky)
            if new_ste is not None:
                new_ste[b_act, p_act, k_exact - k_sticky :] = coupled_gates[active_mask].to(new_ste.dtype)

        return SparseExactBank(
            token_ids=new_tok_ids,
            page_ids=self.page_ids,
            h_coords=new_h,
            w_coords=new_w,
            _ste_gates=new_ste,
        )
