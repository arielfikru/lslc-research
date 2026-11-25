"""Multi-Resolution SpatialLatentProjector, QueryToTOCRouter, and Layer-Ordered BookPrefixAssembler."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LSLCConfig
from .encoder import SparseExactBank
from .rope_3d import PageRoPE1D


@dataclass
class RouterOutput:
    """Output of QueryToTOCRouter partitioning P pages into Deep, Skim, and TOC indices."""

    logits: torch.Tensor  # [B, P]
    probs: torch.Tensor  # [B, P]
    deep_idx: torch.Tensor  # [B, k_deep]
    skim_idx: torch.Tensor  # [B, k_skim]
    toc_idx: torch.Tensor  # [B, k_toc]


@dataclass
class AssemblyOutput:
    """Mode A prefix embeddings plus exact slice boundaries for prefix KV-cache invariance checks."""

    prefix_embeds: torch.Tensor  # [B, prefix_len, d_llm]
    immutable_prefix_len: int  # Length of [TOC | Skim | Deep l=0 | Deep l=1]
    slice_bounds: dict[str, tuple[int, int]]


class _LayerProjectorMLP(nn.Module):
    def __init__(self, in_dim: int, d_llm: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialLatentProjector(nn.Module):
    """Projects book tensor Z [B, P, 3, C, 4, 4] into TOC (1x1), Skim (2x2), and layer-ordered Deep Read."""

    def __init__(self, cfg: LSLCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c = cfg.latent_channels

        # IMPORTANT: TOC and Skim read ONLY consolidated structural layers Z[:, :, :2] (l=0 Static + l=1 Entity)
        # so sticky-note delta-updates on dynamic layer l=2 preserve strict max_abs_diff == 0.0 prefix invariance!
        self.toc_proj = _LayerProjectorMLP(2 * c, cfg.d_llm)
        self.toc_slot_attn = nn.Linear(2 * c, 1) if getattr(cfg, "router_mode", "v2_legacy") == "content_attn" else None
        self.skim_proj = _LayerProjectorMLP(2 * c, cfg.d_llm)
        self.skim_pos = nn.Parameter(torch.randn(1, 1, 4, cfg.d_llm) * 0.02)

        # Deep Read uses three independent layer projections (4 + 4 + 16 = 24 dense tokens/page)
        self.deep_l0_proj = _LayerProjectorMLP(c, cfg.d_llm)
        self.deep_l0_pos = nn.Parameter(torch.randn(1, 1, 4, cfg.d_llm) * 0.02)

        self.deep_l1_proj = _LayerProjectorMLP(c, cfg.d_llm)
        self.deep_l1_pos = nn.Parameter(torch.randn(1, 1, 4, cfg.d_llm) * 0.02)

        self.deep_l2_proj = _LayerProjectorMLP(c, cfg.d_llm)
        self.deep_l2_pos = nn.Parameter(torch.randn(1, 1, cfg.num_spatial_slots, cfg.d_llm) * 0.02)

    def _project_structural_2x2(self, z_book: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project consolidated structural layers l=0 and l=1 onto 2x2 grids [B, P, 4, d_llm]."""
        bsz, pages, _, c, h, w = z_book.shape
        l0_grid = F.avg_pool2d(z_book[:, :, 0].reshape(bsz * pages, c, h, w), kernel_size=2)
        l0_toks = self.deep_l0_proj(l0_grid.flatten(2).transpose(1, 2).reshape(bsz, pages, 4, c)) + self.deep_l0_pos

        l1_grid = F.avg_pool2d(z_book[:, :, 1].reshape(bsz * pages, c, h, w), kernel_size=2)
        l1_toks = self.deep_l1_proj(l1_grid.flatten(2).transpose(1, 2).reshape(bsz, pages, 4, c)) + self.deep_l1_pos
        return l0_toks, l1_toks

    def project_toc(self, z_book: torch.Tensor) -> torch.Tensor:
        """Level 0 TOC (1x1) nested over structural layers l=0,1: [B, P, 3, C, H, W] -> [B, P, 1, d_llm]."""
        bsz, pages, _, c, h, w = z_book.shape
        l0_toks, l1_toks = self._project_structural_2x2(z_book)
        struct_1x1 = 0.5 * (l0_toks + l1_toks).mean(dim=2, keepdim=True)
        struct = z_book[:, :, :2].reshape(bsz * pages, 2 * c, h, w)
        if self.toc_slot_attn is not None:
            struct_slots = struct.flatten(2).transpose(1, 2).reshape(bsz, pages, h * w, 2 * c)
            slot_w = F.softmax(self.toc_slot_attn(struct_slots), dim=2)
            pooled = (slot_w * struct_slots).sum(dim=2, keepdim=True)
        else:
            pooled = F.avg_pool2d(struct, kernel_size=(h, w)).reshape(bsz, pages, 1, 2 * c)
        return struct_1x1 + self.toc_proj(pooled)

    def project_skim(self, z_book: torch.Tensor) -> torch.Tensor:
        """Level 1 Skim (2x2) nested over structural layers l=0,1: [B, P, 3, C, H, W] -> [B, P, 4, d_llm]."""
        bsz, pages, _, c, h, w = z_book.shape
        l0_toks, l1_toks = self._project_structural_2x2(z_book)
        struct_2x2 = 0.5 * (l0_toks + l1_toks)
        struct = z_book[:, :, :2].reshape(bsz * pages, 2 * c, h, w)
        pooled = F.avg_pool2d(struct, kernel_size=2)  # [B*P, 2*C, 2, 2]
        tokens = pooled.flatten(2).transpose(1, 2).reshape(bsz, pages, 4, 2 * c)
        return struct_2x2 + self.skim_proj(tokens) + self.skim_pos

    def project_deep(self, z_book: torch.Tensor) -> dict[str, torch.Tensor]:
        """Level 2 Deep Read keeping l=0 (4 tok), l=1 (4 tok), l=2 (16 tok) strictly separate."""
        bsz, pages, _, c, h, w = z_book.shape
        l0_toks, l1_toks = self._project_structural_2x2(z_book)

        l2_raw = z_book[:, :, 2].reshape(bsz * pages, c, h, w).flatten(2).transpose(1, 2).reshape(bsz, pages, h * w, c)
        l2_toks = self.deep_l2_proj(l2_raw) + self.deep_l2_pos

        return {"l0": l0_toks, "l1": l1_toks, "l2": l2_toks}


def _run_fp32_mlp(mlp: nn.Sequential, x: torch.Tensor, training: bool) -> torch.Tensor:
    """Execute [LayerNorm, Dropout, Linear, GELU, Linear] strictly in FP32 to prevent BF16 logit quantization."""
    ln, drop, fc1, _, fc2 = mlp
    h = F.layer_norm(x.float(), (x.shape[-1],), ln.weight.float(), ln.bias.float())
    h = F.dropout(h, p=float(drop.p), training=training)
    h = F.gelu(F.linear(h, fc1.weight.float(), fc1.bias.float()))
    return F.linear(h, fc2.weight.float(), fc2.bias.float())


class QueryToTOCRouter(nn.Module):
    """Routes user query against all page TOC embeddings using 1D PageRoPE (v2_legacy) or permutation-invariant cosine attention (content_attn)."""

    def __init__(
        self,
        d_llm: int,
        d_route: int = 256,
        router_mode: str = "v2_legacy",
        use_contextual_query_encoding: bool = False,
    ) -> None:
        super().__init__()
        if router_mode not in ("v2_legacy", "content_attn"):
            raise ValueError(f"router_mode must be 'v2_legacy' or 'content_attn', got {router_mode!r}.")
        self.d_route = d_route
        self.router_mode = router_mode
        self.use_contextual_query_encoding = bool(use_contextual_query_encoding)
        use_bias = router_mode == "v2_legacy"
        self.q_proj = nn.Linear(d_llm, d_route, bias=use_bias)
        self.k_proj = nn.Linear(d_llm, d_route, bias=use_bias)
        self.page_rope = PageRoPE1D(d_route)
        if router_mode == "content_attn":
            with torch.no_grad():
                self.k_proj.weight.copy_(self.q_proj.weight)
            self.q_norm = nn.LayerNorm(d_llm)
            self.k_norm = nn.LayerNorm(d_llm)
            self.q_token_gate = nn.Linear(d_llm, 1)
            self.bank_key_gate = nn.Linear(d_llm, 1)
            self.bank_key_alpha = nn.Parameter(torch.tensor(0.5))
            self.logit_scale = nn.Parameter(torch.tensor(math.log(4.0)))
        else:
            self.q_norm = None
            self.k_norm = None
            self.q_token_gate = None
            self.bank_key_gate = None
            self.bank_key_alpha = None
            self.logit_scale = None
        if router_mode == "content_attn" and self.use_contextual_query_encoding:
            self.q_ctx_gate = nn.Linear(d_llm, 1)
            self.q_ctx_mlp = nn.Sequential(
                nn.LayerNorm(d_llm),
                nn.Dropout(0.1),
                nn.Linear(d_llm, d_route),
                nn.GELU(),
                nn.Linear(d_route, d_route),
            )
            self.k_ctx_mlp = nn.Sequential(
                nn.LayerNorm(d_llm),
                nn.Dropout(0.1),
                nn.Linear(d_llm, d_route),
                nn.GELU(),
                nn.Linear(d_route, d_route),
            )
            self.k_toc_mlp = nn.Sequential(
                nn.LayerNorm(d_llm),
                nn.Dropout(0.1),
                nn.Linear(d_llm, d_route),
                nn.GELU(),
                nn.Linear(d_route, d_route),
            )
            self.k_bank_mlp = nn.Sequential(
                nn.LayerNorm(d_llm),
                nn.Dropout(0.1),
                nn.Linear(d_llm, d_route),
                nn.GELU(),
                nn.Linear(d_route, d_route),
            )
            for mlp in (self.q_ctx_mlp, self.k_ctx_mlp, self.k_toc_mlp, self.k_bank_mlp):
                nn.init.normal_(mlp[4].weight, mean=0.0, std=0.02)
                nn.init.zeros_(mlp[4].bias)
        else:
            self.q_ctx_gate = None
            self.q_ctx_mlp = None
            self.k_ctx_mlp = None
            self.k_toc_mlp = None
            self.k_bank_mlp = None
        self.scale = d_route ** -0.5

    def get_logit_scale(self) -> torch.Tensor:
        """Return active positive logit scale in float32."""
        if self.router_mode == "content_attn":
            if self.logit_scale is not None:
                return self.logit_scale.float().exp().clamp(max=50.0)
            return torch.tensor(12.0, device=self.q_proj.weight.device, dtype=torch.float32)
        return torch.tensor(self.scale, device=self.q_proj.weight.device, dtype=torch.float32)

    def encode_query(
        self,
        query_embed: torch.Tensor,
        query_token_embeds: torch.Tensor | None = None,
        contextual_query_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode query embeddings (and optional mid-layer contextual states) into normalized [B, d_route]."""
        if self.router_mode != "content_attn":
            q_in = query_embed if torch.is_autocast_enabled() else query_embed.to(self.q_proj.weight.dtype)
            return self.q_proj(q_in)

        base_tok = contextual_query_embeds if contextual_query_embeds is not None else query_token_embeds
        if base_tok is not None and self.q_token_gate is not None:
            q_tok = base_tok.float()
            q_w = F.softmax(
                F.linear(q_tok, self.q_token_gate.weight.float(), self.q_token_gate.bias.float()),
                dim=1,
            )
            q_gated = (q_w * q_tok).sum(dim=1)
            q_in = 0.5 * q_tok.mean(dim=1) + 0.5 * q_gated if contextual_query_embeds is not None else q_gated
        else:
            q_in = query_embed.float()

        if self.q_norm is not None:
            q_in = F.layer_norm(q_in, (q_in.shape[-1],), self.q_norm.weight.float(), self.q_norm.bias.float())

        q_raw = F.linear(q_in, self.q_proj.weight.float())
        if self.q_ctx_mlp is not None and self.q_ctx_gate is not None:
            if contextual_query_embeds is not None:
                q_ctx_tok = contextual_query_embeds.float()
            elif query_token_embeds is not None:
                q_ctx_tok = query_token_embeds.float()
            else:
                q_ctx_tok = query_embed.unsqueeze(1).float()
            qc_w = F.softmax(
                F.linear(q_ctx_tok, self.q_ctx_gate.weight.float(), self.q_ctx_gate.bias.float()),
                dim=1,
            )
            q_ctx_pool = 0.5 * (qc_w * q_ctx_tok).sum(dim=1) + 0.5 * q_ctx_tok[:, -1, :]
            q_raw = q_raw + _run_fp32_mlp(self.q_ctx_mlp, q_ctx_pool, self.training)
        return F.normalize(q_raw, dim=-1)

    def encode_keys(
        self,
        toc_embeds: torch.Tensor,
        exact_bank_embeds: torch.Tensor | None = None,
        center_across_pages: bool = True,
        contextual_page_keys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode [B, P, 1, d_llm] TOC embeddings (+ optional exact bank & mid-layer page keys) into normalized [B, P, d_route]."""
        if self.router_mode != "content_attn":
            t_in = toc_embeds.squeeze(2)
            if not torch.is_autocast_enabled():
                t_in = t_in.to(self.k_proj.weight.dtype)
            return self.k_proj(t_in)

        num_pages = toc_embeds.shape[1]
        toc_raw = toc_embeds.squeeze(2).float()
        b_pool: torch.Tensor | None = None
        if (
            exact_bank_embeds is not None
            and exact_bank_embeds.numel() > 0
            and self.bank_key_gate is not None
            and self.bank_key_alpha is not None
        ):
            b_emb = exact_bank_embeds.float()
            b_w = F.softmax(
                F.linear(b_emb, self.bank_key_gate.weight.float(), self.bank_key_gate.bias.float()),
                dim=2,
            )
            b_pool = (b_w * b_emb).sum(dim=2)

        if contextual_page_keys is not None:
            k_ctx = contextual_page_keys.float()
            is_multi_span = k_ctx.dim() == 4
            if self.q_norm is not None:
                k_ctx_in = F.layer_norm(
                    k_ctx, (k_ctx.shape[-1],), self.q_norm.weight.float(), self.q_norm.bias.float()
                )
            else:
                k_ctx_in = k_ctx
            k_raw = F.linear(k_ctx_in, self.q_proj.weight.float())
            if self.k_ctx_mlp is not None:
                k_raw = k_raw + _run_fp32_mlp(self.k_ctx_mlp, k_ctx, self.training)
        else:
            is_multi_span = False
            p_key = toc_raw
            if b_pool is not None and self.bank_key_alpha is not None:
                p_key = p_key + self.bank_key_alpha.float() * b_pool
            if self.k_norm is not None:
                p_key = F.layer_norm(p_key, (p_key.shape[-1],), self.k_norm.weight.float(), self.k_norm.bias.float())
            k_raw = F.linear(p_key, self.k_proj.weight.float())

        if self.k_toc_mlp is not None:
            toc_res = _run_fp32_mlp(self.k_toc_mlp, toc_raw, self.training)
            k_raw = k_raw + (toc_res.unsqueeze(2) if is_multi_span else toc_res)
        if b_pool is not None and self.k_bank_mlp is not None:
            bank_res = _run_fp32_mlp(self.k_bank_mlp, b_pool, self.training)
            k_raw = k_raw + (bank_res.unsqueeze(2) if is_multi_span else bank_res)
        if center_across_pages and num_pages > 1:
            if is_multi_span:
                k_raw = k_raw - k_raw.mean(dim=(1, 2), keepdim=True)
            else:
                k_raw = k_raw - k_raw.mean(dim=1, keepdim=True)
        return F.normalize(k_raw, dim=-1)

    def forward(
        self,
        query_embed: torch.Tensor,
        toc_embeds: torch.Tensor,
        k_deep: int,
        k_skim: int,
        query_token_embeds: torch.Tensor | None = None,
        exact_bank_embeds: torch.Tensor | None = None,
        salience_route_weight: float = 8.0,
        delta_bank_embeds: torch.Tensor | None = None,
        contextual_query_embeds: torch.Tensor | None = None,
        contextual_page_keys: torch.Tensor | None = None,
        external_dense_logits: torch.Tensor | None = None,
    ) -> RouterOutput:
        """Route [B, d_llm] query over [B, P, 1, d_llm] page TOC tokens (+ optional in-VRAM SparseExactBank MaxSim)."""
        bsz, num_pages, _, _ = toc_embeds.shape
        device = toc_embeds.device
        if not torch.is_autocast_enabled():
            toc_embeds = toc_embeds.to(self.k_proj.weight.dtype)
            query_embed = query_embed.to(self.q_proj.weight.dtype)

        if external_dense_logits is not None:
            logits = external_dense_logits.to(device=device, dtype=torch.float32)
        elif self.router_mode == "content_attn":
            # Keep routing strictly in float32 to prevent BF16 logit quantization collapse
            q = self.encode_query(
                query_embed,
                query_token_embeds=query_token_embeds,
                contextual_query_embeds=contextual_query_embeds,
            ).unsqueeze(1)
            k = self.encode_keys(
                toc_embeds,
                exact_bank_embeds=exact_bank_embeds,
                center_across_pages=True,
                contextual_page_keys=contextual_page_keys,
            )
            scale = self.get_logit_scale()
            if k.dim() == 4:
                # MaxSim across S sub-page spans: max_s cos(q, k_span[p, s])
                logits = torch.einsum("bd,bpsd->bps", q.squeeze(1), k).amax(dim=-1) * scale  # float32 [B, P]
            else:
                logits = torch.bmm(q, k.transpose(1, 2)).squeeze(1) * scale  # float32 [B, P]
        else:
            k = self.k_proj(toc_embeds.squeeze(2))  # [B, P, d_route]
            q = self.q_proj(query_embed).unsqueeze(1)  # [B, 1, d_route]

            k = self.page_rope(k, torch.arange(num_pages, device=device))
            q = self.page_rope(q, torch.tensor([num_pages], device=device))

            logits = torch.bmm(q, k.transpose(1, 2)).squeeze(1) * self.scale  # [B, P]

        if (
            query_token_embeds is not None
            and exact_bank_embeds is not None
            and exact_bank_embeds.numel() > 0
            and salience_route_weight > 0.0
            and num_pages > 1
        ):
            q_norm = F.normalize(query_token_embeds.float(), dim=-1)  # [B, M, d]
            page_sigs = torch.cat((exact_bank_embeds.float(), toc_embeds.float()), dim=2)  # [B, P, K+1, d]
            b_norm = F.normalize(page_sigs, dim=-1)
            sim = torch.einsum("bmd,bpkd->bpmk", q_norm, b_norm)  # [B, P, M, K+1]
            max_per_q = sim.amax(dim=-1)  # [B, P, M]
            if self.router_mode == "content_attn":
                # Pure-GPU thresholded Hit-Inverse-DF^2 + contiguous bigram/trigram matching over
                # immutable original page key banks and (optionally) live post-delta sticky banks.
                hit1 = torch.sigmoid((max_per_q - 0.85) * 20.0)  # [B, P, M]
                sim_bank = sim[:, :, :, :-1]  # [B, P, M, K] (source-ordered exact bank tokens)
                hit_b = torch.sigmoid((sim_bank - 0.85) * 20.0)
                bi = (
                    (hit_b[:, :, :-1, :-1] * hit_b[:, :, 1:, 1:]).amax(dim=-1)
                    if (query_token_embeds.shape[1] >= 2 and exact_bank_embeds.shape[2] >= 2)
                    else None
                )
                tri = (
                    (hit_b[:, :, :-2, :-2] * hit_b[:, :, 1:-1, 1:-1] * hit_b[:, :, 2:, 2:]).amax(dim=-1)
                    if (query_token_embeds.shape[1] >= 3 and exact_bank_embeds.shape[2] >= 3)
                    else None
                )
                if delta_bank_embeds is not None and delta_bank_embeds.numel() > 0:
                    d_norm = F.normalize(delta_bank_embeds.float(), dim=-1)
                    sim_d = torch.einsum("bmd,bpkd->bpmk", q_norm, d_norm)
                    hit_d = torch.sigmoid((sim_d - 0.85) * 20.0)
                    hit1 = torch.maximum(hit1, hit_d.amax(dim=-1))
                    if bi is not None and delta_bank_embeds.shape[2] >= 2:
                        bi_d = (hit_d[:, :, :-1, :-1] * hit_d[:, :, 1:, 1:]).amax(dim=-1)
                        bi = torch.maximum(bi, bi_d)
                    if tri is not None and delta_bank_embeds.shape[2] >= 3:
                        tri_d = (
                            hit_d[:, :, :-2, :-2] * hit_d[:, :, 1:-1, 1:-1] * hit_d[:, :, 2:, 2:]
                        ).amax(dim=-1)
                        tri = torch.maximum(tri, tri_d)
                df1 = hit1.sum(dim=1, keepdim=True)  # [B, 1, M]
                w1 = (df1 > 0.3).to(hit1.dtype) / (df1 + 0.5).pow(2)
                sal_score = (hit1 * w1).sum(dim=-1) / (w1.sum(dim=-1) + 1e-6)  # [B, P]
                if bi is not None:
                    df2 = bi.sum(dim=1, keepdim=True)
                    w2 = (df2 > 0.3).to(bi.dtype) / (df2 + 0.5).pow(2)
                    sal_score = sal_score + 2.0 * ((bi * w2).sum(dim=-1) / (w2.sum(dim=-1) + 1e-6))
                if tri is not None:
                    df3 = tri.sum(dim=1, keepdim=True)
                    w3 = (df3 > 0.3).to(tri.dtype) / (df3 + 0.5).pow(2)
                    sal_score = sal_score + 3.0 * ((tri * w3).sum(dim=-1) / (w3.sum(dim=-1) + 1e-6))
            else:
                top_m = min(4, max_per_q.shape[-1])
                sal_score = torch.topk(max_per_q, k=top_m, dim=-1).values.mean(dim=-1)  # [B, P]
            if self.router_mode == "content_attn":
                sal_std = sal_score.std(dim=-1, keepdim=True)
                sal_z = (sal_score - sal_score.mean(dim=-1, keepdim=True)) / sal_std.clamp(min=0.35)
                sal_conf = (sal_score.amax(dim=-1, keepdim=True) / 1.5).clamp(0.0, 1.0)
                dense_f = logits.float()
                dense_std = dense_f.std(dim=-1, keepdim=True)
                dense_z = torch.where(
                    dense_std > 1e-6,
                    (dense_f - dense_f.mean(dim=-1, keepdim=True)) / (dense_std + 1e-6),
                    torch.zeros_like(dense_f),
                )
                logits = (dense_z + (salience_route_weight * sal_conf) * sal_z).to(logits.dtype)
            else:
                sal_std = sal_score.std(dim=-1, keepdim=True)
                sal_z = torch.where(
                    sal_std > 1e-6,
                    (sal_score - sal_score.mean(dim=-1, keepdim=True)) / (sal_std + 1e-6),
                    torch.zeros_like(sal_score),
                )
                logits = logits + (salience_route_weight * sal_z).to(logits.dtype)

        probs = F.softmax(logits, dim=-1)

        kd = min(num_pages, max(0, k_deep))
        ks = min(max(0, num_pages - kd), max(0, k_skim))

        sorted_idx = torch.argsort(logits, dim=-1, descending=True)
        # Order deep_idx by ascending router score so the #1 highest-scoring page (sorted_idx[:, 0])
        # is always placed last (immediately adjacent to the user query across all needle depths 0.0..1.0).
        deep_idx = sorted_idx[:, :kd].flip(dims=[-1])
        skim_idx = torch.sort(sorted_idx[:, kd : kd + ks], dim=-1).values
        toc_idx = torch.sort(sorted_idx[:, kd + ks :], dim=-1).values
        return RouterOutput(logits=logits, probs=probs, deep_idx=deep_idx, skim_idx=skim_idx, toc_idx=toc_idx)


def _gather_book_pages(z_book: torch.Tensor, page_indices: torch.Tensor) -> torch.Tensor:
    """Gather selected latent pages [B, K, 3, C, H, W] from [B, P, 3, C, H, W] prior to MLP projection."""
    bsz, _, layers, ch, h, w = z_book.shape
    k = page_indices.shape[1]
    if k == 0:
        return z_book.new_zeros((bsz, 0, layers, ch, h, w))
    idx = page_indices.view(bsz, k, 1, 1, 1, 1).expand(bsz, k, layers, ch, h, w)
    return torch.gather(z_book, dim=1, index=idx)


def _gather_page_tokens(tensor: torch.Tensor, page_indices: torch.Tensor) -> torch.Tensor:
    """Gather [B, K, T_per_page, d_llm] from [B, P, T_per_page, d_llm] -> flattened [B, K*T_per_page, d_llm]."""
    bsz, _, tok_per_page, d_llm = tensor.shape
    k = page_indices.shape[1]
    if k == 0:
        return tensor.new_zeros((bsz, 0, d_llm))
    idx = page_indices.unsqueeze(-1).unsqueeze(-1).expand(bsz, k, tok_per_page, d_llm)
    return torch.gather(tensor, dim=1, index=idx).reshape(bsz, k * tok_per_page, d_llm)


class BookPrefixAssembler(nn.Module):
    """Assembles Mode A soft-prompt prefix in strict volatility order for KV prefix-cache synergy."""

    def __init__(self, cfg: LSLCConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        z_book: torch.Tensor,
        projector: SpatialLatentProjector,
        router_out: RouterOutput,
        exact_bank: SparseExactBank | None = None,
        embed_tokens_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        hw_coord_embed: nn.Embedding | None = None,
        mode: str = "full",
        training_ste: bool = True,
        hierarchical_chapter_size: int | None = None,
        flat_toc_max_tokens: int | None = None,
        toc_embeds: torch.Tensor | None = None,
        exact_bank_attenuation: str = "prob_sq",
    ) -> AssemblyOutput:
        """Assemble Mode A causal prefix: [TOC | Skim | Deep l0 | Deep l1 | Deep l2 | Exact]."""
        if exact_bank_attenuation not in ("prob_sq", "none"):
            raise ValueError(
                f"exact_bank_attenuation must be 'prob_sq' or 'none', got {exact_bank_attenuation!r}"
            )
        bsz, num_pages, _, _, _, _ = z_book.shape
        device = z_book.device

        if mode == "exact_only":
            if exact_bank is None or router_out.deep_idx.shape[1] == 0:
                empty = torch.zeros((bsz, 0, self.cfg.d_llm), device=device)
                return AssemblyOutput(prefix_embeds=empty, immutable_prefix_len=0, slice_bounds={"exact": (0, 0)})
            if embed_tokens_fn is None or hw_coord_embed is None:
                raise ValueError("embed_tokens_fn and hw_coord_embed must be provided when exact_bank is set.")
            exact_toks = exact_bank.gather_embeds(
                router_out.deep_idx,
                embed_tokens_fn=embed_tokens_fn,
                hw_coord_embed=hw_coord_embed,
                grid_w=self.cfg.grid_w,
                training_ste=training_ste,
            )
            if exact_bank_attenuation == "prob_sq":
                kd = router_out.deep_idx.shape[1]
                k_ex = exact_bank.token_ids.shape[-1]
                p_deep = torch.gather(router_out.probs, dim=1, index=router_out.deep_idx)
                r_conf = (p_deep / p_deep.amax(dim=-1, keepdim=True).clamp(min=1e-6)) ** 2
                exact_toks = (
                    exact_toks.view(bsz, kd, k_ex, -1) * r_conf.to(exact_toks.dtype).unsqueeze(-1).unsqueeze(-1)
                ).view(bsz, kd * k_ex, -1)
            return AssemblyOutput(
                prefix_embeds=exact_toks,
                immutable_prefix_len=0,
                slice_bounds={"exact": (0, exact_toks.shape[1])},
            )

        toc_all = projector.project_toc(z_book) if toc_embeds is None else toc_embeds

        if mode == "toc_only":
            seq = toc_all.reshape(bsz, num_pages, self.cfg.d_llm)
            return AssemblyOutput(
                prefix_embeds=seq,
                immutable_prefix_len=seq.shape[1],
                slice_bounds={"toc": (0, seq.shape[1])},
            )

        if mode == "skim_only":
            skim_all = projector.project_skim(z_book).reshape(bsz, num_pages * 4, self.cfg.d_llm)
            return AssemblyOutput(
                prefix_embeds=skim_all,
                immutable_prefix_len=skim_all.shape[1],
                slice_bounds={"skim": (0, skim_all.shape[1])},
            )

        if mode == "deep_only":
            deep = projector.project_deep(z_book)
            l0 = deep["l0"].reshape(bsz, num_pages * 4, self.cfg.d_llm)
            l1 = deep["l1"].reshape(bsz, num_pages * 4, self.cfg.d_llm)
            l2 = deep["l2"].reshape(bsz, num_pages * self.cfg.num_spatial_slots, self.cfg.d_llm)
            imm_len = l0.shape[1] + l1.shape[1]
            seq = torch.cat((l0, l1, l2), dim=1)
            return AssemblyOutput(
                prefix_embeds=seq,
                immutable_prefix_len=imm_len,
                slice_bounds={
                    "deep_l0": (0, l0.shape[1]),
                    "deep_l1": (l0.shape[1], imm_len),
                    "deep_l2": (imm_len, seq.shape[1]),
                },
            )

        if mode != "full":
            raise ValueError(f"Unknown BookPrefixAssembler mode: {mode}")

        # Gather-first O(k_deep + k_skim) projection instead of O(P) full-book MLP pass
        if router_out.skim_idx.shape[1] > 0:
            z_skim = _gather_book_pages(z_book, router_out.skim_idx)
            skim_toks = projector.project_skim(z_skim).reshape(bsz, -1, self.cfg.d_llm)
        else:
            skim_toks = z_book.new_zeros((bsz, 0, self.cfg.d_llm))

        if router_out.deep_idx.shape[1] > 0:
            z_deep = _gather_book_pages(z_book, router_out.deep_idx)
            deep_sel = projector.project_deep(z_deep)
            l0_toks = deep_sel["l0"].reshape(bsz, -1, self.cfg.d_llm)
            l1_toks = deep_sel["l1"].reshape(bsz, -1, self.cfg.d_llm)
            l2_toks = deep_sel["l2"].reshape(bsz, -1, self.cfg.d_llm)
        else:
            l0_toks = z_book.new_zeros((bsz, 0, self.cfg.d_llm))
            l1_toks = z_book.new_zeros((bsz, 0, self.cfg.d_llm))
            l2_toks = z_book.new_zeros((bsz, 0, self.cfg.d_llm))

        extra_toc_bounds: dict[str, tuple[int, int]] = {}
        if hierarchical_chapter_size is not None and hierarchical_chapter_size > 0 and num_pages > hierarchical_chapter_size:
            chapter_size = int(hierarchical_chapter_size)
            num_chapters = (num_pages + chapter_size - 1) // chapter_size
            chapter_parts = []
            for c in range(num_chapters):
                s_idx = c * chapter_size
                e_idx = min(num_pages, s_idx + chapter_size)
                chapter_parts.append(toc_all[:, s_idx:e_idx, 0, :].mean(dim=1, keepdim=True))
            chapter_tocs = torch.cat(chapter_parts, dim=1)  # [B, num_chapters, d_llm]

            per_batch_exp: list[list[int]] = []
            for b in range(bsz):
                deep_set = set(router_out.deep_idx[b].tolist())
                skim_set = set(router_out.skim_idx[b].tolist())
                sel_chapters = sorted({p // chapter_size for p in deep_set})
                exp_pages_b = [
                    p
                    for c in sel_chapters
                    for p in range(c * chapter_size, min(num_pages, (c + 1) * chapter_size))
                    if p not in deep_set and p not in skim_set
                ]
                per_batch_exp.append(exp_pages_b)

            max_exp = max((len(x) for x in per_batch_exp), default=0)
            if max_exp == 0:
                expanded_toc_toks = toc_all.new_zeros((bsz, 0, self.cfg.d_llm))
            else:
                exp_tensors = []
                for b in range(bsz):
                    pages_b = per_batch_exp[b]
                    if pages_b:
                        idx_b = torch.tensor(pages_b, device=device, dtype=torch.long)
                        t_b = toc_all[b, idx_b, 0, :]  # [n_exp, d_llm]
                    else:
                        t_b = toc_all.new_zeros((0, self.cfg.d_llm))
                    if len(pages_b) < max_exp:
                        pad_b = toc_all.new_zeros((max_exp - len(pages_b), self.cfg.d_llm))
                        t_b = torch.cat((t_b, pad_b), dim=0)
                    exp_tensors.append(t_b)
                expanded_toc_toks = torch.stack(exp_tensors, dim=0)  # [B, max_exp, d_llm]

            toc_toks = torch.cat((chapter_tocs, expanded_toc_toks), dim=1)
            extra_toc_bounds["chapter_toc"] = (0, chapter_tocs.shape[1])
            extra_toc_bounds["expanded_toc"] = (chapter_tocs.shape[1], toc_toks.shape[1])
        else:
            toc_idx_use = router_out.toc_idx
            if flat_toc_max_tokens is not None and flat_toc_max_tokens > 0 and toc_idx_use.shape[1] > flat_toc_max_tokens:
                # Uniformly subsample unrouted flat TOC indices to match a target token budget (ablation control)
                n_toc = toc_idx_use.shape[1]
                sel_pos = torch.linspace(0, n_toc - 1, steps=int(flat_toc_max_tokens), device=device).long()
                toc_idx_use = toc_idx_use[:, sel_pos]
            toc_toks = _gather_page_tokens(toc_all, toc_idx_use)

        # (skim_toks and deep l0/l1/l2 tokens were projected gather-first above)

        # Provide straight-through gradient path into QueryToTOCRouter via dynamic l=2 Deep Read tokens
        # without altering l0/l1/Skim/TOC prefix invariance when DeltaUpdater edits l=2.
        if training_ste and router_out.deep_idx.shape[1] > 0:
            kd = router_out.deep_idx.shape[1]
            p_deep = torch.gather(router_out.probs, dim=1, index=router_out.deep_idx)  # [B, kd]
            ste_r = (p_deep + (1.0 - p_deep).detach()).unsqueeze(-1).expand(bsz, kd, self.cfg.num_spatial_slots)
            l2_toks = l2_toks * ste_r.reshape(bsz, kd * self.cfg.num_spatial_slots, 1).to(l2_toks.dtype)

        parts = [toc_toks, skim_toks, l0_toks, l1_toks, l2_toks]
        bounds: dict[str, tuple[int, int]] = dict(extra_toc_bounds)
        cursor = 0
        for name, part in zip(("toc", "skim", "deep_l0", "deep_l1", "deep_l2"), parts):
            nxt = cursor + part.shape[1]
            bounds[name] = (cursor, nxt)
            cursor = nxt

        immutable_prefix_len = bounds["deep_l1"][1]

        if exact_bank is not None and router_out.deep_idx.shape[1] > 0:
            if embed_tokens_fn is None or hw_coord_embed is None:
                raise ValueError("embed_tokens_fn and hw_coord_embed must be provided when exact_bank is set.")
            exact_toks = exact_bank.gather_embeds(
                router_out.deep_idx,
                embed_tokens_fn=embed_tokens_fn,
                hw_coord_embed=hw_coord_embed,
                grid_w=self.cfg.grid_w,
                training_ste=training_ste,
            )
            if exact_bank_attenuation == "prob_sq":
                # Attenuate runner-up deep pages' exact footnotes by normalized router probability
                # while keeping the #1 winning page at weight == 1.0000 (100% uncorrupted verbatim embeddings).
                kd = router_out.deep_idx.shape[1]
                k_ex = exact_bank.token_ids.shape[-1]
                p_deep = torch.gather(router_out.probs, dim=1, index=router_out.deep_idx)  # [B, kd]
                r_conf = (p_deep / p_deep.amax(dim=-1, keepdim=True).clamp(min=1e-6)) ** 2
                exact_toks = (exact_toks.view(bsz, kd, k_ex, -1) * r_conf.to(exact_toks.dtype).unsqueeze(-1).unsqueeze(-1)).view(
                    bsz, kd * k_ex, -1
                )
            parts.append(exact_toks)
            bounds["exact"] = (cursor, cursor + exact_toks.shape[1])
        else:
            bounds["exact"] = (cursor, cursor)

        prefix_embeds = torch.cat(parts, dim=1) if parts else torch.zeros((bsz, 0, self.cfg.d_llm), device=device)
        return AssemblyOutput(
            prefix_embeds=prefix_embeds,
            immutable_prefix_len=immutable_prefix_len,
            slice_bounds=bounds,
        )


def compute_router_infonce_loss(
    router: QueryToTOCRouter,
    q_vecs: torch.Tensor,
    k_vecs: torch.Tensor,
    target_page_idx: torch.Tensor,
    *,
    topic_ids: torch.Tensor | None = None,
    in_batch_weight: float = 1.0,
    symmetric_in_batch: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute intra-book + in-batch hard-negative InfoNCE contrastive loss for QueryToTOCRouter (3D or 4D multi-span keys)."""
    bsz = k_vecs.shape[0]
    num_pages = k_vecs.shape[1]
    device = q_vecs.device
    scale = router.get_logit_scale().to(device=device, dtype=torch.float32)

    q_f = q_vecs.float()
    k_f = k_vecs.float()
    tgt = target_page_idx.to(device=device, dtype=torch.long)

    # 1. Intra-book hard-negative InfoNCE over all P pages in the same book
    if k_f.dim() == 4:
        intra_logits = torch.einsum("bd,bpsd->bps", q_f, k_f).amax(dim=-1) * scale  # [B, P]
    else:
        intra_logits = torch.bmm(q_f.unsqueeze(1), k_f.transpose(1, 2)).squeeze(1) * scale  # [B, P]
    loss_intra = F.cross_entropy(intra_logits, tgt)
    intra_top1 = float((intra_logits.argmax(dim=-1) == tgt).float().mean().item())

    # 2. In-batch hard-negative InfoNCE over positive pages across distinct books
    if bsz > 1 and in_batch_weight > 0.0:
        batch_arange = torch.arange(bsz, device=device)
        pos_k = k_f[batch_arange, tgt]  # [B, d_route] or [B, S, d_route]
        if pos_k.dim() == 3:
            sim_bb = torch.einsum("id,jsd->ijs", q_f, pos_k).amax(dim=-1) * scale  # [B, B]
        else:
            sim_bb = torch.matmul(q_f, pos_k.transpose(0, 1)) * scale  # [B, B]
        if topic_ids is not None:
            t_ids = topic_ids.to(device=device)
            eye = torch.eye(bsz, dtype=torch.bool, device=device)
            same_topic_offdiag = (t_ids.unsqueeze(1) == t_ids.unsqueeze(0)) & (~eye)
            sim_bb = sim_bb.masked_fill(same_topic_offdiag, -1e4)
        labels_bb = batch_arange
        loss_q2k = F.cross_entropy(sim_bb, labels_bb)
        if symmetric_in_batch:
            loss_k2q = F.cross_entropy(sim_bb.transpose(0, 1), labels_bb)
            loss_inbatch = 0.5 * (loss_q2k + loss_k2q)
        else:
            loss_inbatch = loss_q2k
        inbatch_top1 = float((sim_bb.argmax(dim=-1) == labels_bb).float().mean().item())
        loss_total = loss_intra + float(in_batch_weight) * loss_inbatch
    else:
        loss_inbatch = torch.zeros((), device=device, dtype=torch.float32)
        inbatch_top1 = 1.0
        loss_total = loss_intra

    metrics = {
        "loss_intra": float(loss_intra.detach().item()),
        "loss_inbatch": float(loss_inbatch.detach().item()),
        "logit_scale": float(scale.detach().item()),
        "intra_top1": intra_top1,
        "inbatch_top1": inbatch_top1,
        "num_pages": float(num_pages),
    }
    return loss_total, metrics
