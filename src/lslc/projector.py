"""Multi-Resolution SpatialLatentProjector, QueryToTOCRouter, and Layer-Ordered BookPrefixAssembler."""

from __future__ import annotations

from dataclasses import dataclass
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


class QueryToTOCRouter(nn.Module):
    """Routes user query against all page TOC embeddings using 1D PageRoPE."""

    def __init__(self, d_llm: int, d_route: int = 256) -> None:
        super().__init__()
        self.d_route = d_route
        self.q_proj = nn.Linear(d_llm, d_route)
        self.k_proj = nn.Linear(d_llm, d_route)
        self.page_rope = PageRoPE1D(d_route)
        self.scale = d_route ** -0.5

    def forward(
        self,
        query_embed: torch.Tensor,
        toc_embeds: torch.Tensor,
        k_deep: int,
        k_skim: int,
        query_token_embeds: torch.Tensor | None = None,
        exact_bank_embeds: torch.Tensor | None = None,
        salience_route_weight: float = 8.0,
    ) -> RouterOutput:
        """Route [B, d_llm] query over [B, P, 1, d_llm] page TOC tokens (+ optional in-VRAM SparseExactBank MaxSim)."""
        bsz, num_pages, _, _ = toc_embeds.shape
        device = toc_embeds.device
        if not torch.is_autocast_enabled():
            toc_embeds = toc_embeds.to(self.k_proj.weight.dtype)
            query_embed = query_embed.to(self.q_proj.weight.dtype)

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
            top_m = min(4, max_per_q.shape[-1])
            sal_score = torch.topk(max_per_q, k=top_m, dim=-1).values.mean(dim=-1)  # [B, P]
            sal_z = (sal_score - sal_score.mean(dim=-1, keepdim=True)) / (sal_score.std(dim=-1, keepdim=True) + 1e-6)
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
    ) -> AssemblyOutput:
        """Assemble Mode A causal prefix: [TOC | Skim | Deep l0 | Deep l1 | Deep l2 | Exact]."""
        bsz, num_pages, _, _, _, _ = z_book.shape
        device = z_book.device
        toc_all = projector.project_toc(z_book)

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

        skim_all = projector.project_skim(z_book)
        deep_all = projector.project_deep(z_book)

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

        skim_toks = _gather_page_tokens(skim_all, router_out.skim_idx)
        l0_toks = _gather_page_tokens(deep_all["l0"], router_out.deep_idx)
        l1_toks = _gather_page_tokens(deep_all["l1"], router_out.deep_idx)
        l2_toks = _gather_page_tokens(deep_all["l2"], router_out.deep_idx)

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
