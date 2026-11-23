"""DeltaUpdater: Sticky-Note in-place spatial delta-patch on dynamic layer l=2 (with optional l=1 gate)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LSLCConfig
from .encoder import PerceiverCrossAttentionBlock, SpatialLayerSelfAttentionBlock


class DeltaUpdater(nn.Module):
    """Applies sparse gated residual patches Z_{p,2} <- (1 - G_p)*Z_{p,2} + G_p*dZ_p while protecting l=0."""

    def __init__(self, cfg: LSLCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.page_in_proj = nn.Linear(cfg.latent_channels, cfg.d_enc)
        self.edit_in_proj = nn.Linear(cfg.d_llm, cfg.d_enc)
        self.edit_spatial_proj = nn.Linear(cfg.d_enc, cfg.d_enc)
        self.edit_tail_proj = nn.Linear(cfg.d_enc, cfg.d_enc)
        self.layer_emb = nn.Embedding(cfg.num_layers, cfg.d_enc)

        self.cross_block = PerceiverCrossAttentionBlock(cfg.d_enc, cfg.enc_heads, cfg.enc_dropout)
        self.self_block = SpatialLayerSelfAttentionBlock(cfg)

        # Sticky-note gate & small-norm initialized delta head targeting dynamic layer l=2
        # Spatial sticky-note prior: bottom strip (h = H-1, 25% of grid) defaults open (+0.5),
        # while upper rows (h = 0..H-2, 75% of grid) default closed (-2.5), yielding mean(G_p) ~ 0.21.
        self.gate_head = nn.Conv2d(cfg.d_enc, 1, kernel_size=1)
        nn.init.normal_(self.gate_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_head.bias)
        spatial_prior = torch.full((1, 1, cfg.grid_h, cfg.grid_w), -2.5)
        spatial_prior[:, :, -1, :] = 0.5
        self.register_buffer("sticky_spatial_prior", spatial_prior)

        self.delta_head = nn.Conv2d(cfg.d_enc, cfg.latent_channels, kernel_size=1)
        nn.init.normal_(self.delta_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.delta_head.bias)

        if cfg.allow_entity_update:
            self.entity_gate_head = nn.Conv2d(cfg.d_enc, 1, kernel_size=1)
            nn.init.constant_(self.entity_gate_head.bias, cfg.delta_gate_bias_init - 0.5)
            self.entity_delta_head = nn.Conv2d(cfg.d_enc, cfg.latent_channels, kernel_size=1)
            nn.init.normal_(self.entity_delta_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.entity_delta_head.bias)

    def forward(
        self,
        z_page: torch.Tensor,
        edit_embeds: torch.Tensor,
        allow_entity_update: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update single page Z_page [B, 3, C, H, W] using revision turn [B, T_e, d_llm]."""
        bsz, layers, c, h, w = z_page.shape
        if layers != 3:
            raise ValueError(f"Expected 3 semantic layers, got {layers}.")

        if not torch.is_autocast_enabled():
            z_page = z_page.to(self.page_in_proj.weight.dtype)
            edit_embeds = edit_embeds.to(self.edit_in_proj.weight.dtype)

        # Flatten [B, L, C, H, W] -> [B, L*H*W, d_enc]
        z_seq = z_page.permute(0, 1, 3, 4, 2).contiguous().view(bsz, layers * h * w, c)
        layer_ids = torch.arange(layers, device=z_page.device).repeat_interleave(h * w)
        z_enc = self.page_in_proj(z_seq) + self.layer_emb(layer_ids).unsqueeze(0)
        edit_enc = self.edit_in_proj(edit_embeds)

        updated_enc = self.self_block(self.cross_block(z_enc, edit_enc))
        updated_grid = updated_enc.view(bsz, layers, h, w, self.cfg.d_enc).permute(0, 1, 4, 2, 3)

        # Direct order-preserving spatial projection of revision tokens onto the bottom sticky-note strip (h = H-1)
        # Combines full-turn 4-bin summary + 1-to-1 un-averaged tail-W token alignment on (h = H-1, w = 0..W-1)
        edit_binned = F.adaptive_avg_pool1d(edit_enc.transpose(1, 2), output_size=w).transpose(1, 2)
        if edit_enc.shape[1] >= w:
            tail_enc = edit_enc[:, -w:, :]
        else:
            pad_len = w - edit_enc.shape[1]
            tail_enc = F.pad(edit_enc, (0, 0, pad_len, 0))
        edit_strip = (self.edit_spatial_proj(edit_binned) + self.edit_tail_proj(tail_enc)).transpose(1, 2).view(
            bsz, self.cfg.d_enc, 1, w
        )
        edit_spatial = torch.zeros((bsz, self.cfg.d_enc, h, w), device=z_page.device, dtype=updated_grid.dtype)
        edit_spatial[:, :, -1:, :] = edit_strip.to(updated_grid.dtype)

        # Layer l=0 (Static) is ALWAYS strictly write-protected
        z_l0 = z_page[:, 0]

        # Layer l=2 (Dynamic Sticky-Note) receives localized gated state override
        l2_feat = updated_grid[:, 2] + edit_spatial
        prior = self.sticky_spatial_prior.to(dtype=l2_feat.dtype, device=l2_feat.device)
        gate_l2 = torch.sigmoid(self.gate_head(l2_feat) + prior)  # [B, 1, H, W]
        delta_l2 = self.delta_head(l2_feat)  # [B, C, H, W]
        # Pre-gate channel-wise RMS alignment prevents cumulative activation-norm drift across k >= 5 sequential edits
        old_rms = z_page[:, 2].float().pow(2).mean(dim=1, keepdim=True).add(1e-6).sqrt()
        delta_rms = delta_l2.float().pow(2).mean(dim=1, keepdim=True).add(1e-6).sqrt()
        delta_l2_normed = (delta_l2.float() * (old_rms / delta_rms).clamp(0.25, 4.0)).to(delta_l2.dtype)
        z_l2 = ((1.0 - gate_l2) * z_page[:, 2] + gate_l2 * delta_l2_normed).to(z_page.dtype)

        # Layer l=1 (Entity) stays untouched unless explicitly enabled
        do_entity = self.cfg.allow_entity_update if allow_entity_update is None else allow_entity_update
        if do_entity and hasattr(self, "entity_gate_head"):
            l1_feat = updated_grid[:, 1]
            gate_l1 = torch.sigmoid(self.entity_gate_head(l1_feat))
            delta_l1 = self.entity_delta_head(l1_feat)
            z_l1 = ((1.0 - gate_l1) * z_page[:, 1] + gate_l1 * delta_l1).to(z_page.dtype)
        else:
            z_l1 = z_page[:, 1]

        z_new = torch.stack((z_l0, z_l1, z_l2), dim=1).to(z_page.dtype)
        return z_new, gate_l2.squeeze(1)

    def update_book_page(
        self,
        z_book: torch.Tensor,
        page_indices: torch.Tensor,
        edit_embeds: torch.Tensor,
        allow_entity_update: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update selected page [B] inside book [B, P, 3, C, H, W] without touching other pages."""
        bsz = z_book.shape[0]
        batch_arange = torch.arange(bsz, device=z_book.device)
        target_pages = z_book[batch_arange, page_indices]
        new_pages, gates = self.forward(target_pages, edit_embeds, allow_entity_update=allow_entity_update)

        z_book_new = z_book.clone()
        z_book_new[batch_arange, page_indices] = new_pages.to(z_book_new.dtype)
        return z_book_new, gates
