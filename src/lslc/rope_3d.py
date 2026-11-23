"""3D Factorized Spatial-Layer RoPE (l, h, w) and 1D Page-Index RoPE."""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dimension in half and rotate [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding matching last-dimension shapes of x, cos, sin."""
    return (x * cos) + (rotate_half(x) * sin)


def _axis_cos_sin(positions: torch.Tensor, dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    if dim % 2 != 0:
        raise ValueError(f"Rotary axis dim must be even, got {dim}.")
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim))
    freqs = torch.outer(positions.float(), inv_freq)  # [N, half]
    emb = torch.cat((freqs, freqs), dim=-1)  # [N, dim]
    return emb.cos(), emb.sin()


def _default_3d_split(head_dim: int) -> tuple[int, int, int]:
    if head_dim % 2 != 0 or head_dim < 6:
        raise ValueError(f"head_dim must be even and >= 6 for 3D RoPE, got {head_dim}.")
    if head_dim == 64:
        return (16, 24, 24)
    pairs = head_dim // 2
    p_l = max(1, pairs // 4)
    p_h = (pairs - p_l) // 2
    p_w = pairs - p_l - p_h
    return (p_l * 2, p_h * 2, p_w * 2)


class FactorizedRoPE3D(nn.Module):
    """Factorized 3D rotary embedding over (layer l, row h, col w) book coordinates."""

    def __init__(
        self,
        head_dim: int,
        num_layers: int = 3,
        grid_h: int = 4,
        grid_w: int = 4,
        split: tuple[int, int, int] | None = None,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.split = _default_3d_split(head_dim) if split is None else split
        if sum(self.split) != head_dim or any(d % 2 != 0 or d <= 0 for d in self.split):
            raise ValueError(f"Invalid 3D RoPE split {self.split} for head_dim={head_dim}.")

        coords_l, coords_h, coords_w = torch.meshgrid(
            torch.arange(num_layers),
            torch.arange(grid_h),
            torch.arange(grid_w),
            indexing="ij",
        )
        flat_l = coords_l.reshape(-1)
        flat_h = coords_h.reshape(-1)
        flat_w = coords_w.reshape(-1)

        cos_l, sin_l = _axis_cos_sin(flat_l, self.split[0], base=base)
        cos_h, sin_h = _axis_cos_sin(flat_h, self.split[1], base=base)
        cos_w, sin_w = _axis_cos_sin(flat_w, self.split[2], base=base)

        self.register_buffer("cos_l", cos_l, persistent=False)
        self.register_buffer("sin_l", sin_l, persistent=False)
        self.register_buffer("cos_h", cos_h, persistent=False)
        self.register_buffer("sin_h", sin_h, persistent=False)
        self.register_buffer("cos_w", cos_w, persistent=False)
        self.register_buffer("sin_w", sin_w, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate tensor x of shape [B, heads, L*H*W, head_dim]."""
        expected_slots = self.num_layers * self.grid_h * self.grid_w
        if x.shape[-2] != expected_slots:
            raise ValueError(f"Expected {expected_slots} slots for FactorizedRoPE3D, got {x.shape[-2]}.")

        d_l, d_h, d_w = self.split
        x_l, x_h, x_w = torch.split(x, [d_l, d_h, d_w], dim=-1)
        dtype = x.dtype

        y_l = apply_rotary_emb(x_l, self.cos_l.to(dtype).unsqueeze(0).unsqueeze(0), self.sin_l.to(dtype).unsqueeze(0).unsqueeze(0))
        y_h = apply_rotary_emb(x_h, self.cos_h.to(dtype).unsqueeze(0).unsqueeze(0), self.sin_h.to(dtype).unsqueeze(0).unsqueeze(0))
        y_w = apply_rotary_emb(x_w, self.cos_w.to(dtype).unsqueeze(0).unsqueeze(0), self.sin_w.to(dtype).unsqueeze(0).unsqueeze(0))
        return torch.cat((y_l, y_h, y_w), dim=-1)


class PageRoPE1D(nn.Module):
    """1D Rotary Position Embedding applied over page indices inside QueryToTOCRouter."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"PageRoPE1D dim must be even, got {dim}.")
        self.dim = dim
        self.base = base

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate x [..., S, dim] using 1D integer positions [S]."""
        cos, sin = _axis_cos_sin(positions.to(x.device), self.dim, base=self.base)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        return apply_rotary_emb(x, cos, sin)
