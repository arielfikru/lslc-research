"""Lightweight zero-dependency LoRA wrappers for controlled LSLC co-adaptation studies."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear layer with trainable fp32 low-rank matrices A and B."""

    def __init__(self, base: nn.Linear, *, r: int = 16, alpha: float = 32.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank r must be > 0, got {r}")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.r)
        self.enabled = True
        self._merged = False

        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty((self.r, base.in_features), dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, self.r), dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _delta_weight(self) -> torch.Tensor:
        return (self.lora_B.detach().float() @ self.lora_A.detach().float()) * self.scaling

    @torch.no_grad()
    def merge(self) -> bool:
        """Fold LoRA delta into frozen base weight in-place for zero-overhead inference timing."""
        if self._merged or self.r <= 0:
            return False
        delta = self._delta_weight().to(self.base.weight.dtype)
        self.base.weight.data.add_(delta)
        self._merged = True
        return True

    @torch.no_grad()
    def unmerge(self) -> bool:
        """Restore original frozen base weight by subtracting previously merged LoRA delta."""
        if not self._merged or self.r <= 0:
            return False
        delta = self._delta_weight().to(self.base.weight.dtype)
        self.base.weight.data.sub_(delta)
        self._merged = False
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self._merged or not self.enabled or self.r <= 0:
            return out
        delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scaling
        return out + delta.to(out.dtype)


def inject_lora(
    model: nn.Module,
    *,
    r: int = 16,
    alpha: float = 32.0,
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
) -> list[str]:
    """Replace matching leaf nn.Linear modules with LoRALinear wrappers in-place."""
    if r <= 0:
        raise ValueError(f"LoRA rank r must be > 0, got {r}")

    targets = set(target_modules)
    matches: list[tuple[nn.Module, str, str, nn.Linear]] = []
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent._modules.items()):
            if child_name in targets and isinstance(child, nn.Linear):
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                matches.append((parent, child_name, full_name, child))

    if not matches:
        raise ValueError(f"No nn.Linear modules matched target_modules={target_modules!r}")

    replaced: list[str] = []
    for parent, attr_name, full_name, linear_mod in matches:
        setattr(parent, attr_name, LoRALinear(linear_mod, r=r, alpha=alpha))
        replaced.append(full_name)
    return sorted(replaced)


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return all trainable lora_A and lora_B parameters attached to model."""
    return [
        p
        for mod in model.modules()
        if isinstance(mod, LoRALinear)
        for p in (mod.lora_A, mod.lora_B)
        if p.requires_grad
    ]


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    """Toggle LoRA residual contributions across all LoRALinear layers in model."""
    flag = bool(enabled)
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            mod.enabled = flag


def merge_lora(model: nn.Module) -> int:
    """Merge all unmerged LoRALinear weights into base linear layers and return merged count."""
    count = 0
    for mod in model.modules():
        if isinstance(mod, LoRALinear) and mod.merge():
            count += 1
    return count


def unmerge_lora(model: nn.Module) -> int:
    """Unmerge all merged LoRALinear layers back into frozen base + active LoRA adapters."""
    count = 0
    for mod in model.modules():
        if isinstance(mod, LoRALinear) and mod.unmerge():
            count += 1
    return count


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect CPU contiguous LoRA adapter tensors keyed by module path."""
    state: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            prefix = f"{name}." if name else ""
            state[f"{prefix}lora_A"] = mod.lora_A.detach().cpu().contiguous()
            state[f"{prefix}lora_B"] = mod.lora_B.detach().cpu().contiguous()
    return state
