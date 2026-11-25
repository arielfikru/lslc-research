"""Configuration presets, precision selection, and analytical memory accounting for LSLC."""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass
class LSLCConfig:
    """Configuration for Layered Spatial Latent Caching (LSLC)."""

    # Frozen LLM backbone geometry (default: Qwen/Qwen3-1.7B)
    model_name: str = "Qwen/Qwen3-1.7B"
    d_llm: int = 2048
    num_llm_layers: int = 28
    num_kv_heads: int = 8
    llm_head_dim: int = 128

    # LSLC Living-Book geometry (12 KiB/page FP16 = 48 B/raw token)
    page_size: int = 256
    num_layers: int = 3  # l=0 Static, l=1 Entity, l=2 Dynamic/Sticky-Note
    latent_channels: int = 128
    grid_h: int = 4
    grid_w: int = 4

    # Trainable encoder / router dimensions
    d_enc: int = 512
    enc_heads: int = 8
    enc_layers: int = 2
    enc_dropout: float = 0.0
    d_route: int = 256

    # Sparse Exact-Recall Footnote Bank (8 B/token = 128 B/page)
    exact_ratio: float = 0.0625
    k_exact: int = 16
    salience_gamma_init: float = 0.5

    # Hierarchical routing budgets
    k_deep: int = 2
    k_skim: int = 2

    # Matryoshka multi-scale dropout probabilities: (TOC 1x1, Skim 2x2, Deep)
    matryoshka_probs: tuple[float, float, float] = (0.35, 0.45, 0.20)

    # Composite loss weights
    lambda_rec: float = 1.0
    lambda_mat: float = 0.5
    lambda_sp: float = 0.05
    lambda_route: float = 0.25

    # Optimization defaults
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float, float] | tuple[float, float] = (0.9, 0.95)
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0

    # Delta-updater defaults
    allow_entity_update: bool = False
    delta_gate_bias_init: float = -1.5

    # v3.0.0 optional contextual edit encoding, Condition-B distillation, and tokenizer-agnostic / position-agnostic modes
    use_contextual_edit_encoding: bool = False
    contextual_edit_layer: int | None = None
    lambda_cond_b_distill: float = 0.0
    salience_mode: str = "v2_legacy"
    delta_pooling_mode: str = "v2_tail"
    router_mode: str = "v2_legacy"
    use_contextual_query_encoding: bool = False
    contextual_query_layer: int | None = None
    contextual_page_spans: int = 1
    salience_span_len: int = 1

    def __post_init__(self) -> None:
        if self.salience_span_len < 1:
            raise ValueError(f"salience_span_len must be >= 1, got {self.salience_span_len}.")
        if self.contextual_page_spans < 1:
            raise ValueError(f"contextual_page_spans must be >= 1, got {self.contextual_page_spans}.")
        if self.salience_mode not in ("v2_legacy", "learned_surprisal"):
            raise ValueError(f"salience_mode must be 'v2_legacy' or 'learned_surprisal', got {self.salience_mode!r}.")
        if self.delta_pooling_mode not in ("v2_tail", "cross_attn"):
            raise ValueError(f"delta_pooling_mode must be 'v2_tail' or 'cross_attn', got {self.delta_pooling_mode!r}.")
        if self.router_mode not in ("v2_legacy", "content_attn"):
            raise ValueError(f"router_mode must be 'v2_legacy' or 'content_attn', got {self.router_mode!r}.")
        if self.page_size % (self.grid_h * self.grid_w) != 0:
            raise ValueError(
                f"page_size ({self.page_size}) must be divisible by grid_h * grid_w ({self.grid_h * self.grid_w})."
            )
        if self.num_layers != 3:
            raise ValueError(f"LSLC v1 requires exactly 3 semantic layers (Static, Entity, Dynamic); got {self.num_layers}.")
        if self.d_enc % self.enc_heads != 0:
            raise ValueError(f"d_enc ({self.d_enc}) must be divisible by enc_heads ({self.enc_heads}).")
        if self.k_exact != 16 and self.exact_ratio == 0.0625:
            self.exact_ratio = self.k_exact / float(self.page_size)
        elif self.k_exact == 16 and self.exact_ratio != 0.0625:
            self.k_exact = max(1, int(round(self.page_size * self.exact_ratio)))

    @property
    def num_spatial_slots(self) -> int:
        return self.grid_h * self.grid_w

    @property
    def num_book_slots(self) -> int:
        return self.num_layers * self.grid_h * self.grid_w

    @property
    def tokens_per_span(self) -> int:
        return self.page_size // self.num_spatial_slots

    @property
    def deep_dense_tokens_per_page(self) -> int:
        # 2x2 Static (4) + 2x2 Entity (4) + 4x4 Dynamic (16) = 24
        l0_toks = (self.grid_h // 2) * (self.grid_w // 2)
        l1_toks = (self.grid_h // 2) * (self.grid_w // 2)
        l2_toks = self.grid_h * self.grid_w
        return l0_toks + l1_toks + l2_toks

    @property
    def kv_bytes_per_token_fp16(self) -> int:
        return 2 * self.num_llm_layers * self.num_kv_heads * self.llm_head_dim * 2

    @property
    def book_bytes_per_page_fp16(self) -> int:
        return self.num_layers * self.latent_channels * self.grid_h * self.grid_w * 2

    @property
    def exact_bytes_per_page(self) -> int:
        # (token_id: int32 [4B], page_id: int16 [2B], h: uint8 [1B], w: uint8 [1B]) = 8 B/tok
        return self.k_exact * 8

    @classmethod
    def qwen3_1_7b(cls, **overrides) -> "LSLCConfig":
        """Primary MVE backbone preset: Qwen/Qwen3-1.7B (112 KiB/tok FP16 KV)."""
        params = dict(
            model_name="Qwen/Qwen3-1.7B",
            d_llm=2048,
            num_llm_layers=28,
            num_kv_heads=8,
            llm_head_dim=128,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def qwen3_4b(cls, **overrides) -> "LSLCConfig":
        """Scaling validation backbone preset: Qwen/Qwen3-4B (144 KiB/tok FP16 KV)."""
        params = dict(
            model_name="Qwen/Qwen3-4B",
            d_llm=2560,
            num_llm_layers=36,
            num_kv_heads=8,
            llm_head_dim=128,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def llama31_8b(cls, **overrides) -> "LSLCConfig":
        """Cross-family 8B replication backbone preset: Llama-3.1-8B-Instruct (128 KiB/tok FP16 KV)."""
        params = dict(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            d_llm=4096,
            num_llm_layers=32,
            num_kv_heads=8,
            llm_head_dim=128,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def tiny_mock(cls, **overrides) -> "LSLCConfig":
        """Fast offline CPU/MPS mock preset preserving exact LSLC page/layer geometry."""
        params = dict(
            model_name="tiny-mock-qwen3",
            d_llm=128,
            num_llm_layers=2,
            num_kv_heads=2,
            llm_head_dim=32,
            d_enc=64,
            enc_heads=4,
            enc_layers=2,
            d_route=64,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def from_model_name(cls, model_name: str, **overrides) -> "LSLCConfig":
        lowered = model_name.lower()
        if "llama" in lowered or "8b" in lowered:
            return cls.llama31_8b(model_name=model_name, **overrides)
        if "4b" in lowered:
            return cls.qwen3_4b(model_name=model_name, **overrides)
        if "mock" in lowered or "tiny" in lowered:
            return cls.tiny_mock(model_name=model_name, **overrides)
        return cls.qwen3_1_7b(model_name=model_name, **overrides)

    def sync_from_hf_config(self, hf_config) -> None:
        """Align LLM geometry dynamically from a loaded HuggingFace model config."""
        if hasattr(hf_config, "hidden_size"):
            self.d_llm = int(hf_config.hidden_size)
        if hasattr(hf_config, "num_hidden_layers"):
            self.num_llm_layers = int(hf_config.num_hidden_layers)
        if hasattr(hf_config, "num_key_value_heads"):
            self.num_kv_heads = int(hf_config.num_key_value_heads)
        if hasattr(hf_config, "head_dim") and hf_config.head_dim is not None:
            self.llm_head_dim = int(hf_config.head_dim)
        elif hasattr(hf_config, "num_attention_heads"):
            self.llm_head_dim = int(self.d_llm // int(hf_config.num_attention_heads))
        if self.use_contextual_edit_encoding and self.contextual_edit_layer is None:
            self.contextual_edit_layer = max(1, self.num_llm_layers // 2)


def select_amp_dtype(device: torch.device | str) -> tuple[torch.dtype, bool]:
    """Select (autocast_dtype, use_grad_scaler) automatically for T4 vs L4/A100 vs CPU/MPS."""
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False
        return torch.float16, True
    return torch.float32, False


def compute_analytical_memory_stats(
    cfg: LSLCConfig,
    num_tokens: int,
    k_deep: int | None = None,
    k_skim: int | None = None,
    use_exact_bank: bool = True,
) -> dict[str, float | int]:
    """Compute exact end-to-end VRAM footprint (persistent book + sparse bank + active LLM KV)."""
    pages = math.ceil(num_tokens / cfg.page_size)
    kd = min(pages, cfg.k_deep if k_deep is None else k_deep)
    ks = min(max(0, pages - kd), cfg.k_skim if k_skim is None else k_skim)
    kt = max(0, pages - kd - ks)

    active_dense_tokens = kt * 1 + ks * 4 + kd * cfg.deep_dense_tokens_per_page
    active_exact_tokens = (kd * cfg.k_exact) if use_exact_bank else 0
    active_total_tokens = active_dense_tokens + active_exact_tokens

    kv_b = cfg.kv_bytes_per_token_fp16
    book_bytes = pages * cfg.book_bytes_per_page_fp16
    exact_bank_bytes = (pages * cfg.exact_bytes_per_page) if use_exact_bank else 0
    active_kv_bytes = active_total_tokens * kv_b
    lslc_net_bytes = book_bytes + exact_bank_bytes + active_kv_bytes

    full_fp16_kv_bytes = num_tokens * kv_b
    int4_kv_bytes = full_fp16_kv_bytes // 4
    h2o_20pct_bytes = int(round(num_tokens * 0.20 * kv_b))

    mib = 1024.0 * 1024.0
    return {
        "num_tokens": num_tokens,
        "pages": pages,
        "k_toc": kt,
        "k_skim": ks,
        "k_deep": kd,
        "active_dense_tokens": active_dense_tokens,
        "active_exact_tokens": active_exact_tokens,
        "active_total_tokens": active_total_tokens,
        "kv_kib_per_token": kv_b / 1024.0,
        "book_mib": book_bytes / mib,
        "exact_bank_mib": exact_bank_bytes / mib,
        "active_kv_mib": active_kv_bytes / mib,
        "lslc_net_mib": lslc_net_bytes / mib,
        "full_fp16_kv_mib": full_fp16_kv_bytes / mib,
        "int4_kv_mib": int4_kv_bytes / mib,
        "h2o_20pct_mib": h2o_20pct_bytes / mib,
        "ratio_vs_fp16": full_fp16_kv_bytes / max(1, lslc_net_bytes),
        "ratio_vs_int4": int4_kv_bytes / max(1, lslc_net_bytes),
    }
