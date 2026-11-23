"""Layered Spatial Latent Caching (LSLC) — Modular PyTorch MVE Package."""

from .config import LSLCConfig, compute_analytical_memory_stats, select_amp_dtype
from .datasets import (
    ChunkedTextPageDataset,
    LSLCEpisode,
    LivingBookStateOverrideDataset,
    SyntheticExactSemanticNIAHDataset,
)
from .delta_updater import DeltaUpdater
from .encoder import LayeredLatentEncoder, SalienceGate, SparseExactBank
from .projector import (
    AssemblyOutput,
    BookPrefixAssembler,
    QueryToTOCRouter,
    RouterOutput,
    SpatialLatentProjector,
)
from .rope_3d import FactorizedRoPE3D, PageRoPE1D
from .trainer import (
    FrozenQwenLSLCWrapper,
    LSLCLossOutput,
    LSLCSystem,
    TinyMockCausalLM,
    estimate_int4_kv_bytes,
    greedy_generate_from_prefix,
    h2o_evict_keep_indices,
    load_frozen_qwen3,
    streamingllm_keep_indices,
)

LayeredSpatialLatentCache = LSLCSystem

__all__ = [
    "AssemblyOutput",
    "BookPrefixAssembler",
    "ChunkedTextPageDataset",
    "DeltaUpdater",
    "FactorizedRoPE3D",
    "FrozenQwenLSLCWrapper",
    "LSLCConfig",
    "LSLCEpisode",
    "LSLCLossOutput",
    "LSLCSystem",
    "LayeredSpatialLatentCache",
    "LayeredLatentEncoder",
    "LivingBookStateOverrideDataset",
    "PageRoPE1D",
    "QueryToTOCRouter",
    "RouterOutput",
    "SalienceGate",
    "SparseExactBank",
    "SpatialLatentProjector",
    "SyntheticExactSemanticNIAHDataset",
    "TinyMockCausalLM",
    "compute_analytical_memory_stats",
    "estimate_int4_kv_bytes",
    "greedy_generate_from_prefix",
    "h2o_evict_keep_indices",
    "load_frozen_qwen3",
    "select_amp_dtype",
    "streamingllm_keep_indices",
]
