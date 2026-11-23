"""Unified LSLCSystem, TinyMockCausalLM, FrozenQwenLSLCWrapper, Composite Losses, and Baseline Helpers."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import random
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LSLCConfig
from .delta_updater import DeltaUpdater
from .encoder import LayeredLatentEncoder, SalienceGate, SparseExactBank
from .projector import AssemblyOutput, BookPrefixAssembler, QueryToTOCRouter, RouterOutput, SpatialLatentProjector


@dataclass
class LSLCLossOutput:
    """Breakdown of composite LSLC training step losses and telemetry."""

    loss_total: torch.Tensor
    loss_ntp: torch.Tensor
    loss_rec: torch.Tensor
    loss_mat: torch.Tensor
    loss_sparse: torch.Tensor
    loss_route: torch.Tensor
    loss_salience: torch.Tensor
    matryoshka_mode: str
    mean_gate: float
    exact_bank_bytes: int
    immutable_prefix_len: int


class LSLCSystem(nn.Module):
    """Collects all 5 trainable LSLC modules + spatial (h, w) exact-index coordinate embedding + AE cue."""

    def __init__(self, cfg: LSLCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = LayeredLatentEncoder(cfg)
        self.salience_gate = SalienceGate(cfg)
        self.projector = SpatialLatentProjector(cfg)
        self.router = QueryToTOCRouter(cfg.d_llm, d_route=cfg.d_route)
        self.delta_updater = DeltaUpdater(cfg)
        self.assembler = BookPrefixAssembler(cfg)
        self.hw_coord_embed = nn.Embedding(cfg.num_spatial_slots, cfg.d_llm)
        # Keep initial spatial coordinate perturbation small (<10% of LLM token embedding norm)
        # so verbatim SparseExactBank tokens remain immediately readable by the frozen LLM.
        nn.init.normal_(self.hw_coord_embed.weight, mean=0.0, std=0.002)
        self.ae_cue_embed = nn.Parameter(torch.randn(1, 1, cfg.d_llm) * 0.02)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_book(
        self,
        page_token_ids: torch.Tensor,
        embed_tokens_fn,
        surprisal_pages: torch.Tensor | None = None,
        use_exact_bank: bool = True,
        vocab_size: int | None = None,
    ) -> tuple[torch.Tensor, SparseExactBank | None, torch.Tensor]:
        """Encode [B, P, Tp] -> (Z_book [B, P, 3, C, 4, 4], SparseExactBank, toc_embeds [B, P, 1, d_llm])."""
        pages_embeds = embed_tokens_fn(page_token_ids)
        z_book = self.encoder.encode_pages(pages_embeds)
        exact_bank = (
            SparseExactBank.from_pages(
                page_token_ids,
                self.salience_gate,
                pages_embeds,
                surprisal_pages=surprisal_pages,
                vocab_size=vocab_size,
            )
            if use_exact_bank
            else None
        )
        toc_embeds = self.projector.project_toc(z_book)
        return z_book, exact_bank, toc_embeds

    def apply_page_delta(
        self,
        z_book: torch.Tensor,
        page_indices: torch.Tensor,
        edit_token_ids: torch.Tensor,
        embed_tokens_fn,
        allow_entity_update: bool | None = None,
        exact_bank: SparseExactBank | None = None,
        vocab_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, SparseExactBank]:
        edit_embeds = embed_tokens_fn(edit_token_ids)
        z_book_new, gates = self.delta_updater.update_book_page(
            z_book, page_indices, edit_embeds, allow_entity_update=allow_entity_update
        )
        if exact_bank is not None:
            new_bank = exact_bank.update_page_from_delta(
                page_indices=page_indices,
                edit_token_ids=edit_token_ids,
                edit_embeds=edit_embeds,
                salience_gate=self.salience_gate,
                delta_gates=gates,
                grid_h=self.cfg.grid_h,
                grid_w=self.cfg.grid_w,
                vocab_size=vocab_size,
            )
            return z_book_new, gates, new_bank
        return z_book_new, gates

    def route_and_assemble(
        self,
        z_book: torch.Tensor,
        toc_embeds: torch.Tensor,
        query_embed: torch.Tensor,
        exact_bank: SparseExactBank | None,
        embed_tokens_fn,
        k_deep: int | None = None,
        k_skim: int | None = None,
        mode: str = "full",
        training_ste: bool = True,
        router_out_override: RouterOutput | None = None,
        hierarchical_chapter_size: int | None = None,
        query_input_ids: torch.Tensor | None = None,
        salience_route_weight: float = 8.0,
        flat_toc_max_tokens: int | None = None,
    ) -> tuple[RouterOutput, AssemblyOutput]:
        kd = self.cfg.k_deep if k_deep is None else k_deep
        ks = self.cfg.k_skim if k_skim is None else k_skim
        if router_out_override is not None:
            router_out = router_out_override
        else:
            q_tok_emb = None
            bank_emb = None
            if exact_bank is not None and embed_tokens_fn is not None and salience_route_weight > 0.0:
                q_tok_emb = (
                    embed_tokens_fn(query_input_ids)
                    if query_input_ids is not None
                    else query_embed.unsqueeze(1)
                )
                bank_emb = embed_tokens_fn(exact_bank.token_ids.clamp(min=0).to(z_book.device))
            router_out = self.router(
                query_embed,
                toc_embeds,
                k_deep=kd,
                k_skim=ks,
                query_token_embeds=q_tok_emb,
                exact_bank_embeds=bank_emb,
                salience_route_weight=salience_route_weight,
            )
        assembly = self.assembler(
            z_book=z_book,
            projector=self.projector,
            router_out=router_out,
            exact_bank=exact_bank,
            embed_tokens_fn=embed_tokens_fn,
            hw_coord_embed=self.hw_coord_embed,
            mode=mode,
            training_ste=training_ste,
            hierarchical_chapter_size=hierarchical_chapter_size,
            flat_toc_max_tokens=flat_toc_max_tokens,
        )
        return router_out, assembly


class _TinyAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, dim = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(bsz, seq, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(attn.transpose(1, 2).contiguous().view(bsz, seq, dim))
        x = x + self.ff(self.ln2(x))
        return x


class TinyMockCausalLM(nn.Module):
    """Lightweight HuggingFace-compatible CausalLM mock for offline CPU/MPS verification."""

    def __init__(
        self,
        vocab_size: int = 512,
        d_llm: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        num_kv_heads: int = 2,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=d_llm,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=d_llm // num_heads,
            use_cache=False,
        )
        self.embed_tokens = nn.Embedding(vocab_size, d_llm)
        self.layers = nn.ModuleList([_TinyAttentionBlock(d_llm, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_llm)
        self.lm_head = nn.Linear(d_llm, vocab_size, bias=False)
        self.gradient_checkpointing_enabled = False
        self.input_require_grads_enabled = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True

    def enable_input_require_grads(self) -> None:
        self.input_require_grads_enabled = True

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
    ) -> SimpleNamespace:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds.")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds

        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


def load_frozen_qwen3(cfg: LSLCConfig, load_in_4bit: bool = False, device: torch.device | str = "cuda"):
    """Load and strictly freeze a HuggingFace Qwen3 backbone (BF16/FP16 or 4-bit NF4 for T4)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        kwargs["device_map"] = "auto"
    else:
        dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else (torch.float16 if torch.device(device).type == "cuda" else torch.float32)
        )
        kwargs["torch_dtype"] = dtype

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    if not load_in_4bit:
        llm.to(device)
    cfg.sync_from_hf_config(llm.config)
    return llm, tokenizer


class FrozenQwenLSLCWrapper(nn.Module):
    """Wraps a strictly frozen LLM + trainable LSLCSystem with exact prefix masking and composite loss."""

    def __init__(self, cfg: LSLCConfig, llm: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.llm = llm
        self.freeze_llm_()
        self.lslc = LSLCSystem(cfg)
        self._rng = random.Random(42)

    def freeze_llm_(self) -> None:
        for param in self.llm.parameters():
            param.requires_grad_(False)
        self.llm.eval()
        if hasattr(self.llm, "config"):
            self.llm.config.use_cache = False
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable()
        if hasattr(self.llm, "enable_input_require_grads"):
            self.llm.enable_input_require_grads()

    def _run_llm_with_prefix(
        self,
        prefix_embeds: torch.Tensor,
        turn_ids: torch.Tensor,
        turn_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Strictly pass inputs_embeds (NEVER input_ids alongside inputs_embeds) and -100 prefix mask."""
        embed_fn = self.llm.get_input_embeddings()
        target_dtype = embed_fn.weight.dtype
        turn_embeds = embed_fn(turn_ids).to(target_dtype)
        prefix_cast = prefix_embeds.to(target_dtype)

        bsz, prefix_len, _ = prefix_cast.shape
        turn_len = turn_ids.shape[1]

        inputs_embeds = torch.cat((prefix_cast, turn_embeds), dim=1)
        attention_mask = torch.ones((bsz, prefix_len + turn_len), device=inputs_embeds.device, dtype=torch.long)

        raw_labels = turn_ids if turn_labels is None else turn_labels
        prefix_ignore = torch.full((bsz, prefix_len), -100, device=inputs_embeds.device, dtype=torch.long)
        labels = torch.cat((prefix_ignore, raw_labels), dim=1)

        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return out.loss

    def _sample_matryoshka_mode(self) -> str:
        modes = ("toc_only", "skim_only", "deep_only")
        return self._rng.choices(modes, weights=self.cfg.matryoshka_probs, k=1)[0]

    def forward_step(
        self,
        page_token_ids: torch.Tensor,
        query_ids: torch.Tensor,
        answer_ids: torch.Tensor,
        target_page_idx: torch.Tensor | None = None,
        edit_page_idx: torch.Tensor | None = None,
        edit_token_ids: torch.Tensor | None = None,
        edit_steps: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_exact_bank: bool = True,
        matryoshka_mode: str | None = None,
        run_reconstruction: bool = True,
        run_matryoshka: bool = True,
        distill_salience: bool = True,
        lambda_sal: float = 1.0,
    ) -> LSLCLossOutput:
        """Execute one full forward pass computing NTP, Reconstruction, Matryoshka, Sparse Gate, Salience, and Router losses."""
        embed_fn = self.llm.get_input_embeddings()
        vocab_size = int(getattr(self.llm.config, "vocab_size", 32000))
        bsz, num_pages, _ = page_token_ids.shape
        device = page_token_ids.device

        z_book, exact_bank, _ = self.lslc.encode_book(
            page_token_ids, embed_tokens_fn=embed_fn, use_exact_bank=use_exact_bank, vocab_size=vocab_size
        )

        # Optional Track 4 Sticky-Note Delta-Update on l=2 (single or multi-step sequential recurrence)
        seq_edits: list[tuple[torch.Tensor, torch.Tensor]] = []
        if edit_steps is not None:
            seq_edits = list(edit_steps)
        elif edit_page_idx is not None and edit_token_ids is not None:
            seq_edits = [(edit_page_idx, edit_token_ids)]

        last_edit_page: torch.Tensor | None = None
        last_edit_tokens: torch.Tensor | None = None
        if seq_edits:
            gate_list: list[torch.Tensor] = []
            for e_p, e_tok in seq_edits:
                if exact_bank is not None:
                    z_book, g_p, exact_bank = self.lslc.apply_page_delta(
                        z_book, e_p, e_tok, embed_tokens_fn=embed_fn, exact_bank=exact_bank, vocab_size=vocab_size
                    )
                else:
                    z_book, g_p = self.lslc.apply_page_delta(z_book, e_p, e_tok, embed_tokens_fn=embed_fn)
                gate_list.append(g_p)
                last_edit_page, last_edit_tokens = e_p, e_tok
            gate_stack = torch.stack(gate_list, dim=0)
            mean_g = gate_stack.mean()
            # Target-Sparsity Band Loss keeping mean(G_p) inside [0.12, 0.26] (never collapsing to 0 or 1)
            band_excess = F.relu(mean_g - 0.26) + F.relu(0.12 - mean_g)
            loss_sparse = 15.0 * (band_excess + 10.0 * (band_excess ** 2)) + self.cfg.lambda_sp * 0.01 * mean_g
            mean_gate = float(mean_g.detach().item())
        else:
            loss_sparse = torch.zeros((), device=device)
            mean_gate = 0.0

        # Project TOC after optional delta (TOC depends strictly on structural l=0,1 so prefix stays invariant)
        toc_embeds = self.lslc.projector.project_toc(z_book)
        query_embed = embed_fn(query_ids).mean(dim=1)  # [B, d_llm]

        router_out, assembly = self.lslc.route_and_assemble(
            z_book=z_book,
            toc_embeds=toc_embeds,
            query_embed=query_embed,
            exact_bank=exact_bank,
            embed_tokens_fn=embed_fn,
            mode="full",
            training_ste=True,
        )

        # 1. Next-Token Prediction loss on [query | answer] (mask query positions with -100)
        turn_ids = torch.cat((query_ids, answer_ids), dim=1)
        query_ignore = torch.full_like(query_ids, -100)
        turn_labels = torch.cat((query_ignore, answer_ids), dim=1)
        loss_ntp = self._run_llm_with_prefix(assembly.prefix_embeds, turn_ids, turn_labels=turn_labels)

        # 2. Page Autoencoding Reconstruction loss from Deep Read representation + learnable AE cue
        if run_reconstruction:
            if last_edit_page is not None and last_edit_tokens is not None:
                rec_page = int(last_edit_page[0].item())
                rec_tokens = last_edit_tokens
            else:
                rec_page = self._rng.randrange(num_pages) if target_page_idx is None else int(target_page_idx[0].item())
                rec_tokens = page_token_ids[:, rec_page, : min(64, self.cfg.page_size)]
            deep_dict = self.lslc.projector.project_deep(z_book[:, rec_page : rec_page + 1])
            cue = self.lslc.ae_cue_embed.expand(bsz, -1, -1)
            rec_prefix = torch.cat((deep_dict["l0"], deep_dict["l1"], deep_dict["l2"]), dim=2).squeeze(1)
            rec_prefix = torch.cat((rec_prefix, cue), dim=1)
            loss_rec = self._run_llm_with_prefix(rec_prefix, rec_tokens)
        else:
            loss_rec = torch.zeros((), device=device)

        # 3. Matryoshka Multi-Scale Zoom loss
        chosen_mode = self._sample_matryoshka_mode() if matryoshka_mode is None else matryoshka_mode
        if run_matryoshka:
            _, mat_assembly = self.lslc.route_and_assemble(
                z_book=z_book,
                toc_embeds=toc_embeds,
                query_embed=query_embed,
                exact_bank=None,
                embed_tokens_fn=embed_fn,
                mode=chosen_mode,
                training_ste=False,
            )
            loss_mat = self._run_llm_with_prefix(mat_assembly.prefix_embeds, turn_ids, turn_labels=turn_labels)
        else:
            loss_mat = torch.zeros((), device=device)

        # 4. Auxiliary Router supervision when target_page_idx is known
        if target_page_idx is not None:
            loss_route = F.cross_entropy(router_out.logits, target_page_idx.to(device))
        else:
            loss_route = torch.zeros((), device=device)

        # 5. Self-Supervised Frozen-LLM Surprisal + Micro-Span Distillation for SalienceGate (zero inference overhead)
        if use_exact_bank and distill_salience:
            sal_page = self._rng.randrange(num_pages) if target_page_idx is None else int(target_page_idx[0].item())
            sal_ids = page_token_ids[:, sal_page, :]
            with torch.no_grad():
                sal_logits_llm = self.llm(input_ids=sal_ids).logits
                shift_l = sal_logits_llm[:, :-1, :].contiguous()
                shift_t = sal_ids[:, 1:].contiguous()
                per_tok_nll = F.cross_entropy(
                    shift_l.view(-1, shift_l.shape[-1]), shift_t.view(-1), reduction="none"
                ).view(bsz, sal_ids.shape[1] - 1)
                surprisal = F.pad(per_tok_nll, (1, 0), value=0.0)
                surp_smooth = 0.5 * surprisal + 0.5 * F.avg_pool1d(
                    surprisal.unsqueeze(1), kernel_size=7, stride=1, padding=3
                ).squeeze(1)
                if vocab_size > 2048:
                    is_struct = (sal_ids < 128).float().unsqueeze(1)
                    struct_burst = F.avg_pool1d(is_struct, kernel_size=11, stride=1, padding=5).squeeze(1)
                    surp_smooth = surp_smooth + 6.0 * struct_burst
                k_target = min(self.cfg.k_exact, sal_ids.shape[1])
                top_surp_idx = torch.topk(surp_smooth, k=k_target, dim=-1).indices
                target_mask = torch.zeros_like(surprisal)
                target_mask.scatter_(1, top_surp_idx, 1.0)
                sal_embeds = embed_fn(sal_ids)

            gate_logits = self.lslc.salience_gate.score_logits(
                sal_embeds, surprisal=None, token_ids=sal_ids, vocab_size=vocab_size
            )
            pos_weight = torch.tensor(8.0, device=device, dtype=gate_logits.dtype)
            loss_salience = F.binary_cross_entropy_with_logits(
                gate_logits, target_mask.to(gate_logits.dtype), pos_weight=pos_weight
            )
        else:
            loss_salience = torch.zeros((), device=device)

        loss_total = (
            loss_ntp
            + self.cfg.lambda_rec * loss_rec
            + self.cfg.lambda_mat * loss_mat
            + loss_sparse
            + self.cfg.lambda_route * loss_route
            + lambda_sal * loss_salience
        )
        return LSLCLossOutput(
            loss_total=loss_total,
            loss_ntp=loss_ntp.detach(),
            loss_rec=loss_rec.detach(),
            loss_mat=loss_mat.detach(),
            loss_sparse=loss_sparse.detach(),
            loss_route=loss_route.detach(),
            loss_salience=loss_salience.detach(),
            matryoshka_mode=chosen_mode,
            mean_gate=mean_gate,
            exact_bank_bytes=0 if exact_bank is None else exact_bank.storage_bytes(),
            immutable_prefix_len=assembly.immutable_prefix_len,
        )


@torch.no_grad()
def greedy_generate_from_prefix(
    llm: nn.Module,
    prefix_embeds: torch.Tensor,
    query_ids: torch.Tensor,
    max_new_tokens: int = 16,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Autoregressive greedy decoding from continuous [prefix_embeds | query_ids] with KV-cache reuse."""
    embed_fn = llm.get_input_embeddings()
    llm_dtype = embed_fn.weight.dtype
    prefix_cast = prefix_embeds.to(llm_dtype)
    q_embeds = embed_fn(query_ids).to(llm_dtype)
    inp_embeds = torch.cat((prefix_cast, q_embeds), dim=1)
    bsz, init_len, _ = inp_embeds.shape
    device = inp_embeds.device
    attn_mask = torch.ones((bsz, init_len), device=device, dtype=torch.long)

    prev_use_cache = getattr(llm.config, "use_cache", False) if hasattr(llm, "config") else False
    if hasattr(llm, "config"):
        llm.config.use_cache = True

    generated: list[torch.Tensor] = []
    try:
        out = llm(inputs_embeds=inp_embeds, attention_mask=attn_mask, use_cache=True)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok)
        pkv = getattr(out, "past_key_values", None)

        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and bool((next_tok == eos_token_id).all()):
                break
            attn_mask = torch.cat((attn_mask, torch.ones((bsz, 1), device=device, dtype=torch.long)), dim=1)
            if pkv is not None:
                out = llm(input_ids=next_tok, attention_mask=attn_mask, past_key_values=pkv, use_cache=True)
                pkv = getattr(out, "past_key_values", None)
            else:
                inp_embeds = torch.cat((inp_embeds, embed_fn(next_tok).to(llm_dtype)), dim=1)
                out = llm(inputs_embeds=inp_embeds, attention_mask=attn_mask)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_tok)
    finally:
        if hasattr(llm, "config"):
            llm.config.use_cache = prev_use_cache

    return torch.cat(generated, dim=1)


# --- Baseline Accounting & Token-Selection Helpers (Tracks 0B / 0C / 0D) ---

def estimate_int4_kv_bytes(cfg: LSLCConfig, num_tokens: int) -> int:
    return (num_tokens * cfg.kv_bytes_per_token_fp16) // 4


def streamingllm_keep_indices(seq_len: int, sink_tokens: int = 4, window_tokens: int = 512) -> torch.Tensor:
    """Select attention sinks + recent sliding window token indices."""
    if seq_len <= sink_tokens + window_tokens:
        return torch.arange(seq_len, dtype=torch.long)
    sinks = torch.arange(sink_tokens, dtype=torch.long)
    recent = torch.arange(seq_len - window_tokens, seq_len, dtype=torch.long)
    return torch.cat((sinks, recent), dim=0)


def h2o_evict_keep_indices(
    cumulative_attn_scores: torch.Tensor,
    budget_ratio: float = 0.20,
    recent_ratio: float = 0.10,
) -> torch.Tensor:
    """Retain recent tokens + top Heavy-Hitter attention tokens up to total budget_ratio."""
    seq_len = cumulative_attn_scores.shape[-1]
    total_keep = max(1, int(round(seq_len * budget_ratio)))
    recent_keep = min(total_keep, max(1, int(round(seq_len * recent_ratio))))
    hh_keep = max(0, total_keep - recent_keep)

    recent_start = seq_len - recent_keep
    recent_idx = torch.arange(recent_start, seq_len, device=cumulative_attn_scores.device)
    if hh_keep == 0 or recent_start == 0:
        return recent_idx

    past_scores = cumulative_attn_scores[:recent_start]
    _, hh_idx = torch.topk(past_scores, k=min(hh_keep, recent_start), sorted=False)
    return torch.sort(torch.cat((hh_idx, recent_idx), dim=0)).values
