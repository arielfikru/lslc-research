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
    loss_cond_b: torch.Tensor | None = None


class LSLCSystem(nn.Module):
    """Collects all 5 trainable LSLC modules + spatial (h, w) exact-index coordinate embedding + AE cue."""

    def __init__(self, cfg: LSLCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = LayeredLatentEncoder(cfg)
        self.salience_gate = SalienceGate(cfg)
        self.projector = SpatialLatentProjector(cfg)
        self.router = QueryToTOCRouter(
            cfg.d_llm,
            d_route=cfg.d_route,
            router_mode=cfg.router_mode,
            use_contextual_query_encoding=cfg.use_contextual_query_encoding,
        )
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
        special_token_ids: tuple[int, ...] | None = None,
        salience_span_len: int | None = None,
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
                special_token_ids=special_token_ids,
                salience_span_len=salience_span_len,
            )
            if use_exact_bank
            else None
        )
        toc_embeds = self.projector.project_toc(z_book)
        return z_book, exact_bank, toc_embeds

    def reencode_page(
        self,
        z_book: torch.Tensor,
        toc_embeds: torch.Tensor,
        exact_bank: SparseExactBank | None,
        page_idx: int,
        new_page_token_ids: torch.Tensor,
        embed_tokens_fn,
        vocab_size: int | None = None,
        in_place: bool = True,
        special_token_ids: tuple[int, ...] | None = None,
        salience_span_len: int | None = None,
    ) -> tuple[torch.Tensor, SparseExactBank | None, torch.Tensor]:
        """Modularly re-encode a single page into an existing LSLC Book without touching other pages."""
        num_pages = z_book.shape[1]
        if page_idx < 0 or page_idx >= num_pages:
            raise IndexError(f"page_idx {page_idx} out of range for book with {num_pages} pages")
        if new_page_token_ids.dim() == 2:
            page_ids_3d = new_page_token_ids.unsqueeze(1)
        elif new_page_token_ids.dim() == 3 and new_page_token_ids.shape[1] == 1:
            page_ids_3d = new_page_token_ids
        else:
            raise ValueError(
                f"new_page_token_ids must have shape [B, Tp] or [B, 1, Tp], got {tuple(new_page_token_ids.shape)}"
            )

        z_new, bank_new, toc_new = self.encode_book(
            page_ids_3d,
            embed_tokens_fn=embed_tokens_fn,
            use_exact_bank=(exact_bank is not None),
            vocab_size=vocab_size,
            special_token_ids=special_token_ids,
            salience_span_len=salience_span_len,
        )

        if in_place:
            z_out = z_book
            toc_out = toc_embeds
            bank_out = exact_bank
        else:
            z_out = z_book.clone()
            toc_out = toc_embeds.clone()
            bank_out = (
                None
                if exact_bank is None
                else SparseExactBank(
                    token_ids=exact_bank.token_ids.clone(),
                    page_ids=exact_bank.page_ids.clone(),
                    h_coords=exact_bank.h_coords.clone(),
                    w_coords=exact_bank.w_coords.clone(),
                    _ste_gates=None if exact_bank._ste_gates is None else exact_bank._ste_gates.clone(),
                    base_token_ids=(
                        None if exact_bank.base_token_ids is None else exact_bank.base_token_ids.clone()
                    ),
                )
            )

        z_out[:, page_idx : page_idx + 1] = z_new.to(z_out.dtype)
        toc_out[:, page_idx : page_idx + 1] = toc_new.to(toc_out.dtype)
        if bank_out is not None and bank_new is not None:
            bank_out.token_ids[:, page_idx : page_idx + 1] = bank_new.token_ids.to(bank_out.token_ids.dtype)
            bank_out.h_coords[:, page_idx : page_idx + 1] = bank_new.h_coords.to(bank_out.h_coords.dtype)
            bank_out.w_coords[:, page_idx : page_idx + 1] = bank_new.w_coords.to(bank_out.w_coords.dtype)
            bank_out.page_ids[:, page_idx : page_idx + 1] = int(page_idx)
            if bank_out._ste_gates is not None and bank_new._ste_gates is not None:
                bank_out._ste_gates[:, page_idx : page_idx + 1] = bank_new._ste_gates.to(bank_out._ste_gates.dtype)
            if bank_out.base_token_ids is not None:
                bank_out.base_token_ids[:, page_idx : page_idx + 1] = bank_new.token_ids.to(
                    bank_out.base_token_ids.dtype
                )
        return z_out, bank_out, toc_out

    def apply_page_delta(
        self,
        z_book: torch.Tensor,
        page_indices: torch.Tensor,
        edit_token_ids: torch.Tensor,
        embed_tokens_fn,
        allow_entity_update: bool | None = None,
        exact_bank: SparseExactBank | None = None,
        vocab_size: int | None = None,
        edit_embeds_override: torch.Tensor | None = None,
        in_place: bool = False,
        bank_write_threshold: float = 0.45,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, SparseExactBank]:
        raw_edit_embeds = embed_tokens_fn(edit_token_ids)
        delta_edit_embeds = raw_edit_embeds if edit_embeds_override is None else edit_embeds_override
        z_book_new, gates = self.delta_updater.update_book_page(
            z_book, page_indices, delta_edit_embeds, allow_entity_update=allow_entity_update, in_place=in_place
        )
        if exact_bank is not None:
            new_bank = exact_bank.update_page_from_delta(
                page_indices=page_indices,
                edit_token_ids=edit_token_ids,
                edit_embeds=raw_edit_embeds,
                salience_gate=self.salience_gate,
                delta_gates=gates,
                grid_h=self.cfg.grid_h,
                grid_w=self.cfg.grid_w,
                vocab_size=vocab_size,
                in_place=in_place,
                bank_write_threshold=bank_write_threshold,
                edit_token_weights=getattr(self.delta_updater, "_last_edit_attn", None),
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
        exact_bank_attenuation: str = "prob_sq",
        contextual_query_embeds: torch.Tensor | None = None,
        contextual_page_keys: torch.Tensor | None = None,
        external_dense_logits: torch.Tensor | None = None,
    ) -> tuple[RouterOutput, AssemblyOutput]:
        kd = self.cfg.k_deep if k_deep is None else k_deep
        ks = self.cfg.k_skim if k_skim is None else k_skim
        if router_out_override is not None:
            router_out = router_out_override
        else:
            q_tok_emb = (
                embed_tokens_fn(query_input_ids)
                if (query_input_ids is not None and embed_tokens_fn is not None)
                else None
            )
            bank_emb = None
            delta_bank_emb = None
            if exact_bank is not None and embed_tokens_fn is not None and (salience_route_weight > 0.0 or self.cfg.router_mode == "content_attn"):
                if q_tok_emb is None:
                    q_tok_emb = query_embed.unsqueeze(1)
                bank_emb = embed_tokens_fn(exact_bank.routing_token_ids().clamp(min=0).to(z_book.device))
                if salience_route_weight > 0.0 and exact_bank.base_token_ids is not None:
                    delta_bank_emb = embed_tokens_fn(exact_bank.token_ids.clamp(min=0).to(z_book.device))
            router_out = self.router(
                query_embed,
                toc_embeds,
                k_deep=kd,
                k_skim=ks,
                query_token_embeds=q_tok_emb,
                exact_bank_embeds=bank_emb,
                salience_route_weight=salience_route_weight,
                delta_bank_embeds=delta_bank_emb,
                contextual_query_embeds=contextual_query_embeds,
                contextual_page_keys=contextual_page_keys,
                external_dense_logits=external_dense_logits,
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
            toc_embeds=toc_embeds,
            exact_bank_attenuation=exact_bank_attenuation,
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

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor | None]:
        bsz, seq, dim = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(bsz, seq, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        past_len = 0
        if past_key_value is not None:
            pk, pv = past_key_value
            past_len = int(pk.shape[2])
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)
        present = (k, v) if use_cache else None

        total_len = int(k.shape[2])
        if output_attentions or past_len > 0:
            scale = 1.0 / (self.head_dim ** 0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, Q, K]
            if seq > 1:
                q_pos = torch.arange(past_len, past_len + seq, device=x.device).unsqueeze(1)
                k_pos = torch.arange(total_len, device=x.device).unsqueeze(0)
                causal_mask = k_pos > q_pos
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn_weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
            attn = torch.matmul(attn_weights, v)
        else:
            attn_weights = None
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        x = x + self.out(attn.transpose(1, 2).contiguous().view(bsz, seq, dim))
        x = x + self.ff(self.ln2(x))
        return x, present, (attn_weights if output_attentions else None)


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
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, position_ids
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds.")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds

        want_cache = bool(self.config.use_cache if use_cache is None else use_cache)
        all_hidden: list[torch.Tensor] | None = [x] if output_hidden_states else None
        all_attns: list[torch.Tensor] | None = [] if output_attentions else None
        next_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if want_cache else None

        for layer_idx, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[layer_idx]
            x, present, attn_w = layer(
                x,
                past_key_value=layer_past,
                use_cache=want_cache,
                output_attentions=output_attentions,
            )
            if all_hidden is not None:
                all_hidden.append(x)
            if all_attns is not None and attn_w is not None:
                all_attns.append(attn_w)
            if next_cache is not None and present is not None:
                next_cache.append(present)
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
        return SimpleNamespace(
            loss=loss,
            logits=logits,
            past_key_values=tuple(next_cache) if next_cache is not None else None,
            hidden_states=tuple(all_hidden) if all_hidden is not None else None,
            attentions=tuple(all_attns) if all_attns is not None else None,
        )


@torch.no_grad()
def extract_contextual_edit_embeds(
    llm: nn.Module,
    edit_token_ids: torch.Tensor,
    layer_idx: int | None = None,
) -> torch.Tensor:
    """Extract frozen LLM mid-layer hidden states for a revision turn and RMS-align to embed_tokens scale."""
    embed_fn = llm.get_input_embeddings()
    raw_embeds = embed_fn(edit_token_ids)
    out = llm(input_ids=edit_token_ids, output_hidden_states=True, use_cache=False)
    hs = getattr(out, "hidden_states", None)
    if not hs or len(hs) < 2:
        return raw_embeds.detach()

    idx = (len(hs) - 1) // 2 if layer_idx is None else max(1, min(int(layer_idx), len(hs) - 1))
    h = hs[idx].float()
    target_rms = raw_embeds.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    src_rms = h.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    aligned = (h * (target_rms / src_rms)).to(raw_embeds.dtype)
    return aligned.detach()


@torch.no_grad()
def extract_contextual_page_keys(
    llm: nn.Module,
    page_token_ids: torch.Tensor,
    layer_idx: int | None = None,
    pad_token_id: int | None = None,
    chunk_size: int = 16,
    num_page_spans: int = 1,
) -> torch.Tensor:
    """Extract masked mean-pooled frozen LLM mid-layer vectors [B, P, d_llm] or [B, P, S, d_llm] once at page ingest."""
    embed_fn = llm.get_input_embeddings()
    bsz, num_pages, page_len = page_token_ids.shape
    spans = max(1, int(num_page_spans))
    if num_pages == 0:
        d_llm = int(embed_fn.weight.shape[1])
        empty_shape = (bsz, 0, d_llm) if spans == 1 else (bsz, 0, spans, d_llm)
        return torch.zeros(empty_shape, device=page_token_ids.device, dtype=embed_fn.weight.dtype)

    flat_ids = page_token_ids.reshape(bsz * num_pages, page_len)
    backbone = getattr(llm, "model", None)
    use_backbone = backbone is not None and callable(backbone)
    step = max(1, int(chunk_size))
    pooled_chunks: list[torch.Tensor] = []

    for s in range(0, flat_ids.shape[0], step):
        chunk = flat_ids[s : s + step]
        if pad_token_id is not None:
            mask = chunk != int(pad_token_id)
            empty_rows = ~mask.any(dim=1)
            if empty_rows.any():
                mask = mask.clone()
                mask[empty_rows, 0] = True
        else:
            mask = torch.ones_like(chunk, dtype=torch.bool)

        raw_embeds = embed_fn(chunk)
        mask_f = mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        raw_pool = (raw_embeds.float() * mask_f).sum(dim=1) / denom

        if use_backbone:
            out = backbone(input_ids=chunk, attention_mask=mask.long(), output_hidden_states=True, use_cache=False)
        else:
            out = llm(input_ids=chunk, attention_mask=mask.long(), output_hidden_states=True, use_cache=False)
        hs = getattr(out, "hidden_states", None)
        has_hs = bool(hs) and len(hs) >= 2
        if has_hs:
            idx = (len(hs) - 1) // 2 if layer_idx is None else max(1, min(int(layer_idx), len(hs) - 1))
            h_seq = hs[idx].float()
        else:
            h_seq = raw_embeds.float()

        h_page_pool = (h_seq * mask_f).sum(dim=1) / denom
        page_target_rms = raw_pool.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        page_src_rms = h_page_pool.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        aligned_page = h_page_pool * (page_target_rms / page_src_rms)

        if spans == 1:
            pooled_chunks.append(aligned_page.to(raw_embeds.dtype))
        else:
            span_vecs: list[torch.Tensor] = []
            for sp_i in range(spans):
                st = (sp_i * page_len) // spans
                en = ((sp_i + 1) * page_len) // spans
                if en <= st:
                    span_vecs.append(aligned_page)
                    continue
                m_sp = mask_f[:, st:en, :]
                cnt_sp = m_sp.sum(dim=1)  # [chunk_n, 1]
                valid_sp = cnt_sp > 0.5
                h_sp = (h_seq[:, st:en, :] * m_sp).sum(dim=1) / cnt_sp.clamp(min=1.0)
                raw_sp = (raw_embeds[:, st:en, :].float() * m_sp).sum(dim=1) / cnt_sp.clamp(min=1.0)
                t_rms = torch.where(
                    valid_sp,
                    raw_sp.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt(),
                    page_target_rms,
                )
                s_rms = h_sp.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
                aligned_sp = torch.where(valid_sp, h_sp * (t_rms / s_rms), aligned_page)
                span_vecs.append(aligned_sp)
            pooled_chunks.append(torch.stack(span_vecs, dim=1).to(raw_embeds.dtype))

    out_cat = torch.cat(pooled_chunks, dim=0)
    if spans == 1:
        return out_cat.reshape(bsz, num_pages, -1).detach()
    return out_cat.reshape(bsz, num_pages, spans, -1).detach()


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

    def _run_llm_with_prefix_and_logits(
        self,
        prefix_embeds: torch.Tensor,
        turn_ids: torch.Tensor,
        turn_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return out.loss, out.logits

    def _run_llm_with_prefix(
        self,
        prefix_embeds: torch.Tensor,
        turn_ids: torch.Tensor,
        turn_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss, _ = self._run_llm_with_prefix_and_logits(prefix_embeds, turn_ids, turn_labels=turn_labels)
        return loss

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
        edit_accepted_flags: list[bool] | None = None,
        verified_final_pages: torch.Tensor | None = None,
        lambda_cond_b: float | None = None,
        distill_temp: float = 2.0,
        distill_mode: str = "cond_b",
        router_query_ids: torch.Tensor | None = None,
        salience_span_mask: torch.Tensor | None = None,
    ) -> LSLCLossOutput:
        """Execute one full forward pass computing NTP, Reconstruction, Matryoshka, Sparse Gate, Salience, Router, and optional Distillation losses."""
        if distill_mode not in ("none", "cond_b", "cond_a_verified_text"):
            raise ValueError(f"Unsupported distill_mode: {distill_mode!r}")
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

        if edit_accepted_flags is not None and len(edit_accepted_flags) != len(seq_edits):
            raise ValueError(
                f"edit_accepted_flags length ({len(edit_accepted_flags)}) must match seq_edits ({len(seq_edits)})."
            )

        last_edit_page: torch.Tensor | None = None
        last_edit_tokens: torch.Tensor | None = None
        if seq_edits:
            gate_list: list[torch.Tensor] = []
            step_losses: list[torch.Tensor] = []
            flags = [True] * len(seq_edits) if edit_accepted_flags is None else list(edit_accepted_flags)
            for (e_p, e_tok), is_accepted in zip(seq_edits, flags):
                edit_override = (
                    extract_contextual_edit_embeds(self.llm, e_tok, layer_idx=self.cfg.contextual_edit_layer)
                    if self.cfg.use_contextual_edit_encoding
                    else None
                )
                if exact_bank is not None:
                    z_book, g_p, exact_bank = self.lslc.apply_page_delta(
                        z_book,
                        e_p,
                        e_tok,
                        embed_tokens_fn=embed_fn,
                        exact_bank=exact_bank,
                        vocab_size=vocab_size,
                        edit_embeds_override=edit_override,
                    )
                else:
                    z_book, g_p = self.lslc.apply_page_delta(
                        z_book, e_p, e_tok, embed_tokens_fn=embed_fn, edit_embeds_override=edit_override
                    )
                gate_list.append(g_p)
                g_logits = getattr(self.lslc.delta_updater, "_last_gate_logits", None)
                if hasattr(self.lslc.delta_updater, "_last_gate_logits"):
                    self.lslc.delta_updater._last_gate_logits = None
                if g_logits is not None:
                    strip_logits = g_logits[:, -1, :]  # [B, W] bottom sticky-note strip
                    upper_logits = g_logits[:, :-1, :]  # [B, H-1, W] upper sparse rows
                    target_strip = torch.full_like(strip_logits, 0.90 if is_accepted else 0.05)
                    target_upper = torch.full_like(upper_logits, 0.02)
                    loss_strip = F.binary_cross_entropy_with_logits(strip_logits, target_strip)
                    loss_upper = F.binary_cross_entropy_with_logits(upper_logits, target_upper)
                    step_losses.append(1.0 * loss_strip + 2.0 * loss_upper)
                else:
                    g_mean = g_p.mean()
                    if is_accepted:
                        band_excess = F.relu(g_mean - 0.26) + F.relu(0.12 - g_mean)
                        step_losses.append(2.0 * (band_excess + 5.0 * (band_excess ** 2)))
                    else:
                        strip_mean = g_p[:, -1, :].mean()
                        step_losses.append(2.0 * g_mean + 4.0 * strip_mean)
                if is_accepted:
                    last_edit_page, last_edit_tokens = e_p, e_tok
            gate_stack = torch.stack(gate_list, dim=0)
            loss_sparse = torch.stack(step_losses).mean()
            mean_gate = float(gate_stack.mean().detach().item())
        else:
            loss_sparse = torch.zeros((), device=device)
            mean_gate = 0.0

        # Project TOC after optional delta (TOC depends strictly on structural l=0,1 so prefix stays invariant)
        toc_embeds = self.lslc.projector.project_toc(z_book)
        r_q_ids = query_ids if router_query_ids is None else router_query_ids.to(device)
        query_embed = embed_fn(r_q_ids).mean(dim=1)  # [B, d_llm]
        ctx_q_embeds = None
        if self.cfg.use_contextual_query_encoding:
            ctx_q_layer = (
                self.cfg.contextual_query_layer
                if self.cfg.contextual_query_layer is not None
                else self.cfg.contextual_edit_layer
            )
            ctx_q_embeds = extract_contextual_edit_embeds(self.llm, r_q_ids, layer_idx=ctx_q_layer)

        train_sal_weight = 0.0 if self.cfg.salience_mode == "learned_surprisal" else 8.0
        router_out, assembly = self.lslc.route_and_assemble(
            z_book=z_book,
            toc_embeds=toc_embeds,
            query_embed=query_embed,
            exact_bank=exact_bank,
            embed_tokens_fn=embed_fn,
            mode="full",
            training_ste=True,
            query_input_ids=r_q_ids,
            salience_route_weight=train_sal_weight,
            contextual_query_embeds=ctx_q_embeds,
        )

        # 1. Next-Token Prediction loss on [query | answer] (mask query positions with -100)
        turn_ids = torch.cat((query_ids, answer_ids), dim=1)
        query_ignore = torch.full_like(query_ids, -100)
        turn_labels = torch.cat((query_ignore, answer_ids), dim=1)
        loss_ntp, student_logits = self._run_llm_with_prefix_and_logits(
            assembly.prefix_embeds, turn_ids, turn_labels=turn_labels
        )

        # Optional Distillation from verified state ("cond_b" re-encoded or "cond_a_verified_text" raw page)
        w_cond_b = self.cfg.lambda_cond_b_distill if lambda_cond_b is None else float(lambda_cond_b)
        loss_cond_b = torch.zeros((), device=device)
        if verified_final_pages is not None and w_cond_b > 0.0 and distill_mode != "none":
            ans_len = answer_ids.shape[1]
            s_pref_len = assembly.prefix_embeds.shape[1]
            s_start = s_pref_len + query_ids.shape[1] - 1
            student_ans_logits = student_logits[:, s_start : s_start + ans_len, :]
            temp = max(1e-3, float(distill_temp))
            batch_arange = torch.arange(bsz, device=device)

            if distill_mode == "cond_b":
                with torch.no_grad():
                    z_b, bank_b, toc_b = self.lslc.encode_book(
                        verified_final_pages.to(device),
                        embed_tokens_fn=embed_fn,
                        use_exact_bank=use_exact_bank,
                        vocab_size=vocab_size,
                    )
                    _, asm_b = self.lslc.route_and_assemble(
                        z_book=z_b,
                        toc_embeds=toc_b,
                        query_embed=query_embed,
                        exact_bank=bank_b,
                        embed_tokens_fn=embed_fn,
                        mode="full",
                        training_ste=False,
                        router_out_override=router_out,
                    )
                    _, teacher_logits = self._run_llm_with_prefix_and_logits(
                        asm_b.prefix_embeds, turn_ids, turn_labels=turn_labels
                    )
                    b_pref_len = asm_b.prefix_embeds.shape[1]
                    t_start = b_pref_len + query_ids.shape[1] - 1
                    teacher_ans_logits = teacher_logits[:, t_start : t_start + ans_len, :]

                vocab_dim = student_ans_logits.shape[-1]
                kl_ans = (
                    F.kl_div(
                        F.log_softmax(student_ans_logits.float().reshape(-1, vocab_dim) / temp, dim=-1),
                        F.softmax(teacher_ans_logits.float().reshape(-1, vocab_dim) / temp, dim=-1),
                        reduction="batchmean",
                    )
                    * (temp ** 2)
                )
                latent_mse = torch.zeros((), device=device)
                if last_edit_page is not None:
                    ep_idx = last_edit_page.to(device)
                    latent_mse = F.mse_loss(
                        z_book[batch_arange, ep_idx, 2].float(),
                        z_b[batch_arange, ep_idx, 2].detach().float(),
                    )
                loss_cond_b = kl_ans + 0.5 * latent_mse
            else:
                if target_page_idx is not None:
                    ep_target_idx = target_page_idx.to(device)
                elif last_edit_page is not None:
                    ep_target_idx = last_edit_page.to(device)
                else:
                    ep_target_idx = torch.zeros((bsz,), dtype=torch.long, device=device)
                with torch.no_grad():
                    verified_page_ids = verified_final_pages.to(device)[batch_arange, ep_target_idx]
                    teacher_prefix = embed_fn(verified_page_ids)
                    _, teacher_logits = self._run_llm_with_prefix_and_logits(
                        teacher_prefix, turn_ids, turn_labels=turn_labels
                    )
                    a_pref_len = teacher_prefix.shape[1]
                    t_start = a_pref_len + query_ids.shape[1] - 1
                    teacher_ans_logits = teacher_logits[:, t_start : t_start + ans_len, :]

                vocab_dim = student_ans_logits.shape[-1]
                loss_cond_b = (
                    F.kl_div(
                        F.log_softmax(student_ans_logits.float().reshape(-1, vocab_dim) / temp, dim=-1),
                        F.softmax(teacher_ans_logits.float().reshape(-1, vocab_dim) / temp, dim=-1),
                        reduction="batchmean",
                    )
                    * (temp ** 2)
                )

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
                if self.cfg.salience_mode == "learned_surprisal":
                    # Tokenizer-agnostic pure frozen-LLM surprisal with light span smoothing (zero token-ID rules)
                    surp_smooth = 0.7 * surprisal + 0.3 * F.avg_pool1d(
                        surprisal.unsqueeze(1), kernel_size=3, stride=1, padding=1
                    ).squeeze(1)
                    if salience_span_mask is not None:
                        span_m = salience_span_mask[:, sal_page, :].to(device=device, dtype=surp_smooth.dtype)
                        surp_smooth = surp_smooth + 10.0 * span_m
                else:
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
            + w_cond_b * loss_cond_b
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
            loss_cond_b=loss_cond_b.detach(),
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
