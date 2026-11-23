"""Deterministic datasets and synthetic Exact/Semantic NIAH & Living-Book State-Override generators."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any
import torch
from torch.utils.data import Dataset


@dataclass
class LSLCEpisode:
    """Single training or evaluation episode for LSLC."""

    context_pages: torch.Tensor  # [P, Tp]
    query_ids: torch.Tensor  # [T_q]
    answer_ids: torch.Tensor  # [T_a]
    target_page_idx: int
    metadata: dict[str, Any]
    edit_pages: list[tuple[int, torch.Tensor]] | None = None  # [(page_idx, edit_ids)]


def _hash_token(text: str, vocab_size: int, reserve_special: int = 32) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    val = int.from_bytes(digest[:4], "little")
    return reserve_special + (val % (vocab_size - reserve_special))


def _encode_text_or_mock(text: str, tokenizer: Any | None, vocab_size: int) -> list[int]:
    if tokenizer is not None:
        return list(tokenizer.encode(text, add_special_tokens=False))
    return [_hash_token(tok, vocab_size) for tok in text.split()]


def _sample_background_page(
    rng: random.Random,
    page_size: int,
    background_token_pool: list[int] | None,
    fallback_vocab: list[int],
) -> list[int]:
    if background_token_pool and len(background_token_pool) >= page_size:
        start = rng.randint(0, len(background_token_pool) - page_size)
        return list(background_token_pool[start : start + page_size])
    return [rng.choice(fallback_vocab) for _ in range(page_size)]


class SyntheticExactSemanticNIAHDataset(Dataset):
    """Generates reproducible Exact-NIAH (UUIDs/IDs/ports) and Semantic-NIAH episodes across 5 depths."""

    DEPTHS = (0.0, 0.25, 0.50, 0.75, 1.0)
    EXACT_KEYS = ("API-KEY", "BUILD-ID", "NODE-PORT", "TX-HASH", "SESSION-UUID")
    SEMANTIC_TOPICS = (
        ("primary database failover runs in active-passive mode", "How does the primary database handle failover?", "active-passive"),
        ("nightly backups are encrypted before cross-region replication", "When are backups encrypted relative to replication?", "before"),
        ("telemetry pipelines batch metrics every fifteen seconds", "What interval does telemetry batching use?", "fifteen"),
        ("canary deployments roll back automatically on latency spikes", "What triggers automatic canary rollback?", "latency"),
    )

    def __init__(
        self,
        num_episodes: int = 64,
        num_pages: int = 4,
        page_size: int = 256,
        tokenizer: Any | None = None,
        vocab_size: int = 512,
        seed: int = 42,
        mode: str = "mixed",  # "exact", "semantic", or "mixed"
        background_token_pool: list[int] | None = None,
    ) -> None:
        self.num_episodes = num_episodes
        self.num_pages = num_pages
        self.page_size = page_size
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer) if tokenizer is not None else vocab_size
        self.seed = seed
        self.mode = mode
        self.background_token_pool = background_token_pool
        self.episodes = [self._build_episode(i) for i in range(num_episodes)]

    def _build_episode(self, idx: int) -> LSLCEpisode:
        rng = random.Random(self.seed + idx * 1009)
        depth = self.DEPTHS[idx % len(self.DEPTHS)]
        target_page = min(self.num_pages - 1, int(round(depth * (self.num_pages - 1))))

        if self.mode == "mixed":
            needle_type = "exact" if idx % 2 == 0 else "semantic"
        else:
            needle_type = self.mode

        if needle_type == "exact":
            key_name = self.EXACT_KEYS[idx % len(self.EXACT_KEYS)]
            code_val = f"{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
            needle_text = f"\n[RECORD] {key_name} = {code_val} .\n"
            query_text = f"\nQuestion: Output only the exact code for {key_name}.\nAnswer:"
            answer_text = f" {code_val}"
            answer_str = code_val
        else:
            fact, q, short_ans = self.SEMANTIC_TOPICS[idx % len(self.SEMANTIC_TOPICS)]
            needle_text = f"\n[NOTE] {fact} .\n"
            query_text = f"\nQuestion: {q}\nAnswer:"
            answer_text = f" {fact}"
            answer_str = short_ans

        needle_tokens = _encode_text_or_mock(needle_text, self.tokenizer, self.vocab_size)
        filler_vocab = _encode_text_or_mock(
            "system log entry status nominal routine checkpoint archive module", self.tokenizer, self.vocab_size
        )

        pages: list[list[int]] = []
        needle_span = (0, 0)
        for p in range(self.num_pages):
            page_toks = _sample_background_page(rng, self.page_size, self.background_token_pool, filler_vocab)
            if p == target_page:
                insert_at = min(max(0, self.page_size - len(needle_tokens) - 4), self.page_size // 3)
                end_at = min(self.page_size, insert_at + len(needle_tokens))
                page_toks[insert_at:end_at] = needle_tokens[: end_at - insert_at]
                needle_span = (insert_at, end_at)
            pages.append(page_toks)

        query_ids = _encode_text_or_mock(query_text, self.tokenizer, self.vocab_size)
        answer_ids = _encode_text_or_mock(answer_text, self.tokenizer, self.vocab_size)
        return LSLCEpisode(
            context_pages=torch.tensor(pages, dtype=torch.long),
            query_ids=torch.tensor(query_ids, dtype=torch.long),
            answer_ids=torch.tensor(answer_ids, dtype=torch.long),
            target_page_idx=target_page,
            metadata={
                "needle_type": needle_type,
                "depth": depth,
                "idx": idx,
                "answer_str": answer_str,
                "needle_span": needle_span,
                "needle_token_ids": needle_tokens,
            },
        )

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> LSLCEpisode:
        return self.episodes[idx]


class LivingBookStateOverrideDataset(Dataset):
    """Generates multi-turn state-override episodes testing DeltaUpdater and static l=0 retention."""

    BUDGET_POOL = ("50M", "55M", "60M", "65M", "70M", "75M", "80M", "85M", "90M", "95M")

    def __init__(
        self,
        num_episodes: int = 48,
        num_pages: int = 8,
        page_size: int = 256,
        num_edits: int = 3,
        tokenizer: Any | None = None,
        vocab_size: int = 512,
        seed: int = 123,
        background_token_pool: list[int] | None = None,
    ) -> None:
        self.num_episodes = num_episodes
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_edits = min(num_edits, num_pages)
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer) if tokenizer is not None else vocab_size
        self.seed = seed
        self.background_token_pool = background_token_pool
        self.episodes = [self._build_episode(i) for i in range(num_episodes)]

    def _build_episode(self, idx: int) -> LSLCEpisode:
        rng = random.Random(self.seed + idx * 2003)
        filler_vocab = _encode_text_or_mock(
            "project policy document schedule allocation team specification", self.tokenizer, self.vocab_size
        )

        # Static page 0 axiom that must NEVER be overwritten by later l=2 edits
        static_rule = "\nSTATIC POLICY : security audit requires dual signoff .\n"
        static_toks = _encode_text_or_mock(static_rule, self.tokenizer, self.vocab_size)

        pages: list[list[int]] = []
        for p in range(self.num_pages):
            toks = _sample_background_page(rng, self.page_size, self.background_token_pool, filler_vocab)
            if p == 0:
                toks[: len(static_toks)] = static_toks[: self.page_size]
            pages.append(toks)

        edit_page = 1 if self.num_pages > 1 else 0
        initial_val = f"{rng.randint(10, 45)}M"
        init_stmt = f"\nSTATE BUDGET INITIAL : {initial_val} .\n"
        init_toks = _encode_text_or_mock(init_stmt, self.tokenizer, self.vocab_size)
        pages[edit_page][: len(init_toks)] = init_toks[: self.page_size]

        sampled_vals = rng.sample(self.BUDGET_POOL, k=min(self.num_edits, len(self.BUDGET_POOL)))
        edits: list[tuple[int, torch.Tensor]] = []
        latest_val = initial_val
        for latest_val in sampled_vals:
            edit_text = f"REVISION OVERRIDE : budget updated to {latest_val} ."
            edit_ids = torch.tensor(_encode_text_or_mock(edit_text, self.tokenizer, self.vocab_size), dtype=torch.long)
            edits.append((edit_page, edit_ids))

        # Alternate between querying overridden state and checking untouched static policy
        check_static = idx % 3 == 0
        if check_static:
            query_text = "\nQuestion: What does the static security audit policy require? Answer:"
            answer_text = " dual signoff"
            answer_str = "dual signoff"
            target_page = 0
        else:
            query_text = "\nQuestion: What is the current updated budget allocation? Answer:"
            answer_text = f" {latest_val}"
            answer_str = latest_val
            target_page = edit_page

        return LSLCEpisode(
            context_pages=torch.tensor(pages, dtype=torch.long),
            query_ids=torch.tensor(_encode_text_or_mock(query_text, self.tokenizer, self.vocab_size), dtype=torch.long),
            answer_ids=torch.tensor(_encode_text_or_mock(answer_text, self.tokenizer, self.vocab_size), dtype=torch.long),
            target_page_idx=target_page,
            metadata={
                "is_overridden_query": not check_static,
                "latest_val": latest_val,
                "answer_str": answer_str,
                "num_edits": self.num_edits,
            },
            edit_pages=edits,
        )

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> LSLCEpisode:
        return self.episodes[idx]


class ChunkedTextPageDataset(Dataset):
    """Slices a token stream into P modular 256-token pages plus continuation tokens."""

    def __init__(self, token_ids: list[int], num_pages: int = 4, page_size: int = 256, turn_len: int = 64) -> None:
        self.num_pages = num_pages
        self.page_size = page_size
        self.turn_len = turn_len
        window = num_pages * page_size + turn_len
        if len(token_ids) < window:
            raise ValueError(f"Need at least {window} tokens, got {len(token_ids)}.")
        self.samples: list[LSLCEpisode] = []
        stride = max(page_size, window // 2)
        for start in range(0, len(token_ids) - window + 1, stride):
            ctx = torch.tensor(token_ids[start : start + num_pages * page_size], dtype=torch.long).view(
                num_pages, page_size
            )
            cont = torch.tensor(token_ids[start + num_pages * page_size : start + window], dtype=torch.long)
            split_q = max(1, turn_len // 4)
            self.samples.append(
                LSLCEpisode(
                    context_pages=ctx,
                    query_ids=cont[:split_q],
                    answer_ids=cont[split_q:],
                    target_page_idx=num_pages - 1,
                    metadata={"start": start},
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> LSLCEpisode:
        return self.samples[idx]
