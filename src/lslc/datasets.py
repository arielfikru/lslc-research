"""Deterministic datasets and synthetic Exact/Semantic NIAH & Living-Book State-Override generators."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import re
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


class AdversarialLivingBookDataset(Dataset):
    """Shortcut-free state-override dataset with randomized locations, static facts, neighbor projects, and accepted/rejected proposals."""

    TRAIN_STATIC_POLICIES = (
        ("security audit requires dual signoff", "What does the static security audit policy require?", "dual signoff"),
        ("production secrets use aes-256-gcm encryption", "Which cipher does the static secrets policy mandate?", "aes-256-gcm"),
        ("database snapshots run every six hours", "How often does the static snapshot policy run?", "every six hours"),
        ("incident logs are retained for thirty days", "How long are incident logs retained under static policy?", "thirty days"),
        ("on-call escalation pages pagerduty tier-1", "Where does static escalation route alerts?", "pagerduty tier-1"),
        ("internal rpc traffic enforces zero trust", "What network rule does static RPC policy enforce?", "zero trust"),
        ("release freezes begin every friday noon", "When does the static release freeze start?", "friday noon"),
        ("container artifacts are signed via sigstore cosign", "How are container artifacts signed under static policy?", "sigstore cosign"),
    )
    EVAL_STATIC_POLICIES = (
        ("primary compliance region is eu-frankfurt", "What is the static primary compliance region?", "eu-frankfurt"),
        ("signing keys rotate every ninety days", "What is the static key rotation interval?", "ninety days"),
        ("failover clusters stay in active-passive mode", "In what mode does the static failover cluster operate?", "active-passive"),
        ("ci pipelines require unit and fuzz checks", "Which checks does static CI policy mandate?", "unit and fuzz"),
        ("backup archives replicate to cold glacier", "Where do static backup archives replicate?", "cold glacier"),
        ("api rate limits default to five hundred rps", "What is the static default API rate limit?", "five hundred rps"),
        ("canary traffic shifts in ten percent steps", "In what increments does static canary traffic shift?", "ten percent"),
        ("schema migrations lock tables for fifteen seconds", "How long may static schema migrations lock tables?", "fifteen seconds"),
    )
    STATIC_POLICIES = TRAIN_STATIC_POLICIES + EVAL_STATIC_POLICIES

    TRAIN_PROJECTS = ("Atlas", "Borealis", "Cygnus", "Draco", "Helios", "Kepler")
    EVAL_PROJECTS = ("Lyra", "Orion", "Polaris", "Vega", "Rigel", "Altair")
    PROJECTS = TRAIN_PROJECTS + EVAL_PROJECTS

    TRAIN_BUDGET_POOL = ("50M", "55M", "60M", "65M", "70M", "75M")
    EVAL_BUDGET_POOL = ("80M", "85M", "90M", "95M", "105M", "115M")
    BUDGET_POOL = TRAIN_BUDGET_POOL + EVAL_BUDGET_POOL

    TRAIN_SYMBOLIC_STATES = (
        "canary-staging",
        "shadow-mirror",
        "blue-green-prod",
        "quorum-degraded",
        "rollback-lockdown",
        "hot-standby",
    )
    EVAL_SYMBOLIC_STATES = (
        "us-west-oregon",
        "eu-central-frankfurt",
        "ap-southeast-singapore",
        "dual-region-mesh",
        "draining-maintenance",
        "read-only-replica",
    )
    SYMBOLIC_STATES = TRAIN_SYMBOLIC_STATES + EVAL_SYMBOLIC_STATES

    _PROC_PROJ_HEADS = (
        "Kestrel", "Vanguard", "Meridian", "Halcyon", "Solstice", "Cobalt",
        "Vertex", "Nimbus", "Sovereign", "Bastion", "Stratos", "Caldera",
        "Vector", "Zephyr", "Beacon", "Prism", "Harbor", "Tundra",
    )
    _PROC_PROJ_TAILS = (
        "Core", "Mesh", "Vault", "Grid", "Pulse", "Edge", "Bridge", "Forge", "Node", "Sync"
    )
    _PROC_STATE_A = (
        "active", "warm", "canary", "blue", "shadow", "edge", "burst",
        "sync", "staged", "live", "fast", "hybrid", "core", "multi",
    )
    _PROC_STATE_B = (
        "cluster", "tier", "failover", "mirror", "node", "shard",
        "proxy", "bridge", "vault", "pool", "ring", "gateway",
    )
    _PROC_STATE_C = (
        "tokyo", "london", "zurich", "dublin", "seoul", "osaka",
        "alpha", "beta", "gamma", "delta", "nordic", "sydney",
    )
    _PROC_STATIC_TEMPLATES = (
        ("audit retention", "How many days are static {topic} logs kept?", "{val} days", ("seven", "twelve", "eighteen", "twenty-four", "forty-five", "sixty")),
        ("encryption suite", "Which algorithm does the static {topic} policy require?", "{val}", ("aes-192-gcm", "chacha20-poly1305", "camellia-256", "aria-256-gcm", "sm4-gcm")),
        ("snapshot cadence", "What is the static {topic} interval?", "every {val}", ("two hours", "four hours", "eight hours", "twelve hours", "twenty minutes")),
        ("alert routing", "Where does static {topic} dispatch urgent pages?", "{val}", ("opsgenie-tier2", "victorops-primary", "oncall-sre-bot", "dispatch-alpha", "noc-west-queue")),
        ("artifact signing", "How are static {topic} releases signed?", "{val}", ("gpg-hardware-key", "kms-ecdsa-p384", "notation-x509", "cosign-kms-vault")),
        ("network mode", "What protocol mode does static {topic} enforce?", "{val}", ("mtls-strict", "wireguard-mesh", "quic-verified", "ipsec-tunnel")),
    )
    _PROC_STATIC_TOPICS = (
        "billing", "telemetry", "ingress", "checkout", "ledger", "storage",
        "auth", "webhook", "scheduler", "gateway", "cache", "search",
    )

    TRAIN_ACCEPT_BUDGET_TEMPLATES = (
        "Board approved raising {proj} budget from {old} to {new} .",
        "Finance signed off on moving {proj} spending cap from {old} to {new} .",
        "Executive directive: {proj} allocation is now updated to {new} .",
        "Treasury confirmed {proj} quarterly limit stands at {new} .",
        "Leadership ratified {new} as the active budget for {proj} .",
        "REVISION OVERRIDE : {proj} budget updated to {new} .",
    )
    EVAL_ACCEPT_BUDGET_TEMPLATES = (
        "Budget committee enacted {new} for {proj}, replacing {old} effective immediately .",
        "Official ledger update: {proj} authorized funding shifted from {old} to {new} .",
        "Controller memo: set {proj} ceiling to {new} for this cycle .",
        "Finalized audit amendment places {proj} budget at {new} .",
        "Signed addendum elevates {proj} cap from {old} up to {new} .",
        "Operations confirmed {proj} operating allowance is revised to {new} .",
    )

    TRAIN_REJECT_BUDGET_TEMPLATES = (
        "Proposal to raise {proj} budget from {old} to {new} was rejected by finance .",
        "Motion to set {proj} allocation to {new} failed review; prior figure stands .",
        "Treasury vetoed moving {proj} cap from {old} to {new} .",
        "Draft request for {proj} spending ceiling {new} was declined without changes .",
        "Finance committee turned down shifting {proj} limit to {new} .",
    )
    EVAL_REJECT_BUDGET_TEMPLATES = (
        "Amendment seeking to bump {proj} funding from {old} to {new} was voted down .",
        "Audit board denied {proj} request for {new}, leaving existing cap intact .",
        "Suggested increase of {proj} ceiling to {new} did not pass sign-off .",
        "No approval was granted to move {proj} budget from {old} to {new} .",
        "Proposed {proj} limit {new} remains unratified and inactive .",
    )

    TRAIN_ACCEPT_SYMBOLIC_TEMPLATES = (
        "Deployment controller switched {proj} from {old} to {new} .",
        "Release lead promoted {proj} runtime mode to {new} .",
        "Infrastructure update: {proj} state is now {new} .",
        "REVISION OVERRIDE : {proj} deployment state updated to {new} .",
    )
    EVAL_ACCEPT_SYMBOLIC_TEMPLATES = (
        "Orchestrator transitioned {proj} topology from {old} into {new} .",
        "Cluster health update puts {proj} actively in {new} .",
        "Cutover complete: {proj} now serves under {new} .",
        "Signed rollout notice moves {proj} to {new} .",
    )

    TRAIN_REJECT_SYMBOLIC_TEMPLATES = (
        "Proposal to switch {proj} from {old} to {new} was aborted before rollout .",
        "Attempt to move {proj} deployment state to {new} was rejected by SRE .",
        "Canary check blocked transitioning {proj} to {new} .",
        "Rollback guard cancelled moving {proj} into {new} .",
    )
    EVAL_REJECT_SYMBOLIC_TEMPLATES = (
        "Planned cutover of {proj} from {old} toward {new} was called off .",
        "Release gate refused promoting {proj} to {new} ; prior mode holds .",
        "Staging request to place {proj} in {new} failed approval .",
        "Unapproved shift of {proj} runtime to {new} was discarded .",
    )

    QUERY_TYPES = (
        "accepted_override",
        "rejected_proposal",
        "neighbor_preservation",
        "randomized_static_retention",
    )

    def __init__(
        self,
        num_episodes: int = 48,
        num_pages: int = 6,
        page_size: int = 256,
        num_edits: int = 3,
        mode: str = "budget",  # "budget" or "symbolic"
        split: str = "train",  # "train" or "eval"
        tokenizer: Any | None = None,
        vocab_size: int = 512,
        seed: int = 2026,
        background_token_pool: list[int] | None = None,
        style: str = "standard",  # "standard", "mixed_natural", or "headerless_eval"
    ) -> None:
        if num_pages < 4:
            raise ValueError(f"AdversarialLivingBookDataset requires num_pages >= 4, got {num_pages}.")
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}.")
        if mode not in ("budget", "symbolic"):
            raise ValueError(f"mode must be 'budget' or 'symbolic', got {mode!r}.")
        if style not in ("standard", "mixed_natural", "headerless_eval"):
            raise ValueError(f"style must be 'standard', 'mixed_natural', or 'headerless_eval', got {style!r}.")
        self.num_episodes = num_episodes
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_edits = num_edits
        self.mode = mode
        self.split = split
        self.style = style
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer) if tokenizer is not None else vocab_size
        self.seed = seed
        self.background_token_pool = background_token_pool
        self.episodes = [self._build_episode(i) for i in range(num_episodes)]

    def _select_templates(self, accepted: bool) -> tuple[str, ...]:
        if self.mode == "budget":
            if self.split == "train":
                return self.TRAIN_ACCEPT_BUDGET_TEMPLATES if accepted else self.TRAIN_REJECT_BUDGET_TEMPLATES
            return self.EVAL_ACCEPT_BUDGET_TEMPLATES if accepted else self.EVAL_REJECT_BUDGET_TEMPLATES
        if self.split == "train":
            return self.TRAIN_ACCEPT_SYMBOLIC_TEMPLATES if accepted else self.TRAIN_REJECT_SYMBOLIC_TEMPLATES
        return self.EVAL_ACCEPT_SYMBOLIC_TEMPLATES if accepted else self.EVAL_REJECT_SYMBOLIC_TEMPLATES

    def _sample_train_projects(self, rng: random.Random) -> tuple[str, str]:
        forbidden = set(self.EVAL_PROJECTS)
        chosen: list[str] = []
        while len(chosen) < 2:
            if rng.random() < 0.15:
                cand = rng.choice(self.TRAIN_PROJECTS)
            else:
                cand = f"{rng.choice(self._PROC_PROJ_HEADS)}-{rng.choice(self._PROC_PROJ_TAILS)}"
            if cand not in forbidden and cand not in chosen:
                chosen.append(cand)
        return chosen[0], chosen[1]

    def _sample_train_static_policy(self, rng: random.Random) -> tuple[str, str, str]:
        forbidden_ans = {ans for _, _, ans in self.EVAL_STATIC_POLICIES}
        while True:
            if rng.random() < 0.2:
                rule, q, ans = rng.choice(self.TRAIN_STATIC_POLICIES)
            else:
                kind, q_tmpl, ans_tmpl, vals = rng.choice(self._PROC_STATIC_TEMPLATES)
                topic = rng.choice(self._PROC_STATIC_TOPICS)
                val = rng.choice(vals)
                ans = ans_tmpl.format(val=val)
                q = q_tmpl.format(topic=topic)
                rule = f"{topic} {kind} policy uses {ans}"
            if ans not in forbidden_ans:
                return rule, q, ans

    def _sample_train_budget_values(self, rng: random.Random, count: int) -> list[str]:
        forbidden = set(self.EVAL_BUDGET_POOL) | {f"{n}M" for n in range(26, 41)}
        units = ("M", "M", "M", "K", "GB", "ms")
        chosen: list[str] = []
        while len(chosen) < count:
            unit = rng.choice(units)
            num = rng.randint(11, 989)
            cand = f"{num}{unit}"
            if cand not in forbidden and cand not in chosen:
                chosen.append(cand)
        return chosen

    def _sample_train_symbolic_values(self, rng: random.Random, count: int) -> list[str]:
        forbidden = set(self.EVAL_SYMBOLIC_STATES) | {
            "passive-failover",
            "dual-region-mesh",
            "read-only-replica",
        }
        chosen: list[str] = []
        while len(chosen) < count:
            shape = rng.randrange(10)
            a = rng.choice(self._PROC_STATE_A)
            b = rng.choice(self._PROC_STATE_B)
            c = rng.choice(self._PROC_STATE_C)
            if shape == 0:
                cand = rng.choice(self.TRAIN_SYMBOLIC_STATES)
            elif shape <= 2:
                cand = f"{a}-{c}"
            elif shape <= 4:
                cand = f"{a}-{b}-{c}"
            elif shape <= 6:
                cand = f"{a}_{b}_{c}"
            elif shape <= 8:
                cand = f"{a.capitalize()}{b.capitalize()}{c.capitalize()}"
            else:
                cand = rng.choice((f"zone {c}{rng.randint(2, 9)}", f"tier {rng.randint(2, 9)} {a}", f"{b} v{rng.randint(2, 9)}"))
            if cand not in forbidden and cand not in chosen:
                chosen.append(cand)
        return chosen

    def _choose_stmt_format(self, rng: random.Random) -> str:
        if self.style == "standard":
            return "record"
        if self.style == "mixed_natural":
            p = rng.random()
            if p < 0.30:
                return "record"
            return "prose" if p < 0.65 else "code"
        return "eval_prose" if rng.random() < 0.5 else "eval_code"

    @staticmethod
    def _format_fact_stmt(
        kind: str,
        *,
        fmt: str,
        proj: str = "",
        val: str = "",
        rule: str = "",
    ) -> str:
        if kind == "static":
            if fmt == "prose":
                return f"\nPer current engineering standards, {rule} .\n"
            if fmt == "code":
                return f"\nSTATIC_POLICY_RULE = \"{rule}\"\n"
            if fmt == "eval_prose":
                return f"\nEngineering compliance baseline establishes that {rule} .\n"
            if fmt == "eval_code":
                return f"\nSYSTEM_STANDARDS[\"default\"] = \"{rule}\"\n"
            return f"\nPOLICY: {rule} .\n"
        if kind == "budget":
            if fmt == "prose":
                return f"\nFinance updated {proj} quarterly spending allowance to {val} .\n"
            if fmt == "code":
                return f"\nPROJECT_BUDGET[\"{proj}\"] = \"{val}\"\n"
            if fmt == "eval_prose":
                return f"\nEffective this fiscal quarter, {proj} operates with an authorized allocation of {val} .\n"
            if fmt == "eval_code":
                return f"\nRUNTIME_ALLOCATIONS[\"{proj}\"] = {{\"cap\": \"{val}\"}}\n"
            return f"\nRECORD: Project {proj} approved budget = {val} .\n"
        # symbolic
        if fmt == "prose":
            return f"\nFollowing the architecture review, {proj} operates under {val} mode .\n"
        if fmt == "code":
            return f"\nDEPLOY_STATE[\"{proj}\"] = \"{val}\"\n"
        if fmt == "eval_prose":
            return f"\nUnder the current production topology, {proj} routes live workloads via {val} .\n"
        if fmt == "eval_code":
            return f"\nCLUSTER_MODE[\"{proj}\"] = \"{val}\"\n"
        return f"\nRECORD: Project {proj} approved deployment state = {val} .\n"

    def _mark_span_mask(
        self,
        mask_row: torch.Tensor,
        stmt_toks: list[int],
        offset: int,
        priority_subs: tuple[str, ...],
    ) -> None:
        n_tok = min(len(stmt_toks), self.page_size - offset)
        if n_tok <= 0:
            return
        mask_row[offset : offset + n_tok] = 0.5
        window = stmt_toks[:n_tok]
        matched_any = False
        for sub in priority_subs:
            if not sub:
                continue
            for probe in (f" {sub}", f"\"{sub}", sub):
                sub_ids = _encode_text_or_mock(probe, self.tokenizer, self.vocab_size)
                m = len(sub_ids)
                if m == 0 or m > n_tok:
                    continue
                for j in range(0, n_tok - m + 1):
                    if window[j : j + m] == sub_ids:
                        mask_row[offset + j : offset + j + m] = 1.0
                        matched_any = True
        if not matched_any:
            tail = min(8, n_tok)
            mask_row[offset + n_tok - tail : offset + n_tok] = 1.0

    def _build_episode(self, idx: int) -> LSLCEpisode:
        rng = random.Random(self.seed + idx * 3001 + (0 if self.split == "train" else 99991))
        filler_vocab = _encode_text_or_mock(
            "system specification charter milestone ledger compliance service endpoint", self.tokenizer, self.vocab_size
        )

        # Randomize distinct page placements and draw strictly disjoint pools between train and eval splits
        static_page, proj_a_page, proj_b_page = rng.sample(range(self.num_pages), 3)
        if self.split == "train":
            static_rule, static_q, static_ans = self._sample_train_static_policy(rng)
            proj_a, proj_b = self._sample_train_projects(rng)
        else:
            static_rule, static_q, static_ans = rng.choice(self.EVAL_STATIC_POLICIES)
            proj_a, proj_b = rng.sample(self.EVAL_PROJECTS, 2)

        fmt_static = self._choose_stmt_format(rng)
        fmt_a = self._choose_stmt_format(rng)
        fmt_b = self._choose_stmt_format(rng)

        if self.mode == "budget":
            if self.split == "train":
                drawn = self._sample_train_budget_values(rng, self.num_edits + 3)
                init_a, init_b = drawn[0], drawn[1]
                pool = drawn[2:]
            else:
                init_a = f"{26 + rng.randrange(15)}M"
                init_b = f"{26 + rng.randrange(15)}M"
                pool = list(self.EVAL_BUDGET_POOL)
            init_a_stmt = self._format_fact_stmt("budget", fmt=fmt_a, proj=proj_a, val=init_a)
            init_b_stmt = self._format_fact_stmt("budget", fmt=fmt_b, proj=proj_b, val=init_b)
        else:
            if self.split == "train":
                drawn = self._sample_train_symbolic_values(rng, self.num_edits + 3)
                init_a, init_b = drawn[0], drawn[1]
                pool = drawn[2:]
            else:
                init_a = "passive-failover"
                init_b = rng.choice(("dual-region-mesh", "read-only-replica"))
                pool = [s for s in self.EVAL_SYMBOLIC_STATES if s != init_b]
            init_a_stmt = self._format_fact_stmt("symbolic", fmt=fmt_a, proj=proj_a, val=init_a)
            init_b_stmt = self._format_fact_stmt("symbolic", fmt=fmt_b, proj=proj_b, val=init_b)

        static_stmt = self._format_fact_stmt("static", fmt=fmt_static, rule=static_rule)
        static_toks = _encode_text_or_mock(static_stmt, self.tokenizer, self.vocab_size)
        init_a_toks = _encode_text_or_mock(init_a_stmt, self.tokenizer, self.vocab_size)
        init_b_toks = _encode_text_or_mock(init_b_stmt, self.tokenizer, self.vocab_size)

        pages: list[list[int]] = []
        bg_proj_a: list[int] = []
        ctx_sal_mask = torch.zeros((self.num_pages, self.page_size), dtype=torch.float32)
        static_offset = rng.randint(0, max(0, self.page_size - len(static_toks)))
        proj_a_offset = rng.randint(0, max(0, self.page_size - max(len(init_a_toks), 28)))
        proj_b_offset = rng.randint(0, max(0, self.page_size - len(init_b_toks)))
        for p in range(self.num_pages):
            toks = _sample_background_page(rng, self.page_size, self.background_token_pool, filler_vocab)
            if p == static_page:
                n_s = min(len(static_toks), self.page_size - static_offset)
                toks[static_offset : static_offset + n_s] = static_toks[:n_s]
                self._mark_span_mask(ctx_sal_mask[p], static_toks, static_offset, (static_ans,))
            elif p == proj_a_page:
                bg_proj_a = list(toks)
                n_a = min(len(init_a_toks), self.page_size - proj_a_offset)
                toks[proj_a_offset : proj_a_offset + n_a] = init_a_toks[:n_a]
                self._mark_span_mask(ctx_sal_mask[p], init_a_toks, proj_a_offset, (proj_a, init_a))
            elif p == proj_b_page:
                n_b = min(len(init_b_toks), self.page_size - proj_b_offset)
                toks[proj_b_offset : proj_b_offset + n_b] = init_b_toks[:n_b]
                self._mark_span_mask(ctx_sal_mask[p], init_b_toks, proj_b_offset, (proj_b, init_b))
            pages.append(toks)

        query_type = self.QUERY_TYPES[idx % len(self.QUERY_TYPES)]
        candidate_vals = rng.sample(pool, k=min(self.num_edits + 1, len(pool)))

        # Build sequence of accepted & rejected revision turns on proj_a_page
        edits: list[tuple[int, torch.Tensor]] = []
        accepted_flags: list[bool] = []
        cur_val_a = init_a
        decoy_str: str | None = None

        for step_i in range(self.num_edits):
            is_last = step_i == (self.num_edits - 1)
            if query_type == "rejected_proposal" and is_last:
                is_accepted = False
            elif query_type == "accepted_override" and is_last:
                is_accepted = True
            else:
                is_accepted = (step_i % 2 == 0) if self.num_edits >= 2 else True

            cand = candidate_vals[step_i]
            if cand == cur_val_a:
                cand = candidate_vals[-1]
            tmpl = rng.choice(self._select_templates(is_accepted))
            e_text = tmpl.format(proj=proj_a, old=cur_val_a, new=cand)
            e_ids = torch.tensor(_encode_text_or_mock(e_text, self.tokenizer, self.vocab_size), dtype=torch.long)
            edits.append((proj_a_page, e_ids))
            accepted_flags.append(is_accepted)
            if is_accepted:
                cur_val_a = cand
            elif is_last:
                decoy_str = cand

        # Construct ground-truth verified_final_pages reflecting only accepted edits on proj_a_page
        verified_pages = [list(pt) for pt in pages]
        verified_pages[proj_a_page] = list(bg_proj_a)
        final_a_stmt = self._format_fact_stmt(
            "budget" if self.mode == "budget" else "symbolic",
            fmt=fmt_a,
            proj=proj_a,
            val=cur_val_a,
        )
        final_a_toks = _encode_text_or_mock(final_a_stmt, self.tokenizer, self.vocab_size)
        final_offset_a = min(proj_a_offset, max(0, self.page_size - len(final_a_toks)))
        n_fa = min(len(final_a_toks), self.page_size - final_offset_a)
        verified_pages[proj_a_page][final_offset_a : final_offset_a + n_fa] = final_a_toks[:n_fa]
        ver_sal_mask = ctx_sal_mask.clone()
        ver_sal_mask[proj_a_page].zero_()
        self._mark_span_mask(ver_sal_mask[proj_a_page], final_a_toks, final_offset_a, (proj_a, cur_val_a))

        instr = "\nBased strictly on the records above, answer with only the exact value.\nQuestion: "
        if query_type == "randomized_static_retention":
            clean_q_text = static_q
            a_text = f" {static_ans}\n"
            ans_str = static_ans
            target_p = static_page
        elif query_type == "neighbor_preservation":
            clean_q_text = (
                f"What is the approved budget for {proj_b}?"
                if self.mode == "budget"
                else f"What is the approved deployment state for {proj_b}?"
            )
            a_text = f" {init_b}\n"
            ans_str = init_b
            target_p = proj_b_page
        else:
            clean_q_text = (
                f"What is the approved budget for {proj_a}?"
                if self.mode == "budget"
                else f"What is the approved deployment state for {proj_a}?"
            )
            a_text = f" {cur_val_a}\n"
            ans_str = cur_val_a
            target_p = proj_a_page

        q_text = f"{instr}{clean_q_text} Answer:"
        router_query_ids = torch.tensor(_encode_text_or_mock(clean_q_text, self.tokenizer, self.vocab_size), dtype=torch.long)
        ans_ids_list = _encode_text_or_mock(a_text, self.tokenizer, self.vocab_size)
        eos_id = getattr(self.tokenizer, "eos_token_id", None) if self.tokenizer is not None else None
        if eos_id is not None and (not ans_ids_list or int(ans_ids_list[-1]) != int(eos_id)):
            ans_ids_list.append(int(eos_id))

        return LSLCEpisode(
            context_pages=torch.tensor(pages, dtype=torch.long),
            query_ids=torch.tensor(_encode_text_or_mock(q_text, self.tokenizer, self.vocab_size), dtype=torch.long),
            answer_ids=torch.tensor(ans_ids_list, dtype=torch.long),
            target_page_idx=target_p,
            metadata={
                "split": self.split,
                "style": self.style,
                "mode": self.mode,
                "query_type": query_type,
                "is_overridden_query": query_type in ("accepted_override", "rejected_proposal"),
                "answer_str": ans_str,
                "decoy_str": decoy_str,
                "static_page": static_page,
                "proj_a_page": proj_a_page,
                "proj_b_page": proj_b_page,
                "static_offset": static_offset,
                "proj_a_offset": final_offset_a,
                "proj_b_offset": proj_b_offset,
                "proj_a": proj_a,
                "proj_b": proj_b,
                "router_query_ids": router_query_ids,
                "edit_accepted_flags": accepted_flags,
                "verified_final_pages": torch.tensor(verified_pages, dtype=torch.long),
                "context_salience_span_mask": ctx_sal_mask,
                "verified_salience_span_mask": ver_sal_mask,
            },
            edit_pages=edits,
        )

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> LSLCEpisode:
        return self.episodes[idx]


PARAPHRASE_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "across", "after", "against", "along", "among", "an", "and",
        "are", "around", "as", "at", "be", "been", "before", "behind", "being", "below",
        "beneath", "beside", "between", "beyond", "but", "by", "can", "could", "did", "do",
        "does", "doing", "done", "down", "during", "each", "either", "for", "from", "had",
        "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i",
        "if", "in", "inside", "into", "is", "it", "its", "itself", "just", "may", "me",
        "might", "more", "most", "much", "must", "my", "near", "no", "nor", "not", "now",
        "of", "off", "on", "once", "only", "or", "other", "our", "ours", "out", "outside",
        "over", "own", "same", "she", "should", "since", "so", "some", "such", "than",
        "that", "the", "their", "theirs", "them", "then", "there", "these", "they", "this",
        "those", "through", "to", "too", "toward", "towards", "under", "underneath",
        "unlike", "until", "up", "upon", "very", "was", "we", "were", "what", "when",
        "where", "which", "while", "who", "whom", "why", "will", "with", "within",
        "without", "would", "you", "your", "yours",
    }
)


def content_words(text: str) -> set[str]:
    """Extract normalized non-stopword alphanumeric content words (>1 char)."""
    return {
        w
        for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) > 1 and w not in PARAPHRASE_STOPWORDS
    }


def _simple_plural_stem(word: str) -> str:
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


class ZeroOverlapParaphraseRoutingBenchmark(Dataset):
    """Pure-paraphrase semantic routing benchmark enforcing 0 content-word overlap between query and target page."""

    # 64 training topics (passages, queries) with strictly zero shared content words per topic
    TRAIN_TOPICS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
        (
            "cardiac_rhythm",
            (
                "A small battery-powered pacemaker delivers rhythmic electrical voltage to keep sluggish heartbeats steady.",
                "Surgical rhythm generators stimulate myocardial chambers whenever native cardiac contraction falters.",
            ),
            (
                "Which medical device corrects abnormally slow pulse rates using tiny shock impulses?",
                "How do doctors stabilize weak ventricular beats with an implanted stimulator?",
            ),
        ),
        (
            "solar_cells",
            (
                "Silicon wafers inside rooftop panels knock electrons loose whenever bright sunlight strikes them.",
                "Semiconductor plates convert daytime photons directly into usable direct current electricity.",
            ),
            (
                "How do photovoltaic arrays generate electric power from daylight rays?",
                "What happens inside crystalline modules exposed to midday illumination?",
            ),
        ),
        (
            "volcanic_magma",
            (
                "Scorching liquid rock rises beneath fractured crust until trapped gas blasts ash high into the sky.",
                "Pressurized underground magma chambers rupture violently along weak tectonic fissures.",
            ),
            (
                "Why do dormant peaks explode with cinders and superheated lava?",
                "What causes subterranean molten stone to erupt through mountain vents?",
            ),
        ),
        (
            "coral_bleaching",
            (
                "Overheated ocean currents force colorful symbiotic algae out of stony reef skeletons, leaving them pale.",
                "Rising sea temperatures expel tiny photosynthetic partners living inside tropical limestone colonies.",
            ),
            (
                "Why do marine polyps turn stark white during prolonged heatwaves?",
                "What damages underwater calcium habitats when coastal waters grow unusually warm?",
            ),
        ),
        (
            "public_key_crypto",
            (
                "Anyone may lock a message using an open encryption credential, yet only the matching secret unlocks it.",
                "Two-key ciphers pair a freely shared locking key with an exclusively held decryption token.",
            ),
            (
                "How can strangers send confidential notes without previously exchanging a private password?",
                "Explain asymmetric mathematical scrambling where broadcasting one code enables secure inbox delivery.",
            ),
        ),
        (
            "lunar_tides",
            (
                "Gravitational pull from Earth's orbiting satellite lifts coastal sea levels twice every day.",
                "Lunar attraction tugs ocean basins outward, producing alternating high and low shoreline surges.",
            ),
            (
                "What celestial force drives predictable six-hour swells along sandy beaches?",
                "Why does maritime water rise and fall cyclically beneath the night moon?",
            ),
        ),
        (
            "airplane_lift",
            (
                "Curved upper wings accelerate passing airflow, lowering pressure above so heavy aircraft climb.",
                "Tilted airfoils deflect rushing air downward, creating upward aerodynamic force beneath jets.",
            ),
            (
                "How do massive metal planes stay aloft during high-speed flight?",
                "What physics principle lets winged vehicles overcome gravity?",
            ),
        ),
        (
            "yeast_fermentation",
            (
                "Microscopic fungi consume dissolved malt sugars inside sealed vats, releasing alcohol and bubbles.",
                "Anaerobic brewer cultures metabolize sweet wort into ethanol alongside effervescent carbon dioxide.",
            ),
            (
                "How does bread dough swell and craft beer gain fizz without oxygen?",
                "What biological reaction turns grain syrup into spirited beverages?",
            ),
        ),
        (
            "violin_acoustics",
            (
                "Rosin-coated horsehair grips taut gut cords, vibrating the hollow spruce soundbox loudly.",
                "Drawing a sticky bow across metal wires excites wooden resonance chambers inside orchestral instruments.",
            ),
            (
                "Why does sliding a haired stick produce rich musical tones from carved timber?",
                "How do acoustic fiddles amplify tiny string vibrations for concert halls?",
            ),
        ),
        (
            "steel_tempering",
            (
                "Reheating quenched iron blades softens brittle martensite crystals so swords resist shattering.",
                "Controlled furnace warming relieves internal stress inside hardened carbon alloys, boosting toughness.",
            ),
            (
                "Why do blacksmiths bake freshly cooled metal tools a second time?",
                "How does moderate post-quench heat keep razor edges from snapping easily?",
            ),
        ),
        (
            "appellate_review",
            (
                "Higher judges examine trial transcripts for legal mistakes rather than hearing fresh witness testimony.",
                "Review panels inspect lower courtroom records solely to spot procedural errors or misapplied statutes.",
            ),
            (
                "What happens when losing litigants ask senior courts to overturn verdicts?",
                "Why do supreme benches rarely summon new evidence during judicial challenges?",
            ),
        ),
        (
            "monsoon_winds",
            (
                "Warming July continents pull moisture-laden sea breezes inland, unleashing months of torrential rain.",
                "Seasonal land-ocean temperature flips reverse regional gusts across southern Asia, soaking farmlands.",
            ),
            (
                "What atmospheric shift brings heavy downpours to tropical peninsulas each autumn?",
                "Why do prevailing coastal storms abruptly change direction between winter and summer?",
            ),
        ),
        (
            "photosynthesis_chlorophyll",
            (
                "Green leaf pigments capture red and blue light rays to assemble glucose from air and water.",
                "Plant chloroplasts harness sunshine to convert atmospheric carbon dioxide into nourishing starches.",
            ),
            (
                "How do forest trees manufacture sweet food solely from sunlight and rain?",
                "Why do botanical foliage cells appear emerald while building carbohydrates?",
            ),
        ),
        (
            "glacier_moraine",
            (
                "Slow-moving ice rivers grind valley boulders and dump long gravel ridges along their melting snouts.",
                "Advancing alpine snowpacks bulldoze bedrock debris into curved earthen mounds when climates warm.",
            ),
            (
                "How do ancient frozen sheets leave behind rocky hills after retreating?",
                "What geologic process carves U-shaped canyons and deposits unsorted stone heaps?",
            ),
        ),
        (
            "carbon_dating",
            (
                "Measuring leftover radioactive isotopes in buried charcoal reveals when ancient campfires burned.",
                "Physicists count decaying atomic nuclei inside preserved bones to estimate prehistoric ages.",
            ),
            (
                "How do archaeologists determine how old unearthed wooden artifacts really are?",
                "What laboratory clock estimates fossils from vanished civilizations?",
            ),
        ),
        (
            "fiber_optics",
            (
                "Hair-thin glass strands trap rapid laser pulses via total internal reflection across continents.",
                "Silica cables bounce light signals cleanly down curved undersea trenches with minimal attenuation.",
            ),
            (
                "How does internet traffic travel beneath oceans at photonic speed?",
                "Why do bright beams stay confined inside bent transparent conduits?",
            ),
        ),
        (
            "vapor_refrigeration",
            (
                "Evaporating liquid coolant absorbs indoor thermal energy before an electric compressor squeezes gas outside.",
                "Phase-changing fluids soak up kitchen heat inside coils and vent thermal energy outdoors.",
            ),
            (
                "How do household freezers chill food compartments while warming rear grates?",
                "What thermodynamic cycle moves warmth out of insulated cabinets?",
            ),
        ),
        (
            "vitamin_c_scurvy",
            (
                "Fresh citrus fruits supply ascorbic acid needed to weave sturdy collagen fibers in gums and skin.",
                "Sailors eating limes avoided bleeding tissues because tropical produce repairs connective proteins.",
            ),
            (
                "Why did long ocean voyages cause loose teeth before naval crews carried lemons?",
                "Which dietary nutrient prevents brittle blood vessels and slow wound healing?",
            ),
        ),
        (
            "container_shipping",
            (
                "Standardized steel boxes transfer seamlessly between cargo ships, freight trains, and highway trailers.",
                "Intermodal metal crates eliminated slow dockside repacking, slashing global port turnaround times.",
            ),
            (
                "What logistical invention revolutionized international sea trade after 1956?",
                "Why can cranes unload ocean vessels directly onto waiting railcars?",
            ),
        ),
        (
            "mercator_projection",
            (
                "Stretching polar latitudes vertically keeps compass bearings straight on flat nautical wall charts.",
                "Cylindrical map grids preserve local angles for sailors while exaggerating Greenland's true area.",
            ),
            (
                "Why do Arctic islands look gigantic compared to Africa on common classroom atlases?",
                "Which cartographic trick trades accurate landmass size for constant navigation headings?",
            ),
        ),
        (
            "jacquard_loom",
            (
                "Punched cardboard strips automated intricate silk brocade patterns decades before electronic computers.",
                "Perforated wooden cards lifted selected warp threads automatically during nineteenth-century weaving.",
            ),
            (
                "Which textile machine inspired early programmable memory using stiff paper holes?",
                "How did French artisans fabricate ornate floral tapestries without manual errors?",
            ),
        ),
        (
            "seismograph_waves",
            (
                "Fast compression tremors reach monitoring stations minutes before destructive rolling surface shocks.",
                "Subterranean fault ruptures emit primary acoustic jolts ahead of slower sideways ground swaying.",
            ),
            (
                "How do earthquake warning networks alert distant cities seconds before buildings shake?",
                "Why do geological detectors register two distinct arrivals from deep crustal slips?",
            ),
        ),
        (
            "bee_pollination",
            (
                "Fuzzy foraging insects dust sticky yellow pollen between blossoms while sipping sweet nectar.",
                "Honeybees carry powdery male gametes across orchard flowers, enabling crisp apple harvests.",
            ),
            (
                "Why do fruit farmers rent buzzing hives every spring?",
                "How do winged pollinators boost seed yields across wildflower meadows?",
            ),
        ),
        (
            "hydraulic_brakes",
            (
                "Incompressible oil pressure multiplies modest pedal pushes into crushing force against spinning rotors.",
                "Sealed glycol lines transmit foot pressure evenly across four axle calipers to halt heavy trucks.",
            ),
            (
                "How does tapping a small car lever stop two tons of rolling steel?",
                "What fluid mechanics law amplifies driver effort inside automotive wheel cylinders?",
            ),
        ),
        (
            "penicillin_mold",
            (
                "Accidental blue mold colonies dissolved nearby Staphylococcus bacteria on forgotten glass dishes.",
                "Mycelial secretions rupture fragile peptidoglycan walls inside multiplying pathogenic microbes.",
            ),
            (
                "How did an untidy petri plate yield humanity's first life-saving antibiotic?",
                "Why do fungal extracts kill infectious germs without harming patients?",
            ),
        ),
        (
            "cathode_crt",
            (
                "Magnetic yokes steer speeding electron beams across glowing phosphor screens line by line.",
                "Vacuum tubes paint television images when invisible particle streams strike fluorescent dots.",
            ),
            (
                "How did bulky glass monitors draw moving pictures before flat panels existed?",
                "What steered charged particles inside vintage arcade displays?",
            ),
        ),
        (
            "vaccine_memory",
            (
                "Attenuated viral fragments train adaptive immune lymphocytes to neutralize real pathogens rapidly later.",
                "Weakened antigens prime long-lived B-lymphocytes so subsequent infections trigger instant antibodies.",
            ),
            (
                "How does an early medical shot protect people from future contagious outbreaks?",
                "Why do white blood cells react much faster after prior harmless exposure?",
            ),
        ),
        (
            "arch_bridge",
            (
                "Curved stone ribs channel heavy roadway loads sideways into solid canyon cliffs.",
                "Masonry wedges lock tightly under downward weight, transferring stress into rocky abutments.",
            ),
            (
                "Why can ancient Roman aqueducts support huge traffic without steel cables?",
                "How does a semicircular span stay standing when no mortar glues its keystones?",
            ),
        ),
        (
            "osmosis_roots",
            (
                "Salty root hairs draw dilute soil moisture inward across semipermeable plant membranes.",
                "Fine botanical fibers absorb groundwater passively whenever internal solutes exceed damp earth.",
            ),
            (
                "How do thirsty garden shrubs pull rainwater upward from muddy beds without pumps?",
                "Why does pouring excessive fertilizer brine dehydrate delicate stems?",
            ),
        ),
        (
            "gyroscope_inertia",
            (
                "Rapidly spinning brass flywheels resist tilting, keeping spacecraft pointed steadily along fixed stars.",
                "High angular momentum stabilizes rolling ships and inertial navigation pods against sudden jolts.",
            ),
            (
                "How do satellites maintain orientation without relying on magnetic compasses?",
                "Why does a fast-whirling top refuse to topple sideways on a table?",
            ),
        ),
        (
            "pasteurization_milk",
            (
                "Briefly heating raw dairy kills harmful microbes while preserving creamy flavor and proteins.",
                "Flash thermal treatment eliminates spoilage germs inside bottled beverages without boiling.",
            ),
            (
                "How do creameries make fresh milk safe for store shelves without ruining taste?",
                "Which nineteenth-century food preservation method stopped tuberculosis transmission?",
            ),
        ),
        (
            "lightning_static",
            (
                "Colliding hailstones inside tall thunderheads separate electrical charges until giant sparks jump.",
                "Updraft friction strips electrons across storm clouds, discharging sudden bolts toward ground.",
            ),
            (
                "What builds huge voltage differences inside dark summer cumulo-nimbus skies?",
                "Why do violent atmospheric tempests produce blinding flashes followed by thunderclaps?",
            ),
        ),
        (
            "mri_superconduct",
            (
                "Liquid-helium-cooled coils align hydrogen protons inside living tissue without ionizing radiation.",
                "Powerful clinical magnets flip water nuclei briefly to map soft brain anatomy cleanly.",
            ),
            (
                "How do hospital scanners photograph torn knee ligaments safely without X-rays?",
                "Why must diagnostic imaging suites remain free of iron objects?",
            ),
        ),
        (
            "catalytic_converter",
            (
                "Honeycomb platinum meshes oxidize toxic carbon monoxide exhaust into benign gases.",
                "Precious metal coatings inside muffler canisters scrub smog-forming nitrogen oxides from tailpipes.",
            ),
            (
                "How do modern gasoline cars reduce poisonous tailpipe fumes before venting outdoors?",
                "Why do thieves steal metallic cylinder units bolted beneath parked sedans?",
            ),
        ),
        (
            "hibernation_torpor",
            (
                "Ground squirrels plunge body temperature near freezing all winter to conserve stored fat.",
                "Dormant alpine mammals slow heartbeats to three per minute until spring thaw arrives.",
            ),
            (
                "How do tiny woodland rodents survive months of snow without eating food?",
                "Why does metabolic slowdown protect hibernating bears from muscle wasting?",
            ),
        ),
        (
            "inkjet_piezo",
            (
                "Flexing quartz crystals squirt minuscule ink droplets onto passing paper sheets cleanly.",
                "Tiny thermal heaters boil dye bubbles inside printhead nozzles, firing sharp text dots.",
            ),
            (
                "How do desktop color printers place clear photographs without touching pages?",
                "What microscopic actuator ejects pigment drops at rapid frequency?",
            ),
        ),
        (
            "compass_magnetite",
            (
                "Suspended lodestone needles swing toward Earth's geomagnetic poles even beneath overcast skies.",
                "Magnetized iron slivers floating on cork let medieval mariners steer open seas confidently.",
            ),
            (
                "How did ancient navigators hold steady headings during foggy nights away from stars?",
                "Why does a freely pivoting pointer align north-south across ocean voyages?",
            ),
        ),
        (
            "cloud_seeding",
            (
                "Airplanes disperse fine silver iodide crystals inside damp clouds to trigger falling snowflakes.",
                "Artificial ice nuclei encourage supercooled atmospheric droplets to freeze and drop as rain.",
            ),
            (
                "How can weather modification crews coax extra precipitation over parched reservoirs?",
                "Why do highland planes release microscopic dust particles into winter storms?",
            ),
        ),
        (
            "transistor_switch",
            (
                "Applying tiny voltage to a silicon gate opens or blocks heavy current without fragile glass bulbs.",
                "Microscopic doped semiconductor junctions amplify weak signals and execute binary logic.",
            ),
            (
                "Which solid-state component replaced hot vacuum tubes inside early radios and computers?",
                "How do modern processor chips toggle billions of digital states per second?",
            ),
        ),
        (
            "blood_clotting",
            (
                "Sticky platelet fragments weave tough fibrin nets across torn vessels to halt bleeding.",
                "Coagulation cascades turn dissolved plasma proteins into insoluble mesh plugs over scrapes.",
            ),
            (
                "How does skin seal fresh cuts rapidly before dangerous fluid loss occurs?",
                "Why do small needle pricks stop oozing within minutes?",
            ),
        ),
        (
            "canal_locks",
            (
                "Watertight chambers raise or lower river barges between uneven hilltop elevation steps.",
                "Flooding sealed concrete basins lifts heavy freighters smoothly over steep coastal dams.",
            ),
            (
                "How do ocean vessels climb inland waterways connecting two seas at different altitudes?",
                "What engineering gates let ships cross Panama's mountainous divide?",
            ),
        ),
        (
            "mirage_refraction",
            (
                "Sun-baked asphalt warms air layers right above highways, deflecting sky light upward like puddles.",
                "Steep thermal density gradients curve distant rays so desert travelers see illusory pools.",
            ),
            (
                "Why do dry summer roads look wet ahead even though pavement stays bone-dry?",
                "What optical bending creates shimmering phantom lakes across hot sand dunes?",
            ),
        ),
        (
            "sonar_submarine",
            (
                "Underwater vessels ping sharp acoustic clicks and time returning echoes off rocky trenches.",
                "Hydrophone arrays map sunken wrecks by measuring how quickly sound bounces back.",
            ),
            (
                "How do deep-sea crews navigate pitch-black abyssal canyons without headlights?",
                "Why does passive listening reveal distant propeller blades miles away?",
            ),
        ),
        (
            "compost_humus",
            (
                "Aerobic microbes digest kitchen scraps and dry autumn leaves into rich, crumbly garden humus.",
                "Thermophilic bacteria break down yard clippings inside aerated piles, enriching poor topsoil.",
            ),
            (
                "How do organic growers recycle vegetable peels into fertile dark earth?",
                "Why do backyard decomposition bins grow noticeably warm inside their centers?",
            ),
        ),
        (
            "doppler_siren",
            (
                "Approaching ambulances squeeze sound waves tighter, raising pitch until speeding past listeners.",
                "Moving sirens compress ahead acoustic crests and stretch trailing ripples downward in tone.",
            ),
            (
                "Why does a racing train horn drop suddenly from high shrillness to low rumble?",
                "How does motion alter perceived frequency when emergency trucks pass pedestrians?",
            ),
        ),
        (
            "geothermal_vent",
            (
                "Deep boreholes tap scalding underground steam pockets near magmatic bedrock to spin turbines.",
                "Circulating piped water through hot subterranean granite produces steady renewable electricity.",
            ),
            (
                "How does Iceland generate round-the-clock power from earth's internal heat?",
                "Why can volcanic islands heat entire neighborhoods cleanly without burning coal?",
            ),
        ),
        (
            "centrifuge_cream",
            (
                "Whirling metal drums fling denser watery milk outward while lighter butterfat collects centrally.",
                "Rapid rotational acceleration separates suspended blood cells cleanly inside laboratory tubes.",
            ),
            (
                "How do dairy factories skim heavy cream within seconds instead of waiting overnight?",
                "Which spinning bench instrument isolates red plasma layers for medical tests?",
            ),
        ),
        (
            "bimetallic_thermostat",
            (
                "Bonded brass and steel strips curl unevenly when warmed, snapping furnace switches off.",
                "Two welded metals expand at unequal rates, bending mechanically to regulate room heaters.",
            ),
            (
                "How do simple wall dials click boilers off automatically without digital microchips?",
                "Why do laminated metallic coils curve predictably inside vintage ovens?",
            ),
        ),
        (
            "amber_fossils",
            (
                "Ancient pine resin entombed primordial beetles intact before hardening into golden gemstones.",
                "Fossilized tree sap shields delicate dinosaur feathers from decay across millions of years.",
            ),
            (
                "How were fragile prehistoric mosquitoes preserved in three dimensions without crushing?",
                "What sticky botanical secretion hardens into translucent orange jewelry?",
            ),
        ),
        (
            "barometer_pressure",
            (
                "Falling mercury columns warn captains that low-pressure storm fronts are approaching rapidly.",
                "Aneroid metal capsules flex inward whenever dense fair-weather air pushes down harder.",
            ),
            (
                "How do sailors forecast sudden squalls hours before dark clouds appear?",
                "What tabletop gauge tracks subtle atmospheric weight shifts indoors?",
            ),
        ),
        (
            "anodizing_aluminum",
            (
                "Electrolytic acid tanks grow tough, dye-absorbing oxide armor directly onto aluminum frames.",
                "Submerging metal laptop shells in charged solutions seals porous surfaces against scratches.",
            ),
            (
                "How do manufacturers color sleek metallic gadgets vibrantly without peeling paint?",
                "What electrochemical bath prevents lightweight alloy parts from corroding outdoors?",
            ),
        ),
        (
            "mangrove_salt",
            (
                "Coastal mangrove roots filter harsh seawater salt while breathing through snorkel-like aerial stilts.",
                "Tidal swamp trees excrete excess brine through waxy leaves and stabilize muddy shorelines.",
            ),
            (
                "How do tropical shoreline forests thrive standing directly in brackish ocean surf?",
                "Which hardy wetland shrubs shield villages from violent cyclone storm surges?",
            ),
        ),
        (
            "thermos_vacuum",
            (
                "Evacuated double glass walls silvered inside block conduction, convection, and radiant heat loss.",
                "Airless metallic flasks keep morning coffee piping hot during freezing alpine hikes.",
            ),
            (
                "Why can insulated travel bottles preserve ice cubes for twenty-four hours?",
                "How does removing air between twin stainless shells stop thermal transfer?",
            ),
        ),
        (
            "geodesic_dome",
            (
                "Triangulated strut networks distribute roof weight evenly without needing interior pillars.",
                "Interlocking spherical polygons enclose huge exhibition halls from remarkably light aluminum.",
            ),
            (
                "Why can Buckminster Fuller's round shelters withstand fierce blizzards using minimal timber?",
                "How do architectural triangles achieve rigid strength across wide spans?",
            ),
        ),
        (
            "enzyme_detergent",
            (
                "Cold-water laundry proteases snip stubborn egg and grass stains into rinseable fragments.",
                "Bio-engineered catalysts break down greasy food residues without scalding hot wash cycles.",
            ),
            (
                "How do modern washing powders remove protein spots cleanly at chilly temperatures?",
                "Why do biological soap formulas brighten clothes gently compared to harsh bleach?",
            ),
        ),
        (
            "kinetic_flywheel",
            (
                "Regenerative hybrid buses spin massive carbon rotors during braking to boost uphill launches.",
                "Vacuum-housed inertial wheels store surplus grid energy briefly via pure rotation.",
            ),
            (
                "How can racing cars capture stopping momentum without heavy chemical batteries?",
                "What rotating mechanical bank smooths brief electrical fluctuations?",
            ),
        ),
        (
            "aqueduct_siphon",
            (
                "Pressurized lead pipes dipped across deep valleys before climbing up opposite hills smoothly.",
                "Sealed inverted conduits carried mountain spring water under rivers into Roman baths.",
            ),
            (
                "How did ancient engineers cross wide gorges without building towering stone arches?",
                "Why does continuous enclosed plumbing push liquids uphill to lower outlets?",
            ),
        ),
        (
            "aurora_magnetosphere",
            (
                "Solar wind particles funnel down polar magnetic field lines, exciting thin oxygen into green curtains.",
                "Charged cosmic ions collide with high nitrogen atoms above Arctic skies, glowing vividly.",
            ),
            (
                "What paints shimmering ribbons across winter nights near northern tundra towns?",
                "Why do geomagnetic storms trigger colorful sky displays far from the equator?",
            ),
        ),
        (
            "cork_bark",
            (
                "Mediterranean oak harvesters strip spongy outer bark every nine years without felling trees.",
                "Honeycomb suberin cells make natural wine stoppers buoyant, elastic, and waterproof.",
            ),
            (
                "How do Iberian forests supply resilient bottle plugs sustainably for centuries?",
                "Why does stripped tree rind regrow cleanly without harming living trunks?",
            ),
        ),
        (
            "quartz_piezo_watch",
            (
                "Coin-cell current vibrates a tiny tuning-fork crystal at exactly 32,768 cycles per tick.",
                "Microscopic mineral oscillators divide steady electronic pulses to step wristwatch hands accurately.",
            ),
            (
                "Why do inexpensive battery timepieces keep better time than coiled spring chronometers?",
                "How does a vibrating silicon-dioxide sliver regulate digital seconds?",
            ),
        ),
        (
            "drip_irrigation",
            (
                "Perforated hoses trickle measured drops straight onto crop roots, slashing desert evaporation.",
                "Low-pressure emitters feed orchard shrubs slowly beneath mulch instead of spraying open air.",
            ),
            (
                "How do arid farms grow lush tomatoes using half the usual river allocation?",
                "Why do buried watering tubes prevent salty mineral crusts across dry fields?",
            ),
        ),
        (
            "ballistic_pendulum",
            (
                "Firing projectiles into suspended wooden blocks converts swift momentum into measurable swing height.",
                "Measuring how far heavy clay targets arc lets physicists calculate bullet velocity cleanly.",
            ),
            (
                "How did nineteenth-century labs determine muzzle speed without high-frame cameras?",
                "Why does capturing a fast pellet inside hanging timber reveal its impact pace?",
            ),
        ),
        (
            "parabolic_dish",
            (
                "Paraboloid aluminum reflectors bounce weak satellite signals onto one central receiver horn.",
                "Concave mirror troughs focus scattered sunshine onto oil pipes to boil steam turbines.",
            ),
            (
                "Why do backyard television antennas use bowl-shaped metal screens?",
                "How do curved dishes gather faint radio whispers from distant spacecraft?",
            ),
        ),
        (
            "leavened_sourdough",
            (
                "Wild lactobacilli acidify fermented grain pastes, creating tangy flavor alongside natural rise.",
                "Symbiotic starter jars pair hardy microbes with sour bacteria for chewy artisan boules.",
            ),
            (
                "Why does rustic bakery bread taste pleasantly sharp without commercial yeast packets?",
                "How do living flour-and-water cultures preserve loaves naturally against mold?",
            ),
        ),
    )

    # 24 completely unseen evaluation topics, strictly domain-disjoint from TRAIN_TOPICS
    EVAL_TOPICS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
        (
            "bat_echolocation",
            (
                "Nocturnal bats emit ultrasonic chirps and decode returning echoes to snatch flying moths midair.",
                "High-frequency vocal clicks bounce off tiny insects, guiding winged mammals cleanly through pitch-black caves.",
            ),
            (
                "How do cave-dwelling flyers hunt agile prey in total darkness without eyesight?",
                "Why do rapid sound reflections let night hunters dodge slender tree branches?",
            ),
        ),
        (
            "lymphatic_drainage",
            (
                "Valved body vessels collect leaked capillary plasma and return filtered lymph toward neck veins.",
                "Muscle contractions squeeze excess interstitial serum through immune nodes back into circulation.",
            ),
            (
                "How does human anatomy clear swollen tissue fluid without a central pump?",
                "Which network transports white cells and dietary fats toward collarbones?",
            ),
        ),
        (
            "submersible_ballast",
            (
                "Flooding external saddle tanks replaces buoyant air with seawater so diving hulls sink smoothly.",
                "Compressed air blasts water out of steel compartments whenever deep craft ascend surfaceward.",
            ),
            (
                "How do ocean explorers control depth without burning heavy propeller fuel?",
                "Why does venting trapped gas let underwater vessels settle onto seabeds?",
            ),
        ),
        (
            "permafrost_methane",
            (
                "Thawing Arctic tundra soil awakens dormant microbes that release potent greenhouse gases rapidly.",
                "Mild northern peat bogs decompose ancient roots, venting trapped hydrocarbon bubbles skyward.",
            ),
            (
                "Why do melting Siberian plains accelerate planetary climate warming?",
                "What happens when frozen polar turf softens after centuries of ice?",
            ),
        ),
        (
            "ceramic_glaze",
            (
                "Firing powdered feldspar and silica coats porous stoneware clay in waterproof glassy enamel.",
                "Kiln flames melt metallic oxide dusts across pottery bowls, sealing microscopic pores permanently.",
            ),
            (
                "How do studio potters make earthen mugs watertight and glossy inside ovens?",
                "Why do raw terracotta jugs leak unless baked with mineral coatings?",
            ),
        ),
        (
            "stratospheric_ozone",
            (
                "Upper-atmosphere triatomic oxygen protects terrestrial DNA by absorbing harsh ultraviolet solar rays.",
                "High-altitude O3 molecules intercept burning UV photons long before reaching sunlit skin.",
            ),
            (
                "Why did banning chlorofluorocarbon sprays help heal polar sky shields?",
                "What atmospheric blanket blocks cancer-causing radiation from outer space?",
            ),
        ),
        (
            "origami_miura",
            (
                "Repeating parallelogram creases let rigid solar sails fold compactly inside narrow rocket fairings.",
                "Interlocking diagonal folds unfold huge orbital arrays smoothly with one gentle pull.",
            ),
            (
                "How do aerospace engineers pack wide spacecraft panels tightly before liftoff?",
                "Which Japanese paper-folding pattern inspires deployable satellite wings?",
            ),
        ),
        (
            "circadian_melatonin",
            (
                "Evening darkness cues pineal glands to secrete sleep hormones while blue screens suppress them.",
                "Retinal sensors signal brain pacemakers when dusk arrives, triggering drowsy nighttime chemistry.",
            ),
            (
                "Why does scrolling bright phones at midnight delay natural slumber?",
                "How do human biological clocks synchronize rest cycles with sunset?",
            ),
        ),
        (
            "suspension_cables",
            (
                "Vertical hanger ropes transfer heavy deck weight upward onto twin steel catenary strands.",
                "Anchored towers carry huge tensile forces from sagging main cables down into bedrock.",
            ),
            (
                "How do sleek Golden Gate spans cross mile-wide straits without mid-channel piers?",
                "Why must massive concrete blocks anchor both ends of hanging roadway wires?",
            ),
        ),
        (
            "bioluminescent_plankton",
            (
                "Disturbed coastal dinoflagellates flash electric cyan sparks whenever breaking waves jostle them.",
                "Luciferin oxidation inside microscopic marine cells lights nighttime surf along dark bays.",
            ),
            (
                "Why do ocean paddles glow vivid neon blue during calm summer midnights?",
                "What chemical reaction makes tiny sea organisms sparkle when startled?",
            ),
        ),
        (
            "maglev_superconductor",
            (
                "Repelling magnetic guideways float sleek commuter trains millimeters above tracks without wheel friction.",
                "Alternating track coils pull levitated carriages forward silently past five hundred kilometers hourly.",
            ),
            (
                "How do futuristic express rails glide smoothly without metal-on-metal contact?",
                "Why can electromagnetic propulsion reach aircraft speeds along ground corridors?",
            ),
        ),
        (
            "dendrochronology_rings",
            (
                "Counting wide wet-year tree bands against narrow drought rings dates historic timber cabins.",
                "Concentric oak growth layers preserve accurate multi-century rainfall records inside ancient beams.",
            ),
            (
                "How do historians pinpoint exact felling seasons for medieval roof rafters?",
                "Why do trunk cross-sections reveal forgotten megadroughts cleanly?",
            ),
        ),
        (
            "desalination_reverse",
            (
                "High-pressure pumps squeeze salty brine through tight polymer sheets, leaving pure drinking water.",
                "Semipermeable membranes block dissolved sodium ions when coastal plants exceed osmotic backpressure.",
            ),
            (
                "How do arid desert cities turn bitter ocean surf into fresh tap supplies?",
                "Why do seawater purification stations consume substantial electric power?",
            ),
        ),
        (
            "subduction_trench",
            (
                "Dense basaltic ocean plates dive beneath buoyant granite continents, carving abyssal seafloor trenches.",
                "Sinking lithosphere slabs drag oceanic crust deep into hot mantle zones near island arcs.",
            ),
            (
                "Why does Earth's Pacific rim host its deepest underwater canyons?",
                "What happens where colliding tectonic floors slide downward under coastal ranges?",
            ),
        ),
        (
            "iris_biometrics",
            (
                "Near-infrared cameras photograph intricate crypt patterns around human pupils for secure identity checks.",
                "Unique fibrous muscle textures inside colored eye rings remain stable from childhood onward.",
            ),
            (
                "How do airport security gates verify travelers instantly without touching glass pads?",
                "Why do ocular scanners stay accurate even when people wear clear contact lenses?",
            ),
        ),
        (
            "aquifer_recharge",
            (
                "Porous sandstone beds soak up winter snowmelt slowly, refilling deep rural well reserves.",
                "Permeable gravel basins filter surface runoff downward into confined subterranean water tables.",
            ),
            (
                "Why can paving sprawling parking lots dry up nearby farm boreholes?",
                "How does rainwater replenish hidden underground reservoirs cleanly?",
            ),
        ),
        (
            "solar_eclipse_corona",
            (
                "When lunar discs cover blinding photospheres, wispy pearly plasma halos shine clearly overhead.",
                "Brief midday totality reveals scorching magnetic loops arcing above our star's hidden rim.",
            ),
            (
                "Why can astronomers glimpse faint solar atmospheres only during rare celestial alignments?",
                "What causes sudden twilight chills and crescent shadows beneath trees?",
            ),
        ),
        (
            "pendulum_escapement",
            (
                "Toothed anchor wheels release steady spring ticks each time swinging brass bobs reverse.",
                "Swinging weights regulate gear trains precisely so grandfather clocks mark accurate minutes.",
            ),
            (
                "How did seventeenth-century clockmakers prevent coiled mainsprings from unwinding instantly?",
                "Why does lengthening a hanging rod make vintage hall timepieces run slower?",
            ),
        ),
        (
            "anechoic_wedges",
            (
                "Deep foam pyramids line test chambers to swallow stray echoes completely for microphone calibration.",
                "Suspended wire floors isolate acoustic labs from building rumble and bouncing wall reflections.",
            ),
            (
                "How do audio engineers build eerily silent rooms where heartbeats sound loud?",
                "Why do pointed fiberglass baffles prevent standing sound waves indoors?",
            ),
        ),
        (
            "kombucha_scoby",
            (
                "Symbiotic gelatinous mats ferment sugared black tea into tart sparkling probiotic tonic.",
                "Cellulose pellicles house cooperating bacteria and yeasts that brew acidic amber drinks.",
            ),
            (
                "How does sweetened brewed leaves turn fizzy and vinegary inside countertop jars?",
                "What living rubbery disc floats atop fermented herbal beverages?",
            ),
        ),
        (
            "ballast_keel",
            (
                "Heavy lead fins bolted beneath racing yachts counter sideways gale force, keeping masts upright.",
                "Deep weighted blades stop heeled sailboats from capsizing during fierce crosswinds.",
            ),
            (
                "How can narrow wooden sloops lean forty degrees in squalls without flipping upside down?",
                "Why do modern regatta hulls carry tons of dense metal underwater?",
            ),
        ),
        (
            "mycorrhizal_fungi",
            (
                "Underground fungal threads trade soil phosphorus to tall pines in exchange for sweet sap.",
                "Microscopic root networks link spruce saplings, shuttling minerals across shaded forest floors.",
            ),
            (
                "How do towering conifers gather scarce nutrients through thin rocky dirt?",
                "Why do woodland mushrooms sprout reliably beneath healthy evergreen stands?",
            ),
        ),
        (
            "capstan_winch",
            (
                "Wrapping hemp ropes three turns around rotating drums multiplies deckhand hauling grip exponentially.",
                "Friction coils around vertical barrels let small crews hoist massive anchors safely.",
            ),
            (
                "How could few sailors haul heavy wooden yards aloft using simple grooved cylinders?",
                "Why does adding extra loops around bollards prevent taut mooring lines from slipping?",
            ),
        ),
        (
            "bokeh_aperture",
            (
                "Wide lens openings shrink focal depth, melting distant streetlamps into creamy circular discs.",
                "Rounded iris blades turn out-of-focus background highlights into soft luminous orbs.",
            ),
            (
                "How do portrait photographers blur messy crowds behind sharp smiling faces?",
                "Why do polygonal camera diaphragms shape glowing evening lights differently?",
            ),
        ),
    )

    NEUTRAL_BACKGROUND_SENTENCES: tuple[str, ...] = (
        "Municipal clerks filed quarterly stationery receipts inside grey cabinets.",
        "Tuesday afternoon schedules listed routine elevator inspections across building four.",
        "Corridor bulletin boards displayed updated cafeteria menus for visiting guests.",
        "Inventory ledgers recorded twelve spare oak chairs in basement storage.",
        "Groundskeepers trimmed tidy boxwood hedges beside the brick courtyard.",
        "Archive binders held alphabetical attendance sheets from autumn seminars.",
        "Custodial crews mopped terrazzo hallways promptly after evening lectures.",
        "Receptionists sorted incoming courier parcels onto numbered wooden shelves.",
        "Maintenance staff replaced flickering hallway bulbs along the eastern wing.",
        "Campus shuttles departed every twenty minutes from the north plaza.",
        "Librarians shelved returned hardcover novels before opening doors Wednesday morning.",
        "Facilities coordinators ordered fresh whiteboard markers for seminar rooms.",
        "Security guards logged routine badge checks at the lobby turnstiles.",
        "Bookkeepers balanced petty cash envelopes prior to Friday noon.",
        "Carpenters tightened loose brass hinges on heavy courtyard doors.",
        "Postal clerks stamped outgoing envelopes with standard regional postmarks.",
        "Janitors emptied recycling bins near the third-floor copy alcove.",
        "Event planners stacked folding tables neatly behind auditorium curtains.",
        "Clerical assistants updated telephone directory extensions across four departments.",
        "Gardeners swept fallen maple twigs off brick garden footpaths.",
        "Supply managers counted unopened reams of legal paper on Monday.",
        "Technicians checked projector cables inside lecture hall seven before class.",
        "Reception desks offered printed campus maps to arriving visitors.",
        "Auditors verified serial numbers printed on office desk lamps.",
    )

    @classmethod
    def assert_disjoint_splits(cls) -> None:
        """Verify 100% disjoint topics/text and strict 0 content-word overlap across every (passage, query)."""
        train_ids = {t[0] for t in cls.TRAIN_TOPICS}
        eval_ids = {t[0] for t in cls.EVAL_TOPICS}
        assert len(cls.TRAIN_TOPICS) >= 64, f"Expected >=64 train topics, got {len(cls.TRAIN_TOPICS)}"
        assert len(cls.EVAL_TOPICS) >= 24, f"Expected >=24 eval topics, got {len(cls.EVAL_TOPICS)}"
        assert len(train_ids) == len(cls.TRAIN_TOPICS), "Duplicate topic_id in TRAIN_TOPICS"
        assert len(eval_ids) == len(cls.EVAL_TOPICS), "Duplicate topic_id in EVAL_TOPICS"
        assert train_ids.isdisjoint(eval_ids), f"Overlapping topic IDs: {train_ids & eval_ids}"

        train_texts = {s for _, ps, qs in cls.TRAIN_TOPICS for s in (*ps, *qs)}
        eval_texts = {s for _, ps, qs in cls.EVAL_TOPICS for s in (*ps, *qs)}
        assert train_texts.isdisjoint(eval_texts), "Overlapping text between train and eval topics"

        for split_name, bank in (("train", cls.TRAIN_TOPICS), ("eval", cls.EVAL_TOPICS)):
            for tid, passages, queries in bank:
                for p_txt in passages:
                    p_words = content_words(p_txt)
                    p_stems = {_simple_plural_stem(w) for w in p_words}
                    for q_txt in queries:
                        q_words = content_words(q_txt)
                        overlap = q_words & p_words
                        assert overlap == set(), (
                            f"[{split_name}:{tid}] non-zero content overlap {overlap} between "
                            f"query={q_txt!r} and passage={p_txt!r}"
                        )
                        q_stems = {_simple_plural_stem(w) for w in q_words}
                        stem_overlap = q_stems & p_stems
                        assert stem_overlap == set(), (
                            f"[{split_name}:{tid}] plural/stem overlap {stem_overlap} between "
                            f"query={q_txt!r} and passage={p_txt!r}"
                        )

    def __init__(
        self,
        num_episodes: int = 24,
        num_pages: int = 16,
        page_size: int = 256,
        split: str = "eval",
        tokenizer: Any | None = None,
        vocab_size: int = 512,
        seed: int = 2026,
        background_token_pool: list[int] | None = None,
        num_distractor_topics: int | None = None,
    ) -> None:
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}.")
        if num_pages < 2:
            raise ValueError(f"num_pages must be >= 2, got {num_pages}.")
        self.assert_disjoint_splits()
        self.num_episodes = num_episodes
        self.num_pages = num_pages
        self.page_size = page_size
        self.split = split
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer) if tokenizer is not None else vocab_size
        self.seed = seed
        self.background_token_pool = background_token_pool
        self.topic_bank = self.TRAIN_TOPICS if split == "train" else self.EVAL_TOPICS
        self.num_distractor_topics = (
            min(num_pages - 1, 7 if num_pages <= 16 else 15)
            if num_distractor_topics is None
            else min(num_pages - 1, max(0, int(num_distractor_topics)))
        )
        self._bg_sentences = self._prepare_bg_sentences()
        self.episodes = [self._build_episode(i) for i in range(num_episodes)]

    def _prepare_bg_sentences(self) -> list[str]:
        sents: list[str] = []
        if self.tokenizer is not None and self.background_token_pool and len(self.background_token_pool) >= 64:
            stride = 32
            for start in range(0, min(len(self.background_token_pool) - 32, 3200), stride):
                chunk_txt = self.tokenizer.decode(self.background_token_pool[start : start + 32], skip_special_tokens=True)
                chunk_txt = re.sub(r"\s+", " ", chunk_txt).strip()
                if len(chunk_txt) >= 24 and not re.fullmatch(r"(?:t\d+\s*)+", chunk_txt):
                    sents.append(chunk_txt)
        sents.extend(self.NEUTRAL_BACKGROUND_SENTENCES)
        return sents

    def _encode_fixed_page(self, text: str) -> list[int]:
        toks = _encode_text_or_mock(text, self.tokenizer, self.vocab_size)
        pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0) if self.tokenizer is not None else 0
        if len(toks) >= self.page_size:
            return toks[: self.page_size]
        return toks + [pad_id] * (self.page_size - len(toks))

    def _build_filtered_page_text(
        self,
        rng: random.Random,
        core_sentence: str,
        forbidden_words: set[str],
    ) -> str:
        safe_bg = [s for s in self._bg_sentences if not (content_words(s) & forbidden_words)]
        if not safe_bg:
            safe_bg = [s for s in self.NEUTRAL_BACKGROUND_SENTENCES if not (content_words(s) & forbidden_words)]
        if not safe_bg:
            safe_bg = ["Routine inventory logs recorded standard shelf counts."]

        before_sents = [rng.choice(safe_bg) for _ in range(rng.randint(1, 2))]
        after_sents = [rng.choice(safe_bg) for _ in range(rng.randint(2, 4))]
        # Keep core_sentence early (< page_size tokens) and fill page_size with real prose (0 pad tokens)
        parts = before_sents + [core_sentence] + after_sents
        text = " ".join(parts)
        for _ in range(24):
            if len(_encode_text_or_mock(text, self.tokenizer, self.vocab_size)) >= self.page_size:
                break
            text = text + " " + rng.choice(safe_bg)
        return text

    def _build_episode(self, idx: int) -> LSLCEpisode:
        rng = random.Random(self.seed + idx * 4099 + (0 if self.split == "train" else 77773))
        n_topics = len(self.topic_bank)
        topic_idx = idx % n_topics if (self.split == "eval" or idx < n_topics) else rng.randrange(n_topics)
        topic_id, passages, queries = self.topic_bank[topic_idx]
        p_var = (
            (idx // n_topics) % len(passages)
            if (self.split == "eval" or idx < n_topics * len(passages))
            else rng.randrange(len(passages))
        )
        q_var = (idx // (n_topics * len(passages))) % len(queries) if self.split == "eval" else rng.randrange(len(queries))
        passage_text = passages[p_var]
        query_text = queries[q_var]

        q_words = content_words(query_text)
        target_page = rng.randrange(self.num_pages)

        other_page_indices = [p for p in range(self.num_pages) if p != target_page]
        rng.shuffle(other_page_indices)
        distractor_pages = other_page_indices[: self.num_distractor_topics]
        other_topic_indices = [ti for ti in range(n_topics) if ti != topic_idx]
        rng.shuffle(other_topic_indices)

        page_texts: list[str] = [""] * self.num_pages
        page_texts[target_page] = self._build_filtered_page_text(rng, passage_text, q_words)

        distractor_topic_map: dict[int, str] = {}
        for d_i, p_idx in enumerate(distractor_pages):
            d_tid, d_passages, _ = self.topic_bank[other_topic_indices[d_i % len(other_topic_indices)]]
            d_passage = rng.choice(d_passages)
            page_texts[p_idx] = self._build_filtered_page_text(rng, d_passage, q_words)
            distractor_topic_map[p_idx] = d_tid

        for p_idx in range(self.num_pages):
            if not page_texts[p_idx]:
                page_texts[p_idx] = self._build_filtered_page_text(rng, rng.choice(self.NEUTRAL_BACKGROUND_SENTENCES), q_words)

        # Hard assertion: zero content-word overlap between query and target page text
        overlap_raw = q_words & content_words(page_texts[target_page])
        assert overlap_raw == set(), f"Zero-overlap violation in episode {idx} ({topic_id}): {overlap_raw}"

        page_token_lists = [self._encode_fixed_page(t) for t in page_texts]
        if self.tokenizer is not None:
            mock_wrapped = getattr(self.tokenizer, "_next_id", 0) > getattr(self.tokenizer, "vocab_size", 1 << 30)
            decoded_target = self.tokenizer.decode(page_token_lists[target_page], skip_special_tokens=True)
            if not mock_wrapped and (q_words & content_words(decoded_target)):
                # Fallback to purely neutral sentences + passage if BPE pool decoding created an accidental overlap
                safe_neutral = [s for s in self.NEUTRAL_BACKGROUND_SENTENCES if not (content_words(s) & q_words)]
                rebuilt = " ".join(([rng.choice(safe_neutral)] * 2) + [passage_text] + ([rng.choice(safe_neutral)] * 4))
                page_texts[target_page] = rebuilt
                page_token_lists[target_page] = self._encode_fixed_page(rebuilt)
                decoded_target = self.tokenizer.decode(page_token_lists[target_page], skip_special_tokens=True)
            if not mock_wrapped:
                assert q_words & content_words(decoded_target) == set(), (
                    f"Decoded target page violated zero-overlap in episode {idx} ({topic_id}): "
                    f"{q_words & content_words(decoded_target)}"
                )

        rq_ids = torch.tensor(_encode_text_or_mock(query_text, self.tokenizer, self.vocab_size), dtype=torch.long)
        ans_ids = torch.tensor(_encode_text_or_mock(f" {passage_text[:64]}", self.tokenizer, self.vocab_size), dtype=torch.long)
        return LSLCEpisode(
            context_pages=torch.tensor(page_token_lists, dtype=torch.long),
            query_ids=rq_ids,
            answer_ids=ans_ids,
            target_page_idx=target_page,
            metadata={
                "split": self.split,
                "topic_id": topic_id,
                "topic_idx": topic_idx,
                "query_text": query_text,
                "target_passage": passage_text,
                "target_page_text": page_texts[target_page],
                "page_texts": page_texts,
                "distractor_pages": distractor_topic_map,
                "router_query_ids": rq_ids.clone(),
                "num_pages": self.num_pages,
            },
        )

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> LSLCEpisode:
        return self.episodes[idx]
