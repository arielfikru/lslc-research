"""Cross-file API migration benchmark and coding-agent harness for LSLC v3.

Evaluates multi-turn coding agents on maintaining cross-file contracts after an
API signature change across realistic 38-64 file (8K-16K token) Python repos:
updating multiple production callers and mock-based unit tests using only
working memory + retrieved/re-encoded contract & session decisions while
preserving an unrelated homonym function in another module.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Callable


@dataclass
class MigrationTask:
    """Single cross-file API migration episode."""

    task_id: str
    split: str
    pkg_name: str
    symbol_name: str
    old_signature: str
    new_signature: str
    new_kwarg: str
    default_value: str
    prompt: str
    files: dict[str, str]
    gold_files: dict[str, str]
    hidden_tests: str
    caller_files: list[str]
    homonym_file: str
    decoy_value: str = ""
    core_file: str = ""
    core_old_content: str = ""
    core_new_content: str = ""
    initial_files: dict[str, str] = field(default_factory=dict)
    session_notes_path: str = ""
    session_notes_initial: str = ""
    session_turns: list[dict[str, object]] = field(default_factory=list)


@dataclass
class MigrationEvalResult:
    """Verification and efficiency telemetry for one migration episode."""

    task_id: str
    patch_correct: bool
    stale_signature_used: bool
    homonym_preserved: bool
    hidden_test_passed: bool
    failure_reason: str = ""
    files_read_total: int = 0
    repeated_file_reads: int = 0
    chars_read_total: int = 0
    search_calls: int = 0
    memory_update_ms: float = 0.0
    wall_clock_ms: float = 0.0
    decoy_value_used: bool = False
    active_prefix_tokens: int = 0
    peak_vram_mb: float = 0.0


@dataclass(frozen=True)
class _MigrationSpec:
    task_id: str
    pkg_name: str
    symbol_name: str
    arg_name: str
    arg_default_expr: str
    new_kwarg: str
    kwarg_value_expr: str
    decoy_value_expr: str
    homonym_return_prefix: str


_TRAIN_SPECS: tuple[_MigrationSpec, ...] = (
    _MigrationSpec(
        task_id="train_01_parse_manifest",
        pkg_name="pkgalpha",
        symbol_name="parse_manifest",
        arg_name="raw_path",
        arg_default_expr='"manifest.json"',
        new_kwarg="strict",
        kwarg_value_expr="True",
        decoy_value_expr="False",
        homonym_return_prefix="legacy-manifest",
    ),
    _MigrationSpec(
        task_id="train_02_build_token",
        pkg_name="pkgbeta",
        symbol_name="build_token",
        arg_name="subject",
        arg_default_expr='"worker-1"',
        new_kwarg="scope",
        kwarg_value_expr='"read"',
        decoy_value_expr='"admin"',
        homonym_return_prefix="legacy-token",
    ),
    _MigrationSpec(
        task_id="train_03_dispatch_job",
        pkg_name="pkggamma",
        symbol_name="dispatch_job",
        arg_name="payload",
        arg_default_expr='"sync"',
        new_kwarg="priority",
        kwarg_value_expr='"high"',
        decoy_value_expr='"low"',
        homonym_return_prefix="legacy-job",
    ),
    _MigrationSpec(
        task_id="train_04_fetch_schema",
        pkg_name="pkgdelta",
        symbol_name="fetch_schema",
        arg_name="name",
        arg_default_expr='"events"',
        new_kwarg="version",
        kwarg_value_expr='"v2"',
        decoy_value_expr='"v0"',
        homonym_return_prefix="legacy-schema",
    ),
)

_EVAL_SPECS: tuple[_MigrationSpec, ...] = (
    _MigrationSpec(
        task_id="eval_01_load_config",
        pkg_name="cfgflow",
        symbol_name="load_config",
        arg_name="path",
        arg_default_expr='"app.toml"',
        new_kwarg="env",
        kwarg_value_expr='"prod"',
        decoy_value_expr='"dev"',
        homonym_return_prefix="legacy-cfg",
    ),
    _MigrationSpec(
        task_id="eval_02_open_session",
        pkg_name="sesskit",
        symbol_name="open_session",
        arg_name="endpoint",
        arg_default_expr='"db://primary"',
        new_kwarg="tenant",
        kwarg_value_expr='"acme"',
        decoy_value_expr='"guest"',
        homonym_return_prefix="legacy-sess",
    ),
    _MigrationSpec(
        task_id="eval_03_emit_metric",
        pkg_name="telemetryx",
        symbol_name="emit_metric",
        arg_name="metric_id",
        arg_default_expr='"latency_ms"',
        new_kwarg="region",
        kwarg_value_expr='"us-east"',
        decoy_value_expr='"eu-west"',
        homonym_return_prefix="legacy-metric",
    ),
    _MigrationSpec(
        task_id="eval_04_resolve_route",
        pkg_name="routepath",
        symbol_name="resolve_route",
        arg_name="uri",
        arg_default_expr='"/v1/items"',
        new_kwarg="cluster",
        kwarg_value_expr='"edge"',
        decoy_value_expr='"canary"',
        homonym_return_prefix="legacy-route",
    ),
    _MigrationSpec(
        task_id="eval_05_acquire_lease",
        pkg_name="locksmith",
        symbol_name="acquire_lease",
        arg_name="resource",
        arg_default_expr='"shard-0"',
        new_kwarg="owner",
        kwarg_value_expr='"scheduler"',
        decoy_value_expr='"fallback"',
        homonym_return_prefix="legacy-lease",
    ),
    _MigrationSpec(
        task_id="eval_06_render_invoice",
        pkg_name="billcraft",
        symbol_name="render_invoice",
        arg_name="account_id",
        arg_default_expr='"acct-9"',
        new_kwarg="currency",
        kwarg_value_expr='"USD"',
        decoy_value_expr='"EUR"',
        homonym_return_prefix="legacy-invoice",
    ),
)

_SAFE_TIER_VALUES = ("staging", "local", "blue", "green", "sandbox", "archive", "r1", "r3")
_SAFE_CHANNEL_VALUES = ("alpha-q", "beta-q", "mesh-0", "mesh-1", "pipe-a", "pipe-b", "hub-0", "hub-1")


def _generate_distractor_files(spec: _MigrationSpec, num_distractors: int) -> dict[str, str]:
    """Generate realistic distractor Python modules (~240-260 tok each) with zero target leakage."""
    if num_distractors <= 0:
        return {}

    pkg = spec.pkg_name
    forbidden_words = {
        spec.symbol_name,
        spec.new_kwarg,
        spec.kwarg_value_expr.strip("\"'"),
        spec.decoy_value_expr.strip("\"'"),
    }
    distractors: dict[str, str] = {}

    for idx in range(num_distractors):
        kind = idx % 8
        suffix = f"{idx:02d}"
        tier = _SAFE_TIER_VALUES[idx % len(_SAFE_TIER_VALUES)]
        chan = _SAFE_CHANNEL_VALUES[(idx * 3) % len(_SAFE_CHANNEL_VALUES)]
        timeout_ms = 120 + (idx * 35) % 900
        retry_limit = 2 + (idx % 5)
        batch_cap = 16 + (idx % 6) * 8

        if kind == 0:
            path = f"{pkg}/config/profiles_{suffix}.py"
            body = (
                f'"""Configuration profile utilities module {suffix} for {pkg}."""\n\n'
                f"from dataclasses import dataclass\n\n\n"
                f"@dataclass\n"
                f"class ProfileEntry{suffix}:\n"
                f'    """Static tuning descriptor for stage `{tier}`."""\n\n'
                f"    profile_key: str\n"
                f"    tier_label: str = \"{tier}\"\n"
                f"    timeout_ms: int = {timeout_ms}\n"
                f"    retry_budget: int = {retry_limit}\n\n"
                f"    def to_headers(self, *, namespace: str = \"{chan}\") -> dict[str, str]:\n"
                f"        return {{\n"
                f'            "x-profile": self.profile_key,\n'
                f'            "x-tier": self.tier_label,\n'
                f'            "x-namespace": namespace,\n'
                f'            "x-timeout": str(self.timeout_ms),\n'
                f"        }}\n\n\n"
                f"def load_profile_bundle_{suffix}(profile_key: str, *, tier_label: str = \"{tier}\") -> dict[str, object]:\n"
                f'    """Build validated configuration descriptor dictionary."""\n'
                f"    entry = ProfileEntry{suffix}(profile_key=profile_key, tier_label=tier_label)\n"
                f"    headers = entry.to_headers(namespace=\"{chan}\")\n"
                f'    return {{"entry": entry.profile_key, "tier": entry.tier_label, "headers": headers, "cap": {batch_cap}}}\n'
            )
        elif kind == 1:
            path = f"{pkg}/middleware/guard_{suffix}.py"
            body = (
                f'"""Request guard and rate-budget middleware {suffix} for {pkg}."""\n\n'
                f"from dataclasses import dataclass\n\n\n"
                f"@dataclass\n"
                f"class GuardBudget{suffix}:\n"
                f'    """Sliding token counter for inbound requests."""\n\n'
                f"    channel_id: str = \"{chan}\"\n"
                f"    max_burst: int = {batch_cap}\n"
                f"    timeout_ms: int = {timeout_ms}\n\n"
                f"    def allow_request(self, weight: int, *, namespace: str = \"{tier}\") -> dict[str, object]:\n"
                f"        remaining = max(0, self.max_burst - max(0, weight))\n"
                f"        accepted_flag = 1 if remaining > 0 else 0\n"
                f"        return {{\n"
                f'            "channel": self.channel_id,\n'
                f'            "namespace": namespace,\n'
                f'            "remaining": remaining,\n'
                f'            "accepted": accepted_flag,\n'
                f"        }}\n\n\n"
                f"def inspect_guard_state_{suffix}(weight: int, *, channel_id: str = \"{chan}\") -> int:\n"
                f'    """Return bounded remaining token allowance for diagnostics."""\n'
                f"    guard = GuardBudget{suffix}(channel_id=channel_id)\n"
                f"    outcome = guard.allow_request(weight)\n"
                f'    return int(outcome["remaining"]) + {retry_limit}\n'
            )
        elif kind == 2:
            path = f"{pkg}/schemas/payload_{suffix}.py"
            body = (
                f'"""Typed payload serializer schema {suffix} for {pkg}."""\n\n'
                f"from dataclasses import dataclass\n\n\n"
                f"@dataclass\n"
                f"class PayloadEnvelope{suffix}:\n"
                f'    """Canonical wire frame for internal telemetry."""\n\n'
                f"    record_id: str\n"
                f"    format_mode: str = \"compact\"\n"
                f"    shard_index: int = {idx % 7}\n"
                f"    retry_limit: int = {retry_limit}\n\n"
                f"    def encode_frame(self, *, channel_id: str = \"{chan}\") -> dict[str, str]:\n"
                f"        return {{\n"
                f'            "id": self.record_id,\n'
                f'            "mode": self.format_mode,\n'
                f'            "channel": channel_id,\n'
                f'            "shard": str(self.shard_index),\n'
                f"        }}\n\n\n"
                f"def normalize_payload_{suffix}(record_id: str, *, format_mode: str = \"compact\") -> dict[str, str]:\n"
                f'    """Normalize identifier text before queue handoff."""\n'
                f"    cleaned = record_id.strip().lower()\n"
                f"    frame = PayloadEnvelope{suffix}(record_id=cleaned, format_mode=format_mode)\n"
                f"    return frame.encode_frame(channel_id=\"{chan}\")\n"
            )
        elif kind == 3:
            path = f"{pkg}/cli/commands_{suffix}.py"
            body = (
                f'"""Administrative CLI subcommand helpers {suffix} for {pkg}."""\n\n\n'
                f"def build_cli_flags_{suffix}(target_name: str, *, tier_label: str = \"{tier}\", dry_run: int = 0) -> list[str]:\n"
                f'    """Assemble deterministic subprocess flags without side effects."""\n'
                f"    flags = [\n"
                f'        "--target", target_name.strip(),\n'
                f'        "--tier", tier_label,\n'
                f'        "--timeout-ms", "{timeout_ms}",\n'
                f'        "--retries", "{retry_limit}",\n'
                f"    ]\n"
                f"    if dry_run > 0:\n"
                f'        flags.append("--check-only")\n'
                f"    return flags\n\n\n"
                f"def format_cli_summary_{suffix}(target_name: str, *, channel_id: str = \"{chan}\") -> dict[str, str]:\n"
                f'    """Produce structured CLI execution preview."""\n'
                f"    args = build_cli_flags_{suffix}(target_name, tier_label=\"{tier}\")\n"
                f"    return {{\n"
                f'        "command": " ".join(args),\n'
                f'        "channel": channel_id,\n'
                f'        "batch_cap": "{batch_cap}",\n'
                f"    }}\n"
            )
        elif kind == 4:
            path = f"{pkg}/adapters/storage_{suffix}.py"
            body = (
                f'"""Key-value shard adapter helpers {suffix} for {pkg}."""\n\n'
                f"from dataclasses import dataclass, field\n\n\n"
                f"@dataclass\n"
                f"class MemoryShardAdapter{suffix}:\n"
                f'    """In-memory mock persistence adapter for queue records."""\n\n'
                f"    namespace: str = \"{tier}\"\n"
                f"    timeout_ms: int = {timeout_ms}\n"
                f"    records: dict[str, str] = field(default_factory=dict)\n\n"
                f"    def write_slot(self, key: str, value: str, *, zone_tag: str = \"{chan}\") -> str:\n"
                f"        composite = f\"{{self.namespace}}:{{zone_tag}}:{{key}}\"\n"
                f"        self.records[composite] = value\n"
                f"        return composite\n\n"
                f"    def read_slot(self, key: str, *, zone_tag: str = \"{chan}\") -> str:\n"
                f"        composite = f\"{{self.namespace}}:{{zone_tag}}:{{key}}\"\n"
                f'        return self.records.get(composite, "missing")\n\n\n'
                f"def snapshot_adapter_{suffix}(key: str, *, namespace: str = \"{tier}\") -> dict[str, str]:\n"
                f'    """Capture synthetic fetch/store round-trip state."""\n'
                f"    adapter = MemoryShardAdapter{suffix}(namespace=namespace)\n"
                f'    slot = adapter.write_slot(key, "ok")\n'
                f'    return {{"slot": slot, "value": adapter.read_slot(key)}}\n'
            )
        elif kind == 5:
            path = f"{pkg}/workers/pipeline_{suffix}.py"
            body = (
                f'"""Asynchronous batch worker utilities {suffix} for {pkg}."""\n\n\n'
                f"def partition_workload_{suffix}(items: list[str], *, batch_cap: int = {batch_cap}) -> list[list[str]]:\n"
                f'    """Split item keys into bounded uniform worker slices."""\n'
                f"    step = max(1, batch_cap)\n"
                f"    return [items[pos : pos + step] for pos in range(0, len(items), step)]\n\n\n"
                f"def summarize_batch_{suffix}(items: list[str], *, channel_id: str = \"{chan}\") -> dict[str, object]:\n"
                f'    """Compute worker partitioning metrics for runner health."""\n'
                f"    chunks = partition_workload_{suffix}(items, batch_cap={batch_cap})\n"
                f"    return {{\n"
                f'        "channel": channel_id,\n'
                f'        "num_items": len(items),\n'
                f'        "num_chunks": len(chunks),\n'
                f'        "retry_limit": {retry_limit},\n'
                f'        "timeout_ms": {timeout_ms},\n'
                f"    }}\n"
            )
        elif kind == 6:
            path = f"{pkg}/utils/formatting_{suffix}.py"
            body = (
                f'"""Deterministic label and header formatting helper {suffix} for {pkg}."""\n\n\n'
                f"def sanitize_label_{suffix}(raw_text: str, *, default_tag: str = \"{tier}\") -> str:\n"
                f'    """Collapse whitespace and normalize separators safely."""\n'
                f"    stripped = raw_text.strip()\n"
                f"    if not stripped:\n"
                f"        return default_tag\n"
                f'    return "_".join(stripped.lower().split())\n\n\n'
                f"def compose_audit_line_{suffix}(actor: str, action: str, *, channel_id: str = \"{chan}\") -> str:\n"
                f'    """Render structured single-line log summary."""\n'
                f"    safe_actor = sanitize_label_{suffix}(actor)\n"
                f"    safe_action = sanitize_label_{suffix}(action)\n"
                f'    return f"channel={{channel_id}} actor={{safe_actor}} action={{safe_action}} cap={batch_cap}"\n'
            )
        else:
            path = f"tests/test_module_{suffix}.py"
            body = (
                f'"""Auxiliary standalone unit checks module {suffix} for {pkg}."""\n\n'
                f"import unittest\n\n\n"
                f"class TestAuxiliaryHelpers{suffix}(unittest.TestCase):\n"
                f"    def test_budget_clamp_behavior(self) -> None:\n"
                f"        cap = {batch_cap}\n"
                f"        used = {retry_limit}\n"
                f"        remaining = max(0, cap - used)\n"
                f"        self.assertGreater(remaining, 0)\n"
                f"        self.assertEqual(remaining + used, cap)\n\n"
                f"    def test_tier_label_roundtrip(self) -> None:\n"
                f'        tier_label = "{tier}"\n'
                f'        channel_id = "{chan}"\n'
                f'        combined = f"{{tier_label}}:{{channel_id}}:{timeout_ms}"\n'
                f"        parts = combined.split(\":\")\n"
                f"        self.assertEqual(len(parts), 3)\n"
                f"        self.assertEqual(parts[0], tier_label)\n"
            )

        body += (
            f"\n\n"
            f"def describe_module_bounds_{suffix}(*, zone_tag: str = \"{chan}\") -> dict[str, object]:\n"
            f'    """Return bounded operational metadata for module `{suffix}`."""\n'
            f"    return {{\n"
            f'        "module": "{suffix}",\n'
            f'        "package": "{pkg}",\n'
            f'        "tier": "{tier}",\n'
            f'        "zone": zone_tag,\n'
            f'        "timeout_ms": {timeout_ms},\n'
            f'        "retry_limit": {retry_limit},\n'
            f'        "batch_cap": {batch_cap},\n'
            f"    }}\n\n\n"
            f"def clamp_module_metric_{suffix}(value: int, *, low_bound: int = 0, top_bound: int = {batch_cap}) -> int:\n"
            f'    """Clamp integer telemetry metric into inclusive operational limits."""\n'
            f"    return max(low_bound, min(top_bound, int(value)))\n"
        )

        words = set(re.findall(r"[A-Za-z0-9_-]+", body))
        overlap = words & forbidden_words
        if overlap:
            raise AssertionError(
                f"Distractor {path} leaked forbidden task tokens {sorted(overlap)} for {spec.task_id}"
            )
        distractors[path] = body

    return distractors


def _build_session_turns(spec: _MigrationSpec, split: str, task_idx: int) -> list[dict[str, object]]:
    pkg = spec.pkg_name
    sym = spec.symbol_name
    kw = spec.new_kwarg
    kw_val = spec.kwarg_value_expr
    decoy_val = spec.decoy_value_expr

    if split == "train":
        acc_text = (
            f"Engineering lead ratified {kw}={kw_val} as the active caller default for {pkg}.{sym}."
        )
        rej_text = (
            f"Motion to configure {pkg}.{sym} caller default as {kw}={decoy_val} "
            f"was vetoed during code review."
        )
    else:
        acc_text = (
            f"Architecture committee finalized {pkg}.{sym} caller default: "
            f"set {kw}={kw_val} effective immediately."
        )
        if task_idx % 2 == 0:
            rej_text = (
                f"Proposal to set {pkg}.{sym} caller default to {kw}={decoy_val} "
                f"failed approval and was discarded; prior ratified default holds."
            )
        else:
            rej_text = (
                f"Draft proposal to set {pkg}.{sym} caller default to {kw}={decoy_val} "
                f"was denied by review board and discarded."
            )

    acc_turn: dict[str, object] = {
        "turn_id": 0,
        "accepted": True,
        "proposed_value": kw_val,
        "text": acc_text,
    }
    rej_turn: dict[str, object] = {
        "turn_id": 1,
        "accepted": False,
        "proposed_value": decoy_val,
        "text": rej_text,
    }
    turns = [acc_turn, rej_turn] if (task_idx % 2 == 0) else [rej_turn, acc_turn]
    for tidx, t in enumerate(turns, start=1):
        t["turn_id"] = tidx
    return turns


def _build_task_from_spec(
    spec: _MigrationSpec,
    split: str,
    *,
    task_idx: int = 0,
    num_distractor_files: int = 32,
) -> MigrationTask:
    pkg = spec.pkg_name
    sym = spec.symbol_name
    arg = spec.arg_name
    kw = spec.new_kwarg
    kw_val = spec.kwarg_value_expr
    decoy_val = spec.decoy_value_expr

    old_sig = f"{sym}({arg})"
    new_sig = f"{sym}({arg}, *, {kw})"

    init_py = (
        f'"""Package {pkg}."""\n'
        f"from .core import {sym}\n\n"
        f'__all__ = ["{sym}"]\n'
    )
    core_old_py = (
        f'"""Core implementation for {pkg}."""\n\n\n'
        f"def {sym}({arg}: str) -> dict[str, str]:\n"
        f'    """Load or execute `{sym}` under the legacy single-argument contract."""\n'
        f'    return {{"{arg}": {arg}}}\n'
    )
    core_py = (
        f'"""Core implementation for {pkg}."""\n\n\n'
        f"def {sym}({arg}: str, *, {kw}: str) -> dict[str, str]:\n"
        f'    """Load or execute `{sym}` with mandatory `{kw}` context."""\n'
        f'    return {{"{arg}": {arg}, "{kw}": str({kw})}}\n'
    )
    legacy_py = (
        f'"""Unrelated legacy helper with homonym `{sym}` that must NOT change."""\n\n\n'
        f"def {sym}({arg}: str) -> str:\n"
        f'    """Legacy single-argument `{sym}` contract."""\n'
        f'    return f"{spec.homonym_return_prefix}:{{{arg}}}"\n\n\n'
        f"def run_legacy_flow({arg}: str) -> str:\n"
        f"    return {sym}({arg})\n"
    )

    # Zero answer leakage: no TODO comment naming the new kwarg inside caller files!
    service_a_stale = (
        f'"""Production caller A for {pkg}."""\n'
        f"from .core import {sym}\n\n\n"
        f"def run_service_a({arg}: str = {spec.arg_default_expr}) -> dict[str, str]:\n"
        f"    return {sym}({arg})\n"
    )
    service_a_gold = (
        f'"""Production caller A for {pkg}."""\n'
        f"from .core import {sym}\n\n\n"
        f"def run_service_a({arg}: str = {spec.arg_default_expr}, {kw}: str = {kw_val}) -> dict[str, str]:\n"
        f"    return {sym}({arg}, {kw}={kw})\n"
    )

    service_b_stale = (
        f'"""Production caller B (module-qualified import) for {pkg}."""\n'
        f"from . import core\n\n\n"
        f"def run_service_b({arg}: str = {spec.arg_default_expr}) -> dict[str, str]:\n"
        f"    return core.{sym}({arg})\n"
    )
    service_b_gold = (
        f'"""Production caller B (module-qualified import) for {pkg}."""\n'
        f"from . import core\n\n\n"
        f"def run_service_b({arg}: str = {spec.arg_default_expr}, {kw}: str = {kw_val}) -> dict[str, str]:\n"
        f"    return core.{sym}({arg}, {kw}={kw})\n"
    )

    test_stale = (
        f'"""Unit tests for {pkg} callers using a mock replacement for `{sym}`."""\n'
        f"import unittest\n"
        f"from unittest import mock\n\n"
        f"from {pkg} import service_a\n"
        f"from {pkg} import service_b\n\n\n"
        f"class TestServiceCallers(unittest.TestCase):\n"
        f"    def test_callers_use_current_contract(self) -> None:\n"
        f"        seen = []\n\n"
        f"        def fake_{sym}({arg}: str) -> dict[str, str]:\n"
        f"            seen.append(({arg}, None))\n"
        f'            return {{"{arg}": {arg}}}\n\n'
        f'        with mock.patch("{pkg}.service_a.{sym}", side_effect=fake_{sym}), \\\n'
        f'             mock.patch("{pkg}.core.{sym}", side_effect=fake_{sym}):\n'
        f"            out_a = service_a.run_service_a()\n"
        f"            out_b = service_b.run_service_b()\n"
        f"        self.assertEqual(len(seen), 2)\n"
        f'        self.assertIn("{arg}", out_a)\n'
        f'        self.assertIn("{arg}", out_b)\n'
    )
    test_gold = (
        f'"""Unit tests for {pkg} callers using a mock replacement for `{sym}`."""\n'
        f"import unittest\n"
        f"from unittest import mock\n\n"
        f"from {pkg} import service_a\n"
        f"from {pkg} import service_b\n\n\n"
        f"class TestServiceCallers(unittest.TestCase):\n"
        f"    def test_callers_use_current_contract(self) -> None:\n"
        f"        seen = []\n\n"
        f"        def fake_{sym}({arg}: str, *, {kw}: str) -> dict[str, str]:\n"
        f"            seen.append(({arg}, {kw}))\n"
        f'            return {{"{arg}": {arg}, "{kw}": {kw}}}\n\n'
        f'        with mock.patch("{pkg}.service_a.{sym}", side_effect=fake_{sym}), \\\n'
        f'             mock.patch("{pkg}.core.{sym}", side_effect=fake_{sym}):\n'
        f"            out_a = service_a.run_service_a()\n"
        f"            out_b = service_b.run_service_b()\n"
        f"        self.assertEqual(len(seen), 2)\n"
        f'        self.assertEqual(out_a["{kw}"], {kw_val})\n'
        f'        self.assertEqual(out_b["{kw}"], {kw_val})\n'
    )

    hidden_tests = (
        f'"""Hidden grading suite for {pkg}.{sym} migration."""\n'
        f"import inspect\n"
        f"import unittest\n"
        f"from {pkg} import core, legacy_compat, service_a, service_b\n\n\n"
        f"class HiddenMigrationVerification(unittest.TestCase):\n"
        f"    def test_core_and_callers_pass_end_to_end(self) -> None:\n"
        f"        res_a = service_a.run_service_a()\n"
        f"        res_b = service_b.run_service_b()\n"
        f'        self.assertIn("{kw}", res_a)\n'
        f'        self.assertIn("{kw}", res_b)\n'
        f'        self.assertEqual(str(res_a["{kw}"]), str({kw_val}))\n'
        f'        self.assertEqual(str(res_b["{kw}"]), str({kw_val}))\n'
        f"        sig = inspect.signature(core.{sym})\n"
        f'        self.assertEqual(sig.parameters["{kw}"].kind, inspect.Parameter.KEYWORD_ONLY)\n\n'
        f"    def test_homonym_remains_intact(self) -> None:\n"
        f"        sig_legacy = inspect.signature(legacy_compat.{sym})\n"
        f'        self.assertEqual(list(sig_legacy.parameters.keys()), ["{arg}"])\n'
        f'        out = legacy_compat.run_legacy_flow("sample")\n'
        f'        self.assertEqual(out, "{spec.homonym_return_prefix}:sample")\n'
    )

    distractors = _generate_distractor_files(spec, num_distractor_files)
    core_file = f"{pkg}/core.py"

    files = {
        f"{pkg}/__init__.py": init_py,
        core_file: core_py,
        f"{pkg}/service_a.py": service_a_stale,
        f"{pkg}/service_b.py": service_b_stale,
        f"{pkg}/legacy_compat.py": legacy_py,
        "tests/test_service.py": test_stale,
        **distractors,
    }
    initial_files = dict(files)
    initial_files[core_file] = core_old_py

    gold_files = {
        f"{pkg}/__init__.py": init_py,
        core_file: core_py,
        f"{pkg}/service_a.py": service_a_gold,
        f"{pkg}/service_b.py": service_b_gold,
        f"{pkg}/legacy_compat.py": legacy_py,
        "tests/test_service.py": test_gold,
        **distractors,
    }

    # Zero answer leakage in prompt: contract & approved default live ONLY in memory.
    prompt = (
        f"In package `{pkg}`, `{pkg}/core.py:{sym}` was recently migrated to a new "
        f"keyword-only API contract, and the team finalized the required caller default "
        f"during the architecture review session. Consult the updated `{pkg}/core.py` "
        f"and architecture session notes in memory to update `{pkg}/service_a.py`, "
        f"`{pkg}/service_b.py`, and `tests/test_service.py` to match the ratified contract "
        f"and approved caller default, while leaving `{pkg}/legacy_compat.py:{sym}` untouched."
    )

    session_notes_path = f"{pkg}/SESSION_NOTES.md"
    session_notes_initial = (
        f"# Architecture Review Notes: {pkg}\n"
        f"- Package `{pkg}` maintains `{pkg}/core.py:{sym}` and callers `{pkg}/service_a.py`, "
        f"`{pkg}/service_b.py`, and `tests/test_service.py`.\n"
        f"- Policy: `{pkg}/legacy_compat.py:{sym}` is a separate homonym adapter and must remain unchanged.\n"
        f"- Status: API keyword-only migration and caller default ratification are pending review entries below."
    )
    session_turns = _build_session_turns(spec, split=split, task_idx=task_idx)

    return MigrationTask(
        task_id=spec.task_id,
        split=split,
        pkg_name=pkg,
        symbol_name=sym,
        old_signature=old_sig,
        new_signature=new_sig,
        new_kwarg=kw,
        default_value=kw_val,
        prompt=prompt,
        files=files,
        gold_files=gold_files,
        hidden_tests=hidden_tests,
        caller_files=[f"{pkg}/service_a.py", f"{pkg}/service_b.py", "tests/test_service.py"],
        homonym_file=f"{pkg}/legacy_compat.py",
        decoy_value=decoy_val,
        core_file=core_file,
        core_old_content=core_old_py,
        core_new_content=core_py,
        initial_files=initial_files,
        session_notes_path=session_notes_path,
        session_notes_initial=session_notes_initial,
        session_turns=session_turns,
    )


class CrossFileAPIMigrationBenchmark:
    """Repository generator for disjoint train/eval cross-file API migrations."""

    TRAIN_SPECS: tuple[_MigrationSpec, ...] = _TRAIN_SPECS
    EVAL_SPECS: tuple[_MigrationSpec, ...] = _EVAL_SPECS

    def __init__(self, split: str = "eval", num_distractor_files: int = 32) -> None:
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        if num_distractor_files < 0:
            raise ValueError(f"num_distractor_files must be >= 0, got {num_distractor_files}")
        self.split = split
        self.num_distractor_files = int(num_distractor_files)
        specs = self.TRAIN_SPECS if split == "train" else self.EVAL_SPECS
        self.tasks: list[MigrationTask] = [
            _build_task_from_spec(
                s,
                split,
                task_idx=idx,
                num_distractor_files=self.num_distractor_files,
            )
            for idx, s in enumerate(specs)
        ]

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> MigrationTask:
        return self.tasks[idx]

    @classmethod
    def assert_disjoint_splits(cls) -> None:
        """Raise ValueError if train and eval specs share packages, symbols, kwargs, or defaults."""
        for spec in cls.TRAIN_SPECS + cls.EVAL_SPECS:
            if spec.kwarg_value_expr == spec.decoy_value_expr:
                raise ValueError(f"Spec {spec.task_id} must use distinct approved vs decoy values")
        train_pkgs = {s.pkg_name for s in cls.TRAIN_SPECS}
        eval_pkgs = {s.pkg_name for s in cls.EVAL_SPECS}
        train_syms = {s.symbol_name for s in cls.TRAIN_SPECS}
        eval_syms = {s.symbol_name for s in cls.EVAL_SPECS}
        train_kws = {s.new_kwarg for s in cls.TRAIN_SPECS}
        eval_kws = {s.new_kwarg for s in cls.EVAL_SPECS}
        train_vals = {s.kwarg_value_expr for s in cls.TRAIN_SPECS}
        eval_vals = {s.kwarg_value_expr for s in cls.EVAL_SPECS}
        train_decoys = {s.decoy_value_expr for s in cls.TRAIN_SPECS}
        eval_decoys = {s.decoy_value_expr for s in cls.EVAL_SPECS}
        if (
            (train_pkgs & eval_pkgs)
            or (train_syms & eval_syms)
            or (train_kws & eval_kws)
            or (train_vals & eval_vals)
            or (train_decoys & eval_decoys)
        ):
            raise ValueError("TRAIN_SPECS and EVAL_SPECS must have strictly disjoint names and values")


def _extract_fn_signature_shape(src: str, symbol: str) -> tuple[list[str], list[str], bool] | None:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == symbol:
            pos = [a.arg for a in node.args.posonlyargs + node.args.args]
            kwonly = [a.arg for a in node.args.kwonlyargs]
            has_varkw = node.args.kwarg is not None
            return pos, kwonly, has_varkw
    return None


def _matches_exact_literal(node: ast.AST | None, target_val: object) -> bool:
    if node is None:
        return False
    try:
        parsed = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return False
    return type(parsed) is type(target_val) and parsed == target_val


def _check_decoy_value_used(task: MigrationTask, candidate_files: dict[str, str]) -> bool:
    if not task.decoy_value:
        return False
    try:
        decoy_lit = ast.literal_eval(task.decoy_value)
    except (ValueError, SyntaxError):
        return False

    kw = task.new_kwarg
    sym = task.symbol_name
    for path in task.caller_files:
        src = candidate_files.get(path)
        if not src:
            continue
        try:
            tree = ast.parse(src, filename=path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                all_args = list(node.args.posonlyargs) + list(node.args.args)
                for a_node, d_node in zip(reversed(all_args), reversed(node.args.defaults)):
                    if a_node.arg == kw and _matches_exact_literal(d_node, decoy_lit):
                        return True
                for a_node, d_node in zip(node.args.kwonlyargs, node.args.kw_defaults):
                    if a_node.arg == kw and _matches_exact_literal(d_node, decoy_lit):
                        return True
            elif isinstance(node, ast.Call):
                func = node.func
                is_sym_or_caller = (
                    (isinstance(func, ast.Name) and (func.id == sym or func.id.startswith("run_service_")))
                    or (isinstance(func, ast.Attribute) and (func.attr == sym or func.attr.startswith("run_service_")))
                )
                if is_sym_or_caller:
                    for kw_node in node.keywords:
                        if kw_node.arg == kw and _matches_exact_literal(kw_node.value, decoy_lit):
                            return True
            elif path == "tests/test_service.py" and isinstance(node, ast.Constant):
                if isinstance(node.value, type(decoy_lit)) and node.value == decoy_lit:
                    return True
    return False


def _check_caller_files_stale(task: MigrationTask, candidate_files: dict[str, str]) -> tuple[bool, str]:
    sym = task.symbol_name
    kw = task.new_kwarg

    for path in task.caller_files:
        src = candidate_files.get(path)
        if src is None:
            return True, f"Missing required file {path}"
        try:
            tree = ast.parse(src, filename=path)
        except SyntaxError as exc:
            return True, f"SyntaxError in {path}: {exc}"

        saw_call_or_fake = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                is_target = (
                    (isinstance(func, ast.Name) and func.id == sym)
                    or (isinstance(func, ast.Attribute) and func.attr == sym)
                )
                if is_target:
                    saw_call_or_fake = True
                    kw_names = {k.arg for k in node.keywords if k.arg is not None}
                    has_double_star = any(k.arg is None for k in node.keywords)
                    if kw not in kw_names and not has_double_star:
                        return True, f"Stale call to {sym}() without `{kw}=` in {path}"
            elif path == "tests/test_service.py" and isinstance(node, ast.FunctionDef):
                if node.name == f"fake_{sym}" or node.name == sym:
                    saw_call_or_fake = True
                    kwonly = [a.arg for a in node.args.kwonlyargs]
                    pos_and_kw = [a.arg for a in node.args.args]
                    has_varkw = node.args.kwarg is not None
                    if kw not in kwonly and kw not in pos_and_kw and not has_varkw:
                        return True, f"Stale mock `{node.name}` missing `{kw}` in {path}"

        if not saw_call_or_fake:
            return True, f"No call or mock for `{sym}` found in {path}"

    return False, ""


def _check_homonym_preserved(task: MigrationTask, candidate_files: dict[str, str]) -> tuple[bool, str]:
    orig_src = task.files[task.homonym_file]
    cand_src = candidate_files.get(task.homonym_file)
    if cand_src is None:
        return False, f"Missing homonym file {task.homonym_file}"
    try:
        orig_shape = _extract_fn_signature_shape(orig_src, task.symbol_name)
        cand_shape = _extract_fn_signature_shape(cand_src, task.symbol_name)
    except SyntaxError as exc:
        return False, f"SyntaxError in homonym file {task.homonym_file}: {exc}"
    if orig_shape is None or cand_shape != orig_shape:
        return False, f"Homonym `{task.homonym_file}:{task.symbol_name}` signature was altered"
    if f"{task.new_kwarg}=" in cand_src:
        return False, f"Homonym file `{task.homonym_file}` contains migrated keyword `{task.new_kwarg}=`"
    return True, ""


def verify_migration_patch(
    task: MigrationTask,
    candidate_files: dict[str, str],
    *,
    run_subprocess: bool = True,
    timeout_sec: float = 10.0,
) -> MigrationEvalResult:
    """Verify patch via AST stale-signature/homonym checks and subprocess unit tests."""
    stale, stale_reason = _check_caller_files_stale(task, candidate_files)
    homonym_ok, homonym_reason = _check_homonym_preserved(task, candidate_files)
    decoy_used = _check_decoy_value_used(task, candidate_files)

    hidden_passed = False
    test_reason = ""
    if run_subprocess:
        with tempfile.TemporaryDirectory() as tmpdir:
            for rel_path, content in candidate_files.items():
                abs_path = os.path.join(tmpdir, rel_path)
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                with open(abs_path, "w", encoding="utf-8") as fh:
                    fh.write(content)
            tests_dir = os.path.join(tmpdir, "tests")
            if os.path.isdir(tests_dir):
                tests_init = os.path.join(tests_dir, "__init__.py")
                if not os.path.exists(tests_init):
                    with open(tests_init, "w", encoding="utf-8") as fh:
                        fh.write("")
            hidden_path = os.path.join(tmpdir, "_hidden_verify.py")
            with open(hidden_path, "w", encoding="utf-8") as fh:
                fh.write(task.hidden_tests)
            env = dict(os.environ)
            env["PYTHONPATH"] = tmpdir
            proc = subprocess.run(
                [sys.executable, "-m", "unittest", "_hidden_verify", "tests.test_service"],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            hidden_passed = proc.returncode == 0
            if not hidden_passed:
                err_tail = (proc.stderr or proc.stdout or "").strip().splitlines()
                test_reason = " | ".join(err_tail[-3:])[:240] if err_tail else f"unittest exited with {proc.returncode}"
    else:
        hidden_passed = (not stale) and homonym_ok and (not decoy_used)

    reasons = [r for r in (stale_reason, homonym_reason, test_reason) if r]
    if decoy_used:
        reasons.append(f"Rejected decoy value {task.decoy_value} used")
    patch_correct = (not stale) and homonym_ok and hidden_passed
    return MigrationEvalResult(
        task_id=task.task_id,
        patch_correct=patch_correct,
        stale_signature_used=stale,
        homonym_preserved=homonym_ok,
        hidden_test_passed=hidden_passed,
        failure_reason="; ".join(reasons),
        decoy_value_used=decoy_used,
    )


class CodingAgentHarness:
    """Tool-matched execution harness tracking reads, searches, and LSLC memory overhead."""

    def __init__(
        self,
        task: MigrationTask,
        on_read: Callable[[str, str], float] | None = None,
        on_edit: Callable[[str, str, str], float] | None = None,
        *,
        use_initial_files: bool = False,
    ) -> None:
        self.task = task
        base_files = task.initial_files if (use_initial_files and task.initial_files) else task.files
        self.files: dict[str, str] = dict(base_files)
        self._on_read = on_read
        self._on_edit = on_edit
        self._seen_paths: set[str] = set()
        self.files_read_total = 0
        self.repeated_file_reads = 0
        self.chars_read_total = 0
        self.search_calls = 0
        self.memory_update_ms = 0.0
        self.active_prefix_tokens = 0
        self.peak_vram_mb = 0.0
        self._t0 = time.perf_counter()

    def list_files(self) -> list[str]:
        return sorted(self.files.keys())

    def search_code(self, pattern: str, regex: bool = False) -> list[dict[str, object]]:
        self.search_calls += 1
        compiled = re.compile(pattern) if regex else None
        hits: list[dict[str, object]] = []
        for path in sorted(self.files.keys()):
            for line_no, line in enumerate(self.files[path].splitlines(), start=1):
                matched = bool(compiled.search(line)) if compiled is not None else (pattern in line)
                if matched:
                    hits.append({"path": path, "line": line_no, "text": line})
        return hits

    def read_file(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(f"Unknown task file: {path}")
        content = self.files[path]
        self.files_read_total += 1
        if path in self._seen_paths:
            self.repeated_file_reads += 1
        else:
            self._seen_paths.add(path)
        self.chars_read_total += len(content)
        if self._on_read is not None:
            self.memory_update_ms += float(self._on_read(path, content) or 0.0)
        return content

    def apply_file_edit(self, path: str, new_content: str) -> None:
        if path not in self.files:
            raise FileNotFoundError(f"Unknown task file: {path}")
        old_content = self.files[path]
        self.files[path] = new_content
        if self._on_edit is not None:
            self.memory_update_ms += float(self._on_edit(path, old_content, new_content) or 0.0)

    def evaluate(self, *, run_subprocess: bool = True, timeout_sec: float = 10.0) -> MigrationEvalResult:
        res = verify_migration_patch(
            self.task,
            self.files,
            run_subprocess=run_subprocess,
            timeout_sec=timeout_sec,
        )
        res.files_read_total = self.files_read_total
        res.repeated_file_reads = self.repeated_file_reads
        res.chars_read_total = self.chars_read_total
        res.search_calls = self.search_calls
        res.memory_update_ms = self.memory_update_ms
        res.wall_clock_ms = (time.perf_counter() - self._t0) * 1000.0
        res.active_prefix_tokens = self.active_prefix_tokens
        res.peak_vram_mb = self.peak_vram_mb
        return res


__all__ = [
    "CodingAgentHarness",
    "CrossFileAPIMigrationBenchmark",
    "MigrationEvalResult",
    "MigrationTask",
    "verify_migration_patch",
]
