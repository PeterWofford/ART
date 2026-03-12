from __future__ import annotations

from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field
from rich import box
from rich.console import Console
from rich.table import Table
import torch

from .megatron_forward_trace import ForwardTraceCapture

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path(REPO_ROOT / ".local/megatron_lora_correctness")
ORACLE_MOE_ROUTING_BUNDLE_DIRNAME = "oracle_moe_routing_replay"
ORACLE_REPLAY_TOPOLOGY_SUFFIX = "oracle_replay"

REGENERATE_ENV = "ART_REGENERATE_MEGATRON_ORACLE"
EXTENDED_TOPOLOGIES_ENV = "ART_MEGATRON_ORACLE_ENABLE_EXTENDED_TOPOLOGIES"
SENSITIVITY_MUTATION_ENV = "ART_MEGATRON_ORACLE_MUTATION"

DEFAULT_SENSITIVITY_MUTATION = "drop_finalize"
SUPPORTED_SENSITIVITY_MUTATIONS = (DEFAULT_SENSITIVITY_MUTATION,)
SensitivityMutation = str

REQUIRED_PACKED_TENSOR_FILES = (
    "tokens.pt",
    "group_ids.pt",
    "parent_ids.pt",
    "input_pos.pt",
    "assistant_mask.pt",
    "logprobs.pt",
    "advantages.pt",
    "weights.pt",
)
NON_FINITE_METRIC_VALUE = 1e30
EXPERT_TABLE_ROW_LIMIT = 8
EXPERT_TRIPLET_PARAM_RE = re.compile(
    r"layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\."
)
PHASE_PRINT_ORDER = {
    "forward": 0,
    "router_scores": 1,
    "router_topk_ids": 2,
    "outputs": 3,
    "losses": 4,
    "grads": 5,
    "deltas": 6,
}


class Topology(BaseModel):
    """Defines distributed topology settings for one Megatron run variant."""

    model_config = ConfigDict(frozen=True)

    tp: int
    ep: int
    etp: int = 1
    dp: int = 1
    sp: bool = False
    cp: int = 1
    pp: int = 1
    vpp: int = 1

    def resolved_expert_dp(self) -> int:
        """Derives expert data parallel size from topology/world-size constraints."""
        attention_world = self.tp * self.cp * self.pp * self.dp
        expert_divisor = self.etp * self.ep * self.pp
        if attention_world % expert_divisor != 0:
            raise ValueError(
                "Invalid topology for Megatron expert parallelism: "
                f"world_size={attention_world} is not divisible by "
                f"etp*ep*pp={expert_divisor}."
            )
        return attention_world // expert_divisor

    def slug(self) -> str:
        return (
            f"tp{self.tp}_ep{self.ep}_etp{self.etp}"
            f"_dp{self.dp}_edp{self.resolved_expert_dp()}"
            f"_cp{self.cp}_pp{self.pp}_vpp{self.vpp}_sp{int(self.sp)}"
        )

    def world_size(self) -> int:
        # Mirrors Megatron parallel-state sizing:
        # attention side: world = tp * pp * cp * dp
        # expert side must also divide this world size (validated in resolved_expert_dp()).
        attention_world = self.tp * self.cp * self.pp * self.dp
        self.resolved_expert_dp()
        return attention_world


class PackedTensorConfig(BaseModel):
    """Controls synthetic packed tensor generation used by oracle harness runs."""

    num_sequences: int = 8
    sequence_length: int = 256
    prefill_tokens: int = 64
    decode_tokens: int = 64
    vocab_high: int = 8192


class LoraConfig(BaseModel):
    """Configures LoRA adapter dimensions and targeted module families."""

    rank: int = 1
    alpha: int = 32
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class ToleranceProfile(BaseModel):
    """Defines row-level pass/fail thresholds for variant comparison phases."""

    relative_l2: float = 1e-2
    mean_abs_pct: float = 1.0


class OracleCaseConfig(BaseModel):
    """Contains all deterministic run parameters for one oracle case."""

    base_model: str
    num_layers: int = 4
    seed: int = 20260305
    num_steps: int = 2
    learning_rate: float = 1e-3
    beta: float = 0.0
    loss_scale: float = 1e4
    packed_tensors: PackedTensorConfig = Field(default_factory=PackedTensorConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    tolerances: ToleranceProfile = Field(default_factory=ToleranceProfile)


class DiskPackedTensorsSpec(BaseModel):
    """Describes packed tensor artifacts persisted on disk for reuse."""

    dir: str
    num_sequences: int
    sequence_length: int
    pixel_values: tuple[int, list[int]] | None = None
    image_grid_thw: tuple[int, list[int]] | None = None


class CaseArtifacts(BaseModel):
    """Holds stable case-level artifact paths used by all variants."""

    case_id: str
    case_dir: str
    packed_tensors: DiskPackedTensorsSpec
    shared_init_adapter_path: str


class WorkerRunRequest(BaseModel):
    """Defines one distributed worker invocation for generating variant artifacts."""

    case_id: str
    case_config: OracleCaseConfig
    topology: Topology
    topology_dir: str
    packed_tensors: DiskPackedTensorsSpec
    shared_init_adapter_path: str
    mutation: SensitivityMutation | None = None
    moe_routing_replay_path: str | None = None
    moe_routing_replay_strict: bool = True
    capture_moe_routing_bundle_path: str | None = None


class StepTrace(BaseModel):
    """Tracks per-step trace artifact filenames and loss metadata."""

    step_index: int
    loss: float
    probs_corr: float
    output_file: str
    grads_file: str
    deltas_file: str
    lora_file: str


class RunManifest(BaseModel):
    """Records run metadata and per-step trace references for one topology output."""

    case_id: str
    base_model: str
    num_layers: int
    topology: str
    world_size: int
    seed: int
    num_steps: int
    packed_tensors: DiskPackedTensorsSpec
    tolerances: ToleranceProfile
    steps: list[StepTrace]


class MetricRow(BaseModel):
    """Represents one comparable unit (param/module/global) for one phase and step."""

    case_id: str
    variant: str
    topology: str
    oracle_topology: str
    step_index: int
    phase: str
    param: str
    numel: float
    mean_abs_diff: float
    relative_l2: float
    typical_abs_scale: float
    mean_abs_pct: float
    topk_mismatch_fraction: float | None = None
    top1_mismatch_fraction: float | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    pass_signal: bool = True
    failure_reasons: list[str] = Field(default_factory=list)


class VariantSpec(BaseModel):
    """Declares how to execute and evaluate one candidate variant against the oracle."""

    name: str
    topology: Topology
    thresholds_by_phase: dict[str, dict[str, float]]
    output_slug: str | None = None
    reference_slug: str | None = None
    mutation: SensitivityMutation | None = None
    expected_signal: Literal["pass", "fail"] = "pass"

    def resolved_output_slug(self) -> str:
        if self.output_slug is not None:
            return self.output_slug
        return _topology_output_slug(self.topology, self.mutation)

    def resolved_reference_slug(self) -> str:
        if self.reference_slug is not None:
            return self.reference_slug
        return ORACLE_TOPOLOGY.slug()


class VariantReport(BaseModel):
    """Captures full comparison output for one variant run."""

    case_id: str
    variant: str
    topology: str
    reference_topology: str
    expected_signal: Literal["pass", "fail"]
    signal: Literal["pass", "fail"]
    pass_count: int
    fail_count: int
    step_summaries: dict[int, dict[str, Any]]
    metrics: list[MetricRow]


class DiffAccumulator:
    """Accumulates diff statistics across tensors and router-id mismatch counters."""

    def __init__(self) -> None:
        self.numel = 0
        self.abs_sum = 0.0
        self.diff_sq_sum = 0.0
        self.ref_sq_sum = 0.0
        self.ref_abs_sum = 0.0
        self.router_topk_total = 0
        self.router_topk_mismatch = 0
        self.router_top1_total = 0
        self.router_top1_mismatch = 0

    def update(self, reference, candidate) -> None:  # type: ignore[no-untyped-def]
        """Adds one tensor pair into the accumulator."""
        ref = reference.detach().float()
        cand = candidate.detach().float()
        diff = (cand - ref).abs()
        if diff.numel() == 0:
            return
        self.numel += int(diff.numel())
        self.abs_sum += float(diff.sum().item())
        self.diff_sq_sum += float((cand - ref).square().sum().item())
        self.ref_sq_sum += float(ref.square().sum().item())
        self.ref_abs_sum += float(ref.abs().sum().item())

    def update_router_ids(self, reference_ids, candidate_ids) -> None:  # type: ignore[no-untyped-def]
        """Adds router top-k id mismatch counts into the accumulator."""
        self.router_topk_total += int(reference_ids.numel())
        self.router_topk_mismatch += int((reference_ids != candidate_ids).sum().item())
        if reference_ids.ndim >= 2 and reference_ids.shape[1] > 0:
            self.router_top1_total += int(reference_ids.shape[0])
            self.router_top1_mismatch += int(
                (reference_ids[:, 0] != candidate_ids[:, 0]).sum().item()
            )

    def as_summary(self) -> dict[str, float]:
        """Returns normalized summary values for one row."""
        if self.numel == 0:
            topk_fraction = 0.0
            top1_fraction = 0.0
        else:
            topk_fraction = (
                self.router_topk_mismatch / self.router_topk_total
                if self.router_topk_total > 0
                else 0.0
            )
            top1_fraction = (
                self.router_top1_mismatch / self.router_top1_total
                if self.router_top1_total > 0
                else 0.0
            )
        if self.numel == 0:
            return {
                "numel": 0.0,
                "mean_abs_diff": 0.0,
                "relative_l2": 0.0,
                "typical_abs_scale": 0.0,
                "mean_abs_pct": 0.0,
                "topk_mismatch_fraction": topk_fraction,
                "top1_mismatch_fraction": top1_fraction,
            }
        mean_abs = self.abs_sum / self.numel
        typical_abs = self.ref_abs_sum / self.numel
        mean_abs_pct = (mean_abs / (typical_abs + 1e-12)) * 100.0
        return {
            "numel": _finite_metric(float(self.numel), default=0.0),
            "mean_abs_diff": _finite_metric(mean_abs),
            "relative_l2": _finite_metric(
                (self.diff_sq_sum**0.5) / max(self.ref_sq_sum**0.5, 1e-12)
            ),
            "typical_abs_scale": _finite_metric(typical_abs, default=0.0),
            "mean_abs_pct": _finite_metric(mean_abs_pct),
            "topk_mismatch_fraction": _finite_metric(topk_fraction, default=1.0),
            "top1_mismatch_fraction": _finite_metric(top1_fraction, default=1.0),
        }


T = TypeVar("T")


def _require_not_none(value: T | None, name: str) -> T:
    if value is None:
        raise RuntimeError(f"{name} is None")
    return value


TOPOLOGIES = [
    Topology(tp=1, ep=1, etp=1, dp=1, sp=False),
    Topology(tp=2, ep=1, etp=1, dp=1, sp=True),
    Topology(tp=1, ep=2, etp=1, dp=2, sp=False),
    Topology(tp=2, ep=2, etp=1, dp=2, sp=True),
]
EXTENDED_TOPOLOGIES = [
    Topology(tp=1, ep=1, etp=1, dp=2, sp=False),
    Topology(tp=2, ep=1, etp=1, dp=2, sp=True),
]
ORACLE_TOPOLOGY = TOPOLOGIES[0]
SENSITIVITY_TOPOLOGY = TOPOLOGIES[1]


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def sensitivity_mutations() -> list[SensitivityMutation]:
    """Parses sensitivity mutation selectors from env as a CSV list."""
    raw = os.environ.get(SENSITIVITY_MUTATION_ENV)
    if raw is None or raw.strip() == "":
        return []
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return [DEFAULT_SENSITIVITY_MUTATION]
    mutations = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unsupported = [
        mutation
        for mutation in mutations
        if mutation not in SUPPORTED_SENSITIVITY_MUTATIONS
    ]
    if not unsupported:
        return mutations
    supported = ", ".join(SUPPORTED_SENSITIVITY_MUTATIONS)
    raise ValueError(
        f"Unsupported {SENSITIVITY_MUTATION_ENV} value '{raw}'. "
        f"Supported values: {supported}, CSV of supported values, 1/true/yes/on."
    )


def sensitivity_enabled() -> bool:
    return bool(sensitivity_mutations())


def extended_topologies_enabled() -> bool:
    """Returns whether extended topologies are enabled for the suite."""
    return _truthy(os.environ.get(EXTENDED_TOPOLOGIES_ENV))


def regenerate_requested() -> bool:
    return _truthy(os.environ.get(REGENERATE_ENV))


def case_config(
    base_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
) -> OracleCaseConfig:
    """Builds the deterministic default oracle case config."""
    return OracleCaseConfig(base_model=base_model)


def available_gpu_count() -> int:
    import torch

    return int(torch.cuda.device_count())


def stable_case_id(case_config: OracleCaseConfig) -> str:
    """Builds a deterministic case id from case config contents."""
    payload = case_config.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    model_tag = (
        case_config.base_model.replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )
    return f"{model_tag}_{digest}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_packed_tensors(
    config: PackedTensorConfig,
    seed: int,
) -> dict[str, Any]:
    """Generates deterministic synthetic packed tensors used in integration runs."""
    import torch

    if config.num_sequences <= 1:
        raise ValueError("num_sequences must be greater than 1")
    shape = (config.num_sequences, config.sequence_length)
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        low=10,
        high=config.vocab_high,
        size=shape,
        dtype=torch.long,
        generator=generator,
    )
    group_ids = torch.zeros(shape, dtype=torch.long)
    parent_ids = torch.full(shape, -1, dtype=torch.long)
    input_pos = (
        torch.arange(config.sequence_length, dtype=torch.long)
        .unsqueeze(0)
        .expand(config.num_sequences, -1)
        .clone()
    )
    prefix_length = max(1, min(config.sequence_length - 1, config.prefill_tokens))
    decode_span = max(1, config.decode_tokens)
    cursor = prefix_length
    branch = 1
    while cursor < config.sequence_length:
        end = min(config.sequence_length, cursor + decode_span)
        group_ids[:, cursor:end] = branch
        parent_ids[:, cursor:end] = 0
        cursor = end
        branch += 1
    assistant_mask = torch.zeros(shape, dtype=torch.bool)
    assistant_mask[:, prefix_length:] = True
    logprobs = (
        torch.randn(
            shape,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.25
        - 1.75
    )
    advantages = (
        torch.randn(
            shape,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.1
        + 1.0
    )
    weights = torch.ones(shape, dtype=torch.float32)
    return {
        "tokens": tokens,
        "group_ids": group_ids,
        "parent_ids": parent_ids,
        "input_pos": input_pos,
        "assistant_mask": assistant_mask,
        "logprobs": logprobs,
        "advantages": advantages,
        "weights": weights,
        "pixel_values": [None] * config.num_sequences,
        "image_grid_thw": [None] * config.num_sequences,
    }


def _create_packed_tensors(
    case_config: OracleCaseConfig,
    packed_dir: Path,
) -> DiskPackedTensorsSpec:
    """Persists packed tensors to disk and returns their descriptor."""
    from art.preprocessing.pack import PackedTensors, packed_tensors_to_dir

    packed_tensors = cast(
        PackedTensors,
        _build_packed_tensors(case_config.packed_tensors, case_config.seed),
    )
    descriptor = packed_tensors_to_dir(packed_tensors, str(packed_dir))
    return DiskPackedTensorsSpec.model_validate(descriptor)


def ensure_case_artifacts(case_config: OracleCaseConfig) -> CaseArtifacts:
    """Ensures stable case-level artifacts (input tensors) are present and reusable."""
    case_id = stable_case_id(case_config)
    case_dir = ARTIFACT_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_json(case_dir / "case_config.json", case_config.model_dump(mode="json"))

    descriptor_path = case_dir / "packed_tensors.json"
    if descriptor_path.exists():
        packed_spec = DiskPackedTensorsSpec.model_validate(_read_json(descriptor_path))
    else:
        packed_spec = _create_packed_tensors(case_config, case_dir / "packed_tensors")
        _write_json(descriptor_path, packed_spec.model_dump(mode="json"))

    shared_init_path = case_dir / "shared_init" / "adapter_model.safetensors"
    shared_init_path.parent.mkdir(parents=True, exist_ok=True)
    return CaseArtifacts(
        case_id=case_id,
        case_dir=str(case_dir),
        packed_tensors=packed_spec,
        shared_init_adapter_path=str(shared_init_path),
    )


def _replace_topology_dir(path: Path) -> None:
    """Resets one topology output directory before regeneration."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "traces").mkdir(parents=True, exist_ok=True)


def _topology_output_slug(
    topology: Topology,
    mutation: SensitivityMutation | None = None,
) -> str:
    """Builds output slug for a topology and optional mutation variant."""
    return topology.slug() if mutation is None else f"{topology.slug()}__{mutation}"


def _load_manifest(topology_dir: Path) -> RunManifest:
    """Loads one run manifest for a topology output directory."""
    manifest_path = topology_dir / "manifest.json"
    return RunManifest.model_validate(_read_json(manifest_path))


def _load_output_tensor(topology_dir: Path, step: StepTrace):
    """Loads one output trace tensor referenced by a step trace entry."""
    import torch

    path = topology_dir / step.output_file
    return torch.load(path, map_location="cpu")


def _load_safetensor_map(path: Path) -> dict[str, Any]:
    """Loads one safetensor map from disk."""
    from safetensors.torch import load_file

    return load_file(str(path))


def _align_sequence_parallel(reference, candidate):  # type: ignore[no-untyped-def]
    """Aligns sequence-parallel-shaped tensors so diff computation is topology-agnostic."""
    if reference.shape == candidate.shape:
        return candidate
    if (
        candidate.ndim == reference.ndim + 1
        and candidate.shape[0] * candidate.shape[1] == reference.shape[0]
        and tuple(candidate.shape[2:]) == tuple(reference.shape[1:])
    ):
        return candidate.reshape(reference.shape)
    return None


def _is_moe_base_forward_param(name: str) -> bool:
    """Returns whether this forward param is a base MoE expert internal tensor."""
    if ".mlp.experts." not in name:
        return False
    if any(token in name for token in (".router", ".gate_lora", ".up_lora", ".lora")):
        return False
    return ".linear_fc1" in name or ".linear_fc2" in name


def _lookup_call_by_index(
    trace: dict[str, list[dict[str, Any]]],
    module_name: str,
    call_index: int,
) -> dict[str, Any] | None:
    calls = trace.get(module_name)
    if calls is None:
        return None
    for call in calls:
        if int(call.get("call_index", -1)) == call_index:
            return call
    if 0 <= call_index < len(calls):
        return calls[call_index]
    return None


def _router_module_name_for_expert_module(module_name: str) -> str | None:
    if ".mlp.experts.linear_fc1" in module_name:
        return module_name.replace(".mlp.experts.linear_fc1", ".mlp.router")
    if ".mlp.experts.linear_fc2" in module_name:
        return module_name.replace(".mlp.experts.linear_fc2", ".mlp.router")
    return None


def _build_moe_row_identities(
    *,
    module_name: str,
    call_index: int,
    trace: dict[str, list[dict[str, Any]]],
    row_splits: list[int] | None,
) -> list[tuple[int, int, int]] | None:
    router_module_name = _router_module_name_for_expert_module(module_name)
    if router_module_name is None:
        return None
    router_call = _lookup_call_by_index(trace, router_module_name, call_index)
    if router_call is None:
        return None
    router_topk_ids = router_call.get("router_topk_ids")
    if not isinstance(router_topk_ids, torch.Tensor) or router_topk_ids.ndim != 2:
        return None
    token_splits_raw = router_call.get("router_topk_ids__row_splits")
    if row_splits is None:
        if isinstance(token_splits_raw, list):
            row_splits = [
                int(v) * int(router_topk_ids.shape[1]) for v in token_splits_raw
            ]
        else:
            row_splits = [int(router_topk_ids.numel())]
    if isinstance(token_splits_raw, list):
        token_splits = [int(v) for v in token_splits_raw]
    else:
        topk = int(router_topk_ids.shape[1])
        token_splits = [int(v) // topk for v in row_splits]
    if len(row_splits) != len(token_splits):
        return None
    row_cursor = 0
    token_cursor = 0
    identities: list[tuple[int, int, int]] = []
    for row_count, token_count in zip(row_splits, token_splits):
        local_ids = router_topk_ids[token_cursor : token_cursor + token_count]
        token_cursor += token_count
        local_identities: list[tuple[int, int, int]] = []
        max_expert = int(local_ids.max().item()) if local_ids.numel() > 0 else -1
        for expert_id in range(max_expert + 1):
            expert_rows = (local_ids == expert_id).nonzero(as_tuple=False)
            for token_offset, slot_index in expert_rows.tolist():
                local_identities.append(
                    (expert_id, token_cursor - token_count + token_offset, slot_index)
                )
        if len(local_identities) != row_count:
            return None
        identities.extend(local_identities)
        row_cursor += row_count
    if row_cursor != sum(row_splits):
        return None
    return identities


def _canonicalize_moe_base_forward_tensor(
    *,
    module_name: str,
    call_index: int,
    tensor: torch.Tensor,
    trace: dict[str, list[dict[str, Any]]],
    call: dict[str, Any],
) -> torch.Tensor:
    if not _is_moe_base_forward_param(module_name):
        return tensor
    if tensor.ndim != 2:
        return tensor
    row_splits_raw = call.get("primary_output__row_splits")
    row_splits = (
        [int(v) for v in row_splits_raw] if isinstance(row_splits_raw, list) else None
    )
    identities = _build_moe_row_identities(
        module_name=module_name,
        call_index=call_index,
        trace=trace,
        row_splits=row_splits,
    )
    if identities is None or len(identities) != int(tensor.shape[0]):
        return tensor
    order = sorted(range(len(identities)), key=lambda index: identities[index])
    return tensor[order]


def _minimal_param_name(name: str) -> str:
    """Returns a shorter but 1:1 param/module identifier for report readability."""
    return name.removeprefix("base_model.model.model.").replace(
        "module.module.decoder.", ""
    )


def _load_forward_trace(
    topology_dir: Path, step_index: int
) -> dict[str, list[dict[str, Any]]]:
    """Loads one merged forward-trace file for a given step."""
    trace_path = topology_dir / "traces" / f"forward_trace_step_{step_index:03d}.pt"
    return ForwardTraceCapture.load_trace(trace_path)


def _threshold_string(thresholds: dict[str, float]) -> str:
    """Formats threshold dicts into compact table cells."""
    if not thresholds:
        return "-"
    return ", ".join(f"{key}<={value:.3g}" for key, value in sorted(thresholds.items()))


def _finite_metric(value: float, *, default: float = NON_FINITE_METRIC_VALUE) -> float:
    """Maps NaN/Inf metric values to a large finite sentinel for JSON-safe reports."""
    value_f = float(value)
    if math.isnan(value_f):
        return default
    if math.isinf(value_f):
        return default if value_f > 0 else -default
    return value_f


def _triplet_expert_key(param: str) -> tuple[int, int] | None:
    """Returns (layer, expert_id) for expert up/gate/down params."""
    match = EXPERT_TRIPLET_PARAM_RE.search(param)
    if match is None:
        return None
    return int(match.group("layer")), int(match.group("expert"))


class VariantRunner:
    """Runs oracle/candidate variants and emits row-level comparison reports."""

    def __init__(
        self,
        *,
        case_config: OracleCaseConfig,
        console: Console | None = None,
    ) -> None:
        self.case_config = case_config
        self.case_artifacts = ensure_case_artifacts(case_config)
        self.case_id = self.case_artifacts.case_id
        self.case_dir = Path(self.case_artifacts.case_dir)
        self.oracle_slug = ORACLE_TOPOLOGY.slug()
        self.oracle_dir = self.case_dir / self.oracle_slug
        self.oracle_routing_bundle_dir = (
            self.case_dir / ORACLE_MOE_ROUTING_BUNDLE_DIRNAME
        )
        self.shared_init_path = Path(self.case_artifacts.shared_init_adapter_path)
        self.console = console or Console(width=160)
        self._oracle_initialized = False
        self._oracle_regenerated = False

    def _run_topology(
        self,
        *,
        topology: Topology,
        output_slug: str,
        mutation: SensitivityMutation | None,
        replay_bundle_dir: Path | None,
        capture_bundle_dir: Path | None,
        regenerate: bool,
    ) -> Path:
        """Executes one topology worker run and returns its output directory."""
        topology_dir = self.case_dir / output_slug
        manifest_path = topology_dir / "manifest.json"
        if manifest_path.exists() and not regenerate:
            return topology_dir
        _replace_topology_dir(topology_dir)
        request = WorkerRunRequest(
            case_id=self.case_id,
            case_config=self.case_config,
            topology=topology,
            topology_dir=str(topology_dir),
            packed_tensors=self.case_artifacts.packed_tensors,
            shared_init_adapter_path=str(self.shared_init_path),
            mutation=mutation,
            moe_routing_replay_path=(
                None if replay_bundle_dir is None else str(replay_bundle_dir)
            ),
            moe_routing_replay_strict=True,
            capture_moe_routing_bundle_path=(
                None if capture_bundle_dir is None else str(capture_bundle_dir)
            ),
        )
        from .megatron_oracle_worker import run_worker_subprocess

        run_worker_subprocess(request, topology_dir, repo_root=REPO_ROOT)
        return topology_dir

    def ensure_oracle(self) -> Path:
        """Ensures oracle capture and canonical replay artifacts exist exactly once per session."""
        regenerate = regenerate_requested()
        if self._oracle_initialized and (not regenerate or self._oracle_regenerated):
            return self.oracle_dir
        if regenerate and self.shared_init_path.exists():
            self.shared_init_path.unlink()
        bundle_manifest = self.oracle_routing_bundle_dir / "manifest.json"
        oracle_manifest = self.oracle_dir / "manifest.json"
        need_capture = (
            regenerate
            or not bundle_manifest.exists()
            or not self.shared_init_path.exists()
        )
        run_oracle_topology = partial(
            self._run_topology,
            topology=ORACLE_TOPOLOGY,
            mutation=None,
            regenerate=True,
        )
        if need_capture:
            run_oracle_topology(
                output_slug=f"{self.oracle_slug}__oracle_capture",
                replay_bundle_dir=None,
                capture_bundle_dir=self.oracle_routing_bundle_dir,
            )
        if regenerate or not oracle_manifest.exists():
            run_oracle_topology(
                output_slug=self.oracle_slug,
                replay_bundle_dir=self.oracle_routing_bundle_dir,
                capture_bundle_dir=None,
            )
        self._oracle_initialized = True
        self._oracle_regenerated = self._oracle_regenerated or regenerate
        return self.oracle_dir

    def ensure_variant_artifacts(
        self,
        variant: VariantSpec,
    ) -> Path:
        """Ensures oracle prerequisites and candidate artifacts for one variant."""
        self.ensure_oracle()
        output_slug = variant.resolved_output_slug()
        if output_slug == self.oracle_slug and variant.mutation is None:
            return self.oracle_dir
        return self._run_topology(
            topology=variant.topology,
            output_slug=output_slug,
            mutation=variant.mutation,
            replay_bundle_dir=self.oracle_routing_bundle_dir,
            capture_bundle_dir=None,
            regenerate=True,
        )

    @staticmethod
    def _apply_thresholds(row: MetricRow, thresholds: dict[str, float]) -> None:
        """Evaluates row thresholds using AND semantics over all configured keys."""
        row.thresholds = dict(thresholds)
        if not thresholds:
            row.pass_signal = True
            row.failure_reasons = []
            return
        payload = row.model_dump(mode="python")
        reasons: list[str] = []
        for key, limit in sorted(thresholds.items()):
            value = payload.get(key)
            if not isinstance(value, (int, float)):
                reasons.append(f"{key}=missing")
                continue
            if float(value) > float(limit):
                reasons.append(f"{key}={float(value):.6g}>{float(limit):.6g}")
        row.pass_signal = len(reasons) == 0
        row.failure_reasons = reasons

    @staticmethod
    def _inf_summary() -> dict[str, float]:
        """Builds a large-error finite summary for structural mismatches."""
        return {
            "numel": 0.0,
            "mean_abs_diff": NON_FINITE_METRIC_VALUE,
            "relative_l2": NON_FINITE_METRIC_VALUE,
            "typical_abs_scale": 0.0,
            "mean_abs_pct": NON_FINITE_METRIC_VALUE,
            "topk_mismatch_fraction": 1.0,
            "top1_mismatch_fraction": 1.0,
        }

    def _build_metric_row(
        self,
        *,
        variant: VariantSpec,
        step_index: int,
        phase: str,
        param: str,
        summary: dict[str, float],
        structural_failure: str | None = None,
    ) -> MetricRow:
        """Builds one metric row and applies per-phase thresholds."""
        row = MetricRow(
            case_id=self.case_id,
            variant=variant.name,
            topology=variant.resolved_output_slug(),
            oracle_topology=variant.resolved_reference_slug(),
            step_index=step_index,
            phase=phase,
            param=param,
            numel=summary["numel"],
            mean_abs_diff=summary["mean_abs_diff"],
            relative_l2=summary["relative_l2"],
            typical_abs_scale=summary["typical_abs_scale"],
            mean_abs_pct=summary["mean_abs_pct"],
            topk_mismatch_fraction=summary.get("topk_mismatch_fraction"),
            top1_mismatch_fraction=summary.get("top1_mismatch_fraction"),
        )
        self._apply_thresholds(row, variant.thresholds_by_phase.get(phase, {}))
        if structural_failure is not None:
            row.pass_signal = False
            row.failure_reasons = [structural_failure, *row.failure_reasons]
        return row

    def _build_metric_rows_from_tensor_pairs(
        self,
        *,
        variant: VariantSpec,
        step_index: int,
        phase: str,
        pairs: list[tuple[str, Any, Any]],
        router_ids: bool = False,
    ) -> list[MetricRow]:
        """Builds rows from named tensor pairs with one shared diff path."""
        rows: list[MetricRow] = []
        for name, reference, candidate in pairs:
            param_name = _minimal_param_name(name)
            reference_aligned = reference
            candidate_aligned = candidate
            aligned_candidate = _align_sequence_parallel(
                reference_aligned, candidate_aligned
            )
            if aligned_candidate is None:
                rows.append(
                    self._build_metric_row(
                        variant=variant,
                        step_index=step_index,
                        phase=phase,
                        param=param_name,
                        summary=self._inf_summary(),
                        structural_failure="shape mismatch",
                    )
                )
                continue
            accumulator = DiffAccumulator()
            if router_ids:
                accumulator.update_router_ids(reference_aligned, aligned_candidate)
            else:
                accumulator.update(reference_aligned, aligned_candidate)
            rows.append(
                self._build_metric_row(
                    variant=variant,
                    step_index=step_index,
                    phase=phase,
                    param=param_name,
                    summary=accumulator.as_summary(),
                )
            )
        return rows

    def _check_matching_keys(
        self,
        reference: dict[str, Any],
        candidate: dict[str, Any],
        variant: VariantSpec,
        step_index: int,
        phase: str,
    ) -> tuple[bool, list[MetricRow] | None]:
        """Checks if the keys of two tensor maps match and builds a metric row if they don't."""
        reference_keys = set(reference.keys())
        candidate_keys = set(candidate.keys())
        if reference_keys != candidate_keys:
            missing = sorted(reference_keys - candidate_keys)
            extra = sorted(candidate_keys - reference_keys)
            return False, [
                self._build_metric_row(
                    variant=variant,
                    step_index=step_index,
                    phase=phase,
                    param="__keys__",
                    summary=self._inf_summary(),
                    structural_failure=f"missing={missing[:5]} extra={extra[:5]}",
                )
            ]
        return True, None

    def _build_metric_rows_from_tensor_maps(
        self,
        *,
        variant: VariantSpec,
        step_index: int,
        phase: str,
        reference: dict[str, Any],
        candidate: dict[str, Any],
        router_ids: bool = False,
    ) -> list[MetricRow]:
        """Builds rows from two keyed tensor maps through a unified compare path."""
        matching, rows = self._check_matching_keys(
            reference, candidate, variant, step_index, phase
        )
        if not matching:
            return rows if rows is not None else []
        pairs = [
            (key, reference[key], candidate[key])
            for key in sorted(set(reference.keys()))
        ]
        return self._build_metric_rows_from_tensor_pairs(
            variant=variant,
            step_index=step_index,
            phase=phase,
            pairs=pairs,
            router_ids=router_ids,
        )

    @staticmethod
    def _flatten_forward_trace_tensors(
        trace: dict[str, list[dict[str, Any]]],
        *,
        value_key: str,
    ) -> dict[str, Any]:
        """Flattens per-module forward trace calls into a deterministic tensor map."""
        flattened: dict[str, Any] = {}
        for module_name in sorted(trace.keys()):
            for call_offset, call in enumerate(trace[module_name]):
                tensor = call.get(value_key)
                if tensor is None:
                    continue
                call_index = call.get("call_index", call_offset)
                if value_key == "primary_output" and isinstance(tensor, torch.Tensor):
                    tensor = _canonicalize_moe_base_forward_tensor(
                        module_name=module_name,
                        call_index=int(call_index),
                        tensor=tensor,
                        trace=trace,
                        call=call,
                    )
                flattened[f"{module_name}.call_{call_index}"] = tensor
        return flattened

    @staticmethod
    def _build_step_summaries(rows: list[MetricRow]) -> dict[int, dict[str, Any]]:
        """Builds step-indexed payloads directly from row model dumps."""
        step_summaries: dict[int, dict[str, Any]] = {}
        for row in rows:
            step_entry = step_summaries.setdefault(row.step_index, {})
            phase_entry = cast(dict[str, Any], step_entry.setdefault(row.phase, {}))
            phase_entry[row.param] = row.model_dump(mode="json")
        return step_summaries

    def compare_variant(self, variant: VariantSpec) -> VariantReport:
        """Compares one candidate variant against its reference topology."""
        reference_slug = variant.resolved_reference_slug()
        topology_slug = variant.resolved_output_slug()
        reference_dir = self.case_dir / reference_slug
        topology_dir = self.case_dir / topology_slug
        reference_manifest = _load_manifest(reference_dir)
        topology_manifest = _load_manifest(topology_dir)
        rows: list[MetricRow] = []
        if len(reference_manifest.steps) != len(topology_manifest.steps):
            rows.append(
                self._build_metric_row(
                    variant=variant,
                    step_index=0,
                    phase="step_count",
                    param="__step_count__",
                    summary=self._inf_summary(),
                    structural_failure=(
                        f"reference={len(reference_manifest.steps)} "
                        f"candidate={len(topology_manifest.steps)}"
                    ),
                )
            )

        import torch

        for reference_step, topology_step in zip(
            reference_manifest.steps, topology_manifest.steps
        ):
            step_index = reference_step.step_index
            reference_trace = _load_forward_trace(reference_dir, step_index)
            topology_trace = _load_forward_trace(topology_dir, step_index)
            map_phase_inputs = [
                (
                    "outputs",
                    {"logprobs": _load_output_tensor(reference_dir, reference_step)},
                    {"logprobs": _load_output_tensor(topology_dir, topology_step)},
                    False,
                ),
                (
                    "losses",
                    {"loss": torch.tensor([reference_step.loss], dtype=torch.float32)},
                    {"loss": torch.tensor([topology_step.loss], dtype=torch.float32)},
                    False,
                ),
                (
                    "grads",
                    _load_safetensor_map(reference_dir / reference_step.grads_file),
                    _load_safetensor_map(topology_dir / topology_step.grads_file),
                    False,
                ),
                (
                    "deltas",
                    _load_safetensor_map(reference_dir / reference_step.deltas_file),
                    _load_safetensor_map(topology_dir / topology_step.deltas_file),
                    False,
                ),
                *[
                    (
                        phase,
                        self._flatten_forward_trace_tensors(
                            reference_trace,
                            value_key=value_key,
                        ),
                        self._flatten_forward_trace_tensors(
                            topology_trace,
                            value_key=value_key,
                        ),
                        phase == "router_topk_ids",
                    )
                    for phase, value_key in (
                        ("forward", "primary_output"),
                        ("router_scores", "router_topk_scores"),
                        ("router_topk_ids", "router_topk_ids"),
                    )
                ],
            ]
            for phase, reference_map, candidate_map, router_ids in map_phase_inputs:
                rows.extend(
                    self._build_metric_rows_from_tensor_maps(
                        variant=variant,
                        step_index=step_index,
                        phase=phase,
                        reference=reference_map,
                        candidate=candidate_map,
                        router_ids=router_ids,
                    )
                )
        pass_count = sum(1 for row in rows if row.pass_signal)
        fail_count = len(rows) - pass_count
        signal: Literal["pass", "fail"] = "pass" if fail_count == 0 else "fail"
        return VariantReport(
            case_id=self.case_id,
            variant=variant.name,
            topology=topology_slug,
            reference_topology=reference_slug,
            expected_signal=variant.expected_signal,
            signal=signal,
            pass_count=pass_count,
            fail_count=fail_count,
            step_summaries=self._build_step_summaries(rows),
            metrics=rows,
        )

    @staticmethod
    def assert_expected_signal(report: VariantReport, context: str) -> None:
        """Raises when observed run signal diverges from variant expectation."""
        if report.signal == report.expected_signal:
            return
        if report.signal == "fail":
            first_failure = next(row for row in report.metrics if not row.pass_signal)
            raise AssertionError(
                f"{context}: topology={report.topology} phase={first_failure.phase} "
                f"step={first_failure.step_index} param={first_failure.param} "
                f"reasons={'; '.join(first_failure.failure_reasons)}"
            )
        raise AssertionError(
            f"{context}: expected_signal={report.expected_signal} "
            f"observed_signal={report.signal} topology={report.topology}"
        )

    def _write_variant_report(self, topology_dir: Path, report: VariantReport) -> None:
        """Persists full variant report JSON for debugging and regression inspection."""
        _write_json(
            topology_dir / "variant_report.json", report.model_dump(mode="json")
        )

    def print_report(self, report: VariantReport) -> None:
        """Prints a row-level table with expert rows subsampled by highest relative_l2."""
        non_expert_rows: list[MetricRow] = []
        triplet_rows: list[tuple[tuple[int, int], MetricRow]] = []
        for row in report.metrics:
            expert_key = _triplet_expert_key(row.param)
            if expert_key is None:
                non_expert_rows.append(row)
                continue
            triplet_rows.append((expert_key, row))

        scores_by_layer: dict[int, dict[int, float]] = {}
        for (layer, expert_id), row in triplet_rows:
            layer_scores = scores_by_layer.setdefault(layer, {})
            layer_scores[expert_id] = max(
                layer_scores.get(expert_id, float("-inf")), row.relative_l2
            )

        selected_experts: set[tuple[int, int]] = set()
        for layer, expert_scores in scores_by_layer.items():
            top_experts = sorted(
                expert_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:EXPERT_TABLE_ROW_LIMIT]
            for expert_id, _score in top_experts:
                selected_experts.add((layer, expert_id))

        selected_triplet_rows = [
            row for expert_key, row in triplet_rows if expert_key in selected_experts
        ]
        table_rows = non_expert_rows + selected_triplet_rows
        detail_table = Table(
            title=(
                f"Variant Report | variant={report.variant} "
                f"| topology={report.topology} | signal={report.signal} "
                f"| selected_experts={len(selected_experts)} "
                f"(top {EXPERT_TABLE_ROW_LIMIT} per layer)"
            ),
            box=box.SIMPLE_HEAVY,
            show_lines=False,
        )
        detail_table.add_column("Step", justify="right")
        detail_table.add_column("Phase", style="cyan")
        detail_table.add_column("Param")
        detail_table.add_column("Status")
        detail_table.add_column("relative_l2", justify="right")
        detail_table.add_column("mean_abs_pct", justify="right")
        detail_table.add_column("typical_abs", justify="right")
        # detail_table.add_column("Thresholds")
        detail_table.add_column("Failure")
        sorted_rows = sorted(
            table_rows,
            key=lambda row: (
                row.step_index,
                PHASE_PRINT_ORDER.get(row.phase, 999),
                row.param,
                row.pass_signal,
            ),
        )
        for row in sorted_rows:
            status_text = (
                "[green]PASS[/green]" if row.pass_signal else "[red]FAIL[/red]"
            )
            failure_text = "" if row.pass_signal else "; ".join(row.failure_reasons)
            detail_table.add_row(
                str(row.step_index),
                row.phase,
                row.param,
                status_text,
                f"{row.relative_l2:.6g}",
                f"{row.mean_abs_pct:.6g}%",
                f"{row.typical_abs_scale:.6g}",
                # _threshold_string(row.thresholds),  # disabled for now to avoid clutter, neat to keep though
                failure_text,
            )
        self.console.print(detail_table)

    def run_variant(
        self,
        variant: VariantSpec,
    ) -> VariantReport:
        """Runs a variant end-to-end, writes JSON report, and prints row table."""
        topology_dir = self.ensure_variant_artifacts(variant)
        report = self.compare_variant(variant)
        self._write_variant_report(topology_dir, report)
        self.print_report(report)
        return report

    def run_suite(
        self,
        variants: list[VariantSpec],
    ) -> list[VariantReport]:
        """Runs variants in order and stops at the first unexpected signal."""
        reports: list[VariantReport] = []
        for variant in variants:
            report = self.run_variant(variant)
            reports.append(report)
            self.assert_expected_signal(report, "Megatron oracle suite mismatch")
        return reports


def _default_phase_thresholds(
    case_cfg: OracleCaseConfig,
) -> dict[str, dict[str, float]]:
    """Builds default per-phase (fwd, grad, outputs, losses, deltas) threshold dictionaries."""
    default = {
        "relative_l2": case_cfg.tolerances.relative_l2,
        "mean_abs_pct": case_cfg.tolerances.mean_abs_pct,
    }
    return {
        key: default for key in ["outputs", "losses", "grads", "deltas", "forward"]
    } | {
        "router_scores": {"mean_abs_pct": 0.0},
        "router_topk_ids": {
            "topk_mismatch_fraction": 0.0,
            "top1_mismatch_fraction": 0.0,
        },
    }


def _suite_variants(case_cfg: OracleCaseConfig) -> list[VariantSpec]:
    """Builds the standard oracle suite variant ordering."""
    thresholds = _default_phase_thresholds(case_cfg)
    variants = [
        VariantSpec(
            name="oracle_replay_parity",
            topology=ORACLE_TOPOLOGY,
            output_slug=_topology_output_slug(
                ORACLE_TOPOLOGY, ORACLE_REPLAY_TOPOLOGY_SUFFIX
            ),
            thresholds_by_phase=thresholds,
        )
    ]
    for topology in TOPOLOGIES[1:] + (
        EXTENDED_TOPOLOGIES if extended_topologies_enabled() else []
    ):
        variants.append(
            VariantSpec(
                name=f"topology_{topology.slug()}",
                topology=topology,
                thresholds_by_phase=thresholds,
            )
        )
    return variants


def run_suite(
    *,
    case_config: OracleCaseConfig,
) -> list[VariantReport]:
    """Runs replay parity and topology variants with fail-fast assertions."""
    runner = VariantRunner(case_config=case_config)
    return runner.run_suite(_suite_variants(case_config))


def run_sensitivity_suite(
    *,
    case_config: OracleCaseConfig,
    mutations: list[SensitivityMutation],
) -> list[VariantReport]:
    """Runs a list of sensitivity mutations and expects each to fail."""
    runner = VariantRunner(case_config=case_config)
    thresholds = _default_phase_thresholds(case_config)
    variants = [
        VariantSpec(
            name=f"sensitivity_{mutation}",
            topology=SENSITIVITY_TOPOLOGY,
            mutation=mutation,
            expected_signal="fail",
            thresholds_by_phase=thresholds,
        )
        for mutation in mutations
    ]
    return runner.run_suite(variants)
