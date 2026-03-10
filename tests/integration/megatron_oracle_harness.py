from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Any, Callable, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path(REPO_ROOT / ".local/megatron_lora_oracles")

REGENERATE_ENV = "ART_REGENERATE_MEGATRON_ORACLE"
BASE_MODEL_ENV = "ART_MEGATRON_ORACLE_BASE_MODEL"
DP_SUPPORT_ENV = "ART_MEGATRON_ORACLE_ENABLE_DP_PHASE_B"
SENSITIVITY_MUTATION_ENV = "ART_MEGATRON_ORACLE_MUTATION"

SensitivityMutation = Literal["drop_finalize"]

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


class Topology(BaseModel):
    model_config = ConfigDict(frozen=True)

    tp: int
    ep: int
    etp: int = 1
    dp: int = 1
    sp: int = 0
    phase: Literal["A", "B"] = "A"

    def slug(self) -> str:
        return f"tp{self.tp}_ep{self.ep}_etp{self.etp}_dp{self.dp}_sp{self.sp}"

    def world_size(self) -> int:
        return self.tp * self.ep * self.etp * self.dp


class PackedTensorConfig(BaseModel):
    num_sequences: int = 8
    sequence_length: int = 256
    prefill_tokens: int = 64
    decode_tokens: int = 64
    vocab_high: int = 8192


class LoraConfig(BaseModel):
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
    outputs_abs: float = 1e-2
    outputs_rel: float = 1e-2
    losses_abs: float = 1e-4
    losses_rel: float = 1e-4
    grads_abs: float = 1e-2
    grads_rel: float = 1e-2
    deltas_abs: float = 1e-2
    deltas_rel: float = 1e-2


class OracleCaseConfig(BaseModel):
    base_model: str
    seed: int = 20260305
    num_steps: int = 3
    learning_rate: float = 5e-6
    beta: float = 0.0
    packed_tensors: PackedTensorConfig = Field(default_factory=PackedTensorConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    tolerances: ToleranceProfile = Field(default_factory=ToleranceProfile)


class DiskPackedTensorsSpec(BaseModel):
    dir: str
    num_sequences: int
    sequence_length: int
    pixel_values: tuple[int, list[int]] | None = None
    image_grid_thw: tuple[int, list[int]] | None = None


class CaseArtifacts(BaseModel):
    case_id: str
    case_dir: str
    packed_tensors: DiskPackedTensorsSpec
    shared_init_adapter_path: str


class WorkerRunRequest(BaseModel):
    case_id: str
    case_config: OracleCaseConfig
    topology: Topology
    topology_dir: str
    packed_tensors: DiskPackedTensorsSpec
    shared_init_adapter_path: str
    allow_create_shared_init: bool = False
    mutation: SensitivityMutation | None = None


class StepTrace(BaseModel):
    step_index: int
    loss: float
    probs_corr: float
    output_file: str
    grads_file: str
    deltas_file: str
    lora_file: str


class RunManifest(BaseModel):
    case_id: str
    base_model: str
    topology: str
    world_size: int
    seed: int
    num_steps: int
    packed_tensors: DiskPackedTensorsSpec
    tolerances: ToleranceProfile
    steps: list[StepTrace]


class ComparisonFailure(BaseModel):
    case_id: str
    topology: str
    oracle_topology: str
    metric: Literal["outputs", "losses", "grads", "lora_deltas"]
    step_index: int
    key: str
    max_abs_error: float
    max_rel_error: float
    abs_tolerance: float
    rel_tolerance: float
    message: str


PHASE_A_TOPOLOGIES = [
    Topology(tp=1, ep=1, etp=1, dp=1, sp=0, phase="A"),
    Topology(tp=2, ep=1, etp=1, dp=1, sp=1, phase="A"),
    Topology(tp=1, ep=2, etp=1, dp=1, sp=0, phase="A"),
    Topology(tp=2, ep=2, etp=1, dp=1, sp=1, phase="A"),
]
PHASE_B_TOPOLOGIES = [
    Topology(tp=1, ep=1, etp=1, dp=2, sp=0, phase="B"),
    Topology(tp=2, ep=1, etp=1, dp=2, sp=1, phase="B"),
]
ORACLE_TOPOLOGY = PHASE_A_TOPOLOGIES[0]
SENSITIVITY_TOPOLOGY = PHASE_A_TOPOLOGIES[1]


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def sensitivity_mutation() -> SensitivityMutation | None:
    raw = os.environ.get(SENSITIVITY_MUTATION_ENV)
    if raw is None or raw.strip() == "":
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return "drop_finalize"
    if normalized == "drop_finalize":
        return "drop_finalize"
    raise ValueError(
        f"Unsupported {SENSITIVITY_MUTATION_ENV} value '{raw}'. "
        "Supported values: drop_finalize, 1/true/yes/on."
    )


def sensitivity_enabled() -> bool:
    return sensitivity_mutation() is not None


def phase_b_dp_enabled() -> bool:
    return _truthy(os.environ.get(DP_SUPPORT_ENV))


def regenerate_requested() -> bool:
    return _truthy(os.environ.get(REGENERATE_ENV))


def default_case_config() -> OracleCaseConfig:
    def _env_float(name: str, default: str) -> float:
        return float(os.environ.get(name, default))

    tolerances = ToleranceProfile(
        outputs_abs=_env_float("ART_MEGATRON_ORACLE_OUTPUTS_ABS_TOL", "1e-2"),
        outputs_rel=_env_float("ART_MEGATRON_ORACLE_OUTPUTS_REL_TOL", "1e-2"),
        losses_abs=_env_float("ART_MEGATRON_ORACLE_LOSSES_ABS_TOL", "1e-4"),
        losses_rel=_env_float("ART_MEGATRON_ORACLE_LOSSES_REL_TOL", "1e-4"),
        grads_abs=_env_float("ART_MEGATRON_ORACLE_GRADS_ABS_TOL", "1e-2"),
        grads_rel=_env_float("ART_MEGATRON_ORACLE_GRADS_REL_TOL", "1e-2"),
        deltas_abs=_env_float("ART_MEGATRON_ORACLE_DELTAS_ABS_TOL", "1e-2"),
        deltas_rel=_env_float("ART_MEGATRON_ORACLE_DELTAS_REL_TOL", "1e-2"),
    )
    return OracleCaseConfig(
        base_model=os.environ.get(
            BASE_MODEL_ENV,
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
        ),
        seed=int(os.environ.get("ART_MEGATRON_ORACLE_SEED", "20260305")),
        num_steps=int(os.environ.get("ART_MEGATRON_ORACLE_NUM_STEPS", "3")),
        learning_rate=float(os.environ.get("ART_MEGATRON_ORACLE_LR", "5e-6")),
        beta=float(os.environ.get("ART_MEGATRON_ORACLE_BETA", "0.0")),
        tolerances=tolerances,
    )


def available_gpu_count() -> int:
    import torch

    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def stable_case_id(case_config: OracleCaseConfig) -> str:
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
        json.dump(payload, handle, indent=2, sort_keys=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_packed_tensors(
    config: PackedTensorConfig,
    seed: int,
) -> dict[str, Any]:
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
    from art.preprocessing.pack import PackedTensors, packed_tensors_to_dir

    packed_tensors = cast(
        PackedTensors,
        _build_packed_tensors(case_config.packed_tensors, case_config.seed),
    )
    descriptor = packed_tensors_to_dir(packed_tensors, str(packed_dir))
    return DiskPackedTensorsSpec.model_validate(descriptor)


def _validate_packed_tensor_files(spec: DiskPackedTensorsSpec) -> None:
    tensor_dir = Path(spec.dir)
    for filename in REQUIRED_PACKED_TENSOR_FILES:
        file_path = tensor_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing packed tensor file: {file_path}")


def ensure_case_artifacts(case_config: OracleCaseConfig) -> CaseArtifacts:
    case_id = stable_case_id(case_config)
    case_dir = ARTIFACT_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_json(case_dir / "case_config.json", case_config.model_dump(mode="json"))

    descriptor_path = case_dir / "packed_tensors.json"
    if descriptor_path.exists():
        packed_spec = DiskPackedTensorsSpec.model_validate(_read_json(descriptor_path))
        _validate_packed_tensor_files(packed_spec)
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
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "traces").mkdir(parents=True, exist_ok=True)


def _run_worker_subprocess(request: WorkerRunRequest, topology_dir: Path) -> None:
    request_path = topology_dir / "run_request.json"
    _write_json(request_path, request.model_dump(mode="json"))

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(request.topology.world_size()),
        str(Path(__file__).resolve()),
        "--worker-run",
        "--run-request",
        str(request_path),
    ]
    run = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    combined_output = f"{run.stdout}\n{run.stderr}".strip()
    (topology_dir / "worker.log").write_text(combined_output + "\n", encoding="utf-8")
    if run.returncode != 0:
        tail = "\n".join(combined_output.splitlines()[-80:])
        raise RuntimeError(
            f"Topology run failed for {request.topology.slug()} with exit code "
            f"{run.returncode}.\n{tail}"
        )


def ensure_topology_artifacts(
    case_config: OracleCaseConfig,
    topology: Topology,
    *,
    regenerate: bool = False,
    mutation: SensitivityMutation | None = None,
) -> Path:
    case_artifacts = ensure_case_artifacts(case_config)
    case_dir = Path(case_artifacts.case_dir)
    topology_dir = case_dir / topology.slug()
    manifest_path = topology_dir / "manifest.json"
    if manifest_path.exists() and not regenerate:
        return topology_dir

    _replace_topology_dir(topology_dir)
    shared_init_path = Path(case_artifacts.shared_init_adapter_path)
    allow_create_shared_init = topology.slug() == ORACLE_TOPOLOGY.slug()
    if not allow_create_shared_init and not shared_init_path.exists():
        ensure_topology_artifacts(
            case_config=case_config,
            topology=ORACLE_TOPOLOGY,
            regenerate=False,
            mutation=None,
        )
    if not allow_create_shared_init and not shared_init_path.exists():
        raise FileNotFoundError(
            f"Oracle shared adapter missing after oracle generation: {shared_init_path}"
        )
    if mutation is not None and topology.slug() == ORACLE_TOPOLOGY.slug():
        raise RuntimeError("Sensitivity mutation cannot be applied to oracle topology")

    request = WorkerRunRequest(
        case_id=case_artifacts.case_id,
        case_config=case_config,
        topology=topology,
        topology_dir=str(topology_dir),
        packed_tensors=case_artifacts.packed_tensors,
        shared_init_adapter_path=str(shared_init_path),
        allow_create_shared_init=allow_create_shared_init,
        mutation=mutation,
    )
    _run_worker_subprocess(request, topology_dir)
    if not manifest_path.exists():
        raise RuntimeError(f"Missing manifest after run: {manifest_path}")
    return topology_dir


def ensure_oracle_reference_artifacts(
    *,
    case_config: OracleCaseConfig,
    regenerate: bool = False,
) -> Path:
    return ensure_topology_artifacts(
        case_config=case_config,
        topology=ORACLE_TOPOLOGY,
        regenerate=regenerate,
        mutation=None,
    )


def _load_manifest(topology_dir: Path) -> RunManifest:
    manifest_path = topology_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing topology manifest: {manifest_path}")
    return RunManifest.model_validate(_read_json(manifest_path))


def _load_output_tensor(topology_dir: Path, step: StepTrace):
    import torch

    path = topology_dir / step.output_file
    if not path.exists():
        raise FileNotFoundError(f"Missing output trace: {path}")
    return torch.load(path, map_location="cpu")


def _load_safetensor_map(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    if not path.exists():
        raise FileNotFoundError(f"Missing safetensor trace: {path}")
    return load_file(str(path))


def _tensor_error(reference, candidate) -> tuple[float, float]:
    ref = reference.detach().float()
    cand = candidate.detach().float()
    if ref.shape != cand.shape:
        return float("inf"), float("inf")
    if ref.numel() == 0:
        return 0.0, 0.0
    diff = (cand - ref).abs()
    max_abs = float(diff.max().item())
    max_rel = float((diff / ref.abs().clamp_min(1e-12)).max().item())
    return max_abs, max_rel


def _build_failure(
    *,
    case_id: str,
    topology: str,
    metric: Literal["outputs", "losses", "grads", "lora_deltas"],
    step_index: int,
    key: str,
    max_abs_error: float,
    max_rel_error: float,
    abs_tolerance: float,
    rel_tolerance: float,
    message: str,
) -> ComparisonFailure:
    return ComparisonFailure(
        case_id=case_id,
        topology=topology,
        oracle_topology=ORACLE_TOPOLOGY.slug(),
        metric=metric,
        step_index=step_index,
        key=key,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
        message=message,
    )


def _compare_tensor_pair(
    *,
    case_id: str,
    topology: str,
    metric: Literal["outputs", "losses", "grads", "lora_deltas"],
    step_index: int,
    key: str,
    reference,
    candidate,
    abs_tolerance: float,
    rel_tolerance: float,
) -> ComparisonFailure | None:
    max_abs, max_rel = _tensor_error(reference, candidate)
    if max_abs <= abs_tolerance or max_rel <= rel_tolerance:
        return None
    return _build_failure(
        case_id=case_id,
        topology=topology,
        metric=metric,
        step_index=step_index,
        key=key,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
        message=f"{metric} mismatch at step {step_index}, key '{key}'",
    )


def _compare_tensor_maps(
    *,
    case_id: str,
    topology: str,
    metric: Literal["grads", "lora_deltas"],
    step_index: int,
    reference: dict[str, Any],
    candidate: dict[str, Any],
    abs_tolerance: float,
    rel_tolerance: float,
) -> ComparisonFailure | None:
    ref_keys = set(reference.keys())
    cand_keys = set(candidate.keys())
    if ref_keys != cand_keys:
        missing = sorted(ref_keys - cand_keys)
        extra = sorted(cand_keys - ref_keys)
        return _build_failure(
            case_id=case_id,
            topology=topology,
            metric=metric,
            step_index=step_index,
            key="__keys__",
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            abs_tolerance=abs_tolerance,
            rel_tolerance=rel_tolerance,
            message=(
                f"{metric} key mismatch at step {step_index}; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            ),
        )
    for key in sorted(ref_keys):
        failure = _compare_tensor_pair(
            case_id=case_id,
            topology=topology,
            metric=metric,
            step_index=step_index,
            key=key,
            reference=reference[key],
            candidate=candidate[key],
            abs_tolerance=abs_tolerance,
            rel_tolerance=rel_tolerance,
        )
        if failure is not None:
            return failure
    return None


def _write_failure_report(topology_dir: Path, failure: ComparisonFailure) -> None:
    _write_json(topology_dir / "failure_report.json", failure.model_dump(mode="json"))


def compare_topology_to_oracle(
    *,
    case_config: OracleCaseConfig,
    topology: Topology,
) -> ComparisonFailure | None:
    if topology.slug() == ORACLE_TOPOLOGY.slug():
        return None

    case_id = stable_case_id(case_config)
    case_dir = ARTIFACT_ROOT / case_id
    oracle_dir = case_dir / ORACLE_TOPOLOGY.slug()
    topology_dir = case_dir / topology.slug()

    oracle_manifest = _load_manifest(oracle_dir)
    topology_manifest = _load_manifest(topology_dir)
    if len(oracle_manifest.steps) != len(topology_manifest.steps):
        return _build_failure(
            case_id=case_id,
            topology=topology.slug(),
            metric="losses",
            step_index=0,
            key="__step_count__",
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            abs_tolerance=case_config.tolerances.losses_abs,
            rel_tolerance=case_config.tolerances.losses_rel,
            message=(
                "Step count mismatch: "
                f"oracle={len(oracle_manifest.steps)} vs "
                f"topology={len(topology_manifest.steps)}"
            ),
        )

    import torch

    for oracle_step, topology_step in zip(
        oracle_manifest.steps, topology_manifest.steps
    ):
        step_index = oracle_step.step_index
        oracle_outputs = _load_output_tensor(oracle_dir, oracle_step)
        topology_outputs = _load_output_tensor(topology_dir, topology_step)
        failure = _compare_tensor_pair(
            case_id=case_id,
            topology=topology.slug(),
            metric="outputs",
            step_index=step_index,
            key="logprobs",
            reference=oracle_outputs,
            candidate=topology_outputs,
            abs_tolerance=case_config.tolerances.outputs_abs,
            rel_tolerance=case_config.tolerances.outputs_rel,
        )
        if failure is not None:
            return failure

        oracle_loss = torch.tensor([oracle_step.loss], dtype=torch.float32)
        topology_loss = torch.tensor([topology_step.loss], dtype=torch.float32)
        failure = _compare_tensor_pair(
            case_id=case_id,
            topology=topology.slug(),
            metric="losses",
            step_index=step_index,
            key="loss",
            reference=oracle_loss,
            candidate=topology_loss,
            abs_tolerance=case_config.tolerances.losses_abs,
            rel_tolerance=case_config.tolerances.losses_rel,
        )
        if failure is not None:
            return failure

        for metric, oracle_file, topo_file, abs_tol, rel_tol in (
            (
                "grads",
                oracle_step.grads_file,
                topology_step.grads_file,
                case_config.tolerances.grads_abs,
                case_config.tolerances.grads_rel,
            ),
            (
                "lora_deltas",
                oracle_step.deltas_file,
                topology_step.deltas_file,
                case_config.tolerances.deltas_abs,
                case_config.tolerances.deltas_rel,
            ),
        ):
            failure = _compare_tensor_maps(
                case_id=case_id,
                topology=topology.slug(),
                metric=metric,
                step_index=step_index,
                reference=_load_safetensor_map(oracle_dir / oracle_file),
                candidate=_load_safetensor_map(topology_dir / topo_file),
                abs_tolerance=abs_tol,
                rel_tolerance=rel_tol,
            )
            if failure is not None:
                return failure
    return None


def run_and_compare_topology(
    *,
    case_config: OracleCaseConfig,
    topology: Topology,
    regenerate: bool = False,
) -> None:
    ensure_oracle_reference_artifacts(
        case_config=case_config,
        regenerate=regenerate and topology.slug() == ORACLE_TOPOLOGY.slug(),
    )
    ensure_topology_artifacts(
        case_config=case_config,
        topology=topology,
        regenerate=regenerate,
        mutation=None,
    )
    failure = compare_topology_to_oracle(case_config=case_config, topology=topology)
    if failure is None:
        return
    topology_dir = ARTIFACT_ROOT / failure.case_id / topology.slug()
    _write_failure_report(topology_dir, failure)
    raise AssertionError(
        "Megatron oracle mismatch: "
        f"topology={failure.topology}, metric={failure.metric}, "
        f"step={failure.step_index}, key={failure.key}, "
        f"max_abs={failure.max_abs_error:.6g}, "
        f"max_rel={failure.max_rel_error:.6g}, "
        f"tol_abs={failure.abs_tolerance:.6g}, "
        f"tol_rel={failure.rel_tolerance:.6g}"
    )


def run_sensitivity_check(
    *,
    case_config: OracleCaseConfig,
    regenerate: bool = False,
) -> None:
    mutation = sensitivity_mutation()
    if mutation is None:
        raise RuntimeError(
            f"Sensitivity check requires {SENSITIVITY_MUTATION_ENV} to be set"
        )

    ensure_oracle_reference_artifacts(
        case_config=case_config,
        regenerate=regenerate,
    )
    ensure_topology_artifacts(
        case_config=case_config,
        topology=SENSITIVITY_TOPOLOGY,
        regenerate=True,
        mutation=mutation,
    )
    failure = compare_topology_to_oracle(
        case_config=case_config,
        topology=SENSITIVITY_TOPOLOGY,
    )
    if failure is None:
        raise AssertionError(
            "Sensitivity mutation did not produce an oracle mismatch. "
            f"mutation={mutation}, topology={SENSITIVITY_TOPOLOGY.slug()}"
        )


def _set_deterministic_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _merge_sharded_dicts(shards_by_rank: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    merged: dict[str, list[Any]] = {}
    for rank_shards in shards_by_rank:
        for key, tensor in rank_shards.items():
            merged.setdefault(key, []).append(tensor.detach().cpu())
    full_state: dict[str, Any] = {}
    for key, shards in merged.items():
        if len(shards) == 1:
            full_state[key] = shards[0].contiguous()
            continue
        concat_dim = 1 if ".lora_A." in key else 0
        full_state[key] = torch.cat(shards, dim=concat_dim).contiguous()
    return full_state


def _gather_full_state(local_state: dict[str, Any]) -> dict[str, Any] | None:
    import torch

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    torch.distributed.gather_object(local_state, gathered, dst=0)
    if rank != 0:
        return None
    assert gathered is not None
    entries = [entry for entry in gathered if entry is not None]
    return _merge_sharded_dicts(entries)


def _collect_lora_state(model_chunks: list[Any]) -> dict[str, Any] | None:
    local_state: dict[str, Any] = {}
    for chunk in model_chunks:
        for module in chunk.modules():
            if not hasattr(module, "sharded_lora_state_dict"):
                continue
            module_state = module.sharded_lora_state_dict()
            for key, value in module_state.items():
                if key in local_state:
                    raise RuntimeError(
                        f"Duplicate LoRA key while collecting state: {key}"
                    )
                local_state[key] = value.detach().cpu()
    return _gather_full_state(local_state)


def _collect_lora_grads(model_chunks: list[Any]) -> dict[str, Any] | None:
    from megatron.core import parallel_state as ps

    from art.megatron.lora import LoRA

    local_grads: dict[str, Any] = {}
    for chunk in model_chunks:
        for module in chunk.modules():
            if not isinstance(module, LoRA):
                continue
            grad_a = (
                module.A_T.grad
                if module.A_T.grad is not None
                else module.A_T.new_zeros(module.A_T.shape)
            )
            grad_b = (
                module.B_T.grad
                if module.B_T.grad is not None
                else module.B_T.new_zeros(module.B_T.shape)
            )
            if module.num_local_experts > 1:
                if ps.get_expert_data_parallel_rank() != 0:
                    continue
                for expert in range(module.num_local_experts):
                    prefix = module.adapter_model_prefix.format(
                        expert=expert + module._expert_offset
                    )
                    local_grads[f"{prefix}.lora_A.weight"] = (
                        grad_a[expert].detach().cpu().T
                    )
                    local_grads[f"{prefix}.lora_B.weight"] = (
                        grad_b[expert].detach().cpu().T
                    )
            else:
                if ps.get_data_parallel_rank() != 0:
                    continue
                local_grads[f"{module.adapter_model_prefix}.lora_A.weight"] = (
                    grad_a.detach().cpu().T
                )
                local_grads[f"{module.adapter_model_prefix}.lora_B.weight"] = (
                    grad_b.detach().cpu().T
                )
    return _gather_full_state(local_grads)


def _validate_adapter_exact(
    expected_state: dict[str, Any],
    adapter_model: dict[str, Any],
) -> None:
    expected_keys = set(expected_state.keys())
    adapter_keys = set(adapter_model.keys())
    missing = sorted(expected_keys - adapter_keys)
    extra = sorted(adapter_keys - expected_keys)
    if missing or extra:
        raise KeyError(
            f"Adapter keys mismatch: missing={missing[:5]} extra={extra[:5]}"
        )


def _validate_loaded_state_matches_adapter(
    loaded_state: dict[str, Any],
    adapter_model: dict[str, Any],
) -> None:
    import torch

    for key in sorted(adapter_model.keys()):
        if key not in loaded_state:
            raise KeyError(f"Loaded LoRA state missing key: {key}")
        if not torch.equal(loaded_state[key].cpu(), adapter_model[key].cpu()):
            max_abs, max_rel = _tensor_error(adapter_model[key], loaded_state[key])
            raise RuntimeError(
                f"Loaded LoRA state mismatch for key '{key}' "
                f"(max_abs={max_abs:.6g}, max_rel={max_rel:.6g})"
            )


def _configure_provider(provider: Any, topology: Topology) -> None:
    provider.tensor_model_parallel_size = topology.tp
    provider.expert_model_parallel_size = topology.ep
    provider.expert_tensor_parallel_size = topology.etp
    provider.pipeline_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.sequence_parallel = bool(topology.sp)
    if hasattr(provider, "attention_dropout"):
        provider.attention_dropout = 0.0
    if hasattr(provider, "hidden_dropout"):
        provider.hidden_dropout = 0.0


def _delta_state(
    initial_state: dict[str, Any],
    current_state: dict[str, Any],
) -> dict[str, Any]:
    initial_keys = set(initial_state.keys())
    current_keys = set(current_state.keys())
    if initial_keys != current_keys:
        missing = sorted(initial_keys - current_keys)
        extra = sorted(current_keys - initial_keys)
        raise KeyError(
            f"LoRA state keys changed during training: missing={missing[:3]} extra={extra[:3]}"
        )
    return {
        key: current_state[key].detach().cpu() - initial_state[key].detach().cpu()
        for key in sorted(initial_keys)
    }


@contextmanager
def _mutation_hook(
    megatron_train_module: Any,
    mutation: SensitivityMutation | None,
    pre_optimizer_step_hook: Callable[[], None] | None = None,
):
    original_finalize = megatron_train_module._finalize_grads
    original_optimizer_step = megatron_train_module._optimizer_step

    if mutation == "drop_finalize":
        megatron_train_module._finalize_grads = lambda _model: None
    elif mutation is not None:
        raise ValueError(f"Unsupported mutation: {mutation}")

    if pre_optimizer_step_hook is not None:

        def _patched_optimizer_step(optimizer: Any, learning_rate: float):
            pre_optimizer_step_hook()
            return original_optimizer_step(optimizer, learning_rate)

        megatron_train_module._optimizer_step = _patched_optimizer_step

    if mutation is None:
        if pre_optimizer_step_hook is None:
            yield
            return
    try:
        yield
    finally:
        megatron_train_module._finalize_grads = original_finalize
        megatron_train_module._optimizer_step = original_optimizer_step


def _worker_run(request: WorkerRunRequest) -> None:
    from megatron.core.optimizer import OptimizerConfig
    from safetensors.torch import load_file, save_file
    import torch

    from art import dev, types
    from art.megatron import train as megatron_train
    from art.preprocessing.pack import packed_tensors_from_dir

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    _set_deterministic_seed(request.case_config.seed)

    world_size = torch.distributed.get_world_size()
    if world_size != request.topology.world_size():
        raise RuntimeError(
            f"World size mismatch: expected {request.topology.world_size()}, got {world_size}"
        )

    runtime = megatron_train.build_training_runtime(
        model_identifier=request.case_config.base_model,
        provider_configure=lambda provider: _configure_provider(
            provider, request.topology
        ),
        optimizer_config=OptimizerConfig(
            bf16=True,
            lr=request.case_config.learning_rate,
            adam_beta1=0.9,
            adam_beta2=0.99,
            clip_grad=0.1,
            weight_decay=0.1,
        ),
        print_env=False,
        print_optimizer_stats=False,
    )
    model_chunks = runtime.model
    optimizer = runtime.optimizer

    topology_dir = Path(request.topology_dir)
    traces_dir = topology_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    shared_init_path = Path(request.shared_init_adapter_path)
    if not shared_init_path.exists():
        if not request.allow_create_shared_init:
            raise FileNotFoundError(
                f"Missing oracle shared adapter at {shared_init_path}"
            )
        initial_state = _collect_lora_state(model_chunks)
        if torch.distributed.get_rank() == 0:
            assert initial_state is not None
            shared_init_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(initial_state, str(shared_init_path))
    torch.distributed.barrier()
    if not shared_init_path.exists():
        raise FileNotFoundError(f"Shared init adapter not created: {shared_init_path}")

    adapter_model = load_file(str(shared_init_path))
    expected_state = _collect_lora_state(model_chunks)
    if torch.distributed.get_rank() == 0:
        assert expected_state is not None
        _validate_adapter_exact(expected_state, adapter_model)
    torch.distributed.barrier()

    megatron_train.load_adapter_into_model(model_chunks, adapter_model)
    loaded_state = _collect_lora_state(model_chunks)
    if torch.distributed.get_rank() == 0:
        assert loaded_state is not None
        _validate_loaded_state_matches_adapter(loaded_state, adapter_model)
    torch.distributed.barrier()

    packed_tensors = packed_tensors_from_dir(
        **request.packed_tensors.model_dump(exclude_none=True)
    )
    initial_lora_state = _collect_lora_state(model_chunks)
    if torch.distributed.get_rank() == 0 and initial_lora_state is None:
        raise RuntimeError("Failed to collect initial LoRA state on rank 0")

    train_config = types.TrainConfig(
        learning_rate=request.case_config.learning_rate,
        beta=request.case_config.beta,
        kl_penalty_coef=0.0,
    )
    experimental_config: dev.TrainConfig = {}
    step_traces: list[StepTrace] = []
    captured_grads: dict[str, Any] | None = None

    def _capture_lora_grads() -> None:
        nonlocal captured_grads
        captured_grads = _collect_lora_grads(model_chunks)

    with _mutation_hook(
        megatron_train,
        request.mutation,
        pre_optimizer_step_hook=_capture_lora_grads,
    ):
        for step_index in range(request.case_config.num_steps):
            sample_index = step_index % request.packed_tensors.num_sequences
            inputs = megatron_train.select_indexed_inputs(packed_tensors, sample_index)
            captured_grads = None

            step_result = megatron_train.run_training_step(
                model_chunks=model_chunks,
                optimizer=optimizer,
                learning_rate=train_config.learning_rate,
                inputs=inputs,
                config=train_config,
                experimental_config=experimental_config,
                ref_logprobs=None,
            )
            if torch.distributed.get_rank() == 0 and captured_grads is None:
                raise RuntimeError("Failed to collect LoRA grads on rank 0")

            current_lora_state = _collect_lora_state(model_chunks)
            if torch.distributed.get_rank() == 0 and current_lora_state is None:
                raise RuntimeError("Failed to collect current LoRA state on rank 0")

            if torch.distributed.get_rank() == 0:
                assert captured_grads is not None
                assert initial_lora_state is not None
                assert current_lora_state is not None
                output_rel = Path("traces") / f"output_step_{step_index:03d}.pt"
                grads_rel = Path("traces") / f"grads_step_{step_index:03d}.safetensors"
                deltas_rel = (
                    Path("traces") / f"deltas_step_{step_index:03d}.safetensors"
                )
                lora_rel = Path(f"lora_step_{step_index:03d}.safetensors")

                torch.save(
                    step_result.new_logprobs.detach().cpu().float(),
                    topology_dir / output_rel,
                )
                save_file(captured_grads, str(topology_dir / grads_rel))
                deltas = _delta_state(initial_lora_state, current_lora_state)
                save_file(deltas, str(topology_dir / deltas_rel))
                save_file(current_lora_state, str(topology_dir / lora_rel))

                step_traces.append(
                    StepTrace(
                        step_index=step_index,
                        loss=float(step_result.reduced_loss.item()),
                        probs_corr=step_result.probs_corr,
                        output_file=str(output_rel),
                        grads_file=str(grads_rel),
                        deltas_file=str(deltas_rel),
                        lora_file=str(lora_rel),
                    )
                )
            torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        manifest = RunManifest(
            case_id=request.case_id,
            base_model=request.case_config.base_model,
            topology=request.topology.slug(),
            world_size=request.topology.world_size(),
            seed=request.case_config.seed,
            num_steps=request.case_config.num_steps,
            packed_tensors=request.packed_tensors,
            tolerances=request.case_config.tolerances,
            steps=step_traces,
        )
        _write_json(topology_dir / "manifest.json", manifest.model_dump(mode="json"))
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def _run_worker_cli(run_request_path: Path) -> None:
    request = WorkerRunRequest.model_validate(_read_json(run_request_path))
    _worker_run(request)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Megatron oracle harness worker")
    parser.add_argument("--worker-run", action="store_true")
    parser.add_argument("--run-request", type=Path)
    return parser.parse_args(argv)


def _main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if not args.worker_run:
        raise SystemExit("This module is intended for test imports or --worker-run")
    if args.run_request is None:
        raise SystemExit("--run-request is required with --worker-run")
    _run_worker_cli(args.run_request)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
