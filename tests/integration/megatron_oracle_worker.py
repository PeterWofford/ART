from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Callable

import numpy as np

from art.megatron.routing_replay import (
    ParallelTopology as ReplayParallelTopology,
)
from art.megatron.routing_replay import (
    build_bundle_from_forward_trace_dir,
)

from .megatron_forward_trace import ForwardTraceCapture
from .megatron_oracle_harness import (
    OracleCaseConfig,
    RunManifest,
    SensitivityMutation,
    StepTrace,
    Topology,
    WorkerRunRequest,
    _read_json,
    _require_not_none,
    _write_json,
)


def run_worker_subprocess(
    request: WorkerRunRequest,
    topology_dir: Path,
    *,
    repo_root: Path,
) -> None:
    """Runs one distributed worker subprocess and stores combined logs."""
    request_path = topology_dir / "run_request.json"
    _write_json(request_path, request.model_dump(mode="json"))
    worker_module = "integration.megatron_oracle_worker"
    worker_cwd = repo_root / "tests"

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(request.topology.world_size()),
        "-m",
        worker_module,
        "--worker-run",
        "--run-request",
        str(request_path),
    ]
    run = subprocess.run(
        command,
        cwd=str(worker_cwd),
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


def _set_deterministic_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _merge_sharded_dicts(shards_by_rank: list[dict[str, Any]]) -> dict[str, Any]:
    """Merges rank-sharded LoRA tensors into a full state dict on rank 0."""
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
    """Gathers local state dicts to rank 0 and merges them."""
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
    """Collects full LoRA adapter state for validation and delta computation."""
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
    """Collects full LoRA gradient tensors across all ranks."""
    from art.megatron.lora import LoRA

    local_grads: dict[str, Any] = {}
    for chunk in model_chunks:
        for module in chunk.modules():
            if not isinstance(module, LoRA):
                continue
            for key, param, expert in module._export_items():  # type: ignore[attr-defined]
                if not hasattr(param, "main_grad"):
                    raise RuntimeError(
                        f"LoRA param missing main_grad attribute for key '{key}'"
                    )
                grad = param.main_grad
                if grad is None:
                    raise RuntimeError(f"LoRA param main_grad is None for key '{key}'")
                if hasattr(grad, "_local_tensor"):
                    grad = grad._local_tensor
                local_grads[key] = (
                    grad[expert].detach().cpu().T
                    if expert is not None
                    else grad.detach().cpu().T
                )
    return _gather_full_state(local_grads)


def _validate_loaded_state_matches_adapter(
    loaded_state: dict[str, Any],
    adapter_model: dict[str, Any],
) -> None:
    """Checks loaded model LoRA state exactly matches adapter tensors and keys."""
    import torch

    for key in sorted(adapter_model.keys()):
        assert torch.equal(loaded_state[key].cpu(), adapter_model[key].cpu()), (
            f"Loaded LoRA state mismatch for key '{key}'"
        )


def _configure_provider(
    provider: Any,
    topology: Topology,
    case_config: OracleCaseConfig,
) -> None:
    """Applies deterministic topology/model overrides to provider config."""
    provider.tensor_model_parallel_size = topology.tp
    provider.expert_model_parallel_size = topology.ep
    provider.expert_tensor_parallel_size = topology.etp
    # These are intentionally pinned to 1 for now; switching to topology-driven
    # values is the single lever to start CP/PP coverage in the harness.
    provider.pipeline_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.sequence_parallel = topology.sp
    provider.num_layers = case_config.num_layers
    if hasattr(provider, "attention_dropout"):
        provider.attention_dropout = 0.0
    if hasattr(provider, "hidden_dropout"):
        provider.hidden_dropout = 0.0


def _build_optimizer_config(case_config: OracleCaseConfig):
    """Builds Megatron optimizer settings for deterministic harness runs."""
    from megatron.core.optimizer import OptimizerConfig

    optimizer_kwargs = dict(
        lr=case_config.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        clip_grad=0.1,
        weight_decay=0.1,
    )
    return OptimizerConfig(
        bf16=True,
        **optimizer_kwargs,
    )


def _assert_runtime_configuration(
    model_chunks: list[Any],
    case_config: OracleCaseConfig,
) -> None:
    """Validates runtime model depth equals requested oracle case config."""
    observed_num_layers: set[int] = set()

    for chunk in model_chunks:
        module: Any = chunk
        while hasattr(module, "module"):
            module = module.module
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "num_layers"):
            observed_num_layers.add(int(config.num_layers))

    if observed_num_layers != {case_config.num_layers}:
        raise RuntimeError(
            "Runtime num_layers mismatch: "
            f"requested={case_config.num_layers}, observed={sorted(observed_num_layers)}"
        )


def _delta_state(
    initial_state: dict[str, Any],
    current_state: dict[str, Any],
) -> dict[str, Any]:
    """Computes LoRA parameter deltas while enforcing stable key sets."""
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
    loss_scale: float = 1.0,
):
    """Applies optional sensitivity mutation hooks around training steps."""
    original_finalize = megatron_train_module._finalize_grads
    original_optimizer_step = megatron_train_module._optimizer_step
    original_loss_fn = megatron_train_module.loss_fn

    if mutation == "drop_finalize":
        megatron_train_module._finalize_grads = lambda _model: None
    elif mutation is not None:
        raise ValueError(f"Unsupported mutation: {mutation}")

    if pre_optimizer_step_hook is not None:

        def _patched_optimizer_step(optimizer: Any, learning_rate: float):
            pre_optimizer_step_hook()
            return original_optimizer_step(optimizer, learning_rate)

        megatron_train_module._optimizer_step = _patched_optimizer_step

    if loss_scale <= 0:
        raise ValueError(f"loss_scale must be > 0, got {loss_scale}")
    if loss_scale != 1.0:

        def _scaled_loss_fn(*args: Any, **kwargs: Any):
            loss = original_loss_fn(*args, **kwargs)
            return loss.model_copy(
                update={
                    "mean_policy_loss": loss.mean_policy_loss * loss_scale,
                    "mean_kl": loss.mean_kl * loss_scale,
                    "policy_loss_sum": loss.policy_loss_sum * loss_scale,
                }
            )

        megatron_train_module.loss_fn = _scaled_loss_fn

    if mutation is None:
        if pre_optimizer_step_hook is None and loss_scale == 1.0:
            yield
            return
    try:
        yield
    finally:
        megatron_train_module._finalize_grads = original_finalize
        megatron_train_module._optimizer_step = original_optimizer_step
        megatron_train_module.loss_fn = original_loss_fn


def _worker_run(request: WorkerRunRequest) -> None:
    """Executes one full distributed training trace generation worker run."""
    from safetensors.torch import load_file, save_file
    import torch

    from art import dev, types
    from art.megatron import train as megatron_train
    from art.preprocessing.pack import packed_tensors_from_dir

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    _set_deterministic_seed(request.case_config.seed)

    runtime = megatron_train.build_training_runtime(
        model_identifier=request.case_config.base_model,
        provider_configure=lambda provider: _configure_provider(
            provider, request.topology, request.case_config
        ),
        optimizer_config=_build_optimizer_config(request.case_config),
        print_env=False,
        print_optimizer_stats=False,
    )
    model_chunks = runtime.model
    optimizer = runtime.optimizer
    megatron_train.configure_moe_routing_replay(
        runtime,
        replay_bundle_path=request.moe_routing_replay_path,
        strict=request.moe_routing_replay_strict,
    )
    _assert_runtime_configuration(model_chunks, request.case_config)

    topology_dir = Path(request.topology_dir)
    traces_dir = topology_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    # setup the shared initial lora
    shared_init_path = Path(request.shared_init_adapter_path)
    if not shared_init_path.exists():
        initial_state = _collect_lora_state(model_chunks)
        if torch.distributed.get_rank() == 0:
            shared_init_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                _require_not_none(initial_state, "initial_state"),
                str(shared_init_path),
            )
    torch.distributed.barrier()

    # load the shared initial lora into the model and validate we can collect it from the model
    adapter_model = load_file(str(shared_init_path))
    megatron_train.load_adapter_into_model(model_chunks, adapter_model, optimizer)
    loaded_state = _collect_lora_state(model_chunks)
    if torch.distributed.get_rank() == 0:
        _validate_loaded_state_matches_adapter(
            _require_not_none(loaded_state, "loaded_state"), adapter_model
        )
    torch.distributed.barrier()

    # load the inputs
    packed_tensors = packed_tensors_from_dir(
        **request.packed_tensors.model_dump(exclude_none=True)
    )
    initial_lora_state = loaded_state

    train_config = types.TrainConfig(
        learning_rate=request.case_config.learning_rate,
        beta=request.case_config.beta,
        kl_penalty_coef=0.0,
    )
    experimental_config: dev.TrainConfig = {}
    step_traces: list[StepTrace] = []
    captured_grads: dict[str, Any] | None = None
    forward_trace_capture = ForwardTraceCapture(model_chunks, enabled=True)

    def _capture_lora_grads() -> None:
        nonlocal captured_grads
        captured_grads = _collect_lora_grads(model_chunks)

    with _mutation_hook(
        megatron_train,
        request.mutation,
        pre_optimizer_step_hook=_capture_lora_grads,
        loss_scale=request.case_config.loss_scale,
    ):
        for step_index in range(request.case_config.num_steps):
            forward_trace_capture.set_step(step_index)
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
                step_index=step_index,
                sample_index=sample_index,
                moe_routing_replay_controller=runtime.moe_routing_replay_controller,
            )
            forward_trace_capture.save_current_step(traces_dir)
            torch.distributed.barrier()
            current_lora_state = _collect_lora_state(model_chunks)

            if torch.distributed.get_rank() == 0:
                # save artifacts (outputs, grads, lora deltas, current lora)
                grads = _require_not_none(captured_grads, "captured_grads")
                initial_state = _require_not_none(
                    initial_lora_state, "initial_lora_state"
                )
                current_state = _require_not_none(
                    current_lora_state, "current_lora_state"
                )
                deltas = _delta_state(initial_state, current_state)

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
                save_file(grads, str(topology_dir / grads_rel))
                save_file(deltas, str(topology_dir / deltas_rel))
                save_file(current_state, str(topology_dir / lora_rel))

                # build and append the step trace
                step_traces.append(
                    StepTrace(
                        step_index=step_index,
                        loss=float(
                            step_result.reduced_loss.item()
                            / request.case_config.loss_scale
                        ),
                        probs_corr=step_result.probs_corr,
                        output_file=str(output_rel),
                        grads_file=str(grads_rel),
                        deltas_file=str(deltas_rel),
                        lora_file=str(lora_rel),
                    )
                )
            torch.distributed.barrier()

    forward_trace_capture.close()

    if torch.distributed.get_rank() == 0:
        # build and save the moe routing replay bundle
        if request.capture_moe_routing_bundle_path is not None:
            replay_bundle = build_bundle_from_forward_trace_dir(
                traces_dir=traces_dir,
                num_steps=request.case_config.num_steps,
                topology=ReplayParallelTopology.model_validate(
                    request.topology.model_dump(
                        include={"tp", "ep", "etp", "dp", "sp", "cp", "pp", "vpp"},
                        mode="python",
                    )
                ),
            )
            replay_bundle.to_dir(request.capture_moe_routing_bundle_path)

        # build and save the run manifest
        manifest = RunManifest(
            case_id=request.case_id,
            base_model=request.case_config.base_model,
            num_layers=request.case_config.num_layers,
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


def run_worker_cli(run_request_path: Path) -> None:
    """Loads a worker request and dispatches worker execution."""
    request = WorkerRunRequest.model_validate(_read_json(run_request_path))
    _worker_run(request)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parses worker CLI arguments."""
    parser = argparse.ArgumentParser(description="Megatron oracle harness worker")
    parser.add_argument("--worker-run", action="store_true")
    parser.add_argument("--run-request", type=Path)
    return parser.parse_args(argv)


def _main(argv: list[str]) -> int:
    """CLI entry for worker-only execution mode."""
    args = _parse_args(argv)
    if not args.worker_run:
        raise SystemExit("This module is intended for test imports or --worker-run")
    if args.run_request is None:
        raise SystemExit("--run-request is required with --worker-run")
    run_worker_cli(args.run_request)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
