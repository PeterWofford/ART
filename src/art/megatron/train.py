# isort: off
import os


def _set_cache_dir(env_var: str, default_path: str) -> None:
    if not os.environ.get(env_var):
        os.environ[env_var] = os.path.expanduser(default_path)
    os.makedirs(os.environ[env_var], exist_ok=True)


os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
_set_cache_dir("TORCHINDUCTOR_CACHE_DIR", "~/.cache/torchinductor")
_set_cache_dir("TRITON_CACHE_DIR", "~/.triton/cache")
# isort: on

import gc
import json
import math
import shutil
import time
from typing import Any, Callable, cast

from megatron.core import parallel_state as ps
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer.module import MegatronModule
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load_file, save_file
import torch
from torch._inductor.runtime.cache_dir_utils import cache_dir as inductor_cache_dir

from art import dev, types
from art.loss import loss_fn, shift_tensor
from art.megatron.finalize_grads import finalize_model_grads_extended
from art.megatron.flex_attention import create_shared_prefix_attention_state
from art.megatron.lora import apply_lora_adapters
from art.megatron.offload import OffloadState, offload_to_cpu, reload_to_gpu
from art.megatron.provider import get_provider
from art.preprocessing.pack import (
    DiskPackedTensors,
    PackedTensors,
    packed_tensors_from_dir,
)

DEFAULT_MODEL_IDENTIFIER = "Qwen/Qwen3-30B-A3B-Instruct-2507"


class TrainingJob(BaseModel):
    lora_path: str
    optimizer_state_path: str
    disk_packed_tensors: DiskPackedTensors
    config: types.TrainConfig
    experimental_config: dev.TrainConfig


class TrainingRuntime(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: Any
    model: list[MegatronModule]
    optimizer: Any
    rank: int
    world_size: int


class TrainStepResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    reduced_loss: torch.Tensor
    probs_corr: float
    new_logprobs: torch.Tensor
    update_successful: bool
    grad_norm: float
    num_zeros_in_grad: int | None


def print0(rank: int, *values: Any) -> None:
    if rank == 0:
        print(*values)


def freeze_model(model_chunks: list[MegatronModule]) -> list[MegatronModule]:
    for module in model_chunks:
        for param in module.parameters():
            param.requires_grad = False
    return model_chunks


def _install_gpt_preprocess_hook(model_chunks: list[MegatronModule]) -> None:
    for chunk in model_chunks:
        module: Any = chunk
        while not isinstance(module, GPTModel) and hasattr(module, "module"):
            module = module.module
        if not isinstance(module, GPTModel):
            continue
        preprocess = module._preprocess

        def preprocess_hook(*args, _preprocess=preprocess, **kwargs):
            preproc_output = list(_preprocess(*args, **kwargs))
            preproc_output[0].requires_grad = True  # type: ignore[index]
            table = preproc_output[1]  # [S, B, 1, D]  # type: ignore[index]
            embedding_dim = table.size(-1)
            table_flat = table.view(table.size(0), embedding_dim)
            position_ids = kwargs["position_ids"]  # [B, S]
            batch_size, sequence_length = position_ids.shape
            gathered = table_flat.index_select(0, position_ids.reshape(-1))
            gathered = (
                gathered.view(batch_size, sequence_length, embedding_dim)
                .permute(1, 0, 2)
                .contiguous()
            )
            preproc_output[1] = gathered.unsqueeze(2)  # [S, B, 1, D]
            return tuple(preproc_output)

        module._preprocess = preprocess_hook  # type: ignore[attr-defined]


def _default_optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        bf16=True,
        lr=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        clip_grad=0.1,
        weight_decay=0.1,
    )


def build_training_runtime(
    *,
    model_identifier: str | None = None,
    provider_configure: Callable[[Any], None] | None = None,
    optimizer_config: OptimizerConfig | None = None,
    print_env: bool = True,
    print_optimizer_stats: bool = True,
) -> TrainingRuntime:
    provider = get_provider(
        model_identifier or os.environ.get("MODEL_IDENTIFIER", DEFAULT_MODEL_IDENTIFIER)
    )
    if provider_configure is not None:
        provider_configure(provider)
    provider.register_pre_wrap_hook(freeze_model)
    provider.register_pre_wrap_hook(
        lambda chunks: apply_lora_adapters(chunks, provider)
    )

    model = cast(
        list[MegatronModule],
        provider.provide_distributed_model(
            ddp_config=DistributedDataParallelConfig(),
            data_parallel_random_init=False,
        ),
    )

    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before building runtime"
        )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    if rank == 0 and print_env:
        print("TORCHINDUCTOR_CACHE_DIR:", os.environ["TORCHINDUCTOR_CACHE_DIR"])
        print("Resolved inductor cache_dir():", inductor_cache_dir())
        print("TRITON_CACHE_DIR:", os.environ["TRITON_CACHE_DIR"])

    _install_gpt_preprocess_hook(model)

    optimizer = get_megatron_optimizer(
        config=optimizer_config or _default_optimizer_config(),
        model_chunks=model,
    )

    if rank == 0 and print_optimizer_stats:
        num_params = sum(
            p.numel()
            for group in optimizer.param_groups
            if not group["is_decoupled_lr"]
            for p in group["params"]
        )
        print(f"Number of parameters in optimizer: {num_params:,}")
        total_params = sum(p.numel() for module in model for p in module.parameters())
        percent = (num_params / total_params) * 100 if total_params > 0 else 0
        print(f"Optimizer parameters as percent of total: {percent:0.2f}%")

    return TrainingRuntime(
        provider=provider,
        model=model,
        optimizer=optimizer,
        rank=rank,
        world_size=world_size,
    )


def iter_modules(model_chunks: list[MegatronModule]) -> Any:
    for chunk in model_chunks:
        for module in chunk.modules():
            yield module


def load_adapter_into_model(
    model_chunks: list[MegatronModule],
    adapter_model: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for module in iter_modules(model_chunks):
            if hasattr(module, "load_lora"):
                module.load_lora(adapter_model)  # type: ignore[attr-defined]


def collect_sharded_lora_state(
    model_chunks: list[MegatronModule],
    adapter_model: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    sharded_state_dict: dict[str, torch.Tensor] = {}
    sharded_state_manifest: dict[str, dict[str, Any]] = {}
    for module in iter_modules(model_chunks):
        if hasattr(module, "sharded_lora_state_dict"):
            module_sharded_lora_state_dict: dict[str, torch.Tensor] = (
                module.sharded_lora_state_dict()  # type: ignore[attr-defined]
            )
            for key, value in module_sharded_lora_state_dict.items():
                target_dtype = (
                    adapter_model[key].dtype if key in adapter_model else value.dtype
                )
                sharded_state_dict[key] = value.to(target_dtype)
        if hasattr(module, "sharded_lora_manifest"):
            module_sharded_lora_manifest: dict[str, dict[str, Any]] = (
                module.sharded_lora_manifest()  # type: ignore[attr-defined]
            )
            sharded_state_manifest.update(module_sharded_lora_manifest)
    return sharded_state_dict, sharded_state_manifest


def select_indexed_inputs(packed_tensors: PackedTensors, index: int) -> PackedTensors:
    return PackedTensors(  # type: ignore[call-arg]
        **{
            key: value[index : index + 1]
            for key, value in packed_tensors.items()
            if isinstance(value, torch.Tensor)
        },
        pixel_values=[None],
        image_grid_thw=[None],
    )


def _move_inputs_to_device(inputs: PackedTensors, device: torch.device) -> None:
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)  # type: ignore[index]


def _finalize_grads(model_chunks: list[MegatronModule]) -> None:
    finalize_model_grads_extended(cast(list[torch.nn.Module], model_chunks))


def _optimizer_step(
    optimizer: Any,
    learning_rate: float,
) -> tuple[bool, float, int | None]:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
    update_successful, grad_norm, num_zeros_in_grad = cast(
        tuple[bool, float, int | None], optimizer.step()
    )
    optimizer.zero_grad()
    return update_successful, grad_norm, num_zeros_in_grad


def _reduce_loss(loss: torch.Tensor) -> torch.Tensor:
    reduced_loss = loss.detach().clone()
    torch.distributed.all_reduce(reduced_loss, op=torch.distributed.ReduceOp.AVG)
    return reduced_loss


def run_training_step(
    *,
    model_chunks: list[MegatronModule],
    optimizer: Any,
    learning_rate: float,
    inputs: PackedTensors,
    config: types.TrainConfig,
    experimental_config: dev.TrainConfig,
    ref_logprobs: torch.Tensor | None = None,
) -> TrainStepResult:
    device = next(model_chunks[0].parameters()).device
    _move_inputs_to_device(inputs, device)

    attention_state = create_shared_prefix_attention_state(
        group_ids=inputs["group_ids"],
        parent_ids=inputs["parent_ids"],
    )
    attention_mask = torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=device)

    for chunk in model_chunks:
        chunk.zero_grad_buffer()  # ty: ignore[call-non-callable]

    new_logprobs: torch.Tensor = -model_chunks[0](
        input_ids=inputs["tokens"],
        position_ids=inputs["input_pos"],
        attention_mask=attention_mask,
        labels=shift_tensor(inputs["tokens"], 0),
        extra_block_kwargs={"attention_bias": attention_state},
    )

    loss_info = loss_fn(
        inputs,  # ty: ignore[invalid-argument-type]
        new_logprobs,
        ref_logprobs,
        None,
        experimental_config,
    )
    loss = loss_info.mean_policy_loss + config.beta * loss_info.mean_kl
    loss.backward()
    _finalize_grads(model_chunks)
    update_successful, grad_norm, num_zeros_in_grad = _optimizer_step(
        optimizer,
        learning_rate,
    )
    reduced_loss = _reduce_loss(loss)

    return TrainStepResult(
        reduced_loss=reduced_loss,
        probs_corr=float(loss_info.probs_corr.item()),
        new_logprobs=new_logprobs,
        update_successful=update_successful,
        grad_norm=grad_norm,
        num_zeros_in_grad=num_zeros_in_grad,
    )


def _run_service_loop(runtime: TrainingRuntime) -> None:
    offload_state = OffloadState()
    offload_to_cpu(runtime.model, runtime.optimizer, runtime.rank, offload_state)

    while True:
        torch.distributed.barrier()
        jobs_dir = "/tmp/megatron_training_jobs"
        os.makedirs(jobs_dir, exist_ok=True)
        job_names = sorted(
            job_name for job_name in os.listdir(jobs_dir) if job_name.endswith(".json")
        )
        if not job_names:
            time.sleep(1)
            continue

        wake_lock_path = "/tmp/megatron_vllm_waking"
        while os.path.exists(wake_lock_path):
            time.sleep(0.2)

        reload_to_gpu(runtime.model, runtime.optimizer, runtime.rank, offload_state)

        job_name = job_names[0]
        job_path = os.path.join(jobs_dir, job_name)
        with open(job_path, "rb") as handle:
            job = TrainingJob.model_validate_json(handle.read())
        config = job.config
        experimental_config = job.experimental_config

        print0(runtime.rank, "Loaded job from", job_path)
        print0(runtime.rank, "Job:", job)

        adapter_model_path = f"{job.lora_path}/adapter_model.safetensors"
        if not os.path.exists(adapter_model_path):
            raise FileNotFoundError(f"No adapter model found at {adapter_model_path}")
        print0(runtime.rank, "Loading adapter model from", adapter_model_path)
        adapter_model = load_file(adapter_model_path)
        load_adapter_into_model(runtime.model, adapter_model)

        optimizer_shard_path = os.path.join(
            job.optimizer_state_path,
            f"{runtime.rank + 1:02d}-of-{runtime.world_size:02d}.pt",
        )
        if os.path.exists(optimizer_shard_path):
            print("Loading optimizer state from", optimizer_shard_path)
            runtime.optimizer.load_state_dict(torch.load(optimizer_shard_path))
        else:
            print(
                "No optimizer state found at",
                optimizer_shard_path,
                "- resetting optimizer for new run",
            )
            runtime.optimizer.optimizer.state.clear()
            runtime.optimizer.reload_model_params()

        print0(
            runtime.rank, "Loading packed tensors from", job.disk_packed_tensors["dir"]
        )
        packed_tensors = packed_tensors_from_dir(**job.disk_packed_tensors)
        num_sequences = job.disk_packed_tensors["num_sequences"]

        dp_rank = ps.get_data_parallel_rank()
        dp_world_size = ps.get_data_parallel_world_size()
        num_indices = math.ceil(num_sequences / dp_world_size)
        indices = list(range(dp_rank, num_sequences, dp_world_size))
        if not indices:
            indices = [dp_rank % num_sequences]
        repeat = math.ceil(num_indices / len(indices))
        indices = (indices * repeat)[:num_indices]

        for index in indices:
            inputs = select_indexed_inputs(packed_tensors, index)
            step_result = run_training_step(
                model_chunks=runtime.model,
                optimizer=runtime.optimizer,
                learning_rate=config.learning_rate,
                inputs=inputs,
                config=config,
                experimental_config=experimental_config,
                ref_logprobs=None,
            )
            print0(
                runtime.rank,
                "Correlation between old and new probabilities:",
                step_result.probs_corr,
            )

            if runtime.rank == 0:
                with open(
                    "/tmp/megatron_training_log.jsonl", "a+", encoding="utf-8"
                ) as log_file:
                    log_msg = json.dumps(
                        {
                            "loss": step_result.reduced_loss.item(),
                            "grad_norm": step_result.grad_norm,
                            "probs_corr": step_result.probs_corr,
                        }
                    )
                    print("Logging", log_msg)
                    log_file.write(log_msg + "\n")

        sharded_state_dict, sharded_state_manifest = collect_sharded_lora_state(
            runtime.model,
            adapter_model,
        )
        shard_path = os.path.join(
            job.lora_path,
            f"adapter_model-{runtime.rank + 1:02d}-of-{runtime.world_size:02d}.safetensors",
        )
        manifest_path = os.path.join(
            job.lora_path,
            f"adapter_manifest-{runtime.rank + 1:02d}-of-{runtime.world_size:02d}.json",
        )
        print("Saving adapter shard to", shard_path)
        save_file(sharded_state_dict, shard_path)
        print("Saving adapter shard manifest to", manifest_path)
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(sharded_state_manifest, manifest_file, sort_keys=True)

        print("Saving optimizer shard to", optimizer_shard_path)
        os.makedirs(job.optimizer_state_path, exist_ok=True)
        torch.save(runtime.optimizer.state_dict(), optimizer_shard_path)

        offload_to_cpu(runtime.model, runtime.optimizer, runtime.rank, offload_state)

        del packed_tensors
        del adapter_model
        if "inputs" in locals():
            del inputs
        gc.collect()
        torch.cuda.empty_cache()

        torch.distributed.barrier()
        if runtime.rank == 0:
            os.remove(job_path)
            with open(
                "/tmp/megatron_training_log.jsonl", "a+", encoding="utf-8"
            ) as log_file:
                log_file.write("all done\n")
            shutil.rmtree(job.disk_packed_tensors["dir"])


def main() -> None:
    runtime = build_training_runtime(
        model_identifier=os.environ.get("MODEL_IDENTIFIER", DEFAULT_MODEL_IDENTIFIER)
    )
    _run_service_loop(runtime)


if __name__ == "__main__":
    main()
