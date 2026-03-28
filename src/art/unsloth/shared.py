import asyncio
from dataclasses import dataclass
import os
import time
from typing import Any, AsyncIterator, Iterable, Literal, cast

from datasets import Dataset
import nest_asyncio
import peft
import torch
from torch.optim import Optimizer
from transformers import GenerationMixin, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from trl import GRPOConfig, GRPOTrainer

from .. import dev, types
from ..preprocessing.inputs import TrainInputs, create_train_inputs
from ..preprocessing.pack import (
    DiskPackedTensors,
    PackedTensors,
    packed_tensors_from_dir,
)
from ..preprocessing.tokenize import SFTBatch
from .train import gc_and_empty_cuda_cache, train

nest_asyncio.apply()


class CausalLM(PreTrainedModel, GenerationMixin):
    """Dummy class for type checking."""

    pass


@dataclass
class UnslothTrainContext:
    model: CausalLM
    tokenizer: PreTrainedTokenizerBase
    peft_model: peft.peft_model.PeftModelForCausalLM
    trainer: GRPOTrainer
    inputs_queue: asyncio.Queue[TrainInputs]
    results_queue: asyncio.Queue[dict[str, float]]
    train_task: asyncio.Task[None] | None = None
    warmup_pending: bool = True
    last_training_mode: Literal["sft", "rl"] | None = None
    _is_offloaded: bool = False
    _pinned_buffers: dict[str, torch.Tensor] | None = None

    def offload_to_cpu(self) -> None:
        if self._is_offloaded:
            return

        if self._pinned_buffers is None:
            self._pinned_buffers = {}

        for name, param in self.peft_model.named_parameters():
            if param.device.type != "cuda":
                continue
            if (
                name not in self._pinned_buffers
                or self._pinned_buffers[name].shape != param.shape
            ):
                self._pinned_buffers[name] = torch.empty(
                    param.shape,
                    dtype=param.dtype,
                    device="cpu",
                    pin_memory=True,
                )
            self._pinned_buffers[name].copy_(param.data, non_blocking=True)
            param.data = self._pinned_buffers[name]

        optimizer = getattr(self.trainer, "optimizer", None)
        if optimizer is not None and hasattr(optimizer, "state"):
            for param_id, state in optimizer.state.items():
                for key, value in state.items():
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.device.type != "cuda"
                    ):
                        continue
                    buffer_key = f"opt_{id(param_id)}_{key}"
                    if (
                        buffer_key not in self._pinned_buffers
                        or self._pinned_buffers[buffer_key].shape != value.shape
                    ):
                        self._pinned_buffers[buffer_key] = torch.empty(
                            value.shape,
                            dtype=value.dtype,
                            device="cpu",
                            pin_memory=True,
                        )
                    self._pinned_buffers[buffer_key].copy_(value, non_blocking=True)
                    state[key] = self._pinned_buffers[buffer_key]

        torch.cuda.synchronize()
        self._is_offloaded = True
        gc_and_empty_cuda_cache()

    def reload_to_gpu(self, device: str = "cuda:0") -> None:
        if not self._is_offloaded:
            return

        for _, param in self.peft_model.named_parameters():
            if param.device.type != "cpu":
                continue
            gpu_tensor = torch.empty(param.shape, dtype=param.dtype, device=device)
            gpu_tensor.copy_(param.data, non_blocking=True)
            param.data = gpu_tensor

        optimizer = getattr(self.trainer, "optimizer", None)
        if optimizer is not None and hasattr(optimizer, "state"):
            for state in optimizer.state.values():
                for key, value in state.items():
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.device.type != "cpu"
                    ):
                        continue
                    gpu_tensor = torch.empty(
                        value.shape, dtype=value.dtype, device=device
                    )
                    gpu_tensor.copy_(value, non_blocking=True)
                    state[key] = gpu_tensor

        torch.cuda.synchronize()
        self._is_offloaded = False

    async def load_lora_adapter(self, lora_path: str) -> None:
        try:
            await self.results_queue.join()
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        try:
            import importlib

            load_safetensors = importlib.import_module("safetensors.torch").load_file
        except Exception:
            load_safetensors = None  # type: ignore[assignment]

        state_dict = None
        st_path = os.path.join(lora_path, "adapter_model.safetensors")
        bin_path = os.path.join(lora_path, "adapter_model.bin")
        alt_st_path = os.path.join(lora_path, "model.safetensors")
        alt_bin_path = os.path.join(lora_path, "pytorch_model.bin")
        try:
            if os.path.exists(st_path) and load_safetensors is not None:
                state_dict = load_safetensors(st_path, device="cpu")
            elif os.path.exists(bin_path):
                state_dict = torch.load(bin_path, map_location="cpu")  # type: ignore[call-arg]
            elif os.path.exists(alt_st_path) and load_safetensors is not None:
                state_dict = load_safetensors(alt_st_path, device="cpu")
            elif os.path.exists(alt_bin_path):
                state_dict = torch.load(alt_bin_path, map_location="cpu")  # type: ignore[call-arg]
            else:
                raise FileNotFoundError(f"No adapter weights found in {lora_path}")
        except Exception as exc:
            raise RuntimeError(f"Failed to load LoRA adapter weights: {exc}") from exc

        with torch.no_grad():
            self.peft_model.zero_grad(set_to_none=True)
            optimizer = getattr(self.trainer, "optimizer", None)
            if optimizer is not None:
                optimizer = getattr(optimizer, "optimizer", optimizer)
                if hasattr(optimizer, "zero_grad"):
                    optimizer.zero_grad(set_to_none=True)  # type: ignore[arg-type]
                if hasattr(optimizer, "state") and isinstance(optimizer.state, dict):
                    optimizer.state.clear()

        try:
            try:
                from peft.utils.save_and_load import (
                    set_peft_model_state_dict as _set_peft_model_state_dict,
                )
            except Exception:
                from peft import (
                    set_peft_model_state_dict as _set_peft_model_state_dict,  # type: ignore
                )

            active_adapter = getattr(self.peft_model, "active_adapter", "default")
            _set_peft_model_state_dict(
                self.peft_model,
                state_dict,
                adapter_name=active_adapter,
            )
            self.peft_model.set_adapter(active_adapter)
        except Exception as exc:
            raise RuntimeError(f"Failed to set LoRA weights in-place: {exc}") from exc

        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    async def load_optimizer_state(self, checkpoint_dir: str) -> None:
        try:
            await self.results_queue.join()
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
            self.trainer.optimizer.load_state_dict(optimizer_state)

    def save_lora_adapter(self, lora_path: str) -> None:
        self.trainer.save_model(lora_path)

    def save_optimizer_state(self, checkpoint_dir: str) -> None:
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        torch.save(self.trainer.optimizer.state_dict(), optimizer_path)


def create_unsloth_train_context(
    *,
    init_args: dict[str, Any],
    peft_args: dict[str, Any],
    trainer_args: dict[str, Any],
    use_fast_model: bool = False,
) -> UnslothTrainContext:
    import unsloth

    loader_cls = unsloth.FastModel if use_fast_model else unsloth.FastLanguageModel
    model, tokenizer = cast(
        tuple[CausalLM, PreTrainedTokenizerBase],
        loader_cls.from_pretrained(**init_args),
    )

    if (
        hasattr(model, "peft_config")
        and getattr(model, "peft_config", None) is not None
    ):
        peft_model = cast(peft.peft_model.PeftModelForCausalLM, model)
    else:
        peft_model = cast(
            peft.peft_model.PeftModelForCausalLM,
            loader_cls.get_peft_model(model, **peft_args),
        )

    if not hasattr(peft_model, "warnings_issued"):
        peft_model.warnings_issued = {}  # type: ignore[attr-defined]

    trainer = GRPOTrainer(
        model=peft_model,  # type: ignore[arg-type]
        reward_funcs=[],
        args=GRPOConfig(**trainer_args),
        train_dataset=Dataset.from_list([{"prompt": ""} for _ in range(10_000_000)]),
        processing_class=tokenizer,
    )
    if trainer.optimizer is None:
        trainer.create_optimizer()

    inputs_queue: asyncio.Queue[TrainInputs] = asyncio.Queue()
    results_queue: asyncio.Queue[dict[str, float]] = asyncio.Queue()

    def _async_prepare_inputs(*_: Any, **__: Any) -> dict[str, torch.Tensor]:
        async def get_inputs() -> TrainInputs:
            return await inputs_queue.get()

        inputs = asyncio.run(get_inputs())
        return cast(dict[str, torch.Tensor], inputs)

    trainer._prepare_inputs = _async_prepare_inputs

    return UnslothTrainContext(
        model=model,
        tokenizer=tokenizer,
        peft_model=peft_model,
        trainer=trainer,
        inputs_queue=inputs_queue,
        results_queue=results_queue,
    )


def _get_trainer_optimizer(ctx: UnslothTrainContext) -> Optimizer:
    optimizer = cast(Optimizer | None, getattr(ctx.trainer, "optimizer", None))
    if optimizer is None:
        raise RuntimeError("Trainer optimizer must be initialized before training")
    return optimizer


def _reset_optimizer_if_mode_changed(
    ctx: UnslothTrainContext,
    mode: Literal["sft", "rl"],
) -> None:
    mode_changed = ctx.last_training_mode is not None and ctx.last_training_mode != mode
    if mode_changed:
        _get_trainer_optimizer(ctx).state.clear()
    ctx.last_training_mode = mode


def _precalculate_new_logprobs(
    ctx: UnslothTrainContext,
    packed_tensors: PackedTensors,
    config: types.TrainConfig,
    _config: dev.TrainConfig,
) -> torch.Tensor:
    return torch.cat(
        [
            ctx.trainer.compute_loss(
                ctx.peft_model,
                TrainInputs(  # ty:ignore[missing-typed-dict-key]
                    **{
                        key: value[offset : offset + 1]
                        for key, value in packed_tensors.items()
                        if isinstance(value, torch.Tensor)
                    },
                    pixel_values=packed_tensors["pixel_values"][offset : offset + 1],
                    image_grid_thw=packed_tensors["image_grid_thw"][
                        offset : offset + 1
                    ],
                    config=config,
                    _config=_config,
                    return_new_logprobs=True,
                ),
            )
            for offset in range(0, packed_tensors["tokens"].shape[0])
        ]
    ).to("cpu")


async def run_unsloth_rl_training(
    ctx: UnslothTrainContext,
    disk_packed_tensors: DiskPackedTensors,
    config: types.TrainConfig,
    _config: dev.TrainConfig,
    verbose: bool = False,
) -> AsyncIterator[dict[str, float]]:
    _reset_optimizer_if_mode_changed(ctx, "rl")
    optimizer = _get_trainer_optimizer(ctx)
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = 0.1

    packed_tensors = packed_tensors_from_dir(**disk_packed_tensors)
    await ctx.results_queue.join()

    if ctx.train_task is None:
        ctx.train_task = asyncio.create_task(
            train(
                trainer=ctx.trainer,
                results_queue=ctx.results_queue,
            )
        )

    warmup = ctx.warmup_pending
    precalculate_logprobs = _config.get("precalculate_logprobs", False)

    for offset in range(0, packed_tensors["tokens"].shape[0]):
        for _ in range(2 if warmup else 1):
            if precalculate_logprobs and not warmup:
                packed_tensors["original_logprobs"] = packed_tensors["logprobs"]  # type: ignore[index]
                packed_tensors["logprobs"] = _precalculate_new_logprobs(
                    ctx,
                    packed_tensors,
                    config,
                    _config,
                )
                precalculate_logprobs = False

            ctx.inputs_queue.put_nowait(
                create_train_inputs(packed_tensors, offset, config, _config, warmup)
            )

            done, _ = await asyncio.wait(
                [
                    asyncio.create_task(ctx.results_queue.get()),
                    ctx.train_task,
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if verbose:
                print(
                    "Done waiting for a result from the queue or for the training task to, presumably, raise an exception"
                )
            for task in done:
                result = task.result()
                assert result is not None, "The training task should never finish."
                ctx.results_queue.task_done()
                if warmup:
                    gc_and_empty_cuda_cache()
                    await asyncio.sleep(0.1)
                    warmup = False
                    ctx.warmup_pending = False
                else:
                    yield result


async def run_unsloth_sft_training(
    ctx: UnslothTrainContext,
    batches: Iterable[SFTBatch],
    verbose: bool = False,
    *,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
) -> AsyncIterator[dict[str, float]]:
    _reset_optimizer_if_mode_changed(ctx, "sft")
    optimizer = _get_trainer_optimizer(ctx)

    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"

    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = weight_decay

    ctx.peft_model.train()
    device = next(ctx.peft_model.parameters()).device

    for batch_idx, batch in enumerate(batches):
        batch_start_time = time.perf_counter()
        batch_loss = 0.0

        for param_group in optimizer.param_groups:
            param_group["lr"] = batch.learning_rate

        num_trainable_tokens = torch.tensor(
            batch.num_trainable_tokens,
            dtype=torch.long,
            device=device,
        )

        for trajectory_tensor in batch.trajectory_tensors:
            input_ids = trajectory_tensor["input_ids"].to(device)
            attention_mask = trajectory_tensor["attention_mask"].to(device)
            labels = trajectory_tensor["labels"].to(device)

            outputs = ctx.peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                num_items_in_batch=num_trainable_tokens,
            )
            loss = outputs.loss
            loss.backward()
            batch_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            ctx.peft_model.parameters(),
            max_grad_norm,
        ).item()

        optimizer.step()
        optimizer.zero_grad()

        batch_time = time.perf_counter() - batch_start_time
        tokens_per_second = (
            batch.num_trainable_tokens / batch_time if batch_time > 0 else 0.0
        )

        if verbose:
            print(
                f"Batch {batch_idx}: loss={batch_loss:.4f}, lr={batch.learning_rate:.2e}, "
                f"grad_norm={grad_norm:.4f}, tok/s={tokens_per_second:.1f}"
            )

        yield {
            "loss": batch_loss,
            "learning_rate": batch.learning_rate,
            "grad_norm": grad_norm,
            "num_trajectories": float(batch.num_trajectories),
            "num_trainable_tokens": float(batch.num_trainable_tokens),
            "tokens_per_second": tokens_per_second,
        }
