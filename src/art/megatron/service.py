import asyncio
from dataclasses import dataclass
import datetime
from functools import cached_property
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, AsyncIterator

from peft.tuners.lora.config import LoraConfig
from pydantic import BaseModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from vllm import AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.v1.engine.async_llm import AsyncLLM

from .. import dev, types
from ..local.checkpoints import get_last_checkpoint_dir
from ..preprocessing.pack import DiskPackedTensors
from ..preprocessing.tokenize import SFTBatch
from ..unsloth.service import do_sleep, do_wake_up, gc_and_empty_cuda_cache
from ..utils.get_model_step import get_step_from_dir
from ..utils.output_dirs import get_step_checkpoint_dir
from ..vllm import get_llm, openai_server_task, run_on_workers


class MegatronTrainingJob(BaseModel):
    """Job format for communication with train.py"""

    lora_path: str
    optimizer_state_path: str
    disk_packed_tensors: DiskPackedTensors
    config: types.TrainConfig
    experimental_config: dev.TrainConfig


@dataclass
class MegatronService:
    model_name: str
    base_model: str
    config: dev.InternalModelConfig
    output_dir: str
    _is_sleeping: bool = False
    _latest_step: int = 0
    _lora_id_counter: int = 1
    _megatron_process: asyncio.subprocess.Process | None = None
    _optimizer_state_path: str | None = None

    def _next_lora_id(self) -> int:
        self._lora_id_counter += 1
        return self._lora_id_counter

    def _get_optimizer_state_path(self) -> str:
        if self._optimizer_state_path is not None:
            return self._optimizer_state_path
        self._optimizer_state_path = os.path.join(self.output_dir, "optimizer_states")
        os.makedirs(self._optimizer_state_path, exist_ok=True)
        return self._optimizer_state_path

    def _default_lora_adapter_config(self) -> LoraConfig:
        # Keep in sync with LoRA settings in megatron/train.py.
        return LoraConfig(
            r=1,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
        )

    def _load_base_model_config(self) -> dict[str, Any]:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(self.base_model, "config.json")
        with open(config_path) as f:
            return json.load(f)

    def _create_qwen_moe_identity_lora(
        self, lora_path: str, config_dict: dict[str, Any]
    ) -> None:
        text_config = config_dict.get("text_config")
        if not isinstance(text_config, dict):
            text_config = config_dict

        hidden_size = int(text_config["hidden_size"])
        num_hidden_layers = int(text_config["num_hidden_layers"])
        num_attention_heads = int(text_config["num_attention_heads"])
        num_key_value_heads = int(
            text_config.get("num_key_value_heads")
            or text_config.get("num_query_groups")
            or num_attention_heads
        )
        kv_channels = int(
            text_config.get("kv_channels") or (hidden_size // num_attention_heads)
        )
        num_experts = int(text_config["num_experts"])
        expert_intermediate_size_raw = (
            text_config.get("moe_intermediate_size")
            or text_config.get("intermediate_size")
            or text_config.get("ffn_hidden_size")
        )
        if expert_intermediate_size_raw is None:
            raise ValueError(
                "Unable to infer MoE intermediate size from base model config."
            )
        expert_intermediate_size = int(expert_intermediate_size_raw)

        q_out_features = kv_channels * num_attention_heads
        kv_out_features = kv_channels * num_key_value_heads
        rank = int(self._default_lora_adapter_config().r)
        dtype = torch.bfloat16

        def _init_a(*shape: int) -> torch.Tensor:
            tensor = torch.empty(shape, dtype=dtype)
            torch.nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
            return tensor

        tensors: dict[str, torch.Tensor] = {}
        for layer_idx in range(num_hidden_layers):
            layer_prefix = f"base_model.model.model.layers.{layer_idx}"

            for proj_name, out_features in (
                ("q_proj", q_out_features),
                ("k_proj", kv_out_features),
                ("v_proj", kv_out_features),
                ("o_proj", hidden_size),
            ):
                proj_prefix = f"{layer_prefix}.self_attn.{proj_name}"
                tensors[f"{proj_prefix}.lora_A.weight"] = _init_a(rank, hidden_size)
                tensors[f"{proj_prefix}.lora_B.weight"] = torch.zeros(
                    out_features, rank, dtype=dtype
                )

            experts_prefix = f"{layer_prefix}.mlp.experts"
            for expert_idx in range(num_experts):
                expert_prefix = f"{experts_prefix}.{expert_idx}"

                for proj_name in ("gate_proj", "up_proj"):
                    proj_prefix = f"{expert_prefix}.{proj_name}"
                    tensors[f"{proj_prefix}.lora_A.weight"] = _init_a(rank, hidden_size)
                    tensors[f"{proj_prefix}.lora_B.weight"] = torch.zeros(
                        expert_intermediate_size, rank, dtype=dtype
                    )

                down_prefix = f"{expert_prefix}.down_proj"
                tensors[f"{down_prefix}.lora_A.weight"] = _init_a(
                    rank, expert_intermediate_size
                )
                tensors[f"{down_prefix}.lora_B.weight"] = torch.zeros(
                    hidden_size, rank, dtype=dtype
                )

        os.makedirs(lora_path, exist_ok=True)
        save_file(tensors, os.path.join(lora_path, "adapter_model.safetensors"))

    def _adapter_has_weights(self, lora_path: str) -> bool:
        adapter_path = os.path.join(lora_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            return False
        try:
            with safe_open(adapter_path, framework="pt") as adapter_file:
                for key in adapter_file.keys():
                    tensor = adapter_file.get_tensor(key)
                    if torch.any(tensor != 0):
                        return True
        except Exception:
            return False
        return False

    def _create_identity_lora(self, lora_path: str) -> None:
        config_dict = self._load_base_model_config()
        if config_dict.get("model_type") == "qwen3_5_moe":
            self._create_qwen_moe_identity_lora(lora_path, config_dict)
            return

        # Create an identity (zero) LoRA using PEFT so vLLM can load it.
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM

        lora_config = self._default_lora_adapter_config()
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        peft_model = get_peft_model(model, lora_config)
        # Keep LoRA A initialized (trainable) and zero only B for identity.
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.data.zero_()
        os.makedirs(lora_path, exist_ok=True)
        peft_model.save_pretrained(lora_path)
        del peft_model, model
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _ensure_identity_lora(self, lora_path: str) -> None:
        if self._adapter_has_weights(lora_path):
            return
        self._create_identity_lora(lora_path)

    def _ensure_lora_adapter_config(
        self, lora_path: str, *, source_path: str | None = None
    ) -> None:
        config_path = os.path.join(lora_path, "adapter_config.json")
        if os.path.exists(config_path):
            return
        os.makedirs(lora_path, exist_ok=True)
        if source_path is not None:
            source_config = os.path.join(source_path, "adapter_config.json")
            if os.path.exists(source_config):
                shutil.copy(source_config, config_path)
                return
        config_dict = self._default_lora_adapter_config().to_dict()
        # PEFT emits target_modules as a set; normalize for JSON output.
        if isinstance(config_dict.get("target_modules"), set):
            config_dict["target_modules"] = sorted(config_dict["target_modules"])
        if config_dict.get("base_model_name_or_path") is None:
            config_dict["base_model_name_or_path"] = self.base_model
        with open(config_path, "w") as f:
            json.dump(config_dict, f)

    async def _add_lora_aliases(
        self, llm: AsyncLLM, step: int, checkpoint_dir: str
    ) -> None:
        added = await llm.add_lora(
            LoRARequest(
                lora_name=f"{self.model_name}@{step}",
                lora_int_id=self._next_lora_id(),
                lora_path=checkpoint_dir,
            )
        )
        if not added:
            raise RuntimeError(f"Failed to add LoRA adapter for step {step}")
        self._latest_step = step

    async def register_lora_for_step(self, step: int, checkpoint_dir: str) -> None:
        llm = await self.llm
        await llm.pause_generation()
        await self._add_lora_aliases(llm, step, checkpoint_dir)
        await llm.resume_generation()

    async def _ensure_megatron_running(self) -> None:
        """Lazily start Megatron training process if not running."""
        if self._megatron_process is not None:
            if self._megatron_process.returncode is None:
                return
            self._megatron_process = None

        try:
            import megatron.bridge  # type: ignore

            setup_cmd = ""
        except ImportError:
            setup_script = Path(__file__).parent / "setup.sh"
            setup_cmd = f"bash {setup_script} && "

        subprocess.run(["pkill", "-9", "megatron-service"], check=False)
        train_script = Path(__file__).parent / "train.py"
        num_gpus = torch.cuda.device_count()
        os.environ["MODEL_IDENTIFIER"] = self.base_model

        command = (
            f"{setup_cmd}uv run torchrun --nproc_per_node {num_gpus} {train_script}"
        )
        self._megatron_process = await asyncio.create_subprocess_shell(command)

    async def start_openai_server(
        self, config: dev.OpenAIServerConfig | None
    ) -> tuple[str, int]:
        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
            self._latest_step = 0
        else:
            self._latest_step = get_step_from_dir(self.output_dir)
        self._ensure_identity_lora(lora_path)
        self._ensure_lora_adapter_config(lora_path)

        lora_path_for_server = (
            lora_path if self._adapter_has_weights(lora_path) else None
        )
        model_name_for_server = self.model_name
        if lora_path_for_server is None:
            # When we cannot materialize an identity adapter for a new model
            # architecture, keep the step-qualified model alias available.
            model_name_for_server = f"{self.model_name}@{self._latest_step}"
        server_config = dev.get_openai_server_config(
            model_name=model_name_for_server,
            base_model=self.base_model,
            log_file=f"{self.output_dir}/logs/vllm.log",
            lora_path=lora_path_for_server,
            config=config,
        )
        await openai_server_task(engine=await self.llm, config=server_config)
        return (
            server_config.get("server_args", {}).get("host") or "0.0.0.0",
            server_config.get("server_args", {}).get("port", 8000),
        )

    async def vllm_engine_is_sleeping(self) -> bool:
        return self._is_sleeping

    async def train(
        self,
        disk_packed_tensors: DiskPackedTensors,
        config: types.TrainConfig,
        _config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        llm = await self.llm
        await llm.pause_generation()
        await llm.reset_prefix_cache()
        await run_on_workers(llm, do_sleep, level=2)
        self._is_sleeping = True
        gc_and_empty_cuda_cache()

        # Start Megatron after vLLM has freed GPU memory.
        await self._ensure_megatron_running()

        lora_path = get_last_checkpoint_dir(self.output_dir)
        if lora_path is None:
            lora_path = get_step_checkpoint_dir(self.output_dir, 0)
        self._ensure_lora_adapter_config(lora_path)

        self._optimizer_state_path = self._get_optimizer_state_path()

        jobs_dir = "/tmp/megatron_training_jobs"
        os.makedirs(jobs_dir, exist_ok=True)
        for job_name in os.listdir(jobs_dir):
            if job_name.endswith(".json"):
                os.remove(os.path.join(jobs_dir, job_name))
        job = MegatronTrainingJob(
            lora_path=lora_path,
            optimizer_state_path=self._optimizer_state_path,
            disk_packed_tensors=disk_packed_tensors,
            config=config,
            experimental_config=_config,
        )
        job_path = os.path.join(jobs_dir, f"{datetime.datetime.now().isoformat()}.json")
        with open(job_path, "w") as f:
            f.write(job.model_dump_json())

        num_lines = 0
        while True:
            await asyncio.sleep(0.1)
            if (
                self._megatron_process is None
                or self._megatron_process.returncode is not None
            ):
                exit_code = (
                    self._megatron_process.returncode
                    if self._megatron_process is not None
                    else "unknown"
                )
                raise RuntimeError(
                    "Megatron training process exited while waiting for completion "
                    f"signal (exit code: {exit_code})."
                )
            try:
                with open("/tmp/megatron_training_log.jsonl", "a+") as log_file:
                    log_file.seek(0)
                    lines = log_file.readlines()[num_lines:]
                    for line in lines:
                        if line := line.strip():
                            if line == "all done":
                                self._merge_lora_adapter(lora_path)
                                os.remove("/tmp/megatron_training_log.jsonl")
                                break
                            num_lines += 1
                            yield json.loads(line)
                    else:
                        continue
                    break
            except FileNotFoundError:
                continue

        next_step = self._latest_step + 1
        new_checkpoint_dir = get_step_checkpoint_dir(self.output_dir, next_step)
        os.makedirs(new_checkpoint_dir, exist_ok=True)
        shutil.copy(
            f"{lora_path}/adapter_model.safetensors",
            f"{new_checkpoint_dir}/adapter_model.safetensors",
        )
        self._ensure_lora_adapter_config(new_checkpoint_dir, source_path=lora_path)

        wake_lock_path = "/tmp/megatron_vllm_waking"
        try:
            with open(wake_lock_path, "w") as lock_file:
                lock_file.write("waking vllm\n")
            await run_on_workers(llm, do_wake_up)
            self._is_sleeping = False
        finally:
            if os.path.exists(wake_lock_path):
                os.remove(wake_lock_path)

        await self._add_lora_aliases(llm, next_step, new_checkpoint_dir)
        await llm.resume_generation()

    # SFT not supported for MegatronService
    async def train_sft(
        self,
        batches: list[Any],
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        raise NotImplementedError("SFT training is not supported for MegatronService")
        yield {}  # Make this a generator

    def _merge_lora_adapter(self, lora_path: str) -> None:
        """Merge sharded LoRA adapters from distributed training."""
        base_dir = Path(lora_path)
        shard_filenames = sorted(base_dir.glob("adapter_model-*-of-*.safetensors"))
        if not shard_filenames:
            return

        adapter_model_path = base_dir / "adapter_model.safetensors"
        sharded_tensors: dict[str, list[torch.Tensor]] = {}

        for filename in shard_filenames:
            with safe_open(filename, framework="pt") as file:
                for key in file.keys():
                    tensor = file.get_tensor(key)
                    sharded_tensors.setdefault(key, []).append(tensor)

        adapter_model: dict[str, torch.Tensor] = {}
        if adapter_model_path.exists():
            adapter_model = load_file(adapter_model_path)

        for key, tensors in sharded_tensors.items():
            tensor = torch.cat(tensors, dim=1 if "lora_A" in key else 0)
            adapter_model[key] = tensor

        save_file(adapter_model, adapter_model_path)
        for filename in shard_filenames:
            filename.unlink()

    @cached_property
    def llm(self) -> asyncio.Task[AsyncLLM]:
        engine_args = {
            **self.config.get("engine_args", {}),
            "enable_lora": True,
            "max_loras": self.config.get("engine_args", {}).get("max_loras", 2),
        }
        for key in ["enable_log_requests", "disable_log_requests"]:
            engine_args.pop(key, None)
        return asyncio.create_task(get_llm(AsyncEngineArgs(**engine_args)))  # type: ignore
