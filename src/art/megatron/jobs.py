from typing import Literal

from pydantic import BaseModel

from .. import dev, types
from ..preprocessing.pack import DiskPackedTensors
from .routing_replay import MoeRoutingReplayBundle

DEFAULT_TRAINING_LOG_PATH = "/tmp/megatron_training_log.jsonl"
DEFAULT_JOBS_DIR = "/tmp/megatron_training_jobs"
DEFAULT_VLLM_WAKE_LOCK_PATH = "/tmp/megatron_vllm_waking"


class MegatronTrainingJob(BaseModel):
    lora_path: str
    optimizer_state_path: str
    disk_packed_tensors: DiskPackedTensors
    config: types.TrainConfig
    experimental_config: dev.TrainConfig
    moe_routing_replay_path: str | None = None
    moe_routing_replay_strict: bool = True
    log_path: str = DEFAULT_TRAINING_LOG_PATH


MegatronTrainingJob.model_rebuild(
    force=True,
    _types_namespace={"MoeRoutingReplayBundle": MoeRoutingReplayBundle},
)


class MegatronSFTTrainingJob(BaseModel):
    job_type: Literal["sft"] = "sft"
    lora_path: str
    optimizer_state_path: str
    sft_data_dir: str
    num_batches: int
    learning_rates: list[float]
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    log_path: str = DEFAULT_TRAINING_LOG_PATH
