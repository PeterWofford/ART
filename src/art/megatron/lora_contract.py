import math
from dataclasses import dataclass
from typing import Any

import torch

DEFAULT_LORA_RANK = 1
DEFAULT_LORA_ALPHA = 32
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_LORA_LAYER_ROOT = "base_model.model.model.layers"


def layer_prefix(layer_idx: int) -> str:
    return f"{_LORA_LAYER_ROOT}.{layer_idx}"


def self_attn_prefix(layer_idx: int) -> str:
    return f"{layer_prefix(layer_idx)}.self_attn"


def self_attn_proj_prefix(layer_idx: int, proj_name: str) -> str:
    return f"{self_attn_prefix(layer_idx)}.{proj_name}"


def experts_prefix(layer_idx: int) -> str:
    return f"{layer_prefix(layer_idx)}.mlp.experts"


def expert_prefix(layer_idx: int, expert_idx: int) -> str:
    return f"{experts_prefix(layer_idx)}.{expert_idx}"


def default_target_modules() -> list[str]:
    return list(DEFAULT_TARGET_MODULES)


@dataclass(frozen=True)
class QwenMoELoraSpec:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    kv_channels: int
    num_experts: int
    expert_intermediate_size: int

    @property
    def q_out_features(self) -> int:
        return self.kv_channels * self.num_attention_heads

    @property
    def kv_out_features(self) -> int:
        return self.kv_channels * self.num_key_value_heads


def qwen_moe_lora_spec_from_config(config_dict: dict[str, Any]) -> QwenMoELoraSpec:
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
        text_config.get("kv_channels")
        or text_config.get("head_dim")
        or (hidden_size // num_attention_heads)
    )
    num_experts = int(text_config["num_experts"])
    expert_intermediate_size_raw = (
        text_config.get("moe_intermediate_size")
        or text_config.get("intermediate_size")
        or text_config.get("ffn_hidden_size")
    )
    if expert_intermediate_size_raw is None:
        raise ValueError("Unable to infer MoE intermediate size from base model config.")
    expert_intermediate_size = int(expert_intermediate_size_raw)

    return QwenMoELoraSpec(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        kv_channels=kv_channels,
        num_experts=num_experts,
        expert_intermediate_size=expert_intermediate_size,
    )


def build_qwen_moe_identity_lora_tensors(
    config_dict: dict[str, Any],
    *,
    rank: int = DEFAULT_LORA_RANK,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    spec = qwen_moe_lora_spec_from_config(config_dict)

    def _init_a(*shape: int) -> torch.Tensor:
        tensor = torch.empty(shape, dtype=dtype)
        torch.nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
        return tensor

    tensors: dict[str, torch.Tensor] = {}
    for layer_idx in range(spec.num_hidden_layers):
        for proj_name, in_features, out_features in (
            ("q_proj", spec.hidden_size, spec.q_out_features),
            ("k_proj", spec.hidden_size, spec.kv_out_features),
            ("v_proj", spec.hidden_size, spec.kv_out_features),
            ("o_proj", spec.q_out_features, spec.hidden_size),
        ):
            proj_prefix = self_attn_proj_prefix(layer_idx, proj_name)
            tensors[f"{proj_prefix}.lora_A.weight"] = _init_a(rank, in_features)
            tensors[f"{proj_prefix}.lora_B.weight"] = torch.zeros(
                out_features, rank, dtype=dtype
            )

        for expert_idx in range(spec.num_experts):
            proj_root = expert_prefix(layer_idx, expert_idx)
            for proj_name in ("gate_proj", "up_proj"):
                proj_prefix = f"{proj_root}.{proj_name}"
                tensors[f"{proj_prefix}.lora_A.weight"] = _init_a(rank, spec.hidden_size)
                tensors[f"{proj_prefix}.lora_B.weight"] = torch.zeros(
                    spec.expert_intermediate_size, rank, dtype=dtype
                )

            down_prefix = f"{proj_root}.down_proj"
            tensors[f"{down_prefix}.lora_A.weight"] = _init_a(
                rank, spec.expert_intermediate_size
            )
            tensors[f"{down_prefix}.lora_B.weight"] = torch.zeros(
                spec.hidden_size, rank, dtype=dtype
            )

    return tensors

