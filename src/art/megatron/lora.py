from typing import Sequence

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.transformer_layer import TransformerLayer
import torch

from .lora_adapters import (
    LoRA,
    MLPExpertsLinearFC1LoRA,
    MLPExpertsLinearFC2LoRA,
    SelfAttentionLinearProjLoRA,
    SelfAttentionLinearQKVLoRA,
)
from .lora_contract import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_RANK,
    experts_prefix,
    layer_prefix,
    self_attn_prefix,
    self_attn_proj_prefix,
)


def apply_lora_adapters(
    model: Sequence[torch.nn.Module],
    provider: GPTModelProvider,
) -> None:
    with torch.no_grad():
        for chunk in model:
            for module in chunk.modules():
                if isinstance(module, TransformerLayer):
                    layer_idx = module.layer_number - 1
                    adapter_model_prefix = layer_prefix(layer_idx)
                    assert isinstance(module.self_attention, SelfAttention)
                    self_attention_linear_proj = module.self_attention.linear_proj
                    if not isinstance(self_attention_linear_proj, TERowParallelLinear):
                        self_attention_linear_proj = (
                            self_attention_linear_proj.linear_proj
                        )
                        assert isinstance(
                            self_attention_linear_proj, TERowParallelLinear
                        )
                    module.self_attention.linear_proj = SelfAttentionLinearProjLoRA(
                        adapter_model_prefix=self_attn_proj_prefix(layer_idx, "o_proj"),
                        linear_proj=self_attention_linear_proj,
                        rank=DEFAULT_LORA_RANK,
                        alpha=DEFAULT_LORA_ALPHA,
                        provider=provider,
                    )
                    self_attention_linear_qkv = module.self_attention.linear_qkv
                    if not isinstance(
                        self_attention_linear_qkv, TELayerNormColumnParallelLinear
                    ):
                        self_attention_linear_qkv = self_attention_linear_qkv.linear_qkv
                        assert isinstance(
                            self_attention_linear_qkv, TELayerNormColumnParallelLinear
                        )
                    module.self_attention.linear_qkv = SelfAttentionLinearQKVLoRA(
                        adapter_model_prefix=self_attn_prefix(layer_idx),
                        linear_qkv=self_attention_linear_qkv,
                        rank=DEFAULT_LORA_RANK,
                        alpha=DEFAULT_LORA_ALPHA,
                        provider=provider,
                    )
                    assert isinstance(module.mlp.experts, TEGroupedMLP)
                    mlp_experts_linear_fc1 = module.mlp.experts.linear_fc1
                    if not isinstance(
                        mlp_experts_linear_fc1,
                        TEColumnParallelGroupedLinear,  # type: ignore
                    ):
                        mlp_experts_linear_fc1 = mlp_experts_linear_fc1.linear_fc1
                        assert isinstance(
                            mlp_experts_linear_fc1,
                            TEColumnParallelGroupedLinear,  # type: ignore
                        )
                    module.mlp.experts.linear_fc1 = MLPExpertsLinearFC1LoRA(
                        adapter_model_prefix=experts_prefix(layer_idx),
                        linear_fc1=mlp_experts_linear_fc1,
                        rank=DEFAULT_LORA_RANK,
                        alpha=DEFAULT_LORA_ALPHA,
                        num_local_experts=module.mlp.experts.num_local_experts,
                    )
                    mlp_experts_linear_fc2 = module.mlp.experts.linear_fc2
                    if not isinstance(
                        mlp_experts_linear_fc2,
                        TERowParallelGroupedLinear,  # type: ignore
                    ):
                        mlp_experts_linear_fc2 = mlp_experts_linear_fc2.linear_fc2
                        assert isinstance(
                            mlp_experts_linear_fc2,
                            TERowParallelGroupedLinear,  # type: ignore
                        )
                    module.mlp.experts.linear_fc2 = MLPExpertsLinearFC2LoRA(
                        adapter_model_prefix=experts_prefix(layer_idx),
                        linear_fc2=mlp_experts_linear_fc2,
                        rank=DEFAULT_LORA_RANK,
                        alpha=DEFAULT_LORA_ALPHA,
                        num_local_experts=module.mlp.experts.num_local_experts,
                    )
