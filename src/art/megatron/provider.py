import copy
import inspect

from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.spec_utils import ModuleSpec
import torch

from art.megatron.flex_attention import FlexDotProductAttention


def get_provider(model: str) -> GPTModelProvider:
    bridge = AutoBridge.from_hf_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    assert isinstance(bridge._model_bridge, Qwen3MoEBridge), (
        "Only Qwen3 MoE models are supported"
    )
    provider = bridge.to_megatron_provider()
    base_layer_spec = provider.transformer_layer_spec

    def _flex_attention_layer_spec(
        config: GPTModelProvider, vp_stage: int | None = None
    ) -> ModuleSpec:
        if isinstance(base_layer_spec, ModuleSpec):
            layer_spec = copy.deepcopy(base_layer_spec)
        elif "vp_stage" in inspect.signature(base_layer_spec).parameters:
            layer_spec = base_layer_spec(config, vp_stage=vp_stage)
        else:
            layer_spec = base_layer_spec(config)

        # Keep Megatron's standard layer stack and replace only core attention.
        layer_spec = copy.deepcopy(layer_spec)
        layer_spec.submodules.self_attention.submodules.core_attention = (  # type: ignore[union-attr]
            FlexDotProductAttention
        )
        return layer_spec

    provider.transformer_layer_spec = _flex_attention_layer_spec
    provider.attention_backend = AttnBackend.auto
    provider.recompute_granularity = "full"
    provider.recompute_method = "uniform"
    provider.recompute_num_layers = 1
    provider.tensor_model_parallel_size = min(2, torch.cuda.device_count())
    provider.context_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = torch.cuda.device_count()
    provider.expert_tensor_parallel_size = 1
    provider.moe_shared_expert_overlap = True
    provider.moe_router_dtype = "fp32"
    if provider.tensor_model_parallel_size > 1:
        provider.sequence_parallel = True
    return provider
