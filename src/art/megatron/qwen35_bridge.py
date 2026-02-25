import logging
import re
from typing import Any

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    MegatronParamMapping,
)
from megatron.bridge.models.conversion.utils import get_module_and_param_from_name
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
from megatron.bridge.models.qwen.qwen_provider import Qwen3MoEModelProvider
from megatron.core.transformer.module import MegatronModule

logger = logging.getLogger(__name__)
_LOGGED_LINEAR_QKV_SKIP = False


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _text_config(cfg: Any) -> Any:
    text_cfg = _cfg_get(cfg, "text_config")
    if text_cfg is not None:
        return text_cfg
    return cfg


def _as_int(value: Any, *, fallback: int | None = None) -> int:
    if value is None:
        if fallback is None:
            raise ValueError("Expected integer config value, got None")
        return fallback
    return int(value)


def _resize_dim_to_target(
    tensor: torch.Tensor,
    *,
    dim: int,
    target_size: int,
    param_name: str,
) -> torch.Tensor:
    current_size = tensor.shape[dim]
    if current_size == target_size:
        return tensor
    if current_size > target_size:
        logger.warning(
            "Truncating %s along dim=%s from %s to %s",
            param_name,
            dim,
            current_size,
            target_size,
        )
        return tensor.narrow(dim, 0, target_size)

    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_size - current_size
    logger.warning(
        "Padding %s along dim=%s from %s to %s",
        param_name,
        dim,
        current_size,
        target_size,
    )
    pad = torch.zeros(*pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=dim)


class _DtypeSafeAutoMapping(AutoMapping):
    """AutoMapping variant that pre-aligns source dtype with destination dtype.

    This avoids TP scatter failures when Megatron parameters are float32 while
    HF tensors are bfloat16 (a known path in row-parallel mappings).
    """

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: torch.nn.Module,
    ) -> torch.Tensor:
        if self.tp_size == 1 or self.tp_rank == 0:
            normalized_param = self._normalize_expert_param_name(self.megatron_param)
            _, target_param = get_module_and_param_from_name(
                megatron_module, normalized_param
            )
            if hf_weights.dtype != target_param.dtype:
                hf_weights = hf_weights.to(target_param.dtype)
        return super().hf_to_megatron(hf_weights, megatron_module)


class _Qwen35FusedExpertFC1Mapping(MegatronParamMapping[torch.Tensor]):
    """Map fused gate_up_proj tensor to Megatron expert FC1 packed layout.

    HF source tensor shape is expected to be:
      [num_experts, 2 * moe_hidden, hidden]
    while each Megatron expert shard expects:
      [2 * moe_hidden, hidden]
    """

    def _validate_patterns(self) -> None:
        # Allow wildcard count mismatch:
        #   megatron: ...layers.*...weight*
        #   hf:       ...layers.*...gate_up_proj
        return

    def resolve(self, captures: tuple[str, ...]) -> "_Qwen35FusedExpertFC1Mapping":
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(resolved_megatron_param, str(resolved_hf_param))

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: torch.nn.Module,
    ) -> torch.Tensor:
        match = re.search(r"weight(\d+)$", self.megatron_param)
        if match is None:
            raise ValueError(f"Unable to parse expert index from {self.megatron_param}")
        expert_idx = int(match.group(1))
        if hf_weights.ndim != 3:
            raise ValueError(
                f"Expected fused expert tensor ndim=3 for {self.hf_param}, got {hf_weights.ndim}"
            )
        expert_weights = hf_weights[expert_idx]
        gate, up = torch.chunk(expert_weights, 2, dim=0)
        fused_fc1 = torch.cat([gate, up], dim=0)

        delegate = _DtypeSafeAutoMapping(self.megatron_param, self.megatron_param)
        return delegate.hf_to_megatron(fused_fc1, megatron_module)

    def megatron_to_hf(
        self,
        megatron_weights: torch.Tensor | None,
        megatron_module: MegatronModule | None,
    ) -> dict[str, torch.Tensor]:
        # Export path is intentionally deferred until we validate import path.
        return {}


class _Qwen35FusedExpertFC2Mapping(MegatronParamMapping[torch.Tensor]):
    """Map fused down_proj tensor to Megatron expert FC2 packed layout.

    HF source tensor shape is expected to be:
      [num_experts, hidden, moe_hidden]
    while each Megatron expert shard expects:
      [hidden, moe_hidden]
    """

    def _validate_patterns(self) -> None:
        # Allow wildcard count mismatch:
        #   megatron: ...layers.*...weight*
        #   hf:       ...layers.*...down_proj
        return

    def resolve(self, captures: tuple[str, ...]) -> "_Qwen35FusedExpertFC2Mapping":
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(resolved_megatron_param, str(resolved_hf_param))

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: torch.nn.Module,
    ) -> torch.Tensor:
        match = re.search(r"weight(\d+)$", self.megatron_param)
        if match is None:
            raise ValueError(f"Unable to parse expert index from {self.megatron_param}")
        expert_idx = int(match.group(1))
        if hf_weights.ndim != 3:
            raise ValueError(
                f"Expected fused expert tensor ndim=3 for {self.hf_param}, got {hf_weights.ndim}"
            )
        expert_fc2 = hf_weights[expert_idx]

        delegate = _DtypeSafeAutoMapping(self.megatron_param, self.megatron_param)
        return delegate.hf_to_megatron(expert_fc2, megatron_module)

    def megatron_to_hf(
        self,
        megatron_weights: torch.Tensor | None,
        megatron_module: MegatronModule | None,
    ) -> dict[str, torch.Tensor]:
        # Export path is intentionally deferred until we validate import path.
        return {}


class _Qwen35AttentionProjMapping(MegatronParamMapping[torch.Tensor]):
    """Map HF attention out projections to Megatron linear_proj layout."""

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: torch.nn.Module,
    ) -> torch.Tensor:
        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
        expected_cols = target_param.shape[1] * self.tp_size
        aligned = _resize_dim_to_target(
            hf_weights, dim=1, target_size=expected_cols, param_name=self.megatron_param
        )
        delegate = _DtypeSafeAutoMapping(self.megatron_param, self.megatron_param)
        return delegate.hf_to_megatron(aligned, megatron_module)

    def megatron_to_hf(
        self,
        megatron_weights: torch.Tensor | None,
        megatron_module: MegatronModule | None,
    ) -> dict[str, torch.Tensor]:
        # Export path is intentionally deferred until we validate import path.
        return {}


class _Qwen35LinearAttentionQKVMapping(MegatronParamMapping[torch.Tensor]):
    """Map linear-attention fused QKV to legacy Megatron linear_qkv layout."""

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: torch.nn.Module,
    ) -> torch.Tensor | None:
        global _LOGGED_LINEAR_QKV_SKIP
        if not _LOGGED_LINEAR_QKV_SKIP:
            logger.warning(
                "Skipping Qwen3.5 linear-attention QKV import; source layout does not match legacy Megatron Qwen3 MoE linear_qkv."
            )
            _LOGGED_LINEAR_QKV_SKIP = True
        # Qwen3.5 linear-attention QKV layout is not equivalent to the legacy
        # Qwen3 MoE Megatron linear_qkv layout. Skip base-weight import for this
        # tensor to avoid silent, destructive projection truncation.
        return None

    def megatron_to_hf(
        self,
        megatron_weights: torch.Tensor | None,
        megatron_module: MegatronModule | None,
    ) -> dict[str, torch.Tensor]:
        # Export path is intentionally deferred until we validate import path.
        return {}


class _Qwen35SyntheticNormMapping(MegatronParamMapping[torch.Tensor]):
    """Synthesize unit-norm scale for layers without HF q/k norm tensors."""

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: torch.nn.Module,
    ) -> torch.Tensor:
        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
        return torch.ones_like(target_param, dtype=target_param.dtype, device=target_param.device)

    def megatron_to_hf(
        self,
        megatron_weights: torch.Tensor | None,
        megatron_module: MegatronModule | None,
    ) -> dict[str, torch.Tensor]:
        # Export path is intentionally deferred until we validate import path.
        return {}


class _Qwen35FullAttentionQKVMapping(MegatronParamMapping[dict[str, torch.Tensor]]):
    """Map full-attention Q/K/V tensors into Megatron linear_qkv layout."""

    def hf_to_megatron(
        self,
        hf_weights: dict[str, torch.Tensor],
        megatron_module: torch.nn.Module,
    ) -> torch.Tensor:
        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
        expected_rows = target_param.shape[0] * self.tp_size

        q = hf_weights["q"]
        k = hf_weights["k"]
        v = hf_weights["v"]

        # Qwen3.5 full attention packs q_proj as [query, gate] per head.
        # Extract only query channels to map into Megatron linear_qkv.
        config = self._get_config(megatron_module)
        num_heads = int(getattr(config, "num_attention_heads", 0) or 0)
        head_dim = int(getattr(config, "kv_channels", 0) or 0)
        q_query_rows = num_heads * head_dim
        if (
            q_query_rows > 0
            and q.shape[0] == q_query_rows * 2
            and q.shape[0] % max(num_heads, 1) == 0
        ):
            q = (
                q.view(num_heads, head_dim * 2, q.shape[1])[:, :head_dim, :]
                .contiguous()
                .view(q_query_rows, q.shape[1])
            )
        else:
            expected_q_rows = max(expected_rows - k.shape[0] - v.shape[0], 0)
            q = _resize_dim_to_target(
                q,
                dim=0,
                target_size=expected_q_rows,
                param_name=f"{self.megatron_param}.q",
            )

        fused_qkv = torch.cat([q, k, v], dim=0)
        fused_qkv = _resize_dim_to_target(
            fused_qkv,
            dim=0,
            target_size=expected_rows,
            param_name=self.megatron_param,
        )
        delegate = _DtypeSafeAutoMapping(self.megatron_param, self.megatron_param)
        return delegate.hf_to_megatron(fused_qkv, megatron_module)

    def megatron_to_hf(
        self,
        megatron_weights: torch.Tensor | None,
        megatron_module: MegatronModule | None,
    ) -> dict[str, torch.Tensor]:
        # Export path is intentionally deferred until we validate import path.
        return {}


class Qwen35MoEBridge(Qwen3MoEBridge):
    """Local bridge skeleton for Qwen3.5 MoE.

    This bridge intentionally focuses on the HF -> Megatron import path first.
    It normalizes Qwen3.5 config nesting (text_config) and maps the most relevant
    tensors to make provider/model bring-up observable.
    """

    _cached_num_layers: int = 0
    _cached_layer_types: list[str] = []

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Qwen3MoEModelProvider:
        root_cfg = hf_pretrained.config
        text_cfg = _text_config(root_cfg)

        hidden_size = _as_int(_cfg_get(text_cfg, "hidden_size"))
        num_layers = _as_int(_cfg_get(text_cfg, "num_hidden_layers"))
        num_attention_heads = _as_int(_cfg_get(text_cfg, "num_attention_heads"))
        num_query_groups = _as_int(
            _cfg_get(text_cfg, "num_key_value_heads", _cfg_get(text_cfg, "num_query_groups")),
            fallback=num_attention_heads,
        )
        num_moe_experts = _as_int(_cfg_get(text_cfg, "num_experts"))
        moe_router_topk = _as_int(_cfg_get(text_cfg, "num_experts_per_tok"), fallback=8)

        moe_ffn_hidden_size = _as_int(
            _cfg_get(
                text_cfg,
                "moe_intermediate_size",
                _cfg_get(text_cfg, "ffn_hidden_size", _cfg_get(text_cfg, "intermediate_size")),
            )
        )
        ffn_hidden_size = _as_int(
            _cfg_get(
                text_cfg,
                "intermediate_size",
                _cfg_get(text_cfg, "ffn_hidden_size", moe_ffn_hidden_size * moe_router_topk),
            )
        )
        vocab_size = _as_int(_cfg_get(text_cfg, "vocab_size", _cfg_get(root_cfg, "vocab_size")))
        seq_length = _as_int(
            _cfg_get(
                text_cfg,
                "max_position_embeddings",
                _cfg_get(root_cfg, "max_position_embeddings", 32768),
            )
        )
        kv_channels = _as_int(
            _cfg_get(text_cfg, "head_dim", _cfg_get(text_cfg, "kv_channels")),
            fallback=hidden_size // num_attention_heads,
        )
        layer_types = _cfg_get(text_cfg, "layer_types", [])
        self._qwen35_num_layers = num_layers
        self._qwen35_layer_types = (
            [str(layer_type) for layer_type in layer_types]
            if isinstance(layer_types, list)
            else []
        )
        type(self)._cached_num_layers = self._qwen35_num_layers
        type(self)._cached_layer_types = list(self._qwen35_layer_types)

        provider = Qwen3MoEModelProvider(
            num_layers=num_layers,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            num_moe_experts=num_moe_experts,
            moe_router_topk=moe_router_topk,
            init_method_std=float(_cfg_get(text_cfg, "initializer_range", 0.02)),
            layernorm_epsilon=float(_cfg_get(text_cfg, "rms_norm_eps", 1e-6)),
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(vocab_size),
            rotary_base=float(_cfg_get(text_cfg, "rope_theta", 1000000.0)),
            share_embeddings_and_output_weights=bool(
                _cfg_get(root_cfg, "tie_word_embeddings", False)
            ),
            vocab_size=vocab_size,
            seq_length=seq_length,
            # dtype is defined on the top-level HF config for Qwen3.5 wrappers.
            fp16=(self.dtype_from_hf(root_cfg, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(root_cfg, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(root_cfg, default=torch.float32),
            generation_config=hf_pretrained.generation_config,
            qk_layernorm=True,
            moe_grouped_gemm=True,
            kv_channels=kv_channels,
        )
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = [
            # Embeddings / head / final norm
            _DtypeSafeAutoMapping(
                megatron_param="embedding.word_embeddings.weight",
                hf_param="model.language_model.embed_tokens.weight",
            ),
            _DtypeSafeAutoMapping(
                megatron_param="output_layer.weight",
                hf_param="lm_head.weight",
            ),
            _DtypeSafeAutoMapping(
                megatron_param="decoder.final_layernorm.weight",
                hf_param="model.language_model.norm.weight",
            ),
            # Layer norms and router
            _DtypeSafeAutoMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                hf_param="model.language_model.layers.*.input_layernorm.weight",
            ),
            _DtypeSafeAutoMapping(
                megatron_param="decoder.layers.*.mlp.router.weight",
                hf_param="model.language_model.layers.*.mlp.gate.weight",
            ),
            _DtypeSafeAutoMapping(
                megatron_param="decoder.layers.*.pre_mlp_layernorm.weight",
                hf_param="model.language_model.layers.*.post_attention_layernorm.weight",
            ),
            # MoE experts: custom fused mappings.
            _Qwen35FusedExpertFC1Mapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                hf_param="model.language_model.layers.*.mlp.experts.gate_up_proj",
            ),
            _Qwen35FusedExpertFC2Mapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                hf_param="model.language_model.layers.*.mlp.experts.down_proj",
            ),
        ]

        layer_types = list(
            getattr(self, "_qwen35_layer_types", [])
            or getattr(type(self), "_cached_layer_types", [])
        )
        num_layers = int(
            getattr(self, "_qwen35_num_layers", 0)
            or getattr(type(self), "_cached_num_layers", 0)
        )

        if num_layers > 0 and len(layer_types) == num_layers:
            for layer_idx, layer_type in enumerate(layer_types):
                if layer_type == "linear_attention":
                    mapping_list.extend(
                        [
                            _Qwen35LinearAttentionQKVMapping(
                                megatron_param=f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight",
                                hf_param=f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight",
                            ),
                            _Qwen35AttentionProjMapping(
                                megatron_param=f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight",
                                hf_param=f"model.language_model.layers.{layer_idx}.linear_attn.out_proj.weight",
                            ),
                            _Qwen35SyntheticNormMapping(
                                megatron_param=f"decoder.layers.{layer_idx}.self_attention.q_layernorm.weight",
                                hf_param=f"model.language_model.layers.{layer_idx}.input_layernorm.weight",
                            ),
                            _Qwen35SyntheticNormMapping(
                                megatron_param=f"decoder.layers.{layer_idx}.self_attention.k_layernorm.weight",
                                hf_param=f"model.language_model.layers.{layer_idx}.input_layernorm.weight",
                            ),
                        ]
                    )
                else:
                    mapping_list.extend(
                        [
                            _Qwen35FullAttentionQKVMapping(
                                megatron_param=f"decoder.layers.{layer_idx}.self_attention.linear_qkv.weight",
                                hf_param={
                                    "q": f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight",
                                    "k": f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight",
                                    "v": f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight",
                                },
                            ),
                            _Qwen35AttentionProjMapping(
                                megatron_param=f"decoder.layers.{layer_idx}.self_attention.linear_proj.weight",
                                hf_param=f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight",
                            ),
                            _DtypeSafeAutoMapping(
                                megatron_param=f"decoder.layers.{layer_idx}.self_attention.q_layernorm.weight",
                                hf_param=f"model.language_model.layers.{layer_idx}.self_attn.q_norm.weight",
                            ),
                            _DtypeSafeAutoMapping(
                                megatron_param=f"decoder.layers.{layer_idx}.self_attention.k_layernorm.weight",
                                hf_param=f"model.language_model.layers.{layer_idx}.self_attn.k_norm.weight",
                            ),
                        ]
                    )
        else:
            # Fallback to architecture-agnostic wildcard mappings if layer types
            # are unavailable in the loaded config.
            mapping_list.extend(
                [
                    _Qwen35LinearAttentionQKVMapping(
                        megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                        hf_param="model.language_model.layers.*.linear_attn.in_proj_qkv.weight",
                    ),
                    _DtypeSafeAutoMapping(
                        megatron_param="decoder.layers.*.self_attention.linear_proj.weight",
                        hf_param="model.language_model.layers.*.linear_attn.out_proj.weight",
                    ),
                    _DtypeSafeAutoMapping(
                        megatron_param="decoder.layers.*.self_attention.q_layernorm.weight",
                        hf_param="model.language_model.layers.*.self_attn.q_norm.weight",
                    ),
                    _DtypeSafeAutoMapping(
                        megatron_param="decoder.layers.*.self_attention.k_layernorm.weight",
                        hf_param="model.language_model.layers.*.self_attn.k_norm.weight",
                    ),
                ]
            )
        return MegatronMappingRegistry(*mapping_list)

