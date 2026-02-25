from __future__ import annotations

import logging

from megatron.bridge.models.gpt_provider import GPTModelProvider
import torch

from .attention_policy import resolve_attention_backend
from .provider_flags import ProviderFlags

logger = logging.getLogger(__name__)


def configure_provider(
    provider: GPTModelProvider,
    *,
    flags: ProviderFlags,
    bridge: object | None,
    debug: bool = False,
) -> None:
    provider.attention_backend = resolve_attention_backend(
        flags.attention_backend, bridge=bridge, debug=debug
    )
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
    if flags.skip_hf_base_weights:
        # In random-init mode there is no HF load hook, so provider config must
        # finalize init methods before model construction.
        provider.perform_initialization = True
        if hasattr(provider, "finalize"):
            provider.finalize()
    if debug:
        logger.warning(
            "Provider configured tp=%s pp=%s ep=%s etp=%s sequence_parallel=%s "
            "attention_backend=%s recompute=%s/%s/%s",
            provider.tensor_model_parallel_size,
            provider.pipeline_model_parallel_size,
            provider.expert_model_parallel_size,
            provider.expert_tensor_parallel_size,
            getattr(provider, "sequence_parallel", False),
            provider.attention_backend,
            provider.recompute_granularity,
            provider.recompute_method,
            provider.recompute_num_layers,
        )
