import logging
import os

from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
import torch

from .bridge_patches import register_qwen35_bridge_patches
from .provider_config import configure_provider
from .provider_contract import require_supported_qwen_bridge
from .provider_flags import get_provider_flags

logger = logging.getLogger(__name__)
_MEGATRON_DEBUG = os.environ.get("ART_MEGATRON_DEBUG", "0") == "1"


def get_provider(model: str) -> GPTModelProvider:
    register_qwen35_bridge_patches()
    flags = get_provider_flags()
    if _MEGATRON_DEBUG:
        logger.warning("Resolving Megatron provider for model=%s", model)
    bridge = AutoBridge.from_hf_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if _MEGATRON_DEBUG:
        logger.warning(
            "AutoBridge resolved architecture=%s bridge=%s.%s",
            getattr(bridge, "_causal_lm_architecture", "<unknown>"),
            type(bridge._model_bridge).__module__,
            type(bridge._model_bridge).__name__,
        )
    require_supported_qwen_bridge(bridge)
    if flags.skip_hf_base_weights:
        logger.warning(
            "ART_MEGATRON_SKIP_HF_BASE_WEIGHTS=1 -> skipping HF base-weight import; "
            "training model is random-initialized for control-plane validation."
        )
    provider = bridge.to_megatron_provider(load_weights=not flags.skip_hf_base_weights)
    configure_provider(provider, flags=flags, bridge=bridge, debug=_MEGATRON_DEBUG)
    return provider
