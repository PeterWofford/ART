import logging
import os

logger = logging.getLogger(__name__)
_MEGATRON_DEBUG = os.environ.get("ART_MEGATRON_DEBUG", "0") == "1"
_PATCHES_REGISTERED = False

# Known Qwen3.5 architecture identifiers seen in configs and model cards.
_QWEN35_ARCH_ALIASES = (
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoEForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoEForCausalLM",
)
_QWEN35_MODEL_TYPE_ALIASES = (
    "qwen3_5_moe",
    "qwen3_5_moe_text",
)


def register_qwen35_bridge_patches() -> None:
    """Register local bridge aliases for Qwen3.5 model architectures.

    This is intentionally idempotent. It lets us map Qwen3.5 architecture names to
    a local bridge implementation without editing megatron-bridge internals.
    """

    global _PATCHES_REGISTERED
    if _PATCHES_REGISTERED:
        return

    import transformers
    from transformers import Qwen3MoeConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from megatron.bridge.models.conversion.model_bridge import (
        get_model_bridge,
        register_bridge_implementation,
    )
    from megatron.core.models.gpt.gpt_model import GPTModel

    from .qwen35_bridge import Qwen35MoEBridge

    for model_type in _QWEN35_MODEL_TYPE_ALIASES:
        try:
            CONFIG_MAPPING.register(model_type, Qwen3MoeConfig, exist_ok=True)
        except TypeError:
            # Older transformers mapping may not accept exist_ok.
            try:
                CONFIG_MAPPING.register(model_type, Qwen3MoeConfig)
            except Exception:
                pass
        except Exception:
            pass
        if _MEGATRON_DEBUG:
            logger.warning(
                "Registered config alias model_type=%s -> %s",
                model_type,
                Qwen3MoeConfig.__name__,
            )

    # AutoBridge._causal_lm_architecture tries getattr(transformers, architecture_name).
    # Ensure those symbols exist for Qwen3.5 architecture names.
    for alias in _QWEN35_ARCH_ALIASES:
        if not hasattr(transformers, alias):
            placeholder_cls = type(alias, (), {"__module__": "transformers"})
            setattr(transformers, alias, placeholder_cls)
            if _MEGATRON_DEBUG:
                logger.warning(
                    "Registered transformers architecture placeholder %s",
                    alias,
                )

    registry = getattr(get_model_bridge, "_exact_types", {})
    for source_name in _QWEN35_ARCH_ALIASES:
        if source_name in registry:
            continue
        register_bridge_implementation(
            source=source_name,
            target=GPTModel,
            bridge_class=Qwen35MoEBridge,
        )
        if _MEGATRON_DEBUG:
            logger.warning(
                "Registered local Megatron bridge alias source=%s bridge=%s",
                source_name,
                Qwen35MoEBridge.__name__,
            )

    _PATCHES_REGISTERED = True
