from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any

from megatron.core.transformer.enums import AttnBackend

logger = logging.getLogger(__name__)

_NVTE_ENV_KEYS = (
    "NVTE_FUSED_ATTN",
    "NVTE_FLASH_ATTN",
    "NVTE_UNFUSED_ATTN",
    "NVTE_DEBUG",
    "NVTE_DEBUG_LEVEL",
)


def _config_to_dict(config: Any) -> dict[str, Any] | None:
    if config is None:
        return None
    if isinstance(config, dict):
        return config
    if hasattr(config, "to_dict"):
        try:
            return config.to_dict()
        except Exception:
            return None
    return getattr(config, "__dict__", None)


def _bridge_config_dict(bridge: Any) -> dict[str, Any] | None:
    for attr in ("_hf_config", "hf_config", "config", "_config"):
        cfg = getattr(bridge, attr, None)
        cfg_dict = _config_to_dict(cfg)
        if cfg_dict:
            return cfg_dict
    return None


def _extract_head_dim(config_dict: dict[str, Any]) -> int | None:
    text_cfg = config_dict.get("text_config") or config_dict
    head_dim = text_cfg.get("head_dim")
    if head_dim is not None:
        return int(head_dim)
    hidden_size = text_cfg.get("hidden_size")
    num_heads = text_cfg.get("num_attention_heads")
    if hidden_size is not None and num_heads:
        return int(hidden_size) // int(num_heads)
    return None


def _extract_attention_notes(config_dict: dict[str, Any]) -> dict[str, Any]:
    text_cfg = config_dict.get("text_config") or config_dict
    return {
        "head_dim": _extract_head_dim(config_dict),
        "num_attention_heads": text_cfg.get("num_attention_heads"),
        "num_key_value_heads": text_cfg.get("num_key_value_heads"),
        "full_attention_interval": text_cfg.get("full_attention_interval"),
        "layer_types": text_cfg.get("layer_types"),
    }


def _flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _nvte_env() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in _NVTE_ENV_KEYS}


def resolve_attention_backend(
    requested: str,
    *,
    bridge: Any | None,
    debug: bool = False,
) -> AttnBackend:
    normalized = requested.strip().lower()
    if not hasattr(AttnBackend, normalized):
        logger.warning(
            "Unknown ART_MEGATRON_ATTENTION_BACKEND=%s; falling back to fused.",
            requested,
        )
        normalized = "fused"

    backend = getattr(AttnBackend, normalized)
    config_dict = _bridge_config_dict(bridge) if bridge is not None else None
    nvte_env = _nvte_env()
    if config_dict:
        notes = _extract_attention_notes(config_dict)
        head_dim = notes.get("head_dim")
        if normalized == "fused" and head_dim and head_dim > 128:
            logger.warning(
                "Requested fused attention with head_dim=%s. Transformer Engine may "
                "disable fused kernels at this head dimension. Consider "
                "ART_MEGATRON_ATTENTION_BACKEND=flash or auto.",
                head_dim,
            )
        if normalized == "fused" and nvte_env.get("NVTE_FUSED_ATTN") == "0":
            logger.warning(
                "NVTE_FUSED_ATTN=0 disables fused attention kernels. Set it to 1 "
                "to allow fused backend selection."
            )
        if normalized == "flash" and not _flash_attn_available():
            logger.warning(
                "Requested flash attention but flash-attn is not installed."
            )
        if normalized == "flash" and nvte_env.get("NVTE_FLASH_ATTN") == "0":
            logger.warning(
                "NVTE_FLASH_ATTN=0 disables flash attention kernels. Set it to 1 "
                "to allow flash backend selection."
            )
        if debug:
            logger.warning(
                "Attention config: %s (flash_attn=%s nvte_env=%s)",
                notes,
                _flash_attn_available(),
                nvte_env,
            )
    elif debug:
        logger.warning(
            "Attention config: <unknown> (flash_attn=%s nvte_env=%s)",
            _flash_attn_available(),
            nvte_env,
        )
    return backend
