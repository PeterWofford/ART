import os
from dataclasses import dataclass

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_ATTN_BACKEND_VALUES = frozenset({"flash", "fused", "unfused", "local", "auto"})


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "0")
    return value.strip().lower() in _TRUE_VALUES


def _env_attention_backend() -> str:
    value = os.environ.get("ART_MEGATRON_ATTENTION_BACKEND", "auto")
    normalized = value.strip().lower()
    if normalized in _ATTN_BACKEND_VALUES:
        return normalized
    return "fused"


@dataclass(frozen=True)
class ProviderFlags:
    """Environment-driven flags for Megatron provider bring-up."""

    skip_hf_base_weights: bool
    attention_backend: str


def get_provider_flags() -> ProviderFlags:
    """Resolve provider flags from environment variables.

    ART_MEGATRON_SKIP_HF_BASE_WEIGHTS=1:
        Build Megatron model with random initialization and skip HF->Megatron
        base-weight import. Useful for control-plane and plumbing validation
        when a bridge is still under development.

    ART_MEGATRON_ATTENTION_BACKEND:
        Select Megatron attention backend. Supported values are:
        flash, fused, unfused, local, auto. Defaults to auto.

    NVTE_*:
        Transformer Engine backend knobs (e.g. NVTE_FUSED_ATTN,
        NVTE_FLASH_ATTN, NVTE_UNFUSED_ATTN) can further enable/disable
        specific kernels. Use NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 to inspect
        backend selection in logs.
    """

    return ProviderFlags(
        skip_hf_base_weights=_env_flag("ART_MEGATRON_SKIP_HF_BASE_WEIGHTS"),
        attention_backend=_env_attention_backend(),
    )

