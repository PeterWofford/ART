from typing import Any

# Names kept as strings so this module stays lightweight and does not
# trigger heavy Megatron imports on its own.
_SUPPORTED_BRIDGE_NAMES = frozenset(
    {
        "Qwen3MoEBridge",
        "Qwen35MoEBridge",
    }
)


def _bridge_identity(model_bridge: object) -> str:
    cls = type(model_bridge)
    return f"{cls.__module__}.{cls.__name__}"


def _is_supported_bridge_type(model_bridge: object) -> bool:
    bridge_cls = type(model_bridge)
    if bridge_cls.__name__ in _SUPPORTED_BRIDGE_NAMES:
        return True
    return any(base.__name__ in _SUPPORTED_BRIDGE_NAMES for base in bridge_cls.__mro__[1:])


def require_supported_qwen_bridge(auto_bridge: Any) -> object:
    """Validate AutoBridge output against our local Qwen bridge contract.

    This keeps provider.py simple while letting us broaden support without
    repeatedly changing provider assertions.
    """

    model_bridge = getattr(auto_bridge, "_model_bridge", None)
    if model_bridge is None:
        raise AssertionError("AutoBridge resolved no _model_bridge instance")

    if _is_supported_bridge_type(model_bridge):
        return model_bridge

    architecture = getattr(auto_bridge, "_causal_lm_architecture", "<unknown>")
    supported = ", ".join(sorted(_SUPPORTED_BRIDGE_NAMES))
    raise AssertionError(
        "Unsupported bridge for Qwen provider path: "
        f"architecture={architecture} "
        f"resolved={_bridge_identity(model_bridge)} "
        f"supported=[{supported}]"
    )

