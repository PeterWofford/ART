"""Validation functions for model configuration."""

from .model import InternalModelConfig


def is_dedicated_mode(config: InternalModelConfig) -> bool:
    """Return True if the config specifies dedicated mode (separate training and inference GPUs)."""
    return "trainer_gpu_ids" in config and "inference_gpu_ids" in config


def _parse_dedicated_int_arg(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer in dedicated mode")
    return value


def validate_dedicated_config(config: InternalModelConfig) -> None:
    """Validate dedicated mode GPU configuration.

    Raises ValueError if the configuration is invalid.
    Does nothing if neither trainer_gpu_ids nor inference_gpu_ids is set (shared mode).
    """
    has_trainer = "trainer_gpu_ids" in config
    has_inference = "inference_gpu_ids" in config

    if has_trainer != has_inference:
        raise ValueError(
            "trainer_gpu_ids and inference_gpu_ids must both be set or both unset"
        )

    if not has_trainer:
        return

    trainer_gpu_ids = config["trainer_gpu_ids"]
    inference_gpu_ids = config["inference_gpu_ids"]

    if not trainer_gpu_ids:
        raise ValueError("trainer_gpu_ids must be non-empty")

    if not inference_gpu_ids:
        raise ValueError("inference_gpu_ids must be non-empty")

    if set(trainer_gpu_ids) & set(inference_gpu_ids):
        raise ValueError("trainer_gpu_ids and inference_gpu_ids must not overlap")
    inference_gpu_count = len(inference_gpu_ids)

    if trainer_gpu_ids[0] != 0:
        raise ValueError(
            "trainer_gpu_ids must start at GPU 0 (training runs in-process)"
        )

    expected = list(range(len(trainer_gpu_ids)))
    if trainer_gpu_ids != expected:
        raise ValueError(
            "trainer_gpu_ids must be contiguous starting from 0 (e.g., [0], [0,1])"
        )

    # Reject settings that are incompatible with dedicated mode
    if config.get("init_args", {}).get("fast_inference"):
        raise ValueError(
            "fast_inference is incompatible with dedicated mode "
            "(dedicated mode runs vLLM as a subprocess, not in-process)"
        )

    if config.get("engine_args", {}).get("enable_sleep_mode"):
        raise ValueError(
            "enable_sleep_mode is incompatible with dedicated mode "
            "(dedicated mode runs vLLM on a separate GPU, sleep/wake is not needed)"
        )

    engine_args = config.get("engine_args", {})
    tensor_parallel_size = engine_args.get("tensor_parallel_size")
    tp_size = 1
    if tensor_parallel_size is not None:
        tp_size = _parse_dedicated_int_arg("tensor_parallel_size", tensor_parallel_size)

    if tp_size > 1:
        if tp_size != inference_gpu_count:
            raise ValueError(
                "tensor_parallel_size must equal len(inference_gpu_ids) "
                f"({inference_gpu_count}) in dedicated mode"
            )
        for key in ("data_parallel_size", "data_parallel_size_local"):
            value = engine_args.get(key)
            if value is None:
                continue
            parsed_value = _parse_dedicated_int_arg(key, value)
            if parsed_value > 1:
                raise ValueError(
                    f"{key} must be 1 or unset when tensor_parallel_size > 1 "
                    "in dedicated mode"
                )
        return

    for key in ("data_parallel_size", "data_parallel_size_local"):
        value = engine_args.get(key)
        if value is None:
            continue
        parsed_value = _parse_dedicated_int_arg(key, value)
        if parsed_value != inference_gpu_count:
            raise ValueError(
                f"{key} must equal len(inference_gpu_ids) ({inference_gpu_count}) "
                "in dedicated mode"
            )
