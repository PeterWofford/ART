from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run
    from art.model import TrainableModel

logger = logging.getLogger(__name__)

_ALIAS_CACHE: dict[tuple[str, str, str], set[str]] = {}
_IN_FLIGHT_ALIASES: set[tuple[str, str, str, str]] = set()
_ALIAS_LOCK = threading.Lock()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def checkpoint_upload_enabled() -> bool:
    return _env_bool("UPLOAD_CHECKPOINTS_TO_WANDB", False)


def checkpoint_upload_every_steps() -> int:
    value = os.environ.get("UPLOAD_CHECKPOINTS_EVERY_STEPS")
    if value is not None and value.strip():
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("UPLOAD_CHECKPOINTS_EVERY_STEPS must be positive")
        return parsed

    fallback_eval_every = os.environ.get("EVAL_EVERY")
    if fallback_eval_every is not None and fallback_eval_every.strip():
        parsed = int(fallback_eval_every)
        if parsed <= 0:
            raise ValueError("EVAL_EVERY must be positive when used as checkpoint upload fallback")
        return parsed

    return 20


def should_upload_checkpoint_step(step: int) -> bool:
    if not checkpoint_upload_enabled():
        return False
    every_steps = checkpoint_upload_every_steps()
    return step % every_steps == 0


def _artifact_cache_key(entity: str, project: str, artifact_name: str) -> tuple[str, str, str]:
    return (entity, project, artifact_name)


def _get_existing_aliases(entity: str, project: str, artifact_name: str) -> set[str]:
    cache_key = _artifact_cache_key(entity, project, artifact_name)
    cached = _ALIAS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import wandb

    aliases: set[str] = set()
    try:
        api = wandb.Api()
        collection = api.artifact_collection(
            type_name="lora",
            name=f"{entity}/{project}/{artifact_name}",
        )
        for version in collection.versions():
            aliases.update(version.aliases)
    except wandb.errors.CommError:
        pass

    _ALIAS_CACHE[cache_key] = aliases
    return aliases


def _mark_aliases_uploaded(entity: str, project: str, artifact_name: str, aliases: list[str]) -> None:
    cache_key = _artifact_cache_key(entity, project, artifact_name)
    with _ALIAS_LOCK:
        cached = _ALIAS_CACHE.setdefault(cache_key, set())
        cached.update(aliases)


def maybe_upload_checkpoint_to_wandb(
    *,
    model: "TrainableModel",
    checkpoint_path: str,
    step: int,
    force: bool = False,
    verbose: bool = False,
) -> bool:
    if not checkpoint_upload_enabled():
        return False
    if not force and not should_upload_checkpoint_step(step):
        return False
    if "WANDB_API_KEY" not in os.environ:
        logger.warning(
            "Checkpoint upload requested for %s step %s but WANDB_API_KEY is missing; skipping.",
            model.name,
            step,
        )
        return False

    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.is_dir():
        logger.warning(
            "Checkpoint upload requested for %s step %s but path does not exist: %s",
            model.name,
            step,
            checkpoint_dir,
        )
        return False

    import wandb

    wandb_run = model._get_wandb_run()
    if wandb_run is None:
        logger.warning(
            "Checkpoint upload requested for %s step %s but W&B run is unavailable; skipping.",
            model.name,
            step,
        )
        return False

    entity = (
        os.environ.get("UPLOAD_CHECKPOINTS_WANDB_ENTITY")
        or getattr(wandb_run, "entity", None)
        or model.entity
    )
    if not entity:
        api = wandb.Api()
        entity = api.default_entity
    if not entity:
        raise ValueError("Unable to resolve W&B entity for checkpoint upload")

    artifact_name = os.environ.get("UPLOAD_CHECKPOINTS_ARTIFACT_NAME", model.name)
    alias = f"step{step}"
    existing_aliases = _get_existing_aliases(entity, model.project, artifact_name)
    cache_key = _artifact_cache_key(entity, model.project, artifact_name)
    pending_key = (*cache_key, alias)
    with _ALIAS_LOCK:
        cached_aliases = _ALIAS_CACHE.setdefault(cache_key, set())
        if alias in existing_aliases or alias in cached_aliases or pending_key in _IN_FLIGHT_ALIASES:
            if verbose:
                print(f"Checkpoint artifact already exists or is in-flight for {artifact_name}:{alias}; skipping upload.")
            return False
        _IN_FLIGHT_ALIASES.add(pending_key)

    metadata: dict[str, object] = {"wandb.base_model": model.base_model}
    artifact = wandb.Artifact(
        artifact_name,
        type="lora",
        metadata=metadata,
        storage_region="coreweave-us",
    )
    artifact.add_dir(str(checkpoint_dir))
    aliases = [alias, "latest"]
    artifact_handle = wandb_run.log_artifact(artifact, aliases=aliases)
    upload_succeeded = False
    try:
        try:
            artifact_handle.wait()
        except ValueError as exc:
            if "Unable to fetch artifact" in str(exc):
                logger.warning("W&B artifact fetch warning after uploading %s:%s: %s", artifact_name, alias, exc)
            else:
                raise
        upload_succeeded = True
    finally:
        with _ALIAS_LOCK:
            _IN_FLIGHT_ALIASES.discard(pending_key)

    if upload_succeeded:
        _mark_aliases_uploaded(entity, model.project, artifact_name, aliases)
        if verbose:
            print(f"Uploaded checkpoint artifact {entity}/{model.project}/{artifact_name}:{alias}")
    return upload_succeeded
