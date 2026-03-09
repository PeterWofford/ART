from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .trajectories import TrajectoryGroup

TRAIN_GRADIENT_STEPS_KEY = "data/step_num_gradient_steps"

_SCENARIO_ID_CANDIDATE_KEYS = (
    "scenario_id",
    "scenario_scenario_id",
    "scenario_idx",
    "scenario_scenario_idx",
)

TRAIN_METRIC_KEY_RENAMES = {
    "reward": "reward/mean",
    "reward_std_dev": "reward/std_dev",
    "exception_rate": "reward/exception_rate",
    "policy_loss": "loss/train",
    "loss": "loss/train",
    "entropy": "loss/entropy",
    "kl_div": "loss/kl_div",
    "kl_policy_ref": "loss/kl_policy_ref",
    "grad_norm": "loss/grad_norm",
    "learning_rate": "loss/learning_rate",
    "tokens_per_second": "throughput/train_tok_per_sec",
    "num_groups_submitted": "train/num_groups_submitted",
    "num_groups_trainable": "train/num_groups_trainable",
    "num_trajectories": "train/num_trajectories",
    "num_trainable_tokens": "train/num_trainable_tokens",
    "train_tokens": "data/step_trainer_tokens",
    "num_datums": "data/step_num_datums",
}


def rename_train_metric_key(metric: str) -> str:
    if metric.startswith("group_metric_"):
        return f"reward/group_{metric[len('group_metric_'):]}"
    return TRAIN_METRIC_KEY_RENAMES.get(metric, metric)


def rename_train_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {rename_train_metric_key(key): float(value) for key, value in metrics.items()}


@dataclass(frozen=True)
class TrajectoryBatchSummary:
    num_scenarios: int
    num_trajectories: int
    num_groups_submitted: int
    num_groups_trainable: int
    scenario_ids: list[str]


def summarize_trajectory_groups(
    trajectory_groups: Iterable[TrajectoryGroup],
) -> TrajectoryBatchSummary:
    groups = list(trajectory_groups)
    scenario_ids: list[str] = []
    seen_scenario_ids: set[str] = set()

    for group in groups:
        scenario_id = _extract_scenario_id(group)
        if scenario_id is None or scenario_id in seen_scenario_ids:
            continue
        seen_scenario_ids.add(scenario_id)
        scenario_ids.append(scenario_id)

    return TrajectoryBatchSummary(
        num_scenarios=len(groups),
        num_trajectories=sum(len(group.trajectories) + len(group.exceptions) for group in groups),
        num_groups_submitted=len(groups),
        num_groups_trainable=sum(1 for group in groups if _group_is_trainable(group)),
        scenario_ids=scenario_ids,
    )


def build_data_metrics_from_summary(
    summary: TrajectoryBatchSummary,
    *,
    include_trainable_groups: bool,
) -> dict[str, float]:
    metrics = {
        "data/step_num_scenarios": float(summary.num_scenarios),
        "data/step_num_trajectories": float(summary.num_trajectories),
        "data/step_num_groups_submitted": float(summary.num_groups_submitted),
    }
    if include_trainable_groups:
        metrics["data/step_num_groups_trainable"] = float(summary.num_groups_trainable)
    return metrics


def build_train_metrics_from_summary(
    summary: TrajectoryBatchSummary,
) -> dict[str, float]:
    return {
        "train/num_groups_submitted": float(summary.num_groups_submitted),
        "train/num_groups_trainable": float(summary.num_groups_trainable),
        "train/num_trajectories": float(summary.num_trajectories),
    }


def _group_is_trainable(group: TrajectoryGroup) -> bool:
    rewards = [trajectory.reward for trajectory in group.trajectories]
    return len(rewards) > 1 and len(set(rewards)) > 1


def _extract_scenario_id(group: TrajectoryGroup) -> str | None:
    for metadata in [group.metadata, *(trajectory.metadata for trajectory in group.trajectories)]:
        scenario_id = _extract_scenario_id_from_metadata(metadata)
        if scenario_id is not None:
            return scenario_id
    return None


def _extract_scenario_id_from_metadata(
    metadata: dict[str, Any],
) -> str | None:
    for key in _SCENARIO_ID_CANDIDATE_KEYS:
        value = metadata.get(key)
        if value is not None:
            return str(value)

    for key, value in metadata.items():
        if value is None:
            continue
        if key.endswith("scenario_id") or key.endswith("scenario_idx"):
            return str(value)
    return None
