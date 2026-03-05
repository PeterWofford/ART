TRAIN_GRADIENT_STEPS_KEY = "data/step_num_gradient_steps"

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
