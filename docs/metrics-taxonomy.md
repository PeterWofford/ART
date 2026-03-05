# Metrics Taxonomy (Phase 1)

Phase 1 introduces sectioned metric namespaces and hierarchical cost rollups.

## Sections

- `reward/*`
- `loss/*`
- `throughput/*`
- `costs/*`
- `time/*`
- `data/*`
- `train/*`, `val/*`, `test/*`

## Train Key Mapping

Current training code emits the following canonical keys:

- `reward` -> `reward/mean`
- `reward_std_dev` -> `reward/std_dev`
- `exception_rate` -> `reward/exception_rate`
- `group_metric_<k>` -> `reward/group_<k>`
- `policy_loss` / `loss` -> `loss/train`
- `entropy` -> `loss/entropy`
- `kl_div` -> `loss/kl_div`
- `kl_policy_ref` -> `loss/kl_policy_ref`
- `grad_norm` -> `loss/grad_norm`
- `learning_rate` -> `loss/learning_rate`
- `tokens_per_second` -> `throughput/train_tok_per_sec`
- `num_groups_submitted` -> `train/num_groups_submitted`
- `num_groups_trainable` -> `train/num_groups_trainable`
- `num_trajectories` -> `train/num_trajectories`
- `num_trainable_tokens` -> `train/num_trainable_tokens`
- `train_tokens` -> `data/step_trainer_tokens`
- `num_datums` -> `data/step_num_datums`
- `num_gradient_steps` -> `data/step_num_gradient_steps`

## Cost Rollups

Cost leaves can be logged with either:

- hierarchical keys, e.g. `costs/train/llm_judge/correctness`
- legacy component keys, e.g. `costs_prefill`, `costs_sample`

ART rolls costs up automatically:

- parent rollups (for example `costs/train`, `costs/all`)
- cumulative keys with `_cum` suffix (for example `costs/all_cum`)

## End-to-End Smoke Test

Run:

```bash
uv run python examples/metrics_taxonomy_smoke.py
```

This writes a local history file and, if `WANDB_API_KEY` is set, logs the same metrics to W&B.
