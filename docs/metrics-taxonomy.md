# Metrics Taxonomy (Phase 1-3)

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

## Metrics Added By ART

ART now emits the following metrics from library internals where the data is available:

- `reward/*` aggregates from `model.log(..., split="train")`
- `loss/*` from trainer backends
- `time/wall_clock_sec` and `training_step` on every logged row
- `time/step_trainer_s` for training calls
- `time/step_wall_s`, `time/step_actor_s`, `time/step_eval_s` from `PipelineTrainer`
- `data/step_num_scenarios`, `data/step_num_trajectories`, `data/step_num_groups_submitted`
- `data/step_num_groups_trainable` for train splits
- `data/cum_num_unique_scenarios` when scenario IDs are present in group or trajectory metadata
- `data/step_trainer_tokens` where the backend knows the trainer token count
- `throughput/cum_trainer_idle_s`, `throughput/cum_actor_idle_s`
- `throughput/avg_trainer_tok_per_s`, `throughput/avg_actor_tok_per_s` when both token and time inputs are available

Some metrics remain user-owned because ART cannot infer them reliably for every workflow, especially actor token usage outside the pipeline trainer.

## User Helpers

Use the builder helpers for step-level metrics that only user code can know:

```python
builder = model.metrics_builder()

with builder.measure("time/step_actor_s"):
    result = await run_rollouts()

builder.add_data(
    step_actor_tokens=result.actor_tokens,
    scenario_ids=result.scenario_ids,
)

builder.add_idle_times(step_actor_idle_s=result.actor_idle_s)
```

If these metrics are logged before the next `model.log(...)` flush, ART will also emit the cumulative and derived throughput metrics automatically.

## End-to-End Smoke Test

Run:

```bash
uv run python examples/metrics_taxonomy_smoke.py
```

This writes a local history file and, if `WANDB_API_KEY` is set, logs the same metrics to W&B.

## API Cost Decorator (Phase 2/3)

Use `@track_api_cost` to automatically write judge/API spend into `costs/{train|eval}/...`.

```python
from art.metrics import track_api_cost

@track_api_cost(
    source="llm_judge/correctness",
    provider="openai",
    prompt_price_per_million=1.0,
    completion_price_per_million=2.0,
)
async def run_judge(client, messages):
    return await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
```

Activate metric cost context while running train/eval logic:

```python
train_token = model.activate_metrics_context("train")
try:
    await run_judge(client, train_messages)
finally:
    train_token.var.reset(train_token)

eval_token = model.activate_metrics_context("eval")
try:
    await run_judge(client, eval_messages)
finally:
    eval_token.var.reset(eval_token)
```

The next `model.log(...)` flush for that step will include:

- `costs/train/llm_judge/correctness` (or `costs/eval/...`)
- hierarchical rollups like `costs/train`, `costs/all`
- cumulative keys like `costs/all_cum`

Built-in providers:

- OpenAI usage (`prompt_tokens`, `completion_tokens`)
- Anthropic usage (`input_tokens`, `output_tokens`)

You can override pricing per decorator call or configure builder-level defaults:

```python
builder = model.metrics_builder()
builder.register_token_pricing("openai", prompt_per_million=1.2, completion_per_million=4.8)
builder.register_cost_extractor("openai", lambda response: 0.001)  # optional custom extractor
```
