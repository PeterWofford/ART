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

## Backend Output

ART backends emit canonical sectioned keys directly. The canonical training keys include:

- `reward/mean`
- `reward/std_dev`
- `reward/exception_rate`
- `reward/group_<k>`
- `loss/train`
- `loss/entropy`
- `loss/kl_div`
- `loss/kl_policy_ref`
- `loss/grad_norm`
- `loss/learning_rate`
- `train/num_groups_submitted`
- `train/num_groups_trainable`
- `train/num_trajectories`
- `train/num_trainable_tokens`
- `data/step_trainer_tokens`
- `data/step_num_datums`
- `data/step_num_gradient_steps`

## Cost Rollups

Cost leaves can be logged with hierarchical keys, for example:

- hierarchical keys, e.g. `costs/train/llm_judge/correctness`

ART rolls costs up automatically:

- parent rollups (for example `costs/train`, `costs/all`)
- cumulative keys under the `cum/` namespace (for example `costs/cum/all`)

## Metrics Added By ART

ART now emits the following metrics from library internals where the data is available:

- `reward/*` aggregates from `model.log(..., split="train")`
- `loss/*` from trainer backends
- `time/wall_clock_sec` and `training_step` on every logged row
- `time/step_trainer_s` for training calls
- `time/step_wall_s` from `PipelineTrainer` and `LocalBackend` train-step logs
- `time/step_actor_s`, `time/step_eval_s` from `PipelineTrainer`
- `data/step_num_scenarios`, `data/step_num_trajectories`, `data/step_num_groups_submitted`
- `data/step_num_groups_trainable` for train splits
- `data/cum/num_unique_scenarios` when `scenario_id` is present in group or trajectory metadata
- `data/step_trainer_tokens` where the backend knows the trainer token count
- `costs/gpu` on `LocalBackend` train-step logs when ART can resolve GPU pricing
- `throughput/cum/trainer_idle_s`, `throughput/cum/actor_idle_s`
- `throughput/avg_trainer_tok_per_s`, `throughput/avg_actor_tok_per_s` when both token and time inputs are available

Some metrics remain user-owned because ART cannot infer them reliably for every workflow, especially actor token usage outside the pipeline trainer.

For automatic GPU cost on `LocalBackend`, ART currently auto-detects H200s at
$3/hour per GPU. For other GPU types, pass `gpu_cost_per_hour_usd=...` to
`LocalBackend(...)` if you want ART to emit `costs/gpu` instead of skipping it.

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

## API Cost Decorator (Phase 2/3)

Use `@track_api_cost` to automatically write judge/API spend into `costs/{train|eval}/...`.

```python
from art.metrics import track_api_cost

@track_api_cost(
    source="llm_judge/correctness",
    provider="openai",
    model_name="openai/gpt-oss-20b",
)
async def run_judge(client, messages):
    return await client.chat.completions.create(
        model="gpt-oss-20b",
        messages=messages,
    )
```

Activate metric cost context while running train/eval logic:

```python
with model.metrics_builder("train").activate_context():
    await run_judge(client, train_messages)

with model.metrics_builder("eval").activate_context():
    await run_judge(client, eval_messages)
```

The next `model.log(...)` flush for that step will include:

- `costs/train/llm_judge/correctness` (or `costs/eval/...`)
- hierarchical rollups like `costs/train`, `costs/all`
- cumulative keys like `costs/cum/all`

Built-in usage extraction:

- OpenAI usage (`prompt_tokens`, `completion_tokens`)
- Anthropic usage (`input_tokens`, `output_tokens`)

Pricing is model-aware by default. ART will use the configured model pricing from
`art.costs.MODEL_PRICING` when it can resolve a concrete model name, and it
raises instead of guessing when pricing is missing.

You can still override pricing per decorator call or register model-specific
pricing on the builder:

```python
builder = model.metrics_builder()
builder.register_model_pricing(
    "anthropic/my-custom-judge",
    prompt_per_million=1.2,
    completion_per_million=4.8,
)
builder.register_cost_extractor("openai", lambda response: 0.001)  # optional custom extractor
```
