---
name: train-sft
description: SFT training reference for the ART framework. Use when the user asks to create, write, or help with an SFT training script, fine-tune a model, train from a JSONL dataset, do distillation, or anything related to supervised fine-tuning.
---

# SFT Training Wizard

You are guiding the user through setting up Supervised Fine-Tuning (SFT) for a language model using the ART framework. Act as an interactive wizard: ask questions, validate inputs, and generate a complete runnable script.

## Step 1: Determine Training Scenario

Ask the user which SFT scenario they want using AskUserQuestion:

1. **Train from a JSONL file** — They have a dataset file with chat-formatted examples
2. **Train from inline trajectories** — They want to define a small number of examples directly in code
3. **Distillation** — They want to train a smaller model using outputs from a larger teacher model

## Step 2: Select and Validate Dataset (JSONL scenario)

**IMPORTANT**: Do NOT assume a dataset. Wait for the user to choose or provide one.

Search for `.jsonl` files in the working directory using Glob (`**/*.jsonl`). If there are more than 5 results, show only the 5 most recently modified files. Present the found files as options using AskUserQuestion, showing the full file path as each option label. Always include "Provide my own file path" as the last option. If no `.jsonl` files are found, just ask the user to provide a path.

Once the user has selected or provided a file path, validate it silently — do NOT ask for confirmation before running validation commands. Use this script to validate and count rows:

```python
import json, sys
ROLES = {"system", "user", "assistant", "developer", "tool", "function"}
errors = []
for i, line in enumerate(open(sys.argv[1]), 1):
    try:
        r = json.loads(line)
        msgs = r.get("input", r).get("messages", [])
        assert isinstance(msgs, list) and msgs, "no messages"
        for j, m in enumerate(msgs):
            assert m.get("role") in ROLES, f"messages[{j}]: invalid role {m.get('role')!r}"
            assert m.get("content") or m.get("function_call") or m.get("tool_calls"), f"messages[{j}]: no content"
        if "input" not in r:
            assert msgs[-1]["role"] == "assistant", "last message must be from assistant"
        tools = r.get("tools")
        if tools is not None:
            assert isinstance(tools, list), "tools must be a list"
    except Exception as e:
        errors.append(f"  Line {i}: {e}")
print(f"{len(errors)} error(s):\n" + "\n".join(errors) if errors else f"Valid! {i} rows")
sys.exit(1 if errors else 0)
```

The JSONL format supports these fields per row:
- **`messages`** (required): List of chat messages
- **`tools`** (optional): List of tool/function definitions for tool-call training
- **`response_format`** (optional): Structured output schema (not used during training, but useful as metadata)

Report the row count and validation result to the user. Do NOT read the whole dataset file. Do NOT name the dataset. If the format is wrong, help them fix it or convert their data.

## Step 3: Gather Base Parameters

Do NOT ask the user to review or confirm their answers after collecting them — just proceed to the next step.

- **Base model**: Recommend ONLY these models:
  - `OpenPipe/Qwen3-14B-Instruct`
  - `Qwen/Qwen3-30B-A3B-Instruct-2507`
  - `meta-llama/Llama-3.1-8B-Instruct`
- **Project name**: A name for this training project (default: `sft-project`)
- **Run name**: A static, descriptive name (e.g., `agent-001`, `pii-redactor-001`, `math-tutor-001`). Ask the user for a meaningful name. Do NOT generate random names.

For **distillation** also ask:
- **Teacher model**: The larger model to distill from (e.g., an OpenRouter model)
- **Teacher API base URL and key**: If using a third-party provider
- **Prompts**: What prompts to send to the teacher model

For **inline trajectories** also ask:
- **Training examples**: Help them construct message pairs (system/user/assistant turns)

## Step 4: Gather Hyperparameters

Only ask these AFTER the dataset has been validated and the row count is known. Use the row count to compute sensible defaults.

Run this Python snippet via Bash to compute defaults (replace `NUM_ROWS` with the actual row count). Do NOT show any formulas or calculation steps to the user — only show the final values.

```python
import math, sys
n = int(sys.argv[1])
epochs = max(1, min(10, round(10000 / n)))
batch_size = 2
total_steps = math.ceil(n * epochs / batch_size)
steps_per_epoch = math.ceil(n / batch_size)
warmup_steps = max(10, min(1000, round(steps_per_epoch * 0.05)))
warmup_ratio = round(warmup_steps / total_steps, 4)
print(f"epochs={epochs} batch_size={batch_size} lr=2e-4 schedule=linear warmup_ratio={warmup_ratio}")
```

Present the output values to the user, then ask using AskUserQuestion:
- **Use defaults (Recommended)** — show all values in the description
- **Customize** — adjust individual hyperparameters

If they choose "Customize", ask which parameters to change.

### For inline trajectories:
- **Learning rate**: (default: `5e-5`)
- **Batch size**: (default: `1`)

## Step 5: Generate the Training Script

Write a complete, runnable Python script. Use the patterns below. Every script MUST:
- Call `await backend.close()` at the end so the process doesn't hang
- Print post-training info and usage examples (see shared block below)

### Post-training block (append to ALL scripts before `backend.close()`):
```python
    # --- Training complete ---
    step = await model.get_step()
    inference_name = model.get_inference_name()
    client = model.openai_client()

    print("\n" + "=" * 60)
    print("SFT TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:          {inference_name}")
    print(f"  Base model:     <BASE_MODEL>")
    print(f"  Training step:  {step}")
    print(f"  Inference URL:  {client.base_url}")
    print(f"  W&B run:        https://wandb.ai/<YOUR_TEAM>/<PROJECT_NAME>/runs/<RUN_NAME>")
    print("=" * 60)

    print("\n--- Python usage (openai SDK) ---\n")
    print(f'''\
from openai import OpenAI

client = OpenAI(
    base_url="{client.base_url}",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="{inference_name}",
    messages=[
        {{"role": "user", "content": "Your prompt here"}},
    ],
)
print(response.choices[0].message.content)
''')

    print("--- curl usage ---\n")
    print(f'''\
curl {client.base_url}chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{inference_name}",
    "messages": [
      {{"role": "user", "content": "Your prompt here"}}
    ]
  }}'
''')

    await backend.close()
```

### JSONL file training pattern:
```python
"""SFT training script generated by /train-sft wizard."""
import asyncio
import art
from art.local import LocalBackend
from art.utils.sft import train_sft_from_file

async def main():
    backend = LocalBackend()
    model = art.TrainableModel(
        name="<RUN_NAME>",
        project="<PROJECT_NAME>",
        base_model="<BASE_MODEL>",
        _internal_config=art.dev.InternalModelConfig(
            engine_args={"gpu_memory_utilization": 0.7},
        ),
    )
    await model.register(backend)

    await train_sft_from_file(
        model=model,
        file_path="<FILE_PATH>",
        epochs=<EPOCHS>,
        batch_size=<BATCH_SIZE>,
        peak_lr=<PEAK_LR>,
        schedule_type="<SCHEDULE_TYPE>",
        warmup_ratio=<WARMUP_RATIO>,
        verbose=True,
    )

    # ... post-training block + backend.close() ...

if __name__ == "__main__":
    asyncio.run(main())
```

### Inline trajectories pattern:
```python
"""SFT training script generated by /train-sft wizard."""
import asyncio
import art
from art.local import LocalBackend

async def main():
    backend = LocalBackend()
    model = art.TrainableModel(
        name="<RUN_NAME>",
        project="<PROJECT_NAME>",
        base_model="<BASE_MODEL>",
        _internal_config=art.dev.InternalModelConfig(
            engine_args={"gpu_memory_utilization": 0.7},
        ),
    )
    await model.register(backend)

    trajectories = [
        art.Trajectory(
            messages_and_choices=[
                {"role": "user", "content": "<USER_MESSAGE>"},
                {"role": "assistant", "content": "<ASSISTANT_RESPONSE>"},
            ],
            tools=<TOOLS_OR_NONE>,
            reward=0.0,
        ),
    ]

    await model.train_sft(
        trajectories,
        config=art.TrainSFTConfig(
            learning_rate=<LEARNING_RATE>,
            batch_size=<BATCH_SIZE>,
        ),
        verbose=True,
    )

    # ... post-training block + backend.close() ...

if __name__ == "__main__":
    asyncio.run(main())
```

### Distillation pattern:
```python
"""Distillation SFT script generated by /train-sft wizard."""
import asyncio, os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import art
from art.local import LocalBackend

load_dotenv()

async def main():
    teacher_client = AsyncOpenAI(
        api_key=os.environ["<API_KEY_ENV_VAR>"],
        base_url="<TEACHER_API_BASE>",
    )
    prompts = ["<PROMPT_1>", "<PROMPT_2>"]

    trajectories = []
    for prompt in prompts:
        completion = await teacher_client.chat.completions.create(
            model="<TEACHER_MODEL>",
            messages=[{"role": "user", "content": prompt}],
        )
        trajectories.append(
            art.Trajectory(
                messages_and_choices=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion.choices[0].message.content},
                ],
                tools=<TOOLS_OR_NONE>,
                reward=0.0,
            )
        )

    backend = LocalBackend()
    model = art.TrainableModel(
        name="<RUN_NAME>",
        project="<PROJECT_NAME>",
        base_model="<STUDENT_BASE_MODEL>",
        _internal_config=art.dev.InternalModelConfig(
            engine_args={"gpu_memory_utilization": 0.7},
        ),
    )
    await model.register(backend)

    await model.train_sft(
        trajectories,
        config=art.TrainSFTConfig(learning_rate=2e-4),
        verbose=True,
    )

    # ... post-training block + backend.close() ...

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 6: Write and Offer to Run

1. Write the script to a file (suggest `sft_train.py`)
2. Ask the user if they want to run it now with `uv run python <script_path>`
3. If yes, run it **directly using the Bash tool** (do NOT delegate to a Task subagent) so training logs stream live to the user. Use a **2-minute timeout**. If it times out, check progress and decide whether to continue.
4. **GPU memory errors**: If training fails with OOM, lower `gpu_memory_utilization` in the existing `_internal_config` (e.g. from `0.7` to `0.5`).
5. **Stale GPU memory**: If available GPU memory looks too small, previous training runs may still be occupying memory. Before retrying, run `nvidia-smi` to check, and if needed kill leftover processes with `kill <pid>` to free memory.

## Important Notes

- LocalBackend requires a GPU.
