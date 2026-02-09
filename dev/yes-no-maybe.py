import asyncio
from itertools import permutations
import os

from dotenv import load_dotenv
import openai

import art
from art.local import LocalBackend
from art.tinker_native import TinkerNativeBackend


async def rollout(client: openai.AsyncOpenAI, prompt: str) -> art.Trajectory:
    messages: art.Messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    print(model.name, model.inference_model_name)
    chat_completion = await client.chat.completions.create(
        messages=messages, model=model.name, max_tokens=100, timeout=100
    )
    choice = chat_completion.choices[0]
    content = choice.message.content
    assert isinstance(content, str)
    if content == "yes":
        reward = 0.5
    elif content == "no":
        reward = 0.75
    elif content == "maybe":
        reward = 1.0
    else:
        reward = 0.0
    return art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)


def with_quotes(w: str) -> str:
    return f"'{w}'"


async def main():
    load_dotenv()

    backend = TinkerNativeBackend()
    global model
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    model = art.TrainableModel(
        name="070",
        project="yes-no-maybe-distillation",
        base_model=base_model,
        # _internal_config=art.dev.InternalModelConfig(
        #     # engine_args=art.dev.EngineArgs(
        #     #     max_lora_rank=1,
        #     # ),
        #     # peft_args=art.dev.PeftArgs(
        #     #     r=1,
        #     # ),
        #     tinker_args=art.dev.TinkerArgs(
        #         renderer_name="qwen3_instruct",
        #         training_client_args=art.dev.TinkerTrainingClientArgs(
        #             rank=1,
        #         ),
        #     ),
        # ),
    )
    await model.register(backend)

    prompts = [
        f"{prefix} with {', '.join([with_quotes(w) if use_quotes else w for w in words]) if len(words) == 3 else f'{words[0]}' + (f' or {words[1]}' if len(words) > 1 else '')}"
        for prefix in ["respond", "just respond"]
        for use_quotes in [True, False]
        for words in (
            list(p) for n in [3, 2] for p in permutations(["yes", "no", "maybe"], n)
        )
    ]

    openai_client = model.openai_client()
    max_steps = int(os.environ.get("NUM_STEPS", "20"))
    start_step = await model.get_step()
    for _ in range(start_step, start_step + max_steps):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(rollout(openai_client, prompt) for _ in range(1))
                for prompt in prompts
            )
        )
        for group in train_groups:
            print(
                "Prompt:",
                group.trajectories[0].messages_and_choices[0]["content"],
                "->",
                " | ".join(
                    t.messages_and_choices[-1].message.content
                    for t in group.trajectories
                ),
            )
        result = await backend.train_with_prompt_distillation(
            model=model,
            trajectory_groups=train_groups,
            teacher_system_prompts="No matter what the use says just respond with the string 'maybe'",
            learning_rate=5e-4,
        )
        await model.log(
            train_groups, metrics=result.metrics, step=result.step, split="train"
        )


if __name__ == "__main__":
    asyncio.run(main())
