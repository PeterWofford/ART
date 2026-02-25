from __future__ import annotations

import asyncio
import os
import re

import openai

import art

_ANSWER_RE = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)
_REWARD_BY_LABEL = {
    "yes": 0.5,
    "no": 0.75,
    "maybe": 1.0,
}
_DEBUG_COMPLETION_LIMIT = int(os.environ.get("DEBUG_COMPLETION_SAMPLES", "0"))
_debug_completion_logged = 0


def extract_label(content: str) -> str | None:
    match = _ANSWER_RE.search(content.strip().lower())
    if match is None:
        return None
    return match.group(1)


async def rollout(
    client: openai.AsyncOpenAI,
    model_name: str,
    prompt: str,
    *,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    timeout_s: float,
    temperature: float,
) -> art.Trajectory:
    messages: art.Messages = [
        {
            "role": "system",
            "content": (
                "Reply with exactly one lowercase word: yes, no, or maybe. "
                "Never include any other words."
            ),
        },
        {
            "role": "user",
            "content": f"/no_think {prompt}. Reply with one word: yes, no, or maybe.",
        },
    ]
    async with semaphore:
        chat_completion = await client.chat.completions.create(
            messages=messages,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout_s,
            stop=["\n"],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    choice = chat_completion.choices[0]
    content = choice.message.content
    assert isinstance(content, str)
    label = extract_label(content)
    global _debug_completion_logged
    if _debug_completion_logged < _DEBUG_COMPLETION_LIMIT:
        _debug_completion_logged += 1
        print(
            f"sample_completion[{_debug_completion_logged}]: "
            f"raw={content!r} parsed_label={label!r}"
        )
    reward = _REWARD_BY_LABEL[label] if label is not None else 0.0
    return art.Trajectory(
        messages_and_choices=[*messages, choice],
        reward=reward,
        metrics={
            "reward": reward,
            "matched_label": float(label is not None),
            "label_yes": float(label == "yes"),
            "label_no": float(label == "no"),
            "label_maybe": float(label == "maybe"),
        },
    )
