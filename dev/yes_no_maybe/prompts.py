from __future__ import annotations

from itertools import permutations
from typing import Iterable, Sequence


def with_quotes(word: str) -> str:
    return f"'{word}'"


def _format_words(words: Sequence[str], *, use_quotes: bool) -> str:
    if len(words) == 3:
        items = [with_quotes(word) if use_quotes else word for word in words]
        return ", ".join(items)
    if len(words) == 2:
        return f"{words[0]} or {words[1]}"
    if words:
        return words[0]
    raise ValueError("Prompt word list is empty.")


def build_prompt_variants() -> list[str]:
    prompts: list[str] = []
    for prefix in ["respond", "just respond"]:
        for use_quotes in [True, False]:
            for n in [3, 2]:
                for word_tuple in permutations(["yes", "no", "maybe"], n):
                    words = list(word_tuple)
                    prompts.append(f"{prefix} with {_format_words(words, use_quotes=use_quotes)}")
    return prompts


def slice_prompts(
    prompts: Iterable[str],
    *,
    offset: int = 0,
    limit: int = 0,
) -> list[str]:
    sliced = list(prompts)
    if offset > 0:
        sliced = sliced[offset:]
    if limit > 0:
        sliced = sliced[:limit]
    return sliced
