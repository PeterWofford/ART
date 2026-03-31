"""Check registry. Each check is a callable: conversations in, issues out."""

from __future__ import annotations

from typing import Callable

from ..models import Conversation, Issue

CheckFn = Callable[[list[tuple[int, Conversation]]], list[Issue]]

_registry: dict[str, dict] = {}


def register(check_id: str, name: str, description: str):
    """Decorator to register a check function."""

    def decorator(fn: CheckFn) -> CheckFn:
        _registry[check_id] = {
            "id": check_id,
            "name": name,
            "description": description,
            "fn": fn,
        }
        return fn

    return decorator


def get_all_checks() -> dict[str, dict]:
    return dict(_registry)


def run_checks(
    conversations: list[tuple[int, Conversation]],
    include: set[str] | None = None,
    exclude: set[str] | None = None,
) -> list[Issue]:
    """Run selected checks and return all issues."""
    issues: list[Issue] = []
    for check_id, check in _registry.items():
        if include is not None:
            if not any(
                check_id == inc or check_id.startswith(inc + ".") for inc in include
            ):
                continue
        if exclude is not None:
            if any(
                check_id == exc or check_id.startswith(exc + ".") for exc in exclude
            ):
                continue
        issues.extend(check["fn"](conversations))
    return issues


# Import check modules so they self-register
from . import (  # noqa: E402, F401
    completeness,
    diversity,
    pii,
    quality,
    schema,
    tool_integrity,
)
