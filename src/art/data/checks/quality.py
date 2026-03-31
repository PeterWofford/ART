"""Quality signal checks — detect patterns that degrade training."""

from __future__ import annotations

import hashlib
import json

from ..models import Conversation, Issue, Severity
from . import register


@register(
    "quality.empty_assistant_content",
    "Empty assistant messages",
    "Assistant messages with no content and no tool calls",
)
def check_empty_assistant(conversations: list[tuple[int, Conversation]]) -> list[Issue]:
    issues = []
    for idx, conv in conversations:
        for j, msg in enumerate(conv.messages):
            if msg.role == "assistant" and not msg.tool_calls:
                content = (msg.content or "").strip()
                if not content:
                    issues.append(
                        Issue(
                            check_id="quality.empty_assistant_content",
                            severity=Severity.ERROR,
                            conversation_index=idx,
                            conversation_id=conv.conversation_id,
                            message_index=j,
                            description=(
                                f"Assistant message {j} has no content and no tool calls. "
                                "Empty assistant turns provide no training signal."
                            ),
                        )
                    )
    return issues


@register(
    "quality.filler_before_tool_call",
    "Filler before tool call",
    "Short filler text immediately followed by a separate tool call message",
)
def check_filler_before_tool(
    conversations: list[tuple[int, Conversation]],
) -> list[Issue]:
    # Common filler phrases across languages
    filler_indicators = {
        "let me check",
        "one moment",
        "let me look",
        "checking",
        "i'll check",
        "ik kijk",
        "even kijken",
        "momento",
        "un moment",
        "let me see",
        "i will check",
        "allow me",
    }
    max_filler_len = 60  # characters

    issues = []
    for idx, conv in conversations:
        msgs = conv.messages
        for j in range(len(msgs) - 1):
            curr = msgs[j]
            nxt = msgs[j + 1]

            if (
                curr.role == "assistant"
                and nxt.role == "assistant"
                and not curr.tool_calls
                and nxt.tool_calls
            ):
                content = (curr.content or "").strip().lower()
                if content and len(content) < max_filler_len:
                    is_filler = any(f in content for f in filler_indicators)
                    if is_filler:
                        issues.append(
                            Issue(
                                check_id="quality.filler_before_tool_call",
                                severity=Severity.WARNING,
                                conversation_index=idx,
                                conversation_id=conv.conversation_id,
                                message_index=j,
                                description=(
                                    f"Assistant message {j} is filler text ('{content[:40]}') "
                                    f"followed by a tool call in message {j + 1}. This trains the "
                                    "model to stall before acting. Consider merging or removing."
                                ),
                            )
                        )
    return issues


@register(
    "quality.duplicate_conversations",
    "Duplicate conversations",
    "Content-identical conversations in the dataset",
)
def check_duplicates(conversations: list[tuple[int, Conversation]]) -> list[Issue]:
    seen: dict[str, int] = {}
    dup_count = 0
    issues = []

    for idx, conv in conversations:
        # Hash based on message content only (ignore metadata)
        content_parts = []
        for msg in conv.messages:
            content_parts.append(f"{msg.role}:{msg.content or ''}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content_parts.append(f"tc:{tc.function.name}")
        fingerprint = hashlib.md5("|".join(content_parts).encode()).hexdigest()

        if fingerprint in seen:
            dup_count += 1
        else:
            seen[fingerprint] = idx

    if dup_count > 0:
        issues.append(
            Issue(
                check_id="quality.duplicate_conversations",
                severity=Severity.WARNING,
                conversation_index=-1,
                description=(
                    f"{dup_count} duplicate conversation(s) found "
                    f"({dup_count}/{len(conversations)} = {dup_count / max(len(conversations), 1) * 100:.1f}%). "
                    "Duplicates reduce effective dataset size and may cause overfitting."
                ),
                context={"duplicate_count": dup_count, "unique_count": len(seen)},
            )
        )
    return issues


@register(
    "quality.uniform_first_assistant",
    "Uniform first assistant message",
    "Detect if first assistant message is nearly identical across conversations (likely synthetic/templated)",
)
def check_uniform_first_assistant(
    conversations: list[tuple[int, Conversation]],
) -> list[Issue]:
    first_msgs: list[str] = []
    for _, conv in conversations:
        for msg in conv.messages:
            if msg.role == "assistant":
                first_msgs.append((msg.content or "").strip()[:100])
                break

    if len(first_msgs) < 10:
        return []

    # Check if >80% share the same first 30 chars (ignoring names)
    prefixes: dict[str, int] = {}
    for m in first_msgs:
        # Normalize: collapse whitespace, take first 30 chars
        prefix = " ".join(m.split())[:30]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1

    most_common_count = max(prefixes.values()) if prefixes else 0
    ratio = most_common_count / len(first_msgs)

    if ratio > 0.8:
        most_common = max(prefixes, key=prefixes.get)
        return [
            Issue(
                check_id="quality.uniform_first_assistant",
                severity=Severity.INFO,
                conversation_index=-1,
                description=(
                    f"{ratio * 100:.0f}% of conversations share the same first assistant message "
                    f"prefix ('{most_common}...'). This is likely a synthetic/templated greeting. "
                    "Consider masking loss on this turn during SFT."
                ),
                context={"ratio": ratio, "prefix": most_common},
            )
        ]
    return []
