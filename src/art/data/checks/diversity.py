"""Diversity analysis checks."""

from __future__ import annotations

from collections import Counter
import hashlib

from ..models import Conversation, Issue, Severity
from . import register


def _tool_sequence_fingerprint(conv: Conversation) -> str:
    """Create a fingerprint from the sequence of tool calls."""
    calls = []
    for msg in conv.messages:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append(tc.function.name)
    return ">".join(calls) if calls else "<no_tools>"


@register(
    "diversity.tool_call_coverage",
    "Tool call coverage",
    "Distribution of tool usage across the dataset",
)
def check_tool_coverage(conversations: list[tuple[int, Conversation]]) -> list[Issue]:
    all_defined: set[str] = set()
    call_counts: Counter[str] = Counter()

    for _, conv in conversations:
        if conv.tools:
            for t in conv.tools:
                all_defined.add(t.function.name)
        for msg in conv.messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    call_counts[tc.function.name] += 1

    if not call_counts:
        return []

    issues = []

    # Coverage: what fraction of defined tools are actually called
    if all_defined:
        called = set(call_counts.keys())
        uncalled = all_defined - called
        coverage = len(called & all_defined) / len(all_defined)

        if uncalled:
            issues.append(
                Issue(
                    check_id="diversity.tool_call_coverage",
                    severity=Severity.INFO,
                    conversation_index=-1,
                    description=(
                        f"Tool coverage: {coverage * 100:.0f}% ({len(called & all_defined)}/{len(all_defined)}). "
                        f"Unused tools: {', '.join(sorted(uncalled))}."
                    ),
                    context={
                        "coverage": coverage,
                        "tool_counts": dict(call_counts.most_common()),
                        "unused": sorted(uncalled),
                    },
                )
            )

    # Distribution skew
    total_calls = sum(call_counts.values())
    if total_calls > 0:
        top_tool, top_count = call_counts.most_common(1)[0]
        top_ratio = top_count / total_calls
        if top_ratio > 0.7 and len(call_counts) > 1:
            issues.append(
                Issue(
                    check_id="diversity.tool_call_coverage",
                    severity=Severity.WARNING,
                    conversation_index=-1,
                    description=(
                        f"Tool usage is heavily skewed: '{top_tool}' accounts for "
                        f"{top_ratio * 100:.0f}% of all tool calls ({top_count}/{total_calls}). "
                        "The model may underperform on less-represented tools."
                    ),
                )
            )

    return issues


@register(
    "diversity.scenario_fingerprinting",
    "Scenario diversity",
    "Measure unique conversation scenarios based on tool-call sequences",
)
def check_scenario_diversity(
    conversations: list[tuple[int, Conversation]],
) -> list[Issue]:
    if len(conversations) < 20:
        return []

    fingerprints: Counter[str] = Counter()
    for _, conv in conversations:
        fp = _tool_sequence_fingerprint(conv)
        fingerprints[fp] += 1

    n_unique = len(fingerprints)
    n_total = len(conversations)
    largest_cluster_fp, largest_cluster_size = fingerprints.most_common(1)[0]
    diversity_score = 1.0 - (largest_cluster_size / n_total)

    issues = []

    if n_unique < n_total * 0.1:
        severity = Severity.WARNING
    else:
        severity = Severity.INFO

    top_3 = fingerprints.most_common(3)
    top_desc = "; ".join(
        f"'{fp[:40]}' ({count} convos, {count / n_total * 100:.0f}%)"
        for fp, count in top_3
    )

    issues.append(
        Issue(
            check_id="diversity.scenario_fingerprinting",
            severity=severity,
            conversation_index=-1,
            description=(
                f"{n_unique} unique tool-call sequences from {n_total} conversations "
                f"(diversity score: {diversity_score:.2f}). "
                f"Largest clusters: {top_desc}."
            ),
            context={
                "unique_scenarios": n_unique,
                "total_conversations": n_total,
                "diversity_score": diversity_score,
                "top_clusters": [{"fingerprint": fp, "count": c} for fp, c in top_3],
            },
        )
    )

    return issues
