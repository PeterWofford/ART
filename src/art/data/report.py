"""Report generation: JSON and human-readable output."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Any

from rich.console import Console
from rich.table import Table

from .models import Issue, Severity


def compute_readiness(
    conversations_count: int,
    issues: list[Issue],
    tools_defined: int = 0,
    tools_called: int = 0,
) -> dict[str, Any]:
    """Compute training readiness metrics from issues."""
    if conversations_count == 0:
        return {"verdict": "NOT_READY", "reason": "No conversations loaded"}

    # Conversations with errors
    error_convos: set[int] = set()
    warning_convos: set[int] = set()
    for issue in issues:
        if issue.conversation_index < 0:
            continue  # dataset-level issue
        if issue.severity == Severity.ERROR:
            error_convos.add(issue.conversation_index)
        elif issue.severity == Severity.WARNING:
            warning_convos.add(issue.conversation_index)

    usable_rate = 1.0 - len(error_convos) / conversations_count
    warning_rate = len(warning_convos) / conversations_count

    tool_coverage = (tools_called / tools_defined) if tools_defined > 0 else None

    # Check for dataset-level errors (e.g., missing tool calls entirely)
    dataset_errors = [
        i for i in issues if i.conversation_index < 0 and i.severity == Severity.ERROR
    ]

    # Verdict
    if dataset_errors:
        verdict = "NOT_READY"
    elif usable_rate >= 0.9:
        verdict = "READY"
    elif usable_rate < 0.5:
        verdict = "NOT_READY"
    else:
        verdict = "NEEDS_REVIEW"

    return {
        "verdict": verdict,
        "usable_conversation_rate": round(usable_rate, 3),
        "usable_conversations": conversations_count - len(error_convos),
        "conversations_with_errors": len(error_convos),
        "conversations_with_warnings": len(warning_convos),
        "warning_rate": round(warning_rate, 3),
        "tool_coverage": round(tool_coverage, 3) if tool_coverage is not None else None,
    }


def build_json_report(
    input_path: str,
    conversations_count: int,
    total_messages: int,
    issues: list[Issue],
    checks_run: list[str],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    """Build the full JSON report."""
    # Summarize by check
    by_check: dict[str, dict] = {}
    for issue in issues:
        cid = issue.check_id
        if cid not in by_check:
            by_check[cid] = {
                "severity": issue.severity.value,
                "count": 0,
                "affected_conversations": set(),
            }
        by_check[cid]["count"] += 1
        if issue.conversation_index >= 0:
            by_check[cid]["affected_conversations"].add(issue.conversation_index)

    # Convert sets to counts
    summary = {}
    for cid, data in sorted(by_check.items()):
        summary[cid] = {
            "severity": data["severity"],
            "count": data["count"],
            "affected_conversations": len(data["affected_conversations"]),
        }

    by_severity = Counter(i.severity.value for i in issues)

    return {
        "metadata": {
            "input_path": input_path,
            "total_conversations": conversations_count,
            "total_messages": total_messages,
            "checks_run": checks_run,
        },
        "readiness": readiness,
        "summary_by_severity": dict(by_severity),
        "summary_by_check": summary,
        "issues": [i.model_dump(mode="json") for i in issues],
    }


def print_human_report(
    console: Console,
    input_path: str,
    conversations_count: int,
    total_messages: int,
    issues: list[Issue],
    readiness: dict[str, Any],
) -> None:
    """Print a human-readable report to the console."""
    console.print()
    console.print("[bold]=== Training Data Validation Report ===[/bold]")
    console.print()
    console.print(f"  Input:   {input_path}")
    console.print(f"  Convos:  {conversations_count:,} ({total_messages:,} messages)")
    console.print()

    # Readiness
    verdict = readiness["verdict"]
    color = {"READY": "green", "NEEDS_REVIEW": "yellow", "NOT_READY": "red"}.get(
        verdict, "white"
    )
    console.print(f"  [bold {color}]--- Readiness: {verdict} ---[/bold {color}]")
    console.print()
    console.print(
        f"    Usable conversations:  {readiness['usable_conversation_rate'] * 100:.1f}%  "
        f"({readiness['usable_conversations']:,} / {conversations_count:,})"
    )
    if readiness.get("tool_coverage") is not None:
        console.print(
            f"    Tool coverage:         {readiness['tool_coverage'] * 100:.0f}%"
        )
    console.print()

    # Group issues by check, sort by severity then count
    by_check: dict[str, list[Issue]] = defaultdict(list)
    for issue in issues:
        by_check[issue.check_id].append(issue)

    severity_order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
    sorted_checks = sorted(
        by_check.items(),
        key=lambda x: (severity_order.get(x[1][0].severity, 9), -len(x[1])),
    )

    if sorted_checks:
        console.print("  [bold]--- Top Issues ---[/bold]")
        console.print()

        for check_id, check_issues in sorted_checks[:10]:
            sev = check_issues[0].severity
            sev_label = {
                "error": "[red]ERROR[/red]",
                "warning": "[yellow]WARN [/yellow]",
                "info": "[blue]INFO [/blue]",
            }
            label = sev_label.get(sev.value, sev.value.upper())

            # Count affected conversations
            affected = len(
                {
                    i.conversation_index
                    for i in check_issues
                    if i.conversation_index >= 0
                }
            )
            count_str = (
                f"{affected} convos" if affected > 0 else f"{len(check_issues)} issues"
            )

            console.print(f"    {label}  [bold]{check_id}[/bold]  {count_str}")

            # Show first issue's description as the detail
            desc = check_issues[0].description
            # For dataset-level issues, show the full description
            if check_issues[0].conversation_index < 0:
                for line in desc.split(". "):
                    console.print(f"           {line.strip()}")
            else:
                console.print(f"           {desc[:120]}")
            console.print()
    else:
        console.print("  [green]No issues found![/green]")
        console.print()
