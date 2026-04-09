#!/usr/bin/env python3
"""Apply CI-only uv build overrides to a pyproject.toml file."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rewrite CI-sensitive uv extra-build-variables in pyproject.toml."
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        required=True,
        help="Path to the pyproject.toml file to rewrite in place.",
    )
    parser.add_argument(
        "--apex-parallel-build",
        type=int,
        required=True,
        help="Value to write for APEX_PARALLEL_BUILD.",
    )
    parser.add_argument(
        "--apex-nvcc-threads",
        type=int,
        required=True,
        help="Value to write for NVCC_APPEND_FLAGS=--threads <n>.",
    )
    return parser


def _replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise SystemExit(f"Expected exactly one {label} entry in pyproject.toml.")
    return updated


def main() -> int:
    args = _build_parser().parse_args()
    if args.apex_parallel_build <= 0:
        raise SystemExit("--apex-parallel-build must be a positive integer.")
    if args.apex_nvcc_threads <= 0:
        raise SystemExit("--apex-nvcc-threads must be a positive integer.")
    if not args.pyproject.is_file():
        raise SystemExit(f"pyproject file not found: {args.pyproject}")

    text = args.pyproject.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        r'APEX_PARALLEL_BUILD = "[0-9]+"',
        f'APEX_PARALLEL_BUILD = "{args.apex_parallel_build}"',
        "APEX_PARALLEL_BUILD",
    )
    text = _replace_once(
        text,
        r'NVCC_APPEND_FLAGS = "--threads [0-9]+"',
        f'NVCC_APPEND_FLAGS = "--threads {args.apex_nvcc_threads}"',
        "NVCC_APPEND_FLAGS",
    )
    args.pyproject.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
