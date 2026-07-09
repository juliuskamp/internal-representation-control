"""Shared helpers for the jsonl-based, resumable scripts."""

import json
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Parse a jsonl file, skipping unparseable lines (a truncated trailing
    line from a killed writer) — used to resume appendable outputs."""
    with open(path) as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def run_bounded(tasks: Iterable, work: Callable, concurrency: int) -> Iterator:
    """Map `work` over `tasks` on `concurrency` worker threads, yielding
    results as they complete (completion order, not input order).

    Keeps ~2x `concurrency` tasks in flight, so producing the next task (which
    may do disk I/O) overlaps with execution instead of all tasks being
    materialized upfront."""
    tasks = iter(tasks)
    concurrency = max(1, concurrency)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        inflight = set()

        def refill() -> None:
            try:
                inflight.add(ex.submit(work, next(tasks)))
            except StopIteration:
                pass

        for _ in range(concurrency * 2):
            refill()
        while inflight:
            finished, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in finished:
                yield fut.result()
                refill()
