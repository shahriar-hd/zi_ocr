"""Stage 02 CLI - generate the text, one worker thread per API key.

Runs on the laptop. This is the only stage that costs money, so it is the one
most carefully made resumable: every sample lands in its own JSON file the
instant it arrives, and a re-run skips whatever is already on disk. A crash at
3 a.m. costs the in-flight requests and nothing else.

Work distribution is a single queue drained by one thread per key. No work is
partitioned up front, so a fast key simply takes more samples than a throttled
one, and a key that hits its quota drops out of the pool without stalling the
others.

Usage::

    python -m fa_ocr_lora.diverse.generation.run                  # all plans
    python -m fa_ocr_lora.diverse.generation.run --limit 20       # a taste
    python -m fa_ocr_lora.diverse.generation.run --dry-run        # no API calls
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

from ...dataset.common.io_utils import sample_rng, write_json
from ..config import DiverseConfig, load_diverse_config
from ..planning.sampler import Plan, read_plans
from .client import (
    Completion,
    FatalRequestError,
    GenerationError,
    KeyWorker,
    ResponseCache,
    load_api_keys,
    mask_key,
)
from .prompts import SYSTEM_PROMPT, build_prompt, prompt_hash

__all__ = ["RunCounters", "generate_texts", "main"]

LOGGER: Final[logging.Logger] = logging.getLogger("fa_diverse.generate")


class RunCounters:
    """Thread-safe tallies for one generation run.

    Attributes:
        skipped: Samples already on disk.
        written: Samples generated this run.
        cached: Samples served from the response cache.
        failed: Samples that exhausted their retries.
    """

    __slots__ = ("skipped", "written", "cached", "failed", "by_profile", "_lock")

    def __init__(self, skipped: int = 0) -> None:
        """Start the tallies.

        Args:
            skipped: Samples already on disk before the run began.
        """
        self.skipped = skipped
        self.written = 0
        self.cached = 0
        self.failed = 0
        self.by_profile: Counter[str] = Counter()
        self._lock = threading.Lock()

    def record(self, completion: Completion) -> int:
        """Record a successful sample.

        Args:
            completion: The reply that was written.

        Returns:
            The new ``written`` total.
        """
        with self._lock:
            self.written += 1
            if completion.cached:
                self.cached += 1
            self.by_profile[completion.profile] += 1
            return self.written

    def record_failure(self) -> None:
        """Record a sample that could not be generated."""
        with self._lock:
            self.failed += 1

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-serialisable summary."""
        return {
            "skipped": self.skipped,
            "written": self.written,
            "cached": self.cached,
            "failed": self.failed,
            "by_profile": dict(self.by_profile.most_common()),
        }


def text_path_for(config: DiverseConfig, plan: Plan) -> Path:
    """Return the per-sample output path.

    Sharded by the plan index's thousands digit, so a 150k-sample run does not
    put 150,000 files in one directory.

    Args:
        config: The active configuration.
        plan: The plan being generated.

    Returns:
        The JSON file this sample occupies.
    """
    shard = plan.index // 1000
    return config.paths.texts_dir / f"{shard:04d}" / f"{plan.plan_id}.json"


def _record(
    plan: Plan, prompt: str, completion: Completion, key: str
) -> dict[str, object]:
    """Build the per-sample record written to disk.

    The plan is embedded rather than referenced, so stage 03 and the molab
    render stage need only this one file per sample - which is what makes the
    Drive handoff a single directory rather than a join.

    Args:
        plan: The plan.
        prompt: The prompt that was sent.
        completion: The reply.
        key: Cache key, recorded for auditing.

    Returns:
        The record.
    """
    return {
        "plan_id": plan.plan_id,
        "index": plan.index,
        "text_raw": completion.text,
        "llm_model": completion.model,
        "profile": completion.profile,
        "temperature": completion.temperature,
        "finish_reason": completion.finish_reason,
        "prompt_hash": key,
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "attempts": completion.attempts,
        "cached": completion.cached,
        "plan": plan.as_dict(),
    }


def generate_texts(
    config: DiverseConfig,
    *,
    limit: int | None = None,
    overwrite: bool = False,
    keys_file: Path | None = None,
    dry_run: bool = False,
) -> RunCounters:
    """Generate text for every plan not already on disk.

    Args:
        config: The active configuration.
        limit: Process at most this many pending plans.
        overwrite: Regenerate samples already on disk.
        keys_file: Optional file of API keys.
        dry_run: Build and print prompts without calling the API.

    Returns:
        The run tallies.

    Raises:
        GenerationError: If no API key is available, or every key is exhausted
            before any sample was produced.
    """
    plans = read_plans(config.paths.plans_jsonl)
    profiles = {profile.name: profile for profile in config.generation.profiles}

    pending = [
        plan for plan in plans if overwrite or not text_path_for(config, plan).is_file()
    ]
    counters = RunCounters(skipped=len(plans) - len(pending))
    if limit is not None:
        pending = pending[:limit]

    if dry_run:
        LOGGER.info(
            "Dry run: %d pending plan(s); showing the first 2 prompts", len(pending)
        )
        for plan in pending[:2]:
            LOGGER.info("")
            LOGGER.info("=" * 70)
            LOGGER.info(
                "%s  category=%s  profile=%s", plan.plan_id, plan.category, plan.profile
            )
            LOGGER.info("=" * 70)
            LOGGER.info("%s", build_prompt(plan))
        return counters

    api_keys = load_api_keys(config.generation.api_key_env, keys_file)
    LOGGER.info(
        "%d plan(s): %d already on disk, %d to generate",
        len(plans),
        counters.skipped,
        len(pending),
    )
    LOGGER.info(
        "%d API key(s) [%s], %d profile(s), %d rpm/key, %d worker(s)/key",
        len(api_keys),
        ", ".join(mask_key(key) for key in api_keys),
        len(profiles),
        config.generation.rpm,
        config.generation.workers_per_key,
    )
    if not pending:
        return counters

    cache = ResponseCache(
        config.paths.llm_cache_dir, enabled=config.generation.cache_enabled
    )
    workers = [
        KeyWorker(
            key,
            base_url=config.generation.base_url,
            timeout=config.generation.timeout_s,
            rpm=config.generation.rpm,
        )
        for key in api_keys
    ]

    work: queue.Queue[Plan] = queue.Queue()
    for plan in pending:
        work.put(plan)

    stop = threading.Event()
    #: Keys retired because their quota is exhausted. A set of labels, not a
    #: list, because several threads share one key and would each try to retire
    #: it; the run ends only when every distinct key is in here.
    retired: set[str] = set()
    retire_lock = threading.Lock()
    #: How many times each plan has been handed back to the queue. A failing
    #: sample must be retried on another key - the first key may simply be
    #: throttled - but it must not be retried forever: a sample that fails on
    #: every key (a model that returns empty for that prompt, say) would
    #: otherwise bounce between workers and the run would never finish. The
    #: first version of this loop had exactly that bug and hung with three
    #: samples outstanding.
    requeues: Counter[str] = Counter()
    requeue_lock = threading.Lock()
    max_requeues = max(2, len(api_keys))
    log_every = max(1, len(pending) // 40)
    started = time.monotonic()

    def run_key(worker: KeyWorker) -> None:
        """Drain the queue using one key until it empties or the key dies.

        Args:
            worker: The key's worker.
        """
        while not stop.is_set():
            try:
                plan = work.get_nowait()
            except queue.Empty:
                return

            profile = profiles.get(plan.profile)
            if profile is None:
                LOGGER.error(
                    "%s names unknown profile %r; re-run stage 01 after editing "
                    "generation.profiles",
                    plan.plan_id,
                    plan.profile,
                )
                counters.record_failure()
                continue

            prompt = build_prompt(plan)
            key = prompt_hash(prompt, profile.name, config.seed)
            completion = cache.get(key)

            if completion is None:
                rng = sample_rng(config.seed, "generate", plan.index)
                try:
                    completion = worker.complete(
                        prompt,
                        system_prompt=SYSTEM_PROMPT,
                        profile=profile,
                        max_tokens=config.generation.max_tokens,
                        max_retries=config.generation.max_retries,
                        rng=rng,
                    )
                except FatalRequestError as error:
                    # The request is invalid, so no other key will serve it.
                    # Requeueing would only repeat the same 400 once per key.
                    counters.record_failure()
                    LOGGER.error("%s: %s", plan.plan_id, error)
                    continue
                except GenerationError as error:
                    if worker.cooling:
                        # The key is out of quota, not the sample's fault:
                        # hand it back unconditionally and retire this key.
                        work.put(plan)
                        with retire_lock:
                            retired.add(worker.label)
                            alive = len(workers) - len(retired)
                        LOGGER.warning(
                            "key %s retired (%d key(s) still working)",
                            worker.label,
                            alive,
                        )
                        if alive <= 0:
                            LOGGER.error("every key is in cooldown; stopping")
                            stop.set()
                        return
                    with requeue_lock:
                        requeues[plan.plan_id] += 1
                        attempts = requeues[plan.plan_id]
                    if attempts < max_requeues:
                        # Another key may do better; this one may just be
                        # having a bad minute.
                        work.put(plan)
                        LOGGER.warning(
                            "%s: %s -- requeued (%d/%d)",
                            plan.plan_id,
                            error,
                            attempts,
                            max_requeues,
                        )
                    else:
                        counters.record_failure()
                        LOGGER.error(
                            "%s: %s -- giving up after %d key(s)",
                            plan.plan_id,
                            error,
                            attempts,
                        )
                    continue
                except Exception:  # noqa: BLE001 - a thread must not die silently
                    LOGGER.exception(
                        "key %s: unexpected failure on %s", worker.label, plan.plan_id
                    )
                    counters.record_failure()
                    continue
                cache.put(key, completion)

            write_json(
                text_path_for(config, plan), _record(plan, prompt, completion, key)
            )
            written = counters.record(completion)
            if written % log_every == 0 or written == len(pending):
                elapsed = max(1e-6, time.monotonic() - started)
                LOGGER.info(
                    "  %d/%d written (%.1f/min, %d cached, %d failed)",
                    written,
                    len(pending),
                    written / elapsed * 60.0,
                    counters.cached,
                    counters.failed,
                )

    LOGGER.info(
        "Starting %d worker thread(s): %d key(s) x %d/key",
        len(workers) * config.generation.workers_per_key,
        len(workers),
        config.generation.workers_per_key,
    )
    threads = [
        threading.Thread(
            target=run_key,
            args=(worker,),
            name=f"key-{worker.label}-{slot}",
            daemon=True,
        )
        for worker in workers
        for slot in range(config.generation.workers_per_key)
    ]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; finishing in-flight requests. Re-run to resume.")
        stop.set()
        for thread in threads:
            thread.join(timeout=config.generation.timeout_s)
        raise

    unserved = work.qsize()
    if unserved:
        counters.failed += unserved
        LOGGER.warning("%d plan(s) left unserved; re-run to retry them", unserved)

    LOGGER.info("")
    LOGGER.info(
        "Per-key throttle waits: %s",
        ", ".join(f"{worker.label}: {worker.throttled}" for worker in workers),
    )
    LOGGER.info("Cache hits: %d", cache.hits)
    return counters


def consolidate(config: DiverseConfig) -> int:
    """Roll the per-sample JSON files into one ``texts.jsonl``.

    The per-sample files are what make the stage resumable; the single JSONL is
    what stage 03 streams and what the Drive mirror ships. Both are kept.

    Args:
        config: The active configuration.

    Returns:
        The number of records written.
    """
    target = config.paths.texts_jsonl
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for path in sorted(config.paths.texts_dir.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                LOGGER.warning("skipping unreadable %s: %s", path.name, error)
                continue
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(target)
    LOGGER.info("Wrote %s (%d record(s))", target, count)
    return count


def _report(config: DiverseConfig, counters: RunCounters) -> None:
    """Print and persist the stage 02 summary.

    Args:
        config: The active configuration.
        counters: The run tallies.
    """
    summary = counters.as_dict()
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("GENERATION SUMMARY")
    LOGGER.info("=" * 60)
    LOGGER.info("skipped (on disk)  %d", counters.skipped)
    LOGGER.info("written            %d", counters.written)
    LOGGER.info("  of which cached  %d", counters.cached)
    LOGGER.info("failed             %d", counters.failed)
    LOGGER.info("by profile         %s", summary["by_profile"])
    write_json(
        config.paths.work_dir / config.paths.GENERATION / "generation_summary.json",
        summary,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None, help="path to diverse_config.yaml")
    parser.add_argument(
        "--limit", type=int, default=None, help="process at most N pending plans"
    )
    parser.add_argument("--rpm", type=int, default=None, help="override generation.rpm")
    parser.add_argument(
        "--api-keys-file", default=None, help="file with one API key per line"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="regenerate samples already on disk"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore and do not write the response cache",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print prompts without calling the API"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    # The openai client logs one INFO line per HTTP request through httpx,
    # which at 150k samples buries every line this stage actually wants read.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    config = load_diverse_config(args.config)
    generation = config.generation
    if args.rpm is not None:
        generation = replace(generation, rpm=args.rpm)
    if args.no_cache:
        generation = replace(generation, cache_enabled=False)
    config = replace(config, generation=generation)

    try:
        counters = generate_texts(
            config,
            limit=args.limit,
            overwrite=args.overwrite,
            keys_file=Path(args.api_keys_file) if args.api_keys_file else None,
            dry_run=args.dry_run,
        )
    except GenerationError as error:
        LOGGER.error("%s", error)
        return 1
    if args.dry_run:
        return 0

    consolidate(config)
    _report(config, counters)
    return 1 if counters.written == 0 and counters.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
