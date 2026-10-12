"""The LLM transport: a key pool, per-key pacing, cooldown and a disk cache.

Generalised from ``dataset/synth/text_generator.py``, which already solved the
hard part. Its design is kept because it is right: **one worker thread per API
key, each holding at most one request in flight.** That makes rate limiting
trivial. A throttled key sleeps in place and the other keys keep generating;
there is no cursor, no cooldown table and no scheduler lock, because workers
share no scheduling state.

What is new here:

* **Profiles.** A request carries its own model, temperature and ``top_p``,
  drawn per sample in stage 01, so one run spans several generators. The
  letters pipeline had a single model and its output is recognisably uniform.
* **Cooldown with escalation.** A 429 or a quota error puts *that key* to sleep
  for a growing interval and records it, so a key that has exhausted a daily
  quota stops being tried every few seconds while its siblings continue.
  ``Retry-After`` is honoured when the endpoint sends it.
* **An on-disk cache** keyed by ``prompt_hash``. Stage 03 is re-run whenever a
  filter threshold changes, and without a cache that would mean paying for the
  same text repeatedly.

Keys come from ``NVIDIA_API_KEY``, which may itself hold a comma- or
whitespace-separated list, plus ``NVIDIA_API_KEY_1`` .. ``_32``, plus an
optional file. Nothing logs a key: :func:`mask_key` is the only way one reaches
a log line.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ..config import GenerationProfile

__all__ = [
    "Completion",
    "GenerationError",
    "KeyWorker",
    "ResponseCache",
    "load_api_keys",
    "mask_key",
]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

#: Numbered key variants probed in the environment.
MAX_NUMBERED_KEYS: Final[int] = 32

#: HTTP statuses worth retrying. 429 is rate limiting; 5xx is the endpoint
#: having a bad minute. A 400 or 404 is a request this code got wrong and
#: retrying it just wastes the quota.
RETRY_STATUS: Final[frozenset[int]] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: Statuses that mean "this key is out of budget", as opposed to "slow down".
#: These get the long cooldown, because retrying in four seconds is pointless.
QUOTA_STATUS: Final[frozenset[int]] = frozenset({402, 403})

#: Base and cap of the exponential backoff, in seconds.
BACKOFF_BASE: Final[float] = 4.0
BACKOFF_CAP: Final[float] = 120.0

#: Cooldown applied to a key that reports an exhausted quota. Long, because the
#: alternative is burning the run's wall-clock on a key that cannot serve it.
QUOTA_COOLDOWN: Final[float] = 900.0

#: Substrings in an error body that mean a quota rather than a rate limit.
_QUOTA_HINTS: Final[tuple[str, ...]] = (
    "quota",
    "credit",
    "insufficient",
    "exceeded your",
    "out of funds",
    "billing",
)


class GenerationError(RuntimeError):
    """Raised when a sample could not be generated after every attempt."""


def load_api_keys(env_var: str, keys_file: Path | None = None) -> list[str]:
    """Collect every API key available for this run.

    Three sources, so a pool can be supplied whichever way is convenient: the
    base variable (which may hold a comma- or whitespace-separated list),
    numbered variants ``<VAR>_1`` .. ``<VAR>_32``, and an optional file with one
    key per line.

    Args:
        env_var: Base environment variable name, e.g. ``NVIDIA_API_KEY``.
        keys_file: Optional file holding one key per line; ``#`` starts a
            comment.

    Returns:
        The keys, de-duplicated, in discovery order.

    Raises:
        GenerationError: If no key was found anywhere.
    """
    found: list[str] = []

    def absorb(raw: str | None) -> None:
        """Split one raw value on commas and whitespace into individual keys."""
        for piece in re.split(r"[,\s]+", (raw or "").strip()):
            if piece:
                found.append(piece)

    absorb(os.getenv(env_var))
    for index in range(1, MAX_NUMBERED_KEYS + 1):
        absorb(os.getenv(f"{env_var}_{index}"))
    if keys_file is not None:
        if not keys_file.is_file():
            raise GenerationError(f"API key file {keys_file} not found")
        for line in keys_file.read_text(encoding="utf-8").splitlines():
            absorb(line.split("#", 1)[0])

    keys = list(dict.fromkeys(found))
    if not keys:
        raise GenerationError(
            f"No API key found. Set {env_var} (one key, or several separated by "
            f"commas), or {env_var}_1 / {env_var}_2 ..., or pass --api-keys-file."
        )
    return keys


def mask_key(key: str) -> str:
    """Render an API key safely for logs.

    Args:
        key: The full key.

    Returns:
        A short masked form such as ``...a1b2``.
    """
    return f"...{key[-4:]}" if len(key) > 4 else "..."


@dataclass(frozen=True, slots=True)
class Completion:
    """One model reply and its accounting.

    Attributes:
        text: The reply content, stripped.
        model: Model that produced it.
        profile: Profile name used.
        temperature: Temperature used.
        finish_reason: Endpoint's finish reason; ``"cache"`` for a cache hit.
        prompt_tokens: Prompt token count, ``0`` when unreported.
        completion_tokens: Completion token count, ``0`` when unreported.
        attempts: Attempts spent, including the successful one.
        cached: Whether this came from the on-disk cache.
    """

    text: str
    model: str
    profile: str
    temperature: float
    finish_reason: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 1
    cached: bool = False


class ResponseCache:
    """On-disk cache of model replies, keyed by prompt hash.

    Sharded two levels deep by the key's first four hex characters, so a
    150k-sample run does not put 150,000 files in one directory - which makes
    ``ls`` unusable and some filesystems slow.
    """

    __slots__ = ("root", "enabled", "_hits", "_lock")

    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        """Prepare the cache directory.

        Args:
            root: Cache root.
            enabled: Whether reads and writes happen at all.
        """
        self.root = root
        self.enabled = enabled
        self._hits = 0
        self._lock = threading.Lock()
        if enabled:
            root.mkdir(parents=True, exist_ok=True)

    @property
    def hits(self) -> int:
        """Return the number of cache hits so far."""
        return self._hits

    def _path(self, key: str) -> Path:
        """Return the file holding one cache entry.

        Args:
            key: Prompt hash.

        Returns:
            The sharded path.
        """
        return self.root / key[:2] / key[2:4] / f"{key}.json"

    def get(self, key: str) -> Completion | None:
        """Look up a cached reply.

        Args:
            key: Prompt hash.

        Returns:
            The cached completion, or ``None`` on a miss or an unreadable
            entry. A corrupt entry is a miss, not an error: the cost of
            re-requesting is one API call, and the cost of raising is the run.
        """
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        with self._lock:
            self._hits += 1
        return Completion(
            text=payload.get("text", ""),
            model=payload.get("model", ""),
            profile=payload.get("profile", ""),
            temperature=float(payload.get("temperature", 0.0)),
            finish_reason="cache",
            prompt_tokens=int(payload.get("prompt_tokens", 0)),
            completion_tokens=int(payload.get("completion_tokens", 0)),
            attempts=0,
            cached=True,
        )

    def put(self, key: str, completion: Completion) -> None:
        """Store a reply.

        Args:
            key: Prompt hash.
            completion: The reply to store.
        """
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(
                    {
                        "text": completion.text,
                        "model": completion.model,
                        "profile": completion.profile,
                        "temperature": completion.temperature,
                        "finish_reason": completion.finish_reason,
                        "prompt_tokens": completion.prompt_tokens,
                        "completion_tokens": completion.completion_tokens,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as error:
            LOGGER.debug("cache write failed for %s: %s", key, error)


def _retry_after_seconds(error: object) -> float | None:
    """Read a ``Retry-After`` header from an API error, if present.

    Args:
        error: The exception raised by the client.

    Returns:
        The requested delay in seconds, or ``None``.
    """
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _is_quota(status: int | None, message: str) -> bool:
    """Decide whether a failure means an exhausted quota.

    A rate limit says "slow down" and clears in seconds; a quota says "stop"
    and clears in hours. Treating them alike either hammers a dead key or
    needlessly idles a live one, so the distinction is worth guessing at from
    the message when the status alone is ambiguous.

    Args:
        status: HTTP status, if known.
        message: The error text, lowercased by the caller or not.

    Returns:
        ``True`` when the key appears to be out of budget.
    """
    if status in QUOTA_STATUS:
        return True
    lowered = message.lower()
    return status == 429 and any(hint in lowered for hint in _QUOTA_HINTS)


class KeyWorker:
    """One API key, its own client, and the requests it issues sequentially.

    Each key gets exactly one worker and a worker holds one request in flight
    at a time, so rate limiting needs no shared state: a throttled key sleeps
    in place while every other key keeps working.

    Attributes:
        label: Masked key, safe to log.
        throttled: Times this key has been made to wait.
        cooldown_until: Monotonic time before which this key should not be
            used, set when a quota error is seen.
    """

    __slots__ = (
        "label",
        "throttled",
        "cooldown_until",
        "_client",
        "_config",
        "_min_interval",
        "_next_slot",
    )

    def __init__(self, key: str, *, base_url: str, timeout: float, rpm: int) -> None:
        """Bind a client to one API key.

        Args:
            key: The API key this worker owns.
            base_url: OpenAI-compatible endpoint.
            timeout: Per-request socket timeout in seconds.
            rpm: Requests per minute this key may issue.

        Raises:
            GenerationError: If the ``openai`` package is unavailable.
        """
        try:
            from openai import OpenAI
        except ImportError as error:
            raise GenerationError(
                "the openai package is required: pip install 'fa-ocr-lora[text]'"
            ) from error

        self.label = mask_key(key)
        self.throttled = 0
        self.cooldown_until = 0.0
        self._client = OpenAI(
            base_url=base_url,
            api_key=key,
            timeout=timeout,
            max_retries=0,  # retries, backoff and cooldown are handled here
        )
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next_slot = 0.0

    @property
    def cooling(self) -> bool:
        """Report whether this key is currently in a quota cooldown."""
        return time.monotonic() < self.cooldown_until

    def _pace(self) -> None:
        """Sleep until this key is allowed to issue its next request."""
        now = time.monotonic()
        if self.cooldown_until > now:
            time.sleep(self.cooldown_until - now)
        if self._min_interval <= 0.0:
            return
        wait = self._next_slot - time.monotonic()
        if wait > 0.0:
            time.sleep(wait)
        self._next_slot = time.monotonic() + self._min_interval

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str,
        profile: GenerationProfile,
        max_tokens: int,
        max_retries: int,
        rng: random.Random,
    ) -> Completion:
        """Request one completion, waiting out this key's own limits.

        Args:
            prompt: The user prompt.
            system_prompt: The system prompt.
            profile: Model, temperature and ``top_p`` for this request.
            max_tokens: Output ceiling.
            max_retries: Attempts before giving up.
            rng: Per-sample RNG, used to jitter the backoff so a fleet of
                workers released by the same 429 does not retry in lockstep.

        Returns:
            The completion.

        Raises:
            GenerationError: If every attempt failed.
        """
        last_error = "no attempt was made"
        for attempt in range(max_retries):
            self._pace()
            try:
                response = self._client.chat.completions.create(
                    model=profile.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=profile.temperature,
                    top_p=profile.top_p,
                    max_tokens=max_tokens,
                    # Deliberately no response_format: this endpoint's
                    # guided-JSON decoder truncates strings at 512 characters,
                    # and the reply here is plain text anyway.
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except Exception as error:  # noqa: BLE001 - classified below
                status = getattr(error, "status_code", None)
                message = str(error)
                if _is_quota(status, message):
                    self.cooldown_until = time.monotonic() + QUOTA_COOLDOWN
                    self.throttled += 1
                    LOGGER.warning(
                        "key %s: quota exhausted (%s); cooling down %.0fs while the "
                        "other keys continue",
                        self.label,
                        status or "no status",
                        QUOTA_COOLDOWN,
                    )
                    raise GenerationError(f"key {self.label} is out of quota") from error
                if status is not None and status not in RETRY_STATUS:
                    raise GenerationError(
                        f"API rejected the request ({status}): {message}"
                    ) from error
                delay = _retry_after_seconds(error) or min(
                    BACKOFF_CAP, BACKOFF_BASE * 2**attempt
                )
                delay += rng.uniform(0.0, 1.5)
                self.throttled += 1
                last_error = (
                    f"transport error ({status or 'connection'}): {message[:160]}"
                )
                LOGGER.warning(
                    "key %s: %s -- waiting %.1fs", self.label, last_error, delay
                )
                time.sleep(delay)
                continue

            if not response.choices:
                last_error = "reply carried no choices"
                LOGGER.debug(
                    "key %s: %s (attempt %d)", self.label, last_error, attempt + 1
                )
                continue

            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            if choice.finish_reason == "length" and not text:
                # A reasoning model can spend its whole budget on hidden
                # tokens and return nothing. Say so plainly rather than
                # reporting an empty sample.
                last_error = (
                    f"reply hit the {max_tokens}-token ceiling with no content; "
                    f"{profile.model} may be spending the budget on reasoning"
                )
                LOGGER.warning("key %s: %s", self.label, last_error)
                continue
            if not text:
                last_error = "model returned an empty reply"
                LOGGER.debug(
                    "key %s: %s (attempt %d)", self.label, last_error, attempt + 1
                )
                continue

            usage = getattr(response, "usage", None)
            return Completion(
                text=text,
                model=profile.model,
                profile=profile.name,
                temperature=profile.temperature,
                finish_reason=str(choice.finish_reason or "stop"),
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                attempts=attempt + 1,
            )

        raise GenerationError(f"after {max_retries} attempt(s): {last_error}")
