"""Group-aware train/test assignment for the exported corpus.

A letter is rendered as one *or more* pages, and every page carries the same
letterhead, the same signature and the same synthesised table. Splitting the
corpus by page - shuffle the rows, slice off a tenth - therefore leaks: page 2
of a letter lands in ``test`` while page 1 trained the model, and the eval reads
several points better than the model deserves.

:class:`SplitPlan` fixes the unit of assignment at the *group* rather than the
row. Four properties follow, and each is a failure mode worth naming:

* **Every page of one letter shares a split**, because the hashed key is the
  letter, not the page.
* **The group is ``source_sample_id``, not ``sample_id``.** A plaintext file
  re-rendered under two sample ids - a second pass, an augmentation sweep -
  would otherwise put near-identical pages on both sides of the line.
  ``group_by`` is the knob for the case where per-render assignment is what is
  genuinely wanted.
* **Adding data never moves an existing sample**, because assignment reads only
  the id: no ordering, no corpus size, no row count. This matters directly here
  - plaintext is still being added, and a shuffle-and-slice split would
  silently re-deal the whole corpus on the next export.
* **``blake2b``, never :func:`hash`.** Python salts ``str.__hash__`` per
  process, so a plan built on :func:`hash` would assign a different split on
  every run and the leak would come back through the resume path.

The plan is content-addressed by :meth:`SplitPlan.fingerprint`, which the export
stores in its state file. A resume whose plan does not match that fingerprint is
an error: appending a second batch under changed fractions is exactly the leak
this module exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final, Iterable, Mapping

from ...config import (
    HF_GROUP_KEYS,
    HF_SPLIT_TOLERANCE,
    ConfigError,
    HuggingFaceConfig,
    normalize_split_fractions,
)

__all__ = ["SplitPlan"]

#: Bumped only if the hashing recipe changes. It is part of the fingerprint, so
#: a bump invalidates every stored plan rather than silently re-dealing samples
#: that a resume would then treat as already assigned.
FINGERPRINT_VERSION: Final[str] = "1"

#: Digest width in bytes. Eight bytes is more entropy than any corpus needs;
#: see :meth:`SplitPlan.position` for why the last split still has to catch the
#: top of the range.
_DIGEST_BYTES: Final[int] = 8


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """A deterministic, order-independent assignment of samples to splits.

    Attributes:
        fractions: ``(name, fraction)`` pairs, e.g. ``(("train", 0.9),
            ("test", 0.1))``. A mapping is accepted too and is converted on
            construction. Pairs are reordered into :data:`HF_SPLITS` order so
            the fingerprint does not depend on how the YAML happened to be
            written.
        group_by: Which id the assignment is keyed on, one of
            :data:`HF_GROUP_KEYS`.
        salt: Mixed into the hash. Changing it re-deals the entire corpus, so
            it belongs in the config file and should not be touched once a
            dataset has been published.
    """

    fractions: tuple[tuple[str, float], ...]
    group_by: str = "source_sample_id"
    salt: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fractions",
            normalize_split_fractions(self.fractions, field="split plan"),
        )
        if self.group_by not in HF_GROUP_KEYS:
            raise ConfigError(
                f"split group_by must be one of {sorted(HF_GROUP_KEYS)}, "
                f"got {self.group_by!r}"
            )
        if not isinstance(self.salt, str):
            raise ConfigError(f"split salt must be a string, got {self.salt!r}")

    @classmethod
    def from_mapping(
        cls,
        fractions: Mapping[str, float],
        *,
        group_by: str = "source_sample_id",
        salt: str = "",
    ) -> "SplitPlan":
        """Build a plan from a ``{name: fraction}`` mapping."""
        return cls(fractions=tuple(fractions.items()), group_by=group_by, salt=salt)

    @classmethod
    def from_settings(cls, settings: HuggingFaceConfig) -> "SplitPlan":
        """Build the plan the config asks for.

        The config validates the same things this class does, so a plan built
        this way cannot fail where a config load succeeded. It is a separate
        step rather than a config property because ``config/`` must not import
        anything under ``dataset/export/``.
        """
        return cls(
            fractions=settings.splits,
            group_by=settings.group_by,
            salt=settings.split_salt,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitPlan":
        """Rebuild a plan from its :meth:`as_dict` form in the state file."""
        try:
            fractions = payload["splits"]
        except KeyError as exc:
            raise ConfigError(f"stored split plan is missing {exc}") from exc
        if isinstance(fractions, Mapping):
            fractions = tuple(fractions.items())
        return cls(
            fractions=tuple(tuple(pair) for pair in fractions),
            group_by=str(payload.get("group_by", "source_sample_id")),
            salt=str(payload.get("salt", "")),
        )

    def names(self) -> tuple[str, ...]:
        """Return the split names in :data:`HF_SPLITS` order."""
        return tuple(name for name, _ in self.fractions)

    def fraction_of(self, name: str) -> float:
        """Return the requested fraction for one split.

        Raises:
            KeyError: If the split is not part of this plan.
        """
        for candidate, fraction in self.fractions:
            if candidate == name:
                return fraction
        raise KeyError(name)

    def group_key(self, *, sample_id: str, source_sample_id: str) -> str:
        """Return the id this plan hashes for the given sample."""
        if self.group_by == "sample_id":
            return sample_id
        return source_sample_id

    def position(self, key: str) -> float:
        """Map a group key onto ``[0.0, 1.0]``.

        Eight digest bytes exceed a float's 53-bit mantissa, so the very top of
        the range rounds up and this can return exactly ``1.0``. That is
        harmless because :meth:`assign` falls through to the last split rather
        than indexing by the cumulative bound, but it is the reason the walk is
        written that way.
        """
        digest = hashlib.blake2b(
            f"{self.salt}:{key}".encode("utf-8"), digest_size=_DIGEST_BYTES
        ).digest()
        return int.from_bytes(digest, "big") / float(1 << (_DIGEST_BYTES * 8))

    def assign(self, *, sample_id: str, source_sample_id: str) -> str:
        """Return the split this sample belongs to.

        Both ids are required so callers stay uniform across ``group_by``
        settings; only the one named by :attr:`group_by` is hashed.
        """
        position = self.position(
            self.group_key(sample_id=sample_id, source_sample_id=source_sample_id)
        )
        cumulative = 0.0
        for name, fraction in self.fractions[:-1]:
            cumulative += fraction
            if position < cumulative:
                return name
        # The last split catches the float tail of the cumulative walk as well
        # as the rounding described in `position`.
        return self.fractions[-1][0]

    def as_dict(self) -> dict[str, Any]:
        """Serialize the plan for the export state file and ``dataset_info``."""
        return {
            "version": FINGERPRINT_VERSION,
            "splits": [[name, fraction] for name, fraction in self.fractions],
            "group_by": self.group_by,
            "salt": self.salt,
            "fingerprint": self.fingerprint(),
        }

    def fingerprint(self) -> str:
        """Return a stable digest of everything that changes an assignment.

        Fractions are rounded before hashing, so a share arrived at as
        ``1 - 0.1`` fingerprints the same as one written ``0.9``. Nine decimal
        places is far finer than :data:`HF_SPLIT_TOLERANCE` lets two plans
        differ by and still be accepted as the same deal.
        """
        payload = json.dumps(
            {
                "version": FINGERPRINT_VERSION,
                "splits": [
                    [name, round(fraction, 9)] for name, fraction in self.fractions
                ],
                "group_by": self.group_by,
                "salt": self.salt,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def describe(self) -> str:
        """Return a one-line human summary for logs and error messages."""
        parts = ", ".join(f"{name}={fraction:g}" for name, fraction in self.fractions)
        salt = f", salt={self.salt!r}" if self.salt else ""
        return f"{parts} by {self.group_by}{salt}"
