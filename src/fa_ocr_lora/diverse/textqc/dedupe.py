"""Near-duplicate detection over the cleaned corpus.

Exact duplicates are easy and rare; the real problem is a model that writes the
*same paragraph with three words changed* a hundred times. That is invisible to
a hash, it inflates the corpus without adding information, and because the
duplicates share a plan's genre they concentrate in one part of the
distribution - so the model over-fits exactly the slice that looked largest.

MinHash over character shingles is implemented here rather than imported.
``datasketch`` would do it, but it is another dependency for a hundred lines
of arithmetic, and a self-contained implementation can be seeded from the run
seed - which matters, because a randomly-seeded hash family makes the
deduplication decision differ between runs over the same corpus.

Two thresholds, because within a genre a model repeats itself far more and a
looser global threshold would let those through: ``dedupe_threshold`` applies
across the corpus and ``dedupe_threshold_same_genre`` applies within one genre.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from typing import Final

from ..text_policy import ZWNJ, normalise_for_eval

__all__ = ["MinHashIndex", "shingles", "signature"]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

#: A 61-bit Mersenne prime, the modulus of the hash family. Large enough that
#: collisions between distinct shingles are negligible at corpus scale.
_PRIME: Final[int] = (1 << 61) - 1

#: 32-bit mask applied to the base shingle hash before the permutations.
_MASK: Final[int] = (1 << 32) - 1


def shingles(text: str, size: int) -> frozenset[str]:
    """Build the character shingle set of a text.

    Character shingles rather than word shingles, deliberately. Persian
    compounds join with ZWNJ and inflection changes word boundaries, so two
    texts that a reader calls identical can share few whole words; character
    shingles are robust to both.

    Args:
        text: The text, which is normalised here so two samples differing only
            in digit script or Arabic yeh compare equal.
        size: Characters per shingle.

    Returns:
        The distinct shingles. Empty when the text is shorter than one
        shingle, which makes every very short text non-duplicable - correct,
        because a six-word banner legitimately repeats common phrases and
        deduplicating banners would empty the category.
    """
    body = normalise_for_eval(text).replace(ZWNJ, "")
    body = " ".join(body.split())
    if len(body) < size:
        return frozenset()
    return frozenset(body[index : index + size] for index in range(len(body) - size + 1))


def _hash_shingle(shingle: str) -> int:
    """Hash one shingle to 32 bits.

    Args:
        shingle: The shingle.

    Returns:
        A 32-bit integer. ``blake2b`` rather than :func:`hash`, because the
        built-in hash is salted per process and would make deduplication
        non-reproducible across runs.
    """
    digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") & _MASK


def _permutations(count: int, seed: int) -> list[tuple[int, int]]:
    """Build the ``(a, b)`` coefficients of the hash family.

    Derived from the run seed so the same corpus deduplicates identically on
    every machine and every re-run.

    Args:
        count: Number of permutations, i.e. the signature length.
        seed: The run seed.

    Returns:
        ``count`` coefficient pairs with ``a`` non-zero.
    """
    out: list[tuple[int, int]] = []
    for index in range(count):
        material = hashlib.blake2b(
            f"{seed}:minhash:{index}".encode(), digest_size=16
        ).digest()
        first = int.from_bytes(material[:8], "big") % (_PRIME - 1) + 1
        second = int.from_bytes(material[8:], "big") % _PRIME
        out.append((first, second))
    return out


def signature(
    shingle_set: Iterable[str], permutations: Sequence[tuple[int, int]]
) -> tuple[int, ...]:
    """Compute the MinHash signature of a shingle set.

    Args:
        shingle_set: The shingles.
        permutations: Coefficient pairs from :func:`_permutations`.

    Returns:
        One minimum per permutation. All-``_PRIME`` for an empty set, which
        never matches a real signature.
    """
    hashed = [_hash_shingle(item) for item in shingle_set]
    if not hashed:
        return tuple(_PRIME for _ in permutations)
    return tuple(
        min((first * value + second) % _PRIME for value in hashed)
        for first, second in permutations
    )


class MinHashIndex:
    """Banded LSH index over MinHash signatures.

    Attributes:
        permutations: Number of hash permutations.
        bands: Number of bands the signature is split into.
    """

    __slots__ = (
        "permutations",
        "bands",
        "_rows",
        "_coefficients",
        "_buckets",
        "_signatures",
    )

    def __init__(self, *, permutations: int, seed: int, bands: int | None = None) -> None:
        """Prepare the hash family and the bucket table.

        Args:
            permutations: Signature length. 128 gives a usable S-curve around
                a 0.7-0.8 threshold.
            seed: The run seed, so banding is reproducible.
            bands: Bands to split the signature into; defaults to a value whose
                implied threshold sits near 0.7. More bands means more
                candidate pairs and higher recall at higher cost.

        Raises:
            ValueError: If ``bands`` does not divide ``permutations``, because
                a ragged final band silently changes the threshold.
        """
        self.permutations = permutations
        self.bands = bands if bands is not None else max(1, permutations // 8)
        if permutations % self.bands:
            raise ValueError(
                f"bands ({self.bands}) must divide permutations ({permutations}); "
                "a ragged band changes the effective threshold without saying so"
            )
        self._rows = permutations // self.bands
        self._coefficients = _permutations(permutations, seed)
        self._buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
        self._signatures: dict[str, tuple[int, ...]] = {}

    def _bands_of(self, sig: Sequence[int]) -> Iterator[tuple[int, int]]:
        """Yield the band keys of a signature.

        Args:
            sig: A MinHash signature.

        Yields:
            ``(band_index, band_hash)`` pairs.
        """
        for band in range(self.bands):
            chunk = sig[band * self._rows : (band + 1) * self._rows]
            digest = hashlib.blake2b(
                b"".join(value.to_bytes(8, "big") for value in chunk), digest_size=8
            ).digest()
            yield band, int.from_bytes(digest, "big")

    @staticmethod
    def _similarity(left: Sequence[int], right: Sequence[int]) -> float:
        """Estimate Jaccard similarity from two signatures.

        Args:
            left: First signature.
            right: Second signature.

        Returns:
            The fraction of positions that agree, which is an unbiased
            estimator of the Jaccard index.
        """
        if not left:
            return 0.0
        agree = sum(1 for a, b in zip(left, right, strict=True) if a == b)
        return agree / len(left)

    def query(self, sig: Sequence[int], threshold: float) -> tuple[str, float] | None:
        """Find an indexed item similar to a signature.

        Args:
            sig: The candidate's signature.
            threshold: Minimum estimated Jaccard similarity to count as a
                duplicate.

        Returns:
            ``(sample_id, similarity)`` of the first match above the
            threshold, or ``None``.
        """
        seen: set[str] = set()
        best: tuple[str, float] | None = None
        for key in self._bands_of(sig):
            for candidate in self._buckets.get(key, ()):
                if candidate in seen:
                    continue
                seen.add(candidate)
                score = self._similarity(sig, self._signatures[candidate])
                if score >= threshold and (best is None or score > best[1]):
                    best = (candidate, score)
        return best

    def add(self, sample_id: str, sig: Sequence[int]) -> None:
        """Index a signature.

        Args:
            sample_id: Identifier to return from :meth:`query`.
            sig: Its signature.
        """
        frozen = tuple(sig)
        self._signatures[sample_id] = frozen
        for key in self._bands_of(frozen):
            self._buckets[key].append(sample_id)

    def signature_of(self, text: str, shingle_size: int) -> tuple[int, ...]:
        """Compute a text's signature with this index's hash family.

        Args:
            text: The text.
            shingle_size: Characters per shingle.

        Returns:
            The signature.
        """
        return signature(shingles(text, shingle_size), self._coefficients)

    def __len__(self) -> int:
        """Return the number of indexed items."""
        return len(self._signatures)
