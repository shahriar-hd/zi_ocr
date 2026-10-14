"""Stage 01 - draw plans from the axes, with coverage and capacity control.

Two mechanisms make this more than a random draw.

Capacity clamping
-----------------

An era weight is a request; its pool size is a limit. ``literary`` has ~916
injectable words, so with ``max_uses_per_word = 30`` it can supply at most
27,480 word-slots however much weight the YAML asks for. A 150k-sample run
needs roughly a million slots, so an unclamped 15% literary weight would
either reuse 160 words a hundred times each or quietly substitute other eras.

:func:`clamp_era_weights` computes each era's capacity, caps its weight, and
redistributes the surplus to eras with headroom - logging every change. This is
what lets ``total_samples`` be raised from 150 to 150,000 without the
stratification silently degrading.

Coverage feedback
-----------------

Within the clamped budget, words are drawn to satisfy
``min_uses_per_word <= uses <= max_uses_per_word``. The sampler keeps a live
use count and prefers under-covered words, which both guarantees the floor and
stops a handful of words dominating the corpus. A word at its ceiling is
removed from the pool entirely rather than down-weighted, because a soft
preference still produces outliers over a million draws.

Determinism
-----------

Every draw comes from ``sample_rng(seed, purpose, index)``, so plan *i* is
identical no matter how many plans precede it or in what order the file is
written. Re-running after a config change rewrites the same plans for the
unchanged prefix.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from ...dataset.common.io_utils import sample_rng, weighted_choice
from ..config import BANDS, ERAS, DiverseConfig
from .axes import (
    OUTPUT_FORMS,
    PERSONAS,
    REGISTERS,
    genres_for,
    length_words,
    lengths_for,
)

__all__ = [
    "LexiconEntry",
    "Plan",
    "PlanSampler",
    "clamp_era_weights",
    "load_pool",
]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

#: Minimum injectable words an era needs before the sampler will draw from it.
#: Below this the era cannot contribute meaningful variety and its weight is
#: redistributed rather than spent reusing a dozen words.
_MIN_ERA_POOL: Final[int] = 25


@dataclass(frozen=True, slots=True)
class LexiconEntry:
    """One injectable word, as loaded from ``lexicon.parquet``.

    Attributes:
        word_id: Stable identifier recorded in the plan and the manifest.
        word: The headword.
        pos: Part of speech.
        era: Era/register class.
        domain: Domain label, or ``""``.
        freq_band: Frequency band.
    """

    word_id: str
    word: str
    pos: str
    era: str
    domain: str
    freq_band: str


@dataclass(slots=True)
class Plan:
    """One sample's full specification, before any text exists.

    Every field is recorded in ``plans.jsonl`` and carried through to the
    manifest, so a surprising sample can be explained from its plan alone.

    Attributes:
        plan_id: Stable identifier, ``plan_%07d``.
        index: Zero-based draw index; the seed source.
        seed: Per-sample seed derived from the run seed and the index.
        category: Visual category, which also selects the genre pool.
        genre: Persian genre name, injected into the prompt verbatim.
        register: Persian register name.
        persona: Invented speaker.
        length_target: Named length band.
        length_min: Target word floor.
        length_max: Target word ceiling.
        output_form: How the text is arranged as text.
        features: Content features requested, as names.
        digit_script: Which digit script the sample will be rendered in.
        keep_diacritics: Whether tashkeel survives into ``text_render``.
        arabic_forms: Whether the Arabic-to-Persian letter fold is skipped.
        drop_zwnj: Whether ZWNJ is deliberately removed.
        profile: Generation profile name.
        word_ids: Lexicon words requested, by id.
        words: The same words, as strings, for the prompt.
        word_eras: Era of each requested word, parallel to ``words``.
        word_bands: Band of each requested word, parallel to ``words``.
        word_domains: Domain of each requested word, parallel to ``words``.
        off_domain: Whether a deliberately mismatched word was included.
    """

    plan_id: str
    index: int
    seed: int
    category: str
    genre: str
    register: str
    persona: str
    length_target: str
    length_min: int
    length_max: int
    output_form: str
    features: list[str]
    digit_script: str
    keep_diacritics: bool
    arabic_forms: bool
    drop_zwnj: bool
    profile: str
    word_ids: list[str]
    words: list[str]
    word_eras: list[str]
    word_bands: list[str]
    word_domains: list[str]
    off_domain: bool

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-serialisable form written to ``plans.jsonl``."""
        return asdict(self)


def load_pool(path: Path) -> list[LexiconEntry]:
    """Load the injectable words from ``lexicon.parquet``.

    Only rows with ``in_pool`` are returned. The excluded rows are the
    high-frequency core the LLM writes unprompted; they remain in the file as
    evidence but are never spent on a prompt slot.

    Args:
        path: Path to ``lexicon.parquet``.

    Returns:
        The injectable entries.

    Raises:
        FileNotFoundError: If the lexicon has not been built.
        RuntimeError: If pyarrow is unavailable, or the pool is empty.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `python -m fa_ocr_lora.diverse.lexicon.build` first"
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required to read lexicon.parquet: pip install 'fa-ocr-lora[hf]'"
        ) from error

    table = pq.read_table(
        path, columns=["word_id", "word", "pos", "era", "domain", "freq_band", "in_pool"]
    )
    entries = [
        LexiconEntry(
            word_id=row["word_id"],
            word=row["word"],
            pos=row["pos"],
            era=row["era"],
            domain=row["domain"],
            freq_band=row["freq_band"],
        )
        for row in table.to_pylist()
        if row["in_pool"]
    ]
    if not entries:
        raise RuntimeError(f"{path} holds no injectable words; check lexicon.in_pool")
    return entries


def clamp_era_weights(
    requested: Mapping[str, float],
    pool_sizes: Mapping[str, int],
    *,
    total_slots: int,
    max_uses_per_word: int,
) -> tuple[dict[str, float], dict[str, str]]:
    """Cap each era's weight at what its word pool can actually supply.

    Args:
        requested: Era weights from the configuration, summing to 1.0.
        pool_sizes: Injectable word count per era.
        total_slots: Word-slots the whole run will consume, i.e.
            ``total_samples * mean_words_per_prompt``.
        max_uses_per_word: Per-word ceiling from the lexicon configuration.

    Returns:
        ``(weights, notes)``. ``weights`` sums to 1.0 and respects every
        capacity; ``notes`` explains each adjustment, for logging and for the
        coverage report.
    """
    notes: dict[str, str] = {}
    capacity = {era: pool_sizes.get(era, 0) * max_uses_per_word for era in requested}
    weights = dict(requested)

    # Eras too small to contribute variety are zeroed outright.
    for era in list(weights):
        if weights[era] > 0 and pool_sizes.get(era, 0) < _MIN_ERA_POOL:
            notes[era] = (
                f"zeroed: only {pool_sizes.get(era, 0)} injectable word(s), "
                f"below the {_MIN_ERA_POOL}-word floor"
            )
            weights[era] = 0.0

    if total_slots <= 0:
        return weights, notes

    # Cap, collecting the surplus.
    surplus = 0.0
    headroom: dict[str, float] = {}
    for era, weight in weights.items():
        if weight <= 0:
            continue
        ceiling = capacity[era] / total_slots
        if weight > ceiling:
            notes[era] = (
                f"clamped {weight:.4f} -> {ceiling:.4f}: "
                f"{pool_sizes.get(era, 0)} word(s) "
                f"x {max_uses_per_word} uses = {capacity[era]} slot(s) of {total_slots}"
            )
            surplus += weight - ceiling
            weights[era] = ceiling
        else:
            headroom[era] = ceiling - weight

    # Redistribute the surplus proportionally to remaining headroom. Two passes
    # are enough in practice; a third would mean every era is saturated, which
    # the final renormalisation handles.
    for _ in range(2):
        if surplus <= 1e-12 or not headroom:
            break
        available = sum(headroom.values())
        if available <= 0:
            break
        granted = min(surplus, available)
        for era, room in list(headroom.items()):
            share = granted * (room / available)
            weights[era] += share
            headroom[era] = room - share
            if headroom[era] <= 1e-12:
                del headroom[era]
        surplus -= granted

    total = sum(weights.values())
    if total <= 0:
        raise RuntimeError(
            "every era was zeroed by capacity clamping; the lexicon is too small "
            "for this total_samples, or plan.era_weights names only starved eras"
        )
    if surplus > 1e-9:
        notes["_saturated"] = (
            f"{surplus:.4f} of weight could not be placed: every era is at capacity. "
            f"Lower plan.total_samples, raise lexicon.max_uses_per_word, or widen "
            f"the lexicon."
        )
    return {era: weight / total for era, weight in weights.items()}, notes


class PlanSampler:
    """Draws plans, tracking word coverage as it goes.

    Attributes:
        config: The active configuration.
        uses: Live use count per ``word_id``.
    """

    __slots__ = (
        "config",
        "uses",
        "_by_era",
        "_by_era_band",
        "_by_domain",
        "_era_weights",
        "_era_notes",
        "_category_pool",
        "_profile_pool",
        "_digit_pool",
        "_feature_names",
        "_exhausted",
    )

    def __init__(self, config: DiverseConfig, pool: Sequence[LexiconEntry]) -> None:
        """Index the lexicon and compute the clamped era budget.

        Args:
            config: The active configuration.
            pool: Injectable lexicon entries.
        """
        self.config = config
        self.uses: Counter[str] = Counter()
        self._exhausted: set[str] = set()

        self._by_era: dict[str, list[LexiconEntry]] = defaultdict(list)
        self._by_era_band: dict[tuple[str, str], list[LexiconEntry]] = defaultdict(list)
        self._by_domain: dict[str, list[LexiconEntry]] = defaultdict(list)
        for entry in pool:
            self._by_era[entry.era].append(entry)
            self._by_era_band[(entry.era, entry.freq_band)].append(entry)
            if entry.domain:
                self._by_domain[entry.domain].append(entry)

        words = config.plan.words_per_prompt
        mean_words = (words.minimum + words.maximum) / 2.0
        total_slots = int(round(config.plan.total_samples * mean_words))
        self._era_weights, self._era_notes = clamp_era_weights(
            config.plan.era_weights,
            {era: len(entries) for era, entries in self._by_era.items()},
            total_slots=total_slots,
            max_uses_per_word=config.lexicon.max_uses_per_word,
        )
        for era, note in sorted(self._era_notes.items()):
            LOGGER.warning("era %s: %s", era, note)

        self._category_pool = tuple(config.plan.category_weights.items())
        self._profile_pool = tuple(
            (profile.name, profile.weight) for profile in config.generation.profiles
        )
        self._digit_pool = tuple(config.text_policy.digit_script_probs.items())
        self._feature_names = tuple(config.plan.feature_probabilities)

    @property
    def era_weights(self) -> dict[str, float]:
        """Return the capacity-clamped era weights actually in use."""
        return dict(self._era_weights)

    @property
    def era_notes(self) -> dict[str, str]:
        """Return the explanation of every clamping adjustment."""
        return dict(self._era_notes)

    def _available(self, entries: Sequence[LexiconEntry]) -> list[LexiconEntry]:
        """Filter an entry list to words below the use ceiling.

        Args:
            entries: Candidate entries.

        Returns:
            Entries whose use count is below ``max_uses_per_word``.
        """
        ceiling = self.config.lexicon.max_uses_per_word
        return [entry for entry in entries if self.uses[entry.word_id] < ceiling]

    def _draw_word(
        self, rng: random.Random, *, era: str, band: str, domain: str | None
    ) -> LexiconEntry | None:
        """Draw one word, preferring under-covered candidates.

        The preference is implemented by drawing a small candidate set and
        keeping the least-used member, which is far cheaper than maintaining a
        sorted structure over 124k words and produces the same floor
        guarantee - a word at zero uses wins any comparison it enters, and over
        a run every word enters many.

        Args:
            rng: The sample's RNG.
            era: Preferred era.
            band: Preferred frequency band.
            domain: Preferred domain, or ``None`` for any.

        Returns:
            The chosen entry, or ``None`` when no candidate remains anywhere.
        """
        tiers: list[Sequence[LexiconEntry]] = []
        if domain:
            tiers.append(
                [entry for entry in self._by_domain.get(domain, ()) if entry.era == era]
            )
            tiers.append(self._by_domain.get(domain, ()))
        tiers.append(self._by_era_band.get((era, band), ()))
        tiers.append(self._by_era.get(era, ()))
        # Final fallback: any era at the requested band, then anything at all.
        tiers.append(
            [
                entry
                for other in ERAS
                if self._era_weights.get(other, 0.0) > 0
                for entry in self._by_era_band.get((other, band), ())
            ]
        )

        for tier in tiers:
            candidates = self._available(tier)
            if not candidates:
                continue
            # Sample a few and keep the least-used; ties broken by word_id so
            # the choice is deterministic.
            picks = rng.sample(candidates, 8) if len(candidates) > 8 else list(candidates)
            return min(picks, key=lambda entry: (self.uses[entry.word_id], entry.word_id))
        return None

    def draw(self, index: int) -> Plan:
        """Draw plan number ``index``.

        Args:
            index: Zero-based draw index.

        Returns:
            The plan.
        """
        seed = self.config.seed
        rng = sample_rng(seed, "plan", index)
        plan_config = self.config.plan
        policy = self.config.text_policy

        category = weighted_choice(self._category_pool, rng)
        genre = weighted_choice(genres_for(category), rng)
        register = weighted_choice(REGISTERS, rng)
        persona = weighted_choice(PERSONAS, rng)

        # Length is drawn from the pool the category permits, not from the
        # global pool: a `line` sample asking for 300-600 words is incoherent,
        # and the model obeys the word count over the category.
        length_target = weighted_choice(lengths_for(category), rng)
        low, high = length_words(length_target)

        output_form = weighted_choice(OUTPUT_FORMS, rng)

        # Content features. Emoji is clamped to chat, and a table only makes
        # sense where a layout can carry one.
        features: list[str] = []
        for name in self._feature_names:
            probability = plan_config.feature_probabilities[name]
            if name == "emoji" and category != "chat":
                probability = 0.0
            if name == "table" and category not in {"page", "table_form", "web"}:
                probability = 0.0
            if name == "footnotes" and category not in {"page", "story"}:
                probability = 0.0
            if rng.random() < probability:
                features.append(name)

        digit_script = weighted_choice(self._digit_pool, rng)
        keep_diacritics = rng.random() < policy.diacritics_fraction
        arabic_forms = rng.random() < policy.arabic_forms_fraction
        drop_zwnj = rng.random() < policy.drop_zwnj_fraction
        profile = weighted_choice(self._profile_pool, rng)

        # Words. Count scales with the length target: a six-word banner cannot
        # absorb twelve rare words, and asking it to produces a word salad.
        requested = plan_config.words_per_prompt.draw(rng)
        span = max(1, high)
        cap = max(1, int(round(plan_config.rare_words_per_100.draw(rng) * span / 100.0)))
        count = max(1, min(requested, cap))

        era_pool = tuple(
            (era, weight) for era, weight in self._era_weights.items() if weight > 0
        )
        band_pool = tuple(
            (band, weight)
            for band, weight in plan_config.band_weights.items()
            if weight > 0
        )
        off_domain = rng.random() < plan_config.off_domain_probability

        chosen: list[LexiconEntry] = []
        seen: set[str] = set()
        very_rare_used = 0
        very_rare_cap = plan_config.max_very_rare_per_prompt
        for _ in range(count * 6):
            if len(chosen) >= count:
                break
            era = weighted_choice(era_pool, rng)
            band = weighted_choice(band_pool, rng) if band_pool else "rare"
            # Cap the very rarest words per prompt. A request made entirely of
            # words the model has never seen degenerates into other scripts
            # and Latin transliteration; a couple among commoner words gives it
            # an anchor to write around. Over-quota draws ask for `rare`
            # instead, which keeps corpus-level rarity without breaking the
            # sample.
            if band == "very_rare" and very_rare_used >= very_rare_cap:
                band = "rare"
            domain = None
            if off_domain and self._by_domain:
                domain = rng.choice(sorted(self._by_domain))
            entry = self._draw_word(rng, era=era, band=band, domain=domain)
            if entry is None or entry.word_id in seen:
                continue
            # Enforced on the entry, not on the requested band: _draw_word
            # falls back through broader tiers when a band is thin, and those
            # tiers are not band-filtered, so a `rare` request can legitimately
            # return a very_rare word.
            if entry.freq_band == "very_rare" and very_rare_used >= very_rare_cap:
                continue
            seen.add(entry.word_id)
            chosen.append(entry)
            if entry.freq_band == "very_rare":
                very_rare_used += 1

        for entry in chosen:
            self.uses[entry.word_id] += 1
            if self.uses[entry.word_id] >= self.config.lexicon.max_uses_per_word:
                self._exhausted.add(entry.word_id)

        return Plan(
            plan_id=f"plan_{index:07d}",
            index=index,
            seed=seed,
            category=category,
            genre=genre,
            register=register,
            persona=persona,
            length_target=length_target,
            length_min=low,
            length_max=high,
            output_form=output_form,
            features=features,
            digit_script=digit_script,
            keep_diacritics=keep_diacritics,
            arabic_forms=arabic_forms,
            drop_zwnj=drop_zwnj,
            profile=profile,
            word_ids=[entry.word_id for entry in chosen],
            words=[entry.word for entry in chosen],
            word_eras=[entry.era for entry in chosen],
            word_bands=[entry.freq_band for entry in chosen],
            word_domains=[entry.domain for entry in chosen],
            off_domain=off_domain,
        )

    def draw_all(self) -> Iterator[Plan]:
        """Draw every plan the configuration asks for.

        Yields:
            Plans in index order.
        """
        for index in range(self.config.plan.total_samples):
            yield self.draw(index)

    def coverage(self, pool: Sequence[LexiconEntry]) -> dict[str, object]:
        """Build the coverage report.

        Args:
            pool: The injectable entries the run drew from.

        Returns:
            A JSON-serialisable report: how many words met the floor, which
            eras and bands were actually spent, and every clamping note.
        """
        floor = self.config.lexicon.min_uses_per_word
        by_id = {entry.word_id: entry for entry in pool}
        used = {word_id: count for word_id, count in self.uses.items() if count > 0}

        era_spend: Counter[str] = Counter()
        band_spend: Counter[str] = Counter()
        domain_spend: Counter[str] = Counter()
        for word_id, count in used.items():
            entry = by_id.get(word_id)
            if entry is None:
                continue
            era_spend[entry.era] += count
            band_spend[entry.freq_band] += count
            if entry.domain:
                domain_spend[entry.domain] += count

        return {
            "pool_size": len(pool),
            "words_used": len(used),
            "words_unused": len(pool) - len(used),
            "words_at_or_above_floor": sum(
                1 for count in used.values() if count >= floor
            ),
            "words_at_ceiling": len(self._exhausted),
            "min_uses_per_word": floor,
            "max_uses_per_word": self.config.lexicon.max_uses_per_word,
            "total_word_slots": sum(used.values()),
            "era_weights_requested": dict(self.config.plan.era_weights),
            "era_weights_effective": self.era_weights,
            "era_clamping_notes": self.era_notes,
            "spend_by_era": dict(era_spend.most_common()),
            "spend_by_band": {name: band_spend.get(name, 0) for name in BANDS},
            "spend_by_domain": dict(domain_spend.most_common(30)),
        }


def write_plans(plans: Sequence[Plan], path: Path) -> int:
    """Write plans to JSONL atomically.

    Args:
        plans: Plans in index order.
        path: Destination ``plans.jsonl``.

    Returns:
        The number of plans written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for plan in plans:
            handle.write(json.dumps(plan.as_dict(), ensure_ascii=False) + "\n")
    temporary.replace(path)
    return len(plans)


def read_plans(path: Path) -> list[Plan]:
    """Read plans back from JSONL.

    Args:
        path: Source ``plans.jsonl``.

    Returns:
        The plans in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `python -m fa_ocr_lora.diverse.planning.build` first"
        )
    plans: list[Plan] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                plans.append(Plan(**json.loads(line)))
    return plans
