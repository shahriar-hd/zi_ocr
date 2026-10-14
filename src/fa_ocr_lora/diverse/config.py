"""Typed, strict loader for ``configs/diverse_config.yaml``.

Same contract as :mod:`fa_ocr_lora.config.config`, which this deliberately
mirrors rather than extends: the two pipelines share primitives
(:class:`~fa_ocr_lora.config.config.Range`, :class:`ConfigError`) but not a
schema, and fusing them would mean every knob of the letters pipeline becoming
a valid key here and vice versa. Strictness is the whole point - a typo must
fail at startup, not silently keep a default through a 150k-sample run.

Precedence::

    CLI flag  >  environment  >  diverse_config.yaml  >  dataclass default

Three design rules this loader enforces that the YAML comments only request:

* **A weight map over a closed set is validated against that set.** Writing
  ``category_weights: {pgae: 0.25}`` is an error naming the typo, not a silently
  zero-weighted ``page``.
* **Probability maps that must sum to one, do.** ``category_weights`` and
  ``digit_script_probs`` are shares of the corpus; a set that sums to 0.9 means
  10% of samples would fall through whatever branch happened to be last.
* **Relative weight maps are not normalised.** ``dpi_choices``,
  ``generation.profiles`` and ``paper.tints`` are relative, so a single entry can
  be retuned without renormalising the rest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Final

import yaml

from ..config.base import ConfigError
from ..config.config import IntRange, Range
from ..config.env import (
    ensure_dotenv_loaded,
    env_bool,
    env_int,
    env_path,
    env_str,
)
from ..dataset.common.io_utils import resolve_repo_path

__all__ = [
    "BANDS",
    "CATEGORIES",
    "CONFIG_PATH_ENV_VAR",
    "DEFAULT_CONFIG_PATH",
    "DEGRADATIONS",
    "ERAS",
    "FEATURES",
    "STYLE_CLASSES",
    "WASH_SHAPES",
    "ConfigError",
    "CropConfig",
    "DegradationConfig",
    "DiverseConfig",
    "DriveConfig",
    "FontPoolConfig",
    "GenerationConfig",
    "GenerationProfile",
    "ImageQcConfig",
    "LexiconConfig",
    "LexiconSource",
    "PackagingConfig",
    "PaperConfig",
    "PathsConfig",
    "PlanConfig",
    "RenderConfig",
    "ShowThroughConfig",
    "SplitsConfig",
    "TextPolicyConfig",
    "TextQcConfig",
    "UploadConfig",
    "WashConfig",
    "WatermarkConfig",
    "YellowingConfig",
    "load_diverse_config",
]

#: Default location of this pipeline's config file.
DEFAULT_CONFIG_PATH: Final[str] = "configs/diverse_config.yaml"

#: Environment overrides. Paths and secrets only - everything else is YAML.
CONFIG_PATH_ENV_VAR: Final[str] = "FA_DIVERSE_CONFIG"
_WORK_DIR_ENV_VAR: Final[str] = "FA_DIVERSE_WORK_DIR"
_REPORTS_DIR_ENV_VAR: Final[str] = "FA_DIVERSE_REPORTS_DIR"
_FONTS_DIR_ENV_VAR: Final[str] = "FA_DIVERSE_FONTS_DIR"
_CACHE_DIR_ENV_VAR: Final[str] = "FA_DIVERSE_CACHE_DIR"
_SEED_ENV_VAR: Final[str] = "FA_DIVERSE_SEED"
_TOTAL_ENV_VAR: Final[str] = "FA_DIVERSE_TOTAL_SAMPLES"
_REPO_ENV_VAR: Final[str] = "FA_DIVERSE_HF_REPO"
_PRIVATE_ENV_VAR: Final[str] = "FA_DIVERSE_HF_PRIVATE"
_DRIVE_FOLDER_ID_ENV_VAR: Final[str] = "FA_DIVERSE_DRIVE_FOLDER_ID"

#: Visual categories, which are also the Hugging Face config names. Closed set:
#: a category here with no format module registered in
#: :mod:`fa_ocr_lora.diverse.layouts.registry` would plan samples nothing can
#: render, so both sides validate against this tuple.
CATEGORIES: Final[tuple[str, ...]] = (
    "page",
    "paragraph",
    "line",
    "poetry",
    "web",
    "chat",
    "banner",
    "story",
    "table_form",
    "mixed",
)

#: Frequency bands assigned in stage 00 and sampled in stage 01.
BANDS: Final[tuple[str, ...]] = ("very_common", "common", "uncommon", "rare", "very_rare")

#: Era/register classes, orthogonal to frequency. ``unlabelled`` is not a
#: failure mode, it is most of Lilak: a permissively-licensed spell-check
#: dictionary carries no era metadata, and pretending otherwise would be worse
#: than recording the absence.
ERAS: Final[tuple[str, ...]] = (
    "archaic",
    "literary",
    "dialectal",
    "technical",
    "neologism",
    "unlabelled",
)

#: Content features drawn independently per plan.
FEATURES: Final[tuple[str, ...]] = (
    "numbers",
    "latin_terms",
    "quotations",
    "parentheticals",
    "bullet_list",
    "table",
    "headings",
    "footnotes",
    "diacritics",
    "zwnj_heavy",
    "emoji",
)

#: Degradation effects. Closed, because an effect named here with no
#: implementation would be silently skipped, and the manifest would claim a
#: variation the pixels do not carry.
DEGRADATIONS: Final[tuple[str, ...]] = (
    "paper_texture",
    "stains",
    "folds",
    "ink_bleed",
    "uneven_toner",
    "scanner_noise",
    "skew",
    "perspective",
    "vignette",
    "shadow",
    "blur",
    "low_light",
    "glare",
    "jpeg",
    "rescale",
    "moire",
    "occlusion",
    "highlighter",
    "clutter",
)

#: Font style classes. The layout stage draws a class first and a family
#: second, so adding ten sans families does not make every poetry page sans.
STYLE_CLASSES: Final[tuple[str, ...]] = (
    "sans",
    "naskh",
    "serif",
    "nastaliq",
    "display",
    "mono",
)

#: Shapes the coloured background wash can take.
WASH_SHAPES: Final[tuple[str, ...]] = ("flood", "band", "panel", "bloom")

#: Digit scripts; mirrors :data:`fa_ocr_lora.diverse.text_policy.DIGIT_SCRIPTS`.
_DIGIT_SCRIPTS: Final[tuple[str, ...]] = ("persian", "latin", "arabic_indic", "mixed")

#: Lexicon source kinds this pipeline knows how to parse.
_SOURCE_KINDS: Final[tuple[str, ...]] = (
    "lilak_lexicon",
    "wiktionary_dump",
    "wikipedia_titles",
    "wordlist",
)

#: How far a set of shares may stray from 1.0 before it is an error. Loose
#: enough for hand-written decimals, tight enough to catch a missing entry.
_SHARE_TOLERANCE: Final[float] = 1e-6


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def _mapping(payload: Any, context: str) -> Mapping[str, Any]:
    """Require a YAML mapping.

    Args:
        payload: The YAML value.
        context: Dotted config path, for error messages.

    Returns:
        The mapping, or an empty one when the key was absent.

    Raises:
        ConfigError: If the value is present but not a mapping.
    """
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{context}: expected a mapping, got {type(payload).__name__}")
    return payload


def _section(payload: Any, context: str, allowed: Sequence[str]) -> Mapping[str, Any]:
    """Require a mapping and reject unknown keys.

    Args:
        payload: The YAML value.
        context: Dotted config path, for error messages.
        allowed: Every key this section accepts.

    Returns:
        The validated mapping.

    Raises:
        ConfigError: On a key outside ``allowed``. The message lists what was
            expected, because a bare "unknown key" says nothing about why.
    """
    section = _mapping(payload, context)
    unknown = set(section) - set(allowed)
    if unknown:
        raise ConfigError(
            f"{context}: unknown key(s) {sorted(unknown)}; "
            f"expected any of {sorted(allowed)}"
        )
    return section


def _number(payload: Any, context: str, default: float) -> float:
    """Parse a float, rejecting booleans and non-numbers.

    Args:
        payload: The YAML value; ``None`` falls back to ``default``.
        context: Dotted config path, for error messages.
        default: Value to use when absent.

    Returns:
        The parsed number.

    Raises:
        ConfigError: If the value is not a number.
    """
    if payload is None:
        return default
    if isinstance(payload, bool) or not isinstance(payload, (int, float)):
        raise ConfigError(f"{context}: expected a number, got {payload!r}")
    return float(payload)


def _probability(payload: Any, context: str, default: float) -> float:
    """Parse a probability in ``[0, 1]``.

    Args:
        payload: The YAML value; ``None`` falls back to ``default``.
        context: Dotted config path, for error messages.
        default: Value to use when absent.

    Returns:
        The parsed probability.

    Raises:
        ConfigError: If the value is not a number in ``[0, 1]``.
    """
    value = _number(payload, context, default)
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{context}: expected a probability in [0, 1], got {value}")
    return value


def _positive_int(payload: Any, context: str, default: int) -> int:
    """Parse a strictly positive integer.

    Args:
        payload: The YAML value; ``None`` falls back to ``default``.
        context: Dotted config path, for error messages.
        default: Value to use when absent.

    Returns:
        The parsed integer.

    Raises:
        ConfigError: If the value is not an integer above zero.
    """
    if payload is None:
        return default
    if isinstance(payload, bool) or not isinstance(payload, int):
        raise ConfigError(f"{context}: expected a whole number, got {payload!r}")
    if payload <= 0:
        raise ConfigError(f"{context}: expected a positive number, got {payload}")
    return payload


def _flag(payload: Any, context: str, default: bool) -> bool:
    """Parse a boolean, rejecting truthy strings.

    ``enabled: "no"`` is a YAML string and therefore truthy, which is a classic
    way to turn a feature on while believing it is off.

    Args:
        payload: The YAML value; ``None`` falls back to ``default``.
        context: Dotted config path, for error messages.
        default: Value to use when absent.

    Returns:
        The parsed boolean.

    Raises:
        ConfigError: If the value is not a real boolean.
    """
    if payload is None:
        return default
    if not isinstance(payload, bool):
        raise ConfigError(f"{context}: expected true or false, got {payload!r}")
    return payload


def _text(payload: Any, context: str, default: str) -> str:
    """Parse a string.

    Args:
        payload: The YAML value; ``None`` falls back to ``default``.
        context: Dotted config path, for error messages.
        default: Value to use when absent.

    Returns:
        The stripped string.

    Raises:
        ConfigError: If the value is not a string.
    """
    if payload is None:
        return default
    if not isinstance(payload, str):
        raise ConfigError(f"{context}: expected a string, got {payload!r}")
    return payload.strip()


def _choice(payload: Any, context: str, default: str, allowed: Sequence[str]) -> str:
    """Parse a string constrained to a closed set.

    Args:
        payload: The YAML value; ``None`` falls back to ``default``.
        context: Dotted config path, for error messages.
        default: Value to use when absent.
        allowed: Permitted values.

    Returns:
        The chosen value.

    Raises:
        ConfigError: If the value is outside ``allowed``.
    """
    value = _text(payload, context, default)
    if value not in allowed:
        raise ConfigError(f"{context}: expected one of {sorted(allowed)}, got {value!r}")
    return value


def _weights(
    payload: Any,
    context: str,
    allowed: Sequence[str],
    default: Mapping[str, float],
    *,
    shares: bool,
    require_all: bool = False,
) -> dict[str, float]:
    """Parse a weight map over a closed key set.

    Args:
        payload: The YAML mapping; ``None`` falls back to ``default``.
        context: Dotted config path, for error messages.
        allowed: Permitted keys.
        default: Map used when the key is absent.
        shares: Whether the values are shares of a whole that must sum to 1.0.
            ``False`` means relative weights, which are left unnormalised so a
            single entry can be retuned in isolation.
        require_all: Whether every key in ``allowed`` must be present. Used for
            sets where a missing entry means "silently zero", which is almost
            never what the author meant.

    Returns:
        Key to weight, with absent keys defaulted to ``0.0``.

    Raises:
        ConfigError: On an unknown key, a negative weight, a missing required
            key, or - when ``shares`` - a total that is not 1.0.
    """
    section = (
        _section(payload, context, allowed) if payload is not None else dict(default)
    )
    out: dict[str, float] = {}
    for key in allowed:
        if key in section:
            value = _number(section[key], f"{context}.{key}", 0.0)
            if value < 0.0:
                raise ConfigError(
                    f"{context}.{key}: weight cannot be negative, got {value}"
                )
            out[key] = value
        elif require_all:
            raise ConfigError(
                f"{context}: missing required key {key!r}; "
                "every entry must be explicit so a zero is never an accident"
            )
        else:
            out[key] = float(default.get(key, 0.0))

    total = sum(out.values())
    if total <= 0.0:
        raise ConfigError(f"{context}: weights must not all be zero")
    if shares and abs(total - 1.0) > _SHARE_TOLERANCE:
        raise ConfigError(
            f"{context}: shares must sum to 1.0, got {total:.6f}; "
            "a set that does not sum to one leaves samples falling through "
            "whichever branch happens to be last"
        )
    return out


def _int_weights(
    payload: Any, context: str, default: Mapping[int, float]
) -> dict[int, float]:
    """Parse a weight map whose keys are integers, such as ``dpi_choices``.

    Args:
        payload: The YAML mapping; ``None`` falls back to ``default``.
        context: Dotted config path, for error messages.
        default: Map used when the key is absent.

    Returns:
        Integer key to relative weight.

    Raises:
        ConfigError: On a non-integer key, a negative weight, or an empty map.
    """
    section = _mapping(payload, context)
    if not section:
        return dict(default)
    out: dict[int, float] = {}
    for raw_key, raw_value in section.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as error:
            raise ConfigError(
                f"{context}: keys must be whole numbers, got {raw_key!r}"
            ) from error
        value = _number(raw_value, f"{context}.{raw_key}", 0.0)
        if value < 0.0:
            raise ConfigError(f"{context}.{raw_key}: weight cannot be negative")
        out[key] = value
    if sum(out.values()) <= 0.0:
        raise ConfigError(f"{context}: weights must not all be zero")
    return out


def _string_weights(
    payload: Any, context: str, default: Mapping[str, float]
) -> dict[str, float]:
    """Parse an open-keyed relative weight map, such as ``paper.tints``.

    Unlike :func:`_weights` the key set is open, because the authors of the
    YAML are free to add a paper colour without editing this module.

    Args:
        payload: The YAML mapping; ``None`` falls back to ``default``.
        context: Dotted config path, for error messages.
        default: Map used when the key is absent.

    Returns:
        Key to relative weight.

    Raises:
        ConfigError: On a negative weight or an all-zero map.
    """
    section = _mapping(payload, context)
    if not section:
        return dict(default)
    out: dict[str, float] = {}
    for raw_key, raw_value in section.items():
        value = _number(raw_value, f"{context}.{raw_key}", 0.0)
        if value < 0.0:
            raise ConfigError(f"{context}.{raw_key}: weight cannot be negative")
        out[str(raw_key)] = value
    if sum(out.values()) <= 0.0:
        raise ConfigError(f"{context}: weights must not all be zero")
    return out


def _numbers(payload: Any, context: str, default: Sequence[float]) -> tuple[float, ...]:
    """Parse a list of numbers.

    Args:
        payload: The YAML list; ``None`` falls back to ``default``.
        context: Dotted config path, for error messages.
        default: Value to use when absent.

    Returns:
        The parsed numbers.

    Raises:
        ConfigError: If the value is not a list of numbers.
    """
    if payload is None:
        return tuple(default)
    if not isinstance(payload, Sequence) or isinstance(payload, str):
        raise ConfigError(f"{context}: expected a list, got {payload!r}")
    return tuple(
        _number(item, f"{context}[{index}]", 0.0) for index, item in enumerate(payload)
    )


def _strings(payload: Any, context: str, default: Sequence[str]) -> tuple[str, ...]:
    """Parse a list of strings.

    Args:
        payload: The YAML list; ``None`` falls back to ``default``.
        context: Dotted config path, for error messages.
        default: Value to use when absent.

    Returns:
        The parsed strings.

    Raises:
        ConfigError: If the value is not a list of strings.
    """
    if payload is None:
        return tuple(default)
    if not isinstance(payload, Sequence) or isinstance(payload, str):
        raise ConfigError(f"{context}: expected a list, got {payload!r}")
    return tuple(
        _text(item, f"{context}[{index}]", "") for index, item in enumerate(payload)
    )


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Where every stage reads and writes.

    Attributes:
        work_dir: Root of the stage outputs. The laptop fills its text
            subtrees; the molab runtime fills the render subtrees after pulling
            the text down.
        reports_dir: Coverage, QC and contact-sheet reports.
        fonts_dir: The redistributable OFL pool, *not* the host font
            directory - see the YAML comment.
        cache_dir: Downloaded dumps and the LLM response cache.
    """

    work_dir: Path
    reports_dir: Path
    fonts_dir: Path
    cache_dir: Path

    #: Subdirectory names, fixed rather than configurable. They are part of the
    #: handoff contract between the laptop and the molab runtime, and a run
    #: that renamed one would silently fail to find the other machine's work.
    LEXICON: ClassVar[str] = "lexicon"
    PLANNING: ClassVar[str] = "planning"
    GENERATION: ClassVar[str] = "generation"
    TEXTQC: ClassVar[str] = "textqc"
    LAYOUTS: ClassVar[str] = "layouts"
    RENDER: ClassVar[str] = "render"
    QC: ClassVar[str] = "qc"

    @classmethod
    def from_dict(cls, payload: Any) -> PathsConfig:
        """Build the paths section.

        Args:
            payload: The ``paths`` mapping.

        Returns:
            The parsed section, every path absolute.

        Raises:
            ConfigError: On an unknown key.
        """
        section = _section(
            payload, "paths", ("work_dir", "reports_dir", "fonts_dir", "cache_dir")
        )
        work = _text(section.get("work_dir"), "paths.work_dir", "data/diverse")
        return cls(
            work_dir=resolve_repo_path(work),
            reports_dir=resolve_repo_path(
                _text(section.get("reports_dir"), "paths.reports_dir", "reports/diverse")
            ),
            fonts_dir=resolve_repo_path(
                _text(section.get("fonts_dir"), "paths.fonts_dir", "data/diverse/fonts")
            ),
            cache_dir=resolve_repo_path(
                _text(section.get("cache_dir"), "paths.cache_dir", f"{work}/cache")
            ),
        )

    @property
    def lexicon_parquet(self) -> Path:
        """Return the stage 00 output."""
        return self.work_dir / self.LEXICON / "lexicon.parquet"

    @property
    def lexicon_report(self) -> Path:
        """Return the stage 00 summary JSON."""
        return self.work_dir / self.LEXICON / "lexicon_summary.json"

    @property
    def plans_jsonl(self) -> Path:
        """Return the stage 01 output."""
        return self.work_dir / self.PLANNING / "plans.jsonl"

    @property
    def coverage_json(self) -> Path:
        """Return the stage 01 coverage report."""
        return self.work_dir / self.PLANNING / "coverage.json"

    @property
    def texts_dir(self) -> Path:
        """Return the stage 02 per-sample JSON directory."""
        return self.work_dir / self.GENERATION / "texts"

    @property
    def texts_jsonl(self) -> Path:
        """Return the stage 02 rolled-up JSONL."""
        return self.work_dir / self.GENERATION / "texts.jsonl"

    @property
    def llm_cache_dir(self) -> Path:
        """Return the on-disk LLM response cache."""
        return self.cache_dir / "llm"

    @property
    def texts_clean_jsonl(self) -> Path:
        """Return the stage 03 output."""
        return self.work_dir / self.TEXTQC / "texts_clean.jsonl"

    @property
    def rejections_jsonl(self) -> Path:
        """Return the stage 03 rejection log, one row per dropped sample."""
        return self.work_dir / self.TEXTQC / "rejections.jsonl"

    @property
    def layouts_jsonl(self) -> Path:
        """Return the stage 04 output."""
        return self.work_dir / self.LAYOUTS / "layouts.jsonl"

    @property
    def images_dir(self) -> Path:
        """Return the stage 05 image tree."""
        return self.work_dir / self.RENDER / "images"

    @property
    def annotations_dir(self) -> Path:
        """Return the stage 05 annotation tree."""
        return self.work_dir / self.RENDER / "annotations"

    @property
    def accepted_jsonl(self) -> Path:
        """Return the stage 06 output."""
        return self.work_dir / self.QC / "accepted.jsonl"

    @property
    def image_rejections_jsonl(self) -> Path:
        """Return the stage 06 rejection log."""
        return self.work_dir / self.QC / "rejections.jsonl"

    @property
    def fonts_csv(self) -> Path:
        """Return the font pool catalogue written by the download script."""
        return self.fonts_dir / "fonts.csv"


@dataclass(frozen=True, slots=True)
class LexiconSource:
    """One harvestable source of Persian headwords.

    Attributes:
        name: Short identifier recorded in the ``source`` column.
        kind: One of :data:`_SOURCE_KINDS`; selects the parser.
        url: Where to fetch it.
        licence: SPDX-ish identifier.
        licence_note: Prose recorded in the dataset card, explaining exactly
            what is and is not redistributed.
        attribution: Human-readable credit.
    """

    name: str
    kind: str
    url: str
    licence: str
    licence_note: str
    attribution: str

    @classmethod
    def from_dict(cls, payload: Any, context: str) -> LexiconSource:
        """Build one source entry.

        Args:
            payload: The source mapping.
            context: Dotted config path, for error messages.

        Returns:
            The parsed source.

        Raises:
            ConfigError: On an unknown key, an unknown ``kind``, or a missing
                licence note. The note is mandatory because the dataset card's
                licence reasoning is assembled from these strings, and a source
                with no recorded terms cannot be published.
        """
        section = _section(
            payload,
            context,
            ("name", "kind", "url", "licence", "licence_note", "attribution"),
        )
        note = _text(section.get("licence_note"), f"{context}.licence_note", "")
        if not note:
            raise ConfigError(
                f"{context}.licence_note: required; the dataset card's licence "
                "reasoning is built from it and a source with no recorded terms "
                "cannot be published"
            )
        return cls(
            name=_text(section.get("name"), f"{context}.name", ""),
            kind=_choice(
                section.get("kind"), f"{context}.kind", "wordlist", _SOURCE_KINDS
            ),
            url=_text(section.get("url"), f"{context}.url", ""),
            licence=_text(section.get("licence"), f"{context}.licence", ""),
            licence_note=" ".join(note.split()),
            attribution=_text(section.get("attribution"), f"{context}.attribution", ""),
        )


@dataclass(frozen=True, slots=True)
class LexiconConfig:
    """Stage 00 settings.

    Attributes:
        sources: Harvestable sources, in order.
        frequency_url: Corpus used to compute frequencies and bands.
        frequency_licence: Its licence.
        max_documents: Documents to tokenise for the frequency table.
        length: Permitted headword length in characters.
        band_percentiles: Descending cut-offs naming each band.
        exclude_top_frequency_rank: Words this frequent are never injected into
            a prompt, because the LLM writes them unprompted.
        min_uses_per_word: Coverage floor across the run.
        max_uses_per_word: Coverage ceiling across the run.
    """

    sources: tuple[LexiconSource, ...]
    frequency_url: str
    frequency_licence: str
    max_documents: int
    length: IntRange
    band_percentiles: dict[str, float]
    exclude_top_frequency_rank: int
    min_uses_per_word: int
    max_uses_per_word: int

    def __post_init__(self) -> None:
        """Validate the cross-field invariants this section must hold."""
        if not self.sources:
            raise ConfigError("lexicon.sources: at least one source is required")
        if self.max_uses_per_word < self.min_uses_per_word:
            raise ConfigError(
                "lexicon: max_uses_per_word must be at least min_uses_per_word, got "
                f"{self.max_uses_per_word} < {self.min_uses_per_word}"
            )
        ordered = [self.band_percentiles[name] for name in BANDS[:-1]]
        if ordered != sorted(ordered, reverse=True):
            raise ConfigError(
                "lexicon.band_percentiles: must descend from very_common to rare, got "
                f"{dict(zip(BANDS[:-1], ordered, strict=True))}"
            )

    @classmethod
    def from_dict(cls, payload: Any) -> LexiconConfig:
        """Build the lexicon section.

        Args:
            payload: The ``lexicon`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key.
        """
        section = _section(
            payload,
            "lexicon",
            (
                "sources",
                "frequency",
                "length",
                "band_percentiles",
                "exclude_top_frequency_rank",
                "min_uses_per_word",
                "max_uses_per_word",
            ),
        )
        raw_sources = section.get("sources") or []
        if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, str):
            raise ConfigError("lexicon.sources: expected a list")
        sources = tuple(
            LexiconSource.from_dict(item, f"lexicon.sources[{index}]")
            for index, item in enumerate(raw_sources)
        )
        frequency = _section(
            section.get("frequency"),
            "lexicon.frequency",
            ("url", "licence", "max_documents"),
        )
        return cls(
            sources=sources,
            frequency_url=_text(frequency.get("url"), "lexicon.frequency.url", ""),
            frequency_licence=_text(
                frequency.get("licence"), "lexicon.frequency.licence", ""
            ),
            max_documents=_positive_int(
                frequency.get("max_documents"), "lexicon.frequency.max_documents", 400_000
            ),
            length=IntRange.parse(
                section.get("length"), "lexicon.length", IntRange(2, 24)
            ),
            band_percentiles=_weights(
                section.get("band_percentiles"),
                "lexicon.band_percentiles",
                BANDS[:-1],
                {"very_common": 0.80, "common": 0.55, "uncommon": 0.30, "rare": 0.08},
                shares=False,
                require_all=True,
            ),
            exclude_top_frequency_rank=_positive_int(
                section.get("exclude_top_frequency_rank"),
                "lexicon.exclude_top_frequency_rank",
                5_000,
            ),
            min_uses_per_word=_positive_int(
                section.get("min_uses_per_word"), "lexicon.min_uses_per_word", 2
            ),
            max_uses_per_word=_positive_int(
                section.get("max_uses_per_word"), "lexicon.max_uses_per_word", 30
            ),
        )


@dataclass(frozen=True, slots=True)
class PlanConfig:
    """Stage 01 settings - the creativity engine's parameters.

    Attributes:
        total_samples: How many plans to draw.
        words_per_prompt: Lexicon words injected per request.
        max_very_rare_per_prompt: Hard cap on ``very_rare`` words in one
            request. Handed a prompt of nothing but words it has never seen, a
            model falls back on tokens from other scripts; a couple of very
            rare words among commoner ones gives it something to write around.
        rare_words_per_100: Cap on rare words per 100 output words, so the
            text reads as prose rather than as a glossary.
        off_domain_probability: Chance a word's domain mismatches the genre.
        band_weights: Sampling shares per frequency band.
        era_weights: Sampling shares per era class.
        category_weights: Target corpus share per visual category.
        feature_probabilities: Independent per-feature probabilities.
    """

    total_samples: int
    words_per_prompt: IntRange
    max_very_rare_per_prompt: int
    rare_words_per_100: IntRange
    off_domain_probability: float
    band_weights: dict[str, float]
    era_weights: dict[str, float]
    category_weights: dict[str, float]
    feature_probabilities: dict[str, float]

    @classmethod
    def from_dict(cls, payload: Any) -> PlanConfig:
        """Build the plan section.

        Args:
            payload: The ``plan`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key, or when ``band_weights``,
                ``era_weights`` or ``category_weights`` do not sum to 1.0.
        """
        section = _section(
            payload,
            "plan",
            (
                "total_samples",
                "words_per_prompt",
                "max_very_rare_per_prompt",
                "rare_words_per_100",
                "off_domain_probability",
                "band_weights",
                "era_weights",
                "category_weights",
                "feature_probabilities",
            ),
        )
        features = _section(
            section.get("feature_probabilities"), "plan.feature_probabilities", FEATURES
        )
        return cls(
            total_samples=_positive_int(
                section.get("total_samples"), "plan.total_samples", 150
            ),
            words_per_prompt=IntRange.parse(
                section.get("words_per_prompt"), "plan.words_per_prompt", IntRange(2, 6)
            ),
            max_very_rare_per_prompt=_positive_int(
                section.get("max_very_rare_per_prompt"),
                "plan.max_very_rare_per_prompt",
                2,
            ),
            rare_words_per_100=IntRange.parse(
                section.get("rare_words_per_100"),
                "plan.rare_words_per_100",
                IntRange(3, 8),
            ),
            off_domain_probability=_probability(
                section.get("off_domain_probability"), "plan.off_domain_probability", 0.15
            ),
            band_weights=_weights(
                section.get("band_weights"),
                "plan.band_weights",
                BANDS,
                {
                    "very_common": 0.0,
                    "common": 0.02,
                    "uncommon": 0.13,
                    "rare": 0.40,
                    "very_rare": 0.45,
                },
                shares=True,
                require_all=True,
            ),
            era_weights=_weights(
                section.get("era_weights"),
                "plan.era_weights",
                ERAS,
                {
                    "archaic": 0.18,
                    "literary": 0.15,
                    "dialectal": 0.07,
                    "technical": 0.24,
                    "neologism": 0.14,
                    "unlabelled": 0.22,
                },
                shares=True,
                require_all=True,
            ),
            category_weights=_weights(
                section.get("category_weights"),
                "plan.category_weights",
                CATEGORIES,
                {
                    "page": 0.25,
                    "paragraph": 0.15,
                    "line": 0.15,
                    "poetry": 0.08,
                    "web": 0.10,
                    "chat": 0.08,
                    "banner": 0.07,
                    "story": 0.04,
                    "table_form": 0.05,
                    "mixed": 0.03,
                },
                shares=True,
                require_all=True,
            ),
            feature_probabilities={
                name: _probability(
                    features.get(name), f"plan.feature_probabilities.{name}", 0.0
                )
                for name in FEATURES
            },
        )


@dataclass(frozen=True, slots=True)
class GenerationProfile:
    """One LLM sampling profile.

    More than one is mandatory, not optional. A single generator's style and
    quirks leak into the corpus and become the thing the model learns; the
    previous run had exactly one and its text is recognisably uniform.

    Attributes:
        name: Identifier recorded per sample, and usable as a held-out
            generator in :attr:`SplitsConfig.holdout_generator`.
        model: Endpoint model id.
        temperature: Sampling temperature.
        top_p: Nucleus sampling cut-off.
        weight: Relative share of requests.
    """

    name: str
    model: str
    temperature: float
    top_p: float
    weight: float

    @classmethod
    def from_dict(cls, payload: Any, context: str) -> GenerationProfile:
        """Build one profile.

        Args:
            payload: The profile mapping.
            context: Dotted config path, for error messages.

        Returns:
            The parsed profile.

        Raises:
            ConfigError: On an unknown key or an out-of-range value.
        """
        section = _section(
            payload, context, ("name", "model", "temperature", "top_p", "weight")
        )
        name = _text(section.get("name"), f"{context}.name", "")
        model = _text(section.get("model"), f"{context}.model", "")
        if not name or not model:
            raise ConfigError(f"{context}: both name and model are required")
        temperature = _number(section.get("temperature"), f"{context}.temperature", 0.8)
        if not 0.0 <= temperature <= 2.0:
            raise ConfigError(
                f"{context}.temperature: expected [0, 2], got {temperature}"
            )
        weight = _number(section.get("weight"), f"{context}.weight", 1.0)
        if weight <= 0.0:
            raise ConfigError(
                f"{context}.weight: expected a positive weight, got {weight}"
            )
        return cls(
            name=name,
            model=model,
            temperature=temperature,
            top_p=_probability(section.get("top_p"), f"{context}.top_p", 0.95),
            weight=weight,
        )


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Stage 02 settings.

    Attributes:
        base_url: OpenAI-compatible endpoint.
        api_key_env: Base environment variable holding the key pool.
        profiles: At least two sampling profiles.
        max_tokens: Per-request output ceiling.
        rpm: Requests per minute per key.
        workers_per_key: Concurrent requests sharing one key. The per-key
            pacer holds the combined rate to ``rpm`` regardless, so this fills
            the rate budget during slow replies rather than exceeding it.
        max_retries: Attempts before a sample is recorded as failed.
        timeout_s: Per-request socket timeout.
        cache_enabled: Whether responses are cached on disk by prompt hash.
    """

    base_url: str
    api_key_env: str
    profiles: tuple[GenerationProfile, ...]
    max_tokens: int
    rpm: int
    workers_per_key: int
    max_retries: int
    timeout_s: float
    cache_enabled: bool

    def __post_init__(self) -> None:
        """Validate the cross-field invariants this section must hold."""
        if len(self.profiles) < 2:
            raise ConfigError(
                "generation.profiles: at least two profiles are required. One "
                "generator's style becomes the thing the model learns; the "
                "previous run had a single profile and its text is uniform."
            )
        names = [profile.name for profile in self.profiles]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ConfigError(f"generation.profiles: duplicate name(s) {duplicates}")

    @classmethod
    def from_dict(cls, payload: Any) -> GenerationConfig:
        """Build the generation section.

        Args:
            payload: The ``generation`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key or fewer than two profiles.
        """
        section = _section(
            payload,
            "generation",
            (
                "base_url",
                "api_key_env",
                "profiles",
                "max_tokens",
                "rpm",
                "workers_per_key",
                "max_retries",
                "timeout_s",
                "cache_enabled",
            ),
        )
        raw = section.get("profiles") or []
        if not isinstance(raw, Sequence) or isinstance(raw, str):
            raise ConfigError("generation.profiles: expected a list")
        return cls(
            base_url=_text(
                section.get("base_url"),
                "generation.base_url",
                "https://integrate.api.nvidia.com/v1",
            ),
            api_key_env=_text(
                section.get("api_key_env"), "generation.api_key_env", "NVIDIA_API_KEY"
            ),
            profiles=tuple(
                GenerationProfile.from_dict(item, f"generation.profiles[{index}]")
                for index, item in enumerate(raw)
            ),
            max_tokens=_positive_int(
                section.get("max_tokens"), "generation.max_tokens", 4096
            ),
            rpm=_positive_int(section.get("rpm"), "generation.rpm", 40),
            workers_per_key=_positive_int(
                section.get("workers_per_key"), "generation.workers_per_key", 1
            ),
            max_retries=_positive_int(
                section.get("max_retries"), "generation.max_retries", 6
            ),
            timeout_s=_number(section.get("timeout_s"), "generation.timeout_s", 180.0),
            cache_enabled=_flag(
                section.get("cache_enabled"), "generation.cache_enabled", True
            ),
        )


@dataclass(frozen=True, slots=True)
class TextPolicyConfig:
    """How ``text_render`` differs from what the model wrote.

    Every field here is a *labelled* variation. The point is not that the
    corpus contains Arabic letterforms or undiacriticised text, it is that the
    manifest says which samples do.

    Attributes:
        digit_script_probs: Shares of the four digit scripts.
        diacritics_fraction: Share rendered with tashkeel preserved.
        arabic_forms_fraction: Share rendered without the Arabic-to-Persian
            letter fold.
        keep_zwnj: Whether ZWNJ survives into ``text_render`` by default.
        drop_zwnj_fraction: Share that deliberately drops it anyway.
    """

    digit_script_probs: dict[str, float]
    diacritics_fraction: float
    arabic_forms_fraction: float
    keep_zwnj: bool
    drop_zwnj_fraction: float

    @classmethod
    def from_dict(cls, payload: Any) -> TextPolicyConfig:
        """Build the text policy section.

        Args:
            payload: The ``text_policy`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key.
        """
        section = _section(
            payload,
            "text_policy",
            (
                "digit_script_probs",
                "diacritics_fraction",
                "arabic_forms_fraction",
                "keep_zwnj",
                "drop_zwnj_fraction",
            ),
        )
        return cls(
            digit_script_probs=_weights(
                section.get("digit_script_probs"),
                "text_policy.digit_script_probs",
                _DIGIT_SCRIPTS,
                {"persian": 0.55, "latin": 0.25, "arabic_indic": 0.05, "mixed": 0.15},
                shares=True,
                require_all=True,
            ),
            diacritics_fraction=_probability(
                section.get("diacritics_fraction"),
                "text_policy.diacritics_fraction",
                0.07,
            ),
            arabic_forms_fraction=_probability(
                section.get("arabic_forms_fraction"),
                "text_policy.arabic_forms_fraction",
                0.05,
            ),
            keep_zwnj=_flag(section.get("keep_zwnj"), "text_policy.keep_zwnj", True),
            drop_zwnj_fraction=_probability(
                section.get("drop_zwnj_fraction"), "text_policy.drop_zwnj_fraction", 0.12
            ),
        )


@dataclass(frozen=True, slots=True)
class TextQcConfig:
    """Stage 03 thresholds.

    Attributes:
        persian_ratio_min: Letter-level Persian floor for ordinary samples.
        persian_ratio_min_mixed: Relaxed floor when the plan asked for Latin
            terms or mixed script.
        length_tolerance: Permitted relative deviation from the length target.
        lexicon_hit_rate_min: Share of requested words that must appear.
        max_line_repeat: Identical-line repeats tolerated.
        ngram_size: N for the repetition ratio.
        max_ngram_ratio: Repeated-n-gram share above which text is degenerate.
        dedupe_threshold: Jaccard above which two texts are duplicates.
        dedupe_threshold_same_genre: Stricter threshold within one genre.
        minhash_permutations: MinHash signature length.
        shingle_size: Characters per shingle.
        max_rejection_rate: Stop and report above this rate.
    """

    persian_ratio_min: float
    persian_ratio_min_mixed: float
    length_tolerance: float
    lexicon_hit_rate_min: float
    max_line_repeat: int
    ngram_size: int
    max_ngram_ratio: float
    dedupe_threshold: float
    dedupe_threshold_same_genre: float
    minhash_permutations: int
    shingle_size: int
    max_rejection_rate: float

    @classmethod
    def from_dict(cls, payload: Any) -> TextQcConfig:
        """Build the text QC section.

        Args:
            payload: The ``text_qc`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key.
        """
        keys = (
            "persian_ratio_min",
            "persian_ratio_min_mixed",
            "length_tolerance",
            "lexicon_hit_rate_min",
            "max_line_repeat",
            "ngram_size",
            "max_ngram_ratio",
            "dedupe_threshold",
            "dedupe_threshold_same_genre",
            "minhash_permutations",
            "shingle_size",
            "max_rejection_rate",
        )
        section = _section(payload, "text_qc", keys)
        return cls(
            persian_ratio_min=_probability(
                section.get("persian_ratio_min"), "text_qc.persian_ratio_min", 0.80
            ),
            persian_ratio_min_mixed=_probability(
                section.get("persian_ratio_min_mixed"),
                "text_qc.persian_ratio_min_mixed",
                0.45,
            ),
            length_tolerance=_probability(
                section.get("length_tolerance"), "text_qc.length_tolerance", 0.55
            ),
            lexicon_hit_rate_min=_probability(
                section.get("lexicon_hit_rate_min"), "text_qc.lexicon_hit_rate_min", 0.5
            ),
            max_line_repeat=_positive_int(
                section.get("max_line_repeat"), "text_qc.max_line_repeat", 3
            ),
            ngram_size=_positive_int(section.get("ngram_size"), "text_qc.ngram_size", 8),
            max_ngram_ratio=_probability(
                section.get("max_ngram_ratio"), "text_qc.max_ngram_ratio", 0.18
            ),
            dedupe_threshold=_probability(
                section.get("dedupe_threshold"), "text_qc.dedupe_threshold", 0.80
            ),
            dedupe_threshold_same_genre=_probability(
                section.get("dedupe_threshold_same_genre"),
                "text_qc.dedupe_threshold_same_genre",
                0.70,
            ),
            minhash_permutations=_positive_int(
                section.get("minhash_permutations"), "text_qc.minhash_permutations", 128
            ),
            shingle_size=_positive_int(
                section.get("shingle_size"), "text_qc.shingle_size", 5
            ),
            max_rejection_rate=_probability(
                section.get("max_rejection_rate"), "text_qc.max_rejection_rate", 0.20
            ),
        )


@dataclass(frozen=True, slots=True)
class CropConfig:
    """Derived line, paragraph and word crops.

    Attributes:
        line: Whether to emit line crops.
        paragraph: Whether to emit paragraph crops.
        word_fraction: Share of words that become single-word crops. Small on
            purpose: a tiny image carries little training signal per unit of
            compute.
        padding: Random padding as a fraction of the box height.
    """

    line: bool
    paragraph: bool
    word_fraction: float
    padding: Range

    @classmethod
    def from_dict(cls, payload: Any) -> CropConfig:
        """Build the crop section.

        Args:
            payload: The ``render.derive_crops`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key.
        """
        section = _section(
            payload,
            "render.derive_crops",
            ("line", "paragraph", "word_fraction", "padding"),
        )
        return cls(
            line=_flag(section.get("line"), "render.derive_crops.line", True),
            paragraph=_flag(
                section.get("paragraph"), "render.derive_crops.paragraph", True
            ),
            word_fraction=_probability(
                section.get("word_fraction"), "render.derive_crops.word_fraction", 0.02
            ),
            padding=Range.parse(
                section.get("padding"), "render.derive_crops.padding", Range(0.04, 0.22)
            ),
        )


@dataclass(frozen=True, slots=True)
class FontPoolConfig:
    """How the font pool is sampled and held out.

    Attributes:
        holdout: Families reserved for ``test_unseen_fonts``.
        coverage_min: Required glyph coverage. Below 1.0 would let a tofu box
            through, and a tofu box means the label stops describing the
            pixels - so this is validated to be exactly 1.0.
    """

    holdout: int
    coverage_min: float

    def __post_init__(self) -> None:
        """Validate the cross-field invariants this section must hold."""
        if self.coverage_min < 1.0:
            raise ConfigError(
                "render.fonts.coverage_min: must be 1.0. A value below one permits "
                "a missing glyph, which Chromium renders with a fallback face - "
                "the label then describes letterforms the image does not contain."
            )

    @classmethod
    def from_dict(cls, payload: Any) -> FontPoolConfig:
        """Build the font pool section.

        Args:
            payload: The ``render.fonts`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key.
        """
        section = _section(payload, "render.fonts", ("holdout", "coverage_min"))
        holdout = section.get("holdout", 3)
        if isinstance(holdout, bool) or not isinstance(holdout, int) or holdout < 0:
            raise ConfigError(
                "render.fonts.holdout: expected a non-negative whole number, "
                f"got {holdout!r}"
            )
        return cls(
            holdout=holdout,
            coverage_min=_number(
                section.get("coverage_min"), "render.fonts.coverage_min", 1.0
            ),
        )


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """Stage 05 settings.

    Attributes:
        engine: Shaping engine; ``chromium`` only - see the YAML comment on why
            the Pillow/raqm path was dropped.
        dpi_choices: Relative weights over render resolutions.
        clean_fraction: Share rendered with no degradation at all.
        max_page_time_ms: Per-page render timeout.
        workers: Parallel browser contexts.
        jpeg_quality: Quality range for degraded JPEG output.
        derive_crops: Crop derivation settings.
        fonts: Font pool settings.
    """

    engine: str
    dpi_choices: dict[int, float]
    clean_fraction: float
    max_page_time_ms: int
    workers: int
    jpeg_quality: IntRange
    derive_crops: CropConfig
    fonts: FontPoolConfig

    @classmethod
    def from_dict(cls, payload: Any) -> RenderConfig:
        """Build the render section.

        Args:
            payload: The ``render`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key.
        """
        section = _section(
            payload,
            "render",
            (
                "engine",
                "dpi_choices",
                "clean_fraction",
                "max_page_time_ms",
                "workers",
                "jpeg_quality",
                "derive_crops",
                "fonts",
            ),
        )
        return cls(
            engine=_choice(
                section.get("engine"), "render.engine", "chromium", ("chromium",)
            ),
            dpi_choices=_int_weights(
                section.get("dpi_choices"),
                "render.dpi_choices",
                {96: 0.10, 150: 0.22, 200: 0.18, 300: 0.40, 400: 0.10},
            ),
            clean_fraction=_probability(
                section.get("clean_fraction"), "render.clean_fraction", 0.25
            ),
            max_page_time_ms=_positive_int(
                section.get("max_page_time_ms"), "render.max_page_time_ms", 30_000
            ),
            workers=_positive_int(section.get("workers"), "render.workers", 4),
            jpeg_quality=IntRange.parse(
                section.get("jpeg_quality"), "render.jpeg_quality", IntRange(55, 95)
            ),
            derive_crops=CropConfig.from_dict(section.get("derive_crops")),
            fonts=FontPoolConfig.from_dict(section.get("fonts")),
        )


@dataclass(frozen=True, slots=True)
class ShowThroughConfig:
    """The ghost of the sheet's other side.

    Unlike the letters pipeline, the ghost is **another sample's text**, not a
    mirror of this one. A mirrored copy of the page teaches a model to treat
    the ghost as a reflection of what it is already reading; unrelated ghost
    text forces it to learn that faint mirrored strokes are not content.

    Attributes:
        probability: Chance a degraded sample carries show-through.
        alpha: Ghost ink opacity.
        blur_px: Gaussian blur, in pixels at the sample's DPI.
        hue_shift_deg: Hue rotation relative to the recto ink.
        desaturate: Extra desaturation; 1.0 is fully grey.
        scale: Scale jitter between the two sides.
        offset: Misregistration as a fraction of page size.
        rotate_deg: Rotation jitter.
        mirror_probability: Chance the ghost is mirrored, which is the
            physically correct case for show-through.
    """

    probability: float
    alpha: Range
    blur_px: Range
    hue_shift_deg: Range
    desaturate: Range
    scale: Range
    offset: Range
    rotate_deg: Range
    mirror_probability: float

    @classmethod
    def from_dict(cls, payload: Any) -> ShowThroughConfig:
        """Build the show-through section.

        Args:
            payload: The ``degradation.show_through`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key.
        """
        context = "degradation.show_through"
        section = _section(
            payload,
            context,
            (
                "probability",
                "alpha",
                "blur_px",
                "hue_shift_deg",
                "desaturate",
                "scale",
                "offset",
                "rotate_deg",
                "mirror_probability",
            ),
        )
        return cls(
            probability=_probability(
                section.get("probability"), f"{context}.probability", 0.22
            ),
            alpha=Range.parse(
                section.get("alpha"), f"{context}.alpha", Range(0.05, 0.18)
            ),
            blur_px=Range.parse(
                section.get("blur_px"), f"{context}.blur_px", Range(1.2, 4.5)
            ),
            hue_shift_deg=Range.parse(
                section.get("hue_shift_deg"),
                f"{context}.hue_shift_deg",
                Range(-28.0, 28.0),
            ),
            desaturate=Range.parse(
                section.get("desaturate"), f"{context}.desaturate", Range(0.35, 0.85)
            ),
            scale=Range.parse(
                section.get("scale"), f"{context}.scale", Range(0.97, 1.03)
            ),
            offset=Range.parse(
                section.get("offset"), f"{context}.offset", Range(-0.02, 0.02)
            ),
            rotate_deg=Range.parse(
                section.get("rotate_deg"), f"{context}.rotate_deg", Range(-1.5, 1.5)
            ),
            mirror_probability=_probability(
                section.get("mirror_probability"), f"{context}.mirror_probability", 0.85
            ),
        )


@dataclass(frozen=True, slots=True)
class YellowingConfig:
    """Age yellowing, strongest at the sheet edges.

    Attributes:
        probability: Chance a sample is yellowed.
        strength: How far toward brown the edges go.
    """

    probability: float
    strength: Range

    @classmethod
    def from_dict(cls, payload: Any) -> YellowingConfig:
        """Build the yellowing section.

        Args:
            payload: The ``degradation.paper.yellowing`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key.
        """
        context = "degradation.paper.yellowing"
        section = _section(payload, context, ("probability", "strength"))
        return cls(
            probability=_probability(
                section.get("probability"), f"{context}.probability", 0.30
            ),
            strength=Range.parse(
                section.get("strength"), f"{context}.strength", Range(0.04, 0.26)
            ),
        )


@dataclass(frozen=True, slots=True)
class WashConfig:
    """A coloured wash or panel behind the text.

    Sits above the paper and below the ink, which is what distinguishes it from
    a paper tint: it covers part of the page, so the same sample carries both
    high-contrast and low-contrast text.

    Attributes:
        probability: Chance a sample carries a wash.
        alpha: Wash opacity.
        hues: Hue angles to draw from, in degrees.
        shapes: Relative weights over :data:`WASH_SHAPES`.
    """

    probability: float
    alpha: Range
    hues: tuple[float, ...]
    shapes: dict[str, float]

    @classmethod
    def from_dict(cls, payload: Any) -> WashConfig:
        """Build the wash section.

        Args:
            payload: The ``degradation.paper.wash`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key, or an empty hue list.
        """
        context = "degradation.paper.wash"
        section = _section(payload, context, ("probability", "alpha", "hues", "shapes"))
        hues = _numbers(
            section.get("hues"),
            f"{context}.hues",
            (0, 35, 48, 95, 150, 200, 215, 265, 330),
        )
        if not hues:
            raise ConfigError(f"{context}.hues: at least one hue is required")
        return cls(
            probability=_probability(
                section.get("probability"), f"{context}.probability", 0.18
            ),
            alpha=Range.parse(
                section.get("alpha"), f"{context}.alpha", Range(0.04, 0.20)
            ),
            hues=hues,
            shapes=_weights(
                section.get("shapes"),
                f"{context}.shapes",
                WASH_SHAPES,
                {"flood": 0.30, "band": 0.25, "panel": 0.30, "bloom": 0.15},
                shares=False,
            ),
        )


@dataclass(frozen=True, slots=True)
class WatermarkConfig:
    """A faint repeated word or ornament behind the text.

    Never annotated. It is noise with provenance: the chosen word is recorded
    in ``aug`` so a surprising image can be explained, but it contributes no
    box and no label.

    Attributes:
        probability: Chance a sample carries a watermark.
        alpha: Watermark opacity.
        rotate_deg: Rotation range.
        scale: Size as a fraction of the page's shorter side.
    """

    probability: float
    alpha: Range
    rotate_deg: Range
    scale: Range

    @classmethod
    def from_dict(cls, payload: Any) -> WatermarkConfig:
        """Build the watermark section.

        Args:
            payload: The ``degradation.paper.watermark`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key.
        """
        context = "degradation.paper.watermark"
        section = _section(
            payload, context, ("probability", "alpha", "rotate_deg", "scale")
        )
        return cls(
            probability=_probability(
                section.get("probability"), f"{context}.probability", 0.10
            ),
            alpha=Range.parse(
                section.get("alpha"), f"{context}.alpha", Range(0.03, 0.12)
            ),
            rotate_deg=Range.parse(
                section.get("rotate_deg"), f"{context}.rotate_deg", Range(-40.0, 40.0)
            ),
            scale=Range.parse(
                section.get("scale"), f"{context}.scale", Range(0.25, 0.75)
            ),
        )


@dataclass(frozen=True, slots=True)
class PaperConfig:
    """Paper colour and everything drawn behind the text.

    Attributes:
        tints: Relative weights over base sheet colours.
        yellowing: Age yellowing settings.
        wash: Coloured wash settings.
        watermark: Watermark settings.
    """

    tints: dict[str, float]
    yellowing: YellowingConfig
    wash: WashConfig
    watermark: WatermarkConfig

    @classmethod
    def from_dict(cls, payload: Any) -> PaperConfig:
        """Build the paper section.

        Args:
            payload: The ``degradation.paper`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key.
        """
        section = _section(
            payload, "degradation.paper", ("tints", "yellowing", "wash", "watermark")
        )
        return cls(
            tints=_string_weights(
                section.get("tints"), "degradation.paper.tints", {"#fdfbf6": 1.0}
            ),
            yellowing=YellowingConfig.from_dict(section.get("yellowing")),
            wash=WashConfig.from_dict(section.get("wash")),
            watermark=WatermarkConfig.from_dict(section.get("watermark")),
        )


@dataclass(frozen=True, slots=True)
class DegradationConfig:
    """Everything applied to a rendered page after the clean raster.

    Attributes:
        probabilities: Per-effect probability within a degraded sample.
        strength: Named strength ranges.
        show_through: Verso ghost settings.
        paper: Paper colour and background settings.
    """

    probabilities: dict[str, float]
    strength: dict[str, Range]
    show_through: ShowThroughConfig
    paper: PaperConfig

    #: Strength ranges the renderer expects, with their defaults.
    _STRENGTH_DEFAULTS: ClassVar[Mapping[str, tuple[float, float]]] = {
        "skew_deg": (-3.5, 3.5),
        "perspective": (0.0, 0.06),
        "blur_radius": (0.3, 1.8),
        "noise_sigma": (1.0, 9.0),
        "vignette": (0.05, 0.35),
        "rescale": (0.45, 0.95),
    }

    @classmethod
    def from_dict(cls, payload: Any) -> DegradationConfig:
        """Build the degradation section.

        Args:
            payload: The ``degradation`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key.
        """
        section = _section(
            payload,
            "degradation",
            ("probabilities", "strength", "show_through", "paper"),
        )
        probabilities = _section(
            section.get("probabilities"), "degradation.probabilities", DEGRADATIONS
        )
        strength = _section(
            section.get("strength"), "degradation.strength", tuple(cls._STRENGTH_DEFAULTS)
        )
        return cls(
            probabilities={
                name: _probability(
                    probabilities.get(name), f"degradation.probabilities.{name}", 0.0
                )
                for name in DEGRADATIONS
            },
            strength={
                name: Range.parse(
                    strength.get(name), f"degradation.strength.{name}", Range(low, high)
                )
                for name, (low, high) in cls._STRENGTH_DEFAULTS.items()
            },
            show_through=ShowThroughConfig.from_dict(section.get("show_through")),
            paper=PaperConfig.from_dict(section.get("paper")),
        )


@dataclass(frozen=True, slots=True)
class ImageQcConfig:
    """Stage 06 thresholds.

    Attributes:
        min_x_height_px: Legibility floor.
        min_x_height_px_low_res: Relaxed floor for deliberately small samples.
        min_contrast: Michelson contrast floor after degradation.
        phash_max_distance: Perceptual-hash distance below which two images
            are duplicates.
        sanity_ocr_fraction: Share run through an independent OCR as a
            pipeline sanity check, not as a quality target.
        max_rejection_rate: Stop and report above this rate.
        contact_sheet_per_category: Images per category on the review sheet.
        contact_sheet_thumbnail_px: Thumbnail size.
    """

    min_x_height_px: float
    min_x_height_px_low_res: float
    min_contrast: float
    phash_max_distance: int
    sanity_ocr_fraction: float
    max_rejection_rate: float
    contact_sheet_per_category: int
    contact_sheet_thumbnail_px: int

    @classmethod
    def from_dict(cls, payload: Any) -> ImageQcConfig:
        """Build the image QC section.

        Args:
            payload: The ``image_qc`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key.
        """
        section = _section(
            payload,
            "image_qc",
            (
                "min_x_height_px",
                "min_x_height_px_low_res",
                "min_contrast",
                "phash_max_distance",
                "sanity_ocr_fraction",
                "max_rejection_rate",
                "contact_sheet",
            ),
        )
        sheet = _section(
            section.get("contact_sheet"),
            "image_qc.contact_sheet",
            ("per_category", "thumbnail_px"),
        )
        return cls(
            min_x_height_px=_number(
                section.get("min_x_height_px"), "image_qc.min_x_height_px", 8.0
            ),
            min_x_height_px_low_res=_number(
                section.get("min_x_height_px_low_res"),
                "image_qc.min_x_height_px_low_res",
                5.0,
            ),
            min_contrast=_probability(
                section.get("min_contrast"), "image_qc.min_contrast", 0.12
            ),
            phash_max_distance=_positive_int(
                section.get("phash_max_distance"), "image_qc.phash_max_distance", 4
            ),
            sanity_ocr_fraction=_probability(
                section.get("sanity_ocr_fraction"), "image_qc.sanity_ocr_fraction", 0.01
            ),
            max_rejection_rate=_probability(
                section.get("max_rejection_rate"), "image_qc.max_rejection_rate", 0.20
            ),
            contact_sheet_per_category=_positive_int(
                sheet.get("per_category"), "image_qc.contact_sheet.per_category", 12
            ),
            contact_sheet_thumbnail_px=_positive_int(
                sheet.get("thumbnail_px"), "image_qc.contact_sheet.thumbnail_px", 320
            ),
        )


@dataclass(frozen=True, slots=True)
class SplitsConfig:
    """Stage 07 split plan, including the two hard generalisation tests.

    Attributes:
        ratios: Train/validation/test shares.
        group_by: Manifest column tying a page to its crops and siblings.
        split_salt: Changes the deal without changing the ratios.
        holdout_templates_per_category: Templates reserved for
            ``test_unseen_templates``.
        holdout_generator: Generation profile reserved for
            ``test_unseen_generator``; ``""`` disables that split.
    """

    ratios: dict[str, float]
    group_by: str
    split_salt: str
    holdout_templates_per_category: int
    holdout_generator: str

    @classmethod
    def from_dict(cls, payload: Any) -> SplitsConfig:
        """Build the splits section.

        Args:
            payload: The ``splits`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key, or ratios that do not sum to 1.
        """
        section = _section(
            payload,
            "splits",
            (
                "ratios",
                "group_by",
                "split_salt",
                "holdout_templates_per_category",
                "holdout_generator",
            ),
        )
        holdout = section.get("holdout_templates_per_category", 1)
        if isinstance(holdout, bool) or not isinstance(holdout, int) or holdout < 0:
            raise ConfigError(
                "splits.holdout_templates_per_category: expected a non-negative "
                f"whole number, got {holdout!r}"
            )
        return cls(
            ratios=_weights(
                section.get("ratios"),
                "splits.ratios",
                ("train", "validation", "test"),
                {"train": 0.94, "validation": 0.02, "test": 0.04},
                shares=True,
                require_all=True,
            ),
            group_by=_text(section.get("group_by"), "splits.group_by", "group_id"),
            split_salt=_text(section.get("split_salt"), "splits.split_salt", ""),
            holdout_templates_per_category=holdout,
            holdout_generator=_text(
                section.get("holdout_generator"), "splits.holdout_generator", ""
            ),
        )


@dataclass(frozen=True, slots=True)
class PackagingConfig:
    """Stage 07 shard geometry.

    Attributes:
        output_dir: Export root.
        shard_samples: Soft sample ceiling per shard.
        min_shard_samples: Floor used when the byte ceiling forces a step down.
        min_shard_bytes: A shard does not close on sample count until it also
            reaches this size, so blob-free configs do not become hundreds of
            tiny files.
        max_shard_bytes: Hard byte ceiling.
        compression: Parquet codec.
        views: ``full`` and/or ``ocr_text_only``.
        include_text_corpus: Whether to export the cleaned text as its own
            config.
    """

    output_dir: Path
    shard_samples: int
    min_shard_samples: int
    min_shard_bytes: int
    max_shard_bytes: int
    compression: str
    views: tuple[str, ...]
    include_text_corpus: bool

    def __post_init__(self) -> None:
        """Validate the cross-field invariants this section must hold."""
        if self.min_shard_samples > self.shard_samples:
            raise ConfigError(
                "packaging: min_shard_samples must not exceed shard_samples, got "
                f"{self.min_shard_samples} > {self.shard_samples}"
            )
        if self.min_shard_bytes >= self.max_shard_bytes:
            raise ConfigError(
                "packaging: min_shard_bytes must be below max_shard_bytes, got "
                f"{self.min_shard_bytes} >= {self.max_shard_bytes}"
            )
        unknown = set(self.views) - {"full", "ocr_text_only"}
        if unknown:
            raise ConfigError(f"packaging.views: unknown view(s) {sorted(unknown)}")
        if not self.views:
            raise ConfigError("packaging.views: at least one view is required")

    @classmethod
    def from_dict(cls, payload: Any) -> PackagingConfig:
        """Build the packaging section.

        Args:
            payload: The ``packaging`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key.
        """
        section = _section(
            payload,
            "packaging",
            (
                "output_dir",
                "shard_samples",
                "min_shard_samples",
                "min_shard_bytes",
                "max_shard_bytes",
                "compression",
                "views",
                "include_text_corpus",
            ),
        )
        return cls(
            output_dir=resolve_repo_path(
                _text(
                    section.get("output_dir"), "packaging.output_dir", "data/diverse/hf"
                )
            ),
            shard_samples=_positive_int(
                section.get("shard_samples"), "packaging.shard_samples", 500
            ),
            min_shard_samples=_positive_int(
                section.get("min_shard_samples"), "packaging.min_shard_samples", 200
            ),
            min_shard_bytes=_positive_int(
                section.get("min_shard_bytes"), "packaging.min_shard_bytes", 268_435_456
            ),
            max_shard_bytes=_positive_int(
                section.get("max_shard_bytes"), "packaging.max_shard_bytes", 1_073_741_824
            ),
            compression=_choice(
                section.get("compression"),
                "packaging.compression",
                "zstd",
                ("zstd", "snappy", "gzip", "brotli", "lz4", "none"),
            ),
            views=_strings(
                section.get("views"), "packaging.views", ("full", "ocr_text_only")
            ),
            include_text_corpus=_flag(
                section.get("include_text_corpus"), "packaging.include_text_corpus", True
            ),
        )


@dataclass(frozen=True, slots=True)
class UploadConfig:
    """Stage 08 settings.

    Attributes:
        repo_id: Hub dataset repo; empty until the user fills it in.
        private: Whether the repo starts private. Defaults true, because the
            dataset card and contact sheets must be reviewed first.
        workers: Parallel upload workers.
        delete_after_upload: Whether verified local shards are removed.
        dataset_name: Pretty name for the card.
        license: Dataset licence; must be compatible with the weakest-licensed
            ingredient.
        commit_message: Commit message prefix.
    """

    repo_id: str
    private: bool
    workers: int
    delete_after_upload: bool
    dataset_name: str
    license: str
    commit_message: str

    @classmethod
    def from_dict(cls, payload: Any) -> UploadConfig:
        """Build the upload section.

        Args:
            payload: The ``upload`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key.
        """
        section = _section(
            payload,
            "upload",
            (
                "repo_id",
                "private",
                "workers",
                "delete_after_upload",
                "dataset_name",
                "license",
                "commit_message",
            ),
        )
        return cls(
            repo_id=_text(section.get("repo_id"), "upload.repo_id", ""),
            private=_flag(section.get("private"), "upload.private", True),
            workers=_positive_int(section.get("workers"), "upload.workers", 3),
            delete_after_upload=_flag(
                section.get("delete_after_upload"), "upload.delete_after_upload", False
            ),
            dataset_name=_text(
                section.get("dataset_name"), "upload.dataset_name", "Persian Diverse OCR"
            ),
            license=_text(section.get("license"), "upload.license", "cc-by-sa-4.0"),
            commit_message=_text(
                section.get("commit_message"),
                "upload.commit_message",
                "Add Persian diverse OCR shards",
            ),
        )


@dataclass(frozen=True, slots=True)
class DriveConfig:
    """Google Drive mirroring.

    Attributes:
        enabled: Whether mirroring runs at all.
        folder: Drive folder name, beside the existing letters tree.
        folder_id: Explicit parent id; required for a service account.
        mirror: Which artifact classes are mirrored.
        workers: Parallel upload threads.
        tar_batch: Samples per tarball for the many-small-files artifacts.
    """

    enabled: bool
    folder: str
    folder_id: str
    mirror: dict[str, bool]
    workers: int
    tar_batch: int

    #: Artifact classes that can be mirrored. ``images`` defaults off: the raw
    #: renders are reproducible from (text, seed, config) and would cost
    #: 50-200 GB of upload from the molab runtime, while the LLM text is not
    #: reproducible because it cost money.
    _MIRROR_KEYS: ClassVar[tuple[str, ...]] = (
        "lexicon",
        "plans",
        "texts",
        "texts_clean",
        "shards",
        "images",
        "reports",
    )

    @classmethod
    def from_dict(cls, payload: Any) -> DriveConfig:
        """Build the Drive section.

        Args:
            payload: The ``drive`` mapping.

        Returns:
            The parsed section.

        Raises:
            ConfigError: On any malformed key.
        """
        section = _section(
            payload,
            "drive",
            ("enabled", "folder", "folder_id", "mirror", "workers", "tar_batch"),
        )
        mirror = _section(section.get("mirror"), "drive.mirror", cls._MIRROR_KEYS)
        defaults = {
            "lexicon": True,
            "plans": True,
            "texts": True,
            "texts_clean": True,
            "shards": True,
            "images": False,
            "reports": True,
        }
        return cls(
            enabled=_flag(section.get("enabled"), "drive.enabled", True),
            folder=_text(section.get("folder"), "drive.folder", "fa-diverse-ocr"),
            folder_id=_text(section.get("folder_id"), "drive.folder_id", ""),
            mirror={
                name: _flag(mirror.get(name), f"drive.mirror.{name}", defaults[name])
                for name in cls._MIRROR_KEYS
            },
            workers=_positive_int(section.get("workers"), "drive.workers", 4),
            tar_batch=_positive_int(section.get("tar_batch"), "drive.tar_batch", 2000),
        )


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DiverseConfig:
    """The whole configuration, validated.

    Attributes:
        run_name: Identifier for this corpus; part of the Drive and export
            folder names so two runs never collide.
        seed: Master seed. Every per-sample draw derives from
            ``f"{seed}:{purpose}:{index}"``.
        paths: Where stages read and write.
        lexicon: Stage 00.
        plan: Stage 01.
        generation: Stage 02.
        text_policy: What ``text_render`` does to the model's words.
        text_qc: Stage 03.
        render: Stage 05.
        degradation: Post-raster effects.
        image_qc: Stage 06.
        splits: Stage 07 split plan.
        packaging: Stage 07 shard geometry.
        upload: Stage 08.
        drive: Mirroring.
        source_path: The file this was loaded from, for provenance.
    """

    run_name: str
    seed: int
    paths: PathsConfig
    lexicon: LexiconConfig
    plan: PlanConfig
    generation: GenerationConfig
    text_policy: TextPolicyConfig
    text_qc: TextQcConfig
    render: RenderConfig
    degradation: DegradationConfig
    image_qc: ImageQcConfig
    splits: SplitsConfig
    packaging: PackagingConfig
    upload: UploadConfig
    drive: DriveConfig
    source_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate the cross-field invariants this section must hold."""
        holdout = self.splits.holdout_generator
        if holdout:
            names = {profile.name for profile in self.generation.profiles}
            if holdout not in names:
                raise ConfigError(
                    f"splits.holdout_generator: {holdout!r} is not a generation "
                    f"profile; known profiles are {sorted(names)}"
                )
            if len(names) < 2:
                raise ConfigError(
                    "splits.holdout_generator: holding out the only profile would "
                    "leave the train split with no text"
                )

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, source: Path | None = None
    ) -> DiverseConfig:
        """Build the configuration from a parsed YAML mapping.

        Args:
            payload: The whole file.
            source: Path it came from, recorded for provenance.

        Returns:
            The validated configuration.

        Raises:
            ConfigError: On any unknown or malformed key.
        """
        allowed = (
            "run_name",
            "seed",
            "paths",
            "lexicon",
            "plan",
            "generation",
            "text_policy",
            "text_qc",
            "render",
            "degradation",
            "image_qc",
            "splits",
            "packaging",
            "upload",
            "drive",
        )
        section = _section(payload, "<root>", allowed)
        return cls(
            run_name=_text(section.get("run_name"), "run_name", "fa_diverse_v1"),
            seed=_positive_int(section.get("seed"), "seed", 20_261_010),
            paths=PathsConfig.from_dict(section.get("paths")),
            lexicon=LexiconConfig.from_dict(section.get("lexicon")),
            plan=PlanConfig.from_dict(section.get("plan")),
            generation=GenerationConfig.from_dict(section.get("generation")),
            text_policy=TextPolicyConfig.from_dict(section.get("text_policy")),
            text_qc=TextQcConfig.from_dict(section.get("text_qc")),
            render=RenderConfig.from_dict(section.get("render")),
            degradation=DegradationConfig.from_dict(section.get("degradation")),
            image_qc=ImageQcConfig.from_dict(section.get("image_qc")),
            splits=SplitsConfig.from_dict(section.get("splits")),
            packaging=PackagingConfig.from_dict(section.get("packaging")),
            upload=UploadConfig.from_dict(section.get("upload")),
            drive=DriveConfig.from_dict(section.get("drive")),
            source_path=source,
        )


def _apply_env(config: DiverseConfig) -> DiverseConfig:
    """Overlay environment variables onto a parsed configuration.

    Applied *after* parsing so a bad environment value hits the same
    validators as a bad YAML value, rather than bypassing them.

    Args:
        config: The configuration as parsed from YAML.

    Returns:
        The configuration with environment overrides applied.
    """
    paths = config.paths
    work = env_path(_WORK_DIR_ENV_VAR)
    reports = env_path(_REPORTS_DIR_ENV_VAR)
    fonts = env_path(_FONTS_DIR_ENV_VAR)
    cache = env_path(_CACHE_DIR_ENV_VAR)
    if any(value is not None for value in (work, reports, fonts, cache)):
        paths = PathsConfig(
            work_dir=resolve_repo_path(work) if work else paths.work_dir,
            reports_dir=resolve_repo_path(reports) if reports else paths.reports_dir,
            fonts_dir=resolve_repo_path(fonts) if fonts else paths.fonts_dir,
            cache_dir=resolve_repo_path(cache) if cache else paths.cache_dir,
        )

    plan = config.plan
    total = env_int(_TOTAL_ENV_VAR)
    if total is not None:
        plan = replace(
            plan, total_samples=_positive_int(total, _TOTAL_ENV_VAR, plan.total_samples)
        )

    upload = config.upload
    repo = env_str(_REPO_ENV_VAR)
    private = env_bool(_PRIVATE_ENV_VAR)
    if repo is not None or private is not None:
        upload = replace(
            upload,
            repo_id=repo if repo is not None else upload.repo_id,
            private=private if private is not None else upload.private,
        )

    drive = config.drive
    folder_id = env_str(_DRIVE_FOLDER_ID_ENV_VAR)
    if folder_id is not None:
        drive = replace(drive, folder_id=folder_id)

    seed = env_int(_SEED_ENV_VAR)
    return replace(
        config,
        paths=paths,
        plan=plan,
        upload=upload,
        drive=drive,
        seed=(
            _positive_int(seed, _SEED_ENV_VAR, config.seed)
            if seed is not None
            else config.seed
        ),
    )


def load_diverse_config(
    path: str | Path | None = None, *, apply_env: bool = True
) -> DiverseConfig:
    """Load, validate and return the configuration.

    Args:
        path: Config file; defaults to :data:`FA_DIVERSE_CONFIG` if set, else
            :data:`DEFAULT_CONFIG_PATH`.
        apply_env: Whether to overlay environment overrides. Tests pass
            ``False`` so a developer's ``.env`` cannot change an assertion.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the file is missing, is not a YAML mapping, or holds an
            unknown or malformed key.
    """
    if apply_env:
        ensure_dotenv_loaded()
    if path is None:
        override = env_str(CONFIG_PATH_ENV_VAR) if apply_env else None
        path = override or DEFAULT_CONFIG_PATH
    target = resolve_repo_path(path)
    if not target.is_file():
        raise ConfigError(f"configuration file not found: {target}")
    try:
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"{target}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{target}: expected a YAML mapping at the top level")

    config = DiverseConfig.from_dict(payload, source=target)
    return _apply_env(config) if apply_env else config
