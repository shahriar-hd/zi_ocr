"""Typed loader for ``configs/synth_config.yaml``.

CLAUDE.md 8 forbids hardcoded constants: every knob the synthetic pipeline
turns - page sizes, DPI, probabilities, asset locations - lives in YAML and
arrives here as a frozen, validated dataclass. Loading is strict, so a typo in
the YAML fails at startup rather than silently rendering 20k samples with a
default nobody intended.

Two settings are not the YAML's business: **paths**, which differ per machine,
and **secrets**, which must never be committed. Those come from the
environment via :mod:`fa_ocr_lora.config.env`, and :func:`load_config` overlays
them once the file has been parsed. The full precedence chain is::

    CLI flag  >  environment  >  YAML  >  dataclass default
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from random import Random
from typing import Any, ClassVar, Final, Iterable, Mapping, Sequence

import yaml

from ..dataset.common.io_utils import DEFAULT_SEED, repo_root, resolve_repo_path
from .base import CHART_KINDS, ConfigError
from .env import ensure_dotenv_loaded, env_bool, env_int, env_path, env_str
from .vocabulary import (
    DEFAULT_VOCABULARY_PATH,
    VOCABULARY_PATH_ENV_VAR,
    Vocabulary,
    load_vocabulary,
)

__all__ = [
    "BAR_STYLES",
    "CHART_KINDS",
    "CONFIG_PATH_ENV_VAR",
    "CONTACT_FIELDS",
    "CSS_PPI",
    "DATA_DIR_ENV_VAR",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_VOCABULARY_PATH",
    "FOOTER_LAYOUT_STYLES",
    "HF_COMPRESSIONS",
    "HF_FAMILIES",
    "HF_GROUP_KEYS",
    "HF_OUTPUT_DIR_ENV_VAR",
    "HF_PRIVATE_ENV_VAR",
    "HF_REPO_ID_ENV_VAR",
    "HF_SPLITS",
    "HF_SPLITS_ENV_VAR",
    "HF_SPLIT_TOLERANCE",
    "HF_TOKEN_ENV_VARS",
    "IMAGE_FORMAT_CHOICES",
    "METADATA_FIELDS",
    "PAGE_SIDES",
    "PATH_ENV_VARS",
    "SEED_ENV_VAR",
    "STAMP_ROUND_SHAPES",
    "STAMP_SHAPES",
    "SUBDIR_FIELDS",
    "VOCABULARY_PATH_ENV_VAR",
    "ChartConfig",
    "ConfigError",
    "ElementsConfig",
    "FontSpec",
    "HuggingFaceConfig",
    "ImageFormatConfig",
    "IntRange",
    "LayoutConfig",
    "PageSizeSpec",
    "PathsConfig",
    "ProbabilityConfig",
    "Range",
    "RenderConfig",
    "ScanConfig",
    "SignatureConfig",
    "StampConfig",
    "SynthConfig",
    "TableConfig",
    "ThemeConfig",
    "Vocabulary",
    "load_config",
    "load_synth_config",
    "load_vocabulary",
    "normalize_split_fractions",
    "parse_split_spec",
]

#: Default location of the synthetic pipeline's config file.
DEFAULT_CONFIG_PATH: Final[str] = "configs/synth_config.yaml"

_ORIENTATIONS: Final[frozenset[str]] = frozenset({"portrait", "landscape"})

#: Millimetres per inch, for converting physical page sizes to CSS pixels.
_MM_PER_INCH: Final[float] = 25.4

#: CSS reference resolution. Chromium treats 96 CSS px as one inch, so a
#: device scale factor of ``dpi / 96`` is what turns a CSS layout into a
#: raster at the requested DPI.
CSS_PPI: Final[int] = 96


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    """Fetch a required key, naming the section in the error."""
    if key not in mapping:
        raise ConfigError(f"{context}: missing required key {key!r}")
    return mapping[key]


def _as_probability(value: Any, context: str) -> float:
    """Validate that a value is a probability in ``[0, 1]``."""
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context}: expected a number, got {value!r}") from exc
    if not 0.0 <= probability <= 1.0:
        raise ConfigError(f"{context}: probability must be in [0, 1], got {probability}")
    return probability


#: Accepted spelling of a colour anywhere in the config. Both renderers expect
#: ``#rrggbb`` exactly: CSS would treat a malformed value as no declaration at
#: all and silently fall back, while ``theme.hex_to_rgb`` would raise mid-render
#: on sample 400. Validating the spelling here turns both into a load error.
_HEX_COLOUR: Final[re.Pattern[str]] = re.compile(r"#[0-9A-Fa-f]{6}")


def _as_hex_colour(value: Any, context: str) -> str:
    """Validate that a value is a ``#rrggbb`` colour and return it."""
    colour = str(value).strip()
    if not _HEX_COLOUR.fullmatch(colour):
        raise ConfigError(f"{context}: expected a #rrggbb colour, got {value!r}")
    return colour


@dataclass(frozen=True, slots=True)
class Range:
    """An inclusive numeric range one sample draws a value from.

    Attributes:
        minimum: Low end, inclusive.
        maximum: High end, inclusive. Equal ends mean a constant.
    """

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if self.maximum < self.minimum:
            raise ConfigError(
                f"range must be ascending, got {self.minimum}-{self.maximum}"
            )

    def draw(self, rng: Random) -> float:
        """Return a uniform sample from the range.

        Args:
            rng: The sample's seeded RNG.

        Returns:
            A value in ``[minimum, maximum]``. Drawn even when the range is a
            constant, so a retuned range never shifts later decisions.
        """
        return rng.uniform(self.minimum, self.maximum)

    @classmethod
    def parse(cls, payload: Any, context: str, default: "Range") -> "Range":
        """Build a range from YAML, accepting a scalar or a ``{min, max}`` map.

        Args:
            payload: The YAML value; ``None`` falls back to ``default``.
            context: Dotted config path, for error messages.
            default: Range to use when the key is absent.

        Returns:
            The parsed range.

        Raises:
            ConfigError: If the payload is neither a number nor a mapping with
                numeric ``min``/``max``.
        """
        if payload is None:
            return default
        if isinstance(payload, (int, float)) and not isinstance(payload, bool):
            return cls(float(payload), float(payload))
        if not isinstance(payload, Mapping):
            raise ConfigError(
                f"{context}: expected a number or a {{min, max}} mapping, "
                f"got {payload!r}"
            )
        unknown = set(payload) - {"min", "max"}
        if unknown:
            raise ConfigError(f"{context}: unknown key(s) {sorted(unknown)}")
        try:
            low = float(payload.get("min", default.minimum))
            high = float(payload.get("max", default.maximum))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{context}: min and max must be numbers") from exc
        try:
            return cls(low, high)
        except ConfigError as exc:
            raise ConfigError(f"{context}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class IntRange:
    """An inclusive range of whole numbers one sample draws a value from.

    Separate from :class:`Range` because the draw is a different one, not just
    a rounded version of it: ``rng.randrange`` consumes raw bits from the
    generator where ``rng.uniform`` consumes a float. Substituting one for the
    other would shift every later decision in the sample, so counts (rows in a
    table, points in a series) keep their own type.

    Both ends are inclusive, which is what a reader of the YAML expects from
    ``{min: 4, max: 8}``. :meth:`draw` adds the exclusive-bound offset itself.

    Attributes:
        minimum: Low end, inclusive.
        maximum: High end, inclusive. Equal ends mean a constant.
    """

    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.maximum < self.minimum:
            raise ConfigError(
                f"range must be ascending, got {self.minimum}-{self.maximum}"
            )

    def draw(self, rng: Random) -> int:
        """Return a uniform integer from the range.

        Args:
            rng: The sample's seeded RNG.

        Returns:
            A value in ``[minimum, maximum]``. Drawn even when the range is a
            constant, so a retuned range never shifts later decisions.
        """
        return rng.randrange(self.minimum, self.maximum + 1)

    @classmethod
    def parse(cls, payload: Any, context: str, default: "IntRange") -> "IntRange":
        """Build a range from YAML, accepting a scalar or a ``{min, max}`` map.

        Args:
            payload: The YAML value; ``None`` falls back to ``default``.
            context: Dotted config path, for error messages.
            default: Range to use when the key is absent.

        Returns:
            The parsed range.

        Raises:
            ConfigError: If the payload is not an integer, or is a mapping whose
                ``min``/``max`` are not integers. A float is rejected rather
                than truncated: ``{min: 4.5}`` is a mistake, not a request.
        """
        if payload is None:
            return default
        if isinstance(payload, bool):
            raise ConfigError(f"{context}: expected a whole number, got {payload!r}")
        if isinstance(payload, int):
            return cls(payload, payload)
        if not isinstance(payload, Mapping):
            raise ConfigError(
                f"{context}: expected a whole number or a {{min, max}} mapping, "
                f"got {payload!r}"
            )
        unknown = set(payload) - {"min", "max"}
        if unknown:
            raise ConfigError(f"{context}: unknown key(s) {sorted(unknown)}")
        bounds: list[int] = []
        for key, fallback in (("min", default.minimum), ("max", default.maximum)):
            value = payload.get(key, fallback)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(
                    f"{context}.{key}: expected a whole number, got {value!r}"
                )
            bounds.append(value)
        try:
            return cls(bounds[0], bounds[1])
        except ConfigError as exc:
            raise ConfigError(f"{context}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class PageSizeSpec:
    """One selectable page geometry, in physical millimetres."""

    name: str
    width_mm: float
    height_mm: float
    orientation: str = "portrait"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.width_mm <= 0 or self.height_mm <= 0:
            raise ConfigError(
                f"page size {self.name!r}: dimensions must be positive, "
                f"got {self.width_mm}x{self.height_mm}"
            )
        if self.orientation not in _ORIENTATIONS:
            raise ConfigError(
                f"page size {self.name!r}: orientation must be one of "
                f"{sorted(_ORIENTATIONS)}, got {self.orientation!r}"
            )
        if self.weight < 0:
            raise ConfigError(f"page size {self.name!r}: weight must be non-negative")

    @property
    def css_width(self) -> float:
        """Page width in CSS pixels."""
        return self.width_mm / _MM_PER_INCH * CSS_PPI

    @property
    def css_height(self) -> float:
        """Page height in CSS pixels."""
        return self.height_mm / _MM_PER_INCH * CSS_PPI

    def device_size(self, dpi: int) -> tuple[int, int]:
        """Return the rasterized ``(width, height)`` in device pixels at ``dpi``."""
        return (
            int(round(self.width_mm / _MM_PER_INCH * dpi)),
            int(round(self.height_mm / _MM_PER_INCH * dpi)),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PageSizeSpec":
        """Build a page size from its YAML mapping."""
        name = str(_require(payload, "name", "page_sizes entry"))
        return cls(
            name=name,
            width_mm=float(_require(payload, "width_mm", f"page size {name!r}")),
            height_mm=float(_require(payload, "height_mm", f"page size {name!r}")),
            orientation=str(payload.get("orientation", "portrait")),
            weight=float(payload.get("weight", 1.0)),
        )


#: Artifact subdirectory names that may be overridden, mapped to the
#: :class:`PathsConfig` field that holds each one. They are settings rather
#: than constants because a consumer expects the tree it expects - a
#: PaddleFarmer corpus is usually handed ``image/`` beside ``json/`` - and
#: renaming a directory in YAML beats moving 20k files afterwards.
SUBDIR_FIELDS: Final[Mapping[str, str]] = {
    "images": "images_subdir",
    "pdf": "pdf_subdir",
    "markdown": "markdown_subdir",
    "annotations": "annotations_subdir",
    "json": "json_subdir",
}


def _as_subdir(value: Any, context: str) -> str:
    """Validate one artifact subdirectory name.

    Args:
        value: The configured name; ``""`` means "write into ``output_dir``".
        context: Dotted config path, for error messages.

    Returns:
        The name with any surrounding whitespace and trailing slash stripped.

    Raises:
        ConfigError: If the name is absolute or tries to escape the output
            directory - an artifact tree must stay self-contained to be
            copyable to a training host in one move. An absolute name is
            rejected rather than reinterpreted as relative, so a mistyped
            path never silently lands somewhere plausible.
    """
    raw = str(value).strip()
    if not raw:
        return ""
    if raw.startswith("/") or Path(raw).is_absolute() or ".." in Path(raw).parts:
        raise ConfigError(
            f"{context}: must be a relative name inside the output directory, "
            f"got {value!r}"
        )
    return raw.rstrip("/")


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Where the pipeline reads assets from and writes artifacts to.

    The ``*_subdir`` fields name the artifact directories under
    ``output_dir``. An empty name puts that artifact in ``output_dir``
    itself, which is the default for the JSONL training files.
    """

    manifest: Path
    plaintext_dir: Path
    logos_dir: Path
    signatures_dir: Path
    texture_dir: Path
    output_dir: Path
    fonts_dir: Path | None = None
    images_subdir: str = "images"
    pdf_subdir: str = "pdf"
    markdown_subdir: str = "markdown"
    annotations_subdir: str = "annotations"
    json_subdir: str = ""

    def __post_init__(self) -> None:
        for key, attribute in SUBDIR_FIELDS.items():
            cleaned = _as_subdir(getattr(self, attribute), f"paths.subdirs.{key}")
            if cleaned != getattr(self, attribute):
                object.__setattr__(self, attribute, cleaned)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PathsConfig":
        """Build the path set, resolving each entry against the repo root."""
        context = "paths"
        fonts_dir = payload.get("fonts_dir")
        subdirs = payload.get("subdirs") or {}
        if not isinstance(subdirs, Mapping):
            raise ConfigError("paths.subdirs must be a mapping")
        unknown = set(subdirs) - set(SUBDIR_FIELDS)
        if unknown:
            raise ConfigError(
                f"paths.subdirs: unknown key(s) {sorted(unknown)}; "
                f"expected any of {sorted(SUBDIR_FIELDS)}"
            )
        overrides = {
            SUBDIR_FIELDS[key]: str(value) for key, value in subdirs.items()
        }
        return cls(
            manifest=resolve_repo_path(_require(payload, "manifest", context)),
            plaintext_dir=resolve_repo_path(_require(payload, "plaintext_dir", context)),
            logos_dir=resolve_repo_path(_require(payload, "logos_dir", context)),
            signatures_dir=resolve_repo_path(
                _require(payload, "signatures_dir", context)
            ),
            texture_dir=resolve_repo_path(_require(payload, "texture_dir", context)),
            output_dir=resolve_repo_path(_require(payload, "output_dir", context)),
            fonts_dir=(
                Path(fonts_dir).expanduser()
                if fonts_dir
                else None
            ),
            **overrides,
        )

    def with_subdirs(self, overrides: Mapping[str, str]) -> "PathsConfig":
        """Return a copy with some artifact subdirectory names replaced.

        Args:
            overrides: Keys from :data:`SUBDIR_FIELDS` mapped to new names.

        Returns:
            The updated paths; ``self`` when ``overrides`` is empty.

        Raises:
            ConfigError: If a key is not an artifact directory.
        """
        if not overrides:
            return self
        unknown = set(overrides) - set(SUBDIR_FIELDS)
        if unknown:
            raise ConfigError(
                f"unknown artifact directory {sorted(unknown)}; "
                f"expected any of {sorted(SUBDIR_FIELDS)}"
            )
        return replace(
            self,
            **{SUBDIR_FIELDS[key]: value for key, value in overrides.items()},
        )

    def _child(self, subdir: str) -> Path:
        """Return ``output_dir / subdir``, or ``output_dir`` when unset."""
        return self.output_dir / subdir if subdir else self.output_dir

    def image_ref(self, name: str) -> str:
        """Return a page filename as it is recorded in the ground truth.

        Args:
            name: Bare page filename, e.g. ``synth_000000_p01.png``. Its suffix
                comes from :attr:`ImageFormatConfig.suffix`.

        Returns:
            The path relative to ``output_dir``, so a training host resolves
            it from the dataset root alone rather than from this checkout.
        """
        return f"{self.images_subdir}/{name}" if self.images_subdir else name

    @property
    def images_dir(self) -> Path:
        """Directory holding one image per rendered page."""
        return self._child(self.images_subdir)

    @property
    def pdf_dir(self) -> Path:
        """Directory holding one PDF per sample."""
        return self._child(self.pdf_subdir)

    @property
    def markdown_dir(self) -> Path:
        """Directory holding the Markdown reconstruction per sample."""
        return self._child(self.markdown_subdir)

    @property
    def annotations_dir(self) -> Path:
        """Directory holding the bounding-box annotation per sample."""
        return self._child(self.annotations_subdir)

    @property
    def json_dir(self) -> Path:
        """Directory holding the JSONL training files."""
        return self._child(self.json_subdir)

    @property
    def render_manifest(self) -> Path:
        """CSV listing every rendered sample."""
        return self.output_dir / "manifest.csv"


#: Output image formats, mapped to the Pillow format name and the file suffix
#: the corpus uses. ``jpg`` is accepted as a spelling of ``jpeg`` because half
#: the world writes it that way, and rejecting it would be pedantry.
_IMAGE_FORMATS: Final[Mapping[str, tuple[str, str]]] = {
    "png": ("PNG", ".png"),
    "jpeg": ("JPEG", ".jpg"),
    "webp": ("WEBP", ".webp"),
}

#: Alternative spellings accepted for a format name.
_IMAGE_FORMAT_ALIASES: Final[Mapping[str, str]] = {"jpg": "jpeg"}

#: Formats that can store an alpha channel. A page capture is RGBA - the point
#: of the transparent ground is that light areas are an absence of ink - so an
#: encoder that cannot hold alpha must be handed an already-flattened image.
_ALPHA_FORMATS: Final[frozenset[str]] = frozenset({"png", "webp"})

#: Every accepted spelling of a format name, aliases included. The CLI uses
#: this for ``--image-format``'s choices so the flag and the YAML key can never
#: accept different sets.
IMAGE_FORMAT_CHOICES: Final[tuple[str, ...]] = tuple(
    sorted(set(_IMAGE_FORMATS) | set(_IMAGE_FORMAT_ALIASES))
)


@dataclass(frozen=True, slots=True)
class ImageFormatConfig:
    """How a page raster is encoded on the way to disk.

    PNG is lossless and the safe default, but a 20k-page PNG corpus is several
    times the size of the same corpus as quality-92 JPEG, which matters when it
    has to reach a Colab runtime. The format is therefore a setting, and every
    consumer of a page filename derives its suffix from here rather than
    assuming ``.png``.

    Attributes:
        format: ``png``, ``jpeg`` (or ``jpg``), or ``webp``.
        quality: Encoder quality for the lossy formats, ``1``-``100``. Ignored
            by PNG and by lossless WebP.
        optimize: Whether to spend extra CPU shrinking the file. Off by
            default: at 20k pages the saving is a few percent and the cost is
            most of the encode time.
        lossless: WebP only - encode losslessly, which makes ``quality`` a
            compression-effort setting rather than a fidelity one.
    """

    format: str = "png"
    quality: int = 92
    optimize: bool = False
    lossless: bool = False

    def __post_init__(self) -> None:
        normalised = _IMAGE_FORMAT_ALIASES.get(
            str(self.format).strip().lower(), str(self.format).strip().lower()
        )
        if normalised not in _IMAGE_FORMATS:
            raise ConfigError(
                f"render.image.format must be one of {sorted(_IMAGE_FORMATS)}, "
                f"got {self.format!r}"
            )
        if normalised != self.format:
            object.__setattr__(self, "format", normalised)
        if not 1 <= self.quality <= 100:
            raise ConfigError(
                f"render.image.quality must be in 1-100, got {self.quality}"
            )

    @property
    def pillow_format(self) -> str:
        """Format name to hand :meth:`PIL.Image.Image.save`."""
        return _IMAGE_FORMATS[self.format][0]

    @property
    def suffix(self) -> str:
        """File suffix for a page raster, including the leading dot."""
        return _IMAGE_FORMATS[self.format][1]

    @property
    def supports_alpha(self) -> bool:
        """Whether this format can store an alpha channel."""
        return self.format in _ALPHA_FORMATS

    @property
    def is_lossless(self) -> bool:
        """Whether encoding preserves the pixels exactly."""
        return self.format == "png" or (self.format == "webp" and self.lossless)

    def save_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments for :meth:`PIL.Image.Image.save`.

        Returns:
            Encoder options for the configured format. JPEG is written at
            4:4:4 (``subsampling=0``) rather than Pillow's default 4:2:0: chroma
            subsampling averages colour over 2x2 blocks, which smears the thin
            strokes and diacritics of Persian type at exactly the scale the OCR
            model has to read. That is fidelity, not taste, so it is not a knob.
        """
        if self.format == "png":
            return {"optimize": self.optimize}
        if self.format == "jpeg":
            return {
                "quality": self.quality,
                "optimize": self.optimize,
                "subsampling": 0,
            }
        return {
            "quality": self.quality,
            "lossless": self.lossless,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImageFormatConfig":
        """Build the image settings from their YAML mapping.

        Args:
            payload: The ``render.image`` mapping.

        Returns:
            The validated settings.

        Raises:
            ConfigError: If a key is unknown or a value is out of range.
        """
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            raise ConfigError(
                f"render.image: unknown key(s) {sorted(unknown)}; "
                f"expected any of {sorted(known)}"
            )
        return cls(
            format=str(payload.get("format", "png")),
            quality=int(payload.get("quality", 92)),
            optimize=bool(payload.get("optimize", False)),
            lossless=bool(payload.get("lossless", False)),
        )


#: Chromium flags the render stage launches with. Font hinting is disabled so a
#: glyph's advance width does not depend on the host's freetype build - two
#: machines must produce the same layout, and therefore the same boxes, from the
#: same seed. In config rather than in code because a container or a CI runner
#: routinely needs one more flag (``--no-sandbox`` is the usual one), and
#: editing Python to add it is the wrong workflow.
_DEFAULT_CHROMIUM_ARGS: Final[tuple[str, ...]] = (
    "--font-render-hinting=none",
    "--disable-lcd-text",
    "--force-color-profile=srgb",
    "--hide-scrollbars",
)


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """Rasterization settings and the page-size pool."""

    dpi: int
    page_sizes: tuple[PageSizeSpec, ...]
    margin_mm_range: tuple[float, float] = (18.0, 26.0)
    max_pages: int = 6
    timeout_ms: int = 30_000
    image: ImageFormatConfig = field(default_factory=ImageFormatConfig)
    chromium_args: tuple[str, ...] = _DEFAULT_CHROMIUM_ARGS

    #: YAML keys this section accepts. Held explicitly because two of them do
    #: not match a field name (``margin_mm`` builds ``margin_mm_range``), so the
    #: field list alone would reject valid files.
    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "dpi",
            "page_sizes",
            "margin_mm",
            "max_pages",
            "timeout_ms",
            "image",
            "chromium_args",
        }
    )

    def __post_init__(self) -> None:
        if self.dpi <= 0:
            raise ConfigError(f"render.dpi must be positive, got {self.dpi}")
        if not self.page_sizes:
            raise ConfigError("render.page_sizes must list at least one page size")
        if sum(spec.weight for spec in self.page_sizes) <= 0:
            raise ConfigError("render.page_sizes weights must sum to a positive value")
        low, high = self.margin_mm_range
        if low <= 0 or high < low:
            raise ConfigError(
                f"render.margin_mm must be an ascending positive range, got {low}-{high}"
            )
        if self.max_pages < 1:
            raise ConfigError(f"render.max_pages must be >= 1, got {self.max_pages}")

    @property
    def device_scale_factor(self) -> float:
        """Chromium scale factor that yields the configured DPI."""
        return self.dpi / CSS_PPI

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RenderConfig":
        """Build the render settings from their YAML mapping.

        Args:
            payload: The ``render`` mapping.

        Returns:
            The validated settings.

        Raises:
            ConfigError: If a key is unknown or a value is invalid. The
                unknown-key check is why the removed ``jpeg_quality`` key now
                fails loudly instead of being silently ignored, as it was.
        """
        unknown = set(payload) - cls._KEYS
        if unknown:
            raise ConfigError(
                f"render: unknown key(s) {sorted(unknown)}; "
                f"expected any of {sorted(cls._KEYS)}"
            )
        sizes = _require(payload, "page_sizes", "render")
        if not isinstance(sizes, Sequence) or isinstance(sizes, (str, bytes)):
            raise ConfigError("render.page_sizes must be a list")
        margin = payload.get("margin_mm", {})
        args = payload.get("chromium_args")
        return cls(
            dpi=int(payload.get("dpi", 300)),
            page_sizes=tuple(PageSizeSpec.from_dict(item) for item in sizes),
            margin_mm_range=(
                float(margin.get("min", 18.0)),
                float(margin.get("max", 26.0)),
            ),
            max_pages=int(payload.get("max_pages", 6)),
            timeout_ms=int(payload.get("timeout_ms", 30_000)),
            image=ImageFormatConfig.from_dict(payload.get("image", {}) or {}),
            chromium_args=(
                tuple(str(item) for item in args)
                if args is not None
                else _DEFAULT_CHROMIUM_ARGS
            ),
        )


@dataclass(frozen=True, slots=True)
class ProbabilityConfig:
    """How often each optional page feature appears."""

    header: float = 0.9
    footer: float = 0.75
    signature: float = 1.0
    stamp: float = 0.55
    texture: float = 0.5
    element: float = 0.1
    header_slogan: float = 0.3
    page_number: float = 0.6
    religious_opener: float = 0.45
    header_category: float = 0.28
    footer_labels: float = 0.72
    sender_line: float = 0.32
    subject_underline: float = 0.34
    metadata_same_side: float = 0.12
    metadata_boxed: float = 0.34
    metadata_labels: float = 0.94

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProbabilityConfig":
        """Build the probability set, validating each value lies in ``[0, 1]``."""
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ConfigError(
                f"probabilities: unknown key(s) {sorted(unknown)}; "
                f"expected any of {sorted(known)}"
            )
        values = {
            key: _as_probability(value, f"probabilities.{key}")
            for key, value in payload.items()
        }
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ScanConfig:
    """Scanner-simulation knobs: how a clean render is degraded into a scan.

    The renderer produces a mathematically perfect page - pure geometry, even
    illumination, no grain - which no scanner ever produces. Training on clean
    pages alone teaches the model that ink is sharp and paper is flat, so this
    section describes the physical damage to reintroduce: ink softening, sensor
    and lens geometry, uneven light, and paper grain.

    Every field is either a probability or a :class:`Range` the sample draws
    from, so the corpus spans the range instead of sitting at one setting.
    """

    enabled: bool = True
    probability: float = 0.85

    # Paper: chosen once per document, because one letter is printed on one
    # sheet of paper.
    paper_tints: tuple[str, ...] = ("#FBF8F1", "#F8F6F0", "#FAFAF7", "#F6F3EA")
    texture_strength: Range = field(default_factory=lambda: Range(0.18, 0.55))
    texture_grayscale: float = 0.5
    texture_autocontrast: float = 0.55
    texture_brightness: Range = field(default_factory=lambda: Range(0.94, 1.1))

    # Ink: softening and bleed-through.
    ink_blur: Range = field(default_factory=lambda: Range(0.3, 1.1))
    ink_lighten: Range = field(default_factory=lambda: Range(0.02, 0.16))
    ink_bleed: float = 0.35
    ink_bleed_alpha: Range = field(default_factory=lambda: Range(0.03, 0.11))
    ink_dilate: float = 0.18

    # Geometry: sheet placement on the platen and lens distortion.
    rotation_deg: Range = field(default_factory=lambda: Range(-0.9, 0.9))
    perspective: Range = field(default_factory=lambda: Range(0.0, 0.005))
    warp_amplitude: Range = field(default_factory=lambda: Range(0.0, 0.009))
    barrel: Range = field(default_factory=lambda: Range(-0.01, 0.014))

    # Lighting and colour response.
    illumination: Range = field(default_factory=lambda: Range(0.04, 0.22))
    vignette: Range = field(default_factory=lambda: Range(0.0, 0.26))
    shadow: float = 0.35
    shadow_strength: Range = field(default_factory=lambda: Range(0.06, 0.26))
    gamma: Range = field(default_factory=lambda: Range(0.9, 1.1))
    brightness: Range = field(default_factory=lambda: Range(0.95, 1.06))
    contrast: Range = field(default_factory=lambda: Range(0.92, 1.05))

    # Sensor noise and the output pass a scanner's own software applies.
    gaussian_sigma: Range = field(default_factory=lambda: Range(0.8, 4.5))
    # Dust specks, in specks per megapixel - a handful on a whole page, not a
    # fraction of every pixel. Salt-and-pepper noise used to live here and was
    # removed: isolated pixels flipped to pure black and white are not a defect
    # any scanner produces, and teaching the adapter to discount that texture
    # buys nothing on a real document. See `render/scan.py::_add_specks`.
    speck_density: Range = field(default_factory=lambda: Range(0.0, 4.0))
    speck_radius: Range = field(default_factory=lambda: Range(0.7, 1.9))
    speck_strength: Range = field(default_factory=lambda: Range(0.18, 0.62))
    jpeg: float = 0.45
    jpeg_quality: Range = field(default_factory=lambda: Range(72.0, 95.0))
    resample: float = 0.35
    resample_scale: Range = field(default_factory=lambda: Range(0.7, 0.95))
    grayscale: float = 0.22
    bilevel: float = 0.06

    #: YAML key -> field name, per section, so the file stays grouped by physical
    #: effect and reads in the scanner's vocabulary while the dataclass stays
    #: flat and prefixed.
    _SECTIONS: ClassVar[Mapping[str, Mapping[str, str]]] = {
        "paper": {
            "tints": "paper_tints",
            "strength": "texture_strength",
            "grayscale": "texture_grayscale",
            "autocontrast": "texture_autocontrast",
            "brightness": "texture_brightness",
        },
        "ink": {
            "blur": "ink_blur",
            "lighten": "ink_lighten",
            "bleed": "ink_bleed",
            "bleed_alpha": "ink_bleed_alpha",
            "dilate": "ink_dilate",
        },
        "geometry": {
            "rotation_deg": "rotation_deg",
            "perspective": "perspective",
            "warp_amplitude": "warp_amplitude",
            "barrel": "barrel",
        },
        "lighting": {
            "illumination": "illumination",
            "vignette": "vignette",
            "shadow": "shadow",
            "shadow_strength": "shadow_strength",
            "gamma": "gamma",
            "brightness": "brightness",
            "contrast": "contrast",
        },
        "noise": {
            "gaussian_sigma": "gaussian_sigma",
            "speck_density": "speck_density",
            "speck_radius": "speck_radius",
            "speck_strength": "speck_strength",
            "jpeg": "jpeg",
            "jpeg_quality": "jpeg_quality",
            "resample": "resample",
            "resample_scale": "resample_scale",
            "grayscale": "grayscale",
            "bilevel": "bilevel",
        },
    }

    #: ``section.key`` -> why it is gone, for keys an older config file may still
    #: carry. Named explicitly so the loader can say what replaced a setting
    #: instead of reporting it as a typo.
    _REMOVED: ClassVar[Mapping[str, str]] = {
        "noise.salt_pepper": (
            "salt-and-pepper noise was removed - flipping isolated pixels to "
            "pure black and white is not a defect a scanner produces. Use "
            "noise.speck_density (specks per megapixel), noise.speck_radius and "
            "noise.speck_strength, which stamp soft multi-pixel dust blots"
        ),
    }

    def __post_init__(self) -> None:
        if not self.paper_tints:
            raise ConfigError("scan.paper.tints must list at least one colour")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScanConfig":
        """Build the scan settings from the ``scan`` section.

        Args:
            payload: The section's mapping, grouped into ``paper``, ``ink``,
                ``geometry``, ``lighting`` and ``noise`` subsections.

        Returns:
            The validated settings.

        Raises:
            ConfigError: On an unknown key, a malformed range, or an out-of-range
                probability.
        """
        unknown = set(payload) - ({"enabled", "probability"} | set(cls._SECTIONS))
        if unknown:
            raise ConfigError(
                f"scan: unknown key(s) {sorted(unknown)}; expected any of "
                f"{sorted({'enabled', 'probability'} | set(cls._SECTIONS))}"
            )
        defaults = cls()
        values: dict[str, Any] = {
            "enabled": bool(payload.get("enabled", defaults.enabled)),
            "probability": _as_probability(
                payload.get("probability", defaults.probability), "scan.probability"
            ),
        }
        for section, aliases in cls._SECTIONS.items():
            block = payload.get(section) or {}
            if not isinstance(block, Mapping):
                raise ConfigError(f"scan.{section}: expected a mapping")
            unknown_keys = set(block) - set(aliases)
            if unknown_keys:
                retired = sorted(
                    key for key in unknown_keys if f"{section}.{key}" in cls._REMOVED
                )
                if retired:
                    reasons = "; ".join(
                        f"scan.{section}.{key}: {cls._REMOVED[f'{section}.{key}']}"
                        for key in retired
                    )
                    raise ConfigError(reasons)
                raise ConfigError(
                    f"scan.{section}: unknown key(s) {sorted(unknown_keys)}; "
                    f"expected any of {sorted(aliases)}"
                )
            for alias, name in aliases.items():
                if alias not in block:
                    continue
                values[name] = cls._parse_field(
                    name, block[alias], f"scan.{section}.{alias}", defaults
                )
        return cls(**values)

    @staticmethod
    def _parse_field(
        name: str, value: Any, context: str, defaults: "ScanConfig"
    ) -> Any:
        """Coerce one YAML value to the type its field declares."""
        current = getattr(defaults, name)
        if name == "paper_tints":
            tints = tuple(str(item).strip() for item in value or ())
            if not tints:
                raise ConfigError(f"{context}: must list at least one colour")
            return tints
        if isinstance(current, Range):
            return Range.parse(value, context, current)
        return _as_probability(value, context)


@dataclass(frozen=True, slots=True)
class FontSpec:
    """One selectable font family.

    Attributes:
        name: The family name as fontconfig reports it, e.g. ``B Nazanin``.
            An empty name in a display pool means "reuse the body family".
        weight: Relative sampling weight, not a percentage.
        scale: Optical-size correction. The legacy B-series faces draw small
            for their nominal point size, so the composer multiplies the
            sampled point size by this to keep families visually comparable.
    """

    name: str
    weight: float = 1.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ConfigError(
                f"theme font {self.name!r}: weight must be > 0, got {self.weight}"
            )
        if self.scale <= 0:
            raise ConfigError(
                f"theme font {self.name!r}: scale must be > 0, got {self.scale}"
            )


def _parse_weight_pool(
    payload: Any,
    context: str,
    default: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    """Parse a weighted choice pool from YAML.

    A weighted pool reads best in YAML as a mapping - ``rule: 30`` - but has to
    reach :func:`~fa_ocr_lora.dataset.common.io_utils.weighted_choice` as an
    ordered sequence of pairs, because a seeded draw over an unordered mapping
    would depend on insertion order. Python preserves YAML's order, so the pairs
    come out in the order the file lists them and the same seed keeps picking the
    same entry.

    Args:
        payload: The YAML value: a ``{name: weight}`` mapping, a sequence of
            ``{name, weight}`` mappings, or ``None`` for the default.
        context: Dotted config path, for error messages.
        default: Pool to use when the key is absent.

    Returns:
        The pool as ordered ``(name, weight)`` pairs.

    Raises:
        ConfigError: If the payload is neither shape, a name is empty, or a
            weight is not a number.
    """
    if payload is None:
        return default
    entries: list[tuple[str, float]] = []
    if isinstance(payload, Mapping):
        items: list[tuple[Any, Any]] = list(payload.items())
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        items = []
        for index, entry in enumerate(payload):
            if not isinstance(entry, Mapping):
                raise ConfigError(
                    f"{context}[{index}]: expected a mapping with name and weight, "
                    f"got {entry!r}"
                )
            items.append((entry.get("name", ""), entry.get("weight", 1.0)))
    else:
        raise ConfigError(
            f"{context}: expected a {{name: weight}} mapping or a list of them, "
            f"got {payload!r}"
        )
    for name, weight in items:
        label = str(name).strip()
        if not label:
            raise ConfigError(f"{context}: every entry needs a name")
        try:
            entries.append((label, float(weight)))
        except (TypeError, ValueError) as error:
            raise ConfigError(
                f"{context}[{label!r}]: weight must be a number, got {weight!r}"
            ) from error
    if not entries:
        raise ConfigError(f"{context}: must list at least one entry")
    return tuple(entries)


def _parse_font_pool(
    payload: Any,
    field: str,
    *,
    allow_empty_name: bool,
) -> tuple[FontSpec, ...]:
    """Parse a weighted font pool from YAML.

    Accepts either a plain list of family names or a list of
    ``{name, weight, scale}`` mappings, so a minimal config stays readable.

    Args:
        payload: The YAML value for this field.
        field: Dotted config path, for error messages.
        allow_empty_name: Whether ``""`` is a legal family (display pools use
            it to mean "reuse the body family").

    Returns:
        The parsed pool.

    Raises:
        ConfigError: If the pool is empty or an entry is malformed.
    """
    if not payload:
        raise ConfigError(f"{field} must list at least one family")
    specs: list[FontSpec] = []
    for index, entry in enumerate(payload):
        if isinstance(entry, str):
            specs.append(FontSpec(name=entry))
            continue
        if not isinstance(entry, Mapping):
            raise ConfigError(
                f"{field}[{index}]: expected a family name or a mapping, "
                f"got {type(entry).__name__}"
            )
        unknown = set(entry) - {"name", "weight", "scale"}
        if unknown:
            raise ConfigError(f"{field}[{index}]: unknown key(s) {sorted(unknown)}")
        name = str(entry.get("name", "")).strip()
        if not name and not allow_empty_name:
            raise ConfigError(f"{field}[{index}]: name is required")
        specs.append(
            FontSpec(
                name=name,
                weight=float(entry.get("weight", 1.0)),
                scale=float(entry.get("scale", 1.0)),
            )
        )
    return tuple(specs)


#: Fallback seal-ink palette for a config file with no ``theme`` section. Named
#: rather than inlined because ``from_dict`` needs the same value: reading
#: ``ThemeConfig.stamp_inks`` off the class yields a slot descriptor, not the
#: default, so a file omitting the key used to crash instead of falling back.
_DEFAULT_STAMP_INKS: Final[tuple[str, ...]] = ("#1B3F94", "#A81E22", "#4B2A7B")

#: Canonical header/footer bar treatments. Defined here rather than in
#: ``layout/theme.py`` so ``theme.bar_weights`` can be validated against them at
#: load time - a mistyped style name would otherwise reach the CSS template as
#: an unknown class and render nothing, silently. ``layout.theme.BarStyle``
#: re-exports these as named attributes for readable call sites.
BAR_STYLES: Final[tuple[str, ...]] = (
    "solid",
    "tinted_band",
    "rule",
    "double_rule",
    "side_stripe",
    "none",
)

#: Relative weights for the bar styles, used when ``theme.bar_weights`` is
#: absent. A plain rule is the most common treatment on real letterheads, so it
#: dominates without excluding the rest.
_DEFAULT_BAR_WEIGHTS: Final[tuple[tuple[str, float], ...]] = (
    ("rule", 30.0),
    ("tinted_band", 22.0),
    ("solid", 16.0),
    ("double_rule", 14.0),
    ("side_stripe", 10.0),
    ("none", 8.0),
)

#: Near-black ink candidates. Pure ``#000`` is rare in print, so the pool leans
#: to very dark neutrals.
_DEFAULT_INK_CANDIDATES: Final[tuple[str, ...]] = (
    "#1A1A1A",
    "#212121",
    "#141414",
    "#232323",
)

#: Page surfaces: warm and cool paper tints. Pure ``#FFFFFF`` is deliberately
#: absent - no sheet of paper is white, and the surface is painted with an alpha
#: so the scanned paper underneath shows through it.
_DEFAULT_SURFACE_CANDIDATES: Final[tuple[str, ...]] = (
    "#FCFAF4",
    "#FBF9F3",
    "#FDFCF8",
    "#FAF9F5",
    "#F9FAFB",
    "#FBFAF6",
)

#: Corner radii the sheet's panels may use. ``0.0`` appears twice because square
#: corners are twice as likely as any rounded value.
_DEFAULT_CORNER_RADII_PX: Final[tuple[float, ...]] = (0.0, 0.0, 2.0, 4.0, 6.0)


#: Which margin a page element hugs. Same rationale as :data:`BAR_STYLES`: the
#: names reach the CSS template as class suffixes, so a typo in
#: ``layout.logo_sides`` must fail at load rather than silently place nothing.
PAGE_SIDES: Final[tuple[str, ...]] = ("right", "center", "left")

#: How the footer arranges its contact fields. ``inline`` runs them on one line,
#: ``stacked`` one per line, ``split`` pushes the last field to the far margin.
FOOTER_LAYOUT_STYLES: Final[tuple[str, ...]] = ("inline", "stacked", "split")

#: The contact fields a letter record can carry, in postal order. Used to
#: validate ``layout.contact_orders``, whose entries name a subset of these.
CONTACT_FIELDS: Final[tuple[str, ...]] = ("address", "phone", "email", "website")

#: The administrative metadata fields, which CLAUDE.md 5.2 makes the
#: highest-priority region of the page. Used to validate
#: ``layout.metadata_orders``: a misspelled key there would drop the field
#: silently, and dropping the registration number is exactly the failure the
#: eval metric is meant to catch.
METADATA_FIELDS: Final[tuple[str, ...]] = (
    "letter_number",
    "registration_number",
    "date",
    "attachment",
    "national_id",
)

#: Border treatments a rubber stamp can take. Same rationale as
#: :data:`BAR_STYLES` - the name selects an SVG border branch, so an unknown one
#: would draw no border at all and the seal would print as bare text.
STAMP_SHAPES: Final[tuple[str, ...]] = (
    "circle",
    "double_circle",
    "oval",
    "rounded_square",
    "square",
    "hexagon",
)

#: Shapes whose ring text follows an arc. Not a knob: text curved around a
#: corner is not something real stamps do, so this is a fact about each shape
#: rather than a choice. Kept beside :data:`STAMP_SHAPES` so the two cannot
#: drift, and validated as a subset of it.
STAMP_ROUND_SHAPES: Final[frozenset[str]] = frozenset(
    {"circle", "double_circle", "oval"}
)

#: Where the letterhead logo sits. Right dominates because that is the head of
#: the line in an RTL page, and so where Iranian letterhead puts the mark.
_DEFAULT_LOGO_SIDES: Final[tuple[tuple[str, float], ...]] = (
    ("right", 52.0),
    ("center", 26.0),
    ("left", 22.0),
)

#: Which margin the closing block hugs. Persian letters sign at the inline-end
#: (left) side far more often than centred.
_DEFAULT_SIGNATURE_SIDES: Final[tuple[tuple[str, float], ...]] = (
    ("left", 58.0),
    ("center", 26.0),
    ("right", 16.0),
)

#: How footer fields are laid out, by relative weight.
_DEFAULT_FOOTER_LAYOUTS: Final[tuple[tuple[str, float], ...]] = (
    ("inline", 62.0),
    ("stacked", 26.0),
    ("split", 12.0),
)

#: Field orders for the footer. Address first is the postal convention; the
#: short forms exist because a one-line footer is very common. Picked with a
#: uniform ``rng.choice``, not a weighted draw, so the order of these entries
#: does not matter - only how many times each shape appears.
_DEFAULT_CONTACT_ORDERS: Final[tuple[tuple[str, ...], ...]] = (
    ("address", "phone", "email", "website"),
    ("address", "phone", "website"),
    ("phone", "email", "website"),
    ("address", "phone"),
    ("phone", "website"),
    ("address", "phone", "email"),
)

#: Field orders for the metadata block. ``letter_number`` before ``date`` is
#: near-universal; the rest genuinely varies between organisations. A field the
#: record leaves empty is dropped, so a longer order is not a taller block.
_DEFAULT_METADATA_ORDERS: Final[tuple[tuple[str, ...], ...]] = (
    ("letter_number", "date", "attachment"),
    ("letter_number", "date", "attachment", "national_id"),
    ("registration_number", "date", "attachment"),
    ("letter_number", "registration_number", "date", "attachment"),
    ("date", "letter_number", "attachment"),
    ("letter_number", "date"),
)


@dataclass(frozen=True, slots=True)
class ThemeConfig:
    """Typography and palette constraints.

    Everything here is taste: which paper tints and near-blacks the sample may
    draw from, how translucent each layer of ink is, how much bigger a heading
    is than the body. The one thing that is *not* taste is
    :attr:`min_contrast_ratio`, which is what stops a seed producing a page no
    model could read.

    The alpha ranges are load-bearing beyond looks. Nothing on a page is opaque
    white: light areas are transparent so the paper texture composited beneath
    the sheet stays visible, and the contrast floor is therefore measured on the
    *composite* rather than on the nominal colour.

    Attributes:
        font_families: Body faces with their sampling weight and optical scale.
        display_font_families: Faces for the subject line and headings; an
            entry with an empty name means "reuse the body face".
        body_font_pt_range: Body point size before the family's scale factor.
        line_height_range: Body line height, as a multiple of the point size.
        min_contrast_ratio: WCAG AA floor for body ink over its surface.
        stamp_inks: Seal-ink colours for the rubber stamp.
        bar_weights: Relative weights for the header bar treatment. The footer
            draws from the same pool minus ``side_stripe``, which runs the full
            page height and so cannot also be a footer.
        ink_candidates: Near-black body inks.
        surface_candidates: Page surface tints.
        corner_radii_px: Panel corner radii to choose between.
        surface_alpha: Opacity the surface tint is painted at.
        ink_alpha: Opacity body ink is painted at.
        tint_alpha: Opacity of tinted bands and panels.
        panel_alpha: Opacity of solid panels.
        rule_alpha: Opacity of rules and hairlines.
        muted_ink_lighten: How far secondary text is lightened from the body
            ink before its own, lower contrast floor is enforced.
        muted_ink_contrast_ratio: Contrast floor for that secondary text. Lower
            than the body's because WCAG AA only demands 4.5 of body copy.
        tint_lighten: How far the brand colour is lightened into a band tint.
        rule_lighten: How far the accent is lightened into a rule colour.
        heading_scale: Heading size as a multiple of the body size.
        subject_scale: Subject-line size as a multiple of the body size.
        rule_width_px: Stroke width for rules.
        justify_probability: Chance the body text is justified rather than
            ragged.
        table_zebra_probability: Chance a table gets alternating row shading.
        darken_step: How much a colour is darkened per attempt while chasing
            the contrast floor.
        max_darken_steps: How many attempts before falling back to near-black.
        bar_text_light: Text colour on a solid brand bar when the bar is dark
            enough to carry it.
        bar_text_dark: Starting colour when it is not, darkened until it passes.
    """

    font_families: tuple[FontSpec, ...] = (FontSpec("B Nazanin"),)
    display_font_families: tuple[FontSpec, ...] = (FontSpec("B Titr"),)
    body_font_pt_range: tuple[float, float] = (10.5, 13.0)
    line_height_range: tuple[float, float] = (1.6, 2.0)
    min_contrast_ratio: float = 4.5
    stamp_inks: tuple[str, ...] = _DEFAULT_STAMP_INKS
    bar_weights: tuple[tuple[str, float], ...] = _DEFAULT_BAR_WEIGHTS
    ink_candidates: tuple[str, ...] = _DEFAULT_INK_CANDIDATES
    surface_candidates: tuple[str, ...] = _DEFAULT_SURFACE_CANDIDATES
    corner_radii_px: tuple[float, ...] = _DEFAULT_CORNER_RADII_PX
    surface_alpha: Range = Range(0.12, 0.34)
    ink_alpha: Range = Range(0.88, 0.97)
    tint_alpha: Range = Range(0.42, 0.78)
    panel_alpha: Range = Range(0.9, 0.99)
    rule_alpha: Range = Range(0.7, 0.94)
    muted_ink_lighten: Range = Range(0.28, 0.42)
    muted_ink_contrast_ratio: float = 3.0
    tint_lighten: Range = Range(0.86, 0.94)
    rule_lighten: Range = Range(0.35, 0.6)
    heading_scale: Range = Range(1.04, 1.16)
    subject_scale: Range = Range(1.08, 1.24)
    rule_width_px: Range = Range(0.8, 2.4)
    justify_probability: float = 0.65
    table_zebra_probability: float = 0.55
    darken_step: float = 0.08
    max_darken_steps: int = 24
    bar_text_light: str = "#FFFFFF"
    bar_text_dark: str = "#1A1A1A"

    #: Keys accepted in the YAML ``theme`` section. The dataclass fields plus
    #: the two that are spelled differently in the file.
    _KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "font_families",
            "display_font_families",
            "body_font_pt",
            "line_height",
            "min_contrast_ratio",
            "stamp_inks",
            "bar_weights",
            "ink_candidates",
            "surface_candidates",
            "corner_radii_px",
            "surface_alpha",
            "ink_alpha",
            "tint_alpha",
            "panel_alpha",
            "rule_alpha",
            "muted_ink_lighten",
            "muted_ink_contrast_ratio",
            "tint_lighten",
            "rule_lighten",
            "heading_scale",
            "subject_scale",
            "rule_width_px",
            "justify_probability",
            "table_zebra_probability",
            "darken_step",
            "max_darken_steps",
            "bar_text_light",
            "bar_text_dark",
        }
    )

    #: Ranges parsed straight from a same-named key, mapped to their default.
    _RANGES: ClassVar[Mapping[str, tuple[float, float]]] = {
        "surface_alpha": (0.12, 0.34),
        "ink_alpha": (0.88, 0.97),
        "tint_alpha": (0.42, 0.78),
        "panel_alpha": (0.9, 0.99),
        "rule_alpha": (0.7, 0.94),
        "muted_ink_lighten": (0.28, 0.42),
        "tint_lighten": (0.86, 0.94),
        "rule_lighten": (0.35, 0.6),
        "heading_scale": (1.04, 1.16),
        "subject_scale": (1.08, 1.24),
        "rule_width_px": (0.8, 2.4),
    }

    def __post_init__(self) -> None:
        if not self.font_families:
            raise ConfigError("theme.font_families must list at least one family")
        if not self.display_font_families:
            raise ConfigError(
                "theme.display_font_families must list at least one family"
            )
        if self.min_contrast_ratio < 1.0:
            raise ConfigError(
                f"theme.min_contrast_ratio must be >= 1, got {self.min_contrast_ratio}"
            )
        if self.muted_ink_contrast_ratio < 1.0:
            raise ConfigError(
                "theme.muted_ink_contrast_ratio must be >= 1, got "
                f"{self.muted_ink_contrast_ratio}"
            )
        for field_name in ("ink_candidates", "surface_candidates", "corner_radii_px"):
            if not getattr(self, field_name):
                raise ConfigError(f"theme.{field_name} must not be empty")
        # Every colour is checked for spelling here rather than at use: a
        # malformed value reaches CSS as an ignored declaration and renders the
        # page in the browser default, which looks plausible and is wrong.
        for field_name in ("ink_candidates", "surface_candidates", "stamp_inks"):
            for index, colour in enumerate(getattr(self, field_name)):
                _as_hex_colour(colour, f"theme.{field_name}[{index}]")
        _as_hex_colour(self.bar_text_light, "theme.bar_text_light")
        _as_hex_colour(self.bar_text_dark, "theme.bar_text_dark")
        if not self.bar_weights:
            raise ConfigError("theme.bar_weights must name at least one style")
        unknown_styles = {style for style, _ in self.bar_weights} - set(BAR_STYLES)
        if unknown_styles:
            raise ConfigError(
                f"theme.bar_weights: unknown style(s) {sorted(unknown_styles)}; "
                f"expected any of {sorted(BAR_STYLES)}"
            )
        if any(weight < 0 for _, weight in self.bar_weights):
            raise ConfigError("theme.bar_weights: weights must not be negative")
        if sum(weight for _, weight in self.bar_weights) <= 0:
            raise ConfigError("theme.bar_weights: at least one weight must be positive")
        # A footer never uses the full-height side stripe, so the pool must hold
        # something else or the footer has nothing to draw.
        footer_total = sum(
            weight for style, weight in self.bar_weights if style != "side_stripe"
        )
        if footer_total <= 0:
            raise ConfigError(
                "theme.bar_weights: side_stripe cannot be the only style; the "
                "footer draws from this pool without it"
            )
        for name in ("justify_probability", "table_zebra_probability"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"theme.{name} must be in 0-1, got {value}")
        if not 0.0 < self.darken_step <= 1.0:
            raise ConfigError(
                f"theme.darken_step must be in (0, 1], got {self.darken_step}"
            )
        if self.max_darken_steps < 1:
            raise ConfigError(
                f"theme.max_darken_steps must be >= 1, got {self.max_darken_steps}"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ThemeConfig":
        """Build the theme constraints from their YAML mapping.

        Args:
            payload: The ``theme`` section.

        Returns:
            The validated constraints.

        Raises:
            ConfigError: If a key is unknown or a value is out of range.
        """
        unknown = set(payload) - cls._KEYS
        if unknown:
            raise ConfigError(
                f"theme: unknown key(s) {sorted(unknown)}; "
                f"expected any of {sorted(cls._KEYS)}"
            )
        font_pt = payload.get("body_font_pt", {})
        line_height = payload.get("line_height", {})
        ranges = {
            name: Range.parse(payload.get(name), f"theme.{name}", Range(*default))
            for name, default in cls._RANGES.items()
        }
        return cls(
            font_families=_parse_font_pool(
                payload.get("font_families") or [FontSpec("B Nazanin").name],
                "theme.font_families",
                allow_empty_name=False,
            ),
            display_font_families=_parse_font_pool(
                payload.get("display_font_families") or [FontSpec("B Titr").name],
                "theme.display_font_families",
                allow_empty_name=True,
            ),
            body_font_pt_range=(
                float(font_pt.get("min", 10.5)),
                float(font_pt.get("max", 13.0)),
            ),
            line_height_range=(
                float(line_height.get("min", 1.6)),
                float(line_height.get("max", 2.0)),
            ),
            min_contrast_ratio=float(payload.get("min_contrast_ratio", 4.5)),
            stamp_inks=tuple(payload.get("stamp_inks", _DEFAULT_STAMP_INKS)),
            bar_weights=_parse_weight_pool(
                payload.get("bar_weights"), "theme.bar_weights", _DEFAULT_BAR_WEIGHTS
            ),
            ink_candidates=tuple(
                payload.get("ink_candidates") or _DEFAULT_INK_CANDIDATES
            ),
            surface_candidates=tuple(
                payload.get("surface_candidates") or _DEFAULT_SURFACE_CANDIDATES
            ),
            corner_radii_px=tuple(
                float(value)
                for value in (
                    payload.get("corner_radii_px") or _DEFAULT_CORNER_RADII_PX
                )
            ),
            muted_ink_contrast_ratio=float(
                payload.get("muted_ink_contrast_ratio", 3.0)
            ),
            justify_probability=float(payload.get("justify_probability", 0.65)),
            table_zebra_probability=float(
                payload.get("table_zebra_probability", 0.55)
            ),
            darken_step=float(payload.get("darken_step", 0.08)),
            max_darken_steps=int(payload.get("max_darken_steps", 24)),
            bar_text_light=str(payload.get("bar_text_light", "#FFFFFF")),
            bar_text_dark=str(payload.get("bar_text_dark", "#1A1A1A")),
            **ranges,
        )


def _parse_orders(
    payload: Any, default: tuple[tuple[str, ...], ...]
) -> tuple[tuple[str, ...], ...]:
    """Normalise a list-of-lists field-order pool.

    Args:
        payload: A sequence of sequences of field names, or ``None``.
        default: Pool to use when the key is absent or empty.

    Returns:
        The pool as nested tuples. Names are not validated here - the owning
        dataclass's ``__post_init__`` checks them against its own vocabulary,
        so a directly-constructed config is validated the same as a parsed one.
    """
    if not payload:
        return default
    return tuple(tuple(str(name) for name in order) for order in payload)


@dataclass(frozen=True, slots=True)
class LayoutConfig:
    """Where things sit on the sheet, and how large they are drawn.

    This is page geometry and placement, kept apart from :class:`ThemeConfig`
    (which decides colour and type) and from :class:`ProbabilityConfig` (which
    decides whether a feature appears at all). The split matters in practice:
    retuning how big a logo prints should never risk touching a contrast floor.

    Every pool here is drawn once per document, so widening one changes the
    variety of the corpus rather than the look of a single page.

    Attributes:
        stripe_mm: Width of the full-height brand stripe.
        logo_mm: Letterhead logo height. Below ~9mm a 24x24 mark stops being
            legible at 300 DPI; above ~22mm it competes with the company name.
        logo_sides: Weighted margin the logo hugs.
        signature_sides: Weighted margin the closing block hugs.
        footer_layouts: Weighted footer arrangement.
        contact_orders: Candidate field orders for the footer, chosen
            uniformly. Fields the record does not carry are dropped, so a
            longer order is not a longer footer.
        metadata_orders: Candidate field orders for the administrative
            metadata block, chosen the same way.
    """

    stripe_mm: Range = Range(6.0, 11.0)
    logo_mm: Range = Range(9.0, 22.0)
    logo_sides: tuple[tuple[str, float], ...] = _DEFAULT_LOGO_SIDES
    signature_sides: tuple[tuple[str, float], ...] = _DEFAULT_SIGNATURE_SIDES
    footer_layouts: tuple[tuple[str, float], ...] = _DEFAULT_FOOTER_LAYOUTS
    contact_orders: tuple[tuple[str, ...], ...] = _DEFAULT_CONTACT_ORDERS
    metadata_orders: tuple[tuple[str, ...], ...] = _DEFAULT_METADATA_ORDERS

    #: Weighted pools, mapped to the vocabulary their names must come from.
    _POOLS: ClassVar[Mapping[str, tuple[str, ...]]] = {
        "logo_sides": PAGE_SIDES,
        "signature_sides": PAGE_SIDES,
        "footer_layouts": FOOTER_LAYOUT_STYLES,
    }

    #: Field-order pools, mapped to the field names their entries may use.
    _ORDERS: ClassVar[Mapping[str, tuple[str, ...]]] = {
        "contact_orders": CONTACT_FIELDS,
        "metadata_orders": METADATA_FIELDS,
    }

    def __post_init__(self) -> None:
        for name, vocabulary in self._POOLS.items():
            pool: tuple[tuple[str, float], ...] = getattr(self, name)
            if not pool:
                raise ConfigError(f"layout.{name} must name at least one option")
            unknown = {value for value, _ in pool} - set(vocabulary)
            if unknown:
                raise ConfigError(
                    f"layout.{name}: unknown value(s) {sorted(unknown)}; "
                    f"expected any of {sorted(vocabulary)}"
                )
            if any(weight < 0 for _, weight in pool):
                raise ConfigError(f"layout.{name}: weights must not be negative")
            if sum(weight for _, weight in pool) <= 0:
                raise ConfigError(f"layout.{name}: at least one weight must be positive")
        for name, fields in self._ORDERS.items():
            orders: tuple[tuple[str, ...], ...] = getattr(self, name)
            if not orders:
                raise ConfigError(f"layout.{name} must list at least one order")
            for order in orders:
                if not order:
                    raise ConfigError(f"layout.{name}: an order must not be empty")
                unknown_fields = set(order) - set(fields)
                if unknown_fields:
                    raise ConfigError(
                        f"layout.{name}: unknown field(s) {sorted(unknown_fields)}; "
                        f"expected any of {sorted(fields)}"
                    )
                if len(set(order)) != len(order):
                    raise ConfigError(
                        f"layout.{name}: {list(order)} repeats a field; it would "
                        "be printed twice"
                    )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LayoutConfig":
        """Build the layout geometry from its YAML mapping.

        Args:
            payload: The ``layout`` section.

        Returns:
            The validated geometry.

        Raises:
            ConfigError: If a key is unknown or a value is out of range.
        """
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ConfigError(
                f"layout: unknown key(s) {sorted(unknown)}; "
                f"expected any of {sorted(known)}"
            )
        orders = payload.get("contact_orders")
        metadata_orders = payload.get("metadata_orders")
        return cls(
            stripe_mm=Range.parse(
                payload.get("stripe_mm"), "layout.stripe_mm", Range(6.0, 11.0)
            ),
            logo_mm=Range.parse(
                payload.get("logo_mm"), "layout.logo_mm", Range(9.0, 22.0)
            ),
            logo_sides=_parse_weight_pool(
                payload.get("logo_sides"), "layout.logo_sides", _DEFAULT_LOGO_SIDES
            ),
            signature_sides=_parse_weight_pool(
                payload.get("signature_sides"),
                "layout.signature_sides",
                _DEFAULT_SIGNATURE_SIDES,
            ),
            footer_layouts=_parse_weight_pool(
                payload.get("footer_layouts"),
                "layout.footer_layouts",
                _DEFAULT_FOOTER_LAYOUTS,
            ),
            contact_orders=_parse_orders(orders, _DEFAULT_CONTACT_ORDERS),
            metadata_orders=_parse_orders(metadata_orders, _DEFAULT_METADATA_ORDERS),
        )


#: Pen colours, by relative weight. Blue-black leads because Iranian
#: administrative practice signs in blue.
_DEFAULT_PEN_INKS: Final[tuple[tuple[str, float], ...]] = (
    ("#12295C", 34.0),
    ("#1B3A7A", 24.0),
    ("#1A1A1A", 18.0),
    ("#0E1D3D", 14.0),
    ("#2B2B38", 10.0),
)


#: Border shapes, by relative weight. The round forms dominate because that is
#: what an Iranian company seal almost always is; the angular ones exist so the
#: model does not learn "stamp == circle".
_DEFAULT_STAMP_SHAPES: Final[tuple[tuple[str, float], ...]] = (
    ("circle", 30.0),
    ("double_circle", 26.0),
    ("oval", 14.0),
    ("rounded_square", 12.0),
    ("square", 10.0),
    ("hexagon", 8.0),
)


@dataclass(frozen=True, slots=True)
class StampConfig:
    """The rubber stamp (مهر) pressed beside the signature.

    Everything here is per-sample variation in how hard and how straight the
    pad was pressed. The stamp's *internal* geometry - its 200-unit viewBox,
    the border and text radii, the ring font sizes - deliberately stays in
    ``layout/stamp.py``: those numbers are only meaningful relative to each
    other, and a YAML edit to one of them would break the mark rather than
    restyle it.

    Attributes:
        shapes: Weighted border treatments.
        width_mm: Rendered width on the page; the mark is square. Real company
            seals run 30-42 mm across, and the range brackets that.
        rotate_deg: Tilt of the whole mark. A stamp is pressed by hand, so this
            is sloppier than a signature's.
        opacity: Overall ink weight. Below ~0.7 the mark stops reading as
            deliberate; at 1.0 it looks printed rather than pressed.
        wobble: How far ``feTurbulence`` shoves the mark around, in viewBox
            units. Two units on a 200-unit box is a visible wobble at the
            border without smearing the text.
        erosion: Threshold on the hole mask, as an alpha multiplier. More
            negative eats less ink: the intercept shifts the whole grain curve
            down, so only its brightest peaks clear zero and survive as holes.
            At ``-1.2`` too much of the field cleared and thin borders came out
            dotted rather than pressed, so the range starts below that.
        stroke_width: Border weight in viewBox units.
        logo_fraction: The logo's share of the box's width.
        ink_gain: Gain applied to the logo's density before it is flooded with
            ink. Density is ``1 - luminance``, so a light brand fill lands near
            0.2 and prints as a watermark next to saturated ring text at low
            gain. At 4.0 anything darker than ~75% luminance inks solid, while
            white knockouts (density 0) and the transparent margin still drop
            out - a rubber pad transfers one weight of ink or none.
        name_prefix_probability: Chance the upper ring reads ``مهر <name>``
            rather than the bare organisation name.
    """

    shapes: tuple[tuple[str, float], ...] = _DEFAULT_STAMP_SHAPES
    width_mm: Range = Range(28.0, 42.0)
    rotate_deg: Range = Range(-18.0, 18.0)
    opacity: Range = Range(0.72, 0.94)
    wobble: Range = Range(1.4, 3.0)
    erosion: Range = Range(-3.0, -1.8)
    stroke_width: Range = Range(2.6, 4.4)
    logo_fraction: float = 0.42
    ink_gain: float = 4.0
    name_prefix_probability: float = 0.35

    def __post_init__(self) -> None:
        if not self.shapes:
            raise ConfigError("stamp.shapes must name at least one shape")
        unknown = {shape for shape, _ in self.shapes} - set(STAMP_SHAPES)
        if unknown:
            raise ConfigError(
                f"stamp.shapes: unknown shape(s) {sorted(unknown)}; "
                f"expected any of {sorted(STAMP_SHAPES)}"
            )
        if any(weight < 0 for _, weight in self.shapes):
            raise ConfigError("stamp.shapes: weights must not be negative")
        if sum(weight for _, weight in self.shapes) <= 0:
            raise ConfigError("stamp.shapes: at least one weight must be positive")
        if not 0.0 < self.logo_fraction <= 1.0:
            raise ConfigError(
                "stamp.logo_fraction must be within (0, 1] - it is the logo's "
                f"share of the seal's box, got {self.logo_fraction}"
            )
        if self.ink_gain <= 0:
            raise ConfigError(
                "stamp.ink_gain must be positive, or the logo would print as "
                f"nothing at all, got {self.ink_gain}"
            )
        _as_probability(self.name_prefix_probability, "stamp.name_prefix_probability")
        if self.opacity.minimum <= 0 or self.opacity.maximum > 1:
            raise ConfigError(
                f"stamp.opacity must stay within (0, 1], got {self.opacity}"
            )
        if self.width_mm.minimum <= 0:
            raise ConfigError(
                f"stamp.width_mm.min must be positive, got {self.width_mm.minimum}"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StampConfig":
        """Build the stamp settings from their YAML mapping.

        Args:
            payload: The ``stamp`` section.

        Returns:
            The validated settings.

        Raises:
            ConfigError: If a key is unknown or a value is out of range.
        """
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ConfigError(
                f"stamp: unknown key(s) {sorted(unknown)}; "
                f"expected any of {sorted(known)}"
            )
        defaults = cls()
        ranges = {
            name: Range.parse(
                payload.get(name), f"stamp.{name}", getattr(defaults, name)
            )
            for name in (
                "width_mm",
                "rotate_deg",
                "opacity",
                "wobble",
                "erosion",
                "stroke_width",
            )
        }
        return cls(
            shapes=_parse_weight_pool(
                payload.get("shapes"), "stamp.shapes", _DEFAULT_STAMP_SHAPES
            ),
            logo_fraction=float(
                payload.get("logo_fraction", defaults.logo_fraction)
            ),
            ink_gain=float(payload.get("ink_gain", defaults.ink_gain)),
            name_prefix_probability=_as_probability(
                payload.get(
                    "name_prefix_probability", defaults.name_prefix_probability
                ),
                "stamp.name_prefix_probability",
            ),
            **ranges,
        )


@dataclass(frozen=True, slots=True)
class SignatureConfig:
    """How a UTSig scan is lifted off its paper and drawn onto the page.

    The first three values govern the *extraction*, not the look: they decide
    which pixels of a scan count as ink at all. They are therefore part of
    ``_extract_alpha``'s cache key in ``layout/signature.py`` - two configs in
    one process must not share a mask cut at different floors.

    Attributes:
        paper_floor: Alpha at or below which a pixel counts as paper rather
            than ink, on the 0-255 scale. Derived from the corpus: the scans'
            background sits at 240-254 (alpha 1-15), while the faintest real
            stroke edges land well above this. Raising it eats thin strokes;
            lowering it prints the scan's grey rectangle onto the letterhead.
        crop_pad: Padding kept around the ink's bounding box, as a fraction of
            the crop, so a rotation does not clip a stroke ending at the edge.
        max_edge_px: Longest edge of the emitted PNG. A 46 mm signature at 300
            DPI needs 543 device pixels, so anything past this is resolution
            the page cannot show; capping holds the base64 payload near 80 KB.
        pen_inks: Weighted pen colours.
        width_mm: Rendered signature width on the page.
        rotate_deg: Tilt applied to the ink. A signature is written by hand
            against a ruled line, so the range is deliberately narrow.
        overlap_mm: How far the signature rides up over the signatory title,
            as real signatures do rather than sitting neatly below it.
    """

    paper_floor: int = 26
    crop_pad: float = 0.02
    max_edge_px: int = 720
    pen_inks: tuple[tuple[str, float], ...] = _DEFAULT_PEN_INKS
    width_mm: Range = Range(28.0, 46.0)
    rotate_deg: Range = Range(-7.0, 7.0)
    overlap_mm: Range = Range(0.0, 5.0)

    def __post_init__(self) -> None:
        if not 0 <= self.paper_floor <= 254:
            raise ConfigError(
                "signature.paper_floor must be within [0, 254]; 255 would cut "
                f"every pixel and leave no ink, got {self.paper_floor}"
            )
        if not 0.0 <= self.crop_pad <= 0.5:
            raise ConfigError(
                f"signature.crop_pad must be within [0.0, 0.5], got {self.crop_pad}"
            )
        if self.max_edge_px < 2:
            raise ConfigError(
                "signature.max_edge_px must be at least 2, or the mask would be "
                f"thumbnailed away, got {self.max_edge_px}"
            )
        if not self.pen_inks:
            raise ConfigError("signature.pen_inks must name at least one colour")
        for colour, weight in self.pen_inks:
            _as_hex_colour(colour, "signature.pen_inks")
            if weight < 0:
                raise ConfigError("signature.pen_inks: weights must not be negative")
        if sum(weight for _, weight in self.pen_inks) <= 0:
            raise ConfigError("signature.pen_inks: at least one weight must be positive")
        if self.width_mm.minimum <= 0:
            raise ConfigError(
                f"signature.width_mm.min must be positive, got {self.width_mm.minimum}"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SignatureConfig":
        """Build the signature settings from their YAML mapping.

        Args:
            payload: The ``signature`` section.

        Returns:
            The validated settings.

        Raises:
            ConfigError: If a key is unknown or a value is out of range.
        """
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ConfigError(
                f"signature: unknown key(s) {sorted(unknown)}; "
                f"expected any of {sorted(known)}"
            )
        defaults = cls()
        return cls(
            paper_floor=int(payload.get("paper_floor", defaults.paper_floor)),
            crop_pad=float(payload.get("crop_pad", defaults.crop_pad)),
            max_edge_px=int(payload.get("max_edge_px", defaults.max_edge_px)),
            pen_inks=_parse_weight_pool(
                payload.get("pen_inks"), "signature.pen_inks", _DEFAULT_PEN_INKS
            ),
            width_mm=Range.parse(
                payload.get("width_mm"), "signature.width_mm", defaults.width_mm
            ),
            rotate_deg=Range.parse(
                payload.get("rotate_deg"), "signature.rotate_deg", defaults.rotate_deg
            ),
            overlap_mm=Range.parse(
                payload.get("overlap_mm"), "signature.overlap_mm", defaults.overlap_mm
            ),
        )


@dataclass(frozen=True, slots=True)
class TableConfig:
    """The synthesised table's shape and the numbers invented for its cells.

    Only the *invented* table is configured here. Roughly 118 of the 2000
    letter bodies carry a real Markdown pipe table, and those are rendered
    verbatim - real data beats invented data as a training target, so nothing
    in this section can touch them.

    The column headers decide which of these ranges a cell draws from, so a
    range only has to be plausible for the columns that name it: an amount
    column is millions of rials, a percentage column is a percentage.

    Attributes:
        rows: Body row count. The column count is not drawn - it is the width
            of the header row picked from the vocabulary.
        amount_millions: Value for a ``مبلغ``/``هزینه`` column, in millions of
            rials; the scale is fixed in code alongside the header wording that
            declares the unit.
        count: Value for a ``تعداد`` column.
        percent: Value for a ``درصد`` column.
        day: Day of the Jalali month for a ``تاریخ``/``زمان`` column.
        year: Jalali year for the same column. The default brackets the corpus
            rather than tracking the present, so a rendered letter stays
            consistent with the dates in its own body.
    """

    rows: IntRange = IntRange(4, 8)
    amount_millions: IntRange = IntRange(5, 949)
    count: IntRange = IntRange(1, 59)
    percent: IntRange = IntRange(10, 99)
    day: IntRange = IntRange(1, 28)
    year: IntRange = IntRange(1401, 1404)

    def __post_init__(self) -> None:
        if self.rows.minimum < 1:
            raise ConfigError(
                "elements.table.rows.min must be at least 1, or the table would "
                f"render as a header with no body, got {self.rows.minimum}"
            )
        if self.day.minimum < 1 or self.day.maximum > 31:
            raise ConfigError(
                f"elements.table.day must stay within [1, 31], got {self.day}"
            )
        for name in ("amount_millions", "count", "percent", "year"):
            bounds: IntRange = getattr(self, name)
            if bounds.minimum < 0:
                raise ConfigError(
                    f"elements.table.{name}.min must not be negative, "
                    f"got {bounds.minimum}"
                )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TableConfig":
        """Build the table settings from their YAML mapping.

        Args:
            payload: The ``elements.table`` section.

        Returns:
            The validated settings.

        Raises:
            ConfigError: If a key is unknown or a value is not a whole number.
        """
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ConfigError(
                f"elements.table: unknown key(s) {sorted(unknown)}; "
                f"expected any of {sorted(known)}"
            )
        defaults = cls()
        return cls(
            **{
                name: IntRange.parse(
                    payload.get(name),
                    f"elements.table.{name}",
                    getattr(defaults, name),
                )
                for name in known
            }
        )


#: Kinds the fallback picks between when the letter body names none. Pie is
#: deliberately absent: a pie chart is only ever drawn when the body explicitly
#: asks for one, because a proportion chart of an invented series is rarely
#: what an administrative letter would have used.
_DEFAULT_FALLBACK_KINDS: Final[tuple[str, ...]] = ("bar", "line")


@dataclass(frozen=True, slots=True)
class ChartConfig:
    """The synthesised chart's kind, length, and invented values.

    As with :class:`TableConfig`, only invented data is configured: the 40
    chart-bearing letters that ship a real label/value table are charted from
    it directly.

    ``fallback_kinds`` is an unweighted tuple rather than a weighted pool
    because the draw is ``rng.choice``, which consumes a different number of
    bits than the one ``weighted_choice`` uses. Preserving that keeps the
    shipped defaults byte-identical to the pre-config pipeline.

    Attributes:
        fallback_kinds: Kinds picked between when the body names no chart type.
        series_points: Number of categories plotted. Capped in code by the
            length of the label vocabulary the series is drawn from, so a range
            wider than the shortest pool simply plots the whole pool.
        percent_values: Value range for a ``درصد`` series.
        count_values: Value range for a ``تعداد`` series.
        amount_values: Value range for any other series, which is money.
    """

    fallback_kinds: tuple[str, ...] = _DEFAULT_FALLBACK_KINDS
    series_points: IntRange = IntRange(4, 6)
    percent_values: IntRange = IntRange(15, 99)
    count_values: IntRange = IntRange(8, 479)
    amount_values: IntRange = IntRange(12, 899)

    def __post_init__(self) -> None:
        if not self.fallback_kinds:
            raise ConfigError(
                "elements.chart.fallback_kinds must name at least one kind"
            )
        unknown = set(self.fallback_kinds) - set(CHART_KINDS)
        if unknown:
            raise ConfigError(
                f"elements.chart.fallback_kinds: unknown kind(s) {sorted(unknown)}; "
                f"expected any of {sorted(CHART_KINDS)}"
            )
        if self.series_points.minimum < 2:
            raise ConfigError(
                "elements.chart.series_points.min must be at least 2, or the "
                f"chart would plot a single point, got {self.series_points.minimum}"
            )
        for name in ("percent_values", "count_values", "amount_values"):
            bounds: IntRange = getattr(self, name)
            if bounds.minimum < 0:
                raise ConfigError(
                    f"elements.chart.{name}.min must not be negative - a "
                    f"negative bar draws below the axis, got {bounds.minimum}"
                )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChartConfig":
        """Build the chart settings from their YAML mapping.

        Args:
            payload: The ``elements.chart`` section.

        Returns:
            The validated settings.

        Raises:
            ConfigError: If a key is unknown, a kind is unrecognised, or a
                count is not a whole number.
        """
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ConfigError(
                f"elements.chart: unknown key(s) {sorted(unknown)}; "
                f"expected any of {sorted(known)}"
            )
        defaults = cls()
        kinds_payload = payload.get("fallback_kinds")
        if kinds_payload is None:
            kinds = defaults.fallback_kinds
        elif isinstance(kinds_payload, str) or not isinstance(kinds_payload, Sequence):
            raise ConfigError(
                "elements.chart.fallback_kinds: expected a list of kinds, "
                f"got {kinds_payload!r}"
            )
        else:
            kinds = tuple(str(kind).strip() for kind in kinds_payload)
        return cls(
            fallback_kinds=kinds,
            **{
                name: IntRange.parse(
                    payload.get(name),
                    f"elements.chart.{name}",
                    getattr(defaults, name),
                )
                for name in known - {"fallback_kinds"}
            },
        )


@dataclass(frozen=True, slots=True)
class ElementsConfig:
    """The optional structured elements attached to a page.

    Attributes:
        table: Synthesised-table settings.
        chart: Synthesised-chart settings.
    """

    table: TableConfig = field(default_factory=TableConfig)
    chart: ChartConfig = field(default_factory=ChartConfig)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ElementsConfig":
        """Build the element settings from their YAML mapping.

        Args:
            payload: The ``elements`` section.

        Returns:
            The validated settings.

        Raises:
            ConfigError: If a sub-section is unknown or malformed.
        """
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ConfigError(
                f"elements: unknown key(s) {sorted(unknown)}; "
                f"expected any of {sorted(known)}"
            )
        return cls(
            table=TableConfig.from_dict(payload.get("table", {}) or {}),
            chart=ChartConfig.from_dict(payload.get("chart", {}) or {}),
        )


#: Split names the HuggingFace tooling recognises without a custom loader.
HF_SPLITS: Final[tuple[str, ...]] = ("train", "validation", "test")

#: Parquet compression codecs ``pyarrow`` ships with in every wheel we target.
#: Deliberately not the full codec list: ``lz4``/``brotli`` builds vary by
#: platform, and a shard nobody can read is worse than a slightly larger one.
HF_COMPRESSIONS: Final[tuple[str, ...]] = ("zstd", "snappy", "gzip", "none")

#: Artifact families the export can pack, each becoming one HuggingFace config
#: (a subdirectory under ``data/``). The render stage writes one image per page
#: and one PDF, Markdown reconstruction and annotation JSON per sample; the text
#: stage upstream writes the source plaintext record. ``metadata`` is the
#: blob-free index table over all of them.
#:
#: Order is the order the export walks them in, and the order they appear in the
#: dataset card, so ``images`` leads: it is the family the Hub's viewer opens.
HF_FAMILIES: Final[tuple[str, ...]] = (
    "images",
    "plaintext",
    "markdown",
    "pdf",
    "annotations",
    "metadata",
)

#: What a split assignment may be keyed on. ``source_sample_id`` is the source
#: plaintext file, which is the grouping that keeps two renders of one letter on
#: the same side of the train/test line; ``sample_id`` is per render, for the
#: case where that is genuinely wanted.
HF_GROUP_KEYS: Final[tuple[str, ...]] = ("source_sample_id", "sample_id")

#: How far split fractions may sum from 1.0 and still be accepted. Loose enough
#: for a third written as ``0.333`` in three places, tight enough that a
#: forgotten split is still an error.
HF_SPLIT_TOLERANCE: Final[float] = 1e-6


def normalize_split_fractions(
    fractions: Any, *, field: str = "huggingface.splits"
) -> tuple[tuple[str, float], ...]:
    """Validate split fractions and return them in :data:`HF_SPLITS` order.

    Canonical ordering matters beyond tidiness: the export fingerprints the
    plan so a resume cannot silently re-deal samples, and a fingerprint that
    depended on the order the YAML happened to list the splits in would make
    two identical plans look different.

    Args:
        fractions: A ``{name: fraction}`` mapping, or any iterable of
            ``(name, fraction)`` pairs.
        field: Name used to prefix error messages, so the caller's own
            vocabulary reaches the user.

    Returns:
        The pairs, ordered by :data:`HF_SPLITS`.

    Raises:
        ConfigError: If a name is unknown or repeated, a fraction is not a
            positive number, or the fractions do not sum to 1.0.
    """
    if isinstance(fractions, Mapping):
        entries: Iterable[Any] = tuple(fractions.items())
    elif isinstance(fractions, (str, bytes)) or not isinstance(fractions, Iterable):
        raise ConfigError(
            f"{field} must be a mapping of split name to fraction, "
            f"got {fractions!r}"
        )
    else:
        entries = tuple(fractions)

    pairs: list[tuple[str, float]] = []
    seen: set[str] = set()
    for entry in entries:
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Iterable):
            raise ConfigError(
                f"{field} entries must be (name, fraction) pairs, got {entry!r}"
            )
        items = tuple(entry)
        if len(items) != 2:
            raise ConfigError(
                f"{field} entries must be (name, fraction) pairs, got {entry!r}"
            )
        name, value = str(items[0]).strip(), items[1]
        if name not in HF_SPLITS:
            raise ConfigError(
                f"{field}: unknown split {name!r}; must be one of "
                f"{sorted(HF_SPLITS)}"
            )
        if name in seen:
            raise ConfigError(f"{field}: split {name!r} listed more than once")
        seen.add(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(
                f"{field}: fraction for {name!r} must be a number, got {value!r}"
            )
        fraction = float(value)
        if fraction <= 0.0:
            raise ConfigError(
                f"{field}: fraction for {name!r} must be > 0, got {fraction:g}; "
                "drop the split instead of giving it no share"
            )
        pairs.append((name, fraction))

    if not pairs:
        raise ConfigError(f"{field} must name at least one split")
    total = sum(fraction for _, fraction in pairs)
    if abs(total - 1.0) > HF_SPLIT_TOLERANCE:
        breakdown = ", ".join(f"{name}={fraction:g}" for name, fraction in pairs)
        raise ConfigError(
            f"{field} must sum to 1.0 (+-{HF_SPLIT_TOLERANCE:g}), got "
            f"{total:g} from {breakdown}"
        )
    return tuple(sorted(pairs, key=lambda pair: HF_SPLITS.index(pair[0])))


@dataclass(frozen=True, slots=True)
class HuggingFaceConfig:
    """How the corpus is packed into Parquet shards for the Hub.

    Three groups of knobs, each answering a different question.

    **What is packed.** ``families`` selects which artifact kinds become
    Parquet: the render tree holds page images, a PDF, a Markdown
    reconstruction and an annotation JSON per sample, and the text stage
    upstream holds the source plaintext record. Each selected family becomes
    one subdirectory - and therefore one loadable HuggingFace config - so
    dropping a family costs a consumer that artifact and nothing else.

    **How it is divided.** ``splits`` gives each split its share, and
    ``group_by`` fixes the unit those shares are drawn over. Keying on
    ``source_sample_id`` is what keeps every page of one letter on the same
    side of the train/test line; keying on ``sample_id`` splits per render and
    leaks whenever one letter was rendered twice. ``split_salt`` re-deals the
    whole corpus, so it belongs in the file and must not be changed once a
    dataset has been published.

    **How large the files are.** Three limits govern sharding and they are not
    interchangeable. ``shard_samples`` is the nominal batch; ``min_shard_bytes``
    holds a shard open past that batch until it is worth being a file at all,
    which is what stops the blob-free families becoming forty 200 KB shards;
    ``max_shard_bytes`` is the hard ceiling. When a full batch would exceed the
    ceiling the writer steps down to ``min_shard_samples`` and says so in the
    log, because a silently truncated export reads as a complete one.

    Attributes:
        output_dir: Where the export tree is written. Kept separate from the
            render ``output_dir`` so an export never lands among the artifacts
            it was built from.
        splits: ``(name, fraction)`` pairs, ordered by :data:`HF_SPLITS`. A
            tuple rather than a mapping because this dataclass is frozen and a
            frozen dataclass holding a ``dict`` is unhashable.
        group_by: Which id the split assignment hashes, from
            :data:`HF_GROUP_KEYS`.
        split_salt: Mixed into that hash. Changing it re-deals every sample.
        families: Artifact families to pack, ordered by :data:`HF_FAMILIES`.
        shard_samples: Nominal samples per shard.
        min_shard_samples: Batch size to fall back to when a full batch would
            breach ``max_shard_bytes``.
        min_shard_bytes: Payload a shard must accumulate before the sample
            count is allowed to close it. ``0`` restores pure sample-count
            sharding.
        max_shard_bytes: Hard ceiling on one shard's accumulated payload,
            measured before compression, so the written file is smaller than
            this rather than larger.
        compression: Parquet codec.
        dataset_name: Name written into the dataset card and
            ``dataset_info.json``.
        license_id: SPDX-style licence id for the card's front matter. Defaults
            to a non-commercial one: the pages composite third-party brand marks
            and real signatures from a research-only corpus, so a permissive
            declaration on the container would contradict the attribution the
            card carries.
        language: Content language for the card's front matter.
        include_elements: Whether the per-page element list travels in the
            ``images`` rows. Turning it off keeps image + text, which is what a
            recognition-only fine-tune reads; the boxes still ship in the
            ``annotations`` family, so nothing is lost from the export.
        write_metadata_csv: Whether to also write the flat ``metadata.csv``
            index. Cheap, and the only readable view of the corpus for someone
            without ``pyarrow``.
        write_splits_csv: Whether to write ``splits.csv``, the bare
            ``sample_id, source_sample_id, split`` assignment. Cheap, and it
            makes the group disjointness checkable without reading Parquet.
        hub_repo_id: ``owner/name`` of the Hub dataset repository the export is
            pushed to. Empty means "not configured", which is the default: an
            export is a local directory until someone names a destination.
            Belongs in ``.env`` rather than the YAML for the same reason the
            token does - it is per-person, not per-corpus.
        hub_private: Whether :func:`create_repo` makes a newly created
            repository private. Ignored for a repository that already exists;
            the Hub does not let an upload flip that flag, deliberately.
        hub_commit_message: First line of the commit each upload creates. The
            family being pushed is appended, so a repository's history reads as
            one line per family rather than six identical entries.
    """

    output_dir: Path = Path("data/hf")
    splits: tuple[tuple[str, float], ...] = (("train", 0.9), ("test", 0.1))
    group_by: str = "source_sample_id"
    split_salt: str = ""
    families: tuple[str, ...] = HF_FAMILIES
    shard_samples: int = 500
    min_shard_samples: int = 250
    min_shard_bytes: int = 64 * 1024**2
    max_shard_bytes: int = 2 * 1024**3
    compression: str = "zstd"
    dataset_name: str = "persian-official-letters-synth"
    license_id: str = "cc-by-nc-4.0"
    language: str = "fa"
    include_elements: bool = True
    write_metadata_csv: bool = True
    write_splits_csv: bool = True
    hub_repo_id: str = ""
    hub_private: bool = False
    hub_commit_message: str = "Upload synthetic Persian letter corpus"

    #: YAML keys whose dataclass field is spelled differently. ``license`` is a
    #: soft keyword in neither Python nor YAML, but ``license_id`` reads better
    #: as an attribute while ``license:`` is what a dataset card calls it.
    _ALIASES: ClassVar[Mapping[str, str]] = {"license": "license_id"}

    #: Keys that used to exist, mapped to what replaced them. A config file
    #: written against the single-split exporter would otherwise fail with a
    #: bare "unknown key", which says nothing about why.
    _REMOVED: ClassVar[Mapping[str, str]] = {
        "split": (
            "splits: the export now writes every split at once, so give shares "
            "instead of one name, e.g. splits: {train: 0.9, test: 0.1}"
        )
    }

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "splits", normalize_split_fractions(self.splits)
        )
        object.__setattr__(self, "families", self._normalized_families())
        if self.group_by not in HF_GROUP_KEYS:
            raise ConfigError(
                f"huggingface.group_by must be one of {sorted(HF_GROUP_KEYS)}, "
                f"got {self.group_by!r}"
            )
        if self.shard_samples < 1:
            raise ConfigError(
                "huggingface.shard_samples must be at least 1, got "
                f"{self.shard_samples}"
            )
        if self.min_shard_samples < 1:
            raise ConfigError(
                "huggingface.min_shard_samples must be at least 1, got "
                f"{self.min_shard_samples}"
            )
        if self.min_shard_samples > self.shard_samples:
            raise ConfigError(
                "huggingface.min_shard_samples must not exceed shard_samples "
                f"({self.min_shard_samples} > {self.shard_samples}); the "
                "step-down is a smaller batch, not a larger one"
            )
        if self.min_shard_bytes < 0:
            raise ConfigError(
                "huggingface.min_shard_bytes must not be negative, got "
                f"{self.min_shard_bytes}"
            )
        if self.max_shard_bytes < 1024:
            raise ConfigError(
                "huggingface.max_shard_bytes must be at least 1024; a ceiling "
                "below one page image would close every shard after one row, "
                f"got {self.max_shard_bytes}"
            )
        if self.min_shard_bytes > self.max_shard_bytes:
            raise ConfigError(
                "huggingface.min_shard_bytes must not exceed max_shard_bytes "
                f"({self.min_shard_bytes} > {self.max_shard_bytes}); a shard "
                "cannot be held open past a ceiling it has already hit"
            )
        if self.compression not in HF_COMPRESSIONS:
            raise ConfigError(
                f"huggingface.compression must be one of {sorted(HF_COMPRESSIONS)}, "
                f"got {self.compression!r}"
            )
        if not self.dataset_name.strip():
            raise ConfigError("huggingface.dataset_name must not be empty")
        if not self.language.strip():
            raise ConfigError("huggingface.language must not be empty")
        object.__setattr__(self, "hub_repo_id", self.hub_repo_id.strip())
        if self.hub_repo_id:
            # Checked here rather than at upload time: a typo in `.env` should
            # surface when the config loads, not after a 40 GB export finishes.
            owner, separator, name = self.hub_repo_id.partition("/")
            if not separator or not owner.strip() or not name.strip():
                raise ConfigError(
                    "huggingface.hub_repo_id must be 'owner/name', got "
                    f"{self.hub_repo_id!r}"
                )
            if "/" in name:
                raise ConfigError(
                    "huggingface.hub_repo_id must hold exactly one '/', got "
                    f"{self.hub_repo_id!r}"
                )

    def _normalized_families(self) -> tuple[str, ...]:
        """Validate ``families`` and return it in :data:`HF_FAMILIES` order."""
        raw = self.families
        # A bare `families: images` is a plausible mistake, and iterating a
        # string would report six unknown families spelled `i`, `m`, `a`, ...
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
            raise ConfigError(
                "huggingface.families must be a list of family names, got "
                f"{raw!r}"
            )
        names: list[str] = []
        for entry in raw:
            name = str(entry).strip()
            if name not in HF_FAMILIES:
                raise ConfigError(
                    f"huggingface.families: unknown family {name!r}; must be "
                    f"one of {sorted(HF_FAMILIES)}"
                )
            if name in names:
                raise ConfigError(
                    f"huggingface.families: {name!r} listed more than once"
                )
            names.append(name)
        if not names:
            raise ConfigError("huggingface.families must name at least one family")
        return tuple(sorted(names, key=HF_FAMILIES.index))

    @property
    def data_dir(self) -> Path:
        """Directory the per-family shard directories live under."""
        return self.output_dir / "data"

    def family_dir(self, family: str) -> Path:
        """Return the directory one family's shards are written into.

        A directory name under ``data/`` is what the Hub reads as a config
        name, so this is also the name ``load_dataset(repo, family)`` takes.
        """
        return self.data_dir / family

    def split_names(self) -> tuple[str, ...]:
        """Return the configured split names, in :data:`HF_SPLITS` order."""
        return tuple(name for name, _ in self.splits)

    def fraction_of(self, split: str) -> float:
        """Return the requested fraction for one split.

        Raises:
            KeyError: If the split is not part of this configuration.
        """
        for name, fraction in self.splits:
            if name == split:
                return fraction
        raise KeyError(split)

    def shard_name(self, split: str, index: int, total: int) -> str:
        """Return a shard's final filename.

        Args:
            split: Split the shard holds.
            index: Zero-based shard number within that split.
            total: How many shards the split produced.

        Returns:
            The HuggingFace-conventional name, e.g.
            ``train-00000-of-00005.parquet``. The total is not known while
            streaming, which is why the writer renames at close rather than
            naming shards as it goes.
        """
        return f"{split}-{index:05d}-of-{total:05d}.parquet"

    def staging_name(self, split: str, index: int) -> str:
        """Return the name a shard carries while the export is still running."""
        return f"{split}-{index:05d}.parquet"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HuggingFaceConfig":
        """Build the export settings from their YAML mapping.

        Args:
            payload: The ``huggingface`` section.

        Returns:
            The validated settings.

        Raises:
            ConfigError: If a key is unknown or a value is out of range.
        """
        # ``fields`` rather than ``__dataclass_fields__``: the latter also holds
        # an entry per ``ClassVar`` pseudo-field, and a message offering
        # ``_ALIASES`` as a config key is noise.
        known = {entry.name for entry in fields(cls)} | set(cls._ALIASES)
        retired = sorted(set(payload) & set(cls._REMOVED))
        if retired:
            replacements = "; ".join(cls._REMOVED[key] for key in retired)
            raise ConfigError(
                f"huggingface: {retired} no longer exists, use {replacements}"
            )
        unknown = set(payload) - known
        if unknown:
            raise ConfigError(
                f"huggingface: unknown key(s) {sorted(unknown)}; "
                f"expected any of {sorted(known)}"
            )
        resolved = {
            cls._ALIASES.get(key, key): value for key, value in payload.items()
        }
        defaults = cls()
        return cls(
            output_dir=resolve_repo_path(
                resolved.get("output_dir", defaults.output_dir)
            ),
            splits=normalize_split_fractions(
                resolved.get("splits", defaults.splits)
            ),
            group_by=str(resolved.get("group_by", defaults.group_by)).strip(),
            split_salt=str(resolved.get("split_salt", defaults.split_salt)),
            # Passed through unconverted: `_normalized_families` has to see a
            # bare string as a mistake, and `tuple("images")` would hide it.
            families=resolved.get("families", defaults.families),
            shard_samples=int(resolved.get("shard_samples", defaults.shard_samples)),
            min_shard_samples=int(
                resolved.get("min_shard_samples", defaults.min_shard_samples)
            ),
            min_shard_bytes=int(
                resolved.get("min_shard_bytes", defaults.min_shard_bytes)
            ),
            max_shard_bytes=int(
                resolved.get("max_shard_bytes", defaults.max_shard_bytes)
            ),
            compression=str(
                resolved.get("compression", defaults.compression)
            ).strip().lower(),
            dataset_name=str(resolved.get("dataset_name", defaults.dataset_name)),
            license_id=str(resolved.get("license_id", defaults.license_id)),
            language=str(resolved.get("language", defaults.language)),
            include_elements=bool(
                resolved.get("include_elements", defaults.include_elements)
            ),
            write_metadata_csv=bool(
                resolved.get("write_metadata_csv", defaults.write_metadata_csv)
            ),
            write_splits_csv=bool(
                resolved.get("write_splits_csv", defaults.write_splits_csv)
            ),
            hub_repo_id=str(resolved.get("hub_repo_id", defaults.hub_repo_id)),
            hub_private=bool(resolved.get("hub_private", defaults.hub_private)),
            hub_commit_message=str(
                resolved.get("hub_commit_message", defaults.hub_commit_message)
            ),
        )



@dataclass(frozen=True, slots=True)
class SynthConfig:
    """The whole synthetic-render configuration."""

    seed: int
    paths: PathsConfig
    render: RenderConfig
    probabilities: ProbabilityConfig = field(default_factory=ProbabilityConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    stamp: StampConfig = field(default_factory=StampConfig)
    signature: SignatureConfig = field(default_factory=SignatureConfig)
    elements: ElementsConfig = field(default_factory=ElementsConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    vocabulary: Vocabulary = field(default_factory=Vocabulary)
    source_path: Path | None = None

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, source_path: Path | None = None
    ) -> "SynthConfig":
        """Build the config from an already-parsed YAML mapping.

        Args:
            payload: Parsed YAML document.
            source_path: File the mapping came from, kept for error messages.

        Returns:
            A validated configuration.

        Raises:
            ConfigError: If a required section or key is missing or invalid, or
                if the ``vocabulary`` file it points at cannot be loaded.
        """
        known_sections = {
            "seed",
            "paths",
            "render",
            "probabilities",
            "theme",
            "layout",
            "stamp",
            "signature",
            "elements",
            "scan",
            "huggingface",
            "vocabulary",
        }
        unknown = set(payload) - known_sections
        if unknown:
            raise ConfigError(
                f"unknown top-level section(s) {sorted(unknown)}; "
                f"expected any of {sorted(known_sections)}"
            )
        return cls(
            seed=int(payload.get("seed", DEFAULT_SEED)),
            paths=PathsConfig.from_dict(_require(payload, "paths", "config")),
            render=RenderConfig.from_dict(_require(payload, "render", "config")),
            probabilities=ProbabilityConfig.from_dict(
                payload.get("probabilities", {}) or {}
            ),
            theme=ThemeConfig.from_dict(payload.get("theme", {}) or {}),
            layout=LayoutConfig.from_dict(payload.get("layout", {}) or {}),
            stamp=StampConfig.from_dict(payload.get("stamp", {}) or {}),
            signature=SignatureConfig.from_dict(payload.get("signature", {}) or {}),
            elements=ElementsConfig.from_dict(payload.get("elements", {}) or {}),
            scan=ScanConfig.from_dict(payload.get("scan", {}) or {}),
            huggingface=HuggingFaceConfig.from_dict(
                payload.get("huggingface", {}) or {}
            ),
            vocabulary=_load_vocabulary_section(payload.get("vocabulary")),
            source_path=source_path,
        )


def _load_vocabulary_section(payload: Any) -> Vocabulary:
    """Resolve the ``vocabulary:`` key into loaded Persian content pools.

    The key is a *path*, not a section: the pools are long enough that keeping
    them in ``synth_config.yaml`` would bury the knobs, and they are edited for
    different reasons. An absent key still loads the default file, so a config
    that never mentions the vocabulary behaves as if it pointed at
    ``configs/vocabulary_fa.yaml``.

    The environment override is applied here rather than in :func:`_apply_env`,
    because for this one setting "parse the YAML" and "read the file" are the
    same step - honouring it later would mean loading one vocabulary only to
    throw it away.

    Args:
        payload: The ``vocabulary`` key's value: a path, or ``None`` when
            absent. An inline mapping is also accepted, which is what keeps the
            test suite from needing a second file on disk.

    Returns:
        The loaded, validated pools.

    Raises:
        ConfigError: If the value is neither a path nor a mapping, or the file
            it names is missing or malformed.
    """
    override = env_str(VOCABULARY_PATH_ENV_VAR, "")
    if override:
        return load_vocabulary(override)
    if payload is None:
        return load_vocabulary(DEFAULT_VOCABULARY_PATH)
    if isinstance(payload, Mapping):
        return Vocabulary.from_dict(payload)
    if isinstance(payload, (str, Path)):
        return load_vocabulary(payload)
    raise ConfigError(
        f"config.vocabulary: expected a path to a vocabulary file, got {payload!r}"
    )


#: Environment variables that re-point one :class:`PathsConfig` entry, mapped
#: to the field each one replaces. A path is the archetypal setting that
#: differs per machine and must not be committed, so the environment - not the
#: YAML - is where a training host says where its volumes are mounted.
PATH_ENV_VARS: Final[Mapping[str, str]] = {
    "FA_OCR_MANIFEST": "manifest",
    "FA_OCR_PLAINTEXT_DIR": "plaintext_dir",
    "FA_OCR_LOGOS_DIR": "logos_dir",
    "FA_OCR_SIGNATURES_DIR": "signatures_dir",
    "FA_OCR_TEXTURE_DIR": "texture_dir",
    "FA_OCR_OUTPUT_DIR": "output_dir",
    "FA_OCR_FONTS_DIR": "fonts_dir",
}

#: Variable naming the config file itself, honoured when no path is passed.
CONFIG_PATH_ENV_VAR: Final[str] = "FA_OCR_CONFIG"

#: Variable that re-anchors every asset path living inside the repository's
#: own ``data/`` tree; see :func:`_reanchor_data_dir`.
DATA_DIR_ENV_VAR: Final[str] = "FA_OCR_DATA_DIR"

#: Variable overriding the master seed.
SEED_ENV_VAR: Final[str] = "FA_OCR_SEED"

#: Variable re-pointing the HuggingFace export tree. Separate from
#: :data:`PATH_ENV_VARS` because it names a *destination* rather than an input
#: asset, and because it belongs to a different config section.
HF_OUTPUT_DIR_ENV_VAR: Final[str] = "FA_OCR_HF_OUTPUT_DIR"

#: Variable overriding the split shares, written ``train=0.9,test=0.1``. Present
#: for the case the YAML cannot cover: a throwaway export on a training host
#: that wants the whole corpus in one split, without editing a file whose
#: fractions a published dataset depends on.
HF_SPLITS_ENV_VAR: Final[str] = "FA_OCR_HF_SPLITS"

#: Variable naming the Hub dataset repository, ``owner/name``. The destination
#: is per-person, so it belongs beside the token rather than in a committed
#: YAML file that everyone with a checkout shares.
HF_REPO_ID_ENV_VAR: Final[str] = "FA_OCR_HF_REPO_ID"

#: Variable deciding whether a *newly created* Hub repository is private.
HF_PRIVATE_ENV_VAR: Final[str] = "FA_OCR_HF_PRIVATE"

#: Variable holding the Hub write token. Spelled the way ``huggingface_hub``
#: itself spells it, so a machine already logged in through the CLI needs no
#: extra configuration and the token never has to appear in ours.
HF_TOKEN_ENV_VARS: Final[tuple[str, ...]] = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
)


def parse_split_spec(spec: str, *, field: str = HF_SPLITS_ENV_VAR) -> tuple[
    tuple[str, float], ...
]:
    """Parse a ``train=0.9,test=0.1`` split specification.

    Used by both the environment override and the CLI flag, so a fraction typed
    in either place fails the same way.

    Args:
        spec: Comma-separated ``name=fraction`` assignments.
        field: Name used to prefix error messages.

    Returns:
        The pairs, validated and ordered by :data:`HF_SPLITS`.

    Raises:
        ConfigError: If the syntax is wrong or the fractions are invalid.
    """
    pairs: list[tuple[str, float]] = []
    for chunk in spec.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        name, separator, raw = entry.partition("=")
        if not separator:
            raise ConfigError(
                f"{field}: expected name=fraction pairs, got {entry!r} "
                "(e.g. train=0.9,test=0.1)"
            )
        try:
            fraction = float(raw.strip())
        except ValueError as exc:
            raise ConfigError(
                f"{field}: fraction for {name.strip()!r} is not a number: "
                f"{raw.strip()!r}"
            ) from exc
        pairs.append((name.strip(), fraction))
    if not pairs:
        raise ConfigError(f"{field}: no splits given in {spec!r}")
    return normalize_split_fractions(pairs, field=field)


def _reanchor_data_dir(paths: PathsConfig, base: Path) -> PathsConfig:
    """Move every path under the repository's ``data/`` tree to ``base``.

    The YAML ships repository-relative asset paths (``data/text/generator.csv``),
    which is right for a checkout that carries its own data and wrong for a
    training host where the corpus lives on a mounted volume. Setting
    ``FA_OCR_DATA_DIR=/mnt/corpus`` turns that into
    ``/mnt/corpus/text/generator.csv``.

    The rule is deliberately narrow: only a path that resolves *inside*
    ``<repo>/data`` is moved. A YAML entry that already points somewhere else,
    and ``fonts_dir`` - which is a host font directory, not corpus data - are
    left exactly as configured.

    Args:
        paths: The paths parsed from YAML.
        base: Replacement root for the ``data/`` tree.

    Returns:
        The re-anchored paths, or ``paths`` when nothing lived under ``data/``.
    """
    old_base = repo_root() / "data"
    moved: dict[str, Path] = {}
    for name in ("manifest", "plaintext_dir", "logos_dir", "signatures_dir",
                 "texture_dir", "output_dir"):
        current: Path = getattr(paths, name)
        try:
            relative = current.relative_to(old_base)
        except ValueError:
            continue
        moved[name] = base / relative
    return replace(paths, **moved) if moved else paths


def _apply_env(config: SynthConfig) -> SynthConfig:
    """Overlay environment overrides onto an already-validated config.

    Applied *after* the YAML is parsed, so a bad value from the environment
    hits the same dataclass validators a bad value in the file would, and the
    error names the same field.

    Order within the paths section matters: :data:`DATA_DIR_ENV_VAR` moves the
    whole tree first, then a specific ``FA_OCR_*_DIR`` replaces one entry, so
    the narrower variable wins.

    Args:
        config: The configuration as loaded from YAML.

    Returns:
        The configuration with any environment overrides applied; ``config``
        itself when none are set.

    Raises:
        ConfigError: If an override is present but unusable.
        EnvError: If an override is present but not of its declared type.
    """
    paths = config.paths
    data_dir = env_path(DATA_DIR_ENV_VAR)
    if data_dir is not None:
        paths = _reanchor_data_dir(paths, data_dir)
    overrides: dict[str, Path] = {}
    for variable, name in PATH_ENV_VARS.items():
        value = env_path(variable)
        if value is not None:
            # ``fonts_dir`` is used as given; the rest are corpus paths and
            # follow the YAML's repository-relative convention.
            overrides[name] = (
                value if name == "fonts_dir" else resolve_repo_path(value)
            )
    if overrides:
        paths = replace(paths, **overrides)

    changes: dict[str, Any] = {}
    if paths is not config.paths:
        changes["paths"] = paths
    seed = env_int(SEED_ENV_VAR)
    if seed is not None:
        changes["seed"] = seed
    hf_output = env_path(HF_OUTPUT_DIR_ENV_VAR)
    hf_splits = env_str(HF_SPLITS_ENV_VAR)
    hf_repo = env_str(HF_REPO_ID_ENV_VAR)
    hf_private = env_bool(HF_PRIVATE_ENV_VAR)
    if (
        hf_output is not None
        or hf_splits is not None
        or hf_repo is not None
        or hf_private is not None
    ):
        hf_changes: dict[str, Any] = {}
        if hf_output is not None:
            hf_changes["output_dir"] = resolve_repo_path(hf_output)
        if hf_splits is not None:
            hf_changes["splits"] = parse_split_spec(hf_splits)
        if hf_repo is not None:
            hf_changes["hub_repo_id"] = hf_repo
        if hf_private is not None:
            hf_changes["hub_private"] = hf_private
        changes["huggingface"] = replace(config.huggingface, **hf_changes)
    return replace(config, **changes) if changes else config


def load_config(
    path: str | Path | None = None, *, apply_env: bool = True
) -> SynthConfig:
    """Load and validate the synthetic pipeline configuration.

    Precedence is ``CLI flag > environment > YAML > dataclass default``. This
    function owns the middle two: it reads the YAML, then overlays whatever the
    environment sets. A caller that has parsed CLI flags applies those last.

    Args:
        path: YAML file to read, absolute or relative to the repository root.
            Defaults to :data:`CONFIG_PATH_ENV_VAR` if that is set, and to
            :data:`DEFAULT_CONFIG_PATH` otherwise - so a CLI ``--config`` must
            pass ``None`` rather than its own default to let the environment
            through.
        apply_env: Whether to overlay environment overrides. Passing ``False``
            loads the file exactly as written, which is what a test comparing
            against the committed YAML wants.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the file is missing, empty, or fails validation.
        EnvError: If ``.env`` is malformed or an override is not of its
            declared type.
    """
    if apply_env:
        ensure_dotenv_loaded()
    if path is None:
        path = (
            env_str(CONFIG_PATH_ENV_VAR, DEFAULT_CONFIG_PATH)
            if apply_env
            else DEFAULT_CONFIG_PATH
        )
    target = resolve_repo_path(path)
    if not target.is_file():
        raise ConfigError(f"config file not found: {target}")
    try:
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{target}: invalid YAML - {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{target}: expected a YAML mapping at the top level")
    config = SynthConfig.from_dict(payload, source_path=target)
    return _apply_env(config) if apply_env else config


#: Former name of :func:`load_config`, kept so external call sites and older
#: notebooks keep working after the loader moved into this package.
load_synth_config = load_config
