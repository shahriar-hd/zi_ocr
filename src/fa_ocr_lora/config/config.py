"""Typed loader for ``configs/synth_config.yaml``.

CLAUDE.md 8 forbids hardcoded constants: every knob the synthetic pipeline
turns - page sizes, DPI, probabilities, asset locations - lives in YAML and
arrives here as a frozen, validated dataclass. Loading is strict, so a typo in
the YAML fails at startup rather than silently rendering 20k samples with a
default nobody intended.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import yaml

from .io_utils import DEFAULT_SEED, resolve_repo_path

__all__ = [
    "ConfigError",
    "SUBDIR_FIELDS",
    "PageSizeSpec",
    "PathsConfig",
    "RenderConfig",
    "ProbabilityConfig",
    "FontSpec",
    "ThemeConfig",
    "SynthConfig",
    "load_synth_config",
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


class ConfigError(ValueError):
    """Raised when the configuration file is missing keys or malformed."""


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
            name: Bare PNG filename, e.g. ``synth_000000_p01.png``.

        Returns:
            The path relative to ``output_dir``, so a training host resolves
            it from the dataset root alone rather than from this checkout.
        """
        return f"{self.images_subdir}/{name}" if self.images_subdir else name

    @property
    def images_dir(self) -> Path:
        """Directory holding one PNG per rendered page."""
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


@dataclass(frozen=True, slots=True)
class RenderConfig:
    """Rasterization settings and the page-size pool."""

    dpi: int
    page_sizes: tuple[PageSizeSpec, ...]
    margin_mm_range: tuple[float, float] = (18.0, 26.0)
    max_pages: int = 6
    jpeg_quality: int = 95
    timeout_ms: int = 30_000

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
        """Build the render settings from their YAML mapping."""
        sizes = _require(payload, "page_sizes", "render")
        if not isinstance(sizes, Sequence) or isinstance(sizes, (str, bytes)):
            raise ConfigError("render.page_sizes must be a list")
        margin = payload.get("margin_mm", {})
        return cls(
            dpi=int(payload.get("dpi", 300)),
            page_sizes=tuple(PageSizeSpec.from_dict(item) for item in sizes),
            margin_mm_range=(
                float(margin.get("min", 18.0)),
                float(margin.get("max", 26.0)),
            ),
            max_pages=int(payload.get("max_pages", 6)),
            jpeg_quality=int(payload.get("jpeg_quality", 95)),
            timeout_ms=int(payload.get("timeout_ms", 30_000)),
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


@dataclass(frozen=True, slots=True)
class ThemeConfig:
    """Typography and palette constraints."""

    font_families: tuple[FontSpec, ...] = (FontSpec("B Nazanin"),)
    display_font_families: tuple[FontSpec, ...] = (FontSpec("B Titr"),)
    body_font_pt_range: tuple[float, float] = (10.5, 13.0)
    line_height_range: tuple[float, float] = (1.6, 2.0)
    min_contrast_ratio: float = 4.5
    stamp_inks: tuple[str, ...] = ("#1B3F94", "#A81E22", "#4B2A7B")

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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ThemeConfig":
        """Build the theme constraints from their YAML mapping."""
        font_pt = payload.get("body_font_pt", {})
        line_height = payload.get("line_height", {})
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
            stamp_inks=tuple(payload.get("stamp_inks", ThemeConfig.stamp_inks)),
        )


@dataclass(frozen=True, slots=True)
class SynthConfig:
    """The whole synthetic-render configuration."""

    seed: int
    paths: PathsConfig
    render: RenderConfig
    probabilities: ProbabilityConfig = field(default_factory=ProbabilityConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
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
            ConfigError: If a required section or key is missing or invalid.
        """
        known_sections = {"seed", "paths", "render", "probabilities", "theme"}
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
            source_path=source_path,
        )


def load_synth_config(path: str | Path = DEFAULT_CONFIG_PATH) -> SynthConfig:
    """Load and validate the synthetic pipeline configuration.

    Args:
        path: YAML file to read, absolute or relative to the repository root.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the file is missing, empty, or fails validation.
    """
    target = resolve_repo_path(path)
    if not target.is_file():
        raise ConfigError(f"config file not found: {target}")
    try:
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{target}: invalid YAML - {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{target}: expected a YAML mapping at the top level")
    return SynthConfig.from_dict(payload, source_path=target)
