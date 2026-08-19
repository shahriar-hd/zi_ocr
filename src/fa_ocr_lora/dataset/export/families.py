"""What each Parquet family carries, and how one sample becomes its rows.

The render tree holds four artifacts per sample - one image per page, a PDF, a
Markdown reconstruction and the annotation JSON - and the text stage upstream
holds the source ``plaintext_XXXXXX.json`` it was all built from. The export
ships every one of them. Each becomes its own *family*: one subdirectory under
``data/``, which is also one loadable HuggingFace config, so
``load_dataset(repo, "pdf")`` is a thing a consumer can ask for and a consumer
who only wants pages never downloads the PDFs.

Three decisions shape this module:

* **The sample header is repeated on every row of every family.** Thirty-odd
  small columns - ids, page geometry, the letter's own fields, the filter flags,
  what the text stage asked the drafting model for - cost a few hundred bytes
  next to a multi-megabyte page image, and they save every consumer a join.
  Someone slicing an error analysis by ``letter_tone`` can do it from the
  ``images`` family alone.
* **A column is declared once.** Parquet needs an Arrow schema and the Hub needs
  a ``datasets`` feature descriptor. Hand-writing both per family is two
  declarations that can drift into disagreement, so :class:`Column` carries the
  kind and both are derived from it.
* **A missing artifact is not fatal to the export.** A missing *page image* is
  an :class:`~.errors.ExportError` - a row with no image is not a training
  example. A missing PDF, Markdown or plaintext file skips that one family's row
  for that sample, is logged, and is counted; a tree shipped without the text
  stage still exports five families.

The ``has_table`` / ``has_chart`` / ``has_signature`` / ``has_stamp`` flags are
derived from the annotation's own element types rather than read back from the
render manifest. The annotation is what shipped, so a flag derived from it
cannot disagree with the boxes sitting in the same row.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Collection, Final, Mapping, Protocol, Sequence

from ...config import HF_FAMILIES, SynthConfig
from ..common.projections import paddle_farmer_records, paddle_records
from ..common.schema import DocumentAnnotation, ElementType
from .errors import ExportError

__all__ = [
    "FAMILIES",
    "HEADER_COLUMNS",
    "Column",
    "FamilySpec",
    "SampleBundle",
    "family_specs",
    "load_bundle",
    "row_payload_bytes",
    "sample_header",
]

_LOGGER: Final[logging.Logger] = logging.getLogger("fa_ocr_lora.export.families")

_MIB: Final[int] = 1024**2


# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Column:
    """One column of one family, in the single place its type is stated.

    Attributes:
        name: Column name, as it appears in Parquet and in the dataset card.
        kind: A key of :data:`_KIND_FEATURES`. Both the Arrow type and the
            ``datasets`` feature descriptor are derived from it, so the two can
            never disagree about what the column holds.
        note: One line for the dataset card's column table. It lives here rather
            than in the template so a new column cannot ship undocumented.
        only_with_elements: Whether the column is present only when the export
            was asked for per-page element lists.
    """

    name: str
    kind: str
    note: str = ""
    only_with_elements: bool = False


#: The element struct as ``datasets`` spells it. ``content`` is deliberately
#: absent: the structured payload of a table or a chart has a different shape per
#: element type, and it ships whole in the ``annotations`` family instead.
_ELEMENT_FEATURE: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "id": {"dtype": "string", "_type": "Value"},
        "type": {"dtype": "string", "_type": "Value"},
        "text": {"dtype": "string", "_type": "Value"},
        "bbox": {"feature": {"dtype": "int32", "_type": "Value"}, "_type": "Sequence"},
        "reading_order": {"dtype": "int32", "_type": "Value"},
    }
)

#: Every column kind, mapped to its ``datasets`` feature descriptor. The Arrow
#: side of the same table is :func:`_arrow_type`; adding a kind means adding to
#: both.
_KIND_FEATURES: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "string": {"dtype": "string", "_type": "Value"},
        "int32": {"dtype": "int32", "_type": "Value"},
        "int64": {"dtype": "int64", "_type": "Value"},
        "bool": {"dtype": "bool", "_type": "Value"},
        "binary": {"dtype": "binary", "_type": "Value"},
        "image": {"_type": "Image"},
        "string_list": {
            "feature": {"dtype": "string", "_type": "Value"},
            "_type": "Sequence",
        },
        "int32_list": {
            "feature": {"dtype": "int32", "_type": "Value"},
            "_type": "Sequence",
        },
        "elements": [dict(_ELEMENT_FEATURE)],
    }
)

#: The element struct again, in the *front matter* spelling.
_ELEMENT_YAML: Final[tuple[Mapping[str, Any], ...]] = (
    MappingProxyType({"name": "id", "dtype": "string"}),
    MappingProxyType({"name": "type", "dtype": "string"}),
    MappingProxyType({"name": "text", "dtype": "string"}),
    MappingProxyType({"name": "bbox", "sequence": "int32"}),
    MappingProxyType({"name": "reading_order", "dtype": "int32"}),
)

#: Every column kind again, in the spelling the dataset card's ``dataset_info``
#: front matter uses. This is *not* :data:`_KIND_FEATURES`: ``datasets`` writes
#: the internal JSON form above into Arrow metadata but a simplified form into
#: YAML - ``dtype: image`` rather than ``{"_type": "Image"}``, ``sequence:`` and
#: ``list:`` rather than nested ``_type`` keys - and reads YAML back with
#: ``Features._from_yaml_list``, which rejects the internal form. Both tables are
#: keyed by the same :class:`Column` kinds and :func:`_check_kind_tables` asserts
#: they stay in step, so a new kind cannot be declared in only one of them.
_KIND_YAML: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "string": {"dtype": "string"},
        "int32": {"dtype": "int32"},
        "int64": {"dtype": "int64"},
        "bool": {"dtype": "bool"},
        "binary": {"dtype": "binary"},
        "image": {"dtype": "image"},
        "string_list": {"sequence": "string"},
        "int32_list": {"sequence": "int32"},
        "elements": {"list": list(_ELEMENT_YAML)},
    }
)


def _check_kind_tables() -> None:
    """Fail at import if the two kind tables have drifted apart.

    Raises:
        ExportError: If a kind is declared in one table and not the other.
    """
    missing = set(_KIND_FEATURES) ^ set(_KIND_YAML)
    if missing:
        raise ExportError(
            "column kinds declared in only one feature table: "
            f"{sorted(missing)}. Add the kind to both _KIND_FEATURES and "
            "_KIND_YAML, and to _arrow_type."
        )


_check_kind_tables()


def _plain(value: Any) -> Any:
    """Return ``value`` with every mapping copied into a plain ``dict``.

    ``yaml.safe_dump`` refuses a :class:`~types.MappingProxyType`, and the kind
    tables are read-only views on purpose, so the YAML writer gets a mutable
    copy rather than the module's own tables.

    Args:
        value: Any nesting of mappings, sequences and scalars.

    Returns:
        The same structure built from ``dict`` and ``list``.
    """
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value



def _arrow_type(pyarrow: Any, kind: str) -> Any:
    """Return the Arrow type for a column kind.

    Args:
        pyarrow: The imported ``pyarrow`` module.
        kind: A key of :data:`_KIND_FEATURES`.

    Returns:
        The ``pyarrow.DataType`` the column is written as.

    Raises:
        ExportError: If the kind is not one this module knows, which can only
            happen after an edit to a family's column list.
    """
    if kind == "string":
        return pyarrow.string()
    if kind == "int32":
        return pyarrow.int32()
    if kind == "int64":
        return pyarrow.int64()
    if kind == "bool":
        return pyarrow.bool_()
    if kind == "binary":
        return pyarrow.binary()
    if kind == "image":
        # HuggingFace's `Image` feature is exactly this struct; the schema
        # metadata is what tells `load_dataset` to decode it as a picture.
        return pyarrow.struct(
            [
                pyarrow.field("bytes", pyarrow.binary()),
                pyarrow.field("path", pyarrow.string()),
            ]
        )
    if kind == "string_list":
        return pyarrow.list_(pyarrow.string())
    if kind == "int32_list":
        return pyarrow.list_(pyarrow.int32())
    if kind == "elements":
        return pyarrow.list_(
            pyarrow.struct(
                [
                    pyarrow.field("id", pyarrow.string()),
                    pyarrow.field("type", pyarrow.string()),
                    pyarrow.field("text", pyarrow.string()),
                    pyarrow.field("bbox", pyarrow.list_(pyarrow.int32())),
                    pyarrow.field("reading_order", pyarrow.int32()),
                ]
            )
        )
    raise ExportError(f"no Arrow type is defined for column kind {kind!r}")


# --------------------------------------------------------------------------- #
# The sample header
# --------------------------------------------------------------------------- #


#: Columns every family carries, repeated on each of a sample's rows.
#:
#: ``letter_fields`` and ``scan`` are JSON strings rather than nested structs
#: because their keys vary by sample - a letter with no table has no table
#: fields, a born-digital page has no ``scan`` - and a struct column would have
#: to be the union of every key any sample ever carried, growing the schema
#: whenever the pipeline learns a new field. The ten flattened ``letter_fields``
#: columns beside them are the ones worth filtering on; the JSON is there so
#: nothing is lost when the pipeline adds an eleventh.
HEADER_COLUMNS: Final[tuple[Column, ...]] = (
    # Identity and provenance.
    Column("sample_id", "string", "Rendered sample id, e.g. `synth_000123`."),
    Column(
        "source_sample_id",
        "string",
        "Id of the source plaintext letter. The split is decided per group of "
        "this key, so every page of one letter shares a split.",
    ),
    Column("split", "string", "`train` or `test`, as assigned by the split plan."),
    Column("seed", "int64", "Per-sample RNG seed; re-renders the page byte-for-byte."),
    # Page geometry.
    Column("dpi", "int32", "Rasterisation density of the page images."),
    Column("page_size", "string", "`a4`, `a3`, `a5` or `letter`."),
    Column("orientation", "string", "`portrait` or `landscape`."),
    Column("page_count", "int32", "Pages the letter rendered to."),
    Column("element_count", "int32", "Annotated elements across the whole sample."),
    Column("texture", "string", "Paper texture applied, or empty for clean white."),
    # The letter's own administrative fields.
    Column("company_name", "string", "Sending organisation, as printed."),
    Column("destination_organization", "string", "Recipient organisation."),
    Column("subject", "string", "The letter's موضوع line."),
    Column("letter_number", "string", "شماره - the outgoing reference number."),
    Column("date", "string", "تاریخ - the Persian date as printed."),
    Column("national_id", "string", "شناسه ملی of the sender, when printed."),
    Column("registration_number", "string", "شماره ثبت of the sender, when printed."),
    Column("signatory_title", "string", "Title under the signature block."),
    Column(
        "overflowed",
        "bool",
        "True when the body did not fit the page budget and was truncated - "
        "useful to exclude from a strict transcription eval.",
    ),
    Column(
        "missing_assets",
        "string_list",
        "Assets the renderer asked for and did not find, e.g. a logo file.",
    ),
    # Filter flags, derived from the element types on the sample's own pages.
    Column("has_table", "bool", "The sample has at least one annotated table."),
    Column("has_chart", "bool", "The sample has at least one annotated chart."),
    Column("has_signature", "bool", "A handwritten signature was composited in."),
    Column("has_stamp", "bool", "A مهر stamp was composited in."),
    # What the text stage asked the drafting model for.
    Column("letter_genre", "string", "Genre the text stage requested, e.g. `notice`."),
    Column("letter_tone", "string", "Tone the text stage requested."),
    Column("chart_table", "string", "Structured element the text stage asked for."),
    Column("logo_id", "string", "Id of the SVG logo used in the letterhead."),
    Column("company_category", "string", "Sector of the sending organisation."),
    Column(
        "generation",
        "string",
        "JSON: which model drafted the letter body, and with what settings.",
    ),
    # Kept whole, for the keys the flattening does not know about.
    Column(
        "letter_fields",
        "string",
        "JSON: every field the renderer resolved, including the ones not "
        "flattened into their own column.",
    ),
    Column(
        "scan",
        "string",
        "JSON: the scanner simulation applied - blur, noise, skew, JPEG "
        "quality - or empty for a born-digital page.",
    ),
)

_HEADER_NAMES: Final[frozenset[str]] = frozenset(
    column.name for column in HEADER_COLUMNS
)

#: Header flag mapped to the element type whose presence sets it.
_FLAG_TYPES: Final[Mapping[str, ElementType]] = MappingProxyType(
    {
        "has_table": ElementType.TABLE,
        "has_chart": ElementType.CHART,
        "has_signature": ElementType.SIGNATURE_IMAGE,
        "has_stamp": ElementType.STAMP,
    }
)

#: ``letter_fields`` keys that become their own header column.
_LETTER_FIELD_COLUMNS: Final[tuple[str, ...]] = (
    "company_name",
    "destination_organization",
    "subject",
    "letter_number",
    "date",
    "national_id",
    "registration_number",
    "signatory_title",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return ``value`` when it is a mapping, an empty one otherwise."""
    return value if isinstance(value, Mapping) else {}


def _text(payload: Mapping[str, Any], key: str) -> str:
    """Read one key as a string, mapping both absent and ``None`` to ``""``."""
    return str(payload.get(key, "") or "")


def _string_list(value: Any) -> list[str]:
    """Coerce a JSON value into a list of strings.

    A scalar is wrapped rather than dropped: ``missing_assets`` is written as a
    list, and a hand-edited annotation holding a bare string should still
    export.
    """
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


def sample_header(
    annotation: DocumentAnnotation,
    *,
    split: str,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the per-sample columns every family repeats on every row.

    Args:
        annotation: The sample's ground truth, which is the authority for
            everything about the rendered page.
        split: The split :class:`~.splits.SplitPlan` assigned the sample to.
        source: The parsed source ``plaintext_XXXXXX.json``, or ``None`` when it
            could not be read. Its six columns are left empty in that case -
            they describe what the text stage asked for, which is context rather
            than ground truth, so losing them does not invalidate a sample.

    Returns:
        A mapping whose keys are exactly the names in :data:`HEADER_COLUMNS`.

    Raises:
        ExportError: If the built header does not match
            :data:`HEADER_COLUMNS`, which can only happen after an edit that
            changed one and not the other.
    """
    fields = annotation.letter_fields
    record = _mapping(source)
    inputs = _mapping(record.get("inputs"))
    letter = _mapping(record.get("letter"))
    generation = _mapping(record.get("generation"))
    present = {element.type for _, element in annotation.iter_elements()}

    header: dict[str, Any] = {
        "sample_id": annotation.sample_id,
        "source_sample_id": annotation.source_sample_id,
        "split": split,
        "seed": annotation.seed,
        "dpi": annotation.dpi,
        "page_size": annotation.page_size,
        "orientation": annotation.orientation,
        "page_count": len(annotation.pages),
        "element_count": annotation.element_count,
        "texture": annotation.texture or "",
        "overflowed": bool(fields.get("overflowed")),
        "missing_assets": _string_list(fields.get("missing_assets")),
        "letter_genre": _text(inputs, "letter_genre"),
        "letter_tone": _text(inputs, "letter_tone"),
        "chart_table": _text(inputs, "chart_table"),
        "logo_id": _text(inputs, "logo_id"),
        "company_category": (
            _text(letter, "company_category") or _text(inputs, "company_category")
        ),
        "generation": (
            json.dumps(dict(generation), ensure_ascii=False) if generation else ""
        ),
        "letter_fields": json.dumps(dict(fields), ensure_ascii=False),
        "scan": (
            json.dumps(annotation.scan, ensure_ascii=False) if annotation.scan else ""
        ),
    }
    for key in _LETTER_FIELD_COLUMNS:
        header[key] = _text(fields, key)
    for flag, element_type in _FLAG_TYPES.items():
        header[flag] = element_type in present
    if set(header) != _HEADER_NAMES:
        missing = sorted(_HEADER_NAMES - set(header))
        extra = sorted(set(header) - _HEADER_NAMES)
        raise ExportError(
            "the sample header no longer matches HEADER_COLUMNS "
            f"(missing {missing}, unexpected {extra})"
        )
    return header


# --------------------------------------------------------------------------- #
# One sample, read off disk once
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SampleBundle:
    """Everything the families need about one sample, read from disk once.

    Six families are written per sample and every one of them repeats the
    header, so the artifacts are read once here rather than once per family.

    Attributes:
        annotation: The sample's ground truth.
        header: The output of :func:`sample_header`.
        split: The split the sample was assigned to.
        page_images: Encoded page images, one per page and in page order. Empty
            when the ``images`` family was not requested.
        pdf: The whole document as one PDF, or ``None`` when absent.
        markdown: The Markdown reconstruction, or ``None`` when absent.
        plaintext: The source record's JSON text, or ``None`` when absent.
        source: That text parsed, or an empty mapping when it was absent or
            unparseable.
        missing: Families whose artifact could not be read, so the caller can
            count what the export left out rather than discovering it later as a
            short row count.
    """

    annotation: DocumentAnnotation
    header: Mapping[str, Any]
    split: str
    page_images: tuple[bytes, ...] = ()
    pdf: bytes | None = None
    markdown: str | None = None
    plaintext: str | None = None
    source: Mapping[str, Any] = MappingProxyType({})
    missing: tuple[str, ...] = ()

    @property
    def sample_id(self) -> str:
        """The render-stage id, for logs and the resume state."""
        return self.annotation.sample_id


def _optional_bytes(path: Path, *, sample_id: str, kind: str) -> bytes | None:
    """Read a per-sample artifact whose absence is tolerated.

    A missing file means the stage that writes it did not run for this sample,
    which costs one family's row. Any *other* ``OSError`` - a permission
    problem, a bad disk - is a broken machine rather than a partial corpus, and
    silently exporting a smaller dataset because of one is the failure this
    module is written to avoid.

    Args:
        path: The artifact.
        sample_id: Render-stage id, for the message.
        kind: Family name, for the message.

    Returns:
        The bytes, or ``None`` when the file does not exist.

    Raises:
        ExportError: If the file exists but cannot be read.
    """
    try:
        return path.read_bytes()
    except FileNotFoundError:
        _LOGGER.warning("%s: no %s artifact at %s", sample_id, kind, path)
        return None
    except OSError as error:
        raise ExportError(f"{sample_id}: cannot read {path}: {error}") from error


def _optional_text(path: Path, *, sample_id: str, kind: str) -> str | None:
    """Read a UTF-8 artifact whose absence is tolerated.

    Args:
        path: The artifact.
        sample_id: Render-stage id, for the message.
        kind: Family name, for the message.

    Returns:
        The decoded text, or ``None`` when the file does not exist.

    Raises:
        ExportError: If the file exists but cannot be read or is not UTF-8. The
            pipeline writes UTF-8, so a file that is not is not one of ours.
    """
    payload = _optional_bytes(path, sample_id=sample_id, kind=kind)
    if payload is None:
        return None
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ExportError(f"{sample_id}: {path} is not UTF-8: {error}") from error


def load_bundle(
    config: SynthConfig,
    annotation: DocumentAnnotation,
    *,
    split: str,
    families: Collection[str],
) -> SampleBundle:
    """Read one sample's artifacts and build its header.

    Only the artifacts the requested families need are read, so an export of
    ``metadata`` alone never touches a page image. The source plaintext record
    is the exception: it feeds six header columns that every family carries, so
    it is read whichever families are active.

    Args:
        config: Loaded configuration, for the artifact directories.
        annotation: The sample's ground truth, already parsed.
        split: The split the sample was assigned to.
        families: Which artifacts to read, as the union of the active families'
            :attr:`FamilySpec.needs`. The names coincide with the family names
            that consume them, so a family absent from this set is one whose
            missing artifact is not even looked for.

    Returns:
        The bundle, with :attr:`SampleBundle.missing` naming any requested
        family whose artifact was absent.

    Raises:
        ExportError: If a page image is missing or unreadable, or any artifact
            exists but cannot be read.
    """
    wanted = set(families)
    sample_id = annotation.sample_id
    paths = config.paths
    missing: list[str] = []

    page_images: tuple[bytes, ...] = ()
    if "images" in wanted:
        images: list[bytes] = []
        for page in annotation.pages:
            image_path = paths.output_dir / page.image_path
            try:
                images.append(image_path.read_bytes())
            except OSError as error:
                raise ExportError(
                    f"{sample_id} page {page.page_index}: cannot read "
                    f"{image_path}: {error}"
                ) from error
        page_images = tuple(images)

    pdf: bytes | None = None
    if "pdf" in wanted:
        pdf = _optional_bytes(
            paths.pdf_dir / f"{sample_id}.pdf", sample_id=sample_id, kind="pdf"
        )
        if pdf is None:
            missing.append("pdf")

    markdown: str | None = None
    if "markdown" in wanted:
        markdown = _optional_text(
            paths.markdown_dir / f"{sample_id}.md",
            sample_id=sample_id,
            kind="markdown",
        )
        if markdown is None:
            missing.append("markdown")

    plaintext_path = paths.plaintext_dir / f"{annotation.source_sample_id}.json"
    plaintext = _optional_text(
        plaintext_path, sample_id=sample_id, kind="plaintext"
    )
    if plaintext is None and "plaintext" in wanted:
        missing.append("plaintext")

    source: Mapping[str, Any] = {}
    if plaintext is not None:
        try:
            parsed = json.loads(plaintext)
        except json.JSONDecodeError as error:
            # Not fatal: the six columns it feeds are context, and the verbatim
            # text still ships. A tree the pipeline wrote cannot reach here at
            # all, since the render read the same file.
            _LOGGER.warning(
                "%s: %s is not valid JSON (%s); its header fields stay empty",
                sample_id,
                plaintext_path,
                error,
            )
            parsed = None
        if isinstance(parsed, Mapping):
            source = parsed
        elif parsed is not None:
            _LOGGER.warning(
                "%s: %s is not a JSON object; its header fields stay empty",
                sample_id,
                plaintext_path,
            )

    return SampleBundle(
        annotation=annotation,
        header=sample_header(annotation, split=split, source=source),
        split=split,
        page_images=page_images,
        pdf=pdf,
        markdown=markdown,
        plaintext=plaintext,
        source=source,
        missing=tuple(missing),
    )


# --------------------------------------------------------------------------- #
# Row builders
# --------------------------------------------------------------------------- #


class RowBuilder(Protocol):
    """Turns one bundle into one family's rows."""

    def __call__(
        self, bundle: SampleBundle, *, include_elements: bool
    ) -> list[dict[str, Any]]:
        """Return the rows, or an empty list when the artifact was absent."""


def _images_rows(
    bundle: SampleBundle, *, include_elements: bool
) -> list[dict[str, Any]]:
    """Build one row per rendered page."""
    annotation = bundle.annotation
    if len(bundle.page_images) != len(annotation.pages):
        raise ExportError(
            f"{bundle.sample_id}: {len(bundle.page_images)} page image(s) read "
            f"for {len(annotation.pages)} annotated page(s)"
        )
    rows: list[dict[str, Any]] = []
    for page, image in zip(annotation.pages, bundle.page_images):
        row: dict[str, Any] = {
            "image": {"bytes": image, "path": page.image_path},
            "text": page.transcription(),
            "page_index": page.page_index,
            "width": page.width,
            "height": page.height,
            "image_bytes": len(image),
            **bundle.header,
        }
        if include_elements:
            row["elements"] = [
                {
                    "id": element.id,
                    "type": str(element.type),
                    "text": element.text,
                    "bbox": element.bbox.as_list(),
                    "reading_order": element.reading_order,
                }
                for element in page.elements
            ]
        rows.append(row)
    return rows


def _plaintext_rows(
    bundle: SampleBundle, *, include_elements: bool
) -> list[dict[str, Any]]:
    """Build the one row carrying the source letter the render was built from."""
    del include_elements
    if bundle.plaintext is None:
        return []
    letter = _mapping(bundle.source.get("letter"))
    return [
        {
            "plaintext": bundle.plaintext,
            "body": _text(letter, "body"),
            "attachments": _text(letter, "attachments"),
            "full_transcription": bundle.annotation.full_transcription,
            **bundle.header,
        }
    ]


def _markdown_rows(
    bundle: SampleBundle, *, include_elements: bool
) -> list[dict[str, Any]]:
    """Build the one row carrying the Markdown reconstruction."""
    del include_elements
    if bundle.markdown is None:
        return []
    return [
        {
            "markdown": bundle.markdown,
            "markdown_bytes": len(bundle.markdown.encode("utf-8")),
            **bundle.header,
        }
    ]


def _pdf_rows(
    bundle: SampleBundle, *, include_elements: bool
) -> list[dict[str, Any]]:
    """Build the one row carrying the whole document as a PDF."""
    del include_elements
    if bundle.pdf is None:
        return []
    return [
        {"pdf": bundle.pdf, "pdf_bytes": len(bundle.pdf), **bundle.header}
    ]


def _annotations_rows(
    bundle: SampleBundle, *, include_elements: bool
) -> list[dict[str, Any]]:
    """Build the one row carrying the complete ground truth.

    ``include_elements`` is deliberately ignored: turning it off slims the
    trainable ``images`` family, and this family is where the geometry keeps
    living so nothing is dropped from the export as a whole.

    The annotation is re-serialised from the parsed dataclass rather than copied
    from the file's bytes, so this column is provably the same ground truth the
    ``images`` rows' boxes came from.
    """
    del include_elements
    annotation = bundle.annotation
    return [
        {
            "annotation": json.dumps(annotation.as_dict(), ensure_ascii=False),
            "full_transcription": annotation.full_transcription,
            "paddle_label": json.dumps(
                paddle_records(annotation), ensure_ascii=False
            ),
            "farmer_label": json.dumps(
                paddle_farmer_records(annotation), ensure_ascii=False
            ),
            **bundle.header,
        }
    ]


def _metadata_rows(
    bundle: SampleBundle, *, include_elements: bool
) -> list[dict[str, Any]]:
    """Build the one blob-free row, for a consumer indexing the corpus."""
    del include_elements
    annotation = bundle.annotation
    text = annotation.full_transcription
    return [
        {
            "image_paths": [page.image_path for page in annotation.pages],
            "page_widths": [page.width for page in annotation.pages],
            "page_heights": [page.height for page in annotation.pages],
            "text_chars": len(text),
            "text_words": len(text.split()),
            **bundle.header,
        }
    ]


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FamilySpec:
    """One artifact kind: its columns, its rows, and how it is buffered.

    Attributes:
        name: Family name, which is also its subdirectory under ``data/`` and
            therefore the config name ``load_dataset`` takes.
        granularity: ``"page"`` when the family writes one row per rendered
            page, ``"sample"`` when it writes one per letter. Documented in the
            card, since it is the difference between a row count and a sample
            count.
        row_group_bytes: Uncompressed payload the family's row-group buffer
            accumulates before it is written. Per family rather than global
            because every configured family holds one open writer per split, so
            one shared budget would be multiplied by twelve.
        columns: Every column in written order, the conditional ones included.
        build_rows: Turns a :class:`SampleBundle` into this family's rows.
        needs: Which artifacts :func:`load_bundle` must read for it.
        summary: One line for the dataset card.
    """

    name: str
    granularity: str
    row_group_bytes: int
    columns: tuple[Column, ...]
    build_rows: RowBuilder
    needs: tuple[str, ...] = ()
    summary: str = ""

    def active(self, *, include_elements: bool) -> tuple[Column, ...]:
        """Return the columns this export actually writes."""
        if include_elements:
            return self.columns
        return tuple(
            column for column in self.columns if not column.only_with_elements
        )

    def payload_columns(self, *, include_elements: bool) -> tuple[Column, ...]:
        """Return the active columns that are *not* part of the shared header.

        The dataset card documents the header once and then only what each
        family adds, so it needs the two groups apart.
        """
        return tuple(
            column
            for column in self.active(include_elements=include_elements)
            if column.name not in _HEADER_NAMES
        )

    def features(self, *, include_elements: bool) -> dict[str, Any]:
        """Return the family's schema in ``datasets`` feature notation.

        Written into the Arrow schema's metadata and into ``dataset_info.json``.
        The card's front matter needs the same schema in a different spelling;
        that is :meth:`yaml_features`.
        """
        return {
            column.name: _KIND_FEATURES[column.kind]
            for column in self.active(include_elements=include_elements)
        }

    def yaml_features(self, *, include_elements: bool) -> list[dict[str, Any]]:
        """Return the family's schema as the dataset card's front matter wants it.

        Same columns, same order, same types as :meth:`features` and
        :meth:`schema` - one :class:`Column` list generates all three - but in
        the simplified notation ``datasets`` reads out of a card's
        ``dataset_info`` block. Order matters: ``load_dataset`` casts each shard
        to the declared features, so a name or a type that disagreed with the
        Parquet would turn a working dataset into a load error.
        """
        return [
            {"name": column.name, **_plain(_KIND_YAML[column.kind])}
            for column in self.active(include_elements=include_elements)
        ]

    def schema(self, pyarrow: Any, *, include_elements: bool) -> Any:
        """Build the Arrow schema this family's shards are written against.

        Declared rather than inferred per batch. Inference would type a column
        from whatever the first batch happened to contain, so a shard whose
        samples all lack a texture would get ``null`` where its neighbour got
        ``string`` - and the two would no longer concatenate.

        Args:
            pyarrow: The imported ``pyarrow`` module.
            include_elements: Whether the conditional columns are present.

        Returns:
            A ``pyarrow.Schema`` carrying the ``datasets`` feature descriptor in
            its metadata, so ``load_dataset`` decodes ``image`` as an image.
        """
        columns = self.active(include_elements=include_elements)
        metadata = {
            b"huggingface": json.dumps(
                {
                    "info": {
                        "features": self.features(
                            include_elements=include_elements
                        )
                    }
                },
                ensure_ascii=False,
            ).encode("utf-8")
        }
        return pyarrow.schema(
            [
                pyarrow.field(column.name, _arrow_type(pyarrow, column.kind))
                for column in columns
            ],
            metadata=metadata,
        )


def _columns(*payload: Column, tail: Sequence[Column] = ()) -> tuple[Column, ...]:
    """Assemble a family's column order: its own payload, the header, then tail.

    Args:
        payload: The family's distinctive columns, blob first so the Hub's
            viewer leads with it.
        tail: Columns that belong after the header, which in practice means the
            conditional element list.

    Returns:
        The full declared order.
    """
    return (*payload, *HEADER_COLUMNS, *tail)


#: Every family, keyed by name and ordered as :data:`HF_FAMILIES` orders them.
FAMILIES: Final[Mapping[str, FamilySpec]] = MappingProxyType(
    {
        "images": FamilySpec(
            name="images",
            granularity="page",
            row_group_bytes=32 * _MIB,
            columns=_columns(
                Column("image", "image", "The rendered page, embedded as bytes."),
                Column("text", "string", "This page's transcription, reading order."),
                Column("page_index", "int32", "0-based index of the page."),
                Column("width", "int32", "Page width in pixels."),
                Column("height", "int32", "Page height in pixels."),
                Column("image_bytes", "int32", "Size of the encoded image."),
                tail=(
                    Column(
                        "elements",
                        "elements",
                        "Per-element id, type, text, `[x0, y0, x1, y1]` box and "
                        "reading order. Present when the export was run with "
                        "`include_elements`.",
                        only_with_elements=True,
                    ),
                ),
            ),
            build_rows=_images_rows,
            needs=("images",),
            summary=(
                "One row per rendered page: the page image, its transcription "
                "in reading order, and its element boxes."
            ),
        ),
        "plaintext": FamilySpec(
            name="plaintext",
            granularity="sample",
            row_group_bytes=8 * _MIB,
            columns=_columns(
                Column(
                    "plaintext",
                    "string",
                    "The text stage's `plaintext_XXXXXX.json` verbatim.",
                ),
                Column("body", "string", "The drafted letter body, on its own."),
                Column("attachments", "string", "The پیوست lines, one per line."),
                Column(
                    "full_transcription",
                    "string",
                    "What actually reached the page, for comparison with `body`.",
                ),
            ),
            build_rows=_plaintext_rows,
            needs=("plaintext",),
            summary=(
                "The source letter the render was built from: the text stage's "
                "record verbatim, plus its body and attachments split out."
            ),
        ),
        "markdown": FamilySpec(
            name="markdown",
            granularity="sample",
            row_group_bytes=8 * _MIB,
            columns=_columns(
                Column("markdown", "string", "The Markdown document."),
                Column("markdown_bytes", "int32", "Its size in UTF-8 bytes."),
            ),
            build_rows=_markdown_rows,
            needs=("markdown",),
            summary=(
                "The Markdown reconstruction of the letter, as the composer "
                "wrote it before rasterisation."
            ),
        ),
        "pdf": FamilySpec(
            name="pdf",
            granularity="sample",
            row_group_bytes=32 * _MIB,
            columns=_columns(
                Column("pdf", "binary", "The whole document as PDF bytes."),
                Column("pdf_bytes", "int32", "Its size in bytes."),
            ),
            build_rows=_pdf_rows,
            needs=("pdf",),
            summary=(
                "The whole document as one PDF, with selectable text, for a "
                "consumer who wants the vector page rather than the raster."
            ),
        ),
        "annotations": FamilySpec(
            name="annotations",
            granularity="sample",
            row_group_bytes=8 * _MIB,
            columns=_columns(
                Column(
                    "annotation",
                    "string",
                    "The full annotation as JSON - every page, every element, "
                    "every box - regardless of `include_elements`.",
                ),
                Column(
                    "full_transcription",
                    "string",
                    "All text on all pages, in reading order.",
                ),
                Column(
                    "paddle_label",
                    "string",
                    "JSON lines: the PaddleOCR-VL projection, one per page.",
                ),
                Column(
                    "farmer_label",
                    "string",
                    "JSON lines: the PaddleFarmer projection, one per page.",
                ),
            ),
            build_rows=_annotations_rows,
            needs=(),
            summary=(
                "The complete ground truth as JSON, plus the PaddleOCR-VL and "
                "PaddleFarmer projections of it."
            ),
        ),
        "metadata": FamilySpec(
            name="metadata",
            granularity="sample",
            row_group_bytes=2 * _MIB,
            columns=_columns(
                Column(
                    "image_paths",
                    "string_list",
                    "Page image paths, relative to the render tree's root.",
                ),
                Column("page_widths", "int32_list", "Width of each page, in pixels."),
                Column("page_heights", "int32_list", "Height of each page, in pixels."),
                Column("text_chars", "int32", "Characters in the transcription."),
                Column("text_words", "int32", "Whitespace-separated tokens in it."),
            ),
            build_rows=_metadata_rows,
            needs=(),
            summary=(
                "One blob-free row per sample: the header, where the pages "
                "live, and how much text they carry."
            ),
        ),
    }
)

if tuple(FAMILIES) != HF_FAMILIES:  # pragma: no cover - guards an edit
    raise RuntimeError(
        "the family registry and HF_FAMILIES disagree: "
        f"{tuple(FAMILIES)} vs {HF_FAMILIES}. The config validates names "
        "against HF_FAMILIES and the exporter looks them up here, so the two "
        "have to list the same families in the same order."
    )

_UNDOCUMENTED: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            f"{spec.name}.{column.name}"
            for spec in FAMILIES.values()
            for column in spec.columns
            if not column.note
        }
    )
)
if _UNDOCUMENTED:  # pragma: no cover - guards an edit
    raise RuntimeError(
        f"these export columns have no note: {list(_UNDOCUMENTED)}. The dataset "
        "card builds its column tables from Column.note, so a column without "
        "one would ship undocumented."
    )


def family_specs(names: Sequence[str]) -> tuple[FamilySpec, ...]:
    """Return the specs for the configured families, in registry order.

    Args:
        names: Family names, as validated by the configuration.

    Returns:
        One spec per name.

    Raises:
        ExportError: If a name has no spec. The configuration rejects unknown
            names, so this is a guard against an edit rather than against user
            input.
    """
    unknown = sorted(set(names) - set(FAMILIES))
    if unknown:
        raise ExportError(
            f"no export family is defined for {unknown}; "
            f"known families are {sorted(FAMILIES)}"
        )
    wanted = set(names)
    return tuple(spec for name, spec in FAMILIES.items() if name in wanted)


# --------------------------------------------------------------------------- #
# Size accounting
# --------------------------------------------------------------------------- #


def _value_bytes(value: Any) -> int:
    """Estimate one cell's uncompressed size.

    ``bool`` is checked before ``int`` because it is a subclass of it, and
    ``str``/``bytes`` before ``Sequence`` for the same reason.
    """
    if value is None:
        return 0
    if isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return 8
    if isinstance(value, float):
        return 8
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, Mapping):
        return sum(_value_bytes(item) for item in value.values())
    if isinstance(value, Sequence):
        return sum(_value_bytes(item) for item in value)
    return 0


def row_payload_bytes(row: Mapping[str, Any]) -> int:
    """Estimate one row's uncompressed payload.

    Used for the shard ceiling, which is stated as a limit on the written file.
    Measuring before compression makes the estimate conservative in the right
    direction: the blobs are already compressed and dominate the total, so the
    file lands slightly under the ceiling rather than over it.

    Args:
        row: A row as built by a family's :attr:`FamilySpec.build_rows`.

    Returns:
        Approximate bytes the row contributes.
    """
    return sum(_value_bytes(value) for value in row.values())
