"""The administrative metadata block: شماره / تاریخ / پیوست and friends.

CLAUDE.md 5.2 names this block the highest-priority field for model accuracy,
so it is modelled more carefully than any other region on the page:

* **Every field carries its own bounding box**, not just the block. A consumer
  can therefore evaluate ``تاریخ`` extraction without segmenting the block
  itself, and the eval stage can score field-level accuracy directly.
* **The label and the value are separate spans inside that field.** Real
  letters vary the separator (``:``, a dotted leader, an empty gap), and a
  model that only ever saw ``شماره: ۱۲۳`` would learn the colon, not the
  number.
* **The label wording is sampled**, because Persian administrative practice
  has no single spelling: ``شماره``, ``شماره نامه`` and ``شمارهٔ نامه`` all
  appear on real letterhead, as do ``پیوست`` and ``ضمائم``.

The block sits top-right or top-left, opposite the logo wherever possible -
the layout convention on Iranian letterhead, and the arrangement that keeps
the two from colliding.
"""

from __future__ import annotations

import html
import random
from dataclasses import dataclass
from typing import Final, Sequence

from ..common.io_utils import to_persian_digits, weighted_choice

__all__ = [
    "MetadataField",
    "MetadataBlockSpec",
    "build_metadata_block",
    "metadata_to_html",
    "metadata_to_markdown",
]

#: Label wordings per field key. The first entry is the most common form on
#: real letterhead, and carries the most weight in :func:`_pick_label`.
_LABEL_VARIANTS: Final[dict[str, tuple[str, ...]]] = {
    "letter_number": ("شماره", "شماره نامه", "شمارهٔ نامه", "شماره اندیکاتور"),
    "registration_number": ("شماره ثبت", "شمارهٔ ثبت", "شماره ثبت نامه"),
    "date": ("تاریخ", "تاریخ نامه", "مورخ"),
    "attachment": ("پیوست", "ضمائم", "پیوست‌ها"),
    "national_id": ("شناسه ملی", "شناسهٔ ملی", "شناسه ملی شرکت"),
}

#: Weights matching ``_LABEL_VARIANTS`` order: the canonical wording dominates
#: so the corpus still reads as real correspondence.
_LABEL_WEIGHTS: Final[tuple[float, ...]] = (60.0, 18.0, 14.0, 8.0)

#: Separator drawn between label and value, with relative weights. The empty
#: string is a real treatment - many letterheads align values in a column and
#: use whitespace alone.
_SEPARATORS: Final[tuple[tuple[str, float], ...]] = (
    (":", 62.0),
    ("", 20.0),
    ("ـــــ", 10.0),
    (".....", 8.0),
)

#: Value for a letter that has attachments.
_ATTACHMENT_PRESENT: Final[tuple[str, ...]] = ("دارد", "دارد (۱ برگ)", "به پیوست")

#: Value for a letter with nothing attached. Real letters print this rather
#: than omitting the row, which is why the field survives an empty value.
_ATTACHMENT_ABSENT: Final[tuple[str, ...]] = ("ندارد", "-", "فاقد پیوست")

#: Field order variants. ``letter_number`` before ``date`` is near-universal;
#: the rest of the ordering genuinely varies between organisations.
_FIELD_ORDERS: Final[tuple[tuple[str, ...], ...]] = (
    ("letter_number", "date", "attachment"),
    ("letter_number", "date", "attachment", "national_id"),
    ("registration_number", "date", "attachment"),
    ("letter_number", "registration_number", "date", "attachment"),
    ("date", "letter_number", "attachment"),
    ("letter_number", "date"),
)

#: Probability that the block is placed on the same side as the logo. Kept low
#: because the two crowd each other, but not zero - real letterhead does it.
_SAME_SIDE_PROBABILITY: Final[float] = 0.12

#: Probability that the block is drawn inside a hairline border.
_BOXED_PROBABILITY: Final[float] = 0.34

#: Probability that labels are rendered at all. A minority of real letters
#: print bare values in a fixed column.
_LABELLED_PROBABILITY: Final[float] = 0.94


def _pick_label(key: str, rng: random.Random) -> str:
    """Pick one wording for a field's label.

    Args:
        key: Field key, e.g. ``letter_number``.
        rng: The sample's seeded RNG.

    Returns:
        A Persian label, or the key itself if it is unknown.
    """
    variants = _LABEL_VARIANTS.get(key)
    if not variants:
        return key
    weights = _LABEL_WEIGHTS[: len(variants)]
    return weighted_choice(tuple(zip(variants, weights)), rng)


@dataclass(frozen=True, slots=True)
class MetadataField:
    """One label/value row of the metadata block.

    Attributes:
        key: Stable machine key, e.g. ``date``. Used as the element id suffix,
            so an evaluator can find a field without matching Persian text.
        label: The Persian label as rendered, which varies per sample.
        value: The field's value, already in Persian digits.
        separator: What is drawn between label and value.
    """

    key: str
    label: str
    value: str
    separator: str = ":"

    @property
    def text(self) -> str:
        """The field as one reading line, for the transcription."""
        if self.separator:
            return f"{self.label}{self.separator} {self.value}".strip()
        return f"{self.label} {self.value}".strip()

    def as_content(self) -> dict[str, str]:
        """Return the field's structured payload for the annotation."""
        return {"key": self.key, "label": self.label, "value": self.value}


@dataclass(frozen=True, slots=True)
class MetadataBlockSpec:
    """The whole metadata block for one letter.

    Attributes:
        fields: The rows, in the order they are rendered.
        side: ``right`` or ``left`` - which margin the block hugs.
        boxed: Whether the block is drawn inside a hairline border, a common
            treatment that changes how the region segments.
        show_labels: Whether labels are rendered at all.
    """

    fields: tuple[MetadataField, ...]
    side: str = "right"
    boxed: bool = False
    show_labels: bool = True

    @property
    def text(self) -> str:
        """The block as reading text, one field per line."""
        return "\n".join(item.text for item in self.fields)

    def field_for(self, key: str) -> MetadataField | None:
        """Return one field by key, or ``None`` when it is not rendered."""
        for item in self.fields:
            if item.key == key:
                return item
        return None

    def as_content(self) -> dict[str, object]:
        """Return the block's structured payload for the annotation."""
        return {
            "side": self.side,
            "boxed": self.boxed,
            "show_labels": self.show_labels,
            "fields": [item.as_content() for item in self.fields],
        }


def build_metadata_block(
    letter_number: str,
    registration_number: str,
    date: str,
    national_id: str,
    has_attachments: bool,
    rng: random.Random,
    logo_side: str = "right",
) -> MetadataBlockSpec:
    """Assemble the metadata block for one letter.

    A field whose value is empty is dropped rather than rendered blank, so the
    annotation never claims a box for text that is not on the page. If that
    leaves nothing at all, the block falls back to the date alone - every real
    administrative letter is dated.

    Args:
        letter_number: ``letter.letter_number`` from the record.
        registration_number: ``letter.registration_number`` from the record.
        date: ``letter.date`` from the record, already Persian-formatted.
        national_id: ``letter.national_id`` from the record.
        has_attachments: Whether the letter enumerates any attachments.
        rng: The sample's seeded RNG, conventionally ``purpose="metadata"``.
        logo_side: Which side the header logo sits on, so the block can take
            the opposite margin.

    Returns:
        The block spec, with at least one field.
    """
    values: dict[str, str] = {
        "letter_number": to_persian_digits(letter_number).strip(),
        "registration_number": to_persian_digits(registration_number).strip(),
        "date": to_persian_digits(date).strip(),
        "national_id": to_persian_digits(national_id).strip(),
        "attachment": rng.choice(
            _ATTACHMENT_PRESENT if has_attachments else _ATTACHMENT_ABSENT
        ),
    }

    order = rng.choice(_FIELD_ORDERS)
    separator = weighted_choice(_SEPARATORS, rng)
    fields = tuple(
        MetadataField(
            key=key,
            label=_pick_label(key, rng),
            value=values[key],
            separator=separator,
        )
        for key in order
        if values.get(key)
    )
    if not fields:
        fields = (
            MetadataField(
                key="date",
                label=_pick_label("date", rng),
                value=values["date"] or "-",
                separator=separator,
            ),
        )

    opposite = "left" if logo_side == "right" else "right"
    side = logo_side if rng.random() < _SAME_SIDE_PROBABILITY else opposite
    return MetadataBlockSpec(
        fields=fields,
        # A centred logo leaves both margins free, so the block keeps the
        # right-hand side that Persian letterhead defaults to.
        side="right" if logo_side == "center" else side,
        boxed=rng.random() < _BOXED_PROBABILITY,
        show_labels=rng.random() < _LABELLED_PROBABILITY,
    )


def metadata_to_html(spec: MetadataBlockSpec, element_id: str = "meta") -> str:
    """Render the metadata block to annotated HTML.

    Args:
        spec: The block to render.
        element_id: Id prefix; each field becomes ``{element_id}.{key}``.

    Returns:
        An HTML fragment whose block and every field carry ``data-ann-*``.
    """
    safe_id = html.escape(element_id, quote=True)
    classes = ["metadata-block", f"metadata-{html.escape(spec.side, quote=True)}"]
    if spec.boxed:
        classes.append("metadata-boxed")
    parts: list[str] = [
        f'<div class="{" ".join(classes)}" data-ann-type="metadata_block" '
        f'data-ann-id="{safe_id}">'
    ]
    for item in spec.fields:
        field_id = html.escape(f"{element_id}.{item.key}", quote=True)
        safe_key = html.escape(item.key, quote=True)
        parts.append(
            f'<div class="metadata-field" data-ann-type="metadata_field" '
            f'data-ann-id="{field_id}" data-ann-key="{safe_key}">'
        )
        if spec.show_labels:
            parts.append(
                f'<span class="metadata-label">{html.escape(item.label)}'
                f"{html.escape(item.separator)}</span>"
            )
        parts.append(
            f'<span class="metadata-value">{html.escape(item.value)}</span></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def metadata_to_markdown(spec: MetadataBlockSpec) -> str:
    """Render the metadata block for the ``.md`` artifact.

    Args:
        spec: The block to render.

    Returns:
        One label/value line per field, joined by hard line breaks.
    """
    lines = [
        f"**{item.label}:** {item.value}" if spec.show_labels else item.value
        for item in spec.fields
    ]
    return "  \n".join(lines)
