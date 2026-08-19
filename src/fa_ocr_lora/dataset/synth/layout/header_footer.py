"""Letterhead and footer: logo placement, company identity, contact fields.

The header and footer are where one letter stops looking like another. Both
are assembled here as specs first and HTML second, so the composer decides
*what* appears and this module owns *how* it is tagged for annotation.

Three decisions are worth stating up front:

* **The logo is embedded as an ``<img>`` data URI, not inlined markup.** 171
  of the 4888 logo files carry their own ``<style>`` block and some carry
  ``id`` attributes; pasting that into the page would leak CSS rules and
  collide ids with whatever else the sample renders. A replaced element is a
  sealed rendering context, so no logo can restyle the letter around it.
* **Monochrome logos get recoloured when the bar would swallow them.** 2901
  files set a single ``fill`` on the root ``<svg>``, so a brand-coloured logo
  on a solid brand-coloured bar renders as a silhouette of nothing. When the
  bar is solid, the root fill is rewritten to the bar's text colour.
  Multi-colour logos keep their own palette - there is no single fill to
  rewrite - which is why the solid bar is not the most common treatment.
* **The page number is a template, not a value.** It reads ``صفحه ۱ از ۳``,
  and neither number is known until pagination has run. The node ships with
  ``data-page-template`` holding ``{page}`` and ``{total}`` placeholders for
  ``paginate.js`` to fill per sheet, and renders a sane single-page string in
  the meantime so a screenshot taken without the script still looks like a
  letter.
"""

from __future__ import annotations

import base64
import html
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final, Sequence

from ....config import HeaderVocabulary, LayoutConfig, ProbabilityConfig
from ...common.io_utils import to_persian_digits, weighted_choice

__all__ = [
    "HeaderSpec",
    "FooterSpec",
    "logo_data_uri",
    "build_header",
    "build_footer",
    "header_to_html",
    "footer_to_html",
    "opener_to_html",
    "header_to_markdown",
    "footer_to_markdown",
]

#: Fallbacks for a caller that does not hand over the loaded config - the test
#: suite, mostly. Read off an instance rather than written out again, so there is
#: exactly one place a default lives: a literal here would be free to drift from
#: the dataclass, and did (this signature said 0.92 while the dataclass said 0.9).
_DEFAULT_LAYOUT: Final[LayoutConfig] = LayoutConfig()
_DEFAULT_PROBABILITIES: Final[ProbabilityConfig] = ProbabilityConfig()

#: Same idea for the Persian wording: the pools live in
#: ``configs/vocabulary_fa.yaml`` and arrive as ``config.vocabulary.header``.
_DEFAULT_VOCABULARY: Final[HeaderVocabulary] = HeaderVocabulary()

#: Field orders for the footer, and how those fields are laid out, both now live
#: in ``layout.contact_orders`` / ``layout.footer_layouts``.

#: Root-tag ``fill="..."`` on an ``<svg>`` element.
#:
#: Intentionally not a knob: this is the grammar of the SVG files on disk, not
#: a stylistic choice.
_ROOT_FILL: Final[re.Pattern[str]] = re.compile(
    r"(<svg\b[^>]*?)\bfill\s*=\s*(['\"])(.*?)\2", re.IGNORECASE | re.DOTALL
)

#: Opening ``<svg`` tag, for injecting a fill when the root has none.
_SVG_OPEN: Final[re.Pattern[str]] = re.compile(r"<svg\b", re.IGNORECASE)


def _set_root_fill(svg_text: str, colour: str) -> str:
    """Force the root ``<svg>`` fill, so a monochrome mark can be recoloured.

    Only the root attribute is touched. A logo that colours its own paths
    keeps them, which is deliberate: rewriting every ``fill`` in the file
    would flatten a multi-colour mark into a blob.

    Args:
        svg_text: The SVG source.
        colour: CSS colour to apply.

    Returns:
        The SVG with its root fill set.
    """
    replaced, count = _ROOT_FILL.subn(
        lambda match: f'{match.group(1)}fill="{colour}"', svg_text, count=1
    )
    if count:
        return replaced
    return _SVG_OPEN.sub(f'<svg fill="{colour}"', svg_text, count=1)


@lru_cache(maxsize=256)
def logo_data_uri(path: Path, colour: str | None = None) -> str:
    """Read one logo SVG and encode it as a ``data:`` URI.

    Cached because a batch reuses the same few hundred marks, and because the
    same logo recoloured for a solid bar is a distinct cache key.

    Args:
        path: The SVG file on disk.
        colour: Root fill override, or ``None`` to keep the file's own colour.

    Returns:
        A base64 ``data:image/svg+xml`` URI, or an empty string when the file
        cannot be read.
    """
    try:
        svg_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    if colour:
        svg_text = _set_root_fill(svg_text, colour)
    encoded = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


@dataclass(frozen=True, slots=True)
class HeaderSpec:
    """The letterhead for one document.

    Attributes:
        enabled: Whether a header is drawn at all.
        company_name: The organisation line, prefix included.
        slogan: Sub-line under the name, empty when the sample has none.
        opener: Religious opener such as ``بسمه تعالی``, empty when absent.
        logo_path: The mark to draw, or ``None`` when the row has no usable
            logo or the sample draws none.
        logo_side: ``right``, ``center`` or ``left``.
        logo_height_mm: Rendered height of the mark, in millimetres.
        bar_style: Treatment from :class:`~..layout.theme.BarStyle`.
        show_category: Whether the company's category line is printed.
        category: The category text, already formatted for display.
    """

    enabled: bool = True
    company_name: str = ""
    slogan: str = ""
    opener: str = ""
    logo_path: Path | None = None
    logo_side: str = "right"
    logo_height_mm: float = 14.0
    bar_style: str = "rule"
    show_category: bool = False
    category: str = ""

    @property
    def has_logo(self) -> bool:
        """Whether a logo image is drawn in the header."""
        return self.logo_path is not None

    @property
    def text(self) -> str:
        """The header as reading text, one line per rendered element."""
        lines = [
            self.company_name,
            self.category if self.show_category else "",
            self.slogan,
        ]
        return "\n".join(line for line in lines if line.strip())

    def as_content(self) -> dict[str, object]:
        """Return the header's structured payload for the annotation."""
        return {
            "company_name": self.company_name,
            "slogan": self.slogan,
            "category": self.category if self.show_category else "",
            "logo_side": self.logo_side,
            "logo_height_mm": self.logo_height_mm,
            "bar_style": self.bar_style,
            "logo": self.logo_path.name if self.logo_path else None,
        }


@dataclass(frozen=True, slots=True)
class FooterSpec:
    """The footer for one document.

    Attributes:
        enabled: Whether a footer is drawn at all.
        fields: ``(key, label, value)`` per contact row, in render order. The
            label is empty when the sample prints bare values.
        layout: ``inline``, ``stacked`` or ``split``.
        bar_style: Treatment from :class:`~..layout.theme.BarStyle`.
        page_number_template: Format string with ``{page}`` and ``{total}``,
            or empty when the sample prints no page number.
    """

    enabled: bool = True
    fields: tuple[tuple[str, str, str], ...] = ()
    layout: str = "inline"
    bar_style: str = "rule"
    page_number_template: str = ""

    @property
    def has_page_number(self) -> bool:
        """Whether a page number is printed."""
        return bool(self.page_number_template)

    @property
    def text(self) -> str:
        """The footer as reading text, one line per field."""
        return "\n".join(
            f"{label}: {value}" if label else value for _, label, value in self.fields
        )

    def as_content(self) -> dict[str, object]:
        """Return the footer's structured payload for the annotation."""
        return {
            "layout": self.layout,
            "bar_style": self.bar_style,
            "fields": [
                {"key": key, "label": label, "value": value}
                for key, label, value in self.fields
            ],
            "page_number_template": self.page_number_template,
        }


def build_header(
    company_name: str,
    category: str,
    logo_path: Path | None,
    bar_style: str,
    rng: random.Random,
    layout: LayoutConfig = _DEFAULT_LAYOUT,
    header_probability: float = _DEFAULT_PROBABILITIES.header,
    slogan_probability: float = _DEFAULT_PROBABILITIES.header_slogan,
    opener_probability: float = _DEFAULT_PROBABILITIES.religious_opener,
    category_probability: float = _DEFAULT_PROBABILITIES.header_category,
    vocabulary: HeaderVocabulary = _DEFAULT_VOCABULARY,
) -> HeaderSpec:
    """Assemble the letterhead for one document.

    Args:
        company_name: The brand name from the letter record.
        category: The company category, used as an optional sub-line.
        logo_path: Resolved logo SVG, or ``None`` when the row has none.
        bar_style: Header bar treatment chosen by the theme.
        rng: The sample's seeded RNG, conventionally ``purpose="header"``.
        layout: Logo placement and size, from ``synth_config.yaml``.
        header_probability: Chance the letter has a header at all.
        slogan_probability: Chance a slogan line is printed.
        opener_probability: Chance a religious opener is printed. Drawn even
            when the header is disabled, because the opener sits above the
            body and survives a bare page.
        category_probability: Chance the category sub-line is printed, for a
            row that carries a category at all.
        vocabulary: The ``header`` pools from ``vocabulary_fa.yaml``: name
            prefixes, openers and slogans.

    Returns:
        The header spec; ``enabled`` is ``False`` for a headerless letter.
    """
    enabled = rng.random() < header_probability
    opener = (
        weighted_choice(vocabulary.religious_openers, rng)
        if rng.random() < opener_probability
        else ""
    )
    prefix = weighted_choice(vocabulary.name_prefixes, rng)
    name = f"{prefix} {company_name}".strip() if company_name else prefix
    slogan = rng.choice(vocabulary.slogans) if rng.random() < slogan_probability else ""
    side = weighted_choice(layout.logo_sides, rng)
    height = round(layout.logo_mm.draw(rng), 1)
    # Drawn before the `category` test rather than after it, so a row with no
    # category advances the stream exactly as far as one that has it.
    wants_category = rng.random() < category_probability
    return HeaderSpec(
        enabled=enabled,
        company_name=name,
        slogan=slogan,
        opener=opener,
        logo_path=logo_path if enabled else None,
        logo_side=side,
        logo_height_mm=height,
        bar_style=bar_style,
        show_category=bool(category) and wants_category,
        category=category,
    )


def build_footer(
    contact_fields: Sequence[tuple[str, str, str]],
    bar_style: str,
    rng: random.Random,
    layout: LayoutConfig = _DEFAULT_LAYOUT,
    footer_probability: float = _DEFAULT_PROBABILITIES.footer,
    page_number_probability: float = _DEFAULT_PROBABILITIES.page_number,
    label_probability: float = _DEFAULT_PROBABILITIES.footer_labels,
    vocabulary: HeaderVocabulary = _DEFAULT_VOCABULARY,
) -> FooterSpec:
    """Assemble the footer for one document.

    Args:
        contact_fields: ``(key, label, value)`` from
            :attr:`~..letter_source.ContactInfo.labelled_fields`; only
            populated fields arrive here.
        bar_style: Footer bar treatment chosen by the theme.
        rng: The sample's seeded RNG, conventionally ``purpose="footer"``.
        layout: Field order and arrangement, from ``synth_config.yaml``.
        footer_probability: Chance the letter has a footer at all.
        page_number_probability: Chance a page number is printed.
        label_probability: Chance the fields are labelled rather than printed
            as bare values.
        vocabulary: The ``header`` pools from ``vocabulary_fa.yaml``, for the
            contact labels and the page-number wordings.

    Returns:
        The footer spec. A letter whose record carries no contact details at
        all still gets a page number if one was drawn, which is what a real
        continuation sheet looks like.
    """
    enabled = rng.random() < footer_probability
    available = {key: (label, value) for key, label, value in contact_fields}
    order = rng.choice(layout.contact_orders)
    show_labels = rng.random() < label_probability
    fields = tuple(
        (
            key,
            available[key][0] if show_labels else "",
            to_persian_digits(available[key][1]),
        )
        for key in order
        if key in available
    )
    # Label wording varies per sample, but only when labels are shown at all.
    if show_labels:
        fields = tuple(
            (key, rng.choice(vocabulary.labels_for(key, label)), value)
            for key, label, value in fields
        )
    template = (
        rng.choice(vocabulary.page_templates)
        if rng.random() < page_number_probability
        else ""
    )
    arrangement = weighted_choice(layout.footer_layouts, rng)
    return FooterSpec(
        enabled=enabled,
        fields=fields,
        layout=arrangement,
        bar_style=bar_style,
        page_number_template=template,
    )


def _logo_html(spec: HeaderSpec, bar_text: str) -> str:
    """Render the logo image, recoloured when it sits on a solid bar.

    Args:
        spec: The header being rendered.
        bar_text: Text colour of a solid bar, used as the recolour target.

    Returns:
        An ``<img>`` element, or an empty string when there is no logo.
    """
    if spec.logo_path is None:
        return ""
    colour = bar_text if spec.bar_style == "solid" else None
    uri = logo_data_uri(spec.logo_path, colour)
    if not uri:
        return ""
    return (
        f'<img class="header-logo" data-ann-type="logo" data-ann-id="header.logo" '
        f'src="{uri}" alt="" style="height:{spec.logo_height_mm}mm">'
    )


def header_to_html(spec: HeaderSpec, bar_text: str = "#FFFFFF") -> str:
    """Render the letterhead to annotated HTML.

    Args:
        spec: The header to render.
        bar_text: Text colour on a solid bar, for logo recolouring.

    Returns:
        An HTML fragment, empty when the sample has no header. The religious
        opener is *not* included: it belongs above the body, and the composer
        places it there via :func:`opener_to_html`.
    """
    if not spec.enabled:
        return ""
    classes = (
        f"letter-header header-{html.escape(spec.bar_style, quote=True)} "
        f"logo-{html.escape(spec.logo_side, quote=True)}"
    )
    parts: list[str] = [
        f'<header class="{classes}" data-ann-type="header_bar" '
        f'data-ann-id="header">'
    ]
    parts.append(_logo_html(spec, bar_text))
    parts.append('<div class="header-identity">')
    if spec.company_name:
        parts.append(
            f'<div class="header-company" data-ann-type="header_company_name" '
            f'data-ann-id="header.company">{html.escape(spec.company_name)}</div>'
        )
    if spec.show_category and spec.category:
        parts.append(
            f'<div class="header-category" data-ann-type="header_slogan" '
            f'data-ann-id="header.category">{html.escape(spec.category)}</div>'
        )
    if spec.slogan:
        parts.append(
            f'<div class="header-slogan" data-ann-type="header_slogan" '
            f'data-ann-id="header.slogan">{html.escape(spec.slogan)}</div>'
        )
    parts.append("</div></header>")
    return "".join(parts)


def opener_to_html(spec: HeaderSpec) -> str:
    """Render the religious opener, which sits above the letter body.

    Args:
        spec: The header carrying the opener text.

    Returns:
        An HTML fragment, empty when the sample has no opener.
    """
    if not spec.opener:
        return ""
    return (
        f'<div class="religious-opener" data-ann-type="religious_opener" '
        f'data-ann-id="opener">{html.escape(spec.opener)}</div>'
    )


def footer_to_html(spec: FooterSpec) -> str:
    """Render the footer to annotated HTML.

    The page-number node carries ``data-page-template`` for ``paginate.js`` to
    fill once the sheet count is known, and meanwhile renders the single-page
    string so the fragment is never blank.

    Args:
        spec: The footer to render.

    Returns:
        An HTML fragment, empty when the sample has no footer.
    """
    if not spec.enabled or (not spec.fields and not spec.has_page_number):
        return ""
    classes = (
        f"letter-footer footer-{html.escape(spec.bar_style, quote=True)} "
        f"footer-{html.escape(spec.layout, quote=True)}"
    )
    parts: list[str] = [
        f'<footer class="{classes}" data-ann-type="footer_bar" '
        f'data-ann-id="footer">'
    ]
    if spec.fields:
        parts.append('<div class="footer-fields">')
        for key, label, value in spec.fields:
            field_id = html.escape(f"footer.{key}", quote=True)
            safe_key = html.escape(key, quote=True)
            parts.append(
                f'<span class="footer-field" data-ann-type="footer_field" '
                f'data-ann-id="{field_id}" data-ann-key="{safe_key}">'
            )
            if label:
                parts.append(
                    f'<span class="footer-label">{html.escape(label)}:</span> '
                )
            parts.append(f"{html.escape(value)}</span>")
        parts.append("</div>")
    if spec.has_page_number:
        one_page = spec.page_number_template.format(
            page=to_persian_digits(1), total=to_persian_digits(1)
        )
        parts.append(
            f'<div class="page-number" data-ann-type="page_number" '
            f'data-ann-id="footer.page_number" '
            f'data-page-template="{html.escape(spec.page_number_template, quote=True)}">'
            f"{html.escape(one_page)}</div>"
        )
    parts.append("</footer>")
    return "".join(parts)


def header_to_markdown(spec: HeaderSpec) -> str:
    """Render the letterhead for the ``.md`` artifact.

    Args:
        spec: The header to render.

    Returns:
        The opener, company name and slogan as Markdown, empty when absent.
    """
    lines: list[str] = []
    if spec.opener:
        lines.extend((spec.opener, ""))
    if spec.enabled and spec.company_name:
        lines.append(f"# {spec.company_name}")
        if spec.show_category and spec.category:
            lines.append(f"*{spec.category}*")
        if spec.slogan:
            lines.append(f"*{spec.slogan}*")
    return "\n".join(lines)


def footer_to_markdown(spec: FooterSpec) -> str:
    """Render the footer for the ``.md`` artifact.

    Args:
        spec: The footer to render.

    Returns:
        A horizontal rule followed by the contact lines, empty when absent.
    """
    if not spec.enabled or not spec.fields:
        return ""
    lines = [
        f"**{label}:** {value}" if label else value for _, label, value in spec.fields
    ]
    return "---\n\n" + "  \n".join(lines)
