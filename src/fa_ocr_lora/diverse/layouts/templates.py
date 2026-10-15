"""HTML templates, one family per visual category.

This is where the corpus gets its visual variety, and it is written as code
rather than as Jinja files on purpose: a template here is a *function* of
(text, palette, fonts, RNG), so it can make structural decisions - two columns
or one, a sidebar or not, a table with or without zebra striping - that a
string template cannot.

Every template follows three rules that the rest of the pipeline depends on:

1. **Annotated regions carry ``data-ann``.** ``harvest.js`` finds text by
   querying ``[data-ann]`` and reads the type from ``data-ann``. A region
   without the attribute contributes no box, which is how decoration
   (rules, avatars, watermarks) stays out of the ground truth.
2. **Nothing may overflow its canvas.** A clipped line would leave the label
   claiming characters the image does not show. Templates size themselves to
   their content, and image QC rejects anything that escapes anyway.
3. **No CSS transform may move an annotated node.** The harvest reads
   post-layout rectangles, but a transform applied after layout is not
   reflected in them. Rotation and skew belong to the degradation stage, which
   maps the boxes through the same matrix as the pixels.

No real brand, logo or app name appears anywhere. The messenger, news-site and
shop styling is invented, which keeps the corpus publishable and stops the
model keying on a real company's letterform.
"""

from __future__ import annotations

import html
import random
from collections.abc import Callable
from typing import Final

from ..text_policy import to_digit_script
from .spec import Palette

__all__ = [
    "TEMPLATE_IDS",
    "build_document",
    "escape",
]


def escape(text: str) -> str:
    """Escape text for HTML, preserving Persian characters and ZWNJ.

    Args:
        text: Raw text.

    Returns:
        The escaped text. Only ``&``, ``<``, ``>`` and quotes are touched;
        ZWNJ and the Persian block pass through unchanged, because they are
        content and must reach the renderer byte-for-byte.
    """
    return html.escape(text, quote=True)


def _ann(kind: str) -> str:
    """Return the annotation attribute for a block type.

    Args:
        kind: A :class:`~fa_ocr_lora.diverse.render.schema.BlockType` value.

    Returns:
        The attribute text to splice into a tag.
    """
    return f' data-ann="{kind}"'


#: Persian UI chrome for invented apps and sites. None names a real product.
_CHAT_NAMES: Final[tuple[str, ...]] = (
    "مهتاب",
    "آرش",
    "نگار",
    "سهیل",
    "پریسا",
    "بهرام",
    "شیرین",
    "کاوه",
    "رویا",
    "فرهاد",
    "ترمه",
    "بیژن",
)
_CHAT_APPS: Final[tuple[str, ...]] = (
    "گپ‌رسان",
    "پیام‌نگار",
    "هم‌کلام",
    "نداپیام",
    "رسانک",
)
_SITE_NAMES: Final[tuple[str, ...]] = (
    "خبرنامه امروز",
    "دیدگاه نو",
    "پایگاه دانش",
    "روزنگار",
    "رسانه آزاد",
    "نشان",
)
_NAV_ITEMS: Final[tuple[str, ...]] = (
    "خانه",
    "اخبار",
    "درباره ما",
    "تماس",
    "آرشیو",
    "پرسش‌ها",
    "فروشگاه",
    "راهنما",
)
_UI_ACTIONS: Final[tuple[str, ...]] = (
    "ذخیره",
    "انصراف",
    "ارسال",
    "ویرایش",
    "حذف",
    "بازگشت",
    "تأیید",
    "جست‌وجو",
)

#: Template identifiers. ``test_unseen_templates`` holds out whole templates,
#: so this list is the closed set the split plan draws from - a template that
#: renders but is absent here could never be held out, and the hard test would
#: quietly measure less than it claims.
TEMPLATE_IDS: Final[dict[str, tuple[str, ...]]] = {
    "page": (
        "page_prose",
        "page_two_column",
        "page_headed_article",
        "page_textbook",
        "page_report",
        "page_newspaper",
    ),
    "paragraph": ("para_plain", "para_boxed", "para_caption", "para_pullquote"),
    "line": ("line_title", "line_label", "line_ui", "line_subtitle"),
    "poetry": ("poem_two_hemistich", "poem_centred", "poem_free", "poem_framed"),
    "web": ("web_article", "web_blog", "web_forum", "web_product", "web_faq"),
    "chat": ("chat_light", "chat_dark", "chat_sms", "chat_group"),
    "banner": ("banner_poster", "banner_social", "banner_sign", "banner_flyer"),
    "story": ("story_chapter", "story_dropcap", "story_plain"),
    "table_form": ("table_grid", "table_invoice", "table_form_fields", "table_timetable"),
    "mixed": ("mixed_technical", "mixed_ui", "mixed_label"),
}


def _base_css(
    palette: Palette, font_css: str, family: str, *, size_px: float, line_height: float
) -> str:
    """Build the stylesheet shared by every template.

    Args:
        palette: The sample's colours.
        font_css: ``@font-face`` rules with the fonts embedded.
        family: Primary CSS family name.
        size_px: Body font size in CSS pixels.
        line_height: Unitless line height.

    Returns:
        The CSS text.
    """
    return f"""{font_css}
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{background:{palette.paper};color:{palette.ink}}}
body{{font-family:'{family}',sans-serif;font-size:{size_px:.1f}px;
line-height:{line_height:.2f};-webkit-font-smoothing:antialiased}}
p{{margin:0 0 0.75em 0}}
h1,h2,h3{{line-height:1.25;margin:0 0 0.5em 0;color:{palette.accent}}}
ul,ol{{margin:0 0 0.75em 1.4em}}
li{{margin:0 0 0.25em 0}}
table{{border-collapse:collapse;width:100%}}
.muted{{color:{palette.muted}}}
.accent{{color:{palette.accent}}}
"""


def _wrap(
    body: str,
    css: str,
    *,
    width: int,
    height: int | None = None,
    pad: int = 0,
    direction: str = "rtl",
) -> str:
    """Assemble a complete HTML document.

    Args:
        body: The document body markup.
        css: The stylesheet.
        width: Canvas width in CSS pixels.
        height: Canvas height, or ``None`` to fit the content.
        pad: Padding inside the canvas, in CSS pixels.
        direction: ``rtl`` or ``ltr``.

    Returns:
        The document.
    """
    sizing = f"width:{width}px;" + (f"height:{height}px;" if height else "")
    return f"""<!DOCTYPE html>
<html dir="{direction}" lang="fa"><head><meta charset="utf-8">
<style>{css}
#canvas{{{sizing}padding:{pad}px;overflow:hidden;position:relative}}</style>
</head><body><div id="canvas">{body}</div></body></html>"""


def _paragraphs(text: str) -> list[str]:
    """Split a text into paragraphs on blank lines.

    Args:
        text: The sample's ``text_render``.

    Returns:
        Non-empty paragraph strings.
    """
    blocks = [block.strip() for block in text.split("\n\n")]
    return [block for block in blocks if block]


def _lines(text: str) -> list[str]:
    """Split a text into non-empty lines.

    Args:
        text: The sample's ``text_render``.

    Returns:
        Non-empty line strings.
    """
    return [line.strip() for line in text.splitlines() if line.strip()]


def _is_table(text: str) -> bool:
    """Report whether a text looks like pipe-delimited table rows.

    Args:
        text: The sample's ``text_render``.

    Returns:
        ``True`` when at least two lines carry a pipe.
    """
    return sum(1 for line in _lines(text) if "|" in line) >= 2


def _is_dialogue(text: str) -> bool:
    """Report whether a text looks like ``name: message`` turns.

    Args:
        text: The sample's ``text_render``.

    Returns:
        ``True`` when at least half the lines carry an early colon.
    """
    lines = _lines(text)
    if len(lines) < 2:
        return False
    turns = sum(1 for line in lines if ":" in line[:24] or "：" in line[:24])
    return turns >= max(2, len(lines) // 2)


def _render_paragraphs(text: str, kind: str = "paragraph") -> str:
    """Render text as annotated paragraphs, honouring list markers.

    A line beginning with a bullet or a number is emitted as a list item
    rather than a paragraph, so the block types in the annotation match what
    the page actually shows.

    Args:
        text: The sample's ``text_render``.
        kind: Block type for ordinary paragraphs.

    Returns:
        The markup.
    """
    out: list[str] = []
    for block in _paragraphs(text):
        lines = _lines(block)
        bullets = [
            line
            for line in lines
            if line[:2] in {"- ", "* ", "• "}
            or (line[:3].rstrip(".)").isdigit() and line[1:3] in {". ", ") ", "- "})
        ]
        if bullets and len(bullets) >= max(2, len(lines) - 1):
            items = "".join(
                f"<li{_ann('list_item')}>{escape(line.lstrip('-*• ').lstrip())}</li>"
                for line in lines
            )
            out.append(f"<ul>{items}</ul>")
        else:
            out.append(f"<p{_ann(kind)}>{escape(block)}</p>")
    return "".join(out)


def _table_markup(text: str, palette: Palette, rng: random.Random) -> str:
    """Render pipe-delimited rows as an HTML table.

    Args:
        text: The sample's ``text_render``.
        palette: The sample's colours.
        rng: The sample's seeded RNG.

    Returns:
        The table markup, with the first row as headers.
    """
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in _lines(text)
        if "|" in line
    ]
    if not rows:
        return _render_paragraphs(text)
    zebra = rng.random() < 0.55
    bordered = rng.random() < 0.75
    border = f"1px solid {palette.muted}" if bordered else "none"
    out = [f'<table style="border:{border}">']
    header, *body = rows
    out.append("<thead><tr>")
    for cell in header:
        out.append(
            f'<th{_ann("table_header")} style="border:{border};padding:6px 8px;'
            f'background:{palette.panel};text-align:right;font-weight:700">{escape(cell)}</th>'
        )
    out.append("</tr></thead><tbody>")
    for index, row in enumerate(body):
        shade = palette.panel if (zebra and index % 2) else "transparent"
        out.append(f'<tr style="background:{shade}">')
        for cell in row:
            out.append(
                f'<td{_ann("table_cell")} style="border:{border};padding:5px 8px;'
                f'text-align:right">{escape(cell)}</td>'
            )
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def build_document(
    template_id: str,
    text: str,
    palette: Palette,
    *,
    font_css: str,
    family: str,
    display_family: str,
    rng: random.Random,
    digit_script: str,
) -> tuple[str, int, int]:
    """Build the HTML document for one template.

    Args:
        template_id: One of the ids in :data:`TEMPLATE_IDS`.
        text: The sample's ``text_render``.
        palette: The sample's colours.
        font_css: ``@font-face`` rules with the fonts embedded.
        family: Primary body family name.
        display_family: Family for headings and display text.
        rng: The sample's seeded RNG.
        digit_script: Digit script for any chrome this template invents, so a
            timestamp or a page number matches the sample's labelled script.

    Returns:
        ``(html, width, height)``. A ``height`` of ``0`` means the renderer
        should size the canvas to the content.

    Raises:
        KeyError: If the template id is unknown.
    """
    builder = _BUILDERS.get(template_id)
    if builder is None:
        raise KeyError(f"unknown template {template_id!r}")
    return builder(text, palette, font_css, family, display_family, rng, digit_script)


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #


def _fit_font_px(
    text: str,
    width: int,
    height: int,
    margin: int,
    line_height: float,
    *,
    low: float,
    high: float,
    columns: int = 1,
) -> tuple[float, bool]:
    """Choose a body size that fits the text in the canvas, or give up on it.

    A fixed-height page with more text than it can hold clips the overflow.
    The pixels then show less than `text_render` claims, which image QC
    correctly rejects - one story page came back with 9% of its label visible.
    Guessing a font size per template and hoping was the cause.

    The estimate is crude on purpose. Persian at size `s` averages about
    `0.52 * s` per character of advance, so a column of width `w` holds
    `w / (0.52 * s)` characters per line, and `n` characters need
    `n * 0.52 * s / w` lines of `s * line_height` each. Setting that equal to
    the available height and solving for `s` gives the largest size that can
    fit. It is approximate, so the result is scaled down slightly and the
    caller still lets image QC have the final say.

    Args:
        text: The text to be laid out.
        width: Canvas width in CSS pixels.
        height: Canvas height in CSS pixels.
        margin: Padding inside the canvas.
        line_height: Unitless line height.
        low: Smallest acceptable body size.
        high: Largest body size, i.e. the size wanted when the text is short.
        columns: Text columns, which multiply the usable width.

    Returns:
        ``(font_px, fits)``. When ``fits`` is ``False`` even ``low`` is too
        large, and the caller should let the canvas grow to the content rather
        than clip it.
    """
    usable_width = max(40, (width - 2 * margin)) * max(1, columns)
    usable_height = max(40, height - 2 * margin)
    characters = max(1, len(text))
    # Allow for paragraph breaks, each of which wastes part of a line.
    characters += 60 * text.count("\n\n")
    ideal = ((usable_width * usable_height) / (0.52 * line_height * characters)) ** 0.5
    ideal *= 0.94  # the estimate ignores widows and heading sizes
    if ideal >= high:
        return high, True
    if ideal <= low:
        return low, False
    return ideal, True


def _page_size(rng: random.Random) -> tuple[int, int, int]:
    """Choose a paper size in CSS pixels at 96 dpi.

    Args:
        rng: The sample's seeded RNG.

    Returns:
        ``(width, height, margin)``.
    """
    from ...dataset.common.io_utils import weighted_choice

    name = weighted_choice(
        (
            ("a4", 0.58),
            ("letter", 0.16),
            ("a5", 0.12),
            ("a4_landscape", 0.08),
            ("b5", 0.06),
        ),
        rng,
    )
    sizes = {
        "a4": (794, 1123),
        "letter": (816, 1056),
        "a5": (559, 794),
        "a4_landscape": (1123, 794),
        "b5": (665, 944),
    }
    width, height = sizes[name]
    margin = rng.randint(42, 86)
    return width, height, margin


def _page_prose(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """Plain prose on a page, the commonest printed form."""
    width, height, margin = _page_size(rng)
    line_height = rng.uniform(1.6, 2.0)
    size_px, fits = _fit_font_px(
        text,
        width,
        height,
        margin,
        line_height,
        low=14.5,
        high=18.5,
        columns=1,
    )
    if not fits:
        # More text than this sheet can hold. Let the canvas grow
        # rather than clip: a tall page is honest, a clipped label is not.
        height = 0
    css = _base_css(palette, font_css, family, size_px=size_px, line_height=line_height)
    justify = "justify" if rng.random() < 0.6 else "right"
    body = f'<div style="text-align:{justify}">{_render_paragraphs(text)}</div>'
    return _wrap(body, css, width=width, height=height, pad=margin), width, height


def _page_two_column(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """Two columns, read right to left."""
    width, height, margin = _page_size(rng)
    gap = rng.randint(22, 40)
    css = _base_css(
        palette,
        font_css,
        family,
        size_px=rng.uniform(13.0, 16.0),
        line_height=rng.uniform(1.55, 1.85),
    )
    css += f"#flow{{column-count:2;column-gap:{gap}px;column-fill:auto;height:100%}}"
    if rng.random() < 0.5:
        css += f"#flow{{column-rule:1px solid {palette.muted}}}"
    body = f'<div id="flow" style="text-align:justify">{_render_paragraphs(text)}</div>'
    return _wrap(body, css, width=width, height=height, pad=margin), width, height


def _page_headed_article(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A headline, a rule, then prose."""
    width, height, margin = _page_size(rng)
    line_height = rng.uniform(1.6, 1.9)
    size_px, fits = _fit_font_px(
        text,
        width,
        height,
        margin,
        line_height,
        low=13.0,
        high=16.0,
        columns=2,
    )
    if not fits:
        # More text than this sheet can hold. Let the canvas grow
        # rather than clip: a tall page is honest, a clipped label is not.
        height = 0
    css = _base_css(palette, font_css, family, size_px=size_px, line_height=line_height)
    paragraphs = _paragraphs(text)
    heading = paragraphs[0][:70] if paragraphs else ""
    rest = "\n\n".join(paragraphs[1:]) if len(paragraphs) > 1 else ""
    rule = (
        f'<div style="height:2px;background:{palette.accent};margin:0 0 18px 0"></div>'
        if rng.random() < 0.7
        else ""
    )
    body = (
        f'<h1{_ann("heading")} style="font-family:\'{display}\';font-size:1.9em">{escape(heading)}</h1>'
        f"{rule}{_render_paragraphs(rest) if rest else ''}"
    )
    return _wrap(body, css, width=width, height=height, pad=margin), width, height


def _page_textbook(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A textbook page: heading, body, a sidebar and a footnote."""
    width, height, margin = _page_size(rng)
    line_height = rng.uniform(1.6, 1.9)
    size_px, fits = _fit_font_px(
        text,
        width,
        height,
        margin,
        line_height,
        low=14.0,
        high=17.0,
        columns=1,
    )
    if not fits:
        # More text than this sheet can hold. Let the canvas grow
        # rather than clip: a tall page is honest, a clipped label is not.
        height = 0
    css = _base_css(palette, font_css, family, size_px=size_px, line_height=line_height)
    paragraphs = _paragraphs(text)
    heading = paragraphs[0][:60] if paragraphs else ""
    rest = "\n\n".join(paragraphs[1:])
    aside = ""
    if len(paragraphs) > 2 and rng.random() < 0.6:
        aside = (
            f'<aside{_ann("quote")} style="float:left;width:34%;margin:0 0 12px 16px;'
            f"padding:12px;background:{palette.panel};border-right:3px solid {palette.accent};"
            f'font-size:0.92em">{escape(paragraphs[-1][:240])}</aside>'
        )
    footnote = (
        f'<div{_ann("footnote")} style="margin-top:18px;padding-top:8px;'
        f'border-top:1px solid {palette.muted};font-size:0.82em;color:{palette.muted}">'
        f'{escape(to_digit_script("1. ", digits))}{escape(paragraphs[-1][:120] if paragraphs else "")}</div>'
        if rng.random() < 0.5
        else ""
    )
    body = (
        f'<h2{_ann("heading")} style="font-family:\'{display}\';font-size:1.5em">{escape(heading)}</h2>'
        f"{aside}{_render_paragraphs(rest)}{footnote}"
    )
    return _wrap(body, css, width=width, height=height, pad=margin), width, height


def _page_report(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A report page, with a table when the text supplies one."""
    width, height, margin = _page_size(rng)
    line_height = rng.uniform(1.5, 1.8)
    size_px, fits = _fit_font_px(
        text,
        width,
        height,
        margin,
        line_height,
        low=13.5,
        high=16.5,
        columns=1,
    )
    if not fits:
        # More text than this sheet can hold. Let the canvas grow
        # rather than clip: a tall page is honest, a clipped label is not.
        height = 0
    css = _base_css(palette, font_css, family, size_px=size_px, line_height=line_height)
    inner = (
        _table_markup(text, palette, rng) if _is_table(text) else _render_paragraphs(text)
    )
    page_no = (
        f'<div{_ann("page_number")} style="position:absolute;bottom:{margin // 2}px;'
        f'left:50%;transform:none;color:{palette.muted};font-size:0.85em">'
        f"{escape(to_digit_script(str(rng.randint(1, 48)), digits))}</div>"
        if rng.random() < 0.6
        else ""
    )
    return (
        _wrap(inner + page_no, css, width=width, height=height, pad=margin),
        width,
        height,
    )


def _page_newspaper(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A dense three-column newspaper page with a masthead."""
    width, height, margin = _page_size(rng)
    line_height = rng.uniform(1.4, 1.65)
    size_px, fits = _fit_font_px(
        text,
        width,
        height,
        margin,
        line_height,
        low=13.0,
        high=16.0,
        columns=1,
    )
    if not fits:
        # More text than this sheet can hold. Let the canvas grow
        # rather than clip: a tall page is honest, a clipped label is not.
        height = 0
    css = _base_css(palette, font_css, family, size_px=size_px, line_height=line_height)
    columns = rng.choice([2, 3])
    css += f"#flow{{column-count:{columns};column-gap:{rng.randint(16, 28)}px;column-fill:auto}}"
    masthead = (
        f'<div{_ann("header")} style="font-family:\'{display}\';font-size:2.1em;'
        f"text-align:center;border-bottom:3px double {palette.ink};padding-bottom:8px;"
        f'margin-bottom:14px">{escape(rng.choice(_SITE_NAMES))}</div>'
    )
    body = (
        masthead
        + f'<div id="flow" style="text-align:justify">{_render_paragraphs(text)}</div>'
    )
    return _wrap(body, css, width=width, height=height, pad=margin), width, height


# --------------------------------------------------------------------------- #
# paragraph / line
# --------------------------------------------------------------------------- #


def _para_plain(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A single paragraph block on a plain ground."""
    width = rng.randint(420, 900)
    pad = rng.randint(18, 48)
    css = _base_css(
        palette,
        font_css,
        family,
        size_px=rng.uniform(15.0, 21.0),
        line_height=rng.uniform(1.6, 2.0),
    )
    body = f'<div style="text-align:{"justify" if rng.random() < 0.5 else "right"}">{_render_paragraphs(text)}</div>'
    return _wrap(body, css, width=width, pad=pad), width, 0


def _para_boxed(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A paragraph inside a bordered or tinted box."""
    width = rng.randint(400, 820)
    pad = rng.randint(14, 32)
    css = _base_css(
        palette,
        font_css,
        family,
        size_px=rng.uniform(14.5, 19.0),
        line_height=rng.uniform(1.6, 1.95),
    )
    radius = rng.choice([0, 4, 8, 14])
    style = (
        f"border:{rng.choice([1, 1, 2])}px solid {palette.muted};border-radius:{radius}px;"
        f"padding:{rng.randint(12, 26)}px;background:{palette.panel if rng.random() < 0.5 else 'transparent'}"
    )
    body = f'<div style="{style}">{_render_paragraphs(text)}</div>'
    return _wrap(body, css, width=width, pad=pad), width, 0


def _para_caption(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A caption under a placeholder figure."""
    width = rng.randint(380, 720)
    css = _base_css(
        palette,
        font_css,
        family,
        size_px=rng.uniform(12.5, 16.0),
        line_height=rng.uniform(1.45, 1.7),
    )
    figure = (
        f'<div style="height:{rng.randint(90, 200)}px;background:{palette.panel};'
        f'border:1px solid {palette.muted};margin-bottom:10px"></div>'
    )
    body = (
        figure
        + f'<div{_ann("caption")} style="color:{palette.muted};text-align:center">{escape(" ".join(_lines(text)))}</div>'
    )
    return _wrap(body, css, width=width, pad=rng.randint(14, 30)), width, 0


def _para_pullquote(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A large pull quote with guillemets."""
    width = rng.randint(420, 780)
    css = _base_css(
        palette,
        font_css,
        family,
        size_px=rng.uniform(19.0, 27.0),
        line_height=rng.uniform(1.45, 1.7),
    )
    body = (
        f'<blockquote{_ann("quote")} style="font-family:\'{display}\';'
        f'border-right:4px solid {palette.accent};padding-right:16px;color:{palette.ink}">'
        f'«{escape(" ".join(_lines(text)))}»</blockquote>'
    )
    return _wrap(body, css, width=width, pad=rng.randint(20, 44)), width, 0


def _line_title(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A single large title line."""
    content = _lines(text)[0] if _lines(text) else text
    width = rng.randint(380, 1000)
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(26.0, 52.0), line_height=1.3
    )
    body = f'<div{_ann("heading")} style="font-family:\'{display}\';white-space:nowrap">{escape(content)}</div>'
    return _wrap(body, css, width=width, pad=rng.randint(12, 34)), width, 0


def _line_label(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A small label, as on a form or a product."""
    content = _lines(text)[0] if _lines(text) else text
    width = rng.randint(220, 620)
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(13.0, 22.0), line_height=1.4
    )
    body = f'<div{_ann("field_label")} style="white-space:nowrap">{escape(content)}</div>'
    return _wrap(body, css, width=width, pad=rng.randint(8, 22)), width, 0


def _line_ui(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A UI string, optionally inside a button."""
    content = _lines(text)[0] if _lines(text) else text
    width = rng.randint(200, 560)
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(13.0, 19.0), line_height=1.5
    )
    if rng.random() < 0.5:
        body = (
            f'<span{_ann("ui_string")} style="display:inline-block;padding:8px 18px;'
            f"background:{palette.accent};color:{palette.paper};border-radius:{rng.choice([3, 6, 20])}px;"
            f'white-space:nowrap">{escape(content)}</span>'
        )
    else:
        body = (
            f'<div{_ann("ui_string")} style="white-space:nowrap">{escape(content)}</div>'
        )
    return _wrap(body, css, width=width, pad=rng.randint(10, 26)), width, 0


def _line_subtitle(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A subtitle line on a dark band, as in video subtitles."""
    content = _lines(text)[0] if _lines(text) else text
    width = rng.randint(520, 1020)
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(20.0, 32.0), line_height=1.4
    )
    css += "html,body{background:#101010}"
    body = (
        f'<div{_ann("ui_string")} style="color:#f4f4f4;text-align:center;'
        f'text-shadow:0 2px 4px rgba(0,0,0,0.9)">{escape(content)}</div>'
    )
    return _wrap(body, css, width=width, pad=rng.randint(16, 34)), width, 0


# --------------------------------------------------------------------------- #
# poetry
# --------------------------------------------------------------------------- #


def _poem_two_hemistich(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """Classical two-hemistich layout: two columns per verse."""
    lines = _lines(text)
    width = rng.randint(620, 980)
    css = _base_css(
        palette,
        font_css,
        family,
        size_px=rng.uniform(17.0, 25.0),
        line_height=rng.uniform(1.9, 2.5),
    )
    rows: list[str] = []
    for index in range(0, len(lines), 2):
        left = lines[index]
        right = lines[index + 1] if index + 1 < len(lines) else ""
        rows.append(
            f'<div style="display:flex;gap:{rng.randint(28, 70)}px;justify-content:space-between">'
            f'<span{_ann("verse")} style="flex:1;text-align:right">{escape(left)}</span>'
            f'<span{_ann("verse")} style="flex:1;text-align:left">{escape(right)}</span></div>'
        )
    body = f'<div style="font-family:\'{display}\'">{"".join(rows)}</div>'
    return _wrap(body, css, width=width, pad=rng.randint(28, 64)), width, 0


def _poem_centred(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """Centred masnavi-style verses."""
    width = rng.randint(480, 840)
    css = _base_css(
        palette,
        font_css,
        family,
        size_px=rng.uniform(18.0, 26.0),
        line_height=rng.uniform(1.9, 2.4),
    )
    body = "".join(
        f'<div{_ann("verse")} style="text-align:center;font-family:\'{display}\'">{escape(line)}</div>'
        for line in _lines(text)
    )
    return _wrap(body, css, width=width, pad=rng.randint(26, 58)), width, 0


def _poem_free(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """Free verse with ragged indentation."""
    width = rng.randint(440, 820)
    css = _base_css(
        palette,
        font_css,
        family,
        size_px=rng.uniform(16.0, 23.0),
        line_height=rng.uniform(1.8, 2.3),
    )
    body = "".join(
        f'<div{_ann("verse")} style="padding-right:{rng.choice([0, 0, 18, 36, 54])}px">{escape(line)}</div>'
        for line in _lines(text)
    )
    return _wrap(body, css, width=width, pad=rng.randint(24, 52)), width, 0


def _poem_framed(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """Verses inside an ornamental double frame."""
    width = rng.randint(520, 860)
    css = _base_css(
        palette,
        font_css,
        family,
        size_px=rng.uniform(17.0, 24.0),
        line_height=rng.uniform(1.9, 2.4),
    )
    inner = "".join(
        f'<div{_ann("verse")} style="text-align:center">{escape(line)}</div>'
        for line in _lines(text)
    )
    body = (
        f'<div style="border:2px solid {palette.accent};padding:6px">'
        f'<div style="border:1px solid {palette.muted};padding:{rng.randint(18, 38)}px;'
        f"font-family:'{display}'\">{inner}</div></div>"
    )
    return _wrap(body, css, width=width, pad=rng.randint(20, 44)), width, 0


# --------------------------------------------------------------------------- #
# web
# --------------------------------------------------------------------------- #


def _web_chrome(palette: Palette, rng: random.Random, digits: str) -> str:
    """Build an invented site header with navigation.

    Args:
        palette: The sample's colours.
        rng: The sample's seeded RNG.
        digits: Digit script for any numerals.

    Returns:
        The header markup.
    """
    nav = "".join(
        f'<span{_ann("link")} style="margin-left:18px;color:{palette.muted}">{escape(item)}</span>'
        for item in rng.sample(_NAV_ITEMS, rng.randint(3, 5))
    )
    return (
        f'<div{_ann("header")} style="display:flex;justify-content:space-between;'
        f"align-items:center;padding-bottom:12px;border-bottom:1px solid {palette.muted};"
        f'margin-bottom:18px"><strong style="color:{palette.accent};font-size:1.25em">'
        f"{escape(rng.choice(_SITE_NAMES))}</strong><span>{nav}</span></div>"
    )


def _web_article(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A web article with chrome and a byline."""
    width = rng.randint(760, 1180)
    css = _base_css(
        palette,
        font_css,
        family,
        size_px=rng.uniform(15.0, 18.5),
        line_height=rng.uniform(1.65, 1.95),
    )
    paragraphs = _paragraphs(text)
    heading = paragraphs[0][:80] if paragraphs else ""
    rest = "\n\n".join(paragraphs[1:])
    byline = (
        f'<div{_ann("chat_meta")} style="color:{palette.muted};font-size:0.86em;margin-bottom:14px">'
        f'{escape(to_digit_script(f"{rng.randint(1,30)} مرداد ۱۴۰۳", digits))}'
        f" · {escape(rng.choice(_CHAT_NAMES))}</div>"
    )
    body = (
        _web_chrome(palette, rng, digits)
        + f'<h1{_ann("heading")} style="font-family:\'{display}\';font-size:1.8em">{escape(heading)}</h1>'
        + byline
        + _render_paragraphs(rest)
    )
    return _wrap(body, css, width=width, pad=rng.randint(26, 56)), width, 0


def _web_blog(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A narrow blog post column."""
    width = rng.randint(620, 880)
    css = _base_css(
        palette,
        font_css,
        family,
        size_px=rng.uniform(16.0, 19.5),
        line_height=rng.uniform(1.7, 2.0),
    )
    body = _render_paragraphs(text)
    return _wrap(body, css, width=width, pad=rng.randint(30, 64)), width, 0


def _web_forum(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A forum thread: stacked posts with authors and timestamps."""
    width = rng.randint(720, 1060)
    css = _base_css(
        palette,
        font_css,
        family,
        size_px=rng.uniform(14.0, 17.0),
        line_height=rng.uniform(1.6, 1.85),
    )
    posts: list[str] = []
    for block in _paragraphs(text):
        name = rng.choice(_CHAT_NAMES)
        stamp = to_digit_script(f"{rng.randint(1, 23)}:{rng.randint(10, 59)}", digits)
        posts.append(
            f'<div style="display:flex;gap:12px;padding:14px 0;border-bottom:1px solid {palette.panel}">'
            f'<div style="width:40px;height:40px;border-radius:50%;background:{palette.panel};flex:none"></div>'
            f'<div style="flex:1"><div{_ann("chat_meta")} style="font-size:0.85em;color:{palette.muted}">'
            f"{escape(name)} · {escape(stamp)}</div>"
            f'<div{_ann("chat_message")}>{escape(block)}</div></div></div>'
        )
    return _wrap("".join(posts), css, width=width, pad=rng.randint(20, 44)), width, 0


def _web_product(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A product page: image placeholder, title, price, description."""
    width = rng.randint(760, 1120)
    css = _base_css(
        palette,
        font_css,
        family,
        size_px=rng.uniform(14.5, 17.5),
        line_height=rng.uniform(1.6, 1.9),
    )
    paragraphs = _paragraphs(text)
    title = paragraphs[0][:60] if paragraphs else ""
    price = to_digit_script(f"{rng.randint(90, 980)},000 تومان", digits)
    body = (
        _web_chrome(palette, rng, digits)
        + '<div style="display:flex;gap:24px">'
        + f'<div style="width:38%;height:{rng.randint(200, 300)}px;background:{palette.panel};'
        f'border:1px solid {palette.muted};flex:none"></div>'
        + f'<div style="flex:1"><h2{_ann("heading")} style="font-family:\'{display}\'">{escape(title)}</h2>'
        + f'<div{_ann("field_value")} style="color:{palette.accent};font-size:1.4em;font-weight:700;'
        f'margin:8px 0 14px">{escape(price)}</div>'
        + _render_paragraphs("\n\n".join(paragraphs[1:]))
        + "</div></div>"
    )
    return _wrap(body, css, width=width, pad=rng.randint(24, 50)), width, 0


def _web_faq(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A FAQ list of question-and-answer pairs."""
    width = rng.randint(680, 1000)
    css = _base_css(
        palette,
        font_css,
        family,
        size_px=rng.uniform(14.5, 17.5),
        line_height=rng.uniform(1.6, 1.9),
    )
    blocks = _paragraphs(text)
    items: list[str] = []
    for block in blocks:
        lines = _lines(block)
        question = lines[0] if lines else ""
        answer = " ".join(lines[1:]) if len(lines) > 1 else ""
        items.append(
            f'<div style="padding:14px 0;border-bottom:1px solid {palette.panel}">'
            f'<div{_ann("heading")} style="font-weight:700;color:{palette.accent}">{escape(question)}</div>'
            + (
                f'<div{_ann("paragraph")} style="margin-top:6px">{escape(answer)}</div>'
                if answer
                else ""
            )
            + "</div>"
        )
    return _wrap("".join(items), css, width=width, pad=rng.randint(22, 48)), width, 0


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #


def _chat_bubbles(
    text: str, palette: Palette, rng: random.Random, digits: str, *, dark: bool
) -> str:
    """Build messenger bubbles from dialogue turns.

    Args:
        text: The sample's ``text_render``.
        palette: The sample's colours.
        rng: The sample's seeded RNG.
        digits: Digit script for the timestamps.
        dark: Whether this is a dark theme.

    Returns:
        The bubble markup.
    """
    lines = _lines(text)
    mine_colour = palette.accent
    theirs = palette.panel
    out: list[str] = []
    for index, line in enumerate(lines):
        if ":" in line[:24]:
            speaker, _, message = line.partition(":")
        else:
            speaker, message = "", line
        message = message.strip() or line
        mine = index % 2 == 1
        align = "flex-end" if mine else "flex-start"
        background = mine_colour if mine else theirs
        colour = palette.paper if mine else palette.ink
        stamp = to_digit_script(
            f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}", digits
        )
        name = (
            f'<div{_ann("chat_meta")} style="font-size:0.74em;opacity:0.75;'
            f'margin-bottom:2px">{escape(speaker.strip())}:</div>'
            # Always drawn when the text supplies a speaker. Showing it only on
            # incoming bubbles left the name in `text_render` but not on the
            # page, so the label claimed words no box covered - image QC caught
            # ten of these on the first run.
            if speaker.strip()
            else ""
        )
        out.append(
            f'<div style="display:flex;justify-content:{align};margin-bottom:8px">'
            f'<div style="max-width:72%;background:{background};color:{colour};'
            f'padding:8px 12px;border-radius:14px">{name}'
            f'<div{_ann("chat_message")}>{escape(message)}</div>'
            f'<div{_ann("chat_meta")} style="font-size:0.7em;opacity:0.7;text-align:left">'
            f"{escape(stamp)}</div></div></div>"
        )
    return "".join(out)


def _chat_light(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A light-theme messenger screenshot."""
    width = rng.randint(360, 520)
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(14.0, 17.0), line_height=1.55
    )
    header = (
        f'<div{_ann("header")} style="padding:10px 14px;background:{palette.accent};'
        f'color:{palette.paper};font-weight:700">{escape(rng.choice(_CHAT_APPS))}'
        f" · {escape(rng.choice(_CHAT_NAMES))}</div>"
    )
    body = (
        header
        + f'<div style="padding:12px">{_chat_bubbles(text, palette, rng, digits, dark=False)}</div>'
    )
    return _wrap(body, css, width=width, pad=0), width, 0


def _chat_dark(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A dark-theme messenger screenshot."""
    width = rng.randint(360, 520)
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(14.0, 17.0), line_height=1.55
    )
    header = (
        f'<div{_ann("header")} style="padding:10px 14px;background:{palette.panel};'
        f'color:{palette.ink};font-weight:700">{escape(rng.choice(_CHAT_NAMES))}</div>'
    )
    body = (
        header
        + f'<div style="padding:12px">{_chat_bubbles(text, palette, rng, digits, dark=True)}</div>'
    )
    return _wrap(body, css, width=width, pad=0), width, 0


def _chat_sms(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """An SMS conversation, plainer than a messenger."""
    width = rng.randint(340, 480)
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(14.5, 17.5), line_height=1.5
    )
    body = f'<div style="padding:14px">{_chat_bubbles(text, palette, rng, digits, dark=False)}</div>'
    return _wrap(body, css, width=width, pad=0), width, 0


def _chat_group(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A group chat, with a member count in the header."""
    width = rng.randint(380, 560)
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(13.5, 16.5), line_height=1.55
    )
    count = to_digit_script(str(rng.randint(3, 240)), digits)
    header = (
        f'<div{_ann("header")} style="padding:10px 14px;background:{palette.panel}">'
        f'<div style="font-weight:700">{escape(rng.choice(["گروه خانوادگی", "هم‌کلاسی‌ها", "تیم پروژه", "همسایه‌ها"]))}</div>'
        f'<div{_ann("chat_meta")} style="font-size:0.76em;color:{palette.muted}">'
        f"{escape(count)} عضو</div></div>"
    )
    body = (
        header
        + f'<div style="padding:12px">{_chat_bubbles(text, palette, rng, digits, dark=False)}</div>'
    )
    return _wrap(body, css, width=width, pad=0), width, 0


# --------------------------------------------------------------------------- #
# banner
# --------------------------------------------------------------------------- #


def _banner_poster(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A poster: huge centred display text."""
    width = rng.randint(600, 1000)
    height = rng.randint(600, 1100)
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(34.0, 64.0), line_height=1.25
    )
    css += f"html,body{{background:{palette.accent};color:{palette.paper}}}"
    body = (
        f'<div style="height:100%;display:flex;flex-direction:column;justify-content:center;'
        f"align-items:center;text-align:center;font-family:'{display}'\">"
        + "".join(
            f'<div{_ann("banner_text")}>{escape(line)}</div>' for line in _lines(text)
        )
        + "</div>"
    )
    return (
        _wrap(body, css, width=width, height=height, pad=rng.randint(30, 70)),
        width,
        height,
    )


def _banner_social(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A square social card."""
    side = rng.choice([600, 720, 800, 1080])
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(28.0, 48.0), line_height=1.3
    )
    css += f"html,body{{background:{palette.panel}}}"
    body = (
        f'<div style="height:100%;display:flex;flex-direction:column;justify-content:center;'
        f"text-align:right;font-family:'{display}'\">"
        + "".join(
            f'<div{_ann("banner_text")}>{escape(line)}</div>' for line in _lines(text)
        )
        + "</div>"
    )
    return _wrap(body, css, width=side, height=side, pad=rng.randint(36, 80)), side, side


def _banner_sign(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A wide shop sign."""
    width = rng.randint(900, 1500)
    height = rng.randint(200, 360)
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(38.0, 72.0), line_height=1.2
    )
    css += f"html,body{{background:{palette.ink};color:{palette.paper}}}"
    body = (
        f'<div style="height:100%;display:flex;justify-content:center;align-items:center;'
        f"text-align:center;font-family:'{display}'\">"
        + "".join(
            f'<div{_ann("banner_text")}>{escape(line)}</div>' for line in _lines(text)
        )
        + "</div>"
    )
    return (
        _wrap(body, css, width=width, height=height, pad=rng.randint(16, 40)),
        width,
        height,
    )


def _banner_flyer(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A flyer: a display headline over smaller body lines."""
    width = rng.randint(520, 860)
    height = rng.randint(700, 1100)
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(17.0, 24.0), line_height=1.5
    )
    lines = _lines(text)
    headline = lines[0] if lines else ""
    rest = lines[1:]
    body = (
        f'<div{_ann("banner_text")} style="font-family:\'{display}\';font-size:2.3em;'
        f'color:{palette.accent};margin-bottom:22px">{escape(headline)}</div>'
        + "".join(f'<div{_ann("paragraph")}>{escape(line)}</div>' for line in rest)
    )
    return (
        _wrap(body, css, width=width, height=height, pad=rng.randint(28, 60)),
        width,
        height,
    )


# --------------------------------------------------------------------------- #
# story
# --------------------------------------------------------------------------- #


def _story_chapter(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A book page opening a chapter."""
    width, height, margin = _page_size(rng)
    line_height = rng.uniform(1.75, 2.1)
    size_px, fits = _fit_font_px(
        text,
        width,
        height,
        margin,
        line_height,
        low=11.5,
        high=13.5,
        columns=2,
    )
    if not fits:
        # More text than this sheet can hold. Let the canvas grow
        # rather than clip: a tall page is honest, a clipped label is not.
        height = 0
    css = _base_css(palette, font_css, family, size_px=size_px, line_height=line_height)
    number = to_digit_script(str(rng.randint(1, 24)), digits)
    head = (
        f'<div{_ann("heading")} style="text-align:center;font-family:\'{display}\';'
        f'margin-bottom:30px"><div style="font-size:0.9em;color:{palette.muted}">فصل {escape(number)}</div>'
        f'<div style="font-size:1.6em">{escape(_lines(text)[0][:50] if _lines(text) else "")}</div></div>'
    )
    body = head + _render_paragraphs("\n\n".join(_paragraphs(text)[1:]))
    return _wrap(body, css, width=width, height=height, pad=margin), width, height


def _story_dropcap(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A book page whose first paragraph opens with a drop cap."""
    width, height, margin = _page_size(rng)
    line_height = rng.uniform(1.75, 2.1)
    size_px, fits = _fit_font_px(
        text,
        width,
        height,
        margin,
        line_height,
        low=15.0,
        high=18.0,
        columns=1,
    )
    if not fits:
        # More text than this sheet can hold. Let the canvas grow
        # rather than clip: a tall page is honest, a clipped label is not.
        height = 0
    css = _base_css(palette, font_css, family, size_px=size_px, line_height=line_height)
    css += (
        f"#first::first-letter{{float:right;font-size:3.1em;line-height:0.86;"
        f"margin-left:8px;color:{palette.accent};font-family:'{display}'}}"
    )
    paragraphs = _paragraphs(text)
    first = (
        f'<p{_ann("paragraph")} id="first" style="text-align:justify">{escape(paragraphs[0])}</p>'
        if paragraphs
        else ""
    )
    body = first + _render_paragraphs("\n\n".join(paragraphs[1:]))
    return _wrap(body, css, width=width, height=height, pad=margin), width, height


def _story_plain(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A plain book page with a page number."""
    width, height, margin = _page_size(rng)
    line_height = rng.uniform(1.8, 2.1)
    size_px, fits = _fit_font_px(
        text,
        width,
        height,
        margin,
        line_height,
        low=15.0,
        high=18.0,
        columns=1,
    )
    if not fits:
        # More text than this sheet can hold. Let the canvas grow
        # rather than clip: a tall page is honest, a clipped label is not.
        height = 0
    css = _base_css(palette, font_css, family, size_px=size_px, line_height=line_height)
    page_no = (
        f'<div{_ann("page_number")} style="position:absolute;bottom:{margin // 2}px;'
        f'width:100%;text-align:center;color:{palette.muted};font-size:0.84em">'
        f"{escape(to_digit_script(str(rng.randint(5, 320)), digits))}</div>"
    )
    body = f'<div style="text-align:justify">{_render_paragraphs(text)}</div>{page_no}'
    return _wrap(body, css, width=width, height=height, pad=margin), width, height


# --------------------------------------------------------------------------- #
# table_form
# --------------------------------------------------------------------------- #


def _table_grid(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A bare data table."""
    width = rng.randint(620, 1080)
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(13.0, 16.5), line_height=1.5
    )
    body = (
        _table_markup(text, palette, rng) if _is_table(text) else _render_paragraphs(text)
    )
    return _wrap(body, css, width=width, pad=rng.randint(18, 44)), width, 0


def _table_invoice(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """An invoice: header fields above a table."""
    width = rng.randint(680, 1000)
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(12.5, 15.5), line_height=1.55
    )
    number = to_digit_script(str(rng.randint(10000, 99999)), digits)
    date = to_digit_script(
        f"۱۴۰۳/{rng.randint(1,12):02d}/{rng.randint(1,29):02d}", digits
    )
    head = (
        f'<div style="display:flex;justify-content:space-between;margin-bottom:16px;'
        f'padding-bottom:10px;border-bottom:2px solid {palette.accent}">'
        f'<div><span{_ann("field_label")} class="muted">شماره: </span>'
        f'<span{_ann("field_value")}>{escape(number)}</span></div>'
        f'<div><span{_ann("field_label")} class="muted">تاریخ: </span>'
        f'<span{_ann("field_value")}>{escape(date)}</span></div></div>'
    )
    body = head + (
        _table_markup(text, palette, rng) if _is_table(text) else _render_paragraphs(text)
    )
    return _wrap(body, css, width=width, pad=rng.randint(22, 48)), width, 0


def _table_form_fields(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A form of label-and-box field rows."""
    width = rng.randint(600, 940)
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(13.0, 16.5), line_height=1.6
    )
    rows: list[str] = []
    for line in _lines(text):
        label, _, value = line.partition("|") if "|" in line else (line, "", "")
        rows.append(
            f'<div style="display:flex;gap:10px;align-items:center;margin-bottom:10px">'
            f'<span{_ann("field_label")} style="width:34%;color:{palette.muted}">{escape(label.strip())}</span>'
            f'<span{_ann("field_value")} style="flex:1;border:1px solid {palette.muted};'
            f'padding:5px 8px;min-height:1.8em">{escape(value.strip())}</span></div>'
        )
    return _wrap("".join(rows), css, width=width, pad=rng.randint(22, 46)), width, 0


def _table_timetable(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A dense timetable with a tinted header row."""
    width = rng.randint(700, 1180)
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(11.5, 14.5), line_height=1.45
    )
    body = (
        _table_markup(text, palette, rng) if _is_table(text) else _render_paragraphs(text)
    )
    return _wrap(body, css, width=width, pad=rng.randint(14, 34)), width, 0


# --------------------------------------------------------------------------- #
# mixed
# --------------------------------------------------------------------------- #


def _mixed_technical(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """Persian technical prose carrying Latin terms."""
    width = rng.randint(680, 1040)
    css = _base_css(
        palette,
        font_css,
        family,
        size_px=rng.uniform(14.0, 17.0),
        line_height=rng.uniform(1.65, 1.95),
    )
    body = _render_paragraphs(text)
    return _wrap(body, css, width=width, pad=rng.randint(24, 52)), width, 0


def _mixed_ui(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A code-switched settings panel."""
    width = rng.randint(520, 860)
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(13.5, 16.5), line_height=1.6
    )
    rows: list[str] = []
    for line in _lines(text):
        rows.append(
            f'<div style="display:flex;justify-content:space-between;padding:10px 0;'
            f'border-bottom:1px solid {palette.panel}">'
            f'<span{_ann("ui_string")}>{escape(line)}</span>'
            f'<span{_ann("ui_string")} style="color:{palette.accent}">'
            f"{escape(rng.choice(_UI_ACTIONS))}</span></div>"
        )
    return _wrap("".join(rows), css, width=width, pad=rng.randint(18, 40)), width, 0


def _mixed_label(
    text: str,
    palette: Palette,
    font_css: str,
    family: str,
    display: str,
    rng: random.Random,
    digits: str,
) -> tuple[str, int, int]:
    """A product label mixing Persian and Latin."""
    width = rng.randint(380, 700)
    css = _base_css(
        palette, font_css, family, size_px=rng.uniform(12.5, 17.0), line_height=1.55
    )
    body = (
        f'<div style="border:2px solid {palette.ink};padding:{rng.randint(14, 28)}px">'
        + "".join(
            f'<div{_ann("field_value")}>{escape(line)}</div>' for line in _lines(text)
        )
        + "</div>"
    )
    return _wrap(body, css, width=width, pad=rng.randint(12, 30)), width, 0


#: One template builder: text, colours, fonts and an RNG in, a document and
#: its canvas size out.
Builder = Callable[
    [str, Palette, str, str, str, random.Random, str], tuple[str, int, int]
]

#: Dispatch table. Every id in :data:`TEMPLATE_IDS` must appear here; the
#: module-level check below enforces it, because a template that is planned but
#: has no builder would fail only once the renderer reached that sample.
_BUILDERS: Final[dict[str, Builder]] = {
    "page_prose": _page_prose,
    "page_two_column": _page_two_column,
    "page_headed_article": _page_headed_article,
    "page_textbook": _page_textbook,
    "page_report": _page_report,
    "page_newspaper": _page_newspaper,
    "para_plain": _para_plain,
    "para_boxed": _para_boxed,
    "para_caption": _para_caption,
    "para_pullquote": _para_pullquote,
    "line_title": _line_title,
    "line_label": _line_label,
    "line_ui": _line_ui,
    "line_subtitle": _line_subtitle,
    "poem_two_hemistich": _poem_two_hemistich,
    "poem_centred": _poem_centred,
    "poem_free": _poem_free,
    "poem_framed": _poem_framed,
    "web_article": _web_article,
    "web_blog": _web_blog,
    "web_forum": _web_forum,
    "web_product": _web_product,
    "web_faq": _web_faq,
    "chat_light": _chat_light,
    "chat_dark": _chat_dark,
    "chat_sms": _chat_sms,
    "chat_group": _chat_group,
    "banner_poster": _banner_poster,
    "banner_social": _banner_social,
    "banner_sign": _banner_sign,
    "banner_flyer": _banner_flyer,
    "story_chapter": _story_chapter,
    "story_dropcap": _story_dropcap,
    "story_plain": _story_plain,
    "table_grid": _table_grid,
    "table_invoice": _table_invoice,
    "table_form_fields": _table_form_fields,
    "table_timetable": _table_timetable,
    "mixed_technical": _mixed_technical,
    "mixed_ui": _mixed_ui,
    "mixed_label": _mixed_label,
}

_declared = {name for names in TEMPLATE_IDS.values() for name in names}
_missing = _declared - set(_BUILDERS)
if _missing:  # pragma: no cover - a module-level contract
    raise RuntimeError(f"templates declared with no builder: {sorted(_missing)}")
_extra = set(_BUILDERS) - _declared
if _extra:  # pragma: no cover
    raise RuntimeError(f"builders not declared in TEMPLATE_IDS: {sorted(_extra)}")
