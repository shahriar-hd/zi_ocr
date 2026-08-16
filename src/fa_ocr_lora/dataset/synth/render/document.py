"""Jinja rendering of the letter templates into one self-contained document.

The two templates are rendered in sequence: ``letter.css.j2`` first, then its
output is injected into ``letter.html.j2`` as an inline ``<style>``. That order
is deliberate - the page must not reference a stylesheet by URL, because a
Chromium context created with ``set_content`` has no base URL to resolve one
against, and a page that renders before its CSS arrives is a page whose boxes
are measured against the wrong layout.

Autoescaping is off. Everything interpolated is either a colour, a number, or a
fragment the layout modules already escaped, and turning it on would corrupt
the CSS: a font stack like ``"B Nazanin", sans-serif`` would come out with its
quotes HTML-encoded and no Persian face would resolve.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Mapping

__all__ = ["TemplateError", "render_document", "template_dir"]

#: Directory holding the shipped templates, as declared in ``pyproject.toml``'s
#: package data.
_TEMPLATE_DIR: Final[Path] = Path(__file__).with_name("templates")

_CSS_TEMPLATE: Final[str] = "letter.css.j2"
_HTML_TEMPLATE: Final[str] = "letter.html.j2"


class TemplateError(RuntimeError):
    """Raised when the templates are missing or fail to render."""


def template_dir() -> Path:
    """Return the directory the letter templates are loaded from."""
    return _TEMPLATE_DIR


@lru_cache(maxsize=1)
def _environment() -> Any:
    """Return the process-wide Jinja environment.

    Cached because compiling the templates is the expensive part and the
    result is immutable; a 20k-sample run reuses it for every page.

    Returns:
        A configured :class:`jinja2.Environment`.

    Raises:
        TemplateError: If Jinja2 is not installed.
    """
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ImportError as error:  # pragma: no cover - environment issue
        raise TemplateError("jinja2 is required to render letters") from error

    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render_document(context: Mapping[str, Any]) -> str:
    """Render one composed letter to a complete HTML document.

    Args:
        context: The context built by
            :func:`..layout.page_composer.compose_letter`.

    Returns:
        A standalone HTML document: inline CSS, inline fonts, inline images.

    Raises:
        TemplateError: If a template is missing or references a value the
            context does not provide. ``StrictUndefined`` makes that a hard
            failure rather than a silently blank page, since a blank palette
            entry would render invisible text and still produce boxes.
    """
    try:
        from jinja2 import TemplateError as JinjaTemplateError
    except ImportError as error:  # pragma: no cover - environment issue
        raise TemplateError("jinja2 is required to render letters") from error

    environment = _environment()
    payload = dict(context)
    try:
        stylesheet = environment.get_template(_CSS_TEMPLATE).render(**payload)
        payload["stylesheet"] = stylesheet
        return environment.get_template(_HTML_TEMPLATE).render(**payload)
    except JinjaTemplateError as error:
        raise TemplateError(f"letter template failed: {error}") from error
