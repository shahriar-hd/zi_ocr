"""Definitions shared by the numbers loader and the words loader.

:mod:`~fa_ocr_lora.config.config` reads ``synth_config.yaml`` and
:mod:`~fa_ocr_lora.config.vocabulary` reads ``vocabulary_fa.yaml``. Both raise
the same exception, and both validate a per-chart-kind mapping against the same
set of kinds - so without this module the two would have to import each other.
Nothing here depends on either of them.

Both names are re-exported from :mod:`~fa_ocr_lora.config.config` and from the
package, so importers keep using ``from ...config import ConfigError`` and need
not know this file exists.
"""

from __future__ import annotations

from typing import Final

__all__ = ["CHART_KINDS", "ConfigError"]


class ConfigError(ValueError):
    """Raised when the configuration file is missing keys or malformed."""


#: Chart kinds the renderer can draw. A kind outside this set would reach
#: ``ChartVocabulary.caption_for`` with no caption registered and render a
#: chart the loader promised was captioned, so both loaders validate against it.
#:
#: Intentionally not a knob: adding a kind here without teaching
#: ``elements/charts.py`` to draw it produces a page with a blank space where
#: the chart should be.
CHART_KINDS: Final[tuple[str, ...]] = ("bar", "line", "pie")
