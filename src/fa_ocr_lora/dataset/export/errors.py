"""The export package's own exception.

Its own module so the layering stays acyclic: :mod:`.families` and
:mod:`.huggingface` both raise it, and ``huggingface`` imports ``families``.
:mod:`.huggingface` re-exports it, so ``from ..export.huggingface import
ExportError`` keeps working for the callers that had it.
"""

from __future__ import annotations

__all__ = ["ExportError"]


class ExportError(RuntimeError):
    """Raised when an export cannot start or cannot be completed.

    Distinct from :class:`~fa_ocr_lora.config.ConfigError`: the configuration
    may be perfectly valid and the corpus still unexportable - no rendered
    samples, a missing page image, or ``pyarrow`` not installed.
    """
