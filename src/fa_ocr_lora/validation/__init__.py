"""Sensitive-field validation for Persian administrative letters.

CER and WER measure how much of a page was read correctly. They cannot measure
whether the شماره ثبت was read correctly, because a page is ~700 characters of
prose against ~20 characters of registration number and date - so a model that
mangles every digit of the metadata block still scores a CER around 3%. CLAUDE.md
5.2 makes that block the highest-priority field. This package is what makes it
separately measurable, and separately auditable at inference time when there is no
ground truth to compare against.

Four questions are asked of every field, kept apart because they answer to
different audiences:

===========================  ===============  ==================================
Question                     Needs reference  Module
===========================  ===============  ==================================
Is the value well formed?    no               :mod:`.fields`
Is it internally consistent? no               :mod:`.iran` (Jalali dates)
How certain was the model?   no               :mod:`.confidence`, :mod:`.recorder`
Does it match the truth?     yes              :mod:`.fields`, :mod:`.report`
===========================  ===============  ==================================

The first three survive into production, which is the point: a
``شماره ثبت:`` followed by nothing numeric, a تاریخ of Esfand 30 in a common year,
or a value whose weakest token the model gave a 0.2 probability are all faults that
can be flagged on a document nobody has ever transcribed.

Rules live in ``configs/field_validation.yaml`` - label wordings and value
patterns, because a registration-number format is an organisational convention
rather than a fact, and CLAUDE.md 8 forbids burying either in a module.

Everything re-exported here is free of torch, so the Phase 3 inference layer can
apply the identical rules without importing the training package. The one piece
that needs the framework - :class:`~fa_ocr_lora.validation.recorder.ConfidenceRecorder`,
which hooks generation to record per-token certainty - lives in :mod:`.recorder`
and must be imported from there explicitly.
"""

from __future__ import annotations

from .confidence import (
    FieldConfidence,
    LazySpanConfidence,
    TokenConfidence,
    span_confidence,
    token_spans,
)
from .config import (
    DEFAULT_FIELD_CONFIG_PATH,
    FIELD_CONFIG_PATH_ENV_VAR,
    FieldCheck,
    FieldKind,
    FieldSpec,
    FieldValidationConfig,
    load_field_validation_config,
)
from .fields import (
    ExtractedField,
    FieldComparison,
    compare_fields,
    extract_fields,
)
from .iran import (
    all_digits_identical,
    digits_of,
    jalali_date_is_valid,
    jalali_is_leap,
    jalali_month_length,
    normalize_digits,
    parse_jalali_date,
)
from .report import (
    DEFAULT_CONFIDENCE_FLOOR,
    DEFAULT_MAX_FINDINGS,
    FieldAudit,
    FieldAuditor,
    FieldFinding,
    FieldStats,
    FindingKind,
    summarize_fields,
)

__all__ = [
    "DEFAULT_CONFIDENCE_FLOOR",
    "DEFAULT_FIELD_CONFIG_PATH",
    "DEFAULT_MAX_FINDINGS",
    "FIELD_CONFIG_PATH_ENV_VAR",
    "ExtractedField",
    "FieldAudit",
    "FieldAuditor",
    "FieldCheck",
    "FieldComparison",
    "FieldConfidence",
    "FieldFinding",
    "FieldKind",
    "FieldSpec",
    "FieldStats",
    "FieldValidationConfig",
    "FindingKind",
    "LazySpanConfidence",
    "TokenConfidence",
    "all_digits_identical",
    "compare_fields",
    "digits_of",
    "extract_fields",
    "jalali_date_is_valid",
    "jalali_is_leap",
    "jalali_month_length",
    "load_field_validation_config",
    "normalize_digits",
    "parse_jalali_date",
    "span_confidence",
    "summarize_fields",
    "token_spans",
]
