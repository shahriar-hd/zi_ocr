"""Typed loader for ``configs/field_validation.yaml``.

The sensitive-field audit is driven entirely from that file, and it lives in its
own file rather than as a section of ``lora_config.yaml`` for one reason: the
audit is not a training setting. Phase 3 inference has to apply the identical
patterns to a document it has never seen, with no manifest, no split and no
learning rate - so it must be able to load these rules without loading a
training profile.

What is configurable and what is not follows from how brittle each part is.

*Label wordings and value patterns are configurable*, because they are
organisational conventions rather than facts. The شماره ثبت of one ministry is
five bare digits; another's carries a year prefix and a letter class. CLAUDE.md 8
forbids burying either in a module, and the user asking for "matching against the
official registration-number pattern" is asking for a value that differs per
deployment. The defaults shipped in the YAML describe *this corpus*, and the file
says so.

*The semantic checks are not configurable*, only selectable. There is one right
answer for whether a Jalali date exists, so the choice offered is which check
applies to a field, not what the check does.

Loading is strict in the same way the training loader is: an unknown key or an
uncompilable pattern is an error at load time, because the alternative is an
audit that silently stops finding a field and reports a perfect score for it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import yaml

from ..config.base import ConfigError
from ..dataset.common.io_utils import resolve_repo_path

__all__ = [
    "DEFAULT_FIELD_CONFIG_PATH",
    "FIELD_CONFIG_PATH_ENV_VAR",
    "FieldCheck",
    "FieldKind",
    "FieldSpec",
    "FieldValidationConfig",
    "load_field_validation_config",
]

#: Repository-relative default, used when no path is given.
DEFAULT_FIELD_CONFIG_PATH: Final[str] = "configs/field_validation.yaml"

#: Variable naming the file, honoured when no path is passed. Lets a deployment
#: substitute its own registration-number pattern without editing the repository.
FIELD_CONFIG_PATH_ENV_VAR: Final[str] = "FA_OCR_FIELD_CONFIG"

#: Fallbacks for the two optional top-level keys. Module constants rather than
#: reads of the dataclass attributes, because ``slots=True`` replaces class-level
#: defaults with slot descriptors - ``FieldValidationConfig.separators`` is a
#: descriptor, not a tuple.
_DEFAULT_SEPARATORS: Final[tuple[str, ...]] = (":", "")
_DEFAULT_MAX_GAP_CHARS: Final[int] = 12
_DEFAULT_REQUIRE_FIELD_CONTEXT: Final[bool] = True


class FieldKind(StrEnum):
    """The sensitive fields this layer audits.

    The five names match the metadata-block keys the synthetic generator emits
    (``letter_number``, ``registration_number``, ``date``, ``national_id``) plus
    ``amount``, which appears in table cells rather than the metadata block and
    so is found by pattern alone.
    """

    LETTER_NUMBER = "letter_number"
    REGISTRATION_NUMBER = "registration_number"
    DATE = "date"
    NATIONAL_ID = "national_id"
    AMOUNT = "amount"


class FieldCheck(StrEnum):
    """Semantic checks a field may be subject to beyond its pattern.

    ``NONE`` is not an absence of checking - a field with no semantic check is
    still pattern-matched, still compared against the reference, and still
    screened for a repeated-digit run. It means only that nothing further about
    the value is decidable from the value alone.
    """

    NONE = "none"
    JALALI_DATE = "jalali_date"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """How one sensitive field is recognised and checked.

    Attributes:
        kind: Which field this describes.
        labels: Wordings that introduce the value in running text, e.g.
            ``شماره ثبت``. Empty for a field that carries no label and is found
            by its value pattern alone.
        pattern: Compiled value pattern. Must match the *whole* value; the
            extractor anchors it, so a pattern needs no ``^``/``$``.
            Match group 0 is taken as the value.
        check: Semantic check applied to a matched value.
        description: Human-readable note, echoed into the audit report so a
            reader of the JSON knows what the pattern was meant to describe.
    """

    kind: FieldKind
    labels: tuple[str, ...]
    pattern: re.Pattern[str]
    check: FieldCheck = FieldCheck.NONE
    description: str = ""

    @property
    def is_labelled(self) -> bool:
        """Whether this field is located by a label rather than by pattern alone.

        Returns:
            ``True`` if the spec carries at least one label wording.
        """
        return bool(self.labels)


@dataclass(frozen=True, slots=True)
class FieldValidationConfig:
    """Everything the audit needs, validated.

    Attributes:
        specs: One spec per audited field, in the file's order.
        separators: Strings that may sit between a label and its value, beyond
            whitespace. The empty string is a real member: many letterheads align
            values in a column and use whitespace alone.
        max_gap_chars: How far past a label the extractor will look for a value
            before giving up. Bounded so that a label at the end of one line does
            not capture a number from two paragraphs later.
        require_field_context: Whether a label with no readable value must look
            like a field introduction before it is reported as malformed. Every
            label here is also an ordinary Persian word, so with this off the
            audit reports ``در تاریخ ۱۳۸۹ مبنی بر`` as a malformed date -
            measured at 67% of all malformed findings on this corpus's reference
            text. Off is the literal reading of the rules and is kept available
            for exactly that: measuring how much the rule suppresses.
        source_path: File this was loaded from, or ``None`` for the defaults.
    """

    specs: tuple[FieldSpec, ...]
    separators: tuple[str, ...] = _DEFAULT_SEPARATORS
    max_gap_chars: int = _DEFAULT_MAX_GAP_CHARS
    require_field_context: bool = _DEFAULT_REQUIRE_FIELD_CONTEXT
    source_path: Path | None = None

    def spec_for(self, kind: FieldKind) -> FieldSpec | None:
        """Look up one field's spec.

        Args:
            kind: The field wanted.

        Returns:
            The spec, or ``None`` if this configuration does not audit that
            field.
        """
        for spec in self.specs:
            if spec.kind is kind:
                return spec
        return None

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a JSON-serialisable mapping.

        Written into the audit report so a saved result carries the rules that
        produced it - a field-validity rate is not interpretable without the
        pattern it was measured against.

        Returns:
            A mapping suitable for ``json.dump``.
        """
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "separators": list(self.separators),
            "max_gap_chars": self.max_gap_chars,
            "require_field_context": self.require_field_context,
            "fields": {
                spec.kind.value: {
                    "labels": list(spec.labels),
                    "pattern": spec.pattern.pattern,
                    "check": spec.check.value,
                    "description": spec.description,
                }
                for spec in self.specs
            },
        }


def _reject_unknown(
    payload: Mapping[str, Any], allowed: Sequence[str], where: str
) -> None:
    """Fail on any key outside the allowed set.

    Args:
        payload: The mapping to check.
        allowed: Keys this mapping may carry.
        where: Dotted path used in the error message.

    Raises:
        ConfigError: On the first unknown key, naming the legal alternatives.
    """
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {', '.join(unknown)}; "
            f"expected one of {', '.join(sorted(allowed))}"
        )


def _as_str_tuple(payload: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
    """Read a list-of-strings key, defaulting to empty.

    Args:
        payload: The mapping to read.
        key: Key to read.
        where: Dotted path used in the error message.

    Returns:
        The values as a tuple, empty when the key is absent.

    Raises:
        ConfigError: If the value is not a sequence of strings.
    """
    raw = payload.get(key, ())
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise ConfigError(f"{where}.{key}: expected a list of strings, got {raw!r}")
    values = []
    for item in raw:
        if not isinstance(item, str):
            raise ConfigError(f"{where}.{key}: expected strings, got {item!r}")
        values.append(item)
    return tuple(values)


def _compile(payload: Mapping[str, Any], where: str) -> re.Pattern[str]:
    """Compile a field's value pattern.

    Args:
        payload: The field's mapping.
        where: Dotted path used in the error message.

    Returns:
        The compiled pattern.

    Raises:
        ConfigError: If the pattern is missing, not a string, uncompilable, or
            matches the empty string - the last because a pattern that matches
            nothing at all would report every value as present and correct.
    """
    raw = payload.get("pattern")
    if not isinstance(raw, str) or not raw:
        raise ConfigError(f"{where}.pattern: expected a non-empty regex string")
    try:
        compiled = re.compile(raw)
    except re.error as exc:
        raise ConfigError(f"{where}.pattern: invalid regex ({exc})") from exc
    if compiled.fullmatch(""):
        raise ConfigError(
            f"{where}.pattern: matches the empty string, so every value would "
            "be reported valid; make at least one character required"
        )
    return compiled


def _parse_spec(kind_name: str, payload: Mapping[str, Any]) -> FieldSpec:
    """Build one field spec from its YAML mapping.

    Args:
        kind_name: The mapping's key, which must name a :class:`FieldKind`.
        payload: The field's settings.

    Returns:
        The validated spec.

    Raises:
        ConfigError: On an unknown field name, an unknown key, a bad pattern or
            an unknown check.
    """
    try:
        kind = FieldKind(kind_name)
    except ValueError as exc:
        legal = ", ".join(member.value for member in FieldKind)
        raise ConfigError(
            f"fields.{kind_name}: not an auditable field; expected one of {legal}"
        ) from exc
    where = f"fields.{kind_name}"
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(payload).__name__}")
    _reject_unknown(payload, ("labels", "pattern", "check", "description"), where)

    raw_check = payload.get("check", FieldCheck.NONE.value)
    try:
        check = FieldCheck(raw_check)
    except ValueError as exc:
        legal = ", ".join(member.value for member in FieldCheck)
        raise ConfigError(
            f"{where}.check: unknown check {raw_check!r}; expected one of {legal}"
        ) from exc

    description = payload.get("description", "")
    if not isinstance(description, str):
        raise ConfigError(f"{where}.description: expected a string, got {description!r}")

    return FieldSpec(
        kind=kind,
        labels=_as_str_tuple(payload, "labels", where),
        pattern=_compile(payload, where),
        check=check,
        description=description,
    )


def load_field_validation_config(
    path: str | Path | None = None,
) -> FieldValidationConfig:
    """Read and validate the field-validation rules.

    Args:
        path: File to read. When ``None``, the ``FA_OCR_FIELD_CONFIG``
            environment variable is consulted, then
            :data:`DEFAULT_FIELD_CONFIG_PATH`.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the file is missing, empty, invalid YAML, or carries an
            unknown key, a bad pattern or no fields at all.
    """
    chosen = path if path is not None else os.environ.get(FIELD_CONFIG_PATH_ENV_VAR)
    resolved = resolve_repo_path(chosen or DEFAULT_FIELD_CONFIG_PATH)
    if not resolved.is_file():
        raise ConfigError(f"field validation config not found: {resolved}")
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{resolved}: invalid YAML ({exc})") from exc
    if payload is None:
        raise ConfigError(f"{resolved}: file is empty")
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{resolved}: expected a mapping at the top level")
    _reject_unknown(
        payload,
        ("fields", "separators", "max_gap_chars", "require_field_context"),
        "top level",
    )

    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, Mapping) or not raw_fields:
        raise ConfigError(
            f"{resolved}: fields: expected a non-empty mapping of field name to "
            "settings; a config that audits nothing would report a clean run"
        )
    specs = tuple(_parse_spec(name, spec) for name, spec in raw_fields.items())

    separators = (
        _as_str_tuple(payload, "separators", "top level")
        if "separators" in payload
        else _DEFAULT_SEPARATORS
    )
    raw_gap = payload.get("max_gap_chars", _DEFAULT_MAX_GAP_CHARS)
    gap_ok = isinstance(raw_gap, int) and not isinstance(raw_gap, bool)
    if not gap_ok or not 0 <= raw_gap <= 64:
        raise ConfigError(f"max_gap_chars: expected an integer in 0-64, got {raw_gap!r}")

    raw_context = payload.get("require_field_context", _DEFAULT_REQUIRE_FIELD_CONTEXT)
    if not isinstance(raw_context, bool):
        raise ConfigError(
            f"require_field_context: expected true or false, got {raw_context!r}"
        )

    return FieldValidationConfig(
        specs=specs,
        separators=separators,
        max_gap_chars=raw_gap,
        require_field_context=raw_context,
        source_path=resolved,
    )
