"""Drive the letter generator and build the sample manifest.

Run this to produce the plain-text half of the synthetic dataset:

* letters are written by :mod:`fa_ocr_lora.dataset.synth.text_generator` to
  ``data/text/plaintext/plaintext_000000.json`` (one file per sample, six-digit
  index, resumable -- re-running fills only the gaps);
* every letter is then joined back to its source row in
  ``data/thesvg/logos.csv`` and listed in ``data/text/generator.csv``, the
  manifest downstream phases (page composition, annotation, training) read to
  resolve a sample to its inputs and its logo artwork.

The file doubles as a pytest module. The offline checks cover the reply parser
and the Persian-numeral synthesizers and need no credentials; the end-to-end
test calls the live API, writes to a pytest ``tmp_path`` rather than ``data/``,
and is skipped unless an API key is configured.

Run it as a script (writes into ``data/``)::

    python test/test_text_generator.py --count 20

Or under pytest (offline checks only)::

    pytest test/test_text_generator.py -k "not end_to_end" -v
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Final, Sequence

#: Make ``import fa_ocr_lora`` work when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fa_ocr_lora.dataset.synth import text_generator as gen  # noqa: E402

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

#: Source of the sender-company pool, produced by scripts/download_thesvg_logo.py.
DEFAULT_LOGOS_CSV: Final[Path] = REPO_ROOT / "data" / "thesvg" / "logos.csv"
#: Where the generator writes one JSON file per letter.
DEFAULT_OUTPUT_DIR: Final[Path] = REPO_ROOT / "data" / "text" / "plaintext"
#: Manifest joining each letter back to its logo row.
DEFAULT_MANIFEST: Final[Path] = REPO_ROOT / "data" / "text" / "generator.csv"

#: Root of the UTSig signature corpus, scanned recursively for ``.tif`` files.
DEFAULT_SIGNATURES_DIR: Final[Path] = REPO_ROOT / "data" / "UTSig"

#: Share of letters that carry a signature. Not every administrative letter is
#: signed in practice, and CLAUDE.md 5.2 calls for the signature to be added
#: randomly, so the remainder deliberately gets an empty ``signature_path``.
SIGNATURE_PROBABILITY: Final[float] = 0.7

DEFAULT_SAMPLE_COUNT: Final[int] = 3

#: Columns of the manifest, in the order they are written.
#:
#: A deliberate middle ground: the long fields (``body``, ``attachments``) stay
#: in the JSON and only ``sample_path`` points at them, because inlining them
#: would make this file ~50 MB for a 20k run, duplicate the corpus on disk, and
#: embed multi-line Persian text in CSV cells. Everything short enough to sort,
#: filter or eyeball is here, so the common questions -- which letters carry a
#: table, what subjects were produced, how long each body is -- are answered
#: without opening 20,000 files.
MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "sample_index",
    "sample_path",
    "logo_path",
    "logo_title",
    "logo_categories",
    "subject",
    "destination_organization",
    "letter_number",
    "date",
    "national_id",
    "registration_number",
    "signatory_title",
    "chart_table",
    "letter_genre",
    "letter_tone",
    "body_chars",
    "has_attachments",
    "filled_fields",
    "signature_path",
)


def load_signatures(signatures_dir: Path) -> list[Path]:
    """Collect every signature image under the UTSig corpus.

    Both ``Genuine`` and ``Forgery`` are used: the page composer only needs a
    plausible ink shape, and a forged signature looks no less like handwriting
    than a genuine one, so the whole corpus widens the visual variety.

    Args:
        signatures_dir: Root of the UTSig tree.

    Returns:
        Sorted paths of the signature images; empty if the corpus is absent.
    """
    if not signatures_dir.is_dir():
        return []
    return sorted(signatures_dir.rglob("*.tif"))


def assign_signature(
    sample_index: int,
    signatures: Sequence[Path],
    seed: int = gen.DEFAULT_SEED,
    probability: float = SIGNATURE_PROBABILITY,
) -> str:
    """Pick the signature image for one sample, or none.

    Derived from ``(seed, sample_index)`` rather than drawn from a shared RNG,
    so a sample keeps the same signature no matter when it is generated, in
    which order, or how many times the manifest is rebuilt. That is what lets
    the backfill script fill old rows without changing them on the next run.

    Args:
        sample_index: Zero-based sample index.
        signatures: Pool returned by :func:`load_signatures`.
        seed: Master seed, matching the generator's own.
        probability: Chance this sample carries a signature at all.

    Returns:
        A repo-relative path, or ``""`` when the letter is unsigned.
    """
    if not signatures:
        return ""
    rng = random.Random(f"{seed}:signature:{sample_index}")
    if rng.random() >= probability:
        return ""
    chosen = rng.choice(signatures)
    try:
        return str(chosen.relative_to(REPO_ROOT))
    except ValueError:
        #: Pool lives outside the repo (a test fixture); keep the path usable.
        return str(chosen)


def build_generator_manifest(
    output_dir: Path,
    manifest_path: Path,
    signatures_dir: Path = DEFAULT_SIGNATURES_DIR,
) -> int:
    """Write one manifest row per generated letter.

    Each row carries the logo's path, title and categories alongside the
    letter's identifying fields and the path of its JSON file, so a later phase
    can pair a letter with its artwork -- and answer "which samples have a
    table?" or "what subjects came out?" -- without opening every JSON. The
    long fields stay in the JSON; ``sample_path`` points at them.

    Args:
        output_dir: Directory holding the ``plaintext_*.json`` files.
        manifest_path: CSV file to write; parent directories are created.
        signatures_dir: Root of the UTSig corpus supplying ``signature_path``.

    Returns:
        The number of rows written.

    Raises:
        FileNotFoundError: If the output directory holds no letter.
    """
    signatures = load_signatures(signatures_dir)
    if not signatures:
        print(f"WARNING: no signatures found under {signatures_dir}; column left empty")

    rows: list[dict[str, str]] = []
    for path in sorted(output_dir.glob("plaintext_*.json")):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        inputs = record["inputs"]
        letter = record["letter"]
        filled = record.get("generation", {}).get("filled_fields") or []
        rows.append(
            {
                "sample_id": record["sample_id"],
                "sample_index": str(record["sample_index"]),
                "sample_path": str(path),
                "logo_path": inputs["svg_path"],
                "logo_title": inputs["company_name"],
                "logo_categories": inputs["company_category"],
                "subject": letter["subject"],
                "destination_organization": letter["destination_organization"],
                "letter_number": letter["letter_number"],
                "date": letter["date"],
                "national_id": letter["national_id"],
                "registration_number": letter["registration_number"],
                "signatory_title": letter["signatory_title"],
                "chart_table": inputs["chart_table"],
                "letter_genre": inputs["letter_genre"],
                "letter_tone": inputs["letter_tone"],
                "body_chars": str(len(letter["body"])),
                "has_attachments": "yes" if letter["attachments"].strip() else "no",
                "filled_fields": "|".join(filled),
                "signature_path": assign_signature(record["sample_index"], signatures),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No plaintext_*.json files found under {output_dir}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def run(
    output_dir: Path,
    logos_csv: Path,
    sample_count: int,
    manifest_path: Path,
    extra_args: list[str] | None = None,
) -> int:
    """Generate letters, then build the manifest describing them.

    Args:
        output_dir: Directory that receives the ``plaintext_*.json`` files.
        logos_csv: Logo manifest supplying the sender-company pool.
        sample_count: Number of letters the output directory should hold.
        manifest_path: CSV file the manifest is written to.
        extra_args: Additional ``--`` flags forwarded to the generator CLI.

    Returns:
        The number of manifest rows written.

    Raises:
        SystemExit: If the generator exits non-zero.
    """
    args = [
        "--output-dir", str(output_dir),
        "--logos-csv", str(logos_csv),
        "--count", str(sample_count),
    ]
    if extra_args:
        args.extend(extra_args)

    print(f"Generating {sample_count} letter(s) into {output_dir}")
    code = gen.main(args)
    if code != 0:
        raise SystemExit(f"text_generator.main() exited with code {code}")

    written = len(list(output_dir.glob("plaintext_*.json")))
    print(f"Building manifest at {manifest_path} ({written} letter(s) on disk)")
    return build_generator_manifest(output_dir, manifest_path)


# --------------------------------------------------------------------------- #
# Offline tests -- no API key required
# --------------------------------------------------------------------------- #


def _letter_payload(**overrides: Any) -> dict[str, Any]:
    """Build a syntactically complete letter payload for parser tests.

    Args:
        **overrides: Fields to replace in the default payload.

    Returns:
        A payload accepted by :func:`fa_ocr_lora...validate_letter`.
    """
    payload: dict[str, Any] = {key: "مقدار تستی" for key in gen.REQUIRED_KEYS}
    payload["subject"] = "موضوع تستی"
    payload["destination_organization"] = "اداره کل تست"
    payload["body"] = "متن " * 200
    payload["contact_info"] = {key: "x" for key in gen.CONTACT_KEYS}
    payload.update(overrides)
    return payload


def test_parse_model_json_plain() -> None:
    """A bare JSON object parses unchanged."""
    payload = gen.parse_model_json(json.dumps({"subject": "x"}, ensure_ascii=False))
    assert payload["subject"] == "x"


def test_parse_model_json_wrapped_in_fences_and_prose() -> None:
    """Code fences and surrounding prose are stripped before parsing."""
    raw = 'خروجی:\n```json\n{"subject": "x"}\n```\nپایان.'
    assert gen.parse_model_json(raw)["subject"] == "x"


def test_parse_model_json_ignores_trailing_brace() -> None:
    """A stray brace after the object does not corrupt the capture.

    The previous greedy ``{.*}`` regex swallowed the trailing text and failed
    to parse; the brace-matching scan stops at the object's real end.
    """
    raw = '{"subject": "x"} توضیح اضافه } پایان'
    assert gen.parse_model_json(raw)["subject"] == "x"


def test_parse_model_json_recovers_truncated_reply() -> None:
    """A reply cut off mid-object is repaired rather than discarded.

    The body is the expensive part of the letter, so salvaging a truncated
    draft avoids paying for a whole regeneration.
    """
    raw = '{"subject": "x", "body": "متن کامل که ناگهان قطع'
    payload = gen.parse_model_json(raw)
    assert payload["subject"] == "x"
    assert payload["body"].startswith("متن کامل")


def test_parse_model_json_rejects_garbage() -> None:
    """Text holding no JSON object raises rather than returning junk."""
    try:
        gen.parse_model_json("سلام، هیچ جیسونی اینجا نیست.")
    except gen.TextGenerationError:
        return
    raise AssertionError("expected TextGenerationError")


def test_synthesized_numerals_contain_no_ascii_digits() -> None:
    """Locally generated numerals stay entirely in Persian-Indic digits.

    Zero-padding after transliteration used to splice an ASCII ``0`` into the
    date, which would have corrupted the OCR ground truth.
    """
    import random

    for seed in range(200):
        rng = random.Random(seed)
        for value in (gen.synthesize_jalali_date(rng), gen.synthesize_letter_number(rng)):
            assert not any(char.isdigit() and char.isascii() for char in value), value


def test_clean_body_strips_openers_and_header_lines() -> None:
    """Ceremonial openers and repeated header lines never reach the body.

    The page composer renders the letterhead, so an opener left in the text
    would appear twice on the page and drift into the OCR ground truth.
    """
    content = "احتراماً به استناد مصوبه شماره ۱۲۳ اعلام می‌دارد که..."
    openers = (
        "بسم الله الرحمن الرحیم",
        "بسم‌الله‌الرحمن‌الرحیم",
        "بسم الله",
        "بسم رب الشهدا",
        "به نام خدا",
        "به نام خداوند متعال",
        "به نام خداوند بخشنده مهربان",
        "بنام خداوند",
        "باسمه تعالی",
        "بسمه تعالی",
        "هو الله",
        "هوالعلیم",
        "به یاری خداوند متعال",
        "یا علی",
    )
    for opener in openers:
        for separator in ("\n", "\n\n", "\n \n"):
            assert gen.clean_body(f"{opener}{separator}{content}") == content, opener

    #: The opener must never be stripped *partially*. A pattern treating
    #: «الرحمن الرحیم» as optional turned "بسم الله الرحمن الرحیم اداری"
    #: into the mangled "الرحمن الرحیم اداری", which then reached the body.
    for tail in ("اداری", "قانون اساسی جمهوری اسلامی ایران", "عدالت و قانون‌گذاری"):
        cleaned = gen.clean_body(f"بسم الله الرحمن الرحیم {tail}\n\n{content}")
        assert cleaned == content, cleaned
        assert "بسم" not in cleaned and "الرحیم" not in cleaned, cleaned
        assert gen.clean_body(f"بسمه تعالی {tail}\n{content}") == content

    #: Header lines belong in their own JSON fields.
    stacked = f"بسم الله\nشماره: ۱۲۳/الف\nتاریخ: ۱۴۰۳/۰۵/۱۲\nموضوع: درخواست\n{content}"
    assert gen.clean_body(stacked) == content

    #: A body already starting with real content is returned untouched, and a
    #: later mention of these words is not treated as a header.
    assert gen.clean_body(content) == content
    keep = f"{content}\nموضوع: این متن باید بماند"
    assert gen.clean_body(keep) == keep


def test_fill_missing_fields_is_deterministic_and_company_consistent() -> None:
    """Repairs derive from the sample seed and the sender's own slug."""
    import random

    config = gen.GeneratorConfig(output_dir=Path("/tmp"), logos_csv=DEFAULT_LOGOS_CSV)
    company = gen.CompanyProfile(
        logo_id="thesvg_acme", slug="acme", title="Acme", categories=("Software",)
    )
    inputs = gen.build_sample_inputs(0, config, [company])

    def repair() -> dict[str, Any]:
        letter = {key: "" for key in gen.REQUIRED_KEYS}
        letter["contact_info"] = {key: "" for key in gen.CONTACT_KEYS}
        gen.fill_missing_fields(letter, inputs, random.Random("fixed"))
        return letter

    first, second = repair(), repair()
    assert first == second, "same seed must produce the same repair"
    assert first["signatory_title"], "signatory_title must be filled"
    assert "acme" in first["contact_info"]["email"], first["contact_info"]["email"]


def test_assign_signature_is_deterministic_and_partial() -> None:
    """Signature choice is seed-derived and covers only part of the corpus.

    Determinism is what lets the backfill script write the same path a fresh
    rebuild would, so old and new manifest rows agree.
    """
    pool = [Path(f"/tmp/sig/{n}.tif") for n in range(50)]

    assigned = [assign_signature(i, pool) for i in range(400)]
    assert assigned == [assign_signature(i, pool) for i in range(400)], "must be reproducible"

    signed = [path for path in assigned if path]
    assert signed, "some letters must be signed"
    assert len(signed) < len(assigned), "some letters must be unsigned"
    #: 70% target; allow slack so the test is not flaky on a finite sample.
    assert 0.6 < len(signed) / len(assigned) < 0.8, len(signed) / len(assigned)
    assert all(path.endswith(".tif") for path in signed)

    assert assign_signature(0, []) == "", "no corpus means no signature"


def test_generator_manifest_rows(tmp_path: Path) -> None:
    """The manifest lists one row per letter with six-digit sample ids."""
    config = gen.GeneratorConfig(output_dir=tmp_path, logos_csv=DEFAULT_LOGOS_CSV)
    company = gen.CompanyProfile(
        logo_id="thesvg_acme",
        slug="acme",
        title="Acme",
        categories=("Software",),
        svg_path="svg/acme.svg",
    )
    for index in range(2):
        inputs = gen.build_sample_inputs(index, config, [company])
        letter = gen.validate_letter(_letter_payload(), inputs)
        gen.write_record(
            gen.output_path_for(config, inputs),
            gen.build_record(letter, inputs, config, 1, []),
        )

    manifest = tmp_path / "generator.csv"
    assert build_generator_manifest(tmp_path, manifest) == 2

    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert tuple(rows[0]) == MANIFEST_COLUMNS
    assert rows[0]["sample_id"] == "plaintext_000000"
    assert rows[0]["logo_path"] == "svg/acme.svg"
    assert rows[0]["logo_title"] == "Acme"
    assert rows[0]["body_chars"].isdigit() and int(rows[0]["body_chars"]) > 0
    assert Path(rows[0]["sample_path"]).is_file()


def test_text_generation_end_to_end(tmp_path: Path) -> None:
    """Generate two letters against the live API and manifest them.

    Writes to ``tmp_path``, never to ``data/``. Skipped without credentials.

    Args:
        tmp_path: Pytest-provided scratch directory.
    """
    import pytest

    gen.load_dotenv_if_available()
    if not os.getenv(gen.DEFAULT_API_KEY_ENV) and not os.getenv("RUN_API_TESTS"):
        pytest.skip(f"{gen.DEFAULT_API_KEY_ENV} is not set; skipping the live API test.")
    if not DEFAULT_LOGOS_CSV.is_file():
        pytest.skip(f"{DEFAULT_LOGOS_CSV} is missing; run scripts/download_thesvg_logo.py first.")

    rows = run(
        output_dir=tmp_path,
        logos_csv=DEFAULT_LOGOS_CSV,
        sample_count=2,
        manifest_path=tmp_path / "generator.csv",
    )
    assert rows == 2


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    """Generate letters into ``data/`` and write the manifest.

    Returns:
        ``0`` on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logos-csv", type=Path, default=DEFAULT_LOGOS_CSV,
                        help="Logo manifest supplying the sender-company pool.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory the plaintext_*.json files are written to.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="CSV file joining each letter back to its logo row.")
    parser.add_argument("--count", type=int, default=DEFAULT_SAMPLE_COUNT,
                        help="Number of letters the output directory should hold.")
    args, passthrough = parser.parse_known_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logos manifest:  {args.logos_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample manifest:  {args.manifest}")

    rows = run(
        output_dir=args.output_dir,
        logos_csv=args.logos_csv,
        sample_count=args.count,
        manifest_path=args.manifest,
        extra_args=passthrough or None,
    )
    print(f"Done: {rows} row(s) in {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
