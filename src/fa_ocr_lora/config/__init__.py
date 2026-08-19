"""Configuration: typed YAML, Persian content pools, and ``.env``.

This package is the single place the pipeline learns anything it was not
handed as an argument. CLAUDE.md 8 forbids hardcoded constants, and that rule
is split three ways here:

* :mod:`~fa_ocr_lora.config.config` - the typed, validated loader for
  ``configs/synth_config.yaml``: every knob that changes how a sample looks.
* :mod:`~fa_ocr_lora.config.vocabulary` - the Persian content pools from
  ``configs/vocabulary_fa.yaml``: every string that gets printed on a page.
  Kept in its own file so retuning wording never risks touching a probability.
* :mod:`~fa_ocr_lora.config.env` - ``.env`` loading and typed accessors, for
  the two things a YAML file in a git repository must not hold: machine
  specific paths, and secrets.

Precedence is fixed and applies to every setting that appears in more than one
of them::

    CLI flag  >  environment  >  YAML  >  dataclass default

Environment overrides are applied *after* the YAML is parsed, so a bad value
from ``.env`` fails the same validators a bad value in the file would.

Import direction is one-way: this package may read
``fa_ocr_lora.dataset.common.io_utils``, and nothing under ``dataset/`` is
imported from here beyond that, so there is no cycle.
"""

from __future__ import annotations

from .config import (
    BAR_STYLES,
    CHART_KINDS,
    CONFIG_PATH_ENV_VAR,
    CONTACT_FIELDS,
    CSS_PPI,
    DATA_DIR_ENV_VAR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_VOCABULARY_PATH,
    FOOTER_LAYOUT_STYLES,
    HF_COMPRESSIONS,
    HF_FAMILIES,
    HF_GROUP_KEYS,
    HF_OUTPUT_DIR_ENV_VAR,
    HF_PRIVATE_ENV_VAR,
    HF_REPO_ID_ENV_VAR,
    HF_SPLITS,
    HF_SPLITS_ENV_VAR,
    HF_SPLIT_TOLERANCE,
    HF_TOKEN_ENV_VARS,
    IMAGE_FORMAT_CHOICES,
    METADATA_FIELDS,
    PAGE_SIDES,
    PATH_ENV_VARS,
    SEED_ENV_VAR,
    STAMP_ROUND_SHAPES,
    STAMP_SHAPES,
    SUBDIR_FIELDS,
    VOCABULARY_PATH_ENV_VAR,
    ChartConfig,
    ConfigError,
    ElementsConfig,
    FontSpec,
    HuggingFaceConfig,
    ImageFormatConfig,
    IntRange,
    LayoutConfig,
    PageSizeSpec,
    PathsConfig,
    ProbabilityConfig,
    Range,
    RenderConfig,
    ScanConfig,
    SignatureConfig,
    StampConfig,
    SynthConfig,
    TableConfig,
    ThemeConfig,
    Vocabulary,
    load_config,
    load_synth_config,
    load_vocabulary,
    normalize_split_fractions,
    parse_split_spec,
)
from .env import (
    EnvError,
    ensure_dotenv_loaded,
    env_bool,
    env_float,
    env_int,
    env_list,
    env_path,
    env_str,
    load_dotenv,
)
from .vocabulary import (
    ChartVocabulary,
    ElementsVocabulary,
    HeaderVocabulary,
    LetterVocabulary,
    MetadataVocabulary,
    StampVocabulary,
    TableVocabulary,
)

__all__ = [
    "BAR_STYLES",
    "CHART_KINDS",
    "CONFIG_PATH_ENV_VAR",
    "CONTACT_FIELDS",
    "CSS_PPI",
    "DATA_DIR_ENV_VAR",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_VOCABULARY_PATH",
    "FOOTER_LAYOUT_STYLES",
    "HF_COMPRESSIONS",
    "HF_FAMILIES",
    "HF_GROUP_KEYS",
    "HF_OUTPUT_DIR_ENV_VAR",
    "HF_PRIVATE_ENV_VAR",
    "HF_REPO_ID_ENV_VAR",
    "HF_SPLITS",
    "HF_SPLITS_ENV_VAR",
    "HF_SPLIT_TOLERANCE",
    "HF_TOKEN_ENV_VARS",
    "IMAGE_FORMAT_CHOICES",
    "METADATA_FIELDS",
    "PAGE_SIDES",
    "PATH_ENV_VARS",
    "SEED_ENV_VAR",
    "STAMP_ROUND_SHAPES",
    "STAMP_SHAPES",
    "SUBDIR_FIELDS",
    "VOCABULARY_PATH_ENV_VAR",
    "ChartConfig",
    "ChartVocabulary",
    "ConfigError",
    "ElementsConfig",
    "ElementsVocabulary",
    "EnvError",
    "FontSpec",
    "HeaderVocabulary",
    "HuggingFaceConfig",
    "ImageFormatConfig",
    "IntRange",
    "LayoutConfig",
    "LetterVocabulary",
    "MetadataVocabulary",
    "PageSizeSpec",
    "PathsConfig",
    "ProbabilityConfig",
    "Range",
    "RenderConfig",
    "ScanConfig",
    "SignatureConfig",
    "StampConfig",
    "StampVocabulary",
    "SynthConfig",
    "TableConfig",
    "TableVocabulary",
    "ThemeConfig",
    "Vocabulary",
    "ensure_dotenv_loaded",
    "env_bool",
    "env_float",
    "env_int",
    "env_list",
    "env_path",
    "env_str",
    "load_config",
    "load_dotenv",
    "load_synth_config",
    "load_vocabulary",
    "normalize_split_fractions",
    "parse_split_spec",
]
