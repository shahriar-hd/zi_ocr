"""Stage 00 - harvesting the Persian lexicon.

The point of this stage is the *long tail*. An instruction-tuned model writes
the high-frequency core of Persian unprompted, so a prompt slot spent on a
common word buys no vocabulary the model has not already seen. What it never
reaches for is archaic and classical words, dialect forms, domain terminology
and the Academy's modern coinages - so that is what is harvested here, and
:attr:`~fa_ocr_lora.diverse.config.LexiconConfig.exclude_top_frequency_rank`
throws the common core away.
"""
