"""Stage 03 - text QC and the two-field text policy.

Decides, once, what the label says. ``text_render`` is exactly the character
sequence the renderer draws; ``text_norm`` is the evaluation form. Keeping both
is what stops a model being punished for a distinction no reader makes, and
stops the label drifting from the pixels.

The filters here are not cosmetic. The first live generation run leaked Cyrillic
and CJK into Persian prose in a third of samples despite an explicit
Persian-only instruction, and one such sample still measured 80.6% Persian
letters - so contamination is rejected as a *kind*, not a quantity.
"""
