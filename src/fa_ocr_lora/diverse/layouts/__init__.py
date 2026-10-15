"""Stage 04 - layout planning: format, template, fonts, colours, degradation.

A registry of format modules, one per visual category. Each takes a cleaned
text plus a seeded RNG and returns a render spec: an HTML document, the fonts
it needs, and the degradation to apply afterwards. The variety here is the
second half of the dataset's diversity - stage 01 varies what the text *says*,
this varies what it *looks like*.
"""
