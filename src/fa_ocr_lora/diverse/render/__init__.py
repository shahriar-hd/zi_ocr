"""Stage 05 - rendering, annotation harvest and degradation.

Chromium through Playwright does the shaping. That is not a convenience: it is
the only engine in reach that gets Persian bidi, joining, ZWNJ and mixed-script
runs right, and Pillow here is not built with libraqm so it would draw
disconnected letters.

Annotations come from the renderer, never from guessing: ``harvest.js`` reads
``Range.getClientRects()`` per text node, so a line's box is where the browser
actually put it. Any geometric degradation is then applied to the pixels and
the polygons through the same mapping, which is what keeps the ground truth
attached to the ink.
"""
