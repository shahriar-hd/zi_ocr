"""Rendering back end: Jinja templates, embedded fonts, and Chromium.

This subpackage is the only place that knows *how* a letter becomes pixels.
Everything above it - the layout and element modules - produces HTML fragments
and structured content and never touches a browser, which is what keeps those
modules testable without Playwright installed.
"""

from __future__ import annotations
