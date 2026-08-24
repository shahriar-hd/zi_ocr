"""Package entry point, so ``python -m fa_ocr_lora.inference`` works.

Exists to give the module-run form a single canonical spelling. Running
``python -m fa_ocr_lora.inference.predict`` imports ``predict`` twice — once as
``fa_ocr_lora.inference.predict`` while resolving the package, then again as
``__main__`` — which Python reports as a ``RuntimeWarning`` on every invocation.
The warning is harmless but it is the first thing a user sees, and a CLI whose
own start-up prints a warning does not read as finished. Dispatching from here
imports the module exactly once.
"""

from __future__ import annotations

from .predict import main

if __name__ == "__main__":
    raise SystemExit(main())
