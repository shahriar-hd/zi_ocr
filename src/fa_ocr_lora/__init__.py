"""PaddleOCR-VL fine-tuning and inference for Persian administrative text.

Importing anything below this package runs :func:`_align_transformers_with_tensorflow`
first, so the training modules stay importable on hosts whose TensorFlow is newer
than transformers expects.
"""

from __future__ import annotations

import importlib.util
import os
import warnings

__all__: list[str] = []


def _align_transformers_with_tensorflow() -> None:
    """Make ``from transformers import Trainer`` survivable with Keras 3 installed.

    transformers 4.57 imports its TensorFlow stack as soon as TensorFlow is
    importable, then refuses Keras 3 (it wants the backwards-compatible
    ``tf-keras``), so the import raises before a model is ever loaded. Nothing here
    uses TensorFlow - this is a torch + PEFT trainer - so when the Keras 2 shim is
    missing the framework is switched off **for transformers only**: TensorFlow stays
    installed and importable on its own.

    Installing ``tf-keras`` removes the reason for this at the source, and
    ``scripts/train_molab.sh`` does exactly that (pinned to the installed TensorFlow
    minor) before any stage runs; the guard then returns without touching anything.

    Returns:
        None. The environment of the process is updated in place.
    """
    if os.environ.get("USE_TF"):
        return  # a caller - a driver script or the kernel - already decided
    if importlib.util.find_spec("tensorflow") is None:
        return  # transformers never looks for tf_keras without TensorFlow
    if importlib.util.find_spec("tf_keras") is not None:
        return  # Keras 2 shim present: transformers can use TensorFlow as designed
    os.environ["USE_TF"] = "0"
    os.environ.setdefault("USE_TORCH", "1")
    # FORCE_TF_AVAILABLE outranks USE_TF, so it would undo the line above.
    os.environ.pop("FORCE_TF_AVAILABLE", None)
    warnings.warn(
        "TensorFlow is installed without tf_keras, which transformers 4.57 refuses "
        "(Keras 3). USE_TF=0 keeps this package importable; install tf-keras to give "
        "transformers its TensorFlow support back - see scripts/train_molab.sh env.",
        RuntimeWarning,
        stacklevel=2,
    )


_align_transformers_with_tensorflow()
