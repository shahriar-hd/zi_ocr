"""Is the detached training run still alive? ``kill -0`` alone cannot say.

A molab run is started with ``setsid nohup ... &`` from a shell that exits at
once, so the training process is orphaned and reparented to whatever the
container runs as PID 1. When nothing reaps, an exited run stays in the process
table as a zombie: the pid still exists, ``kill -0`` still answers for it, and
every consumer that asks "is it running?" reports yes for a process that will
never run again. That is why ``scripts/train_molab.sh status`` called a run
eleven minutes finished "alive", and why the notebook's monitor cell kept
polling instead of evaluating the adapter it had just watched train.

The state field of ``/proc/<pid>/stat`` is the truth: ``Z`` means the process
exited and only its exit status is left to collect, so there is nothing left to
wait for.
"""

from __future__ import annotations

import os

__all__ = ["is_running", "pid_state"]

# /proc is Linux-only. Elsewhere there is no state field to consult and the
# signal is the only evidence available.
_HAS_PROC = os.path.exists("/proc/self/stat")


def pid_state(pid: int) -> str:
    """Classify a process id as ``"alive"``, ``"zombie"`` or ``"gone"``.

    Args:
        pid: Process id to classify.

    Returns:
        ``"alive"`` - the pid exists and can still do work; ``"zombie"`` - the
        pid exists but the process has already exited and nobody reaped it;
        ``"gone"`` - there is no such pid.

    Raises:
        ValueError: If ``pid`` is not positive. ``os.kill(0, ...)`` means the
            caller's own process group and a negative pid means every process
            the caller may signal, neither of which is a process to inspect.
    """
    if pid <= 0:
        raise ValueError(f"pid must be positive, got {pid}")
    try:
        os.kill(pid, 0)  # signal 0 = existence check, nothing is delivered
    except ProcessLookupError:
        return "gone"
    except PermissionError:
        return "alive"  # it exists, it is just not ours to signal
    if not _HAS_PROC:
        return "alive"
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            raw = handle.read()
    except FileNotFoundError:
        return "gone"  # exited between the two calls
    except OSError:
        return "alive"
    # The comm field is parenthesised and may itself contain spaces or ')', so
    # the state is whatever follows the last ')': no later field contains one.
    fields = raw.rsplit(")", 1)[-1].split()
    if not fields:
        return "alive"
    return "zombie" if fields[0] == "Z" else "alive"


def is_running(pid: int) -> bool:
    """True while ``pid`` is a process that can still do work.

    A zombie cannot: it has already run, and only its exit status is waiting to
    be collected.

    Args:
        pid: Process id to test.

    Returns:
        Whether the process is alive rather than exited-but-unreaped or absent.
    """
    return pid_state(pid) == "alive"
