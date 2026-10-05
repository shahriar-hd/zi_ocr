"""Tests for ``fa_ocr_lora.utils.process``: what "still running" actually means.

The load-bearing case is the zombie. A molab run is orphaned when the shell that
started it exits, and the container's PID 1 does not reap, so the run stays in
the process table after it is done - `kill -0` succeeds, and everything built on
that answered "alive" for a process that had finished hours earlier. Each test
here builds the state it asserts on rather than trusting a fixture pid.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from fa_ocr_lora.utils.process import is_running, pid_state

_HAS_PROC = os.path.exists("/proc/self/stat")


def _stat_state(pid: int) -> str:
    """Read the state field of ``/proc/<pid>/stat`` without reaping anything."""
    with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
        return handle.read().rsplit(")", 1)[-1].split()[0]


def test_own_pid_is_alive() -> None:
    """The process asking the question is, by definition, running."""
    assert pid_state(os.getpid()) == "alive"
    assert is_running(os.getpid()) is True


def test_reaped_pid_is_gone() -> None:
    """A pid that was collected is no longer in the table at all."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    pid = proc.pid
    proc.wait()
    assert pid_state(pid) == "gone"
    assert is_running(pid) is False


@pytest.mark.skipif(not _HAS_PROC, reason="no /proc to read the state field from")
def test_unreaped_child_is_a_zombie_and_not_running() -> None:
    """An exited-but-unreaped child answers `kill -0` yet cannot do any work.

    This is the state the monitor cell was stuck in: the signal says the pid
    exists, ``pid_state`` must not be fooled by it.
    """
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    deadline = time.time() + 30
    while time.time() < deadline:
        if _stat_state(proc.pid) == "Z":
            break
        time.sleep(0.01)
    else:
        proc.wait()
        pytest.fail("child never exited; could not construct a zombie")

    assert pid_state(proc.pid) == "zombie"
    assert is_running(proc.pid) is False
    # What the old check trusted - the pid does exist, it just will never run.
    os.kill(proc.pid, 0)
    proc.wait()


@pytest.mark.skipif(not _HAS_PROC, reason="no /proc to read the state field from")
def test_thread_state_is_alive() -> None:
    """A live child is reported as running while it is actually executing."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert _stat_state(proc.pid) in {"S", "R", "D"}
        assert pid_state(proc.pid) == "alive"
        assert is_running(proc.pid) is True
    finally:
        proc.kill()
        proc.wait()


def test_non_positive_pid_is_refused() -> None:
    """0 would mean this process group and -1 every process we may signal."""
    with pytest.raises(ValueError, match="positive"):
        pid_state(0)
    with pytest.raises(ValueError, match="positive"):
        pid_state(-1)
