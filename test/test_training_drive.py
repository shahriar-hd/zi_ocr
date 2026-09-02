"""Unit tests for the Google Drive checkpoint mirror.

None of these touch the network. A fake in-memory Drive stands in for
:class:`~fa_ocr_lora.drive.DriveClient`, which is enough because the properties
worth defending are all about *ordering and lifetime*, not about HTTP:

**A half-uploaded checkpoint must never be resumed from.** The ``LATEST`` marker
is the only thing resume trusts, so it must appear strictly after every file of a
checkpoint has landed - and must not appear at all when one of them failed.

**The uploader must survive the trainer deleting the checkpoint underneath it.**
``save_total_limit`` unlinks old checkpoint directories the moment a new one is
written, which on a slow uplink happens while the upload is still reading. Staging
by hard link is what makes that safe, and the test deletes the original to prove
it.

**An upload failure must not kill training.** Six hours of GPU time is worth more
than one checkpoint copy, so a failure is counted and logged, never raised.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import pytest

from fa_ocr_lora.config.base import ConfigError
from fa_ocr_lora.drive import DriveError
from fa_ocr_lora.training.checkpoint_sync import (
    LATEST_MARKER,
    STAGING_DIRNAME,
    CheckpointSync,
    _checkpoint_step,
    _mime_type,
)
from fa_ocr_lora.training.config import DriveConfig
from fa_ocr_lora.training.loop import DriveSyncCallback, run_log_file


class FakeDrive:
    """An in-memory stand-in for :class:`~fa_ocr_lora.drive.DriveClient`.

    Implements only the surface :class:`CheckpointSync` uses. Guarded by a lock
    because ``_upload_many`` calls it from a thread pool, so an unsynchronised
    dict would make the tests flaky rather than wrong.
    """

    def __init__(self, *, fail_on: set[str] | None = None) -> None:
        """Build an empty Drive whose uploads fail for the named files."""
        self.folders: dict[str, dict[str, object]] = {
            "root": {"name": "root", "parent": "", "files": {}}
        }
        self.fail_on = fail_on or set()
        self.uploads: list[str] = []
        self.deleted: list[str] = []
        self._next = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------- helpers
    def _id(self, prefix: str) -> str:
        self._next += 1
        return f"{prefix}{self._next}"

    def _files(self, folder_id: str) -> dict[str, tuple[str, int, bytes]]:
        return self.folders[folder_id]["files"]  # type: ignore[return-value]

    def _child(self, parent: str, name: str) -> str | None:
        for fid, meta in self.folders.items():
            if meta["parent"] == parent and meta["name"] == name:
                return fid
        return None

    # -------------------------------------------------------- DriveClient API
    def ensure_folder(self, relative_path: str, root_id: str) -> str:
        current = root_id
        for segment in [p for p in relative_path.split("/") if p]:
            with self._lock:
                found = self._child(current, segment)
                if found is None:
                    found = self._id("folder-")
                    self.folders[found] = {
                        "name": segment,
                        "parent": current,
                        "files": {},
                    }
            current = found
        return current

    def find_folder(self, relative_path: str, root_id: str) -> str | None:
        current = root_id
        for segment in [p for p in relative_path.split("/") if p]:
            current = self._child(current, segment)  # type: ignore[assignment]
            if current is None:
                return None
        return current

    def list_folder(self, folder_id: str) -> dict[str, tuple[str, int]]:
        with self._lock:
            return {
                name: (fid, size)
                for name, (fid, size, _) in self._files(folder_id).items()
            }

    def list_subfolders(self, folder_id: str) -> dict[str, str]:
        return {
            meta["name"]: fid  # type: ignore[misc]
            for fid, meta in self.folders.items()
            if meta["parent"] == folder_id
        }

    def upload(
        self,
        path: Path,
        name: str,
        folder_id: str,
        mime_type: str = "application/octet-stream",
        existing_id: str | None = None,
    ) -> str:
        if name in self.fail_on:
            raise DriveError(f"synthetic failure uploading {name}")
        payload = path.read_bytes()
        with self._lock:
            file_id = existing_id or self._id("file-")
            self._files(folder_id)[name] = (file_id, len(payload), payload)
            self.uploads.append(name)
        return file_id

    def download(self, file_id: str, destination: Path) -> int:
        for meta in self.folders.values():
            for _name, (fid, size, payload) in meta["files"].items():  # type: ignore[union-attr]
                if fid == file_id:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(payload)
                    return size
        raise DriveError(f"no such file {file_id}")

    def read_text(self, file_id: str) -> str:
        for meta in self.folders.values():
            for _name, (fid, _size, payload) in meta["files"].items():  # type: ignore[union-attr]
                if fid == file_id:
                    return payload.decode("utf-8")
        raise DriveError(f"no such file {file_id}")

    def delete(self, file_id: str) -> None:
        self.deleted.append(file_id)
        self.folders.pop(file_id, None)


def make_checkpoint(root: Path, step: int, *, optimizer: bool = True) -> Path:
    """Write a plausible checkpoint directory and return it."""
    directory = root / f"checkpoint-{step}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_model.safetensors").write_bytes(b"w" * 64)
    (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
    (directory / "trainer_state.json").write_text(
        f'{{"global_step": {step}}}', encoding="utf-8"
    )
    if optimizer:
        (directory / "optimizer.pt").write_bytes(b"o" * 256)
        (directory / "scheduler.pt").write_bytes(b"s" * 16)
    return directory


def start_sync(tmp_path: Path, fake: FakeDrive, **kwargs: object) -> CheckpointSync:
    """Build a sync bound to the fake Drive, bypassing authentication.

    Mirrors what :meth:`CheckpointSync.start` does apart from the credential
    round-trip, which is the one part that cannot run offline.
    """
    sync = CheckpointSync(
        run_name="run-a",
        output_dir=tmp_path / "out",
        root_folder="fa-ocr-lora",
        **kwargs,  # type: ignore[arg-type]
    )
    sync._client = fake  # type: ignore[assignment]
    root = fake.ensure_folder("fa-ocr-lora", "root")
    sync._run_folder_id = fake.ensure_folder("run-a", root)
    sync._thread = threading.Thread(target=sync._work, daemon=True)
    sync._thread.start()
    return sync


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_drive_config_defaults_to_off_and_non_interactive():
    config = DriveConfig.parse({})
    assert config.enabled is False
    # False is the load-bearing default: run_local_server() on a headless box
    # binds a port nobody can reach and hangs until the run is killed.
    assert config.interactive_auth is False
    assert config.resume_from_drive is True


def test_drive_config_rejects_an_unknown_key():
    with pytest.raises(ConfigError) as excinfo:
        DriveConfig.parse({"enabeld": True})
    assert "drive" in str(excinfo.value)
    assert "enabeld" in str(excinfo.value)


def test_drive_config_rejects_a_negative_worker_count():
    with pytest.raises(ConfigError):
        DriveConfig.parse({"workers": 0})


def test_run_name_falls_back_to_the_output_directory_name():
    # Without this a smoke test writing to runs/smoke would overwrite an
    # overnight run's checkpoints in Drive.
    config = DriveConfig.parse({})
    assert config.resolved_run_name(Path("runs/lora-a100")) == "lora-a100"


def test_an_explicit_run_name_wins_over_the_directory():
    config = DriveConfig.parse({"run_name": "nightly"})
    assert config.resolved_run_name(Path("runs/lora-a100")) == "nightly"


def test_optional_paths_accept_null_and_resolve_when_given():
    assert DriveConfig.parse({"token_path": None}).token_path is None
    resolved = DriveConfig.parse({"token_path": "secrets/t.json"}).token_path
    assert resolved is not None
    assert resolved.is_absolute()


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("checkpoint-1200", 1200),
        ("checkpoint-0", 0),
        ("adapter_only", -1),
        ("checkpoint-final", -1),
        (LATEST_MARKER, -1),
    ],
)
def test_checkpoint_step_parses_the_trainer_naming_convention(name, expected):
    assert _checkpoint_step(name) == expected


def test_mime_type_falls_back_to_opaque_bytes():
    assert _mime_type(Path("adapter_model.safetensors")) == "application/octet-stream"
    assert _mime_type(Path("trainer_state.json")) == "application/json"


# --------------------------------------------------------------------------- #
# Staging - the save_total_limit race
# --------------------------------------------------------------------------- #


def test_staging_survives_the_trainer_deleting_the_checkpoint(tmp_path):
    fake = FakeDrive()
    sync = CheckpointSync(run_name="run-a", output_dir=tmp_path / "out")
    checkpoint = make_checkpoint(tmp_path / "ckpts", 100)

    staged = sync._stage(checkpoint)
    # This is exactly what save_total_limit does, and it is why staging exists.
    import shutil

    shutil.rmtree(checkpoint)

    assert not checkpoint.exists()
    assert (staged / "adapter_model.safetensors").read_bytes() == b"w" * 64
    assert sorted(p.name for p in staged.iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
    ]
    assert fake.uploads == []


def test_staging_uses_hard_links_rather_than_copies(tmp_path):
    # Not cosmetic: a 1.5 GB checkpoint copied every 500 steps would spend more
    # disk bandwidth than the training reads.
    sync = CheckpointSync(run_name="run-a", output_dir=tmp_path / "out")
    checkpoint = make_checkpoint(tmp_path / "ckpts", 100)
    staged = sync._stage(checkpoint)
    original = checkpoint / "adapter_model.safetensors"
    assert (staged / "adapter_model.safetensors").stat().st_ino == original.stat().st_ino


def test_staging_can_drop_optimizer_state(tmp_path):
    sync = CheckpointSync(
        run_name="run-a", output_dir=tmp_path / "out", upload_optimizer_state=False
    )
    staged = sync._stage(make_checkpoint(tmp_path / "ckpts", 100))
    names = sorted(p.name for p in staged.iterdir())
    assert "optimizer.pt" not in names
    assert "scheduler.pt" not in names
    assert "adapter_model.safetensors" in names


def test_staging_mirrors_nested_directories(tmp_path):
    sync = CheckpointSync(run_name="run-a", output_dir=tmp_path / "out")
    checkpoint = make_checkpoint(tmp_path / "ckpts", 100)
    (checkpoint / "extra").mkdir()
    (checkpoint / "extra" / "tokenizer.json").write_text("{}", encoding="utf-8")
    staged = sync._stage(checkpoint)
    assert (staged / "extra" / "tokenizer.json").is_file()


# --------------------------------------------------------------------------- #
# Upload ordering - the marker
# --------------------------------------------------------------------------- #


def test_a_checkpoint_is_mirrored_and_then_published(tmp_path):
    fake = FakeDrive()
    sync = start_sync(tmp_path, fake)
    sync.enqueue(make_checkpoint(tmp_path / "ckpts", 200))
    sync.close(timeout=10)

    run_folder = fake.find_folder("fa-ocr-lora/run-a", "root")
    remote = fake.find_folder("checkpoint-200", run_folder)
    assert sorted(fake.list_folder(remote)) == [
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
    ]
    marker = fake.list_folder(run_folder)[LATEST_MARKER]
    assert fake.read_text(marker[0]).strip() == "checkpoint-200"
    assert sync.stats.checkpoints_uploaded == 1
    assert sync.stats.files_uploaded == 5
    assert sync.stats.failures == 0


def test_the_marker_is_written_after_the_files_not_before(tmp_path):
    fake = FakeDrive()
    sync = start_sync(tmp_path, fake)
    sync.enqueue(make_checkpoint(tmp_path / "ckpts", 200))
    sync.close(timeout=10)
    # Resume reads the marker, so its position in this list is the whole
    # guarantee: an instance that dies mid-upload leaves the previous complete
    # checkpoint as the newest one Drive admits to.
    assert fake.uploads[-1] == LATEST_MARKER
    assert LATEST_MARKER not in fake.uploads[:-1]


def test_a_failed_file_leaves_no_marker_and_does_not_raise(tmp_path):
    fake = FakeDrive(fail_on={"optimizer.pt"})
    sync = start_sync(tmp_path, fake)
    sync.enqueue(make_checkpoint(tmp_path / "ckpts", 200))
    sync.close(timeout=10)

    run_folder = fake.find_folder("fa-ocr-lora/run-a", "root")
    assert LATEST_MARKER not in fake.list_folder(run_folder)
    assert sync.stats.failures == 1
    assert sync.stats.checkpoints_uploaded == 0


def test_publish_false_mirrors_without_becoming_a_resume_target(tmp_path):
    fake = FakeDrive()
    sync = start_sync(tmp_path, fake)
    sync.enqueue(make_checkpoint(tmp_path / "ckpts", 200))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"final")
    sync.enqueue(adapter, publish=False)
    sync.close(timeout=10)

    run_folder = fake.find_folder("fa-ocr-lora/run-a", "root")
    assert fake.find_folder("adapter", run_folder) is not None
    marker = fake.list_folder(run_folder)[LATEST_MARKER]
    # Still the checkpoint. Publishing the adapter would point the next resume at
    # a directory with no trainer state.
    assert fake.read_text(marker[0]).strip() == "checkpoint-200"


def test_an_identical_remote_file_is_skipped_on_re_upload(tmp_path):
    fake = FakeDrive()
    sync = start_sync(tmp_path, fake)
    checkpoint = make_checkpoint(tmp_path / "ckpts", 200)
    sync.enqueue(checkpoint)
    sync.close(timeout=10)
    first = sync.stats.files_uploaded

    sync2 = start_sync(tmp_path, fake)
    sync2.enqueue(checkpoint)
    sync2.close(timeout=10)
    assert first == 5
    assert sync2.stats.files_uploaded == 0
    assert sync2.stats.files_skipped == 5


def test_the_log_is_mirrored_alongside_the_checkpoint(tmp_path):
    fake = FakeDrive()
    log = tmp_path / "train.log"
    log.write_text("step 1\n", encoding="utf-8")
    sync = start_sync(tmp_path, fake, log_path=log)
    sync.enqueue(make_checkpoint(tmp_path / "ckpts", 200))
    sync.close(timeout=10)

    run_folder = fake.find_folder("fa-ocr-lora/run-a", "root")
    logs = fake.find_folder("logs", run_folder)
    assert logs is not None
    assert "train.log" in fake.list_folder(logs)


def test_staged_leftovers_from_a_killed_run_are_re_queued(tmp_path):
    # Staging survives kill -9, and a directory still there is by definition one
    # whose upload never finished.
    fake = FakeDrive()
    staging = tmp_path / "out" / STAGING_DIRNAME
    orphan = staging / "checkpoint-50"
    orphan.mkdir(parents=True)
    (orphan / "adapter_model.safetensors").write_bytes(b"x" * 8)

    sync = start_sync(tmp_path, fake)
    sync._requeue_stale()
    sync.close(timeout=10)

    run_folder = fake.find_folder("fa-ocr-lora/run-a", "root")
    assert fake.find_folder("checkpoint-50", run_folder) is not None


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #


def test_pruning_keeps_the_newest_remote_checkpoints(tmp_path):
    fake = FakeDrive()
    sync = start_sync(tmp_path, fake, keep_remote=2)
    for step in (100, 200, 300, 400):
        sync.enqueue(make_checkpoint(tmp_path / "ckpts", step))
    sync.close(timeout=20)

    run_folder = fake.find_folder("fa-ocr-lora/run-a", "root")
    remaining = sorted(fake.list_subfolders(run_folder))
    assert remaining == ["checkpoint-300", "checkpoint-400"]


def test_keep_remote_zero_keeps_everything(tmp_path):
    fake = FakeDrive()
    sync = start_sync(tmp_path, fake, keep_remote=0)
    for step in (100, 200, 300):
        sync.enqueue(make_checkpoint(tmp_path / "ckpts", step))
    sync.close(timeout=20)

    run_folder = fake.find_folder("fa-ocr-lora/run-a", "root")
    assert len(fake.list_subfolders(run_folder)) == 3


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #


def test_fetch_resume_downloads_the_published_checkpoint(tmp_path):
    fake = FakeDrive()
    sync = start_sync(tmp_path, fake)
    sync.enqueue(make_checkpoint(tmp_path / "ckpts", 200))
    sync.enqueue(make_checkpoint(tmp_path / "ckpts", 400))
    sync.close(timeout=20)

    fresh = start_sync(tmp_path, fake)
    destination = tmp_path / "fresh-instance"
    destination.mkdir()
    resumed = fresh.fetch_resume(destination)
    fresh.close(timeout=10)

    assert resumed == destination / "checkpoint-400"
    assert (resumed / "adapter_model.safetensors").read_bytes() == b"w" * 64
    assert (resumed / "optimizer.pt").read_bytes() == b"o" * 256


def test_fetch_resume_reuses_local_files_of_the_right_size(tmp_path):
    fake = FakeDrive()
    sync = start_sync(tmp_path, fake)
    sync.enqueue(make_checkpoint(tmp_path / "ckpts", 200))
    sync.close(timeout=10)

    destination = tmp_path / "partial"
    target = destination / "checkpoint-200"
    target.mkdir(parents=True)
    (target / "optimizer.pt").write_bytes(b"o" * 256)  # already the right size

    fresh = start_sync(tmp_path, fake)
    resumed = fresh.fetch_resume(destination)
    fresh.close(timeout=10)

    assert resumed is not None
    # The size-matched file was not re-fetched, the four others were.
    assert (target / "adapter_model.safetensors").is_file()


def test_fetch_resume_returns_none_when_drive_holds_nothing(tmp_path):
    fake = FakeDrive()
    sync = start_sync(tmp_path, fake)
    resumed = sync.fetch_resume(tmp_path / "fresh")
    sync.close(timeout=10)
    assert resumed is None


def test_fetch_resume_falls_back_to_the_highest_numbered_directory(tmp_path):
    # A run that predates the marker, or one killed before its first publish.
    fake = FakeDrive()
    run_folder = fake.ensure_folder("fa-ocr-lora/run-a", "root")
    for step in (100, 900, 300):
        folder = fake.ensure_folder(f"checkpoint-{step}", run_folder)
        source = tmp_path / "src"
        source.mkdir(exist_ok=True)
        blob = source / "adapter_model.safetensors"
        blob.write_bytes(b"z" * 8)
        fake.upload(blob, blob.name, folder)

    sync = start_sync(tmp_path, fake)
    resumed = sync.fetch_resume(tmp_path / "fresh")
    sync.close(timeout=10)
    assert resumed is not None
    assert resumed.name == "checkpoint-900"


def test_a_marker_naming_a_missing_folder_does_not_resume(tmp_path):
    fake = FakeDrive()
    sync = start_sync(tmp_path, fake)
    sync.enqueue(make_checkpoint(tmp_path / "ckpts", 200))
    sync.close(timeout=10)

    run_folder = fake.find_folder("fa-ocr-lora/run-a", "root")
    fake.delete(fake.find_folder("checkpoint-200", run_folder))

    fresh = start_sync(tmp_path, fake)
    resumed = fresh.fetch_resume(tmp_path / "fresh")
    fresh.close(timeout=10)
    assert resumed is None


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_client_access_before_start_says_so():
    sync = CheckpointSync(run_name="run-a", output_dir=Path("out"))
    with pytest.raises(DriveError, match="start"):
        _ = sync.client


def test_enqueue_before_start_is_a_no_op(tmp_path):
    sync = CheckpointSync(run_name="run-a", output_dir=tmp_path / "out")
    sync.enqueue(make_checkpoint(tmp_path / "ckpts", 100))
    assert sync.stats.checkpoints_uploaded == 0
    assert not (tmp_path / "out" / STAGING_DIRNAME).exists()


def test_close_is_idempotent(tmp_path):
    fake = FakeDrive()
    sync = start_sync(tmp_path, fake)
    sync.close(timeout=10)
    sync.close(timeout=10)  # run_training's finally block reaches this
    assert sync._thread is None


# --------------------------------------------------------------------------- #
# Run log
# --------------------------------------------------------------------------- #


def test_run_log_file_captures_records_and_detaches(tmp_path):
    root = logging.getLogger()
    before = len(root.handlers)
    with run_log_file(tmp_path) as path:
        logging.getLogger("fa_ocr_lora.test").info("peak 30089 MiB")
        assert len(root.handlers) == before + 1
    assert len(root.handlers) == before
    assert "peak 30089 MiB" in path.read_text(encoding="utf-8")
    assert path == tmp_path / "logs" / "train.log"


def test_run_log_file_appends_so_a_resume_keeps_its_history(tmp_path):
    with run_log_file(tmp_path) as path:
        logging.getLogger("fa_ocr_lora.test").info("first run")
    with run_log_file(tmp_path):
        logging.getLogger("fa_ocr_lora.test").info("resumed run")
    text = path.read_text(encoding="utf-8")
    assert "first run" in text
    assert "resumed run" in text


# --------------------------------------------------------------------------- #
# The Trainer hook
# --------------------------------------------------------------------------- #


class RecordingSync:
    """Captures what the callback would hand to a real mirror."""

    def __init__(self) -> None:
        """Start with nothing enqueued."""
        self.calls: list[tuple[Path, bool]] = []

    def enqueue(self, checkpoint_dir: Path, *, publish: bool = True) -> None:
        """Record the call instead of uploading."""
        self.calls.append((checkpoint_dir, publish))


class _State:
    """The one Trainer state attribute the callback reads."""

    def __init__(self, global_step: int) -> None:
        """Bind the step number."""
        self.global_step = global_step


def test_on_save_enqueues_the_checkpoint_for_this_step(tmp_path):
    # This single hook is what covers both save_strategy: steps and
    # save_strategy: epoch - Trainer calls it after _save_checkpoint either way.
    recorder = RecordingSync()
    make_checkpoint(tmp_path, 1500)
    callback = DriveSyncCallback(recorder, tmp_path)  # type: ignore[arg-type]
    callback.on_save(None, _State(1500), None)
    assert recorder.calls == [(tmp_path / "checkpoint-1500", True)]


def test_on_save_ignores_a_checkpoint_that_is_not_there(tmp_path):
    recorder = RecordingSync()
    callback = DriveSyncCallback(recorder, tmp_path)  # type: ignore[arg-type]
    callback.on_save(None, _State(99), None)  # nothing written for step 99
    assert recorder.calls == []
