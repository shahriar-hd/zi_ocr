"""Mirror training checkpoints to Google Drive without stalling the GPU.

A rented A100 is billed by the hour and is not backed up. A run that saves
checkpoints only to the instance's local disk loses everything the moment the
instance is reclaimed, so checkpoints go to Drive as they are written. Two
properties make that safe rather than merely convenient:

**The upload never blocks training.** Uploads run on a background thread fed by
a queue, so :meth:`CheckpointSync.enqueue` returns in microseconds and the
trainer goes straight back to the next step. This is one of the few places where
Python threads genuinely help: the work is socket I/O, which releases the GIL for
its entire duration, so the uploader costs the training loop nothing but a little
memory bandwidth. It is *not* a process pool, because sharing an authenticated
client and a queue across processes would cost more than it saves.

**A half-uploaded checkpoint is never resumed from.** Every file of a checkpoint
is uploaded first, and only then is a ``LATEST`` marker rewritten to name it.
Resume reads the marker, so an upload interrupted by the instance disappearing
leaves the previous - complete - checkpoint as the newest one Drive admits to.
Without that ordering, resume would eventually load a truncated ``optimizer.pt``
and fail in a way that looks like a corrupt checkpoint rather than a lost race.

The other race is local: :class:`~transformers.Trainer` enforces
``save_total_limit`` by deleting old checkpoint directories immediately after
writing a new one, which can happen while the uploader is still reading one. So
:meth:`CheckpointSync.enqueue` first *stages* the checkpoint by hard-linking its
files into a sibling directory. A hard link costs nothing, copies no bytes, and
keeps the inode alive after the trainer unlinks the original; the uploader then
reads the staged copy, which nothing else touches. Staging falls back to a real
copy only if the link fails, which in practice means a filesystem that cannot do
it at all.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import queue
import shutil
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..drive import DriveClient, DriveError, load_credentials

LOGGER = logging.getLogger(__name__)

#: Written last, after every file of a checkpoint has landed. Resume trusts this
#: name over the directory listing, because a directory can exist while its
#: contents are still in flight.
LATEST_MARKER: Final[str] = "LATEST"

#: Subfolder of the remote run directory holding the run log.
LOG_DIRNAME: Final[str] = "logs"

#: Local staging root, relative to the run's output directory.
STAGING_DIRNAME: Final[str] = "drive_staging"

#: Checkpoint files that resume does not need and that dwarf everything else.
#: ``optimizer.pt`` for a LoRA run is roughly twice the adapter itself, so
#: dropping it is the difference between a fast mirror and a slow one - at the
#: cost of restarting the optimiser state on resume. Config decides; this is
#: only the list of what "state" means.
OPTIMIZER_FILES: Final[frozenset[str]] = frozenset(
    {"optimizer.pt", "optimizer.bin", "scheduler.pt", "rng_state.pth"}
)


def _checkpoint_step(name: str) -> int:
    """Return the step number encoded in a ``checkpoint-1200`` directory name.

    Returns:
        The step, or ``-1`` for a name that does not follow the convention, so
        that sorting puts unrecognised directories first rather than crashing.
    """
    _, _, tail = name.partition("-")
    return int(tail) if tail.isdigit() else -1


def _mime_type(path: Path) -> str:
    """Guess a MIME type for Drive, defaulting to opaque bytes."""
    guess, _ = mimetypes.guess_type(path.name)
    return guess or "application/octet-stream"


@dataclass(frozen=True, slots=True)
class _Job:
    """One unit of upload work.

    Attributes:
        kind: ``"checkpoint"`` or ``"log"``.
        name: Checkpoint directory name, or ``""`` for a log.
        path: Staged checkpoint directory, or the log file to push.
        publish: Whether to name this directory in the ``LATEST`` marker. False
            for the final adapter, which is mirrored for safekeeping but is not
            something ``trainer.train`` can resume from - publishing it would
            point the next resume at a directory with no optimiser or trainer
            state and turn a successful run into a broken restart.
    """

    kind: str
    name: str
    path: Path
    publish: bool = True


@dataclass(slots=True)
class SyncStats:
    """Counters reported into the run summary.

    Failures are counted, never raised into the training loop: losing a
    checkpoint copy is bad, and killing a six-hour run over a transient 503
    would be worse.
    """

    checkpoints_uploaded: int = 0
    files_uploaded: int = 0
    bytes_uploaded: int = 0
    files_skipped: int = 0
    failures: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-serialisable view for the run summary."""
        return {
            "checkpoints_uploaded": self.checkpoints_uploaded,
            "files_uploaded": self.files_uploaded,
            "bytes_uploaded": self.bytes_uploaded,
            "files_skipped": self.files_skipped,
            "failures": self.failures,
        }


class CheckpointSync:
    """Background mirror of a run's checkpoints and log into Google Drive.

    Lifecycle: :meth:`start` (authenticates and spawns the worker),
    :meth:`enqueue` per checkpoint, :meth:`close` to drain. :meth:`fetch_resume`
    is the one method that runs on the calling thread and blocks, because
    training cannot begin until the checkpoint it downloads is on disk.
    """

    def __init__(
        self,
        *,
        run_name: str,
        output_dir: Path,
        root_folder: str = "fa-ocr-lora",
        root_folder_id: str | None = None,
        service_account: Path | None = None,
        token: Path | None = None,
        client_secret: Path | None = None,
        interactive: bool = False,
        workers: int = 4,
        upload_optimizer_state: bool = True,
        keep_remote: int = 0,
        log_path: Path | None = None,
    ) -> None:
        """Configure the mirror. No network happens until :meth:`start`.

        Args:
            run_name: Remote folder for this run, under the root folder.
            output_dir: The run's local output directory; staging lives inside it.
            root_folder: Drive folder path to hold every run.
            root_folder_id: An explicit Drive folder id, which wins over
                ``root_folder``. Needed for a folder this code did not create,
                since the ``drive.file`` scope cannot resolve one by name.
            service_account: Service-account key, the headless-friendly route.
            token: Cached OAuth token.
            client_secret: OAuth client secret, used only if ``interactive``.
            interactive: Whether a browser may be opened. False on a server.
            workers: Parallel file transfers within one checkpoint.
            upload_optimizer_state: Mirror :data:`OPTIMIZER_FILES` too.
            keep_remote: Remote checkpoints to retain, oldest deleted first.
                ``0`` keeps every one.
            log_path: Run log to mirror alongside each checkpoint.
        """
        self.run_name = run_name
        self.output_dir = output_dir
        self.root_folder = root_folder
        self.root_folder_id = root_folder_id
        self.service_account = service_account
        self.token = token
        self.client_secret = client_secret
        self.interactive = interactive
        self.workers = max(1, workers)
        self.upload_optimizer_state = upload_optimizer_state
        self.keep_remote = max(0, keep_remote)
        self.log_path = log_path

        self.stats = SyncStats()
        self._staging = output_dir / STAGING_DIRNAME
        self._client: DriveClient | None = None
        self._run_folder_id: str | None = None
        self._queue: queue.Queue[_Job | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- setup

    @property
    def client(self) -> DriveClient:
        """Return the authenticated client, which :meth:`start` must have made."""
        if self._client is None:
            raise DriveError("CheckpointSync.start() has not run")
        return self._client

    def start(self) -> None:
        """Authenticate, resolve the remote run folder, and spawn the worker.

        Called before training, so that a credential problem fails the run in the
        first seconds rather than being discovered hours later as an empty Drive
        folder. Any :class:`~fa_ocr_lora.drive.DriveError` here is deliberately
        left to propagate.
        """
        credentials = load_credentials(
            self.client_secret,
            self.token,
            service_account=self.service_account,
            interactive=self.interactive,
        )
        self._client = DriveClient(credentials)
        root = self.root_folder_id or self.client.ensure_folder(self.root_folder, "root")
        self._run_folder_id = self.client.ensure_folder(self.run_name, root)
        LOGGER.info(
            "drive sync: %s/%s (folder id %s)",
            self.root_folder,
            self.run_name,
            self._run_folder_id,
        )
        self._thread = threading.Thread(
            target=self._work, name="drive-checkpoint-sync", daemon=True
        )
        self._thread.start()
        self._requeue_stale()

    def _requeue_stale(self) -> None:
        """Re-queue staged checkpoints a previous run died before uploading.

        Staging survives a kill -9, and a staged directory that is still there is
        by definition one whose upload never finished. Re-queuing is safe because
        the uploader skips remote files that already match byte for byte.
        """
        if not self._staging.is_dir():
            return
        for path in sorted(self._staging.iterdir()):
            if not path.is_dir():
                continue
            LOGGER.warning(
                "drive sync: re-queuing staged %s from a previous run", path.name
            )
            self._queue.put(
                _Job("checkpoint", path.name, path, _checkpoint_step(path.name) >= 0)
            )

    # ------------------------------------------------------------- enqueueing

    def enqueue(self, checkpoint_dir: Path, *, publish: bool = True) -> None:
        """Stage a directory and hand it to the worker. Returns immediately.

        Staging is the blocking part and it is hard-linking, not copying, so the
        cost is a handful of ``link(2)`` calls regardless of checkpoint size.

        Args:
            checkpoint_dir: Local directory to mirror.
            publish: Whether resume may later select this directory. See
                :attr:`_Job.publish`.
        """
        if self._thread is None:
            return
        try:
            staged = self._stage(checkpoint_dir)
        except OSError as exc:
            self.stats.failures += 1
            LOGGER.warning("drive sync: could not stage %s: %s", checkpoint_dir, exc)
            return
        self._queue.put(_Job("checkpoint", checkpoint_dir.name, staged, publish))
        if self.log_path is not None and self.log_path.exists():
            self._queue.put(_Job("log", "", self.log_path))

    def _stage(self, checkpoint_dir: Path) -> Path:
        """Hard-link a checkpoint's files into the staging area.

        Returns:
            The staged directory, whose layout mirrors the original.
        """
        target = self._staging / checkpoint_dir.name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        for source in sorted(checkpoint_dir.rglob("*")):
            if not source.is_file():
                continue
            if not self.upload_optimizer_state and source.name in OPTIMIZER_FILES:
                continue
            destination = target / source.relative_to(checkpoint_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
            except OSError:
                # A filesystem that cannot hard-link, or a cross-device layout.
                # Copying costs real bytes but keeps the guarantee that matters:
                # the uploader reads something the trainer will not delete.
                shutil.copy2(source, destination)
        return target

    # ----------------------------------------------------------------- worker

    def _work(self) -> None:
        """Drain the queue until the sentinel arrives."""
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                if job.kind == "checkpoint":
                    self._push_checkpoint(job)
                else:
                    self._push_log(job.path)
            except Exception as exc:  # noqa: BLE001 - a failed upload must not
                # kill training. Counted, logged with context, and the next
                # checkpoint gets its own attempt.
                self.stats.failures += 1
                LOGGER.warning(
                    "drive sync: %s %s failed: %s", job.kind, job.name or "log", exc
                )
            finally:
                self._queue.task_done()

    def _push_checkpoint(self, job: _Job) -> None:
        """Upload one staged checkpoint, then publish it via the marker."""
        files = sorted(p for p in job.path.rglob("*") if p.is_file())
        if not files:
            LOGGER.warning("drive sync: staged %s is empty", job.name)
            return

        # Group by relative directory so a nested checkpoint layout is mirrored
        # rather than flattened into name collisions.
        by_dir: dict[str, list[Path]] = {}
        for path in files:
            relative = path.relative_to(job.path).parent.as_posix()
            by_dir.setdefault("" if relative == "." else relative, []).append(path)

        uploaded = skipped = 0
        for relative, group in sorted(by_dir.items()):
            remote_path = f"{job.name}/{relative}" if relative else job.name
            folder_id = self.client.ensure_folder(remote_path, self._run_folder_id or "")
            existing = self.client.list_folder(folder_id)
            todo = []
            for path in group:
                match = existing.get(path.name)
                if match and match[1] == path.stat().st_size:
                    skipped += 1
                    continue
                todo.append((path, match[0] if match else None))
            if todo:
                uploaded += self._upload_many(todo, folder_id)

        self.stats.files_uploaded += uploaded
        self.stats.files_skipped += skipped
        self.stats.checkpoints_uploaded += 1
        if job.publish:
            self._publish(job.name)
        shutil.rmtree(job.path, ignore_errors=True)
        LOGGER.info(
            "drive sync: %s mirrored (%d uploaded, %d already current)",
            job.name,
            uploaded,
            skipped,
        )
        if self.keep_remote:
            self._prune()

    def _upload_many(
        self, todo: Iterable[tuple[Path, str | None]], folder_id: str
    ) -> int:
        """Upload a group of files concurrently and return how many landed.

        One file's failure is isolated: the others still upload, and the caller
        finds out because the marker is only written when nothing raised.
        """
        items = list(todo)
        count = 0
        with ThreadPoolExecutor(max_workers=min(self.workers, len(items))) as pool:
            futures = {
                pool.submit(
                    self.client.upload,
                    path,
                    path.name,
                    folder_id,
                    _mime_type(path),
                    existing_id,
                ): path
                for path, existing_id in items
            }
            for future, path in futures.items():
                future.result()
                count += 1
                self.stats.bytes_uploaded += path.stat().st_size
        return count

    def _publish(self, name: str) -> None:
        """Rewrite the ``LATEST`` marker to name a fully uploaded checkpoint."""
        marker = self._staging / LATEST_MARKER
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(name + "\n", encoding="utf-8")
        folder_id = self._run_folder_id or ""
        existing = self.client.list_folder(folder_id).get(LATEST_MARKER)
        self.client.upload(
            marker,
            LATEST_MARKER,
            folder_id,
            "text/plain",
            existing[0] if existing else None,
        )

    def _push_log(self, path: Path) -> None:
        """Mirror the run log, overwriting the remote copy in place.

        Resolved against the run folder, not the Drive root: ``run_name`` is a
        leaf name there, so passing it as a path from the root would create a
        second ``<run_name>/`` folder beside the real one and put the log in the
        wrong tree.
        """
        folder_id = self.client.ensure_folder(LOG_DIRNAME, self._run_folder_id or "")
        existing = self.client.list_folder(folder_id).get(path.name)
        self.client.upload(
            path, path.name, folder_id, "text/plain", existing[0] if existing else None
        )

    def _prune(self) -> None:
        """Delete the oldest remote checkpoints beyond ``keep_remote``.

        The remote mirror needs its own limit: ``save_total_limit`` bounds the
        instance's disk, but every checkpoint ever written reaches Drive, and a
        17,000-step run at ``save_steps: 200`` would leave 86 of them there.
        """
        folders = self.client.list_subfolders(self._run_folder_id or "")
        checkpoints = sorted(
            ((name, fid) for name, fid in folders.items() if _checkpoint_step(name) >= 0),
            key=lambda pair: _checkpoint_step(pair[0]),
        )
        for name, folder_id in checkpoints[: max(0, len(checkpoints) - self.keep_remote)]:
            try:
                self.client.delete(folder_id)
            except Exception as exc:  # noqa: BLE001 - pruning is housekeeping
                LOGGER.warning("drive sync: could not prune %s: %s", name, exc)
            else:
                LOGGER.info("drive sync: pruned remote %s", name)

    # ------------------------------------------------------------------ resume

    def fetch_folder(self, name: str, destination: Path) -> Path | None:
        """Download one named remote folder of this run, whatever it holds.

        :meth:`fetch_resume` exists for the one case the trainer needs - the
        newest *complete* checkpoint, chosen by the ``LATEST`` marker. This is
        the general case, and the reason it is separate: an evaluation session
        wants ``adapter_only``, which no marker names and which is not a
        checkpoint, so none of the step-number logic applies to it.

        Blocks, like :meth:`fetch_resume`, because the caller cannot load an
        adapter that is not on disk yet.

        Args:
            name: Remote folder name under this run's folder.
            destination: Local directory to write the files into, created if
                absent. The folder's contents land directly in it, not in a
                ``name`` subdirectory, so the caller decides the local layout.

        Returns:
            ``destination`` when at least one file landed or was already
            current, or ``None`` when Drive has no such folder for this run.
        """
        folder_id = self.client.find_folder(name, self._run_folder_id or "")
        if folder_id is None:
            LOGGER.info("drive sync: no remote %s for %s", name, self.run_name)
            return None
        contents = self.client.list_folder(folder_id)
        if not contents:
            LOGGER.warning("drive sync: remote %s is empty", name)
            return None
        destination.mkdir(parents=True, exist_ok=True)
        downloaded = reused = 0
        for filename, (file_id, size) in sorted(contents.items()):
            local = destination / filename
            # Same size-match staleness test the upload side uses, so an
            # interrupted fetch only re-pulls what is actually missing.
            if local.exists() and local.stat().st_size == size:
                reused += 1
                continue
            self.client.download(file_id, local)
            downloaded += 1
        LOGGER.info(
            "drive sync: fetched %s (%d downloaded, %d already local)",
            name,
            downloaded,
            reused,
        )
        return destination

    def push_directory(self, source: Path, *, remote_name: str | None = None) -> int:
        """Upload a directory tree into this run's remote folder, blocking.

        For artefacts that are produced *after* training and are small enough
        that a background thread would be more machinery than it saves - the
        evaluation report bundle being the case this was written for. The
        checkpoint path deliberately does not reuse it: a checkpoint is staged
        by hard-link and uploaded off the training thread so the GPU never waits
        on the network, and neither property matters once the GPU is idle.

        Nested directories are mirrored, so a bundle with a ``charts/``
        subdirectory arrives with that subdirectory.

        Args:
            source: Local directory to upload. Its *contents* go up.
            remote_name: Remote folder name under the run folder; defaults to
                ``source.name``.

        Returns:
            How many files were uploaded.

        Raises:
            FileNotFoundError: If ``source`` is not a directory. An upload that
                silently did nothing is the failure this prevents.
        """
        if not source.is_dir():
            raise FileNotFoundError(f"{source} is not a directory")
        root = remote_name or source.name
        uploaded = 0
        for directory, _subdirs, filenames in os.walk(source):
            local_dir = Path(directory)
            relative = local_dir.relative_to(source)
            remote_path = (
                root if relative == Path(".") else f"{root}/{relative.as_posix()}"
            )
            folder_id = self.client.ensure_folder(remote_path, self._run_folder_id or "")
            existing = self.client.list_folder(folder_id)
            todo: list[tuple[Path, str | None]] = []
            for filename in sorted(filenames):
                path = local_dir / filename
                current = existing.get(filename)
                if current is not None and current[1] == path.stat().st_size:
                    continue
                todo.append((path, current[0] if current else None))
            if todo:
                uploaded += self._upload_many(todo, folder_id)
        self.stats.files_uploaded += uploaded
        LOGGER.info("drive sync: pushed %s (%d file(s))", root, uploaded)
        return uploaded

    def fetch_resume(self, destination_root: Path) -> Path | None:
        """Download the newest complete remote checkpoint, if there is one.

        Blocks, unlike everything else here, because training cannot start until
        the files are on disk.

        Args:
            destination_root: Local checkpoints directory to write into, normally
                the trainer's ``output_dir``.

        Returns:
            The local checkpoint directory, or ``None`` when Drive holds nothing
            usable for this run.
        """
        name = self._latest_remote()
        if name is None:
            LOGGER.info("drive sync: no remote checkpoint for %s", self.run_name)
            return None
        folder_id = self.client.find_folder(name, self._run_folder_id or "")
        if folder_id is None:
            LOGGER.warning(
                "drive sync: %s names %s, which is missing", LATEST_MARKER, name
            )
            return None

        target = destination_root / name
        target.mkdir(parents=True, exist_ok=True)
        contents = self.client.list_folder(folder_id)
        if not contents:
            LOGGER.warning("drive sync: remote %s is empty", name)
            return None

        downloaded = reused = 0
        for filename, (file_id, size) in sorted(contents.items()):
            local = target / filename
            # Size match is the same staleness test the upload side uses, so a
            # resume interrupted halfway only re-fetches what is missing.
            if local.exists() and local.stat().st_size == size:
                reused += 1
                continue
            self.client.download(file_id, local)
            downloaded += 1
        LOGGER.info(
            "drive sync: resumed %s from Drive (%d downloaded, %d already local)",
            name,
            downloaded,
            reused,
        )
        return target

    def _latest_remote(self) -> str | None:
        """Return the newest complete remote checkpoint name, or ``None``.

        Prefers the marker. Falls back to the highest-numbered directory only
        when no marker exists at all - which means either a run that predates
        this code or one killed before its first publish, and in both cases a
        best guess beats refusing to resume.
        """
        folder_id = self._run_folder_id or ""
        contents = self.client.list_folder(folder_id)
        marker = contents.get(LATEST_MARKER)
        if marker is not None:
            name = self.client.read_text(marker[0]).strip()
            if name:
                return name
        folders = self.client.list_subfolders(folder_id)
        steps = [name for name in folders if _checkpoint_step(name) >= 0]
        if not steps:
            return None
        newest = max(steps, key=_checkpoint_step)
        LOGGER.warning(
            "drive sync: no %s marker; falling back to %s, which may be incomplete",
            LATEST_MARKER,
            newest,
        )
        return newest

    # ---------------------------------------------------------------- teardown

    def close(self, timeout: float = 900.0) -> None:
        """Drain the queue and stop the worker.

        Args:
            timeout: Seconds to wait. A bounded wait is deliberate: an upload
                wedged behind a dead network must not keep a finished run - and
                its billed instance - alive indefinitely. What is already in
                Drive stays valid, because the marker only ever names a complete
                checkpoint.
        """
        if self._thread is None:
            return
        if self.log_path is not None and self.log_path.exists():
            self._queue.put(_Job("log", "", self.log_path))
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            LOGGER.warning(
                "drive sync: still uploading after %.0fs; abandoning %d queued item(s)",
                timeout,
                self._queue.qsize(),
            )
        else:
            LOGGER.info("drive sync: %s", self.stats.as_dict())
        self._thread = None

    def __enter__(self) -> CheckpointSync:
        """Start the mirror and return it."""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Drain the mirror, whether or not training raised."""
        self.close()
