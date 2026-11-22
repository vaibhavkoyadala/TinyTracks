"""Resumable, verified model downloads.

Design:
- A model is a set of files. Each file streams into `<filename>.part`, and is renamed
  into place only after its size and SHA-256 match the catalog. A model is `ready` only
  when every file is in place, so a half-written or corrupt model is never reported ready.
- Interrupted transfers resume with an HTTP Range request from the bytes already on disk.
- Network errors retry with exponential backoff. The user can pause at any point.
- Progress is written to SQLite (throttled), so the UI and any restart see the same state.
- On startup, downloads that were in flight when the app last stopped resume automatically.
"""
import hashlib
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from catalog import CATALOG
from store import Store

CHUNK = 256 * 1024
HASH_CHUNK = 1024 * 1024
MAX_ATTEMPTS = 6
PROGRESS_INTERVAL = 0.25
HTTP_TIMEOUT = 30
USER_AGENT = "TinyTracks/0.1"
SPACE_MARGIN = 64 * 1024 * 1024


class Cancelled(Exception):
    """Raised inside a worker when the user pauses."""


class UnknownModel(KeyError):
    pass


class Busy(RuntimeError):
    """The operation conflicts with a download that is currently running."""


class NoSpace(RuntimeError):
    """Not enough free disk space for the remaining bytes (plus a safety margin)."""


class DownloadManager:
    def __init__(self, store: Store, weights_dir: Path):
        self.store = store
        self.weights_dir = weights_dir
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}

    @classmethod
    def open(cls, data_dir: Path) -> "DownloadManager":
        store = Store(data_dir / "tinytracks.db")
        store.upsert_catalog(CATALOG)
        mgr = cls(store, data_dir / "weights")
        mgr._reconcile()
        return mgr

    # ----- public API -------------------------------------------------------

    def list(self) -> list[dict]:
        return [self._present(m) for m in self.store.list_models()]

    def get(self, model_id: str) -> dict:
        model = self._require(model_id)
        out = self._present(model)
        out["files"] = [
            {
                "filename": f["filename"],
                "size_bytes": f["size_bytes"],
                "bytes_done": min(f["bytes_done"], f["size_bytes"]),
                "done": bool(f["done"]),
            }
            for f in model["files"]
        ]
        return out

    def start(self, model_id: str) -> dict:
        """Start (or resume) a download. Idempotent: a running download is returned as-is."""
        model = self._require(model_id)
        with self._lock:
            if model_id in self._threads or model["status"] == "ready":
                return self.get(model_id)
            d = self._dir(model)
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            remaining = sum(f["size_bytes"] - self._disk_offset(d, f) for f in model["files"])
            free = shutil.disk_usage(self.weights_dir).free
            if free < remaining + SPACE_MARGIN:
                raise NoSpace(
                    f"needs {remaining + SPACE_MARGIN:,} bytes free; {free:,} available"
                )
            ev = threading.Event()
            self._cancel[model_id] = ev
            self.store.update_model(model_id, status="downloading", error=None)
            t = threading.Thread(target=self._run, args=(model_id, ev), name=f"dl-{model_id}", daemon=True)
            self._threads[model_id] = t
            t.start()
        return self.get(model_id)

    def pause(self, model_id: str) -> dict:
        self._require(model_id)
        with self._lock:
            thread = self._threads.get(model_id)
            ev = self._cancel.get(model_id)
        if thread is not None:
            ev.set()
            thread.join(timeout=10)
            if thread.is_alive():
                raise Busy("download did not stop in time; try again")
        return self.get(model_id)

    def delete(self, model_id: str) -> dict:
        model = self._require(model_id)
        with self._lock:
            if model_id in self._threads:
                raise Busy("pause the download before deleting it")
            shutil.rmtree(self._dir(model), ignore_errors=True)
            self.store.reset_files(model_id)
            self.store.update_model(model_id, status="not_downloaded", error=None)
        return self.get(model_id)

    def shutdown(self) -> None:
        """Stop all transfers. Partial files stay on disk for the next launch to resume."""
        with self._lock:
            events = list(self._cancel.values())
            threads = list(self._threads.values())
        for ev in events:
            ev.set()
        for t in threads:
            t.join(timeout=10)

    # ----- internals --------------------------------------------------------

    def _require(self, model_id: str) -> dict:
        model = self.store.get_model(model_id)
        if model is None:
            raise UnknownModel(model_id)
        return model

    def _dir(self, model: dict) -> Path:
        return self.weights_dir / model["id"]

    @staticmethod
    def _paths(d: Path, f: dict) -> tuple[Path, Path]:
        final = d / f["filename"]
        return final, final.with_name(final.name + ".part")

    def _disk_offset(self, d: Path, f: dict) -> int:
        """Bytes of this file that are already on disk (final or partial)."""
        final, part = self._paths(d, f)
        if final.is_file() and final.stat().st_size == f["size_bytes"]:
            return f["size_bytes"]
        if part.exists():
            return min(part.stat().st_size, f["size_bytes"])
        return 0

    def _present(self, model: dict) -> dict:
        files = model["files"]
        size = sum(f["size_bytes"] for f in files)
        done = sum(min(f["bytes_done"], f["size_bytes"]) for f in files)
        ready = model["status"] == "ready"
        return {
            "id": model["id"],
            "name": model["name"],
            "family": model["family"],
            "description": model["description"],
            "repo": model["repo"],
            "revision": model["revision"],
            "file_count": len(files),
            "size_bytes": size,
            "status": model["status"],
            "bytes_done": done,
            "percent": round(100.0 * done / size, 2) if size else 0.0,
            "local_dir": str(self._dir(model)) if ready else None,
            "error": model["error"],
        }

    def _reconcile(self) -> None:
        """Bring DB state in line with the files on disk, then resume interrupted work."""
        resume: list[str] = []
        for m in self.store.list_models():
            d = self._dir(m)
            all_done = True
            for f in m["files"]:
                final, _ = self._paths(d, f)
                complete = final.is_file() and final.stat().st_size == f["size_bytes"]
                all_done &= complete
                self.store.update_file(
                    m["id"], f["filename"],
                    bytes_done=self._disk_offset(d, f), done=int(complete),
                )

            was_active = m["status"] in ("downloading", "verifying")
            if all_done:
                self.store.update_model(m["id"], status="ready", error=None)
                continue
            if m["status"] == "failed":
                status = "failed"
            elif any(self._disk_offset(d, f) for f in m["files"]):
                status = "paused"
            else:
                status = "not_downloaded"
            self.store.update_model(m["id"], status=status, error=m["error"] if status == "failed" else None)
            if was_active:
                resume.append(m["id"])

        for model_id in resume:
            try:
                self.start(model_id)
            except NoSpace as exc:
                self.store.update_model(model_id, status="failed", error=str(exc)[:500])

    def _record_disk_progress(self, model_id: str) -> None:
        model = self._require(model_id)
        d = self._dir(model)
        for f in model["files"]:
            self.store.update_file(model_id, f["filename"], bytes_done=self._disk_offset(d, f))

    def _run(self, model_id: str, ev: threading.Event) -> None:
        try:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    self._download_all(model_id, ev)
                    return
                except Cancelled:
                    self._record_disk_progress(model_id)
                    self.store.update_model(model_id, status="paused")
                    return
                except Exception as exc:  # network, disk, or integrity failure
                    self._record_disk_progress(model_id)
                    if attempt == MAX_ATTEMPTS:
                        self.store.update_model(model_id, status="failed", error=str(exc)[:500])
                        return
                    self.store.update_model(model_id, error=f"attempt {attempt} failed: {exc}"[:500])
                    if ev.wait(min(2 ** attempt, 30)):
                        self._record_disk_progress(model_id)
                        self.store.update_model(model_id, status="paused")
                        return
        finally:
            with self._lock:
                self._threads.pop(model_id, None)
                self._cancel.pop(model_id, None)

    def _download_all(self, model_id: str, ev: threading.Event) -> None:
        model = self._require(model_id)
        d = self._dir(model)
        d.mkdir(parents=True, exist_ok=True)
        for f in model["files"]:
            final, _ = self._paths(d, f)
            if final.is_file() and final.stat().st_size == f["size_bytes"]:
                continue  # already verified on an earlier run
            self.store.update_model(model_id, status="downloading")
            self._download_file(model_id, d, f, ev)
        self.store.update_model(model_id, status="ready", error=None)

    def _download_file(self, model_id: str, d: Path, f: dict, ev: threading.Event) -> None:
        final, part = self._paths(d, f)
        size = f["size_bytes"]
        name = f["filename"]

        offset = part.stat().st_size if part.exists() else 0
        if offset > size:  # cannot be a valid prefix; start clean
            part.unlink()
            offset = 0
        if offset < size:
            self._fetch(model_id, f, part, offset, ev)

        got = part.stat().st_size
        if got < size:  # connection dropped early: keep the prefix, the retry resumes
            raise IOError(f"{name}: connection closed early at {got} of {size} bytes")
        if got > size:
            part.unlink()
            raise IOError(f"{name}: larger than expected ({got} > {size} bytes); restarting")

        self.store.update_model(model_id, status="verifying")
        if _sha256_file(part) != f["sha256"]:
            part.unlink()
            raise IOError(f"{name}: sha256 mismatch; discarded and will restart")

        os.replace(part, final)  # atomic: `final` only ever holds verified bytes
        self.store.update_file(model_id, name, bytes_done=size, done=1)

    def _fetch(self, model_id: str, f: dict, part: Path, offset: int, ev: threading.Event) -> None:
        size = f["size_bytes"]
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        req = urllib.request.Request(f["source_url"], headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=HTTP_TIMEOUT)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and offset:  # our prefix is not acceptable; start over next attempt
                part.unlink(missing_ok=True)
            raise

        with resp:
            if offset and resp.status != 206:
                offset = 0  # server ignored Range and sent the whole file
            done = offset
            last_report = time.monotonic()
            with open(part, "ab" if offset else "wb") as fh:
                while True:
                    if ev.is_set():
                        raise Cancelled()
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= PROGRESS_INTERVAL:
                        self.store.update_file(model_id, f["filename"], bytes_done=min(done, size))
                        last_report = now
        self.store.update_file(model_id, f["filename"], bytes_done=min(done, size))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(HASH_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
