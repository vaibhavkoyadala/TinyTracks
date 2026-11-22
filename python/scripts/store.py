"""SQLite database that records every model, its files, and download state.

SQLite is a single local file, needs no server process, and ships with Python,
so it suits a desktop app better than a MySQL server would.
"""
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 5

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    family       TEXT NOT NULL,
    description  TEXT NOT NULL,
    repo         TEXT NOT NULL,
    revision     TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'not_downloaded',
    error        TEXT,
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS model_files (
    model_id     TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    filename     TEXT NOT NULL,
    position     INTEGER NOT NULL,
    source_url   TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL,
    sha256       TEXT NOT NULL,
    bytes_done   INTEGER NOT NULL DEFAULT 0,
    done         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (model_id, filename)
);

CREATE TABLE IF NOT EXISTS request_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL NOT NULL,
    server          TEXT NOT NULL,
    method          TEXT NOT NULL,
    path            TEXT NOT NULL,
    status          INTEGER,
    duration_ms     REAL,
    model_id        TEXT,
    client          TEXT,
    request_bytes   INTEGER,
    response_bytes  INTEGER
);
CREATE INDEX IF NOT EXISTS request_log_ts ON request_log (ts);

CREATE TABLE IF NOT EXISTS settings (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmark_jobs (
    id            TEXT PRIMARY KEY,
    model_id      TEXT NOT NULL,
    widths        TEXT NOT NULL,
    gen_tokens    INTEGER NOT NULL,
    prompt        TEXT NOT NULL,
    status        TEXT NOT NULL,
    current_width INTEGER,
    error         TEXT,
    auto          INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS benchmark_results (
    job_id               TEXT NOT NULL REFERENCES benchmark_jobs(id) ON DELETE CASCADE,
    beam                 INTEGER NOT NULL,
    search_s             REAL,
    prefill_ms           REAL,
    decode_ms_per_token  REAL,
    decode_tok_s         REAL,
    tokens               INTEGER,
    created_at           REAL NOT NULL,
    PRIMARY KEY (job_id, beam)
);

CREATE TABLE IF NOT EXISTS kernels (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL REFERENCES benchmark_jobs(id) ON DELETE CASCADE,
    model_id    TEXT NOT NULL,
    beam        INTEGER NOT NULL,
    device      TEXT NOT NULL,
    ast_hash    TEXT NOT NULL,
    opts        TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""

# not_downloaded -> downloading -> verifying -> ready
#                  \-> paused (user or interrupted) / failed (retries exhausted)
STATUSES = ("not_downloaded", "downloading", "paused", "verifying", "ready", "failed")


class Store:
    def __init__(self, db_path: Path):
        self.path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            version = c.execute("PRAGMA user_version").fetchone()[0]
            if version < 2:
                # Pre-release schema: only catalog rows lived here, and they are re-seeded
                # from catalog.py on every start. Weights on disk are untouched.
                c.executescript("DROP TABLE IF EXISTS model_files; DROP TABLE IF EXISTS models;")
            c.executescript(SCHEMA)  # idempotent: adds any new tables
            job_cols = [r["name"] for r in c.execute("PRAGMA table_info(benchmark_jobs)").fetchall()]
            if "auto" not in job_cols:  # schema v4
                c.execute("ALTER TABLE benchmark_jobs ADD COLUMN auto INTEGER NOT NULL DEFAULT 0")
            if version < SCHEMA_VERSION:
                c.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def _conn(self):
        # Autocommit mode: each statement is its own transaction. Connections are
        # per-call, so worker threads never share one.
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
        finally:
            conn.close()

    def upsert_catalog(self, entries: list[dict]) -> None:
        """Sync catalog rows. Download state is kept; file metadata follows the catalog."""
        now = time.time()
        with self._conn() as c:
            c.execute("BEGIN")
            try:
                for e in entries:
                    c.execute(
                        """
                        INSERT INTO models (id, name, family, description, repo, revision, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            name=excluded.name, family=excluded.family, description=excluded.description,
                            repo=excluded.repo, revision=excluded.revision, updated_at=excluded.updated_at
                        """,
                        (e["id"], e["name"], e["family"], e["description"], e["repo"], e["revision"], now),
                    )
                    names = []
                    for pos, f in enumerate(e["files"]):
                        names.append(f["filename"])
                        c.execute(
                            """
                            INSERT INTO model_files (model_id, filename, position, source_url, size_bytes, sha256)
                            VALUES (?, ?, ?, ?, ?, ?)
                            ON CONFLICT(model_id, filename) DO UPDATE SET
                                position=excluded.position, source_url=excluded.source_url,
                                size_bytes=excluded.size_bytes, sha256=excluded.sha256
                            """,
                            (e["id"], f["filename"], pos, f["source_url"], f["size_bytes"], f["sha256"]),
                        )
                    placeholders = ", ".join("?" for _ in names)
                    c.execute(
                        f"DELETE FROM model_files WHERE model_id = ? AND filename NOT IN ({placeholders})",
                        (e["id"], *names),
                    )
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise

    def _attach_files(self, c: sqlite3.Connection, rows) -> list[dict]:
        models = []
        for r in rows:
            m = dict(r)
            m["files"] = [
                dict(f) for f in c.execute(
                    "SELECT * FROM model_files WHERE model_id = ? ORDER BY position", (m["id"],)
                ).fetchall()
            ]
            models.append(m)
        return models

    def list_models(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM models ORDER BY family, name").fetchall()
            return self._attach_files(c, rows)

    def get_model(self, model_id: str) -> dict | None:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchall()
            models = self._attach_files(c, rows)
        return models[0] if models else None

    def update_model(self, model_id: str, **fields) -> None:
        if not fields:
            return
        if "status" in fields and fields["status"] not in STATUSES:
            raise ValueError(f"unknown status {fields['status']!r}")
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE models SET {cols} WHERE id = ?", (*fields.values(), model_id))

    def update_file(self, model_id: str, filename: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(
                f"UPDATE model_files SET {cols} WHERE model_id = ? AND filename = ?",
                (*fields.values(), model_id, filename),
            )

    def reset_files(self, model_id: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE model_files SET bytes_done = 0, done = 0 WHERE model_id = ?", (model_id,))

    # ----- settings ---------------------------------------------------------

    def get_setting(self, key: str) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_setting(self, key: str, value: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                      "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    # ----- benchmarks -------------------------------------------------------

    def create_job(self, job: dict) -> None:
        now = time.time()
        with self._conn() as c:
            c.execute(
                "INSERT INTO benchmark_jobs (id, model_id, widths, gen_tokens, prompt, status, current_width, "
                "error, auto, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (job["id"], job["model_id"], job["widths"], job["gen_tokens"], job["prompt"], job["status"],
                 None, None, int(job.get("auto", False)), now, now),
            )

    def update_job(self, job_id: str, **fields) -> None:
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE benchmark_jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))

    def get_job(self, job_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM benchmark_jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_jobs(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM benchmark_jobs ORDER BY created_at DESC").fetchall()]

    def add_result(self, job_id: str, result: dict) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO benchmark_results (job_id, beam, search_s, prefill_ms, decode_ms_per_token, "
                "decode_tok_s, tokens, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, result["beam"], result["search_s"], result["prefill_ms"], result["decode_ms_per_token"],
                 result["decode_tok_s"], result["tokens"], time.time()),
            )

    def list_results(self, job_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM benchmark_results WHERE job_id = ? ORDER BY beam", (job_id,)).fetchall()
        return [dict(r) for r in rows]

    def add_kernels(self, job_id: str, model_id: str, beam: int, device: str, kernels: list[tuple[str, str]]) -> None:
        now = time.time()
        with self._conn() as c:
            c.execute("BEGIN")
            try:
                for ast_hash, opts in kernels:
                    c.execute(
                        "INSERT INTO kernels (job_id, model_id, beam, device, ast_hash, opts, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (job_id, model_id, beam, device, ast_hash, opts, now),
                    )
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise

    def list_kernels(self, model_id: str | None = None) -> list[dict]:
        with self._conn() as c:
            if model_id:
                rows = c.execute("SELECT * FROM kernels WHERE model_id = ? ORDER BY id", (model_id,)).fetchall()
            else:
                rows = c.execute("SELECT * FROM kernels ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    # ----- request log ------------------------------------------------------

    def add_request(self, row: dict) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO request_log (ts, server, method, path, status, duration_ms, model_id, client, "
                "request_bytes, response_bytes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (row["ts"], row["server"], row["method"], row["path"], row["status"], row["duration_ms"],
                 row["model_id"], row["client"], row["request_bytes"], row["response_bytes"]),
            )

    def list_requests(self, limit: int = 200, before_id: int | None = None, server: str | None = None,
                      errors_only: bool = False) -> list[dict]:
        where, args = [], []
        if before_id:
            where.append("id < ?")
            args.append(before_id)
        if server:
            where.append("server = ?")
            args.append(server)
        if errors_only:
            where.append("status >= 400")
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        with self._conn() as c:
            rows = c.execute(f"SELECT * FROM request_log {clause} ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        return [dict(r) for r in rows]

    def request_summary(self, since: float) -> dict:
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS total, SUM(status >= 400) AS errors, AVG(duration_ms) AS avg_ms, "
                "SUM(server = 'openai') AS openai FROM request_log WHERE ts >= ?", (since,)).fetchone()
        return {"total": row["total"] or 0, "errors": row["errors"] or 0,
                "avg_ms": round(row["avg_ms"], 1) if row["avg_ms"] is not None else None,
                "openai": row["openai"] or 0}

    def prune_requests(self, keep: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM request_log WHERE id <= (SELECT MAX(id) FROM request_log) - ?", (keep,))

