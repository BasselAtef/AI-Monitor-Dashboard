"""
Database access layer.

Speaks SQLite for local development and PostgreSQL for deployment, chosen by
whether DATABASE_URL is set. Callers use one small surface:

    with db.cursor() as cur:
        cur.execute("SELECT ... WHERE user_id = %s", (uid,))
        rows = cur.fetchall()

Two details differ between the two engines and are handled here rather than at
every call site:

  placeholders  SQLite uses ?, PostgreSQL uses %s. Write %s always; it is
                translated for SQLite.

  row access    Rows come back as dicts on both engines, so indexing by column
                name behaves the same. sqlite3.Row and psycopg tuples are not
                interchangeable, and plain dicts keep existing code honest.

The date-offset and hourly-bucket SQL lives here too, since those differ more
than the placeholder syntax and are easy to get subtly wrong.
"""

import os
import sqlite3
import threading
import time
from contextlib import contextmanager

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
TZ_OFFSET_HOURS = int(os.environ.get("TZ_OFFSET_HOURS", "3"))

# Connection pool size. Keep this at or below the Postgres connection limit
# your host allows minus whatever else uses that database; a pool larger than
# the server limit shows up as "too many connections" at deploy time.
# Railway trial plans allow very few, so this is usually lowered there.
POOL_MIN = int(os.environ.get("DB_POOL_MIN", "1"))
POOL_MAX = int(os.environ.get("DB_POOL_MAX", "10"))

# Startup waits, kept short so a slow database cannot stall the boot for long.
POOL_WAIT_SECONDS = float(os.environ.get("DB_POOL_WAIT", "15"))
POOL_CLOSE_TIMEOUT = float(os.environ.get("DB_POOL_CLOSE", "5"))

_pool = None
_pool_lock = threading.Lock()


def using_postgres() -> bool:
    """True when DATABASE_URL points at PostgreSQL."""
    return DATABASE_URL.startswith(("postgres://", "postgresql://"))


# --------------------------------------------------------------------------
# local-time SQL, per dialect
# --------------------------------------------------------------------------
def local_time_sql(column: str) -> str:
    """SQL expression shifting a UTC timestamp into display time."""
    if using_postgres():
        return f"({column} + INTERVAL '{TZ_OFFSET_HOURS} hours')"
    return f"datetime({column}, '+{TZ_OFFSET_HOURS} hours')"


def hour_bucket_sql(column: str) -> str:
    """SQL expression truncating a timestamp to the hour, as text."""
    if using_postgres():
        return f"to_char({column} + INTERVAL '{TZ_OFFSET_HOURS} hours', 'YYYY-MM-DD HH24:00:00')"
    return f"strftime('%Y-%m-%d %H:00:00', datetime({column}, '+{TZ_OFFSET_HOURS} hours'))"


def placeholder_style() -> str:
    return "%s"


# --------------------------------------------------------------------------
# connection handling
# --------------------------------------------------------------------------
def get_pool():
    """Lazily build the PostgreSQL pool.

    Opening a Postgres connection costs a TCP handshake plus auth, so doing it
    per request would dominate latency and exhaust connections under load.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                from psycopg_pool import ConnectionPool

                candidate = ConnectionPool(
                    DATABASE_URL,
                    min_size=POOL_MIN,
                    max_size=POOL_MAX,
                    timeout=10,
                    kwargs={"autocommit": False},
                    open=True,
                )
                try:
                    candidate.wait(timeout=POOL_WAIT_SECONDS)
                except Exception:
                    # Never publish a pool that did not come up, or every
                    # later attempt would reuse it and fail identically.
                    try:
                        candidate.close(timeout=1)
                    except Exception:
                        pass
                    raise
                _pool = candidate
    return _pool


def close_pool():
    """Drop the pool. Bounded wait so shutdown cannot hang on a stuck worker."""
    global _pool
    pool, _pool = _pool, None
    if pool is not None:
        try:
            pool.close(timeout=POOL_CLOSE_TIMEOUT)
        except Exception:
            pass


class Cursor:
    """Thin wrapper giving both engines the same cursor behaviour."""

    def __init__(self, raw, sqlite_mode: bool):
        self._raw = raw
        self._sqlite = sqlite_mode
        self.lastrowid = None

    def execute(self, sql: str, params=()):
        # Callers write %s; SQLite wants ?.
        if self._sqlite:
            sql = sql.replace("%s", "?")
        self._raw.execute(sql, params or ())
        return self

    def fetchone(self):
        row = self._raw.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def fetchall(self):
        return [self._row_to_dict(r) for r in self._raw.fetchall()]

    def _row_to_dict(self, row):
        if isinstance(row, dict):
            return row
        desc = self._raw.description or []
        return {col[0]: row[i] for i, col in enumerate(desc)}

    def __getattr__(self, name):
        # lastrowid support
        return getattr(self._raw, name)


@contextmanager
def cursor():
    """Yield a Cursor inside a transaction, committing on success."""
    if using_postgres():
        from psycopg.rows import dict_row

        with get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                wrapped = Cursor(cur, sqlite_mode=False)
                try:
                    yield wrapped
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                return

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        wrapped = Cursor(conn.cursor(), sqlite_mode=True)
        yield wrapped
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------
DB_PATH = "monitor.db"

SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS api_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    provider TEXT,
    model TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    latency_ms INTEGER,
    cost_usd REAL,
    status TEXT,
    error_message TEXT,
    user_id INTEGER
);

CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    anomaly_type TEXT,
    severity TEXT,
    description TEXT,
    call_id INTEGER,
    user_id INTEGER
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active INTEGER DEFAULT 1,
    user_id INTEGER
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    google_sub TEXT UNIQUE NOT NULL,
    email TEXT NOT NULL,
    name TEXT,
    picture TEXT,
    ingest_token_hash TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_calls_user ON api_calls(user_id);
CREATE INDEX IF NOT EXISTS idx_anom_user ON anomalies(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""

SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS api_calls (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    provider TEXT,
    model TEXT,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    cost_usd NUMERIC(12, 8) DEFAULT 0,
    status TEXT,
    error_message TEXT,
    user_id BIGINT
);

CREATE TABLE IF NOT EXISTS anomalies (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    anomaly_type TEXT,
    severity TEXT,
    description TEXT,
    call_id BIGINT,
    user_id BIGINT
);

CREATE TABLE IF NOT EXISTS sessions (
    id BIGSERIAL PRIMARY KEY,
    name TEXT,
    start_time TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    user_id BIGINT
);

CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    google_sub TEXT UNIQUE NOT NULL,
    email TEXT NOT NULL,
    name TEXT,
    picture TEXT,
    ingest_token_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_calls_user ON api_calls(user_id);
CREATE INDEX IF NOT EXISTS idx_anom_user ON anomalies(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


def init_db(retries: int = 5, delay: float = 2.0) -> None:
    """Create tables and indexes if they are missing.

    Runs at import, so on a fresh deploy the database may still be starting.
    Retry briefly rather than dying with a pool timeout that says nothing about
    the actual cause.
    """
    schema = SCHEMA_POSTGRES if using_postgres() else SCHEMA_SQLITE
    statements = [s for s in schema.split(";") if s.strip()]

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with cursor() as cur:
                for statement in statements:
                    cur.execute(statement)
            return
        except Exception as exc:
            last_error = exc
            # A schema error will not fix itself; only retry connection trouble.
            if using_postgres() and attempt < retries:
                print(
                    f"[db] schema init attempt {attempt}/{retries} failed "
                    f"({type(exc).__name__}); retrying in {delay}s",
                    flush=True,
                )
                # Discard the pool. A pool that failed to open is left closed,
                # and reusing it would make every later attempt fail the same
                # way instead of reconnecting.
                close_pool()
                time.sleep(delay)

    raise SystemExit(
        "Cannot start: could not prepare the database schema.\n"
        f"  backend : {'PostgreSQL' if using_postgres() else 'SQLite'}\n"
        f"  url     : {_redacted_url()}\n"
        f"  error   : {type(last_error).__name__}: {last_error}\n\n"
        + (
            "If the database is still starting, the deploy will retry on the\n"
            "next restart. Check that DATABASE_URL is correct and that the\n"
            "Postgres plugin is attached to this service.\n"
            if using_postgres()
            else "Check that the directory is writable.\n"
        )
    ) from last_error


def _redacted_url() -> str:
    """DATABASE_URL with any password masked, safe to log."""
    if not DATABASE_URL:
        return "(unset)"
    try:
        scheme, _, rest = DATABASE_URL.partition("://")
        if "@" in rest:
            creds, _, host = rest.partition("@")
            user, _, _pw = creds.partition(":")
            rest = f"{user}:***@{host}"
        return f"{scheme}://{rest}"
    except Exception:
        return "(unparseable)"


def table_columns(table: str) -> set:
    """Column names for a table, used by the migration check."""
    if using_postgres():
        with cursor() as cur:
            cur.execute(
                """SELECT column_name FROM information_schema.columns
                   WHERE table_name = %s""",
                (table,),
            )
            return {r["column_name"] for r in cur.fetchall()}

    with cursor() as cur:
        cur.execute(f"PRAGMA table_info({table})")
        return {r["name"] for r in cur.fetchall()}
