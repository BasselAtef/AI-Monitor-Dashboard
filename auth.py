"""
Google OAuth sign-in and per-user data isolation.

Two separate auth paths, because they serve different callers:

  Browser routes (/, /api/stats, ...)  -> Google OAuth session cookie
  /api/log                             -> per-user ingest token in a header

/api/log cannot use a session cookie: ai_monitor.py is a machine POSTing from
another program, with no browser and no way to complete a redirect.

Credentials come from the environment and are required. If they are missing the
app refuses to start rather than silently serving everyone's data to anyone.
"""

import hashlib
import hmac
import os
import secrets
import sqlite3
from functools import wraps

from authlib.integrations.flask_client import OAuth

# Load .env for local development. Values already in the real environment win,
# which is what production hosts set. Missing .env is not an error.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.environ.get(
    "GOOGLE_REDIRECT_URI", "http://localhost:5000/auth/google/callback"
)
SECRET_KEY = os.environ.get("SECRET_KEY", "")
SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "0") == "1"

# Hosts we accept a sign-in started on. Google must still have a registered
# redirect URI for each one, but keeping the session cookie alive across
# localhost/127.0.0.1 avoids a confusing state mismatch during local dev.
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def is_local_host(host: str) -> bool:
    return (host or "").split(":")[0].lower() in LOCAL_HOSTS

REQUIRED_ENV = (
    ("GOOGLE_CLIENT_ID", GOOGLE_CLIENT_ID),
    ("GOOGLE_CLIENT_SECRET", GOOGLE_CLIENT_SECRET),
    ("SECRET_KEY", SECRET_KEY),
)


def assert_configured() -> None:
    """Fail fast if auth is not usable. Called once at startup."""
    missing = [name for name, value in REQUIRED_ENV if not value]
    if missing:
        raise SystemExit(
            "Cannot start: missing required auth configuration.\n"
            + "\n".join(f"  - {name}" for name in missing)
            + "\n\nSet them in your environment (see .env.example), or add them\n"
            "in the hosting provider's dashboard as environment variables.\n"
        )
    if len(SECRET_KEY) < 32:
        raise SystemExit(
            "Cannot start: SECRET_KEY must be at least 32 characters.\n"
            "Generate one with:\n"
            "  python -c \"import secrets; print(secrets.token_hex(32))\"\n"
        )


def where_to_get_credentials() -> str:
    """Guidance shown when auth config is incomplete."""
    return f"""
Setup
-----
1. Get your OAuth client from Google Cloud Console:
     https://console.cloud.google.com/apis/credentials
   Create (or open) an OAuth 2.0 Client ID, type "Web application".

2. Add this under "Authorized redirect URIs":
     http://localhost:5000/auth/google/callback

3. Create a .env file next to app.py (PowerShell):
     Copy-Item .env.example .env
   then open .env and fill in:
     GOOGLE_CLIENT_ID=...apps.googleusercontent.com
     GOOGLE_CLIENT_SECRET=...
     SECRET_KEY=<64 random hex chars>
     GOOGLE_REDIRECT_URI=http://localhost:5000/auth/google/callback

   Generate SECRET_KEY with:
     python -c "import secrets; print(secrets.token_hex(32))"

   Missing: {', '.join(name for name, value in REQUIRED_ENV if not value) or 'nothing'}
""".strip()


oauth = OAuth()


def init_oauth(app: "Flask"):  # noqa: F821
    """Register the Google OAuth client on the app."""
    app.config.update(
        SECRET_KEY=SECRET_KEY,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        # Keep the session cookie off plaintext HTTP once you have a real domain.
        SESSION_COOKIE_SECURE=SESSION_COOKIE_SECURE,
        PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 14,
    )

    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        access_token_url="https://oauth2.googleapis.com/token",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    # Binds the OAuth registry to this app; without it every authorize call
    # raises "OAuth is not init with Flask app".
    oauth.init_app(app)


# --------------------------------------------------------------------------
# users table
# --------------------------------------------------------------------------
def init_users_table(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               google_sub TEXT UNIQUE NOT NULL,
               email TEXT NOT NULL,
               name TEXT,
               picture TEXT,
               ingest_token_hash TEXT NOT NULL,
               created_at DATETIME DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------
# ingest tokens
# --------------------------------------------------------------------------
def hash_token(token: str) -> str:
    """Store only a hash, so a database leak does not hand over write access."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_ingest_token() -> str:
    return "aim_" + secrets.token_urlsafe(32)


def get_or_create_user(db_path: str, google_sub: str, email: str,
                       name: str = "", picture: str = "") -> tuple:
    """Return (user_id, email, name, picture, ingest_token).

    Returns an existing user unchanged, or creates one with a fresh token.
    The token is returned only on creation.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM users WHERE google_sub = ?", (google_sub,)
    ).fetchone()

    if row:
        result = (row["id"], row["email"], row["name"], row["picture"], None)
        conn.close()
        return result

    token = new_ingest_token()
    conn.execute(
        """INSERT INTO users
               (google_sub, email, name, picture, ingest_token_hash)
           VALUES (?, ?, ?, ?, ?)""",
        (google_sub, email, name, picture, hash_token(token)),
    )
    conn.commit()
    user_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return (user_id, email, name, picture, token)


def user_for_token(db_path: str, token: str):
    """Resolve an ingest token to a user id, or None. Constant-time compare."""
    if not token:
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, ingest_token_hash FROM users").fetchall()
    conn.close()

    target = hash_token(token)
    for row in rows:
        if hmac.compare_digest(row["ingest_token_hash"], target):
            return row["id"]
    return None


def get_user(db_path: str, user_id: int):
    """Fetch a user by id, for displaying account info."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, email, name, picture, created_at FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    conn.close()
    return row


def rotate_ingest_token(db_path: str, user_id: int) -> str:
    """Issue a new ingest token, invalidating the old one."""
    token = new_ingest_token()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE users SET ingest_token_hash = ? WHERE id = ?",
        (hash_token(token), user_id),
    )
    conn.commit()
    conn.close()
    return token
