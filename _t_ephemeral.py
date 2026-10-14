"""Tests for the redeploy data loss.

Symptom: "every time I redeploy I can't find the old data of API calls".

Cause: with no DATABASE_URL, using_postgres() returns False and the app uses
SQLite at monitor.db. On a PaaS the container filesystem is ephemeral, so each
redeploy starts from an empty file. Nothing warned about it.

These tests assert the failure is now loud, and that /api/health reveals which
backend is live without leaking credentials.
"""
import io
import os
import subprocess
import sys
import tempfile

DASH = r"D:\Antigravity\PROJECTS\ai-monitor-dashboard"

fails = []


def check(label, cond, extra=""):
    print(f"  [{'ok' if cond else 'RED'}] {label}" + (f"  {extra}" if extra else ""))
    if not cond:
        fails.append(label)


def boot(env_overrides, code="import db, app; print('OK')"):
    """Run the app in a clean directory with the given environment."""
    env = dict(os.environ)
    env.pop("DATABASE_URL", None)
    for k in ("RAILWAY_ENVIRONMENT", "RENDER", "FLY_APP_NAME", "ALLOW_SQLITE"):
        env.pop(k, None)
    env.update(env_overrides)
    env.setdefault("GOOGLE_CLIENT_ID", "x")
    env.setdefault("GOOGLE_CLIENT_SECRET", "y")
    return subprocess.run([sys.executable, "-c", code], cwd=DASH, env=env,
                          capture_output=True, text=True, timeout=120)


print("=" * 72)
print("1. local development still works")
print("=" * 72)
r = boot({})
check("boots with no DATABASE_URL locally", r.returncode == 0, r.stdout.strip()[:60])
check("prints a persistence hint", "SQLite" in r.stdout or "monitor.db" in r.stdout,
      r.stdout.strip()[:100])

print()
print("=" * 72)
print("2. a PaaS deploy without a database now refuses to start")
print("=" * 72)
for env_name in ("RAILWAY_ENVIRONMENT", "RENDER", "FLY_APP_NAME", "DYNO"):
    r = boot({env_name: "production"})
    refused = r.returncode != 0
    explains = "DATABASE_URL" in (r.stderr or "") or "DATABASE_URL" in (r.stdout or "")
    check(f"refuses on {env_name}", refused, f"exit={r.returncode}")
    check(f"  and the message names DATABASE_URL", explains)

print()
print("=" * 72)
print("3. the refusal explains the consequence, not just the fix")
print("=" * 72)
r = boot({"RAILWAY_ENVIRONMENT": "production"})
msg = (r.stdout or "") + (r.stderr or "")
check("says the data would be lost", "redeploy" in msg.lower() or "empty" in msg.lower())
check("offers an escape hatch", "ALLOW_SQLITE" in msg)

print()
print("=" * 72)
print("4. ALLOW_SQLITE=1 opts out, for a genuinely file-backed deploy")
print("=" * 72)
r = boot({"RAILWAY_ENVIRONMENT": "production", "ALLOW_SQLITE": "1"})
check("starts when explicitly allowed", r.returncode == 0, r.stdout.strip()[:60])

print()
print("=" * 72)
print("5. a real DATABASE_URL is never blocked")
print("=" * 72)
PG = "postgresql://aim:aimpw@127.0.0.1:55432/aimdb"
r = boot({"RAILWAY_ENVIRONMENT": "production", "DATABASE_URL": PG},
         code="import db, app; print('BACKEND=postgres' if db.using_postgres() else 'BACKEND=sqlite')")
combined = (r.stdout or "") + (r.stderr or "")
check("the ephemeral guard does not fire", "no PostgreSQL DATABASE_URL" not in combined,
      combined.strip()[:80])
check("it boots against real Postgres", r.returncode == 0, combined.strip()[:100])
check("and selects postgres", "BACKEND=postgres" in combined, combined.strip()[:60])

print()
print("=" * 72)
print("6. /api/health reports the backend without leaking anything")
print("=" * 72)
os.chdir(DASH)
os.environ.pop("DATABASE_URL", None)
os.environ["GOOGLE_CLIENT_ID"] = "x"
os.environ["GOOGLE_CLIENT_SECRET"] = "y"
os.environ.pop("RAILWAY_ENVIRONMENT", None)
import contextlib  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import app as application

application.app.config.update(TESTING=True)
c = application.app.test_client()

res = c.get("/api/health")
body = res.get_json()
print(f"  GET /api/health -> {res.status_code} {body}")
check("returns 200 without authentication", res.status_code == 200)
check("names the backend", body.get("backend") == "sqlite")
check("says persistence is false", body.get("persistent") is False)
check("includes a warning", bool(body.get("warning")))

raw = res.get_data(as_text=True)
check("does not expose DATABASE_URL", "DATABASE_URL" not in raw
      or "postgresql" not in raw)
check("does not expose any password", "://" not in raw or "@" not in raw)
check("does not leak SECRET_KEY", "SECRET_KEY" not in raw)

print()
print("=" * 72)
print("7. it reports postgres when that is what is configured")
print("=" * 72)
saved = dict(os.environ)
os.environ["DATABASE_URL"] = PG
try:
    for mod in ("db", "app", "auth", "pricing"):
        sys.modules.pop(mod, None)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import db as db2
        import app as application2

    application2.app.config.update(TESTING=True)
    c2 = application2.app.test_client()
    body2 = c2.get("/api/health").get_json()
    raw2 = c2.get("/api/health").get_data(as_text=True)
    print(f"  GET /api/health -> {body2}")
    check("reports postgresql", body2.get("backend") == "postgresql")
    check("reports persistent", body2.get("persistent") is True)
    check("no warning when persistent", body2.get("warning") is None)
    check("the password is not echoed", "aimpw" not in raw2)
    check("the host is not echoed", "127.0.0.1" not in raw2)
finally:
    os.environ.clear()
    os.environ.update(saved)

print()
print("FAIL: " + ", ".join(fails) if fails else "PASS: data loss is now prevented")