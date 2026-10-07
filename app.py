"""
AI System Health Dashboard
Real-time monitoring for LLM API calls with cost tracking, latency analysis, and anomaly detection

Multi-tenant: each Google account sees only its own calls, sessions, and anomalies.
"""

import os
from functools import wraps

from flask import Flask, render_template, jsonify, request, make_response
from flask import session, redirect, url_for
from flask_cors import CORS
from datetime import datetime, timedelta, timezone
import sqlite3
import json
import time
import threading
from collections import defaultdict
import statistics

from monitor_wrapper import build_wrapper
from auth import (
    assert_configured,
    where_to_get_credentials,
    init_oauth,
    init_users_table,
    is_local_host,
    oauth,
    get_or_create_user,
    user_for_token,
    get_user,
    rotate_ingest_token,
    GOOGLE_REDIRECT_URI,
)

# Timezone offset (Egypt/Alexandria = UTC+3)
# Change this if you're in a different timezone
TZ_OFFSET_HOURS = 3  # Egypt: UTC+3, adjust as needed

def to_local_time(utc_timestamp_str):
    """Convert UTC timestamp string to local time"""
    if not utc_timestamp_str:
        return None
    try:
        dt = datetime.fromisoformat(utc_timestamp_str.replace('Z', '+00:00'))
        local_dt = dt + timedelta(hours=TZ_OFFSET_HOURS)
        return local_dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return utc_timestamp_str

def bootstrap():
    """Refuse to boot without usable auth credentials.

    Prints actionable guidance rather than a stack trace, since the most
    common cause is simply not having filled in .env yet.
    """
    try:
        assert_configured()
    except SystemExit as exc:
        print(f"\n{exc}\n", flush=True)
        print(where_to_get_credentials(), "\n", flush=True)
        raise SystemExit(1) from None


bootstrap()

app = Flask(__name__)
init_oauth(app)

# Browser reads only; the UI is same-origin. The telemetry endpoint accepts a
# cross-origin ingest token, so CORS stays scoped to that one route.
CORS(app, resources={r"/api/log": {"origins": "*"}})

# Database setup
DB_PATH = "monitor.db"


def current_user_id():
    """The signed-in user id, or None."""
    return session.get("user_id")


def login_required(view):
    """Guard a browser route behind a Google sign-in."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user_id() is None:
            if request.path.startswith("/api/"):
                return jsonify({"error": "authentication required"}), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


def token_from_request():
    """Pull the ingest token out of the header or query string.

    The query fallback exists because some shell tools and cron jobs find
    header flags awkward. Both paths are equally unguessable.
    """
    header = request.headers.get("X-Ingest-Token", "")
    if header:
        return header.strip()
    return (request.args.get("token") or "").strip()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS api_calls
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                  user_id INTEGER)""")

    c.execute("""CREATE TABLE IF NOT EXISTS anomalies
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  anomaly_type TEXT,
                  severity TEXT,
                  description TEXT,
                  call_id INTEGER,
                  user_id INTEGER)""")

    c.execute("""CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT,
                 start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                 is_active INTEGER DEFAULT 1,
                 user_id INTEGER)""")

    # Databases created before multi-user support lack these columns; add them
    # in place so existing rows survive. Their user_id stays NULL, which is
    # never served to anyone.
    for table in ("api_calls", "anomalies", "sessions"):
        cols = {r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
        if "user_id" not in cols:
            c.execute(f"ALTER TABLE {table} ADD COLUMN user_id INTEGER")

    c.execute("CREATE INDEX IF NOT EXISTS idx_calls_user ON api_calls(user_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_anom_user ON anomalies(user_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")

    conn.commit()
    conn.close()


init_db()
init_users_table(DB_PATH)
# Cost calculation (per 1M tokens)
COST_TABLE = {
    "groq/llama-3.1-70b": {"input": 0.59, "output": 0.79},
    "groq/llama-3.1-8b": {"input": 0.05, "output": 0.08},
    "gemini/gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini/gemini-1.5-pro": {"input": 0.35, "output": 1.05},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.150, "output": 0.600},
}

def calculate_cost(provider, model, prompt_tokens, completion_tokens):
    """Cost in USD. Matches the full model id against the table by prefix, so
    real ids like 'llama-3.1-8b-instant' resolve to 'groq/llama-3.1-8b'."""
    provider = (provider or "").lower()
    model = (model or "").lower()

    rates = None
    for key, value in COST_TABLE.items():
        prefix, wanted = key.split("/", 1)
        if prefix == provider and model.startswith(wanted):
            # Longest match wins, so gpt-4o does not swallow gpt-4o-mini.
            if rates is None or len(wanted) > len(rates[0]):
                rates = (wanted, value)

    if rates is None:
        return 0.0

    table = rates[1]
    input_cost = (prompt_tokens / 1_000_000) * table["input"]
    output_cost = (completion_tokens / 1_000_000) * table["output"]
    return round(input_cost + output_cost, 8)

def detect_anomalies(call_id, latency_ms, cost_usd, provider, user_id):
    """Detect performance and cost anomalies"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get recent calls for baseline (last 100 calls for same provider)
    # Baseline is per user and per provider, so one tenant's traffic never
    # skews another's thresholds.
    c.execute('''SELECT latency_ms, cost_usd FROM api_calls 
                 WHERE provider = ? AND status = 'success' AND user_id = ?
                 ORDER BY timestamp DESC LIMIT 100''', (provider, user_id))
    recent = c.fetchall()
    
    if len(recent) < 10:  # Need baseline data
        conn.close()
        return
    
    latencies = [r[0] for r in recent]
    costs = [r[1] for r in recent]
    
    avg_latency = statistics.mean(latencies)
    stdev_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
    
    avg_cost = statistics.mean(costs)
    stdev_cost = statistics.stdev(costs) if len(costs) > 1 else 0
    
    # Latency spike detection (> 2 standard deviations)
    if stdev_latency > 0 and latency_ms > avg_latency + (2 * stdev_latency):
        c.execute('''INSERT INTO anomalies (anomaly_type, severity, description, call_id, user_id)
                     VALUES (?, ?, ?, ?, ?)''',
                  ("latency_spike", "warning", 
                   f"Latency {latency_ms}ms is {round((latency_ms/avg_latency - 1) * 100)}% above baseline {round(avg_latency)}ms",
                   call_id, user_id))
    
    # Cost spike detection
    if stdev_cost > 0 and cost_usd > avg_cost + (2 * stdev_cost):
        c.execute('''INSERT INTO anomalies (anomaly_type, severity, description, call_id, user_id)
                     VALUES (?, ?, ?, ?, ?)''',
                  ("cost_spike", "critical",
                   f"Cost ${cost_usd} is {round((cost_usd/avg_cost - 1) * 100)}% above baseline ${round(avg_cost, 4)}",
                   call_id, user_id))
    
    conn.commit()
    conn.close()

# --------------------------------------------------------------------------
# auth routes
# --------------------------------------------------------------------------
@app.route('/login')
def login():
    if current_user_id():
        return redirect(url_for("index"))

    # Keep the callback on the same host the user started from. Cookies are
    # scoped per host, so mixing localhost and 127.0.0.1 across the two hops
    # loses the session and trips Authlib's state check.
    redirect_uri = GOOGLE_REDIRECT_URI
    if is_local_host(request.host):
        scheme = request.scheme
        redirect_uri = f"{scheme}://{request.host}/auth/google/callback"

    return oauth.google.authorize_redirect(redirect_uri=redirect_uri)


@app.route('/auth/google/callback')
def google_callback():
    state = request.args.get("state")

    if not state or not any(
        k.startswith("_state_google_") and k.endswith(state)
        for k in list(session.keys())
    ):
        # The stored state is gone: cookie expired, host changed between the
        # two hops, or the callback URL was reloaded. Retrying the browser
        # step once almost always works, so offer that instead of a dead end.
        session.pop("_fresh_state", None)
        return render_template(
            "login.html",
            error="That sign-in link expired or was already used. "
                  "Click below to start a fresh one.",
            retry=True,
        ), 400

    try:
        token = oauth.google.authorize_access_token()
    except Exception as exc:
        message = str(exc)
        if "mismatching_state" in message:
            return render_template(
                "login.html",
                error="Sign-in state check failed. This usually means the browser "
                      "sent the callback to a different address than the one you "
                      "started from. Try again using the same address "
                      "(localhost or 127.0.0.1, not both).",
                retry=True,
            ), 400
        return render_template("login.html", error=f"Sign-in failed: {message}"), 400

    info = token.get("userinfo")
    if not info or not info.get("sub"):
        return render_template("login.html", error="Google did not return an account."), 400

    # Google verifies email; treat the address as unverified unless it says so.
    verified = bool(info.get("email_verified"))
    if not verified:
        return render_template(
            "login.html",
            error="Your Google email is not verified. Verify it at "
                  "myaccount.google.com/email and try again.",
        ), 400

    user_id, email, name, _, token_issued = get_or_create_user(
        DB_PATH, info["sub"], info.get("email", ""),
        info.get("name", ""), info.get("picture", ""),
    )
    session["user_id"] = user_id
    session["email"] = email
    session.permanent = True

    if token_issued:
        # Shown once, so the user can put it in their wrapper.
        session["new_ingest_token"] = token_issued
    return redirect(url_for("index"))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route('/api/account')
@login_required
def account():
    uid = current_user_id()
    row = get_user(DB_PATH, uid)
    return jsonify({
        "id": uid,
        "email": row["email"] if row else session.get("email"),
        "name": row["name"] if row else None,
        "picture": row["picture"] if row else None,
        # Returned only in the response right after first sign-in.
        "ingest_token": session.pop("new_ingest_token", None),
    })


@app.route('/api/account/token/rotate', methods=['POST'])
@login_required
def rotate_token():
    """Issue a new ingest token. The old one stops working immediately."""
    token = rotate_ingest_token(DB_PATH, current_user_id())
    return jsonify({"ingest_token": token})


# --------------------------------------------------------------------------
# telemetry ingest (token auth, not session auth)
# --------------------------------------------------------------------------
@app.route('/api/log', methods=['POST'])
def log_api_call():
    """Record one LLM call. Authenticated by ingest token, since the caller is
    ai_monitor.py rather than a browser."""
    user_id = user_for_token(DB_PATH, token_from_request())
    if user_id is None:
        return jsonify({
            "success": False,
            "error": "invalid ingest token. Sign in and copy your token from /api/account",
        }), 401

    data = request.get_json(silent=True) or {}
    provider = data.get('provider', 'unknown')
    model = data.get('model', 'unknown')

    prompt_tokens = int(data.get('prompt_tokens', 0) or 0)
    completion_tokens = int(data.get('completion_tokens', 0) or 0)
    total_tokens = prompt_tokens + completion_tokens
    latency_ms = int(data.get('latency_ms', 0) or 0)
    status = data.get('status', 'success')
    error_message = data.get('error_message', None)

    cost_usd = calculate_cost(provider, model, prompt_tokens, completion_tokens)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO api_calls
                 (provider, model, prompt_tokens, completion_tokens, total_tokens,
                  latency_ms, cost_usd, status, error_message, user_id)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (provider, model, prompt_tokens, completion_tokens, total_tokens,
               latency_ms, cost_usd, status, error_message, user_id))
    call_id = c.lastrowid
    conn.commit()
    conn.close()

    if status == 'success':
        threading.Thread(
            target=detect_anomalies,
            args=(call_id, latency_ms, cost_usd, provider, user_id),
            daemon=True,
        ).start()

    return jsonify({"success": True, "call_id": call_id, "cost_usd": cost_usd})


# --------------------------------------------------------------------------
# dashboard reads (session auth)
# --------------------------------------------------------------------------
@app.route('/')
def index():
    """Landing page when signed out, dashboard when signed in."""
    if current_user_id() is None:
        return render_template('landing.html')
    return render_template('index.html')


@app.route('/api/stats')
@login_required
def get_stats():
    """Statistics for the signed-in user only."""
    uid = current_user_id()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    session_id = request.args.get('session_id', None)
    if session_id:
        c.execute('SELECT start_time FROM sessions WHERE id = ? AND user_id = ?',
                  (session_id, uid))
        result = c.fetchone()
        cutoff = result[0] if result else datetime.utcnow() - timedelta(hours=24)
    else:
        hours = int(request.args.get('hours', 24))
        cutoff = datetime.utcnow() - timedelta(hours=hours)

    c.execute("""SELECT COUNT(*),
                 SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END),
                 SUM(total_tokens),
                 SUM(cost_usd),
                 AVG(latency_ms)
                 FROM api_calls
                 WHERE timestamp > ? AND user_id = ?""", (cutoff, uid))

    total_calls, successful, total_tokens, total_cost, avg_latency = c.fetchone()
    total_calls = total_calls or 0
    successful = successful or 0
    total_tokens = total_tokens or 0
    total_cost = total_cost or 0.0
    avg_latency = avg_latency or 0

    success_rate = (successful / total_calls * 100) if total_calls > 0 else 100

    c.execute("""SELECT provider, COUNT(*), SUM(cost_usd), AVG(latency_ms)
                 FROM api_calls
                 WHERE timestamp > ? AND status = 'success' AND user_id = ?
                 GROUP BY provider""", (cutoff, uid))

    provider_stats = []
    for row in c.fetchall():
        provider_stats.append({
            "provider": row[0],
            "calls": row[1],
            "cost": round(row[2] or 0, 4),
            "avg_latency_ms": round(row[3] or 0, 1)
        })

    c.execute("""SELECT datetime(timestamp, ?) as local_time, anomaly_type, severity, description
                 FROM anomalies
                 WHERE user_id = ?
                 ORDER BY timestamp DESC LIMIT 10""", (f"+{TZ_OFFSET_HOURS} hours", uid))

    anomalies = []
    for row in c.fetchall():
        anomalies.append({
            "timestamp": row[0],
            "type": row[1],
            "severity": row[2],
            "description": row[3]
        })

    c.execute("""SELECT strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                 COUNT(*) as count
                 FROM api_calls
                 WHERE timestamp > ? AND user_id = ?
                 GROUP BY hour
                 ORDER BY hour""", (cutoff, uid))

    hourly_volume = [{"hour": row[0], "count": row[1]} for row in c.fetchall()]
    conn.close()

    return jsonify({
        "total_calls": total_calls,
        "success_rate": round(success_rate, 2),
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "provider_stats": provider_stats,
        "anomalies": anomalies,
        "hourly_volume": hourly_volume
    })


@app.route('/api/recent-calls')
@login_required
def get_recent_calls():
    """Recent calls for the signed-in user only."""
    uid = current_user_id()
    limit = int(request.args.get('limit', 50))

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"""SELECT datetime(timestamp, '+{TZ_OFFSET_HOURS} hours'), provider, model,
                         total_tokens, latency_ms, cost_usd, status, error_message
                 FROM api_calls
                 WHERE user_id = ?
                 ORDER BY timestamp DESC LIMIT ?""", (uid, limit))

    calls = []
    for row in c.fetchall():
        calls.append({
            "timestamp": row[0],
            "provider": row[1],
            "model": row[2],
            "tokens": row[3],
            "latency_ms": row[4],
            "cost_usd": round(row[5], 8),
            "status": row[6],
            "error": row[7]
        })

    conn.close()
    return jsonify(calls)


@app.route('/api/simulate')
@login_required
def simulate_traffic():
    """Generate sample traffic for demo purposes."""
    import random

    providers = [
        ("groq", "llama-3.1-70b"),
        ("groq", "llama-3.1-8b"),
        ("gemini", "gemini-1.5-flash"),
        ("openai", "gpt-4o-mini")
    ]

    uid = current_user_id()
    generated = 0
    for _ in range(20):
        provider, model = random.choice(providers)
        prompt_tokens = random.randint(100, 2000)
        completion_tokens = random.randint(50, 1000)
        latency_ms = random.randint(200, 5000)
        status = "success" if random.random() > 0.05 else "error"

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""INSERT INTO api_calls
                     (provider, model, prompt_tokens, completion_tokens, total_tokens,
                      latency_ms, cost_usd, status, error_message, user_id)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (provider, model, prompt_tokens, completion_tokens,
                   prompt_tokens + completion_tokens, latency_ms,
                   calculate_cost(provider, model, prompt_tokens, completion_tokens),
                   status, "Timeout" if status == "error" else None, uid))
        conn.commit()
        conn.close()
        generated += 1

    return jsonify({"success": True, "generated": generated})


# --------------------------------------------------------------------------
# sessions (session auth, scoped per user)
# --------------------------------------------------------------------------
@app.route('/api/sessions/start', methods=['POST'])
@login_required
def start_session():
    """Start a new monitoring session for this user only."""
    uid = current_user_id()
    data = request.json or {}
    session_name = data.get('name', f"Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE sessions SET is_active = 0 WHERE user_id = ?', (uid,))
    c.execute('INSERT INTO sessions (name, start_time, user_id) VALUES (?, ?, ?)',
              (session_name, datetime.utcnow(), uid))
    session_id = c.lastrowid
    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "session_id": session_id,
        "name": session_name,
        "start_time": datetime.utcnow().isoformat()
    })


@app.route('/api/sessions/active')
@login_required
def get_active_session():
    """This user's active session, if any."""
    uid = current_user_id()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"""SELECT id, name, datetime(start_time, '+{TZ_OFFSET_HOURS} hours')
                 FROM sessions
                 WHERE is_active = 1 AND user_id = ?
                 ORDER BY start_time DESC LIMIT 1""", (uid,))
    result = c.fetchone()
    conn.close()

    if result:
        return jsonify({"session_id": result[0], "name": result[1], "start_time": result[2]})
    return jsonify({"session_id": None})


@app.route('/api/sessions/stop', methods=['POST'])
@login_required
def stop_session():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE sessions SET is_active = 0 WHERE user_id = ?', (current_user_id(),))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


# --------------------------------------------------------------------------
# wrapper download (session auth; bakes in the caller's own token)
# --------------------------------------------------------------------------
@app.route('/api/projects/add', methods=['POST'])
@login_required
def add_project():
    """Generate ai_monitor.py, pre-filled with this user's ingest token.

    Download-only by design: a browser cannot hand a page an absolute local
    path, and a server on someone else's machine could not use one anyway.
    """
    uid = current_user_id()
    data = request.json or {}
    project_name = data.get('project_name') or 'my_project'
    monitor_url = (request.args.get('monitor_url')
                   or data.get('monitor_url')
                   or request.host_url.rstrip('/'))

    row = get_user(DB_PATH, uid)
    fresh_token = session.pop("new_ingest_token", None)
    ingest_token = fresh_token or request.args.get('token_preview') or ''

    source = build_wrapper(project_name, monitor_url, ingest_token)

    response = make_response(source)
    response.headers['Content-Type'] = 'text/x-python; charset=utf-8'
    response.headers['Content-Disposition'] = 'attachment; filename=ai_monitor.py'
    return response



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
