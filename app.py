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
import json
import time
import threading
from collections import defaultdict
import statistics

import db
import pricing
from db import cursor, init_db, table_columns, local_time_sql, hour_bucket_sql
from monitor_wrapper import build_wrapper
from auth import (
    assert_configured,
    where_to_get_credentials,
    init_oauth,
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
TZ_OFFSET_HOURS = db.TZ_OFFSET_HOURS  # Egypt: UTC+3, adjust as needed

init_db()

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

# Pull in LiteLLM now, in the background, so the first call to an unpriced
# model does not stall its telemetry POST on a five-second import. Daemon, so it
# never holds up shutdown, and its failure is ignored: pricing is a fallback,
# not a precondition for serving.
threading.Thread(target=pricing.warm_up, daemon=True).start()

# Browser reads only; the UI is same-origin. The telemetry endpoint accepts a
# cross-origin ingest token, so CORS stays scoped to that one route.
CORS(app, resources={r"/api/log": {"origins": "*"}})

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

# Schema creation and the local-time SQL live in db.py, which speaks both
# SQLite and PostgreSQL.


# Cost calculation (per 1M tokens)
COST_TABLE = {
    # Groq
    "groq/llama-3.3-70b": {"input": 0.59, "output": 0.79},
    "groq/llama-3.1-70b": {"input": 0.59, "output": 0.79},
    "groq/llama-3.1-8b": {"input": 0.05, "output": 0.08},
    "groq/mixtral-8x7b": {"input": 0.24, "output": 0.24},
    "groq/qwen3.8-27b": {"input": 0.80, "output": 4.00},
    # Gemini
    "gemini/gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "gemini/gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini/gemini-2.0-pro": {"input": 0.50, "output": 1.50},
    "gemini/gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini/gemini-1.5-pro": {"input": 0.35, "output": 1.05},
    # OpenAI
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "openai/gpt-4.5": {"input": 75.00, "output": 150.00},
    "openai/o1": {"input": 15.00, "output": 60.00},
    "openai/o3-mini": {"input": 1.10, "output": 4.40},
    # Anthropic
    "anthropic/claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "anthropic/claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "anthropic/claude-3-opus": {"input": 15.00, "output": 75.00},
    # DeepSeek
    "deepseek/deepseek-v3": {"input": 0.14, "output": 0.28},
    "deepseek/deepseek-r1": {"input": 0.55, "output": 2.19},
    # Mistral
    "mistral/mistral-large": {"input": 2.00, "output": 6.00},
    "mistral/mistral-small": {"input": 0.20, "output": 0.60},
    # Local inference is free; listed so it is distinguished from "unknown".
    "ollama/*": {"input": 0.0, "output": 0.0},
}

def _format_ts(value):
    """Render a timestamp as 'YYYY-MM-DD HH:MM:SS' for the JSON response.

    SQLite hands back a string; PostgreSQL returns a real datetime, which
    jsonify cannot serialise on its own.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def price_for(provider, model):
    """Resolve a rate table, or None when the model is not priced anywhere.

    COST_TABLE is checked first and wins when it matches. LiteLLM is the
    fallback for models the table has not caught up with, such as a provider
    releasing a new id between releases of this file.

    Matching within the table is by longest prefix, so 'gpt-4o' is not
    swallowed by the 'gpt-4o-mini' row and 'llama-3.1-8b-instant' resolves to
    the 8b row.

    Some providers namespace their ids ('qwen/qwen3.8-27b' on Groq), so the
    segment after the last slash is tried as well. Without that, a table key
    of 'groq/qwen3.8-27b' would never match, and the call would silently read
    as unpriced.
    """
    table = _table_rate(provider, model)
    if table is not None:
        return table
    return pricing.lookup_rates(provider, model)


def _table_rate(provider, model):
    """COST_TABLE-only lookup, with no fallback."""
    provider = (provider or "").lower()
    model = (model or "").lower()

    candidates = [model]
    if "/" in model:
        tail = model.rsplit("/", 1)[1]
        if tail and tail != model:
            candidates.append(tail)

    best = None
    for key, value in COST_TABLE.items():
        prefix, wanted = key.split("/", 1)
        if prefix != provider:
            continue
        wildcard = wanted.endswith("*")
        # Longest literal match wins, so gpt-4o beats gpt-4o-mini and a
        # catch-all never shadows a specific entry.
        weight = 0 if wildcard else len(wanted)
        for candidate in candidates:
            if wildcard:
                matched = bool(candidate)
            else:
                matched = candidate.startswith(wanted)
            if matched and (best is None or weight > best[0]):
                best = (weight, value)
    return best[1] if best else None


def calculate_cost(provider, model, prompt_tokens, completion_tokens):
    """Cost in USD, or None when the model has no entry in COST_TABLE.

    None matters: returning 0.0 for an unpriced model makes a real bill look
    free. Callers surface the difference instead of quietly reporting zero.
    """
    table = price_for(provider, model)
    if table is None:
        return None

    input_cost = (prompt_tokens / 1_000_000) * table["input"]
    output_cost = (completion_tokens / 1_000_000) * table["output"]
    return round(input_cost + output_cost, 8)

def detect_anomalies(call_id, latency_ms, cost_usd, provider, user_id):
    """Detect performance and cost anomalies"""
    with cursor() as c:
        # Get recent calls for baseline (last 100 calls for same provider)
        # Baseline is per user and per provider, so one tenant's traffic never
        # skews another's thresholds.
        c.execute('''SELECT latency_ms, cost_usd FROM api_calls
                     WHERE provider = %s AND status = 'success' AND user_id = %s
                     ORDER BY timestamp DESC, id DESC LIMIT 100''', (provider, user_id))
        recent = c.fetchall()

        if len(recent) < 10:  # Need baseline data
            return

        # cost_usd is Decimal on PostgreSQL; normalise before arithmetic.
        latencies = [float(r["latency_ms"]) for r in recent]
        costs = [float(r["cost_usd"]) for r in recent if r["cost_usd"] is not None]

        avg_latency = statistics.mean(latencies)
        stdev_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0

        avg_cost = statistics.mean(costs) if costs else 0
        stdev_cost = statistics.stdev(costs) if len(costs) > 1 else 0

        # Latency spike detection (> 2 standard deviations)
        if stdev_latency > 0 and latency_ms > avg_latency + (2 * stdev_latency):
            c.execute('''INSERT INTO anomalies (anomaly_type, severity, description, call_id, user_id)
                         VALUES (%s, %s, %s, %s, %s)''',
                      ("latency_spike", "warning",
                       f"Latency {latency_ms}ms is {round((latency_ms/avg_latency - 1) * 100)}% above baseline {round(avg_latency)}ms",
                       call_id, user_id))

        # Cost spike detection. Skip when this call has no priced rate, so an
        # unpriced model does not trigger a spike against a priced baseline.
        if cost_usd is not None and stdev_cost > 0 and float(cost_usd) > avg_cost + (2 * stdev_cost):
            c.execute('''INSERT INTO anomalies (anomaly_type, severity, description, call_id, user_id)
                         VALUES (%s, %s, %s, %s, %s)''',
                      ("cost_spike", "critical",
                       f"Cost ${cost_usd} is {round((float(cost_usd)/avg_cost - 1) * 100)}% above baseline ${round(avg_cost, 4)}",
                       call_id, user_id))

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
        info["sub"], info.get("email", ""),
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
    row = get_user(uid)
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
    token = rotate_ingest_token(current_user_id())
    return jsonify({"ingest_token": token})


# --------------------------------------------------------------------------
# telemetry ingest (token auth, not session auth)
# --------------------------------------------------------------------------
@app.route('/api/log', methods=['POST'])
def log_api_call():
    """Record one LLM call. Authenticated by ingest token, since the caller is
    ai_monitor.py rather than a browser."""
    user_id = user_for_token(token_from_request())
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

    with cursor() as c:
        c.execute("""INSERT INTO api_calls
                     (provider, model, prompt_tokens, completion_tokens, total_tokens,
                      latency_ms, cost_usd, status, error_message, user_id)
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                     RETURNING id""",
                  (provider, model, prompt_tokens, completion_tokens, total_tokens,
                   latency_ms, cost_usd, status, error_message, user_id))
        call_id = c.fetchone()["id"]

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

    with cursor() as c:
        session_id = request.args.get('session_id', None)
        if session_id:
            c.execute('SELECT start_time FROM sessions WHERE id = %s AND user_id = %s',
                      (session_id, uid))
            result = c.fetchone()
            cutoff = result["start_time"] if result else datetime.utcnow() - timedelta(hours=24)
        else:
            hours = int(request.args.get('hours', 24))
            cutoff = datetime.utcnow() - timedelta(hours=hours)

        c.execute("""SELECT COUNT(*) AS total_calls,
                     SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successful,
                     SUM(total_tokens) AS total_tokens,
                     SUM(cost_usd) AS total_cost,
                     AVG(latency_ms) AS avg_latency
                     FROM api_calls
                     WHERE timestamp > %s AND user_id = %s""", (cutoff, uid))
        agg = c.fetchone() or {}

        total_calls = agg.get("total_calls") or 0
        successful = agg.get("successful") or 0
        total_tokens = agg.get("total_tokens") or 0
        total_cost = agg.get("total_cost") or 0.0
        avg_latency = agg.get("avg_latency") or 0

        success_rate = (successful / total_calls * 100) if total_calls > 0 else 100

        c.execute("""SELECT provider, COUNT(*) AS calls, SUM(cost_usd) AS cost,
                            AVG(latency_ms) AS avg_latency,
                            SUM(CASE WHEN cost_usd IS NULL THEN 1 ELSE 0 END) AS unpriced
                     FROM api_calls
                     WHERE timestamp > %s AND status = 'success' AND user_id = %s
                     GROUP BY provider""", (cutoff, uid))

        provider_stats = []
        for row in c.fetchall():
            calls = row["calls"] or 0
            unpriced = row["unpriced"] or 0
            # Cost is per model, so a provider group can hold both priced and
            # unpriced calls. Any unpriced call makes this group's total
            # incomplete, so report n/a rather than a subtotal that looks whole.
            # float() also normalises PostgreSQL's Decimal.
            cost = None if unpriced else round(float(row["cost"] or 0), 4)
            provider_stats.append({
                "provider": row["provider"],
                "calls": calls,
                "cost": cost,
                "avg_latency_ms": round(float(row["avg_latency"] or 0), 1),
                "unpriced_calls": unpriced,
            })

        # Named separately from provider_stats, which groups by provider.
        c.execute("""SELECT DISTINCT provider, model FROM api_calls
                     WHERE timestamp > %s AND user_id = %s AND cost_usd IS NULL
                     LIMIT 10""", (cutoff, uid))
        unpriced_models = [f"{r['provider']}/{r['model']}" for r in c.fetchall()]

        c.execute(f"""SELECT {local_time_sql('timestamp')} AS local_time,
                             anomaly_type, severity, description
                      FROM anomalies
                      WHERE user_id = %s
                      ORDER BY timestamp DESC, id DESC LIMIT 10""", (uid,))

        anomalies = []
        for row in c.fetchall():
            anomalies.append({
                "timestamp": _format_ts(row["local_time"]),
                "type": row["anomaly_type"],
                "severity": row["severity"],
                "description": row["description"]
            })

        bucket = hour_bucket_sql('timestamp')
        c.execute(f"""SELECT {bucket} AS hour, COUNT(*) AS count
                      FROM api_calls
                      WHERE timestamp > %s AND user_id = %s
                      GROUP BY hour
                      ORDER BY hour""", (cutoff, uid))

        hourly_volume = [{"hour": row["hour"], "count": row["count"]} for row in c.fetchall()]

    # The total is only meaningful when every call in the window is priced.
    total_cost = None if unpriced_models else round(float(total_cost or 0), 4)

    return jsonify({
        "total_calls": total_calls,
        "success_rate": round(success_rate, 2),
        "total_tokens": total_tokens,
        # PostgreSQL NUMERIC arrives as Decimal, which jsonify renders as a string.
        # Coerce to float so the API contract is numeric on both backends.
        "total_cost": total_cost,
        "avg_latency_ms": round(avg_latency, 1),
        "provider_stats": provider_stats,
        "unpriced_models": unpriced_models,
        "anomalies": anomalies,
        "hourly_volume": hourly_volume
    })


@app.route('/api/recent-calls')
@login_required
def get_recent_calls():
    """Recent calls for the signed-in user only."""
    uid = current_user_id()
    limit = int(request.args.get('limit', 50))

    conn_local = local_time_sql('timestamp')
    with cursor() as c:
        # id breaks ties: SQLite timestamps have second precision, so several
        # rows written in the same second otherwise sort unpredictably.
        c.execute(f"""SELECT {conn_local} AS local_time, provider, model,
                             total_tokens, latency_ms, cost_usd, status, error_message
                      FROM api_calls
                      WHERE user_id = %s
                      ORDER BY timestamp DESC, id DESC LIMIT %s""", (uid, limit))

        calls = []
        for row in c.fetchall():
            calls.append({
                "timestamp": _format_ts(row["local_time"]),
                "provider": row["provider"],
                "model": row["model"],
                "tokens": row["total_tokens"],
                "latency_ms": row["latency_ms"],
                "cost_usd": (None if row["cost_usd"] is None
                            else round(float(row["cost_usd"]), 8)),
                "status": row["status"],
                "error": row["error_message"]
            })

    return jsonify(calls)


@app.route('/api/simulate')
@login_required
def simulate_traffic():
    """Generate sample traffic for demo purposes."""
    import random

    providers = [
        ("groq", "llama-3.3-70b"),
        ("groq", "llama-3.1-8b"),
        ("gemini", "gemini-3.8-flash"),
        ("gemini", "gemini-2.0-flash"),
        ("gemini", "gemini-1.5-flash"),
        ("openai", "gpt-4o"),
        ("openai", "gpt-4o-mini"),
        ("openai", "o3-mini"),
        ("anthropic", "claude-3-5-sonnet"),
        ("deepseek", "deepseek-r1"),
    ]

    uid = current_user_id()
    generated = 0
    for _ in range(20):
        provider, model = random.choice(providers)
        prompt_tokens = random.randint(100, 2000)
        completion_tokens = random.randint(50, 1000)
        latency_ms = random.randint(200, 5000)
        status = "success" if random.random() > 0.05 else "error"

        with cursor() as c:
            c.execute("""INSERT INTO api_calls
                         (provider, model, prompt_tokens, completion_tokens, total_tokens,
                          latency_ms, cost_usd, status, error_message, user_id)
                         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                      (provider, model, prompt_tokens, completion_tokens,
                       prompt_tokens + completion_tokens, latency_ms,
                       calculate_cost(provider, model, prompt_tokens, completion_tokens),
                       status, "Timeout" if status == "error" else None, uid))
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

    started = datetime.utcnow()
    with cursor() as c:
        c.execute('UPDATE sessions SET is_active = %s WHERE user_id = %s',
                  (False, uid))
        c.execute("""INSERT INTO sessions (name, start_time, user_id)
                     VALUES (%s, %s, %s) RETURNING id""",
                  (session_name, started, uid))
        session_id = c.fetchone()["id"]

    return jsonify({
        "success": True,
        "session_id": session_id,
        "name": session_name,
        "start_time": started.isoformat()
    })


@app.route('/api/sessions/active')
@login_required
def get_active_session():
    """This user's active session, if any."""
    uid = current_user_id()
    bucket = local_time_sql('start_time')
    with cursor() as c:
        c.execute(f"""SELECT id, name, {bucket} AS local_start
                     FROM sessions
                     WHERE is_active = %s AND user_id = %s
                     ORDER BY start_time DESC, id DESC LIMIT 1""", (True, uid))
        result = c.fetchone()

    if result:
        return jsonify({
            "session_id": result["id"],
            "name": result["name"],
            "start_time": _format_ts(result["local_start"]),
        })
    return jsonify({"session_id": None})


@app.route('/api/sessions/stop', methods=['POST'])
@login_required
def stop_session():
    with cursor() as c:
        c.execute('UPDATE sessions SET is_active = %s WHERE user_id = %s',
                  (False, current_user_id()))
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

    # Token sources, in order: freshly issued this session, then one the caller
    # pasted in. Refuse to emit a wrapper that cannot authenticate, since an
    # empty INGEST_TOKEN makes the file silently inert.
    fresh_token = session.pop("new_ingest_token", None)
    supplied = (data.get('ingest_token') or '').strip()
    ingest_token = fresh_token or supplied

    if not ingest_token:
        return jsonify({
            'success': False,
            'error': 'No ingest token. Copy it from My Account (or rotate it '
                     'for a fresh one) and paste it here.',
        }), 400

    source = build_wrapper(project_name, monitor_url, ingest_token)

    response = make_response(source)
    response.headers['Content-Type'] = 'text/x-python; charset=utf-8'
    response.headers['Content-Disposition'] = 'attachment; filename=ai_monitor.py'
    response.headers['X-Ingest-Token-Embedded'] = 'true'
    return response



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
