"""
AI System Health Dashboard
Real-time monitoring for LLM API calls with cost tracking, latency analysis, and anomaly detection
"""

from flask import Flask, render_template, jsonify, request, make_response
from flask_cors import CORS
from datetime import datetime, timedelta, timezone
import sqlite3
import json
import time
import threading
from collections import defaultdict
import statistics

from monitor_wrapper import build_wrapper

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

app = Flask(__name__)
CORS(app)

# Database setup
DB_PATH = "monitor.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS api_calls
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
                  error_message TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS anomalies
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  anomaly_type TEXT,
                  severity TEXT,
                  description TEXT,
                  call_id INTEGER)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                  is_active INTEGER DEFAULT 1)''')
    conn.commit()
    conn.close()

init_db()

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

def detect_anomalies(call_id, latency_ms, cost_usd, provider):
    """Detect performance and cost anomalies"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get recent calls for baseline (last 100 calls for same provider)
    c.execute('''SELECT latency_ms, cost_usd FROM api_calls 
                 WHERE provider = ? AND status = 'success' 
                 ORDER BY timestamp DESC LIMIT 100''', (provider,))
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
        c.execute('''INSERT INTO anomalies (anomaly_type, severity, description, call_id)
                     VALUES (?, ?, ?, ?)''',
                  ("latency_spike", "warning", 
                   f"Latency {latency_ms}ms is {round((latency_ms/avg_latency - 1) * 100)}% above baseline {round(avg_latency)}ms",
                   call_id))
    
    # Cost spike detection
    if stdev_cost > 0 and cost_usd > avg_cost + (2 * stdev_cost):
        c.execute('''INSERT INTO anomalies (anomaly_type, severity, description, call_id)
                     VALUES (?, ?, ?, ?)''',
                  ("cost_spike", "critical",
                   f"Cost ${cost_usd} is {round((cost_usd/avg_cost - 1) * 100)}% above baseline ${round(avg_cost, 4)}",
                   call_id))
    
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/log', methods=['POST'])
def log_api_call():
    """Log an LLM API call"""
    data = request.json
    
    provider = data.get('provider', 'unknown')
    model = data.get('model', 'unknown')

    prompt_tokens = data.get('prompt_tokens', 0)
    completion_tokens = data.get('completion_tokens', 0)
    total_tokens = prompt_tokens + completion_tokens
    latency_ms = data.get('latency_ms', 0)
    status = data.get('status', 'success')
    error_message = data.get('error_message', None)

    cost_usd = calculate_cost(provider, model, prompt_tokens, completion_tokens)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO api_calls 
                 (provider, model, prompt_tokens, completion_tokens, total_tokens, 
                  latency_ms, cost_usd, status, error_message)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (provider, model, prompt_tokens, completion_tokens, total_tokens,
               latency_ms, cost_usd, status, error_message))
    
    call_id = c.lastrowid
    conn.commit()
    conn.close()
    
    # Run anomaly detection in background
    if status == 'success':
        threading.Thread(target=detect_anomalies, args=(call_id, latency_ms, cost_usd, provider)).start()
    
    return jsonify({"success": True, "call_id": call_id, "cost_usd": cost_usd})

@app.route('/api/stats')
def get_stats():
    """Get dashboard statistics"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check for active session
    session_id = request.args.get('session_id', None)
    if session_id:
        c.execute('SELECT start_time FROM sessions WHERE id = ?', (session_id,))
        result = c.fetchone()
        if result:
            cutoff = result[0]
        else:
            cutoff = datetime.utcnow() - timedelta(hours=24)
    else:
        # Time filter (default: last 24 hours)
        # Use UTC for cutoff since database stores timestamps in UTC
        hours = int(request.args.get('hours', 24))
        cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    # Total calls and success rate
    c.execute('''SELECT COUNT(*), 
                 SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END),
                 SUM(total_tokens),
                 SUM(cost_usd),
                 AVG(latency_ms)
                 FROM api_calls 
                 WHERE timestamp > ?''', (cutoff,))
    
    total_calls, successful, total_tokens, total_cost, avg_latency = c.fetchone()
    total_calls = total_calls or 0
    successful = successful or 0
    total_tokens = total_tokens or 0
    total_cost = total_cost or 0.0
    avg_latency = avg_latency or 0
    
    success_rate = (successful / total_calls * 100) if total_calls > 0 else 100
    
    # Provider breakdown
    c.execute('''SELECT provider, COUNT(*), SUM(cost_usd), AVG(latency_ms)
                 FROM api_calls 
                 WHERE timestamp > ? AND status = 'success'
                 GROUP BY provider''', (cutoff,))
    
    provider_stats = []
    for row in c.fetchall():
        provider_stats.append({
            "provider": row[0],
            "calls": row[1],
            "cost": round(row[2] or 0, 4),
            "avg_latency_ms": round(row[3] or 0, 1)
        })
    
    # Recent anomalies
    c.execute('''SELECT datetime(timestamp, '+3 hours') as local_time, anomaly_type, severity, description
                 FROM anomalies 
                 ORDER BY timestamp DESC LIMIT 10''')
    
    anomalies = []
    for row in c.fetchall():
        anomalies.append({
            "timestamp": row[0],
            "type": row[1],
            "severity": row[2],
            "description": row[3]
        })
    
    # Hourly call volume (last 24 hours)
    c.execute('''SELECT strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                 COUNT(*) as count
                 FROM api_calls
                 WHERE timestamp > ?
                 GROUP BY hour
                 ORDER BY hour''', (cutoff,))
    
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
def get_recent_calls():
    """Get recent API call logs"""
    limit = int(request.args.get('limit', 50))
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT datetime(timestamp, '+3 hours') as local_time, provider, model, total_tokens, latency_ms, 
                 cost_usd, status, error_message
                 FROM api_calls 
                 ORDER BY timestamp DESC LIMIT ?''', (limit,))
    
    calls = []
    for row in c.fetchall():
        calls.append({
            "timestamp": row[0],
            "provider": row[1],
            "model": row[2],
            "tokens": row[3],
            "latency_ms": row[4],
            "cost_usd": round(row[5], 6),
            "status": row[6],
            "error": row[7]
        })
    
    conn.close()
    return jsonify(calls)

@app.route('/api/simulate')
def simulate_traffic():
    """Generate sample traffic for demo purposes"""
    import random
    
    providers = [
        ("groq", "llama-3.1-70b"),
        ("groq", "llama-3.1-8b"),
        ("gemini", "gemini-1.5-flash"),
        ("openai", "gpt-4o-mini")
    ]
    
    for _ in range(20):
        provider, model = random.choice(providers)
        prompt_tokens = random.randint(100, 2000)
        completion_tokens = random.randint(50, 1000)
        latency_ms = random.randint(200, 5000)
        status = "success" if random.random() > 0.05 else "error"
        
        data = {
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "status": status,
            "error_message": "Timeout" if status == "error" else None
        }
        
        # Log the call
        with app.test_request_context('/api/log', json=data):
            log_api_call()
    
    return jsonify({"success": True, "generated": 20})

@app.route('/api/projects/add', methods=['POST'])
def add_project():
    """Generate the ai_monitor.py wrapper and return it as a download.

    Download-only by design. A browser cannot hand a web page an absolute
    local path, and a server on someone else's machine could not use one
    anyway, so writing straight into a project folder is not reachable from
    here. The user downloads the file and drops it in.
    """
    data = request.json or {}
    project_name = data.get('project_name') or 'my_project'
    monitor_url = request.args.get('monitor_url') or data.get('monitor_url') or request.host_url.rstrip('/')

    source = build_wrapper(project_name, monitor_url)

    response = make_response(source)
    response.headers['Content-Type'] = 'text/x-python; charset=utf-8'
    response.headers['Content-Disposition'] = 'attachment; filename=ai_monitor.py'
    return response

@app.route('/api/sessions/start', methods=['POST'])
def start_session():
    """Start a new monitoring session"""
    data = request.json or {}
    session_name = data.get('name', f"Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Deactivate all previous sessions
    c.execute('UPDATE sessions SET is_active = 0')
    
    # Create new session
    c.execute('INSERT INTO sessions (name, start_time) VALUES (?, ?)', 
              (session_name, datetime.utcnow()))
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
def get_active_session():
    """Get the currently active session"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''SELECT id, name, datetime(start_time, '+3 hours') as local_time 
                 FROM sessions 
                 WHERE is_active = 1 
                 ORDER BY start_time DESC LIMIT 1''')
    result = c.fetchone()
    conn.close()
    
    if result:
        return jsonify({
            "session_id": result[0],
            "name": result[1],
            "start_time": result[2]
        })
    else:
        return jsonify({"session_id": None})

@app.route('/api/sessions/stop', methods=['POST'])
def stop_session():
    """Stop the active session and return to time-based filtering"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE sessions SET is_active = 0')
    conn.commit()
    conn.close()
    
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
