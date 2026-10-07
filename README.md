# AI System Health Dashboard

Real-time monitoring dashboard for LLM API calls with cost tracking, latency analysis, and anomaly detection.

## Features

- **Real-time Performance Monitoring**: Track API call latency, token usage, and success rates
- **Cost Tracking**: Automatic cost calculation for Groq, Gemini, and OpenAI models
- **Anomaly Detection**: Statistical detection of latency spikes and cost anomalies (>2 standard deviations)
- **Provider Breakdown**: Compare performance and costs across different LLM providers
- **Production-Ready**: SQLite persistence, REST API, auto-refresh dashboard

## Tech Stack

- **Backend**: Flask (Python), SQLite
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Monitoring**: Threading for async anomaly detection, statistical analysis

## Installation

```bash
cd ai-monitor-dashboard
pip install -r requirements.txt
```

## Usage

### 1. Start the Dashboard

```bash
python app.py
```

Open http://localhost:5000 in your browser.

### 2. Generate Sample Data

Click "Generate Sample Data" button to populate the dashboard with simulated API calls.

### 3. Add a Project

Click **Add Project**, name it, and download `ai_monitor.py`. Drop the file into
your project folder, then swap one call:

```python
from ai_monitor import monitored_groq_call

result = monitored_groq_call(
    api_key=GROQ_KEY,
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Hello"}],
)
```

The wrapper returns the provider's raw JSON, so the surrounding code is unchanged.
It covers Groq, OpenAI, Ollama, Gemini, and any OpenAI-compatible endpoint via
`monitored_chat()`.

**Your credentials stay on your machine.** Calls go straight from your app to the
provider, and only metadata (tokens, latency, cost, status) reaches the dashboard.
The dashboard never calls an LLM and holds no provider keys.

Point the wrapper at a hosted dashboard with one environment variable:

```
AI_MONITOR_URL=https://your-dashboard.up.railway.app
```

### Sessions

Start a named session to track one experiment in isolation — A/B testing two
models, measuring a feature's cost, or benchmarking a code path. Stats then
report only calls made since that point.

## API Endpoints

- `POST /api/log` - Log an API call
- `GET /api/stats?hours=24` - Get dashboard statistics
- `GET /api/recent-calls?limit=50` - Get recent call logs
- `POST /api/projects/add` - Download the `ai_monitor.py` wrapper
- `POST /api/sessions/start` / `stop` / `GET /api/sessions/active`
- `GET /api/simulate` - Generate sample data (for demo)

## Credential Safety

Provider exceptions sometimes echo the request back, and `google.generativeai`
puts the key in the URL query string. The wrapper runs every error message
through `redact_secrets()` before it leaves the machine, covering `?key=`,
`Bearer` tokens, `Authorization`/`x-api-key` headers, and the `gsk_`/`sk-`/
`AIza`/`hf_` provider prefixes. Diagnostic text is preserved.

Monitoring failures are swallowed: a dead dashboard never breaks an LLM call.
Real provider errors still raise, so your own error handling sees them.

## Anomaly Detection

The system automatically detects:
- **Latency Spikes**: Calls >2 standard deviations above baseline
- **Cost Spikes**: Calls >2 standard deviations above baseline cost

Requires at least 10 baseline calls per provider for statistical accuracy.

## Deployment

### Local Development
Already running! Just use `python app.py`.

### Production (Railway/Render)
1. Add `Procfile`:
   ```
   web: gunicorn app:app
   ```
2. Update `requirements.txt` to add `gunicorn`
3. Deploy to Railway or Render (free tier available)

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## Cost Calculation

Prices per 1M tokens. Model ids are matched by longest prefix, so
`llama-3.1-8b-instant` resolves to the `llama-3.1-8b` row, and `gpt-4o` does not
get swallowed by `gpt-4o-mini`. These are local estimates for reporting, not
charges — the dashboard never calls a provider.

| Provider/Model | Input | Output |
|----------------|-------|--------|
| Groq Llama 3.1 70B | $0.59 | $0.79 |
| Groq Llama 3.1 8B | $0.05 | $0.08 |
| Gemini 1.5 Flash | $0.075 | $0.30 |
| Gemini 1.5 Pro | $0.35 | $1.05 |
| OpenAI GPT-4o | $2.50 | $10.00 |
| OpenAI GPT-4o Mini | $0.15 | $0.60 |

## Timezone

All timestamps display in **Egypt time (UTC+3)**. Change `TZ_OFFSET_HOURS` in
`app.py` to match your location. Stored values stay in UTC, so the time-range
filters remain correct regardless of display offset.

## Database

Runs on SQLite locally and PostgreSQL in production, chosen automatically by
whether `DATABASE_URL` is set. `db.py` holds the differences: placeholder style,
schema dialect, the local-time expressions, and connection pooling.

```bash
# local: no DATABASE_URL, uses monitor.db
python app.py

# production
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
```

PostgreSQL gets a bounded connection pool, since opening a connection per
request would exhaust the database's connection limit under load.

## Known Limitations

- **No authentication.** Anyone with the URL can read all logs. Add auth before
  exposing this publicly.
- **Redaction is a regex net, not a guarantee.** It covers the common credential
  shapes; an unusual error string could still slip something through.
- **Migration is additive.** `init_db` creates missing tables and columns; it
  does not move existing SQLite rows into PostgreSQL. Export and re-ingest if
  you want your history.

## Why This Project?

Built to demonstrate production AI monitoring skills for junior AI engineer roles:
- Shows understanding of LLM API costs and optimization
- Demonstrates production observability practices
- Proves ability to build full-stack AI tooling
- Real-time statistical analysis and anomaly detection
- RESTful API design for easy integration

## License

MIT
