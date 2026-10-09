# AI System Monitor

Real-time monitoring dashboard for LLM API calls with cost tracking, latency analysis, and anomaly detection. Multi-tenant, with Google sign-in.

## Features

- **Real-time Performance Monitoring**: Track API call latency, token usage, and success rates
- **Cost Tracking**: Automatic cost calculation for six paid providers (Groq, Gemini, OpenAI, Anthropic, DeepSeek, Mistral), plus a free rate for local Ollama
- **Anomaly Detection**: Statistical detection of latency spikes and cost anomalies (>2 standard deviations)
- **Provider Breakdown**: Compare performance and costs across different LLM providers
- **Multi-tenant**: Every Google account gets its own calls, costs, sessions, and ingest token
- **Sessions**: Scope metrics to one experiment, feature, or A/B test
- **Drop-in Wrapper**: Download `ai_monitor.py` and swap one call to start reporting
- **Credential Safety**: Provider keys never leave your code, and error text is scrubbed of secrets before it is sent
- **Production-Ready**: PostgreSQL with a connection pool, REST API, auto-refresh dashboard

## Tech Stack

- **Backend**: Flask (Python), PostgreSQL (production) / SQLite (local)
- **Auth**: Authlib with Google OAuth 2.0
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Monitoring**: Background threads for async anomaly detection, statistical analysis

## Installation

```bash
git clone https://github.com/BasselAtef/AI-Monitor-Dashboard.git
cd AI-Monitor-Dashboard
pip install -r requirements.txt
cp .env.example .env
```

Then fill in `.env`. Sign-in is required, so the app refuses to start without
these:

```bash
# 1. Create an OAuth 2.0 client (type: Web application) at
#    https://console.cloud.google.com/apis/credentials
# 2. Add this authorized redirect URI:
#    http://localhost:5000/auth/google/callback
GOOGLE_CLIENT_ID=1234567890-xxxx.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-xxxxxxxxxxxx
GOOGLE_REDIRECT_URI=http://localhost:5000/auth/google/callback

# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY=<64 random hex chars>
```

## Usage

### 1. Start the Dashboard

```bash
python app.py
```

Open http://localhost:5000 and sign in with Google.

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

`Procfile` and `requirements.txt` are already set up, so a Railway or Render
deploy needs no code changes.

1. Deploy the repo. The included `Procfile` starts gunicorn, and gunicorn is
   already pinned in `requirements.txt`.
2. Add a **PostgreSQL** plugin. The `DATABASE_URL` it injects switches the app
   from SQLite to Postgres automatically; nothing else changes.
3. Set these environment variables:

   ```
   GOOGLE_CLIENT_ID=...
   GOOGLE_CLIENT_SECRET=...
   SECRET_KEY=...                    # python -c "import secrets; print(secrets.token_hex(32))"
   GOOGLE_REDIRECT_URI=https://<your-app>.up.railway.app/auth/google/callback
   SESSION_COOKIE_SECURE=1           # serves the session cookie over HTTPS only
   TZ_OFFSET_HOURS=3
   DB_POOL_MAX=5                     # keep under your Postgres connection limit
   ```

4. Register the deployed callback URI in Google Cloud under
   Authorized redirect URIs. It must match `GOOGLE_REDIRECT_URI` exactly.

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

Resolution order:

1. **`COST_TABLE` in `app.py`.** The hand-checked table below, always
   authoritative. It wins even where LiteLLM disagrees.
2. **LiteLLM's price map** (`pricing.py`), for ids the table has not caught up
   with, such as a provider shipping a new model. Per-token rates are scaled by
   1M to match the table's shape.
3. **`n/a`.** When neither source knows the model, the call is left unpriced
   rather than recorded as `$0.00`, which would read as free.

`COST_TABLE` staying first is deliberate: it is explicit and reviewable, so the
numbers you report can be checked by reading one file. LiteLLM covers the long
tail nobody hand-maintains.

LiteLLM is a heavy import (~5s, pulls `openai`, `tiktoken`, `boto3`,
`huggingface-hub`). `pricing.py` therefore imports it lazily and warms it in a
background thread at boot, so it never sits on the telemetry path. If it is
missing, every route still works and unpriced models report `n/a`.

| Provider/Model              | Input  | Output  |
| --------------------------- | ------ | ------- |
| Gemini 3.8 Flash            | $0.75  | $3.75   |
| Gemini 2.0 Flash            | $0.10  | $0.40   |
| Gemini 2.0 Pro              | $0.50  | $1.50   |
| Gemini 1.5 Flash            | $0.075 | $0.30   |
| Gemini 1.5 Pro              | $0.35  | $1.05   |
| OpenAI GPT-4o               | $2.50  | $10.00  |
| OpenAI GPT-4o Mini          | $0.15  | $0.60   |
| OpenAI GPT-4.5              | $75.00 | $150.00 |
| OpenAI o1                   | $15.00 | $60.00  |
| OpenAI o3-mini              | $1.10  | $4.40   |
| Anthropic Claude 3.5 Sonnet | $3.00  | $15.00  |
| Anthropic Claude 3.5 Haiku  | $0.80  | $4.00   |
| DeepSeek V3                 | $0.14  | $0.28   |
| DeepSeek R1                 | $0.55  | $2.19   |
| Groq Llama 3.3 70B          | $0.59  | $0.79   |
| Groq Llama 3.1 70B          | $0.59  | $0.79   |
| Groq Llama 3.1 8B           | $0.05  | $0.08   |
| Mistral Large               | $2.00  | $6.00   |
| Qwen 3.8                    | $0.80  | $4.00   |

Resolved from LiteLLM rather than the table, shown here for reference:

| Provider/Model              | Input  | Output  |
| --------------------------- | ------ | ------- |
| Groq-hosted GPT-OSS 20B     | $0.075 | $0.30   |
| Groq-hosted GPT-OSS 120B    | $0.15  | $0.60   |
| OpenAI GPT-4.1              | $2.00  | $8.00   |

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

- **Redaction is a regex net, not a guarantee.** It covers the common credential
  shapes (`?key=`, `Bearer`, `x-api-key`, and provider key prefixes); an unusual
  error string could still slip something through.
- **Rate limiting is per-connection, not per-account.** An ingest token can post
  without limit. Add throttling before exposing a shared instance widely.
- **Migration is additive.** `init_db` creates missing tables and columns; it
  does not move existing SQLite rows into PostgreSQL. Export and re-ingest if
  you want your history.
- **Pricing is a local snapshot.** Rates in `COST_TABLE` are estimates for
  reporting, not billing, and they drift. A model missing from the table falls
  back to LiteLLM's price map, and if neither knows it, the call reports `n/a`
  rather than a misleading `$0.00`.

## Why This Project?

Built to demonstrate production AI monitoring skills for junior AI engineer roles:

- Shows understanding of LLM API costs and optimization
- Demonstrates production observability practices
- Proves ability to build full-stack AI tooling
- Real-time statistical analysis and anomaly detection
- RESTful API design for easy integration

## License

MIT
