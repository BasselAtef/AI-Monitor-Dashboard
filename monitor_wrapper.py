"""
Generates the ai_monitor.py wrapper file that users download or drop into projects.

Kept separate from app.py so the endpoint stays small and the wrapper source is
easy to read, edit, and test on its own.
"""

WRAPPER_TEMPLATE = '''"""
ai_monitor.py - LLM monitoring wrapper for {project_name}
Generated: {generated_at}

WHAT THIS DOES
    Wraps LLM API calls so each one reports provider, model, token counts,
    latency, cost, and errors to your AI Monitor dashboard.

SETUP
    1. Keep this file in the same folder as your app.
    2. Your ingest token is already filled in below. If you commit this file to
       git, override the token with an environment variable instead:

        os.environ["AI_MONITOR_TOKEN"] = "aim_..."

    3. Swap your raw API call for the monitored version:

        from ai_monitor import monitored_groq_call

        result = monitored_groq_call(
            api_key=GROQ_KEY,
            model="llama-3.1-8b-instant",
            messages=[{{"role": "user", "content": "Hello"}}],
        )

    The wrapper returns the provider's raw JSON response, so the rest of your
    code works the same way. Monitoring failures never raise.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

# Base URL of your dashboard. Override with the AI_MONITOR_URL env var.
MONITOR_URL = os.environ.get("AI_MONITOR_URL", "{monitor_url}").rstrip("/")

# Your personal ingest token. Pre-filled when you downloaded this file.
# Prefer the AI_MONITOR_TOKEN env var so the token is not committed to git.
INGEST_TOKEN = os.environ.get("AI_MONITOR_TOKEN", "{ingest_token}").strip()
LOG_ENDPOINT = f"{{MONITOR_URL}}/api/log"

# Never let a slow or dead dashboard block an LLM call.
LOG_TIMEOUT_SECONDS = 2

# Token fallback for providers that do not return usage metadata.
CHARS_PER_TOKEN = 4

# No price table lives here on purpose. Cost is computed server-side from
# COST_TABLE, which is the single source of truth. Keeping a second copy in
# every downloaded wrapper meant the two could silently disagree.

LOG_HEADERS = {{"X-Ingest-Token": INGEST_TOKEN}} if INGEST_TOKEN else {{}}


def _is_auth_error(exc: Exception) -> bool:
    return getattr(exc, "response", None) is not None and exc.response.status_code in (401, 403)


def _warn_auth_once() -> None:
    """Tell the user once if telemetry is being rejected, then stay quiet."""
    global _AUTH_WARNED
    if _AUTH_WARNED:
        return
    _AUTH_WARNED = True
    print(
        "[ai_monitor] Dashboard rejected the ingest token "
        "(401). Sign in at your dashboard, copy the token from /api/account, "
        "and set AI_MONITOR_TOKEN. LLM calls are unaffected."
    )


_AUTH_WARNED = False


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------
def _estimate_tokens(text: str) -> int:
    """Rough token estimate for providers that omit usage metadata."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def redact_secrets(text: str) -> str:
    """Strip credentials out of an error string before it leaves the machine.

    Provider exceptions sometimes echo the request back. Two realistic shapes:
      - google.generativeai puts the key in the URL: .../models/x?key=SECRET
      - a debug repr can include headers: {{'Authorization': 'Bearer SECRET'}}
    """
    if not text:
        return ""

    patterns = [
        # key=SECRET in a query string
        r"([?&]key=)[^&\\s\\"']+",
        r"([?&]api_key=)[^&\\s\\"']+",
        # "Bearer SECRET" and bare token after an auth header name
        r"(Bearer\\s+)[A-Za-z0-9._\\-]{{8,}}",
        r"((?:api[_-]?key|apikey|secret|token|password|passwd)\\s*[=:]\\s*)[^\\s,;)\\}}\\"']+",
        # provider key prefixes
        r"\\b(?:gsk|sk|xai|AIza|hf|r8_)[A-Za-z0-9._\\-]{{8,}}",
    ]

    def _replace(match: "re.Match") -> str:
        # Patterns that capture a prefix keep it (so "?key=" stays readable);
        # the rest are replaced wholesale.
        return (match.group(1) if match.lastindex else "") + "REDACTED"

    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, _replace, cleaned, flags=re.IGNORECASE)
    return cleaned


def log_call(
    provider: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: int = 0,
    status: str = "success",
    error: Optional[str] = None,
) -> None:
    """Send one call to the dashboard. Never raises.

    The full model id is sent on purpose: the dashboard matches it against its
    pricing table by prefix, including namespaced ids such as
    'qwen/qwen3.8-27b', so the cost resolves correctly.
    """
    payload = {{
        "provider": provider,
        "model": model or "unknown",
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "latency_ms": int(latency_ms or 0),
        "status": status,
        "error_message": redact_secrets(error)[:500] or None,
    }}
    if not INGEST_TOKEN:
        # Nothing to authenticate with; fail quietly rather than spam 401s.
        return
    try:
        response = requests.post(
            LOG_ENDPOINT, json=payload, headers=LOG_HEADERS, timeout=LOG_TIMEOUT_SECONDS
        )
        if response.status_code in (401, 403):
            _warn_auth_once()
    except Exception:
        # Monitoring is best-effort. A dead dashboard must not break the app.
        pass


def _extract_usage(response_json: Dict[str, Any]) -> Any:
    """Pull a usage dict out of an OpenAI-compatible response."""
    usage = response_json.get("usage")
    if isinstance(usage, dict):
        return usage
    metadata = response_json.get("usageMetadata")
    if isinstance(metadata, dict):
        return {{
            "prompt_tokens": metadata.get("promptTokenCount", 0),
            "completion_tokens": metadata.get("candidatesTokenCount", 0),
        }}
    return {{}}


# --------------------------------------------------------------------------
# OpenAI-compatible provider (Groq, OpenAI, Mistral, Together, ...)
# --------------------------------------------------------------------------
def monitored_chat(
    provider: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    base_url: str,
    headers: Optional[Dict[str, str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Call any OpenAI-compatible /chat/completions endpoint with monitoring."""
    url = f"{{base_url.rstrip('/')}}/chat/completions"
    request_headers = {{"Content-Type": "application/json"}}
    request_headers.update(headers or {{}})

    body: Dict[str, Any] = {{"model": model, "messages": messages}}
    body.update(kwargs)

    started = time.time()
    try:
        response = requests.post(
            url, headers=request_headers, json=body, timeout=kwargs.pop("timeout", 120)
        )
        latency_ms = int((time.time() - started) * 1000)
        response.raise_for_status()
        result = response.json()

        usage = _extract_usage(result)
        log_call(
            provider=provider,
            model=model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            status="success",
        )
        return result

    except Exception as exc:
        latency_ms = int((time.time() - started) * 1000)
        log_call(
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            status="error",
            error=str(exc),
        )
        raise


def monitored_groq_call(api_key: str, model: str = "llama-3.1-8b-instant",
                        messages: Optional[List[Dict[str, str]]] = None, **kwargs):
    """Groq cloud LLM call, monitored. Returns raw Groq JSON."""
    return monitored_chat(
        provider="groq",
        api_key=api_key,
        model=model,
        messages=messages or [],
        base_url="https://api.groq.com/openai/v1",
        headers={{"Authorization": f"Bearer {{api_key}}"}},
        **kwargs,
    )


def monitored_openai_call(api_key: str, model: str = "gpt-4o-mini",
                          messages: Optional[List[Dict[str, str]]] = None, **kwargs):
    """OpenAI chat completion, monitored. Returns raw OpenAI JSON."""
    return monitored_chat(
        provider="openai",
        api_key=api_key,
        model=model,
        messages=messages or [],
        base_url="https://api.openai.com/v1",
        headers={{"Authorization": f"Bearer {{api_key}}"}},
        **kwargs,
    )


# --------------------------------------------------------------------------
# Ollama (local)
# --------------------------------------------------------------------------
def monitored_ollama_call(model: str = "llama3.2", prompt: str = "",
                          host: str = "http://localhost:11434", **kwargs) -> Dict[str, Any]:
    """Local Ollama generation call, monitored. Tokens are estimated."""
    body: Dict[str, Any] = {{"model": model, "prompt": prompt, "stream": False}}
    body.update(kwargs)

    started = time.time()
    try:
        response = requests.post(
            f"{{host.rstrip('/')}}/api/generate", json=body, timeout=kwargs.pop("timeout", 300)
        )
        latency_ms = int((time.time() - started) * 1000)
        response.raise_for_status()
        result = response.json()

        text = result.get("response", "")
        log_call(
            provider="ollama",
            model=model,
            prompt_tokens=_estimate_tokens(prompt),
            completion_tokens=_estimate_tokens(text),
            latency_ms=latency_ms,
            status="success",
        )
        return result

    except Exception as exc:
        latency_ms = int((time.time() - started) * 1000)
        log_call(provider="ollama", model=model, latency_ms=latency_ms,
                 status="error", error=str(exc))
        raise


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------
def monitored_gemini_call(api_key: str, model: str = "gemini-1.5-flash",
                          prompt: str = "") -> Dict[str, Any]:
    """Gemini generate_content call, monitored. Tokens are estimated.

    Requires: pip install google-generativeai
    """
    started = time.time()
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        response = genai.GenerativeModel(model).generate_content(prompt)
        latency_ms = int((time.time() - started) * 1000)

        text = getattr(response, "text", "") or ""
        usage = getattr(response, "usage_metadata", None)

        log_call(
            provider="gemini",
            model=model,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or _estimate_tokens(prompt),
            completion_tokens=getattr(usage, "candidates_token_count", 0) or _estimate_tokens(text),
            latency_ms=latency_ms,
            status="success",
        )
        return {{"text": text, "raw": response}}

    except Exception as exc:
        latency_ms = int((time.time() - started) * 1000)
        log_call(provider="gemini", model=model, latency_ms=latency_ms,
                 status="error", error=str(exc))
        raise
'''


def build_wrapper(
    project_name: str = "my_project",
    monitor_url: str = "http://localhost:5000",
    ingest_token: str = "",
) -> str:
    """Return the wrapper source, filled in for one project.

    ingest_token is pre-filled so a first-time user can drop the file in and
    have it work immediately. It stays overridable via the AI_MONITOR_TOKEN
    environment variable, which is the better habit once the file is committed.
    """
    from datetime import datetime

    return WRAPPER_TEMPLATE.format(
        project_name=project_name or "my_project",
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        monitor_url=monitor_url.rstrip("/"),
        ingest_token=ingest_token or "",
    )
