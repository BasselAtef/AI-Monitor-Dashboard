"""Fallback pricing lookup via LiteLLM's bundled price map.

`COST_TABLE` in app.py stays the primary source: it is hand-checked, has no
dependency, and is what the dashboard reports by default. This module is only
consulted when `COST_TABLE` has no entry, which happens whenever a provider
ships a model the table has not caught up with.

Two behaviours of LiteLLM's API drive the implementation:

1. `cost_per_token(model=m)` called without token counts returns `(0.0, 0.0)`
   for every model, priced or not. Passing counts of 1 returns the actual
   per-token rates. Reporting the no-argument form would mark every paid model
   free, so the probe always supplies counts.

2. Unknown models raise instead of returning None, and coverage is partial:
   `get_model_info` resolves the OpenAI ids and the Groq-hosted `openai/*` ids,
   but raises `ModelNotMappedError` for several ids that `COST_TABLE` does price
   (gemini-2.0-flash, claude-3-5-sonnet). Those never reach this module, since
   the table matched first.

The optional dependency is imported lazily inside the lookup. If LiteLLM is not
installed the dashboard still serves every route, and unpriced models keep
reporting `n/a` exactly as before.
"""

import threading

# LiteLLM prints a banner and a deprecation notice on import. Neither is useful
# in server logs, and the banner interleaves with gunicorn's own output.
_CACHE = {}
_CACHE_LIMIT = 500
_LOCK = threading.Lock()

# Signals that the optional LiteLLM import has finished, one way or the other.
# Presence in sys.modules is not a readiness signal, for the reason given in
# warm_up().
_READY = threading.Event()


def _candidates(provider, model):
    """Model id spellings to try, most specific first.

    The wrapper reports the provider separately from the model, but providers
    often namespace the model themselves ('openai/gpt-oss-20b' served by Groq),
    so the id may need to be joined, used alone, or stripped of its namespace.
    """
    provider = (provider or "").strip().lower()
    model = (model or "").strip().lower()
    if not model or model == "unknown":
        return []

    # Already carries a provider namespace; do not double it up.
    if model.startswith(provider + "/") and provider:
        return [model]

    options = []
    if provider:
        options.append(f"{provider}/{model}")
        options.append(provider)
    options.append(model)
    if "/" in model:
        tail = model.rsplit("/", 1)[1]
        if tail and tail != model:
            if provider:
                options.append(f"{provider}/{tail}")
            options.append(tail)

    seen = set()
    ordered = []
    for option in options:
        if option and option not in seen:
            seen.add(option)
            ordered.append(option)
    return ordered


def _probe(candidate):
    """Return {'input': usd per 1M tokens, 'output': ...} or None.

    LiteLLM returns the cost for the supplied token counts, so one token of
    each direction is exactly the per-token rate. Multiplying by 1e6 converts
    to the per-1M shape the rest of the app uses.
    """
    try:
        import litellm
    except Exception:
        # Not installed, or the install is broken. Leave the model unpriced
        # rather than taking the dashboard down over a cosmetic number.
        return None

    try:
        litellm.suppress_debug_info = True
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=candidate, prompt_tokens=1, completion_tokens=1
        )
    except Exception:
        # ModelNotMappedError, BadRequestError, or anything a future release
        # raises. An unknown price is a normal outcome here, not an error.
        return None

    if prompt_cost is None and completion_cost is None:
        return None
    try:
        return {
            "input": float(prompt_cost or 0.0) * 1_000_000,
            "output": float(completion_cost or 0.0) * 1_000_000,
        }
    except (TypeError, ValueError):
        return None


def lookup_rates(provider, model):
    """Rates for a model missing from COST_TABLE, or None if still unknown.

    Results are cached per provider/model, including misses. A miss is the
    expensive case: without caching, every logged call for an unrecognised
    model re-walks the candidate list, and /api/log runs once per LLM call.
    """
    key = ((provider or "").strip().lower(), (model or "").strip().lower())
    if not key[1]:
        return None

    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]

    # This runs inside /api/log. Pricing is cosmetic here, so no failure in the
    # fallback is allowed to propagate: an exception would 500 the ingest route
    # and drop the caller's telemetry entirely. An unknown price is the correct
    # outcome for anything that goes wrong here.
    rates = None
    try:
        for candidate in _candidates(*key):
            rates = _probe(candidate)
            if rates is not None:
                break
    except Exception:
        rates = None

    with _LOCK:
        # Bounded so a flood of unique model ids cannot grow memory forever.
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.clear()
        _CACHE[key] = rates
    return rates


def known(provider, model):
    """True when LiteLLM can price this model, without doing the lookup twice."""
    return lookup_rates(provider, model) is not None


def ready():
    """True once the warm-up import has settled, successfully or not."""
    return _READY.is_set()


def clear_cache():
    """Drop cached lookups, so a refreshed price map takes effect."""
    with _LOCK:
        _CACHE.clear()


def warm_up():
    """Import LiteLLM off the request path.

    The import costs about five seconds. Paying it inside /api/log would stall
    the caller's telemetry POST on the first unpriced model, which is worse than
    the `n/a` the fallback exists to remove. Importing here in a background
    thread means the first real lookup hits a warm module.

    Failures are ignored: the lookup is optional, and a warm-up that does not
    complete just leaves the lazy import to do its job later.
    """
    try:
        import litellm

        litellm.suppress_debug_info = True
        ok = True
    except Exception:
        ok = False
    finally:
        # Set last, and always. Checking `import litellm` in sys.modules is not
        # a readiness test: Python registers the module there before the import
        # body finishes, so a caller can observe a half-initialised module.
        _READY.set()
    return ok