#!/usr/bin/env python3
"""free-model-aggregator-02 -- one stdlib-only live free-model aggregator.

What this program does, from first principles:

* It reads a trial brief from the file named by MAKEFUN_PROMPT_FILE.
  The brief names the providers (wire, auth, base URL, key) and a
  sealed quality table (rating, context length, tool / image / JSON
  support) for the free models those providers expose.
* It fetches every provider's model list and registers one merged
  catalog with the trial environment (POST $MAKEFUN_ENV_URL/register).
* It serves POST /v1/chat/completions (OpenAI wire, JSON and SSE)
  and POST /v1/messages (Anthropic wire, JSON and SSE), plus
  GET /v1/models in the wire the auth header names
  (``Authorization: Bearer`` -> OpenAI list, ``x-api-key`` ->
  Anthropic list).
* It routes by score: a request's needs (tools, images, JSON mode,
  estimated context) are counted against every (model, provider)
  route; routes that meet everything rank by sealed quality, then by
  provider health (recent failures push a provider down), then by
  context length. A client-named model that is eligible goes first.
  It translates between OpenAI, Anthropic and Gemini provider wires
  as needed.
* It fails over with a per-provider circuit breaker: 402 or an auth
  failure opens it for the trial, 429 honours Retry-After, 5xx or a
  timeout opens it briefly, 400 / 404 / a parse failure kills just
  that model. After a failed sweep it waits for the shortest live
  breaker (capped) and sweeps again. Every model call goes to a
  provider under MAKEFUN_ENV_URL, never anywhere else.
* It polls GET $MAKEFUN_ENV_URL/work until the work is done, then
  exits 0.

Run it::

    python3 main.py

Print the catalog document the server registers, byte for byte::

    python3 main.py --list

The server is deterministic: no wall clock in any routing decision
and no unseeded randomness. Response ids and streaming chunk
boundaries derive from MAKEFUN_TRIAL_SEED plus a per-request counter.
Python 3.13, standard library only, no installs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PORT = 8080
DEFAULT_HOST = "0.0.0.0"
PROVIDER_TIMEOUT_SEC = 30.0
REGISTER_ATTEMPTS = 5
REGISTER_RETRY_DELAY_SEC = 1.0
WORK_POLL_INTERVAL_SEC = 1.0
MAX_SWEEPS = 3
# A breaker opened "for the trial" stays open far longer than any trial.
BREAKER_TRIAL_OPEN_SEC = 3600.0
# A breaker opened briefly after a 5xx / timeout / 429.
BREAKER_SHORT_OPEN_SEC = 30.0
# Longest we ever wait for a breaker between sweeps.
MAX_BREAKER_WAIT_SEC = 5.0
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_DEFAULT_MAX_TOKENS = 1024
GEMINI_API_VERSION = "v1beta"
# Rough token accounting used only for eligibility estimates.
CHARS_PER_TOKEN = 4
IMAGE_TOKENS_ESTIMATE = 1000
TOOL_OVERHEAD_TOKENS = 200
DEFAULT_CONTEXT_LENGTH = 8192
DEFAULT_RATING = 0.5


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------

def stable_digest(seed, *parts):
    """SHA-256 hex digest of the seed joined with the given parts."""
    material = "\x1f".join([str(seed)] + [str(p) for p in parts])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def stable_int(seed, *parts):
    """A deterministic non-negative integer derived from seed + parts."""
    return int(stable_digest(seed, *parts)[:12], 16)


def deterministic_id(prefix, seed, counter, kind):
    """A response id derived only from seed, counter and kind."""
    return prefix + stable_digest(seed, counter, kind)[:24]


def deterministic_created(seed):
    """A fixed ``created`` timestamp derived only from the seed."""
    return 1700000000 + (stable_int(seed, "created") % 1000000)


def deterministic_created_iso(seed):
    """A fixed ISO-8601 timestamp derived only from the seed."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(deterministic_created(seed)))


def deterministic_chunk_size(seed, counter):
    """Streaming chunk size in characters, derived from seed + counter."""
    return 16 + (stable_int(seed, counter, "chunk") % 48)


def clamp_number(value, low, high, default):
    """Coerce ``value`` to a number inside [low, high] or return default."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    if number < low:
        return low
    if number > high:
        return high
    return number


def as_bool(value, default=False):
    """Best-effort bool coercion; anything unrecognised yields default."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "y", "on"):
            return True
        if lowered in ("0", "false", "no", "n", "off", ""):
            return False
    return default


def as_int(value, default=0):
    """Best-effort int coercion; anything unrecognised yields default."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "").replace("_", "")
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                return default
    return default


def as_float(value, default=0.0):
    """Best-effort float coercion; anything unrecognised yields default."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return default
        return number if math.isfinite(number) else default
    return default


def pick_first(mapping, keys, default=None):
    """Return the first present, non-None value among ``keys`` in mapping."""
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def safe_json_loads(text):
    """Parse JSON text, returning None instead of raising on bad input."""
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def estimate_text_tokens(text):
    """Rough token estimate for plain text: one token per four chars."""
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def message_text_parts(message):
    """Yield the (kind, text-or-url) content pieces of one message."""
    content = message.get("content") if isinstance(message, dict) else None
    if content is None:
        return
    if isinstance(content, str):
        yield ("text", content)
        return
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "text" and isinstance(part.get("text"), str):
                yield ("text", part["text"])
            elif kind == "input_text" and isinstance(part.get("text"), str):
                yield ("text", part["text"])
            elif kind == "image_url":
                url = part.get("image_url")
                if isinstance(url, dict):
                    yield ("image", url.get("url") or "")
                elif isinstance(url, str):
                    yield ("image", url)
            elif kind == "image":
                source = part.get("source") or {}
                if isinstance(source, dict):
                    yield ("image", source.get("data") or source.get("url") or "")
            elif kind == "tool_result":
                inner = part.get("content")
                if isinstance(inner, str):
                    yield ("text", inner)
                elif isinstance(inner, list):
                    for sub in inner:
                        if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                            yield ("text", sub["text"])
            elif kind == "tool_use":
                blob = part.get("input")
                if isinstance(blob, str):
                    yield ("text", blob)
                elif blob is not None:
                    yield ("text", json.dumps(blob, sort_keys=True))
            elif isinstance(part.get("text"), str):
                yield ("text", part["text"])


def estimate_messages_tokens(messages, extra_tokens=0):
    """Estimate the prompt tokens of a message list, plus ``extra_tokens``."""
    total = 0
    if not isinstance(messages, list):
        return extra_tokens
    for message in messages:
        if not isinstance(message, dict):
            continue
        total += 4  # per-message framing overhead
        role = message.get("role")
        if isinstance(role, str):
            total += estimate_text_tokens(role)
        for kind, value in message_text_parts(message):
            if kind == "text":
                total += estimate_text_tokens(value)
            else:
                total += IMAGE_TOKENS_ESTIMATE
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if isinstance(call, dict):
                    total += TOOL_OVERHEAD_TOKENS // 4
                    fn = call.get("function") or {}
                    if isinstance(fn, dict):
                        total += estimate_text_tokens(
                            str(fn.get("name") or "")
                            + str(fn.get("arguments") or "")
                        )
    return total + extra_tokens


def request_needs(request):
    """Summarise what a normalised request needs from a route.

    Returns a dict with has_tools, has_images, wants_json and the
    estimated prompt tokens (including the requested max_tokens, so a
    route must fit prompt + completion inside its context window).
    """
    messages = request.get("messages") or []
    tools = request.get("tools") or []
    has_tools = bool(tools)
    has_images = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        for kind, _value in message_text_parts(message):
            if kind == "image":
                has_images = True
    extra = 0
    if has_tools:
        extra += TOOL_OVERHEAD_TOKENS
        try:
            extra += estimate_text_tokens(json.dumps(tools, sort_keys=True))
        except (TypeError, ValueError):
            extra += 0
    max_tokens = request.get("max_tokens")
    completion_budget = as_int(max_tokens, 0) if max_tokens is not None else 0
    estimated = estimate_messages_tokens(messages, extra_tokens=extra)
    return {
        "has_tools": has_tools,
        "has_images": has_images,
        "wants_json": bool(request.get("wants_json")),
        "estimated_tokens": estimated + max(0, completion_budget),
        "prompt_tokens": estimated,
    }


def log_line(message):
    """Write one diagnostics line to stderr, flushed at once."""
    sys.stderr.write(str(message) + "\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Trial brief: providers and the sealed quality table
# ---------------------------------------------------------------------------

WIRE_ALIASES = {
    "openai": "openai",
    "openai-compatible": "openai",
    "openai_compatible": "openai",
    "oai": "openai",
    "openrouter": "openai",
    "anthropic": "anthropic",
    "anthropic-compatible": "anthropic",
    "anthropic_compatible": "anthropic",
    "claude": "anthropic",
    "gemini": "gemini",
    "google": "gemini",
    "google-gemini": "gemini",
    "google_genai": "gemini",
    "genai": "gemini",
    "vertex": "gemini",
}


def normalize_wire(value, default="openai"):
    """Map a provider's wire label to openai / anthropic / gemini."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key in WIRE_ALIASES:
            return WIRE_ALIASES[key]
    return default


def resolve_provider_url(raw_base, env_url):
    """Resolve a provider base URL against the trial environment URL.

    Absolute URLs are used as given; relative ones are joined onto
    the environment URL, so every model call stays under
    MAKEFUN_ENV_URL.
    """
    if isinstance(raw_base, str) and raw_base.strip():
        base = raw_base.strip()
        if base.startswith("http://") or base.startswith("https://"):
            return base.rstrip("/")
        if env_url:
            return urllib.parse.urljoin(env_url.rstrip("/") + "/", base.lstrip("/"))
        return base.rstrip("/")
    return (env_url or "").rstrip("/")


def provider_key_from_env(wire):
    """Stand-in / real key for a wire, from the environment if set."""
    if wire == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY") or ""
    if wire == "gemini":
        return (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        )
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or ""
    )


class Provider:
    """One provider named by the trial brief."""

    def __init__(self, name, wire, base_url, api_key, raw_models=None):
        self.name = name
        self.wire = wire
        self.base_url = base_url
        self.api_key = api_key or ""
        self.raw_models = list(raw_models or [])
        self.fetched_models = []
        self.list_failed = False

    def __repr__(self):  # pragma: no cover - diagnostics only
        return "Provider(%r, wire=%r, base=%r)" % (
            self.name,
            self.wire,
            self.base_url,
        )

    def models_url(self, path):
        """Join ``path`` onto this provider's base URL."""
        return urllib.parse.urljoin(self.base_url.rstrip("/") + "/", path.lstrip("/"))

    def list_paths(self):
        """Candidate model-list paths for this wire, best first."""
        if self.wire == "gemini":
            return ["v1beta/models", "v1/models", "models"]
        return ["v1/models", "models"]

    def auth_headers(self):
        """Auth headers for this provider's wire."""
        headers = {}
        key = self.api_key or provider_key_from_env(self.wire)
        if not key:
            return headers
        if self.wire == "anthropic":
            headers["x-api-key"] = key
            headers["anthropic-version"] = ANTHROPIC_VERSION
        elif self.wire == "gemini":
            headers["x-goog-api-key"] = key
        else:
            headers["Authorization"] = "Bearer " + key
        return headers

    def call_headers(self, extra=None):
        """Headers for a model call on this wire (auth + content type)."""
        headers = {"Content-Type": "application/json"}
        headers.update(self.auth_headers())
        if self.wire == "anthropic":
            headers.setdefault("anthropic-version", ANTHROPIC_VERSION)
        if extra:
            headers.update(extra)
        return headers


def normalize_provider_entry(entry, index, env_url):
    """Normalise one raw provider entry from the brief into a Provider."""
    if isinstance(entry, str):
        entry = {"name": entry}
    if not isinstance(entry, dict):
        entry = {}
    name = pick_first(entry, ["name", "id", "provider", "provider_id"])
    if not isinstance(name, str) or not name.strip():
        name = "provider-%d" % (index + 1)
    wire = normalize_wire(
        pick_first(entry, ["wire", "type", "protocol", "kind", "api", "format"])
    )
    raw_base = pick_first(
        entry,
        ["base_url", "baseUrl", "url", "endpoint", "base", "api_base", "host"],
    )
    base_url = resolve_provider_url(raw_base, env_url)
    api_key = pick_first(
        entry,
        ["api_key", "apiKey", "key", "token", "auth_key", "secret"],
        "",
    )
    if not isinstance(api_key, str):
        api_key = str(api_key) if api_key is not None else ""
    raw_models = pick_first(entry, ["models", "model_ids", "model_list"], [])
    if isinstance(raw_models, dict):
        raw_models = list(raw_models.keys())
    if not isinstance(raw_models, list):
        raw_models = []
    models = []
    for item in raw_models:
        if isinstance(item, str) and item.strip():
            models.append(item.strip())
        elif isinstance(item, dict):
            model_id = pick_first(item, ["id", "name", "model"])
            if isinstance(model_id, str) and model_id.strip():
                models.append(model_id.strip())
    return Provider(name.strip(), wire, base_url, api_key, models)


def normalize_quality_entry(raw):
    """Normalise one quality-table entry into canonical capability flags."""
    if not isinstance(raw, dict):
        raw = {}
    rating = as_float(
        pick_first(raw, ["rating", "quality", "score", "rank", "value"]),
        DEFAULT_RATING,
    )
    rating = clamp_number(rating, 0.0, 100.0, DEFAULT_RATING)
    context = as_int(
        pick_first(
            raw,
            ["context_length", "context", "max_context", "context_window",
             "max_tokens", "max_context_length"],
        ),
        DEFAULT_CONTEXT_LENGTH,
    )
    if context <= 0:
        context = DEFAULT_CONTEXT_LENGTH
    tools = as_bool(
        pick_first(
            raw,
            ["supports_tools", "tools", "tool_support", "tool_calling",
             "function_calling", "supports_function_calling"],
            False,
        )
    )
    images = as_bool(
        pick_first(
            raw,
            ["supports_images", "images", "image_support", "vision",
             "supports_vision", "image", "multimodal"],
            False,
        )
    )
    json_ok = as_bool(
        pick_first(
            raw,
            ["supports_json", "json", "json_support", "json_mode",
             "supports_json_mode", "structured_output"],
            True,
        )
    )
    return {
        "rating": rating,
        "context_length": context,
        "supports_tools": tools,
        "supports_images": images,
        "supports_json": json_ok,
    }


def extract_quality_table(raw_brief):
    """Find the sealed quality table inside a brief of unknown shape.

    The table may live under ``quality``, ``sealed_quality``,
    ``quality_table`` or ``sealed.quality``; its values may be nested
    one level deeper under ``models``. Anything unrecognised yields
    an empty table and the defaults take over.
    """
    table = {}
    if not isinstance(raw_brief, dict):
        return table
    candidates = []
    for key in ("quality", "sealed_quality", "quality_table", "model_quality"):
        value = raw_brief.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    sealed = raw_brief.get("sealed")
    if isinstance(sealed, dict):
        for key in ("quality", "quality_table", "models"):
            value = sealed.get(key)
            if isinstance(value, dict):
                candidates.append(value)
    models_block = raw_brief.get("models")
    if isinstance(models_block, dict):
        candidates.append(models_block)
    for candidate in candidates:
        inner = candidate
        if isinstance(candidate.get("models"), dict):
            inner = candidate["models"]
        for model_id, entry in inner.items():
            if not isinstance(model_id, str):
                continue
            table[model_id] = normalize_quality_entry(entry)
    return table


def extract_providers(raw_brief, env_url):
    """Find the provider list inside a brief of unknown shape."""
    entries = []
    if not isinstance(raw_brief, dict):
        return entries
    raw = raw_brief.get("providers")
    if isinstance(raw, dict):
        # {"openai": {...}, "anthropic": {...}} style.
        for name, value in raw.items():
            if isinstance(value, dict):
                merged = dict(value)
                merged.setdefault("name", name)
                entries.append(merged)
            elif isinstance(value, str):
                entries.append({"name": name, "base_url": value})
    elif isinstance(raw, list):
        entries = [item for item in raw if isinstance(item, (dict, str))]
    if not entries:
        # A single provider written flat at the top level.
        if pick_first(raw_brief, ["base_url", "baseUrl", "url", "endpoint"]):
            entries = [raw_brief]
    providers = []
    for index, entry in enumerate(entries):
        providers.append(normalize_provider_entry(entry, index, env_url))
    return providers


def load_brief(path, env_url):
    """Load and normalise the trial brief file.

    Returns (providers, quality_table, seed). A missing or unreadable
    brief yields empty providers / quality and the seed from the
    environment, so the server can still start and report the problem.
    """
    seed = os.environ.get("MAKEFUN_TRIAL_SEED") or "default-seed"
    providers = []
    quality = {}
    raw = {}
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, ValueError) as exc:
            log_line("brief: could not read %s: %s" % (path, exc))
            raw = {}
    elif path:
        log_line("brief: file not found: %s" % path)
    if isinstance(raw, dict):
        brief_seed = pick_first(raw, ["trial_seed", "seed", "trialSeed"])
        if isinstance(brief_seed, str) and brief_seed.strip():
            seed = brief_seed.strip()
        elif brief_seed is not None:
            seed = str(brief_seed)
        providers = extract_providers(raw, env_url)
        quality = extract_quality_table(raw)
    return providers, quality, seed


# ---------------------------------------------------------------------------
# HTTP client and the merged catalog
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """A provider call failed; ``kind`` classifies the failure."""

    def __init__(self, kind, status=None, message="", retry_after=None):
        super().__init__(message or kind)
        self.kind = kind
        self.status = status
        self.message = message or kind
        self.retry_after = retry_after


def parse_retry_after(headers):
    """Parse a Retry-After header value into seconds, or None."""
    if not headers:
        return None
    value = None
    try:
        value = headers.get("Retry-After")
    except AttributeError:
        value = None
    if value is None:
        return None
    try:
        seconds = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if seconds < 0 or math.isnan(seconds) or math.isinf(seconds):
        return None
    return seconds


def classify_status(status, body_text=""):
    """Classify an HTTP status into a ProviderError kind."""
    if status in (401, 403):
        return "auth"
    if status == 402:
        return "payment"
    if status == 404:
        return "not_found"
    if status == 429:
        return "rate_limit"
    if status == 400 or status == 422:
        return "bad_request"
    if status is not None and 500 <= status <= 599:
        return "server"
    if status is not None and 400 <= status <= 499:
        return "bad_request"
    return "server"


def http_request(method, url, headers=None, payload=None, timeout=PROVIDER_TIMEOUT_SEC):
    """One HTTP call via urllib.

    Returns (status, parsed_json_or_None, raw_text, headers_dict).
    Network errors raise ProviderError(kind="timeout") or ("network").
    HTTP error statuses are returned, not raised, so the caller can
    classify them.
    """
    data = None
    request_headers = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return response.status, safe_json_loads(raw), raw, dict(response.headers)
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        headers_out = dict(exc.headers) if exc.headers else {}
        return exc.code, safe_json_loads(raw), raw, headers_out
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        reason = str(getattr(exc, "reason", exc) or exc)
        if "timed out" in reason.lower() or isinstance(exc, (socket.timeout, TimeoutError)):
            raise ProviderError("timeout", None, "timed out calling %s" % url)
        raise ProviderError("network", None, "network error calling %s: %s" % (url, reason))
    except OSError as exc:
        raise ProviderError("network", None, "os error calling %s: %s" % (url, exc))


def extract_model_ids(payload):
    """Pull model ids out of a provider list response of unknown shape."""
    ids = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("data")
        if items is None:
            items = payload.get("models")
        if items is None:
            items = payload.get("items")
        if items is None:
            items = []
    else:
        items = []
    if not isinstance(items, list):
        return ids
    for item in items:
        if isinstance(item, str) and item.strip():
            ids.append(item.strip())
        elif isinstance(item, dict):
            model_id = pick_first(item, ["id", "name", "model"])
            if isinstance(model_id, str) and model_id.strip():
                model_id = model_id.strip()
                if model_id.startswith("models/"):
                    model_id = model_id[len("models/"):]
                ids.append(model_id)
    seen = set()
    unique = []
    for model_id in ids:
        if model_id not in seen:
            seen.add(model_id)
            unique.append(model_id)
    return unique


def fetch_provider_models(provider):
    """Fetch one provider's model list; fall back to the brief's list."""
    last_error = None
    for path in provider.list_paths():
        url = provider.models_url(path)
        try:
            status, payload, _raw, _headers = http_request(
                "GET", url, headers=provider.auth_headers()
            )
        except ProviderError as exc:
            last_error = exc
            continue
        if status is not None and 200 <= status < 300 and payload is not None:
            ids = extract_model_ids(payload)
            if ids:
                provider.fetched_models = ids
                provider.list_failed = False
                return ids
            last_error = ProviderError("parse", status, "empty model list at %s" % url)
        else:
            last_error = ProviderError(
                classify_status(status), status, "list failed at %s" % url
            )
    provider.list_failed = True
    if provider.raw_models:
        provider.fetched_models = list(provider.raw_models)
        return list(provider.raw_models)
    if last_error is not None:
        log_line("models: %s list failed: %s" % (provider.name, last_error))
    return []


def strip_provider_prefix(model_id, provider):
    """Remove a provider-name prefix (``openai/``) from a model id."""
    if not isinstance(model_id, str):
        return ""
    prefix = provider.name.strip().lower() + "/"
    if model_id.lower().startswith(prefix):
        return model_id[len(prefix):]
    return model_id


def build_catalog(providers, quality):
    """Merge every provider's models into one deterministic catalog.

    One entry per model id, listing every (provider, provider-model)
    route that serves it. Quality comes from the sealed table keyed
    by merged id, then by provider model id, then by defaults.
    Entries sort by model id so the document is byte-stable.
    """
    merged = {}
    for provider in providers:
        model_ids = provider.fetched_models or provider.raw_models
        for provider_model in model_ids:
            merged_id = strip_provider_prefix(provider_model, provider)
            entry = merged.get(merged_id)
            if entry is None:
                entry = {
                    "id": merged_id,
                    "providers": [],
                    "routes": [],
                    "rating": DEFAULT_RATING,
                    "context_length": DEFAULT_CONTEXT_LENGTH,
                    "supports_tools": False,
                    "supports_images": False,
                    "supports_json": True,
                }
                merged[merged_id] = entry
            if provider.name not in entry["providers"]:
                entry["providers"].append(provider.name)
            route = {
                "provider": provider.name,
                "model": provider_model,
                "wire": provider.wire,
            }
            if route not in entry["routes"]:
                entry["routes"].append(route)
    for model_id, entry in merged.items():
        raw_quality = quality.get(model_id)
        if raw_quality is None:
            for route in entry["routes"]:
                raw_quality = quality.get(route["model"])
                if raw_quality is not None:
                    break
        if raw_quality is not None:
            entry["rating"] = raw_quality["rating"]
            entry["context_length"] = raw_quality["context_length"]
            entry["supports_tools"] = raw_quality["supports_tools"]
            entry["supports_images"] = raw_quality["supports_images"]
            entry["supports_json"] = raw_quality["supports_json"]
        entry["providers"] = sorted(entry["providers"])
        entry["routes"] = sorted(
            entry["routes"], key=lambda r: (r["provider"], r["model"])
        )
    catalog = [merged[key] for key in sorted(merged.keys())]
    return catalog


def catalog_document(catalog):
    """The exact document registered with the trial environment."""
    return {"models": catalog}


def catalog_json_text(catalog):
    """The catalog document as stable, byte-for-byte JSON text."""
    return json.dumps(catalog_document(catalog), sort_keys=True, indent=2) + "\n"


# ---------------------------------------------------------------------------
# Trial environment: register the catalog, poll for work
# ---------------------------------------------------------------------------

def register_catalog(env_url, catalog, seed):
    """Register the merged catalog; retry briefly, then report failure.

    Returns True when the environment accepted the catalog. A failure
    is logged but does not stop the server: the trial may still drive
    the server directly, and a hard exit here would lose everything.
    """
    if not env_url:
        log_line("register: no MAKEFUN_ENV_URL set, skipping registration")
        return False
    url = urllib.parse.urljoin(env_url.rstrip("/") + "/", "register")
    payload = catalog_document(catalog)
    payload["trial_seed"] = seed
    last_status = None
    for attempt in range(1, REGISTER_ATTEMPTS + 1):
        try:
            status, _body, _raw, _headers = http_request("POST", url, payload=payload)
        except ProviderError as exc:
            log_line("register: attempt %d failed: %s" % (attempt, exc))
            last_status = None
        else:
            last_status = status
            if status is not None and 200 <= status < 300:
                log_line(
                    "register: ok (%d models) at %s" % (len(catalog), url)
                )
                return True
            log_line("register: attempt %d got HTTP %s" % (attempt, status))
        if attempt < REGISTER_ATTEMPTS:
            time.sleep(REGISTER_RETRY_DELAY_SEC)
    log_line(
        "register: giving up after %d attempts (last HTTP %s)"
        % (REGISTER_ATTEMPTS, last_status)
    )
    return False


def work_payload_done(payload):
    """Decide whether a /work response means the trial is finished."""
    if payload is None:
        return False
    if isinstance(payload, bool):
        return payload
    if isinstance(payload, dict):
        if payload.get("done") is True:
            return True
        status = payload.get("status")
        if isinstance(status, str) and status.strip().lower() in (
            "done",
            "complete",
            "completed",
            "finished",
            "no_work",
            "empty",
        ):
            return True
        if payload.get("remaining") == 0 and "remaining" in payload:
            return True
        work = payload.get("work")
        if isinstance(work, list) and not work and payload.get("done") is not False:
            # An explicit empty work list with no "done": false means
            # there is nothing left to do.
            if "status" in payload or "done" in payload:
                return True
        if payload.get("finished") is True:
            return True
    return False


def poll_work_until_done(env_url, stop_event, interval=WORK_POLL_INTERVAL_SEC):
    """Poll GET $MAKEFUN_ENV_URL/work until done or told to stop.

    Returns True when the work finished, False when stopped early or
    when there is no environment URL to poll. A 404 on /work means
    the environment has no work endpoint: keep serving instead of
    exiting, and report that once.
    """
    if not env_url:
        return False
    url = urllib.parse.urljoin(env_url.rstrip("/") + "/", "work")
    missing_reported = False
    while not stop_event.is_set():
        try:
            status, payload, _raw, _headers = http_request("GET", url)
        except ProviderError as exc:
            log_line("work: poll failed: %s" % exc)
            stop_event.wait(interval)
            continue
        if status == 404:
            if not missing_reported:
                log_line("work: no /work endpoint at %s, serving only" % url)
                missing_reported = True
            return False
        if status is not None and 200 <= status < 300:
            if work_payload_done(payload):
                log_line("work: done")
                return True
        else:
            log_line("work: poll got HTTP %s" % status)
        stop_event.wait(interval)
    return False


# ---------------------------------------------------------------------------
# Circuit breakers, provider health and route ranking
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """One provider's breaker: closed, briefly open, or open for trial."""

    def __init__(self, name):
        self.name = name
        self.open_until_mono = None
        self.open_for_trial = False
        self.failures = 0
        self.successes = 0

    def is_open(self, now_mono):
        if self.open_for_trial:
            return True
        if self.open_until_mono is None:
            return False
        return now_mono < self.open_until_mono

    def open_briefly(self, seconds, now_mono):
        until = now_mono + max(0.0, seconds)
        if self.open_until_mono is None or until > self.open_until_mono:
            self.open_until_mono = until

    def open_for_trial_now(self):
        self.open_for_trial = True
        self.open_until_mono = None

    def record_success(self):
        self.successes += 1

    def record_failure(self):
        self.failures += 1


class BreakerStore:
    """All providers' breakers, keyed by provider name."""

    def __init__(self):
        self.breakers = {}

    def breaker(self, name):
        item = self.breakers.get(name)
        if item is None:
            item = CircuitBreaker(name)
            self.breakers[name] = item
        return item

    def is_open(self, name):
        return self.breaker(name).is_open(time.monotonic())

    def open_briefly(self, name, seconds):
        self.breaker(name).open_briefly(seconds, time.monotonic())

    def open_for_trial(self, name):
        self.breaker(name).open_for_trial_now()

    def record_success(self, name):
        self.breaker(name).record_success()

    def record_failure(self, name):
        self.breaker(name).record_failure()

    def health_key(self, name):
        """Sort key: open breakers last, then fewest failures first."""
        item = self.breaker(name)
        open_flag = 1 if item.is_open(time.monotonic()) else 0
        return (open_flag, item.failures, -item.successes)

    def shortest_open_wait(self):
        """Seconds until the earliest live brief breaker closes, or None."""
        now = time.monotonic()
        waits = []
        for item in self.breakers.values():
            if item.open_for_trial:
                continue
            if item.open_until_mono is not None and now < item.open_until_mono:
                waits.append(item.open_until_mono - now)
        if not waits:
            return None
        return min(waits)


def route_meets(route_entry, needs):
    """Check one catalog entry against a request's needs."""
    if needs["has_tools"] and not route_entry["supports_tools"]:
        return False
    if needs["has_images"] and not route_entry["supports_images"]:
        return False
    if needs["wants_json"] and not route_entry["supports_json"]:
        return False
    if needs["estimated_tokens"] > route_entry["context_length"]:
        return False
    return True


def route_sort_key(route, entry, breakers, wanted_first):
    """Ranking: wanted model first, then rating, health, context."""
    return (
        0 if wanted_first else 1,
        -entry["rating"],
        breakers.health_key(route["provider"]),
        -entry["context_length"],
        route["provider"],
        route["model"],
    )


def rank_routes(catalog, needs, wanted_model, breakers, include_open=False):
    """Rank every route that can serve this request.

    Eligibility is relaxed in stages when nothing meets everything:
    first the JSON requirement goes, then images, then tools, then
    the context fit (smallest overflow first). A client-named model
    that is eligible goes first. Breaker-open providers sort last
    unless ``include_open`` lets them back in for a fresh sweep.
    """
    entries = {entry["id"]: entry for entry in catalog}
    wanted = (wanted_model or "").strip()
    stages = [
        {"tools": True, "images": True, "json": True, "context": True},
        {"tools": True, "images": True, "json": False, "context": True},
        {"tools": True, "images": False, "json": False, "context": True},
        {"tools": False, "images": False, "json": False, "context": True},
        {"tools": False, "images": False, "json": False, "context": False},
    ]
    for stage in stages:
        relaxed = {
            "has_tools": needs["has_tools"] and stage["tools"],
            "has_images": needs["has_images"] and stage["images"],
            "wants_json": needs["wants_json"] and stage["json"],
            "estimated_tokens": needs["estimated_tokens"]
            if stage["context"]
            else 0,
        }
        candidates = []
        for entry in catalog:
            if not route_meets(entry, relaxed):
                continue
            for route in entry["routes"]:
                if not include_open and breakers.is_open(route["provider"]):
                    continue
                wanted_first = bool(wanted) and entry["id"] == wanted
                candidates.append(
                    (route_sort_key(route, entry, breakers, wanted_first),
                     route, entry)
                )
        if candidates:
            candidates.sort(key=lambda item: item[0])
            ranked = []
            for _key, route, entry in candidates:
                ranked.append(
                    {
                        "model": entry["id"],
                        "provider": route["provider"],
                        "provider_model": route["model"],
                        "wire": route["wire"],
                        "rating": entry["rating"],
                        "context_length": entry["context_length"],
                    }
                )
            return ranked
    return []


def sort_candidates_with_overflow(catalog, needs, wanted_model, breakers):
    """Last-resort ranking when even the relaxed stages found nothing.

    Every route is a candidate; routes whose context overflows least
    come first, so the closest fit is tried before the worst.
    """
    wanted = (wanted_model or "").strip()
    scored = []
    for entry in catalog:
        for route in entry["routes"]:
            if breakers.is_open(route["provider"]):
                continue
            overflow = max(
                0, needs["estimated_tokens"] - entry["context_length"]
            )
            wanted_first = bool(wanted) and entry["id"] == wanted
            key = (
                0 if wanted_first else 1,
                overflow,
                -entry["rating"],
                breakers.health_key(route["provider"]),
                route["provider"],
                route["model"],
            )
            scored.append((key, route, entry))
    scored.sort(key=lambda item: item[0])
    return [
        {
            "model": entry["id"],
            "provider": route["provider"],
            "provider_model": route["model"],
            "wire": route["wire"],
            "rating": entry["rating"],
            "context_length": entry["context_length"],
        }
        for _key, route, entry in scored
    ]


# ---------------------------------------------------------------------------
# Request normalisation and wire translation (requests)
# ---------------------------------------------------------------------------

def normalize_openai_body(body):
    """Normalise an OpenAI chat-completions body into internal form."""
    if not isinstance(body, dict):
        body = {}
    messages = body.get("messages")
    if not isinstance(messages, list):
        messages = []
    clean_messages = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        clean = {"role": str(message.get("role") or "user")}
        content = message.get("content")
        if content is not None:
            clean["content"] = content
        if isinstance(message.get("tool_calls"), list):
            clean["tool_calls"] = message["tool_calls"]
        if isinstance(message.get("tool_call_id"), str):
            clean["tool_call_id"] = message["tool_call_id"]
        if isinstance(message.get("name"), str):
            clean["name"] = message["name"]
        clean_messages.append(clean)
    response_format = body.get("response_format")
    wants_json = False
    json_schema = None
    if isinstance(response_format, dict):
        rtype = response_format.get("type")
        if rtype in ("json_object", "json_schema"):
            wants_json = True
        if rtype == "json_schema" and isinstance(
            response_format.get("json_schema"), dict
        ):
            json_schema = response_format["json_schema"]
    tools = body.get("tools")
    if not isinstance(tools, list):
        tools = []
    return {
        "source_wire": "openai",
        "model": body.get("model") if isinstance(body.get("model"), str) else None,
        "messages": clean_messages,
        "system": None,
        "tools": tools,
        "wants_json": wants_json,
        "json_schema": json_schema,
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "stream": bool(body.get("stream")),
        "stop": body.get("stop"),
    }


def anthropic_content_to_openai(blocks):
    """Convert Anthropic content blocks into OpenAI-style parts."""
    parts = []
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        return ""
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append({"type": "text", "text": block["text"]})
        elif kind == "image":
            source = block.get("source") or {}
            if isinstance(source, dict) and source.get("type") == "base64":
                media = source.get("media_type") or "image/png"
                data = source.get("data") or ""
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:%s;base64,%s" % (media, data)
                        },
                    }
                )
            elif isinstance(source, dict) and source.get("url"):
                parts.append(
                    {"type": "image_url", "image_url": {"url": source["url"]}}
                )
        elif kind == "tool_use":
            parts.append(
                {
                    "type": "text",
                    "text": "[tool use %s: %s]"
                    % (
                        block.get("name") or "",
                        json.dumps(block.get("input") or {}, sort_keys=True),
                    ),
                }
            )
        elif kind == "tool_result":
            inner = block.get("content")
            if isinstance(inner, str):
                parts.append({"type": "text", "text": inner})
            elif isinstance(inner, list):
                for sub in inner:
                    if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                        parts.append({"type": "text", "text": sub["text"]})
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts


def normalize_anthropic_body(body):
    """Normalise an Anthropic messages body into internal form."""
    if not isinstance(body, dict):
        body = {}
    messages = []
    system = body.get("system")
    system_text = ""
    if isinstance(system, str):
        system_text = system
    elif isinstance(system, list):
        texts = [
            block.get("text")
            for block in system
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        system_text = "\n".join(texts)
    if system_text:
        messages.append({"role": "system", "content": system_text})
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        raw_messages = []
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        content = message.get("content")
        tool_calls = []
        tool_results = []
        text_parts = []
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "tool_use":
                    tool_calls.append(
                        {
                            "id": str(block.get("id") or ""),
                            "type": "function",
                            "function": {
                                "name": str(block.get("name") or ""),
                                "arguments": json.dumps(
                                    block.get("input") or {}, sort_keys=True
                                ),
                            },
                        }
                    )
                elif kind == "tool_result":
                    tool_results.append(block)
                else:
                    text_parts.append(block)
            converted = anthropic_content_to_openai(text_parts)
            clean = {"role": role, "content": converted}
            if tool_calls:
                clean["tool_calls"] = tool_calls
            if role == "user" and tool_results:
                # A tool result rides as a tool message after the user turn.
                messages.append(clean)
                for block in tool_results:
                    result_text = ""
                    inner = block.get("content")
                    if isinstance(inner, str):
                        result_text = inner
                    elif isinstance(inner, list):
                        result_text = "\n".join(
                            sub.get("text", "")
                            for sub in inner
                            if isinstance(sub, dict)
                            and isinstance(sub.get("text"), str)
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(block.get("tool_use_id") or ""),
                            "content": result_text,
                        }
                    )
                continue
            messages.append(clean)
        else:
            messages.append({"role": role, "content": content or ""})
    tools = body.get("tools")
    openai_tools = []
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(tool.get("name") or ""),
                        "description": str(tool.get("description") or ""),
                        "parameters": tool.get("input_schema") or {},
                    },
                }
            )
    return {
        "source_wire": "anthropic",
        "model": body.get("model") if isinstance(body.get("model"), str) else None,
        "messages": messages,
        "system": system_text or None,
        "tools": openai_tools,
        "wants_json": False,
        "json_schema": None,
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "stream": bool(body.get("stream")),
        "stop": body.get("stop_sequences"),
    }


def split_system_messages(messages):
    """Split leading system messages off; return (system_text, rest)."""
    system_parts = []
    rest = []
    seen_non_system = False
    for message in messages:
        if not seen_non_system and message.get("role") == "system":
            for _kind, value in message_text_parts(message):
                system_parts.append(value)
        else:
            seen_non_system = True
            rest.append(message)
    return "\n".join(system_parts), rest


def openai_to_anthropic_messages(messages):
    """Convert internal (OpenAI-style) messages to Anthropic messages."""
    system_text, rest = split_system_messages(messages)
    converted = []
    pending_tool_results = []
    for message in rest:
        role = message.get("role")
        if role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id") or "",
                    "content": message.get("content") or "",
                }
            )
            continue
        if role == "assistant" and message.get("tool_calls"):
            blocks = []
            text = ""
            for _kind, value in message_text_parts(message):
                text += value
            if text:
                blocks.append({"type": "text", "text": text})
            for call in message["tool_calls"]:
                fn = call.get("function") or {}
                args = safe_json_loads(fn.get("arguments") or "") or {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id") or "",
                        "name": fn.get("name") or "",
                        "input": args if isinstance(args, dict) else {},
                    }
                )
            converted.append({"role": "assistant", "content": blocks})
            continue
        anthropic_role = "assistant" if role == "assistant" else "user"
        blocks = []
        if pending_tool_results:
            blocks.extend(pending_tool_results)
            pending_tool_results = []
        for kind, value in message_text_parts(message):
            if kind == "text":
                blocks.append({"type": "text", "text": value})
            elif kind == "image":
                if value.startswith("data:"):
                    header, _, data = value.partition(",")
                    media = "image/png"
                    match = re.match(r"data:([^;]+);base64", header)
                    if match:
                        media = match.group(1)
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media,
                                "data": data,
                            },
                        }
                    )
                else:
                    blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": value},
                        }
                    )
        if not blocks:
            blocks = [{"type": "text", "text": ""}]
        converted.append({"role": anthropic_role, "content": blocks})
    if pending_tool_results:
        converted.append({"role": "user", "content": pending_tool_results})
    # Anthropic requires alternating roles; merge consecutive same-role turns.
    merged = []
    for item in converted:
        if merged and merged[-1]["role"] == item["role"]:
            merged[-1]["content"].extend(item["content"])
        else:
            merged.append({"role": item["role"], "content": list(item["content"])})
    return system_text, merged


def openai_tools_to_anthropic(tools):
    """Convert OpenAI-style tools to Anthropic-style tools."""
    converted = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") or tool
        if not isinstance(fn, dict):
            continue
        converted.append(
            {
                "name": fn.get("name") or "",
                "description": fn.get("description") or "",
                "input_schema": fn.get("parameters") or {"type": "object"},
            }
        )
    return converted


def openai_to_gemini_contents(messages):
    """Convert internal messages to Gemini contents + system instruction."""
    system_text, rest = split_system_messages(messages)
    contents = []
    for message in rest:
        role = message.get("role")
        gemini_role = "model" if role == "assistant" else "user"
        parts = []
        tool_calls = message.get("tool_calls") or []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            args = safe_json_loads(fn.get("arguments") or "") or {}
            parts.append(
                {
                    "functionCall": {
                        "name": fn.get("name") or "",
                        "args": args if isinstance(args, dict) else {},
                    }
                }
            )
        if role == "tool":
            parts.append(
                {
                    "functionResponse": {
                        "name": message.get("name") or "",
                        "response": {"content": message.get("content") or ""},
                    }
                }
            )
        for kind, value in message_text_parts(message):
            if kind == "text":
                parts.append({"text": value})
            elif kind == "image":
                if value.startswith("data:"):
                    header, _, data = value.partition(",")
                    media = "image/png"
                    match = re.match(r"data:([^;]+);base64", header)
                    if match:
                        media = match.group(1)
                    parts.append(
                        {"inlineData": {"mimeType": media, "data": data}}
                    )
        if not parts:
            parts = [{"text": ""}]
        if contents and contents[-1]["role"] == gemini_role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": gemini_role, "parts": parts})
    if not contents:
        contents = [{"role": "user", "parts": [{"text": ""}]}]
    return system_text, contents


def openai_tools_to_gemini(tools):
    """Convert OpenAI-style tools to Gemini function declarations."""
    declarations = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") or tool
        if not isinstance(fn, dict):
            continue
        declarations.append(
            {
                "name": fn.get("name") or "",
                "description": fn.get("description") or "",
                "parameters": fn.get("parameters") or {"type": "object"},
            }
        )
    if not declarations:
        return None
    return [{"functionDeclarations": declarations}]


def build_openai_provider_request(provider, route, request):
    """Build (url, headers, payload) for an OpenAI-wire provider call."""
    url = provider.models_url("v1/chat/completions")
    payload = {
        "model": route["provider_model"],
        "messages": request["messages"],
        "stream": bool(request.get("stream")),
    }
    if request.get("tools"):
        payload["tools"] = request["tools"]
    if request.get("max_tokens") is not None:
        payload["max_tokens"] = as_int(request["max_tokens"], 0) or None
    if request.get("temperature") is not None:
        payload["temperature"] = clamp_number(request["temperature"], 0.0, 2.0, 1.0)
    if request.get("wants_json"):
        if request.get("json_schema"):
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": request["json_schema"],
            }
        else:
            payload["response_format"] = {"type": "json_object"}
    if request.get("stop"):
        payload["stop"] = request["stop"]
    return url, provider.call_headers(), payload


def build_anthropic_provider_request(provider, route, request):
    """Build (url, headers, payload) for an Anthropic-wire provider call."""
    url = provider.models_url("v1/messages")
    system_text, anthropic_messages = openai_to_anthropic_messages(
        request["messages"]
    )
    payload = {
        "model": route["provider_model"],
        "messages": anthropic_messages,
        "max_tokens": as_int(request.get("max_tokens"), 0)
        or ANTHROPIC_DEFAULT_MAX_TOKENS,
        "stream": bool(request.get("stream")),
    }
    if system_text:
        payload["system"] = system_text
    if request.get("tools"):
        payload["tools"] = openai_tools_to_anthropic(request["tools"])
    if request.get("temperature") is not None:
        payload["temperature"] = clamp_number(request["temperature"], 0.0, 1.0, 1.0)
    if request.get("stop"):
        payload["stop_sequences"] = (
            request["stop"] if isinstance(request["stop"], list) else [request["stop"]]
        )
    return url, provider.call_headers(), payload


def build_gemini_provider_request(provider, route, request):
    """Build (url, headers, payload) for a Gemini-wire provider call."""
    system_text, contents = openai_to_gemini_contents(request["messages"])
    stream = bool(request.get("stream"))
    action = "streamGenerateContent" if stream else "generateContent"
    if stream:
        url = provider.models_url(
            "v1beta/models/%s:%s" % (route["provider_model"], action)
        )
    else:
        url = provider.models_url(
            "%s/models/%s:%s" % (GEMINI_API_VERSION, route["provider_model"], action)
        )
    payload = {"contents": contents}
    if system_text:
        payload["system_instruction"] = {"parts": [{"text": system_text}]}
    generation_config = {}
    if request.get("temperature") is not None:
        generation_config["temperature"] = clamp_number(
            request["temperature"], 0.0, 2.0, 1.0
        )
    if request.get("max_tokens") is not None:
        generation_config["maxOutputTokens"] = as_int(request["max_tokens"], 0) or None
    if request.get("wants_json"):
        generation_config["responseMimeType"] = "application/json"
        if request.get("json_schema"):
            schema = request["json_schema"]
            if isinstance(schema, dict) and isinstance(schema.get("schema"), dict):
                generation_config["responseSchema"] = schema["schema"]
    if request.get("stop"):
        stops = request["stop"] if isinstance(request["stop"], list) else [request["stop"]]
        generation_config["stopSequences"] = [str(s) for s in stops]
    if generation_config:
        payload["generationConfig"] = generation_config
    tools = openai_tools_to_gemini(request.get("tools"))
    if tools:
        payload["tools"] = tools
    headers = provider.call_headers()
    if stream:
        headers["Accept"] = "text/event-stream"
    return url, headers, payload


def build_provider_request(provider, route, request):
    """Dispatch to the right wire builder for a route."""
    if route["wire"] == "anthropic":
        return build_anthropic_provider_request(provider, route, request)
    if route["wire"] == "gemini":
        return build_gemini_provider_request(provider, route, request)
    return build_openai_provider_request(provider, route, request)


# ---------------------------------------------------------------------------
# Wire translation (responses): parse any provider wire into one shape
# ---------------------------------------------------------------------------

def empty_result(model):
    """A blank internal result for a given model id."""
    return {
        "text": "",
        "tool_calls": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "finish": "stop",
        "model": model,
    }


def normalize_tool_calls(raw_calls, model):
    """Normalise provider tool calls into internal {id, name, arguments}."""
    calls = []
    for index, call in enumerate(raw_calls or []):
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        if isinstance(fn, dict):
            name = fn.get("name") or call.get("name") or ""
            arguments = fn.get("arguments")
        else:
            name = call.get("name") or ""
            arguments = call.get("arguments")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, sort_keys=True)
        if not isinstance(arguments, str):
            arguments = ""
        call_id = call.get("id") or "call_%s_%d" % (model, index)
        calls.append(
            {"id": str(call_id), "name": str(name), "arguments": arguments}
        )
    return calls


def parse_openai_response(payload, model):
    """Parse an OpenAI-wire completion payload into internal form."""
    result = empty_result(model)
    if not isinstance(payload, dict):
        raise ProviderError("parse", None, "non-object OpenAI response")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderError("parse", None, "OpenAI response without choices")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        message = {}
    content = message.get("content")
    if isinstance(content, str):
        result["text"] = content
    elif isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        result["text"] = "".join(texts)
    result["tool_calls"] = normalize_tool_calls(message.get("tool_calls"), model)
    finish = choice.get("finish_reason") or "stop"
    if result["tool_calls"] and finish == "stop":
        finish = "tool_calls"
    result["finish"] = finish if finish in ("stop", "length", "tool_calls", "content_filter") else "stop"
    usage = payload.get("usage") or {}
    if isinstance(usage, dict):
        prompt = as_int(usage.get("prompt_tokens"), 0)
        completion = as_int(usage.get("completion_tokens"), 0)
        result["usage"] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
    if isinstance(payload.get("model"), str) and payload["model"]:
        result["model"] = payload["model"]
    return result


def parse_openai_stream_text(raw_text, model):
    """Assemble an OpenAI SSE stream body into an internal result."""
    result = empty_result(model)
    texts = []
    tool_deltas = {}
    finish = "stop"
    usage = None
    for line in raw_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        payload = safe_json_loads(data)
        if not isinstance(payload, dict):
            continue
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            delta = choice.get("delta") or {}
            if isinstance(delta, dict):
                if isinstance(delta.get("content"), str):
                    texts.append(delta["content"])
                for call in delta.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    index = as_int(call.get("index"), 0)
                    slot = tool_deltas.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    fn = call.get("function") or {}
                    if isinstance(fn, dict):
                        if isinstance(fn.get("name"), str):
                            slot["name"] = fn["name"]
                        if isinstance(fn.get("arguments"), str):
                            slot["arguments"] += fn["arguments"]
                    if isinstance(call.get("id"), str):
                        slot["id"] = call["id"]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
        if isinstance(payload.get("usage"), dict):
            usage = payload["usage"]
    result["text"] = "".join(texts)
    ordered = [tool_deltas[key] for key in sorted(tool_deltas.keys())]
    result["tool_calls"] = [
        {
            "id": slot["id"] or "call_%s_%d" % (model, index),
            "name": slot["name"],
            "arguments": slot["arguments"],
        }
        for index, slot in enumerate(ordered)
    ]
    if result["tool_calls"] and finish == "stop":
        finish = "tool_calls"
    result["finish"] = finish if finish in ("stop", "length", "tool_calls", "content_filter") else "stop"
    if usage is not None:
        prompt = as_int(usage.get("prompt_tokens"), 0)
        completion = as_int(usage.get("completion_tokens"), 0)
        result["usage"] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
    else:
        result["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": estimate_text_tokens(result["text"]),
            "total_tokens": estimate_text_tokens(result["text"]),
        }
    return result

def parse_anthropic_response(payload, model):
    """Parse an Anthropic-wire message payload into internal form."""
    result = empty_result(model)
    if not isinstance(payload, dict):
        raise ProviderError("parse", None, "non-object Anthropic response")
    if payload.get("type") == "error" or isinstance(payload.get("error"), dict):
        error = payload.get("error") or {}
        message = error.get("message") if isinstance(error, dict) else ""
        raise ProviderError("parse", None, "Anthropic error: %s" % (message or ""))
    texts = []
    raw_calls = []
    for block in payload.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif block.get("type") == "tool_use":
            raw_calls.append(
                {
                    "id": block.get("id") or "",
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": json.dumps(block.get("input") or {}, sort_keys=True),
                    },
                }
            )
    result["text"] = "".join(texts)
    result["tool_calls"] = normalize_tool_calls(raw_calls, model)
    stop = payload.get("stop_reason") or "end_turn"
    if result["tool_calls"]:
        result["finish"] = "tool_calls"
    elif stop in ("max_tokens",):
        result["finish"] = "length"
    else:
        result["finish"] = "stop"
    usage = payload.get("usage") or {}
    if isinstance(usage, dict):
        prompt = as_int(usage.get("input_tokens"), 0)
        completion = as_int(usage.get("output_tokens"), 0)
        result["usage"] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
    if isinstance(payload.get("model"), str) and payload["model"]:
        result["model"] = payload["model"]
    return result


def parse_anthropic_stream_text(raw_text, model):
    """Assemble an Anthropic SSE stream body into an internal result."""
    result = empty_result(model)
    texts = []
    tool_blocks = {}
    current_index = None
    current_kind = None
    stop_reason = "end_turn"
    usage = None
    event = None
    for line in raw_text.splitlines():
        line = line.strip()
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
            continue
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        payload = safe_json_loads(data)
        if not isinstance(payload, dict):
            continue
        if event == "message_start":
            message = payload.get("message") or {}
            if isinstance(message, dict) and isinstance(message.get("usage"), dict):
                usage = message["usage"]
        elif event == "content_block_start":
            current_index = as_int(payload.get("index"), 0)
            block = payload.get("content_block") or {}
            if isinstance(block, dict):
                current_kind = block.get("type")
                if current_kind == "tool_use":
                    tool_blocks[current_index] = {
                        "id": block.get("id") or "",
                        "name": block.get("name") or "",
                        "arguments": "",
                    }
        elif event == "content_block_delta":
            delta = payload.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            dtype = delta.get("type")
            index = as_int(payload.get("index"), current_index or 0)
            if dtype == "text_delta" and isinstance(delta.get("text"), str):
                texts.append(delta["text"])
            elif dtype == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                slot = tool_blocks.setdefault(
                    index, {"id": "", "name": "", "arguments": ""}
                )
                slot["arguments"] += delta["partial_json"]
        elif event == "message_delta":
            delta = payload.get("delta") or {}
            if isinstance(delta, dict) and delta.get("stop_reason"):
                stop_reason = delta["stop_reason"]
            if isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
    result["text"] = "".join(texts)
    ordered = [tool_blocks[key] for key in sorted(tool_blocks.keys())]
    result["tool_calls"] = [
        {
            "id": slot["id"] or "call_%s_%d" % (model, index),
            "name": slot["name"],
            "arguments": slot["arguments"],
        }
        for index, slot in enumerate(ordered)
    ]
    if result["tool_calls"]:
        result["finish"] = "tool_calls"
    elif stop_reason == "max_tokens":
        result["finish"] = "length"
    else:
        result["finish"] = "stop"
    if usage is not None:
        prompt = as_int(usage.get("input_tokens"), 0)
        completion = as_int(usage.get("output_tokens"), 0)
        result["usage"] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
    else:
        result["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": estimate_text_tokens(result["text"]),
            "total_tokens": estimate_text_tokens(result["text"]),
        }
    return result


def parse_gemini_response(payload, model):
    """Parse a Gemini generateContent payload into internal form."""
    result = empty_result(model)
    if not isinstance(payload, dict):
        raise ProviderError("parse", None, "non-object Gemini response")
    if isinstance(payload.get("error"), dict):
        error = payload["error"]
        raise ProviderError(
            classify_status(as_int(error.get("code"), 500)),
            as_int(error.get("code"), 500),
            "Gemini error: %s" % (error.get("message") or ""),
        )
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ProviderError("parse", None, "Gemini response without candidates")
    candidate = candidates[0] if isinstance(candidates[0], dict) else {}
    content = candidate.get("content") or {}
    parts = content.get("parts") if isinstance(content, dict) else None
    texts = []
    raw_calls = []
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif isinstance(part.get("functionCall"), dict):
                call = part["functionCall"]
                raw_calls.append(
                    {
                        "id": "",
                        "function": {
                            "name": call.get("name") or "",
                            "arguments": json.dumps(call.get("args") or {}, sort_keys=True),
                        },
                    }
                )
    result["text"] = "".join(texts)
    result["tool_calls"] = normalize_tool_calls(raw_calls, model)
    finish = candidate.get("finishReason") or "STOP"
    if result["tool_calls"]:
        result["finish"] = "tool_calls"
    elif finish == "MAX_TOKENS":
        result["finish"] = "length"
    elif finish in ("SAFETY", "RECITATION", "OTHER"):
        result["finish"] = "content_filter"
    else:
        result["finish"] = "stop"
    usage = payload.get("usageMetadata") or {}
    if isinstance(usage, dict):
        prompt = as_int(usage.get("promptTokenCount"), 0)
        completion = as_int(usage.get("candidatesTokenCount"), 0)
        result["usage"] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }
    return result


def parse_gemini_stream_text(raw_text, model):
    """Assemble Gemini streamGenerateContent SSE into an internal result."""
    result = empty_result(model)
    texts = []
    raw_calls = []
    finish = "STOP"
    usage = None
    for line in raw_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        payload = safe_json_loads(data)
        if not isinstance(payload, dict):
            continue
        piece = None
        try:
            piece = parse_gemini_response(payload, model)
        except ProviderError:
            continue
        texts.append(piece["text"])
        raw_calls.extend(piece["tool_calls"])
        finish = piece["finish"]
        usage = piece["usage"]
    result["text"] = "".join(texts)
    result["tool_calls"] = raw_calls
    result["finish"] = finish
    if usage is not None:
        result["usage"] = usage
    else:
        result["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": estimate_text_tokens(result["text"]),
            "total_tokens": estimate_text_tokens(result["text"]),
        }
    return result


def call_provider(provider, route, request):
    """Call one route and return the internal result.

    Raises ProviderError classified by the outcome: auth / payment /
    rate_limit / server / timeout / bad_request / not_found / parse.
    """
    url, headers, payload = build_provider_request(provider, route, request)
    stream = bool(request.get("stream"))
    try:
        status, parsed, raw, response_headers = http_request(
            "POST", url, headers=headers, payload=payload
        )
    except ProviderError:
        raise
    if status is None or not (200 <= status < 300):
        kind = classify_status(status, raw or "")
        retry_after = parse_retry_after(response_headers)
        detail = ""
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict):
                detail = str(error.get("message") or "")
            elif isinstance(error, str):
                detail = error
            elif isinstance(parsed.get("message"), str):
                detail = parsed["message"]
        raise ProviderError(
            kind, status, detail or ("HTTP %s from %s" % (status, provider.name)),
            retry_after=retry_after,
        )
    wire = route["wire"]
    try:
        if wire == "anthropic":
            if stream:
                return parse_anthropic_stream_text(raw or "", route["model"])
            return parse_anthropic_response(parsed, route["model"])
        if wire == "gemini":
            if stream:
                return parse_gemini_stream_text(raw or "", route["model"])
            return parse_gemini_response(parsed, route["model"])
        if stream:
            return parse_openai_stream_text(raw or "", route["model"])
        return parse_openai_response(parsed, route["model"])
    except ProviderError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ProviderError("parse", status, "could not parse %s reply: %s" % (wire, exc))


# ---------------------------------------------------------------------------
# Failover sweep: try routes in rank order until one answers
# ---------------------------------------------------------------------------

class NoRouteError(Exception):
    """No route in the catalog could serve this request at all."""


class AllFailedError(Exception):
    """Every route failed; carries the last failure for the error reply."""

    def __init__(self, last_error):
        super().__init__(str(last_error))
        self.last_error = last_error


class App:
    """Shared server state: catalog, providers, breakers, counters."""

    def __init__(self, providers, catalog, quality, seed, env_url):
        self.providers = {provider.name: provider for provider in providers}
        self.provider_list = list(providers)
        self.catalog = catalog
        self.quality = quality
        self.seed = seed
        self.env_url = env_url
        self.breakers = BreakerStore()
        self.dead_models = set()
        self.counter = 0
        self.counter_lock = threading.Lock()
        self.stop_event = threading.Event()

    def next_counter(self):
        with self.counter_lock:
            self.counter += 1
            return self.counter

    def note_failure(self, provider_name, error):
        """Apply the breaker rules for one failed provider call."""
        self.breakers.record_failure(provider_name)
        if error.kind in ("auth", "payment"):
            # 402 or an auth failure opens the breaker for the trial.
            self.breakers.open_for_trial(provider_name)
        elif error.kind == "rate_limit":
            wait = error.retry_after
            if wait is None:
                wait = BREAKER_SHORT_OPEN_SEC
            self.breakers.open_briefly(provider_name, min(wait, 120.0))
        elif error.kind in ("server", "timeout", "network"):
            self.breakers.open_briefly(provider_name, BREAKER_SHORT_OPEN_SEC)

    def note_model_failure(self, model_id, error):
        """A 400 / 404 / parse failure kills just that model."""
        if error.kind in ("bad_request", "not_found", "parse"):
            self.dead_models.add(model_id)

    def sweep(self, request):
        """Route one normalised request, failing over across routes.

        Returns (internal_result, route). Raises NoRouteError when the
        catalog has nothing to try, or AllFailedError after every
        sweep failed.
        """
        needs = request_needs(request)
        wanted = request.get("model")
        candidates = rank_routes(
            self.catalog, needs, wanted, self.breakers, include_open=False
        )
        if not candidates:
            candidates = sort_candidates_with_overflow(
                self.catalog, needs, wanted, self.breakers
            )
        candidates = [
            route for route in candidates if route["model"] not in self.dead_models
        ]
        if not candidates:
            raise NoRouteError("no eligible route for this request")
        last_error = None
        for sweep_index in range(MAX_SWEEPS):
            for route in candidates:
                provider = self.providers.get(route["provider"])
                if provider is None:
                    continue
                if self.breakers.is_open(route["provider"]):
                    continue
                try:
                    result = call_provider(provider, route, request)
                except ProviderError as exc:
                    last_error = exc
                    self.note_failure(route["provider"], exc)
                    self.note_model_failure(route["model"], exc)
                    log_line(
                        "route: %s via %s failed (%s): %s"
                        % (route["model"], route["provider"], exc.kind, exc.message)
                    )
                    continue
                self.breakers.record_success(route["provider"])
                return result, route
            # The whole sweep failed: wait for the shortest live breaker
            # (capped), then sweep again with breakers allowed back in.
            wait = self.breakers.shortest_open_wait()
            if wait is None or sweep_index + 1 >= MAX_SWEEPS:
                break
            capped = min(wait, MAX_BREAKER_WAIT_SEC)
            if capped > 0:
                time.sleep(capped)
            candidates = rank_routes(
                self.catalog, needs, wanted, self.breakers, include_open=True
            )
            if not candidates:
                candidates = sort_candidates_with_overflow(
                    self.catalog, needs, wanted, self.breakers
                )
            candidates = [
                route
                for route in candidates
                if route["model"] not in self.dead_models
            ]
            if not candidates:
                break
        if last_error is not None:
            raise AllFailedError(last_error)
        raise NoRouteError("no route answered")


# ---------------------------------------------------------------------------
# Response builders: client-facing JSON and SSE, both wires
# ---------------------------------------------------------------------------

def openai_tool_calls_json(tool_calls):
    """Render internal tool calls as OpenAI message tool_calls."""
    rendered = []
    for call in tool_calls:
        rendered.append(
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
            }
        )
    return rendered


def build_openai_response(result, model, seed, counter):
    """Render an internal result as an OpenAI chat-completion object."""
    message = {"role": "assistant", "content": result["text"]}
    if result["tool_calls"]:
        message["tool_calls"] = openai_tool_calls_json(result["tool_calls"])
    return {
        "id": deterministic_id("chatcmpl-", seed, counter, "openai"),
        "object": "chat.completion",
        "created": deterministic_created(seed),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result["finish"],
            }
        ],
        "usage": result["usage"],
    }


def build_openai_stream_chunks(result, model, seed, counter):
    """Render an internal result as OpenAI SSE chunk payloads."""
    response_id = deterministic_id("chatcmpl-", seed, counter, "openai")
    created = deterministic_created(seed)
    size = deterministic_chunk_size(seed, counter)
    text = result["text"]
    pieces = [text[i:i + size] for i in range(0, len(text), size)] or [""]
    chunks = []
    first = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
        ],
    }
    chunks.append("data: " + json.dumps(first, sort_keys=True) + "\n\n")
    for piece in pieces:
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": piece}, "finish_reason": None}
            ],
        }
        chunks.append("data: " + json.dumps(chunk, sort_keys=True) + "\n\n")
    if result["tool_calls"]:
        for index, call in enumerate(result["tool_calls"]):
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": call["name"],
                                        "arguments": call["arguments"],
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            chunks.append("data: " + json.dumps(chunk, sort_keys=True) + "\n\n")
    last = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": result["finish"]}
        ],
        "usage": result["usage"],
    }
    chunks.append("data: " + json.dumps(last, sort_keys=True) + "\n\n")
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks)


def anthropic_stop_reason(finish):
    """Map an internal finish reason to an Anthropic stop reason."""
    if finish == "tool_calls":
        return "tool_use"
    if finish == "length":
        return "max_tokens"
    if finish == "content_filter":
        return "stop"
    return "end_turn"


def build_anthropic_response(result, model, seed, counter):
    """Render an internal result as an Anthropic message object."""
    content = []
    if result["text"]:
        content.append({"type": "text", "text": result["text"]})
    for call in result["tool_calls"]:
        arguments = safe_json_loads(call["arguments"])
        if not isinstance(arguments, dict):
            arguments = {}
        content.append(
            {
                "type": "tool_use",
                "id": call["id"],
                "name": call["name"],
                "input": arguments,
            }
        )
    if not content:
        content = [{"type": "text", "text": ""}]
    return {
        "id": deterministic_id("msg_", seed, counter, "anthropic"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": anthropic_stop_reason(result["finish"]),
        "stop_sequence": None,
        "usage": {
            "input_tokens": result["usage"]["prompt_tokens"],
            "output_tokens": result["usage"]["completion_tokens"],
        },
    }


def build_anthropic_stream_events(result, model, seed, counter):
    """Render an internal result as Anthropic SSE events."""
    message_id = deterministic_id("msg_", seed, counter, "anthropic")
    size = deterministic_chunk_size(seed, counter)
    text = result["text"]
    pieces = [text[i:i + size] for i in range(0, len(text), size)] or [""]
    events = []
    start = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": result["usage"]["prompt_tokens"],
                "output_tokens": 0,
            },
        },
    }
    events.append("event: message_start\ndata: " + json.dumps(start, sort_keys=True) + "\n\n")
    block_index = 0
    text_block = {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    events.append("event: content_block_start\ndata: " + json.dumps(text_block, sort_keys=True) + "\n\n")
    for piece in pieces:
        delta = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        }
        events.append("event: content_block_delta\ndata: " + json.dumps(delta, sort_keys=True) + "\n\n")
    stop_block = {"type": "content_block_stop", "index": 0}
    events.append("event: content_block_stop\ndata: " + json.dumps(stop_block, sort_keys=True) + "\n\n")
    block_index = 1
    for call in result["tool_calls"]:
        start_block = {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {
                "type": "tool_use",
                "id": call["id"],
                "name": call["name"],
                "input": {},
            },
        }
        events.append("event: content_block_start\ndata: " + json.dumps(start_block, sort_keys=True) + "\n\n")
        delta = {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": "input_json_delta", "partial_json": call["arguments"]},
        }
        events.append("event: content_block_delta\ndata: " + json.dumps(delta, sort_keys=True) + "\n\n")
        stop_block = {"type": "content_block_stop", "index": block_index}
        events.append("event: content_block_stop\ndata: " + json.dumps(stop_block, sort_keys=True) + "\n\n")
        block_index += 1
    message_delta = {
        "type": "message_delta",
        "delta": {
            "stop_reason": anthropic_stop_reason(result["finish"]),
            "stop_sequence": None,
        },
        "usage": {"output_tokens": result["usage"]["completion_tokens"]},
    }
    events.append("event: message_delta\ndata: " + json.dumps(message_delta, sort_keys=True) + "\n\n")
    stop_event = {"type": "message_stop"}
    events.append("event: message_stop\ndata: " + json.dumps(stop_event, sort_keys=True) + "\n\n")
    return "".join(events)


def openai_error_body(status, message, code=None):
    """An OpenAI-style error object."""
    return {
        "error": {
            "message": message,
            "type": "server_error" if status >= 500 else "invalid_request_error",
            "code": code or ("server_error" if status >= 500 else "invalid_request"),
        }
    }


def anthropic_error_body(message, error_type="api_error"):
    """An Anthropic-style error object."""
    return {"type": "error", "error": {"type": error_type, "message": message}}


def error_status_for_failure(error):
    """Map a provider failure kind to the client-facing HTTP status."""
    mapping = {
        "auth": 401,
        "payment": 402,
        "rate_limit": 429,
        "bad_request": 400,
        "not_found": 404,
        "parse": 502,
        "server": 502,
        "timeout": 504,
        "network": 502,
    }
    return mapping.get(error.kind, 502)


def openai_models_list(catalog, seed):
    """The merged catalog as an OpenAI models list."""
    created = deterministic_created(seed)
    return {
        "object": "list",
        "data": [
            {
                "id": entry["id"],
                "object": "model",
                "created": created,
                "owned_by": entry["providers"][0] if entry["providers"] else "aggregator",
            }
            for entry in catalog
        ],
    }


def anthropic_models_list(catalog, seed):
    """The merged catalog as an Anthropic models list."""
    created = deterministic_created_iso(seed)
    return {
        "data": [
            {
                "id": entry["id"],
                "type": "model",
                "display_name": entry["id"],
                "created_at": created,
            }
            for entry in catalog
        ],
        "has_more": False,
        "first_id": catalog[0]["id"] if catalog else None,
        "last_id": catalog[-1]["id"] if catalog else None,
    }


# ---------------------------------------------------------------------------
# HTTP server: the client-facing endpoints
# ---------------------------------------------------------------------------

class AggregatorHandler(BaseHTTPRequestHandler):
    """Handles /v1/chat/completions, /v1/messages and /v1/models."""

    protocol_version = "HTTP/1.1"
    server_version = "FreeModelAggregator/1.0"

    @property
    def app(self):
        return self.server.app

    def log_message(self, _format, *_args):  # keep stdout clean
        return

    def _read_json_body(self):
        length = self.headers.get("Content-Length")
        try:
            size = int(length) if length is not None else 0
        except ValueError:
            size = 0
        if size <= 0:
            return {}
        raw = self.rfile.read(size)
        parsed = safe_json_loads(raw.decode("utf-8", errors="replace"))
        if parsed is None:
            return None
        return parsed

    def _send_json(self, status, payload):
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, status, text):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth_wire(self):
        """Pick the list wire from the auth header: x-api-key wins."""
        if self.headers.get("x-api-key"):
            return "anthropic"
        return "openai"

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/v1/models":
            if self._auth_wire() == "anthropic":
                self._send_json(
                    200, anthropic_models_list(self.app.catalog, self.app.seed)
                )
            else:
                self._send_json(
                    200, openai_models_list(self.app.catalog, self.app.seed)
                )
            return
        if parsed.path in ("/", "/health", "/healthz"):
            self._send_json(200, {"status": "ok", "models": len(self.app.catalog)})
            return
        self._send_json(404, openai_error_body(404, "not found", "not_found"))

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/v1/chat/completions":
            self._handle_chat_completions()
            return
        if parsed.path == "/v1/messages":
            self._handle_messages()
            return
        self._send_json(404, openai_error_body(404, "not found", "not_found"))

    def _handle_chat_completions(self):
        body = self._read_json_body()
        if body is None:
            self._send_json(400, openai_error_body(400, "invalid JSON body"))
            return
        if not isinstance(body, dict):
            self._send_json(400, openai_error_body(400, "body must be an object"))
            return
        request = normalize_openai_body(body)
        if not request["messages"]:
            self._send_json(
                400, openai_error_body(400, "messages must not be empty")
            )
            return
        counter = self.app.next_counter()
        try:
            result, route = self.app.sweep(request)
        except NoRouteError as exc:
            self._send_json(404, openai_error_body(404, str(exc), "model_not_found"))
            return
        except AllFailedError as exc:
            error = exc.last_error
            status = error_status_for_failure(error)
            self._send_json(status, openai_error_body(status, error.message))
            return
        model = route["model"]
        if request["stream"]:
            self._send_sse(
                200, build_openai_stream_chunks(result, model, self.app.seed, counter)
            )
        else:
            self._send_json(
                200, build_openai_response(result, model, self.app.seed, counter)
            )

    def _handle_messages(self):
        body = self._read_json_body()
        if body is None:
            self._send_json(
                400, anthropic_error_body("invalid JSON body", "invalid_request_error")
            )
            return
        if not isinstance(body, dict):
            self._send_json(
                400,
                anthropic_error_body("body must be an object", "invalid_request_error"),
            )
            return
        request = normalize_anthropic_body(body)
        if not request["messages"]:
            self._send_json(
                400,
                anthropic_error_body("messages must not be empty", "invalid_request_error"),
            )
            return
        counter = self.app.next_counter()
        try:
            result, route = self.app.sweep(request)
        except NoRouteError as exc:
            self._send_json(
                404, anthropic_error_body(str(exc), "not_found_error")
            )
            return
        except AllFailedError as exc:
            error = exc.last_error
            status = error_status_for_failure(error)
            error_type = "overloaded_error" if status in (429, 502, 504) else "api_error"
            self._send_json(status, anthropic_error_body(error.message, error_type))
            return
        model = route["model"]
        if request["stream"]:
            self._send_sse(
                200,
                build_anthropic_stream_events(result, model, self.app.seed, counter),
            )
        else:
            self._send_json(
                200, build_anthropic_response(result, model, self.app.seed, counter)
            )


def make_server(app, host, port):
    """Build the threaded HTTP server bound to host:port."""
    server = ThreadingHTTPServer((host, port), AggregatorHandler)
    server.app = app
    server.daemon_threads = True
    return server


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv):
    """Command-line flags for the server."""
    parser = argparse.ArgumentParser(
        description="One stdlib-only server merging free-model catalogs."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the catalog document the server registers, then exit.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind (default: $MAKEFUN_HOST or 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind (default: $MAKEFUN_PORT, $PORT or 8080).",
    )
    parser.add_argument(
        "--brief",
        default=None,
        help="Trial brief file (default: $MAKEFUN_PROMPT_FILE).",
    )
    parser.add_argument(
        "--env-url",
        default=None,
        help="Trial environment URL (default: $MAKEFUN_ENV_URL).",
    )
    return parser.parse_args(argv)


def resolve_port(args):
    """Port precedence: flag, MAKEFUN_PORT, PORT, default."""
    if args.port is not None:
        return args.port
    for name in ("MAKEFUN_PORT", "PORT"):
        value = os.environ.get(name)
        if value:
            try:
                return int(value)
            except ValueError:
                continue
    return DEFAULT_PORT


def resolve_host(args):
    """Host precedence: flag, MAKEFUN_HOST, default."""
    if args.host:
        return args.host
    return os.environ.get("MAKEFUN_HOST") or DEFAULT_HOST


def build_app(env_url, brief_path):
    """Load the brief, fetch every provider's models, build the app."""
    providers, quality, seed = load_brief(brief_path, env_url)
    for provider in providers:
        fetch_provider_models(provider)
    catalog = build_catalog(providers, quality)
    log_line(
        "catalog: %d models from %d providers"
        % (len(catalog), len(providers))
    )
    return App(providers, catalog, quality, seed, env_url), seed


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    env_url = args.env_url or os.environ.get("MAKEFUN_ENV_URL") or ""
    brief_path = args.brief or os.environ.get("MAKEFUN_PROMPT_FILE") or ""
    app, seed = build_app(env_url, brief_path)
    if args.list:
        sys.stdout.write(catalog_json_text(app.catalog))
        sys.stdout.flush()
        return 0
    register_catalog(env_url, app.catalog, seed)
    host = resolve_host(args)
    port = resolve_port(args)
    server = make_server(app, host, port)
    log_line("listening on http://%s:%d" % (host, port))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        done = poll_work_until_done(env_url, app.stop_event)
    except KeyboardInterrupt:
        done = False
    finally:
        app.stop_event.set()
        server.shutdown()
        server.server_close()
    if done:
        return 0
    if not env_url:
        # No trial environment: serving until interrupted is the job.
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
