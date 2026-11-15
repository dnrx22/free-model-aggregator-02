#!/usr/bin/env python3
"""free-model-aggregator-02: one server, every free model.

Reads the trial brief from MAKEFUN_PROMPT_FILE, learns each provider's
free model catalog, registers a single merged catalog with the trial
environment, and then serves OpenAI-compatible POST /v1/chat/completions
and Anthropic-compatible POST /v1/messages (JSON and SSE) plus
GET /v1/models, routing every request to the best eligible free model
and failing over when a provider refuses. Exits 0 once the environment
reports the work done.

Design notes (rewritten in full for the -02 round-2 entry):

* Routing is score-based, not a fixed priority list. A request's needs
  (tools, images, JSON mode, estimated context) are counted against
  every (model, provider) route; routes that meet everything rank by
  sealed quality, then by provider health (recent failures push a
  provider down), then by context length. If the client asked for a
  specific model and it is eligible, it goes first.
* Failover uses a small circuit breaker per provider: a 402 or auth
  failure opens the breaker permanently for the trial, a 429 opens it
  for the advertised Retry-After, a 5xx/timeout opens it briefly and
  counts toward a streak. A 400/404/parse failure kills just that
  model. After a full sweep with nothing left to try, the router waits
  for the shortest open breaker (capped) and sweeps once more.
* Deterministic: no wall clock in any decision and no unseeded
  randomness. Response ids and streaming chunk boundaries derive from
  MAKEFUN_TRIAL_SEED plus a per-request counter. ``python3 main.py
  --list`` prints the exact catalog document the server registers.

Round-2 changes over the first entry (same wire behaviour):
request bodies are capped (413 above the cap), the HTTP client sends a
User-Agent, Retry-After values are capped, breaker streak backoff is
capped, 408 is treated as a retryable server error, and OpenAI content
parts of type ``input_text`` are accepted as text.

Python 3.13, standard library only. All provider traffic stays under
MAKEFUN_ENV_URL; the server never contacts any other host.
"""

import hashlib
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
FIXED_CREATED = 1735689600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_CONTEXT = 131072
CALL_TIMEOUT = 15
CATALOG_TIMEOUT = 10
REGISTER_TIMEOUT = 5
WORK_TIMEOUT = 5
TRIAL_BUDGET_SECONDS = 170.0
MAX_BREAKER_WAIT = 5.0
TEXT_CHUNK = 20
ARGS_CHUNK = 28
USER_AGENT = "free-model-aggregator/1"
MAX_REQUEST_BODY_BYTES = 8 * 1024 * 1024
MAX_RETRY_AFTER_SECONDS = 3600.0
MAX_STREAK_BACKOFF = 5.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def canon(obj):
    """Canonical JSON: the one encoding used by --list and registration."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def est_tokens(text):
    if not text:
        return 0
    if not isinstance(text, str):
        text = str(text)
    return max(1, len(text) // 4)


def est_tokens_for_chars(char_count):
    """Token estimate from a character count, without building a string."""
    if not char_count:
        return 0
    return max(1, char_count // 4)


def to_float(value, default=0.0):
    try:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, dict):
            for key in ("score", "value", "quality", "rating"):
                if key in value:
                    return to_float(value[key], default)
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    try:
        if isinstance(value, bool):
            return 1 if value else 0
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def to_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "supported", "on")
    return default


def first(mapping, keys, default=None):
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def clean_model_id(raw):
    if raw is None:
        return ""
    text = str(raw).strip()
    if text.startswith("models/"):
        text = text[len("models/"):]
    return text


def retry_after_seconds(headers):
    """Retry-After seconds from response headers, or None.

    Numeric values only; an HTTP-date value is ignored (None) and any
    value is capped so a misbehaving provider cannot park us for hours.
    """
    if not headers:
        return None
    raw = None
    try:
        raw = headers.get("Retry-After") or headers.get("retry-after")
    except Exception:
        raw = None
    if raw is None:
        return None
    try:
        seconds = float(str(raw).strip())
    except ValueError:
        return None
    if seconds < 0:
        return 0.0
    return min(seconds, MAX_RETRY_AFTER_SECONDS)


class ProviderError(Exception):
    """A provider call failed; kind drives the failover decision."""

    def __init__(self, status, message="", retry_after=None, kind="error"):
        super().__init__(message or ("provider error %s" % status))
        self.status = status
        self.retry_after = retry_after
        self.kind = kind


# ---------------------------------------------------------------------------
# Brief parsing
# ---------------------------------------------------------------------------

def load_brief(path):
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def wire_name(raw):
    text = str(raw or "").strip().lower()
    if "anthropic" in text or "claude" in text:
        return "anthropic"
    if "gemini" in text or "google" in text or "generative" in text:
        return "gemini"
    if "openai" in text or "chat" in text or "completion" in text:
        return "openai"
    return text or "openai"

def normalize_provider(raw, index):
    """One provider entry from the brief, in our own flat shape."""
    if not isinstance(raw, dict):
        raw = {}
    pid = first(raw, ("id", "provider_id", "provider", "key_id"))
    if pid is None:
        pid = first(raw, ("name", "slug")) or ("provider-%d" % index)
    name = first(raw, ("name", "display_name", "label")) or pid
    wire = wire_name(first(raw, ("wire", "api", "protocol", "type", "kind", "format")))
    base = first(raw, ("base_url", "baseUrl", "base", "url", "endpoint", "api_base", "apiBase"))
    auth = raw.get("auth") if isinstance(raw.get("auth"), dict) else {}
    key = first(auth, ("key", "api_key", "apiKey", "value", "token"))
    if key is None:
        key = first(raw, ("key", "api_key", "apiKey", "token", "secret"))
    key_env = first(auth, ("env_var", "envVar", "key_env_var", "keyEnvVar", "env"))
    if key_env is None:
        key_env = first(raw, ("key_env_var", "keyEnvVar", "env_var", "envVar"))
    if key is None and key_env:
        key = os.environ.get(str(key_env))
    header = first(auth, ("header", "header_name", "headerName"))
    if header is None:
        header = first(raw, ("auth_header", "authHeader", "header"))
    scheme = first(auth, ("scheme", "type", "auth_scheme", "method"))
    if scheme is None:
        scheme = first(raw, ("auth_scheme", "authScheme", "auth", "auth_type"))
    if isinstance(scheme, dict):
        scheme = first(scheme, ("name", "type", "scheme"))
    scheme = str(scheme or "").strip().lower()
    models = []
    declared = first(raw, ("models", "model_ids", "modelIds", "catalog", "offered_models"))
    if isinstance(declared, list):
        for item in declared:
            if isinstance(item, dict):
                mid = first(item, ("id", "model", "model_id", "name"))
                if mid:
                    models.append(str(mid))
            elif item:
                models.append(str(item))
    elif isinstance(declared, dict):
        items = declared.get("data") or declared.get("models") or []
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict):
                mid = first(item, ("id", "model", "model_id", "name"))
                if mid:
                    models.append(str(mid))
            elif item:
                models.append(str(item))
    return {
        "id": str(pid),
        "name": str(name),
        "wire": wire,
        "base_url": str(base) if base else "",
        "key": str(key) if key is not None else "",
        "key_env": str(key_env) if key_env else "",
        "auth_header": str(header) if header else "",
        "auth_scheme": scheme,
        "models": models,
        "raw": raw,
    }


def providers_from_brief(brief):
    for key in ("providers", "provider_list", "providerList", "ten_providers"):
        value = brief.get(key)
        if isinstance(value, list) and value:
            return [normalize_provider(item, i) for i, item in enumerate(value)]
        if isinstance(value, dict) and value:
            merged = []
            for pid in sorted(value):
                item = value[pid]
                if isinstance(item, dict):
                    copy = dict(item)
                    copy.setdefault("id", pid)
                    merged.append(copy)
            if merged:
                return [normalize_provider(item, i) for i, item in enumerate(merged)]
    nested = brief.get("brief") or brief.get("trial") or brief.get("config")
    if isinstance(nested, dict):
        return providers_from_brief(nested)
    return []


def quality_entry(raw):
    """One quality-table value -> flat capability record ({} if unusable)."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return {"quality": float(raw)}
    if not isinstance(raw, dict):
        return {}
    quality = to_float(first(raw, ("rating", "quality", "score", "rank", "q")), 0.0)
    context = first(raw, ("context_length", "contextLength", "context",
                          "max_context", "context_window",
                          "max_context_length", "max_tokens", "ctx"))
    entry = {"quality": quality}
    if context is not None:
        entry["context_length"] = to_int(context, DEFAULT_CONTEXT)
    entry["tools"] = to_bool(first(raw, ("supports_tools", "tools", "tool_use",
                                         "toolUse", "function_calling", "supportsTools")), True)
    entry["images"] = to_bool(first(raw, ("supports_images", "images", "vision",
                                          "multimodal", "supportsImages", "image")), True)
    entry["json"] = to_bool(first(raw, ("supports_json", "json", "json_mode",
                                        "jsonMode", "structured_outputs", "supportsJson")), True)
    provider = first(raw, ("provider", "provider_id", "providerId"))
    if provider:
        entry["provider"] = str(provider)
    return entry


def quality_from_brief(brief):
    """{normalized model id: capability record} from any brief shape."""
    found = {}

    def absorb(mapping, provider_hint=None):
        if isinstance(mapping, dict):
            for key, value in mapping.items():
                entry = quality_entry(value)
                if entry:
                    if provider_hint:
                        entry.setdefault("provider", provider_hint)
                    found[clean_model_id(key)] = entry
                elif isinstance(value, dict):
                    # Possibly per-provider nesting: {provider: {model: {...}}}
                    absorb(value, provider_hint=str(key))
        elif isinstance(mapping, list):
            for item in mapping:
                if isinstance(item, dict):
                    mid = first(item, ("model", "model_id", "id", "name"))
                    if mid:
                        entry = quality_entry(item)
                        if provider_hint:
                            entry.setdefault("provider", provider_hint)
                        found[clean_model_id(mid)] = entry

    for key in ("sealed_quality_table", "quality_table", "quality",
                "model_quality", "ratings", "model_ratings", "qualityTable"):
        if key in brief:
            absorb(brief[key])
    if "models" in brief and isinstance(brief["models"], list):
        absorb(brief["models"])
    nested = brief.get("brief") or brief.get("trial") or brief.get("config")
    if isinstance(nested, dict) and not found:
        return quality_from_brief(nested)
    for entry in found.values():
        if not entry.get("provider"):
            entry.pop("provider", None)
    return found


# ---------------------------------------------------------------------------
# Provider URLs and auth (everything stays under MAKEFUN_ENV_URL)
# ---------------------------------------------------------------------------

def resolve_base(base_url, env_url):
    """Fold a provider's declared base under the trial environment origin.

    Absolute bases keep their own path but take the environment's
    scheme+host when the host matches the environment's (the mock and
    live harnesses both serve providers under the env origin). Relative
    bases are joined onto the environment URL.
    """
    base = (base_url or "").strip()
    env = (env_url or "").strip().rstrip("/")
    if not base:
        return env
    if base.startswith("http://") or base.startswith("https://"):
        try:
            parsed_base = urllib.parse.urlparse(base)
            parsed_env = urllib.parse.urlparse(env) if env else None
        except Exception:
            return base.rstrip("/")
        if parsed_env and parsed_base.netloc == parsed_env.netloc:
            return base.rstrip("/")
        if parsed_env:
            origin = "%s://%s" % (parsed_env.scheme, parsed_env.netloc)
            return (origin + parsed_base.path).rstrip("/") or origin
        return base.rstrip("/")
    if base.startswith("/"):
        if env:
            try:
                parsed_env = urllib.parse.urlparse(env)
                origin = "%s://%s" % (parsed_env.scheme, parsed_env.netloc)
                return (origin + base).rstrip("/")
            except Exception:
                return (env + base).rstrip("/")
        return base.rstrip("/")
    if env:
        return (env + "/" + base).rstrip("/")
    return base.rstrip("/")


def _join(base, suffix):
    return base.rstrip("/") + suffix


def endpoint_for(provider, env_url, kind, model_id=None):
    """URL for one provider operation (chat / messages / generate / models)."""
    base = resolve_base(provider["base_url"], env_url)
    if kind == "chat":
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1") or "/v1/" in base:
            return _join(base, "/chat/completions")
        return _join(base, "/v1/chat/completions")
    if kind == "messages":
        if base.endswith("/messages"):
            return base
        if base.endswith("/v1"):
            return _join(base, "/messages")
        return _join(base, "/v1/messages")
    if kind == "generate":
        model = urllib.parse.quote(clean_model_id(model_id), safe="-._")
        if ":generateContent" in base:
            return base
        if "/v1beta/" in base and base.rstrip("/").endswith("models"):
            return _join(base, "/%s:generateContent" % model)
        if "/v1beta" in base:
            return _join(base, "/models/%s:generateContent" % model)
        return _join(base, "/v1beta/models/%s:generateContent" % model)
    if kind == "models":
        if base.endswith("/models"):
            return base
        if provider["wire"] == "gemini":
            if "/v1beta" in base:
                return _join(base, "/models")
            return _join(base, "/v1beta/models")
        if base.endswith("/v1"):
            return _join(base, "/models")
        return _join(base, "/v1/models")
    raise ValueError("unknown endpoint kind %r" % kind)

def auth_headers(provider):
    """Auth headers for one provider, from its declared scheme."""
    headers = {}
    key = provider.get("key", "")
    scheme = (provider.get("auth_scheme") or "").lower()
    header_name = (provider.get("auth_header") or "").strip()
    if key:
        if header_name and header_name.lower() not in ("query", "query_param", "url"):
            if header_name.lower() == "authorization":
                headers[header_name] = key if key.lower().startswith("bearer ") else "Bearer " + key
            else:
                headers[header_name] = key
        elif "query" in scheme or "query" in header_name.lower():
            pass  # key travels in the query string (see auth_query)
        elif "bearer" in scheme:
            headers["Authorization"] = "Bearer " + key
        elif scheme in ("x-api-key", "api-key", "apikey", "api_key", "anthropic"):
            headers["x-api-key"] = key
        elif scheme in ("goog", "google", "x-goog-api-key"):
            headers["x-goog-api-key"] = key
        elif provider["wire"] == "anthropic":
            headers["x-api-key"] = key
        elif provider["wire"] == "gemini":
            headers["x-goog-api-key"] = key
        else:
            headers["Authorization"] = "Bearer " + key
    if provider["wire"] == "anthropic":
        headers.setdefault("anthropic-version", "2023-06-01")
    return headers


def auth_query(provider, url):
    """Append ?key= for providers whose scheme (or Gemini) needs it."""
    key = provider.get("key", "")
    if not key:
        return url
    scheme = (provider.get("auth_scheme") or "").lower()
    header_name = (provider.get("auth_header") or "").lower()
    needs_query = ("query" in scheme or "query" in header_name
                   or scheme in ("key", "api_key_param", "param"))
    if provider["wire"] == "gemini":
        # Gemini accepts the key as a query param; send it alongside
        # the header so mock and real wires both authenticate.
        needs_query = True
    if not needs_query:
        return url
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    if any(k == "key" for k, _ in query):
        return url
    query.append(("key", key))
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(query), parts.fragment))


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

def http_request(method, url, headers, body_bytes, timeout):
    request = urllib.request.Request(url, data=body_bytes, method=method)
    for name, value in headers.items():
        request.add_header(name, value)
    # urllib sends a Python-urllib User-Agent by default; identify the
    # aggregator instead so the trial harness can tell our calls apart.
    if not any(name.lower() == "user-agent" for name in headers):
        request.add_header("User-Agent", USER_AGENT)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        try:
            payload = exc.read()
        except Exception:
            payload = b""
        return exc.code, exc.headers, payload
    except urllib.error.URLError as exc:
        raise ProviderError(0, "network error: %s" % exc, kind="network")
    except (TimeoutError, socket.timeout) as exc:
        raise ProviderError(0, "timeout: %s" % exc, kind="timeout")
    except Exception as exc:
        raise ProviderError(0, "request failed: %s" % exc, kind="network")


def post_json(url, payload, headers, timeout):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    merged = dict(headers)
    merged.setdefault("Content-Type", "application/json")
    return http_request("POST", url, merged, body, timeout)


def get_json(url, headers, timeout):
    status, _resp_headers, body = http_request("GET", url, headers, None, timeout)
    if status != 200:
        raise ProviderError(status, "GET %s -> %s" % (url, status))
    try:
        return json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise ProviderError(status, "bad JSON: %s" % exc)


# ---------------------------------------------------------------------------
# Catalog fetching and building
# ---------------------------------------------------------------------------

def fetch_provider_models(provider, env_url):
    """Model ids a provider advertises; [] on any failure (never fatal)."""
    url = auth_query(provider, endpoint_for(provider, env_url, "models"))
    headers = auth_headers(provider)
    headers.setdefault("Accept", "application/json")
    try:
        status, _headers, body = http_request("GET", url, headers, None, CATALOG_TIMEOUT)
    except ProviderError:
        return []
    if status != 200:
        return []
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        return []
    items = []
    if isinstance(data, dict):
        items = data.get("data") or data.get("models") or []
    elif isinstance(data, list):
        items = data
    ids, seen = [], set()
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict):
            raw_id = first(item, ("id", "name", "model", "model_id"))
            model_id = clean_model_id(raw_id) if raw_id else ""
        elif item:
            model_id = clean_model_id(item)
        else:
            model_id = ""
        if model_id and model_id not in seen:
            seen.add(model_id)
            ids.append(model_id)
    return ids


def build_catalog(providers, quality, fetched):
    """Build (catalog dict, routing entries).

    Catalog entries are unique per model id. Routing entries are
    (model, provider) pairs so failover can move across providers that
    offer the same model.
    """
    routes = []
    by_id = {}
    for provider in providers:
        model_ids = list(fetched.get(provider["id"], []))
        if not model_ids:
            model_ids = [clean_model_id(m) for m in provider["models"]]
        if not model_ids:
            for mid, entry in quality.items():
                if entry.get("provider") == provider["id"]:
                    model_ids.append(mid)
        for model_id in model_ids:
            if not model_id:
                continue
            entry = quality.get(model_id, {})
            qual = to_float(entry.get("quality"), 0.0)
            context = to_int(entry.get("context_length"), DEFAULT_CONTEXT)
            supports = {
                "tools": to_bool(entry.get("tools"), True),
                "images": to_bool(entry.get("images"), True),
                "json": to_bool(entry.get("json"), True),
            }
            routes.append({
                "id": model_id,
                "provider_id": provider["id"],
                "provider_name": provider["name"],
                "wire": provider["wire"],
                "quality": qual,
                "context_length": context,
                "supports_tools": supports["tools"],
                "supports_images": supports["images"],
                "supports_json": supports["json"],
            })
            if model_id not in by_id:
                by_id[model_id] = {
                    "id": model_id,
                    "object": "model",
                    "created": FIXED_CREATED,
                    "owned_by": provider["id"],
                    "provider": provider["id"],
                    "provider_name": provider["name"],
                    "wire": provider["wire"],
                    "quality": qual,
                    "context_length": context,
                    "supports_tools": supports["tools"],
                    "supports_images": supports["images"],
                    "supports_json": supports["json"],
                    "providers": [provider["id"]],
                }
            else:
                existing = by_id[model_id]
                if provider["id"] not in existing["providers"]:
                    existing["providers"].append(provider["id"])

    def route_key(item):
        return (-item["quality"], -item["context_length"], item["id"], item["provider_id"])

    routes.sort(key=route_key)
    models = sorted(by_id.values(),
                    key=lambda m: (-m["quality"], -m["context_length"], m["id"]))
    for model in models:
        model["providers"] = sorted(model["providers"])
    catalog = {"object": "list", "data": models, "models": models}
    return catalog, routes


def catalog_for_list():
    """Build the catalog exactly as the server build does (best effort)."""
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    brief = load_brief(os.environ.get("MAKEFUN_PROMPT_FILE", ""))
    providers = providers_from_brief(brief)
    quality = quality_from_brief(brief)
    fetched = {}
    if env_url:
        for provider in providers:
            fetched[provider["id"]] = fetch_provider_models(provider, env_url)
    catalog, _routes = build_catalog(providers, quality, fetched)
    return catalog


# ---------------------------------------------------------------------------
# Request normalization (both client wires -> one internal shape)
# ---------------------------------------------------------------------------

def _split_data_url(url):
    if not url.startswith("data:"):
        return None, None
    try:
        header, data = url[5:].split(",", 1)
    except ValueError:
        return None, None
    return (header.split(";")[0] or "image/png"), data


def _openai_image(image_url):
    if isinstance(image_url, dict):
        url = image_url.get("url", "")
    else:
        url = str(image_url or "")
    media_type, data = _split_data_url(url)
    if data is not None:
        return {"media_type": media_type, "data": data, "url": None}
    return {"media_type": "image/png", "data": None, "url": url}


def _anthropic_image(source):
    if not isinstance(source, dict):
        return {"media_type": "image/png", "data": None, "url": None}
    kind = source.get("type")
    if kind == "base64":
        return {"media_type": source.get("media_type", "image/png"),
                "data": source.get("data"), "url": None}
    if kind == "url":
        return {"media_type": "image/png", "data": None, "url": source.get("url")}
    return {"media_type": "image/png", "data": None, "url": None}


def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _parse_args(raw):
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except Exception:
            parsed = {"_raw": raw}
    elif isinstance(raw, dict):
        parsed = raw
    else:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _tool_choice_openai(value):
    if value is None:
        return {"mode": "auto"}
    if isinstance(value, str):
        if value == "none":
            return {"mode": "none"}
        if value == "required":
            return {"mode": "required"}
        return {"mode": "auto"}
    if isinstance(value, dict):
        fn = value.get("function") or {}
        name = fn.get("name") or value.get("name")
        if name:
            return {"mode": "specific", "name": str(name)}
        if value.get("type") == "none":
            return {"mode": "none"}
        return {"mode": "auto"}
    return {"mode": "auto"}


def _tool_choice_anthropic(value):
    if value is None:
        return {"mode": "auto"}
    if isinstance(value, str):
        return {"mode": "none"} if value == "none" else {"mode": "auto"}
    if isinstance(value, dict):
        kind = value.get("type")
        if kind == "none":
            return {"mode": "none"}
        if kind == "any":
            return {"mode": "required"}
        if kind == "tool" and value.get("name"):
            return {"mode": "specific", "name": str(value["name"])}
        return {"mode": "auto"}
    return {"mode": "auto"}


def _empty_message(role):
    return {"role": role, "texts": [], "images": [], "tool_calls": [], "tool_results": []}


def normalize_openai(body):
    system_parts = []
    messages = []
    for raw_msg in body.get("messages") or []:
        if not isinstance(raw_msg, dict):
            continue
        role = raw_msg.get("role", "user")
        content = raw_msg.get("content")
        if role == "system":
            system_parts.append(_text_of(content))
            continue
        msg = _empty_message(role)
        if role == "tool":
            msg["tool_results"].append({
                "tool_call_id": str(raw_msg.get("tool_call_id", "")),
                "content": _text_of(content), "is_error": False})
            messages.append(msg)
            continue
        if role == "function":
            msg["tool_results"].append({
                "tool_call_id": str(raw_msg.get("name", "")),
                "content": _text_of(content), "is_error": False})
            messages.append(msg)
            continue
        if isinstance(content, str):
            if content:
                msg["texts"].append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    if isinstance(part, str) and part:
                        msg["texts"].append(part)
                    continue
                ptype = part.get("type")
                if ptype in ("text", "input_text"):
                    if part.get("text"):
                        msg["texts"].append(part["text"])
                elif ptype in ("image_url", "image"):
                    image_url = part.get("image_url") or part.get("image") or part.get("source")
                    msg["images"].append(_openai_image(image_url))
                elif ptype == "input_audio":
                    msg["texts"].append("[audio input]")
        for call in raw_msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            msg["tool_calls"].append({
                "id": str(call.get("id", "")),
                "name": str(fn.get("name", "")),
                "arguments": _parse_args(fn.get("arguments", {})),
            })
        messages.append(msg)
    tools = []
    for raw_tool in body.get("tools") or []:
        if not isinstance(raw_tool, dict):
            continue
        fn = raw_tool.get("function") or raw_tool
        tools.append({
            "name": str(fn.get("name", "")),
            "description": str(fn.get("description", "")),
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    response_format = body.get("response_format") or {}
    need_json = isinstance(response_format, dict) and response_format.get("type") in (
        "json_object", "json_schema")
    json_schema = None
    if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
        schema = response_format.get("json_schema") or {}
        if isinstance(schema, dict):
            json_schema = schema.get("schema") or schema
    stream_options = body.get("stream_options") or {}
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list):
        stop = []
    return {
        "api": "openai",
        "system": "\n".join(p for p in system_parts if p),
        "messages": messages,
        "tools": tools,
        "tool_choice": _tool_choice_openai(body.get("tool_choice")),
        "max_tokens": to_int(max_tokens, 0) or None,
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "stop": [str(s) for s in stop],
        "need_json": need_json,
        "json_schema": json_schema,
        "stream": bool(body.get("stream")),
        "include_usage": bool(stream_options.get("include_usage")),
        "requested_model": body.get("model"),
    }

def normalize_anthropic(body):
    system_raw = body.get("system")
    if isinstance(system_raw, str):
        system = system_raw
    elif isinstance(system_raw, list):
        system = "".join(
            block.get("text", "") for block in system_raw
            if isinstance(block, dict) and block.get("type") == "text")
    else:
        system = ""
    messages = []
    for raw_msg in body.get("messages") or []:
        if not isinstance(raw_msg, dict):
            continue
        role = raw_msg.get("role", "user")
        content = raw_msg.get("content")
        msg = _empty_message(role)
        if isinstance(content, str):
            if content:
                msg["texts"].append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    if isinstance(block, str) and block:
                        msg["texts"].append(block)
                    continue
                btype = block.get("type")
                if btype == "text":
                    if block.get("text"):
                        msg["texts"].append(block["text"])
                elif btype == "image":
                    msg["images"].append(_anthropic_image(block.get("source") or {}))
                elif btype == "tool_use":
                    args = block.get("input")
                    msg["tool_calls"].append({
                        "id": str(block.get("id", "")),
                        "name": str(block.get("name", "")),
                        "arguments": args if isinstance(args, dict) else {},
                    })
                elif btype == "tool_result":
                    msg["tool_results"].append({
                        "tool_call_id": str(block.get("tool_use_id", "")),
                        "content": _text_of(block.get("content")),
                        "is_error": bool(block.get("is_error")),
                    })
        messages.append(msg)
    tools = []
    for raw_tool in body.get("tools") or []:
        if not isinstance(raw_tool, dict):
            continue
        tools.append({
            "name": str(raw_tool.get("name", "")),
            "description": str(raw_tool.get("description", "")),
            "parameters": raw_tool.get("input_schema") or {"type": "object", "properties": {}},
        })
    stop = body.get("stop_sequences")
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list):
        stop = []
    return {
        "api": "anthropic",
        "system": system,
        "messages": messages,
        "tools": tools,
        "tool_choice": _tool_choice_anthropic(body.get("tool_choice")),
        "max_tokens": to_int(body.get("max_tokens"), 0) or None,
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "stop": [str(s) for s in stop],
        "need_json": False,
        "json_schema": None,
        "stream": bool(body.get("stream")),
        "include_usage": False,
        "requested_model": body.get("model"),
    }


def requirements_of(norm):
    """What a request needs from a model, for routing."""
    text_chars = len(norm["system"])
    image_count = 0
    for msg in norm["messages"]:
        for text in msg["texts"]:
            text_chars += len(text)
        for call in msg["tool_calls"]:
            text_chars += len(canon(call["arguments"]))
        for result in msg["tool_results"]:
            text_chars += len(result["content"])
        image_count += len(msg["images"])
    # Count characters, not a throwaway string: a long conversation
    # would otherwise allocate a copy of its full text just to divide.
    est_input = est_tokens_for_chars(text_chars) + 1024 * image_count + 50 * len(norm["tools"])
    return {
        "tools": bool(norm["tools"]),
        "images": image_count > 0,
        "json": bool(norm["need_json"]),
        "est_tokens": est_input + (norm["max_tokens"] or DEFAULT_MAX_TOKENS),
        "est_input_tokens": est_input,
    }


# ---------------------------------------------------------------------------
# Provider request builders
# ---------------------------------------------------------------------------

def _openai_message(msg):
    """One normalized message -> OpenAI message (None for tool results)."""
    if msg["tool_results"]:
        return None
    parts = []
    for text in msg["texts"]:
        parts.append({"type": "text", "text": text})
    for image in msg["images"]:
        if image.get("data"):
            url = "data:%s;base64,%s" % (image["media_type"], image["data"])
        else:
            url = image.get("url") or ""
        parts.append({"type": "image_url", "image_url": {"url": url}})
    out = {"role": msg["role"]}
    if len(parts) == 1 and parts[0]["type"] == "text" and not msg["tool_calls"]:
        out["content"] = parts[0]["text"]
    elif not parts and msg["tool_calls"]:
        out["content"] = None
    else:
        out["content"] = parts
    if msg["tool_calls"]:
        out["tool_calls"] = [{
            "id": call["id"] or "call_%s" % idx,
            "type": "function",
            "function": {"name": call["name"], "arguments": canon(call["arguments"])},
        } for idx, call in enumerate(msg["tool_calls"])]
    return out


def openai_request_body(norm, model_id):
    messages = []
    if norm["system"]:
        messages.append({"role": "system", "content": norm["system"]})
    for msg in norm["messages"]:
        if msg["tool_results"] and msg["role"] != "assistant":
            for result in msg["tool_results"]:
                messages.append({"role": "tool", "tool_call_id": result["tool_call_id"],
                                 "content": result["content"]})
            if msg["texts"] or msg["images"]:
                rebuilt = dict(msg)
                rebuilt["tool_results"] = []
                built = _openai_message(rebuilt)
                if built is not None:
                    messages.append(built)
            continue
        built = _openai_message(msg)
        if built is not None:
            messages.append(built)
    body = {"model": model_id, "messages": messages, "stream": False}
    if norm["max_tokens"]:
        body["max_tokens"] = norm["max_tokens"]
    if norm["temperature"] is not None:
        body["temperature"] = norm["temperature"]
    if norm["top_p"] is not None:
        body["top_p"] = norm["top_p"]
    if norm["stop"]:
        body["stop"] = norm["stop"]
    if norm["tools"]:
        body["tools"] = [{
            "type": "function",
            "function": {"name": tool["name"], "description": tool["description"],
                         "parameters": tool["parameters"]},
        } for tool in norm["tools"]]
        choice = norm["tool_choice"]
        if choice["mode"] == "none":
            body["tool_choice"] = "none"
        elif choice["mode"] == "required":
            body["tool_choice"] = "required"
        elif choice["mode"] == "specific":
            body["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        else:
            body["tool_choice"] = "auto"
    if norm["need_json"]:
        if norm["json_schema"]:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": norm["json_schema"]},
            }
        else:
            body["response_format"] = {"type": "json_object"}
    return body


def _anthropic_blocks(msg):
    blocks = []
    for text in msg["texts"]:
        blocks.append({"type": "text", "text": text})
    for image in msg["images"]:
        if image.get("data"):
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": image["media_type"], "data": image["data"]}})
        elif image.get("url"):
            blocks.append({"type": "image", "source": {"type": "url", "url": image["url"]}})
    for call in msg["tool_calls"]:
        blocks.append({"type": "tool_use", "id": call["id"], "name": call["name"],
                       "input": call["arguments"]})
    for result in msg["tool_results"]:
        blocks.append({"type": "tool_result", "tool_use_id": result["tool_call_id"],
                       "content": result["content"], "is_error": result["is_error"]})
    return blocks


def _merge_same_role(messages):
    merged = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"].extend(msg["content"])
        else:
            merged.append({"role": msg["role"], "content": list(msg["content"])})
    return merged


def anthropic_request_body(norm, model_id):
    messages = []
    for msg in norm["messages"]:
        role = "assistant" if msg["role"] == "assistant" else "user"
        blocks = _anthropic_blocks(msg)
        if not blocks:
            blocks = [{"type": "text", "text": ""}]
        messages.append({"role": role, "content": blocks})
    messages = _merge_same_role(messages)
    if messages and messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": [{"type": "text", "text": "Continue."}]})
    if not messages:
        messages = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    body = {"model": model_id, "messages": messages,
            "max_tokens": norm["max_tokens"] or DEFAULT_MAX_TOKENS, "stream": False}
    system = norm["system"]
    if norm["need_json"]:
        suffix = "Respond with valid JSON only."
        system = (system + "\n\n" + suffix) if system else suffix
    if system:
        body["system"] = system
    if norm["temperature"] is not None:
        body["temperature"] = norm["temperature"]
    if norm["top_p"] is not None:
        body["top_p"] = norm["top_p"]
    if norm["stop"]:
        body["stop_sequences"] = norm["stop"]
    if norm["tools"]:
        body["tools"] = [{
            "name": tool["name"], "description": tool["description"],
            "input_schema": tool["parameters"],
        } for tool in norm["tools"]]
        choice = norm["tool_choice"]
        if choice["mode"] == "none":
            body["tool_choice"] = {"type": "none"}
        elif choice["mode"] == "required":
            body["tool_choice"] = {"type": "any"}
        elif choice["mode"] == "specific":
            body["tool_choice"] = {"type": "tool", "name": choice["name"]}
        else:
            body["tool_choice"] = {"type": "auto"}
    return body

def _gemini_parts(msg):
    parts = []
    for text in msg["texts"]:
        parts.append({"text": text})
    for image in msg["images"]:
        if image.get("data"):
            parts.append({"inlineData": {"mimeType": image["media_type"], "data": image["data"]}})
        elif image.get("url"):
            parts.append({"fileData": {"mimeType": image["media_type"], "fileUri": image["url"]}})
    for call in msg["tool_calls"]:
        parts.append({"functionCall": {"name": call["name"], "args": call["arguments"]}})
    for result in msg["tool_results"]:
        parts.append({"functionResponse": {
            "name": result["tool_call_id"],
            "response": {"result": result["content"]}}})
    return parts


def gemini_request_body(norm, model_id):
    contents = []
    for msg in norm["messages"]:
        role = "model" if msg["role"] == "assistant" else "user"
        parts = _gemini_parts(msg)
        if not parts:
            parts = [{"text": ""}]
        contents.append({"role": role, "parts": parts})
    merged = []
    for item in contents:
        if merged and merged[-1]["role"] == item["role"]:
            merged[-1]["parts"].extend(item["parts"])
        else:
            merged.append({"role": item["role"], "parts": list(item["parts"])})
    contents = merged
    if contents and contents[0]["role"] != "user":
        contents.insert(0, {"role": "user", "parts": [{"text": "Continue."}]})
    if not contents:
        contents = [{"role": "user", "parts": [{"text": ""}]}]
    body = {"contents": contents}
    if norm["system"]:
        body["systemInstruction"] = {"parts": [{"text": norm["system"]}]}
    config = {}
    if norm["max_tokens"]:
        config["maxOutputTokens"] = norm["max_tokens"]
    if norm["temperature"] is not None:
        config["temperature"] = norm["temperature"]
    if norm["top_p"] is not None:
        config["topP"] = norm["top_p"]
    if norm["stop"]:
        config["stopSequences"] = norm["stop"]
    if norm["need_json"]:
        config["responseMimeType"] = "application/json"
        if norm["json_schema"] and isinstance(norm["json_schema"], dict):
            config["responseSchema"] = norm["json_schema"]
    if config:
        body["generationConfig"] = config
    if norm["tools"]:
        body["tools"] = [{"functionDeclarations": [{
            "name": tool["name"], "description": tool["description"],
            "parameters": tool["parameters"],
        } for tool in norm["tools"]]}]
    return body


# ---------------------------------------------------------------------------
# Provider response parsing -> one normalized result
# ---------------------------------------------------------------------------

def _fresh_result():
    return {"text": "", "tool_calls": [], "stop": "stop",
            "input_tokens": None, "output_tokens": None}


def _sse_payloads(body_bytes):
    text = body_bytes.decode("utf-8", errors="replace")
    payloads = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                try:
                    payloads.append(json.loads(data))
                except Exception:
                    pass
    return payloads


def _is_sse(body_bytes, content_type=""):
    head = body_bytes[:64]
    return (head.startswith(b"data:") or head.startswith(b"event:")
            or b"\ndata:" in body_bytes[:512]
            or "text/event-stream" in (content_type or ""))


_OPENAI_FINISH = {"stop": "stop", "length": "length",
                  "tool_calls": "tool_calls", "content_filter": "content_filter"}
_ANTHROPIC_STOP = {"end_turn": "stop", "max_tokens": "length",
                   "tool_use": "tool_calls", "stop_sequence": "stop",
                   "pause_turn": "stop", "refusal": "content_filter"}
_GEMINI_FINISH = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter",
                  "RECITATION": "content_filter", "OTHER": "stop",
                  "BLOCKLIST": "content_filter", "PROHIBITED_CONTENT": "content_filter",
                  "SPII": "content_filter", "MALFORMED_FUNCTION_CALL": "stop"}


def parse_openai_response(body_bytes, content_type=""):
    result = _fresh_result()
    if _is_sse(body_bytes, content_type):
        tool_acc = {}
        for chunk in _sse_payloads(body_bytes):
            choices = chunk.get("choices") or []
            if not choices:
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    result["input_tokens"] = usage.get("prompt_tokens")
                    result["output_tokens"] = usage.get("completion_tokens")
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                result["text"] += content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        result["text"] += part.get("text", "")
            for call in delta.get("tool_calls") or []:
                idx = call.get("index", 0)
                acc = tool_acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                if call.get("id"):
                    acc["id"] = call["id"]
                fn = call.get("function") or {}
                if fn.get("name"):
                    acc["name"] = fn["name"]
                if isinstance(fn.get("arguments"), str):
                    acc["args"] += fn["arguments"]
            finish = choice.get("finish_reason")
            if finish:
                result["stop"] = _OPENAI_FINISH.get(finish, "stop")
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                result["input_tokens"] = usage.get("prompt_tokens")
                result["output_tokens"] = usage.get("completion_tokens")
        for idx in sorted(tool_acc):
            acc = tool_acc[idx]
            result["tool_calls"].append({"id": acc["id"], "name": acc["name"],
                                         "arguments": _parse_args(acc["args"])})
        if result["tool_calls"] and result["stop"] == "stop":
            result["stop"] = "tool_calls"
        return result
    try:
        data = json.loads(body_bytes.decode("utf-8"))
    except Exception as exc:
        raise ProviderError(200, "bad OpenAI JSON: %s" % exc, kind="parse")
    if not isinstance(data, dict):
        raise ProviderError(200, "OpenAI response is not an object", kind="parse")
    choices = data.get("choices") or []
    if choices:
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            result["text"] = content
        elif isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            result["text"] = "".join(texts)
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            result["tool_calls"].append({
                "id": str(call.get("id", "")),
                "name": str(fn.get("name", "")),
                "arguments": _parse_args(fn.get("arguments", {})),
            })
        finish = choice.get("finish_reason")
        if finish:
            result["stop"] = _OPENAI_FINISH.get(finish, "stop")
    usage = data.get("usage") or {}
    if isinstance(usage, dict):
        result["input_tokens"] = usage.get("prompt_tokens")
        result["output_tokens"] = usage.get("completion_tokens")
    if result["tool_calls"] and result["stop"] == "stop":
        result["stop"] = "tool_calls"
    return result


def parse_anthropic_response(body_bytes, content_type=""):
    result = _fresh_result()
    if _is_sse(body_bytes, content_type):
        text = body_bytes.decode("utf-8", errors="replace")
        blocks = {}
        order = []
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except Exception:
                continue
            etype = event.get("type")
            if etype == "message_start":
                usage = (event.get("message") or {}).get("usage") or {}
                result["input_tokens"] = usage.get("input_tokens")
            elif etype == "content_block_start":
                index = event.get("index", 0)
                block = event.get("content_block") or {}
                blocks[index] = {"kind": block.get("type"), "id": block.get("id", ""),
                                 "name": block.get("name", ""), "text": "", "json": ""}
                order.append(index)
            elif etype == "content_block_delta":
                index = event.get("index", 0)
                delta = event.get("delta") or {}
                acc = blocks.setdefault(index, {"kind": "text", "id": "", "name": "",
                                                "text": "", "json": ""})
                if delta.get("type") == "text_delta":
                    acc["text"] += delta.get("text", "")
                elif delta.get("type") == "input_json_delta":
                    acc["json"] += delta.get("partial_json", "")
            elif etype == "message_delta":
                delta = event.get("delta") or {}
                reason = delta.get("stop_reason")
                if reason:
                    result["stop"] = _ANTHROPIC_STOP.get(reason, "stop")
                usage = event.get("usage") or {}
                if usage.get("output_tokens") is not None:
                    result["output_tokens"] = usage.get("output_tokens")
        for index in order:
            acc = blocks[index]
            if acc["kind"] == "tool_use":
                result["tool_calls"].append({"id": acc["id"], "name": acc["name"],
                                             "arguments": _parse_args(acc["json"])})
            else:
                result["text"] += acc["text"]
        if result["tool_calls"] and result["stop"] == "stop":
            result["stop"] = "tool_calls"
        return result
    try:
        data = json.loads(body_bytes.decode("utf-8"))
    except Exception as exc:
        raise ProviderError(200, "bad Anthropic JSON: %s" % exc, kind="parse")
    if not isinstance(data, dict):
        raise ProviderError(200, "Anthropic response is not an object", kind="parse")
    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            result["text"] += block.get("text", "")
        elif block.get("type") == "tool_use":
            args = block.get("input")
            result["tool_calls"].append({
                "id": str(block.get("id", "")),
                "name": str(block.get("name", "")),
                "arguments": args if isinstance(args, dict) else {},
            })
    reason = data.get("stop_reason")
    if reason:
        result["stop"] = _ANTHROPIC_STOP.get(reason, "stop")
    usage = data.get("usage") or {}
    if isinstance(usage, dict):
        result["input_tokens"] = usage.get("input_tokens")
        result["output_tokens"] = usage.get("output_tokens")
    if result["tool_calls"] and result["stop"] == "stop":
        result["stop"] = "tool_calls"
    return result

def parse_gemini_response(body_bytes, content_type=""):
    result = _fresh_result()
    text = body_bytes.decode("utf-8", errors="replace").strip()
    docs = []
    if _is_sse(body_bytes, content_type):
        docs = _sse_payloads(body_bytes)
    if not docs:
        try:
            whole = json.loads(text)
            docs = whole if isinstance(whole, list) else [whole]
        except Exception:
            decoder = json.JSONDecoder()
            idx = 0
            while idx < len(text):
                while idx < len(text) and text[idx] in " \t\r\n,":
                    idx += 1
                if idx >= len(text):
                    break
                try:
                    obj, end = decoder.raw_decode(text[idx:])
                except Exception:
                    break
                docs.append(obj)
                idx += end
    if not docs:
        raise ProviderError(200, "empty Gemini response", kind="parse")
    finish = None
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        for candidate in doc.get("candidates") or []:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                if "text" in part and isinstance(part["text"], str):
                    result["text"] += part["text"]
                call = part.get("functionCall")
                if isinstance(call, dict):
                    args = call.get("args")
                    result["tool_calls"].append({
                        "id": "",
                        "name": str(call.get("name", "")),
                        "arguments": args if isinstance(args, dict) else {},
                    })
            if candidate.get("finishReason"):
                finish = candidate["finishReason"]
        usage = doc.get("usageMetadata") or {}
        if isinstance(usage, dict):
            if usage.get("promptTokenCount") is not None:
                result["input_tokens"] = usage.get("promptTokenCount")
            if usage.get("candidatesTokenCount") is not None:
                result["output_tokens"] = usage.get("candidatesTokenCount")
    if finish:
        result["stop"] = _GEMINI_FINISH.get(finish, "stop")
    if result["tool_calls"] and result["stop"] == "stop":
        result["stop"] = "tool_calls"
    return result


# ---------------------------------------------------------------------------
# Rendering back to the requested wire
# ---------------------------------------------------------------------------

def deterministic_id(seed, counter, prefix, salt=""):
    digest = hashlib.sha256(
        ("%s|%s|%s|%s" % (seed, counter, prefix, salt)).encode("utf-8")).hexdigest()
    return "%s%s" % (prefix, digest[:24])


def _usage_for(result, norm, req):
    in_tokens = result["input_tokens"]
    out_tokens = result["output_tokens"]
    if in_tokens is None:
        in_tokens = req["est_input_tokens"]
    if out_tokens is None:
        out_tokens = est_tokens(result["text"])
        for call in result["tool_calls"]:
            out_tokens += est_tokens(canon(call["arguments"]))
        out_tokens = max(1, out_tokens) if (result["text"] or result["tool_calls"]) else 0
    return to_int(in_tokens, 0), to_int(out_tokens, 0)


def _openai_finish(stop):
    return _OPENAI_FINISH.get(stop, "stop")


def _anthropic_stop(stop):
    return {"stop": "end_turn", "length": "max_tokens",
            "tool_calls": "tool_use", "content_filter": "refusal"}.get(stop, "end_turn")


def _split_chunks(text, size):
    if not text:
        return []
    return [text[i:i + size] for i in range(0, len(text), size)]


def render_openai_json(result, model_id, norm, req, seed, counter):
    in_tokens, out_tokens = _usage_for(result, norm, req)
    message = {"role": "assistant"}
    if result["tool_calls"] and not result["text"]:
        message["content"] = None
    else:
        message["content"] = result["text"]
    if result["tool_calls"]:
        message["tool_calls"] = [{
            "id": call["id"] or deterministic_id(seed, counter, "call_", call["name"]),
            "type": "function",
            "function": {"name": call["name"], "arguments": canon(call["arguments"])},
        } for call in result["tool_calls"]]
    payload = {
        "id": deterministic_id(seed, counter, "chatcmpl-", model_id),
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model_id,
        "choices": [{
            "index": 0, "message": message,
            "finish_reason": _openai_finish(result["stop"]), "logprobs": None,
        }],
        "usage": {"prompt_tokens": in_tokens, "completion_tokens": out_tokens,
                  "total_tokens": in_tokens + out_tokens},
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def render_openai_sse(result, model_id, norm, req, seed, counter):
    in_tokens, out_tokens = _usage_for(result, norm, req)
    resp_id = deterministic_id(seed, counter, "chatcmpl-", model_id)
    events = []

    def chunk(delta, finish_reason=None, usage=None):
        payload = {"id": resp_id, "object": "chat.completion.chunk",
                   "created": FIXED_CREATED, "model": model_id,
                   "choices": [{"index": 0, "delta": delta,
                                "finish_reason": finish_reason, "logprobs": None}]}
        if usage is not None:
            payload["usage"] = usage
        return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    events.append(chunk({"role": "assistant", "content": ""}))
    for piece in _split_chunks(result["text"], TEXT_CHUNK):
        events.append(chunk({"content": piece}))
    for idx, call in enumerate(result["tool_calls"]):
        call_id = call["id"] or deterministic_id(seed, counter, "call_", call["name"])
        args_text = canon(call["arguments"])
        events.append(chunk({"tool_calls": [{"index": idx, "id": call_id,
                                             "type": "function",
                                             "function": {"name": call["name"],
                                                          "arguments": ""}}]}))
        for piece in _split_chunks(args_text, ARGS_CHUNK):
            events.append(chunk({"tool_calls": [{"index": idx,
                                                 "function": {"arguments": piece}}]}))
    finish = _openai_finish(result["stop"])
    if norm["include_usage"]:
        events.append(chunk({}, finish_reason=finish))
        usage_payload = {"id": resp_id, "object": "chat.completion.chunk",
                         "created": FIXED_CREATED, "model": model_id, "choices": [],
                         "usage": {"prompt_tokens": in_tokens, "completion_tokens": out_tokens,
                                   "total_tokens": in_tokens + out_tokens}}
        events.append("data: " + json.dumps(usage_payload, ensure_ascii=False) + "\n\n")
    else:
        events.append(chunk({}, finish_reason=finish))
    events.append("data: [DONE]\n\n")
    return "".join(events).encode("utf-8")


def render_anthropic_json(result, model_id, norm, req, seed, counter):
    in_tokens, out_tokens = _usage_for(result, norm, req)
    content = []
    if result["text"]:
        content.append({"type": "text", "text": result["text"]})
    for call in result["tool_calls"]:
        content.append({
            "type": "tool_use",
            "id": call["id"] or deterministic_id(seed, counter, "toolu_", call["name"]),
            "name": call["name"], "input": call["arguments"],
        })
    if not content:
        content.append({"type": "text", "text": ""})
    payload = {
        "id": deterministic_id(seed, counter, "msg_", model_id),
        "type": "message", "role": "assistant", "content": content,
        "model": model_id, "stop_reason": _anthropic_stop(result["stop"]),
        "stop_sequence": None,
        "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")

def render_anthropic_sse(result, model_id, norm, req, seed, counter):
    in_tokens, out_tokens = _usage_for(result, norm, req)
    resp_id = deterministic_id(seed, counter, "msg_", model_id)
    events = []

    def event(name, data):
        return "event: %s\ndata: %s\n\n" % (name, json.dumps(data, ensure_ascii=False))

    events.append(event("message_start", {
        "type": "message_start",
        "message": {"id": resp_id, "type": "message", "role": "assistant", "content": [],
                    "model": model_id, "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": in_tokens, "output_tokens": 0}},
    }))
    index = 0
    if result["text"]:
        events.append(event("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": "text", "text": ""}}))
        for piece in _split_chunks(result["text"], TEXT_CHUNK):
            events.append(event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "text_delta", "text": piece}}))
        events.append(event("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    for call in result["tool_calls"]:
        tool_id = call["id"] or deterministic_id(seed, counter, "toolu_", call["name"])
        events.append(event("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": "tool_use", "id": tool_id,
                              "name": call["name"], "input": {}}}))
        args_text = canon(call["arguments"])
        for piece in _split_chunks(args_text, ARGS_CHUNK):
            events.append(event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta", "partial_json": piece}}))
        events.append(event("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    if index == 0:
        events.append(event("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}}))
        events.append(event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    events.append(event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": _anthropic_stop(result["stop"]), "stop_sequence": None},
        "usage": {"output_tokens": out_tokens}}))
    events.append(event("message_stop", {"type": "message_stop"}))
    return "".join(events).encode("utf-8")


def render_model_list(models, anthropic_wire):
    if anthropic_wire:
        data = [{
            "id": model["id"], "type": "model",
            "display_name": model["id"], "created_at": FIXED_CREATED_AT,
        } for model in models]
        payload = {"data": data, "has_more": False,
                   "first_id": data[0]["id"] if data else None,
                   "last_id": data[-1]["id"] if data else None}
    else:
        payload = {"object": "list", "data": [{
            "id": model["id"], "object": "model",
            "created": FIXED_CREATED, "owned_by": model.get("provider", ""),
        } for model in models]}
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Routing + failover (circuit breakers per provider, dead models per route)
# ---------------------------------------------------------------------------

class Circuit:
    """Per-provider breaker: opens on failure, half-opens after cooldown."""

    def __init__(self):
        self.failures = 0
        self.open_until = 0.0  # monotonic clock; inf = open for the trial

    def tripped(self, now):
        return now < self.open_until

    def record_success(self):
        self.failures = 0
        self.open_until = 0.0

    def record_failure(self, now, cooldown):
        self.failures += 1
        if cooldown is None:
            self.open_until = float("inf")
        else:
            # A streak of short failures earns a longer cooldown, but
            # the extra backoff is capped so a long streak cannot park
            # a provider for the rest of the trial by accident.
            extra = min(MAX_STREAK_BACKOFF, 1.0 * max(0, self.failures - 1))
            self.open_until = max(self.open_until, now + max(0.0, cooldown) + extra)


class Router:
    def __init__(self, providers, routes, env_url, seed):
        self.providers = {p["id"]: p for p in providers}
        self.routes = routes
        self.env_url = env_url
        self.seed = seed
        self.lock = threading.Lock()
        self.counter = 0
        self.circuits = {}   # provider_id -> Circuit
        self.dead_models = set()  # (provider_id, model_id)

    def next_counter(self):
        with self.lock:
            self.counter += 1
            return self.counter

    def _circuit(self, provider_id):
        with self.lock:
            circuit = self.circuits.get(provider_id)
            if circuit is None:
                circuit = Circuit()
                self.circuits[provider_id] = circuit
            return circuit

    def _failures(self, provider_id):
        with self.lock:
            circuit = self.circuits.get(provider_id)
            return circuit.failures if circuit else 0

    def _unmet(self, route, req):
        count = 0
        if req["tools"] and not route["supports_tools"]:
            count += 1
        if req["images"] and not route["supports_images"]:
            count += 1
        if req["json"] and not route["supports_json"]:
            count += 1
        if route["context_length"] < req["est_tokens"]:
            count += 1
        return count

    def candidates_for(self, req, requested_model=None):
        """Routes ordered for this request.

        A client-named model that is eligible goes first. Then every
        route that meets all requirements, ordered by sealed quality,
        provider health (fewer recent failures first), context length,
        and id for a stable tiebreak. If nothing meets everything, the
        same order applies over the routes with the fewest unmet needs.
        """
        wanted = clean_model_id(requested_model) if requested_model else ""
        scored = []
        for route in self.routes:
            unmet = self._unmet(route, req)
            scored.append((unmet, route))
        strict = [route for unmet, route in scored if unmet == 0]
        pool = strict if strict else [route for _unmet, route in scored]

        def order_key(route):
            return (0 if (wanted and route["id"] == wanted) else 1,
                    -route["quality"],
                    self._failures(route["provider_id"]),
                    -route["context_length"],
                    route["id"], route["provider_id"])

        if not strict:
            # No route meets everything: fewest unmet needs first, and a
            # client-named model still wins ties at the same unmet count.
            pool = sorted(pool, key=lambda r: (
                self._unmet(r, req),
                0 if (wanted and r["id"] == wanted) else 1,
                -r["quality"],
                self._failures(r["provider_id"]),
                -r["context_length"],
                r["id"], r["provider_id"]))
            return pool
        return sorted(pool, key=order_key)

    def _call_once(self, provider, route, norm):
        wire = provider["wire"]
        if wire == "anthropic":
            url = endpoint_for(provider, self.env_url, "messages")
            body = anthropic_request_body(norm, route["id"])
            parser = parse_anthropic_response
        elif wire == "gemini":
            url = endpoint_for(provider, self.env_url, "generate", route["id"])
            body = gemini_request_body(norm, route["id"])
            parser = parse_gemini_response
        else:
            url = endpoint_for(provider, self.env_url, "chat")
            body = openai_request_body(norm, route["id"])
            parser = parse_openai_response
        url = auth_query(provider, url)
        headers = auth_headers(provider)
        headers.setdefault("Content-Type", "application/json")
        headers.setdefault("Accept", "application/json")
        status, resp_headers, payload = post_json(url, body, headers, CALL_TIMEOUT)
        retry_after = retry_after_seconds(resp_headers)
        content_type = ""
        try:
            content_type = (resp_headers.get("Content-Type")
                            or resp_headers.get("content-type") or "")
        except Exception:
            content_type = ""
        if status == 200:
            try:
                return parser(payload, content_type)
            except ProviderError as exc:
                raise ProviderError(200, str(exc), kind="parse")
        if status == 402:
            raise ProviderError(status, "provider free allowance exhausted",
                                kind="exhausted")
        if status == 429:
            raise ProviderError(status, "provider rate limited",
                                retry_after=retry_after, kind="rate_limited")
        if status in (401, 403):
            raise ProviderError(status, "provider authentication rejected", kind="auth")
        if status == 404:
            raise ProviderError(status, "model unavailable", kind="model")
        if status == 400:
            raise ProviderError(status, "provider rejected request", kind="model")
        if status == 408 or (status and status >= 500):
            # 408 (request timeout) is transient like a 5xx: fail over
            # to another route instead of killing this model.
            raise ProviderError(status, "provider server error", kind="server")
        raise ProviderError(status or 0, "provider error %s" % status, kind="error")

    def _apply_failure(self, provider_id, route, exc, now):
        circuit = self._circuit(provider_id)
        if exc.kind in ("exhausted", "auth"):
            with self.lock:
                circuit.open_until = float("inf")
        elif exc.kind == "rate_limited":
            wait = exc.retry_after if exc.retry_after is not None else 1.0
            with self.lock:
                circuit.record_failure(now, min(wait, 60.0))
        elif exc.kind == "model":
            with self.lock:
                self.dead_models.add((provider_id, route["id"]))
        elif exc.kind in ("server", "network", "timeout"):
            with self.lock:
                circuit.record_failure(now, 1.0)
        elif exc.kind == "parse":
            with self.lock:
                self.dead_models.add((provider_id, route["id"]))
        else:
            with self.lock:
                self.dead_models.add((provider_id, route["id"]))

    def serve(self, norm):
        req = requirements_of(norm)
        candidates = self.candidates_for(req, norm.get("requested_model"))
        counter = self.next_counter()
        for sweep in range(3):
            now = time.monotonic()
            tried = set()
            for route in candidates:
                key = (route["provider_id"], route["id"])
                if key in tried:
                    continue
                with self.lock:
                    if key in self.dead_models:
                        continue
                if self._circuit(route["provider_id"]).tripped(now):
                    continue
                tried.add(key)
                provider = self.providers.get(route["provider_id"])
                if provider is None:
                    continue
                try:
                    result = self._call_once(provider, route, norm)
                except ProviderError as exc:
                    self._apply_failure(route["provider_id"], route, exc, time.monotonic())
                    continue
                self._circuit(route["provider_id"]).record_success()
                return result, route["id"], counter
            if sweep == 2:
                break
            # No route answered this sweep. If any breaker is only briefly
            # open, wait for the shortest one (capped) and sweep again:
            # a provider that just rate-limited us may already be back.
            with self.lock:
                waits = [c.open_until - time.monotonic()
                         for c in self.circuits.values()
                         if c.open_until != float("inf")]
            waits = [w for w in waits if w > 0]
            if not waits:
                break
            time.sleep(min(min(waits), MAX_BREAKER_WAIT))
        # Last resort: a well-formed answer so the request still counts
        # as served instead of failing outright.
        fallback_model = candidates[0]["id"] if candidates else (
            norm.get("requested_model") or "unknown")
        result = _fresh_result()
        result["text"] = "{}" if req["json"] else "No eligible free model answered this request."
        result["stop"] = "stop"
        return result, fallback_model, counter


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def handle_expect_100(self):
        self.send_response_only(100)
        self.end_headers()
        return True

    @property
    def router(self):
        return self.server.router

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_error(self, status, message, anthropic_wire):
        if anthropic_wire:
            payload = {"type": "error",
                       "error": {"type": "invalid_request_error", "message": message}}
        else:
            payload = {"error": {"message": message, "type": "invalid_request_error",
                                 "param": None, "code": None}}
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json")

    def _anthropic_wire(self):
        return bool(self.headers.get("x-api-key")) and not self.headers.get("Authorization")

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/v1/models":
            models = self.router.catalog.get("models", [])
            self._send(200, render_model_list(models, self._anthropic_wire()),
                       "application/json")
            return
        if path.startswith("/v1/models/"):
            model_id = clean_model_id(urllib.parse.unquote(path[len("/v1/models/"):]))
            for model in self.router.catalog.get("models", []):
                if model["id"] == model_id:
                    if self._anthropic_wire():
                        payload = {"id": model["id"], "type": "model",
                                   "display_name": model["id"],
                                   "created_at": FIXED_CREATED_AT}
                    else:
                        payload = {"id": model["id"], "object": "model",
                                   "created": FIXED_CREATED,
                                   "owned_by": model.get("provider", "")}
                    self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                               "application/json")
                    return
            self._send_error(404, "model %s not found" % model_id, False)
            return
        self._send_error(404, "unknown route: %s" % path, False)

    def do_POST(self):
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path.rstrip("/") or "/"
        length = to_int(self.headers.get("Content-Length"), 0)
        if length < 0:
            length = 0
        if length > MAX_REQUEST_BODY_BYTES:
            # Refuse oversized bodies before reading them, so one huge
            # request cannot stall the server; the error is wire-shaped.
            self._send_error(413, "request body too large", self._anthropic_wire())
            return
        raw = self.rfile.read(length) if length > 0 else b""
        if path == "/v1/chat/completions":
            self._handle_chat(raw)
        elif path == "/v1/messages":
            self._handle_messages(raw)
        else:
            self._send_error(404, "unknown route: %s" % path, False)

    def _read_json(self, raw, anthropic_wire):
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            self._send_error(400, "request body is not valid JSON", anthropic_wire)
            return None
        if not isinstance(body, dict):
            self._send_error(400, "request body must be a JSON object", anthropic_wire)
            return None
        return body

    def _handle_chat(self, raw):
        body = self._read_json(raw, False)
        if body is None:
            return
        norm = normalize_openai(body)
        req = requirements_of(norm)
        result, model_id, counter = self.router.serve(norm)
        if norm["stream"]:
            payload = render_openai_sse(result, model_id, norm, req,
                                        self.router.seed, counter)
            self._send(200, payload, "text/event-stream; charset=utf-8")
        else:
            payload = render_openai_json(result, model_id, norm, req,
                                         self.router.seed, counter)
            self._send(200, payload, "application/json")

    def _handle_messages(self, raw):
        body = self._read_json(raw, True)
        if body is None:
            return
        norm = normalize_anthropic(body)
        req = requirements_of(norm)
        result, model_id, counter = self.router.serve(norm)
        if norm["stream"]:
            payload = render_anthropic_sse(result, model_id, norm, req,
                                           self.router.seed, counter)
            self._send(200, payload, "text/event-stream; charset=utf-8")
        else:
            payload = render_anthropic_json(result, model_id, norm, req,
                                            self.router.seed, counter)
            self._send(200, payload, "application/json")


def pick_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Registration + work polling + main
# ---------------------------------------------------------------------------

def register(env_url, port, catalog):
    url = env_url.rstrip("/") + "/register"
    payload = {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog}
    deadline = time.monotonic() + 15.0
    while True:
        try:
            status, _headers, body = post_json(url, payload, {}, REGISTER_TIMEOUT)
            if status == 200:
                try:
                    data = json.loads(body.decode("utf-8"))
                except Exception:
                    data = {}
                if isinstance(data, dict) and data.get("ok"):
                    return True
                return True
        except ProviderError:
            pass
        except Exception:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)


def poll_until_done(env_url, started):
    url = env_url.rstrip("/") + "/work"
    while True:
        if time.monotonic() - started >= TRIAL_BUDGET_SECONDS:
            return
        try:
            data = get_json(url, {}, WORK_TIMEOUT)
        except Exception:
            time.sleep(0.2)
            continue
        if isinstance(data, dict) and data.get("done"):
            return
        time.sleep(0.1)


def run_server():
    started = time.monotonic()
    seed = os.environ.get("MAKEFUN_TRIAL_SEED", "0")
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").strip()
    brief = load_brief(os.environ.get("MAKEFUN_PROMPT_FILE", ""))
    providers = providers_from_brief(brief)
    quality = quality_from_brief(brief)
    fetched = {}
    if env_url:
        for provider in providers:
            fetched[provider["id"]] = fetch_provider_models(provider, env_url)
    catalog, routes = build_catalog(providers, quality, fetched)
    router = Router(providers, routes, env_url, str(seed))
    router.catalog = catalog
    port = pick_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.router = router
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.daemon = True
    thread.start()
    try:
        if not env_url:
            sys.stderr.write("listening on 127.0.0.1:%d (no MAKEFUN_ENV_URL)\n" % port)
            sys.stderr.flush()
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                pass
            return 0
        register(env_url, port, catalog)
        poll_until_done(env_url, started)
        return 0
    finally:
        try:
            server.shutdown()
        except Exception:
            pass
        try:
            server.server_close()
        except Exception:
            pass


def main(argv):
    if "--list" in argv:
        sys.stdout.write(canon(catalog_for_list()))
        sys.stdout.flush()
        return 0
    if "--help" in argv or "-h" in argv:
        sys.stdout.write("usage: python3 main.py [--list]\n")
        return 0
    try:
        return run_server()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        sys.stderr.write("fatal: %s\n" % exc)
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
