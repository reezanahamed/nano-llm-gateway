"""nano-llm-gateway: one key, every LLM.

An LLM gateway is a translator plus a switchboard. Your app speaks ONE dialect
(OpenAI's, the de-facto standard) to ONE endpoint with ONE key. The gateway
looks at the model name, picks the right provider, translates the request into
that provider's dialect, calls it with the real key, and translates the answer
back. If the provider fails, it tries the next one. That's the whole idea --
LiteLLM and OpenRouter add scale and features on top, but this is the core.

Read top to bottom. Four sections: CONFIG -> ADAPTERS -> ROUTER -> SERVER.
"""
import json
import os
import time
import uuid

import requests  # the one dependency: plain HTTP calls, nothing hidden

# ============================================================================
# 1. CONFIG -- keys, routing table, price table
# ============================================================================

def load_env(path=".env"):
    """A .env file is just KEY=VALUE lines. Ten lines replace python-dotenv."""
    env = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
    return env

_env = load_env()
def key(name):
    """Real environment variables win over .env, so `export X=...` works too."""
    return os.environ.get(name) or _env.get(name, "")

# The routing + price table. This is the gateway's entire "knowledge":
# which provider serves each model, and what it costs (USD per 1M tokens).
# Prices checked 2026-07-03 -- they change, so re-check before trusting them.
MODELS = {
    # model id              provider     $/1M in  $/1M out
    "gpt-5.4-nano":        ("openai",     0.20,    1.25),
    "gpt-5.4-mini":        ("openai",     0.75,    4.50),
    "claude-haiku-4-5":    ("anthropic",  1.00,    5.00),
    "gemini-2.5-flash-lite": ("google",   0.10,    0.40),
    "deepseek-v4-flash":   ("deepseek",   0.14,    0.28),
}

# Friendly names, so users can say "claude-haiku" without memorizing versions.
ALIASES = {
    "gpt-nano": "gpt-5.4-nano",
    "gpt-mini": "gpt-5.4-mini",
    "claude-haiku": "claude-haiku-4-5",
    "gemini-flash-lite": "gemini-2.5-flash-lite",
    "deepseek": "deepseek-v4-flash",
}

# Which .env key unlocks each provider. No key = provider silently skipped.
PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# ============================================================================
# 2. PROVIDER ADAPTERS -- one function per dialect
# ============================================================================
# Every adapter has the same contract: take OpenAI-style (model, messages,
# max_tokens), speak the provider's native HTTP dialect, and return the
# normalized triple (text, input_tokens, output_tokens). Token counts come
# from the provider's own usage field -- every provider reports them, because
# that's what they bill you on. The normalization is what makes the router
# and cost tracker provider-agnostic.

def call_openai(model, messages, max_tokens, base_url="https://api.openai.com",
                key_name="OPENAI_API_KEY", tokens_param="max_completion_tokens"):
    """OpenAI quirk: their newest models renamed `max_tokens` to
    `max_completion_tokens` -- OpenAI broke its own dialect over time, which
    is exactly why gateways exist. Auth is a simple Bearer header. The extra
    parameters let OTHER OpenAI-compatible providers reuse this adapter."""
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {key(key_name)}"},
        json={"model": model, "messages": messages, tokens_param: max_tokens},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data["choices"][0]["message"]["content"],
            data["usage"]["prompt_tokens"],
            data["usage"]["completion_tokens"])

def call_deepseek(model, messages, max_tokens):
    """DeepSeek copied OpenAI's dialect wholesale (most newer providers do --
    that's how OpenAI's format became the industry standard). So adding a
    whole new provider is these five lines: the OpenAI adapter pointed at a
    different address. Only wrinkle: DeepSeek kept the classic `max_tokens`."""
    return call_openai(model, messages, max_tokens,
                       base_url="https://api.deepseek.com",
                       key_name="DEEPSEEK_API_KEY", tokens_param="max_tokens")

def call_anthropic(model, messages, max_tokens):
    """Anthropic quirks: (1) the system prompt is a separate top-level field,
    not a message with role "system"; (2) `max_tokens` is REQUIRED, not
    optional; (3) auth uses an `x-api-key` header plus a version header."""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [m for m in messages if m["role"] != "system"],
    }
    if system:
        body["system"] = system
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key("ANTHROPIC_API_KEY"),
                 "anthropic-version": "2023-06-01"},
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    # The response is a list of typed blocks (text, tool calls...). Join the text.
    text = "".join(b["text"] for b in data["content"] if b["type"] == "text")
    return (text,
            data["usage"]["input_tokens"],
            data["usage"]["output_tokens"])

def call_gemini(model, messages, max_tokens):
    """Gemini speaks a whole different language: messages are `contents`,
    each with a list of `parts`; the assistant role is called "model"; the
    system prompt is `systemInstruction`; and the model name goes in the URL,
    not the body. Same job, third dialect."""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user",
         "parts": [{"text": m["content"]}]}
        for m in messages if m["role"] != "system"
    ]
    body = {"contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key("GEMINI_API_KEY")},
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    text = "".join(p.get("text", "")
                   for p in data["candidates"][0]["content"]["parts"])
    return (text,
            data["usageMetadata"]["promptTokenCount"],
            data["usageMetadata"]["candidatesTokenCount"])

# The switchboard: provider name -> adapter function.
ADAPTERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_gemini,
    "deepseek": call_deepseek,
}

# ============================================================================
# 3. ROUTER -- pick a provider, fail over, count the money
# ============================================================================
# The failover order when a provider errors or has no key. The requested
# model is always tried first; these are the backups, cheapest-ish first.
FALLBACKS = ["gemini-2.5-flash-lite", "deepseek-v4-flash",
             "gpt-5.4-nano", "claude-haiku-4-5"]

def route(model, messages, max_tokens=1024):
    """The gateway's brain. Resolve the model name, then walk the chain:
    requested model first, then each fallback. A provider is skipped if its
    key is missing, and abandoned if its call raises -- either way the next
    one gets a shot. Returns (model_used, text, in_tokens, out_tokens, cost).
    """
    requested = ALIASES.get(model, model)
    if requested not in MODELS:
        raise ValueError(f"unknown model {model!r}; try one of: "
                         + ", ".join(list(MODELS) + list(ALIASES)))
    chain = [requested] + [m for m in FALLBACKS if m != requested]
    tried = []
    for m in chain:
        provider, price_in, price_out = MODELS[m]
        if not key(PROVIDER_KEYS[provider]):
            tried.append(f"{m} (no key)")
            continue
        try:
            text, tok_in, tok_out = ADAPTERS[provider](m, messages, max_tokens)
        except Exception as exc:
            tried.append(f"{m} ({exc})")
            print(f"[gateway] {m} failed -> next in chain ({exc})")
            continue
        # Cost = tokens x price. Providers report exact token counts in every
        # response because that's what they bill on -- we just do the math.
        cost = (tok_in * price_in + tok_out * price_out) / 1_000_000
        failover = f" FAILOVER({requested}->{m})" if m != requested else ""
        print(f"[gateway] {m} via {provider}: {tok_in} in + {tok_out} out"
              f" = ${cost:.6f}{failover}")
        return m, text, tok_in, tok_out, cost
    raise RuntimeError("all providers failed: " + "; ".join(tried))

# ============================================================================
# 4. HTTP SERVER -- an OpenAI-shaped front door
# ============================================================================
# We expose ONE endpoint, POST /v1/chat/completions, and answer in OpenAI's
# exact JSON shape. That shape is the entire trick: any OpenAI SDK will talk
# to us if you just change its base_url. Errors are OpenAI-shaped too, so
# SDK error handling keeps working. Stdlib server -- no framework needed for
# one route. "Threading" means slow LLM calls don't block each other.
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class GatewayHandler(BaseHTTPRequestHandler):

    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        # OpenAI's error envelope, so SDKs raise their normal typed errors.
        self._send(status, {"error": {"message": message,
                                      "type": "invalid_request_error"}})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._error(404, f"no such endpoint: {self.path}")
        # Auth: the client proves it knows the ONE master key. The real
        # provider keys never leave this process -- that's the security win.
        if self.headers.get("Authorization") != f"Bearer {key('MASTER_KEY')}":
            return self._error(401, "invalid or missing master key")
        try:
            req = json.loads(self.rfile.read(
                int(self.headers.get("Content-Length", 0))))
            model, messages = req["model"], req["messages"]
        except (ValueError, KeyError) as exc:
            return self._error(400, f"bad request body: {exc}")
        try:
            used, text, tok_in, tok_out, _cost = route(
                model, messages, req.get("max_tokens", 1024))
        except ValueError as exc:          # unknown model
            return self._error(400, str(exc))
        except RuntimeError as exc:        # every provider failed
            return self._error(502, str(exc))
        # The response, shaped exactly like OpenAI's. `model` tells the truth
        # about who actually answered -- important when failover kicked in.
        self._send(200, {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": used,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": tok_in, "completion_tokens": tok_out,
                      "total_tokens": tok_in + tok_out},
        })

    def log_message(self, *args):
        pass  # silence the default per-request access log; the router logs better

if __name__ == "__main__":
    if not key("MASTER_KEY"):
        raise SystemExit("Set MASTER_KEY in .env first (any string you invent).")
    ready = [p for p, k in PROVIDER_KEYS.items() if key(k)]
    print(f"nano-llm-gateway on http://localhost:8000  "
          f"(providers with keys: {', '.join(ready) or 'NONE'})")
    ThreadingHTTPServer(("127.0.0.1", 8000), GatewayHandler).serve_forever()
