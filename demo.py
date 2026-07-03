"""Demo: the same question to every LLM you have a key for, through ONE
gateway with ONE key -- then a provider "dies" and the gateway recovers.

Run `python gateway.py` in another terminal first, then `python demo.py`.
"""
import requests
from gateway import ALIASES, MODELS, PROVIDER_KEYS, key

URL = "http://localhost:8000/v1/chat/completions"
QUESTION = "In one short sentence: why is the sky blue?"

def ask(model):
    """One OpenAI-style call to the gateway. Returns (model_used, text, cost)."""
    resp = requests.post(
        URL,
        headers={"Authorization": f"Bearer {key('MASTER_KEY')}"},
        json={"model": model, "max_tokens": 500,
              "messages": [{"role": "user", "content": QUESTION}]},
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    used = data["model"]  # who really answered (may differ after failover)
    text = data["choices"][0]["message"]["content"].strip().splitlines()[0]
    _, price_in, price_out = MODELS[used]
    usage = data["usage"]
    cost = (usage["prompt_tokens"] * price_in
            + usage["completion_tokens"] * price_out) / 1_000_000
    return used, text, cost

have_key = [m for m, (prov, _, _) in MODELS.items() if key(PROVIDER_KEYS[prov])]
missing = [m for m in MODELS if m not in have_key]

print(f'Q: "{QUESTION}"\n')
for model in have_key:
    used, text, cost = ask(model)
    print(f"  {used:24s} ${cost:.6f}   {text[:70]}")

if missing:
    dead = missing[0]
    print(f"\n--- failover: asking for {dead}, whose provider has no key ---")
    used, text, cost = ask(dead)
    print(f"  {dead} is down... rescued by {used} (${cost:.6f})")
    print(f"  {text[:70]}")
else:
    print("\n(all providers have keys -- break one in .env to see failover)")
