# nano-llm-gateway: one key, every LLM

A **~300-line readable Python file** that does what LLM gateways like LiteLLM and OpenRouter do: your app uses one master key, and the gateway routes to Claude, GPT, Gemini, or DeepSeek, with automatic failover and live cost tracking. Built to be **read**, not deployed.

![demo](assets/demo.gif)

Every serious AI app sits behind a gateway, but the popular ones are huge codebases: great tools, hard to learn from. This is the whole idea in one file you can read in one sitting: how the request formats get translated, how failover works, and where your money goes on every call.

## Quickstart

```bash
git clone https://github.com/reezanahamed/nano-llm-gateway && cd nano-llm-gateway
pip install -r requirements.txt        # just `requests`
cp .env.example .env                   # add the API keys you have (any 1+ works)
python gateway.py                      # gateway runs at http://localhost:8000, keep it running
```

Then in a second terminal:

```bash
python demo.py                         # same question → Claude, GPT & Gemini, side by side
```

## What it does

| Feature | What you get |
|---|---|
| One key for everything | Your apps use one master key; real provider keys stay in `.env` |
| Speaks OpenAI | Any OpenAI SDK works. Change `base_url`, keep your code |
| Auto-failover | A provider errors → the next one answers, automatically |
| Live cost tracking | Every request logs tokens used and cost in $ |
| Easy to extend | DeepSeek took ~5 lines to add. Any OpenAI-compatible provider is the same |
| Actually readable | ~300 lines, one file, one dependency |

Point any OpenAI client at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-master-key")
client.chat.completions.create(model="claude-haiku", messages=[...])  # yes, Claude via the OpenAI SDK
```

## How it works

A gateway is a **translator plus a switchboard**. Claude, GPT, and Gemini all do the same job but speak different dialects: different URLs, auth headers, and JSON shapes. The gateway accepts one common format (OpenAI's, the de-facto standard), looks at the model name to pick a provider, translates the request into that provider's dialect, and translates the answer back. If the call fails, it moves down a fallback list and tries the next provider. Since every provider reports token usage in its response, the gateway multiplies tokens by each model's price and logs the cost of every call.

That's the entire trick. The production gateways add scale, streaming, and guardrails on top, but this is the core. Read `gateway.py` top to bottom; the section comments walk you through it.

## Limitations (honest ones)

- No streaming responses
- No rate limiting, retries, or queueing
- Price table is hardcoded. Prices change, so check before trusting the numbers
- Single process, nothing persisted
- **For understanding, not production.** In production, use [LiteLLM](https://github.com/BerriAI/litellm).

## License

MIT
