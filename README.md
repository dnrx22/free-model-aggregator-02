# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/dnrx22/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/dnrx22/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One stdlib-only server merging free-model catalogs, routing each request to the best eligible model, failing over on refusal.

## What it does

- Reads the trial brief from `MAKEFUN_PROMPT_FILE`: providers (wire, auth,
  base URL, key), a sealed quality table (rating, context length, tool /
  image / JSON support).
- Fetches each provider's model list and registers one merged catalog
  with the trial environment (`POST $MAKEFUN_ENV_URL/register`).
- Serves `POST /v1/chat/completions` (OpenAI wire, JSON + SSE) and
  `POST /v1/messages` (Anthropic wire, JSON + SSE), plus
  `GET /v1/models` in the wire the auth header names
  (`Authorization: Bearer` -> OpenAI list, `x-api-key` -> Anthropic list).
- Routes by score: a request's needs (tools, images, JSON mode, estimated
  context) are counted against every (model, provider) route; routes that
  meet everything rank by sealed quality, then by provider health
  (recent failures push a provider down), then by context length. A
  client-named model that is eligible goes first. Translates between
  OpenAI, Anthropic, and Gemini provider wires as needed.
- Fails over with a per-provider circuit breaker: 402 or auth failure
  opens it for the trial, 429 honors Retry-After, 5xx/timeout opens it
  briefly, 400/404/parse failure kills just that model. After a failed
  sweep it waits for the shortest live breaker (capped) and sweeps again.
  Every model call goes to a provider under `MAKEFUN_ENV_URL`, never
  anywhere else.
- Polls `GET $MAKEFUN_ENV_URL/work` until the work is done, then exits 0.

## Run it

```sh
python3 main.py
```

The sandbox sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`, and
`MAKEFUN_TRIAL_SEED`. Python 3.13, standard library only, no installs.

Print the catalog document the server registers, byte for byte:

```sh
python3 main.py --list
```

The server is deterministic: no wall clock in any decision and no
unseeded randomness. Response ids and streaming chunk boundaries derive
from `MAKEFUN_TRIAL_SEED` plus a per-request counter.

---

<sub>Entered on [make.fun](https://make.fun/?ref=0drwcpsw&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
