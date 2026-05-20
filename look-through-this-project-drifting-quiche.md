# TradingAgents — Phased Refactor Plan

## Context

This codebase is the *TradingAgents: Multi-Agents LLM Financial Trading Framework* (arXiv 2412.20138, paper PDF in `trash/`). It's a LangGraph orchestration of analyst → researcher → trader → risk → portfolio-manager agents that emits a structured trading recommendation; **it has no order execution layer of any kind** (verified — no `binance`, `ccxt`, or `place_order` anywhere in `tradingagents/`).

You want to evolve it into a **crypto** research + execution system that runs on your **Claude Max subscription** (no API spend) and trades a small Binance subaccount. We're splitting that into three build phases plus a phase 4 for safety + going live.

**Critical constraint you flagged:** the swap must use the **Claude CLI binary** (`claude -p ...`), not the Claude Agent SDK. The SDK still bills the Anthropic API and would not consume your Max subscription.

---

## Phase 1 — Add Claude CLI as an LLM provider

**Goal:** any agent that today calls `llm.invoke()` / `llm.bind_tools()` / `llm.with_structured_output()` should work unchanged when `config["llm_provider"] = "claude_cli"`.

### Approach
Build a thin LangChain `BaseChatModel` adapter that shells out to the `claude` CLI. We do **not** use the Claude Agent SDK (it bills the API).

### Files
- **NEW** `tradingagents/llm_clients/claude_cli_client.py` (~200–300 LOC)
  - `ClaudeCLIClient(BaseLLMClient)` — analogous to `AnthropicClient` in `tradingagents/llm_clients/anthropic_client.py:26`.
  - `ChatClaudeCLI(BaseChatModel)` — the actual LangChain chat-model adapter.
- **EDIT** `tradingagents/llm_clients/factory.py:35` — add a `claude_cli` branch.
- **EDIT** `tradingagents/llm_clients/model_catalog.py` — add a `claude_cli` entry so `validate_model` doesn't warn.
- **EDIT** `tradingagents/default_config.py:5` — document `"claude_cli"` as an option; deep/quick model strings become CLI `--model` values (`opus`, `sonnet`, `haiku`).
- **EDIT** `tradingagents/graph/trading_graph.py:132` — add a `claude_cli` branch in `_get_provider_kwargs` (probably a no-op — no thinking/effort knob equivalent).

### Adapter implementation notes
- Run `claude -p <prompt> --output-format stream-json --model <model>` via `subprocess.Popen`. Parse stream-json line by line — it gives you structured `assistant` / `tool_use` / `tool_result` events instead of a raw text blob.
- For multi-turn agent loops (Bull/Bear debate, analyst tool loop), use `claude --resume <session-id>` to keep one CLI session alive across turns rather than spawning a new process per message.
- Implement the three LangChain hooks the codebase actually depends on:
  1. **`_generate(messages, ...)`** — plain text in, text out. Easy.
  2. **`bind_tools(tools)`** — store the tool list on the wrapper. On each turn, inject the tool schemas into the system prompt as JSON, parse the model's response for tool-call JSON blocks, execute the matching `@tool` locally, append the result, re-prompt. (You're reimplementing what LangChain normally does for you because the CLI doesn't expose the tool-use protocol natively.)
  3. **`with_structured_output(schema)`** — inject the Pydantic schema into the prompt with `"Return ONLY JSON conforming to: …"`, then parse with `langchain_core.utils.json.parse_json_markdown` and validate against the Pydantic model. Used by `agents/trader/trader.py:18`, the Research Manager, and the Portfolio Manager.

### Verification
- `python main.py` with `config["llm_provider"] = "claude_cli"` and `deep_think_llm = "sonnet"` produces a non-empty `PortfolioDecision`.
- Smoke test each capability separately: a tool-using analyst (Market), a structured-output node (Trader), a multi-turn debate (Bull/Bear).
- `top` should show `claude` processes spawning, not network calls to api.anthropic.com.

---

## Phase 2 — Adapt the analytics layer for crypto

**Goal:** `ta.propagate("BTCUSDT", today)` produces a `PortfolioDecision` using crypto-appropriate data.

### What needs to change
| Component | Action |
|---|---|
| Market Analyst (technicals) | Keep as-is — `stockstats` works on any OHLCV |
| News Analyst | Swap data source to a crypto news provider (CryptoPanic free tier or NewsAPI with crypto query) |
| Social Media Analyst | Keep — crypto Twitter/Reddit signal is arguably stronger than for stocks |
| **Fundamentals Analyst** | **Replace.** Crypto has no balance sheet / income / insider tx. Two options: (a) repurpose into an **on-chain analyst** using CoinGecko free tier (market cap, volume, supply, dev activity) and optionally Glassnode for richer metrics; (b) just disable for crypto symbols |
| Bull/Bear, Research Mgr, Trader, Risk, PM | Keep as-is — pure reasoning over reports |

### Files
- **NEW** `tradingagents/dataflows/binance.py` — public REST `/api/v3/klines` (no API key needed for market data) → OHLCV dataframe; reuse `stockstats` for indicators. Mirror the shape of `tradingagents/dataflows/y_finance.py`.
- **NEW** `tradingagents/dataflows/coingecko.py` — on-chain / market metadata for the repurposed fundamentals path.
- **NEW** `tradingagents/dataflows/cryptopanic.py` (or similar) — crypto news.
- **EDIT** `tradingagents/dataflows/interface.py:31` — register `"binance"` under `core_stock_apis` and `technical_indicators`; `"coingecko"` under `fundamental_data`; `"cryptopanic"` under `news_data`.
- **EDIT** `tradingagents/agents/analysts/fundamentals_analyst.py` — either branch on symbol shape (heuristic: ends in `USDT`/`USDC`/`BTC`/`USD`) to pull on-chain instead of equity fundamentals, or wire a separate `crypto_fundamentals_analyst` and select it per asset class.
- **EDIT** `tradingagents/default_config.py:5` — add `"asset_class": "equities" | "crypto"` and a crypto preset for `data_vendors`.

### Notes
- Symbol convention: agents already preserve ticker strings verbatim (`agents/utils/agent_utils.py:37`), so `BTCUSDT` flows through fine.
- Date handling: crypto markets are 24/7, but daily candles still work. If you later want intraday, you'll need to pass an `interval` param down through the data layer.

### Verification
- `python -c "from tradingagents.dataflows.binance import get_crypto_data; print(get_crypto_data('BTCUSDT','2026-04-01','2026-04-30','1d').tail())"` returns a dataframe.
- `ta.propagate("BTCUSDT", "2026-04-30")` runs end-to-end and produces a `PortfolioDecision` referencing on-chain/news context, not equity fundamentals.

---

## Phase 3 — Add a Binance execution layer

**Goal:** turn the `PortfolioDecision` JSON into actual orders on a Binance subaccount. Read-only / signal-only by default; live execution gated behind explicit config + safety clamps (which are themselves Phase 4).

### Files
- **NEW** `tradingagents/execution/__init__.py`
- **NEW** `tradingagents/execution/binance_executor.py`:
  - `BinanceExecutor(api_key, api_secret, testnet=True, dry_run=True, allow_list=("BTCUSDT","ETHUSDT"))`
  - `execute_decision(decision: PortfolioDecision, symbol: str) -> OrderResult`
  - Maps the structured `PortfolioDecision` (already has action / confidence / sizing fields) → a Binance order spec.
  - In `dry_run`, logs the would-be order to `~/.tradingagents/execution/orders.jsonl` and returns a fake `OrderResult` — does NOT call Binance.
  - In live mode, calls `python-binance`'s `client.create_order(...)`.
- **NEW** `tradingagents/execution/order_log.py` — append-only JSONL log of every intended + actual order, with the originating `PortfolioDecision` snapshot. This becomes the substrate for reflection.
- **EDIT** `main.py` — optional `--execute` flag wires the executor in after `propagate()` returns.

### Subaccount setup (manual, one-time, on Binance UI)
1. Sub Accounts → Create Sub Account → fund with the small test amount.
2. API Management on the subaccount → create key with **Spot Trading enabled, Withdrawals disabled, IP-restricted** to your machine.
3. Store as `BINANCE_API_KEY` / `BINANCE_API_SECRET` in `.env` (already loaded by `main.py:7`).

### Verification (Phase 3 only — without safety yet)
- Dry-run end-to-end: `ta.propagate("BTCUSDT", today)` → executor logs an intended order to `orders.jsonl`. No Binance call made.
- Testnet: point executor at `testnet.binance.vision`, place a single tiny order manually, confirm fill round-trip.

---

## Phase 4 — Safety, scheduling, testnet soak, then live

This is the "make it not blow up" phase and is the longest one in calendar time even though it's not the largest in code.

### Safety guards (deterministic, OUTSIDE the LLM)
- **Symbol allow-list** — hard-coded set, e.g. `{"BTCUSDT","ETHUSDT"}` only.
- **Max notional per trade** — e.g. ≤ $20.
- **Max position size** per symbol.
- **Daily loss kill-switch** — once realized PnL crosses a threshold, executor refuses all new orders for the rest of the UTC day; logs and alerts.
- **Rate limit on decisions** — ignore signals if the previous one fired < N minutes ago.
- **Sanity checks on the decision JSON** — reject unknown actions, sizing > max, missing stop-loss, etc.
- All guards live in `binance_executor.py` and run on every order; the LLM cannot see or modify them.

### Scheduling
- Add `tradingagents/runner.py` with a simple loop: every N minutes, for each symbol in the watchlist, run `ta.propagate(symbol, today)`, hand the decision to the executor.
- Run it as a `launchd` job on macOS (or `tmux` + a shell loop for the MVP).

### Testnet soak
- Run the full pipeline against Binance testnet for **at least a week** with the kill-switch + clamps active.
- Review the `orders.jsonl` log + Binance testnet fills daily — look for: oversized orders that the clamps caught, decisions that contradict the analyst reports, decisions that fire too frequently.

### Going live
- Only flip `testnet=False` after testnet soak shows zero clamp violations *and* you're comfortable with the decision quality.
- Start with the smallest possible notional cap.
- Monitor the first 48 hours closely; have the kill-switch threshold set conservatively.

---

## Critical files to know across all phases

- LLM factory (Phase 1): `tradingagents/llm_clients/factory.py:11`
- Anthropic client (template for the CLI client): `tradingagents/llm_clients/anthropic_client.py:26`
- Tool-using analyst (Phase 1 must support this): `tradingagents/agents/analysts/fundamentals_analyst.py`
- Structured-output call site (Phase 1 must support this): `tradingagents/agents/trader/trader.py:18`
- Vendor router (Phase 2 plug-in point): `tradingagents/dataflows/interface.py:119`
- Existing data vendor template (Phase 2): `tradingagents/dataflows/y_finance.py`
- Config (touched in every phase): `tradingagents/default_config.py:5`
- Trading graph wiring (Phase 1): `tradingagents/graph/trading_graph.py:85`
- Entry point: `main.py:27` — confirms framework outputs a decision; execution layer is net-new in Phase 3.

---

## Suggested order of work

1. **Phase 1 first**, in isolation, against existing equity flow (`NVDA`) — confirms the CLI swap doesn't regress anything before we add new variables.
2. **Phase 2 second**, still no execution, end-to-end on `BTCUSDT` in dry analysis mode.
3. **Phase 3 third** in dry-run only — orders go to JSONL log, not Binance.
4. **Phase 4 last** — safety guards, then testnet soak, then live with tiny caps.
