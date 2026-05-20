# TradingAgents — Manual Runbook

How to run the framework by hand for crypto research on your Claude Max
subscription. Two entry points are supported; both produce a structured
`PortfolioDecision` and (optionally) hand it to a dry-run / testnet
Binance executor.

---

## 1. Prerequisites

```bash
# from the repo root
source .venv/bin/activate          # uses the project's Python 3.12 venv
pip install -e .                   # only needed once / after dep changes
which claude && claude --version   # required: local Claude Code CLI ≥ 2.1
```

The `claude` binary is what backs the `claude_cli` provider — it consumes
your Max / Pro subscription. **No API spend.**

---

## 2. Two ways to run

### Option A — Interactive CLI (recommended)

Best when you want a rich live UI and per-team reports auto-saved to disk.

```bash
.venv/bin/tradingagents            # short form (installed script)
# or, equivalent:
.venv/bin/python -m cli.main
```

You'll be walked through 7–8 questions:

| Step | What it asks | What to pick for crypto |
|------|--------------|--------------------------|
| 1 | **Ticker** | `BTCUSDT`, `ETHUSDT`, `BTC`, `ETH-USD`, `SOL/USDT`, etc. The framework auto-detects crypto from the symbol shape and flips data vendors accordingly. |
| 2 | **Analysis date** | `YYYY-MM-DD`. For backtest-style use, pick a past date. Defaults to today. |
| 3 | **Output language** | English by default. Internal debate always stays English regardless. |
| 4 | **Analysts** | Space to toggle, `a` for all. For crypto, the **News + Market + Fundamentals** trio is the most useful baseline; Social media analyst is optional. |
| 5 | **Research depth** | Shallow (1 round) / Medium (3) / Deep (5). Each round multiplies runtime. Start at Shallow. |
| 6 | **LLM provider** | **`Claude CLI (Max/Pro subscription, no API spend)`** — top of the list. |
| 7 | **Quick + Deep models** | `haiku` for quick, `sonnet` or `opus` for deep. Opus is highest quality / slowest. |
| 8 | Provider-specific thinking knobs | Skipped for `claude_cli` (no equivalent). |

Optional flags:

```bash
.venv/bin/tradingagents --checkpoint              # resume on crash
.venv/bin/tradingagents --clear-checkpoints       # force fresh start
```

### Option B — Edit `main.py` and run

Faster to iterate when you're rerunning the same config repeatedly. Open
`main.py`, edit the **Mode 2** block:

```python
config["llm_provider"] = "claude_cli"
config["deep_think_llm"] = "sonnet"        # opus | sonnet
config["quick_think_llm"] = "haiku"
config["max_debate_rounds"] = 1            # bump to 3 or 5 for deeper runs
config["max_risk_discuss_rounds"] = 1
config["data_vendors"] = {
    "core_stock_apis": "binance",
    "technical_indicators": "binance",
    "fundamental_data": "coingecko",
    "news_data": "crypto_news",
}
symbol, trade_date, run_executor = "BTCUSDT", "2026-04-30", False
```

Then:

```bash
.venv/bin/python main.py
```

The final decision prints to stdout. The interactive CLI also saves the
intermediate reports to disk (see §4) — `main.py` does not, by default.

---

## 3. Where outputs land

### Interactive CLI

All artifacts live under `~/.tradingagents/logs/<TICKER>/<DATE>/`:

```
~/.tradingagents/logs/BTCUSDT/2026-04-30/
├── complete_report.md                ← the full readable report
├── message_tool.log                  ← every LLM message + tool call
├── errors.log                        ← warnings + full traceback on crash
├── reports/<section>.md              ← raw per-section saves
├── 1_analysts/
│   ├── market.md
│   ├── news.md
│   └── fundamentals.md
├── 2_research/
│   ├── bull.md
│   ├── bear.md
│   └── manager.md                    ← Research Manager verdict
├── 3_trading/
│   └── trader.md                     ← Trader investment plan
├── 4_risk/
│   ├── aggressive.md
│   ├── conservative.md
│   └── neutral.md
└── 5_portfolio/
    └── decision.md                   ← Final Portfolio Manager decision
```

**Read `complete_report.md` first** — it's the single concatenated
markdown with everything in order.

### `main.py` path

Prints `final_state["final_trade_decision"]` to stdout. Does **not** save
per-section markdown files (the CLI does). Memory log (across runs) still
goes to `~/.tradingagents/memory/trading_memory.md`.

---

## 4. Configuration knobs (reference)

All of these live in `tradingagents/default_config.py` and can be
overridden per run by editing `main.py` or letting the interactive CLI
set them.

| Key | What it does | Common values |
|---|---|---|
| `llm_provider` | Which LLM backend | `claude_cli`, `openai`, `anthropic`, `google`, `xai`, `deepseek`, `qwen`, `glm`, `openrouter`, `azure`, `ollama` |
| `deep_think_llm` | Model for high-stakes nodes (Trader, PM, Risk debate) | `opus`, `sonnet` |
| `quick_think_llm` | Model for fast nodes (Analyst tool loops, signal extraction) | `haiku`, `sonnet` |
| `max_debate_rounds` | Bull/Bear back-and-forth rounds | 1 (fast) – 5 (thorough) |
| `max_risk_discuss_rounds` | Aggressive/Conservative/Neutral rounds | 1 (fast) – 5 |
| `asset_class` | Frames the Fundamentals Analyst's prompt | `auto` (recommended), `crypto`, `equities` |
| `data_vendors` | Per-category data source map | crypto preset above; equity uses `yfinance` |
| `output_language` | Language for final reports | `English` (default), `Russian`, `中文`, ... |
| `checkpoint_enabled` | Save state after each node | `True` for long deep runs you might want to resume |
| `results_dir` | Where reports go | `~/.tradingagents/logs` (set via `TRADINGAGENTS_RESULTS_DIR`) |
| `memory_log_max_entries` | Cap on persistent memory log | `None` (no cap) or e.g. `200` |

### Crypto symbols the data layer recognises

- **Quote suffixes (auto-detected as crypto):** `USDT`, `USDC`, `BUSD`,
  `FDUSD`, `TUSD`, `DAI`, `USD`
- **Bare base hints (auto-detected as crypto):** `BTC`, `ETH`, `SOL`,
  `BNB`, `XRP`, `ADA`, `DOGE`, `AVAX`, `LINK`, `MATIC`, `DOT`, `TRX`,
  `LTC`, `BCH`, `ATOM`, `NEAR`, `APT`, `ARB`, `OP`, `SUI`, `AAVE`, `UNI`
- **Accepted shapes:** `BTCUSDT`, `BTC-USD`, `BTC/USDT`, `BTC` — all
  normalise to the same Binance pair internally.

### Crypto data sources used

| Category | Source | Notes |
|---|---|---|
| Klines (OHLCV) | Binance public REST `/api/v3/klines` | No key needed; ~1000 daily candles cached per symbol |
| Indicators | `stockstats` over Binance OHLCV | Same indicator catalog as the equity path |
| Fundamentals snapshot | CoinGecko `/coins/{id}` | Market data, supply, community, dev activity |
| Market microstructure | CoinGecko `/coins/{id}/tickers` | Exchange concentration, spreads |
| News | RSS: CoinDesk, CoinTelegraph, Bitcoin Magazine | Override via `CRYPTONEWS_FEEDS=url1,url2,...` |
| Insider transactions | No-op (doesn't apply to crypto) | Returns an explanatory placeholder |

### Optional env vars

| Var | Purpose |
|---|---|
| `COINGECKO_API_KEY` | Use CoinGecko Pro tier (higher rate limit) |
| `CRYPTONEWS_FEEDS` | Comma-separated RSS feeds to override the default news list |
| `TRADINGAGENTS_RESULTS_DIR` | Override where reports save |
| `TRADINGAGENTS_CACHE_DIR` | Override where OHLCV / coin lists cache |
| `TRADINGAGENTS_MEMORY_LOG_PATH` | Override the persistent decision-log path |
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | Only needed for the executor (testnet or live) |

---

## 5. Reading the report

Each run produces a 5-tier rating plus a structured rationale:

| Rating | Meaning | Executor mapping |
|---|---|---|
| `Buy` | High-conviction long | 100% of `quote_budget_usdt` → BUY |
| `Overweight` | Lean long | 50% of `quote_budget_usdt` → BUY |
| `Hold` | No-op | No order placed |
| `Underweight` | Lean short / trim | 50% of held base qty → SELL |
| `Sell` | High-conviction exit | 100% of held base qty → SELL |

The Portfolio Manager's final report (under `5_portfolio/decision.md`)
also contains:

- **Executive Summary** — one paragraph of what to actually do
- **Investment Thesis** — why, with citations of analyst arguments
- **Price Target** — explicit number
- **Time Horizon** — e.g. "2–4 weeks"

---

## 6. Trying multiple pairs

The CLI is one ticker per run. To sweep several, just rerun:

```bash
for sym in BTCUSDT ETHUSDT SOLUSDT; do
  .venv/bin/python -c "
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
config = DEFAULT_CONFIG.copy()
config['llm_provider'] = 'claude_cli'
config['deep_think_llm'] = 'sonnet'
config['quick_think_llm'] = 'haiku'
config['max_debate_rounds'] = 1
config['max_risk_discuss_rounds'] = 1
config['data_vendors'] = {
    'core_stock_apis': 'binance',
    'technical_indicators': 'binance',
    'fundamental_data': 'coingecko',
    'news_data': 'crypto_news',
}
ta = TradingAgentsGraph(['market','fundamentals','news'], config=config, debug=False)
state, signal = ta.propagate('$sym', '2026-04-30')
print(f'\n=== $sym ===')
print(f'Signal: {signal}')
print(state['final_trade_decision'])
"
done
```

Or simpler: run the interactive CLI three times, once per ticker — each
run gets its own folder under `~/.tradingagents/logs/`.

---

## 7. Performance expectations

Per run on `claude_cli` with Sonnet/Haiku and Shallow depth:

- 3 analysts (market, fundamentals, news) with tool loops: **~5–8 min**
- Bull/Bear debate + Research Manager: **~3–5 min**
- Trader + 3-way risk debate + Portfolio Manager: **~5–8 min**
- **Total: ~15–25 min per ticker**

Medium depth (~3 rounds) doubles that. Deep (~5 rounds) quadruples.
Opus is roughly 2x slower than Sonnet but produces noticeably more
nuanced final decisions.

---

## 8. Common issues

| Symptom | Cause / fix |
|---|---|
| `Quote not found for symbol: BTCUSDT` warning during startup | Benign — `_resolve_pending_entries` tries yfinance once for prior memory-log entries; returns `None` and moves on. |
| `claude CLI returned non-JSON output` | Local `claude` CLI hit a rate limit or an interactive prompt. Re-run; if persistent, check `claude --version` ≥ 2.1. |
| `No Binance data found for symbol 'X'` | Pair doesn't exist on Binance (e.g. `XYZUSDT` for an unlisted token). Check on `binance.com`. |
| CoinGecko rate-limit (`429`) | Free tier throttles ~10–30/min. Set `COINGECKO_API_KEY` for Pro. |
| Empty news report | Pair name doesn't match RSS title/body. Add an alias to `_NAME_ALIASES` in `tradingagents/dataflows/crypto_news.py`. |
| Run gets stuck on a single analyst forever | Tool loop is iterating; check `~/.tradingagents/logs/<TICKER>/<DATE>/message_tool.log` for the latest tool call. |
| Report says "tool unavailable" / "could not be loaded" / only some indicators returned | A weak model gave up on the multi-tool loop. The prompt now forbids these phrases, but if it recurs, switch the **quick-thinking** model from `haiku` to `sonnet` (CLI step 7 or `config["quick_think_llm"] = "sonnet"` in `main.py`). Sonnet is ~2× slower but reliable on multi-step tool sequences. |

---

## 9. After you're happy with text-only runs

When you've validated the analysis quality, the next stages are:

1. **Stage 2 — Demo execution.** Set up Binance testnet keys, flip
   `run_executor=True` and `dry_run=False`, and let the executor place a
   small testnet order using your last decision. Code is ready —
   see §6 of [`/Users/nechtas/.claude/plans/look-through-this-project-drifting-quiche.md`](../../.claude/plans/look-through-this-project-drifting-quiche.md)
   for the testnet setup steps. The executor already handles real
   `LOT_SIZE` / `MIN_NOTIONAL` filters and auto-fetches your held base
   balance for SELLs.
2. **Stage 3 — Phase 4 safety guards + live.** Not yet built. Symbol
   allow-list, max-notional clamps, daily kill-switch, decision
   rate-limit, then a week of testnet soak before going live.
