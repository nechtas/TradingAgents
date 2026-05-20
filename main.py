"""Example entrypoint for TradingAgents.

Three modes are shown. Pick one by editing the config below; each is a
small set of overrides on ``DEFAULT_CONFIG``.

  1. Equity research (default in DEFAULT_CONFIG)        — yfinance + paid LLM API
  2. Crypto research on Claude Max subscription         — Binance / CoinGecko / RSS + claude_cli
  3. Crypto research + dry-run Binance execution        — adds the executor on top of (2)

For (3), ``BinanceExecutor(dry_run=True)`` writes orders to
``~/.tradingagents/execution/orders.jsonl`` instead of placing them.
DO NOT flip ``dry_run=False`` until the Phase-4 safety guards land.
"""

from dotenv import load_dotenv

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.execution import BinanceExecutor
from tradingagents.graph.trading_graph import TradingAgentsGraph

load_dotenv()


# ---- choose a mode -----------------------------------------------------

config = DEFAULT_CONFIG.copy()

# Mode 1 — equity research with the paid OpenAI API (the upstream default).
# config["llm_provider"] = "openai"
# config["deep_think_llm"] = "gpt-5.4-mini"
# config["quick_think_llm"] = "gpt-5.4-mini"
# symbol, trade_date, run_executor = "NVDA", "2024-05-10", False

# Mode 2 — crypto research on the local Claude CLI (Claude Max subscription).
config["llm_provider"] = "claude_cli"
config["deep_think_llm"] = "sonnet"        # CLI alias; "opus" is also fine
config["quick_think_llm"] = "haiku"
config["max_debate_rounds"] = 1
config["max_risk_discuss_rounds"] = 1
config["data_vendors"] = {
    "core_stock_apis": "binance",
    "technical_indicators": "binance",
    "fundamental_data": "coingecko",
    "news_data": "crypto_news",
}
symbol, trade_date, run_executor = "BTCUSDT", "2026-04-30", False

# Mode 3 — crypto research + dry-run Binance execution. Same as Mode 2 but
# also hands the final decision to the executor. Flip ``run_executor = True``.
# run_executor = True


# ---- run ---------------------------------------------------------------

ta = TradingAgentsGraph(
    selected_analysts=["market", "fundamentals", "news"],
    debug=False,
    config=config,
)

final_state, signal = ta.propagate(symbol, trade_date)
print("\nFinal decision:")
print(final_state["final_trade_decision"])
print(f"\nProcessed signal: {signal}")


# ---- optional: dry-run execution on Binance ----------------------------

if run_executor:
    # ``quote_budget_usdt`` is the strategy's hard ceiling for a single Buy.
    # Keep it tiny while you iterate; it's the only sizing knob this MVP
    # exposes (Phase 4 adds notional caps, allow-lists, kill-switches).
    executor = BinanceExecutor(testnet=True, dry_run=True)
    result = executor.execute_decision(
        rating=signal,
        symbol=symbol,
        quote_budget_usdt=20.0,
        decision_text=final_state["final_trade_decision"],
        decision_date=trade_date,
        # held_base_qty defaults to None → executor pulls live balance from
        # Binance when API keys are set. Pass an explicit float to override.
    )
    print(f"\nExecutor: side={result.side} qty={result.quantity} "
          f"placed={result.placed} dry_run={result.dry_run} "
          f"reason={result.skipped_reason or result.error or 'ok'}")


# Memorize mistakes and reflect — opt-in.
# ta.reflect_and_remember(1000)  # parameter is the position returns
