"""CoinGecko-backed crypto fundamentals.

Replaces the equity-only Fundamentals/BalanceSheet/Cashflow/Income tools
when the framework is run on a crypto symbol. Uses CoinGecko's free public
endpoints — no API key required for the calls below, though they're
rate-limited (~10–30/min). For higher volume you can plug in a Pro key
via the ``COINGECKO_API_KEY`` env var.

Symbol → coin-id resolution: CoinGecko addresses coins by an internal
slug ("bitcoin", "ethereum", "solana"). We hit ``/api/v3/coins/list``
once and cache the mapping locally so subsequent runs don't re-fetch.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Annotated, Optional

import requests

from .config import get_config

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.coingecko.com/api/v3"
_COIN_LIST_CACHE_NAME = "coingecko-coin-list.json"
_COIN_LIST_TTL_SECONDS = 7 * 24 * 60 * 60  # 1 week


def _api_key_header() -> dict:
    key = os.environ.get("COINGECKO_API_KEY")
    return {"x-cg-pro-api-key": key} if key else {}


def _get(url: str, params: Optional[dict] = None, timeout: int = 20) -> dict:
    resp = requests.get(url, params=params, headers=_api_key_header(), timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _load_coin_list() -> list[dict]:
    config = get_config()
    cache_dir = config["data_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, _COIN_LIST_CACHE_NAME)

    if os.path.exists(path):
        if (time.time() - os.path.getmtime(path)) < _COIN_LIST_TTL_SECONDS:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)

    data = _get(f"{_BASE_URL}/coins/list")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return data


# Symbols that have many homonyms on CoinGecko (ETH the test net, BTC pre-fork
# tokens, ...). The user almost always means these canonical mainnet coins,
# so we hard-pin them to skip the disambiguation step.
_CANONICAL_OVERRIDES = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "MATIC": "matic-network",
    "DOT": "polkadot",
    "TRX": "tron",
    "LTC": "litecoin",
    "BCH": "bitcoin-cash",
    "ATOM": "cosmos",
    "NEAR": "near",
    "APT": "aptos",
    "ARB": "arbitrum",
    "OP": "optimism",
    "SUI": "sui",
}


def _strip_quote(symbol: str) -> str:
    """Strip the quote-asset suffix from a trading-pair symbol.

    Example: BTCUSDT → BTC, ETH-USD → ETH, SOL/USDT → SOL.
    """
    s = symbol.strip().upper().replace("-", "").replace("/", "")
    for q in ("USDT", "BUSD", "USDC", "FDUSD", "TUSD", "USD"):
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)]
    return s


def resolve_coin_id(symbol: str) -> Optional[str]:
    """Resolve a ticker (BTC, BTCUSDT, ETH-USD...) to a CoinGecko coin id."""
    base = _strip_quote(symbol)
    if base in _CANONICAL_OVERRIDES:
        return _CANONICAL_OVERRIDES[base]

    coins = _load_coin_list()
    matches = [c for c in coins if c.get("symbol", "").upper() == base]
    if not matches:
        return None
    # When multiple coins share a ticker, prefer ones whose id matches symbol
    # exactly; otherwise return the first (usually the most-cap-weighted).
    for c in matches:
        if c.get("id", "").lower() == base.lower():
            return c["id"]
    return matches[0]["id"]


def get_crypto_fundamentals(
    symbol: Annotated[str, "crypto symbol or pair, e.g. BTC, BTCUSDT"],
    curr_date: Annotated[str, "Current date YYYY-mm-dd (informational)"] = None,
) -> str:
    """High-level snapshot: market data + supply + ATH/ATL + community.

    This is the closest crypto analogue to ``get_fundamentals`` for an
    equity. The Fundamentals Analyst is given this report instead of a
    balance sheet when the framework runs in crypto mode.
    """
    coin_id = resolve_coin_id(symbol)
    if not coin_id:
        return f"Could not resolve '{symbol}' to a CoinGecko coin id."

    try:
        data = _get(
            f"{_BASE_URL}/coins/{coin_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "true",
                "developer_data": "true",
                "sparkline": "false",
            },
        )
    except Exception as exc:
        return f"Error fetching CoinGecko data for {coin_id}: {exc}"

    md = data.get("market_data", {})
    cd = data.get("community_data", {})
    dd = data.get("developer_data", {})

    def _usd(field: str):
        v = (md.get(field) or {}).get("usd")
        return v if v is not None else "N/A"

    def _pct(field: str):
        v = md.get(field)
        return f"{v:.2f}%" if isinstance(v, (int, float)) else "N/A"

    fields = [
        ("Name", data.get("name")),
        ("Symbol", (data.get("symbol") or "").upper()),
        ("CoinGecko id", coin_id),
        ("Market cap rank", data.get("market_cap_rank")),
        ("Categories", ", ".join(data.get("categories") or []) or "N/A"),
        ("Genesis date", data.get("genesis_date") or "N/A"),
        ("Hashing algorithm", data.get("hashing_algorithm") or "N/A"),
        ("", ""),
        ("Current price (USD)", _usd("current_price")),
        ("Market cap (USD)", _usd("market_cap")),
        ("Fully diluted valuation (USD)", _usd("fully_diluted_valuation")),
        ("24h trading volume (USD)", _usd("total_volume")),
        ("24h price change", _pct("price_change_percentage_24h")),
        ("7d price change", _pct("price_change_percentage_7d")),
        ("30d price change", _pct("price_change_percentage_30d")),
        ("1y price change", _pct("price_change_percentage_1y")),
        ("All-time high (USD)", _usd("ath")),
        ("ATH change %", _pct("ath_change_percentage")),
        ("ATH date (USD)", (md.get("ath_date") or {}).get("usd", "N/A")),
        ("All-time low (USD)", _usd("atl")),
        ("Circulating supply", md.get("circulating_supply")),
        ("Total supply", md.get("total_supply")),
        ("Max supply", md.get("max_supply")),
        ("", ""),
        ("Twitter followers", cd.get("twitter_followers")),
        ("Reddit subscribers", cd.get("reddit_subscribers")),
        ("Reddit avg active users (48h)", cd.get("reddit_accounts_active_48h")),
        ("Telegram channel users", cd.get("telegram_channel_user_count")),
        ("", ""),
        ("Github forks", dd.get("forks")),
        ("Github stars", dd.get("stars")),
        ("Github subscribers", dd.get("subscribers")),
        ("Total contributors", dd.get("total_issues")),
        ("Pull requests merged", dd.get("pull_requests_merged")),
        ("Commits last 4 weeks", dd.get("commit_count_4_weeks")),
    ]

    body_lines = []
    for label, value in fields:
        if label == "":
            body_lines.append("")
            continue
        if value is None:
            continue
        body_lines.append(f"{label}: {value}")

    header = (
        f"# Crypto Fundamentals for {(data.get('symbol') or symbol).upper()} "
        f"({data.get('name', coin_id)})\n"
        f"# Source: CoinGecko ({coin_id})"
        + (f" — as of {curr_date}" if curr_date else "")
        + f"\n# Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + "\n".join(body_lines)


def get_crypto_market_metrics(
    symbol: Annotated[str, "crypto symbol or pair"],
    curr_date: Annotated[str, "current date (informational)"] = None,
) -> str:
    """Drill-down market microstructure: top exchanges, liquidity, volume mix.

    Plays the role that ``get_balance_sheet`` plays for an equity — a
    deeper structural look beyond the high-level overview. For a crypto
    asset that means: where is it traded, how concentrated is liquidity,
    how does volume split across pairs.
    """
    coin_id = resolve_coin_id(symbol)
    if not coin_id:
        return f"Could not resolve '{symbol}' to a CoinGecko coin id."

    try:
        data = _get(
            f"{_BASE_URL}/coins/{coin_id}/tickers",
            params={"include_exchange_logo": "false", "depth": "true"},
        )
    except Exception as exc:
        return f"Error fetching CoinGecko tickers for {coin_id}: {exc}"

    tickers = data.get("tickers") or []
    if not tickers:
        return f"No exchange ticker data for {coin_id}"

    # Aggregate top venues by 24h converted-USD volume.
    by_exchange: dict[str, dict] = {}
    for t in tickers:
        ex = (t.get("market") or {}).get("name", "Unknown")
        vol = (t.get("converted_volume") or {}).get("usd") or 0.0
        spread = t.get("bid_ask_spread_percentage") or 0.0
        slot = by_exchange.setdefault(ex, {"volume": 0.0, "pairs": 0, "min_spread": None})
        slot["volume"] += float(vol)
        slot["pairs"] += 1
        if isinstance(spread, (int, float)):
            cur_min = slot["min_spread"]
            slot["min_spread"] = spread if cur_min is None else min(cur_min, spread)

    rows = sorted(by_exchange.items(), key=lambda kv: kv[1]["volume"], reverse=True)
    total_vol = sum(v["volume"] for _, v in rows) or 1.0

    lines = ["| Rank | Exchange | 24h Volume USD | % of total | # Pairs | Best spread % |",
             "|------|----------|----------------|------------|---------|---------------|"]
    for i, (ex, slot) in enumerate(rows[:15], start=1):
        spread = slot["min_spread"]
        spread_str = f"{spread:.4f}" if isinstance(spread, (int, float)) else "N/A"
        lines.append(
            f"| {i} | {ex} | {slot['volume']:,.0f} | {slot['volume'] / total_vol * 100:.2f}% "
            f"| {slot['pairs']} | {spread_str} |"
        )

    header = (
        f"# Crypto Market Microstructure for {(data.get('name') or coin_id).upper()}\n"
        f"# Source: CoinGecko /coins/{coin_id}/tickers"
        + (f" — as of {curr_date}" if curr_date else "")
        + f"\n# Total tickers: {len(tickers)} across {len(by_exchange)} exchanges\n\n"
    )
    return header + "\n".join(lines)


def get_crypto_developer_activity(
    symbol: Annotated[str, "crypto symbol or pair"],
    freq: Annotated[str, "ignored — kept for interface compatibility"] = "quarterly",
    curr_date: Annotated[str, "current date (informational)"] = None,
) -> str:
    """Developer-activity report — the crypto cousin of ``get_cashflow``.

    Cash flow doesn't exist for a network; the closest analogue to
    "value being produced/consumed" is on-chain activity and developer
    output. We surface git activity from CoinGecko's developer_data field.
    """
    coin_id = resolve_coin_id(symbol)
    if not coin_id:
        return f"Could not resolve '{symbol}' to a CoinGecko coin id."

    try:
        data = _get(
            f"{_BASE_URL}/coins/{coin_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "market_data": "false",
                "community_data": "false",
                "developer_data": "true",
                "sparkline": "false",
            },
        )
    except Exception as exc:
        return f"Error fetching CoinGecko developer data for {coin_id}: {exc}"

    dd = data.get("developer_data", {}) or {}
    code_added = dd.get("code_additions_deletions_4_weeks", {}) or {}

    fields = [
        ("Forks", dd.get("forks")),
        ("Stars", dd.get("stars")),
        ("Subscribers", dd.get("subscribers")),
        ("Total issues", dd.get("total_issues")),
        ("Closed issues", dd.get("closed_issues")),
        ("Pull requests merged", dd.get("pull_requests_merged")),
        ("Pull request contributors", dd.get("pull_request_contributors")),
        ("Commit count (4 weeks)", dd.get("commit_count_4_weeks")),
        ("Code additions (4 weeks)", code_added.get("additions")),
        ("Code deletions (4 weeks)", code_added.get("deletions")),
    ]

    body = "\n".join(f"{label}: {value}" for label, value in fields if value is not None)
    header = (
        f"# Developer Activity for {(data.get('name') or coin_id).upper()}\n"
        f"# Source: CoinGecko ({coin_id})"
        + (f" — as of {curr_date}" if curr_date else "")
        + "\n\n"
    )
    return header + body


def get_crypto_supply_metrics(
    symbol: Annotated[str, "crypto symbol or pair"],
    freq: Annotated[str, "ignored — kept for interface compatibility"] = "quarterly",
    curr_date: Annotated[str, "current date (informational)"] = None,
) -> str:
    """Token supply + price-history snapshot — the crypto cousin of ``get_income_statement``.

    Income statements don't apply to networks; the analogue we surface is
    issuance/supply pressure (circulating vs total vs max) plus return
    statistics that show "earnings power" of holding the asset.
    """
    coin_id = resolve_coin_id(symbol)
    if not coin_id:
        return f"Could not resolve '{symbol}' to a CoinGecko coin id."

    try:
        data = _get(
            f"{_BASE_URL}/coins/{coin_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false",
                "sparkline": "false",
            },
        )
    except Exception as exc:
        return f"Error fetching CoinGecko market data for {coin_id}: {exc}"

    md = data.get("market_data", {}) or {}

    def _usd(f):
        return (md.get(f) or {}).get("usd", "N/A")

    fields = [
        ("Circulating supply", md.get("circulating_supply")),
        ("Total supply", md.get("total_supply")),
        ("Max supply", md.get("max_supply") or "Uncapped"),
        ("Market cap (USD)", _usd("market_cap")),
        ("Fully diluted valuation (USD)", _usd("fully_diluted_valuation")),
        ("Market cap / FDV ratio", md.get("market_cap_fdv_ratio")),
        ("Total volume (24h, USD)", _usd("total_volume")),
        ("Price change 24h (USD)", _usd("price_change_24h_in_currency")),
        ("Price change 7d (%)", md.get("price_change_percentage_7d")),
        ("Price change 14d (%)", md.get("price_change_percentage_14d")),
        ("Price change 30d (%)", md.get("price_change_percentage_30d")),
        ("Price change 60d (%)", md.get("price_change_percentage_60d")),
        ("Price change 200d (%)", md.get("price_change_percentage_200d")),
        ("Price change 1y (%)", md.get("price_change_percentage_1y")),
    ]
    body = "\n".join(
        f"{label}: {value}" for label, value in fields if value is not None
    )
    header = (
        f"# Supply & Returns for {(data.get('name') or coin_id).upper()}\n"
        f"# Source: CoinGecko ({coin_id})"
        + (f" — as of {curr_date}" if curr_date else "")
        + "\n\n"
    )
    return header + body
