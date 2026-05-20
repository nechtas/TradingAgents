"""Binance public-data adapter for crypto OHLCV and indicators.

Mirrors the shape of ``y_finance.py`` so the existing vendor-routing layer
in ``interface.py`` can swap implementations without touching the agents.

Only the public REST endpoints are used here. They require no API key, no
authentication, and have generous rate limits — which is what we want for
research/backtesting. The ``execution/`` package will introduce
authenticated calls separately when we wire up live trading.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Annotated

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from stockstats import wrap

from .config import get_config
from .stockstats_utils import _clean_dataframe

logger = logging.getLogger(__name__)

_KLINES_URL = "https://api.binance.com/api/v3/klines"
# Binance returns max 1000 candles per call. For 5y of daily candles
# (~1825) we'd need pagination; for the default 5y window with a 1d
# interval we cap at 1000 trailing candles which is ~2.7 years and good
# enough for trend analysis. Bump this if you need a longer history.
_MAX_LIMIT = 1000


def _normalize_symbol(symbol: str) -> str:
    """Normalize a user-supplied symbol to Binance's flat USDT-pair form.

    Accepts "BTC", "BTCUSDT", "BTC-USD", "BTC/USDT", "btcusdt" and
    returns "BTCUSDT". Defaults to USDT pair when the user only gives
    the base asset.
    """
    s = symbol.strip().upper().replace("-", "").replace("/", "")
    # Common quote suffixes Binance lists. Order matters: longest first.
    quotes = ("USDT", "BUSD", "USDC", "FDUSD", "TUSD", "BTC", "ETH", "BNB")
    for q in quotes:
        if s.endswith(q) and len(s) > len(q):
            return s
    # Treat USD as a request for USDT pair (Binance has no plain USD pair).
    if s.endswith("USD"):
        return s[:-3] + "USDT"
    # Bare base asset → assume USDT pair.
    return s + "USDT"


def _fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = _MAX_LIMIT,
) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    resp = requests.get(_KLINES_URL, params=params, timeout=20)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return pd.DataFrame(
            columns=["Date", "Open", "High", "Low", "Close", "Volume"]
        )

    # Binance kline schema:
    # [open_time, open, high, low, close, volume, close_time, quote_volume,
    #  trades, taker_buy_base, taker_buy_quote, ignore]
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "Open", "High", "Low", "Close", "Volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    return df


def load_crypto_ohlcv(symbol: str, curr_date: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch and cache crypto OHLCV up to ``curr_date``.

    Same caching and look-ahead-guard semantics as
    ``stockstats_utils.load_ohlcv`` so the indicator path doesn't care
    which asset class it's looking at.
    """
    config = get_config()
    sym = _normalize_symbol(symbol)
    curr_dt = pd.to_datetime(curr_date)

    # Cache window: a fixed trailing 1000 daily candles (~2.7y) up to today.
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    end_ms = int(today.timestamp() * 1000)
    start_ms = end_ms - _MAX_LIMIT * 24 * 60 * 60 * 1000  # 1000 days back

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{sym}-Binance-{interval}-{today.strftime('%Y-%m-%d')}.csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
    else:
        data = _fetch_klines(sym, interval, start_ms, end_ms)
        data.to_csv(data_file, index=False, encoding="utf-8")

    data = _clean_dataframe(data)
    data = data[data["Date"] <= curr_dt]
    return data


def get_crypto_data(
    symbol: Annotated[str, "crypto symbol, e.g. BTCUSDT or BTC"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Return Binance OHLCV CSV string for the requested window.

    Signature mirrors ``y_finance.get_YFin_data_online`` so the vendor
    router can substitute it transparently.
    """
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")
    sym = _normalize_symbol(symbol)

    try:
        data = load_crypto_ohlcv(sym, end_date)
    except Exception as exc:  # network / API outage
        return f"Error fetching Binance data for {sym}: {exc}"

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    data = data[(data["Date"] >= start_dt) & (data["Date"] <= end_dt)]

    if data.empty:
        return f"No Binance data found for symbol '{sym}' between {start_date} and {end_date}"

    for col in ("Open", "High", "Low", "Close"):
        data[col] = data[col].round(4)

    csv_string = data.to_csv(index=False)
    header = (
        f"# Crypto OHLCV for {sym} from {start_date} to {end_date} (source: Binance)\n"
        f"# Total records: {len(data)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string


def get_crypto_indicators(
    symbol: Annotated[str, "crypto symbol, e.g. BTCUSDT"],
    indicator: Annotated[str, "stockstats indicator name, e.g. close_50_sma, macd, rsi"],
    curr_date: Annotated[str, "Current date YYYY-mm-dd"],
    look_back_days: Annotated[int, "How many days to look back"],
) -> str:
    """Compute a stockstats indicator over a trailing window from Binance OHLCV.

    Same indicator catalog as ``y_finance.get_stock_stats_indicators_window``;
    crypto markets are 24/7 so every date is a trading day.
    """
    sym = _normalize_symbol(symbol)
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)

    try:
        data = load_crypto_ohlcv(sym, curr_date)
    except Exception as exc:
        return f"Error fetching Binance data for {sym}: {exc}"

    if data.empty:
        return f"No Binance data available for {sym} on {curr_date}"

    df = wrap(data.copy())
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    try:
        df[indicator]  # trigger calculation
    except Exception as exc:
        return f"Indicator '{indicator}' is not supported by stockstats: {exc}"

    by_date = {row["Date"]: row[indicator] for _, row in df.iterrows()}

    lines = []
    cursor = curr_dt
    while cursor >= start_dt:
        date_str = cursor.strftime("%Y-%m-%d")
        value = by_date.get(date_str, "N/A: Binance returned no candle (rare for 24/7 markets)")
        if pd.isna(value):
            value = "N/A"
        lines.append(f"{date_str}: {value}")
        cursor -= relativedelta(days=1)

    return (
        f"## {indicator} values for {sym} from {start_dt.strftime('%Y-%m-%d')} to {curr_date} "
        f"(source: Binance):\n\n" + "\n".join(lines)
    )
