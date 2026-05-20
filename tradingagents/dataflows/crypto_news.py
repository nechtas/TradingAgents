"""Crypto news via public RSS feeds.

Uses CoinDesk and CoinTelegraph public RSS — both keyless and unmetered
(within reasonable polite use). For per-coin filtering we match on the
ticker symbol and the asset's full name where we know it.

A single optional knob: set ``CRYPTONEWS_FEEDS`` (comma-separated URLs)
to override the default feed list — handy for adding a specialist feed
or pinning to one source for testing.

Insider transactions don't apply to crypto. The ``get_crypto_insider``
shim returns an explanatory empty result so the existing news_data
toolset still routes cleanly.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Annotated, Iterable
from xml.etree import ElementTree as ET

import requests
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

_DEFAULT_FEEDS = (
    "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    "https://cointelegraph.com/rss",
    "https://bitcoinmagazine.com/feed",
)

# Map common ticker → list of full names / aliases used by news outlets, so
# per-coin filtering catches "Bitcoin" articles when the user asked for BTC.
_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("bitcoin",),
    "ETH": ("ethereum", "ether"),
    "SOL": ("solana",),
    "BNB": ("bnb", "binance coin"),
    "XRP": ("xrp", "ripple"),
    "ADA": ("cardano",),
    "DOGE": ("dogecoin",),
    "AVAX": ("avalanche",),
    "LINK": ("chainlink",),
    "MATIC": ("polygon", "matic"),
    "DOT": ("polkadot",),
    "TRX": ("tron",),
    "LTC": ("litecoin",),
    "BCH": ("bitcoin cash",),
    "ATOM": ("cosmos",),
    "NEAR": ("near protocol",),
    "APT": ("aptos",),
    "ARB": ("arbitrum",),
    "OP": ("optimism",),
    "SUI": ("sui",),
    "AAVE": ("aave",),
    "UNI": ("uniswap",),
}


def _strip_quote(symbol: str) -> str:
    s = symbol.strip().upper().replace("-", "").replace("/", "")
    for q in ("USDT", "BUSD", "USDC", "FDUSD", "TUSD", "USD"):
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)]
    return s


def _feed_urls() -> tuple[str, ...]:
    override = os.environ.get("CRYPTONEWS_FEEDS")
    if override:
        return tuple(u.strip() for u in override.split(",") if u.strip())
    return _DEFAULT_FEEDS


def _parse_pub_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _fetch_feed(url: str) -> list[dict]:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "TradingAgents/0.2 (research)"},
            timeout=20,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("RSS fetch failed for %s: %s", url, exc)
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        logger.warning("RSS parse failed for %s: %s", url, exc)
        return []

    items = []
    # Standard RSS 2.0 layout: rss/channel/item
    for item in root.iter("item"):
        title = _strip_html((item.findtext("title") or "").strip())
        link = (item.findtext("link") or "").strip()
        description = _strip_html((item.findtext("description") or "").strip())
        pub_date = _parse_pub_date(item.findtext("pubDate"))
        if not title:
            continue
        items.append(
            {
                "title": title,
                "link": link,
                "summary": description,
                "pub_date": pub_date,
                "source": _source_name_from_url(url),
            }
        )
    return items


def _source_name_from_url(url: str) -> str:
    if "coindesk" in url:
        return "CoinDesk"
    if "cointelegraph" in url:
        return "CoinTelegraph"
    if "bitcoinmagazine" in url:
        return "Bitcoin Magazine"
    return url


def _aggregate_articles() -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for url in _feed_urls():
        for art in _fetch_feed(url):
            key = art["link"] or art["title"]
            if key in seen:
                continue
            seen.add(key)
            out.append(art)
    return out


def _matches_symbol(article: dict, base_symbol: str) -> bool:
    needles = [base_symbol.lower()]
    needles.extend(a.lower() for a in _NAME_ALIASES.get(base_symbol.upper(), ()))
    haystack = (article.get("title", "") + " " + article.get("summary", "")).lower()
    return any(n in haystack for n in needles)


def _format(articles: Iterable[dict], start_dt: datetime, end_dt: datetime) -> str:
    chunks: list[str] = []
    for art in articles:
        pub_dt = art.get("pub_date")
        if pub_dt and not (start_dt <= pub_dt <= end_dt + relativedelta(days=1)):
            continue
        date_str = pub_dt.strftime("%Y-%m-%d") if pub_dt else "?"
        title = art["title"]
        source = art.get("source", "rss")
        summary = art.get("summary", "")
        if len(summary) > 600:
            summary = summary[:600].rstrip() + "..."
        link = art.get("link", "")
        block = f"### {title} (source: {source}, {date_str})"
        if summary:
            block += f"\n{summary}"
        if link:
            block += f"\nLink: {link}"
        chunks.append(block)
    return "\n\n".join(chunks)


def get_crypto_news(
    ticker: Annotated[str, "crypto symbol or pair, e.g. BTC, BTCUSDT"],
    start_date: Annotated[str, "Start date YYYY-mm-dd"],
    end_date: Annotated[str, "End date YYYY-mm-dd"],
) -> str:
    """Per-coin news. Mirrors ``get_news_yfinance`` signature."""
    base = _strip_quote(ticker)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    articles = _aggregate_articles()
    filtered = [a for a in articles if _matches_symbol(a, base)]
    body = _format(filtered, start_dt, end_dt)
    if not body:
        return f"No crypto news found for {base} between {start_date} and {end_date}"
    return f"## {base} Crypto News, {start_date} to {end_date} (sources: RSS):\n\n{body}"


def get_crypto_global_news(
    curr_date: Annotated[str, "Current date YYYY-mm-dd"],
    look_back_days: Annotated[int, "How many days to look back"] = 7,
    limit: Annotated[int, "Max articles"] = 15,
) -> str:
    """Macro crypto-market news. Mirrors ``get_global_news_yfinance`` signature."""
    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = end_dt - relativedelta(days=look_back_days)
    articles = _aggregate_articles()
    body = _format(articles[:limit], start_dt, end_dt)
    if not body:
        return f"No global crypto news found near {curr_date}"
    return (
        f"## Global Crypto Market News, {start_dt.strftime('%Y-%m-%d')} to {curr_date} "
        f"(sources: RSS):\n\n{body}"
    )


def get_crypto_insider(
    ticker: Annotated[str, "crypto symbol or pair"],
) -> str:
    """No-op shim for the insider-transactions slot.

    Insider transactions are an equity-market construct and don't apply
    to crypto. We return an explanatory message so the news-data tool
    routing stays consistent and the analyst can move on.
    """
    base = _strip_quote(ticker)
    return (
        f"# Insider Transactions for {base}\n\n"
        "Insider transactions are an equity-market construct and do not "
        "apply to cryptocurrencies. Consider on-chain whale-wallet "
        "tracking or exchange in/out-flow analysis as the crypto-native "
        "analogue (not currently wired)."
    )
