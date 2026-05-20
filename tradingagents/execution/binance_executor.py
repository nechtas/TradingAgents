"""Translate a PortfolioDecision into a Binance spot order.

Phase-3 scope: end-to-end plumbing only. The executor is **dry-run by
default** — it logs the would-be order to ``orders.jsonl`` and returns
a synthesized OrderResult without touching Binance. To actually place
an order:

  * For testnet: ``BinanceExecutor(api_key, api_secret, testnet=True, dry_run=False)``
  * For live: ``BinanceExecutor(api_key, api_secret, testnet=False, dry_run=False)``
    — but DO NOT do this until the Phase-4 safety guards are in place.

Subaccount setup (manual, on the Binance UI):
  1. Sub Accounts → Create Sub Account → fund with the small test amount.
  2. API Management on the subaccount → create a key with **Spot Trading
     enabled, Withdrawals disabled, IP-restricted** to your machine.
  3. Set ``BINANCE_API_KEY`` and ``BINANCE_API_SECRET`` in ``.env``.

Sizing: the rating maps to a simple fraction of the per-strategy USDT
budget the user passes via ``quote_budget_usdt``. This is intentionally
crude; a more sophisticated sizer (Kelly, vol-targeted, …) belongs
upstream of the executor where it can see broader portfolio state.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from .order_log import OrderLog

logger = logging.getLogger(__name__)


# Common quote-asset suffixes on Binance, longest-first so BTCUSDT strips to
# BTC (not BTCUS) and ETHUSDC strips to ETH (not ETHUS). Used by the
# auto-balance lookup to derive the base asset for get_asset_balance.
_QUOTE_SUFFIXES = ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "BTC", "ETH", "BNB")


def _strip_quote_asset(symbol: str) -> str:
    s = symbol.upper()
    for q in _QUOTE_SUFFIXES:
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)]
    return s


# Map the Portfolio Manager's 5-tier rating to (side, fraction) where
# fraction is a unitless multiplier on the strategy's quote budget for a
# Buy-side rating, or on the held base position for a Sell-side rating.
_RATING_TO_ACTION: dict[str, tuple[Optional[str], float]] = {
    "Buy":         ("BUY",  1.0),
    "Overweight":  ("BUY",  0.5),
    "Hold":        (None,   0.0),
    "Underweight": ("SELL", 0.5),
    "Sell":        ("SELL", 1.0),
}


@dataclass
class OrderResult:
    """Outcome of a single ``execute_decision`` call.

    Always returned (never None) so callers can log it uniformly. Whether
    the order was actually sent is in ``placed``; whether the desk
    intended to act is in ``side``.
    """

    symbol: str
    side: Optional[str]            # "BUY" / "SELL" / None (hold)
    rating: str                    # the input rating string
    placed: bool                   # True if a real Binance order went out
    dry_run: bool
    testnet: bool
    quantity: float = 0.0
    quote_amount: float = 0.0      # USDT spent (BUY) or expected to receive (SELL)
    order_type: str = "MARKET"
    binance_response: Optional[dict] = None
    skipped_reason: Optional[str] = None
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


class BinanceExecutor:
    """Convert a ``PortfolioDecision`` rating into a Binance spot order.

    The executor knows nothing about the LLM pipeline. It takes a rating
    string (``"Buy"``, ``"Overweight"``, ``"Hold"``, ``"Underweight"``,
    ``"Sell"``), a symbol, the strategy's USDT budget, and whatever
    metadata you want preserved in the audit log.

    Phase-4 will wrap ``execute_decision`` with safety clamps; for now
    we only enforce ``dry_run`` and the per-call ``quote_budget_usdt``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        *,
        testnet: bool = True,
        dry_run: bool = True,
        order_log_path: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("BINANCE_API_KEY")
        self.api_secret = api_secret or os.environ.get("BINANCE_API_SECRET")
        self.testnet = testnet
        self.dry_run = dry_run
        self.log = OrderLog(order_log_path)
        self._client = None  # lazy — only built if a live order is needed
        # Per-symbol filter cache: stepSize, minQty, minNotional. Populated
        # on first use (one REST call per symbol per executor lifetime).
        self._symbol_filters: dict[str, dict] = {}

    # ---- public API ----------------------------------------------------

    def execute_decision(
        self,
        rating: str,
        symbol: str,
        quote_budget_usdt: float,
        *,
        decision_text: Optional[str] = None,
        decision_date: Optional[str] = None,
        held_base_qty: Optional[float] = None,
        extra: Optional[dict] = None,
    ) -> OrderResult:
        """Place a spot order based on the Portfolio Manager rating.

        :param rating: "Buy" / "Overweight" / "Hold" / "Underweight" / "Sell"
        :param symbol: Binance symbol, e.g. "BTCUSDT"
        :param quote_budget_usdt: hard ceiling on USDT this call may spend
        :param decision_text: optional rendered markdown to embed in the log
        :param decision_date: optional ISO date the decision applies to
        :param held_base_qty: current base-asset holdings; used for Sell sizing.
            When None (default) and a SELL is needed, the executor auto-fetches
            the live balance from Binance (requires API keys). Pass an explicit
            float to override the auto-fetch (e.g. dry-run rehearsal without keys).
        :param extra: arbitrary extra fields to embed in the log record
        """
        side, fraction = _RATING_TO_ACTION.get(rating, (None, 0.0))
        result = OrderResult(
            symbol=symbol,
            side=side,
            rating=rating,
            placed=False,
            dry_run=self.dry_run,
            testnet=self.testnet,
            extra=extra or {},
        )

        try:
            if side is None:
                result.skipped_reason = "rating maps to no action"
                self._log(result, decision_text, decision_date)
                return result

            if side == "BUY":
                quote_amount = quote_budget_usdt * fraction
                if quote_amount <= 0:
                    result.skipped_reason = "quote_budget_usdt is zero or negative"
                    self._log(result, decision_text, decision_date)
                    return result
                # Pre-flight MIN_NOTIONAL: Binance rejects orders worth less
                # than this. Catch it now instead of after the round-trip.
                min_notional = self._get_symbol_filters(symbol)["min_notional"]
                if min_notional and quote_amount < min_notional:
                    result.skipped_reason = (
                        f"quote_amount {quote_amount:.2f} < MIN_NOTIONAL "
                        f"{min_notional:.2f} for {symbol}"
                    )
                    self._log(result, decision_text, decision_date)
                    return result
                result.quote_amount = quote_amount
                result.quantity = self._estimate_base_qty(symbol, quote_amount)

                if self.dry_run:
                    result.binance_response = {
                        "dry_run": True,
                        "would_send": {
                            "symbol": symbol,
                            "side": "BUY",
                            "type": "MARKET",
                            "quoteOrderQty": round(quote_amount, 2),
                        },
                    }
                else:
                    client = self._get_client()
                    result.binance_response = client.create_order(
                        symbol=symbol,
                        side="BUY",
                        type="MARKET",
                        quoteOrderQty=round(quote_amount, 2),
                    )
                    result.placed = True

            elif side == "SELL":
                if held_base_qty is None:
                    # Auto-fetch from Binance when API keys are available;
                    # otherwise the caller has to pass it explicitly.
                    if self.api_key and self.api_secret:
                        held_base_qty = self._fetch_base_balance(symbol)
                        result.extra.setdefault(
                            "auto_held_base_qty", held_base_qty
                        )
                    else:
                        result.skipped_reason = (
                            "held_base_qty=None and no Binance API keys "
                            "configured — cannot size SELL. Either pass "
                            "held_base_qty explicitly or set "
                            "BINANCE_API_KEY/SECRET in .env."
                        )
                        self._log(result, decision_text, decision_date)
                        return result
                qty = held_base_qty * fraction
                if qty <= 0:
                    result.skipped_reason = (
                        f"held_base_qty {held_base_qty} yields no SELL qty "
                        f"for fraction {fraction}. Nothing to sell."
                    )
                    self._log(result, decision_text, decision_date)
                    return result
                qty = self._truncate_qty(symbol, qty)
                filters = self._get_symbol_filters(symbol)
                if filters["min_qty"] and qty < filters["min_qty"]:
                    result.skipped_reason = (
                        f"sell qty {qty} < LOT_SIZE.minQty {filters['min_qty']} "
                        f"for {symbol}"
                    )
                    self._log(result, decision_text, decision_date)
                    return result
                price = self._estimate_price(symbol)
                notional = qty * price
                if filters["min_notional"] and notional < filters["min_notional"]:
                    result.skipped_reason = (
                        f"sell notional {notional:.2f} < MIN_NOTIONAL "
                        f"{filters['min_notional']:.2f} for {symbol}"
                    )
                    self._log(result, decision_text, decision_date)
                    return result
                result.quantity = qty
                result.quote_amount = notional

                if self.dry_run:
                    result.binance_response = {
                        "dry_run": True,
                        "would_send": {
                            "symbol": symbol,
                            "side": "SELL",
                            "type": "MARKET",
                            "quantity": qty,
                        },
                    }
                else:
                    client = self._get_client()
                    result.binance_response = client.create_order(
                        symbol=symbol,
                        side="SELL",
                        type="MARKET",
                        quantity=qty,
                    )
                    result.placed = True

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Order execution failed for %s rating=%s", symbol, rating)

        self._log(result, decision_text, decision_date)
        return result

    # ---- internals -----------------------------------------------------

    def _get_client(self):
        if self._client is None:
            if not self.api_key or not self.api_secret:
                raise RuntimeError(
                    "BINANCE_API_KEY / BINANCE_API_SECRET not set; cannot place a "
                    "live order. Use dry_run=True or set the environment variables."
                )
            try:
                from binance.client import Client  # python-binance
            except ImportError as exc:
                raise RuntimeError(
                    "python-binance is required for live execution. "
                    "Install with: pip install python-binance"
                ) from exc
            self._client = Client(self.api_key, self.api_secret, testnet=self.testnet)
        return self._client

    def _estimate_price(self, symbol: str) -> float:
        """Best-effort spot price for sizing/logging. No fail-loud — sizing
        for SELL only depends on held_base_qty; price is informational."""
        try:
            import requests
            base = "https://testnet.binance.vision" if self.testnet else "https://api.binance.com"
            resp = requests.get(f"{base}/api/v3/ticker/price", params={"symbol": symbol}, timeout=10)
            resp.raise_for_status()
            return float(resp.json().get("price", 0))
        except Exception as exc:
            logger.warning("Could not fetch %s price for sizing: %s", symbol, exc)
            return 0.0

    def _estimate_base_qty(self, symbol: str, quote_amount: float) -> float:
        price = self._estimate_price(symbol)
        return quote_amount / price if price > 0 else 0.0

    def _fetch_base_balance(self, symbol: str) -> float:
        """Fetch the current free balance of ``symbol``'s base asset.

        Uses the authenticated python-binance client. The base asset is
        whatever's left after stripping the quote-asset suffix (so BTCUSDT
        → BTC). Returns 0.0 on any error — sizing then trips the
        "nothing to sell" branch with a clear log entry rather than
        guessing.
        """
        base_asset = _strip_quote_asset(symbol)
        try:
            client = self._get_client()
            bal = client.get_asset_balance(asset=base_asset) or {}
            return float(bal.get("free", 0) or 0)
        except Exception as exc:
            logger.warning(
                "Could not fetch %s balance for %s: %s",
                base_asset, symbol, exc,
            )
            return 0.0

    def _get_symbol_filters(self, symbol: str) -> dict:
        """Fetch and cache LOT_SIZE / MIN_NOTIONAL filters for ``symbol``.

        Binance rejects orders that don't match the per-symbol filters.
        We pull them from the public exchangeInfo endpoint (no auth, no
        rate-limit concern at our cadence) and cache for the executor
        lifetime. Falls back to a permissive default on network failure
        so dry-run flows continue to work offline.
        """
        cached = self._symbol_filters.get(symbol)
        if cached is not None:
            return cached

        defaults = {"step_size": 1e-6, "min_qty": 0.0, "min_notional": 0.0}
        try:
            import requests
            base = (
                "https://testnet.binance.vision" if self.testnet
                else "https://api.binance.com"
            )
            resp = requests.get(
                f"{base}/api/v3/exchangeInfo",
                params={"symbol": symbol},
                timeout=10,
            )
            resp.raise_for_status()
            symbols = resp.json().get("symbols") or []
            if not symbols:
                logger.warning("exchangeInfo returned no symbol entry for %s", symbol)
                self._symbol_filters[symbol] = defaults
                return defaults

            filters = {f["filterType"]: f for f in symbols[0].get("filters", [])}
            lot = filters.get("LOT_SIZE", {})
            # MIN_NOTIONAL was renamed to NOTIONAL on newer symbols; handle both.
            notional_filter = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL", {})
            out = {
                "step_size": float(lot.get("stepSize") or defaults["step_size"]),
                "min_qty": float(lot.get("minQty") or defaults["min_qty"]),
                "min_notional": float(
                    notional_filter.get("minNotional") or defaults["min_notional"]
                ),
            }
            self._symbol_filters[symbol] = out
            return out
        except Exception as exc:
            logger.warning(
                "Could not fetch exchangeInfo for %s (%s); using permissive defaults",
                symbol, exc,
            )
            self._symbol_filters[symbol] = defaults
            return defaults

    def _truncate_qty(self, symbol: str, qty: float) -> float:
        """Truncate ``qty`` down to the symbol's LOT_SIZE stepSize.

        We truncate (floor) rather than round so a SELL never exceeds the
        held balance — overshoot would be rejected by Binance. The step
        comes from exchangeInfo so it matches what the exchange enforces.
        """
        step = self._get_symbol_filters(symbol)["step_size"] or 1e-6
        return math.floor(qty / step) * step

    def _log(
        self,
        result: OrderResult,
        decision_text: Optional[str],
        decision_date: Optional[str],
    ) -> None:
        record = {
            "symbol": result.symbol,
            "rating": result.rating,
            "side": result.side,
            "placed": result.placed,
            "dry_run": result.dry_run,
            "testnet": result.testnet,
            "quantity": result.quantity,
            "quote_amount": result.quote_amount,
            "order_type": result.order_type,
            "binance_response": result.binance_response,
            "skipped_reason": result.skipped_reason,
            "error": result.error,
            "decision_date": decision_date,
            "decision_text_excerpt": (decision_text or "")[:500],
            **result.extra,
        }
        self.log.write(record)
