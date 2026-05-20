"""Trade-execution layer.

Bridges the LLM pipeline's recommendation (a ``PortfolioDecision``
markdown string + a parsed rating) to a concrete order on a Binance
spot subaccount. By default the executor runs in dry-run mode and
testnet endpoint, so wiring it up doesn't risk real funds; flipping to
live trading is an explicit opt-in via the ``BinanceExecutor``
constructor.

Phase-3 scope: end-to-end plumbing with the order log, dry-run mode,
and testnet support. Safety clamps (notional caps, daily loss
kill-switch, symbol allow-list, rate limits) land in Phase 4 — DO NOT
flip ``dry_run=False, testnet=False`` until those are in place.
"""

from .binance_executor import BinanceExecutor, OrderResult
from .order_log import OrderLog

__all__ = ["BinanceExecutor", "OrderResult", "OrderLog"]
