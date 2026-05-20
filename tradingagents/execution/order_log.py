"""Append-only JSONL log of every intended and executed order.

This is the audit trail and the substrate for reflection on past trades.
Every executor call writes one record — whether the order was a real
fill, a dry-run, a refusal by a safety clamp, or an error.

Format choice: JSONL keeps the file append-only without locking, easy
to ``tail -f`` during testnet soak, and parseable line-by-line in any
language. Don't switch this to JSON or SQLite without a strong reason;
the simplicity is the point.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class OrderLog:
    """Thread-safe append-only writer for execution events."""

    _DEFAULT_REL_PATH = "execution/orders.jsonl"

    def __init__(self, log_path: Optional[str] = None):
        if log_path is None:
            home = os.environ.get("TRADINGAGENTS_HOME") or os.path.join(
                os.path.expanduser("~"), ".tradingagents"
            )
            log_path = os.path.join(home, self._DEFAULT_REL_PATH)
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        record = {**record, "logged_at": datetime.now(timezone.utc).isoformat()}
        line = json.dumps(record, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        logger.info("order log: %s", line)
