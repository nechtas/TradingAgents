from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    asset_class = detect_asset_class(ticker)
    if asset_class == "crypto":
        return (
            f"The instrument to analyze is the cryptocurrency `{ticker}` "
            "(a Binance trading pair or bare crypto symbol). "
            "Use this exact ticker in every tool call, report, and recommendation. "
            "Crypto markets trade 24/7 and have no balance sheets or insider "
            "filings — reason from on-chain metrics, supply schedule, market "
            "microstructure, and news flow instead."
        )
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )


_CRYPTO_QUOTE_SUFFIXES = ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USD")
_CRYPTO_BASE_HINTS = {
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "MATIC",
    "DOT", "TRX", "LTC", "BCH", "ATOM", "NEAR", "APT", "ARB", "OP", "SUI",
    "AAVE", "UNI",
}


def detect_asset_class(ticker: str) -> str:
    """Return "crypto" or "equities" based on config + symbol shape.

    Respects an explicit ``asset_class`` setting in config; falls back to a
    cheap symbol-shape heuristic. Used so the Fundamentals Analyst (and any
    future asset-aware logic) can adapt its prompt without the operator
    having to set the config field manually for every run.
    """
    from tradingagents.dataflows.config import get_config

    cfg_value = (get_config().get("asset_class") or "auto").lower()
    if cfg_value in ("crypto", "equities", "equity"):
        return "equities" if cfg_value in ("equities", "equity") else "crypto"

    s = (ticker or "").upper().replace("-", "").replace("/", "").replace(".", "")
    if any(s.endswith(q) and len(s) > len(q) for q in _CRYPTO_QUOTE_SUFFIXES):
        return "crypto"
    if s in _CRYPTO_BASE_HINTS:
        return "crypto"
    return "equities"

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
