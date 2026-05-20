from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    detect_asset_class,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config


_EQUITY_SYSTEM_MESSAGE = (
    "You are a researcher tasked with analyzing fundamental information over the past week about a company."
    " Please write a comprehensive report of the company's fundamental information such as financial documents,"
    " company profile, basic company financials, and company financial history to gain a full view of the"
    " company's fundamental information to inform traders. Make sure to include as much detail as possible."
    " Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
    " Make sure to append a Markdown table at the end of the report to organize key points in the report,"
    " organized and easy to read."
    " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`,"
    " `get_cashflow`, and `get_income_statement` for specific financial statements."
)

# Crypto has no balance sheet / income / cashflow. The same tool names route
# through interface.py to CoinGecko endpoints that return the closest
# analogues — described inline below so the model knows what each call
# actually delivers and frames the report in crypto terms.
_CRYPTO_SYSTEM_MESSAGE = (
    "You are a researcher analyzing the fundamentals of a cryptocurrency asset."
    " Crypto assets have no balance sheets, income statements, or cash flows in the equity sense;"
    " instead, write a report covering: protocol overview and category, market data (price, market cap,"
    " volume, ATH/ATL), supply schedule (circulating / total / max, dilution risk), market microstructure"
    " (where it trades, liquidity concentration, spreads), developer activity (commits, PRs, contributors),"
    " and community signals. Provide specific, actionable insights with supporting evidence."
    " Use the available tools — note that for crypto symbols these route to CoinGecko under the hood:"
    " `get_fundamentals` returns a market + supply + community snapshot,"
    " `get_balance_sheet` returns market-microstructure / venue-concentration data,"
    " `get_cashflow` returns developer activity (commits, PR throughput),"
    " `get_income_statement` returns supply metrics and rolling returns."
    " Append a Markdown table at the end summarizing the key takeaways."
)


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker)
        asset_class = detect_asset_class(ticker)

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        body = (
            _CRYPTO_SYSTEM_MESSAGE if asset_class == "crypto"
            else _EQUITY_SYSTEM_MESSAGE
        )
        system_message = (body + get_language_instruction(),)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
