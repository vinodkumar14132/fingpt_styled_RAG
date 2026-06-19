
from langchain_core.tools import tool

@tool
def calculator(expression: str):
    """Useful for mathematical calculations"""
    return str(eval(expression))
@tool
def stock_price(ticker: str):
    """Useful for stock prices"""

    import yfinance as yf

    stock = yf.Ticker(ticker)

    return str(
        stock.info.get(
            "currentPrice",
            "Not Available"
        )
    )
@tool
def transcript_search(query: str):
    """Useful for TCS transcript questions"""

    return "AI revenue crossed $2.3 billion"


print(calculator.invoke("12*25"))

print(
    transcript_search.invoke(
        "What is TCS revenue?"
    )
)