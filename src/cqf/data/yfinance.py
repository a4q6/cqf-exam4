import pandas as pd
import yfinance
from cqf.const import DATA_ST, DATA_ET


def get_daily_ohlc(
    ticker="JPY=X", st: str = DATA_ST, et: str = DATA_ET
) -> pd.DataFrame:
    """Query given ticker's OHLC data from YFinance.

    Args:
        ticker (str, optional): Defaults to "JPY=X".
        st (str, optional): Defaults to DATA_ST.
        et (str, optional): Defaults to DATA_ET.

    Returns:
        pd.DataFrame: OHLCV dataframe.
    """
    ohlc = (
        yfinance.Ticker(ticker)
        .history(start=st, end=et)
        .reset_index()
        .rename(lambda x: x.lower(), axis=1)
        .set_index("date")
    )
    # ohlc["ticker"] = "".join(filter(str.isalnum, ticker))
    ohlc["ticker"] = ticker
    return ohlc
