import pandas as pd
import numpy as np
import datetime


def to_ny17_date(time_index: pd.DatetimeIndex) -> np.ndarray:
    # return date with Daystart=NY17:00
    index = pd.DatetimeIndex(pd.to_datetime(time_index))
    if str(index.tz) == "EST":
        return (index + datetime.timedelta(hours=7)).floor("D").tz_localize(None)
    elif str(index.tz) == "UTC":
        return (
            (index.tz_convert("US/Eastern") + datetime.timedelta(hours=7))
            .floor("D")
            .tz_localize(None)
        )
    elif index.tz is None:
        return (
            (
                index.tz_localize("UTC").tz_convert("US/Eastern")
                + datetime.timedelta(hours=7)
            )
            .floor("D")
            .tz_localize(None)
        )
    else:
        raise ValueError()


def is_ny17_weekdays(time_index: pd.DatetimeIndex) -> np.ndarray:
    return pd.DatetimeIndex(to_ny17_date(time_index)).weekday < 5


def is_around_eoy(time_index: pd.DatetimeIndex) -> np.ndarray:
    # return True from Christmas to 3rd Jan.
    dates = pd.DatetimeIndex(to_ny17_date(time_index))
    return ((dates.month == 1) & (dates.day <= 3)) | (
        (dates.month == 12) & (25 <= dates.day)
    )


def is_in_convid(index: pd.DatetimeIndex) -> np.ndarray:
    return ("2020.02.01" < index) & (index < "2020.04.01")


# def to_confidence_quantile(prediction: pd.Series) -> pd.Series:
#     # [!] just return quantile level for simplicity. qunatization should be done only with in-sample.
#     conf = pd.qcut(
#         prediction.pred.abs(),
#         q=np.arange(0, 1.01, 0.01).round(4),
#         labels=False,
#         duplicates="drop",
#     )
#     conf = (conf / conf.max()) * 100
#     return conf


# def calculate_macd(df: pd.DataFrame, short_window=12, long_window=26, signal_window=9):
#     df["shortEMA"] = (
#         df["close"].ewm(span=short_window, min_periods=1, adjust=False).mean()
#     )
#     df["longEMA"] = (
#         df["close"].ewm(span=long_window, min_periods=1, adjust=False).mean()
#     )
#     df["macd"] = df["shortEMA"] - df["longEMA"]
#     df["signal"] = (
#         df["macd"].ewm(span=signal_window, min_periods=1, adjust=False).mean()
#     )
#     return df.signal.rename("macd")


# def calculate_rsi(df: pd.DataFrame, window=14) -> pd.Series:
#     delta = df["close"].diff(1)
#     gain = delta.where(delta > 0, 0)
#     loss = -delta.where(delta < 0, 0)
#     avg_gain = gain.rolling(window=window, min_periods=1).mean()
#     avg_loss = loss.rolling(window=window, min_periods=1).mean()
#     rs = avg_gain / avg_loss
#     df["rsi"] = 100 - (100 / (1 + rs))
#     return df.rsi


def cleanse_ohlc_1min(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Cleanse raw OHLC data to force have 1440 recrods per day.

    Args:
        ohlc (pd.DataFrame): raw ohlc

    Returns:
        pd.DataFrame: cleansed ohlc data which has 1440 records per day.
    """
    normalized_ohlc = []
    for ticker in ohlc.ticker.unique():
        new_ohlc = (
            ohlc.query("ticker==@ticker")
            .set_index("timestamp")
            .sort_index()
            .resample("1min")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        )  # .assign(ticker=ticker)
        new_ohlc["close"] = new_ohlc.close.fillna(
            method="ffill"
        )  # fill null-close by latest close
        new_ohlc = new_ohlc.fillna(
            method="bfill", axis=1
        )  # fill null-ohl by latest close
        new_ohlc["ticker"] = ticker
        normalized_ohlc.append(new_ohlc)
    normalized_ohlc = pd.concat(normalized_ohlc).reset_index()
    normalized_ohlc = normalized_ohlc.loc[
        is_ny17_weekdays(normalized_ohlc.timestamp)
    ]  # drop weekends data
    return normalized_ohlc


def resample_ohlc_timeframe(
    ohlc: pd.DataFrame, timeframe="1h", offset_timeframe="0h"
) -> pd.DataFrame:
    """Convert OHLC's timeframe for example "1min" -> "1Hour".

    Args:
        ohlc (pd.DataFrame):
        timeframe (str, optional): Defaults to "1h".
        offset_timeframe (str, optional): resampling offset.

    Returns:
        pd.DataFrame: _description_
    """
    new_ohlc = pd.concat(
        [
            ohlc.query("ticker==@ticker")
            .set_index("timestamp")
            .shift(freq=offset_timeframe)
            .groupby(pd.Grouper(freq=timeframe))
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .shift(freq=f"-{offset_timeframe}")
            .assign(ticker=ticker)
            for ticker in ohlc.ticker.unique()
        ]
    ).reset_index()
    new_ohlc = new_ohlc.loc[is_ny17_weekdays(new_ohlc.timestamp)]
    return new_ohlc


def build_rolling_ohlc(ohlc: pd.DataFrame, timeframe="3h") -> pd.DataFrame:
    """Build rolling OHLC from OHLC bar"""
    ohlc_rolls = []
    for ticker in ohlc.ticker.unique():
        tmp_ohlc = ohlc.query("ticker==@ticker").set_index("timestamp")
        ohlc_roll = tmp_ohlc.rolling(timeframe).agg(
            {"open": lambda x: x.iloc[0], "high": "max", "low": "min"}
        )
        ohlc_roll = ohlc_roll.join(tmp_ohlc.close)[["open", "high", "low", "close"]]
        ohlc_roll = ohlc_roll.assign(ticker=ticker)
        ohlc_rolls.append(ohlc_roll)
    ohlc_rolls = pd.concat(ohlc_rolls).reset_index().assign(timeframe=timeframe)
    return ohlc_rolls


def attach_ohlc_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Add oc/hl/hige feature columns

    Args:
        ohlc (pd.DataFrame): DataFrame[open,high,low,close]

    Returns:
        pd.DataFrame: _description_
    """
    ohlc["oc_pct"] = (ohlc.close - ohlc.open) / ohlc.open * 1e2
    ohlc["hl_pct"] = (ohlc.high - ohlc.low) / ohlc.open * 1e2
    ohlc["hige_upper_pct"] = (
        (ohlc.high - ohlc[["open", "close"]].max(axis=1)).abs() / ohlc.open * 1e2
    )
    ohlc["hige_lower_pct"] = (
        (ohlc.low - ohlc[["open", "close"]].min(axis=1)).abs() / ohlc.open * 1e2
    )
    return ohlc
