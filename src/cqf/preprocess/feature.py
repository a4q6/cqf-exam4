import pandas as pd
import numpy as np


def calc_ohlcv_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    assert ohlc.ticker.nunique() == 1
    ticker = ohlc.ticker.iloc[0]
    features = (
        ((ohlc.close - ohlc.open) / ohlc.open)
        .rename(f"OC_{ticker}")
        .to_frame()
        .join(((ohlc.high - ohlc.low) / ohlc.open).rename(f"HL_{ticker}"))
        # .join(((ohlc.close - ohlc.low) / ohlc.open).abs().rename(f"LC_{ticker}"))
        # .join(((ohlc.close - ohlc.high) / ohlc.open).abs().rename(f"HC_{ticker}"))
        .join(
            (
                (ohlc.high - np.maximum(ohlc.open, ohlc.close)).abs()
                - (ohlc.low - np.minimum(ohlc.open, ohlc.close)).abs()
            )
            .div(ohlc.open)
            .rename(f"OS_{ticker}")
        )
    )
    if ohlc.volume.eq(0).sum() < ohlc.shape[0] * 0.9:
        features = features.join(ohlc.volume.rename(f"V_{ticker}"))
    else:
        features = features.assign(**{f"V_{ticker}": 0})
    features.columns.name = "feature"
    return features
