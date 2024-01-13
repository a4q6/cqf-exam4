import pandas as pd
import datetime
import shutil
from pathlib import Path
from typing import List, Union, Sequence, Iterable
from pandas import IndexSlice as ix

from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF

data_path = Path(__file__).parent.joinpath("storage")


def download_data(
    ticker: str,
    year: int,
) -> str:
    """Download 1min OHLC data from histdata.com to `data`"""
    path = dl(
        year=str(year),
        pair=ticker,
        platform=P.META_TRADER,
        time_frame=TF.ONE_MINUTE,
        output_directory=data_path.joinpath("zip"),
    )
    shutil.unpack_archive(path, extract_dir="data/")
    return path


def load_histdata(
    ticker: str = "USDJPY",
    year: Union[int, List[int]] = 2015,
) -> pd.DataFrame:
    """Load 1min-OHLC data. (Download from histdata if not exists.)"""
    if isinstance(year, Iterable):
        return pd.concat([load_histdata(ticker=ticker, year=_year) for _year in year])

    path = data_path.joinpath(f"DAT_MT_{ticker.upper()}_M1_{year}.csv")
    if not Path(path).exists():
        download_data(ticker=ticker, year=year)

    data = pd.read_csv(
        path, names=["date", "time", "open", "high", "low", "close", "_"]
    )
    data["timestamp"] = pd.to_datetime(data.date + " " + data.time).dt.tz_localize(
        "EST"
    )
    data["ticker"] = ticker.upper()
    data["date"] = pd.to_datetime(data["date"])
    return data[["timestamp", "ticker", "open", "high", "low", "close"]]
