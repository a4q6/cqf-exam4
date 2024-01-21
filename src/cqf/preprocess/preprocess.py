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


def apply_rolling_lognormalize1(data: pd.DataFrame, window="90d") -> pd.DataFrame:
    assert isinstance(data.index, pd.DatetimeIndex)
    log_data = np.log(data.add(1))
    log_normed = (log_data - log_data.rolling(window).mean()) / log_data.rolling(
        window
    ).std()
    log_normed_clipped = np.clip(log_normed, -4, +4)
    return log_normed_clipped


def apply_rolling_lognormalize2(data: pd.DataFrame, window="90d") -> pd.DataFrame:
    log = np.log(data.add(1))
    log_norm = (log - log.rolling(window).mean()) / log.rolling(window).std()
    return log_norm
