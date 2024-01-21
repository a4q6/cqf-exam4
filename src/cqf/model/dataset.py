import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        n_lookback: int = 20,
        target_months: List[int] = list(range(1, 13)),
    ):
        assert len(x) == len(y)
        assert isinstance(x.index, pd.DatetimeIndex)
        assert isinstance(y.index, pd.DatetimeIndex)
        self.n_lookback = n_lookback
        self.x = x
        self.y = y

        # cast to timeseries array
        self.tickers = sorted(x.columns.get_level_values(0).unique())
        self.y_array = y.values.reshape(1, -1)  # y: ndarray[1, #Time]
        self.x_array = np.concatenate(  # X: ndarray[#Ticker, #Time, #Feature]
            [
                x[ticker].values.reshape(1, x[ticker].shape[0], -1)
                for ticker in self.tickers
            ]
        )

        # create target month mapper
        self.time_index = y.index[n_lookback - 1 :]
        self.sample_flag = self.time_index.month.isin(target_months)
        self.n_samples = int(self.sample_flag.sum())
        self.sample_index_map = (
            # torch sample index(int) -> time series index(int)
            pd.Series(self.sample_flag)
            .cumsum()
            .sub(1)
            .to_frame("sample_index")
            .query("0<=sample_index")
            .reset_index()
            .groupby("sample_index")
            .first()["index"]
            .to_dict()
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i: int):
        idx = self.sample_index_map[i]
        features = self.x_array[:, idx : idx + self.n_lookback, :]
        label = self.y_array[:, idx + self.n_lookback - 1]
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return features, label


class TimeSeriesDataset1D(Dataset):
    def __init__(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        n_lookback: int = 20,
        target_months: List[int] = list(range(1, 13)),
    ):
        assert len(x) == len(y)
        assert isinstance(x.index, pd.DatetimeIndex)
        assert isinstance(y.index, pd.DatetimeIndex)
        self.n_lookback = n_lookback
        self.x = x
        self.y = y

        # cast to timeseries array
        self.tickers = sorted(x.columns.get_level_values(0).unique())
        self.y_array = y.values.reshape(1, -1)  # y: ndarray[1, #Time]
        self.x_array = x.values  # x: [#Time, #Ticker * #FeaturePerTicker]

        # create target month mapper
        self.time_index = y.index[n_lookback - 1 :]
        self.sample_flag = self.time_index.month.isin(target_months)
        self.n_samples = int(self.sample_flag.sum())
        self.sample_index_map = (
            # torch sample index(int) -> time series index(int)
            pd.Series(self.sample_flag)
            .cumsum()
            .sub(1)
            .to_frame("sample_index")
            .query("0<=sample_index")
            .reset_index()
            .groupby("sample_index")
            .first()["index"]
            .to_dict()
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i: int):
        idx = self.sample_index_map[i]
        features = self.x_array[idx : idx + self.n_lookback]
        label = self.y_array[:, idx + self.n_lookback - 1]
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return features, label
