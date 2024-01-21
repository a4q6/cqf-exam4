import pandas as pd
import numpy as np
from typing import Iterable
from sklearn.metrics import log_loss, roc_auc_score

from cqf.preprocess.preprocess import to_ny17_date


def calc_annu_sharpe(pls: pd.Series) -> float:
    dates = to_ny17_date(pls.index)
    daily_pls = pls.groupby(dates).resample("1d").sum()
    return daily_pls.mean() / (daily_pls.std() + 1e-7) * np.sqrt(252)


def prediction_summary(predictions: pd.DataFrame):
    return pd.DataFrame(
        [
            {
                "up_signal_ratio": predictions.pred_label.sum() / predictions.shape[0],
                "bce_loss": log_loss(predictions.label, predictions.pred),
                "avg_pl": predictions.pl.mean(),
                "annu_sharpe": calc_annu_sharpe(predictions.pl),
                "spearman_corr": predictions[["pred", "ret"]]
                .corr(method="spearman")
                .iloc[0, 1],
                "auc": roc_auc_score(predictions.label, predictions.pred),
            }
        ]
    )
