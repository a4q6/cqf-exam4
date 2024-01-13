import pandas as pd
import numpy as np
from cqf.preprocess.preprocess import to_ny17_date
from typing import Iterable


def calc_annu_sharpe(pls: pd.Series) -> float:
    dates = to_ny17_date(pls.index)
    daily_pls = pls.groupby(dates).resample("1d").sum()
    return daily_pls.mean() / daily_pls.std() * np.sqrt(252)


# def calc_max_draw_down(pls: pd.Series) -> float:
#     return (pls.cumsum().cummax() - pls.cumsum()).max()


# def calc_performance_summary(
#     signals: pd.DataFrame,
#     pl_name="pl",
#     signal_name="pred",
#     confidence_threshold=[0, 90],
# ) -> pd.DataFrame:
#     if isinstance(confidence_threshold, Iterable):
#         results = pd.concat(
#             [
#                 calc_performance_summary(
#                     signals,
#                     pl_name=pl_name,
#                     signal_name=signal_name,
#                     confidence_threshold=theta,
#                 ).set_axis([100 - theta])
#                 for theta in confidence_threshold
#             ]
#         )
#         results.index.name = "coverage"
#         return results
#     else:
#         _subset = (
#             signals.loc[
#                 confidence_threshold <= to_confidence_quantile(signals[signal_name])
#             ]
#             if signal_name is not None
#             else signals
#         )
#         return (
#             _subset[pl_name]
#             .describe()
#             .drop(["std", "75%", "25%"])
#             .rename({"50%": "med"})
#             .rename(lambda x: f"pl_{x}")
#             .to_frame("")
#             .T.assign(
#                 pl_total=_subset[pl_name].sum(),
#                 hit_ratio=calc_hit_ratio(_subset[pl_name]),
#                 sharpe=calc_annu_sharpe(_subset[pl_name]),
#                 max_dd=calc_max_draw_down(_subset[pl_name]),
#             )
#             .round(3)
#         )
