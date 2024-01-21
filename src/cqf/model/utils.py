import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader
from typing import List
from tqdm.auto import tqdm
from pathlib import Path

from cqf.model.dataset import TimeSeriesDataset1D
from cqf.utils import to_hash


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def run_train(model, criterion, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


def build_inference(
    model: torch.nn.Module,
    data_loader: TimeSeriesDataset1D,
    returns: pd.Series,
) -> pd.DataFrame:
    """

    Args:
        model (torch.nn.Module):
        data_loader (TimeSeriesDataset1D):
        returns (pd.Series):

    Returns:
        pd.DataFrame:
    """
    predictions = []
    label = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            predictions.append(model(inputs).numpy())
            label.append(labels.numpy())
    predictions = np.concatenate(predictions).reshape(
        -1,
    )
    label = np.concatenate(label).reshape(
        -1,
    )
    predictions = pd.DataFrame(
        {"date": data_loader.dataset.time_index, "pred": predictions, "label": label}
    )
    predictions = predictions.join(returns.rename("ret"), on="date")
    predictions["pred_label"] = (predictions.pred > 0.5).astype(int)
    predictions["pl"] = predictions.pred_label * predictions.ret
    return predictions.set_index("date")


def load_model(
    epoch,
    n_lookback=30,
    hidden_size=32,
    num_layers=3,
    seed=7,
) -> torch.nn.Module:
    params = {
        "epoch": epoch,
        "hidden_size": hidden_size,
        "n_lookback": n_lookback,
        "num_layers": num_layers,
        "seed": seed,
    }
    model_path = Path("artifacts").joinpath(f"model_{to_hash(params)}.pth")
    return torch.load(model_path)
