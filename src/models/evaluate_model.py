from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true_array, y_pred_array))
    rmse = float(math.sqrt(mean_squared_error(y_true_array, y_pred_array)))
    r2 = float(r2_score(y_true_array, y_pred_array))
    safe_mask = np.abs(y_true_array) > 1e-6
    mape = float(np.mean(np.abs((y_true_array[safe_mask] - y_pred_array[safe_mask]) / y_true_array[safe_mask])) * 100)
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def terminal_level_metrics(
    metadata: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    frame = metadata[["terminal_id", "forecast_horizon"]].copy()
    frame["actual"] = np.asarray(y_true, dtype=float)
    frame["prediction"] = np.asarray(y_pred, dtype=float)
    results: dict[str, dict[str, float]] = {}
    for terminal_id, group in frame.groupby("terminal_id"):
        results[str(terminal_id)] = regression_metrics(group["actual"], group["prediction"])
    for horizon, group in frame.groupby("forecast_horizon"):
        results[str(horizon)] = regression_metrics(group["actual"], group["prediction"])
    return results

