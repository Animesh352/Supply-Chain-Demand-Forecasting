from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from evidently import Report
    from evidently.metric_preset import DataDriftPreset
except Exception:  # pragma: no cover
    Report = None
    DataDriftPreset = None


@dataclass
class FoldResult:
    fold: int
    mae: float
    rmse: float
    rmsse: float


def rolling_time_series_folds(
    df: pd.DataFrame,
    date_col: str,
    n_folds: int = 3,
    min_train_days: int = 365,
    val_days: int = 28,
) -> Generator[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp], None, None]:
    """Generate expanding-window train/validation date ranges."""
    unique_dates = np.array(sorted(df[date_col].unique()))
    if len(unique_dates) < min_train_days + n_folds * val_days:
        raise ValueError("Insufficient history for requested folds/min_train_days/val_days.")

    for fold_idx in range(n_folds):
        train_end_idx = min_train_days + fold_idx * val_days
        val_end_idx = train_end_idx + val_days

        train_start = pd.Timestamp(unique_dates[0])
        train_end = pd.Timestamp(unique_dates[train_end_idx - 1])
        val_start = pd.Timestamp(unique_dates[train_end_idx])
        val_end = pd.Timestamp(unique_dates[val_end_idx - 1])
        yield train_start, train_end, val_start, val_end


def rmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: np.ndarray,
) -> float:
    """Root Mean Squared Scaled Error using in-sample naive one-step scale."""
    numerator = np.mean((y_true - y_pred) ** 2)

    diffs = np.diff(y_train_history)
    denom = np.mean(diffs**2) if len(diffs) > 0 else 0.0

    if denom <= 1e-12:
        return float(np.sqrt(numerator))

    return float(np.sqrt(numerator / denom))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: np.ndarray,
) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmsse_value = rmsse(y_true, y_pred, y_train_history=y_train_history)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "rmsse": float(rmsse_value),
    }


def run_basic_drift_detection(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: List[str],
) -> Dict[str, float | bool | str]:
    """Runs a lightweight data drift check with Evidently."""
    if Report is None or DataDriftPreset is None:
        return {
            "drift_detected": False,
            "drift_share": 0.0,
            "status": "evidently_not_available",
        }

    ref = reference_df[columns].dropna().copy()
    cur = current_df[columns].dropna().copy()

    if ref.empty or cur.empty:
        return {
            "drift_detected": False,
            "drift_share": 0.0,
            "status": "insufficient_data",
        }

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    result = report.as_dict()

    drift_section = result.get("metrics", [{}])[0].get("result", {})
    dataset_drift = bool(drift_section.get("dataset_drift", False))
    drift_share = float(drift_section.get("share_of_drifted_columns", 0.0))

    return {
        "drift_detected": dataset_drift,
        "drift_share": drift_share,
        "status": "ok",
    }
