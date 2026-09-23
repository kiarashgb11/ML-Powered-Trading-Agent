"""Regression and prediction-ranking diagnostics for predictive signal only."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2 or np.std(actual) == 0 or np.std(predicted) == 0:
        return math.nan
    return float(np.corrcoef(actual, predicted)[0, 1])


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    """Compute the Phase 2 metric contract on finite values only."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[finite]
    predicted = predicted[finite]
    if not len(actual):
        return {name: math.nan for name in ("mae", "rmse", "r2", "pearson", "spearman", "direction_accuracy")} | {"observations": 0}
    return {
        "observations": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "pearson": _correlation(actual, predicted),
        "spearman": (
            float(spearmanr(actual, predicted).statistic)
            if len(actual) > 1 and np.std(actual) > 0 and np.std(predicted) > 0
            else math.nan
        ),
        "direction_accuracy": float(np.mean(np.sign(actual) == np.sign(predicted))),
    }


def prediction_deciles(actual: np.ndarray, predicted: np.ndarray) -> list[dict[str, Any]]:
    """Summarize realized returns across rank-based prediction deciles."""
    order = np.argsort(predicted, kind="stable")
    groups = np.array_split(order, 10)
    rows: list[dict[str, Any]] = []
    for index, positions in enumerate(groups, start=1):
        rows.append(
            {
                "decile": index,
                "observations": int(len(positions)),
                "mean_prediction": float(np.mean(predicted[positions])) if len(positions) else math.nan,
                "mean_actual": float(np.mean(actual[positions])) if len(positions) else math.nan,
            }
        )
    return rows
