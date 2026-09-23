"""Transparent zero, Ridge, and histogram-gradient-boosting baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from futures_ml.features.pipeline import FORBIDDEN_MODEL_COLUMNS


@dataclass
class ZeroRegressor:
    """Reference estimator that always predicts no future price change."""

    def fit(self, features: Any, target: Any) -> ZeroRegressor:
        return self

    def predict(self, features: Any) -> np.ndarray:
        return np.zeros(len(features), dtype=float)


def validate_feature_columns(features: list[str] | tuple[str, ...]) -> None:
    """Reject labels and diagnostic targets before constructing X."""
    forbidden = sorted(set(features) & set(FORBIDDEN_MODEL_COLUMNS))
    if forbidden:
        raise ValueError(f"Target/label columns are forbidden from X: {forbidden}")


def make_preprocessor(
    numeric_features: list[str], *, scale_numeric: bool, include_root: bool
) -> ColumnTransformer:
    """Create train-fitted numeric and root transformations."""
    validate_feature_columns(numeric_features)
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, Any, list[str]]] = [
        ("numeric", Pipeline(numeric_steps), numeric_features)
    ]
    if include_root:
        transformers.append(
            (
                "root",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["root"],
            )
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def make_ridge_pipeline(
    numeric_features: list[str], *, alpha: float = 1.0, include_root: bool = True
) -> Pipeline:
    """Ridge receives train-median imputation, standardization, and root one-hot."""
    return Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(numeric_features, scale_numeric=True, include_root=include_root),
            ),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def make_hist_gradient_boosting_pipeline(
    numeric_features: list[str],
    *,
    include_root: bool = True,
    random_seed: int = 42,
    **parameters: Any,
) -> Pipeline:
    """HGB receives train-median imputation and root one-hot, but no scaling."""
    return Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(numeric_features, scale_numeric=False, include_root=include_root),
            ),
            (
                "model",
                HistGradientBoostingRegressor(random_state=random_seed, **parameters),
            ),
        ]
    )
