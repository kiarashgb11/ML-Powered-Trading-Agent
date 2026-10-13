from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl

from futures_ml.evaluation.walk_forward import (
    build_walk_forward_folds,
    fold_evaluation_expression,
    fold_training_expression,
)
from futures_ml.features.ohlcv_features import MODEL_FEATURES_A
from futures_ml.features.phase2_1 import EXPERIMENT_FEATURES, RAW_ANCHOR_FEATURES
from futures_ml.models.baselines import make_hist_gradient_boosting_pipeline

START = datetime(2021, 9, 23, tzinfo=timezone.utc)
BOUNDARIES = tuple(
    datetime(year, month, 23, tzinfo=timezone.utc)
    for year, month in (
        (2022, 9),
        (2023, 3),
        (2023, 9),
        (2024, 3),
        (2024, 9),
        (2025, 3),
        (2025, 9),
        (2026, 3),
        (2026, 9),
    )
)


def folds():
    return build_walk_forward_folds(
        data_start=START,
        evaluation_boundaries=BOUNDARIES,
        purge_minutes=75,
    )


def test_fold_definitions_are_deterministic() -> None:
    assert folds() == folds()
    assert len(folds()) == 8


def test_every_evaluation_strictly_follows_training_with_purge() -> None:
    for fold in folds():
        assert fold.training_end_exclusive < fold.evaluation_start
        assert fold.evaluation_start - fold.training_end_exclusive == timedelta(minutes=150)


def test_evaluation_folds_do_not_overlap() -> None:
    defined = folds()
    for previous, current in zip(defined, defined[1:], strict=False):
        assert previous.evaluation_end_exclusive <= current.evaluation_start
        assert previous.training_end_exclusive < current.training_end_exclusive


def test_fold_filters_never_share_a_timestamp() -> None:
    fold = folds()[0]
    frame = pl.DataFrame(
        {
            "timestamp": [
                fold.training_end_exclusive - timedelta(minutes=1),
                fold.training_end_exclusive,
                fold.boundary,
                fold.evaluation_start - timedelta(minutes=1),
                fold.evaluation_start,
            ]
        }
    )
    train = frame.filter(fold_training_expression(fold))
    evaluation = frame.filter(fold_evaluation_expression(fold))
    assert train.height == 1
    assert evaluation.height == 1
    assert set(train["timestamp"].to_list()).isdisjoint(evaluation["timestamp"].to_list())


def test_phase2_2_features_are_exactly_frozen_phase2_1_features() -> None:
    assert EXPERIMENT_FEATURES["A_1m"] == MODEL_FEATURES_A
    assert EXPERIMENT_FEATURES["B_1m"] == (*MODEL_FEATURES_A, *RAW_ANCHOR_FEATURES)
    assert len(EXPERIMENT_FEATURES["A_1m"]) == 35
    assert len(EXPERIMENT_FEATURES["B_1m"]) == 53


def test_preprocessing_is_refit_as_distinct_fold_state() -> None:
    first = make_hist_gradient_boosting_pipeline(
        ["x"],
        include_root=True,
        random_seed=42,
        learning_rate=0.08,
        max_iter=2,
        max_leaf_nodes=3,
        min_samples_leaf=2,
        l2_regularization=0.1,
    )
    second = make_hist_gradient_boosting_pipeline(
        ["x"],
        include_root=True,
        random_seed=42,
        learning_rate=0.08,
        max_iter=2,
        max_leaf_nodes=3,
        min_samples_leaf=2,
        l2_regularization=0.1,
    )
    first_preprocessor = first.named_steps["preprocessor"]
    second_preprocessor = second.named_steps["preprocessor"]
    first_preprocessor.fit(
        pl.DataFrame({"root": ["NQ"] * 4, "x": [0.0, 1.0, None, 3.0]})
    )
    second_preprocessor.fit(
        pl.DataFrame({"root": ["NQ"] * 4, "x": [100.0, 101.0, None, 103.0]}),
    )
    first_median = first_preprocessor.named_transformers_["numeric"].named_steps[
        "imputer"
    ].statistics_[0]
    second_median = second_preprocessor.named_transformers_["numeric"].named_steps[
        "imputer"
    ].statistics_[0]
    assert first is not second
    assert first_median == 1.0
    assert second_median == 101.0
