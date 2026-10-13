from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from futures_ml.data.processed import add_contract_segments
from futures_ml.evaluation.splits import assign_split, calculate_global_boundaries
from futures_ml.features.cross_market import add_cross_market_features
from futures_ml.features.phase2_1 import (
    EXPERIMENT_FEATURES,
    RAW_ANCHOR_FEATURES,
    RELATIVE_ANCHOR_FEATURES,
    add_relative_market_features,
)
from futures_ml.labels.future_returns import add_future_labels
from futures_ml.models.baselines import (
    make_preprocessor,
    make_ridge_pipeline,
    validate_feature_columns,
)
from futures_ml.models.phase2_1_training import validate_identical_sample_rows
from futures_ml.models.streaming_ridge import fit_streaming_ridge
from futures_ml.models.training import deterministic_even_sample

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def bars(minutes: list[int], instruments: list[int] | None = None) -> pl.DataFrame:
    instruments = instruments or [1] * len(minutes)
    close = [100.0 + minute for minute in minutes]
    return add_contract_segments(
        pl.DataFrame(
            {
                "timestamp": [START + timedelta(minutes=value) for value in minutes],
                "root": ["NQ"] * len(minutes),
                "instrument_id": instruments,
                "open": close,
                "high": [value + 1 for value in close],
                "low": [value - 1 for value in close],
                "close": close,
                "volume": [10 + value for value in minutes],
            }
        )
    )


def test_one_minute_target_uses_exact_t_plus_one() -> None:
    result = add_future_labels(bars([0, 1, 2]))
    assert result[0, "future_return_1m"] == pytest.approx(math.log(101 / 100))


def test_one_minute_target_does_not_use_next_available_row() -> None:
    result = add_future_labels(bars([0, 2]))
    assert result[0, "future_return_1m"] is None


def test_one_minute_target_cannot_cross_contract_segment() -> None:
    result = add_future_labels(bars([0, 1], [1, 2]))
    assert result[0, "future_return_1m"] is None


def test_phase2_1_feature_schemas_are_controlled() -> None:
    assert len(EXPERIMENT_FEATURES["A_1m"]) == 35
    assert len(RAW_ANCHOR_FEATURES) == 18
    assert len(RELATIVE_ANCHOR_FEATURES) == 18
    assert len(EXPERIMENT_FEATURES["B_1m"]) == 53
    assert len(EXPERIMENT_FEATURES["C_1m"]) == 53
    assert "NQ_ES_return_spread_5m" not in EXPERIMENT_FEATURES["B_1m"]
    assert set(RAW_ANCHOR_FEATURES).isdisjoint(RELATIVE_ANCHOR_FEATURES)


def test_relative_feature_uses_target_minus_anchor() -> None:
    frame = pl.DataFrame(
        {
            "root": ["CL"],
            "log_return_1m": [0.003],
            "log_return_5m": [0.006],
            "log_return_15m": [0.009],
            **{
                name: [0.001]
                for name in RAW_ANCHOR_FEATURES
            },
        }
    )
    result = add_relative_market_features(frame)
    assert result[0, "relative_to_ES_return_5m"] == pytest.approx(0.005)


def test_self_anchor_is_zero_only_when_both_values_are_valid() -> None:
    frame = pl.DataFrame(
        {
            "root": ["NQ", "NQ"],
            "log_return_1m": [0.003, 0.003],
            "log_return_5m": [0.006, 0.006],
            "log_return_15m": [0.009, 0.009],
            **{name: [0.001, 0.001] for name in RAW_ANCHOR_FEATURES},
        }
    ).with_columns(pl.Series("NQ_return_1m", [0.003, None]))
    result = add_relative_market_features(frame)
    assert result[0, "relative_to_NQ_return_1m"] == pytest.approx(0.0)
    assert result[1, "relative_to_NQ_return_1m"] is None


def test_exact_cross_market_join_does_not_forward_fill() -> None:
    own = bars([0, 2]).select("timestamp", "root")
    anchors = pl.DataFrame(
        {"timestamp": [START + timedelta(minutes=1)], "ES_return_1m": [0.1]}
    )
    result = add_cross_market_features(own, anchors)
    assert result["ES_return_1m"].null_count() == 2


def test_future_labels_are_rejected_from_x() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_feature_columns(["log_return_1m", "future_return_1m"])


def test_preprocessor_is_unchanged_by_validation_values() -> None:
    preprocessor = make_preprocessor(["x"], scale_numeric=True, include_root=False)
    preprocessor.fit(pl.DataFrame({"x": [1.0, 2.0, None]}))
    imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    before = imputer.statistics_.copy()
    preprocessor.transform(pl.DataFrame({"x": [None, 1_000_000.0]}))
    np.testing.assert_array_equal(before, imputer.statistics_)


def test_common_sampling_is_deterministic_and_identity_checked() -> None:
    frame = pl.DataFrame(
        {
            "root": ["NQ"] * 10,
            "timestamp": [START + timedelta(minutes=index) for index in range(10)],
            "value": range(10),
        }
    )
    first = deterministic_even_sample(frame, 4)
    second = deterministic_even_sample(frame, 4)
    assert first.equals(second)
    validate_identical_sample_rows({"A_1m": first, "B_1m": second, "C_1m": first})
    with pytest.raises(AssertionError, match="differ"):
        validate_identical_sample_rows({"A_1m": first, "B_1m": frame.tail(4)})


def test_purge_boundaries_remain_75_minutes() -> None:
    boundaries = calculate_global_boundaries(
        START,
        START + timedelta(days=10),
        purge_minutes=75,
    )
    near = pl.DataFrame(
        {
            "timestamp": [
                boundaries.train_validation_boundary - timedelta(minutes=74),
                boundaries.train_validation_boundary + timedelta(minutes=74),
                boundaries.validation_test_boundary - timedelta(minutes=74),
                boundaries.validation_test_boundary + timedelta(minutes=74),
            ]
        }
    )
    assert assign_split(near, boundaries)["split"].null_count() == 4


def test_streaming_full_ridge_matches_sklearn_ridge() -> None:
    frame = pl.DataFrame(
        {
            "root": ["NQ", "ES", "NQ", "ES", "NQ", "ES"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "z": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
        }
    )
    target = np.asarray([0.1, -0.1, 0.2, -0.2, 0.3, -0.3])
    sklearn_model = make_ridge_pipeline(["x", "z"], alpha=1.0, include_root=True)
    sklearn_model.fit(frame, target)
    streaming = fit_streaming_ridge(
        lambda: iter([(frame, target)]),
        features=("x", "z"),
        roots=("NQ", "ES"),
        medians=np.asarray([3.5, 3.5]),
        alpha=1.0,
    )
    np.testing.assert_allclose(
        streaming.predict(frame),
        sklearn_model.predict(frame),
        rtol=1e-10,
        atol=1e-12,
    )
