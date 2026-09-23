from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from futures_ml.data.processed import add_contract_segments
from futures_ml.evaluation.splits import (
    assign_split,
    calculate_global_boundaries,
    split_expression,
)
from futures_ml.features.cross_market import add_cross_market_features
from futures_ml.features.ohlcv_features import (
    add_exact_log_returns,
    add_time_features,
    add_volume_features,
)
from futures_ml.labels.future_returns import add_future_labels
from futures_ml.models.baselines import make_preprocessor, validate_feature_columns

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


def test_five_minute_return_feature_uses_t_minus_five() -> None:
    nanosecond_bars = bars(list(range(7))).with_columns(
        pl.col("timestamp").cast(pl.Datetime("ns", "UTC"))
    )
    result = add_exact_log_returns(nanosecond_bars, (5,))
    assert result[5, "log_return_5m"] == pytest.approx(math.log(105 / 100))


def test_five_minute_target_uses_t_plus_five() -> None:
    nanosecond_bars = bars(list(range(7))).with_columns(
        pl.col("timestamp").cast(pl.Datetime("ns", "UTC"))
    )
    result = add_future_labels(nanosecond_bars)
    assert result[0, "future_return_5m"] == pytest.approx(math.log(105 / 100))


def test_missing_timestamps_do_not_shift_horizons() -> None:
    result = add_exact_log_returns(bars([0, 1, 2, 3, 4, 6, 7, 8, 9, 10]), (5,))
    row = result.filter(pl.col("timestamp") == START + timedelta(minutes=10))
    assert row[0, "log_return_5m"] is None


def test_rolling_features_never_use_future_rows() -> None:
    original = bars(list(range(25)))
    modified = original.with_columns(
        pl.when(pl.col("timestamp") == START + timedelta(minutes=24))
        .then(1_000_000)
        .otherwise(pl.col("volume"))
        .alias("volume")
    )
    before = add_volume_features(original)
    after = add_volume_features(modified)
    assert before[23, "rolling_volume_mean_20"] == after[23, "rolling_volume_mean_20"]


def test_feature_cannot_cross_contract_segment() -> None:
    result = add_exact_log_returns(bars(list(range(8)), [1, 1, 1, 1, 1, 2, 2, 2]), (5,))
    assert result[5, "log_return_5m"] is None


def test_target_cannot_cross_contract_segment() -> None:
    result = add_future_labels(bars(list(range(8)), [1, 1, 1, 1, 1, 2, 2, 2]))
    assert result[0, "future_return_5m"] is None


def test_target_columns_cannot_enter_feature_matrix() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_feature_columns(["log_return_1m", "future_return_5m"])


def test_preprocessing_statistics_are_fit_only_on_train() -> None:
    preprocessor = make_preprocessor(["x"], scale_numeric=False, include_root=False)
    preprocessor.fit(pl.DataFrame({"x": [1.0, 2.0, None]}))
    imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.statistics_[0] == pytest.approx(1.5)
    assert preprocessor.transform(pl.DataFrame({"x": [None, 10_000.0]}))[0, 0] == pytest.approx(1.5)


def test_cross_market_features_never_use_later_timestamp() -> None:
    own = bars([0, 1]).select("timestamp", "root")
    anchors = pl.DataFrame(
        {
            "timestamp": [START + timedelta(minutes=1)],
            "ES_return_5m": [0.25],
        }
    )
    result = add_cross_market_features(own, anchors)
    assert result[0, "ES_return_5m"] is None
    assert result[1, "ES_return_5m"] == 0.25


def test_cross_market_join_is_exact_not_asof() -> None:
    own = bars([0, 2]).select("timestamp", "root")
    anchors = pl.DataFrame(
        {"timestamp": [START + timedelta(minutes=1)], "ES_return_1m": [0.1]}
    )
    result = add_cross_market_features(own, anchors)
    assert result["ES_return_1m"].null_count() == 2


def test_split_boundaries_are_purged() -> None:
    boundaries = calculate_global_boundaries(START, START + timedelta(days=10), purge_minutes=75)
    near_boundary = pl.DataFrame(
        {
            "timestamp": [
                boundaries.train_validation_boundary - timedelta(minutes=1),
                boundaries.train_validation_boundary + timedelta(minutes=1),
            ]
        }
    )
    assert assign_split(near_boundary, boundaries)["split"].null_count() == 2


def test_test_period_remains_separate() -> None:
    boundaries = calculate_global_boundaries(START, START + timedelta(days=10), purge_minutes=75)
    frame = pl.DataFrame(
        {
            "timestamp": [
                boundaries.validation_test_boundary - timedelta(minutes=76),
                boundaries.validation_test_boundary + timedelta(minutes=76),
            ]
        }
    )
    assert frame.filter(split_expression("validation", boundaries)).height == 1
    assert frame.filter(split_expression("test", boundaries)).height == 1


def test_roll_event_and_segment_ids_are_contiguous() -> None:
    result = bars([0, 1, 2, 3, 4], [10, 10, 20, 20, 10])
    assert result["contract_segment_id"].to_list() == [1, 1, 2, 2, 3]
    assert result["roll_event"].to_list() == [False, False, True, False, True]


def test_time_cycle_uses_full_day_without_integer_overflow() -> None:
    frame = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 6, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 2, 18, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 3, 0, 0, tzinfo=timezone.utc),
            ]
        }
    )
    result = add_time_features(frame)
    assert result["hour_sin"].min() <= -0.95
    assert result["hour_sin"].max() >= 0.95
