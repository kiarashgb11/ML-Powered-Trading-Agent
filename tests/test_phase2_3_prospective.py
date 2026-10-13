from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from futures_ml.data.processed import add_contract_segments
from futures_ml.features.cross_market import add_cross_market_features
from futures_ml.features.ohlcv_features import MODEL_FEATURES_A
from futures_ml.features.phase2_1 import (
    EXPERIMENT_FEATURES,
    RAW_ANCHOR_FEATURES,
    RELATIVE_ANCHOR_FEATURES,
)
from futures_ml.labels.future_returns import add_future_labels
from futures_ml.models.phase2_1_training import SOURCE_FEATURES, SOURCE_METADATA, TARGET
from futures_ml.models.prospective import (
    PHASE2_3_EXPERIMENTS,
    PHASE2_3_ROOTS,
    append_unique_events,
    attach_outcomes,
    causal_prediction_eligibility_expression,
    historical_training_expression,
    predict_frozen_model,
    prospective_expression,
    read_event_log,
    selected_row_hash,
    sha256_file,
    verify_frozen_artifacts,
)

START = datetime(2026, 9, 23, tzinfo=timezone.utc)


class PredictOnlyModel:
    def __init__(self) -> None:
        self.seen_columns: list[str] = []

    def fit(self, *_args, **_kwargs):
        raise AssertionError("Prospective evaluation must never call fit")

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        self.seen_columns = frame.columns
        return np.arange(frame.height, dtype=float) / 1_000_000


def source_frame(minutes: list[int]) -> pl.DataFrame:
    rows = len(minutes)
    return pl.DataFrame(
        {
            "timestamp": [START + timedelta(minutes=value) for value in minutes],
            "root": ["ZN"] * rows,
            "instrument_id": [101] * rows,
            "contract_segment_id": [7] * rows,
            **{feature: [0.0] * rows for feature in SOURCE_FEATURES},
            TARGET: [0.9] * rows,
        }
    )


def bars(minutes: list[int], instruments: list[int] | None = None) -> pl.DataFrame:
    instruments = instruments or [1] * len(minutes)
    close = [100.0 + minute for minute in minutes]
    return add_contract_segments(
        pl.DataFrame(
            {
                "timestamp": [START + timedelta(minutes=value) for value in minutes],
                "root": ["ZN"] * len(minutes),
                "instrument_id": instruments,
                "open": close,
                "high": [value + 1 for value in close],
                "low": [value - 1 for value in close],
                "close": close,
                "volume": [10 + value for value in minutes],
            }
        )
    )


def test_phase2_3_config_freezes_only_preselected_roots_and_hgb() -> None:
    config = yaml.safe_load(Path("config/phase2_3.yaml").read_text(encoding="utf-8"))
    assert tuple(config["roots"]) == PHASE2_3_ROOTS == ("ZN", "ZC", "ZB")
    assert tuple(config["experiments"]) == PHASE2_3_EXPERIMENTS
    assert config["training_rows_per_root"] == 200_000
    assert config["modeling"] == {
        "random_seed": 42,
        "hist_gradient_boosting": {
            "learning_rate": 0.08,
            "max_iter": 150,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 100,
            "l2_regularization": 0.1,
        },
    }


def test_phase2_3_feature_sets_are_frozen_and_exclude_c() -> None:
    assert EXPERIMENT_FEATURES["A_1m"] == MODEL_FEATURES_A
    assert EXPERIMENT_FEATURES["B_1m"] == (*MODEL_FEATURES_A, *RAW_ANCHOR_FEATURES)
    assert len(EXPERIMENT_FEATURES["A_1m"]) == 35
    assert len(EXPERIMENT_FEATURES["B_1m"]) == 53
    assert set(EXPERIMENT_FEATURES["B_1m"]).isdisjoint(RELATIVE_ANCHOR_FEATURES)


def test_target_remains_exact_next_minute_and_same_segment() -> None:
    exact = add_future_labels(bars([0, 1, 3]))
    assert exact[0, TARGET] == pytest.approx(math.log(101 / 100))
    assert exact[1, TARGET] is None
    rollover = add_future_labels(bars([0, 1], [1, 2]))
    assert rollover[0, TARGET] is None


def test_anchor_join_remains_exact_time() -> None:
    own = source_frame([1, 3]).select("timestamp", "root")
    anchor = pl.DataFrame(
        {"timestamp": [START + timedelta(minutes=2)], "ES_return_1m": [0.25]}
    )
    assert add_cross_market_features(own, anchor)["ES_return_1m"].null_count() == 2


def test_cutoff_expressions_strictly_separate_training_and_prospective() -> None:
    frame = source_frame([-1, 0, 1]).with_columns(
        pl.lit(1.0).alias("rolling_vwap_240m")
    )
    historical = frame.filter(historical_training_expression(START))
    prospective = frame.filter(prospective_expression(START))
    assert historical["timestamp"].to_list() == [START - timedelta(minutes=1)]
    assert prospective["timestamp"].to_list() == [START + timedelta(minutes=1)]


def test_prediction_eligibility_uses_only_causal_warmup_state() -> None:
    frame = source_frame([1, 2]).with_columns(
        pl.Series("rolling_vwap_240m", [None, 1.0])
    )
    eligible = frame.filter(causal_prediction_eligibility_expression())
    assert eligible["timestamp"].to_list() == [START + timedelta(minutes=2)]


def test_prediction_uses_only_frozen_features_and_never_refits() -> None:
    model = PredictOnlyModel()
    frame = source_frame([1, 2])
    first = predict_frozen_model(
        model,
        frame,
        model_id="ZN-A-final",
        experiment="A_1m",
        data_version="test",
        recorded_at_utc=START,
    )
    changed_labels = frame.with_columns(pl.lit(-999.0).alias(TARGET))
    second = predict_frozen_model(
        model,
        changed_labels,
        model_id="ZN-A-final",
        experiment="A_1m",
        data_version="test",
        recorded_at_utc=START,
    )
    assert model.seen_columns == list(EXPERIMENT_FEATURES["A_1m"])
    assert TARGET not in model.seen_columns
    assert first["prediction"].to_list() == second["prediction"].to_list()


def test_prediction_outcome_attachment_is_exact_and_deterministic() -> None:
    frame = source_frame([1, 2])
    predictions = predict_frozen_model(
        PredictOnlyModel(),
        frame,
        model_id="ZN-B-final",
        experiment="B_1m",
        data_version="test",
        recorded_at_utc=START,
    )
    outcomes = frame.select(*SOURCE_METADATA, TARGET).filter(
        pl.col("timestamp") == START + timedelta(minutes=1)
    )
    first = attach_outcomes(predictions, outcomes, recorded_at_utc=START)
    second = attach_outcomes(predictions, outcomes, recorded_at_utc=START)
    assert first.equals(second)
    assert first.height == 1
    assert first[0, TARGET] == 0.9
    assert first[0, "timestamp"] == START + timedelta(minutes=1)


def test_append_only_log_never_rewrites_existing_parts(tmp_path: Path) -> None:
    predictions = predict_frozen_model(
        PredictOnlyModel(),
        source_frame([1]),
        model_id="ZN-A-final",
        experiment="A_1m",
        data_version="test",
        recorded_at_utc=START,
    )
    assert append_unique_events(predictions, tmp_path) == 1
    original = next(tmp_path.glob("part-*.parquet"))
    original_hash = sha256_file(original)
    assert append_unique_events(predictions, tmp_path) == 0
    later = predict_frozen_model(
        PredictOnlyModel(),
        source_frame([2]),
        model_id="ZN-A-final",
        experiment="A_1m",
        data_version="test",
        recorded_at_utc=START,
    )
    assert append_unique_events(later, tmp_path) == 1
    assert sha256_file(original) == original_hash
    assert len(list(tmp_path.glob("part-*.parquet"))) == 2
    assert read_event_log(tmp_path).height == 2


def test_frozen_hash_verification_does_not_change_during_prediction(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    rows_path = tmp_path / "rows.parquet"
    model_path.write_bytes(b"immutable model")
    rows_path.write_bytes(b"immutable rows")
    manifest = {
        "models": [
            {
                "model_id": "ZN-A-final",
                "artifact_path": str(model_path),
                "artifact_sha256": sha256_file(model_path),
                "selected_rows_path": str(rows_path),
                "selected_rows_artifact_sha256": sha256_file(rows_path),
            }
        ]
    }
    verify_frozen_artifacts(manifest)
    before = (sha256_file(model_path), sha256_file(rows_path))
    predict_frozen_model(
        PredictOnlyModel(),
        source_frame([1]),
        model_id="ZN-A-final",
        experiment="A_1m",
        data_version="test",
        recorded_at_utc=START,
    )
    verify_frozen_artifacts(manifest)
    assert before == (sha256_file(model_path), sha256_file(rows_path))


def test_selected_training_row_hash_is_deterministic() -> None:
    frame = source_frame([-2, -1]).select(*SOURCE_METADATA, *SOURCE_FEATURES, TARGET)
    assert selected_row_hash(frame) == selected_row_hash(frame.clone())
