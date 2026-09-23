"""External Parquet orchestration for Phase 2 feature datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from futures_ml.data.processed import _atomic_write_parquet
from futures_ml.features.cross_market import CROSS_MARKET_FEATURES
from futures_ml.features.ohlcv_features import MODEL_FEATURES_A, build_own_market_features
from futures_ml.labels.future_returns import add_future_labels

LABEL_COLUMNS = ("future_return_1m", "future_return_5m", "future_return_15m")
METADATA_COLUMNS = (
    "timestamp",
    "root",
    "instrument_id",
    "contract_segment_id",
    "roll_event",
    "segment_start",
    "segment_end",
)
FORBIDDEN_MODEL_COLUMNS = LABEL_COLUMNS + ("direction_5m",)


def read_partitioned_root(base: Path, root: str) -> pl.DataFrame:
    """Lazily scan and collect one root, never the full 16-root dataset."""
    return scan_partitioned_root(base, root).sort("timestamp").collect()


def scan_partitioned_root(base: Path, root: str) -> pl.LazyFrame:
    """Return a lazy root scan so callers can push filters/projections to Parquet."""
    files = sorted((base / f"root={root}").glob("year=*/part-000.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet partitions under {base / f'root={root}'}")
    return pl.scan_parquet(files)


def write_feature_root(frame: pl.DataFrame, base: Path, root: str) -> tuple[str, ...]:
    """Write a feature table partitioned by root and UTC year."""
    outputs: list[str] = []
    years = frame.select(pl.col("timestamp").dt.year().unique().sort()).to_series().to_list()
    for year in years:
        target = base / f"root={root}" / f"year={year}" / "part-000.parquet"
        _atomic_write_parquet(frame.filter(pl.col("timestamp").dt.year() == year), target)
        outputs.append(str(target))
    return tuple(outputs)


def build_experiment_a(frame: pl.DataFrame) -> pl.DataFrame:
    """Build own-market features then exact-time future-return labels."""
    return add_future_labels(build_own_market_features(frame))


def eligibility_expression(target: str = "future_return_5m") -> pl.Expr:
    """Require a target and completed 240-bar segment warm-up.

    Legitimate exact-clock or zero-denominator feature nulls remain in X and are
    handled by preprocessing fitted on train only. Requiring rolling_vwap_240m
    guarantees that no stateful indicator is still in its post-roll warm-up.
    """
    return (
        pl.col(target).is_not_null()
        & pl.col(target).is_finite()
        & pl.col("rolling_vwap_240m").is_not_null()
        & pl.col("rolling_vwap_240m").is_finite()
    )


def summarize_feature_root(frame: pl.DataFrame, root: str) -> dict[str, Any]:
    """Compute eligibility, exact-time loss, warm-up, and null-rate diagnostics."""
    exact_columns = [
        "log_return_1m",
        "log_return_5m",
        "log_return_15m",
        "log_return_30m",
        "log_return_60m",
        *LABEL_COLUMNS,
    ]
    exact_missing = pl.any_horizontal([pl.col(name).is_null() for name in exact_columns])
    # The 240-bar VWAP is the longest stateful warm-up and therefore subsumes
    # the shorter EMA/RSI/ATR/volume warm-ups within a segment.
    warmup_missing = pl.col("rolling_vwap_240m").is_null()
    rollover_guard = (
        ((pl.col("timestamp") - pl.col("segment_start")).dt.total_minutes() < 240)
        | ((pl.col("segment_end") - pl.col("timestamp")).dt.total_minutes() < 15)
    )
    aggregates = frame.select(
        pl.len().alias("rows"),
        eligibility_expression().sum().alias("eligible_rows"),
        exact_missing.sum().alias("rows_missing_exact_timestamp"),
        warmup_missing.sum().alias("rows_with_warmup_null"),
        (
            pl.col("rolling_vwap_240m").is_not_null()
            & pl.col("future_return_5m").is_null()
        )
        .sum()
        .alias("rows_lost_missing_primary_target"),
        rollover_guard.sum().alias("rows_in_roll_guard"),
    ).row(0, named=True)
    null_counts = frame.select(
        *[pl.col(name).null_count().alias(name) for name in [*MODEL_FEATURES_A, *LABEL_COLUMNS]]
    ).row(0, named=True)
    aggregates.update(
        {
            "root": root,
            "null_counts": null_counts,
            "null_rates": {name: count / frame.height for name, count in null_counts.items()},
        }
    )
    return aggregates


def feature_specification() -> list[dict[str, Any]]:
    """Return the machine-readable feature and label data contract."""
    rows: list[dict[str, Any]] = []
    for horizon in (1, 5, 15, 30, 60):
        rows.append(
            {
                "name": f"log_return_{horizon}m",
                "kind": "feature",
                "formula": f"ln(close[t] / close[t-{horizon}m])",
                "source_columns": ["close", "timestamp", "contract_segment_id"],
                "lookback": f"exact {horizon} clock minutes",
                "normalization": "log ratio",
                "experiment": ["A", "B"],
                "code": "features.ohlcv_features.add_exact_log_returns",
            }
        )
    simple = {
        "return_open_to_close": ("(close-open)/open", ["open", "close"], "current bar", "ratio"),
        "range_pct": ("(high-low)/close", ["high", "low", "close"], "current bar", "ratio"),
        "upper_wick_pct": ("(high-max(open,close))/close", ["open", "high", "close"], "current bar", "ratio"),
        "lower_wick_pct": ("(min(open,close)-low)/close", ["open", "low", "close"], "current bar", "ratio"),
        "close_location_in_bar": ("(close-low)/(high-low)", ["high", "low", "close"], "current bar", "unit interval"),
        "close_to_ema_5": ("ln(close/EMA_5)", ["close"], "5 observations", "log ratio"),
        "close_to_ema_20": ("ln(close/EMA_20)", ["close"], "20 observations", "log ratio"),
        "close_to_ema_50": ("ln(close/EMA_50)", ["close"], "50 observations", "log ratio"),
        "ema5_vs_ema20": ("ln(EMA_5/EMA_20)", ["close"], "20 observations", "log ratio"),
        "ema20_vs_ema50": ("ln(EMA_20/EMA_50)", ["close"], "50 observations", "log ratio"),
        "rsi_14": ("100-100/(1+mean(gain,14)/mean(loss,14))", ["close"], "14 observations", "0 to 100"),
        "atr_14_normalized": ("mean(true_range,14)/close", ["high", "low", "close"], "14 observations", "price ratio"),
        "realized_vol_5m": ("std(log_return_1m) over (t-5m,t]", ["close", "timestamp"], "5 clock minutes", "standard deviation"),
        "realized_vol_15m": ("std(log_return_1m) over (t-15m,t]", ["close", "timestamp"], "15 clock minutes", "standard deviation"),
        "realized_vol_60m": ("std(log_return_1m) over (t-60m,t]", ["close", "timestamp"], "60 clock minutes", "standard deviation"),
        "volume": ("current bar volume", ["volume"], "current bar", "none"),
        "rolling_volume_mean_20": ("mean(volume,20)", ["volume"], "20 observations", "none"),
        "rolling_volume_mean_60": ("mean(volume,60)", ["volume"], "60 observations", "none"),
        "volume_zscore_20": ("(volume-mean20)/std20", ["volume"], "20 observations", "z-score"),
        "volume_zscore_60": ("(volume-mean60)/std60", ["volume"], "60 observations", "z-score"),
        "relative_volume_20": ("volume/mean20", ["volume"], "20 observations", "ratio"),
        "close_minus_vwap_60m_pct": ("(close-bar_vwap60)/close", ["high", "low", "close", "volume"], "60 observations", "price ratio"),
        "close_minus_vwap_240m_pct": ("(close-bar_vwap240)/close", ["high", "low", "close", "volume"], "240 observations", "price ratio"),
        "chicago_hour": ("Chicago local hour", ["timestamp"], "current bar", "none"),
        "chicago_minute": ("Chicago local minute", ["timestamp"], "current bar", "none"),
        "chicago_day_of_week": ("Chicago weekday 0=Monday", ["timestamp"], "current bar", "none"),
        "hour_sin": ("sin(2*pi*minute_of_day/1440)", ["timestamp"], "current bar", "cyclical"),
        "hour_cos": ("cos(2*pi*minute_of_day/1440)", ["timestamp"], "current bar", "cyclical"),
        "day_of_week_sin": ("sin(2*pi*weekday/7)", ["timestamp"], "current bar", "cyclical"),
        "day_of_week_cos": ("cos(2*pi*weekday/7)", ["timestamp"], "current bar", "cyclical"),
    }
    code_by_name = {
        **{name: "features.ohlcv_features.add_candle_features" for name in tuple(simple)[:5]},
        **{name: "features.ohlcv_features.add_trend_features" for name in tuple(simple)[5:11]},
        **{name: "features.ohlcv_features.add_volatility_features" for name in tuple(simple)[11:15]},
        **{name: "features.ohlcv_features.add_volume_features" for name in tuple(simple)[15:21]},
        **{name: "features.ohlcv_features.add_vwap_features" for name in tuple(simple)[21:23]},
        **{name: "features.ohlcv_features.add_time_features" for name in tuple(simple)[23:]},
    }
    for name, (formula, sources, lookback, normalization) in simple.items():
        rows.append(
            {
                "name": name,
                "kind": "feature",
                "formula": formula,
                "source_columns": sources,
                "lookback": lookback,
                "normalization": normalization,
                "experiment": ["A", "B"],
                "code": code_by_name[name],
            }
        )
    for name in CROSS_MARKET_FEATURES:
        is_spread = name == "NQ_ES_return_spread_5m"
        rows.append(
            {
                "name": name,
                "kind": "feature",
                "formula": "NQ_return_5m - ES_return_5m" if is_spread else "anchor exact-time log return",
                "source_columns": ["timestamp", "anchor log return"],
                "lookback": "5 clock minutes" if is_spread else "encoded in name",
                "normalization": "log return",
                "experiment": ["B"],
                "code": "features.cross_market.add_cross_market_features",
            }
        )
    for horizon in (1, 5, 15):
        rows.append(
            {
                "name": f"future_return_{horizon}m",
                "kind": "label",
                "formula": f"ln(close[t+{horizon}m] / close[t])",
                "source_columns": ["close", "timestamp", "contract_segment_id"],
                "lookback": "none",
                "horizon": f"exact {horizon} clock minutes",
                "normalization": "log ratio",
                "experiment": ["A", "B"],
                "code": "labels.future_returns.add_future_return",
            }
        )
    return rows


def write_feature_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
