"""Small, causal, rollover-safe OHLCV feature transformations."""

from __future__ import annotations

import math

import polars as pl

RETURN_HORIZONS = (1, 5, 15, 30, 60)
SEGMENT = "contract_segment_id"

MODEL_FEATURES_A = (
    "log_return_1m",
    "log_return_5m",
    "log_return_15m",
    "log_return_30m",
    "log_return_60m",
    "return_open_to_close",
    "range_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "close_location_in_bar",
    "close_to_ema_5",
    "close_to_ema_20",
    "close_to_ema_50",
    "ema5_vs_ema20",
    "ema20_vs_ema50",
    "rsi_14",
    "atr_14_normalized",
    "realized_vol_5m",
    "realized_vol_15m",
    "realized_vol_60m",
    "volume",
    "rolling_volume_mean_20",
    "rolling_volume_mean_60",
    "volume_zscore_20",
    "volume_zscore_60",
    "relative_volume_20",
    "close_minus_vwap_60m_pct",
    "close_minus_vwap_240m_pct",
    "chicago_hour",
    "chicago_minute",
    "chicago_day_of_week",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
)


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator.abs() > 0).then(numerator / denominator).otherwise(None)


def add_exact_log_returns(
    frame: pl.DataFrame, horizons: tuple[int, ...] = RETURN_HORIZONS
) -> pl.DataFrame:
    """Add r_t^h = log(close[t] / close[t-h]) with exact-time segment joins."""
    result = frame
    timestamp_dtype = frame.schema["timestamp"]
    for horizon in horizons:
        past_name = f"_past_close_{horizon}m"
        lookup = frame.select(
            SEGMENT,
            (pl.col("timestamp") + pl.duration(minutes=horizon))
            .cast(timestamp_dtype)
            .alias("timestamp"),
            pl.col("close").alias(past_name),
        )
        result = result.join(lookup, on=[SEGMENT, "timestamp"], how="left").with_columns(
            pl.when((pl.col("close") > 0) & (pl.col(past_name) > 0))
            .then((pl.col("close") / pl.col(past_name)).log())
            .otherwise(None)
            .alias(f"log_return_{horizon}m")
        ).drop(past_name)
    return result


def add_candle_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Add scale-independent candle-shape features at the current bar."""
    body_top = pl.max_horizontal("open", "close")
    body_bottom = pl.min_horizontal("open", "close")
    return frame.with_columns(
        _safe_ratio(pl.col("close") - pl.col("open"), pl.col("open")).alias(
            "return_open_to_close"
        ),
        _safe_ratio(pl.col("high") - pl.col("low"), pl.col("close")).alias("range_pct"),
        _safe_ratio(pl.col("high") - body_top, pl.col("close")).alias("upper_wick_pct"),
        _safe_ratio(body_bottom - pl.col("low"), pl.col("close")).alias("lower_wick_pct"),
        _safe_ratio(pl.col("close") - pl.col("low"), pl.col("high") - pl.col("low")).alias(
            "close_location_in_bar"
        ),
    )


def add_trend_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Add causal EMAs and a 14-observation RSI inside each contract segment."""
    result = frame.with_columns(
        *[
            pl.col("close")
            .ewm_mean(span=span, adjust=False, min_samples=span)
            .over(SEGMENT)
            .alias(f"ema_{span}")
            for span in (5, 20, 50)
        ],
        pl.col("close").diff().over(SEGMENT).alias("_close_change"),
    ).with_columns(
        pl.col("_close_change").clip(lower_bound=0).alias("_gain"),
        (-pl.col("_close_change")).clip(lower_bound=0).alias("_loss"),
    ).with_columns(
        pl.col("_gain").rolling_mean(14, min_samples=14).over(SEGMENT).alias("_avg_gain"),
        pl.col("_loss").rolling_mean(14, min_samples=14).over(SEGMENT).alias("_avg_loss"),
    )
    return result.with_columns(
        (pl.col("close") / pl.col("ema_5")).log().alias("close_to_ema_5"),
        (pl.col("close") / pl.col("ema_20")).log().alias("close_to_ema_20"),
        (pl.col("close") / pl.col("ema_50")).log().alias("close_to_ema_50"),
        (pl.col("ema_5") / pl.col("ema_20")).log().alias("ema5_vs_ema20"),
        (pl.col("ema_20") / pl.col("ema_50")).log().alias("ema20_vs_ema50"),
        pl.when(pl.col("_avg_loss") == 0)
        .then(pl.when(pl.col("_avg_gain") == 0).then(50.0).otherwise(100.0))
        .otherwise(100.0 - 100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss")))
        .alias("rsi_14"),
    ).drop("_close_change", "_gain", "_loss", "_avg_gain", "_avg_loss")


def add_volatility_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Add Wilder-style true range mean and clock-window return volatility."""
    result = frame.with_columns(pl.col("close").shift(1).over(SEGMENT).alias("_prev_close"))
    result = result.with_columns(
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("_prev_close")).abs(),
            (pl.col("low") - pl.col("_prev_close")).abs(),
        ).alias("_true_range")
    ).with_columns(
        pl.col("_true_range")
        .rolling_mean(14, min_samples=14)
        .over(SEGMENT)
        .alias("atr_14")
    )
    result = result.with_columns(
        _safe_ratio(pl.col("atr_14"), pl.col("close")).alias("atr_14_normalized")
    )
    result = result.with_columns(
        *[
            pl.col("log_return_1m")
            .rolling_std_by("timestamp", f"{horizon}m", min_samples=2)
            .over(SEGMENT)
            .alias(f"realized_vol_{horizon}m")
            for horizon in (5, 15, 60)
        ]
    )
    return result.drop("_prev_close", "_true_range")


def add_volume_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Add backward-looking rolling volume levels and normalized surprises."""
    result = frame.with_columns(
        *[
            pl.col("volume")
            .cast(pl.Float64)
            .rolling_mean(window, min_samples=window)
            .over(SEGMENT)
            .alias(f"rolling_volume_mean_{window}")
            for window in (20, 60)
        ],
        *[
            pl.col("volume")
            .cast(pl.Float64)
            .rolling_std(window, min_samples=window)
            .over(SEGMENT)
            .alias(f"_volume_std_{window}")
            for window in (20, 60)
        ],
    )
    return result.with_columns(
        *[
            _safe_ratio(
                pl.col("volume").cast(pl.Float64) - pl.col(f"rolling_volume_mean_{window}"),
                pl.col(f"_volume_std_{window}"),
            ).alias(f"volume_zscore_{window}")
            for window in (20, 60)
        ],
        _safe_ratio(
            pl.col("volume").cast(pl.Float64), pl.col("rolling_volume_mean_20")
        ).alias("relative_volume_20"),
    ).drop("_volume_std_20", "_volume_std_60")


def add_vwap_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Add causal rolling bar-VWAP proxies over 60 and 240 observed bars."""
    result = frame.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("typical_price")
    ).with_columns(
        (pl.col("typical_price") * pl.col("volume").cast(pl.Float64)).alias("_tp_volume")
    )
    expressions: list[pl.Expr] = []
    for window in (60, 240):
        numerator = pl.col("_tp_volume").rolling_sum(window, min_samples=window).over(SEGMENT)
        denominator = (
            pl.col("volume").cast(pl.Float64).rolling_sum(window, min_samples=window).over(SEGMENT)
        )
        expressions.append(_safe_ratio(numerator, denominator).alias(f"rolling_vwap_{window}m"))
    result = result.with_columns(*expressions)
    return result.with_columns(
        *[
            _safe_ratio(pl.col("close") - pl.col(f"rolling_vwap_{window}m"), pl.col("close"))
            .alias(f"close_minus_vwap_{window}m_pct")
            for window in (60, 240)
        ]
    ).drop("_tp_volume")


def add_time_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Add interpretable Chicago-clock and cyclical time features."""
    result = frame.with_columns(
        pl.col("timestamp").dt.convert_time_zone("America/Chicago").alias("chicago_timestamp")
    ).with_columns(
        pl.col("chicago_timestamp").dt.hour().alias("chicago_hour"),
        pl.col("chicago_timestamp").dt.minute().alias("chicago_minute"),
        (pl.col("chicago_timestamp").dt.weekday() - 1).alias("chicago_day_of_week"),
    )
    minute_of_day = (
        pl.col("chicago_hour").cast(pl.Float64) * 60.0
        + pl.col("chicago_minute").cast(pl.Float64)
    )
    day_of_week = pl.col("chicago_day_of_week").cast(pl.Float64)
    return result.with_columns(
        (2 * math.pi * minute_of_day / 1440).sin().alias("hour_sin"),
        (2 * math.pi * minute_of_day / 1440).cos().alias("hour_cos"),
        (2 * math.pi * day_of_week / 7).sin().alias("day_of_week_sin"),
        (2 * math.pi * day_of_week / 7).cos().alias("day_of_week_cos"),
    )


def build_own_market_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply the complete Experiment A causal feature graph."""
    result = add_exact_log_returns(frame)
    result = add_candle_features(result)
    result = add_trend_features(result)
    result = add_volatility_features(result)
    result = add_volume_features(result)
    result = add_vwap_features(result)
    return add_time_features(result)
