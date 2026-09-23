"""Build the canonical one-minute Parquet dataset from immutable DBN files."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import databento as db
import numpy as np
import polars as pl


@dataclass(frozen=True)
class ProcessedRootSummary:
    """Auditable facts for one processed futures root."""

    root: str
    raw_rows: int
    processed_rows: int
    min_timestamp: str
    max_timestamp: str
    contract_segments: int
    rollovers: int
    non_one_minute_intervals: int
    missing_clock_minutes: int
    parquet_files: tuple[str, ...]


def load_manifest_entries(path: Path) -> list[dict[str, Any]]:
    """Read the Phase 1 external manifest and return its entries."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"Manifest has no entries list: {path}")
    return entries


def ohlcv_dbn_paths(entry: dict[str, Any]) -> list[Path]:
    """Extract only immutable DBN payload paths from a manifest entry."""
    paths = [Path(item) for item in str(entry["local_file_path"]).split("|")]
    dbn_paths = sorted(path for path in paths if ".dbn" in path.name)
    if not dbn_paths or any(not path.is_file() for path in dbn_paths):
        raise FileNotFoundError(f"Missing DBN payload(s) for {entry['partition_id']}")
    return dbn_paths


def _chunk_to_frame(chunk: np.ndarray, root: str, price_scale: int) -> pl.DataFrame:
    """Convert one bounded structured-array chunk without using pandas."""
    scale = float(price_scale)
    return pl.DataFrame(
        {
            "timestamp": pl.Series(chunk["ts_event"].astype("int64", copy=False)).cast(
                pl.Datetime("ns", "UTC")
            ),
            "root": pl.Series([root] * len(chunk), dtype=pl.String),
            "instrument_id": chunk["instrument_id"].astype("uint32", copy=False),
            "open": chunk["open"].astype("float64", copy=False) / scale,
            "high": chunk["high"].astype("float64", copy=False) / scale,
            "low": chunk["low"].astype("float64", copy=False) / scale,
            "close": chunk["close"].astype("float64", copy=False) / scale,
            "volume": chunk["volume"].astype("uint64", copy=False),
        }
    )


def read_root_ohlcv(
    paths: Iterable[Path], root: str, *, price_scale: int = 1_000_000_000
) -> pl.DataFrame:
    """Stream DBN chunks and materialize only one root at a time."""
    frames: list[pl.DataFrame] = []
    for path in paths:
        store = db.DBNStore.from_file(path)
        if str(store.schema) != "ohlcv-1m":
            raise ValueError(f"Expected ohlcv-1m, got {store.schema} in {path}")
        for chunk in store.to_ndarray(count=500_000):
            if len(chunk):
                frames.append(_chunk_to_frame(chunk, root, price_scale))
    if not frames:
        raise ValueError(f"No OHLCV records found for {root}")
    return pl.concat(frames, rechunk=True).sort("timestamp")


def add_contract_segments(frame: pl.DataFrame) -> pl.DataFrame:
    """Create contiguous contract segments whenever instrument_id changes."""
    segmented = frame.with_columns(
        (pl.col("instrument_id") != pl.col("instrument_id").shift(1))
        .fill_null(False)
        .alias("roll_event")
    ).with_columns(
        (pl.col("roll_event").cast(pl.UInt32).cum_sum() + 1).alias("contract_segment_id")
    )
    return segmented.with_columns(
        pl.col("timestamp")
        .min()
        .over("contract_segment_id")
        .alias("segment_start"),
        pl.col("timestamp")
        .max()
        .over("contract_segment_id")
        .alias("segment_end"),
    )


def rollover_table(frame: pl.DataFrame) -> pl.DataFrame:
    """Return one row per contiguous contract segment."""
    return (
        frame.group_by("root", "contract_segment_id", maintain_order=True)
        .agg(
            pl.col("instrument_id").first(),
            pl.col("segment_start").first(),
            pl.col("segment_end").first(),
            pl.len().alias("rows"),
        )
        .with_columns((pl.col("contract_segment_id") > 1).alias("roll_event"))
    )


def _atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(path)


def write_partitioned_root(frame: pl.DataFrame, output_root: Path, root: str) -> tuple[str, ...]:
    """Write one compressed Parquet partition per UTC calendar year."""
    years = frame.select(pl.col("timestamp").dt.year().unique().sort()).to_series().to_list()
    outputs: list[str] = []
    for year in years:
        target = output_root / f"root={root}" / f"year={year}" / "part-000.parquet"
        _atomic_write_parquet(frame.filter(pl.col("timestamp").dt.year() == year), target)
        outputs.append(str(target))
    return tuple(outputs)


def summarize_root(
    frame: pl.DataFrame,
    *,
    root: str,
    raw_rows: int,
    parquet_files: tuple[str, ...],
) -> ProcessedRootSummary:
    """Calculate row, coverage, rollover, and missing-clock diagnostics."""
    deltas = frame.select(
        pl.col("timestamp").diff().over("contract_segment_id").dt.total_minutes().alias("minutes")
    )["minutes"]
    non_one = int((deltas.is_not_null() & (deltas != 1)).sum())
    missing_minutes = int(deltas.filter(deltas > 1).sum() - int((deltas > 1).sum()))
    return ProcessedRootSummary(
        root=root,
        raw_rows=raw_rows,
        processed_rows=frame.height,
        min_timestamp=frame["timestamp"].min().isoformat(),
        max_timestamp=frame["timestamp"].max().isoformat(),
        contract_segments=int(frame["contract_segment_id"].max()),
        rollovers=int(frame["roll_event"].sum()),
        non_one_minute_intervals=non_one,
        missing_clock_minutes=missing_minutes,
        parquet_files=parquet_files,
    )


def write_processed_manifest(path: Path, summaries: Iterable[ProcessedRootSummary]) -> None:
    """Write a machine-readable Phase 2 processing manifest."""
    rows = [asdict(summary) for summary in summaries]
    payload = {
        "version": 1,
        "status": "complete",
        "raw_rows": sum(row["raw_rows"] for row in rows),
        "processed_rows": sum(row["processed_rows"] for row in rows),
        "rollovers": sum(row["rollovers"] for row in rows),
        "roots": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
