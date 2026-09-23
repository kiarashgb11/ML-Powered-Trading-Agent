"""Build canonical, rollover-safe, Experiment A/B Parquet datasets."""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import polars as pl
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from futures_ml.config.paths import resolve_data_paths  # noqa: E402
from futures_ml.config.settings import DataSettings  # noqa: E402
from futures_ml.data.processed import (  # noqa: E402
    add_contract_segments,
    load_manifest_entries,
    ohlcv_dbn_paths,
    read_root_ohlcv,
    rollover_table,
    summarize_root,
    write_partitioned_root,
    write_processed_manifest,
)
from futures_ml.features.cross_market import (  # noqa: E402
    ANCHOR_ROOTS,
    add_cross_market_features,
    anchor_return_table,
)
from futures_ml.features.pipeline import (  # noqa: E402
    build_experiment_a,
    feature_specification,
    read_partitioned_root,
    summarize_feature_root,
    write_feature_manifest,
    write_feature_root,
)


def _load_config() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "config" / "phase2.yaml").read_text(encoding="utf-8"))


def _load_existing_summaries(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["root"]: row for row in payload.get("roots", [])}


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Recompute feature partitions while preserving canonical processed Parquet.",
    )
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    config = _load_config()
    data_settings = DataSettings.load(PROJECT_ROOT / "config" / "data.yaml")
    paths = resolve_data_paths(expected_drive=data_settings.expected_drive_windows, create=True)
    paths.configure_process_temp()
    os.environ.setdefault("POLARS_TEMP_DIR", str(paths.temp_root))
    manifest_path = paths.batch_root / "acquisition_manifest.json"
    entries = load_manifest_entries(manifest_path)
    ohlcv = {row["root"]: row for row in entries if row["schema"] == "ohlcv-1m"}
    roots = tuple(config["roots"])
    if set(ohlcv) != set(roots):
        raise RuntimeError(f"OHLCV universe mismatch: {sorted(ohlcv)} != {sorted(roots)}")

    processed_base = paths.processed_data_root / "bars_1m"
    processed_manifest = paths.processed_data_root / "phase2_processed_manifest.json"
    summaries = []
    for root in roots:
        print(f"[processed] {root}", flush=True)
        raw_rows = int(ohlcv[root]["record_count"])
        existing_files = sorted((processed_base / f"root={root}").glob("year=*/part-000.parquet"))
        existing_rows = sum(pl.scan_parquet(path).select(pl.len()).collect().item() for path in existing_files)
        if existing_files and existing_rows == raw_rows:
            frame = read_partitioned_root(processed_base, root)
            outputs = tuple(str(path) for path in existing_files)
        else:
            frame = add_contract_segments(
                read_root_ohlcv(
                    ohlcv_dbn_paths(ohlcv[root]),
                    root,
                    price_scale=int(config["price_scale"]),
                )
            )
            outputs = write_partitioned_root(frame, processed_base, root)
        rolls = int(frame["roll_event"].sum())
        expected_rolls = int(ohlcv[root]["roll_transitions"])
        if rolls != expected_rolls:
            raise RuntimeError(f"STOP: {root} rollover mismatch {rolls} != {expected_rolls}")
        roll_path = paths.processed_data_root / "rollovers" / f"root={root}" / "segments.parquet"
        roll_path.parent.mkdir(parents=True, exist_ok=True)
        rollover_table(frame).write_parquet(roll_path, compression="zstd")
        summaries.append(summarize_root(frame, root=root, raw_rows=raw_rows, parquet_files=outputs))
    if sum(row.rollovers for row in summaries) != 440:
        raise RuntimeError("STOP: total rollover count does not reconcile to Phase 1's 440")
    write_processed_manifest(processed_manifest, summaries)

    experiment_a = paths.features_root / "experiment_a"
    feature_summaries: list[dict] = []
    for root in roots:
        print(f"[features A] {root}", flush=True)
        output_files = sorted((experiment_a / f"root={root}").glob("year=*/part-000.parquet"))
        if output_files and not args.force_features:
            frame = read_partitioned_root(experiment_a, root)
        else:
            frame = build_experiment_a(read_partitioned_root(processed_base, root))
            output_files = [Path(path) for path in write_feature_root(frame, experiment_a, root)]
        summary = summarize_feature_root(frame, root)
        summary["experiment_a_files"] = [str(path) for path in output_files]
        feature_summaries.append(summary)

    print("[features B] constructing exact-timestamp anchor table", flush=True)
    anchor_frames = {root: read_partitioned_root(experiment_a, root) for root in ANCHOR_ROOTS}
    anchors = anchor_return_table(anchor_frames)
    del anchor_frames
    experiment_b = paths.features_root / "experiment_b"
    for summary in feature_summaries:
        root = summary["root"]
        print(f"[features B] {root}", flush=True)
        output_files = sorted((experiment_b / f"root={root}").glob("year=*/part-000.parquet"))
        if output_files and not args.force_features:
            frame_b = read_partitioned_root(experiment_b, root)
        else:
            frame_b = add_cross_market_features(read_partitioned_root(experiment_a, root), anchors)
            output_files = [Path(path) for path in write_feature_root(frame_b, experiment_b, root)]
        summary["experiment_b_files"] = [str(path) for path in output_files]
        summary["cross_market_null_rates"] = frame_b.select(
            *[
                pl.col(name).null_count().truediv(pl.len()).alias(name)
                for name in frame_b.columns
                if name.endswith("_return_1m")
                or name.endswith("_return_5m")
                or name.endswith("_return_15m")
                or name == "NQ_ES_return_spread_5m"
            ]
        ).row(0, named=True)

    payload = {
        "version": config["version"],
        "status": "complete",
        "roots": feature_summaries,
        "total_rows": sum(row["rows"] for row in feature_summaries),
        "total_eligible_rows": sum(row["eligible_rows"] for row in feature_summaries),
        "feature_specification": feature_specification(),
    }
    write_feature_manifest(paths.features_root / "phase2_feature_manifest.json", payload)
    (PROJECT_ROOT / "reports" / "feature_spec.json").write_text(
        json.dumps(feature_specification(), indent=2), encoding="utf-8"
    )
    print(
        f"Complete: {payload['total_rows']:,} rows; "
        f"{payload['total_eligible_rows']:,} Experiment A eligible",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
