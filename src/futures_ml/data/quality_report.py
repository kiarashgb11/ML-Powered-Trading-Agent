"""Generate the post-acquisition data-quality report from the durable manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from futures_ml.config.paths import StorageInfo, resolve_data_paths
from futures_ml.config.settings import DataSettings
from futures_ml.data.acquisition import ManifestEntry
from futures_ml.data.final_acquisition import FINAL_ROOTS, FINAL_SCHEMAS


def _gb(value: int) -> float:
    return value / 1_000_000_000


def generate_quality_report(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    """Aggregate validation evidence into JSON and Markdown reports."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [ManifestEntry(**item) for item in payload["entries"]]
    settings = DataSettings.load(Path("config/data.yaml"))
    paths = resolve_data_paths(expected_drive=settings.expected_drive_windows, create=False)
    storage = StorageInfo.for_path(paths.root)
    by_root: dict[str, list[ManifestEntry]] = defaultdict(list)
    for entry in entries:
        by_root[entry.root].append(entry)
    actual_bytes = sum(entry.file_size_bytes for entry in entries)
    failed = [entry for entry in entries if entry.status != "validated"]
    all_expected = (
        len(entries) == len(FINAL_ROOTS) * len(FINAL_SCHEMAS)
        and {entry.root for entry in entries} == set(FINAL_ROOTS)
        and {entry.schema for entry in entries} == set(FINAL_SCHEMAS)
    )
    integrity_failures = [
        entry
        for entry in entries
        if entry.invalid_ohlc_rows or entry.negative_volume_rows or entry.duplicate_records
    ]
    overall = "PASS" if all_expected and not failed and not integrity_failures else "FAIL"
    report = {
        "overall_validation_status": overall,
        "data_root": str(paths.root),
        "roots": list(FINAL_ROOTS),
        "schemas": list(FINAL_SCHEMAS),
        "partition_count": len(entries),
        "validated_partitions": sum(entry.status == "validated" for entry in entries),
        "failed_or_incomplete_partitions": len(failed),
        "actual_total_disk_bytes": actual_bytes,
        "remaining_disk_free_bytes": storage.free_bytes,
        "entries": [asdict(entry) for entry in entries],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Historical Data Quality Report",
        "",
        f"**Overall validation status: {overall}**",
        "",
        f"- Data root: `{paths.root}`",
        "- Requested range: `2021-09-23` inclusive to `2026-09-23` exclusive UTC",
        f"- Validated partitions: `{report['validated_partitions']}/{len(entries)}`",
        f"- Actual downloaded disk usage: `{_gb(actual_bytes):,.3f} GB`",
        f"- Remaining D: free space: `{_gb(storage.free_bytes):,.3f} GB`",
        f"- Failed or incomplete partitions: `{len(failed)}`",
        "",
        "## Root summary",
        "",
        "| Root | Schemas | Records | Disk GB | Actual coverage | Invalid OHLC | Negative volume | Duplicate bars | Suspicious >4-day gaps | Roll transitions | Metadata coverage | Status |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|:---:|",
    ]
    for root in FINAL_ROOTS:
        rows = by_root[root]
        bar_rows = [entry for entry in rows if entry.schema == "ohlcv-1m"]
        timestamps_min = [entry.min_timestamp for entry in bar_rows if entry.min_timestamp]
        timestamps_max = [entry.max_timestamp for entry in bar_rows if entry.max_timestamp]
        coverage = "n/a"
        if timestamps_min and timestamps_max:
            coverage = f"{min(timestamps_min)} to {max(timestamps_max)}"
        schemas = {entry.schema for entry in rows if entry.status == "validated"}
        metadata_ok = all(
            any(entry.schema == schema and entry.record_count > 0 for entry in rows)
            for schema in ("definition", "statistics", "status")
        )
        root_ok = schemas == set(FINAL_SCHEMAS) and all(
            entry.invalid_ohlc_rows == 0
            and entry.negative_volume_rows == 0
            and entry.duplicate_records == 0
            for entry in rows
        )
        lines.append(
            f"| {root} | {len(schemas)}/4 | {sum(entry.record_count for entry in rows):,} | "
            f"{_gb(sum(entry.file_size_bytes for entry in rows)):,.3f} | {coverage} | "
            f"{sum(entry.invalid_ohlc_rows for entry in rows):,} | "
            f"{sum(entry.negative_volume_rows for entry in rows):,} | "
            f"{sum(entry.duplicate_records for entry in rows):,} | "
            f"{sum(entry.suspicious_gaps for entry in rows):,} | "
            f"{sum(entry.roll_transitions for entry in rows):,} | "
            f"{'complete' if metadata_ok else 'incomplete'} | {'PASS' if root_ok else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Validation interpretation",
            "",
            "- Every downloaded file was matched to Databento's reported size and checksum, then opened as DBN and streamed to count records.",
            "- OHLC checks require `high >= open/close/low`, `low <= open/close`, nonnegative volume, and unique `(timestamp, instrument_id)` bars.",
            "- Roll transitions count changes in `instrument_id` within the volume-ranked continuous OHLCV series. The raw series is not back-adjusted.",
            "- Suspicious gaps are OHLCV timestamp jumps longer than four days. Missing one-minute bars are otherwise not treated as errors because bars require qualifying trades.",
            "- Duplicate-bar checks apply to OHLCV. Definition, statistics, and status schemas may legitimately contain multiple distinct events for one timestamp and instrument.",
            "- Metadata coverage requires nonzero Definition, Statistics, and Status records for every root.",
            "- Coverage in the root table is OHLCV event-time coverage. Definition snapshots can include instruments whose original event timestamps predate the requested start boundary.",
            "",
            "## Failed or incomplete partitions",
            "",
        ]
    )
    if failed:
        lines.extend(f"- `{entry.partition_id}`: {entry.status} - {entry.error}" for entry in failed)
    else:
        lines.append("None.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/data_quality_report.md"))
    args = parser.parse_args()
    load_dotenv(override=False)
    settings = DataSettings.load(Path("config/data.yaml"))
    paths = resolve_data_paths(expected_drive=settings.expected_drive_windows, create=False)
    manifest = args.manifest or paths.batch_root / "acquisition_manifest.json"
    report = generate_quality_report(manifest, args.output)
    print(f"VALIDATION_STATUS={report['overall_validation_status']}")
    print(f"VALIDATED_PARTITIONS={report['validated_partitions']}/{report['partition_count']}")
    print(f"ACTUAL_DISK_GB={_gb(report['actual_total_disk_bytes']):.6f}")
    print(f"D_FREE_GB={_gb(report['remaining_disk_free_bytes']):.6f}")
    return 0 if report["overall_validation_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
