"""Resumable Databento batch acquisition and streaming DBN validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from futures_ml.config.paths import StorageInfo, resolve_data_paths
from futures_ml.config.settings import DataSettings
from futures_ml.data.databento_client import create_historical_client
from futures_ml.data.final_acquisition import FINAL_ROOTS, FINAL_SCHEMAS

LOGGER = logging.getLogger(__name__)
MANIFEST_VERSION = 1
SCHEMA_FOLDERS = {
    "ohlcv-1m": "ohlcv_1m",
    "definition": "definitions",
    "statistics": "statistics",
    "status": "status",
}
TERMINAL_STATUSES = {"validated", "failed"}


@dataclass
class ManifestEntry:
    """Persistent state for one billable root/schema batch request."""

    partition_id: str
    root: str
    schema: str
    start: str
    end: str
    request_symbol: str
    stype_in: str
    estimated_records: int
    estimated_billable_bytes: int
    estimated_cost_usd: float
    status: str = "planned"
    validation_status: str = "pending"
    job_id: str = ""
    job_state: str = ""
    local_file_path: str = ""
    record_count: int = 0
    file_size_bytes: int = 0
    checksum: str = ""
    min_timestamp: str = ""
    max_timestamp: str = ""
    invalid_ohlc_rows: int = 0
    negative_volume_rows: int = 0
    duplicate_records: int = 0
    suspicious_gaps: int = 0
    roll_transitions: int = 0
    mapping_count: int = 0
    download_timestamp_utc: str = ""
    retry_count: int = 0
    error: str = ""


class Manifest:
    """Atomic JSON/CSV manifest persisted on the external data volume."""

    def __init__(self, path: Path, entries: list[ManifestEntry]) -> None:
        self.path = path
        self.entries = entries

    @classmethod
    def from_report(cls, path: Path, report: dict[str, Any]) -> Manifest:
        """Load an existing manifest or create the exact authorized plan."""
        expected = []
        for row in report["rows"]:
            partition_id = f"{row['root']}__{row['schema']}__{row['start']}__{row['end']}"
            expected.append(
                ManifestEntry(
                    partition_id=partition_id,
                    root=row["root"],
                    schema=row["schema"],
                    start=row["start"],
                    end=row["end"],
                    request_symbol=row["request_symbol"],
                    stype_in=row["stype_in"],
                    estimated_records=int(row["estimated_records"]),
                    estimated_billable_bytes=int(row["estimated_billable_bytes"]),
                    estimated_cost_usd=float(row["estimated_cost_usd"]),
                )
            )
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("version") != MANIFEST_VERSION:
                raise RuntimeError("Unsupported acquisition manifest version")
            entries = [ManifestEntry(**item) for item in payload["entries"]]
            expected_ids = {entry.partition_id for entry in expected}
            actual_ids = {entry.partition_id for entry in entries}
            if actual_ids != expected_ids:
                raise RuntimeError("Existing manifest does not match the authorized estimate")
            return cls(path, entries)
        manifest = cls(path, expected)
        manifest.save()
        return manifest

    def save(self) -> None:
        """Atomically replace JSON and its human-readable CSV companion."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": MANIFEST_VERSION,
            "updated_at_utc": _now(),
            "entries": [asdict(entry) for entry in self.entries],
        }
        temp_json = self.path.with_suffix(".json.tmp")
        temp_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temp_json.replace(self.path)

        csv_path = self.path.with_suffix(".csv")
        temp_csv = csv_path.with_suffix(".csv.tmp")
        with temp_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(ManifestEntry)])
            writer.writeheader()
            for entry in self.entries:
                writer.writerow(asdict(entry))
        temp_csv.replace(csv_path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_authorized_report(path: Path, settings: DataSettings) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("download_authorized"):
        raise RuntimeError("The refreshed cost/storage report does not authorize acquisition")
    if float(report["total_cost_usd"]) > settings.budget_limit_usd:
        raise RuntimeError("The refreshed estimate exceeds the hard budget ceiling")
    rows = report.get("rows", [])
    if len(rows) != len(FINAL_ROOTS) * len(FINAL_SCHEMAS):
        raise RuntimeError("The authorized report does not contain the exact 64-request plan")
    if {row["root"] for row in rows} != set(FINAL_ROOTS):
        raise RuntimeError("The authorized report has the wrong roots")
    if {row["schema"] for row in rows} != set(FINAL_SCHEMAS):
        raise RuntimeError("The authorized report has the wrong schemas")
    return report


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(32_000_000):
            digest.update(chunk)
    return digest.hexdigest()


def _download_and_check_files(client: Any, entry: ManifestEntry, output_dir: Path) -> list[Path]:
    details = client.batch.list_files(entry.job_id)
    if not details:
        raise RuntimeError(f"No downloadable files for job {entry.job_id}")
    paths: list[Path] = []
    for detail in details:
        filename = str(detail["filename"])
        downloaded = client.batch.download(
            entry.job_id,
            output_dir=output_dir,
            filename_to_download=filename,
        )
        if len(downloaded) != 1:
            raise RuntimeError(f"Expected one local file for {filename}")
        path = downloaded[0]
        if path.stat().st_size != int(detail["size"]):
            raise RuntimeError(f"Size mismatch for {path}")
        algorithm, _, expected = str(detail["hash"]).partition(":")
        if not algorithm or _hash_file(path, algorithm) != expected:
            raise RuntimeError(f"Checksum mismatch for {path}")
        paths.append(path)
    return paths


def _timestamp_iso(value: int) -> str:
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).isoformat()


def validate_dbn_files(paths: list[Path], schema: str) -> dict[str, int | str]:
    """Stream DBN files and collect integrity and data-quality metrics."""
    import databento as db
    import numpy as np

    dbn_paths = sorted(path for path in paths if ".dbn" in path.name)
    if not dbn_paths:
        raise RuntimeError("Batch job did not contain a DBN data file")
    records = 0
    invalid_ohlc = 0
    negative_volume = 0
    duplicates = 0
    suspicious_gaps = 0
    transitions = 0
    mapping_count = 0
    min_ts: int | None = None
    max_ts: int | None = None
    previous_ts: int | None = None
    previous_instrument: int | None = None
    for path in dbn_paths:
        store = db.DBNStore.from_file(path)
        if str(store.schema) != schema:
            raise RuntimeError(f"Schema mismatch in {path}: {store.schema} != {schema}")
        mapping_count += len(store.mappings)
        for chunk in store.to_ndarray(count=1_000_000):
            if not len(chunk):
                continue
            names = set(chunk.dtype.names or ())
            timestamp_field = "ts_event" if "ts_event" in names else "ts_recv"
            timestamps = chunk[timestamp_field].astype("int64", copy=False)
            instruments = chunk["instrument_id"].astype("int64", copy=False)
            records += len(chunk)
            chunk_min = int(timestamps.min())
            chunk_max = int(timestamps.max())
            min_ts = chunk_min if min_ts is None else min(min_ts, chunk_min)
            max_ts = chunk_max if max_ts is None else max(max_ts, chunk_max)
            if schema == "ohlcv-1m":
                if len(chunk) > 1:
                    duplicates += int(
                        np.count_nonzero(
                            (timestamps[1:] == timestamps[:-1])
                            & (instruments[1:] == instruments[:-1])
                        )
                    )
                if previous_ts is not None:
                    duplicates += int(
                        timestamps[0] == previous_ts and instruments[0] == previous_instrument
                    )
                invalid_ohlc += int(
                    np.count_nonzero(
                        (chunk["high"] < chunk["open"])
                        | (chunk["high"] < chunk["close"])
                        | (chunk["high"] < chunk["low"])
                        | (chunk["low"] > chunk["open"])
                        | (chunk["low"] > chunk["close"])
                    )
                )
                negative_volume += int(np.count_nonzero(chunk["volume"] < 0))
                if len(chunk) > 1:
                    transitions += int(np.count_nonzero(instruments[1:] != instruments[:-1]))
                    suspicious_gaps += int(
                        np.count_nonzero((timestamps[1:] - timestamps[:-1]) > 4 * 24 * 60 * 60 * 1e9)
                    )
                if previous_ts is not None:
                    transitions += int(instruments[0] != previous_instrument)
                    suspicious_gaps += int(timestamps[0] - previous_ts > 4 * 24 * 60 * 60 * 1e9)
            previous_ts = int(timestamps[-1])
            previous_instrument = int(instruments[-1])
    if records == 0:
        raise RuntimeError("DBN partition contained zero records")
    return {
        "record_count": records,
        "min_timestamp": _timestamp_iso(min_ts or 0),
        "max_timestamp": _timestamp_iso(max_ts or 0),
        "invalid_ohlc_rows": invalid_ohlc,
        "negative_volume_rows": negative_volume,
        "duplicate_records": duplicates,
        "suspicious_gaps": suspicious_gaps,
        "roll_transitions": transitions,
        "mapping_count": mapping_count,
    }


def _submit(client: Any, settings: DataSettings, entry: ManifestEntry) -> str:
    result = client.batch.submit_job(
        dataset=settings.dataset,
        symbols=[entry.request_symbol],
        schema=entry.schema,
        start=entry.start,
        end=entry.end,
        encoding="dbn",
        compression="zstd",
        map_symbols=False,
        split_symbols=False,
        split_duration="year",
        delivery="download",
        stype_in=entry.stype_in,
        stype_out="instrument_id",
    )
    job_id = str(result.get("id", ""))
    if not job_id:
        raise RuntimeError(f"Batch submission for {entry.partition_id} returned no job ID")
    return job_id


def run_acquisition(report_path: Path) -> int:
    """Submit missing jobs, download finished jobs, and validate every partition."""
    settings = DataSettings.load(Path("config/data.yaml"))
    report = _load_authorized_report(report_path, settings)
    paths = resolve_data_paths(expected_drive=settings.expected_drive_windows, create=True)
    paths.configure_process_temp()
    required = int(report["required_free_with_headroom_bytes"])
    if StorageInfo.for_path(paths.root).free_bytes < required:
        raise RuntimeError("Storage gate failed immediately before batch submission")
    manifest_path = paths.batch_root / "acquisition_manifest.json"
    if not manifest_path.exists():
        intended = yaml.safe_load(Path("config/intended_plan.yaml").read_text(encoding="utf-8"))
        if not intended.get("download_authorized", False):
            raise RuntimeError(
                "No acquisition manifest exists and download authorization is not active; "
                "refusing to create new paid batch jobs"
            )
    manifest = Manifest.from_report(manifest_path, report)
    client = create_historical_client()

    for entry in manifest.entries:
        if entry.status == "validated" and entry.schema != "ohlcv-1m":
            entry.duplicate_records = 0
        if entry.status == "failed" and entry.job_id and entry.retry_count < 5:
            entry.status = "submitted"
            entry.validation_status = "pending"
    manifest.save()

    for entry in manifest.entries:
        if entry.status in TERMINAL_STATUSES or entry.job_id:
            continue
        LOGGER.info("Submitting %s (%s)", entry.root, entry.schema)
        try:
            entry.job_id = _submit(client, settings, entry)
            entry.status = "submitted"
            entry.job_state = "queued"
            entry.error = ""
            manifest.save()
        except Exception as exc:
            entry.status = "failed"
            entry.error = str(exc)
            manifest.save()
            raise
        time.sleep(3.1)

    while True:
        remaining = [entry for entry in manifest.entries if entry.status not in TERMINAL_STATUSES]
        if not remaining:
            break
        jobs = {
            str(job["id"]): job
            for job in client.batch.list_jobs(states="queued,processing,done,expired", short=False)
        }
        progressed = False
        for entry in remaining:
            job = jobs.get(entry.job_id)
            if job is None:
                entry.error = "Submitted job was not returned by list_jobs"
                manifest.save()
                continue
            state = str(job.get("state", ""))
            entry.job_state = state
            if state in {"queued", "processing"}:
                entry.status = "submitted"
                continue
            if state != "done":
                entry.status = "failed"
                entry.error = str(job.get("error", f"Unexpected job state: {state}"))
                manifest.save()
                progressed = True
                continue
            LOGGER.info("Downloading %s (%s), job %s", entry.root, entry.schema, entry.job_id)
            try:
                output_dir = paths.raw_data_root / SCHEMA_FOLDERS[entry.schema] / entry.root
                local_paths = _download_and_check_files(client, entry, output_dir)
                entry.status = "downloaded"
                entry.local_file_path = "|".join(str(path) for path in local_paths)
                entry.file_size_bytes = sum(path.stat().st_size for path in local_paths)
                entry.checksum = "verified_against_databento_manifest"
                entry.download_timestamp_utc = _now()
                manifest.save()
                metrics = validate_dbn_files(local_paths, entry.schema)
                for key, value in metrics.items():
                    setattr(entry, key, value)
                entry.status = "validated"
                entry.validation_status = "passed"
                entry.error = ""
                manifest.save()
                LOGGER.info("Validated %s (%s): %d records", entry.root, entry.schema, entry.record_count)
            except Exception as exc:
                entry.retry_count += 1
                retryable = entry.retry_count < 5
                entry.status = "submitted" if retryable else "failed"
                entry.validation_status = "pending" if retryable else "failed"
                entry.error = str(exc)
                manifest.save()
                LOGGER.exception(
                    "Partition attempt %d failed%s: %s",
                    entry.retry_count,
                    "; will retry" if retryable else "",
                    entry.partition_id,
                )
            progressed = True
        manifest.save()
        if any(entry.status not in TERMINAL_STATUSES for entry in manifest.entries):
            if not progressed:
                LOGGER.info("Waiting for %d batch jobs", len(remaining))
            time.sleep(15)

    failed = [entry for entry in manifest.entries if entry.status == "failed"]
    return 1 if failed else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("reports/data_cost_report.json"))
    return parser.parse_args()


def main() -> int:
    """CLI entry point for resumable acquisition."""
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(override=False)
    return run_acquisition(_parse_args().report)


if __name__ == "__main__":
    raise SystemExit(main())
