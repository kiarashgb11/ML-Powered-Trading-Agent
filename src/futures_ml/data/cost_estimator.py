"""Authenticated Databento cost, record-count, and billable-size estimator."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv

from futures_ml.config.paths import PathConfigurationError, StorageInfo, resolve_data_paths
from futures_ml.config.settings import DataSettings, SettingsError, UniverseSettings
from futures_ml.data.databento_client import (
    DatabentoConfigurationError,
    create_historical_client,
    verify_dataset_and_schemas,
)

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class EstimateQuery:
    """One symbol/schema metadata request."""

    category: str
    root: str
    request_symbol: str
    schema: str
    stype_in: str
    start: str
    end: str


@dataclass(frozen=True)
class EstimateRow:
    """Cost and size results for one request."""

    category: str
    root: str
    request_symbol: str
    schema: str
    stype_in: str
    start: str
    end: str
    estimated_records: int
    estimated_billable_bytes: int
    estimated_gb: float
    estimated_cost_usd: float


@dataclass(frozen=True)
class EstimateReport:
    """Full estimate with budget and storage decisions."""

    generated_at_utc: str
    dataset: str
    latest_complete_end_exclusive: str
    rows: tuple[EstimateRow, ...]
    total_records: int
    total_billable_bytes: int
    total_billable_gb: float
    total_cost_usd: float
    budget_limit_usd: float
    configured_data_root: str
    storage_drive: str
    disk_total_bytes: int
    disk_free_bytes: int
    expected_working_bytes: int
    required_free_with_headroom_bytes: int
    working_space_multiplier: float
    safety_headroom_multiplier: float
    budget_pass: bool
    storage_pass: bool
    download_authorized: bool


def _subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def latest_complete_end(
    dataset_range: dict[str, Any],
    required_schemas: set[str],
    *,
    today_utc: date | None = None,
) -> date:
    """Find a midnight UTC boundary complete for every required schema."""
    schema_ranges = dataset_range.get("schema", {})
    ends: list[datetime] = []
    for schema in sorted(required_schemas):
        schema_range = schema_ranges.get(schema)
        if not isinstance(schema_range, dict) or not schema_range.get("end"):
            raise ValueError(f"Dataset availability did not include an end for schema {schema!r}")
        ends.append(_parse_timestamp(str(schema_range["end"])))
    current_day = today_utc or datetime.now(timezone.utc).date()
    return min([timestamp.date() for timestamp in ends] + [current_day])


def build_queries(
    universe: UniverseSettings,
    settings: DataSettings,
    end: date,
) -> tuple[EstimateQuery, ...]:
    """Expand the fixed plan into individually auditable symbol/schema requests."""
    categories = {
        "ohlcv-1s": "OHLCV-1s",
        "definition": "Definitions",
        "statistics": "Statistics",
        "status": "Status",
        settings.core_mbp_schema: "MBP-1",
    }
    queries: list[EstimateQuery] = []
    for market in universe.markets:
        for schema in settings.universal_schemas:
            is_definition = schema == "definition"
            symbol_template = (
                settings.definition_template if is_definition else settings.continuous_template
            )
            queries.append(
                EstimateQuery(
                    category=categories.get(schema, schema),
                    root=market.root,
                    request_symbol=symbol_template.format(root=market.root),
                    schema=schema,
                    stype_in="parent" if is_definition else "continuous",
                    start=_subtract_years(end, settings.horizons_years[schema]).isoformat(),
                    end=end.isoformat(),
                )
            )
    for root in universe.core_mbp_markets:
        schema = settings.core_mbp_schema
        queries.append(
            EstimateQuery(
                category=categories.get(schema, schema),
                root=root,
                request_symbol=settings.continuous_template.format(root=root),
                schema=schema,
                stype_in="continuous",
                start=_subtract_years(end, settings.horizons_years[schema]).isoformat(),
                end=end.isoformat(),
            )
        )
    return tuple(queries)


def _with_retry(operation: Callable[[], T], *, description: str, attempts: int = 4) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception:
            if attempt == attempts:
                raise
            delay = 2 ** (attempt - 1)
            LOGGER.warning(
                "%s failed (attempt %d/%d); retrying in %ds", description, attempt, attempts, delay
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def estimate_query(client: Any, settings: DataSettings, query: EstimateQuery) -> EstimateRow:
    """Call all three official metadata estimators for one request."""
    kwargs = {
        "dataset": settings.dataset,
        "start": query.start,
        "end": query.end,
        "symbols": [query.request_symbol],
        "schema": query.schema,
        "stype_in": query.stype_in,
    }
    records = _with_retry(
        lambda: client.metadata.get_record_count(**kwargs),
        description=f"record count for {query.root}/{query.schema}",
    )
    billable_bytes = _with_retry(
        lambda: client.metadata.get_billable_size(**kwargs),
        description=f"billable size for {query.root}/{query.schema}",
    )
    cost = _with_retry(
        lambda: client.metadata.get_cost(**kwargs),
        description=f"cost for {query.root}/{query.schema}",
    )
    return EstimateRow(
        **asdict(query),
        estimated_records=int(records),
        estimated_billable_bytes=int(billable_bytes),
        estimated_gb=round(int(billable_bytes) / 1_000_000_000, 6),
        estimated_cost_usd=round(float(cost), 8),
    )


def build_report(
    *,
    rows: tuple[EstimateRow, ...],
    settings: DataSettings,
    end: date,
    data_root: Path,
    storage: StorageInfo,
) -> EstimateReport:
    """Apply explicit budget and disk-space gates to completed estimates."""
    total_records = sum(row.estimated_records for row in rows)
    total_bytes = sum(row.estimated_billable_bytes for row in rows)
    total_cost = sum(row.estimated_cost_usd for row in rows)
    working_bytes = int(total_bytes * settings.working_space_multiplier)
    required_free = int(working_bytes * settings.safety_headroom_multiplier)
    budget_pass = total_cost <= settings.budget_limit_usd
    storage_pass = storage.free_bytes >= required_free
    return EstimateReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset=settings.dataset,
        latest_complete_end_exclusive=end.isoformat(),
        rows=rows,
        total_records=total_records,
        total_billable_bytes=total_bytes,
        total_billable_gb=round(total_bytes / 1_000_000_000, 6),
        total_cost_usd=round(total_cost, 8),
        budget_limit_usd=settings.budget_limit_usd,
        configured_data_root=str(data_root),
        storage_drive=storage.drive,
        disk_total_bytes=storage.total_bytes,
        disk_free_bytes=storage.free_bytes,
        expected_working_bytes=working_bytes,
        required_free_with_headroom_bytes=required_free,
        working_space_multiplier=settings.working_space_multiplier,
        safety_headroom_multiplier=settings.safety_headroom_multiplier,
        budget_pass=budget_pass,
        storage_pass=storage_pass,
        download_authorized=budget_pass and storage_pass,
    )


def _schema_totals(rows: tuple[EstimateRow, ...]) -> dict[str, dict[str, float | int]]:
    totals: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"records": 0, "billable_bytes": 0, "cost_usd": 0.0}
    )
    for row in rows:
        bucket = totals[row.category]
        bucket["records"] += row.estimated_records
        bucket["billable_bytes"] += row.estimated_billable_bytes
        bucket["cost_usd"] += row.estimated_cost_usd
    return dict(totals)


def _gb(value: int) -> float:
    return value / 1_000_000_000


def write_reports(report: EstimateReport, output_dir: Path) -> None:
    """Write detailed CSV/JSON plus a human-readable Markdown decision report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = report.rows
    totals = _schema_totals(rows)

    csv_path = output_dir / "data_cost_report.csv"
    fieldnames = ["row_type", *EstimateRow.__dataclass_fields__.keys()]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"row_type": "detail", **asdict(row)})
        for category, values in totals.items():
            writer.writerow(
                {
                    "row_type": "schema_total",
                    "category": category,
                    "estimated_records": values["records"],
                    "estimated_billable_bytes": values["billable_bytes"],
                    "estimated_gb": round(_gb(int(values["billable_bytes"])), 6),
                    "estimated_cost_usd": round(float(values["cost_usd"]), 8),
                }
            )
        writer.writerow(
            {
                "row_type": "grand_total",
                "category": "GRAND TOTAL",
                "estimated_records": report.total_records,
                "estimated_billable_bytes": report.total_billable_bytes,
                "estimated_gb": report.total_billable_gb,
                "estimated_cost_usd": report.total_cost_usd,
            }
        )

    json_payload = asdict(report)
    json_payload["schema_totals"] = totals
    with (output_dir / "data_cost_report.json").open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, indent=2)
        handle.write("\n")

    status = "AUTHORIZED" if report.download_authorized else "BLOCKED"
    markdown = [
        "# Databento Data Cost and Storage Report",
        "",
        f"Generated: `{report.generated_at_utc}`",
        f"Dataset: `{report.dataset}`",
        f"Latest complete end boundary (exclusive, UTC): `{report.latest_complete_end_exclusive}`",
        "",
        "## Decision",
        "",
        f"**Download status: {status}**",
        "",
        f"- Budget: `${report.total_cost_usd:,.2f}` estimated / `${report.budget_limit_usd:,.2f}` limit — {'PASS' if report.budget_pass else 'FAIL'}",
        f"- Storage: `{_gb(report.disk_free_bytes):,.2f} GB` free / `{_gb(report.required_free_with_headroom_bytes):,.2f} GB` required with headroom — {'PASS' if report.storage_pass else 'FAIL'}",
        "",
        "No download is performed by the estimator. A downloader may proceed only when this report says AUTHORIZED.",
        "",
        "## Plan totals",
        "",
        "| Plan component | Records | Billable raw GB | Cost USD |",
        "|---|---:|---:|---:|",
    ]
    for category, values in totals.items():
        markdown.append(
            f"| {category} | {int(values['records']):,} | {_gb(int(values['billable_bytes'])):,.3f} | ${float(values['cost_usd']):,.4f} |"
        )
    markdown.extend(
        [
            f"| **GRAND TOTAL** | **{report.total_records:,}** | **{report.total_billable_gb:,.3f}** | **${report.total_cost_usd:,.4f}** |",
            "",
            "## Storage",
            "",
            f"- Configured root: `{report.configured_data_root}`",
            f"- Volume: `{report.storage_drive}`",
            f"- Total capacity: `{_gb(report.disk_total_bytes):,.2f} GB`",
            f"- Available capacity: `{_gb(report.disk_free_bytes):,.2f} GB`",
            f"- Billable raw binary size: `{report.total_billable_gb:,.3f} GB`",
            f"- Expected working space: `{_gb(report.expected_working_bytes):,.3f} GB` ({report.working_space_multiplier:.2f}x raw)",
            f"- Required free space with safety margin: `{_gb(report.required_free_with_headroom_bytes):,.3f} GB` ({report.safety_headroom_multiplier:.2f}x working)",
            "",
            "The API's billable size is uncompressed raw binary used for billing, not a promise of final compressed download size. The working estimate reserves additional room for immutable raw files, processed Parquet, features, and temporary intermediates.",
            "",
            "## Detailed requests",
            "",
            "See `data_cost_report.csv` or `data_cost_report.json` for every symbol/schema request.",
            "",
            "## Methodology notes",
            "",
            "- Estimates come from Databento `get_cost`, `get_record_count`, and `get_billable_size` metadata calls.",
            "- Date ranges are midnight-to-midnight UTC and the end is exclusive.",
            "- OHLCV-1s, statistics, status, and MBP-1 use volume-ranked continuous symbols such as `ES.v.0`.",
            "- Definitions use parent symbols such as `ES.FUT` to cover underlying contracts.",
            "- Databento notes that estimates may over-report ranges not divisible by ten minutes; definition estimates are accurate only for whole-day ranges. This plan uses whole days.",
        ]
    )
    (output_dir / "data_cost_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return parser.parse_args()


def _run(args: argparse.Namespace) -> int:
    """Execute the estimator after command-line parsing and logging setup."""
    load_dotenv(override=False)
    universe = UniverseSettings.load(args.config_dir / "universe.yaml")
    settings = DataSettings.load(args.config_dir / "data.yaml")

    paths = resolve_data_paths(expected_drive=settings.expected_drive_windows, create=False)
    storage = StorageInfo.for_path(paths.root)
    client = create_historical_client()
    required_schemas = {*settings.universal_schemas, settings.core_mbp_schema}
    schemas = verify_dataset_and_schemas(
        client,
        dataset=settings.dataset,
        required_schemas=required_schemas,
    )
    LOGGER.info("Verified %s and %d available schemas", settings.dataset, len(schemas))

    available = client.metadata.get_dataset_range(dataset=settings.dataset)
    end = latest_complete_end(available, required_schemas)
    queries = build_queries(universe, settings, end)
    LOGGER.info("Estimating %d symbol/schema requests through %s (exclusive)", len(queries), end)
    rows = tuple(estimate_query(client, settings, query) for query in queries)

    report = build_report(
        rows=rows,
        settings=settings,
        end=end,
        data_root=paths.root,
        storage=storage,
    )
    write_reports(report, args.output_dir)
    print(f"TOTAL_COST_USD={report.total_cost_usd:.8f}")
    print(f"TOTAL_BILLABLE_GB={report.total_billable_gb:.6f}")
    print(f"TOTAL_RECORDS={report.total_records}")
    print(f"EXPECTED_WORKING_GB={_gb(report.expected_working_bytes):.6f}")
    print(f"AVAILABLE_GB={_gb(report.disk_free_bytes):.6f}")
    print(f"DOWNLOAD_AUTHORIZED={str(report.download_authorized).lower()}")
    return 0 if report.download_authorized else 3


def main() -> int:
    """Run the authenticated estimator with concise safety-stop messages."""
    args = _parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return _run(args)
    except (PathConfigurationError, DatabentoConfigurationError, SettingsError) as exc:
        LOGGER.error("Safety stop: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
