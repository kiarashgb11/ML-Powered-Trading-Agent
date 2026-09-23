"""Final approved 16-market acquisition plan and pre-download safety gate."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from futures_ml.config.paths import PathConfigurationError, StorageInfo, resolve_data_paths
from futures_ml.config.settings import DataSettings, SettingsError
from futures_ml.data.cost_estimator import (
    EstimateQuery,
    build_report,
    estimate_query,
    latest_complete_end,
    write_reports,
)
from futures_ml.data.databento_client import (
    DatabentoConfigurationError,
    create_historical_client,
    verify_dataset_and_schemas,
)

LOGGER = logging.getLogger(__name__)

FINAL_ROOTS = (
    "NQ",
    "ES",
    "RTY",
    "ZF",
    "ZN",
    "ZB",
    "CL",
    "NG",
    "GC",
    "HG",
    "6E",
    "6J",
    "6B",
    "ZC",
    "ZS",
    "ZW",
)
FINAL_SCHEMAS = ("ohlcv-1m", "definition", "statistics", "status")


def subtract_years(value: date, years: int) -> date:
    """Subtract calendar years while handling a leap-day boundary."""
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def build_final_queries(end: date, settings: DataSettings) -> tuple[EstimateQuery, ...]:
    """Build the exact 64 root/schema requests used by the downloader."""
    start = subtract_years(end, 5).isoformat()
    categories = {
        "ohlcv-1m": "OHLCV-1m",
        "definition": "Definitions",
        "statistics": "Statistics",
        "status": "Status",
    }
    queries: list[EstimateQuery] = []
    for root in FINAL_ROOTS:
        for schema in FINAL_SCHEMAS:
            is_definition = schema == "definition"
            template = settings.definition_template if is_definition else settings.continuous_template
            queries.append(
                EstimateQuery(
                    category=categories[schema],
                    root=root,
                    request_symbol=template.format(root=root),
                    schema=schema,
                    stype_in="parent" if is_definition else "continuous",
                    start=start,
                    end=end.isoformat(),
                )
            )
    return tuple(queries)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> int:
    """Refresh all metadata estimates and write the final acquisition gate report."""
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    load_dotenv(override=False)
    try:
        settings = DataSettings.load(args.config_dir / "data.yaml")
        paths = resolve_data_paths(expected_drive=settings.expected_drive_windows, create=False)
        storage = StorageInfo.for_path(paths.root)
        client = create_historical_client()
        verify_dataset_and_schemas(
            client,
            dataset=settings.dataset,
            required_schemas=set(FINAL_SCHEMAS),
        )
        availability = client.metadata.get_dataset_range(dataset=settings.dataset)
        end = latest_complete_end(availability, set(FINAL_SCHEMAS))
        queries = build_final_queries(end, settings)
        LOGGER.info("Refreshing %d exact root/schema requests", len(queries))
        rows = tuple(estimate_query(client, settings, query) for query in queries)
        report = build_report(
            rows=rows,
            settings=settings,
            end=end,
            data_root=paths.root,
            storage=storage,
        )
        write_reports(report, args.reports_dir)
        print("ROOTS=" + ",".join(FINAL_ROOTS))
        print(f"START={queries[0].start}")
        print(f"END_EXCLUSIVE={queries[0].end}")
        print(f"TOTAL_COST_USD={report.total_cost_usd:.8f}")
        print(f"TOTAL_RECORDS={report.total_records}")
        print(f"TOTAL_BILLABLE_GB={report.total_billable_gb:.6f}")
        print(f"EXPECTED_WORKING_GB={report.expected_working_bytes / 1e9:.6f}")
        print(f"REQUIRED_WITH_HEADROOM_GB={report.required_free_with_headroom_bytes / 1e9:.6f}")
        print(f"D_FREE_GB={report.disk_free_bytes / 1e9:.6f}")
        print(f"DOWNLOAD_AUTHORIZED={str(report.download_authorized).lower()}")
        return 0 if report.download_authorized else 3
    except (PathConfigurationError, DatabentoConfigurationError, SettingsError) as exc:
        LOGGER.error("Safety stop: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
