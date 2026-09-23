"""Estimate a review-only revised universe without changing approved configuration."""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from futures_ml.config.paths import StorageInfo, resolve_data_paths
from futures_ml.config.settings import DataSettings
from futures_ml.data.cost_estimator import (
    EstimateQuery,
    EstimateRow,
    estimate_query,
    latest_complete_end,
)
from futures_ml.data.databento_client import (
    create_historical_client,
    verify_dataset_and_schemas,
)

LOGGER = logging.getLogger(__name__)

REVISED_ROOTS = (
    "NQ",
    "ES",
    "RTY",
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
REVISED_MBP_ROOTS = ("NQ", "ES", "CL", "GC")
OHLCV_YEARS = (5, 4)
MBP_MONTHS = (1, 3, 6)


@dataclass(frozen=True)
class PlanSummary:
    """Cost and capacity gates for one independent or combined scenario."""

    plan_id: str
    description: str
    total_records: int
    total_billable_bytes: int
    total_billable_gb: float
    total_cost_usd: float
    expected_working_bytes: int
    required_free_with_headroom_bytes: int
    budget_pass: bool
    storage_pass: bool
    download_authorized: bool = False


def _subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def subtract_months(value: date, months: int) -> date:
    """Subtract whole calendar months while retaining a valid day of month."""
    month_index = value.year * 12 + value.month - 1 - months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _ohlcv_plan_queries(
    *,
    years: int,
    end: date,
    settings: DataSettings,
) -> tuple[EstimateQuery, ...]:
    categories = {
        "ohlcv-1m": "OHLCV-1m",
        "definition": "Definitions",
        "statistics": "Statistics",
        "status": "Status",
    }
    start = _subtract_years(end, years).isoformat()
    queries: list[EstimateQuery] = []
    for root in REVISED_ROOTS:
        for schema in ("ohlcv-1m", "definition", "statistics", "status"):
            is_definition = schema == "definition"
            template = (
                settings.definition_template if is_definition else settings.continuous_template
            )
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


def _mbp_queries(
    *,
    months: int,
    end: date,
    settings: DataSettings,
) -> tuple[EstimateQuery, ...]:
    return tuple(
        EstimateQuery(
            category="MBP-1",
            root=root,
            request_symbol=settings.continuous_template.format(root=root),
            schema="mbp-1",
            stype_in="continuous",
            start=subtract_months(end, months).isoformat(),
            end=end.isoformat(),
        )
        for root in REVISED_MBP_ROOTS
    )


def _estimate(
    client: Any,
    settings: DataSettings,
    *,
    label: str,
    queries: tuple[EstimateQuery, ...],
) -> tuple[EstimateRow, ...]:
    LOGGER.info("Estimating %s (%d rows)", label, len(queries))
    return tuple(estimate_query(client, settings, query) for query in queries)


def summarize(
    *,
    plan_id: str,
    description: str,
    rows: tuple[EstimateRow, ...],
    settings: DataSettings,
    storage: StorageInfo,
) -> PlanSummary:
    """Summarize rows while keeping every revised plan unauthorized."""
    records = sum(row.estimated_records for row in rows)
    billable_bytes = sum(row.estimated_billable_bytes for row in rows)
    cost = sum(row.estimated_cost_usd for row in rows)
    working = int(billable_bytes * settings.working_space_multiplier)
    required = int(working * settings.safety_headroom_multiplier)
    return PlanSummary(
        plan_id=plan_id,
        description=description,
        total_records=records,
        total_billable_bytes=billable_bytes,
        total_billable_gb=round(billable_bytes / 1e9, 6),
        total_cost_usd=round(cost, 8),
        expected_working_bytes=working,
        required_free_with_headroom_bytes=required,
        budget_pass=cost <= settings.budget_limit_usd,
        storage_pass=storage.free_bytes >= required,
    )


def _write_reports(
    *,
    output_dir: Path,
    end: date,
    settings: DataSettings,
    storage: StorageInfo,
    detail_rows: dict[str, tuple[EstimateRow, ...]],
    base_summaries: tuple[PlanSummary, ...],
    mbp_summaries: tuple[PlanSummary, ...],
    combined_summaries: tuple[PlanSummary, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "revised_universe_cost_comparison.csv"
    fields = ("plan_id", *EstimateRow.__dataclass_fields__)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for plan_id, rows in detail_rows.items():
            for row in rows:
                writer.writerow({"plan_id": plan_id, **asdict(row)})

    payload = {
        "dataset": settings.dataset,
        "latest_complete_end_exclusive": end.isoformat(),
        "revised_roots": REVISED_ROOTS,
        "mbp_roots": REVISED_MBP_ROOTS,
        "budget_limit_usd": settings.budget_limit_usd,
        "storage": asdict(storage),
        "working_space_multiplier": settings.working_space_multiplier,
        "safety_headroom_multiplier": settings.safety_headroom_multiplier,
        "base_plans": [asdict(summary) for summary in base_summaries],
        "standalone_mbp_plans": [asdict(summary) for summary in mbp_summaries],
        "combined_plans": [asdict(summary) for summary in combined_summaries],
        "detail_rows": {
            plan_id: [asdict(row) for row in rows] for plan_id, rows in detail_rows.items()
        },
        "warning": "Metadata estimates only. No plan is authorized and no data was downloaded.",
    }
    (output_dir / "revised_universe_cost_comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    def summary_row(summary: PlanSummary) -> str:
        return (
            f"| {summary.plan_id} | ${summary.total_cost_usd:,.2f} | "
            f"{summary.total_records:,} | {summary.total_billable_gb:,.2f} | "
            f"{summary.expected_working_bytes / 1e9:,.2f} | "
            f"{summary.required_free_with_headroom_bytes / 1e9:,.2f} | "
            f"{'PASS' if summary.budget_pass else 'FAIL'} | "
            f"{'PASS' if summary.storage_pass else 'FAIL'} |"
        )

    lines = [
        "# Revised Universe Databento Cost Comparison",
        "",
        "Metadata estimates only. No market data was downloaded, no batch job was submitted, and no scenario is authorized.",
        "",
        f"- Dataset: `{settings.dataset}`",
        f"- End boundary: `{end.isoformat()}` exclusive UTC",
        f"- Revised universe ({len(REVISED_ROOTS)}): {', '.join(REVISED_ROOTS)}",
        f"- MBP universe ({len(REVISED_MBP_ROOTS)}): {', '.join(REVISED_MBP_ROOTS)}",
        f"- Budget threshold: `${settings.budget_limit_usd:,.2f}`",
        f"- D: free: `{storage.free_bytes / 1e9:,.2f} GB` decimal (`{storage.free_bytes / (1024**3):,.2f} GiB`)",
        "",
        "## OHLCV-1m plus Definitions, Statistics, and Status",
        "",
        "Each metadata schema uses the same horizon as its OHLCV plan.",
        "",
        "| Plan | Cost USD | Records | Raw GB | Working GB | Required with headroom GB | Budget | Storage |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|",
        *(summary_row(summary) for summary in base_summaries),
        "",
        "## Standalone MBP-1 estimates for NQ/ES/CL/GC",
        "",
        "| Plan | Cost USD | Records | Raw GB | Working GB | Required with headroom GB | Budget | Storage |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|",
        *(summary_row(summary) for summary in mbp_summaries),
        "",
        "## Combined comparison",
        "",
        "These rows are arithmetic combinations of the independently estimated base and MBP plans.",
        "",
        "| Plan | Cost USD | Records | Raw GB | Working GB | Required with headroom GB | Budget | Storage |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|",
        *(summary_row(summary) for summary in combined_summaries),
        "",
        "## Interpretation",
        "",
        "- `PASS` only means the current estimate is within the configured budget or capacity gate.",
        "- All scenarios remain unauthorized until the user selects one explicitly.",
        "- Cost should be refreshed immediately before any acquisition because the budget margin may be small.",
        "- Billable raw size is not the same as final compressed download size.",
    ]
    (output_dir / "revised_universe_cost_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> int:
    """Run the complete revised comparison through metadata endpoints only."""
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    load_dotenv(override=False)
    settings = DataSettings.load(args.config_dir / "data.yaml")
    paths = resolve_data_paths(expected_drive=settings.expected_drive_windows, create=False)
    storage = StorageInfo.for_path(paths.root)
    client = create_historical_client()
    required_schemas = {"ohlcv-1m", "definition", "statistics", "status", "mbp-1"}
    verify_dataset_and_schemas(
        client,
        dataset=settings.dataset,
        required_schemas=required_schemas,
    )
    availability = client.metadata.get_dataset_range(dataset=settings.dataset)
    end = latest_complete_end(availability, required_schemas)

    detail_rows: dict[str, tuple[EstimateRow, ...]] = {}
    for years in OHLCV_YEARS:
        plan_id = f"BASE-{years}Y"
        detail_rows[plan_id] = _estimate(
            client,
            settings,
            label=f"{years} years of OHLCV-1m plus metadata for 15 markets",
            queries=_ohlcv_plan_queries(years=years, end=end, settings=settings),
        )
    for months in MBP_MONTHS:
        plan_id = f"MBP-{months}M"
        detail_rows[plan_id] = _estimate(
            client,
            settings,
            label=f"{months} month(s) of MBP-1 for NQ/ES/CL/GC",
            queries=_mbp_queries(months=months, end=end, settings=settings),
        )

    base_summaries = tuple(
        summarize(
            plan_id=f"BASE-{years}Y",
            description=f"{years} years OHLCV-1m plus definitions/statistics/status",
            rows=detail_rows[f"BASE-{years}Y"],
            settings=settings,
            storage=storage,
        )
        for years in OHLCV_YEARS
    )
    mbp_summaries = tuple(
        summarize(
            plan_id=f"MBP-{months}M",
            description=f"{months} month(s) MBP-1 for NQ/ES/CL/GC",
            rows=detail_rows[f"MBP-{months}M"],
            settings=settings,
            storage=storage,
        )
        for months in MBP_MONTHS
    )
    combined_summaries = tuple(
        summarize(
            plan_id=f"BASE-{years}Y+MBP-{months}M",
            description=f"{years}-year base plus {months}-month MBP-1",
            rows=detail_rows[f"BASE-{years}Y"] + detail_rows[f"MBP-{months}M"],
            settings=settings,
            storage=storage,
        )
        for years in OHLCV_YEARS
        for months in MBP_MONTHS
    )
    _write_reports(
        output_dir=args.reports_dir,
        end=end,
        settings=settings,
        storage=storage,
        detail_rows=detail_rows,
        base_summaries=base_summaries,
        mbp_summaries=mbp_summaries,
        combined_summaries=combined_summaries,
    )
    for summary in (*base_summaries, *mbp_summaries, *combined_summaries):
        print(
            f"{summary.plan_id}: cost=${summary.total_cost_usd:.2f}, "
            f"raw={summary.total_billable_gb:.2f} GB, "
            f"budget_pass={str(summary.budget_pass).lower()}, "
            f"storage_pass={str(summary.storage_pass).lower()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
