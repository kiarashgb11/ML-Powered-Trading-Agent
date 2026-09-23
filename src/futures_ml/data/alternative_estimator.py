"""Estimate ordered alternatives after the original data plan fails its gates."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from futures_ml.config.paths import StorageInfo, resolve_data_paths
from futures_ml.config.settings import DataSettings, UniverseSettings
from futures_ml.data.cost_estimator import EstimateQuery, EstimateRow, estimate_query
from futures_ml.data.databento_client import (
    create_historical_client,
    verify_dataset_and_schemas,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlternativePlan:
    """A non-authorized alternative preserved for user review."""

    priority: int
    name: str
    description: str
    total_records: int
    total_billable_bytes: int
    total_billable_gb: float
    total_cost_usd: float
    expected_working_bytes: int
    required_free_with_headroom_bytes: int
    budget_pass: bool
    storage_pass: bool
    changes_original_plan: bool = True
    download_authorized: bool = False


def _subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def _queries_for_schema(
    *,
    roots: tuple[str, ...],
    schema: str,
    years: int,
    end: date,
    settings: DataSettings,
) -> tuple[EstimateQuery, ...]:
    category = {"ohlcv-1s": "OHLCV-1s", "ohlcv-1m": "OHLCV-1m", "mbp-1": "MBP-1"}[schema]
    return tuple(
        EstimateQuery(
            category=category,
            root=root,
            request_symbol=settings.continuous_template.format(root=root),
            schema=schema,
            stype_in="continuous",
            start=_subtract_years(end, years).isoformat(),
            end=end.isoformat(),
        )
        for root in roots
    )


def _estimate_rows(
    client: Any,
    settings: DataSettings,
    queries: tuple[EstimateQuery, ...],
    label: str,
) -> tuple[EstimateRow, ...]:
    LOGGER.info("Estimating %s (%d rows)", label, len(queries))
    return tuple(estimate_query(client, settings, query) for query in queries)


def _plan(
    *,
    priority: int,
    name: str,
    description: str,
    rows: tuple[EstimateRow, ...],
    settings: DataSettings,
    storage: StorageInfo,
) -> AlternativePlan:
    total_records = sum(row.estimated_records for row in rows)
    total_bytes = sum(row.estimated_billable_bytes for row in rows)
    total_cost = sum(row.estimated_cost_usd for row in rows)
    working = int(total_bytes * settings.working_space_multiplier)
    required = int(working * settings.safety_headroom_multiplier)
    return AlternativePlan(
        priority=priority,
        name=name,
        description=description,
        total_records=total_records,
        total_billable_bytes=total_bytes,
        total_billable_gb=round(total_bytes / 1_000_000_000, 6),
        total_cost_usd=round(total_cost, 8),
        expected_working_bytes=working,
        required_free_with_headroom_bytes=required,
        budget_pass=total_cost <= settings.budget_limit_usd,
        storage_pass=storage.free_bytes >= required,
    )


def _load_original_rows(path: Path) -> tuple[EstimateRow, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(EstimateRow(**row) for row in payload["rows"])


def estimate_alternatives(
    *,
    client: Any,
    universe: UniverseSettings,
    settings: DataSettings,
    original_rows: tuple[EstimateRow, ...],
    end: date,
    storage: StorageInfo,
) -> tuple[AlternativePlan, ...]:
    """Estimate the requested reduction order plus one V1-aligned candidate."""
    all_roots = tuple(market.root for market in universe.markets)
    core_six = universe.core_mbp_markets
    core_four = ("NQ", "ES", "CL", "GC")
    metadata = tuple(
        row for row in original_rows if row.schema in {"definition", "statistics", "status"}
    )
    ohlcv_5y = tuple(row for row in original_rows if row.schema == "ohlcv-1s")

    mbp_2y = _estimate_rows(
        client,
        settings,
        _queries_for_schema(roots=core_six, schema="mbp-1", years=2, end=end, settings=settings),
        "two years of MBP-1 for six core markets",
    )
    mbp_1y = _estimate_rows(
        client,
        settings,
        _queries_for_schema(roots=core_six, schema="mbp-1", years=1, end=end, settings=settings),
        "one year of MBP-1 for six core markets",
    )
    mbp_1y_four = tuple(row for row in mbp_1y if row.root in core_four)
    ohlcv_4y = _estimate_rows(
        client,
        settings,
        _queries_for_schema(
            roots=all_roots, schema="ohlcv-1s", years=4, end=end, settings=settings
        ),
        "four years of OHLCV-1s for 26 markets",
    )
    ohlcv_3y = _estimate_rows(
        client,
        settings,
        _queries_for_schema(
            roots=all_roots, schema="ohlcv-1s", years=3, end=end, settings=settings
        ),
        "three years of OHLCV-1s for 26 markets",
    )
    ohlcv_1m_5y = _estimate_rows(
        client,
        settings,
        _queries_for_schema(
            roots=all_roots, schema="ohlcv-1m", years=5, end=end, settings=settings
        ),
        "five years of OHLCV-1m for 26 markets",
    )

    inputs = (
        (
            "A1",
            "5y OHLCV-1s + 2y MBP-1 (6 markets)",
            "First prescribed reduction: retain universal five-year 1-second bars and shorten MBP-1 to two years.",
            ohlcv_5y + metadata + mbp_2y,
        ),
        (
            "A2",
            "5y OHLCV-1s + 1y MBP-1 (6 markets)",
            "Second prescribed reduction: retain universal five-year 1-second bars and shorten MBP-1 to one year.",
            ohlcv_5y + metadata + mbp_1y,
        ),
        (
            "A3",
            "5y OHLCV-1s + 1y MBP-1 (NQ/ES/CL/GC)",
            "Third prescribed reduction: retain five-year 1-second bars and reduce the one-year MBP universe to four markets.",
            ohlcv_5y + metadata + mbp_1y_four,
        ),
        (
            "A4",
            "4y OHLCV-1s + 1y MBP-1 (NQ/ES/CL/GC)",
            "Fourth prescribed reduction: shorten universal 1-second history to four years after reducing MBP.",
            ohlcv_4y + metadata + mbp_1y_four,
        ),
        (
            "A5",
            "3y OHLCV-1s + 1y MBP-1 (NQ/ES/CL/GC)",
            "Fifth prescribed reduction: shorten universal 1-second history to three years after reducing MBP.",
            ohlcv_3y + metadata + mbp_1y_four,
        ),
        (
            "B1",
            "5y OHLCV-1m (26 markets), metadata, no MBP-1",
            "V1-oriented candidate: buy the one-minute resolution used by the first model and postpone all MBP-1. This changes the approved source granularity and requires explicit user approval.",
            ohlcv_1m_5y + metadata,
        ),
    )
    return tuple(
        _plan(
            priority=index,
            name=f"{code}: {title}",
            description=details,
            rows=rows,
            settings=settings,
            storage=storage,
        )
        for index, (code, title, details, rows) in enumerate(inputs, start=1)
    )


def estimate_one_minute_extensions(
    *,
    client: Any,
    universe: UniverseSettings,
    settings: DataSettings,
    original_rows: tuple[EstimateRow, ...],
    end: date,
    storage: StorageInfo,
    starting_priority: int,
) -> tuple[AlternativePlan, ...]:
    """Price shorter one-minute candidates without repeating earlier alternatives."""
    all_roots = tuple(market.root for market in universe.markets)
    metadata = tuple(
        row for row in original_rows if row.schema in {"definition", "statistics", "status"}
    )
    plans: list[AlternativePlan] = []
    for offset, years in enumerate((4, 3)):
        ohlcv = _estimate_rows(
            client,
            settings,
            _queries_for_schema(
                roots=all_roots,
                schema="ohlcv-1m",
                years=years,
                end=end,
                settings=settings,
            ),
            f"{years} years of OHLCV-1m for 26 markets",
        )
        plans.append(
            _plan(
                priority=starting_priority + offset,
                name=f"B{offset + 2}: {years}y OHLCV-1m (26 markets), metadata, no MBP-1",
                description=(
                    "V1-oriented candidate using the one-minute resolution consumed by the first "
                    "model, retaining five years of metadata, and postponing MBP-1. This changes "
                    "the approved source granularity and requires explicit user approval."
                ),
                rows=ohlcv + metadata,
                settings=settings,
                storage=storage,
            )
        )
    return tuple(plans)


def write_alternative_reports(
    plans: tuple[AlternativePlan, ...],
    *,
    output_dir: Path,
    budget_limit_usd: float,
    disk_free_bytes: int,
) -> None:
    """Write review-only alternative plan reports in three formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = tuple(AlternativePlan.__dataclass_fields__)
    with (output_dir / "data_cost_alternatives.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(plan) for plan in plans)

    payload = {
        "budget_limit_usd": budget_limit_usd,
        "disk_free_bytes": disk_free_bytes,
        "plans": [asdict(plan) for plan in plans],
        "warning": "Alternatives change the approved plan and are not download authorizations.",
    }
    (output_dir / "data_cost_alternatives.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Databento Alternative Plan Estimates",
        "",
        "The original plan failed both budget and storage gates. These alternatives are estimates for review only. No plan is authorized for download without explicit approval.",
        "",
        f"Budget limit: `${budget_limit_usd:,.2f}`",
        f"Available disk: `{disk_free_bytes / 1_000_000_000:,.2f} GB`",
        "",
        "| Priority | Plan | Cost USD | Raw GB | Working GB | Required with headroom GB | Budget | Storage |",
        "|---:|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    for plan in plans:
        lines.append(
            f"| {plan.priority} | {plan.name} | ${plan.total_cost_usd:,.2f} | "
            f"{plan.total_billable_gb:,.2f} | {plan.expected_working_bytes / 1e9:,.2f} | "
            f"{plan.required_free_with_headroom_bytes / 1e9:,.2f} | "
            f"{'PASS' if plan.budget_pass else 'FAIL'} | "
            f"{'PASS' if plan.storage_pass else 'FAIL'} |"
        )
    lines.extend(["", "## Details", ""])
    for plan in plans:
        lines.extend(
            [
                f"### {plan.name}",
                "",
                plan.description,
                "",
            ]
        )
    (output_dir / "data_cost_alternatives.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> int:
    """Run live metadata estimates for ordered alternative plans."""
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    load_dotenv(override=False)
    universe = UniverseSettings.load(args.config_dir / "universe.yaml")
    settings = DataSettings.load(args.config_dir / "data.yaml")
    paths = resolve_data_paths(expected_drive=settings.expected_drive_windows, create=False)
    storage = StorageInfo.for_path(paths.root)
    original_rows = _load_original_rows(args.reports_dir / "data_cost_report.json")
    end = date.fromisoformat(original_rows[0].end)
    client = create_historical_client()
    verify_dataset_and_schemas(
        client,
        dataset=settings.dataset,
        required_schemas={"ohlcv-1s", "ohlcv-1m", "mbp-1"},
    )
    existing_path = args.reports_dir / "data_cost_alternatives.json"
    if existing_path.exists():
        existing_payload = json.loads(existing_path.read_text(encoding="utf-8"))
        existing_plans = tuple(AlternativePlan(**plan) for plan in existing_payload["plans"])
    else:
        existing_plans = estimate_alternatives(
            client=client,
            universe=universe,
            settings=settings,
            original_rows=original_rows,
            end=end,
            storage=storage,
        )
    existing_names = {plan.name for plan in existing_plans}
    if any(name.startswith(("B2:", "B3:")) for name in existing_names):
        plans = existing_plans
    else:
        plans = existing_plans + estimate_one_minute_extensions(
            client=client,
            universe=universe,
            settings=settings,
            original_rows=original_rows,
            end=end,
            storage=storage,
            starting_priority=len(existing_plans) + 1,
        )
    write_alternative_reports(
        plans,
        output_dir=args.reports_dir,
        budget_limit_usd=settings.budget_limit_usd,
        disk_free_bytes=storage.free_bytes,
    )
    for plan in plans:
        print(
            f"{plan.name}: cost=${plan.total_cost_usd:.2f}, raw={plan.total_billable_gb:.2f} GB, "
            f"budget_pass={str(plan.budget_pass).lower()}, storage_pass={str(plan.storage_pass).lower()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
