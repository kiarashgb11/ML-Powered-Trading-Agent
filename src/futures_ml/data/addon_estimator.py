"""Estimate optional five-year market add-ons to the intended BASE-5Y plan."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

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
from futures_ml.data.revised_plan_estimator import summarize

LOGGER = logging.getLogger(__name__)

ADDON_ROOTS = ("ZF", "6C", "6A")
ADDON_SCHEMAS = ("ohlcv-1m", "definition", "statistics", "status")


@dataclass(frozen=True)
class AddonCombination:
    """One review-only combination layered onto BASE-5Y."""

    addon_roots: tuple[str, ...]
    addon_cost_usd: float
    combined_cost_usd: float
    combined_records: int
    combined_billable_bytes: int
    combined_billable_gb: float
    combined_expected_working_bytes: int
    combined_required_free_with_headroom_bytes: int
    budget_remaining_usd: float
    budget_pass: bool
    storage_pass: bool
    download_authorized: bool = False


def _subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def _addon_queries(
    *,
    root: str,
    end: date,
    settings: DataSettings,
) -> tuple[EstimateQuery, ...]:
    categories = {
        "ohlcv-1m": "OHLCV-1m",
        "definition": "Definitions",
        "statistics": "Statistics",
        "status": "Status",
    }
    start = _subtract_years(end, 5).isoformat()
    queries: list[EstimateQuery] = []
    for schema in ADDON_SCHEMAS:
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


def build_combinations(
    *,
    base_rows: tuple[EstimateRow, ...],
    addon_rows: dict[str, tuple[EstimateRow, ...]],
    settings: DataSettings,
    storage: StorageInfo,
) -> tuple[AddonCombination, ...]:
    """Build every non-empty add-on subset in stable size/name order."""
    base_cost = sum(row.estimated_cost_usd for row in base_rows)
    combinations: list[AddonCombination] = []
    for size in range(1, len(ADDON_ROOTS) + 1):
        for roots in itertools.combinations(ADDON_ROOTS, size):
            selected = tuple(row for root in roots for row in addon_rows[root])
            combined = base_rows + selected
            summary = summarize(
                plan_id="BASE-5Y+" + "+".join(roots),
                description="BASE-5Y with optional roots " + ", ".join(roots),
                rows=combined,
                settings=settings,
                storage=storage,
            )
            addon_cost = sum(row.estimated_cost_usd for row in selected)
            combinations.append(
                AddonCombination(
                    addon_roots=roots,
                    addon_cost_usd=round(addon_cost, 8),
                    combined_cost_usd=summary.total_cost_usd,
                    combined_records=summary.total_records,
                    combined_billable_bytes=summary.total_billable_bytes,
                    combined_billable_gb=summary.total_billable_gb,
                    combined_expected_working_bytes=summary.expected_working_bytes,
                    combined_required_free_with_headroom_bytes=(
                        summary.required_free_with_headroom_bytes
                    ),
                    budget_remaining_usd=round(
                        settings.budget_limit_usd - (base_cost + addon_cost), 8
                    ),
                    budget_pass=summary.budget_pass,
                    storage_pass=summary.storage_pass,
                )
            )
    return tuple(combinations)


def _load_base_report(path: Path) -> tuple[date, tuple[str, ...], tuple[EstimateRow, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    end = date.fromisoformat(payload["latest_complete_end_exclusive"])
    roots = tuple(payload["revised_roots"])
    rows = tuple(EstimateRow(**row) for row in payload["detail_rows"]["BASE-5Y"])
    return end, roots, rows


def _write_reports(
    *,
    output_dir: Path,
    end: date,
    base_roots: tuple[str, ...],
    base_rows: tuple[EstimateRow, ...],
    addon_rows: dict[str, tuple[EstimateRow, ...]],
    combinations: tuple[AddonCombination, ...],
    settings: DataSettings,
    storage: StorageInfo,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = tuple(AddonCombination.__dataclass_fields__)
    with (output_dir / "base5y_addon_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for combination in combinations:
            row = asdict(combination)
            row["addon_roots"] = "+".join(combination.addon_roots)
            writer.writerow(row)

    base_summary = summarize(
        plan_id="BASE-5Y",
        description="Intended five-year 15-market base",
        rows=base_rows,
        settings=settings,
        storage=storage,
    )
    addon_summaries = {
        root: summarize(
            plan_id=f"ADD-{root}",
            description=f"Five-year OHLCV-1m plus metadata for {root}",
            rows=rows,
            settings=settings,
            storage=storage,
        )
        for root, rows in addon_rows.items()
    }
    payload = {
        "dataset": settings.dataset,
        "end_exclusive": end.isoformat(),
        "base_roots": base_roots,
        "addon_roots": ADDON_ROOTS,
        "budget_limit_usd": settings.budget_limit_usd,
        "storage": asdict(storage),
        "base_summary": asdict(base_summary),
        "addon_summaries": {root: asdict(summary) for root, summary in addon_summaries.items()},
        "combinations": [asdict(combination) for combination in combinations],
        "addon_detail_rows": {
            root: [asdict(row) for row in rows] for root, rows in addon_rows.items()
        },
        "warning": "Metadata estimates only. No combination is authorized and no data was downloaded.",
    }
    (output_dir / "base5y_addon_comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# BASE-5Y Optional Market Add-on Comparison",
        "",
        "Metadata estimates only. BASE-5Y is the intended base, but neither it nor any add-on combination is authorized for download by this report.",
        "",
        f"- Dataset: `{settings.dataset}`",
        f"- End boundary: `{end.isoformat()}` exclusive UTC",
        f"- Intended base roots ({len(base_roots)}): {', '.join(base_roots)}",
        f"- Optional roots: {', '.join(ADDON_ROOTS)}",
        f"- Budget threshold: `${settings.budget_limit_usd:,.2f}`",
        f"- Current D: free: `{storage.free_bytes / 1e9:,.2f} GB` decimal (`{storage.free_bytes / (1024**3):,.2f} GiB`)",
        "",
        "## Intended base",
        "",
        f"BASE-5Y is estimated at `${base_summary.total_cost_usd:,.2f}`, `{base_summary.total_billable_gb:,.2f} GB` billable raw, and `{base_summary.required_free_with_headroom_bytes / 1e9:,.2f} GB` required with headroom.",
        "",
        "## Individual add-on estimates",
        "",
        "Each root includes five years of OHLCV-1m, Definitions, Statistics, and Status.",
        "",
        "| Root | Add-on cost USD | Records | Raw GB |",
        "|---|---:|---:|---:|",
    ]
    for root in ADDON_ROOTS:
        summary = addon_summaries[root]
        lines.append(
            f"| {root} | ${summary.total_cost_usd:,.2f} | {summary.total_records:,} | "
            f"{summary.total_billable_gb:,.3f} |"
        )
    lines.extend(
        [
            "",
            "## BASE-5Y plus add-on combinations",
            "",
            "| Add-on roots | Add-on cost USD | Combined cost USD | Budget remaining USD | Combined raw GB | Required with headroom GB | Budget | Storage |",
            "|---|---:|---:|---:|---:|---:|:---:|:---:|",
        ]
    )
    for combination in combinations:
        lines.append(
            f"| {' + '.join(combination.addon_roots)} | ${combination.addon_cost_usd:,.2f} | "
            f"${combination.combined_cost_usd:,.2f} | "
            f"${combination.budget_remaining_usd:,.2f} | "
            f"{combination.combined_billable_gb:,.3f} | "
            f"{combination.combined_required_free_with_headroom_bytes / 1e9:,.2f} | "
            f"{'PASS' if combination.budget_pass else 'FAIL'} | "
            f"{'PASS' if combination.storage_pass else 'FAIL'} |"
        )
    fitting = tuple(combination for combination in combinations if combination.budget_pass)
    lines.extend(["", "## Result", ""])
    if fitting:
        max_count = max(len(combination.addon_roots) for combination in fitting)
        largest = tuple(
            combination for combination in fitting if len(combination.addon_roots) == max_count
        )
        lines.append(
            "Largest fitting combination(s): "
            + ", ".join(" + ".join(combination.addon_roots) for combination in largest)
            + "."
        )
    else:
        lines.append("No add-on combination fits the automatic budget threshold.")
    lines.extend(
        [
            "",
            "All combinations remain review-only. Refresh the estimate immediately before any download.",
        ]
    )
    (output_dir / "base5y_addon_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> int:
    """Estimate the optional roots and all combinations without downloading data."""
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    load_dotenv(override=False)
    settings = DataSettings.load(args.config_dir / "data.yaml")
    paths = resolve_data_paths(expected_drive=settings.expected_drive_windows, create=False)
    storage = StorageInfo.for_path(paths.root)
    report_path = args.reports_dir / "revised_universe_cost_comparison.json"
    base_end, base_roots, base_rows = _load_base_report(report_path)
    if set(ADDON_ROOTS) & set(base_roots):
        raise ValueError("An optional add-on root is already present in BASE-5Y")

    client = create_historical_client()
    required_schemas = set(ADDON_SCHEMAS)
    verify_dataset_and_schemas(
        client,
        dataset=settings.dataset,
        required_schemas=required_schemas,
    )
    current_end = latest_complete_end(
        client.metadata.get_dataset_range(dataset=settings.dataset), required_schemas
    )
    if current_end < base_end:
        raise RuntimeError(
            "Current availability does not cover the saved BASE-5Y boundary; refresh the base "
            "comparison first."
        )

    addon_rows: dict[str, tuple[EstimateRow, ...]] = {}
    for root in ADDON_ROOTS:
        LOGGER.info("Estimating five-year OHLCV-1m plus metadata for %s", root)
        addon_rows[root] = tuple(
            estimate_query(client, settings, query)
            for query in _addon_queries(root=root, end=base_end, settings=settings)
        )
    combinations = build_combinations(
        base_rows=base_rows,
        addon_rows=addon_rows,
        settings=settings,
        storage=storage,
    )
    _write_reports(
        output_dir=args.reports_dir,
        end=base_end,
        base_roots=base_roots,
        base_rows=base_rows,
        addon_rows=addon_rows,
        combinations=combinations,
        settings=settings,
        storage=storage,
    )
    for combination in combinations:
        roots = "+".join(combination.addon_roots)
        print(
            f"{roots}: addon=${combination.addon_cost_usd:.2f}, "
            f"combined=${combination.combined_cost_usd:.2f}, "
            f"remaining=${combination.budget_remaining_usd:.2f}, "
            f"budget_pass={str(combination.budget_pass).lower()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
