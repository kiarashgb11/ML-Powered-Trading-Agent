from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from futures_ml.config.paths import StorageInfo
from futures_ml.config.settings import DataSettings, UniverseSettings
from futures_ml.data.cost_estimator import (
    EstimateRow,
    build_queries,
    build_report,
    latest_complete_end,
    write_reports,
)


def _settings() -> tuple[UniverseSettings, DataSettings]:
    return (
        UniverseSettings.load(Path("config/universe.yaml")),
        DataSettings.load(Path("config/data.yaml")),
    )


def test_plan_has_all_universal_and_core_mbp_requests() -> None:
    universe, settings = _settings()
    queries = build_queries(universe, settings, date(2026, 9, 22))
    assert len(queries) == 26 * 4 + 6
    assert sum(query.schema == "ohlcv-1s" for query in queries) == 26
    assert sum(query.schema == "mbp-1" for query in queries) == 6
    definition = next(query for query in queries if query.schema == "definition")
    assert definition.stype_in == "parent"
    assert definition.request_symbol.endswith(".FUT")


def test_latest_complete_end_uses_earliest_schema_boundary() -> None:
    availability = {
        "schema": {
            "ohlcv-1s": {"end": "2026-09-22T14:00:00Z"},
            "mbp-1": {"end": "2026-09-21T23:00:00Z"},
        }
    }
    result = latest_complete_end(
        availability,
        {"ohlcv-1s", "mbp-1"},
        today_utc=date(2026, 9, 22),
    )
    assert result == date(2026, 9, 21)


def test_report_writes_all_formats_and_applies_gates(tmp_path: Path) -> None:
    _, settings = _settings()
    row = EstimateRow(
        category="OHLCV-1s",
        root="ES",
        request_symbol="ES.v.0",
        schema="ohlcv-1s",
        stype_in="continuous",
        start="2021-09-22",
        end="2026-09-22",
        estimated_records=100,
        estimated_billable_bytes=1_000_000_000,
        estimated_gb=1.0,
        estimated_cost_usd=1.0,
    )
    report = build_report(
        rows=(row,),
        settings=settings,
        end=date(2026, 9, 22),
        data_root=tmp_path,
        storage=StorageInfo(drive="D:", total_bytes=100_000_000_000, free_bytes=50_000_000_000),
    )
    write_reports(report, tmp_path / "reports")
    assert report.download_authorized
    assert (tmp_path / "reports/data_cost_report.csv").exists()
    assert (tmp_path / "reports/data_cost_report.md").exists()
    payload = json.loads((tmp_path / "reports/data_cost_report.json").read_text(encoding="utf-8"))
    assert payload["total_records"] == 100
    assert payload["download_authorized"] is True
