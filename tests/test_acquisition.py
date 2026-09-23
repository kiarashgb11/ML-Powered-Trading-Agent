from __future__ import annotations

import json
from pathlib import Path

import pytest

from futures_ml.data.acquisition import Manifest, _load_authorized_report


def _report() -> dict[str, object]:
    rows = []
    roots = ("NQ", "ES", "RTY", "ZF", "ZN", "ZB", "CL", "NG", "GC", "HG", "6E", "6J", "6B", "ZC", "ZS", "ZW")
    for root in roots:
        for schema in ("ohlcv-1m", "definition", "statistics", "status"):
            rows.append(
                {
                    "root": root,
                    "schema": schema,
                    "start": "2021-09-23",
                    "end": "2026-09-23",
                    "request_symbol": f"{root}.FUT" if schema == "definition" else f"{root}.v.0",
                    "stype_in": "parent" if schema == "definition" else "continuous",
                    "estimated_records": 1,
                    "estimated_billable_bytes": 1,
                    "estimated_cost_usd": 1.0,
                }
            )
    return {"version": 1, "download_authorized": True, "total_cost_usd": 64.0, "rows": rows}


def test_manifest_round_trip_preserves_resume_state(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    first = Manifest.from_report(path, _report())
    first.entries[0].job_id = "job-123"
    first.entries[0].status = "submitted"
    first.save()
    loaded = Manifest.from_report(path, _report())
    assert len(loaded.entries) == 64
    assert loaded.entries[0].job_id == "job-123"
    assert path.with_suffix(".csv").exists()


def test_gate_rejects_unauthorized_report(tmp_path: Path) -> None:
    from futures_ml.config.settings import DataSettings

    report = _report()
    report["download_authorized"] = False
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not authorize"):
        _load_authorized_report(path, DataSettings.load(Path("config/data.yaml")))
