from datetime import date
from pathlib import Path

from futures_ml.config.settings import DataSettings
from futures_ml.data.final_acquisition import FINAL_ROOTS, build_final_queries


def test_final_plan_has_exact_approved_requests() -> None:
    settings = DataSettings.load(Path("config/data.yaml"))
    queries = build_final_queries(date(2026, 9, 23), settings)
    assert len(FINAL_ROOTS) == 16
    assert len(set(FINAL_ROOTS)) == 16
    assert len(queries) == 64
    assert {query.start for query in queries} == {"2021-09-23"}
    assert {query.end for query in queries} == {"2026-09-23"}
    assert sum(query.schema == "ohlcv-1m" for query in queries) == 16
    assert all(
        query.stype_in == "parent" and query.request_symbol.endswith(".FUT")
        for query in queries
        if query.schema == "definition"
    )
