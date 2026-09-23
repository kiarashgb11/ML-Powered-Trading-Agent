from __future__ import annotations

from datetime import date

from futures_ml.data.revised_plan_estimator import subtract_months


def test_subtract_months_handles_year_boundary() -> None:
    assert subtract_months(date(2026, 1, 22), 3) == date(2025, 10, 22)


def test_subtract_months_clamps_end_of_month() -> None:
    assert subtract_months(date(2024, 5, 31), 3) == date(2024, 2, 29)
