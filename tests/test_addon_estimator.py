from __future__ import annotations

from futures_ml.data.addon_estimator import ADDON_ROOTS


def test_addon_roots_match_requested_order() -> None:
    assert ADDON_ROOTS == ("ZF", "6C", "6A")
