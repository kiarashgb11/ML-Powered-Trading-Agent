from __future__ import annotations

from pathlib import Path

import pytest

from futures_ml.config.paths import DataPaths, PathConfigurationError, resolve_data_paths


def test_data_paths_are_all_derived_from_root(tmp_path: Path) -> None:
    paths = DataPaths.from_root(tmp_path / "external-data")
    assert paths.raw_data_root == paths.root / "raw"
    assert paths.processed_data_root == paths.root / "processed"
    assert paths.model_artifact_root == paths.root / "artifacts" / "models"
    assert all(path == paths.root or paths.root in path.parents for path in paths.directories())


def test_missing_data_root_never_falls_back_to_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FUTURES_ML_DATA_ROOT", raising=False)
    with pytest.raises(PathConfigurationError, match="not configured"):
        resolve_data_paths()
