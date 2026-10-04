"""Persist / load the deployable model bundle (the 'champion')."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from churn.config import CHAMPION_DIR


def save_bundle(
    directory: Path, pipe: Pipeline, metadata: dict[str, Any], reference: pd.DataFrame | None = None
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, directory / "model.joblib")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    if reference is not None:
        reference.to_csv(directory / "reference.csv", index=False)


def load_bundle(directory: Path = CHAMPION_DIR) -> tuple[Pipeline, dict[str, Any]]:
    pipe = joblib.load(directory / "model.joblib")
    meta = json.loads((directory / "metadata.json").read_text())
    return pipe, meta


def load_reference(directory: Path = CHAMPION_DIR) -> pd.DataFrame | None:
    f = directory / "reference.csv"
    return pd.read_csv(f, dtype={"SeniorCitizen": str}) if f.exists() else None


def promote(candidate_dir: Path, champion_dir: Path = CHAMPION_DIR) -> None:
    """Atomically replace the champion with the candidate bundle."""
    tmp = champion_dir.with_name(champion_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(candidate_dir, tmp)
    if champion_dir.exists():
        shutil.rmtree(champion_dir)
    tmp.rename(champion_dir)
