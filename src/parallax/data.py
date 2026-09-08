"""Small cached readers for the immutable files shipped with the showcase."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from parallax import config


@lru_cache(maxsize=None)
def _parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(config.DATA_DIR / name)


def tape() -> pd.DataFrame:
    return _parquet("tape_resolved.parquet")


def features() -> pd.DataFrame:
    return _parquet("features.parquet")


def filings() -> pd.DataFrame:
    return _parquet("filings_index.parquet")


@lru_cache(maxsize=None)
def load_json(path: str) -> dict:
    return json.loads((config.PACKAGE_ROOT / path).read_text())


def predictions() -> pd.DataFrame:
    """Return the compatible empty frame: public review lists never use cached scores."""
    return pd.DataFrame(columns=["borrower_id", "period_end", "score_model"])


def text(path: str | Path) -> str:
    return (config.PACKAGE_ROOT / path).read_text()
