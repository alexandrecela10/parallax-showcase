"""Package-local paths and public constants. No environment or service config."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PACKAGE_ROOT / "data"
EVAL_DIR = PACKAGE_ROOT / "eval"
GRAPH_DIR = PACKAGE_ROOT / "graph"
DOCS_DIR = PACKAGE_ROOT / "docs"

TICKERS = ("ARCC", "OBDC", "FSK", "GBDC", "MAIN", "TSLX", "HTGC", "PSEC")
DATA_AS_OF = "2026-09-06"
SEC_ARCHIVES_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{filename}"
)
