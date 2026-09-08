#!/usr/bin/env python3
"""Build the public snapshot from a strict allowlist in the parent Parallax repo.

The deployed assets are already committed under ``showcase``. This maintainer-only
script documents and reproduces the copy step without touching source data.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

SHOWCASE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = SHOWCASE_ROOT.parent

# Only these project files may cross the public-package boundary.
BINARY_ASSETS = {
    "data/tape_resolved.parquet": "data/tape_resolved.parquet",
    "data/features.parquet": "data/features.parquet",
}
JSON_ASSETS = {
    "eval/model_results.json": "eval/model_results.json",
    "eval/er_results.json": "eval/er_results.json",
    "eval/parse_audit_results.json": "eval/parse_audit_results.json",
}
FORBIDDEN_PARTS = {".env", "raw", "mlruns", "team", "handoff", "secret", "certificate", "proxy"}


def _safe_source(relative: str) -> Path:
    """Resolve one allowlisted source and reject risky path components."""
    if relative not in {*BINARY_ASSETS, *JSON_ASSETS, "data/filings_index.parquet",
                        "graph/queries.cypher", "eval/agent_results.json"}:
        raise ValueError(f"not allowlisted: {relative}")
    path = SOURCE_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    lowered = {part.lower() for part in path.relative_to(SOURCE_ROOT).parts}
    if lowered & FORBIDDEN_PARTS:
        raise ValueError(f"forbidden source path: {relative}")
    return path


def _scan_frame(frame: pd.DataFrame, label: str) -> None:
    """Reject local paths or credential-shaped values hidden in table columns."""
    needles = ("/" + "Users/", "file:" + "//", "BEGIN " + "PRIVATE KEY", "GEMINI_" + "API_KEY")
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        values = frame[column].dropna().astype(str)
        if any(values.str.contains(needle, regex=False).any() for needle in needles):
            raise ValueError(f"unsafe text in {label}:{column}")


def _copy_public_data() -> None:
    for source_name, destination_name in BINARY_ASSETS.items():
        source = _safe_source(source_name)
        frame = pd.read_parquet(source)
        _scan_frame(frame, source_name)
        shutil.copyfile(source, SHOWCASE_ROOT / destination_name)

    # Raw filing paths are deliberately omitted. The index retains only fields
    # needed to construct public SEC accession links.
    source = _safe_source("data/filings_index.parquet")
    filings = pd.read_parquet(source).drop(columns=["local_path"], errors="ignore")
    _scan_frame(filings, "data/filings_index.parquet")
    filings.to_parquet(SHOWCASE_ROOT / "data/filings_index.parquet", index=False)


def _copy_evidence() -> None:
    for source_name, destination_name in JSON_ASSETS.items():
        source = _safe_source(source_name)
        payload = json.loads(source.read_text())
        text = json.dumps(payload, indent=1, sort_keys=False) + "\n"
        (SHOWCASE_ROOT / destination_name).write_text(text)


def _write_graph_queries() -> None:
    """Extract the three parameterized queries without local-service or team-record comments."""
    source = _safe_source("graph/queries.cypher")
    text = source.read_text()
    markers = ["// === Q1", "// === Q2", "// === Q3"]
    starts = [text.index(marker) for marker in markers]
    blocks = [text[starts[i]: starts[i + 1] if i + 1 < len(starts) else len(text)]
              for i in range(3)]
    public = [
        "// Parallax public showcase: three saved, parameterized graph questions.",
        "// Schema: (:RawName)-[:RESOLVES_TO]->(:Borrower)",
        "//         (:Lender)-[:HOLDS]->(:Borrower)",
        "//         (:RawName)-[:SIMILAR_TO]-(:RawName)",
        "",
    ]
    for block in blocks:
        lines = block.splitlines()
        public.extend(line.replace(" — ", " - ") for line in lines
                      if line.startswith("// ===") or line.startswith("// Question")
                      or line.startswith("// :param"))
        query_start = next(i for i, line in enumerate(lines) if line.startswith("MATCH "))
        public.extend(line for line in lines[query_start:] if not line.startswith("//"))
        public.append("")
    (SHOWCASE_ROOT / "graph/queries.cypher").write_text("\n".join(public).rstrip() + "\n")


def _write_agent_examples() -> None:
    """Extract four frozen examples without trace IDs, run IDs, tokens, or model calls."""
    source = json.loads(_safe_source("eval/agent_results.json").read_text())
    rows = {row["question_id"]: row for row in source["rows"]}
    choices = [
        ("C2", "Exact-label comparables and neighbour disclosure."),
        ("B1", "The verifier rejected an under-cited aggregate and replaced the answer with a refusal."),
        ("E1", "Archived graph traversal result; the public app does not query Neo4j."),
        ("R2", "A capacity-sized observed-mark list with filing accessions, not model scores."),
    ]
    examples = []
    for question_id, learning in choices:
        row = rows[question_id]
        citations = sorted({accession for claim in row.get("claims", [])
                            for accession in claim.get("accessions", [])})
        examples.append({
            "question_id": question_id,
            "question": row["question"],
            "tool": row["tool_expected"],
            "saved_answer": row["answer"],
            "citations": citations,
            "verifier_pass": bool(row["pass"]),
            "verifier_strikes": int(row["verifier_strikes"]),
            "verifier_reason": row.get("reason") or row.get("why_failed") or "All recorded claims verified.",
            "learning": learning,
        })
    summary = {
        "status": "archived experiment evidence; canned examples; no live model call",
        "source_summary": {
            key: source[key] for key in (
                "n", "repeats_completed", "accuracy", "accuracy_n_pass",
                "citation_groundedness", "n_questions_with_claims", "claims_total",
                "claims_verified", "verifier_strikes", "disagreement_count"
            )
        },
        "examples": examples,
    }
    (SHOWCASE_ROOT / "eval/agent_examples.json").write_text(json.dumps(summary, indent=2) + "\n")


def _scan_outputs() -> None:
    needles = ("/" + "Users/", "file:" + "//", "BEGIN " + "PRIVATE KEY", "AI" + "za", "GEMINI_" + "API_KEY=")
    for path in SHOWCASE_ROOT.rglob("*"):
        if not path.is_file() or path.suffix in {".parquet", ".pyc"}:
            continue
        text = path.read_text(errors="ignore")
        for needle in needles:
            if needle in text:
                raise ValueError(f"unsafe public text {needle!r} in {path.relative_to(SHOWCASE_ROOT)}")


def main() -> None:
    _copy_public_data()
    _copy_evidence()
    _write_graph_queries()
    _write_agent_examples()
    _scan_outputs()
    print("Built allowlisted public snapshot without changing source data.")


if __name__ == "__main__":
    main()
