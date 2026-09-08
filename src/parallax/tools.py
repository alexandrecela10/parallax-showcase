"""Deterministic tape tools used by every interactive showcase screen.

These functions read only packaged Parquet files. They make no service, model, or
network calls. Marks use non-netting, non-equity rows with positive cost.
"""

from __future__ import annotations

import re
from functools import lru_cache

import duckdb
import pandas as pd
from rapidfuzz import fuzz, process

from parallax import config, data

ROW_CAP = 50
THIN_N = 10
DEBT_ROW_LABEL = "is_netting_row = false; instrument_bucket <> 'equity'; cost > 0"


def _period(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d")


def _records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


@lru_cache(maxsize=1)
def coverage() -> dict:
    """Compute snapshot coverage from the packaged tape with in-process DuckDB."""
    tape = data.tape()
    summary = duckdb.sql(
        """SELECT count(DISTINCT filer_ticker), count(DISTINCT period_end),
                  CAST(min(period_end) AS VARCHAR), CAST(max(period_end) AS VARCHAR),
                  count(DISTINCT borrower_id)
           FROM tape"""
    ).fetchone()
    present = set(tape["filer_ticker"].dropna())
    return {
        "bdcs": [ticker for ticker in config.TICKERS if ticker in present],
        "n_filers": int(summary[0]),
        "n_quarters": int(summary[1]),
        "first_quarter": summary[2][:10],
        "last_quarter": summary[3][:10],
        "n_borrowers": int(summary[4]),
        "n_positions": int(len(tape)),
        "data_as_of": config.DATA_AS_OF,
    }


@lru_cache(maxsize=1)
def filing_urls() -> dict[str, str]:
    result = {}
    for row in data.filings().itertuples():
        result[str(row.accession)] = config.SEC_ARCHIVES_URL.format(
            cik=int(row.cik),
            accession_nodash=str(row.accession).replace("-", ""),
            filename=row.primary_document,
        )
    return result


def industry_choices() -> list[str]:
    counts = data.tape().loc[data.tape()["industry"].notna(), "industry"].value_counts()
    return [label for label in counts.index if not re.search(r"\d{4}", str(label))]


_SIZE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(b|bn|billion|mm|m|million|k|thousand)?", re.I)
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6,
         "b": 1e9, "bn": 1e9, "billion": 1e9, "": 1.0}


def parse_size_range(value: str | None) -> tuple[float | None, float | None]:
    if not value or str(value).strip().lower() in {"", "any", "all", "none"}:
        return None, None
    text = str(value).lower().replace(",", "")
    numbers = []
    for match in _SIZE_RE.finditer(text):
        number = float(match.group(1))
        suffix = (match.group(2) or "").lower()
        number *= _MULT[suffix]
        if not suffix and number < 1_000:
            number *= 1e6
        numbers.append(number)
    if not numbers:
        return None, None
    if len(numbers) == 1:
        if any(word in text for word in ("under", "below", "less", "up to", "<")):
            return None, numbers[0]
        return numbers[0], None
    return min(numbers[:2]), max(numbers[:2])


def _normal_label(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(value).lower()).strip()


def _related_labels(request: str, labels: list[str]) -> list[str]:
    want = _normal_label(request)
    tokens = set(want.split()) - {"and", "services", "service", "provider", "providers"}
    if "software" in tokens:
        related = lambda label: "software" in _normal_label(label).split()
    elif {"health", "care"} <= tokens or "healthcare" in tokens:
        related = lambda label: "health care" in _normal_label(label) or "healthcare" in _normal_label(label).split()
    else:
        related = lambda label: bool(tokens & set(_normal_label(label).split()))
    universe = data.tape()["industry"].dropna().unique()
    return sorted(label for label in universe if label not in labels and related(label))


def get_comparables(industry: str, size_range: str, instrument_type: str) -> dict:
    """Reproduce the exact-label, latest-filing comparable-position workflow."""
    tape = data.tape().copy()
    tape["period"] = _period(tape["period_end"])
    labels = sorted(label for label in tape["industry"].dropna().unique()
                    if _normal_label(label) == _normal_label(industry))
    latest = tape.groupby("filer_ticker")["period"].transform("max")
    mask = tape["period"].eq(latest) & tape["spread_bps"].notna() & ~tape["is_netting_row"]
    mask &= tape["industry"].isin(labels) if labels else False
    instrument = _normal_label(instrument_type).replace(" ", "_")
    if instrument not in {"", "any", "all", "none"}:
        mask &= tape["instrument_bucket"].eq(instrument)
    low, high = parse_size_range(size_range)
    if low is not None:
        mask &= tape["principal"].ge(low)
    if high is not None:
        mask &= tape["principal"].le(high)
    matches = tape.loc[mask].sort_values(["spread_bps", "row_id"])

    neighbours = {}
    for label in _related_labels(industry, labels):
        neighbour_mask = tape["period"].eq(latest) & tape["spread_bps"].notna() & ~tape["is_netting_row"]
        neighbour_mask &= tape["industry"].eq(label)
        if instrument not in {"", "any", "all", "none"}:
            neighbour_mask &= tape["instrument_bucket"].eq(instrument)
        if low is not None:
            neighbour_mask &= tape["principal"].ge(low)
        if high is not None:
            neighbour_mask &= tape["principal"].le(high)
        nearby = tape.loc[neighbour_mask]
        if len(nearby):
            neighbours[label] = {
                "n": int(len(nearby)),
                "median_spread_bps": float(nearby["spread_bps"].median()),
                "n_filers": int(nearby["filer_ticker"].nunique()),
            }

    columns = ["accession", "filer_ticker", "period", "borrower_name", "industry",
               "instrument_bucket", "spread_bps", "principal", "cost", "fair_value"]
    return {
        "tool": "get_comparables",
        "period_basis": "latest period each filer reported",
        "n": int(len(matches)),
        "n_filers": int(matches["filer_ticker"].nunique()),
        "p25_spread_bps": None if matches.empty else float(matches["spread_bps"].quantile(.25)),
        "median_spread_bps": None if matches.empty else float(matches["spread_bps"].median()),
        "p75_spread_bps": None if matches.empty else float(matches["spread_bps"].quantile(.75)),
        "industries_matched": {label: int(matches["industry"].eq(label).sum()) for label in labels},
        "related_labels_not_used": neighbours,
        "filer_mix": matches["filer_ticker"].value_counts().to_dict(),
        "thin": len(matches) < THIN_N,
        "accessions": sorted(matches["accession"].dropna().unique()),
        "rows_truncated": len(matches) > ROW_CAP,
        "rows": _records(matches[columns].head(ROW_CAP).rename(columns={"period": "period_end"})),
        "data_as_of": config.DATA_AS_OF,
    }


def _debt_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[~frame["is_netting_row"] & frame["instrument_bucket"].ne("equity")
                     & frame["cost"].fillna(0).gt(0)]


def _normal_name(value: str) -> str:
    """Apply the public resolver's core rule: punctuation and legal suffixes do not identify a firm."""
    tokens = _normal_label(value).split()
    suffixes = {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "lp", "plc"}
    while tokens and tokens[-1] in suffixes:
        tokens.pop()
    return " ".join(tokens)


def _match_borrower(name: str) -> tuple[str | None, float]:
    names = data.tape()["borrower_name"].dropna().unique().tolist()
    normal = {candidate: _normal_name(candidate) for candidate in names}
    want = _normal_name(name)
    candidates = process.extract(want, list(normal.values()), scorer=fuzz.token_set_ratio, limit=10)
    scored = [(candidate, min(fuzz.token_set_ratio(want, candidate),
                              fuzz.token_sort_ratio(want, candidate))) for candidate, _, _ in candidates]
    if not scored:
        return None, 0.0
    best, score = max(scored, key=lambda item: item[1])
    matched = next(candidate for candidate, key in normal.items() if key == best)
    return (matched if score >= 85 else None), float(score)


def get_borrower(name: str) -> dict:
    """Return debt-only holder marks for the best conservative borrower-name match."""
    matched, score = _match_borrower(name)
    if matched is None:
        return {"tool": "get_borrower", "not_found": True, "match_score": score,
                "holders": [], "accessions": [], "data_as_of": config.DATA_AS_OF}
    rows = data.tape().loc[data.tape()["borrower_name"].eq(matched)].copy()
    rows["period_end"] = _period(rows["period_end"])
    holders = []
    for (period, ticker), group in rows.loc[~rows["is_netting_row"]].groupby(["period_end", "filer_ticker"]):
        debt = _debt_rows(group)
        cost, fair_value = debt["cost"].sum(), debt["fair_value"].sum()
        holders.append({
            "period_end": period,
            "filer_ticker": ticker,
            "principal": float(group["principal"].sum()),
            "cost": float(cost),
            "fair_value": float(fair_value),
            "fvc": round(float(fair_value / cost), 4) if cost else None,
            "n_debt_rows": int(len(debt)),
            "fvc_row_filter": DEBT_ROW_LABEL,
            "non_accrual": bool(group["non_accrual_flag"].fillna(False).any()),
            "accessions": sorted(group["accession"].dropna().unique()),
        })
    latest = max(holder["period_end"] for holder in holders)
    latest_marks = [holder["fvc"] for holder in holders
                    if holder["period_end"] == latest and holder["fvc"] is not None]
    return {
        "tool": "get_borrower", "not_found": False, "matched_name": matched,
        "match_score": round(score, 1), "borrower_id": str(rows["borrower_id"].iloc[0]),
        "latest_period": latest, "n_holders_latest": len(latest_marks),
        "dispersion_latest": round(max(latest_marks) - min(latest_marks), 4)
        if len(latest_marks) >= 2 else None,
        "holders": holders, "accessions": sorted(rows["accession"].dropna().unique()),
        "data_as_of": config.DATA_AS_OF,
    }


def review_periods() -> list[str]:
    return sorted(data.features()["period_end"].astype(str).unique())


def review_list(period: str, k: int = 15) -> dict:
    """Rank by observed fvc_min only. No model score or run provenance is loaded."""
    periods = review_periods()
    if period not in periods:
        return {"tool": "review_list", "out_of_coverage": True, "valid_periods": periods,
                "reason": "The packaged feature panel has no row for that quarter.", "rows": [], "n": 0}
    k = max(1, min(int(k), 200))
    features = data.features()

    def top(target: str) -> pd.DataFrame:
        frame = features.loc[features["period_end"].astype(str).eq(target) & features["in_universe"]].copy()
        return frame.sort_values(["fvc_min", "borrower_id"], na_position="last").head(k)

    current = top(period)
    index = periods.index(period)
    previous = periods[index - 1] if index else None
    previous_ids = set(top(previous)["borrower_id"].astype(str)) if previous else set()
    tape = data.tape().copy()
    tape["period"] = _period(tape["period_end"])
    rows = []
    for rank, feature in enumerate(current.itertuples(), start=1):
        source = tape.loc[tape["borrower_id"].astype(str).eq(str(feature.borrower_id))
                          & tape["period"].eq(period) & ~tape["is_netting_row"]]
        detail = get_borrower(feature.borrower_name)
        holders = [holder for holder in detail["holders"] if holder["period_end"] == period]
        rows.append({
            "rank": rank, "borrower_id": str(feature.borrower_id),
            "borrower_name": feature.borrower_name, "industry": feature.industry,
            "fvc_min": None if pd.isna(feature.fvc_min) else float(feature.fvc_min),
            "d_fvc_1q": None if pd.isna(feature.d_fvc_1q) else float(feature.d_fvc_1q),
            "fvc_dispersion": None if pd.isna(feature.fvc_dispersion) else float(feature.fvc_dispersion),
            "n_holders": int(feature.n_holders), "holders": holders,
            "new_since_prev_period": None if previous is None else str(feature.borrower_id) not in previous_ids,
            "accessions": sorted(source["accession"].dropna().unique()),
        })
    return {
        "tool": "review_list", "period": period, "prev_period": previous, "k": k,
        "ranked_by": "observed baseline fvc_min ascending (lowest holder mark first)",
        "model_scores_available": False,
        "n": len(rows),
        "n_new": None if previous is None else sum(row["new_since_prev_period"] for row in rows),
        "n_carried_over": None if previous is None else sum(not row["new_since_prev_period"] for row in rows),
        "rows": rows, "data_as_of": config.DATA_AS_OF,
    }


def sponsor_or_colender_exposure(lender: str) -> dict:
    """Compute a tape-derived co-lender example, not a Neo4j query."""
    ticker = lender.upper().strip()
    # Q1 follows HOLDS edges. A holder can have an equity row, so this relationship
    # example removes netting rows but does not apply the debt-only mark predicate.
    tape = data.tape().loc[~data.tape()["is_netting_row"]].copy()
    tape["period"] = _period(tape["period_end"])
    holdings = tape[["filer_ticker", "borrower_id", "borrower_name", "period", "accession"]].drop_duplicates()
    own = holdings.loc[holdings["filer_ticker"].eq(ticker), ["borrower_id", "period"]].drop_duplicates()
    shared = holdings.merge(own, on=["borrower_id", "period"]).query("filer_ticker != @ticker")
    rows = []
    for other, group in shared.groupby("filer_ticker"):
        rows.append({"co_lender": other, "n_borrowers": int(group["borrower_id"].nunique()),
                     "n_borrower_periods": int(len(group[["borrower_id", "period"]].drop_duplicates())),
                     "accessions": sorted(group["accession"].dropna().unique())[:8]})
    rows.sort(key=lambda row: (-row["n_borrowers"], row["co_lender"]))
    return {"tool": "sponsor_or_colender_exposure", "lender": ticker,
            "source": "tape-derived relationship example; no Neo4j connection",
            "sponsor_identity_available": False, "n_co_lenders": len(rows), "rows": rows}


def graph_examples(lender: str = "ARCC", seed_name: str = "Pluralsight") -> dict:
    """Build compact examples corresponding to the three saved Cypher questions."""
    tape = _debt_rows(data.tape()).copy()
    tape["period"] = _period(tape["period_end"])
    detail = get_borrower(seed_name)
    period = detail.get("latest_period")
    seed_id = detail.get("borrower_id")
    seed_holders = {holder["filer_ticker"] for holder in detail.get("holders", [])
                    if holder["period_end"] == period}
    same_period = tape.loc[tape["period"].eq(period) & tape["filer_ticker"].isin(seed_holders)
                           & tape["borrower_id"].astype(str).ne(str(seed_id))]
    two_hop = []
    for (borrower_id, borrower_name), group in same_period.groupby(["borrower_id", "borrower_name"]):
        shared = sorted(group["filer_ticker"].unique())
        if len(shared) >= 2:
            cost, fair_value = group["cost"].sum(), group["fair_value"].sum()
            two_hop.append({"borrower": borrower_name, "n_shared_lenders": len(shared),
                            "shared_lenders": ", ".join(shared),
                            "fvc": float(fair_value / cost) if cost else None,
                            "accessions": ", ".join(sorted(group["accession"].dropna().unique()))})
    two_hop = sorted(two_hop, key=lambda row: (-row["n_shared_lenders"], row["borrower"]))[:10]

    holder = (tape.groupby(["borrower_id", "borrower_name", "period", "filer_ticker"], as_index=False)
              .agg(cost=("cost", "sum"), fair_value=("fair_value", "sum"),
                   accession=("accession", "first")))
    holder = holder.loc[holder["cost"].ge(1_000_000)]
    holder["fvc"] = holder["fair_value"] / holder["cost"]
    newest = coverage()["last_quarter"]
    dispersion = (holder.loc[holder["period"].eq(newest)]
                  .groupby(["borrower_id", "borrower_name"], as_index=False)
                  .agg(n_holders=("filer_ticker", "nunique"), fvc_min=("fvc", "min"),
                       fvc_max=("fvc", "max")))
    dispersion["dispersion"] = dispersion["fvc_max"] - dispersion["fvc_min"]
    dispersion = dispersion.loc[dispersion["n_holders"].ge(2) & dispersion["dispersion"].ge(.05)]
    dispersion = dispersion.sort_values(["dispersion", "borrower_name"], ascending=[False, True]).head(10)
    return {"co_lending": sponsor_or_colender_exposure(lender)["rows"][:10],
            "two_hop": two_hop, "dispersion": _records(dispersion)}
