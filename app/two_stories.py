"""Two-story Parallax demo: price a loan from BDC comparables, then work a quarterly review list.

Offline and deterministic. It reads only the packaged Parquet snapshot through the showcase
readers in src/parallax. No LLM, no model score, no secret, no network call. SEC links are plain
links a reader may choose to open.

Run: streamlit run app/two_stories.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit runs this file directly, so put this package's src folder on the import path.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parallax import data, tools  # noqa: E402

DEFAULT_CAPACITY = 15  # the brief's review capacity: one analyst, 15 names a quarter
CAPACITY_CHOICES = [10, 15, 20, 25]
INSTRUMENTS = ["first_lien", "second_lien", "unitranche", "subordinated", "any"]
SPREAD_NOTE = (
    "Spread is the contractual margin over the reference rate as filed, in basis points. "
    "It is not an all-in yield: fees and original issue discount are absent."
)


# ---------- small pure helpers (tested directly) ----------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson 95% interval for k successes out of n. None when n is 0."""
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def matched_positions(industry: str, size: str, instrument: str) -> pd.DataFrame:
    """Every row behind tools.get_comparables, not just its 50-row preview.

    Same filter as the showcase tool: exact filed industry label (case and punctuation
    insensitive), latest period each filer reported, a parsed spread, no netting rows,
    optional instrument and principal range. A test checks count and median parity.
    """
    tape = data.tape().copy()
    tape["period"] = pd.to_datetime(tape["period_end"]).dt.strftime("%Y-%m-%d")
    want = tools._normal_label(industry)
    labels = {label for label in tape["industry"].dropna().unique() if tools._normal_label(label) == want}
    latest = tape.groupby("filer_ticker")["period"].transform("max")
    mask = tape["period"].eq(latest) & tape["spread_bps"].notna() & ~tape["is_netting_row"]
    mask &= tape["industry"].isin(labels)
    bucket = tools._normal_label(instrument).replace(" ", "_")
    if bucket not in {"", "any", "all", "none"}:
        mask &= tape["instrument_bucket"].eq(bucket)
    low, high = tools.parse_size_range(size)
    if low is not None:
        mask &= tape["principal"].ge(low)
    if high is not None:
        mask &= tape["principal"].le(high)
    rows = tape.loc[mask].sort_values(["spread_bps", "row_id"]).copy()
    # Mark = fair value / cost on the row. Only defined when cost is positive.
    rows["mark"] = (rows["fair_value"] / rows["cost"]).where(rows["cost"].fillna(0).gt(0))
    rows["filing"] = rows["accession"].map(tools.filing_urls())
    return rows


def lender_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One line per contributing BDC: how many positions, its median spread and mark, its period."""
    table = (rows.groupby("filer_ticker")
             .agg(positions=("row_id", "size"), median_spread_bps=("spread_bps", "median"),
                  median_mark=("mark", "median"), period_end=("period", "max"))
             .reset_index().rename(columns={"filer_ticker": "lender"}))
    table["share_of_positions"] = table["positions"] / table["positions"].sum()
    return table.sort_values(["positions", "lender"], ascending=[False, True]).reset_index(drop=True)


def hindsight(period: str, borrower_ids: list[str]) -> dict:
    """What the next filings showed for the listed names. Observed labels, no model.

    label = a lender with known status flagged a debt position non-accrual at t+1.
    Names without a next-quarter label (absent next quarter) are excluded from the rate.
    """
    panel = data.features()
    quarter = panel.loc[panel["period_end"].astype(str).eq(period) & panel["in_universe"]]
    known = quarter.loc[quarter["label_available"]]
    listed = known.loc[known["borrower_id"].astype(str).isin(borrower_ids)]
    hits = int(listed["label"].eq(True).sum())
    return {
        "n_listed": len(borrower_ids), "n_labelled": int(len(listed)), "hits": hits,
        "ci": wilson(hits, int(len(listed))),
        "base_hits": int(known["label"].eq(True).sum()), "base_n": int(len(known)),
    }


def bps(value: float | None) -> str:
    return "n/a" if value is None or pd.isna(value) else f"S+{value:,.0f}"


# ---------- cached wrappers so reruns stay fast ----------

@st.cache_data(show_spinner=False)
def cached_comparables(industry: str, size: str, instrument: str) -> tuple[dict, pd.DataFrame]:
    return tools.get_comparables(industry, size, instrument), matched_positions(industry, size, instrument)


@st.cache_data(show_spinner=False)
def cached_review(period: str, k: int) -> dict:
    return tools.review_list(period, k)


@st.cache_data(show_spinner=False)
def cached_coverage() -> dict:
    return tools.coverage()


# ---------- page ----------

def header() -> None:
    cov = cached_coverage()
    st.title("BDC loan tape: two jobs")
    st.caption(
        f"Public SEC data: {cov['n_filers']} BDCs ({', '.join(cov['bdcs'])}), {cov['n_quarters']} quarter ends "
        f"{cov['first_quarter']} to {cov['last_quarter']}, {cov['n_positions']:,} filed position rows, "
        f"{cov['n_borrowers']:,} resolved borrowers. Snapshot {cov['data_as_of']}. Not a live feed, not advice."
    )


def story_price() -> None:
    st.header("1. Price a loan from BDC comparables")
    st.markdown(
        "**What you're deciding:** whether a proposed spread for a borrower in a filed industry label sits "
        "inside, above or below what listed BDCs hold, and which lenders and rows that answer rests on."
    )
    labels = tools.industry_choices()
    one, two, three = st.columns(3)
    industry = one.selectbox("Filed industry label", labels,
                             index=labels.index("Software") if "Software" in labels else 0)
    instrument = two.selectbox("Instrument", INSTRUMENTS)
    size = three.text_input("Principal range", value="any", help='e.g. "$20M to $50M", "under $25M", "any"')
    result, rows = cached_comparables(industry, size, instrument)

    a, b, c = st.columns(3)
    a.metric("Comparable positions", f"{result['n']:,}")
    b.metric("Contributing lenders", f"{result['n_filers']}")
    c.metric("Median filed spread (bps)", bps(result["median_spread_bps"]))
    if not result["n"]:
        st.warning("No filed rows match. Try instrument `any` or a nearby label listed below.")
    else:
        st.markdown(
            f"**Spread distribution over all {result['n']:,} positions:** p25 {bps(result['p25_spread_bps'])}, "
            f"median {bps(result['median_spread_bps'])}, p75 {bps(result['p75_spread_bps'])} bps. "
            f"Median mark (fair value / cost): {rows['mark'].median():.3f} over {rows['mark'].notna().sum():,} "
            "rows with positive cost."
        )
        # Caveats come before the evidence so they're read first.
        if result["n_filers"] < 2:
            st.error(f"One lender's book ({', '.join(result['filer_mix'])}), not a market statistic.")
        if result["thin"]:
            st.warning(f"Small sample: {result['n']} positions, fewer than {tools.THIN_N}. Quartiles move with one row.")
        top_share = lender_table(rows)["share_of_positions"].max()
        if result["n_filers"] >= 2 and top_share >= 0.5:
            st.warning(f"One lender supplies {top_share:.0%} of these positions. Read the lender table.")

        st.subheader("Contributing lenders")
        st.dataframe(lender_table(rows), hide_index=True, width="stretch",
                     column_config={"share_of_positions": st.column_config.NumberColumn(format="percent"),
                                    "median_mark": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Each lender's latest filed quarter is used, so periods can differ by lender.")

        st.subheader("Source rows")
        shown = rows[["filer_ticker", "period", "borrower_name", "industry", "instrument_bucket",
                      "spread_bps", "principal", "cost", "fair_value", "mark", "accession", "filing"]]
        st.dataframe(shown, hide_index=True, width="stretch",
                     column_config={"filing": st.column_config.LinkColumn("SEC filing", display_text="open"),
                                    "mark": st.column_config.NumberColumn(format="%.3f")})
        st.caption(SPREAD_NOTE)

    neighbours = result["related_labels_not_used"]
    if neighbours:
        text = "; ".join(f"`{label}` n={v['n']}, median {bps(v['median_spread_bps'])}, {v['n_filers']} lender(s)"
                         for label, v in neighbours.items())
        st.info(f"**Nearby labels, not merged:** {text}. Each BDC names industries its own way; "
                "pooling them would invent a statistic no filing reports.")


def story_review() -> None:
    st.header("2. Quarterly review list")
    st.markdown(
        "**What you're deciding:** which borrowers get your limited review time this quarter. "
        "The list puts the lowest lender mark first, shows what changed and links each name to its filings."
    )
    periods = tools.review_periods()
    one, two = st.columns(2)
    period = one.selectbox("Quarter end", periods,
                           index=periods.index("2026-03-31") if "2026-03-31" in periods else len(periods) - 1)
    k = two.selectbox("Names you can review", CAPACITY_CHOICES, index=CAPACITY_CHOICES.index(DEFAULT_CAPACITY))
    result = cached_review(period, k)
    if result.get("out_of_coverage"):
        st.error(result["reason"])
        return

    a, b, c = st.columns(3)
    a.metric("Names on list", result["n"])
    b.metric("New since prior quarter", "n/a" if result["n_new"] is None else result["n_new"])
    c.metric("Ranked by", "lowest lender mark")
    st.warning(
        "Observed-mark rule, not a prediction. Model scores are withheld: the cached scores failed an identity "
        "check against the model they cite (decision D-018), and the model did not beat this rule."
    )

    table = pd.DataFrame([{
        "rank": row["rank"],
        "borrower": row["borrower_name"],
        "lowest_mark": row["fvc_min"],
        "change_vs_prior_q": row["d_fvc_1q"],
        "lenders": row["n_holders"],
        "lender_marks": " · ".join(f"{h['filer_ticker']} {h['fvc']:.3f}" for h in row["holders"] if h["fvc"] is not None),
        "status": "first quarter" if row["new_since_prev_period"] is None
        else ("NEW" if row["new_since_prev_period"] else "carried over"),
        "filing": tools.filing_urls().get(row["accessions"][0]) if row["accessions"] else None,
    } for row in result["rows"]])
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"filing": st.column_config.LinkColumn("SEC filing", display_text="open"),
                                "lowest_mark": st.column_config.NumberColumn(format="%.3f"),
                                "change_vs_prior_q": st.column_config.NumberColumn(format="%+.3f")})
    st.caption(
        f"Mark = sum(fair value) / sum(cost) per lender on debt rows ({tools.DEBT_ROW_LABEL}). Change = this "
        f"quarter's lowest mark minus {result['prev_period'] or 'n/a'}'s. NEW = not on the prior quarter's list "
        "at the same capacity, not newly distressed."
    )

    # Drill into one name: every lender's mark and every filing behind it.
    pick = st.selectbox("Open one name", [row["borrower_name"] for row in result["rows"]])
    row = next(r for r in result["rows"] if r["borrower_name"] == pick)
    for holder in row["holders"]:
        mark = "n/a" if holder["fvc"] is None else f"{holder['fvc']:.3f}"
        flag = ", non-accrual flagged" if holder["non_accrual"] else ""
        links = ", ".join(f"[{acc}]({tools.filing_urls().get(acc)})" for acc in holder["accessions"])
        st.markdown(f"- **{holder['filer_ticker']}** mark {mark} on {holder['n_debt_rows']} debt row(s){flag}. Filing: {links}")
    if row["n_holders"] < 2:
        st.caption("Single lender: no second opinion on this mark.")

    seen = hindsight(period, [r["borrower_id"] for r in result["rows"]])
    st.subheader("Hindsight check from the next filings")
    if seen["n_labelled"] == 0:
        st.info("No next-quarter filing in this snapshot, so the outcome is unobserved.")
    else:
        lo, hi = seen["ci"]
        st.markdown(
            f"**{seen['hits']} of {seen['n_labelled']}** listed names with a next-quarter record had a debt position "
            f"flagged non-accrual the next quarter (Wilson 95% CI {lo:.2f} to {hi:.2f}). "
            f"Base rate across the quarter: {seen['base_hits']} of {seen['base_n']:,}."
        )
    st.caption(
        "Small samples: one name moves the rate by several points. The rule has no fitted parameters, so every "
        "quarter is an out-of-sample check of the rule, not of a model. The panel ends at 2026-03-31, one quarter "
        "before the newest tape quarter, because a label needs the next filing. Some names are fund vehicles, "
        "not operating companies; the list does not remove them."
    )


def main() -> None:
    st.set_page_config(page_title="BDC loan tape: two jobs", layout="wide")
    header()
    price, review = st.tabs(["1. Price a loan", "2. Review list"])
    with price:
        story_price()
    with review:
        story_review()
    st.divider()
    st.caption("Not in this demo: allocator view, graph queries, archived ML and agent evidence. "
               "See app/streamlit_app.py for the full nine-tab walkthrough.")


# Streamlit (and AppTest) run this file as __main__; a plain import from the tests does not render.
if __name__ == "__main__":
    main()
