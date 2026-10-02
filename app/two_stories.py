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
    "Spread = the interest margin a lender charges above a benchmark rate, in basis points (100 bps = 1%). "
    "It leaves out fees and discounts, so it is not the full return."
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
    st.title("BDC Mark Tape: what other lenders charge and how they value their loans")
    st.caption(
        f"Public SEC data from {cov['n_filers']} BDCs (stock-listed funds that lend to private companies) ({', '.join(cov['bdcs'])}), {cov['n_quarters']} quarter ends "
        f"{cov['first_quarter']} to {cov['last_quarter']}, {cov['n_positions']:,} loans as reported, "
        f"{cov['n_borrowers']:,} borrowers matched across lenders. Snapshot {cov['data_as_of']}. Not live data, not advice."
    )


def story_price() -> None:
    st.header("1. Price a new loan using what other lenders charge")
    st.markdown(
        "**What you're deciding:** whether the interest margin you plan to charge a borrower is in line with, "
        "above or below what listed lenders charge similar companies, and which lenders and loans that rests on."
    )
    labels = tools.industry_choices()
    one, two, three = st.columns(3)
    industry = one.selectbox("Industry (as the lenders label it)", labels,
                             index=labels.index("Software") if "Software" in labels else 0)
    instrument = two.selectbox("Loan type (first lien = repaid first)", INSTRUMENTS)
    size = three.text_input("Loan size", value="any", help='e.g. "$20M to $50M", "under $25M", "any"')
    result, rows = cached_comparables(industry, size, instrument)

    a, b, c = st.columns(3)
    a.metric("Similar loans", f"{result['n']:,}")
    b.metric("Contributing lenders", f"{result['n_filers']}")
    c.metric("Typical margin (median, bps)", bps(result["median_spread_bps"]))
    if not result["n"]:
        st.warning("No reported loans match. Try loan type `any` or a nearby industry listed below.")
    else:
        st.markdown(
            f"**Margin across all {result['n']:,} loans:** lowest quarter below {bps(result['p25_spread_bps'])}, "
            f"median {bps(result['median_spread_bps'])}, top quarter above {bps(result['p75_spread_bps'])} bps. "
            f"Typical valuation (lender's value / amount lent; 1.000 = no loss): {rows['mark'].median():.3f} over "
            f"{rows['mark'].notna().sum():,} loans."
        )
        # Caveats come before the evidence so they're read first.
        if result["n_filers"] < 2:
            st.error(f"One lender's book ({', '.join(result['filer_mix'])}): one lender's view, not the market's.")
        if result["thin"]:
            st.warning(f"Small sample: {result['n']} loans, fewer than {tools.THIN_N}. One loan can move the numbers.")
        top_share = lender_table(rows)["share_of_positions"].max()
        if result["n_filers"] >= 2 and top_share >= 0.5:
            st.warning(f"One lender holds {top_share:.0%} of these loans. Check the lender table.")

        st.subheader("Contributing lenders")
        st.dataframe(lender_table(rows), hide_index=True, width="stretch",
                     column_config={"share_of_positions": st.column_config.NumberColumn(format="percent"),
                                    "median_mark": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Each lender's latest report is used, so dates can differ by lender.")

        st.subheader("Source rows")
        shown = rows[["filer_ticker", "period", "borrower_name", "industry", "instrument_bucket",
                      "spread_bps", "principal", "cost", "fair_value", "mark", "accession", "filing"]]
        st.dataframe(shown, hide_index=True, width="stretch",
                     column_config={"filing": st.column_config.LinkColumn("SEC report", display_text="open"),
                                    "mark": st.column_config.NumberColumn(format="%.3f")})
        st.caption(SPREAD_NOTE)

    neighbours = result["related_labels_not_used"]
    if neighbours:
        text = "; ".join(f"`{label}` n={v['n']}, median {bps(v['median_spread_bps'])}, {v['n_filers']} lender(s)"
                         for label, v in neighbours.items())
        st.info(f"**Similar industry names, kept separate:** {text}. Each lender names industries its own way; "
                "adding them together would create a number no report shows.")


def story_review() -> None:
    st.header("2. Each quarter: which borrowers to check first")
    st.markdown(
        "**What you're deciding:** which borrowers get your limited review time this quarter. "
        "The list puts first the borrowers some lender values lowest, shows what changed, and links each to its report."
    )
    periods = tools.review_periods()
    one, two = st.columns(2)
    period = one.selectbox("Quarter", periods,
                           index=periods.index("2026-03-31") if "2026-03-31" in periods else len(periods) - 1)
    k = two.selectbox("Borrowers you can review", CAPACITY_CHOICES, index=CAPACITY_CHOICES.index(DEFAULT_CAPACITY))
    result = cached_review(period, k)
    if result.get("out_of_coverage"):
        st.error(result["reason"])
        return

    a, b, c = st.columns(3)
    a.metric("Names on list", result["n"])
    b.metric("New since last quarter", "n/a" if result["n_new"] is None else result["n_new"])
    c.metric("Ranked by", "lowest value from any lender")
    st.warning(
        "A simple rule on reported values, not a prediction. Model scores are withheld: the saved scores failed a check "
        "that they came from the model they name (decision D-018), and the model did not beat this rule anyway."
    )

    table = pd.DataFrame([{
        "Rank": row["rank"],
        "Borrower": row["borrower_name"],
        "Lowest value": row["fvc_min"],
        "Change vs last quarter": row["d_fvc_1q"],
        "Lenders": row["n_holders"],
        "Each lender's value": " · ".join(f"{h['filer_ticker']} {h['fvc']:.3f}" for h in row["holders"] if h["fvc"] is not None),
        "Status": "first quarter" if row["new_since_prev_period"] is None
        else ("NEW" if row["new_since_prev_period"] else "also last quarter"),
        "Report": tools.filing_urls().get(row["accessions"][0]) if row["accessions"] else None,
    } for row in result["rows"]])
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"Report": st.column_config.LinkColumn("SEC report", display_text="open"),
                                "Lowest value": st.column_config.NumberColumn(format="%.3f"),
                                "Change vs last quarter": st.column_config.NumberColumn(format="%+.3f")})
    st.caption(
        f"Value = the lender's estimate of what its loans are worth / the amount lent ({tools.DEBT_ROW_LABEL}); "
        f"1.000 means no expected loss. Change = this quarter's lowest value minus {result['prev_period'] or 'n/a'}'s. "
        "NEW = not on last quarter's list of the same length, not necessarily newly in trouble."
    )

    # Drill into one name: every lender's mark and every filing behind it.
    pick = st.selectbox("Open one borrower", [row["borrower_name"] for row in result["rows"]])
    row = next(r for r in result["rows"] if r["borrower_name"] == pick)
    for holder in row["holders"]:
        mark = "n/a" if holder["fvc"] is None else f"{holder['fvc']:.3f}"
        flag = ", marked as not paying interest (non-accrual)" if holder["non_accrual"] else ""
        links = ", ".join(f"[{acc}]({tools.filing_urls().get(acc)})" for acc in holder["accessions"])
        st.markdown(f"- **{holder['filer_ticker']}** values it at {mark} across {holder['n_debt_rows']} loan(s){flag}. Report: {links}")
    if row["n_holders"] < 2:
        st.caption("Only one lender holds this borrower: no second opinion on its value.")

    seen = hindsight(period, [r["borrower_id"] for r in result["rows"]])
    st.subheader("Looking back: did the list catch trouble?")
    if seen["n_labelled"] == 0:
        st.info("No next-quarter report in this snapshot, so the outcome is unknown.")
    else:
        lo, hi = seen["ci"]
        st.markdown(
            f"**{seen['hits']} of {seen['n_labelled']}** listed borrowers stopped paying interest (non-accrual) the next "
            f"quarter (95% range {lo:.2f} to {hi:.2f}). "
            f"Across all borrowers that quarter: {seen['base_hits']} of {seen['base_n']:,}."
        )
    st.caption(
        "Small samples: one borrower moves the rate by several points. The rule has nothing tuned to past data, so "
        "every quarter is a fair test. The check ends at 2026-03-31, a quarter before the newest data, because the "
        "outcome needs the next report. Some borrowers are investment funds, not companies; the list keeps them."
    )


def main() -> None:
    st.set_page_config(page_title="BDC Mark Tape", layout="wide")
    header()
    price, review = st.tabs(["1. Price a new loan", "2. Quarterly review list"])
    with price:
        story_price()
    with review:
        story_review()
    st.divider()
    st.caption("Not in this demo: the investor view, network queries, the archived machine-learning and AI-agent results. "
               "See app/streamlit_app.py for the full nine-tab walkthrough.")


# Streamlit (and AppTest) run this file as __main__; a plain import from the tests does not render.
if __name__ == "__main__":
    main()
