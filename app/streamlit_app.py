"""Nine-tab, offline public walkthrough of the Parallax prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit Community Cloud runs this file directly. Add this repository's src
# directory so the small package works without installing a root project.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parallax import data, tools  # noqa: E402

K = 15
FEATURE_LABELS = {
    "fvc_min": "lowest holder mark",
    "d_fvc_1q": "quarter-over-quarter mark change",
    "fvc_dispersion": "holder-mark dispersion",
    "n_holders": "BDC holders",
}
SPREAD_NOTE = (
    "Spread is the contractual margin over the reference rate as filed. It is not an all-in yield: "
    "fees and original issue discount are absent."
)


@st.cache_data(show_spinner=False)
def packaged_coverage() -> dict:
    return tools.coverage()


@st.cache_data(show_spinner=False)
def comparables(industry: str, size_range: str, instrument: str) -> dict:
    return tools.get_comparables(industry, size_range, instrument)


@st.cache_data(show_spinner=False)
def review(period: str, k: int) -> dict:
    return tools.review_list(period, k)


@st.cache_data(show_spinner=False)
def graph_examples() -> dict:
    return tools.graph_examples()


@st.cache_data(show_spinner=False)
def archived(path: str) -> dict:
    return data.load_json(path)


def coverage_banner() -> str:
    cov = packaged_coverage()
    return (
        f"**Snapshot coverage:** {cov['n_filers']} BDCs ({', '.join(cov['bdcs'])}), "
        f"{cov['n_quarters']} quarter ends from {cov['first_quarter']} to {cov['last_quarter']}, "
        f"{cov['n_positions']:,} filed position rows and {cov['n_borrowers']:,} resolved borrowers. "
        f"Snapshot assembled {cov['data_as_of']}. Public SEC-derived data, not a live market feed."
    )


def filing_link(accession: str) -> str:
    url = tools.filing_urls().get(accession)
    return f"[Open {accession}]({url})" if url else f"`{accession}`"


def accession_list(accessions: list[str], cap: int = 8) -> str:
    links = [filing_link(accession) for accession in accessions[:cap]]
    suffix = f" and {len(accessions) - cap} more" if len(accessions) > cap else ""
    return " · ".join(links) + suffix


def bps(value: float | None) -> str:
    return "n/a" if value is None else f"S+{value:,.0f} bps"


def named_features(row: dict) -> list[str]:
    result = []
    for key in ("fvc_min", "d_fvc_1q", "fvc_dispersion", "n_holders"):
        value = row.get(key)
        if value is None:
            continue
        shown = str(int(value)) if key == "n_holders" else f"{value:.4f}"
        result.append(f"**{FEATURE_LABELS[key]}:** {shown}")
        if len(result) == 3:
            break
    return result


def tab_start_here() -> None:
    st.header("Start here")
    st.info(coverage_banner())
    st.markdown(
        """
### The private-credit problem
Listed business development companies, or BDCs, publish schedules of investments. Those tables expose
private-loan marks and spreads, but each filer uses different labels and borrower spellings. Parallax
turns those filings into one inspectable tape. It helps decide **which evidence to open next**, not which
loan to buy.

### Who this is for
- **Deal underwriter:** compare filed spreads before defending a proposed spread.
- **Monitoring analyst:** fit a quarterly review list to a 15-name review capacity.
- **Allocator:** compare how separate lenders mark a resolved borrower.

### One data spine
`cached SEC filing -> parsed positions -> resolved borrowers -> features -> deterministic views`

The original prototype also explored Neo4j, XGBoost, MLflow and a cited Gemini agent. This public package
does not run those services. It presents a fixed data snapshot, archived evidence and canned agent examples.
"""
    )
    st.subheader("Five-minute walkthrough")
    st.markdown(
        """
1. **Price a deal:** choose Software, any size and first lien. Check position count and filer count before the quartiles.
2. **Review list:** open 2026-03-31. Work from the lowest observed `fvc_min` and inspect holder marks and accessions.
3. **Data + entity resolution:** compare pair-level precision with the wider transitive-component interval.
4. **Knowledge graph:** read the three saved queries beside tape-derived examples. No database is contacted.
5. **ML, MLOps and agent evidence:** read every result as archived experiment evidence, then finish with what failed.
"""
    )
    st.warning(
        "This is a static-data educational showcase. It is not investment advice, a model release, "
        "a live agent or evidence that release gates passed."
    )


def tab_price_a_deal() -> None:
    st.header("Price a deal")
    st.info(coverage_banner())
    st.caption("Exact filing labels are kept separate. Nearby labels are disclosed, never silently merged.")
    labels = tools.industry_choices()
    default = labels.index("Software") if "Software" in labels else 0
    one, two, three = st.columns(3)
    industry = one.selectbox("Industry", labels, index=default)
    size = two.text_input("Size", value="any", help='Examples: "$20M to $50M", "under $25M", or "any".')
    instrument = three.selectbox(
        "Instrument", ["first_lien", "second_lien", "unitranche", "subordinated", "equity", "any"]
    )
    result = comparables(industry, size, instrument)
    c1, c2, c3 = st.columns(3)
    c1.metric("Comparable positions", f"{result['n']:,}")
    c2.metric("Distinct filers", f"{result['n_filers']:,}")
    c3.metric("Data as of", result["data_as_of"])
    if not result["n"]:
        st.warning("No rows match these filters.")
        return

    mix = ", ".join(f"{ticker} {count:,}" for ticker, count in result["filer_mix"].items())
    if result["n_filers"] < 2:
        st.error(f"This is one lender's book ({mix}), not a market statistic.")
        frame = "one lender's book"
    else:
        st.success(f"Observed across {result['n_filers']} filers. Filer mix: {mix}.")
        frame = "matched positions"
    q1, q2, q3 = st.columns(3)
    q1.metric(f"p25 spread ({frame})", bps(result["p25_spread_bps"]))
    q2.metric(f"Median spread ({frame})", bps(result["median_spread_bps"]))
    q3.metric(f"p75 spread ({frame})", bps(result["p75_spread_bps"]))
    st.markdown(
        f"**Distribution over all {result['n']:,} matches:** p25 {bps(result['p25_spread_bps'])}, "
        f"median {bps(result['median_spread_bps'])}, p75 {bps(result['p75_spread_bps'])}."
    )
    if result["thin"]:
        st.warning(f"Thin sample: {result['n']} positions, fewer than {tools.THIN_N}.")

    matched = ", ".join(f"`{label}` ({count:,})" for label, count in result["industries_matched"].items())
    st.markdown(f"**Labels matched:** {matched}")
    neighbours = result["related_labels_not_used"]
    if neighbours:
        text = "; ".join(
            f"`{label}` (n={values['n']:,}, median {bps(values['median_spread_bps'])}, "
            f"{values['n_filers']} filer(s))" for label, values in neighbours.items()
        )
        st.warning(f"**Nearby labels not used:** {text}.")

    rows = pd.DataFrame(result["rows"])
    st.subheader("Underlying filed rows")
    st.caption(
        f"Showing {'the first ' if result['rows_truncated'] else ''}{len(rows):,} row(s). "
        "The metrics above use every match."
    )
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(SPREAD_NOTE)
    st.markdown(f"Accessions behind this result: {accession_list(result['accessions'])}")


def tab_review_list() -> None:
    st.header("Review list")
    st.info(coverage_banner())
    st.caption(
        "Observed-mark baseline only. There is no current model score, probability, run ID or serving-provenance claim."
    )
    periods = tools.review_periods()
    default = periods.index("2026-03-31") if "2026-03-31" in periods else len(periods) - 1
    period = st.selectbox("Quarter end", periods, index=default)
    result = review(period, K)
    if result.get("out_of_coverage"):
        st.error(result["reason"])
        return
    if result["prev_period"]:
        st.subheader(
            f"{result['n_new']} of these {result['n']} are new since {result['prev_period']}"
        )
        st.caption(f"{result['n_carried_over']} carried over at the same review capacity of {K}.")
    st.warning(
        "Ranked by the lowest observed holder mark. This is a triage baseline, not a complete screen: "
        "a borrower marked near par can still move to non-accrual next quarter."
    )
    for row in result["rows"]:
        marker = (
            f"**NEW** since {result['prev_period']}" if row["new_since_prev_period"]
            else f"carried over from {result['prev_period']}" if row["new_since_prev_period"] is False
            else "first panel quarter"
        )
        with st.container(border=True):
            st.markdown(f"**{row['rank']}. {row['borrower_name']}** - {marker}")
            st.markdown("Features, not causes: " + "; ".join(named_features(row)))
            holder_marks = [holder for holder in row["holders"] if holder["fvc"] is not None]
            if len(holder_marks) >= 2:
                marks = " · ".join(
                    f"{holder['filer_ticker']} {holder['fvc']:.4f}" for holder in holder_marks
                )
                st.markdown(f"**Holder marks:** {marks}")
                st.caption(
                    f"Debt-only dispersion {row['fvc_dispersion']:.4f}. Filter: {tools.DEBT_ROW_LABEL}."
                )
            elif holder_marks:
                st.markdown(
                    f"**Single holder mark:** {holder_marks[0]['filer_ticker']} {holder_marks[0]['fvc']:.4f}. "
                    "Dispersion needs at least two holders."
                )
            st.markdown("Filings: " + accession_list(row["accessions"]))
    st.caption("Sorted by observed `fvc_min` ascending. Cached prediction files are not packaged or loaded.")


def tab_data_er() -> None:
    st.header("Data + entity resolution")
    st.info(coverage_banner())
    st.markdown(
        "**Pipeline:** SEC filing index -> Schedule of Investments table detection -> normalized position rows -> "
        "borrower-name candidates -> fuzzy scores -> connected components -> feature panel."
    )
    parse = archived("eval/parse_audit_results.json")
    st.subheader("Parse audit, archived evidence")
    parse_rows = pd.DataFrame(parse["accuracy"])[
        ["field", "correct", "n", "accuracy", "wilson95_lo", "wilson95_hi"]
    ]
    st.dataframe(parse_rows, width="stretch", hide_index=True)
    st.caption(
        "Wilson 95% intervals over a frozen 48-row in-population sample. It was drawn and checked by the "
        "parser author, so it is useful audit evidence, not an independent held-out-filer test."
    )

    er = archived("eval/er_results.json")
    st.subheader("Entity resolution, archived evidence")
    a, b, c = st.columns(3)
    a.metric("Pair precision at threshold 93", f"{er['precision']:.4f}")
    b.metric("Pair recall", f"{er['recall']:.4f}")
    c.metric("Transitive component precision", f"{er['precision_transitive']:.4f}")
    st.markdown(
        f"- Pair precision: {er['tp']}/{er['n_pred_pos']}, Wilson 95% CI "
        f"[{er['ci95'][0]:.4f}, {er['ci95'][1]:.4f}].\n"
        f"- Recall: {er['tp']}/{er['n_label_1_in_curve']}, reported on the 150 sampled in-curve pairs.\n"
        f"- Transitive check: {er['transitive']['correct']}/{er['n_transitive']}, Wilson 95% CI "
        f"[{er['transitive']['ci95'][0]:.4f}, {er['transitive']['ci95'][1]:.4f}]."
    )
    st.warning(
        "Threshold selection was in-sample. A post-label vehicle guard raised precision to 1.0000 on the same "
        "labels, but it was designed from the false positive and is not independent validation. The wider "
        "transitive interval is the honest uncertainty for connected components."
    )
    st.caption(
        f"The guarded tape contains {er['n_borrowers']:,} borrowers; {er['n_borrowers_multi_holder']:,} have "
        f"multiple holders at {er['n_borrowers_multi_holder_period']}."
    )


def _cypher_sections() -> list[tuple[str, str]]:
    text = data.text("graph/queries.cypher")
    markers = ["// === Q1", "// === Q2", "// === Q3"]
    starts = [text.index(marker) for marker in markers]
    blocks = [text[starts[i]: starts[i + 1] if i + 1 < len(starts) else len(text)] for i in range(3)]
    return [("Co-lending network", blocks[0]), ("Two-hop shared-lender exposure", blocks[1]),
            ("Mark dispersion", blocks[2])]


def tab_graph() -> None:
    st.header("Knowledge graph")
    st.markdown(
        """
**Schema**

`(:RawName)-[:RESOLVES_TO]->(:Borrower)`  
`(:Lender)-[:HOLDS {period, principal, cost, fair_value, spread_bps, accession}]->(:Borrower)`  
`(:RawName)-[:SIMILAR_TO {score}]-(:RawName)`

**Why graph instead of only SQL:** entity evidence and exposure are path-shaped. A raw spelling can connect
to a borrower through similarity and resolution edges; shared-lender exposure is a traversal. **Where SQL is
fine:** spread percentiles, holder aggregation and min/max dispersion are ordinary filters and group-bys.
"""
    )
    st.info("Static query walkthrough. The tables below are tape-derived relationship examples; no Neo4j connection runs.")
    examples = graph_examples()
    outputs = [examples["co_lending"], examples["two_hop"], examples["dispersion"]]
    notes = [
        "Example parameter: lender = ARCC. Counts span matching borrower-period relationships.",
        "Example seed: Pluralsight at its latest period; at least two shared lenders.",
        "Example period: latest tape quarter; at least two holder marks and 0.05 dispersion.",
    ]
    for (title, query), output, note in zip(_cypher_sections(), outputs, notes):
        st.subheader(title)
        st.caption(note)
        with st.expander("Saved Cypher"):
            st.code(query, language="cypher")
        st.dataframe(pd.DataFrame(output), width="stretch", hide_index=True)
    st.warning("A path shows a reporting relationship, not contagion, causality or proof that an entity merge is correct.")


def tab_ml_lab() -> None:
    st.header("ML lab")
    results = archived("eval/model_results.json")
    st.info("Archived experiment evidence only. It is not a current serving score or release result.")
    st.markdown(
        """
**Target:** borrower moves to debt non-accrual at quarter *t+1*, using observations at *t*.

**Feature families:** lowest, mean and weighted fair-value-to-cost marks; one- and two-quarter mark changes;
maximum spread; PIK flag; instrument buckets; position size; number of holders; cross-holder mark dispersion;
and industry target encoding.

**Split:** train on earlier quarters, test on 2025-12-31 and 2026-03-31, with deterministic borrower groups so
a borrower cannot appear on both sides of a fold. This protects time and entity boundaries. It does not repair
the in-fold industry target-encoding leakage described below.
"""
    )
    baseline, model, diff = results["rows"]["baseline"], results["rows"]["model"], results["diff"]
    table = [
        {
            "measure": "Recall at precision >= 0.30",
            "baseline": f"not reached; conditional CI [{baseline['recall_at_p30_ci_lo']:.4f}, {baseline['recall_at_p30_ci_hi']:.4f}] on {baseline['recall_at_p30_ci_n']} reached resamples",
            "XGBoost": f"{model['recall_at_p30']:.4f}; CI [{model['recall_at_p30_ci_lo']:.4f}, {model['recall_at_p30_ci_hi']:.4f}] on {model['recall_at_p30_ci_n']} reached resamples",
            "model minus baseline": f"{diff['recall_at_p30']['value']:.4f}; CI [{diff['recall_at_p30']['ci_lo']:.4f}, {diff['recall_at_p30']['ci_hi']:.4f}], n={diff['recall_at_p30']['n']} paired resamples",
        },
        {
            "measure": "Precision@15, 2025-12-31",
            "baseline": f"{baseline['precision_at_15__2025-12-31']:.3f}; CI [{baseline['precision_at_15__2025-12-31_ci_lo']:.3f}, {baseline['precision_at_15__2025-12-31_ci_hi']:.3f}]",
            "XGBoost": f"{model['precision_at_15__2025-12-31']:.3f}; CI [{model['precision_at_15__2025-12-31_ci_lo']:.3f}, {model['precision_at_15__2025-12-31_ci_hi']:.3f}]",
            "model minus baseline": f"{diff['precision_at_15__2025-12-31']['value']:+.4f}; CI [{diff['precision_at_15__2025-12-31']['ci_lo']:.4f}, {diff['precision_at_15__2025-12-31']['ci_hi']:.4f}]",
        },
        {
            "measure": "Precision@15, 2026-03-31",
            "baseline": f"{baseline['precision_at_15__2026-03-31']:.3f}; CI [{baseline['precision_at_15__2026-03-31_ci_lo']:.3f}, {baseline['precision_at_15__2026-03-31_ci_hi']:.3f}]",
            "XGBoost": f"{model['precision_at_15__2026-03-31']:.3f}; CI [{model['precision_at_15__2026-03-31_ci_lo']:.3f}, {model['precision_at_15__2026-03-31_ci_hi']:.3f}]",
            "model minus baseline": f"{diff['precision_at_15__2026-03-31']['value']:+.4f}; CI [{diff['precision_at_15__2026-03-31']['ci_lo']:.4f}, {diff['precision_at_15__2026-03-31']['ci_hi']:.4f}]",
        },
        {
            "measure": "AUC, pooled n=2,838",
            "baseline": f"{baseline['auc']:.4f}; CI [{baseline['auc_ci_lo']:.4f}, {baseline['auc_ci_hi']:.4f}]",
            "XGBoost": f"{model['auc']:.4f}; CI [{model['auc_ci_lo']:.4f}, {model['auc_ci_hi']:.4f}]",
            "model minus baseline": f"{diff['auc']['value']:+.4f}; CI [{diff['auc']['ci_lo']:.4f}, {diff['auc']['ci_hi']:.4f}]",
        },
    ]
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
    st.caption(
        f"Archived JSON: {results['pooled_n_test']:,} held-out borrower-periods, {results['pooled_positives']} positives. "
        f"Intervals use {results['bootstrap_unit']}. Difference rows are bootstrap summaries, not subtraction of rounded cells."
    )
    st.warning(
        "No difference interval excludes zero. XGBoost did not demonstrate an advantage over the observed-mark "
        "baseline. The baseline operating point was not reached; its interval is conditional on resamples where it was reached."
    )
    st.error(
        "Negative-result lesson: industry target encoding included each training row's own label. It looked strong in-fold "
        "and weakened out of fold. Do not tune this away after seeing the held-out quarters; keep the baseline and design "
        "a clean future experiment."
    )


def tab_mlops() -> None:
    st.header("MLOps")
    st.info("Architecture and archived records, not an MLflow endpoint and not a production-readiness claim.")
    st.markdown(
        """
### Intended experiment lifecycle
1. Freeze the tape and feature hashes, code state, filer coverage and data-quality measures.
2. Train baseline first, then one fixed model design on the same grouped time split.
3. Log parameters, metrics, confidence intervals and artifacts to MLflow.
4. Register versions and aliases only after comparing against the baseline.
5. Attach a model card and a quarterly monitoring plan. Keep the human release gate separate.

### What the records taught
Data quality belongs beside model quality. A parse or entity-resolution change can move a metric without any
learning improvement. A registry alias and a hash are useful only if the displayed predictions are tied to the
actual producing run and the historical ancestors remain available.
"""
    )
    st.error(
        "Known failures remain open: cached score identity was stale, and historical lineage depended on mutable or "
        "unavailable ancestors. Removing scores from this showcase avoids a false claim; it does not fix lineage or pass a gate."
    )
    st.subheader("Monitoring concepts from the archived plan")
    st.markdown(
        """
- Watch `fvc_min` drift and null rate, but separate a credit-cycle move from a parser regression.
- Check rows, non-accrual-source coverage and reconciliation per filer, not only in aggregate.
- Labels arrive roughly one quarter later, so top-list movement is only a proxy while waiting.
- Recompute precision@15, recall at required precision and AUC with the same resampling unit when labels arrive.
- Keep manual 15-name analyst review as the decision-level measure. No scheduler or serving path was built.
"""
    )
    with st.expander("Public model-card summary"):
        st.markdown(data.text("docs/model_card.md"))
    with st.expander("Public monitoring-plan summary"):
        st.markdown(data.text("docs/monitoring_plan.md"))


def tab_agent() -> None:
    st.header("Cited agent")
    evidence = archived("eval/agent_examples.json")
    st.info("Canned archived examples only. Selecting a question does not call a model, API, tool service or network.")
    examples = evidence["examples"]
    labels = {f"{item['question_id']} · {item['question']}": item for item in examples}
    selected = labels[st.selectbox("Choose a saved question", list(labels))]
    st.markdown("### Saved answer")
    st.write(selected["saved_answer"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Tool concept", selected["tool"])
    c2.metric("Archived verifier outcome", "PASS" if selected["verifier_pass"] else "FAIL / REFUSAL")
    c3.metric("Verifier strikes", selected["verifier_strikes"])
    st.markdown("**Recorded citations:** " + (accession_list(selected["citations"]) or "none"))
    st.markdown(f"**Verifier note:** {selected['verifier_reason']}")
    st.warning(f"**Learning:** {selected['learning']}")

    summary = evidence["source_summary"]
    st.subheader("Archived evaluation context")
    st.markdown(
        f"The recorded first repeat passed {summary['accuracy_n_pass']} of {summary['n']} questions "
        f"(accuracy {summary['accuracy']:.4f}). Citation groundedness was {summary['citation_groundedness']:.4f} "
        f"as a macro mean over {summary['n_questions_with_claims']} answers with numeric claims. Separately, "
        f"{summary['claims_verified']} of {summary['claims_total']} individual claims verified. "
        f"Seven of 20 questions changed outcome across three repeats."
    )
    st.caption(
        "The macro answer-level mean and claim-level proportion are different estimators. A claim-level Wilson interval "
        "must not be presented as uncertainty for the macro mean. Archived outcomes do not prove current reproducibility."
    )


def tab_lessons() -> None:
    st.header("Lessons + limitations")
    st.markdown(
        """
### Reusable lessons
1. **Start with the practitioner's rule.** Complexity has to beat the lowest filed mark, not just look more technical.
2. **Split on time and group on entity.** Protect both chronological and borrower boundaries.
3. **Put sample size and uncertainty next to metrics.** One name can move a 15-name list sharply.
4. **Measure entity resolution directly.** Pair precision, recall and transitive components answer different questions.
5. **Bind citations to claims.** A real accession beside the wrong aggregate is still wrong.
6. **Treat observability as correctness.** Data hashes, prediction identity and immutable ancestors must connect.
7. **Keep negative results.** The model, agent stability and lineage failures are useful evidence against premature release.

### What failed or stayed uncertain
- XGBoost did not demonstrate superiority over the observed-mark baseline.
- Industry target encoding leaked each training row's own label.
- Entity-resolution threshold selection was in-sample; the transitive check is small and uncertain.
- Cited-agent repeats disagreed on 7 of 20 questions; the verifier also exposed under- and over-citation.
- Cached score identity and historical lineage checks failed and remain backlog.
- Eight BDCs and eight quarter ends are a narrow public sample. Filed marks are not live prices.
- Industry labels contain parser artifacts; the app excludes date-like labels from the pricing control rather than rewriting the tape.

### Next steps
Draw a fresh held-out ER label set, audit a new filer, remove target-encoding leakage in a pre-registered future test,
version predictions with their producing run, make lineage inputs immutable, and repeat agent evaluation before any live use.
"""
    )
    st.warning(
        "Local/showcase disclaimer: this fixed public snapshot is for education and portfolio review. It is not investment "
        "advice, a complete market, a live monitoring system, a deployed model or a live agent."
    )


def main() -> None:
    st.set_page_config(page_title="Parallax Showcase", page_icon=None, layout="wide")
    st.title("Parallax")
    st.caption("Private-credit evidence from public BDC filings, with the failures left visible.")
    tabs = st.tabs([
        "Start here", "Price a deal", "Review list", "Data + entity resolution", "Knowledge graph",
        "ML lab", "MLOps", "Cited agent", "Lessons + limitations",
    ])
    renderers = [tab_start_here, tab_price_a_deal, tab_review_list, tab_data_er, tab_graph,
                 tab_ml_lab, tab_mlops, tab_agent, tab_lessons]
    for tab, renderer in zip(tabs, renderers):
        with tab:
            renderer()


if __name__ == "__main__":
    main()
