# Archived monitoring-plan summary

This plan was not automated or deployed. The prototype had no scheduler or serving path.

## Quarterly input checks

- Compare the `fvc_min` distribution and null rate with prior quarters.
- Check rows, non-accrual source coverage and filing reconciliation per filer.
- Track borrowers entering and leaving each filer's book.
- Re-measure entity resolution and multi-holder coverage on fresh labels.

## While labels are delayed

BDC filings arrive after quarter end, and the next-quarter non-accrual label arrives later still. Until labels
land, compare list turnover and inspect the filings behind surprising marks. These are proxies, not performance.

## When labels arrive

Recompute precision at the 15-name review capacity, recall at the required precision and AUC. Keep the same
borrower-within-quarter bootstrap unit so intervals stay comparable. Compare against the observed-mark baseline.

## Stop conditions

Stop and audit data when a parse money-field accuracy falls below 0.90 or fresh entity-resolution precision falls
below 0.95. Do not promote a learned ranker unless a pre-registered future comparison demonstrates an advantage
over the baseline.

## Human review

An analyst should inspect all 15 names and record which filings changed the decision. That review is closer to
the actual job than request latency or a dashboard-only metric.
