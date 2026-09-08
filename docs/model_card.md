# Archived model-card summary

**Status:** not recommended for deployment. This is historical experiment evidence, not a current model release.

## Intended use

Rank borrower-quarter observations for analyst review of filing evidence. A rank is not a calibrated probability,
a credit decision or a substitute for reading the filing.

## Data and split

The archived experiment used `data/features.parquet`, one row per borrower and feature quarter. Earlier quarters
formed training data. The two held-out quarters were 2025-12-31 and 2026-03-31. Borrowers were assigned to
deterministic groups so one borrower could not appear on both sides of a fold.

## Features

Marks and changes in marks, spread, PIK status, instrument buckets, position size, number of holders,
cross-holder dispersion, observed history and industry target encoding.

## Decision

Keep the observed `fvc_min` baseline. The XGBoost experiment did not demonstrate an advantage: no archived
model-minus-baseline confidence interval excludes zero. The public app does not package or display model scores.

## Material limitations

- Only 41 positives occur in 2,838 pooled held-out borrower-periods.
- Industry target encoding included each training row's own label and showed strong in-fold leakage.
- Scores were not calibrated probabilities.
- Cross-holder dispersion is sparse and can mix lender disagreement with capital-structure differences.
- Cached score identity and historical lineage checks later failed. Hiding scores does not repair those failures.
- No ninth-filer performance or serving latency was measured.
