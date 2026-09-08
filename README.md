# Parallax Showcase

Parallax turns public BDC schedules of investments into an inspectable private-credit loan tape. This standalone
repository is a static-data educational showcase. It is not investment advice, a complete market, a deployed
model or a live agent.

## What is included

- A nine-tab Streamlit walkthrough for the problem, pricing workflow, observed-mark review list, data quality,
  entity resolution, graph design, archived ML evidence, MLOps lessons and canned cited-agent examples.
- A fixed public SEC-derived snapshot: 42,830 position rows, a 10,445-row feature panel and a 64-filing index.
- Archived parse-audit, entity-resolution and model-result JSON files.
- Four curated agent examples extracted from an archived evaluation. They contain no model call or live prompt.
- Three saved Cypher questions plus tape-derived static examples. Neo4j is not contacted.

The package intentionally excludes raw filings, `.env` files, credentials, MLflow stores, usage logs, internal
team records, prediction caches and service endpoints. In particular, `test_predictions.csv` is not included.
The review list is sorted by observed `fvc_min` only and never shows a current model score or run ID.

## Run locally

Use Python 3.11.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app/streamlit_app.py
```

The entrypoint adds this repository's `src` directory to its import path, so no package installation is required.
After the dependencies are installed, the app runs without Docker, `.env`, network access, Gemini, Neo4j or
MLflow. SEC accession links are ordinary links and need network only if a reader chooses to open one.

## Streamlit Community Cloud

1. Publish this directory as the root of a repository.
2. Select Python 3.11 in deployment settings.
3. Set the entrypoint to `app/streamlit_app.py`.
4. Do not add secrets. The app does not read any.

## Evidence boundaries

- **Parse audit:** a frozen, in-population 48-row audit by the parser author. Wilson intervals are displayed.
- **Entity resolution:** threshold 93 was chosen in-sample. Pair precision and recall do not replace the smaller,
  wider transitive-component check. A post-label vehicle guard is disclosed as post hoc.
- **ML:** the archived comparison has 2,838 held-out borrower-periods and 41 positives. No difference interval
  excludes zero. XGBoost did not earn deployment over the observed-mark baseline.
- **Agent:** canned answers preserve verifier outcomes, including a rejected under-cited aggregate. Archived
  repeat disagreement is a limitation, not a live capability claim.
- **MLOps:** stale cached-score identity and mutable historical lineage checks failed. Omitting scores from this
  public app avoids false provenance; it does not fix those failures or pass release gates.

## Repository map

```text
app/streamlit_app.py       Streamlit entrypoint
src/parallax/              offline package-local readers and deterministic tools
data/                      allowlisted public Parquet snapshot
eval/                      archived result JSONs and curated canned examples
graph/queries.cypher       three saved graph questions
docs/                      concise public model-card and monitoring summaries
tests/                     focused offline package and AppTest checks
build_package.py           maintainer-only allowlist builder used to assemble data/evidence
requirements.txt           minimal pinned runtime dependencies
```

## Rebuilding the public snapshot

The deployed repository already contains its assets. Inside the original parent Parallax repository only,
`python showcase/build_package.py` recopies the strict allowlist. The builder drops raw-file paths from the filing
index, extracts only four agent examples, rejects risky paths and scans text outputs. It never deletes or overwrites
source data. A standalone clone does not contain the parent sources, so it does not need to run this command.
