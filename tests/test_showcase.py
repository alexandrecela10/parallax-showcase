"""Focused offline checks for the standalone public package."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
# The parent repository also has a package named parallax. Put this standalone
# subtree first so the test proves the public package does not reach outside it.
sys.path.insert(0, str(ROOT / "src"))
for module_name in [name for name in sys.modules if name == "parallax" or name.startswith("parallax.")]:
    del sys.modules[module_name]

from parallax import config, data, tools  # noqa: E402

APP = ROOT / "app" / "streamlit_app.py"


def page_text(node) -> str:
    parts = []
    for name in ("title", "header", "subheader", "markdown", "caption", "info", "warning", "error", "success"):
        parts.extend(str(element.value) for element in getattr(node, name, []))
    parts.extend(f"{metric.label} {metric.value}" for metric in getattr(node, "metric", []))
    return "\n".join(parts)


def test_package_is_self_contained_and_has_no_predictions():
    assert config.PACKAGE_ROOT == ROOT
    assert data.tape().shape == (42830, 37)
    assert data.features().shape == (10445, 29)
    assert "local_path" not in data.filings().columns
    assert data.predictions().empty
    assert not (ROOT / "data" / "test_predictions.csv").exists()


def test_comparables_keep_exact_labels_and_filer_context():
    software = tools.get_comparables("Software", "any", "first_lien")
    assert software["n"] == 5
    assert software["n_filers"] == 2
    assert software["median_spread_bps"] == 675.0
    assert "Software and Services" in software["related_labels_not_used"]
    assert all(row["accession"] for row in software["rows"])


def test_review_list_is_observed_mark_sorted_with_holder_links():
    result = tools.review_list("2026-03-31", 15)
    assert result["n"] == 15
    assert result["n_new"] == 5
    assert result["model_scores_available"] is False
    marks = [row["fvc_min"] for row in result["rows"]]
    assert marks == sorted(marks)
    for row in result["rows"]:
        assert row["accessions"]
        holder_marks = [holder["fvc"] for holder in row["holders"] if holder["fvc"] is not None]
        assert round(row["fvc_min"], 4) == round(min(holder_marks), 4)


def test_archived_metric_semantics_are_preserved():
    model = data.load_json("eval/model_results.json")
    assert model["rows"]["baseline"]["recall_at_p30"] is None
    assert model["rows"]["baseline"]["recall_at_p30_ci_n"] == 474
    assert model["diff"]["auc"]["excludes_zero"] is False
    er = data.load_json("eval/er_results.json")
    assert er["selection"] == "in-sample"
    assert er["precision_transitive"] == 0.85


def test_app_has_nine_ordered_tabs_and_runs_offline():
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert [tab.label for tab in app.tabs] == [
        "Start here", "Price a deal", "Review list", "Data + entity resolution", "Knowledge graph",
        "ML lab", "MLOps", "Cited agent", "Lessons + limitations",
    ]
    review_text = page_text(app.tabs[2])
    assert "Sorted by observed `fvc_min` ascending" in review_text
    assert not re.search(r"\b[a-f0-9]{32}\b", review_text)
    assert "cached prediction files are not packaged" in review_text.lower()
    agent_text = page_text(app.tabs[7])
    assert "Canned archived examples only" in agent_text
    assert len(app.selectbox) >= 4
