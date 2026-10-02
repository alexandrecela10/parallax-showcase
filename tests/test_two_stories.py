"""Offline checks for the two-story demo (app/two_stories.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
# Same isolation as test_showcase.py: use this subtree's parallax package, not the parent repo's.
sys.path.insert(0, str(ROOT / "src"))
for module_name in [name for name in sys.modules if name == "parallax" or name.startswith("parallax.")]:
    del sys.modules[module_name]

from parallax import data, tools  # noqa: E402

APP = ROOT / "app" / "two_stories.py"

# Import the app module without rendering it (main() only runs as __main__).
_spec = importlib.util.spec_from_file_location("two_stories", APP)
two_stories = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(two_stories)


def page_text(node) -> str:
    parts = []
    for name in ("title", "header", "subheader", "markdown", "caption", "info", "warning", "error", "success"):
        parts.extend(str(element.value) for element in getattr(node, name, []))
    parts.extend(f"{metric.label} {metric.value}" for metric in getattr(node, "metric", []))
    return "\n".join(parts)


def test_full_match_set_agrees_with_showcase_tool():
    # The demo's lender table must rest on exactly the rows the shared tool counts.
    for industry, instrument in [("Software", "first_lien"), ("Insurance", "first_lien"),
                                 ("Software and Services", "any")]:
        tool = tools.get_comparables(industry, "any", instrument)
        rows = two_stories.matched_positions(industry, "any", instrument)
        assert len(rows) == tool["n"]
        assert rows["filer_ticker"].nunique() == tool["n_filers"]
        assert float(rows["spread_bps"].median()) == tool["median_spread_bps"]
        assert rows["filing"].notna().all()


def test_lender_table_sums_to_matches():
    rows = two_stories.matched_positions("Insurance", "any", "first_lien")
    table = two_stories.lender_table(rows)
    assert table["positions"].sum() == len(rows)
    assert abs(table["share_of_positions"].sum() - 1) < 1e-9


def test_hindsight_matches_archived_baseline_precision_at_15():
    archived = data.load_json("eval/model_results.json")["rows"]["baseline"]
    for quarter in ("2025-12-31", "2026-03-31"):
        listed = [row["borrower_id"] for row in tools.review_list(quarter, 15)["rows"]]
        seen = two_stories.hindsight(quarter, listed)
        assert seen["n_labelled"] == 15
        assert round(seen["hits"] / 15, 4) == round(archived[f"precision_at_15__{quarter}"], 4)
        assert seen["base_n"] == archived[f"n_test__{quarter}"]
        assert seen["base_hits"] == archived[f"n_positives__{quarter}"]


def test_wilson_bounds():
    lo, hi = two_stories.wilson(3, 15)
    assert 0 < lo < 0.2 < hi < 1
    assert two_stories.wilson(0, 0) is None


def test_two_story_app_renders_offline_with_caveats():
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert [tab.label for tab in app.tabs] == ["1. Price a new loan", "2. Quarterly review list"]

    price = page_text(app.tabs[0])
    assert "What you're deciding" in price
    assert "Contributing lenders" in price and "Source rows" in price
    assert "Small sample" in price  # Software / first lien has 5 positions

    review = page_text(app.tabs[1])
    assert "What you're deciding" in review
    assert "Model scores are withheld" in review
    assert "3 of 15" not in review  # default quarter 2026-03-31 has 1 of 15
    assert "1 of 15" in review


def test_one_lender_book_warning_fires():
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    app.selectbox[0].set_value("Software and Services")
    app.selectbox[1].set_value("any")
    app.run()
    assert not app.exception
    assert "One lender's book" in page_text(app.tabs[0])


def test_review_capacity_changes_list_length():
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    app.selectbox[3].set_value(10)
    app.run()
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Names on list"] == "10"
