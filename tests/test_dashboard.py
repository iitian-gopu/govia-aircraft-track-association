"""Dashboard tests with a fake API client. Fast and serverless; real end-to-end checks are run separately."""
import ast
import json
import math
import sys
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import charts, client as client_module, compare  # noqa: E402
from dashboard.client import ApiClient, ApiError  # noqa: E402

APP = ROOT / "dashboard" / "app.py"
A, B, C, N = (f"tracklet-{c * 8}-0000-0000" for c in "abcd")


def report(tracklet, index, minute, lat, lon, altitude=10000.0, **extra):
    row = {
        "observation_id": f"obs-{tracklet[9]}{index}", "tracklet_id": tracklet, "observed_at": f"2026-01-10T00:{minute:02d}:00.000Z",
        "received_at": f"2026-01-10T00:{minute:02d}:05.000Z", "latitude": lat, "longitude": lon, "source": "feed-a", "attributes": {"spd": 100.0},
    }
    if altitude is not None:
        row["altitude_ft"] = altitude
    return {**row, **extra}


def candidate(successor, rank=1, score=-1.5, decision="accepted", detail="no competing candidate", **measurements):
    values = dict(gap_s=30.0, endpoint_km=1.0, implied_kmh=120.0, forward_miss_km=0.5, backward_miss_km=0.4, speed_diff_kmh=5.0,
                  course_diff_deg=2.0, altitude_diff_ft=100.0, altitude_miss_ft=50.0, predecessor_reports=3, successor_reports=3, position_miss_km=0.45)
    return {"rank": rank, "successor": successor, "score": score, "decision": decision, "decision_detail": detail, "measurements": {**values, **measurements}}


def tracklet(tracklet_id, successor=None, code=None, candidates=(), gated=None):
    return {
        "tracklet_id": tracklet_id, "status": "accepted" if successor else "unresolved", "accepted_successor": successor,
        "reason_code": "accepted" if successor else code, "reason": f"reason for {code or 'accepted'}", "candidates": list(candidates), "gated_out": gated or {},
    }


def result(method, tracklets, version=None):
    return {
        "method": method, "version": version or f"{method}-v1", "config": {}, "policy": {"min_score": -9.0, "min_margin": 2.0, "plausible_score": -12.0},
        "scenario_id": "scenario-x", "snapshot_id": "snapshot-x", "receipt_cutoff": None, "input_fingerprint": "print-1",
        "links": {t["tracklet_id"]: t["accepted_successor"] for t in tracklets if t["accepted_successor"]}, "tracklets": tracklets,
    }


def response(baseline, learned, fingerprint="print-1", snapshot="snapshot-x", cutoff=None):
    return {
        "analysis": {"scenario_id": "scenario-x", "snapshot_id": snapshot, "receipt_cutoff": cutoff, "input_fingerprint": fingerprint, "tracklet_count": len(baseline["tracklets"]),
                     "observation_count": 0, "supporting_count": 0, "quality": {}},
        "baseline": baseline, "learned": learned,
    }


OBSERVATIONS = [
    report(A, 2, 2, 35.02, -110.02), report(A, 0, 0, 35.00, -110.00), report(A, 1, 1, 35.01, -110.01, altitude=None),
    report(A, 1, 1, 35.01, -110.01, altitude=None),  # identical retransmission
    report(B, 0, 4, 35.04, -110.04), report(B, 1, 5, 35.05, -110.05),
    report(C, 0, 6, 35.06, -110.06), report(C, 1, 7, 35.07, -110.07),
]
SUPPORTING = [
    {"record_id": "support-1", "observation_id": "obs-a0", "source": "telemetry-x", "received_at": "2026-01-10T00:00:10.000Z", "attributes": {"a1": 1.0}},
    {"record_id": "support-1", "observation_id": "obs-a0", "source": "telemetry-x", "received_at": "2026-01-10T00:00:10.000Z", "attributes": {"a1": 1.0}},
]


def baseline_result():
    return result("baseline", [
        tracklet(A, B, candidates=[candidate(B), candidate(C, 2, -7.0, "close_rival", "rival text", forward_miss_km=None)], gated={"gap_too_long": 2}),
        tracklet(B, code="close_rival", candidates=[candidate(C, 1, -3.0, "close_rival", "close to a rival")]),
        tracklet(C, code="no_candidate"),
    ])


def learned_result():
    return result("learned", [
        tracklet(A, code="below_threshold", candidates=[candidate(B, 1, 0.01, "below_threshold", "score 0.010 is below 0.020")]),
        tracklet(B, C, candidates=[candidate(C, 1, 0.8)]),
        tracklet(C, code="no_candidate"),
    ])


def scenario(key):
    partition, name = key.split("/")
    return {"key": key, "partition": partition, "name": name, "scenario_id": "scenario-x", "snapshot_id": f"snapshot-{name}", "receipt_cutoff": None,
            "tracklet_count": 3, "unique_observation_count": 7, "observation_row_count": 8, "supporting_row_count": 2}


class FakeClient:
    """Same public surface as ApiClient. `failures` maps a method name to an ApiError it should raise."""
    base_url = "http://fake"
    timeout = 30.0

    def __init__(self, keys=("validation/scenario_001", "train/scenario_002", "staged_demo/snapshot_1", "staged_demo/snapshot_2"),
                 answers=None, health_body=None):
        self.keys = keys
        self.failures = {}
        self.posts = []
        self.answers = answers or {}
        self.health_body = health_body or {"status": "ok", "ready": True, "problems": []}

    def _maybe_fail(self, name):
        if name in self.failures:
            raise self.failures[name]

    def health(self):
        self._maybe_fail("health")
        return self.health_body

    def scenarios(self):
        self._maybe_fail("scenarios")
        return [scenario(key) for key in self.keys]

    def records(self, partition, name):
        self._maybe_fail("records")
        extra = [report(N, 0, 3, 35.03, -110.03), report(N, 1, 3, 35.035, -110.035)] if name == "snapshot_2" else []
        return {"key": f"{partition}/{name}", "scenario_id": "scenario-x", "snapshot_id": name, "receipt_cutoff": None, "observations": OBSERVATIONS + extra, "supporting_records": SUPPORTING}

    def associate(self, records):
        self._maybe_fail("associate")
        self.posts.append(records["key"])
        answer = self.answers.get(records["key"])
        return answer if answer is not None else response(baseline_result(), learned_result())


def start(fake):
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["api_client"] = fake
    at.run()
    assert not at.exception
    return at


def click_run(at):
    at.button(key="run").click().run()
    assert not at.exception
    return at


def metrics(at):
    return {m.label: m.value for m in at.metric}


def texts(at):
    collections = (at.error, at.warning, at.info, at.success, at.markdown, at.caption)
    return "\n".join(str(element.value) for collection in collections for element in collection)


# Client error mapping

def mock_client(handler):
    return ApiClient("http://api", transport=httpx.MockTransport(handler))


def raises(handler, call=lambda c: c.scenarios()):
    with pytest.raises(ApiError) as caught:
        call(mock_client(handler))
    return caught.value


def test_connection_error_is_backend_unavailable():
    def handler(request):
        raise httpx.ConnectError("refused")
    assert raises(handler).code == client_module.BACKEND_UNAVAILABLE


def test_timeout_is_its_own_code():
    def handler(request):
        raise httpx.ReadTimeout("slow")
    assert raises(handler).code == client_module.TIMEOUT


def test_error_envelope_carries_code_message_and_issues():
    issue = {"code": "bad_latitude", "record_kind": "observation", "message": "out of range", "index": 3, "record_id": "obs-1", "field": "latitude"}
    error = raises(lambda r: httpx.Response(422, json={"error": {"code": "validation_failed", "message": "bad records", "issues": [issue]}}))
    assert (error.code, error.message, error.issues, error.status) == ("validation_failed", "bad records", [issue], 422)


def test_non_json_failure_and_success_are_reported_not_raised_raw():
    assert raises(lambda r: httpx.Response(502, text="<html>bad gateway</html>")).code == "internal_error"
    error = raises(lambda r: httpx.Response(200, text="<html>hello</html>"))
    assert error.code == client_module.BAD_RESPONSE and "<html>" not in error.message


def test_failure_without_envelope():
    assert raises(lambda r: httpx.Response(500, json={"detail": "boom"})).code == "internal_error"
    assert raises(lambda r: httpx.Response(404, json={"detail": "nope"})).code == client_module.BAD_RESPONSE


def test_success_with_the_wrong_shape_is_rejected():
    assert raises(lambda r: httpx.Response(200, json={"items": []})).code == client_module.BAD_RESPONSE
    assert raises(lambda r: httpx.Response(200, json=[1, 2])).code == client_module.BAD_RESPONSE
    records = {"observations": [], "supporting_records": []}
    bad = {"analysis": {}, "baseline": {"tracklets": [], "links": {}}, "learned": {"tracklets": None, "links": {}}}
    assert raises(lambda r: httpx.Response(200, json=bad), lambda c: c.associate(records)).code == client_module.BAD_RESPONSE


def test_health_503_is_an_answer_not_an_error():
    body = {"status": "not_ready", "ready": False, "problems": ["model file missing"]}
    assert mock_client(lambda r: httpx.Response(503, json=body)).health()["problems"] == ["model file missing"]


def test_health_503_carrying_health_fields_and_an_error_is_still_an_answer():
    body = {"status": "unavailable", "ready": False, "problems": ["learned model files are missing"],
            "error": {"code": "model_unavailable", "message": "learned model files are missing", "issues": []}}
    assert mock_client(lambda r: httpx.Response(503, json=body)).health()["ready"] is False


def test_health_503_with_error_envelope_raises_its_code():
    error = raises(lambda r: httpx.Response(503, json={"error": {"code": "model_unavailable", "message": "no model", "issues": []}}), lambda c: c.health())
    assert error.code == "model_unavailable"


def test_associate_503_is_model_unavailable_and_payload_is_this_scenario_only():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(503, json={"error": {"code": "model_unavailable", "message": "no model", "issues": []}})

    records = {"key": "k", "observations": [{"observation_id": "o"}], "supporting_records": [], "scenario_id": "s", "snapshot_id": "n", "receipt_cutoff": None}
    assert raises(handler, lambda c: c.associate(records)).code == "model_unavailable"
    assert seen == {"observations": [{"observation_id": "o"}], "supporting_records": [], "scenario_id": "s", "snapshot_id": "n", "receipt_cutoff": None}


def test_records_path_is_escaped():
    paths = []

    def handler(request):
        paths.append(request.url.raw_path.decode())
        return httpx.Response(200, json={"observations": [], "supporting_records": []})

    mock_client(handler).records("a b", "c/d")
    assert paths == ["/scenarios/a%20b/c%2Fd/records"]


def test_base_url_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("GOVIA_API_URL", "http://example.test:9/")
    assert ApiClient.from_env().base_url == "http://example.test:9"
    monkeypatch.delenv("GOVIA_API_URL")
    assert ApiClient.from_env().base_url == "http://127.0.0.1:8000"


# Snapshot comparison

def link_result(links, ids, method="baseline"):
    tracklets = []
    for tracklet_id in ids:
        successor = links.get(tracklet_id)
        tracklets.append(tracklet(tracklet_id, successor, None if successor else "no_candidate", [candidate(successor)] if successor else []))
    return result(method, tracklets)


def test_compare_finds_new_added_withdrawn_replaced_and_unchanged():
    before = link_result({"a": "c", "d": "e", "f": "g"}, ["a", "c", "d", "e", "f", "g", "h", "i"])
    after = link_result({"a": "n", "n": "c", "d": "e", "h": "i"}, ["a", "c", "d", "e", "f", "g", "h", "i", "n"])
    changes = compare.compare_snapshots(before, after)
    assert changes["new_tracklets"] == ["n"]
    assert changes["removed_tracklets"] == []
    assert sorted((x["predecessor"], x["successor"]) for x in changes["added"]) == [("h", "i"), ("n", "c")]
    assert [(x["predecessor"], x["successor"]) for x in changes["withdrawn"]] == [("f", "g")]
    assert [(x["predecessor"], x["old_successor"], x["new_successor"]) for x in changes["replaced"]] == [("a", "c", "n")]
    assert [(x["predecessor"], x["successor"]) for x in changes["unchanged"]] == [("d", "e")]
    assert changes["changed"] is True
    replaced = changes["replaced"][0]
    assert replaced["new_successor_is_new"] and replaced["new_tracklets_between"]


def test_compare_takes_evidence_from_the_snapshot_that_made_the_link():
    before = result("baseline", [tracklet("a", "c", candidates=[candidate("c", gap_s=100.0)]), tracklet("c", code="no_candidate")])
    after = result("baseline", [tracklet("a", "n", candidates=[candidate("n", gap_s=20.0)]), tracklet("c", code="no_candidate"), tracklet("n", code="no_candidate")])
    changes = compare.compare_snapshots(before, after)
    assert changes["replaced"][0]["before"]["measurements"]["gap_s"] == 100.0
    assert changes["replaced"][0]["after"]["measurements"]["gap_s"] == 20.0
    rows = compare.change_rows(changes)
    assert [(r["Change"], r["Evidence from"], r["Gap (s)"]) for r in rows] == [
        ("Replaced: earlier link", "Snapshot 1", "100.0"), ("Replaced: new link", "Snapshot 2", "20.0"),
    ]


def test_compare_with_no_change_says_so():
    same = link_result({"a": "b"}, ["a", "b"])
    changes = compare.compare_snapshots(same, same)
    assert changes["changed"] is False and not changes["new_tracklets"] and len(changes["unchanged"]) == 1
    assert compare.change_rows(changes) == []


def test_compare_handles_a_new_tracklet_with_no_link_change():
    changes = compare.compare_snapshots(link_result({"a": "b"}, ["a", "b"]), link_result({"a": "b"}, ["a", "b", "n"]))
    assert changes["new_tracklets"] == ["n"] and changes["changed"] is True and not changes["replaced"]


# Plain wording and table rows

@pytest.mark.parametrize("code, category", [
    ("no_candidate", compare.NO_PLAUSIBLE), ("missing_measurement", compare.INSUFFICIENT), ("below_threshold", compare.INSUFFICIENT),
    ("close_rival", compare.SIMILAR), ("successor_taken", compare.CONFLICT), ("predecessor_taken", compare.CONFLICT),
    ("possible_intermediate", compare.POLICY), ("physically_inconsistent", compare.POLICY),
])
def test_reason_categories(code, category):
    assert compare.reason_category(code) == category
    assert compare.reason_meaning(code) and code not in compare.reason_meaning(code)


def test_unknown_reason_code_is_shown_not_hidden():
    assert "strange_code" in compare.reason_category("strange_code")
    assert "strange_code" in compare.reason_meaning("strange_code")


def test_missing_measurement_is_never_zero_and_real_zero_is_kept():
    rows = compare.candidate_rows(tracklet(A, candidates=[candidate(B, forward_miss_km=None, altitude_diff_ft=0.0)]))
    assert rows[0]["Fwd miss (km)"] == "not available"
    assert rows[0]["Alt diff (ft)"] == "0"
    assert compare.fmt(None) == "not available" and compare.fmt(float("nan")) == "not available" and compare.fmt(0.0, 1) == "0.0"


def test_explanations_cover_each_outcome():
    won = tracklet(A, B, candidates=[candidate(B)])
    assert "aaaaaaaa" not in compare.explain_decision(won) and "accepted" in compare.explain_decision(won)
    withheld = tracklet(A, code="close_rival", candidates=[candidate(B, 1, -2.0, "close_rival", "rival details")])
    assert "withheld" in compare.explain_decision(withheld) and "rival details" in compare.explain_decision(withheld)
    runner_up = tracklet(A, C, candidates=[candidate(B, 1, -1.0, "possible_intermediate", "x"), candidate(C, 2, -2.0)])
    assert "Rank 2" in compare.explain_decision(runner_up) and "was not" in compare.explain_decision(runner_up)
    assert compare.explain_decision(tracklet(A, code="no_candidate")) == compare.reason_meaning("no_candidate")


def test_agreement_means_same_successor_or_both_unresolved():
    counts = compare.agreement_counts(baseline_result(), learned_result())
    assert counts == {"agree": 1, "disagree": 2}  # only C agrees: both unresolved


def test_unresolved_groups_follow_the_fixed_order():
    groups = compare.unresolved_groups(result("baseline", [
        tracklet("a", code="close_rival"), tracklet("b", code="no_candidate"), tracklet("c", "d"), tracklet("d", code="predecessor_taken"),
    ]))
    assert list(groups) == [compare.NO_PLAUSIBLE, compare.SIMILAR, compare.CONFLICT]


def test_observation_rows_show_missing_altitude_copies_and_supporting_counts():
    rows = compare.observation_rows(OBSERVATIONS, SUPPORTING, A)
    assert [r["Observation ID"] for r in rows] == ["obs-a0", "obs-a1", "obs-a2"]
    assert rows[1]["Altitude (ft)"] == "not available" and rows[1]["Copies received"] == "2"
    assert rows[0]["Supporting records"] == "1" and rows[2]["Supporting records"] == "0"
    assert json.loads(rows[0]["Attributes"]) == {"spd": 100.0}


def test_supporting_rows_dedupe_and_join_to_the_tracklet():
    assert [r["Record ID"] for r in compare.supporting_rows(OBSERVATIONS, SUPPORTING, A)] == ["support-1"]
    assert compare.supporting_rows(OBSERVATIONS, SUPPORTING, B) == []


# Chart building

def figure(selected=None, highlight=(), options=charts.PlotOptions(), observations=OBSERVATIONS, method_result=None):
    return charts.build_track_figure(observations, method_result or baseline_result(), selected, highlight, options)


def traces(fig, role):
    return [t for t in fig.data if t.meta and t.meta.get("role") == role]


def path(fig, tracklet_id):
    return next(t for t in traces(fig, "path") if t.meta["tracklet_id"] == tracklet_id)


def test_tracks_are_deduped_and_chronological():
    trace = path(figure(), A)
    assert list(trace.y) == [35.00, 35.01, 35.02]
    assert [c[1] for c in trace.customdata] == ["obs-a0", "obs-a1", "obs-a2"]


def test_row_order_does_not_change_the_paths():
    assert list(path(figure(observations=list(reversed(OBSERVATIONS))), A).y) == [35.00, 35.01, 35.02]


def test_hover_data_shows_missing_altitude_and_a_real_zero():
    rows = [report(A, 0, 0, 35.0, -110.0, altitude=None), report(A, 1, 1, 35.1, -110.1, altitude=0.0)]
    customdata = path(figure(observations=rows), A).customdata
    assert customdata[0][3] == "not available" and customdata[1][3] == "0 ft"
    assert customdata[0][0] == A and customdata[0][4] == "feed-a"
    assert "%{customdata[2]}" in path(figure(observations=rows), A).hovertemplate


def test_axes_are_longitude_across_latitude_up_with_corrected_aspect():
    rows = [report(A, 0, 0, 60.0, 10.0), report(A, 1, 1, 60.0, 11.0)]
    fig = figure(observations=rows, method_result=result("baseline", [tracklet(A, code="no_candidate")]))
    assert list(path(fig, A).x) == [10.0, 11.0] and list(path(fig, A).y) == [60.0, 60.0]
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis.scaleratio == pytest.approx(2.0)  # cos(60 degrees) is one half
    assert fig.layout.xaxis.autorange is None and fig.layout.yaxis.autorange is None
    assert "Longitude" in fig.layout.xaxis.title.text and "Latitude" in fig.layout.yaxis.title.text


def test_accepted_link_connects_predecessor_end_to_successor_start():
    links = traces(figure(), "link")
    assert [(t.meta["predecessor"], t.meta["successor"]) for t in links] == [(A, B)]
    assert list(links[0].x) == [-110.02, -110.04] and list(links[0].y) == [35.02, 35.04]
    assert links[0].line.dash == "solid"


def test_links_can_be_hidden():
    assert traces(figure(options=charts.PlotOptions(show_links=False)), "link") == []


def test_competing_candidates_are_dashed_and_only_for_the_selected_tracklet():
    assert traces(figure(), "candidate") == []
    drawn = traces(figure(selected=A), "candidate")
    assert [t.meta["successor"] for t in drawn] == [C]  # B is the accepted link, drawn as a link instead
    assert all(t.line.dash == "dash" for t in drawn)


def test_unresolved_and_selected_styles_differ():
    fig = figure(selected=B)
    assert path(fig, C).line.dash == "dot"
    assert path(fig, A).line.dash == "solid" and path(fig, A).line.color == charts.ACCEPTED
    assert path(fig, B).line.color == charts.INK and path(fig, B).line.width > path(fig, A).line.width
    assert fig.data[-1].meta["role"] == "labels"


def test_highlight_marks_a_new_tracklet():
    fig = figure(highlight=[C])
    assert path(fig, C).name == "New tracklet" and path(fig, C).line.width > path(fig, A).line.width


def test_labels_are_short_and_legend_has_one_entry_per_kind():
    fig = figure(selected=A)
    assert set(traces(fig, "labels")[0].text) == {"aaaaaaaa", "bbbbbbbb", "cccccccc"}
    names = [t.name for t in fig.data if t.showlegend is not False]
    assert len(names) == len(set(names))
    assert "Accepted link" in names and "Tracklet start" in names and "Tracklet end" in names


def test_start_and_end_markers_sit_on_first_and_last_reports():
    fig = figure()
    assert traces(fig, "start")[0].y[0] == 35.00 and traces(fig, "end")[0].y[0] == 35.02


def test_short_labels_stay_unique():
    labels = charts.short_labels(["tracklet-aaaa-1", "tracklet-aaaa-2", "tracklet-bbbb-3"])
    assert len(set(labels.values())) == 3 and labels["tracklet-bbbb-3"] == "bbbb"


def test_empty_figure_builds():
    assert not figure(observations=[], method_result=result("baseline", [])).data


# Whole app, with a fake API

def test_initial_state_prompts_to_run():
    at = start(FakeClient())
    assert "Run association" in texts(at)
    assert at.title[0].value == "Aircraft Track Association — Govia"
    assert at.sidebar.success[0].value == "API ready"
    assert not at.metric


def test_scenarios_are_grouped_by_partition():
    at = start(FakeClient())
    assert list(at.selectbox(key="partition").options) == ["validation", "train", "staged_demo"]
    at.selectbox(key="partition").select("train").run()
    assert not at.exception
    assert list(at.selectbox(key="scenario::train").options) == ["scenario_002"]
    assert "Tracklets: 3" in texts(at) and "Supporting records: 2 rows" in texts(at)


def test_run_renders_summary_plot_inspector_records_and_footer():
    at = click_run(start(FakeClient()))
    assert metrics(at) == {
        "Tracklets": "3", "Accepted (baseline)": "1", "Unresolved (baseline)": "2", "Candidates (baseline)": "3", "Methods agree": "1", "Methods disagree": "2",
    }
    assert len(at.get("plotly_chart")) == 1
    assert "Association inspector" in [s.value for s in at.subheader]
    assert any("Accepted" in s.value for s in at.success)
    candidates = at.dataframe[0].value
    assert candidates["Fwd miss (km)"].tolist() == ["0.50", "not available"]
    assert candidates["Gap (s)"].tolist() == ["30.0", "30.0"]
    assert len(at.expander) >= 2
    assert "Input fingerprint: print-1" in texts(at) and "baseline-v1" in texts(at) and "learned-v1" in texts(at)


def test_train_scenarios_carry_the_in_sample_note():
    at = click_run(start(FakeClient()))
    assert compare.TRAIN_NOTE not in texts(at)
    at.selectbox(key="partition").select("train").run()
    click_run(at)
    assert compare.TRAIN_NOTE in texts(at)


def test_rendered_page_shows_no_evaluation_metrics():
    at = click_run(start(FakeClient()))
    assert not {"precision", "recall"} & set(texts(at).lower().split())


def test_method_switch_changes_view_and_keeps_tracklet():
    at = click_run(start(FakeClient()))
    selector = at.selectbox(key="tracklet::validation/scenario_001")
    selector.select(B).run()
    assert not at.exception
    at.radio(key="method").set_value("learned").run()
    assert not at.exception
    assert metrics(at)["Accepted (learned)"] == "1" and metrics(at)["Candidates (learned)"] == "2"
    assert at.selectbox(key="tracklet::validation/scenario_001").value == B


def test_scale_notes_follow_the_method():
    at = click_run(start(FakeClient()))
    assert "unbounded" in texts(at).lower() or "no lower bound" in texts(at).lower()
    at.radio(key="method").set_value("learned").run()
    assert "uncalibrated" in texts(at)


def test_results_are_cached_until_rerun():
    fake = FakeClient()
    at = click_run(start(fake))
    at.radio(key="method").set_value("learned").run()
    assert fake.posts == ["validation/scenario_001"]
    click_run(at)
    assert fake.posts == ["validation/scenario_001"] * 2


def test_compare_view_filters_and_warns_about_scales():
    at = click_run(start(FakeClient()))
    at.segmented_control(key="view").set_value("Baseline vs learned").run()
    assert not at.exception
    assert "different scales" in texts(at)
    assert at.dataframe[0].value["Agreement"].tolist() == ["Disagree", "Disagree", "Agree"]
    at.radio(key="agree_filter").set_value("Methods agree").run()
    assert not at.exception
    assert at.dataframe[0].value["Agreement"].tolist() == ["Agree"]
    assert any(m.value == "**Baseline**" for m in at.markdown) and any(m.value == "**Learned**" for m in at.markdown)


def test_compare_view_filter_with_no_matches():
    same = response(baseline_result(), baseline_result())
    at = click_run(start(FakeClient(answers={"validation/scenario_001": same})))
    at.segmented_control(key="view").set_value("Baseline vs learned").run()
    at.radio(key="agree_filter").set_value("Methods disagree").run()
    assert not at.exception
    assert "No tracklets match" in texts(at)


def test_unresolved_view_groups_and_filters():
    at = click_run(start(FakeClient()))
    at.segmented_control(key="view").set_value("Unresolved").run()
    assert not at.exception
    page = texts(at)
    assert "does not mean the aircraft stopped" in page
    assert compare.SIMILAR in page and compare.NO_PLAUSIBLE in page
    at.selectbox(key="unresolved_filter").select(compare.SIMILAR).run()
    assert not at.exception
    assert compare.NO_PLAUSIBLE not in [m.value.split("**")[1] for m in at.markdown if m.value.startswith("**")]
    assert at.dataframe[0].value["Reason code"].tolist() == ["close_rival"]
    at.selectbox(key="unresolved_filter").select(compare.CONFLICT).run()
    assert "No unresolved tracklets have this reason" in texts(at)


def test_backend_down_then_recovers_on_rerun():
    fake = FakeClient()
    down = ApiError(client_module.BACKEND_UNAVAILABLE, "Could not reach the API at http://fake.")
    fake.failures = {"health": down, "scenarios": down}
    at = start(fake)
    assert "Backend unavailable" in texts(at)
    assert at.sidebar.error[0].value == "API unavailable"
    assert at.button(key="run").disabled
    fake.failures = {}
    at.button(key="check_api").click().run()
    assert not at.exception
    assert "Backend unavailable" not in texts(at)
    assert at.sidebar.success[0].value == "API ready"
    click_run(at)
    assert metrics(at)["Tracklets"] == "3"


def test_failure_during_run_clears_once_backend_returns():
    fake = FakeClient()
    at = start(fake)
    fake.failures = {"records": ApiError(client_module.BACKEND_UNAVAILABLE, "Could not reach the API.")}
    click_run(at)
    assert "Backend unavailable" in texts(at)
    fake.failures = {}
    at.button(key="check_api").click().run()
    assert "Backend unavailable" not in texts(at) and "Run association" in texts(at)


def test_backend_failure_during_run_is_reported_once():
    fake = FakeClient()
    at = start(fake)
    down = ApiError(client_module.BACKEND_UNAVAILABLE, "Could not reach the API.")
    fake.failures = {"health": down, "records": down}
    click_run(at)
    assert sum("Backend unavailable" in str(e.value) for e in at.error) == 1


def test_cached_result_stays_visible_while_backend_is_down():
    fake = FakeClient()
    at = click_run(start(fake))
    fake.failures = {"health": ApiError(client_module.BACKEND_UNAVAILABLE, "Could not reach the API.")}
    at.button(key="check_api").click().run()
    assert not at.exception
    assert "Backend unavailable" in texts(at) and metrics(at)["Tracklets"] == "3"


def test_model_unavailable_from_health_shows_problems():
    at = start(FakeClient(health_body={"status": "not_ready", "ready": False, "problems": ["learned model file is missing"]}))
    assert "Model unavailable" in texts(at) and "learned model file is missing" in texts(at)
    assert at.sidebar.warning[0].value == "API not ready"


def test_model_unavailable_from_associate():
    fake = FakeClient()
    at = start(fake)
    fake.failures = {"associate": ApiError("model_unavailable", "The learned model could not be loaded.", status=503)}
    click_run(at)
    assert "Model unavailable" in texts(at) and "could not be loaded" in texts(at)


def test_invalid_input_shows_issues():
    fake = FakeClient()
    at = start(fake)
    issue = {"code": "bad_latitude", "record_kind": "observation", "message": "out of range", "index": 3, "record_id": "obs-1", "field": None}
    fake.failures = {"associate": ApiError("validation_failed", "1 record failed validation.", [issue], 422)}
    click_run(at)
    assert not at.exception
    assert "Invalid input" in texts(at)
    table = at.dataframe[0].value
    assert table["field"].tolist() == ["not available"] and table["message"].tolist() == ["out of range"]


def test_server_error_is_named_and_does_not_leak_a_traceback():
    fake = FakeClient()
    at = start(fake)
    fake.failures = {"associate": ApiError("internal_error", "The server failed.", status=500)}
    click_run(at)
    assert not at.exception
    assert "Unexpected server error" in texts(at)


def test_unexpected_exception_in_rendering_is_contained():
    fake = FakeClient()
    at = start(fake)
    fake.failures = {"associate": RuntimeError("secret internals")}
    at.button(key="run").click().run()
    assert not at.exception
    assert "unexpected problem" in texts(at) and "secret internals" not in texts(at)


def test_empty_scenario_list():
    at = start(FakeClient(keys=()))
    assert "lists no scenarios" in texts(at)
    assert at.button(key="run").disabled


def test_empty_response():
    empty = response(result("baseline", []), result("learned", []))
    at = click_run(start(FakeClient(answers={"validation/scenario_001": empty})))
    assert not at.exception
    assert "returned no tracklets" in texts(at)


def staged_answers(second_baseline=None):
    first = response(
        result("baseline", [tracklet(A, B, candidates=[candidate(B, gap_s=200.0)]), tracklet(B, code="no_candidate")]),
        result("learned", [tracklet(A, B, candidates=[candidate(B, 1, 0.3)]), tracklet(B, code="no_candidate")]),
        fingerprint="print-s1", snapshot="snapshot-one", cutoff="2026-01-10T00:10:00Z",
    )
    baseline = second_baseline or result("baseline", [
        tracklet(A, N, candidates=[candidate(N, gap_s=40.0)]), tracklet(N, B, candidates=[candidate(B, gap_s=50.0)]), tracklet(B, code="no_candidate"),
    ])
    second = response(
        baseline, result("learned", [tracklet(A, B, candidates=[candidate(B, 1, 0.3)]), tracklet(B, code="no_candidate"), tracklet(N, code="no_candidate")]),
        fingerprint="print-s2", snapshot="snapshot-two", cutoff="2026-01-10T00:20:00Z",
    )
    return {"staged_demo/snapshot_1": first, "staged_demo/snapshot_2": second}


def open_evidence(at):
    at.segmented_control(key="view").set_value("Additional evidence").run()
    assert not at.exception
    at.button(key="run_staged").click().run()
    assert not at.exception
    return at


def test_staged_comparison_posts_each_snapshot_and_reports_changes():
    fake = FakeClient(answers=staged_answers())
    at = open_evidence(start(fake))
    assert fake.posts == ["staged_demo/snapshot_1", "staged_demo/snapshot_2"]
    values = metrics(at)
    assert values["New tracklets (baseline)"] == "1" and values["Replaced (baseline)"] == "1" and values["Links added (baseline)"] == "1"
    assert values["Replaced (learned)"] == "0" and values["Unchanged (learned)"] == "1"
    assert "not a change of identity" in texts(at)
    assert "print-s1" in at.dataframe[0].value["Fingerprint"].tolist() and "print-s2" in at.dataframe[0].value["Fingerprint"].tolist()
    assert at.dataframe[0].value["Receipt cutoff"].tolist() == ["2026-01-10T00:10:00Z", "2026-01-10T00:20:00Z"]
    assert at.dataframe[1].value["Change"].tolist() == ["Added", "Replaced: earlier link", "Replaced: new link"]
    assert len(at.get("plotly_chart")) == 1
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert any(t.get("name") == "New tracklet" for t in spec["data"])


def test_staged_comparison_says_when_nothing_changed():
    same = staged_answers()["staged_demo/snapshot_1"]
    at = open_evidence(start(FakeClient(answers={"staged_demo/snapshot_1": same, "staged_demo/snapshot_2": same})))
    assert "No link changed under the baseline method" in texts(at) and "No link changed under the learned method" in texts(at)


def test_staged_comparison_shows_failure_of_either_snapshot():
    fake = FakeClient()
    at = start(fake)
    at.segmented_control(key="view").set_value("Additional evidence").run()
    fake.failures = {"associate": ApiError("internal_error", "The server failed.", status=500)}
    at.button(key="run_staged").click().run()
    assert not at.exception
    assert "Unexpected server error" in texts(at)


def test_staged_view_needs_both_snapshots_listed():
    at = start(FakeClient(keys=("validation/scenario_001",)))
    at.segmented_control(key="view").set_value("Additional evidence").run()
    assert "nothing to compare" in texts(at)


# Boundaries

FORBIDDEN = {"baseline", "learned", "association", "candidates", "labels", "training", "evaluation"}


def imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_dashboard_never_imports_inference_modules():
    files = list((ROOT / "dashboard").glob("*.py"))
    assert len(files) >= 4
    for path in files:
        for name in imported_modules(path):
            parts = name.split(".")
            assert parts[0] != "govia_tracks", f"{path.name} imports {name}"
            assert not (parts[0] == "dashboard" and set(parts) & FORBIDDEN), f"{path.name} imports {name}"


def test_dashboard_never_touches_labels_or_evaluation_metrics():
    for path in (ROOT / "dashboard").glob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        for word in ("labels.jsonl", "precision", "recall"):
            assert word not in source, f"{path.name} mentions {word}"


def test_cos_latitude_scale_matches_geometry_at_the_scenario_latitude():
    rows = [report(A, 0, 0, 35.0, -110.0), report(A, 1, 1, 35.0, -110.1)]
    ratio = figure(observations=rows, method_result=result("baseline", [tracklet(A, code="no_candidate")])).layout.yaxis.scaleratio
    assert ratio == pytest.approx(1 / math.cos(math.radians(35.0)))


def test_replacement_through_several_new_tracklets_says_how_many():
    # A -> C becomes A -> N -> M -> C once N and M arrive.
    before = {"links": {"A": "C"}, "tracklets": [tracklet("A"), tracklet("C")]}
    after = {"links": {"A": "N", "N": "M", "M": "C"}, "tracklets": [tracklet("A"), tracklet("N"), tracklet("M"), tracklet("C")]}
    changes = compare.compare_snapshots(before, after)
    assert changes["replaced"][0]["new_tracklets_between"] == 2
    assert all("2 new tracklets now sit between" in row["Note"] for row in compare.change_rows(changes) if row["Change"].startswith("Replaced"))


def test_new_tracklet_that_does_not_lead_back_is_not_called_an_intermediate():
    before = {"links": {"A": "C"}, "tracklets": [tracklet("A"), tracklet("C")]}
    after = {"links": {"A": "N"}, "tracklets": [tracklet("A"), tracklet("N"), tracklet("C")]}
    changes = compare.compare_snapshots(before, after)
    assert changes["replaced"][0]["new_tracklets_between"] == 0
    assert all("newly available" in row["Note"] for row in compare.change_rows(changes))


def test_new_tracklet_chain_that_loops_ends_rather_than_hangs():
    assert compare._new_tracklets_between({"N": "M", "M": "N"}, "N", "C", {"N", "M"}) == 0


def test_withdrawn_link_whose_tracklet_left_says_so_instead_of_an_unknown_reason():
    before = {"links": {"A": "C"}, "tracklets": [tracklet("A"), tracklet("C")]}
    after = {"links": {}, "tracklets": [tracklet("C")]}
    rows = compare.change_rows(compare.compare_snapshots(before, after))
    assert rows[0]["Note"] == "The tracklet is not in snapshot 2."


def test_earlier_results_are_marked_once_the_models_are_unavailable():
    fake = FakeClient()
    at = click_run(start(fake))
    assert "earlier successful analysis" not in texts(at) and metrics(at)
    fake.health_body = {"status": "unavailable", "ready": False, "problems": ["learned model files are missing"]}
    at.run()
    assert not at.exception
    assert "Model unavailable" in texts(at) and "earlier successful analysis" in texts(at)


def test_earlier_results_are_marked_once_the_backend_is_gone():
    fake = FakeClient()
    at = click_run(start(fake))
    fake.failures["health"] = ApiError(client_module.BACKEND_UNAVAILABLE, "Could not reach the API.")
    at.run()
    assert not at.exception
    assert "earlier successful analysis" in texts(at)
