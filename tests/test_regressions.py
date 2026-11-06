"""Regression tests for specific defects, plus selection and degraded-input behaviour pinned where coverage was missing.

Malformed labels used to hang evaluation, so those calls run in a child process with a time limit:
a regression fails here instead of stopping the suite.
"""
import dataclasses
import json
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from govia_tracks.association import (
    ACCEPTED, BELOW_THRESHOLD, CLOSE_RIVAL, MISSING_MEASUREMENT, NO_CANDIDATE, PHYSICALLY_INCONSISTENT, POSSIBLE_INTERMEDIATE, PREDECESSOR_TAKEN, SUCCESSOR_TAKEN,
    AcceptancePolicy, ScoredCandidate, select_links,
)
from govia_tracks.baseline import BaselineConfig, NoiseScales, analyze as analyze_baseline, config_from_dict, config_to_dict, fit_scales, load_config, score_candidate
from govia_tracks.candidates import Candidate, CandidateConfig, PairMeasurements
from govia_tracks.errors import InputValidationError
from govia_tracks.evaluation import LinkCounts, aircraft_chains, aircraft_groups, count_links, labels_without, metrics, split_tracklets, validate_labels
from govia_tracks.labels import load_labels
from govia_tracks.learned import analyze as analyze_learned, load_model, load_policy
from govia_tracks.loader import load_snapshot
from govia_tracks.movement import summarize_tracklet
from govia_tracks.preprocess import prepare_scenario

from conftest import observation

ROOT = Path(__file__).resolve().parent.parent
EPOCH = datetime(2026, 1, 10, tzinfo=timezone.utc)
SPEED = 0.0012  # degrees of latitude per second: about 480 km/h
FLOOR_SCALES = NoiseScales(position_km=0.003, position_km_per_s=0.002, altitude_ft=0.0, altitude_ft_per_s=1.0, speed_kmh=5.0, course_deg=1.0)
TIME_LIMIT_S = 30
LABELS = {"a": "b", "b": "c", "c": None, "x": "y", "y": None, "n": None}

RHO = {"x": "a", "a": "b", "b": "a"}  # one tail into a loop: every walk from x used to run forever
LOOP = {"a": "b", "b": "a"}  # no start, so the group and chain walks used to skip it silently
SELF = {"a": "a"}
UNKNOWN = {"a": "ghost"}
BRANCH = {"a": "c", "b": "c", "c": None}
NOT_A_STRING = {"a": 5, "b": None}
EMPTY = {"a": "", "b": None}


def stamp(seconds):
    return (EPOCH + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def flight(tracklet_id, start_s, count=6, step_s=10, lat=35.0, lon=-108.0, north=SPEED, east=0.0, altitude=15000.0, climb=0.0):
    """Reports on a line defined from time zero, so fragments of one flight line up across gaps."""
    rows = []
    for i in range(count):
        t = start_s + i * step_s
        rows.append(observation(
            observation_id=f"{tracklet_id}-{i}", tracklet_id=tracklet_id, observed_at=stamp(t), received_at=stamp(t + 1),
            latitude=lat + north * t, longitude=lon + east * t, altitude_ft=None if altitude is None else altitude + climb * t,
        ))
    return rows


def scored(a, b, score, acceptable=True):
    return ScoredCandidate(a, b, score, {}, acceptable)


def entry(result, tracklet_id):
    return next(t for t in result["tracklets"] if t["tracklet_id"] == tracklet_id)


def outcome_in_child(call, labels):
    """Run `call` on these labels in a child process. Returns the exception's type name, or "ok". A hang fails the test."""
    code = textwrap.dedent(f"""
        import json
        from govia_tracks.evaluation import aircraft_chains, aircraft_groups, count_links, labels_without
        labels = json.loads({json.dumps(json.dumps(labels))})
        try:
            {call}
            print("ok")
        except Exception as error:
            print(type(error).__name__)
    """)
    try:
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=TIME_LIMIT_S, check=True)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{call} did not finish within {TIME_LIMIT_S} s on {labels}")
    return done.stdout.strip()


# --- Malformed labels are rejected, never looped over ---


@pytest.mark.parametrize("call", [
    "aircraft_groups(labels)", "aircraft_chains(labels)", "count_links({}, labels)", "labels_without(labels, {'b'})", "labels_without(labels, {'a', 'b'})",
])
@pytest.mark.parametrize("labels", [RHO, LOOP], ids=["rho", "loop"])
def test_cyclic_labels_raise_instead_of_hanging(call, labels):
    assert outcome_in_child(call, labels) == "InputValidationError"


@pytest.mark.parametrize("labels,fragment", [
    (RHO, "more than one predecessor"), (LOOP, "cycle"), (SELF, "itself"), (UNKNOWN, "ghost"), (BRANCH, "more than one predecessor"), (NOT_A_STRING, "successor"), (EMPTY, "successor"),
], ids=["rho", "loop", "self", "unknown", "branch", "not_a_string", "empty"])
def test_each_invalid_structure_is_rejected_with_a_clear_message(labels, fragment):
    with pytest.raises(InputValidationError, match=fragment) as caught:
        validate_labels(labels)
    assert {issue.record_kind for issue in caught.value.issues} == {"label"}


def test_every_structural_problem_is_reported_together():
    with pytest.raises(InputValidationError) as caught:
        validate_labels({"a": "ghost", "b": "b", "c": "e", "d": "e", "e": None})
    messages = " ".join(issue.message for issue in caught.value.issues)
    assert len(caught.value.issues) == 3 and "ghost" in messages and "itself" in messages and "more than one predecessor" in messages


def test_valid_structures_pass_and_keep_their_results():
    validate_labels({})
    validate_labels(LABELS)
    assert aircraft_chains(LABELS) == [["a", "b", "c"], ["n"], ["x", "y"]]
    assert aircraft_groups(LABELS) == {"a": "a", "b": "a", "c": "a", "n": "n", "x": "x", "y": "x"}
    assert labels_without(LABELS, {"b"}) == {"a": "c", "c": None, "x": "y", "y": None, "n": None}


@pytest.mark.parametrize("call", [
    aircraft_groups, aircraft_chains, lambda labels: labels_without(labels, set()), lambda labels: labels_without(labels, {"ghost"}), lambda labels: count_links({}, labels),
], ids=["groups", "chains", "without_nothing", "without_the_unknown", "count"])
def test_unknown_successor_is_a_validation_error_not_a_key_error(call):
    with pytest.raises(InputValidationError, match="ghost"):
        call(UNKNOWN)


def write_labels(folder, rows):
    (folder / "labels.jsonl").write_text("".join(json.dumps({"tracklet_id": a, "successor_tracklet_id": b}) + "\n" for a, b in rows.items()), encoding="utf-8")


@pytest.mark.parametrize("labels", [RHO, LOOP, SELF, UNKNOWN, BRANCH, NOT_A_STRING, EMPTY], ids=["rho", "loop", "self", "unknown", "branch", "not_a_string", "empty"])
def test_load_labels_rejects_invalid_structures(tmp_path, labels):
    write_labels(tmp_path, labels)
    with pytest.raises(InputValidationError):
        load_labels(tmp_path)


def test_load_labels_returns_a_valid_file_unchanged(tmp_path):
    write_labels(tmp_path, LABELS)
    assert load_labels(tmp_path) == LABELS


@pytest.mark.data
@pytest.mark.parametrize("partition,folders", [("train", 80), ("validation", 20)])
def test_supplied_development_labels_are_valid(packet, partition, folders):
    found = sorted((packet / "data" / partition).glob("scenario_*"))
    assert len(found) == folders
    for folder in found:
        validate_labels(load_labels(folder))


# --- Wrong links are classified by what they are ---


def categories(m):
    return m["wrong_skips"], m["wrong_other_aircraft"], m["wrong_invalid"]


def test_forward_same_aircraft_link_is_the_only_skip():
    assert categories(metrics(count_links({"a": "c"}, LABELS))) == (1, 0, 0)


@pytest.mark.parametrize("links", [{"c": "a"}, {"b": "a"}, {"c": "b"}], ids=["two_back", "one_back", "last_to_middle"])
def test_backward_same_aircraft_link_is_wrong_but_not_a_skip(links):
    m = metrics(count_links(links, LABELS))
    assert categories(m) == (0, 0, 1) and m["wrong"] == 1 and m["correct"] == 0


def test_self_link_is_wrong_but_not_a_skip():
    m = metrics(count_links({"a": "a"}, LABELS))
    assert categories(m) == (0, 0, 1) and m["wrong"] == 1 and m["unresolved_tracklets"] == 5


@pytest.mark.parametrize("links,unresolved", [
    ({"a": "ghost"}, 5),  # a labeled tracklet did get a link, though a wrong one
    ({"ghost": "a"}, 6),  # no labeled tracklet got a link
    ({"ghost": "phantom"}, 6),
], ids=["unknown_successor", "unknown_predecessor", "both_unknown"])
def test_link_naming_a_tracklet_outside_the_labels_is_wrong_but_not_another_aircraft(links, unresolved):
    m = metrics(count_links(links, LABELS))
    assert categories(m) == (0, 0, 1) and m["wrong"] == 1 and m["accepted"] == 1 and m["unresolved_tracklets"] == unresolved


def test_unresolved_tracklets_is_never_negative():
    m = metrics(count_links({"g1": "g2", "g3": "g4", "g5": "g6"}, {"a": None}))
    assert m["tracklets"] == 1 and m["accepted"] == 3 and m["wrong"] == 3 and m["unresolved_tracklets"] == 1


def test_wrong_categories_sum_to_wrong_and_survive_adding_scenarios():
    total = LinkCounts()
    total.add(count_links({"a": "c", "x": "b", "c": "a", "ghost": "n", "b": "c"}, LABELS, scenario="one"))
    total.add(count_links({"p": "p"}, {"p": None}, scenario="two"))
    m = metrics(total)
    assert m["wrong_skips"] == 1 and m["wrong_other_aircraft"] == 1 and m["wrong_invalid"] == 3 and m["correct"] == 1
    assert m["wrong"] == m["wrong_skips"] + m["wrong_other_aircraft"] + m["wrong_invalid"] == 5
    assert m["accepted"] == 6 and m["tracklets"] == 7 and m["unresolved_tracklets"] == 2  # n and y; p's self-link and a, b, c, x are links
    assert m["recall"] == pytest.approx(1 / 3) and m["precision"] == pytest.approx(1 / 6)


def test_existing_metric_keys_all_remain():
    expected = {
        "precision", "recall", "unresolved_proportion", "candidate_recall", "candidates_per_tracklet", "wrong", "wrong_skips", "wrong_other_aircraft",
        "accepted", "correct", "labeled_links", "labeled_unresolved", "unresolved_tracklets", "tracklets", "candidate_links_retained",
        "candidate_links_excluded", "candidate_pairs",
    }
    assert expected <= set(metrics(count_links({"a": "b"}, LABELS)))


# --- Selection: what a withheld skip does and doesn't do. The first test pins a behaviour kept deliberately; see select_links ---


GUARDED = AcceptancePolicy(min_score=-40, min_margin=4, plausible_score=-120)


def test_withheld_skip_is_not_a_rival_even_when_the_step_closing_the_chain_is_weak():
    candidates = [scored("a", "c", -1), scored("a", "b", -30), scored("b", "c", -100)]
    links, decisions = select_links(candidates, GUARDED)
    assert links == {"a": "b"}
    assert decisions[("a", "c")].code == POSSIBLE_INTERMEDIATE and decisions[("b", "c")].code == BELOW_THRESHOLD


def test_close_rival_blocks_beside_a_withheld_skip_and_the_skip_stays_withheld():
    candidates = [scored("a", "c", -1), scored("a", "b", -5), scored("a", "d", -6), scored("b", "c", -5)]
    links, decisions = select_links(candidates, GUARDED)
    assert links == {"b": "c"}
    assert decisions[("a", "b")].code == CLOSE_RIVAL and decisions[("a", "d")].code == CLOSE_RIVAL
    assert decisions[("a", "c")].code == POSSIBLE_INTERMEDIATE


def test_competing_predecessors_block_each_other_while_a_clear_one_wins():
    close = [scored("a", "x", -5), scored("b", "x", -6)]
    links, decisions = select_links(close, GUARDED)
    assert links == {} and decisions[("a", "x")].code == CLOSE_RIVAL and decisions[("b", "x")].code == CLOSE_RIVAL
    clear = [scored("a", "x", -1), scored("b", "x", -6)]
    links, decisions = select_links(clear, GUARDED)
    assert links == {"a": "x"} and decisions[("b", "x")].code == SUCCESSOR_TAKEN


def test_global_conflicts_respect_one_predecessor_and_successor_in_score_order():
    candidates = [scored("a", "x", -1), scored("a", "y", -10), scored("b", "x", -8), scored("b", "y", -9), scored("c", "y", -30)]
    links, decisions = select_links(candidates, GUARDED)
    # a -> x leads its rivals by 7 or more. That frees b -> y, which then leads c -> y by 21.
    assert links == {"a": "x", "b": "y"}
    assert decisions[("a", "y")].code == PREDECESSOR_TAKEN and decisions[("b", "x")].code == SUCCESSOR_TAKEN and decisions[("c", "y")].code == SUCCESSOR_TAKEN


def test_every_candidate_gets_exactly_one_decision_whatever_happens_to_it():
    candidates = [
        scored("a", "c", -1), scored("a", "b", -30), scored("b", "c", -100), scored("p", "q", -50), scored("r", "s", -1, acceptable=False),
        ScoredCandidate("t", "u", -1, {}, True, "off the path"),
    ]
    _, decisions = select_links(candidates, GUARDED)
    assert {k: v.code for k, v in decisions.items()} == {
        ("a", "c"): POSSIBLE_INTERMEDIATE, ("a", "b"): ACCEPTED, ("b", "c"): BELOW_THRESHOLD, ("p", "q"): BELOW_THRESHOLD,
        ("r", "s"): MISSING_MEASUREMENT, ("t", "u"): PHYSICALLY_INCONSISTENT,
    }


# --- Noise floors on the baseline's scales ---


def test_floor_applies_only_where_the_gap_line_falls_below_it():
    floors = dataclasses.replace(FLOOR_SCALES, position_floor_km=0.1, altitude_floor_ft=30.0)
    config = BaselineConfig(CandidateConfig(), floors, GUARDED, use_speed_course=False)
    # 5 s: line 0.013 km and 5 ft, so the floors set the scale. Misses of 0.2 km and 60 ft are 2 floors each: 4 + 4.
    short = PairMeasurements(5.0, 1.0, 720.0, 0.2, 0.2, 1.0, 1.0, 60.0, 60.0, 6, 6)
    assert score_candidate(Candidate("a", "b", short), config).score == pytest.approx(-8)
    # 200 s: line 0.403 km and 200 ft, both above the floors. Misses of one line each: 1 + 1.
    long = PairMeasurements(200.0, 26.0, 468.0, 0.403, 0.403, 1.0, 1.0, 200.0, 200.0, 6, 6)
    assert score_candidate(Candidate("a", "b", long), config).score == pytest.approx(-2)


def test_scales_without_floors_score_as_before():
    config = BaselineConfig(CandidateConfig(), FLOOR_SCALES, GUARDED, use_speed_course=False)
    short = PairMeasurements(5.0, 1.0, 720.0, 0.013, 0.013, 1.0, 1.0, 5.0, 5.0, 6, 6)
    assert score_candidate(Candidate("a", "b", short), config).score == pytest.approx(-2)


def test_a_config_saved_before_floors_still_loads_with_none():
    data = config_to_dict(BaselineConfig(CandidateConfig(), FLOOR_SCALES, GUARDED))
    del data["scales"]["position_floor_km"], data["scales"]["altitude_floor_ft"]
    assert config_from_dict(data).scales.position_floor_km == 0


def test_floors_are_mean_residuals_of_tracklets_with_enough_reports():
    line = SimpleNamespace(median_residual_km=0.1)
    motions = [
        SimpleNamespace(report_count=9, overall=SimpleNamespace(median_residual_km=0.1), altitude_median_residual_ft=20.0),
        SimpleNamespace(report_count=9, overall=SimpleNamespace(median_residual_km=0.3), altitude_median_residual_ft=None),
        SimpleNamespace(report_count=9, overall=None, altitude_median_residual_ft=40.0),
        SimpleNamespace(report_count=2, overall=line, altitude_median_residual_ft=1000.0),
    ]
    links = [PairMeasurements(g, 13.0, 468.0, 0.2, 0.2, 1.0, 1.0, 100.0, 100.0, 6, 6) for g in (100.0, 200.0, 300.0)]
    scales = fit_scales(links, motions)
    assert scales.position_floor_km == pytest.approx(0.2) and scales.altitude_floor_ft == pytest.approx(30)
    assert fit_scales(links).position_floor_km == 0 and fit_scales(links).altitude_floor_ft == 0


def test_altitude_residual_is_the_typical_distance_from_the_fitted_line():
    # Alternating 30 ft either side of a level line. By hand: the median pair slope and the intercept both come out level.
    rows = flight("a", 0, count=6, altitude=15000.0)
    for i, row in enumerate(rows):
        row["altitude_ft"] += 30.0 if i % 2 == 0 else -30.0
    assert summarize_tracklet(prepare_scenario(rows).tracklets[0]).altitude_median_residual_ft == pytest.approx(30)
    assert summarize_tracklet(prepare_scenario(flight("b", 0, count=6, climb=5.0)).tracklets[0]).altitude_median_residual_ft == pytest.approx(0, abs=1e-6)
    assert summarize_tracklet(prepare_scenario(flight("c", 0, count=6, altitude=None)).tracklets[0]).altitude_median_residual_ft is None


@pytest.mark.data
@pytest.mark.parametrize("partition,at_least", [("train", 270), ("validation", 85)])
def test_a_tracklet_split_at_5_s_rejoins_under_the_saved_baseline(packet, partition, at_least):
    """Fragments of one tracklet, 5 s of reports apart. Before the floors: 134 of 284 on train, and 36 of 88 with 5 wrong on validation."""
    config = load_config(ROOT / "configs" / "baseline.json")
    rejoined = cut_total = wrong = 0
    for folder in sorted((packet / "data" / partition).glob("scenario_*")):
        scenario, labels, cut = split_tracklets(load_snapshot(folder), load_labels(folder), 5.0)
        result = analyze_baseline(scenario, config)
        counts = count_links(result["links"], labels)
        wrong += counts.wrong
        cut_total += len(cut)
        rejoined += sum(result["links"].get(a) == b for a, b in cut)
    assert rejoined >= at_least and wrong == 0 and cut_total >= at_least


# --- Degraded inputs, through both methods with their saved settings ---


@pytest.fixture(scope="module")
def saved_methods():
    baseline_config = load_config(ROOT / "configs" / "baseline.json")
    model = load_model(ROOT / "models")
    policy = load_policy(ROOT / "configs" / "learned.json", model.version)
    return {"baseline": lambda s: analyze_baseline(s, baseline_config), "learned": lambda s: analyze_learned(s, model, policy)}


METHODS = ("baseline", "learned")


def result_of(saved_methods, method, *flights):
    result = saved_methods[method](prepare_scenario([row for f in flights for row in f]))
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert all(t["status"] == ("accepted" if t["accepted_successor"] else "unresolved") for t in result["tracklets"])
    return result


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("a_altitude,b_altitude", [(None, None), (None, 15000.0), (15000.0, None)])
def test_missing_altitude_on_a_clean_continuation_still_links(saved_methods, method, a_altitude, b_altitude):
    result = result_of(saved_methods, method, flight("a", 0, altitude=a_altitude), flight("b", 200, altitude=b_altitude))
    assert result["links"] == {"a": "b"}
    assert entry(result, "a")["candidates"][0]["measurements"]["altitude_miss_ft"] is None


@pytest.mark.parametrize("method", METHODS)
def test_missing_altitude_cant_separate_aligned_rivals_so_nothing_is_forced(saved_methods, method):
    # Identical horizontal fit, and the only signal that could tell them apart is missing.
    result = result_of(saved_methods, method, flight("a", 0, altitude=None), flight("x", 200, lon=-107.9995, altitude=None), flight("y", 200, lon=-108.0005, altitude=None))
    assert result["links"] == {} and entry(result, "a")["reason_code"] == CLOSE_RIVAL


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("a_count,b_count", [(2, 6), (6, 2), (1, 6), (6, 1), (1, 1)])
def test_too_few_reports_stay_unresolved_as_missing_measurement(saved_methods, method, a_count, b_count):
    result = result_of(saved_methods, method, flight("a", 0, count=a_count), flight("b", 200, count=b_count))
    assert result["links"] == {} and entry(result, "a")["reason_code"] == MISSING_MEASUREMENT


@pytest.mark.parametrize("method", METHODS)
def test_a_lone_single_report_has_no_candidate(saved_methods, method):
    result = result_of(saved_methods, method, flight("a", 0, count=1))
    assert result["links"] == {} and entry(result, "a")["reason_code"] == NO_CANDIDATE


def irregular(tracklet_id, times):
    rows = flight(tracklet_id, 0, count=len(times))
    for row, t, i in zip(rows, times, range(len(times))):
        row.update(observed_at=stamp(t), received_at=stamp(t + 1), latitude=35.0 + SPEED * t, observation_id=f"{tracklet_id}-{i}")
    return rows


@pytest.mark.parametrize("method", METHODS)
def test_irregular_timestamps_dont_disturb_a_clean_continuation(saved_methods, method):
    a, b = irregular("a", [0, 1, 4, 30, 33, 90]), irregular("b", [240, 241, 299, 305, 306, 360])
    result = result_of(saved_methods, method, a, b)
    assert result["links"] == {"a": "b"}
    m = entry(result, "a")["candidates"][0]["measurements"]
    assert m["gap_s"] == 150 and m["forward_miss_km"] == pytest.approx(0, abs=0.01) and m["backward_miss_km"] == pytest.approx(0, abs=0.01)


@pytest.mark.parametrize("method", METHODS)
def test_reports_at_one_instant_are_unknown_motion_not_a_crash(saved_methods, method):
    a = irregular("a", [0, 0, 0, 0])
    result = result_of(saved_methods, method, a, flight("b", 200))
    assert result["links"] == {} and entry(result, "a")["reason_code"] == MISSING_MEASUREMENT
    assert entry(result, "a")["candidates"][0]["measurements"]["forward_miss_km"] is None


@pytest.mark.parametrize("method", METHODS)
def test_nothing_plausible_is_no_candidate_with_the_gates_recorded(saved_methods, method):
    result = result_of(saved_methods, method, flight("a", 0), flight("b", 200, lon=-100.0))
    assert result["links"] == {} and entry(result, "a")["reason_code"] == NO_CANDIDATE
    assert entry(result, "a")["gated_out"] and entry(result, "a")["candidates"] == []


@pytest.mark.parametrize("method", METHODS)
def test_several_plausible_candidates_leave_a_close_rival_with_all_listed(saved_methods, method):
    result = result_of(saved_methods, method, flight("a", 0), flight("x", 200, lon=-107.9995), flight("y", 200, lon=-108.0005), flight("z", 200))
    t = entry(result, "a")
    assert result["links"] == {} and t["reason_code"] == CLOSE_RIVAL and len(t["candidates"]) == 3


def test_learned_refuses_a_physically_inconsistent_only_candidate(saved_methods):
    # Starts on a's projected line at 130 s, then flies 45 degrees off it. Longitude degrees are about 0.82 of latitude's near 35N.
    north, east = SPEED * 0.7071, SPEED * 0.7071 / 0.82
    away = flight("b", 130, north=north, east=east, lat=35.0 + (SPEED - north) * 130, lon=-108.0 - east * 130)
    result = result_of(saved_methods, "learned", flight("a", 0), away)
    assert result["links"] == {} and entry(result, "a")["reason_code"] == PHYSICALLY_INCONSISTENT
