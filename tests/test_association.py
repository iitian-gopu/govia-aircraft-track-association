"""Candidate generation, baseline scoring, selection and metrics.

Expected values are worked out by hand from small synthetic flights, not recomputed with the implementation's formulas.
"""
import inspect
import json
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from govia_tracks import association
from govia_tracks.association import (
    BELOW_THRESHOLD, CLOSE_RIVAL, MISSING_MEASUREMENT, NO_CANDIDATE, POSSIBLE_INTERMEDIATE, SUCCESSOR_TAKEN,
    AcceptancePolicy, ScoredCandidate, build_result, check_links, select_links,
)
from govia_tracks.baseline import (
    BaselineConfig, NoiseScales, analyze, config_from_dict, config_to_dict, config_version, fit_scales, load_config, score_candidate,
)
from govia_tracks.candidates import (
    GATE_MAX_GAP, GATE_MAX_PROJECTION, GATE_MAX_SPEED, Candidate, CandidateConfig, PairMeasurements, generate_candidates, measure_pair,
)
from govia_tracks.errors import InputValidationError
from govia_tracks.evaluation import count_links, labels_without, metrics
from govia_tracks.loader import load_snapshot
from govia_tracks.movement import haversine_km, project_position, summarize_tracklet
from govia_tracks.preprocess import prepare_scenario

from conftest import observation

EPOCH = datetime(2026, 1, 10, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parent.parent
SPEED = 0.0012  # degrees of latitude per second: about 480 km/h
SCALES = NoiseScales(position_km=0.1, position_km_per_s=0.002, altitude_ft=50.0, altitude_ft_per_s=1.0, speed_kmh=5.0, course_deg=1.0)
POLICY = AcceptancePolicy(min_score=-40.0, min_margin=4.0, plausible_score=-120.0)
CONFIG = BaselineConfig(CandidateConfig(), SCALES, POLICY, use_speed_course=False)


def stamp(seconds):
    return (EPOCH + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def flight(tracklet, start_s, count=6, step_s=10, lat=35.0, lon=-108.0, north=SPEED, east=0.0, altitude=15000.0, climb=0.0):
    """Reports on a line defined from time zero, so fragments of one flight line up across gaps."""
    rows = []
    for i in range(count):
        t = start_s + i * step_s
        rows.append(observation(
            observation_id=f"{tracklet}-{i}", tracklet_id=tracklet, observed_at=stamp(t), received_at=stamp(t + 1),
            latitude=lat + north * t, longitude=lon + east * t,
            altitude_ft=None if altitude is None else altitude + climb * t,
        ))
    return rows


def run(*flights, config=CONFIG):
    return analyze(prepare_scenario([row for f in flights for row in f]), config)


def tracklet(result, tracklet_id):
    return next(t for t in result["tracklets"] if t["tracklet_id"] == tracklet_id)


def motion(rows):
    return summarize_tracklet(prepare_scenario(rows).tracklets[0])


def pair(**changes):
    values = dict(gap_s=100.0, endpoint_km=13.0, implied_kmh=468.0, forward_miss_km=0.2, backward_miss_km=0.2, speed_diff_kmh=1.0,
                  course_diff_deg=1.0, altitude_diff_ft=100.0, altitude_miss_ft=100.0, predecessor_reports=6, successor_reports=6)
    return PairMeasurements(**{**values, **changes})


def scored(a, b, score, acceptable=True):
    return ScoredCandidate(a, b, score, {}, acceptable)


# --- Candidate generation ---


def test_valid_forward_pair_is_a_candidate_with_known_measurements():
    a, b = flight("a", 0), flight("b", 200)
    candidates = generate_candidates(prepare_scenario(a + b)).candidates
    assert [(c.predecessor, c.successor) for c in candidates] == [("a", "b")]
    m = candidates[0].measurements
    assert m.gap_s == 150  # a ends at 50 s, b starts at 200 s
    # 150 s at 0.0012 deg/s is 0.18 degrees of latitude: 20.015 km.
    assert m.endpoint_km == pytest.approx(20.015, abs=0.01)
    assert m.implied_kmh == pytest.approx(480.4, abs=0.5)
    assert m.forward_miss_km == pytest.approx(0, abs=1e-6) and m.backward_miss_km == pytest.approx(0, abs=1e-6)
    assert m.altitude_diff_ft == pytest.approx(0, abs=1e-6) and m.altitude_miss_ft == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("b_start,why", [(50, "starts exactly when a ends"), (30, "overlaps a"), (-100, "ends before a starts")])
def test_successor_must_start_strictly_after_predecessor_ends(b_start, why):
    a, b = flight("a", 0), flight("b", b_start)
    assert measure_pair(motion(a), motion(b)) is None, why
    pairs = {(c.predecessor, c.successor) for c in generate_candidates(prepare_scenario(a + b)).candidates}
    assert ("a", "b") not in pairs


def test_a_tracklet_is_never_its_own_candidate():
    assert generate_candidates(prepare_scenario(flight("a", 0))).candidates == ()


def test_scenarios_are_analyzed_separately_so_links_cannot_cross():
    first, second = run(flight("a", 0)), run(flight("b", 200))
    assert first["links"] == {} and second["links"] == {}
    with pytest.raises(ValueError, match="outside this scenario"):
        check_links({"a": "b"}, {"a": motion(flight("a", 0))})


def test_very_large_gap_is_gated_and_recorded():
    candidate_set = generate_candidates(prepare_scenario(flight("a", 0) + flight("b", 700)))
    assert candidate_set.candidates == ()
    assert candidate_set.gated_out["a"][GATE_MAX_GAP] == 1


def test_implausible_speed_is_gated_and_recorded():
    far = flight("b", 110, lat=40.0)  # 5 degrees of latitude in 60 s
    candidate_set = generate_candidates(prepare_scenario(flight("a", 0) + far))
    assert candidate_set.candidates == ()
    assert candidate_set.gated_out["a"][GATE_MAX_SPEED] == 1


def test_large_projection_miss_is_gated_and_recorded():
    sideways = flight("b", 200, lon=-107.8)  # about 18 km off the line, at a believable speed
    candidate_set = generate_candidates(prepare_scenario(flight("a", 0) + sideways))
    assert candidate_set.candidates == ()
    assert candidate_set.gated_out["a"] == {GATE_MAX_PROJECTION: 1}


def test_altitude_never_gates_a_candidate():
    assert len(generate_candidates(prepare_scenario(flight("a", 0) + flight("b", 200, altitude=35000.0))).candidates) == 1


def test_heading_wraparound_is_a_small_difference():
    # Courses just west and just east of north, about 359 and 1 degrees. A naive subtraction would give about 358.
    lean = SPEED * 0.0175 / 0.82  # one degree, allowing for shorter longitude degrees near 35N
    m = measure_pair(motion(flight("a", 0, east=-lean)), motion(flight("b", 200, east=lean)))
    assert m.course_diff_deg == pytest.approx(2, abs=0.2)


def test_noisy_endpoint_does_not_distort_the_measurements():
    a = flight("a", 0, count=9)
    a[-1]["latitude"] += 0.2  # last report about 22 km off
    m = measure_pair(motion(a), motion(flight("b", 200)))
    assert m.forward_miss_km < 0.5


def test_generation_uses_only_the_scenario_given():
    parameters = inspect.signature(generate_candidates).parameters
    assert list(parameters) == ["scenario", "config"]


# --- Missing and unreliable measurements ---


def test_missing_altitude_keeps_the_candidate_and_is_not_agreement():
    known_good = run(flight("a", 0), flight("b", 200))
    missing = run(flight("a", 0), flight("b", 200, altitude=None))
    m = tracklet(missing, "a")["candidates"][0]["measurements"]
    assert m["altitude_diff_ft"] is None and m["altitude_miss_ft"] is None
    assert m["score_terms"]["altitude"] == 1.0  # the configured penalty: neither agreement nor disagreement
    assert tracklet(missing, "a")["candidates"][0]["score"] < tracklet(known_good, "a")["candidates"][0]["score"]
    assert missing["links"] == {"a": "b"}


def test_altitude_at_one_report_each_is_compared_directly_rather_than_projected():
    a, b = flight("a", 0, altitude=None), flight("b", 200, altitude=None)
    a[-1]["altitude_ft"], b[0]["altitude_ft"] = 15000.0, 15300.0
    m = measure_pair(motion(a), motion(b))
    assert m.altitude_diff_ft == 300 and m.altitude_miss_ft == 300


@pytest.mark.parametrize("a_count,b_count", [(2, 6), (6, 2)])
def test_too_few_reports_can_never_be_accepted(a_count, b_count):
    result = run(flight("a", 0, count=a_count), flight("b", 200, count=b_count))
    assert result["links"] == {}
    entry = tracklet(result, "a")
    assert entry["status"] == "unresolved" and entry["reason_code"] == MISSING_MEASUREMENT


def test_single_report_tracklets_have_unknown_motion_not_zero_motion():
    m = measure_pair(motion(flight("a", 0, count=1)), motion(flight("b", 200, count=1)))
    assert m.forward_miss_km is None and m.backward_miss_km is None and m.position_miss_km is None
    assert m.speed_diff_kmh is None and m.course_diff_deg is None
    assert m.endpoint_km > 0  # observed positions still give a distance
    result = run(flight("a", 0, count=1), flight("b", 200, count=1))
    assert tracklet(result, "a")["reason_code"] == MISSING_MEASUREMENT


def test_one_sided_motion_uses_the_side_available():
    m = measure_pair(motion(flight("a", 0, count=1)), motion(flight("b", 200)))
    assert m.forward_miss_km is None and m.backward_miss_km == pytest.approx(0, abs=0.01)
    assert m.position_miss_km == m.backward_miss_km


# --- Score ---


def test_score_value_worked_by_hand():
    # Position: scale 0.1 + 0.002 * 100 = 0.3 km, miss 0.6 km, so 2 squared = 4.
    # Altitude: scale 50 + 1 * 100 = 150 ft, miss 450 ft, so 3 squared = 9. Total -13.
    candidate = Candidate("a", "b", pair(forward_miss_km=0.4, backward_miss_km=0.8, altitude_miss_ft=450.0))
    assert score_candidate(candidate, CONFIG).score == pytest.approx(-13)


def test_perfect_continuation_scores_zero_and_higher_is_better():
    perfect = score_candidate(Candidate("a", "b", pair(forward_miss_km=0.0, backward_miss_km=0.0, altitude_miss_ft=0.0)), CONFIG)
    worse_position = score_candidate(Candidate("a", "b", pair(forward_miss_km=0.5, backward_miss_km=0.5, altitude_miss_ft=0.0)), CONFIG)
    worse_altitude = score_candidate(Candidate("a", "b", pair(forward_miss_km=0.0, backward_miss_km=0.0, altitude_miss_ft=500.0)), CONFIG)
    assert perfect.score == 0
    assert worse_position.score < perfect.score and worse_altitude.score < perfect.score


def test_gap_scaled_noise_allows_more_error_on_a_longer_gap():
    near = score_candidate(Candidate("a", "b", pair(gap_s=100.0, forward_miss_km=0.3, backward_miss_km=0.3)), CONFIG)
    far = score_candidate(Candidate("a", "b", pair(gap_s=400.0, forward_miss_km=0.3, backward_miss_km=0.3)), CONFIG)
    assert far.score > near.score


def test_candidates_are_ranked_by_score_in_the_output():
    result = run(flight("a", 0), flight("b", 200), flight("c", 200, lon=-107.99, north=SPEED))
    candidates = tracklet(result, "a")["candidates"]
    assert [c["rank"] for c in candidates] == [1, 2]
    assert candidates[0]["successor"] == "b" and candidates[0]["score"] > candidates[1]["score"]


def test_scale_fit_recovers_a_known_line_and_refuses_a_negative_floor():
    rising = [pair(gap_s=g, forward_miss_km=0.1 + 0.002 * g, backward_miss_km=0.1 + 0.002 * g, altitude_miss_ft=2.0 * g, speed_diff_kmh=4.0, course_diff_deg=2.0)
              for g in (100.0, 200.0, 300.0)]
    scales = fit_scales(rising)
    assert scales.position_km == pytest.approx(0.1) and scales.position_km_per_s == pytest.approx(0.002)
    assert scales.altitude_ft == pytest.approx(0, abs=1e-9) and scales.altitude_ft_per_s == pytest.approx(2)
    assert scales.speed_kmh == 4 and scales.course_deg == 2
    negative = [pair(gap_s=g, altitude_miss_ft=3.0 * g - 100.0) for g in (100.0, 200.0, 300.0)]
    assert fit_scales(negative).altitude_ft == 0


# --- Selection ---


def test_locally_strongest_candidate_creates_a_conflict_the_stronger_link_wins():
    # a and b both prefer c. a's claim is stronger, so b falls back to d.
    links, decisions = select_links([scored("a", "c", -1), scored("b", "c", -6), scored("b", "d", -7)], AcceptancePolicy(-40, 0, -120))
    assert links == {"a": "c", "b": "d"}
    assert decisions[("b", "c")].code == SUCCESSOR_TAKEN


def test_one_successor_per_predecessor_and_one_predecessor_per_successor():
    candidates = [scored(a, b, -random.Random(a + b).random()) for a in "abc" for b in "xyz"]
    links, _ = select_links(candidates, AcceptancePolicy(-40, 0, -120))
    assert len(links) == 3 and len(set(links.values())) == 3


def test_close_rivals_leave_both_unresolved_rather_than_promoting_one():
    links, decisions = select_links([scored("a", "b", -1), scored("a", "c", -2)], AcceptancePolicy(-40, 4, -120))
    assert links == {}
    assert decisions[("a", "b")].code == CLOSE_RIVAL and decisions[("a", "c")].code == CLOSE_RIVAL


def test_close_rival_for_the_same_successor_also_blocks():
    links, decisions = select_links([scored("a", "c", -1), scored("b", "c", -2)], AcceptancePolicy(-40, 4, -120))
    assert links == {}
    assert decisions[("a", "c")].code == CLOSE_RIVAL


def test_clear_lead_is_accepted():
    links, _ = select_links([scored("a", "b", -1), scored("a", "c", -6)], AcceptancePolicy(-40, 4, -120))
    assert links == {"a": "b"}


def test_best_candidate_below_the_minimum_is_withheld():
    links, decisions = select_links([scored("a", "b", -50)], AcceptancePolicy(-40, 0, -120))
    assert links == {}
    assert decisions[("a", "b")].code == BELOW_THRESHOLD


def test_a_rival_that_cannot_be_accepted_still_blocks():
    # Missing measurements stop the rival being accepted, not it being a real alternative.
    links, decisions = select_links([scored("a", "b", -1), scored("a", "c", -2, acceptable=False)], AcceptancePolicy(-40, 4, -120))
    assert links == {} and decisions[("a", "b")].code == CLOSE_RIVAL


def test_exact_ties_resolve_the_same_way_regardless_of_input_order():
    candidates = [scored("b", "x", -1), scored("a", "x", -1), scored("a", "y", -1)]
    policy = AcceptancePolicy(-40, 0, -120)
    expected = select_links(candidates, policy)[0]
    assert expected == {"a": "x"}  # "a" sorts first; b then finds x taken
    for seed in range(5):
        shuffled = list(candidates)
        random.Random(seed).shuffle(shuffled)
        assert select_links(shuffled, policy)[0] == expected


def test_a_tie_with_a_required_margin_accepts_neither():
    links, _ = select_links([scored("a", "x", -1), scored("b", "x", -1)], AcceptancePolicy(-40, 0.5, -120))
    assert links == {}


def test_skip_is_withheld_even_when_it_scores_best():
    candidates = [scored("a", "c", -1), scored("a", "b", -5), scored("b", "c", -5)]
    links, decisions = select_links(candidates, AcceptancePolicy(-40, 0, -120))
    assert links == {"a": "b", "b": "c"}
    assert decisions[("a", "c")].code == POSSIBLE_INTERMEDIATE


def test_implausible_intermediate_does_not_block_a_direct_link():
    candidates = [scored("a", "c", -1), scored("a", "b", -200), scored("b", "c", -5)]
    links, _ = select_links(candidates, AcceptancePolicy(-40, 0, -120))
    assert links == {"a": "c"}


def test_a_withheld_intermediate_still_stops_the_skip():
    # a -> b is held back by a close rival. That must not let a -> c through.
    candidates = [scored("a", "c", -1), scored("a", "b", -5), scored("z", "b", -5.1), scored("b", "c", -5)]
    links, _ = select_links(candidates, AcceptancePolicy(-40, 4, -120))
    assert "a" not in links


def test_physical_chain_links_each_fragment_to_the_next_not_the_one_after():
    result = run(flight("a", 0), flight("b", 200), flight("c", 400))
    assert result["links"] == {"a": "b", "b": "c"}
    skip = next(c for c in tracklet(result, "a")["candidates"] if c["successor"] == "c")
    assert skip["decision"] == POSSIBLE_INTERMEDIATE


def test_altitude_separates_otherwise_identical_rivals():
    tied = run(flight("a", 0), flight("x", 200, lon=-107.999), flight("y", 200, lon=-108.001))
    assert tied["links"] == {} and tracklet(tied, "a")["reason_code"] == CLOSE_RIVAL
    separated = run(flight("a", 0), flight("x", 200, lon=-107.999), flight("y", 200, lon=-108.001, altitude=24000.0))
    assert separated["links"] == {"a": "x"}


def test_scenario_with_nothing_plausible_is_completely_unresolved():
    result = run(flight("a", 0), flight("b", 200, lon=-100.0), flight("c", 400, lat=45.0))
    assert result["links"] == {}
    assert {t["status"] for t in result["tracklets"]} == {"unresolved"}
    assert tracklet(result, "a")["reason_code"] == NO_CANDIDATE
    assert metrics(count_links(result["links"], {"a": None, "b": None, "c": None}))["precision"] is None


def test_last_tracklet_has_no_candidate_and_is_unresolved():
    result = run(flight("a", 0), flight("b", 200))
    assert tracklet(result, "b")["status"] == "unresolved" and tracklet(result, "b")["reason_code"] == NO_CANDIDATE


# --- Invariant checker, which runs on every result ---


def test_check_links_rejects_each_violation():
    motions = {"a": motion(flight("a", 0)), "b": motion(flight("b", 200)), "c": motion(flight("c", 400))}
    check_links({"a": "b", "b": "c"}, motions)
    with pytest.raises(ValueError, match="predecessor"):
        check_links({"a": "c", "b": "c"}, motions)
    with pytest.raises(ValueError, match="self-link"):
        check_links({"a": "a"}, motions)
    with pytest.raises(ValueError, match="forward in time"):
        check_links({"b": "a"}, motions)
    with pytest.raises(ValueError, match="forward in time|cycle"):
        check_links({"a": "b", "b": "a"}, motions)


def test_cycle_is_rejected_even_when_time_order_is_ignored():
    overlapping = {name: motion(flight(name, 0)) for name in "abc"}
    for name, m in overlapping.items():
        object.__setattr__(m, "start_us", 0)
        object.__setattr__(m, "end_us", -1)
    with pytest.raises(ValueError, match="cycle"):
        check_links({"a": "b", "b": "c", "c": "a"}, overlapping)


def test_a_bad_selection_cannot_reach_a_result(monkeypatch):
    monkeypatch.setattr(association, "select_links", lambda *_: ({"b": "a"}, {}))
    scenario = prepare_scenario(flight("a", 0) + flight("b", 200))
    with pytest.raises(ValueError, match="forward in time"):
        build_result("x", "1", {}, scenario, generate_candidates(scenario), [], POLICY)


# --- Determinism and output contract ---


def test_input_order_and_duplicates_do_not_change_the_result():
    rows = flight("a", 0) + flight("b", 200) + flight("c", 200, lon=-107.99) + flight("d", 400)
    expected = json.dumps(analyze(prepare_scenario(rows), CONFIG), sort_keys=True)
    for seed in range(5):
        rng = random.Random(seed)
        noisy = rows + rng.sample(rows, 6)
        rng.shuffle(noisy)
        assert json.dumps(analyze(prepare_scenario(noisy), CONFIG), sort_keys=True) == expected


def test_repeated_runs_are_identical():
    rows = flight("a", 0) + flight("b", 200)
    assert analyze(prepare_scenario(rows), CONFIG) == analyze(prepare_scenario(rows), CONFIG)


def test_result_contract_and_json_round_trip():
    scenario = prepare_scenario(flight("a", 0) + flight("b", 200, altitude=None), scenario_id="scn", snapshot_id="snap")
    result = analyze(scenario, CONFIG)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["method"] == "baseline" and result["version"] == config_version(CONFIG)
    assert result["scenario_id"] == "scn" and result["snapshot_id"] == "snap"
    assert result["input_fingerprint"] == scenario.input_fingerprint
    assert result["policy"] == {"min_score": -40.0, "min_margin": 4.0, "plausible_score": -120.0}
    entry = tracklet(result, "a")
    assert set(entry) == {"tracklet_id", "status", "accepted_successor", "reason_code", "reason", "candidates", "gated_out"}
    assert entry["status"] == "accepted" and entry["accepted_successor"] == "b" and "b" in entry["reason"]
    assert set(entry["candidates"][0]) == {"rank", "successor", "score", "decision", "decision_detail", "measurements"}
    assert entry["candidates"][0]["measurements"]["gap_s"] == 150


def test_any_method_can_use_the_same_result_builder():
    scenario = prepare_scenario(flight("a", 0) + flight("b", 200))
    candidate_set = generate_candidates(scenario)
    other_scores = [ScoredCandidate(c.predecessor, c.successor, 0.9, {"model": "x"}) for c in candidate_set.candidates]
    result = build_result("learned", "v1", {"k": 1}, scenario, candidate_set, other_scores, AcceptancePolicy(0.5, 0.1, 0.2))
    assert result["links"] == {"a": "b"} and result["method"] == "learned"


def test_config_round_trip_and_version_tracks_every_setting():
    assert config_from_dict(json.loads(json.dumps(config_to_dict(CONFIG)))) == CONFIG
    changed = BaselineConfig(CandidateConfig(), SCALES, AcceptancePolicy(-41.0, 4.0, -120.0), use_speed_course=False)
    assert config_version(changed) != config_version(CONFIG)


# --- Labels stay out ---


def test_inference_modules_do_not_load_labels_or_evaluation():
    code = (
        "import sys, govia_tracks.baseline, govia_tracks.candidates, govia_tracks.association; "
        "sys.exit(1 if {'govia_tracks.labels', 'govia_tracks.evaluation'} & set(sys.modules) else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_analyze_takes_no_labels_and_needs_no_labels_file():
    assert list(inspect.signature(analyze).parameters) == ["scenario", "config"]


# --- Metrics ---


LABELS = {"a": "b", "b": "c", "c": None, "x": "y", "y": None, "n": None}


def test_correct_links_count_and_recall_denominator_excludes_null_successors():
    m = metrics(count_links({"a": "b", "b": "c", "x": "y"}, LABELS))
    assert m["correct"] == 3 and m["wrong"] == 0 and m["labeled_links"] == 3
    assert m["precision"] == 1 and m["recall"] == 1 and m["unresolved_proportion"] == 0
    assert m["tracklets"] == 6 and m["unresolved_tracklets"] == 3


def test_skipping_an_intermediate_is_wrong_and_not_recall():
    counts = count_links({"a": "c"}, LABELS)
    m = metrics(counts)
    assert m["correct"] == 0 and m["wrong_skips"] == 1 and m["wrong_other_aircraft"] == 0
    assert m["precision"] == 0 and m["recall"] == 0
    assert m["labeled_unresolved"] == 2  # a has a wrong link, so it is not "unresolved"; b and x are


def test_link_between_aircraft_counts_against_precision_and_recall():
    m = metrics(count_links({"a": "y"}, LABELS))
    assert m["wrong_other_aircraft"] == 1 and m["precision"] == 0 and m["recall"] == 0


def test_link_from_a_tracklet_with_no_successor_is_wrong_and_stays_out_of_the_denominator():
    m = metrics(count_links({"n": "b", "a": "b"}, LABELS))
    assert m["accepted"] == 2 and m["correct"] == 1 and m["wrong"] == 1
    assert m["labeled_links"] == 3 and m["recall"] == pytest.approx(1 / 3) and m["precision"] == 0.5


def test_candidate_recall_denominator_includes_links_excluded_by_gates():
    counts = count_links({"a": "b"}, LABELS, {("a", "b"), ("a", "c"), ("x", "n")}, "scn")
    m = metrics(counts)
    assert m["candidate_links_retained"] == 1 and m["candidate_links_excluded"] == 2
    assert m["candidate_recall"] == pytest.approx(1 / 3) and m["candidate_pairs"] == 3
    assert counts.excluded_links == [("scn", "b", "c"), ("scn", "x", "y")]
    assert m["recall"] == pytest.approx(1 / 3)  # lost links still count in the denominator


def test_a_true_successor_excluded_by_a_gate_is_missed_end_to_end():
    rows = flight("a", 0) + flight("b", 700)  # same line, but past the gap gate
    result = analyze(prepare_scenario(rows), CONFIG)
    pairs = {(t["tracklet_id"], c["successor"]) for t in result["tracklets"] for c in t["candidates"]}
    m = metrics(count_links(result["links"], {"a": "b", "b": None}, pairs))
    assert m["candidate_recall"] == 0 and m["recall"] == 0 and m["labeled_unresolved"] == 1 and m["candidate_links_excluded"] == 1
    assert tracklet(result, "a")["gated_out"] == {GATE_MAX_GAP: 1}


def test_zero_denominators_are_undefined_not_perfect():
    m = metrics(count_links({}, {"a": None}, set()))
    assert m["precision"] is None and m["recall"] is None and m["candidate_recall"] is None and m["unresolved_proportion"] is None
    assert m["accepted"] == 0 and m["labeled_links"] == 0


def test_labels_without_rebuilds_the_next_available_successor():
    labels = {"a": "b", "b": "c", "c": "d", "d": None}
    assert labels_without(labels, {"b", "c"}) == {"a": "d", "d": None}
    assert labels_without(labels, {"d"}) == {"a": "b", "b": "c", "c": None}


# --- Movement additions and the manifest fix ---


def test_altitude_rate_is_recovered_and_ignores_missing_reports():
    rows = flight("a", 0, count=9, step_s=10, climb=5.0, altitude=10000.0)
    rows[3]["altitude_ft"] = None
    m = motion(rows)
    assert m.altitude_rate_ft_s == pytest.approx(5)
    assert m.fitted_start_altitude_ft == pytest.approx(10000) and m.fitted_end_altitude_ft == pytest.approx(10400)


def test_altitude_projection_removes_a_steady_climb_from_the_miss():
    m = measure_pair(motion(flight("a", 0, climb=5.0)), motion(flight("b", 200, climb=5.0)))
    assert m.altitude_diff_ft == pytest.approx(750)  # 150 s at 5 ft/s
    assert m.altitude_miss_ft == pytest.approx(0, abs=1e-6)


def test_project_position_moves_the_expected_distance():
    lat, lon = project_position(35.0, -108.0, 0.0, 0.1, 100.0)  # 10 km north
    assert haversine_km(35.0, -108.0, lat, lon) == pytest.approx(10, abs=0.01) and lon == pytest.approx(-108)
    assert project_position(0.0, 179.99, 1.0, 0.0, 10.0)[1] < 0  # wraps at the antimeridian


def test_malformed_manifest_is_a_validation_error_not_a_key_error(tmp_path):
    (tmp_path / "snapshot.json").write_text(json.dumps({"snapshot_id": "s", "files": {"observations": "o.jsonl"}}), encoding="utf-8")
    with pytest.raises(InputValidationError, match="scenario_id") as caught:
        load_snapshot(tmp_path)
    assert {i.field for i in caught.value.issues} >= {"scenario_id", "files.supporting", "tracklet_count"}


# --- Real data ---


def _all_results(packet):
    config = load_config(ROOT / "configs" / "baseline.json")
    for partition in ("train", "validation"):
        for folder in sorted((packet / "data" / partition).glob("scenario_*")):
            yield partition, folder, load_snapshot(folder), config


@pytest.mark.data
def test_invariants_hold_on_every_development_scenario(packet):
    """Checked here independently of check_links, from the tracklets themselves."""
    scenarios = 0
    for _, _, scenario, config in _all_results(packet):
        result = analyze(scenario, config)
        times = {t.tracklet_id: (t.observations[0].observed_us, t.observations[-1].observed_us) for t in scenario.tracklets}
        links = result["links"]
        assert len(set(links.values())) == len(links)
        for a, b in links.items():
            assert a != b and a in times and b in times and times[b][0] > times[a][1]
            assert tracklet(result, a)["accepted_successor"] == b
        for entry in result["tracklets"]:
            assert (entry["status"] == "accepted") == (entry["accepted_successor"] is not None)
            scores = [c["score"] for c in entry["candidates"]]
            assert scores == sorted(scores, reverse=True)
            assert all(times[c["successor"]][0] > times[entry["tracklet_id"]][1] for c in entry["candidates"])
        scenarios += 1
    assert scenarios == 100


@pytest.mark.data
def test_saved_results_and_metrics_reproduce(packet):
    report = json.loads((ROOT / "reports" / "baseline_metrics.json").read_text(encoding="utf-8"))
    config = load_config(ROOT / "configs" / "baseline.json")
    assert report["version"] == config_version(config)
    saved = json.loads((ROOT / "reports" / "baseline_results_validation.json").read_text(encoding="utf-8"))
    rerun = [analyze(load_snapshot(folder), config) for folder in sorted((packet / "data" / "validation").glob("scenario_*"))]
    assert rerun == saved
