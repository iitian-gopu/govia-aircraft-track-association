"""Learned scorer: training rows, label isolation, schema, artifacts, and behaviour through the shared selection.

Everything here runs on synthetic flights. Scores are never asserted as exact values because they belong to the fitted
model; policies are set relative to the scores the model actually gives, so they exercise thresholds at its own scale.
"""
import ast
import dataclasses
import hashlib
import inspect
import json
import math
import random
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from govia_tracks import learned, training
from govia_tracks.association import (
    BELOW_THRESHOLD, CLOSE_RIVAL, MISSING_MEASUREMENT, NO_CANDIDATE, PHYSICALLY_INCONSISTENT, POSSIBLE_INTERMEDIATE, AcceptancePolicy, ScoredCandidate,
    select_links,
)
from govia_tracks.candidates import Candidate, CandidateConfig, PairMeasurements, generate_candidates
from govia_tracks.evaluation import (
    aircraft_chains, count_links, hide_altitude, hide_altitude_of_some, labels_without, metrics, noisy_altitude, outside_visible_windows,
    random_fragments, split_tracklets, trim_reports, without_tracklets,
)
from govia_tracks.learned import (
    ALTITUDE_FEATURES, FEATURES, MAX_COURSE_DIFF_DEG, MAX_DIRECTIONAL_MISS_KM, METADATA_FILE, MODEL_FILE, analyze, feature_matrix, feature_row, load_model,
    load_policy, physical_objection, score_candidates,
)
from govia_tracks.preprocess import prepare_scenario
from govia_tracks.training import build_model, fit_estimator, save_model, training_examples

from conftest import observation

EPOCH = datetime(2026, 1, 10, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parent.parent
SPEED = 0.0012  # degrees of latitude per second: about 480 km/h
KM_PER_DEG_LAT = 111.19
# Enough trees to learn from a small set, few enough to stay fast.
FIT_PARAMETERS = {"max_depth": 3, "max_iter": 60, "learning_rate": 0.1, "min_samples_leaf": 5}
LENIENT = AcceptancePolicy(min_score=0.2, min_margin=0.05, plausible_score=0.05)
ANYTHING = AcceptancePolicy(min_score=0.0, min_margin=0.0, plausible_score=0.0)  # every candidate clears, every step is plausible


def stamp(seconds):
    return (EPOCH + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def flight(tracklet_id, start_s, count=6, step_s=10, lat=35.0, lon=-108.0, north=SPEED, east=0.0, altitude=15000.0, climb=0.0):
    """Reports on a line defined from time zero, so fragments of one flight line up across gaps."""
    rows = []
    for i in range(count):
        t = start_s + i * step_s
        rows.append(observation(
            observation_id=f"{tracklet_id}-{i}", tracklet_id=tracklet_id, observed_at=stamp(t), received_at=stamp(t + 1),
            latitude=lat + north * t, longitude=lon + east * t,
            altitude_ft=None if altitude is None else altitude + climb * t,
        ))
    return rows


def scenario_of(*flights, **ids):
    return prepare_scenario([row for f in flights for row in f], **ids)


def tracklet(result, tracklet_id):
    return next(t for t in result["tracklets"] if t["tracklet_id"] == tracklet_id)


def candidate_entry(result, a, b):
    return next(c for c in tracklet(result, a)["candidates"] if c["successor"] == b)


def candidate_pairs(result):
    return {(t["tracklet_id"], c["successor"]) for t in result["tracklets"] for c in t["candidates"]}


def pair(**changes):
    values = dict(gap_s=100.0, endpoint_km=13.0, implied_kmh=468.0, forward_miss_km=0.2, backward_miss_km=0.2, speed_diff_kmh=1.0,
                  course_diff_deg=1.0, altitude_diff_ft=100.0, altitude_miss_ft=100.0, predecessor_reports=6, successor_reports=6)
    return PairMeasurements(**{**values, **changes})


def to_json(result):
    return json.dumps(result, sort_keys=True, allow_nan=False)


# --- A synthetic world to fit on: several aircraft close together, each split into fragments ---


def synthetic_scenario(seed, aircraft=5, fragments=3):
    """Returns (rows, labels). Aircraft share a small box and nearly a heading, so other aircraft reach the candidate list.

    Every second aircraft flies the previous one's line at another altitude. Without these, position alone separates
    every pair and the model has no reason to learn anything about altitude.
    """
    rng = random.Random(seed)
    rows, labels = [], {}
    levels = (15000.0, 18000.0, 24000.0, 31000.0)
    for plane in range(aircraft):
        if plane % 2:
            altitude = rng.choice([level for level in levels if level != altitude])
        else:
            heading = math.radians(rng.uniform(-4, 4))
            speed_km_s = rng.uniform(450, 510) / 3600
            north = speed_km_s * math.cos(heading) / KM_PER_DEG_LAT
            east = speed_km_s * math.sin(heading) / (KM_PER_DEG_LAT * math.cos(math.radians(35)))
            lat, lon = 35.0 + rng.uniform(-0.03, 0.03), -108.0 + rng.uniform(-0.03, 0.03)
            altitude = rng.choice(levels)
        start = rng.randint(0, 100)
        names = [f"s{seed}-p{plane}-f{fragment}" for fragment in range(fragments)]
        for fragment, name in enumerate(names):
            for row in flight(name, start, lat=lat, lon=lon, north=north, east=east, altitude=altitude):
                row["latitude"] += rng.gauss(0, 0.0002)  # about 20 m
                row["longitude"] += rng.gauss(0, 0.0002)
                row["altitude_ft"] += rng.gauss(0, 30)
                rows.append(row)
            labels[name] = names[fragment + 1] if fragment + 1 < fragments else None
            start += 50 + rng.randint(100, 200)  # the fragment's duration, then a gap
    return rows, labels


def synthetic_partitions(seeds):
    for seed in seeds:
        rows, labels = synthetic_scenario(seed)
        yield f"scenario_{seed}", prepare_scenario(rows), labels


@pytest.fixture(scope="module")
def examples():
    return training_examples(synthetic_partitions(range(25)))


@pytest.fixture(scope="module")
def model(examples):
    return build_model(examples, FIT_PARAMETERS)


@pytest.fixture(scope="module")
def saved_folder(model, tmp_path_factory):
    folder = tmp_path_factory.mktemp("model")
    save_model(model, folder)
    return folder


def test_the_synthetic_training_set_has_every_kind_of_row(examples):
    assert examples.positives > 100 and examples.negatives_same_aircraft > 50 and examples.negatives_other_aircraft > 100


# --- 1. Training rows ---


A_B_C_X = (flight("a", 0), flight("b", 200), flight("c", 400), flight("x", 200, lon=-107.99))
# A forward pair needs the successor to start after the predecessor ends. a ends at 50, b and x run 200-250, c starts at 400.
# So: a->b, a->x, a->c, b->c, x->c. x is about 0.9 km off the line, well inside the projection gate.
CHAIN_LABELS = {"a": "b", "b": "c", "c": None, "x": None}


def chain_partition(labels=CHAIN_LABELS, name="s1"):
    return name, scenario_of(*A_B_C_X), labels


def targets_by_pair(built, scenario):
    return {(c.predecessor, c.successor): int(t) for c, t in zip(generate_candidates(scenario).candidates, built.targets, strict=True)}


def test_training_rows_have_exact_counts_and_targets():
    name, scenario, labels = chain_partition()
    built = training_examples([(name, scenario, labels)])
    assert built.features.shape == (5, len(FEATURES)) and built.targets.shape == (5,)
    assert targets_by_pair(built, scenario) == {("a", "b"): 1, ("b", "c"): 1, ("a", "c"): 0, ("a", "x"): 0, ("x", "c"): 0}
    assert built.positives == 2 and built.labeled_links == 2
    assert built.negatives_same_aircraft == 1  # the skip a->c
    assert built.negatives_other_aircraft == 2  # a->x and x->c
    assert built.positives + built.negatives_same_aircraft + built.negatives_other_aircraft == len(built.targets)


def test_each_row_holds_the_measurements_of_its_own_pair():
    name, scenario, labels = chain_partition()
    built = training_examples([(name, scenario, labels)])
    expected = [feature_row(c.measurements) for c in generate_candidates(scenario).candidates]
    assert np.array_equal(built.features, np.array(expected), equal_nan=True)


def test_skip_is_a_negative_even_though_its_motion_fits_perfectly():
    built = training_examples([("s", scenario_of(flight("a", 0), flight("b", 200), flight("c", 400)), {"a": "b", "b": "c", "c": None})])
    assert targets_by_pair(built, scenario_of(flight("a", 0), flight("b", 200), flight("c", 400)))[("a", "c")] == 0
    assert built.negatives_same_aircraft == 1 and built.negatives_other_aircraft == 0


def test_a_true_link_lost_by_the_gates_is_labeled_but_makes_no_row():
    far = flight("b", 200, lon=-107.8)  # about 18 km off the line, past the projection gate
    built = training_examples([("s", scenario_of(flight("a", 0), far), {"a": "b", "b": None})])
    assert built.features.shape == (0, len(FEATURES))
    assert built.positives == 0 and built.labeled_links == 1


# --- 2. No leakage between scenarios or from labels into rows ---


def test_identical_tracklet_ids_in_two_scenarios_stay_separate():
    # Same IDs, opposite truth: linked in the first scenario, different aircraft in the second.
    first = ("one", scenario_of(flight("a", 0), flight("b", 200)), {"a": "b", "b": None})
    second = ("two", scenario_of(flight("a", 0), flight("b", 200)), {"a": None, "b": None})
    built = training_examples([first, second])
    assert built.targets.tolist() == [1, 0] and built.groups.tolist() == [0, 1] and built.scenarios == ("one", "two")
    assert built.positives == 1 and built.negatives_other_aircraft == 1


def test_compatible_tracklets_in_different_scenarios_never_pair():
    built = training_examples([("one", scenario_of(flight("a", 0)), {"a": None}), ("two", scenario_of(flight("b", 200)), {"b": None})])
    assert len(built.targets) == 0 and built.scenarios == ("one", "two")


def test_groups_map_each_row_to_its_scenario():
    built = training_examples([chain_partition(name="big"), ("small", scenario_of(flight("a", 0), flight("b", 200)), {"a": "b", "b": None})])
    assert built.groups.tolist() == [0, 0, 0, 0, 0, 1]  # five rows from the first, one from the second
    assert built.scenarios == ("big", "small")


def test_labels_change_targets_but_never_the_rows():
    truth = training_examples([chain_partition()])
    # a's successor is now x. So a and x are one aircraft, b and c another, and a->b, a->c and x->c cross between them.
    other = training_examples([chain_partition({"a": "x", "x": None, "b": "c", "c": None})])
    nothing = training_examples([chain_partition({"a": None, "b": None, "c": None, "x": None})])
    assert truth.features.tobytes() == other.features.tobytes() == nothing.features.tobytes()
    assert (other.positives, other.negatives_same_aircraft, other.negatives_other_aircraft) == (2, 0, 3)
    assert (nothing.positives, nothing.negatives_same_aircraft, nothing.negatives_other_aircraft, nothing.labeled_links) == (0, 0, 5, 0)


# --- 3. Labels can't reach inference ---


def loaded_modules(module):
    code = f"import json, sys, {module}; print(json.dumps(sorted(m for m in sys.modules if m.startswith('govia_tracks'))))"
    done = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, cwd=ROOT)
    return set(json.loads(done.stdout))


def test_importing_inference_does_not_load_labels_evaluation_or_training():
    modules = loaded_modules("govia_tracks.learned")
    assert "govia_tracks.learned" in modules  # the check is looking at the right thing
    assert not modules & {"govia_tracks.labels", "govia_tracks.evaluation", "govia_tracks.training"}


def test_importing_training_does_not_load_labels():
    modules = loaded_modules("govia_tracks.training")
    assert "govia_tracks.training" in modules and "govia_tracks.labels" not in modules


def test_inference_takes_no_labels():
    assert list(inspect.signature(analyze).parameters) == ["scenario", "model", "policy"]
    assert list(inspect.signature(score_candidates).parameters) == ["model", "candidates"]


# --- 4. Fixed schema ---


def edit_metadata(folder, change):
    path = folder / METADATA_FILE
    metadata = json.loads(path.read_text(encoding="utf-8"))
    change(metadata)
    path.write_text(json.dumps(metadata), encoding="utf-8")


def swapped(names):
    return [names[1], names[0], *names[2:]]


@pytest.mark.parametrize("names,why", [
    (swapped(list(FEATURES)), "reordered"),
    (list(FEATURES)[::-1], "reversed"),
    (list(FEATURES)[:-1], "one dropped"),
    ([*FEATURES, "extra_feature"], "one added"),
    (["renamed_feature", *list(FEATURES)[1:]], "one renamed"),
])
def test_load_refuses_a_different_feature_schema(saved_folder, tmp_path, names, why):
    copy = tmp_path / "copy"
    shutil.copytree(saved_folder, copy)
    edit_metadata(copy, lambda metadata: metadata.update(feature_names=names))
    with pytest.raises(ValueError, match="schema"):
        load_model(copy)


def test_load_refuses_a_model_trained_on_a_different_feature_order(examples, tmp_path):
    reversed_names = tuple(reversed(FEATURES))
    reordered = dataclasses.replace(examples, features=examples.features[:, ::-1])
    save_model(build_model(reordered, FIT_PARAMETERS, reversed_names), tmp_path)
    with pytest.raises(ValueError, match="schema"):
        load_model(tmp_path)


def test_load_refuses_changed_artifact_bytes_without_unpickling_them(saved_folder, tmp_path, monkeypatch):
    copy = tmp_path / "copy"
    shutil.copytree(saved_folder, copy)
    assert load_model(copy).version  # intact, it loads
    with (copy / MODEL_FILE).open("ab") as artifact:
        artifact.write(b"\0")

    def refuse(*_):
        raise AssertionError("a changed artifact must not be unpickled")
    monkeypatch.setattr(learned.joblib, "load", refuse)
    with pytest.raises(ValueError, match="hash"):
        load_model(copy)


# --- 5. Missing and non-finite measurements ---


def test_feature_row_follows_the_requested_order():
    m = pair()
    assert feature_row(m, tuple(reversed(FEATURES))) == feature_row(m)[::-1]
    assert feature_row(m, ("gap_s", "endpoint_km")) == [100.0, 13.0]


def test_missing_altitude_is_nan_not_zero():
    row = feature_row(pair(altitude_diff_ft=None, altitude_miss_ft=None))
    for name, value in zip(FEATURES, row, strict=True):
        assert math.isnan(value) == (name in ALTITUDE_FEATURES), name
    assert ALTITUDE_FEATURES <= set(FEATURES)


def test_every_unmeasured_value_is_nan():
    assert math.isnan(feature_row(pair(forward_miss_km=None), ("forward_miss_km",))[0])
    assert feature_row(pair(forward_miss_km=0.0), ("forward_miss_km",)) == [0.0]  # a true zero stays a zero


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_non_finite_measurements_are_errors(bad):
    fields = {f.name for f in dataclasses.fields(PairMeasurements)}
    measured = fields & set(FEATURES)
    assert measured
    for name in measured:
        with pytest.raises(ValueError, match="non-finite"):
            feature_row(pair(**{name: bad}))
        with pytest.raises(ValueError, match=name):  # the message names the measurement when it is the only one asked for
            feature_row(pair(**{name: bad}), (name,))
    with pytest.raises(ValueError, match="non-finite"):
        feature_row(pair(forward_miss_km=bad), ("position_miss_km",))  # derived from it


def test_feature_matrix_shape_holds_with_no_candidates():
    assert feature_matrix([], FEATURES).shape == (0, len(FEATURES))


@pytest.mark.parametrize("a_altitude,b_altitude", [(15000.0, None), (None, 15000.0), (None, None)])
def test_missing_altitude_scores_finite_and_in_range(model, a_altitude, b_altitude):
    result = analyze(scenario_of(flight("a", 0, altitude=a_altitude), flight("b", 200, altitude=b_altitude)), model, LENIENT)
    entry = candidate_entry(result, "a", "b")
    assert entry["measurements"]["altitude_miss_ft"] is None and entry["measurements"]["altitude_diff_ft"] is None
    assert math.isfinite(entry["score"]) and 0 <= entry["score"] <= 1


def test_missing_altitude_still_separates_clean_from_off_path(model):
    rows = flight("a", 0, altitude=None), flight("b", 200, altitude=None), flight("c", 200, lon=-107.97, altitude=None)
    result = analyze(scenario_of(*rows), model, LENIENT)
    assert candidate_entry(result, "a", "b")["score"] > candidate_entry(result, "a", "c")["score"]


def test_missing_altitude_is_not_read_as_a_conflict(model):
    # A 15,000 ft difference with no climb is another aircraft. Unknown altitude should not look like that.
    conflict = analyze(scenario_of(flight("a", 0), flight("b", 200, altitude=30000.0)), model, LENIENT)
    unknown = analyze(scenario_of(flight("a", 0), flight("b", 200, altitude=None)), model, LENIENT)
    assert candidate_entry(unknown, "a", "b")["score"] > candidate_entry(conflict, "a", "b")["score"]


def test_clean_chain_still_links_when_altitude_is_hidden_everywhere(model):
    scenario = hide_altitude(scenario_of(flight("a", 0), flight("b", 200), flight("c", 400)))
    assert analyze(scenario, model, LENIENT)["links"] == {"a": "b", "b": "c"}


@pytest.mark.parametrize("a_count,b_count", [(1, 6), (6, 1), (1, 1), (2, 6), (6, 2)])
def test_too_few_reports_are_never_accepted_even_under_the_loosest_policy(model, a_count, b_count):
    result = analyze(scenario_of(flight("a", 0, count=a_count), flight("b", 200, count=b_count)), model, ANYTHING)
    assert result["links"] == {} and tracklet(result, "a")["reason_code"] == MISSING_MEASUREMENT
    score = candidate_entry(result, "a", "b")["score"]
    assert math.isfinite(score) and 0 <= score <= 1


# --- 6. Altitude-hidden copies ---


class RecordingEstimator:
    """Stands in for the classifier to show exactly what it would be fitted on."""

    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, features, targets, sample_weight=None):
        self.features, self.targets, self.weights = features.copy(), targets.copy(), sample_weight.copy()
        return self


@pytest.fixture
def recording(monkeypatch):
    monkeypatch.setattr(training, "HistGradientBoostingClassifier", RecordingEstimator)


def small_inputs(columns):
    return np.random.default_rng(0).uniform(1, 2, (6, columns)), np.array([1, 0, 0, 1, 0, 1])


@pytest.mark.parametrize("names", [FEATURES, tuple(reversed(FEATURES))], ids=["usual", "reversed"])
def test_hidden_copy_doubles_rows_and_hides_exactly_the_altitude_columns(recording, names):
    features, targets = small_inputs(len(names))
    fitted = fit_estimator(features, targets, feature_names=names)
    assert fitted.features.shape == (12, len(names))
    assert np.array_equal(fitted.features[:6], features)  # originals kept as they were
    assert fitted.targets.tolist() == targets.tolist() * 2  # hiding altitude doesn't change the truth
    hidden = fitted.features[6:]
    assert ALTITUDE_FEATURES & set(names)
    for index, name in enumerate(names):
        if name in ALTITUDE_FEATURES:
            assert np.isnan(hidden[:, index]).all(), name
        else:
            assert np.array_equal(hidden[:, index], features[:, index]), name


def test_hidden_copy_counts_for_half_and_originals_for_one(recording):
    features, targets = small_inputs(len(FEATURES))
    fitted = fit_estimator(features, targets)
    assert fitted.weights.tolist() == [1.0] * 6 + [0.5] * 6
    assert fit_estimator(features, targets, altitude_hidden_copy=False).weights.tolist() == [1.0] * 6


def test_out_of_fold_estimators_never_see_the_scenario_they_score(recording):
    # The first column holds the scenario number, so the rows an estimator was fitted on reveal which scenarios it saw.
    scenarios = tuple(f"s{i}" for i in range(10))
    groups = np.repeat(np.arange(10), 4)
    features = np.column_stack([groups.astype(float)] + [np.ones(40)] * (len(FEATURES) - 1))
    examples = training.TrainingExamples(features, np.tile([1, 0, 0, 0], 10), groups, scenarios, 10, 0, 30, 10)
    by_scenario = training.out_of_fold_estimators(examples)
    assert set(by_scenario) == set(scenarios)
    for position, name in enumerate(scenarios):
        seen = set(by_scenario[name].features[:, 0].astype(int).tolist())
        assert position not in seen
        assert seen  # fitted on the others, and always on whole scenarios because the column equals the group
        assert len(seen) == 8  # five folds of ten scenarios: two held out each time


def test_hidden_copy_leaves_the_inputs_unchanged(recording):
    features, targets = small_inputs(len(FEATURES))
    before = features.copy(), targets.copy()
    fit_estimator(features, targets)
    assert features.tobytes() == before[0].tobytes() and targets.tobytes() == before[1].tobytes()


def test_without_hidden_copy_the_rows_are_as_given(recording):
    features, targets = small_inputs(len(FEATURES))
    fitted = fit_estimator(features, targets, altitude_hidden_copy=False)
    assert np.array_equal(fitted.features, features) and fitted.targets.tolist() == targets.tolist()


def test_parameters_override_the_defaults_and_keep_a_seed(recording):
    features, targets = small_inputs(len(FEATURES))
    fitted = fit_estimator(features, targets, {"max_iter": 7})
    assert fitted.parameters["max_iter"] == 7 and fitted.parameters["random_state"] == training.DEFAULT_PARAMETERS["random_state"]


# --- 7. Determinism ---


def probe_rows(examples):
    missing = examples.features[:5].copy()
    for index, name in enumerate(FEATURES):
        if name in ALTITUDE_FEATURES:
            missing[:, index] = math.nan
    return np.vstack([examples.features, missing])


def test_two_fits_on_the_same_data_are_identical(examples, model):
    again = build_model(examples, FIT_PARAMETERS)
    assert again.version == model.version and again.version.startswith("learned-")
    probe = probe_rows(examples)
    assert again.estimator.predict_proba(probe).tobytes() == model.estimator.predict_proba(probe).tobytes()


def test_version_changes_with_data_settings_or_schema(examples, model):
    changed_value = examples.features.copy()
    changed_value[0, 0] += 1.0
    now_missing = examples.features.copy()
    now_missing[0, 0] = math.nan
    flipped = examples.targets.copy()
    flipped[0] = 1 - flipped[0]
    # Fast settings keep this cheap: only the versions are compared.
    fast = {**FIT_PARAMETERS, "max_iter": 2}
    base = build_model(examples, fast)
    assert base.version != model.version  # parameters
    others = [
        build_model(dataclasses.replace(examples, features=changed_value), fast),
        build_model(dataclasses.replace(examples, features=now_missing), fast),
        build_model(dataclasses.replace(examples, targets=flipped), fast),
        build_model(examples, {**fast, "random_state": 1}),
        build_model(examples, fast, altitude_hidden_copy=False),
        build_model(examples, fast, candidates=CandidateConfig(max_gap_s=500.0)),
        build_model(examples, fast, min_reports=4),
    ]
    versions = [m.version for m in others] + [base.version]
    assert len(set(versions)) == len(versions)


# --- 8. Score meaning ---


def test_clean_continuation_scores_above_a_pair_far_off_the_projected_path(model):
    # c starts when b does, at the same altitude, on a parallel line about 2.7 km east. Only position separates them.
    off_path = flight("c", 200, lon=-107.97)
    result = analyze(scenario_of(flight("a", 0), flight("b", 200), off_path), model, LENIENT)
    clean, far = candidate_entry(result, "a", "b")["score"], candidate_entry(result, "a", "c")["score"]
    assert clean > 0.5 > far  # the fitted model has learned something from the synthetic world


def test_altitude_conflict_scores_below_the_same_track_at_the_same_altitude(model):
    result = analyze(scenario_of(flight("a", 0), flight("b", 200), flight("c", 200, altitude=30000.0)), model, LENIENT)
    assert candidate_entry(result, "a", "b")["score"] > 0.5 > candidate_entry(result, "a", "c")["score"]


def test_ranked_candidates_are_in_descending_order_and_in_range(model):
    rows, _ = synthetic_scenario(201)
    result = analyze(prepare_scenario(rows), model, LENIENT)
    lengths = []
    for entry in result["tracklets"]:
        scores = [c["score"] for c in entry["candidates"]]
        assert scores == sorted(scores, reverse=True)
        assert [c["rank"] for c in entry["candidates"]] == list(range(1, len(scores) + 1))
        assert all(math.isfinite(s) and 0 <= s <= 1 for s in scores)
        lengths.append(len(scores))
    assert max(lengths) >= 2  # ordering was actually tested


def test_a_pairs_score_does_not_depend_on_the_others_or_their_order(model):
    candidates = generate_candidates(prepare_scenario(synthetic_scenario(202)[0])).candidates
    assert len(candidates) >= 3
    together = {(s.predecessor, s.successor): s.score for s in score_candidates(model, candidates)}
    assert {(s.predecessor, s.successor): s.score for s in score_candidates(model, list(reversed(candidates)))} == together
    alone = score_candidates(model, [candidates[0]])[0]
    assert alone.score == together[(alone.predecessor, alone.successor)]
    assert score_candidates(model, []) == []


def test_score_uses_measurements_and_not_tracklet_identity(model):
    measurements = pair()
    first = score_candidates(model, [Candidate("a", "b", measurements)])[0].score
    assert score_candidates(model, [Candidate("zz-9", "qq-1", measurements)])[0].score == first


def test_renaming_tracklets_changes_nothing_but_the_names(model):
    def chain(a, b, c):
        return analyze(scenario_of(flight(a, 0), flight(b, 200), flight(c, 400)), model, LENIENT)
    plain, renamed = chain("a", "b", "c"), chain("z", "y", "x")  # sort order reversed
    assert plain["links"] == {"a": "b", "b": "c"} and renamed["links"] == {"z": "y", "y": "x"}
    assert candidate_entry(plain, "a", "b")["score"] == candidate_entry(renamed, "z", "y")["score"]


# --- 9. Thresholds and margins at the model's own scale ---


def clean_pair_score(model):
    return candidate_entry(analyze(scenario_of(flight("a", 0), flight("b", 200)), model, LENIENT), "a", "b")["score"]


def test_minimum_score_decides_at_the_models_own_scale(model):
    score = clean_pair_score(model)
    rows = flight("a", 0), flight("b", 200)

    above = analyze(scenario_of(*rows), model, AcceptancePolicy(score + 0.001, 0.0, 0.001))
    assert above["links"] == {}
    assert tracklet(above, "a")["status"] == "unresolved" and tracklet(above, "a")["reason_code"] == BELOW_THRESHOLD
    assert candidate_entry(above, "a", "b")["decision"] == BELOW_THRESHOLD

    at = analyze(scenario_of(*rows), model, AcceptancePolicy(score, 0.0, 0.001))
    assert at["links"] == {"a": "b"}  # a pair is only refused when below the minimum


def test_large_margin_between_near_identical_rivals_leaves_both_unresolved(model):
    # x and y straddle the line by about 90 m each and start together, so neither can sit between a and the other.
    rows = flight("a", 0), flight("x", 200, lon=-107.999), flight("y", 200, lon=-108.001)
    result = analyze(scenario_of(*rows), model, AcceptancePolicy(0.0, 0.5, 0.0))
    assert result["links"] == {}
    assert candidate_entry(result, "a", "x")["decision"] == CLOSE_RIVAL and candidate_entry(result, "a", "y")["decision"] == CLOSE_RIVAL
    assert tracklet(result, "a")["reason_code"] == CLOSE_RIVAL


def test_large_margin_between_rival_predecessors_leaves_both_unresolved(model):
    rows = flight("p", 0, lon=-107.999), flight("q", 0, lon=-108.001), flight("r", 200)
    result = analyze(scenario_of(*rows), model, AcceptancePolicy(0.0, 0.5, 0.0))
    assert result["links"] == {}
    assert candidate_entry(result, "p", "r")["decision"] == CLOSE_RIVAL and candidate_entry(result, "q", "r")["decision"] == CLOSE_RIVAL


def test_margin_decides_at_the_models_own_scale(model):
    rows = flight("a", 0), flight("b", 200), flight("c", 200, lon=-107.97)
    scores = analyze(scenario_of(*rows), model, LENIENT)
    lead = candidate_entry(scores, "a", "b")["score"] - candidate_entry(scores, "a", "c")["score"]
    assert lead > 0.02
    assert analyze(scenario_of(*rows), model, AcceptancePolicy(0.0, lead - 0.01, 0.0))["links"] == {"a": "b"}
    blocked = analyze(scenario_of(*rows), model, AcceptancePolicy(0.0, lead + 0.01, 0.0))
    assert blocked["links"] == {} and tracklet(blocked, "a")["reason_code"] == CLOSE_RIVAL


def test_nothing_is_forced_when_nothing_clears(model):
    rows, _ = synthetic_scenario(203)
    result = analyze(prepare_scenario(rows), model, AcceptancePolicy(1.5, 0.0, 0.05))  # scores never exceed 1
    assert result["links"] == {} and candidate_pairs(result)
    for entry in result["tracklets"]:
        assert entry["status"] == "unresolved" and entry["accepted_successor"] is None
        assert all(c["decision"] != "accepted" for c in entry["candidates"])


def test_lone_tracklet_has_no_candidate_and_stays_unresolved(model):
    result = analyze(scenario_of(flight("a", 0)), model, ANYTHING)
    assert result["links"] == {} and tracklet(result, "a")["reason_code"] == NO_CANDIDATE


# --- 10. A tracklet with no true successor, and an attractive wrong candidate ---


def test_ambiguous_no_successor_case_is_left_unresolved_by_a_demanding_policy(model):
    # a's aircraft never reappears. Two other aircraft happen to line up with it, equally well.
    rows = flight("a", 0), flight("x", 200, lon=-107.999), flight("y", 200, lon=-108.001)
    demanding = analyze(scenario_of(*rows), model, AcceptancePolicy(0.2, 0.3, 0.05))
    assert demanding["links"] == {}
    assert tracklet(demanding, "a")["status"] == "unresolved" and tracklet(demanding, "a")["reason_code"] == CLOSE_RIVAL
    m = metrics(count_links(demanding["links"], {"a": None, "x": None, "y": None}))
    assert m["wrong"] == 0 and m["accepted"] == 0 and m["precision"] is None and m["recall"] is None


def test_decoy_with_a_conflicting_altitude_is_refused(model):
    # Horizontally a perfect continuation, but 15,000 ft away with no climb. Only altitude shows it is another aircraft.
    result = analyze(scenario_of(flight("a", 0), flight("x", 200, altitude=30000.0)), model, AcceptancePolicy(0.5, 0.0, 0.05))
    assert result["links"] == {} and tracklet(result, "a")["reason_code"] == BELOW_THRESHOLD


def test_perfect_decoy_is_accepted_and_is_counted_as_wrong(model):
    """The limit of the method, stated openly: a decoy identical in every measurement can't be told from a true link.

    A -> X is a clean continuation by every feature, so a lenient policy accepts it. What matters is that the
    metrics show it: a wrong link that hurts precision, with no effect on the recall denominator.
    """
    rows = flight("a", 0), flight("x", 200), flight("p", 0, lat=45.0), flight("q", 200, lat=45.0)
    result = analyze(scenario_of(*rows), model, LENIENT)
    assert result["links"] == {"a": "x", "p": "q"}
    m = metrics(count_links(result["links"], {"a": None, "x": None, "p": "q", "q": None}, candidate_pairs(result)))
    assert m["accepted"] == 2 and m["correct"] == 1 and m["wrong"] == 1
    assert m["wrong_other_aircraft"] == 1 and m["wrong_skips"] == 0
    assert m["labeled_links"] == 1  # a has no labeled successor, so it isn't in the denominator
    assert m["precision"] == 0.5 and m["recall"] == 1.0

    # And nothing but that wrong link: precision is 0, recall is undefined rather than 0 or 1.
    alone = metrics(count_links({"a": "x"}, {"a": None, "x": None}))
    assert alone["precision"] == 0 and alone["recall"] is None and alone["labeled_links"] == 0 and alone["wrong"] == 1


def test_perfect_decoy_is_held_back_by_a_policy_above_its_score(model):
    score = clean_pair_score(model)
    result = analyze(scenario_of(flight("a", 0), flight("b", 200)), model, AcceptancePolicy(score + 0.001, 0.0, 0.05))
    assert result["links"] == {} and metrics(count_links(result["links"], {"a": None, "b": None}))["wrong"] == 0


# --- 11. Immediate successor ---


def test_chain_links_each_fragment_to_the_next_with_the_skip_withheld(model):
    result = analyze(scenario_of(flight("a", 0), flight("b", 200), flight("c", 400)), model, LENIENT)
    assert result["links"] == {"a": "b", "b": "c"}
    skip = candidate_entry(result, "a", "c")
    assert skip["decision"] == POSSIBLE_INTERMEDIATE and "b" in skip["decision_detail"]
    assert tracklet(result, "c")["status"] == "unresolved" and tracklet(result, "c")["reason_code"] == NO_CANDIDATE


def test_the_skip_stays_withheld_under_the_loosest_policy(model):
    result = analyze(scenario_of(flight("a", 0), flight("b", 200), flight("c", 400)), model, ANYTHING)
    assert result["links"] == {"a": "b", "b": "c"}
    assert candidate_entry(result, "a", "c")["decision"] == POSSIBLE_INTERMEDIATE


def test_with_the_middle_fragment_absent_the_next_available_is_the_successor(model):
    # Same flight without b: c is now the immediate next available tracklet, so nothing withholds the pair.
    result = analyze(scenario_of(flight("a", 0), flight("c", 400)), model, ANYTHING)
    assert result["links"] == {"a": "c"} and candidate_entry(result, "a", "c")["decision"] == "accepted"


# --- 12. Global constraints, from the raw reports and not via check_links ---


def parse_time(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)


def assert_links_valid(links, rows):
    times = {}
    for row in rows:
        first, last = times.get(row["tracklet_id"], (None, None))
        moment = parse_time(row["observed_at"])
        times[row["tracklet_id"]] = (moment if first is None else min(first, moment), moment if last is None else max(last, moment))
    assert len(set(links.values())) == len(links), "a tracklet has two predecessors"
    for a, b in links.items():
        assert a != b, "self-link"
        assert a in times and b in times, "link outside the scenario"
        assert times[b][0] > times[a][1], f"{a} -> {b} does not move forward in time"
    for start in links:
        seen, current = {start}, links[start]
        while current in links:
            assert current not in seen, "cycle"
            seen.add(current)
            current = links[current]


@pytest.mark.parametrize("seed", [301, 302, 303])
@pytest.mark.parametrize("policy", [
    AcceptancePolicy(0.0, 0.0, 2.0),  # nothing counts as an intermediate, so selection accepts as much as it can
    ANYTHING,
    LENIENT,
], ids=["greedy", "anything", "lenient"])
def test_global_constraints_hold_on_a_dense_scenario(model, seed, policy):
    rows, _ = synthetic_scenario(seed, aircraft=8)
    result = analyze(prepare_scenario(rows), model, policy)
    assert_links_valid(result["links"], rows)
    pairs = candidate_pairs(result)
    successors = [b for _, b in pairs]
    predecessors = [a for a, _ in pairs]
    # Without contention the constraints wouldn't be tested.
    assert len(set(successors)) < len(successors) and len(set(predecessors)) < len(predecessors) and result["links"]
    for entry in result["tracklets"]:
        accepted = result["links"].get(entry["tracklet_id"])
        assert entry["accepted_successor"] == accepted and (entry["status"] == "accepted") == (accepted is not None)


def test_greedy_policy_has_to_refuse_some_candidates_in_a_dense_scenario(model):
    rows, _ = synthetic_scenario(301, aircraft=8)
    result = analyze(prepare_scenario(rows), model, AcceptancePolicy(0.0, 0.0, 2.0))
    decisions = {c["decision"] for entry in result["tracklets"] for c in entry["candidates"]}
    assert {"accepted", "successor_taken", "predecessor_taken"} <= decisions


# --- 13. Artifacts ---


def test_save_writes_both_files_with_the_recorded_details(model, saved_folder):
    assert (saved_folder / MODEL_FILE).is_file() and (saved_folder / METADATA_FILE).is_file()
    metadata = json.loads((saved_folder / METADATA_FILE).read_text(encoding="utf-8"))
    assert metadata["model_version"] == model.version
    assert metadata["feature_names"] == list(FEATURES)
    assert isinstance(metadata["parameters"]["random_state"], int)  # a real seed, not None
    assert metadata["parameters"]["max_iter"] == FIT_PARAMETERS["max_iter"]
    assert set(metadata["library_versions"]) >= {"python", "scikit-learn", "numpy", "joblib"}
    assert metadata["artifact_sha256"] == hashlib.sha256((saved_folder / MODEL_FILE).read_bytes()).hexdigest()


def test_reload_keeps_the_version_and_settings(model, saved_folder):
    loaded = load_model(saved_folder)
    assert loaded.version == model.version and loaded.feature_names == model.feature_names
    assert loaded.candidates == model.candidates and loaded.min_reports_for_motion == model.min_reports_for_motion


FRESH_PROCESS = """
import json, sys
import numpy as np
from govia_tracks.association import AcceptancePolicy
from govia_tracks.learned import analyze, load_model
from govia_tracks.preprocess import prepare_scenario

folder, rows_path, matrix_path = sys.argv[1:]
model = load_model(folder)
rows = json.load(open(rows_path))
matrix = np.array(json.load(open(matrix_path)), dtype=np.float64)
print(json.dumps({
    "version": model.version,
    "predictions": model.estimator.predict_proba(matrix)[:, 1].tolist(),
    "result": analyze(prepare_scenario(rows), model, AcceptancePolicy(0.2, 0.05, 0.05)),
}))
"""


def test_a_fresh_process_gets_the_same_exact_predictions(model, examples, saved_folder, tmp_path):
    rows, _ = synthetic_scenario(401)
    matrix = probe_rows(examples)[:200]
    missing = probe_rows(examples)[-5:]  # includes hidden altitude, which JSON carries as NaN
    matrix = np.vstack([matrix, missing])
    (tmp_path / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "matrix.json").write_text(json.dumps(matrix.tolist()), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-c", FRESH_PROCESS, str(saved_folder), str(tmp_path / "rows.json"), str(tmp_path / "matrix.json")],
        check=True, capture_output=True, text=True, cwd=ROOT,
    )
    other = json.loads(done.stdout)
    assert other["version"] == model.version
    assert other["predictions"] == model.estimator.predict_proba(matrix)[:, 1].tolist()  # exact floats; JSON keeps them
    assert other["result"] == analyze(prepare_scenario(rows), model, LENIENT)


# --- 14. Repeated inference ---


def test_repeated_analysis_is_identical(model):
    scenario = prepare_scenario(synthetic_scenario(402)[0])
    assert to_json(analyze(scenario, model, LENIENT)) == to_json(analyze(scenario, model, LENIENT))


def test_reordered_rows_and_identical_duplicates_change_nothing(model):
    rows, _ = synthetic_scenario(403)
    expected = to_json(analyze(prepare_scenario(rows), model, LENIENT))
    for seed in range(3):
        rng = random.Random(seed)
        noisy = rows + rng.sample(rows, 12)
        rng.shuffle(noisy)
        assert to_json(analyze(prepare_scenario(noisy), model, LENIENT)) == expected


def test_inference_never_fits(saved_folder, monkeypatch):
    def refuse(self, *_, **__):
        raise AssertionError("fit called")
    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", refuse)
    with pytest.raises(AssertionError, match="fit called"):  # the guard is live
        fit_estimator(*small_inputs(len(FEATURES)))
    loaded = load_model(saved_folder)
    assert analyze(scenario_of(flight("a", 0), flight("b", 200)), loaded, LENIENT)["links"] == {"a": "b"}


def test_analysis_does_not_change_the_model(examples, model):
    probe = probe_rows(examples)
    before = model.estimator.predict_proba(probe).tobytes()
    version, metadata = model.version, json.dumps(model.metadata, sort_keys=True)
    analyze(prepare_scenario(synthetic_scenario(404)[0]), model, LENIENT)
    assert model.estimator.predict_proba(probe).tobytes() == before
    assert model.version == version and json.dumps(model.metadata, sort_keys=True) == metadata


# --- 15. Metrics on learned results ---


def test_metrics_on_a_correct_chain_with_exact_counts(model):
    # n is 730 km east: every pair with it would need supersonic speed, so it is gated out on both sides.
    rows = flight("a", 0), flight("b", 200), flight("c", 400), flight("n", 200, lon=-100.0)
    result = analyze(scenario_of(*rows), model, LENIENT)
    pairs = candidate_pairs(result)
    assert pairs == {("a", "b"), ("a", "c"), ("b", "c")}
    m = metrics(count_links(result["links"], {"a": "b", "b": "c", "c": None, "n": None}, pairs, "scn"))
    assert (m["accepted"], m["correct"], m["wrong"], m["wrong_skips"], m["wrong_other_aircraft"]) == (2, 2, 0, 0, 0)
    assert (m["labeled_links"], m["labeled_unresolved"], m["tracklets"], m["unresolved_tracklets"]) == (2, 0, 4, 2)
    assert (m["candidate_links_retained"], m["candidate_links_excluded"], m["candidate_pairs"]) == (2, 0, 3)
    assert m["precision"] == 1 and m["recall"] == 1 and m["unresolved_proportion"] == 0 and m["candidate_recall"] == 1
    assert m["candidates_per_tracklet"] == 0.75


def test_metrics_when_the_accepted_links_are_wrong(model):
    # The same accepted a->b and b->c, under labels where a's true successor is c and b flies alone.
    result = analyze(scenario_of(flight("a", 0), flight("b", 200), flight("c", 400)), model, LENIENT)
    assert result["links"] == {"a": "b", "b": "c"}
    m = metrics(count_links(result["links"], {"a": "c", "b": None, "c": None}, candidate_pairs(result)))
    assert (m["accepted"], m["correct"], m["wrong"], m["wrong_other_aircraft"], m["wrong_skips"]) == (2, 0, 2, 2, 0)
    assert m["labeled_links"] == 1 and m["labeled_unresolved"] == 0
    assert m["precision"] == 0 and m["recall"] == 0


def test_an_accepted_skip_is_wrong_and_the_lost_links_still_count(model):
    # b really sits between a and c, but flies about 9 km east of the line, so the projection gate removes a->b and b->c.
    # With nothing to withhold it, a->c clears a policy that accepts anything: a wrong link of the skip kind.
    rows = flight("a", 0), flight("b", 200, lon=-107.9), flight("c", 400)
    result = analyze(scenario_of(*rows), model, ANYTHING)
    assert result["links"] == {"a": "c"}
    counts = count_links(result["links"], {"a": "b", "b": "c", "c": None}, candidate_pairs(result), "scn")
    m = metrics(counts)
    assert (m["accepted"], m["correct"], m["wrong_skips"], m["wrong_other_aircraft"]) == (1, 0, 1, 0)
    assert (m["labeled_links"], m["candidate_links_retained"], m["candidate_links_excluded"]) == (2, 0, 2)
    assert m["labeled_unresolved"] == 1  # b; a has a link, even if it is wrong
    assert m["precision"] == 0 and m["recall"] == 0 and m["candidate_recall"] == 0
    assert counts.excluded_links == [("scn", "a", "b"), ("scn", "b", "c")]


def test_undefined_metrics_are_none_when_the_denominator_is_zero(model):
    result = analyze(scenario_of(flight("a", 0), flight("b", 200)), model, AcceptancePolicy(1.5, 0.0, 0.05))
    assert result["links"] == {}
    no_links = metrics(count_links(result["links"], {"a": None, "b": None}, candidate_pairs(result)))
    assert no_links["accepted"] == 0 and no_links["labeled_links"] == 0
    assert no_links["precision"] is None and no_links["recall"] is None
    assert no_links["unresolved_proportion"] is None and no_links["candidate_recall"] is None
    assert no_links["candidates_per_tracklet"] == 0.5  # the one pair a->b over two tracklets

    labeled = metrics(count_links(result["links"], {"a": "b", "b": None}, candidate_pairs(result)))
    assert labeled["precision"] is None  # nothing accepted, so undefined
    assert labeled["recall"] == 0 and labeled["unresolved_proportion"] == 1  # a link was there to find

    assert metrics(count_links({}, {}))["candidates_per_tracklet"] is None


# --- 16. JSON ---


def test_result_round_trips_through_strict_json_with_missing_measurements(model):
    rows = (
        flight("a", 0, altitude=None), flight("b", 200), flight("single", 200, count=1, altitude=None),
        flight("two", 200, count=2), flight("c", 400, altitude=None),
    )
    scenario = scenario_of(*rows, scenario_id="scn", snapshot_id="snap")
    result = analyze(scenario, model, LENIENT)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["method"] == "learned" and result["version"] == model.version
    assert result["scenario_id"] == "scn" and result["snapshot_id"] == "snap" and result["input_fingerprint"] == scenario.input_fingerprint
    assert result["policy"] == dataclasses.asdict(LENIENT)
    assert result["config"]["features"] == list(FEATURES)
    # The gaps really are in there, as null and not as a number.
    measured = [c["measurements"] for entry in result["tracklets"] for c in entry["candidates"]]
    assert any(m["altitude_miss_ft"] is None for m in measured)
    assert any(m["forward_miss_km"] is None for m in measured) or any(m["backward_miss_km"] is None for m in measured)
    assert all(isinstance(c["score"], float) and math.isfinite(c["score"]) for entry in result["tracklets"] for c in entry["candidates"])


# --- 17. Policy file, and keeping scripts away from held-out data ---


def test_load_policy_reads_the_policy_section(tmp_path):
    path = tmp_path / "learned.json"
    path.write_text(json.dumps({"model_version": "x", "cost_ratio": 5, "policy": {"min_score": 0.7, "min_margin": 0.2, "plausible_score": 0.05}}), encoding="utf-8")
    assert load_policy(path) == AcceptancePolicy(min_score=0.7, min_margin=0.2, plausible_score=0.05)


def test_load_policy_does_not_read_values_outside_the_policy_section(tmp_path):
    path = tmp_path / "learned.json"
    path.write_text(json.dumps({"min_score": 0.7, "min_margin": 0.2, "plausible_score": 0.05}), encoding="utf-8")
    with pytest.raises(KeyError):
        load_policy(path)


@pytest.fixture
def run_learned_tree():
    path = ROOT / "scripts" / "run_learned.py"
    if not path.exists():
        pytest.skip("scripts/run_learned.py doesn't exist yet")
    return path, ast.parse(path.read_text(encoding="utf-8"))


def calls_to(node, name):
    return [c for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == name]


def test_run_script_cannot_reach_held_out_data(run_learned_tree):
    path, tree = run_learned_tree
    # Read as source because the script runs its command on import.
    docstrings = {ast.get_docstring(n, clean=False) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef))}
    literals = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)} - docstrings
    for text in literals:
        lowered = text.lower().replace("\\", "/")
        assert lowered != "test" and "data/test" not in lowered and "staged" not in lowered, text

    kept = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "folders")
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PARTITIONS" for t in n.targets))]
    namespace = {"DATA": None}
    exec(compile(ast.Module(kept, []), str(path), "exec"), namespace)
    assert set(namespace["PARTITIONS"]) == {"train", "validation"}
    for partition in ("test", "staged_demo"):
        with pytest.raises(ValueError):
            namespace["folders"](partition)


def test_run_script_fits_only_on_train(run_learned_tree):
    _, tree = run_learned_tree
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def callers(name):
        return {caller for caller, node in functions.items() if calls_to(node, name)}

    # One helper builds all training rows, and everything that fits takes its rows from that helper.
    assert callers("training_examples") == {"train_examples"}
    assert callers("build_model") == {"train"} and callers("out_of_fold_estimators") == {"model_chooser"}
    assert callers("fit_estimator") == set()
    for call in calls_to(functions["train"], "build_model") + calls_to(functions["model_chooser"], "out_of_fold_estimators"):
        assert isinstance(call.args[0], ast.Call) and call.args[0].func.id == "train_examples"
    # Every folder listed while building training rows names the train partition, literally.
    helper = functions["train_examples"]
    arguments = [call.args[0] for call in calls_to(helper, "folders")]
    assert arguments and all(isinstance(a, ast.Constant) and a.value == "train" for a in arguments)
    assert "validation" not in {n.value for n in ast.walk(helper) if isinstance(n, ast.Constant)}


# --- Thresholds tied to their model, and the stress transforms ---


def test_thresholds_selected_for_another_model_are_refused(tmp_path):
    path = tmp_path / "learned.json"
    path.write_text(json.dumps({"model_version": "learned-a", "policy": {"min_score": 0.1, "min_margin": 0.2, "plausible_score": 0.05}}), encoding="utf-8")
    assert load_policy(path, "learned-a") == AcceptancePolicy(0.1, 0.2, 0.05)
    with pytest.raises(ValueError, match="learned-a"):
        load_policy(path, "learned-b")


def _two_flights():
    return scenario_of(flight("a", 0, altitude=15000.0), flight("b", 200, altitude=None))


def test_hide_altitude_removes_every_altitude_and_leaves_the_original_alone():
    scenario = _two_flights()
    hidden = hide_altitude(scenario)
    assert all(o.altitude_ft is None for t in hidden.tracklets for o in t.observations)
    assert scenario.tracklets[0].observations[0].altitude_ft == 15000.0
    assert hidden.tracklets[0].observations[0].latitude == scenario.tracklets[0].observations[0].latitude


def test_noisy_altitude_keeps_missing_missing_and_repeats_for_a_seed():
    scenario = _two_flights()
    first, second = (noisy_altitude(scenario, 100.0, random.Random("s")) for _ in range(2))
    assert first == second
    assert all(o.altitude_ft is None for o in first.tracklets[1].observations)  # b had none: still none, not noise around zero
    assert all(o.altitude_ft != 15000.0 for o in first.tracklets[0].observations)
    assert abs(first.tracklets[0].observations[0].altitude_ft - 15000.0) < 1000  # 10 sd is far beyond any draw here


def test_without_tracklets_and_rebuilt_labels_agree():
    scenario = scenario_of(flight("a", 0), flight("b", 200), flight("c", 400))
    reduced = without_tracklets(scenario, {"b"})
    assert [t.tracklet_id for t in reduced.tracklets] == ["a", "c"]
    assert labels_without({"a": "b", "b": "c", "c": None}, {"b"}) == {"a": "c", "c": None}


def test_visible_windows_keep_a_contiguous_run_of_each_aircraft():
    labels = {"a1": "a2", "a2": "a3", "a3": "a4", "a4": None, "b1": "b2", "b2": None}
    assert aircraft_chains(labels) == [["a1", "a2", "a3", "a4"], ["b1", "b2"]]
    for seed in range(50):
        dropped = outside_visible_windows(labels, random.Random(seed))
        for chain in aircraft_chains(labels):
            kept = [i for i, t in enumerate(chain) if t not in dropped]
            assert kept, "each aircraft keeps at least one fragment"
            assert kept == list(range(kept[0], kept[-1] + 1)), "no holes inside the window"


def test_random_fragments_is_deterministic_and_respects_the_extremes():
    labels = {name: None for name in "abcdefgh"}
    assert random_fragments(labels, 0.3, random.Random(1)) == random_fragments(labels, 0.3, random.Random(1))
    assert random_fragments(labels, 0.0, random.Random(1)) == set()
    assert random_fragments(labels, 1.0, random.Random(1)) == set(labels)


def test_split_tracklets_rewires_labels_to_the_new_fragments():
    # a has 20 reports over 190 s, so a 10 s gap leaves at least 8 each side. b has 6, too short to split.
    scenario = scenario_of(flight("a", 0, count=20), flight("b", 400))
    split, labels, cut = split_tracklets(scenario, {"a": "b", "b": None}, 10.0)
    assert cut == [("a#1", "a#2")]
    assert labels == {"a#1": "a#2", "a#2": "b", "b": None}
    assert {t.tracklet_id for t in split.tracklets} == {"a#1", "a#2", "b"}
    first, second = (next(t for t in split.tracklets if t.tracklet_id == n).observations for n in ("a#1", "a#2"))
    assert second[0].observed_us - first[-1].observed_us >= 10_000_000  # the gap asked for
    assert all(o.tracklet_id == "a#1" for o in first)


def test_split_tracklets_points_an_earlier_tracklet_at_the_first_half():
    scenario = scenario_of(flight("a", 0), flight("b", 200, count=20))
    _, labels, _ = split_tracklets(scenario, {"a": "b", "b": None}, 10.0)
    assert labels == {"a": "b#1", "b#1": "b#2", "b#2": None}


# --- Physical limits on the learned path ---


def _heading_off(tracklet_id, start_s, degrees, at_lat, at_lon, count=6):
    """A flight passing through (at_lat, at_lon) at start_s, on a course this many degrees east of north."""
    north = SPEED * math.cos(math.radians(degrees))
    east = SPEED * math.sin(math.radians(degrees)) / math.cos(math.radians(at_lat))
    return flight(tracklet_id, start_s, count=count, lat=at_lat - north * start_s, lon=at_lon - east * start_s, north=north, east=east)


def test_a_fragment_on_the_projected_path_but_flying_away_is_refused(model):
    # A ends at 50 s. By 130 s it would be 0.096 deg further north, and b starts exactly there, 45 degrees off course.
    a = flight("a", 0)
    b = _heading_off("b", 130, 45.0, 35.0 + SPEED * 130, -108.0)
    result = analyze(scenario_of(a, b), model, ANYTHING)
    candidate = tracklet(result, "a")["candidates"][0]
    assert candidate["measurements"]["forward_miss_km"] < 0.1  # looks perfect from A's side
    assert candidate["measurements"]["course_diff_deg"] == pytest.approx(45, abs=1)
    assert candidate["decision"] == PHYSICALLY_INCONSISTENT and result["links"] == {}
    assert tracklet(result, "a")["reason_code"] == PHYSICALLY_INCONSISTENT


def test_the_same_flight_on_a_consistent_course_is_not_refused(model):
    result = analyze(scenario_of(flight("a", 0), flight("b", 130)), model, ANYTHING)
    assert result["links"] == {"a": "b"}


@pytest.mark.parametrize("field,value,mentions", [
    ("forward_miss_km", 3.5, "forward"), ("backward_miss_km", 3.5, "backward"), ("course_diff_deg", 16.0, "courses"),
])
def test_physical_limits_each_apply(field, value, mentions):
    good = PairMeasurements(100.0, 13.0, 468.0, 0.2, 0.2, 1.0, 1.0, 100.0, 100.0, 6, 6)
    assert physical_objection(good) is None
    assert mentions in physical_objection(dataclasses.replace(good, **{field: value}))


def test_values_at_the_limit_and_missing_values_raise_no_objection():
    at_limit = PairMeasurements(100.0, 13.0, 468.0, MAX_DIRECTIONAL_MISS_KM, MAX_DIRECTIONAL_MISS_KM, 1.0, MAX_COURSE_DIFF_DEG, 100.0, 100.0, 6, 6)
    missing = PairMeasurements(100.0, 13.0, 468.0, None, None, None, None, None, None, 6, 6)
    assert physical_objection(at_limit) is None and physical_objection(missing) is None


def _sc(a, b, score, ruled_out=None):
    return ScoredCandidate(a, b, score, {}, ruled_out=ruled_out)


def test_a_ruled_out_pair_is_not_a_rival():
    candidates = [_sc("a", "b", 0.99, "too far off"), _sc("a", "c", 0.6)]
    links, decisions = select_links(candidates, AcceptancePolicy(0.1, 0.5, 0.05))
    assert links == {"a": "c"}  # with b as a rival, c's lead would be negative
    assert decisions[("a", "b")].code == PHYSICALLY_INCONSISTENT and decisions[("a", "b")].detail == "too far off"


def test_a_ruled_out_pair_is_not_a_step_in_a_chain():
    # a -> c would be withheld as a skip if a -> b counted as a possible first step.
    candidates = [_sc("a", "c", 0.9), _sc("a", "b", 0.9, "too far off"), _sc("b", "c", 0.9)]
    links, _ = select_links(candidates, AcceptancePolicy(0.1, 0.0, 0.05))
    assert links == {"a": "c"}


def test_hiding_altitude_of_some_tracklets():
    scenario = scenario_of(*(flight(f"t{i}", 0, lon=-108.0 + i) for i in range(40)))
    first = hide_altitude_of_some(scenario, 0.5, random.Random("x"))
    assert first == hide_altitude_of_some(scenario, 0.5, random.Random("x"))
    hidden = [all(o.altitude_ft is None for o in t.observations) for t in first.tracklets]
    assert 5 < sum(hidden) < 35  # some of each
    assert all(all(o.altitude_ft is None for o in t.observations) or all(o.altitude_ft is not None for o in t.observations) for t in first.tracklets)
    assert hide_altitude_of_some(scenario, 0.0, random.Random("x")) == scenario


def test_trim_reports_cuts_both_ends_and_keeps_three():
    scenario = scenario_of(flight("long", 0, count=20), flight("short", 0, count=4, lon=-100.0))
    trimmed = {t.tracklet_id: t.observations for t in trim_reports(scenario, 0.25).tracklets}
    assert [o.observation_id for o in trimmed["long"]] == [f"long-{i}" for i in range(5, 15)]  # 5 off each end of 20
    assert len(trim_reports(scenario, 0.49).tracklets[1].observations) >= 3 and len(trimmed["short"]) >= 3
