"""Deduplication, conflict rejection, ordering, joins, cutoffs and input preservation."""
import copy
import json
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from govia_tracks.errors import InputValidationError
from govia_tracks.loader import load_snapshot, read_jsonl
from govia_tracks.movement import summarize_tracklet
from govia_tracks.preprocess import prepare_scenario

from conftest import observation, supporting


def scenario_rows():
    rows = [
        observation(observation_id=f"a{i}", tracklet_id="t-a", observed_at=f"2026-01-10T10:00:{i:02d}.000Z",
                    received_at=f"2026-01-10T10:01:{i:02d}.000Z", latitude=35 + i * 0.01)
        for i in range(6)
    ]
    rows += [
        observation(observation_id=f"b{i}", tracklet_id="t-b", observed_at=f"2026-01-10T10:05:{i:02d}.000Z",
                    received_at=f"2026-01-10T10:06:{i:02d}.000Z", longitude=-108 + i * 0.01, altitude_ft=None)
        for i in range(4)
    ]
    support = [supporting(record_id="s1", observation_id="a1"), supporting(record_id="s2", observation_id="b2")]
    return rows, support


def snapshot_view(prepared):
    """Everything downstream can see, minus the row-count bookkeeping that retransmissions change."""
    return {
        "tracklets": [asdict(t) for t in prepared.tracklets],
        "motion": [asdict(summarize_tracklet(t)) for t in prepared.tracklets],
        "supporting": {k: [asdict(r) for r in v] for k, v in prepared.supporting.items()},
        "fingerprint": prepared.input_fingerprint,
    }


def test_identical_duplicates_are_dropped_and_counted():
    prepared = prepare_scenario([observation(), observation()], [supporting(), supporting()])
    assert prepared.quality.observation_rows == 2 and prepared.quality.unique_observations == 1
    assert prepared.quality.supporting_rows == 2 and prepared.quality.unique_supporting == 1
    assert len(prepared.supporting["obs-1"]) == 1


def test_key_order_does_not_make_a_duplicate_conflict():
    reordered = dict(reversed(list(observation().items())))
    reordered["attributes"] = dict(reversed(list(reordered["attributes"].items())))
    assert prepare_scenario([observation(), reordered]).quality.unique_observations == 1


@pytest.mark.parametrize("change", [
    {"latitude": 35.0001},
    {"received_at": "2026-01-10T10:00:09.000Z"},
    {"attributes": {"spd": 291.0, "q": 1}},
    {"attributes": {"spd": 290.0, "q": True}},  # 1 and true must not compare equal
    {"tracklet_id": "t-2"},
])
def test_conflicts_are_rejected_and_name_the_id(change):
    with pytest.raises(InputValidationError, match="obs-1") as caught:
        prepare_scenario([observation(), observation(**change)])
    assert [i.code for i in caught.value.issues] == ["conflicting_id"]
    assert caught.value.issues[0].record_id == "obs-1"


def test_missing_versus_null_is_a_conflict():
    absent = observation()
    del absent["altitude_ft"]
    with pytest.raises(InputValidationError):
        prepare_scenario([observation(altitude_ft=None), absent])


def test_supporting_conflict_is_rejected():
    with pytest.raises(InputValidationError, match="sup-1"):
        prepare_scenario([observation()], [supporting(), supporting(attributes={"a1": 1.0})])


def test_reports_are_sorted_by_observation_time_not_receipt_time():
    late_observed_early_received = observation(
        observation_id="x", observed_at="2026-01-10T10:00:30.000Z", received_at="2026-01-10T10:00:31.000Z")
    early_observed_late_received = observation(
        observation_id="y", observed_at="2026-01-10T10:00:10.000Z", received_at="2026-01-10T10:09:00.000Z")
    prepared = prepare_scenario([late_observed_early_received, early_observed_late_received])
    assert [o.observation_id for o in prepared.tracklets[0].observations] == ["y", "x"]
    assert prepared.quality.tracklets_delivered_out_of_order == 1


def test_reordering_and_retransmission_change_nothing_downstream():
    rows, support = scenario_rows()
    expected = snapshot_view(prepare_scenario(rows, support))
    for seed in range(5):
        rng = random.Random(seed)
        noisy_rows = rows + copy.deepcopy(rng.sample(rows, 4))
        noisy_support = support + copy.deepcopy(support[:1])
        rng.shuffle(noisy_rows)
        rng.shuffle(noisy_support)
        assert snapshot_view(prepare_scenario(noisy_rows, noisy_support)) == expected


def test_fingerprint_does_not_depend_on_which_equal_copy_arrives_first():
    whole = observation(attributes={"spd": 290})
    decimal = observation(attributes={"spd": 290.0})
    assert prepare_scenario([whole, decimal]).input_fingerprint == prepare_scenario([decimal, whole]).input_fingerprint


def test_fingerprint_changes_when_content_changes():
    assert prepare_scenario([observation()]).input_fingerprint != prepare_scenario([observation(latitude=35.5)]).input_fingerprint


def test_missing_altitude_stays_missing():
    absent = observation(observation_id="absent", observed_at="2026-01-10T10:00:01.000Z")
    del absent["altitude_ft"]
    rows = [observation(observation_id="null", altitude_ft=None), absent]
    prepared = prepare_scenario(rows)
    assert prepared.quality.altitude_null == 1 and prepared.quality.altitude_absent == 1
    assert [o.altitude_ft for o in prepared.tracklets[0].observations] == [None, None]
    assert "altitude_ft" not in prepared.tracklets[0].observations[1].raw  # original still tells them apart
    motion = summarize_tracklet(prepared.tracklets[0])
    assert motion.altitude_reports == 0 and motion.median_altitude_ft is None


def test_altitude_median_ignores_missing_rather_than_treating_it_as_zero():
    rows = [
        observation(observation_id="a", altitude_ft=10000.0, observed_at="2026-01-10T10:00:00.000Z"),
        observation(observation_id="b", altitude_ft=None, observed_at="2026-01-10T10:00:10.000Z"),
        observation(observation_id="c", altitude_ft=12000.0, observed_at="2026-01-10T10:00:20.000Z"),
    ]
    motion = summarize_tracklet(prepare_scenario(rows).tracklets[0])
    assert motion.altitude_reports == 2 and motion.median_altitude_ft == 11000.0


def test_head_and_tail_altitude_use_the_matching_end_of_the_tracklet():
    values = [10000.0, None, 11000.0, 12000.0, None, 13000.0, 14000.0, 15000.0]
    rows = [
        observation(observation_id=f"o{i}", altitude_ft=value, observed_at=f"2026-01-10T10:00:{i * 5:02d}.000Z")
        for i, value in enumerate(values)
    ]
    motion = summarize_tracklet(prepare_scenario(rows).tracklets[0])
    # Six known altitudes, so a third is two reports at each end.
    assert motion.head_altitude_ft == 10500.0 and motion.tail_altitude_ft == 14500.0


def test_supporting_records_join_only_by_observation_id():
    rows, support = scenario_rows()
    prepared = prepare_scenario(rows, support + [supporting(record_id="orphan", observation_id="elsewhere")])
    assert set(prepared.supporting) == {"a1", "b2"}
    assert [r.record_id for r in prepared.unjoined_supporting] == ["orphan"]
    assert prepared.quality.supporting_without_observation == 1


def test_scenarios_are_prepared_independently():
    # The same IDs in two calls must not interact: scenarios share no state.
    first = prepare_scenario([observation()])
    second = prepare_scenario([observation(latitude=36.0)])
    assert first.tracklets[0].observations[0].latitude == 35.0
    assert second.tracklets[0].observations[0].latitude == 36.0


def test_records_after_the_cutoff_are_refused():
    rows = [observation(received_at="2026-01-10T10:00:05.000Z")]
    assert prepare_scenario(rows, receipt_cutoff="2026-01-10T10:00:05.000Z").receipt_cutoff  # at the cutoff is available
    with pytest.raises(InputValidationError) as caught:
        prepare_scenario(rows, receipt_cutoff="2026-01-10T10:00:04.999Z")
    assert caught.value.issues[0].code == "after_cutoff"
    with pytest.raises(InputValidationError):
        prepare_scenario(rows, [supporting(received_at="2026-01-10T10:00:06.000Z")], receipt_cutoff="2026-01-10T10:00:05.000Z")


def test_originals_and_attributes_are_preserved_unchanged():
    rows, support = scenario_rows()
    rows[0]["attributes"] = {"spd": 1, "nested": {"z": [1, None, "x"]}, "unknown": None}
    rows[0]["extra_top_level"] = {"kept": True}
    expected = copy.deepcopy(rows[0])
    prepared = prepare_scenario(rows, support)
    kept = next(o for t in prepared.tracklets for o in t.observations if o.observation_id == rows[0]["observation_id"])
    assert kept.raw == expected and kept.attributes == expected["attributes"]


def test_supplied_fixture_identical_duplicate(packet):
    rows = read_jsonl(packet / "fixtures" / "identical_duplicate.jsonl")
    assert len(rows) == 2
    assert prepare_scenario(rows).quality.unique_observations == 1


def test_supplied_fixture_conflicting_id(packet):
    rows = read_jsonl(packet / "fixtures" / "conflicting_observation_id.jsonl")
    with pytest.raises(InputValidationError, match="obs-f060e3f5") as caught:
        prepare_scenario(rows)
    assert caught.value.issues[0].code == "conflicting_id"


def test_supplied_fixture_invalid_coordinate(packet):
    rows = read_jsonl(packet / "fixtures" / "invalid_coordinate.jsonl")
    with pytest.raises(InputValidationError) as caught:
        prepare_scenario(rows)
    assert (caught.value.issues[0].code, caught.value.issues[0].field) == ("invalid_field", "latitude")


@pytest.mark.parametrize("text", ['{"a": NaN}', '{"a": 1, "a": 2}', "{oops"])
def test_malformed_jsonl_is_reported_with_its_line(tmp_path, text):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"ok": 1}\n' + text + "\n", encoding="utf-8")
    with pytest.raises(InputValidationError, match="line 2"):
        read_jsonl(path)


def test_snapshot_manifest_mismatch_is_rejected(tmp_path):
    (tmp_path / "observations.jsonl").write_text(json.dumps(observation()) + "\n", encoding="utf-8")
    (tmp_path / "supporting.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "snapshot.json").write_text(json.dumps({
        "snapshot_id": "s", "scenario_id": "c", "partition": "train", "receipt_cutoff": None,
        "files": {"observations": "observations.jsonl", "supporting": "supporting.jsonl", "labels": "labels.jsonl"},
        "tracklet_count": 1, "unique_observation_count": 1, "observation_row_count": 2, "supporting_row_count": 0,
        "labeled_link_count": 0,
    }), encoding="utf-8")
    with pytest.raises(InputValidationError, match="observation_row_count"):
        load_snapshot(tmp_path)


def test_loading_a_snapshot_never_needs_labels(tmp_path):
    (tmp_path / "observations.jsonl").write_text(json.dumps(observation()) + "\n", encoding="utf-8")
    (tmp_path / "supporting.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "snapshot.json").write_text(json.dumps({
        "snapshot_id": "s", "scenario_id": "c", "partition": "train", "receipt_cutoff": None,
        "files": {"observations": "observations.jsonl", "supporting": "supporting.jsonl", "labels": "labels.jsonl"},
        "tracklet_count": 1, "unique_observation_count": 1, "observation_row_count": 1, "supporting_row_count": 0,
        "labeled_link_count": 0,
    }), encoding="utf-8")
    assert not (tmp_path / "labels.jsonl").exists()
    assert len(load_snapshot(tmp_path).tracklets) == 1


def test_input_path_does_not_import_labels():
    """Import the input modules in a fresh interpreter and check the labels module stays unloaded."""
    code = (
        "import sys; import govia_tracks; "
        "sys.exit(1 if 'govia_tracks.labels' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
