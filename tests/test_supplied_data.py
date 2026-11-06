"""The real package through the real loader. Only inputs are read; label files are not opened."""
import json

import pytest

from govia_tracks.loader import iter_partition, load_snapshot
from govia_tracks.movement import summarize_tracklet

pytestmark = pytest.mark.data

# From the package README, typed here so the check does not depend on the loader.
EXPECTED = {"train": (80, 1844, 24143), "validation": (20, 455, 6029), "test": (20, 459, 6059)}


@pytest.mark.parametrize("partition", EXPECTED)
def test_counts_match_the_package_readme(packet, partition):
    scenarios = list(iter_partition(packet / "data", partition))
    assert (
        len(scenarios),
        sum(len(s.tracklets) for s in scenarios),
        sum(s.quality.unique_observations for s in scenarios),
    ) == EXPECTED[partition]
    assert all(s.quality.supporting_without_observation == 0 for s in scenarios)


def test_every_training_tracklet_is_ordered_and_summarizes(packet):
    for scenario in iter_partition(packet / "data", "train"):
        for tracklet in scenario.tracklets:
            times = [o.observed_us for o in tracklet.observations]
            assert times == sorted(times)
            assert len(times) >= 2  # guaranteed by the data contract
            motion = summarize_tracklet(tracklet)
            assert motion.overall is not None and motion.end_us > motion.start_us


def test_staged_snapshots_respect_their_cutoffs_and_only_grow(packet):
    first = load_snapshot(packet / "data" / "staged_demo" / "snapshot_1")
    second = load_snapshot(packet / "data" / "staged_demo" / "snapshot_2")
    assert first.receipt_cutoff and second.receipt_cutoff and first.receipt_cutoff < second.receipt_cutoff
    assert first.input_fingerprint != second.input_fingerprint
    assert len(second.tracklets) == len(first.tracklets) + 1  # the contract: one additional tracklet
    earlier = {o.observation_id: o.raw for t in first.tracklets for o in t.observations}
    later = {o.observation_id: o.raw for t in second.tracklets for o in t.observations}
    assert earlier.keys() < later.keys()
    assert all(json.dumps(earlier[k], sort_keys=True) == json.dumps(later[k], sort_keys=True) for k in earlier)
