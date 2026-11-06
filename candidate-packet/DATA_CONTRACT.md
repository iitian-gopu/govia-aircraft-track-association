# Data Contract and Simplified Assumptions

## Scenarios and partitions

Each scenario is independent. Candidate links must stay within one scenario. Aircraft and scenarios do not cross training, validation, test, demo, or reviewer-held-out boundaries. Source types are intentionally shared across partitions.

Each tracklet belongs to exactly one aircraft and contains at least two observations after deduplication. Same-aircraft tracklets do not overlap. Different aircraft can be very close, move similarly, and have overlapping observation intervals. No real aircraft identity is supplied.

The synthetic motion is smooth, subsonic movement over a bounded region. This dataset does not model takeoffs, landings, weather, aircraft-specific flight envelopes, realistic global coverage, or every possible maneuver. Local planar approximations or standard geodesic calculations are acceptable; document your choice.

Noise is applied to measured positions. Occasional isolated errors can be larger. Sampling is irregular, some interior reports are absent, altitude is incomplete, and optional source measurements can be misleading. Reports and supporting records are shuffled and contain identical retransmissions. The main datasets contain no intentionally conflicting IDs or invalid required fields; those conditions are isolated in fixtures.

## Files and snapshot metadata

- `observations.jsonl`: core reports with source-specific attributes.
- `supporting.jsonl`: optional records joined by observation_id.
- `labels.jsonl`: one row per available tracklet, including null successors.
- `snapshot.json`: snapshot ID, scenario ID, partition, receipt cutoff, file paths, and counts.

A null receipt_cutoff in ordinary partitions means a completed batch containing all listed records. It does not impose an implicit live-inference restriction. Staged snapshots have explicit receipt cutoffs.

## Labels

Example format:

```json
{"tracklet_id":"tracklet-example-a","successor_tracklet_id":"tracklet-example-b"}
{"tracklet_id":"tracklet-example-b","successor_tracklet_id":null}
```

For a given snapshot, the labeled successor is the next available tracklet from the same underlying aircraft in observation-time order. Null means no later same-aircraft tracklet exists in that snapshot. It does not mean the aircraft ceased moving.

Ground-truth physical paths and complete aircraft IDs are not provided to candidates. Labels are supervision for training and evaluation, never inference features.

For all reported metrics, count exact directed successor links. A same-aircraft link that skips an available intermediate tracklet is incorrect. Include all labeled links in recall, even those excluded by candidate generation. See the clarifications for metric definitions.

## Staged demonstration

Both staged folders are complete snapshots, not deltas. Snapshot 2 preserves all records from snapshot 1 and adds delayed evidence. Only one additional tracklet is introduced, along with its supporting records. No previously supplied record is overwritten.

Use each folder's own inputs and labels. Do not union both snapshots before computing first-stage features. A newly available intermediate fragment can replace one direct successor link with two links without changing the inferred aircraft identity.

At least one existing tracklet's ground-truth immediate successor differs between snapshots. This does not guarantee that a particular model will change its output. Explain the actual behavior.

All records in a staged snapshot have received_at at or before its cutoff, including supporting records. The fixed models must not train or retune on either staged snapshot.

## Correctness fixtures

Files under fixtures are independent test inputs, excluded from training and evaluation:

- `identical_duplicate.jsonl`: two identical copies of a valid observation; keep one.
- `conflicting_observation_id.jsonl`: two different reports sharing an observation ID; reject the conflict.
- `invalid_coordinate.jsonl`: an out-of-range latitude; reject it.

These small fixtures do not replace the automated tests requested by the challenge.

## Limits of this dataset

This is a hiring exercise, not a real-world performance benchmark. Synthetic motion and source behavior are simplified. Identifiability varies: some pairs are deliberately difficult given available measurements. Unresolved decisions are valid, but must be weighed against missed correct links.
