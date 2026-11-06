"""Turn the raw records of one scenario into validated, deduplicated, ordered tracklets.

The result depends only on the set of distinct records, not on row order or identical
retransmissions. Any invalid or conflicting record rejects the whole input.
"""
import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

from .errors import InputValidationError, Issue
from .schema import (
    Observation,
    SupportingRecord,
    json_equal,
    parse_observation,
    parse_supporting,
    parse_utc_timestamp,
)


@dataclass(frozen=True)
class Tracklet:
    tracklet_id: str
    observations: tuple[Observation, ...]  # by observation time; ties by observation_id, for ordering only


@dataclass(frozen=True)
class DataQuality:
    """Structurally valid but noteworthy facts. Counted, never corrected."""

    observation_rows: int
    unique_observations: int
    supporting_rows: int
    unique_supporting: int
    altitude_null: int
    altitude_absent: int
    received_before_observed: int
    tracklets_delivered_out_of_order: int
    single_report_tracklets: int
    supporting_without_observation: int


@dataclass(frozen=True)
class PreparedScenario:
    tracklets: tuple[Tracklet, ...]  # by tracklet_id, for deterministic iteration only
    supporting: dict[str, tuple[SupportingRecord, ...]]  # keyed by observation_id
    unjoined_supporting: tuple[SupportingRecord, ...]
    quality: DataQuality
    input_fingerprint: str  # identical for reordered or retransmitted inputs
    scenario_id: str | None = None
    snapshot_id: str | None = None
    receipt_cutoff: str | None = None


def _dedupe(rows, id_field, kind, issues):
    """Compare original records. Rows without a usable ID pass through for validation to report."""
    first_seen: dict[str, int] = {}
    unique: list[tuple[int, object]] = []
    for index, row in enumerate(rows):
        record_id = row.get(id_field) if isinstance(row, dict) else None
        if not isinstance(record_id, str) or not record_id.strip():
            unique.append((index, row))
            continue
        if record_id not in first_seen:
            first_seen[record_id] = index
            unique.append((index, row))
        elif not json_equal(rows[first_seen[record_id]], row):
            issues.append(Issue(
                "conflicting_id", kind,
                f"{kind} rows {first_seen[record_id]} and {index} share {id_field} '{record_id}' with different content",
                index, record_id, id_field,
            ))
    return unique


def _canonical(value):
    """Write whole-number floats as integers, so records that `json_equal` accepts hash alike."""
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _fingerprint(observations, supporting):
    content = {
        "observations": sorted((o.raw for o in observations), key=lambda r: r["observation_id"]),
        "supporting": sorted((s.raw for s in supporting), key=lambda r: r["record_id"]),
    }
    return hashlib.sha256(json.dumps(_canonical(content), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_scenario(
    observation_rows: Iterable,
    supporting_rows: Iterable = (),
    *,
    scenario_id: str | None = None,
    snapshot_id: str | None = None,
    receipt_cutoff: str | None = None,
) -> PreparedScenario:
    """Validate one scenario. Labels are not an input here, by design."""
    observation_rows, supporting_rows = list(observation_rows), list(supporting_rows)
    issues: list[Issue] = []

    cutoff_us = None
    if receipt_cutoff is not None:
        try:
            cutoff_us = parse_utc_timestamp(receipt_cutoff)
        except ValueError as exc:
            issues.append(Issue("invalid_field", "snapshot", f"receipt_cutoff is invalid: {exc}", field="receipt_cutoff"))

    observations: list[Observation] = []
    for index, row in _dedupe(observation_rows, "observation_id", "observation", issues):
        parsed = parse_observation(row, index, issues)
        if parsed is not None:
            observations.append(parsed)
    supporting: list[SupportingRecord] = []
    for index, row in _dedupe(supporting_rows, "record_id", "supporting", issues):
        parsed = parse_supporting(row, index, issues)
        if parsed is not None:
            supporting.append(parsed)

    if cutoff_us is not None:
        for kind, records, id_attr in (("observation", observations, "observation_id"), ("supporting", supporting, "record_id")):
            for record in records:
                if record.received_us > cutoff_us:
                    record_id = getattr(record, id_attr)
                    issues.append(Issue(
                        "after_cutoff", kind,
                        f"{kind} '{record_id}' was received after the snapshot cutoff {receipt_cutoff}",
                        record_id=record_id, field="received_at",
                    ))
    if issues:
        raise InputValidationError(issues)

    delivered: dict[str, list[Observation]] = {}
    for observation in observations:
        delivered.setdefault(observation.tracklet_id, []).append(observation)
    tracklets, out_of_order = [], 0
    for tracklet_id in sorted(delivered):
        ordered = sorted(delivered[tracklet_id], key=lambda o: (o.observed_us, o.observation_id))
        out_of_order += [o.observation_id for o in ordered] != [o.observation_id for o in delivered[tracklet_id]]
        tracklets.append(Tracklet(tracklet_id, tuple(ordered)))

    known = {o.observation_id for o in observations}
    joined: dict[str, list[SupportingRecord]] = {}
    unjoined = []
    for record in sorted(supporting, key=lambda s: s.record_id):
        if record.observation_id in known:
            joined.setdefault(record.observation_id, []).append(record)
        else:
            unjoined.append(record)

    quality = DataQuality(
        observation_rows=len(observation_rows),
        unique_observations=len(observations),
        supporting_rows=len(supporting_rows),
        unique_supporting=len(supporting),
        altitude_null=sum("altitude_ft" in o.raw and o.raw["altitude_ft"] is None for o in observations),
        altitude_absent=sum("altitude_ft" not in o.raw for o in observations),
        received_before_observed=sum(o.received_us < o.observed_us for o in observations),
        tracklets_delivered_out_of_order=out_of_order,
        single_report_tracklets=sum(len(t.observations) < 2 for t in tracklets),
        supporting_without_observation=len(unjoined),
    )
    return PreparedScenario(
        tracklets=tuple(tracklets),
        supporting={key: tuple(value) for key, value in joined.items()},
        unjoined_supporting=tuple(unjoined),
        quality=quality,
        input_fingerprint=_fingerprint(observations, supporting),
        scenario_id=scenario_id,
        snapshot_id=snapshot_id,
        receipt_cutoff=receipt_cutoff,
    )
