"""Pure interpretation of API results: plain wording, table rows, summaries and comparisons.

Nothing here scores or decides; it only reads what the API returned. Numbers are never compared across
methods, because the two scales share no meaning. Missing always renders as "not available", never 0.
"""
import json
import math
import re

from dashboard.charts import NOT_AVAILABLE, short_id, time_key

NO_PLAUSIBLE = "No plausible candidate"
INSUFFICIENT = "Insufficient evidence"
SIMILAR = "Similar competing candidates"
CONFLICT = "Association conflict"
POLICY = "Rejected by acceptance policy"
UNRESOLVED_CATEGORIES = (NO_PLAUSIBLE, INSUFFICIENT, SIMILAR, CONFLICT, POLICY)

# code -> (category, plain meaning, short candidate-table wording). Categories follow the API's own reason wording.
_CODES = {
    "accepted": ("Accepted", "A successor cleared the score and margin requirements.", "Accepted"),
    "no_candidate": (NO_PLAUSIBLE, "No later tracklet passed the physical gates, so nothing could be scored.", "No candidate"),
    "missing_measurement": (INSUFFICIENT, "The best candidate lacks a measurement that acceptance requires.", "Critical measurement missing"),
    "below_threshold": (INSUFFICIENT, "The best candidate's score is below this method's minimum.", "Score below minimum"),
    "close_rival": (SIMILAR, "Another candidate scores too closely, so choosing either would be a guess.", "Close rival"),
    "successor_taken": (CONFLICT, "The best candidate already follows a tracklet with a stronger link.", "Successor already taken"),
    "predecessor_taken": (CONFLICT, "This tracklet already has a stronger accepted link.", "Predecessor already linked"),
    "possible_intermediate": (POLICY, "Another fragment could sit between them, so this link might skip it.", "May skip a fragment"),
    "physically_inconsistent": (POLICY, "The best candidate fails a physical rule of the method.", "Fails a physical check"),
}

SCALE_NOTES = {
    "baseline": "Baseline score: 0 is a perfect fit and there is no lower bound. Higher is more plausible. It is not a probability.",
    "learned": "Learned score: between 0 and 1, uncalibrated, and low on long gaps by design. Higher is more plausible. Do not read it as a probability of being correct.",
}
SCALES_DIFFER = (
    "The two scores use different scales, so never compare numbers across methods. Baseline: 0 is best, unbounded below. "
    "Learned: 0 to 1, uncalibrated, low on long gaps by design. Neither is a confidence value. Compare decisions, not scores."
)
TRAIN_NOTE = "The saved model was fitted on the train partition, so learned results on this scenario are in-sample and look better than they would on unseen data."
UNRESOLVED_NOTE = (
    "Unresolved does not mean the aircraft stopped. It means no successor could be accepted from the evidence in this "
    "snapshot; the tracklet may also simply be the last available fragment."
)
REPLACEMENT_NOTE = (
    "A replaced link can reflect a newly available intermediate fragment, not a change of identity: A → C can become "
    "A → B and B → C once B arrives, and the links describe the immediate next tracklet, not an aircraft."
)
GATE_LABELS = {
    "gap_too_long": "Gap too long",
    "implied_speed_too_high": "Implied speed too high",
    "projection_miss_too_large": "Projection miss too large",
    "not_after_predecessor": "Does not start after this tracklet ends",
}
_TRACKLET_ID = re.compile(r"tracklet-([0-9a-fA-F]{8})(?:-[0-9a-fA-F]+)*")


def reason_category(code) -> str:
    entry = _CODES.get(code)
    return entry[0] if entry else f"Unrecognised reason code: {code}"


def reason_meaning(code) -> str:
    entry = _CODES.get(code)
    return entry[1] if entry else f"The API returned a reason code this dashboard does not know: {code}."


def decision_label(code) -> str:
    entry = _CODES.get(code)
    return entry[2] if entry else str(code)


def shorten_ids(text) -> str:
    """Shorten any full tracklet IDs inside API wording, for tables where the full ID would crowd out the message."""
    return _TRACKLET_ID.sub(r"\1", str(text))


def fmt(value, digits: int = 2) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return NOT_AVAILABLE
    return f"{value:,.{digits}f}"


def _raw(value) -> str:
    return NOT_AVAILABLE if value is None else str(value)


def _attributes(value) -> str:
    return NOT_AVAILABLE if value is None else json.dumps(value, sort_keys=True, ensure_ascii=False)


def by_id(result: dict) -> dict[str, dict]:
    return {t["tracklet_id"]: t for t in result.get("tracklets", [])}


def summarize(result: dict) -> dict[str, int]:
    tracklets = result.get("tracklets", [])
    accepted = sum(t.get("status") == "accepted" for t in tracklets)
    return {
        "tracklets": len(tracklets), "accepted": accepted, "unresolved": len(tracklets) - accepted,
        "candidates": sum(len(t.get("candidates") or []) for t in tracklets),
    }


# The measurements that most often decide a pair come first, so a narrow window hides the least useful ones.
_MEASUREMENT_COLUMNS = (
    ("Gap (s)", "gap_s", 1), ("Fwd miss (km)", "forward_miss_km", 2), ("Back miss (km)", "backward_miss_km", 2),
    ("Alt miss (ft)", "altitude_miss_ft", 0), ("Alt diff (ft)", "altitude_diff_ft", 0), ("Distance (km)", "endpoint_km", 2),
    ("Speed diff (km/h)", "speed_diff_kmh", 1), ("Course diff (deg)", "course_diff_deg", 1),
)
MEASUREMENT_NOTE = (
    "Fwd miss: the distance from this tracklet's projected position to the candidate's start. Back miss: the same, projecting the candidate "
    "backward. Alt miss: the error of projecting altitude across the gap. Alt diff: the altitude change across it. Distance: between the two "
    "facing endpoints."
)


def _measurement_cells(measurements, columns=_MEASUREMENT_COLUMNS) -> dict[str, str]:
    measurements = measurements or {}
    return {title: fmt(measurements.get(field), digits) for title, field, digits in columns}


def candidate_rows(tracklet: dict, compact: bool = False) -> list[dict]:
    columns = _MEASUREMENT_COLUMNS[:4] if compact else _MEASUREMENT_COLUMNS
    rows = []
    for candidate in sorted(tracklet.get("candidates") or [], key=lambda c: c.get("rank", 0)):
        rows.append({
            "Rank": _raw(candidate.get("rank")), "Successor": short_id(candidate.get("successor")), "Score": fmt(candidate.get("score"), 3),
            "Decision": decision_label(candidate.get("decision")), **_measurement_cells(candidate.get("measurements"), columns),
        })
    return rows


def gated_rows(tracklet: dict) -> list[dict]:
    return [
        {"Gate": GATE_LABELS.get(gate, gate), "Pairs removed": str(count)}
        for gate, count in sorted((tracklet.get("gated_out") or {}).items())
    ]


def status_line(tracklet: dict) -> str:
    if tracklet.get("status") == "accepted":
        return f"Accepted. Next tracklet: {short_id(tracklet.get('accepted_successor'))}"
    return f"Unresolved. {reason_category(tracklet.get('reason_code'))}"


def explain_decision(tracklet: dict) -> str:
    """Why the top candidate won or was withheld, using only the API's decisions."""
    candidates = sorted(tracklet.get("candidates") or [], key=lambda c: c.get("rank", 0))
    if not candidates:
        return reason_meaning("no_candidate")
    top = candidates[0]
    top_text = f"Rank {top.get('rank')} candidate {short_id(top.get('successor'))}"
    detail = shorten_ids(top.get("decision_detail") or NOT_AVAILABLE)
    accepted = tracklet.get("accepted_successor")
    if accepted is None:
        return f"{top_text} was withheld: {decision_label(top.get('decision')).lower()} ({detail}). {reason_meaning(top.get('decision'))}"
    if top.get("successor") == accepted:
        return f"{top_text} was accepted: {detail}."
    chosen = next((c for c in candidates if c.get("successor") == accepted), None)
    chosen_text = f"Rank {chosen.get('rank')} candidate" if chosen else "Candidate"
    return f"{chosen_text} {short_id(accepted)} was accepted. {top_text} was not: {decision_label(top.get('decision')).lower()} ({detail})."


def policy_text(result: dict) -> str:
    policy = result.get("policy") or {}
    return (
        f"Acceptance policy for {result.get('method', 'this method')}: score at least {fmt(policy.get('min_score'), 3)}, "
        f"lead over the best remaining rival at least {fmt(policy.get('min_margin'), 3)}."
    )


def _decision_text(tracklet: dict | None) -> str:
    if tracklet is None:
        return NOT_AVAILABLE
    if tracklet.get("status") == "accepted":
        return f"Accepted: {short_id(tracklet.get('accepted_successor'))}"
    return f"Unresolved: {reason_category(tracklet.get('reason_code'))}"


def method_comparison(baseline: dict, learned: dict) -> list[dict]:
    """Per-tracklet decisions. Agreement means the same successor, or both unresolved (reasons may differ)."""
    left, right = by_id(baseline), by_id(learned)
    rows = []
    for tracklet_id in sorted(set(left) | set(right)):
        a, b = left.get(tracklet_id), right.get(tracklet_id)
        agree = a is not None and b is not None and a.get("accepted_successor") == b.get("accepted_successor")
        rows.append({"tracklet_id": tracklet_id, "baseline": _decision_text(a), "learned": _decision_text(b), "agree": agree})
    return rows


def agreement_counts(baseline: dict, learned: dict) -> dict[str, int]:
    rows = method_comparison(baseline, learned)
    agree = sum(row["agree"] for row in rows)
    return {"agree": agree, "disagree": len(rows) - agree}


def unresolved_groups(result: dict) -> dict[str, list[dict]]:
    """{category: tracklets}, in the fixed category order, then any unknown codes. Categories without tracklets are absent."""
    groups: dict[str, list[dict]] = {}
    for tracklet in result.get("tracklets", []):
        if tracklet.get("status") != "accepted":
            groups.setdefault(reason_category(tracklet.get("reason_code")), []).append(tracklet)
    order = {name: i for i, name in enumerate(UNRESOLVED_CATEGORIES)}
    return dict(sorted(groups.items(), key=lambda item: (order.get(item[0], len(order)), item[0])))


def unresolved_rows(tracklets) -> list[dict]:
    rows = []
    for tracklet in tracklets:
        candidates = sorted(tracklet.get("candidates") or [], key=lambda c: c.get("rank", 0))
        rows.append({
            "Tracklet": short_id(tracklet.get("tracklet_id")), "Reason code": str(tracklet.get("reason_code")),
            "Top candidate": short_id(candidates[0].get("successor")) if candidates else "none",
            "Detail": shorten_ids(tracklet.get("reason") or NOT_AVAILABLE),
        })
    return rows


def _candidate_for(tracklet: dict | None, successor) -> dict | None:
    for candidate in (tracklet or {}).get("candidates") or []:
        if candidate.get("successor") == successor:
            return candidate
    return None


def compare_snapshots(before: dict, after: dict) -> dict:
    """What changed in one method's links from an earlier snapshot's result to a later one's.

    Each result comes from its own independent analysis; this only diffs the outputs. A predecessor's link is
    added (none, then one), withdrawn (one, then none), replaced (a different successor) or unchanged.
    Supporting candidates are taken from the snapshot that produced that link.
    """
    before_tracklets, after_tracklets = by_id(before), by_id(after)
    before_links, after_links = before.get("links") or {}, after.get("links") or {}
    new = sorted(set(after_tracklets) - set(before_tracklets))
    added, withdrawn, replaced, unchanged = [], [], [], []
    for predecessor in sorted(set(before_links) | set(after_links)):
        old, current = before_links.get(predecessor), after_links.get(predecessor)
        old_evidence = _candidate_for(before_tracklets.get(predecessor), old)
        new_evidence = _candidate_for(after_tracklets.get(predecessor), current)
        if old == current:
            unchanged.append({"predecessor": predecessor, "successor": current, "before": old_evidence, "after": new_evidence})
        elif old is None:
            added.append({"predecessor": predecessor, "successor": current, "after": new_evidence})
        elif current is None:
            withdrawn.append({
                "predecessor": predecessor, "successor": old, "before": old_evidence,
                "reason_code": after_tracklets.get(predecessor, {}).get("reason_code"),
                "reason": after_tracklets.get(predecessor, {}).get("reason"),
            })
        else:
            replaced.append({
                "predecessor": predecessor, "old_successor": old, "new_successor": current, "before": old_evidence, "after": new_evidence,
                "new_successor_is_new": current in new, "new_tracklets_between": _new_tracklets_between(after_links, current, old, set(new)),
            })
    return {
        "new_tracklets": new, "removed_tracklets": sorted(set(before_tracklets) - set(after_tracklets)),
        "added": added, "withdrawn": withdrawn, "replaced": replaced, "unchanged": unchanged,
        "changed": bool(new or added or withdrawn or replaced or set(before_tracklets) - set(after_tracklets)),
    }


def _new_tracklets_between(links: dict, start, end, new: set) -> int:
    """How many new tracklets the links run through from `start` to `end`. 0 if that path doesn't exist. Stops on a loop."""
    count, current, seen = 0, start, set()
    while current in new and current not in seen:
        seen.add(current)
        count += 1
        current = links.get(current)
    return count if current == end else 0


def _change_row(change: str, predecessor, successor, evidence: dict | None, source: str, note: str) -> dict:
    evidence = evidence or {}
    return {
        "Change": change, "Link": f"{short_id(predecessor)} → {short_id(successor)}", "Evidence from": source,
        "Score": fmt(evidence.get("score"), 3), **_measurement_cells(evidence.get("measurements")), "Note": note,
    }


def change_rows(changes: dict) -> list[dict]:
    """One row per link involved in a change, with the measurements from the snapshot that produced it."""
    rows = []
    for item in changes["added"]:
        note = "Involves the new tracklet." if {item["predecessor"], item["successor"]} & set(changes["new_tracklets"]) else ""
        rows.append(_change_row("Added", item["predecessor"], item["successor"], item.get("after"), "Snapshot 2", note))
    for item in changes["withdrawn"]:
        reason = item.get("reason_code")
        note = "The tracklet is not in snapshot 2." if reason is None else f"Now unresolved: {reason_category(reason)}."
        rows.append(_change_row("Withdrawn", item["predecessor"], item["successor"], item.get("before"), "Snapshot 1", note))
    for item in changes["replaced"]:
        if item["new_tracklets_between"] == 1:
            note = "The new tracklet now sits between the two previously linked tracklets."
        elif item["new_tracklets_between"] > 1:
            note = f"{item['new_tracklets_between']} new tracklets now sit between the two previously linked tracklets."
        elif item["new_successor_is_new"]:
            note = "The new successor is the newly available tracklet."
        else:
            note = "The new successor was already available in snapshot 1."
        rows.append(_change_row("Replaced: earlier link", item["predecessor"], item["old_successor"], item.get("before"), "Snapshot 1", note))
        rows.append(_change_row("Replaced: new link", item["predecessor"], item["new_successor"], item.get("after"), "Snapshot 2", note))
    return rows


def observation_rows(observations, supporting, tracklet_id: str) -> list[dict]:
    """A tracklet's original reports in time order. Identical retransmissions collapse into a copy count."""
    copies: dict[str, int] = {}
    rows: dict[str, dict] = {}
    for row in observations:
        if row.get("tracklet_id") == tracklet_id:
            canonical = json.dumps(row, sort_keys=True, default=str)
            copies[canonical] = copies.get(canonical, 0) + 1
            rows.setdefault(canonical, row)
    counts: dict = {}
    for row in {r.get("record_id"): r for r in supporting}.values():
        counts[row.get("observation_id")] = counts.get(row.get("observation_id"), 0) + 1
    ordered = sorted(rows.items(), key=lambda item: (time_key(item[1].get("observed_at")), str(item[1].get("observation_id"))))
    return [
        {
            "Observation ID": _raw(row.get("observation_id")), "Observed at": _raw(row.get("observed_at")), "Received at": _raw(row.get("received_at")),
            "Latitude": _raw(row.get("latitude")), "Longitude": _raw(row.get("longitude")), "Altitude (ft)": _raw(row.get("altitude_ft")),
            "Source": _raw(row.get("source")), "Attributes": _attributes(row.get("attributes")),
            "Copies received": str(copies[canonical]), "Supporting records": str(counts.get(row.get("observation_id"), 0)),
        }
        for canonical, row in ordered
    ]


def supporting_rows(observations, supporting, tracklet_id: str) -> list[dict]:
    """Supporting records joined to this tracklet's observations, one per record_id, in receipt order."""
    ids = {row.get("observation_id") for row in observations if row.get("tracklet_id") == tracklet_id}
    unique = {row.get("record_id"): row for row in supporting if row.get("observation_id") in ids}
    ordered = sorted(unique.values(), key=lambda r: (time_key(r.get("received_at")), str(r.get("record_id"))))
    return [
        {
            "Record ID": _raw(row.get("record_id")), "Observation ID": _raw(row.get("observation_id")), "Source": _raw(row.get("source")),
            "Received at": _raw(row.get("received_at")), "Attributes": _attributes(row.get("attributes")),
        }
        for row in ordered
    ]
