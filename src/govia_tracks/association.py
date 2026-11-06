"""Deterministic selection of links from scored candidates, shared by every association method.

A method supplies, for each candidate, a score where **higher means more plausible**, and whether
its measurements are good enough to accept at all. Selection knows nothing else about scoring.

Steps:
1. Withhold a pair A -> C when some B could sit between them (A -> B and B -> C are both plausible).
   That link would skip a fragment, and C isn't a rival to B: they could be the same aircraft in sequence.
   This holds even if A -> C outscores A -> B and B -> C is too weak to accept. Links in that position were measured on
   stress cases without heavy altitude noise: 130 of 135 right on train, 7 of 10 on validation (a small sample). Blocking
   them pays at a 5:1 cost only if more than 1 in 6 are wrong, so they are kept.
2. Walk the remaining pairs from highest score to lowest. Accept a pair only if it clears the minimum
   score, its endpoints are free, and it beats every remaining rival on either end by the margin.
   A pair that fails stays a rival, so a close contest leaves both sides unresolved instead of
   promoting the runner-up.

Tracklet IDs break exact score ties so results are repeatable; they carry no evidence. Under any policy
with a positive margin, tied rivals stay unresolved. With a margin of zero, the first by ID is accepted.
"""
from dataclasses import asdict, dataclass

ACCEPTED = "accepted"
UNRESOLVED = "unresolved"

NO_CANDIDATE = "no_candidate"
MISSING_MEASUREMENT = "missing_measurement"
PHYSICALLY_INCONSISTENT = "physically_inconsistent"
BELOW_THRESHOLD = "below_threshold"
POSSIBLE_INTERMEDIATE = "possible_intermediate"
CLOSE_RIVAL = "close_rival"
SUCCESSOR_TAKEN = "successor_taken"
PREDECESSOR_TAKEN = "predecessor_taken"


@dataclass(frozen=True)
class AcceptancePolicy:
    min_score: float  # a pair below this is never accepted
    min_margin: float  # required lead over the best remaining rival
    plausible_score: float  # a pair at or above this counts as a possible step in a chain


@dataclass(frozen=True)
class ScoredCandidate:
    predecessor: str
    successor: str
    score: float
    measurements: dict
    acceptable: bool = True  # False when critical measurements are missing; still a rival, because it may be real
    ruled_out: str | None = None  # why physical rules exclude this pair; then it's neither accepted, nor a rival, nor a step


@dataclass(frozen=True)
class CandidateDecision:
    code: str
    detail: str


def _order(candidate):
    return (-candidate.score, candidate.predecessor, candidate.successor)


def _key(candidate):
    return (candidate.predecessor, candidate.successor)


def select_links(candidates, policy: AcceptancePolicy) -> tuple[dict[str, str], dict[tuple[str, str], CandidateDecision]]:
    """Return ({predecessor: successor}, {(predecessor, successor): decision for every candidate})."""
    plausible = {_key(c) for c in candidates if c.score >= policy.plausible_score and c.ruled_out is None}
    successors: dict[str, list] = {}
    for c in candidates:
        successors.setdefault(c.predecessor, []).append(c)

    decisions: dict[tuple[str, str], CandidateDecision] = {}
    contenders = []
    for c in sorted(candidates, key=_order):
        if c.ruled_out is not None:
            decisions[_key(c)] = CandidateDecision(PHYSICALLY_INCONSISTENT, c.ruled_out)
            continue
        between = sorted(
            step.successor for step in successors[c.predecessor]
            if _key(step) in plausible and (step.successor, c.successor) in plausible
        )
        if between:
            decisions[_key(c)] = CandidateDecision(POSSIBLE_INTERMEDIATE, f"{between[0]} could come between them")
        else:
            contenders.append(c)

    links: dict[str, str] = {}
    taken_successors: dict[str, str] = {}
    for c in contenders:
        if c.predecessor in links:
            decisions[_key(c)] = CandidateDecision(PREDECESSOR_TAKEN, f"{c.predecessor} already links to {links[c.predecessor]}")
        elif c.successor in taken_successors:
            decisions[_key(c)] = CandidateDecision(SUCCESSOR_TAKEN, f"{c.successor} already follows {taken_successors[c.successor]}")
        elif not c.acceptable:
            decisions[_key(c)] = CandidateDecision(MISSING_MEASUREMENT, "critical measurements are missing")
        elif c.score < policy.min_score:
            decisions[_key(c)] = CandidateDecision(BELOW_THRESHOLD, f"score {c.score:.3f} is below {policy.min_score:.3f}")
        else:
            rival = next((
                r for r in contenders
                if r is not c and (r.predecessor == c.predecessor or r.successor == c.successor)
                and r.predecessor not in links and r.successor not in taken_successors
            ), None)  # contenders are sorted, so the first match is the strongest
            if rival is not None and c.score - rival.score < policy.min_margin:
                decisions[_key(c)] = CandidateDecision(
                    CLOSE_RIVAL,
                    f"{rival.predecessor} -> {rival.successor} scores {rival.score:.3f} against {c.score:.3f}; "
                    f"needs a lead of {policy.min_margin:.3f}",
                )
            else:
                links[c.predecessor] = c.successor
                taken_successors[c.successor] = c.predecessor
                decisions[_key(c)] = CandidateDecision(
                    ACCEPTED, "no competing candidate" if rival is None else
                    f"leads {rival.predecessor} -> {rival.successor} by {c.score - rival.score:.3f}",
                )
    return links, decisions


def check_links(links: dict[str, str], motions: dict) -> None:
    """Raise if accepted links break a required invariant. Run on every result, not just in tests."""
    if len(set(links.values())) != len(links):
        raise ValueError("a tracklet has more than one accepted predecessor")
    for a, b in links.items():
        if a == b:
            raise ValueError(f"self-link on {a}")
        if a not in motions or b not in motions:
            raise ValueError(f"link {a} -> {b} refers to a tracklet outside this scenario")
        if motions[b].start_us <= motions[a].end_us:
            raise ValueError(f"link {a} -> {b} does not move forward in time")
    for start in links:
        seen, current = {start}, links[start]
        while current in links:
            if current in seen:
                raise ValueError(f"cycle through {current}")
            seen.add(current)
            current = links[current]


_REASONS = {
    NO_CANDIDATE: "no plausible candidate",
    MISSING_MEASUREMENT: "evidence is insufficient",
    PHYSICALLY_INCONSISTENT: "the best candidate fails a physical check",
    BELOW_THRESHOLD: "evidence is insufficient",
    POSSIBLE_INTERMEDIATE: "the best candidate may skip a fragment",
    CLOSE_RIVAL: "several candidates are similarly plausible",
    SUCCESSOR_TAKEN: "the best candidate has a stronger accepted predecessor",
    PREDECESSOR_TAKEN: "already linked",
}


def build_result(method, version, config, scenario, candidate_set, scored, policy) -> dict:
    """Assemble the JSON-serializable analysis result. Any method producing ScoredCandidates can use this."""
    links, decisions = select_links(scored, policy)
    check_links(links, candidate_set.motions)
    by_predecessor: dict[str, list] = {}
    for c in scored:
        by_predecessor.setdefault(c.predecessor, []).append(c)

    tracklets = []
    for tracklet_id in sorted(candidate_set.motions):
        ranked = sorted(by_predecessor.get(tracklet_id, []), key=_order)
        accepted = links.get(tracklet_id)
        if accepted is not None:
            detail = decisions[(tracklet_id, accepted)].detail
            code, reason = ACCEPTED, f"accepted {accepted}: {detail}"
        elif not ranked:
            code, reason = NO_CANDIDATE, _REASONS[NO_CANDIDATE]
        else:
            decision = decisions[_key(ranked[0])]
            code, reason = decision.code, f"{_REASONS[decision.code]}: {ranked[0].successor}: {decision.detail}"
        tracklets.append({
            "tracklet_id": tracklet_id,
            "status": ACCEPTED if accepted is not None else UNRESOLVED,
            "accepted_successor": accepted,
            "reason_code": code,
            "reason": reason,
            "candidates": [
                {
                    "rank": rank,
                    "successor": c.successor,
                    "score": c.score,
                    "decision": decisions[_key(c)].code,
                    "decision_detail": decisions[_key(c)].detail,
                    "measurements": c.measurements,
                }
                for rank, c in enumerate(ranked, start=1)
            ],
            "gated_out": candidate_set.gated_out.get(tracklet_id, {}),
        })
    return {
        "method": method,
        "version": version,
        "config": config,
        "policy": asdict(policy),
        "scenario_id": scenario.scenario_id,
        "snapshot_id": scenario.snapshot_id,
        "receipt_cutoff": scenario.receipt_cutoff,
        "input_fingerprint": scenario.input_fingerprint,
        "links": dict(sorted(links.items())),
        "tracklets": tracklets,
    }
