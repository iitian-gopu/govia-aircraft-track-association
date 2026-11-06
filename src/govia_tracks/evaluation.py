"""Metrics as defined by the clarifications. Labels enter here and nowhere on the inference path.

Link metrics compare a {predecessor: successor} mapping with labels over a whole scenario, so true links
lost during candidate generation still count against recall.
"""
from dataclasses import asdict, dataclass, field, replace

from .errors import InputValidationError, Issue


@dataclass
class LinkCounts:
    tracklets: int = 0
    labeled_links: int = 0
    accepted: int = 0
    correct: int = 0
    wrong_skips: int = 0  # same aircraft and forward along its chain, but not the immediate successor
    wrong_other_aircraft: int = 0  # both tracklets labeled, on different aircraft
    wrong_invalid: int = 0  # no valid result holds this: a self-link, backward along one aircraft, or a tracklet outside these labels
    labeled_unresolved: int = 0  # labeled successor, but nothing accepted
    candidate_links_retained: int = 0
    candidate_pairs: int = 0
    links_from_unlabeled: int = 0  # accepted, but the predecessor has no label: they can't reduce the unresolved count
    excluded_links: list = field(default_factory=list)  # (scenario, predecessor, successor) lost in candidate generation

    @property
    def wrong(self) -> int:
        return self.accepted - self.correct

    @property
    def unresolved_tracklets(self) -> int:
        """Labeled tracklets with no accepted successor."""
        return self.tracklets - (self.accepted - self.links_from_unlabeled)

    def add(self, other: "LinkCounts") -> None:
        for name, value in asdict(other).items():
            if name == "excluded_links":
                self.excluded_links.extend(other.excluded_links)
            else:
                setattr(self, name, getattr(self, name) + value)


def label_structure_issues(labels: dict[str, str | None]) -> list[Issue]:
    """What stops these labels being separate forward chains: a bad or unknown successor, a self-link, branching, or a cycle."""
    issues = []

    def report(message, tracklet):
        issues.append(Issue("invalid_structure", "label", message, record_id=tracklet))

    following = {}
    predecessors: dict[str, list[str]] = {}
    for tracklet in sorted(labels):
        successor = labels[tracklet]
        if successor is None:
            continue
        if not isinstance(successor, str) or not successor:
            report(f"tracklet '{tracklet}': successor must be null or a tracklet ID, got {successor!r}", tracklet)
        elif successor == tracklet:
            report(f"tracklet '{tracklet}' is its own successor: it follows itself", tracklet)
        elif successor not in labels:
            report(f"tracklet '{tracklet}': successor '{successor}' has no label row", tracklet)
        else:
            following[tracklet] = successor
            predecessors.setdefault(successor, []).append(tracklet)
    for successor, tracklets in sorted(predecessors.items()):
        if len(tracklets) > 1:
            report(f"tracklet '{successor}' has more than one predecessor: {', '.join(tracklets)}", successor)
    walked: dict[str, str] = {}
    for start in sorted(following):
        current = start
        while current in following and current not in walked:
            walked[current] = start
            current = following[current]
        if walked.get(current) == start:
            report(f"successors form a cycle through '{current}'", current)
    return issues


def validate_labels(labels: dict[str, str | None]) -> None:
    """Raise InputValidationError unless these labels are valid chains. The walks below rely on this to finish."""
    issues = label_structure_issues(labels)
    if issues:
        raise InputValidationError(issues)


def aircraft_groups(labels: dict[str, str | None]) -> dict[str, str]:
    """Group tracklets into chains from labels. For describing errors in evaluation only."""
    validate_labels(labels)
    predecessors = {b for b in labels.values() if b is not None}
    groups = {}
    for start in sorted(labels):
        if start not in predecessors:
            current = start
            while current is not None:
                groups[current] = start
                current = labels.get(current)
    return groups


def count_links(links: dict[str, str], labels: dict[str, str | None], candidate_pairs=None, scenario="") -> LinkCounts:
    """Count one scenario. `candidate_pairs` is the set of (predecessor, successor) pairs generated, if known."""
    counts = LinkCounts(tracklets=len(labels), accepted=len(links))
    true_links = {(a, b) for a, b in labels.items() if b is not None}
    places = {tracklet: (aircraft, position) for aircraft, chain in enumerate(aircraft_chains(labels)) for position, tracklet in enumerate(chain)}
    counts.labeled_links = len(true_links)
    for a, b in links.items():
        counts.links_from_unlabeled += a not in places
        if (a, b) in true_links:
            counts.correct += 1
        elif a not in places or b not in places:
            counts.wrong_invalid += 1
        elif places[a][0] != places[b][0]:
            counts.wrong_other_aircraft += 1
        elif places[a][1] < places[b][1]:
            counts.wrong_skips += 1
        else:
            counts.wrong_invalid += 1
    counts.labeled_unresolved = sum(1 for a, _ in true_links if a not in links)
    if candidate_pairs is not None:
        candidate_pairs = set(candidate_pairs)
        counts.candidate_pairs = len(candidate_pairs)
        counts.candidate_links_retained = len(true_links & candidate_pairs)
        counts.excluded_links = sorted((scenario, a, b) for a, b in true_links - candidate_pairs)
    return counts


def labels_without(labels: dict[str, str | None], dropped: set[str]) -> dict[str, str | None]:
    """Labels if some tracklets were never supplied: each remaining tracklet's next available one."""
    validate_labels(labels)
    remaining = {}
    for tracklet, successor in labels.items():
        if tracklet not in dropped:
            while successor in dropped:
                successor = labels[successor]
            remaining[tracklet] = successor
    return remaining


def _with_altitudes(scenario, altitude_of):
    return replace(scenario, tracklets=tuple(
        replace(tracklet, observations=tuple(replace(o, altitude_ft=altitude_of(o)) for o in tracklet.observations))
        for tracklet in scenario.tracklets
    ))


def hide_altitude(scenario):
    """A copy of a scenario with no altitude on any report, for evaluation-only ablation. The original is untouched."""
    return _with_altitudes(scenario, lambda _: None)


def noisy_altitude(scenario, sd_ft, rng):
    """A copy with Gaussian noise on every known altitude. Missing altitude stays missing."""
    return _with_altitudes(scenario, lambda o: None if o.altitude_ft is None else o.altitude_ft + rng.gauss(0, sd_ft))


def hide_altitude_of_some(scenario, probability, rng):
    """A copy where each tracklet independently loses all its altitude with this probability."""
    return replace(scenario, tracklets=tuple(
        replace(t, observations=tuple(replace(o, altitude_ft=None) for o in t.observations)) if rng.random() < probability else t
        for t in scenario.tracklets
    ))


def trim_reports(scenario, share):
    """A copy with this share of reports cut from each end of every tracklet, keeping at least three.
    Gaps widen continuously, moving true links into the 200-280 s band that the supplied data leaves empty."""
    def trimmed(tracklet):
        reports = tracklet.observations
        cut = int(len(reports) * share)
        while cut > 0 and len(reports) - 2 * cut < 3:
            cut -= 1
        return replace(tracklet, observations=reports[cut:len(reports) - cut])
    return replace(scenario, tracklets=tuple(trimmed(t) for t in scenario.tracklets))


def without_tracklets(scenario, dropped):
    """A copy of a scenario as if some tracklets were never supplied."""
    return replace(scenario, tracklets=tuple(t for t in scenario.tracklets if t.tracklet_id not in dropped))


def aircraft_chains(labels):
    """Each aircraft's tracklets in order, from labels. Harness and error description only."""
    validate_labels(labels)
    followers = {b for b in labels.values() if b is not None}
    chains = []
    for start in sorted(labels):
        if start not in followers:
            chain, current = [], start
            while current is not None:
                chain.append(current)
                current = labels[current]
            chains.append(chain)
    return chains


def outside_visible_windows(labels, rng):
    """Tracklets to drop so each aircraft is seen only for a contiguous run of fragments, entering and leaving coverage.

    This creates the case the supplied labels never contain: a tracklet with no true successor but real candidates,
    whose rival successor has lost the true predecessor that would otherwise have claimed it.
    """
    dropped = set()
    for chain in aircraft_chains(labels):
        first, last = sorted((rng.randint(0, len(chain)), rng.randint(0, len(chain))))
        if first == last:
            last = min(len(chain), first + 1)
            first = last - 1
        dropped |= set(chain[:first]) | set(chain[last:])
    return dropped


def random_fragments(labels, probability, rng):
    return {tracklet for tracklet in sorted(labels) if rng.random() < probability}


def split_tracklets(scenario, labels, gap_s, min_half=8):
    """Cut each long-enough tracklet in two around its middle, leaving a short gap. Returns (scenario, labels, links cut).

    The supplied data has no true link under 60 s, so this is the only way to see behaviour on short gaps.
    """
    tracklets, new_labels, cut = [], dict(labels), []
    for tracklet in scenario.tracklets:
        reports = tracklet.observations
        middle = reports[len(reports) // 2].observed_us
        half_gap = gap_s * 500_000
        before = tuple(o for o in reports if o.observed_us < middle - half_gap)
        after = tuple(o for o in reports if o.observed_us > middle + half_gap)
        if len(before) < min_half or len(after) < min_half:
            tracklets.append(tracklet)
            continue
        first, second = tracklet.tracklet_id + "#1", tracklet.tracklet_id + "#2"
        tracklets += [
            replace(tracklet, tracklet_id=first, observations=tuple(replace(o, tracklet_id=first) for o in before)),
            replace(tracklet, tracklet_id=second, observations=tuple(replace(o, tracklet_id=second) for o in after)),
        ]
        new_labels[first], new_labels[second] = second, new_labels.pop(tracklet.tracklet_id)
        cut.append((first, second))
    renamed = {pair[0][:-2]: pair[0] for pair in cut}
    return (
        replace(scenario, tracklets=tuple(tracklets)),
        {a: renamed.get(b, b) for a, b in new_labels.items()},
        cut,
    )


def _ratio(numerator, denominator):
    """A zero denominator is undefined, reported as None alongside its counts. Never a perfect score."""
    return numerator / denominator if denominator else None


def metrics(counts: LinkCounts) -> dict:
    return {
        "precision": _ratio(counts.correct, counts.accepted),
        "recall": _ratio(counts.correct, counts.labeled_links),
        "unresolved_proportion": _ratio(counts.labeled_unresolved, counts.labeled_links),
        "candidate_recall": _ratio(counts.candidate_links_retained, counts.labeled_links),
        "candidates_per_tracklet": _ratio(counts.candidate_pairs, counts.tracklets),
        "wrong": counts.wrong,
        "wrong_skips": counts.wrong_skips,
        "wrong_other_aircraft": counts.wrong_other_aircraft,
        "wrong_invalid": counts.wrong_invalid,
        "accepted": counts.accepted,
        "correct": counts.correct,
        "labeled_links": counts.labeled_links,
        "labeled_unresolved": counts.labeled_unresolved,
        "unresolved_tracklets": counts.unresolved_tracklets,
        "tracklets": counts.tracklets,
        "candidate_links_retained": counts.candidate_links_retained,
        "candidate_links_excluded": len(counts.excluded_links),
        "candidate_pairs": counts.candidate_pairs,
    }
