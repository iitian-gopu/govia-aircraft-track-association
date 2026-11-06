"""Candidate successor pairs and their physical measurements, from one scenario's snapshot.

Shared by every association method. Works from validated tracklets only: labels are not imported,
and tracklet IDs are used only to refer to tracklets and fix iteration order.

Gates are deliberately loose: a pair removed here can't be recovered by any later step, so they
remove only what is physically implausible and leave close calls to the acceptance policy.
Altitude is not gated, so a poor altitude reading can't remove a true link.

Motion comes from the whole-tracklet fit. On training data it projected across gaps far better than
the fit over a tracklet's last third (true-link miss at p95: 0.54 km against 3.38 km).
"""
from dataclasses import asdict, dataclass

from .movement import TrackletMotion, angle_difference_deg, haversine_km, project_position, summarize_tracklet
from .preprocess import PreparedScenario

GATE_TIME_ORDER = "not_after_predecessor"
GATE_MAX_GAP = "gap_too_long"
GATE_MAX_SPEED = "implied_speed_too_high"
GATE_MAX_PROJECTION = "projection_miss_too_large"


@dataclass(frozen=True)
class CandidateConfig:
    max_gap_s: float = 600.0  # straight-line projection is not credible beyond this; longest true link on train is 365 s
    max_implied_kmh: float = 1100.0  # the data contract says subsonic; roughly the speed of sound at altitude
    max_projection_km: float = 5.0  # about 3 degrees of course error over a 350 s gap at 540 km/h; longest true on train is 2 km


@dataclass(frozen=True)
class PairMeasurements:
    """None always means "couldn't be measured", never zero or agreement."""

    gap_s: float  # B's first report minus A's last, by observation time; always positive here
    endpoint_km: float  # great-circle, between fitted endpoints where available
    implied_kmh: float  # endpoint_km over the gap: the speed needed to cross it
    forward_miss_km: float | None  # A's motion projected to B's start, compared with B's start
    backward_miss_km: float | None  # B's motion projected back to A's end, compared with A's end
    speed_diff_kmh: float | None  # between whole-tracklet fitted speeds
    course_diff_deg: float | None  # between whole-tracklet fitted courses, in [0, 180]
    altitude_diff_ft: float | None  # absolute, between fitted altitudes at the two facing ends
    altitude_miss_ft: float | None  # mean error of projecting altitude across the gap, each way
    predecessor_reports: int
    successor_reports: int

    @property
    def position_miss_km(self) -> float | None:
        """Mean of the available directions. Their errors were nearly independent on true training links."""
        known = [m for m in (self.forward_miss_km, self.backward_miss_km) if m is not None]
        return sum(known) / len(known) if known else None


@dataclass(frozen=True)
class Candidate:
    predecessor: str
    successor: str
    measurements: PairMeasurements


@dataclass(frozen=True)
class CandidateSet:
    candidates: tuple[Candidate, ...]  # ordered by predecessor, gap, successor
    gated_out: dict[str, dict[str, int]]  # predecessor -> gate -> count of forward pairs removed
    motions: dict[str, TrackletMotion]


def measure_pair(a: TrackletMotion, b: TrackletMotion) -> PairMeasurements | None:
    """Measure A -> B. Returns None unless B starts strictly after A ends."""
    if b.start_us <= a.end_us:
        return None
    gap_s = (b.start_us - a.end_us) * 1e-6
    a_end = a.fitted_end_position or a.end_position
    b_start = b.fitted_start_position or b.start_position
    endpoint_km = haversine_km(*a_end, *b_start)

    forward = backward = speed_diff = course_diff = None
    if a.overall is not None:
        forward = haversine_km(*project_position(*a_end, a.overall.east_km_s, a.overall.north_km_s, gap_s), *b_start)
    if b.overall is not None:
        backward = haversine_km(*project_position(*b_start, b.overall.east_km_s, b.overall.north_km_s, -gap_s), *a_end)
    if a.overall is not None and b.overall is not None:
        speed_diff = abs(a.overall.speed_kmh - b.overall.speed_kmh)
        if a.overall.course_deg is not None and b.overall.course_deg is not None:
            course_diff = angle_difference_deg(a.overall.course_deg, b.overall.course_deg)

    altitude_diff = altitude_miss = None
    a_altitude = a.fitted_end_altitude_ft if a.fitted_end_altitude_ft is not None else a.median_altitude_ft
    b_altitude = b.fitted_start_altitude_ft if b.fitted_start_altitude_ft is not None else b.median_altitude_ft
    if a_altitude is not None and b_altitude is not None:
        altitude_diff = abs(b_altitude - a_altitude)
        misses = []
        if a.altitude_rate_ft_s is not None:
            misses.append(abs(b_altitude - (a_altitude + a.altitude_rate_ft_s * gap_s)))
        if b.altitude_rate_ft_s is not None:
            misses.append(abs(a_altitude - (b_altitude - b.altitude_rate_ft_s * gap_s)))
        altitude_miss = sum(misses) / len(misses) if misses else altitude_diff

    return PairMeasurements(
        gap_s=gap_s, endpoint_km=endpoint_km, implied_kmh=endpoint_km / gap_s * 3600,
        forward_miss_km=forward, backward_miss_km=backward, speed_diff_kmh=speed_diff, course_diff_deg=course_diff,
        altitude_diff_ft=altitude_diff, altitude_miss_ft=altitude_miss,
        predecessor_reports=a.report_count, successor_reports=b.report_count,
    )


def failed_gates(measurements: PairMeasurements, config: CandidateConfig) -> list[str]:
    """Which gates remove this pair. Unmeasurable values fail no gate; acceptance deals with them."""
    failed = []
    if measurements.gap_s > config.max_gap_s:
        failed.append(GATE_MAX_GAP)
    if measurements.implied_kmh > config.max_implied_kmh:
        failed.append(GATE_MAX_SPEED)
    miss = measurements.position_miss_km
    if miss is not None and miss > config.max_projection_km:
        failed.append(GATE_MAX_PROJECTION)
    return failed


def generate_candidates(scenario: PreparedScenario, config: CandidateConfig = CandidateConfig()) -> CandidateSet:
    """All gated forward pairs within this scenario. Scenarios are processed one at a time, so links can't cross them."""
    motions = {t.tracklet_id: summarize_tracklet(t) for t in scenario.tracklets}
    candidates, gated_out = [], {}
    for a_id, a in motions.items():
        for b_id, b in motions.items():
            if a_id == b_id:
                continue
            measurements = measure_pair(a, b)
            if measurements is None:
                continue
            failed = failed_gates(measurements, config)
            if failed:
                counts = gated_out.setdefault(a_id, {})
                for gate in failed:
                    counts[gate] = counts.get(gate, 0) + 1
            else:
                candidates.append(Candidate(a_id, b_id, measurements))
    candidates.sort(key=lambda c: (c.predecessor, c.measurements.gap_s, c.successor))
    return CandidateSet(tuple(candidates), gated_out, motions)


def measurements_to_dict(measurements: PairMeasurements) -> dict:
    return {**asdict(measurements), "position_miss_km": measurements.position_miss_km}
