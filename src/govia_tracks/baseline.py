"""Rule-based physical continuity score.

Score meaning: **higher is more plausible; 0 is a perfect fit and there is no lower bound.**
It is minus a sum of squared, noise-scaled residuals. It is not a probability, and it has not been
calibrated, so it should not be read as confidence.

Each residual is divided by the size it typically has on a true link, so a value near 1 per term
is ordinary. Scales come from true links in the training partition only; see `fit_scales`.
True links there all span 60 s or more, so a line through them says nothing about shorter gaps. A second
limit, from how far reports sit from their own tracklet's fit, stops the scale falling below report noise.
A missing measurement adds `missing_penalty`, about what an ordinary measurement costs, so missing
evidence is neither agreement nor disagreement.

The time gap is not scored. A gap alone is not evidence that two fragments belong together;
immediacy is handled structurally by the selection step's check for a possible intermediate.
"""
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .association import AcceptancePolicy, ScoredCandidate, build_result
from .candidates import Candidate, CandidateConfig, PairMeasurements, generate_candidates, measurements_to_dict
from .preprocess import PreparedScenario

METHOD = "baseline"
MIN_REPORTS_FOR_FLOOR = 3


@dataclass(frozen=True)
class NoiseScales:
    position_km: float
    position_km_per_s: float  # projection error grows with the gap
    altitude_ft: float
    altitude_ft_per_s: float
    speed_kmh: float
    course_deg: float
    position_floor_km: float = 0.0  # report noise: the position scale doesn't fall below this however short the gap
    altitude_floor_ft: float = 0.0  # likewise for altitude


@dataclass(frozen=True)
class BaselineConfig:
    candidates: CandidateConfig
    scales: NoiseScales
    policy: AcceptancePolicy
    use_altitude: bool = True
    use_speed_course: bool = True
    missing_penalty: float = 1.0
    min_reports_for_motion: int = 3  # a line through fewer reports says little about motion


def _fit_line(gaps, values) -> tuple[float, float]:
    """Least-squares mean value as a + b * gap, with a held at zero or above: no negative noise floor."""
    gap_mean = sum(gaps) / len(gaps)
    value_mean = sum(values) / len(values)
    slope = sum((g - gap_mean) * (v - value_mean) for g, v in zip(gaps, values)) / sum((g - gap_mean) ** 2 for g in gaps)
    intercept = value_mean - slope * gap_mean
    if intercept < 0:
        return 0.0, sum(g * v for g, v in zip(gaps, values)) / sum(g * g for g in gaps)
    return intercept, slope


def fit_scales(true_links: list[PairMeasurements], motions=()) -> NoiseScales:
    """Typical residual sizes, from measurements of true links. The caller selects them from training labels.
    `motions` are training tracklets' summaries, which need no labels: their residuals set the noise floors."""
    position = [m for m in true_links if m.position_miss_km is not None]
    altitude = [m for m in true_links if m.altitude_miss_ft is not None]
    speed = [m.speed_diff_kmh for m in true_links if m.speed_diff_kmh is not None]
    course = [m.course_diff_deg for m in true_links if m.course_diff_deg is not None]
    position_a, position_b = _fit_line([m.gap_s for m in position], [m.position_miss_km for m in position])
    altitude_a, altitude_b = _fit_line([m.gap_s for m in altitude], [m.altitude_miss_ft for m in altitude])
    usable = [m for m in motions if m.report_count >= MIN_REPORTS_FOR_FLOOR]  # fewer lie on their own line whatever the noise
    position_residuals = [m.overall.median_residual_km for m in usable if m.overall is not None]
    altitude_residuals = [m.altitude_median_residual_ft for m in usable if m.altitude_median_residual_ft is not None]
    return NoiseScales(
        position_km=position_a, position_km_per_s=position_b, altitude_ft=altitude_a, altitude_ft_per_s=altitude_b,
        speed_kmh=sum(speed) / len(speed), course_deg=sum(course) / len(course),
        position_floor_km=sum(position_residuals) / len(position_residuals) if position_residuals else 0.0,
        altitude_floor_ft=sum(altitude_residuals) / len(altitude_residuals) if altitude_residuals else 0.0,
    )


def score_terms(m: PairMeasurements, config: BaselineConfig) -> dict[str, float]:
    """Squared, noise-scaled residuals. Larger is worse."""
    s = config.scales

    def term(value, scale):
        return config.missing_penalty if value is None else (value / scale) ** 2

    terms = {"position": term(m.position_miss_km, max(s.position_km + s.position_km_per_s * m.gap_s, s.position_floor_km))}
    if config.use_altitude:
        terms["altitude"] = term(m.altitude_miss_ft, max(s.altitude_ft + s.altitude_ft_per_s * m.gap_s, s.altitude_floor_ft))
    if config.use_speed_course:
        terms["speed"] = term(m.speed_diff_kmh, s.speed_kmh)
        terms["course"] = term(m.course_diff_deg, s.course_deg)
    return terms


def score_candidate(candidate: Candidate, config: BaselineConfig) -> ScoredCandidate:
    m = candidate.measurements
    terms = score_terms(m, config)
    # Both directions: reports spanning no time give no motion, however many there are.
    motion_known = m.forward_miss_km is not None and m.backward_miss_km is not None and min(m.predecessor_reports, m.successor_reports) >= config.min_reports_for_motion
    return ScoredCandidate(
        candidate.predecessor, candidate.successor, -sum(terms.values()),
        {**measurements_to_dict(m), "score_terms": terms}, acceptable=motion_known,
    )


def config_to_dict(config: BaselineConfig) -> dict:
    return asdict(config)


def config_from_dict(data: dict) -> BaselineConfig:
    return BaselineConfig(
        candidates=CandidateConfig(**data["candidates"]),
        scales=NoiseScales(**data["scales"]),
        policy=AcceptancePolicy(**data["policy"]),
        **{k: data[k] for k in ("use_altitude", "use_speed_course", "missing_penalty", "min_reports_for_motion")},
    )


def load_config(path: str | Path) -> BaselineConfig:
    return config_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def config_version(config: BaselineConfig) -> str:
    """Changes whenever any setting changes, so a result can be traced to its configuration."""
    text = json.dumps(config_to_dict(config), sort_keys=True, separators=(",", ":"))
    return f"{METHOD}-{hashlib.sha256(text.encode()).hexdigest()[:12]}"


def analyze(scenario: PreparedScenario, config: BaselineConfig) -> dict:
    """Run the baseline on one scenario. Takes no labels."""
    candidate_set = generate_candidates(scenario, config.candidates)
    scored = [score_candidate(c, config) for c in candidate_set.candidates]
    return build_result(METHOD, config_version(config), config_to_dict(config), scenario, candidate_set, scored, config.policy)
