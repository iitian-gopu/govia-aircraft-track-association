"""Learned candidate-pair scorer, inference side. Uses the shared candidates and selection; only the score differs from the baseline.

Score meaning: **higher is more plausible, in [0, 1].** It is the classifier's predicted probability that a pair
is an immediate successor link. It has not been calibrated, so it should not be read as confidence. True links across
long gaps score low because they resemble skips, so useful thresholds are small; don't compare this with baseline scores.

Nothing here trains, and nothing here reads labels. Fitting lives in `training`.

Artifacts are joblib pickles, so load only files this project generated. `load_model` also checks the
artifact's hash against its metadata, which catches accidental changes but isn't protection against tampering.
"""
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .association import AcceptancePolicy, ScoredCandidate, build_result
from .candidates import CandidateConfig, PairMeasurements, generate_candidates, measurements_to_dict
from .preprocess import PreparedScenario

METHOD = "learned"
MODEL_FILE = "learned_model.joblib"
METADATA_FILE = "learned_model.json"

# Order is part of the model's contract. It is saved with the model and checked on load.
# Chosen with whole-scenario folds on train. Course added nothing, and altitude difference and the mean position miss repeat these.
FEATURES = (
    "gap_s",
    "forward_miss_km",
    "backward_miss_km",
    "altitude_miss_ft",
    "speed_diff_kmh",
)
ALTITUDE_FEATURES = frozenset({"altitude_miss_ft"})

# Physical rules applied to the model's scores, separate from them. Training data rarely holds pairs this far off, so the
# model never learned to refuse one that looks good on its other inputs: a fragment starting on A's path but flying 45
# degrees away scored 0.99. Largest values on true training links: 1.97 km either way and 4.54 degrees.
MAX_DIRECTIONAL_MISS_KM = 3.0
MAX_COURSE_DIFF_DEG = 15.0


def feature_row(measurements: PairMeasurements, features=FEATURES) -> list[float]:
    """Measurements as model inputs. Missing stays NaN, never zero; a non-finite measurement is an error."""
    row = []
    for name in features:
        value = getattr(measurements, name)
        if value is None:
            row.append(math.nan)
        elif not math.isfinite(value):
            raise ValueError(f"non-finite measurement {name}={value}")
        else:
            row.append(float(value))
    return row


def feature_matrix(candidates, features=FEATURES) -> np.ndarray:
    return np.array([feature_row(c.measurements, features) for c in candidates], dtype=np.float64).reshape(len(candidates), len(features))


@dataclass(frozen=True)
class LearnedModel:
    estimator: HistGradientBoostingClassifier
    feature_names: tuple[str, ...] = FEATURES
    candidates: CandidateConfig = CandidateConfig()
    min_reports_for_motion: int = 3
    metadata: dict = field(default_factory=dict)

    @property
    def version(self) -> str:
        return self.metadata["model_version"]


def physical_objection(m: PairMeasurements) -> str | None:
    """Why physical rules exclude a pair, if they do. A measurement that's missing raises no objection here."""
    for name, value in (("forward", m.forward_miss_km), ("backward", m.backward_miss_km)):
        if value is not None and value > MAX_DIRECTIONAL_MISS_KM:
            return f"{name} projection misses by {value:.2f} km; the limit is {MAX_DIRECTIONAL_MISS_KM:.1f} km"
    if m.course_diff_deg is not None and m.course_diff_deg > MAX_COURSE_DIFF_DEG:
        return f"courses differ by {m.course_diff_deg:.1f} degrees; the limit is {MAX_COURSE_DIFF_DEG:.0f}"
    return None


def score_candidates(model: LearnedModel, candidates) -> list[ScoredCandidate]:
    """Score every candidate. A pair's score doesn't depend on the other candidates or their order."""
    if not candidates:
        return []
    scores = model.estimator.predict_proba(feature_matrix(candidates, model.feature_names))[:, 1]
    scored = []
    for candidate, score in zip(candidates, scores):
        m = candidate.measurements
        # Both directions: reports spanning no time give no motion, however many there are.
        reliable = m.forward_miss_km is not None and m.backward_miss_km is not None and min(m.predecessor_reports, m.successor_reports) >= model.min_reports_for_motion
        scored.append(ScoredCandidate(
            candidate.predecessor, candidate.successor, float(score), measurements_to_dict(m), acceptable=reliable, ruled_out=physical_objection(m),
        ))
    return scored


def analyze(scenario: PreparedScenario, model: LearnedModel, policy: AcceptancePolicy) -> dict:
    """Run the learned approach on one scenario. Takes no labels and doesn't train."""
    candidate_set = generate_candidates(scenario, model.candidates)
    config = {
        "candidates": asdict(model.candidates), "features": list(model.feature_names), "min_reports_for_motion": model.min_reports_for_motion,
        "max_directional_miss_km": MAX_DIRECTIONAL_MISS_KM, "max_course_diff_deg": MAX_COURSE_DIFF_DEG,
    }
    scored = score_candidates(model, candidate_set.candidates)
    return build_result(METHOD, model.version, config, scenario, candidate_set, scored, policy)


def load_model(folder: str | Path) -> LearnedModel:
    """Load a locally generated model. Refuses a changed artifact, or a schema that doesn't match this code."""
    folder = Path(folder)
    metadata = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8"))
    if hashlib.sha256((folder / MODEL_FILE).read_bytes()).hexdigest() != metadata["artifact_sha256"]:
        raise ValueError(f"{folder / MODEL_FILE} doesn't match its recorded hash")
    if tuple(metadata["feature_names"]) != FEATURES:
        raise ValueError(f"saved feature schema {metadata['feature_names']} differs from this code's {list(FEATURES)}")
    estimator = joblib.load(folder / MODEL_FILE)
    return LearnedModel(estimator, tuple(metadata["feature_names"]), CandidateConfig(**metadata["candidates"]), metadata["min_reports_for_motion"], metadata)


def load_policy(path: str | Path, model_version: str | None = None) -> AcceptancePolicy:
    """Read acceptance thresholds. With `model_version`, refuse thresholds selected for a different model."""
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    if model_version is not None and saved.get("model_version") != model_version:
        raise ValueError(f"{path} was selected for model {saved.get('model_version')}, not {model_version}")
    data = saved["policy"]
    return AcceptancePolicy(min_score=data["min_score"], min_margin=data["min_margin"], plausible_score=data["plausible_score"])
