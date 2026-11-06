"""Build supervised examples from generated candidates and fit the learned scorer.

Labels arrive as an argument. Every generated candidate becomes one row, with no sampling, so the negatives include
other aircraft and same-aircraft pairs that skip a fragment. Rows carry their scenario, so any split can keep a
whole scenario on one side.
"""
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from .candidates import CandidateConfig, generate_candidates
from .evaluation import aircraft_groups
from .learned import ALTITUDE_FEATURES, FEATURES, METADATA_FILE, METHOD, MODEL_FILE, LearnedModel, feature_row

DEFAULT_PARAMETERS = {
    "max_depth": 3,
    "max_iter": 200,
    "learning_rate": 0.08,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 0,
}
HIDDEN_COPY_WEIGHT = 0.5


@dataclass(frozen=True)
class TrainingExamples:
    features: np.ndarray
    targets: np.ndarray  # 1 if the pair is the labeled immediate successor link
    groups: np.ndarray  # scenario position of each row
    scenarios: tuple[str, ...]
    positives: int
    negatives_same_aircraft: int  # same aircraft, but skipping a fragment
    negatives_other_aircraft: int
    labeled_links: int  # including any lost by candidate generation


def training_examples(partitions, candidate_config: CandidateConfig = CandidateConfig(), features=FEATURES) -> TrainingExamples:
    """`partitions` yields (name, PreparedScenario, labels). Candidates are generated without seeing the labels."""
    rows, targets, groups, names = [], [], [], []
    same_aircraft = other_aircraft = labeled = 0
    for position, (name, scenario, labels) in enumerate(partitions):
        names.append(name)
        chains = aircraft_groups(labels)
        labeled += sum(successor is not None for successor in labels.values())
        for candidate in generate_candidates(scenario, candidate_config).candidates:
            positive = labels[candidate.predecessor] == candidate.successor
            rows.append(feature_row(candidate.measurements, features))
            targets.append(int(positive))
            groups.append(position)
            if not positive:
                if chains[candidate.predecessor] == chains[candidate.successor]:
                    same_aircraft += 1
                else:
                    other_aircraft += 1
    targets = np.array(targets, dtype=np.int64)
    return TrainingExamples(
        np.array(rows, dtype=np.float64).reshape(len(rows), len(features)), targets, np.array(groups, dtype=np.int64), tuple(names),
        int(targets.sum()), same_aircraft, other_aircraft, labeled,
    )


def fit_estimator(features: np.ndarray, targets: np.ndarray, parameters=None, feature_names=FEATURES, altitude_hidden_copy=True):
    """Fit the classifier. No tracklet in training lacks altitude, so without help the model never sees it missing and
    would route NaN arbitrarily. `altitude_hidden_copy` adds every row again, at reduced weight, with altitude missing."""
    weights = np.ones(len(targets))
    if altitude_hidden_copy:
        hidden = features.copy()
        for index, name in enumerate(feature_names):
            if name in ALTITUDE_FEATURES:
                hidden[:, index] = math.nan
        features, targets = np.vstack([features, hidden]), np.concatenate([targets, targets])
        weights = np.concatenate([weights, np.full(len(weights), HIDDEN_COPY_WEIGHT)])
    estimator = HistGradientBoostingClassifier(**{**DEFAULT_PARAMETERS, **(parameters or {})})
    return estimator.fit(features, targets, sample_weight=weights)


def out_of_fold_estimators(examples: TrainingExamples, parameters=None, feature_names=FEATURES, n_splits=5, altitude_hidden_copy=True) -> dict[str, HistGradientBoostingClassifier]:
    """{scenario name: an estimator fitted without that scenario}. Whole scenarios stay on one side of every split.

    A model's scores on its own training data are optimistic, so train results need this to be comparable.
    """
    by_scenario = {}
    for fit_rows, held_rows in GroupKFold(n_splits=n_splits).split(examples.features, groups=examples.groups):
        estimator = fit_estimator(examples.features[fit_rows], examples.targets[fit_rows], parameters, feature_names, altitude_hidden_copy)
        for position in np.unique(examples.groups[held_rows]):
            by_scenario[examples.scenarios[position]] = estimator
    return by_scenario


def build_model(examples: TrainingExamples, parameters=None, feature_names=FEATURES, candidates=CandidateConfig(), min_reports=3, altitude_hidden_copy=True) -> LearnedModel:
    """Fit a model and describe it. The version depends only on settings and training data, so a retrain matches."""
    estimator = fit_estimator(examples.features, examples.targets, parameters, feature_names, altitude_hidden_copy)
    described = {
        "method": METHOD,
        "feature_names": list(feature_names),
        "parameters": {**DEFAULT_PARAMETERS, **(parameters or {})},
        "altitude_hidden_copy": altitude_hidden_copy,
        "hidden_copy_weight": HIDDEN_COPY_WEIGHT if altitude_hidden_copy else None,
        "target": "immediate successor only; a same-aircraft skip is a negative",
        "candidates": asdict(candidates),
        "min_reports_for_motion": min_reports,
        "training": {
            "scenarios": len(examples.scenarios),
            "rows": int(len(examples.targets)),
            "positives": examples.positives,
            "negatives_same_aircraft": examples.negatives_same_aircraft,
            "negatives_other_aircraft": examples.negatives_other_aircraft,
            "labeled_links": examples.labeled_links,
            "data_sha256": hashlib.sha256(np.nan_to_num(examples.features, nan=-1e300).tobytes() + examples.targets.tobytes()).hexdigest(),
        },
    }
    described["model_version"] = f"{METHOD}-" + hashlib.sha256(json.dumps(described, sort_keys=True).encode()).hexdigest()[:12]
    return LearnedModel(estimator, tuple(feature_names), candidates, min_reports, described)


def save_model(model: LearnedModel, folder: str | Path) -> None:
    """Write the fitted estimator and its metadata. Library versions are recorded but aren't part of the model version."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.estimator, folder / MODEL_FILE)
    metadata = {
        **model.metadata,
        "artifact_sha256": hashlib.sha256((folder / MODEL_FILE).read_bytes()).hexdigest(),
        "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
    }
    (folder / METADATA_FILE).write_text(json.dumps(metadata, indent=1, sort_keys=True) + "\n", encoding="utf-8")
