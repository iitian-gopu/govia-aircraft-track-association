"""Association labels, for training and evaluation only.

Deliberately separate from the input path: `loader`, `preprocess`, `schema` and `movement`
must never import this module. A test enforces that.
"""
from pathlib import Path

from .errors import InputValidationError, Issue
from .evaluation import validate_labels
from .loader import read_jsonl


def load_labels(folder: str | Path) -> dict[str, str | None]:
    """Return {tracklet_id: successor_tracklet_id or None} for one snapshot folder. Rejects labels that aren't valid chains."""
    labels: dict[str, str | None] = {}
    issues = []
    for index, row in enumerate(read_jsonl(Path(folder) / "labels.jsonl", "label")):
        tracklet = row.get("tracklet_id") if isinstance(row, dict) else None
        if not isinstance(tracklet, str) or not tracklet or "successor_tracklet_id" not in row:
            issues.append(Issue("invalid_field", "label", f"label row {index}: needs tracklet_id and successor_tracklet_id", index))
        elif tracklet in labels:
            issues.append(Issue("conflicting_id", "label", f"label row {index}: repeated tracklet '{tracklet}'", index, tracklet))
        else:
            labels[tracklet] = row["successor_tracklet_id"]
    if issues:
        raise InputValidationError(issues)
    validate_labels(labels)
    return labels
