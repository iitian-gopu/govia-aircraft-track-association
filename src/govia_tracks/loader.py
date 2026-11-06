"""Read JSONL inputs and snapshot folders. This module never opens a labels file."""
import json
from pathlib import Path
from typing import Iterator

from .errors import InputValidationError, Issue
from .preprocess import PreparedScenario, prepare_scenario


_MANIFEST_COUNTS = ("observation_row_count", "supporting_row_count", "unique_observation_count", "tracklet_count")


def _reject_constant(name):
    raise ValueError(f"{name} is not valid JSON")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key '{key}'")
        result[key] = value
    return result


def read_jsonl(path: str | Path, record_kind: str = "record") -> list:
    """Parse one JSON value per line. All malformed lines are reported together."""
    rows, issues = [], []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        try:
            rows.append(json.loads(line, parse_constant=_reject_constant, object_pairs_hook=_reject_duplicate_keys))
        except ValueError as exc:
            issues.append(Issue("malformed_json", record_kind, f"{Path(path).name} line {number}: {exc}", number - 1))
    if issues:
        raise InputValidationError(issues)
    return rows


def load_snapshot(folder: str | Path) -> PreparedScenario:
    """Load one scenario folder and check its rows against the manifest's descriptive counts."""
    folder = Path(folder)
    manifest = json.loads((folder / "snapshot.json").read_text(encoding="utf-8"))
    required = ("snapshot_id", "scenario_id", "receipt_cutoff", "files", *_MANIFEST_COUNTS)
    absent = [key for key in required if key not in manifest]
    absent += [f"files.{key}" for key in ("observations", "supporting") if key not in manifest.get("files", {})]
    if absent:
        raise InputValidationError([
            Issue("missing_field", "snapshot", f"{folder.name}: snapshot.json lacks '{key}'", field=key) for key in absent
        ])
    observations = read_jsonl(folder / manifest["files"]["observations"], "observation")
    supporting = read_jsonl(folder / manifest["files"]["supporting"], "supporting")
    prepared = prepare_scenario(
        observations, supporting,
        scenario_id=manifest["scenario_id"],
        snapshot_id=manifest["snapshot_id"],
        receipt_cutoff=manifest["receipt_cutoff"],
    )
    actual = {
        "observation_row_count": prepared.quality.observation_rows,
        "supporting_row_count": prepared.quality.supporting_rows,
        "unique_observation_count": prepared.quality.unique_observations,
        "tracklet_count": len(prepared.tracklets),
    }
    mismatches = [
        Issue("count_mismatch", "snapshot", f"{folder.name}: {key} is {value}, manifest says {manifest[key]}", field=key)
        for key, value in actual.items() if manifest[key] != value
    ]
    if mismatches:
        raise InputValidationError(mismatches)
    return prepared


def iter_partition(data_root: str | Path, partition: str) -> Iterator[PreparedScenario]:
    """Yield each scenario of a partition independently, in folder-name order."""
    folders = sorted(Path(data_root, partition).glob("scenario_*"))
    if not folders:
        raise FileNotFoundError(f"no scenario folders under {Path(data_root, partition)}")
    for folder in folders:
        yield load_snapshot(folder)
