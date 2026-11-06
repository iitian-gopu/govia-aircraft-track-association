"""Check the data package without executing packaged code and without reading test or staged labels.

Label files of test and staged snapshots are hashed as bytes but never parsed.

Usage: python scripts/verify_data.py [candidate-packet]
"""
import hashlib
import json
import sys
from pathlib import Path

from govia_tracks.labels import load_labels
from govia_tracks.loader import load_snapshot

root = Path(sys.argv[1] if len(sys.argv) > 1 else "candidate-packet")
hashes = json.loads((root / "checksums.json").read_text(encoding="utf-8"))

missing = sorted(name for name in hashes if not (root / name).exists())
mismatched = sorted(
    name for name in hashes
    if (root / name).exists() and hashlib.sha256((root / name).read_bytes()).hexdigest() != hashes[name]
)
print(f"hashes: {len(hashes)} listed, {len(hashes) - len(missing) - len(mismatched)} match, "
      f"missing={missing}, mismatched={mismatched}")

failures = 0
totals: dict[str, list[int]] = {}
for manifest_path in sorted((root / "data").rglob("snapshot.json")):
    partition = manifest_path.relative_to(root / "data").parts[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared = load_snapshot(manifest_path.parent)  # raises on any row-count or validation failure
    if partition in ("train", "validation"):
        links = sum(v is not None for v in load_labels(manifest_path.parent).values())
        if links != manifest["labeled_link_count"]:
            failures += 1
            print(f"label count mismatch: {manifest_path.parent}")
    row = totals.setdefault(partition, [0, 0, 0])
    row[0] += 1
    row[1] += len(prepared.tracklets)
    row[2] += prepared.quality.unique_observations
for partition, (snapshots, tracklets, observations) in totals.items():
    print(f"{partition}: snapshots={snapshots} tracklets={tracklets} unique_observations={observations}")

# SOURCE_NOTES.md is expected to be missing: Drive holds it as a Google Doc, exported here as "Source Notes.md".
unexpected = [name for name in missing if name != "SOURCE_NOTES.md"]
print("result:", "FAIL" if unexpected or mismatched or failures else "OK")
sys.exit(1 if unexpected or mismatched or failures else 0)
