"""Verify packaged file hashes and basic row counts; Python standard library only."""
import hashlib, json
from pathlib import Path
root=Path(__file__).resolve().parent
hashes=json.loads((root/'checksums.json').read_text())
for name, expected in hashes.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected, name
snapshots=0
for path in (root/'data').rglob('snapshot.json'):
    m=json.loads(path.read_text())
    def read(name): return [json.loads(x) for x in (path.parent/name).read_text().splitlines()]
    obs=read('observations.jsonl'); support=read('supporting.jsonl'); labels=read('labels.jsonl')
    assert len(obs)==m['observation_row_count']
    assert len(support)==m['supporting_row_count']
    assert len({o['observation_id'] for o in obs})==m['unique_observation_count']
    assert len(labels)==m['tracklet_count']
    assert sum(l['successor_tracklet_id'] is not None for l in labels)==m['labeled_link_count']
    snapshots+=1
print(f'OK: {len(hashes)} file hashes and {snapshots} snapshot row counts verified.')
