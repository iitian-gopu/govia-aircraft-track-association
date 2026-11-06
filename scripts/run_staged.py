"""Run both frozen methods on the two staged snapshots, compare their links, then evaluate. Nothing here fits or tunes.

  python scripts/run_staged.py    write reports/staged/{snapshot_1_result,snapshot_2_result,comparison,manifest}.json

Each snapshot is analyzed on its own records alone. Labels are opened only after every decision has been saved,
and only to evaluate. Outputs hold no absolute paths or timestamps, so repeated runs are byte-identical.
"""
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # for the dashboard package, which isn't installed

from dashboard.compare import by_id, compare_snapshots, summarize  # noqa: E402
from govia_tracks import baseline, learned  # noqa: E402
from govia_tracks.evaluation import count_links, metrics  # noqa: E402
from govia_tracks.labels import load_labels  # noqa: E402
from govia_tracks.loader import load_snapshot  # noqa: E402

DATA = ROOT / "candidate-packet" / "data" / "staged_demo"
OUTPUT = ROOT / "reports" / "staged"
BEFORE, AFTER = SNAPSHOTS = ("snapshot_1", "snapshot_2")
METHODS = ("baseline", "learned")
COMMAND = "python scripts/run_staged.py"


def write_json(name, value):
    """Fixed line endings, so a result's hash doesn't depend on the platform. Returns that hash."""
    data = (json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def analyze_snapshot(folder, baseline_config, model, policy):
    """One snapshot's records, and nothing else, through both methods."""
    scenario = load_snapshot(folder)
    return {
        "analysis": {
            "snapshot_id": scenario.snapshot_id, "scenario_id": scenario.scenario_id, "receipt_cutoff": scenario.receipt_cutoff,
            "input_fingerprint": scenario.input_fingerprint, "quality": asdict(scenario.quality),
        },
        "baseline": baseline.analyze(scenario, baseline_config),
        "learned": learned.analyze(scenario, model, policy),
    }


def explain_replacement(item):
    a, old, new, between = item["predecessor"], item["old_successor"], item["new_successor"], item["new_tracklets_between"]
    if between == 1:
        position = f"New tracklet {new} sits between the old pair: snapshot 2 accepts {a} -> {new} and {new} -> {old}."
    elif between > 1:
        position = f"{between} new tracklets sit between the old pair, starting with {new}, and snapshot 2's links run through them to {old}."
    elif item["new_successor_is_new"]:
        position = f"New tracklet {new} is not shown to sit between the old pair: snapshot 2 accepts no path of links from it to {old}."
    else:
        position = f"No new tracklet sits between the old pair: {new} was already available in snapshot 1."
    return (
        f"This method linked {a} to {old} in snapshot 1 and to {new} in snapshot 2. {position} "
        "A link names the immediate next available tracklet, not an aircraft, so this is not an identity change."
    )


def status_changes(before, after):
    old, new = by_id(before), by_id(after)
    return [
        {
            "tracklet_id": tracklet,
            "before": {key: old[tracklet][key] for key in ("status", "accepted_successor", "reason_code", "reason")},
            "after": {key: new[tracklet][key] for key in ("status", "accepted_successor", "reason_code", "reason")},
        }
        for tracklet in sorted(set(old) & set(new)) if old[tracklet]["status"] != new[tracklet]["status"]
    ]


def compare_method(before, after):
    """Label-free: one method's changes between its two independent results."""
    changes = compare_snapshots(before, after)
    tracklets_after = by_id(after)
    replaced = []
    for item in changes["replaced"]:
        old_pair_now = next((c for c in tracklets_after[item["predecessor"]]["candidates"] if c["successor"] == item["old_successor"]), None)
        replaced.append({**item, "old_pair_in_snapshot_2": old_pair_now, "explanation": explain_replacement(item)})
    return {
        "counts": {
            "new_tracklets": len(changes["new_tracklets"]), "removed_tracklets": len(changes["removed_tracklets"]),
            "added": len(changes["added"]), "withdrawn": len(changes["withdrawn"]), "replaced": len(changes["replaced"]),
            "unchanged": len(changes["unchanged"]),
            "unresolved": {BEFORE: summarize(before)["unresolved"], AFTER: summarize(after)["unresolved"]},
        },
        "new_tracklets": [
            {key: tracklets_after[t][key] for key in ("tracklet_id", "status", "accepted_successor", "reason_code", "reason")}
            for t in changes["new_tracklets"]
        ],
        "removed_tracklets": changes["removed_tracklets"],
        "status_changed": status_changes(before, after),
        "added": changes["added"],
        "withdrawn": changes["withdrawn"],
        "replaced": replaced,
        "unchanged": [{"predecessor": item["predecessor"], "successor": item["successor"]} for item in changes["unchanged"]],
    }


def evaluate(result, labels, snapshot):
    pairs = {(t["tracklet_id"], c["successor"]) for t in result["tracklets"] for c in t["candidates"]}
    counts = count_links(result["links"], labels, pairs, snapshot)
    return {**metrics(counts), "excluded_links": [list(link) for link in counts.excluded_links]}


def label_view(tracklet, labels, links):
    """One snapshot's truth for one tracklet. A tracklet the snapshot lacks has no label, which differs from a null successor."""
    return {
        "has_label": tracklet in labels, "true_successor": labels.get(tracklet), "linked_successor": links.get(tracklet),
        "link_correct": links[tracklet] == labels.get(tracklet) if tracklet in links else None,
    }


def ground_truth_changes(comparison, labels, results, method):
    """For each tracklet this method changed. `successor_changed` is null when a snapshot doesn't hold the tracklet."""
    changed = {item["predecessor"] for kind in ("added", "withdrawn", "replaced") for item in comparison[kind]}
    changed |= {item["tracklet_id"] for item in comparison["status_changed"]}
    rows = []
    for tracklet in sorted(changed):
        views = {name: label_view(tracklet, labels[name], results[name][method]["links"]) for name in SNAPSHOTS}
        both = all(views[name]["has_label"] for name in SNAPSHOTS)
        rows.append({
            "tracklet_id": tracklet, **views,
            "successor_changed": views[BEFORE]["true_successor"] != views[AFTER]["true_successor"] if both else None,
        })
    return rows


def main():
    baseline_config = baseline.load_config(ROOT / "configs" / "baseline.json")
    model = learned.load_model(ROOT / "models")
    policy = learned.load_policy(ROOT / "configs" / "learned.json", model.version)

    results, hashes = {}, {}
    for name in SNAPSHOTS:
        results[name] = analyze_snapshot(DATA / name, baseline_config, model, policy)
        hashes[f"{name}_result.json"] = write_json(f"{name}_result.json", results[name])
    methods = {method: compare_method(results[BEFORE][method], results[AFTER][method]) for method in METHODS}

    # Every decision is saved above. Labels are opened from here on, for evaluation only.
    labels = {name: load_labels(DATA / name) for name in SNAPSHOTS}
    for method in METHODS:
        methods[method]["ground_truth_for_changed_tracklets"] = ground_truth_changes(methods[method], labels, results, method)
    hashes["comparison.json"] = write_json("comparison.json", {
        "before": BEFORE, "after": AFTER, "methods": methods,
        "evaluation": {name: {method: evaluate(results[name][method], labels[name], name) for method in METHODS} for name in SNAPSHOTS},
    })
    write_json("manifest.json", {
        "snapshots": {
            name: {key: results[name]["analysis"][key] for key in ("snapshot_id", "scenario_id", "input_fingerprint", "receipt_cutoff")}
            for name in SNAPSHOTS
        },
        "baseline_version": baseline.config_version(baseline_config),
        "learned_version": model.version,
        "decision_settings": {method: {key: results[BEFORE][method][key] for key in ("policy", "config")} for method in METHODS},
        "sha256": hashes,
        "command": COMMAND,
    })
    for method in METHODS:
        print(method, json.dumps(methods[method]["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
