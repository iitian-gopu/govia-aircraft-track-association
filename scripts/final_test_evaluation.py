"""The one-time evaluation on the test partition, with frozen models and settings. Nothing here fits or selects anything.

  python scripts/final_test_evaluation.py freeze  record hashes of the code, settings and model in reports/freeze_manifest.json
  python scripts/final_test_evaluation.py run     check the freeze, evaluate, and write reports/test_evaluation.json

`run` refuses if the code, settings or model differ from the freeze, and refuses to overwrite an existing result:
the first result is the only untouched one. `--after-fix` writes a separate file, which must be disclosed as not untouched.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from govia_tracks import baseline, learned
from govia_tracks.association import check_links
from govia_tracks.candidates import generate_candidates
from govia_tracks.evaluation import LinkCounts, aircraft_groups, count_links, hide_altitude, metrics
from govia_tracks.labels import load_labels
from govia_tracks.loader import load_snapshot

FREEZE = Path("reports/freeze_manifest.json")
RESULT = Path("reports/test_evaluation.json")
AFTER_FIX = Path("reports/test_evaluation_after_fix.json")
TEST = Path("candidate-packet/data/test")
FROZEN_PATHS = ("src/govia_tracks", "configs", "models")


def hashes() -> dict[str, str]:
    files = [p for root in FROZEN_PATHS for p in sorted(Path(root).rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    return {p.as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in files}


def load_methods():
    config = baseline.load_config("configs/baseline.json")
    model = learned.load_model("models")
    policy = learned.load_policy("configs/learned.json", model.version)
    return {
        "baseline": lambda s: baseline.analyze(s, config),
        "learned": lambda s: learned.analyze(s, model, policy),
    }, {"baseline": baseline.config_version(config), "learned": model.version, "learned_policy": vars(policy), "baseline_policy": vars(config.policy)}


def freeze() -> None:
    _, versions = load_methods()
    FREEZE.write_text(json.dumps({"versions": versions, "files": hashes()}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"froze {len(hashes())} files: {versions['baseline']}, {versions['learned']}")


def check_freeze() -> dict:
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    current = hashes()
    changed = sorted(p for p in set(current) | set(frozen["files"]) if current.get(p) != frozen["files"].get(p))
    if changed:
        raise SystemExit(f"refusing to run: changed since the freeze: {changed}")
    return frozen


def trimmed(candidate: dict) -> dict:
    m = candidate["measurements"]
    keep = ("gap_s", "endpoint_km", "forward_miss_km", "backward_miss_km", "speed_diff_kmh", "course_diff_deg", "altitude_diff_ft", "altitude_miss_ft")
    return {"successor": candidate["successor"], "score": candidate["score"], "decision": candidate["decision"], **{k: m[k] for k in keep}}


def by_id(result: dict) -> dict:
    return {t["tracklet_id"]: t for t in result["tracklets"]}


def decision(entry: dict, labels: dict, tracklet: str) -> dict:
    return {"accepted": entry["accepted_successor"], "correct": entry["accepted_successor"] == labels[tracklet], "reason_code": entry["reason_code"], "reason": entry["reason"]}


def run(after_fix: bool) -> None:
    output = AFTER_FIX if after_fix else RESULT
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}: the first result must be preserved")
    frozen = check_freeze()
    methods, versions = load_methods()
    if versions != frozen["versions"]:
        raise SystemExit("refusing to run: loaded versions differ from the freeze")

    totals = {name: LinkCounts() for name in methods}
    hidden = {name: LinkCounts() for name in methods}
    cases = {"wrong_links": [], "labeled_unresolved": [], "disagreements": []}
    scenarios = []
    for folder in sorted(TEST.glob("scenario_*")):
        scenario = load_snapshot(folder)
        # Inference finishes on inputs alone before this scenario's labels are opened.
        results = {name: analyze(scenario) for name, analyze in methods.items()}
        without_altitude = {name: analyze(hide_altitude(scenario)) for name, analyze in methods.items()}
        motions = generate_candidates(scenario).motions
        for result in results.values():
            check_links(result["links"], motions)
        labels = load_labels(folder)
        groups = aircraft_groups(labels)
        scenarios.append({"folder": folder.name, "scenario_id": scenario.scenario_id, "snapshot_id": scenario.snapshot_id, "input_fingerprint": scenario.input_fingerprint})
        for name, result in results.items():
            pairs = {(t["tracklet_id"], c["successor"]) for t in result["tracklets"] for c in t["candidates"]}
            totals[name].add(count_links(result["links"], labels, pairs, folder.name))
            altitude_result = without_altitude[name]
            hidden[name].add(count_links(altitude_result["links"], labels, {(t["tracklet_id"], c["successor"]) for t in altitude_result["tracklets"] for c in t["candidates"]}, folder.name))

        tables = {name: by_id(result) for name, result in results.items()}
        for tracklet, label in sorted(labels.items()):
            entries = {name: tables[name][tracklet] for name in methods}
            accepted = {name: entry["accepted_successor"] for name, entry in entries.items()}
            record = {
                "scenario": folder.name, "tracklet": tracklet, "label": label,
                "baseline": decision(entries["baseline"], labels, tracklet), "learned": decision(entries["learned"], labels, tracklet),
                "top_candidates": {name: [trimmed(c) for c in entry["candidates"][:3]] for name, entry in entries.items()},
            }
            if any(a is not None and a != label for a in accepted.values()):
                record["wrong_kind"] = {name: None if a in (None, label) else ("same_aircraft" if groups[a] == groups[tracklet] else "other_aircraft") for name, a in accepted.items()}
                cases["wrong_links"].append(record)
            if label is not None and any(a is None for a in accepted.values()):
                cases["labeled_unresolved"].append(record)
            if accepted["baseline"] != accepted["learned"]:
                cases["disagreements"].append(record)

    report = {
        "note": "First and only evaluation at the frozen settings. Nothing was fitted or selected on these results.",
        "versions": versions,
        "scenarios": len(scenarios),
        "metrics": {name: {**metrics(totals[name]), "excluded_links": [list(x) for x in totals[name].excluded_links]} for name in methods},
        "stress_altitude_removed": {
            "perturbation": "altitude_ft removed from every report at inference, by evaluation.hide_altitude. Same labels, tracklets, models and settings.",
            "metrics": {name: metrics(hidden[name]) for name in methods},
        },
        "cases": cases,
        "inputs": scenarios,
    }
    output.write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": {k: {m: v for m, v in report["metrics"][k].items() if m != "excluded_links"} for k in report["metrics"]},
                      "stress": report["stress_altitude_removed"]["metrics"], "case_counts": {k: len(v) for k, v in cases.items()}}, indent=1))


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("command", choices=("freeze", "run"))
parser.add_argument("--after-fix", action="store_true", help="write a separate, disclosed result after a correctness fix")
arguments = parser.parse_args()
freeze() if arguments.command == "freeze" else run(arguments.after_fix)
sys.exit(0)
