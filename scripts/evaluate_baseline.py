"""Select baseline settings and evaluate them. Train and validation only: there is no way to point this at test.

  python scripts/evaluate_baseline.py select    fit scales on train, choose settings on validation,
                                                write configs/baseline.json and reports/baseline_selection.json
  python scripts/evaluate_baseline.py evaluate  run the saved configuration on train and validation,
                                                write reports/baseline_metrics.json and reports/baseline_results_validation.json

Selection minimizes (cost_ratio * wrong links + missed links), so the greater cost of a wrong link is an explicit
assumption. Validation decides first. It is small and often can't separate settings, so ties fall to train
(which set the noise scales but not, before this step, any threshold), then to the more cautious setting.

Wrong links are counted at the next-lower margin as well, so a setting is never chosen right beside one that errs.

  balanced      cost ratio 5; the operating policy, which also chooses the feature set
  permissive    cost ratio 1, same features
  conservative  same features; the largest margin, then strictest score, with no wrong links on either partition
                and at least 97% recall across both
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

from govia_tracks.association import AcceptancePolicy, select_links
from govia_tracks.baseline import (
    BaselineConfig, NoiseScales, analyze, config_from_dict, config_to_dict, config_version, fit_scales, load_config, score_candidate,
)
from govia_tracks.candidates import CandidateConfig, generate_candidates
from govia_tracks.evaluation import LinkCounts, count_links, labels_without, metrics
from govia_tracks.labels import load_labels
from govia_tracks.loader import load_snapshot

DATA = Path("candidate-packet/data")
CONFIG = Path("configs/baseline.json")
REPORTS = Path("reports")
BALANCED_RATIO, PERMISSIVE_RATIO = 5, 1
CONSERVATIVE_MIN_RECALL = 0.97
OPERATING_POLICY = "balanced"
MIN_SCORES = (-3, -5, -8, -12, -16, -24, -40)
MIN_MARGINS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
PLAUSIBLE_SCORES = (-8, -16, -30, -60, -120)
VARIANTS = {
    "position": dict(use_altitude=False, use_speed_course=False),
    "position_altitude": dict(use_altitude=True, use_speed_course=False),
    "position_speed_course": dict(use_altitude=False, use_speed_course=True),
    "all": dict(use_altitude=True, use_speed_course=True),
}


def load_partition(partition):
    """Returns (folder name, scenario, labels, candidate set). Labels stay apart from the candidate set."""
    loaded = []
    for folder in sorted((DATA / partition).glob("scenario_*")):
        scenario = load_snapshot(folder)
        loaded.append((folder.name, scenario, load_labels(folder), generate_candidates(scenario, CandidateConfig())))
    return loaded


def sweep_counts(loaded, config, policy):
    total = LinkCounts()
    for name, _, labels, candidate_set in loaded:
        scored = [score_candidate(c, config) for c in candidate_set.candidates]
        links, _ = select_links(scored, policy)
        pairs = {(c.predecessor, c.successor) for c in candidate_set.candidates}
        total.add(count_links(links, labels, pairs, name))
    return total


def select():
    train = load_partition("train")
    validation = load_partition("validation")
    true_links = [c.measurements for _, _, labels, cs in train for c in cs.candidates if labels[c.predecessor] == c.successor]
    scales = fit_scales(true_links, [m for _, _, _, cs in train for m in cs.motions.values()])
    variants = {name: dict(flags) for name, flags in VARIANTS.items()}
    constant_scales = NoiseScales(
        position_km=sum(m.position_miss_km for m in true_links) / len(true_links), position_km_per_s=0.0,
        altitude_ft=sum(m.altitude_miss_ft for m in true_links) / len(true_links), altitude_ft_per_s=0.0,
        speed_kmh=scales.speed_kmh, course_deg=scales.course_deg,
    )

    grid = []
    for scale_name, scale_values in (("gap_scaled", scales), ("constant", constant_scales)):
        for variant, flags in variants.items():
            for plausible in PLAUSIBLE_SCORES:
                for min_score in MIN_SCORES:
                    if plausible > min_score:
                        continue
                    for margin in MIN_MARGINS:
                        policy = AcceptancePolicy(float(min_score), margin, float(plausible))
                        config = BaselineConfig(CandidateConfig(), scale_values, policy, **flags)
                        row = dict(variant=variant, scaling=scale_name, min_score=min_score, min_margin=margin, plausible_score=plausible)
                        for name, loaded in (("validation", validation), ("train", train)):
                            counts = metrics(sweep_counts(loaded, config, policy))
                            row[name] = {k: counts[k] for k in ("wrong", "correct", "accepted", "precision", "recall", "labeled_links")}
                        grid.append(row)

    by_setting = {(r["variant"], r["scaling"], r["min_score"], r["plausible_score"], r["min_margin"]): r for r in grid}

    def guarded_wrong(row, partition):
        """Wrong links at this margin or the next-lower one."""
        index = MIN_MARGINS.index(row["min_margin"])
        lower = by_setting[(row["variant"], row["scaling"], row["min_score"], row["plausible_score"], MIN_MARGINS[max(index - 1, 0)])]
        return max(row[partition]["wrong"], lower[partition]["wrong"])

    def cost(row, partition, ratio):
        return ratio * guarded_wrong(row, partition) + row[partition]["labeled_links"] - row[partition]["correct"]

    def best(rows, ratio):
        return min(rows, key=lambda r: (cost(r, "validation", ratio), cost(r, "train", ratio), -r["min_margin"], -r["min_score"], -r["plausible_score"]))

    balanced = best(grid, BALANCED_RATIO)
    same_features = [r for r in grid if r["variant"] == balanced["variant"] and r["scaling"] == balanced["scaling"]]

    def pooled_recall(row):
        return (row["train"]["correct"] + row["validation"]["correct"]) / (row["train"]["labeled_links"] + row["validation"]["labeled_links"])

    safe = [r for r in same_features if r["train"]["wrong"] == 0 and r["validation"]["wrong"] == 0 and pooled_recall(r) >= CONSERVATIVE_MIN_RECALL]
    policies = {
        "conservative": max(safe, key=lambda r: (r["min_margin"], r["min_score"], pooled_recall(r), r["plausible_score"])),
        "balanced": balanced,
        "permissive": best(same_features, PERMISSIVE_RATIO),
    }
    # Each feature set at the balanced ratio, to show what each signal adds.
    ablation = {
        f"{scaling}/{variant}": best([r for r in grid if r["variant"] == variant and r["scaling"] == scaling], BALANCED_RATIO)
        for scaling in ("gap_scaled", "constant") for variant in variants
    }

    def config_for(row):
        scale_values = scales if row["scaling"] == "gap_scaled" else constant_scales
        policy = AcceptancePolicy(float(row["min_score"]), row["min_margin"], float(row["plausible_score"]))
        return BaselineConfig(CandidateConfig(), scale_values, policy, **variants[row["variant"]])

    chosen = config_for(policies[OPERATING_POLICY])
    CONFIG.parent.mkdir(exist_ok=True)
    CONFIG.write_text(json.dumps({**config_to_dict(chosen), "selected_on": "validation then train (development; both explored in phase 1)",
                                  "cost_ratio": BALANCED_RATIO}, indent=1) + "\n", encoding="utf-8")
    write_json(REPORTS / "baseline_selection.json", dict(
        operating_policy=OPERATING_POLICY, balanced_ratio=BALANCED_RATIO, permissive_ratio=PERMISSIVE_RATIO,
        conservative_min_recall=CONSERVATIVE_MIN_RECALL, policies=policies, ablation=ablation, grid_size=len(grid),
        policy_configs={name: config_to_dict(config_for(row)) for name, row in policies.items()},
    ))
    for name, row in {**policies, **ablation}.items():
        print(f"{name:32s} score>={row['min_score']} margin>={row['min_margin']} plausible>={row['plausible_score']} "
              f"validation wrong={row['validation']['wrong']} correct={row['validation']['correct']} "
              f"train wrong={row['train']['wrong']} correct={row['train']['correct']}")
    print(f"wrote {CONFIG}; grid of {len(grid)}")


def write_json(path, value):
    path.parent.mkdir(exist_ok=True)
    text = json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def evaluate_partition(partition, config):
    total = LinkCounts()
    results = []
    for folder in sorted((DATA / partition).glob("scenario_*")):
        scenario = load_snapshot(folder)
        result = analyze(scenario, config)  # takes no labels
        labels = load_labels(folder)
        pairs = {(t["tracklet_id"], c["successor"]) for t in result["tracklets"] for c in t["candidates"]}
        total.add(count_links(result["links"], labels, pairs, folder.name))
        results.append(result)
    return total, results


def withheld_fragments(partition, config, choose):
    """Run with some tracklets removed, as if never supplied. Labels choose them here, in the harness only."""
    total = LinkCounts()
    no_candidate_null, null_tracklets = 0, 0
    for folder in sorted((DATA / partition).glob("scenario_*")):
        scenario, labels = load_snapshot(folder), load_labels(folder)
        dropped = choose(labels)
        reduced = dataclasses.replace(scenario, tracklets=tuple(t for t in scenario.tracklets if t.tracklet_id not in dropped))
        remaining = labels_without(labels, dropped)
        result = analyze(reduced, config)
        total.add(count_links(result["links"], remaining, {(t["tracklet_id"], c["successor"]) for t in result["tracklets"] for c in t["candidates"]}, folder.name))
        for tracklet in result["tracklets"]:
            if remaining[tracklet["tracklet_id"]] is None:
                null_tracklets += 1
                no_candidate_null += not tracklet["candidates"]
    return {**metrics(total), "null_successor_tracklets": null_tracklets, "null_successor_tracklets_without_candidates": no_candidate_null}


def last_of_each_aircraft(labels):
    return {tracklet for tracklet, successor in labels.items() if successor is None}


def second_of_each_aircraft(labels):
    followers = {b for b in labels.values() if b is not None}
    return {labels[a] for a in labels if a not in followers and labels[a] is not None}


def evaluate():
    config = load_config(CONFIG)
    selection = json.loads((REPORTS / "baseline_selection.json").read_text(encoding="utf-8"))
    policy_metrics = {}
    for name, data in selection["policy_configs"].items():
        policy_config = config_from_dict(data)
        policy_metrics[name] = {p: metrics(evaluate_partition(p, policy_config)[0]) for p in ("train", "validation")}
    report = dict(version=config_version(config), partitions={}, policies=policy_metrics, checks={})
    for partition in ("train", "validation"):
        counts, results = evaluate_partition(partition, config)
        report["partitions"][partition] = dict(metrics=metrics(counts), excluded_links=[list(x) for x in counts.excluded_links])
        # Checks at frozen settings, with no retuning. The real labels' null successors never have a later tracklet to link to.
        report["checks"][partition] = dict(
            true_final_fragment_withheld=withheld_fragments(partition, config, last_of_each_aircraft),
            true_second_fragment_withheld=withheld_fragments(partition, config, second_of_each_aircraft),
            altitude_not_scored=metrics(evaluate_partition(partition, dataclasses.replace(config, use_altitude=False))[0]),
        )
        reason_counts = {}
        for result in results:
            for tracklet in result["tracklets"]:
                reason_counts[tracklet["reason_code"]] = reason_counts.get(tracklet["reason_code"], 0) + 1
        report["partitions"][partition]["reason_counts"] = dict(sorted(reason_counts.items()))
        digest = hashlib.sha256(json.dumps(results, sort_keys=True, allow_nan=False).encode()).hexdigest()
        report["partitions"][partition]["results_sha256"] = digest
        if partition == "validation":
            write_json(REPORTS / "baseline_results_validation.json", results)
    write_json(REPORTS / "baseline_metrics.json", report)
    print(json.dumps(report, indent=1, sort_keys=True))


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("command", choices=("select", "evaluate"))
{"select": select, "evaluate": evaluate}[parser.parse_args().command]()
