"""Train the learned scorer, choose its thresholds, and compare it with the frozen baseline. Train and validation only.

  python scripts/run_learned.py train     fit on train; write models/learned_model.joblib and models/learned_model.json
  python scripts/run_learned.py select    choose thresholds; write configs/learned.json and reports/learned_selection.json
  python scripts/run_learned.py evaluate  compare with the baseline; write reports/learned_metrics.json
  python scripts/run_learned.py stress    compare both methods on perturbed inputs; write reports/learned_stress.json

Thresholds are chosen the same way as the baseline's, with the same 5:1 cost assumption: minimize
(ratio * wrong + missed) on validation, ties to train then to the more cautious setting, counting wrong links at the
next-lower margin too. The saved model has seen train, so every train figure here that is used for a decision or a
comparison comes from out-of-fold models, fitted without the scenario they score. In-sample figures are reported separately.
"""
import argparse
import dataclasses
import hashlib
import json
import random
from pathlib import Path

from govia_tracks.association import AcceptancePolicy, select_links
from govia_tracks.baseline import analyze as baseline_analyze, config_version, load_config
from govia_tracks.candidates import generate_candidates
from govia_tracks.evaluation import (
    LinkCounts, count_links, hide_altitude, hide_altitude_of_some, labels_without, metrics, noisy_altitude, outside_visible_windows,
    random_fragments, split_tracklets, trim_reports, without_tracklets,
)
from govia_tracks.labels import load_labels
from govia_tracks.learned import analyze, load_model, load_policy, score_candidates
from govia_tracks.loader import load_snapshot
from govia_tracks.training import build_model, out_of_fold_estimators, save_model, training_examples

DATA = Path("candidate-packet/data")
MODELS = Path("models")
CONFIG = Path("configs/learned.json")
REPORTS = Path("reports")
PARTITIONS = ("train", "validation")  # test and staged data are not reachable from this script
BALANCED_RATIO, PERMISSIVE_RATIO = 5, 1
CONSERVATIVE_MIN_RECALL = 0.97
MIN_SCORES = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9)
MIN_MARGINS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6)
PLAUSIBLE_SCORES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
STRESS_SEEDS = (0, 1, 2)


def folders(partition):
    if partition not in PARTITIONS:
        raise ValueError(f"only {PARTITIONS} are available here, got {partition!r}")
    return sorted((DATA / partition).glob("scenario_*"))


def write_json(path, value):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def train_examples():
    return training_examples((folder.name, load_snapshot(folder), load_labels(folder)) for folder in folders("train"))


def train():
    model = build_model(train_examples())
    save_model(model, MODELS)
    print(json.dumps({"version": model.version, **model.metadata["training"]}, indent=1))


def model_chooser(model):
    """Returns (partition, scenario name) -> model. Train scenarios get a model that never saw them; validation gets the saved one."""
    held_out = out_of_fold_estimators(train_examples(), model.metadata["parameters"], model.feature_names, altitude_hidden_copy=model.metadata["altitude_hidden_copy"])
    names = {id(estimator): index for index, estimator in enumerate(dict.fromkeys(held_out.values()))}
    fold_models = {
        name: dataclasses.replace(model, estimator=estimator, metadata={**model.metadata, "model_version": f"{model.version}-fold{names[id(estimator)]}"})
        for name, estimator in held_out.items()
    }
    return lambda partition, name: fold_models[name] if partition == "train" else model


def scored_partition(choose, partition):
    """Score once, then every policy reuses the same scores. Returns (scored candidates, labels, candidate pairs, name)."""
    loaded = []
    for folder in folders(partition):
        model = choose(partition, folder.name)
        candidates = generate_candidates(load_snapshot(folder), model.candidates).candidates
        loaded.append((score_candidates(model, candidates), load_labels(folder), {(c.predecessor, c.successor) for c in candidates}, folder.name))
    return loaded


def sweep(loaded, policy):
    total = LinkCounts()
    for scored, labels, pairs, name in loaded:
        total.add(count_links(select_links(scored, policy)[0], labels, pairs, name))
    return total


def select():
    model = load_model(MODELS)
    choose = model_chooser(model)
    loaded = {partition: scored_partition(choose, partition) for partition in PARTITIONS}
    grid = []
    for plausible in PLAUSIBLE_SCORES:
        for min_score in MIN_SCORES:
            if plausible > min_score:
                continue
            for margin in MIN_MARGINS:
                policy = AcceptancePolicy(min_score, margin, plausible)
                row = dict(min_score=min_score, min_margin=margin, plausible_score=plausible)
                for partition in PARTITIONS:
                    counts = metrics(sweep(loaded[partition], policy))
                    row[partition] = {k: counts[k] for k in ("wrong", "wrong_skips", "wrong_other_aircraft", "correct", "accepted", "precision", "recall", "labeled_links")}
                grid.append(row)
    by_setting = {(r["min_score"], r["plausible_score"], r["min_margin"]): r for r in grid}

    def guarded_wrong(row, partition):
        index = MIN_MARGINS.index(row["min_margin"])
        lower = by_setting[(row["min_score"], row["plausible_score"], MIN_MARGINS[max(index - 1, 0)])]
        return max(row[partition]["wrong"], lower[partition]["wrong"])

    def cost(row, partition, ratio):
        return ratio * guarded_wrong(row, partition) + row[partition]["labeled_links"] - row[partition]["correct"]

    def best(ratio):
        return min(grid, key=lambda r: (cost(r, "validation", ratio), cost(r, "train", ratio), -r["min_margin"], -r["min_score"], -r["plausible_score"]))

    def pooled_recall(row):
        return (row["train"]["correct"] + row["validation"]["correct"]) / (row["train"]["labeled_links"] + row["validation"]["labeled_links"])

    safe = [r for r in grid if r["train"]["wrong"] == 0 and r["validation"]["wrong"] == 0 and pooled_recall(r) >= CONSERVATIVE_MIN_RECALL]
    policies = {
        "conservative": max(safe, key=lambda r: (r["min_margin"], r["min_score"], pooled_recall(r), r["plausible_score"])) if safe else None,
        "balanced": best(BALANCED_RATIO),
        "permissive": best(PERMISSIVE_RATIO),
    }
    chosen = policies["balanced"]
    write_json(CONFIG, dict(
        model_version=model.version, cost_ratio=BALANCED_RATIO, selected_on="validation, then out-of-fold train (development; both explored earlier)",
        policy={k: chosen[k] for k in ("min_score", "min_margin", "plausible_score")},
    ))
    write_json(REPORTS / "learned_selection.json", dict(
        model_version=model.version, operating_policy="balanced", balanced_ratio=BALANCED_RATIO, permissive_ratio=PERMISSIVE_RATIO,
        conservative_min_recall=CONSERVATIVE_MIN_RECALL, policies=policies, grid_size=len(grid),
    ))
    for name, row in policies.items():
        print(name, None if row is None else {k: row[k] for k in ("min_score", "min_margin", "plausible_score", "validation", "train")})
    print(f"wrote {CONFIG}; grid of {len(grid)}")


def run_method(partition, analyzer, perturb=lambda s: s):
    """`analyzer(scenario name, scenario)`. Neither method receives labels."""
    total = LinkCounts()
    results = []
    for folder in folders(partition):
        result = analyzer(folder.name, perturb(load_snapshot(folder)))
        pairs = {(t["tracklet_id"], c["successor"]) for t in result["tracklets"] for c in t["candidates"]}
        total.add(count_links(result["links"], load_labels(folder), pairs, folder.name))
        results.append(result)
    return total, results


def digest(results):
    return hashlib.sha256(json.dumps(results, sort_keys=True, allow_nan=False).encode()).hexdigest()


def comparison_methods(partition, choose, policy, baseline_config):
    return {
        "baseline": lambda name, scenario: baseline_analyze(scenario, baseline_config),
        "learned": lambda name, scenario: analyze(scenario, choose(partition, name), policy),
    }


def evaluate():
    model = load_model(MODELS)
    policy = load_policy(CONFIG, model.version)
    baseline_config = load_config("configs/baseline.json")
    choose = model_chooser(model)
    selection = json.loads((REPORTS / "learned_selection.json").read_text(encoding="utf-8"))
    report = dict(
        model_version=model.version, baseline_version=config_version(baseline_config), policy=dataclasses.asdict(policy),
        note="learned on train is out of fold unless a key says in_sample",
        partitions={}, altitude_hidden_at_inference={}, disagreements={}, learned_policies={},
    )
    for name, row in selection["policies"].items():
        if row is not None:
            other = AcceptancePolicy(row["min_score"], row["min_margin"], row["plausible_score"])
            report["learned_policies"][name] = {
                p: metrics(run_method(p, lambda n, s, o=other, p=p: analyze(s, choose(p, n), o))[0]) for p in PARTITIONS
            }
    for partition in PARTITIONS:
        report["partitions"][partition] = {}
        report["altitude_hidden_at_inference"][partition] = {}
        links = {}
        for method, analyzer in comparison_methods(partition, choose, policy, baseline_config).items():
            counts, results = run_method(partition, analyzer)
            report["partitions"][partition][method] = {**metrics(counts), "excluded_links": [list(x) for x in counts.excluded_links], "results_sha256": digest(results)}
            links[method] = {(r["scenario_id"], a, b) for r in results for a, b in r["links"].items()}
            report["altitude_hidden_at_inference"][partition][method] = metrics(run_method(partition, analyzer, hide_altitude)[0])
        report["disagreements"][partition] = dict(
            only_baseline=len(links["baseline"] - links["learned"]), only_learned=len(links["learned"] - links["baseline"]), both=len(links["baseline"] & links["learned"]),
        )
    report["partitions"]["train"]["learned_in_sample"] = metrics(run_method("train", lambda n, s: analyze(s, model, policy))[0])
    write_json(REPORTS / "learned_metrics.json", report)
    print(json.dumps(report, indent=1, sort_keys=True))


def _stress_rows(transform, partition, methods, seeds):
    """Both methods on identical perturbed inputs. `transform(scenario, labels, rng)` returns (scenario, labels)."""
    totals = {name: LinkCounts() for name in methods}
    null_with_candidate = {name: 0 for name in methods}
    null_linked = {name: 0 for name in methods}
    for seed in seeds:
        for folder in folders(partition):
            for name, analyzer in methods.items():
                # Each method gets its own generator from the same seed, so both see identical perturbations.
                scenario, labels = transform(load_snapshot(folder), load_labels(folder), random.Random(f"{seed}/{folder.name}"))
                result = analyzer(folder.name, scenario)
                totals[name].add(count_links(result["links"], labels))
                for tracklet in result["tracklets"]:
                    if labels[tracklet["tracklet_id"]] is None:
                        null_with_candidate[name] += bool(tracklet["candidates"])
                        null_linked[name] += tracklet["accepted_successor"] is not None
    return {
        name: {**metrics(totals[name]), "null_tracklets_with_candidate": null_with_candidate[name], "null_tracklets_linked": null_linked[name]}
        for name in methods
    }


def _dropping(choose):
    def transform(scenario, labels, rng):
        dropped = choose(labels, rng)
        return without_tracklets(scenario, dropped), labels_without(labels, dropped)
    return transform


def stress():
    """Frozen models and settings throughout. Nothing here is fitted or tuned on these results."""
    model = load_model(MODELS)
    policy = load_policy(CONFIG, model.version)
    baseline_config = load_config("configs/baseline.json")
    choose = model_chooser(model)
    window = _dropping(outside_visible_windows)
    cases = {
        "altitude_hidden": (lambda s, l, _: (hide_altitude(s), l), (0,)),
        "altitude_hidden_half_of_tracklets": (lambda s, l, rng: (hide_altitude_of_some(s, 0.5, rng), l), STRESS_SEEDS),
        "trimmed_25pct_each_end": (lambda s, l, _: (trim_reports(s, 0.25), l), (0,)),
        "altitude_noise_100ft": (lambda s, l, rng: (noisy_altitude(s, 100.0, rng), l), STRESS_SEEDS),
        "altitude_noise_300ft": (lambda s, l, rng: (noisy_altitude(s, 300.0, rng), l), STRESS_SEEDS),
        "altitude_noise_1000ft": (lambda s, l, rng: (noisy_altitude(s, 1000.0, rng), l), STRESS_SEEDS),
        "visible_windows": (window, STRESS_SEEDS),
        "visible_windows_altitude_hidden": (lambda s, l, rng: (lambda pair: (hide_altitude(pair[0]), pair[1]))(window(s, l, rng)), STRESS_SEEDS),
        "drop_30pct_of_fragments": (_dropping(lambda l, rng: random_fragments(l, 0.3, rng)), STRESS_SEEDS),
        "split_tracklets_15s": (lambda s, l, _: split_tracklets(s, l, 15.0)[:2], (0,)),
        "split_tracklets_5s": (lambda s, l, _: split_tracklets(s, l, 5.0)[:2], (0,)),
    }
    report = dict(
        model_version=model.version, baseline_version=config_version(baseline_config), seeds=list(STRESS_SEEDS),
        note="learned on train is out of fold; models and settings are frozen; both methods see identical perturbations", cases={},
    )
    for name, (transform, seeds) in cases.items():
        report["cases"][name] = {
            partition: _stress_rows(transform, partition, comparison_methods(partition, choose, policy, baseline_config), seeds)
            for partition in PARTITIONS
        }
        for partition, rows in report["cases"][name].items():
            print(name, partition, {m: {k: rows[m][k] for k in ("correct", "labeled_links", "wrong", "wrong_skips", "wrong_other_aircraft", "null_tracklets_with_candidate", "null_tracklets_linked")} for m in rows})
    write_json(REPORTS / "learned_stress.json", report)


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("command", choices=("train", "select", "evaluate", "stress"))
{"train": train, "select": select, "evaluate": evaluate, "stress": stress}[parser.parse_args().command]()
