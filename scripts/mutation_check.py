"""Check that the tests detect real defects: inject each bug into a temporary copy and expect a failure.

Runs the tests that don't need the data package. Exit code 1 if any mutant survives.

Usage: python scripts/mutation_check.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_S = 60
MUTANTS = [
    ("baseline with no change", None, None, None),
    ("booleans compare equal to numbers", "schema.py", "if isinstance(a, bool) or isinstance(b, bool):", "if False:"),
    ("conflicts silently keep the first record", "preprocess.py", "elif not json_equal(rows[first_seen[record_id]], row):", "elif False:"),
    ("sort by receipt time", "preprocess.py", "key=lambda o: (o.observed_us, o.observation_id)", "key=lambda o: (o.received_us, o.observation_id)"),
    ("no sorting at all", "preprocess.py", "ordered = sorted(delivered[tracklet_id], key=lambda o: (o.observed_us, o.observation_id))", "ordered = delivered[tracklet_id]"),
    ("cutoff excludes records exactly at it", "preprocess.py", "if record.received_us > cutoff_us:", "if record.received_us >= cutoff_us:"),
    ("cutoff not enforced", "preprocess.py", "if record.received_us > cutoff_us:", "if False:"),
    ("orphan supporting records are joined", "preprocess.py", "if record.observation_id in known:", "if True:"),
    ("fingerprint depends on row order", "preprocess.py", 'sorted((o.raw for o in observations), key=lambda r: r["observation_id"])', "[o.raw for o in observations]"),
    ("fingerprint treats 1 and 1.0 differently", "preprocess.py", "json.dumps(_canonical(content),", "json.dumps(content,"),
    ("tail altitude read from the wrong end", "movement.py", "np.median(altitudes[-third:])", "np.median(altitudes[:third])"),
    ("missing altitude becomes zero", "schema.py", "None if altitude is None else float(altitude)", "0.0 if altitude is None else float(altitude)"),
    ("non-ASCII digits accepted in timestamps", "schema.py", 'Z", re.ASCII)', 'Z")'),
    ("API accepts unknown fields such as labels", "api.py", 'extra="forbid"', 'extra="ignore"'),
    ("API echoes blank identifiers", "api.py", "if value is not None and not value.strip():", "if False:"),
    ("API lets empty input through", "api.py", "if not request.observations:", "if False:"),
    ("API infers without its models", "api.py", "if not runtime.ready:", "if False:"),
    ("health always says ready", "api.py", '"ready": runtime.ready,', '"ready": True,'),
    ("API serves the test partition", "api.py", 'PARTITION_FOLDERS = {"train": "scenario_*",', 'PARTITION_FOLDERS = {"test": "scenario_*", "train": "scenario_*",'),
    ("API returns only one method", "api.py", '"learned": learned.analyze(prepared, runtime.model, runtime.policy),', ""),
    ("internal errors leak their text", "api.py", '"An internal error occurred."', "str(exc)"),
    ("absurd altitude reaches the arithmetic", "schema.py", "elif altitude is not None and abs(altitude) > _MAX_ALTITUDE_FT:", "elif False:"),
    ("coordinate ranges not checked", "schema.py", "if not -limit <= value <= limit:", "if False:"),
    ("non-Z offsets accepted", "schema.py", r'(?:\.(\d{1,9}))?Z", re.ASCII)', r'(?:\.(\d{1,9}))?(?:Z|[+-]\d{2}:\d{2})", re.ASCII)'),
    ("booleans accepted as numbers", "schema.py", "isinstance(value, (int, float)) and not isinstance(value, bool) and", "isinstance(value, (int, float)) and"),
    ("no longitude shrinkage with latitude", "movement.py", "* math.cos(math.radians(lat0))", ""),
    ("speed fit not robust to outliers", "movement.py", "vn = float(np.median(", "vn = float(np.mean("),
    ("zero-filled altitude median", "movement.py", "altitudes = [o.altitude_ft for o in reports if o.altitude_ft is not None]", "altitudes = [o.altitude_ft or 0.0 for o in reports]"),
    ("longitude not wrapped at the antimeridian", "movement.py", "((lon - lon0 + 180) % 360 - 180)", "(lon - lon0)"),
    ("course measured from east instead of north", "movement.py", "math.atan2(ve, vn)", "math.atan2(vn, ve)"),
    ("manifest count mismatch ignored", "loader.py", "for key, value in actual.items() if manifest[key] != value", "for key, value in actual.items() if False"),
    ("NaN accepted by the JSONL reader", "loader.py", "parse_constant=_reject_constant, ", ""),
    ("equal start and end times allowed", "candidates.py", "if b.start_us <= a.end_us:", "if b.start_us < a.end_us:"),
    ("gap gate off", "candidates.py", "if measurements.gap_s > config.max_gap_s:", "if False:"),
    ("speed gate off", "candidates.py", "if measurements.implied_kmh > config.max_implied_kmh:", "if False:"),
    ("projection gate off", "candidates.py", "if miss is not None and miss > config.max_projection_km:", "if False:"),
    ("heading wraparound ignored", "candidates.py", "course_diff = angle_difference_deg(a.overall.course_deg, b.overall.course_deg)", "course_diff = abs(a.overall.course_deg - b.overall.course_deg)"),
    ("raw endpoint used instead of fitted", "candidates.py", "a_end = a.fitted_end_position or a.end_position", "a_end = a.end_position"),
    ("altitude not projected", "candidates.py", "altitude_miss = sum(misses) / len(misses) if misses else altitude_diff", "altitude_miss = altitude_diff"),
    ("unprojectable altitude reads as agreement", "candidates.py", "if misses else altitude_diff", "if misses else 0.0"),
    ("missing evidence reads as perfect", "baseline.py", "return config.missing_penalty if value is None", "return 0.0 if value is None"),
    ("score direction reversed", "baseline.py", "candidate.predecessor, candidate.successor, -sum(terms.values()),", "candidate.predecessor, candidate.successor, sum(terms.values()),"),
    ("gap does not widen the scale", "baseline.py", "s.position_km + s.position_km_per_s * m.gap_s", "s.position_km"),
    ("unreliable motion accepted", "baseline.py", "acceptable=motion_known", "acceptable=True"),
    ("negative noise floor allowed", "baseline.py", "if intercept < 0:", "if False:"),
    ("skip guard off", "association.py", "if between:", "if False:"),
    ("margin ignored", "association.py", "if rival is not None and c.score - rival.score < policy.min_margin:", "if False:"),
    ("minimum score ignored", "association.py", "elif c.score < policy.min_score:", "elif False:"),
    ("rivals on the successor side ignored", "association.py", "(r.predecessor == c.predecessor or r.successor == c.successor)", "(r.predecessor == c.predecessor)"),
    ("worst score picked first", "association.py", "return (-candidate.score, candidate.predecessor, candidate.successor)", "return (candidate.score, candidate.predecessor, candidate.successor)"),
    ("no tie-break", "association.py", "return (-candidate.score, candidate.predecessor, candidate.successor)", "return (-candidate.score,)"),
    ("two predecessors for one successor", "association.py", "elif c.successor in taken_successors:", "elif False:"),
    ("invariants not checked on results", "association.py", "    check_links(links, candidate_set.motions)\n", ""),
    ("cycle check off", "association.py", "if current in seen:", "if False:"),
    ("recall divides by accepted", "evaluation.py", '"recall": _ratio(counts.correct, counts.labeled_links)', '"recall": _ratio(counts.correct, counts.accepted)'),
    ("zero denominator reported as perfect", "evaluation.py", "return numerator / denominator if denominator else None", "return numerator / denominator if denominator else 1.0"),
    ("skip counted as correct", "evaluation.py", "if (a, b) in true_links:", "if (a, b) in true_links or (a in places and b in places and places[a][0] == places[b][0]):"),
    ("backward link counted as a skip", "evaluation.py", "elif places[a][1] < places[b][1]:", "elif True:"),
    ("self-link counted as a skip", "evaluation.py", "elif places[a][1] < places[b][1]:", "elif places[a][1] <= places[b][1]:"),
    ("unknown predecessor reduces the unresolved count", "evaluation.py", "return self.tracklets - (self.accepted - self.links_from_unlabeled)", "return self.tracklets - self.accepted"),
    ("label structure not validated", "evaluation.py", "    issues = label_structure_issues(labels)\n    if issues:\n", "    issues = label_structure_issues(labels)\n    if False:\n"),
    ("label cycles not reported", "evaluation.py", "if walked.get(current) == start:", "if False:"),
    ("branching labels not reported", "evaluation.py", "if len(tracklets) > 1:", "if False:"),
    ("load_labels skips structure", "labels.py", "    validate_labels(labels)\n    return labels", "    return labels"),
    ("position noise floor ignored", "baseline.py", "max(s.position_km + s.position_km_per_s * m.gap_s, s.position_floor_km)", "s.position_km + s.position_km_per_s * m.gap_s"),
    ("altitude noise floor ignored", "baseline.py", "max(s.altitude_ft + s.altitude_ft_per_s * m.gap_s, s.altitude_floor_ft)", "s.altitude_ft + s.altitude_ft_per_s * m.gap_s"),
    ("one-sided motion accepted by the baseline", "baseline.py", "motion_known = m.forward_miss_km is not None and m.backward_miss_km is not None and", "motion_known = m.position_miss_km is not None and"),
    ("one-sided motion accepted by the learned scorer", "learned.py", "reliable = m.forward_miss_km is not None and m.backward_miss_km is not None and", "reliable = m.position_miss_km is not None and"),
    ("candidate recall ignores excluded links", "evaluation.py", '"candidate_recall": _ratio(counts.candidate_links_retained, counts.labeled_links)', '"candidate_recall": _ratio(counts.candidate_links_retained, counts.candidate_links_retained)'),
    ("manifest keys not checked", "loader.py", "if absent:", "if False:"),
    ("physical limits not applied", "learned.py", "ruled_out=physical_objection(m),", ""),
    ("course limit off", "learned.py", "if m.course_diff_deg is not None and m.course_diff_deg > MAX_COURSE_DIFF_DEG:", "if False:"),
    ("backward miss limit off", "learned.py", '(("forward", m.forward_miss_km), ("backward", m.backward_miss_km))', '(("forward", m.forward_miss_km),)'),
    ("forward miss limit off", "learned.py", '(("forward", m.forward_miss_km), ("backward", m.backward_miss_km))', '(("backward", m.backward_miss_km),)'),
    ("ruled-out pair still a rival", "association.py", "            decisions[_key(c)] = CandidateDecision(PHYSICALLY_INCONSISTENT, c.ruled_out)\n            continue\n", "            decisions[_key(c)] = CandidateDecision(PHYSICALLY_INCONSISTENT, c.ruled_out)\n"),
    ("ruled-out pair still a chain step", "association.py", "if c.score >= policy.plausible_score and c.ruled_out is None}", "if c.score >= policy.plausible_score}"),
    ("partial hiding hides everything", "evaluation.py", "if rng.random() < probability else t", "if True else t"),
    ("trim can empty a tracklet", "evaluation.py", "while cut > 0 and len(reports) - 2 * cut < 3:", "while False:"),
    ("missing model input becomes zero", "learned.py", "row.append(math.nan)", "row.append(0.0)"),
    ("non-finite model input accepted", "learned.py", "elif not math.isfinite(value):", "elif False:"),
    ("score taken from the wrong class", "learned.py", "[:, 1]", "[:, 0]"),
    ("learned accepts unreliable motion", "learned.py", "acceptable=reliable", "acceptable=True"),
    ("changed artifact loads anyway", "learned.py", 'if hashlib.sha256((folder / MODEL_FILE).read_bytes()).hexdigest() != metadata["artifact_sha256"]:', "if False:"),
    ("different feature schema loads anyway", "learned.py", 'if tuple(metadata["feature_names"]) != FEATURES:', "if False:"),
    ("thresholds from another model accepted", "learned.py", "if model_version is not None and saved.get", "if False and saved.get"),
    ("labels imported by the learned path", "learned.py", "from .preprocess import PreparedScenario", "from .preprocess import PreparedScenario\nfrom . import labels"),
    ("training imported by inference", "learned.py", "from .preprocess import PreparedScenario", "from .preprocess import PreparedScenario\nfrom . import evaluation"),
    ("skips labeled as positive", "training.py", "positive = labels[candidate.predecessor] == candidate.successor", "positive = chains[candidate.predecessor] == chains[candidate.successor]"),
    ("no altitude-hidden training copy", "training.py", "    if altitude_hidden_copy:\n", "    if False:\n"),
    ("hidden copy not down-weighted", "training.py", "np.full(len(weights), HIDDEN_COPY_WEIGHT)", "np.ones(len(weights))"),
    ("out-of-fold model sees its own scenarios", "training.py", "fit_estimator(examples.features[fit_rows], examples.targets[fit_rows],", "fit_estimator(examples.features, examples.targets,"),
    ("no fixed seed", "training.py", '"random_state": 0,', '"random_state": None,'),
    ("noise invents altitude where missing", "evaluation.py", "None if o.altitude_ft is None else o.altitude_ft + rng.gauss(0, sd_ft)", "(o.altitude_ft or 0.0) + rng.gauss(0, sd_ft)"),
    ("split leaves labels pointing at old IDs", "evaluation.py", "{a: renamed.get(b, b) for a, b in new_labels.items()}", "new_labels"),
    ("labels imported by the baseline", "baseline.py", "from .preprocess import PreparedScenario", "from .preprocess import PreparedScenario\nfrom . import labels"),
    ("labels imported on the input path", "__init__.py", "from .errors import InputValidationError, Issue", "from . import labels\nfrom .errors import InputValidationError, Issue"),
]


def run(name, filename, old, new):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        shutil.copytree(ROOT / "src", src, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        for artifacts in ("models", "configs"):  # the API's default root is two levels above the package it runs from
            shutil.copytree(ROOT / artifacts, Path(tmp) / artifacts)
        if filename is not None:
            path = src / "govia_tracks" / filename
            text = path.read_text(encoding="utf-8")
            if text.count(old) < 1:
                return "STALE (pattern not found)"
            path.write_text(text.replace(old, new), encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-x", "--tb=no", "-rfE", "-m", "not data", "-p", "no:cacheprovider", str(ROOT / "tests")],
                cwd=tmp, env={**os.environ, "PYTHONPATH": str(src), "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True, text=True, check=False, timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return "tests fail: FAILED by hanging past the time limit"  # e.g. a loop that the removed check would have stopped
        if result.returncode == 0:
            return "tests pass"
        caught_by = [line for line in result.stdout.splitlines() if line.startswith(("FAILED", "ERROR"))]
        return "tests fail: " + (caught_by[0].split(" - ")[0] if caught_by else "no failure line")


bad = 0
for name, filename, old, new in MUTANTS:
    outcome = run(name, filename, old, new)
    # A collection ERROR means the mutant broke imports rather than being caught by a test.
    ok = outcome == "tests pass" if filename is None else outcome.startswith("tests fail: FAILED")
    bad += not ok
    print(f"{'ok      ' if ok else 'PROBLEM '} {name}: {outcome}")
print(f"{len(MUTANTS) - 1} mutants, {bad} problem(s)")
sys.exit(1 if bad else 0)
