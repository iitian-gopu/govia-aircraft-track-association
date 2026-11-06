"""End to end: real uvicorn and Streamlit processes, the saved models, the supplied records, and real HTTP.

Every process is started on a free port by a fixture, with a time limit on startup, and is always terminated.
Expected values come from raw observations, saved artifacts or the API's own links, never from labels: this file
opens no labels file, and the test partition is only ever asked for to see it refused.
"""
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import httpx
import pytest
import sklearn
from streamlit.testing.v1 import AppTest

from govia_tracks import baseline as baseline_module

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "candidate-packet" / "data"
APP = ROOT / "dashboard" / "app.py"
ASSOCIATE = "/tracklets/associate"
START_LIMIT_S = 60
STOP_LIMIT_S = 15
APP_TIMEOUT_S = 90
ISSUE_FIELDS = {"code", "record_kind", "message", "index", "record_id", "field"}
THROUGH_THE_API = ["train/scenario_001", "validation/scenario_001", "staged_demo/snapshot_1", "staged_demo/snapshot_2"]
VALIDATION = [folder.name for folder in sorted((DATA / "validation").glob("scenario_*"))]
STAGED = ("staged_demo/snapshot_1", "staged_demo/snapshot_2")
METHODS = ("baseline", "learned")
ALTITUDE_MEASUREMENTS = ("altitude_diff_ft", "altitude_miss_ft")
WINDOWS_OR_DRIVE_PATH = re.compile(r"\b[A-Za-z]:(\\|/)")
TRACKLET_ID = re.compile(r"tracklet-[0-9a-fA-F-]+")
# Row-count and delivery facts legitimately describe the rows as delivered, so they alone may differ.
DELIVERY_FACTS = {"observation_rows", "supporting_rows", "tracklets_delivered_out_of_order"}

pytestmark = pytest.mark.data
if not DATA.is_dir():
    pytest.skip("the supplied data package is not present", allow_module_level=True)


# Processes


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def process_env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("GOVIA_ROOT", "GOVIA_API_URL")}
    return {**env, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", **extra}


class Running:
    def __init__(self, process, port, log_path):
        self.process, self.port, self.log_path = process, port, log_path
        self.url = f"http://127.0.0.1:{port}"
        self.http = httpx.Client(base_url=self.url, timeout=60, trust_env=False)

    def log(self) -> str:
        return self.log_path.read_text(encoding="utf-8", errors="replace")


@contextmanager
def running(command, port, log_path, env, ready_path):
    """Start a process, wait for any HTTP answer on `ready_path` within the limit, and always terminate it."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        server = Running(process, port, log_path)
        try:
            deadline = time.monotonic() + START_LIMIT_S
            while True:
                if process.poll() is not None:
                    raise AssertionError(f"exited with {process.returncode} during startup:\n{server.log()[-2000:]}")
                try:
                    server.http.get(ready_path, timeout=2)
                    break
                except httpx.TransportError:
                    if time.monotonic() > deadline:
                        raise AssertionError(f"no answer on {ready_path} within {START_LIMIT_S} s:\n{server.log()[-2000:]}") from None
                    time.sleep(0.2)
            yield server
        finally:
            server.http.close()
            process.terminate()
            try:
                process.wait(STOP_LIMIT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(STOP_LIMIT_S)


def uvicorn_command(port):
    return [sys.executable, "-m", "uvicorn", "govia_tracks.api:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "info"]


@contextmanager
def api_server(log_path, port=None, root=None):
    port = port or free_port()
    env = process_env(**({"GOVIA_ROOT": str(root)} if root else {}))
    with running(uvicorn_command(port), port, log_path, env, "/health") as server:
        yield server


def tree_hashes():
    hashes = {}
    for folder in ("models", "configs", "reports"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file():
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


@pytest.fixture(scope="module", autouse=True)
def artifacts_stay_untouched():
    """Nothing in this run, including serving requests, may train or rewrite a saved artifact."""
    before = tree_hashes()
    yield
    assert tree_hashes() == before


class Api:
    """One real server, with each supplied scenario fetched and analysed once for the whole module."""

    def __init__(self, server):
        self.server, self.url, self.http = server, server.url, server.http
        self._records, self._responses = {}, {}

    def records(self, key):
        if key not in self._records:
            response = self.http.get(f"/scenarios/{key}/records")
            assert response.status_code == 200, response.text
            self._records[key] = response.json()
        return self._records[key]

    def payload(self, key):
        return payload_from(self.records(key))

    def response(self, key):
        if key not in self._responses:
            self._responses[key] = self.http.post(ASSOCIATE, json=self.payload(key))
            assert self._responses[key].status_code == 200, self._responses[key].text
        return self._responses[key]

    def body(self, key):
        return strict_json(self.response(key).text)


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    with api_server(tmp_path_factory.mktemp("api") / "uvicorn.log") as server:
        yield Api(server)
        assert "Traceback" not in server.log()


@pytest.fixture(scope="module")
def no_model_root(tmp_path_factory):
    """A root with the configs and one scenario's input files, and no models. Labels are not copied."""
    root = tmp_path_factory.mktemp("no_models")
    shutil.copytree(ROOT / "configs", root / "configs")
    folder = root / "candidate-packet" / "data" / "validation" / "scenario_001"
    folder.mkdir(parents=True)
    for name in ("observations.jsonl", "supporting.jsonl", "snapshot.json"):
        shutil.copy(DATA / "validation" / "scenario_001" / name, folder / name)
    return root


@pytest.fixture(scope="module")
def no_model_api(no_model_root, tmp_path_factory):
    with api_server(tmp_path_factory.mktemp("no_model_api") / "uvicorn.log", root=no_model_root) as server:
        yield server


# Shared checks


def payload_from(records):
    return copy.deepcopy({key: records[key] for key in ("observations", "supporting_records", "scenario_id", "snapshot_id", "receipt_cutoff")})


def strict_json(text):
    def refuse(name):
        raise AssertionError(f"{name} in a response")
    return json.loads(text, parse_constant=refuse)


def manifest(key):
    return json.loads((DATA / key / "snapshot.json").read_text(encoding="utf-8"))


def assert_no_paths(text, *paths):
    assert "Traceback" not in text and 'File "' not in text
    assert not WINDOWS_OR_DRIVE_PATH.search(text), text[:500]
    for path in (ROOT, *paths):
        assert str(path) not in text and json.dumps(str(path))[1:-1] not in text and Path(path).as_posix() not in text
    for fragment in (".joblib", "learned_model.json", "candidate-packet", "site-packages"):
        assert fragment not in text


def assert_error(response, status, code):
    assert response.status_code == status, response.text
    body = strict_json(response.text)
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "issues"}
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    assert all(set(issue) == ISSUE_FIELDS for issue in body["error"]["issues"])
    assert_no_paths(response.text)
    return body["error"]


def moment_us(text):
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return round(parsed.timestamp() * 1_000_000)


def raw_spans(observations):
    """{tracklet: (first observed, last observed)}, in microseconds, straight from the raw rows."""
    spans = {}
    for row in observations:
        moment = moment_us(row["observed_at"])
        start, end = spans.get(row["tracklet_id"], (moment, moment))
        spans[row["tracklet_id"]] = (min(start, moment), max(end, moment))
    return spans


def finite_or_null(value):
    if isinstance(value, dict):
        return all(finite_or_null(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_or_null(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def assert_links_are_structurally_valid(links, spans):
    assert len(set(links.values())) == len(links), "a tracklet has two accepted predecessors"
    for a, b in links.items():
        assert a != b, f"self-link on {a}"
        assert a in spans and b in spans, f"{a} -> {b} refers to a tracklet not in this input"
        assert spans[b][0] > spans[a][1], f"{a} -> {b} doesn't move strictly forward in time"
    for start in links:
        seen, current = {start}, links[start]
        while current in links:
            assert current not in seen, f"cycle through {current}"
            seen.add(current)
            current = links[current]


def assert_result_invariants(body, observations):
    spans = raw_spans(observations)
    for method in METHODS:
        result = body[method]
        links = result["links"]
        assert finite_or_null(result)
        assert_links_are_structurally_valid(links, spans)
        assert {t["tracklet_id"] for t in result["tracklets"]} == set(spans)
        for tracklet in result["tracklets"]:
            tracklet_id, candidates = tracklet["tracklet_id"], tracklet["candidates"]
            assert tracklet["accepted_successor"] == links.get(tracklet_id)
            assert tracklet["status"] in ("accepted", "unresolved")
            assert isinstance(tracklet["reason"], str) and tracklet["reason"]
            scores = [c["score"] for c in candidates]
            assert scores == sorted(scores, reverse=True), f"{method}: candidates of {tracklet_id} aren't ranked by score"
            assert [c["rank"] for c in candidates] == list(range(1, len(candidates) + 1))
            successors = [c["successor"] for c in candidates]
            assert len(set(successors)) == len(successors)
            for candidate in candidates:
                assert candidate["successor"] != tracklet_id
                assert spans[candidate["successor"]][0] > spans[tracklet_id][1]
                raw_gap_s = (spans[candidate["successor"]][0] - spans[tracklet_id][1]) / 1e6
                assert candidate["measurements"]["gap_s"] == pytest.approx(raw_gap_s, abs=1e-3)
                assert candidate["decision_detail"]
            accepted = [c for c in candidates if c["decision"] == "accepted"]
            if tracklet["status"] == "accepted":
                assert tracklet_id in links and tracklet["reason_code"] == "accepted"
                assert [c["successor"] for c in accepted] == [links[tracklet_id]]
            else:
                assert tracklet_id not in links and tracklet["accepted_successor"] is None
                assert tracklet["reason_code"] != "accepted" and not accepted
                assert tracklet["reason_code"] == (candidates[0]["decision"] if candidates else "no_candidate")
        all_scores = [c["score"] for t in result["tracklets"] for c in t["candidates"]]
        if method == "learned":
            assert all(0.0 <= score <= 1.0 for score in all_scores)
        else:
            assert all(score <= 0.0 for score in all_scores)


# Health


def test_health_is_ready_and_versions_match_the_saved_artifacts(api):
    response = api.http.get("/health")
    assert response.status_code == 200
    body = strict_json(response.text)
    metadata = json.loads((ROOT / "models" / "learned_model.json").read_text(encoding="utf-8"))
    learned_config = json.loads((ROOT / "configs" / "learned.json").read_text(encoding="utf-8"))
    baseline_report = json.loads((ROOT / "reports" / "baseline_metrics.json").read_text(encoding="utf-8"))
    assert body["ready"] is True and body["status"] == "ready" and body["problems"] == []
    assert body["learned"]["available"] is True and body["baseline"]["available"] is True
    assert body["learned"]["version"] == metadata["model_version"] == learned_config["model_version"]
    assert body["learned"]["feature_names"] == metadata["feature_names"]
    assert body["baseline"]["version"] == baseline_report["version"]
    assert body["baseline"]["version"] == baseline_module.config_version(baseline_module.load_config(ROOT / "configs" / "baseline.json"))
    assert hashlib.sha256((ROOT / "models" / "learned_model.joblib").read_bytes()).hexdigest() == metadata["artifact_sha256"]
    assert metadata["library_versions"]["scikit-learn"] == sklearn.__version__


# Supplied scenarios, through records then associate


def test_scenario_list_holds_exactly_the_allowed_partitions(api):
    response = api.http.get("/scenarios")
    assert response.status_code == 200
    partitions = [s["partition"] for s in response.json()["scenarios"]]
    assert {p: partitions.count(p) for p in set(partitions)} == {"train": 80, "validation": 20, "staged_demo": 2}
    assert "label" not in response.text.lower()
    assert_no_paths(response.text)


@pytest.mark.parametrize("key", THROUGH_THE_API)
def test_supplied_scenario_runs_through_records_and_associate(api, key):
    health = api.http.get("/health").json()
    saved = manifest(key)
    records, body, text = api.records(key), api.body(key), api.response(key).text
    assert set(body) == {"analysis", "baseline", "learned"}

    analysis = body["analysis"]
    assert (analysis["scenario_id"], analysis["snapshot_id"], analysis["receipt_cutoff"]) == (saved["scenario_id"], saved["snapshot_id"], saved["receipt_cutoff"])
    assert analysis["tracklet_count"] == saved["tracklet_count"]
    assert analysis["observation_count"] == saved["unique_observation_count"]
    assert analysis["quality"]["observation_rows"] == saved["observation_row_count"] == len(records["observations"])
    assert analysis["quality"]["supporting_rows"] == saved["supporting_row_count"] == len(records["supporting_records"])

    configs = {name: json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8")) for name in METHODS}
    for method in METHODS:
        result = body[method]
        assert result["method"] == method
        assert result["version"] == health[method]["version"]
        assert result["policy"] == configs[method]["policy"]
        assert result["config"], "the decision configuration must be identified"
        assert result["input_fingerprint"] == analysis["input_fingerprint"]
        assert (result["scenario_id"], result["snapshot_id"], result["receipt_cutoff"]) == (saved["scenario_id"], saved["snapshot_id"], saved["receipt_cutoff"])
        assert result["links"], f"{method} linked nothing on {key}"
        assert [c for t in result["tracklets"] for c in t["candidates"]], f"{method} scored nothing on {key}"
    assert_result_invariants(body, records["observations"])
    assert "label" not in text.lower() and "next_tracklet" not in text
    assert_no_paths(text)


@pytest.mark.parametrize("name", VALIDATION)
def test_validation_invariants_hold_from_raw_observations(api, name):
    key = f"validation/{name}"
    assert_result_invariants(api.body(key), api.records(key)["observations"])


def test_all_twenty_validation_scenarios_are_present():
    assert len(VALIDATION) == 20


def test_baseline_responses_equal_the_saved_validation_results(api):
    saved = json.loads((ROOT / "reports" / "baseline_results_validation.json").read_text(encoding="utf-8"))
    assert len(saved) == len(VALIDATION)
    for name, expected in zip(VALIDATION, saved):
        assert api.body(f"validation/{name}")["baseline"] == expected, name


# Order, retransmission and missing altitude, over the wire


def retransmitted_and_shuffled(payload, seed):
    """Shuffled rows plus copies, some with their keys in another order: JSON objects are unordered."""
    rng = random.Random(seed)
    result = copy.deepcopy(payload)
    for field in ("observations", "supporting_records"):
        rows = result[field]
        copies = rng.sample(rows, k=min(25, len(rows)))
        rows.extend(dict(reversed(list(row.items()))) if i % 2 else copy.deepcopy(row) for i, row in enumerate(copies))
        rng.shuffle(rows)
    return result


def assert_same_decisions(again, original):
    assert again["analysis"]["input_fingerprint"] == original["analysis"]["input_fingerprint"]
    for method in METHODS:
        assert again[method]["links"] == original[method]["links"]
        assert again[method] == original[method]
    quiet = lambda analysis: {k: v for k, v in analysis.items() if k != "quality"}  # noqa: E731
    assert quiet(again["analysis"]) == quiet(original["analysis"])
    facts = lambda analysis: {k: v for k, v in analysis["quality"].items() if k not in DELIVERY_FACTS}  # noqa: E731
    assert facts(again["analysis"]) == facts(original["analysis"])


@pytest.mark.parametrize("key", THROUGH_THE_API)
@pytest.mark.parametrize("seed", [1, 2])
def test_shuffled_rows_and_added_duplicates_change_nothing(api, key, seed):
    payload = api.payload(key)
    response = api.http.post(ASSOCIATE, json=retransmitted_and_shuffled(payload, seed))
    assert response.status_code == 200, response.text
    assert_same_decisions(response.json(), api.body(key))


@pytest.mark.parametrize("key", THROUGH_THE_API)
def test_sorted_rows_without_any_duplicates_change_nothing(api, key):
    payload = api.payload(key)
    payload["observations"] = sorted({r["observation_id"]: r for r in payload["observations"]}.values(), key=lambda r: r["observation_id"])
    payload["supporting_records"] = sorted({r["record_id"]: r for r in payload["supporting_records"]}.values(), key=lambda r: r["record_id"])
    response = api.http.post(ASSOCIATE, json=payload)
    assert response.status_code == 200, response.text
    assert_same_decisions(response.json(), api.body(key))


@pytest.mark.parametrize("key", ["validation/scenario_001", "staged_demo/snapshot_2"])
@pytest.mark.parametrize("form", ["absent", "null"])
def test_missing_altitude_on_every_report_is_null_never_zero(api, key, form):
    payload = api.payload(key)
    for row in payload["observations"]:
        row.pop("altitude_ft", None)
        if form == "null":
            row["altitude_ft"] = None
    response = api.http.post(ASSOCIATE, json=payload)
    assert response.status_code == 200, response.text
    body = strict_json(response.text)
    quality = body["analysis"]["quality"]
    unique = body["analysis"]["observation_count"]
    assert (quality["altitude_null"], quality["altitude_absent"]) == ((unique, 0) if form == "null" else (0, unique))
    assert body["analysis"]["input_fingerprint"] != api.body(key)["analysis"]["input_fingerprint"]
    assert_result_invariants(body, payload["observations"])
    penalty = json.loads((ROOT / "configs" / "baseline.json").read_text(encoding="utf-8"))["missing_penalty"]
    for method in METHODS:
        candidates = [c for t in body[method]["tracklets"] for c in t["candidates"]]
        assert candidates
        for candidate in candidates:
            for name in ALTITUDE_MEASUREMENTS:
                assert name in candidate["measurements"] and candidate["measurements"][name] is None
            assert candidate["measurements"]["gap_s"] is not None
            if method == "baseline":
                assert candidate["measurements"]["score_terms"]["altitude"] == penalty


# Errors


def lone_row(rows, id_field):
    """The index of a row with no retransmission, so changing it can't also make a conflict."""
    counts = {}
    for row in rows:
        counts[row[id_field]] = counts.get(row[id_field], 0) + 1
    return next(i for i, row in enumerate(rows) if counts[row[id_field]] == 1)


def changed_observation(**changes):
    def make(payload):
        payload["observations"][lone_row(payload["observations"], "observation_id")].update(changes)
        return {"json": payload}
    return make


def without_observation_field(name):
    def make(payload):
        del payload["observations"][lone_row(payload["observations"], "observation_id")][name]
        return {"json": payload}
    return make


def changed_supporting(**changes):
    def make(payload):
        payload["supporting_records"][lone_row(payload["supporting_records"], "record_id")].update(changes)
        return {"json": payload}
    return make


def raw_body(content):
    return lambda payload: {"content": content, "headers": {"content-type": "application/json"}}


def replaced(**changes):
    def make(payload):
        payload.update(changes)
        return {"json": payload}
    return make


def only(**fields):
    return lambda payload: {"json": fields}


ERROR_MATRIX = [
    ("malformed_json", raw_body(b"{not json"), 400, "invalid_json", None),
    ("empty_body", raw_body(b""), 400, "invalid_json", None),
    ("nan_in_body", raw_body(b'{"observations": [{"latitude": NaN}]}'), 400, "invalid_json", None),
    ("infinite_number", raw_body(b'{"observations": [1e999]}'), 400, "invalid_json", None),
    ("repeated_key", raw_body(b'{"observations": [], "observations": []}'), 400, "invalid_json", None),
    ("not_utf8", raw_body(b"\xff\xfe\xfa"), 400, "invalid_json", None),
    ("array_body", raw_body(b"[]"), 422, "invalid_request", None),
    ("observations_absent", only(supporting_records=[]), 422, "invalid_request", "observations"),
    ("observations_not_a_list", only(observations="nope"), 422, "invalid_request", "observations"),
    ("labels_supplied", replaced(labels=[{"tracklet_id": "secret-value"}]), 422, "invalid_request", "labels"),
    ("snapshot_id_not_text", replaced(snapshot_id=5), 422, "invalid_request", "snapshot_id"),
    ("no_observations", only(observations=[]), 422, "empty_input", None),
    ("empty_observation_id", changed_observation(observation_id=""), 422, "validation_failed", "observation_id"),
    ("blank_tracklet_id", changed_observation(tracklet_id="   "), 422, "validation_failed", "tracklet_id"),
    ("null_tracklet_id", changed_observation(tracklet_id=None), 422, "validation_failed", "tracklet_id"),
    ("numeric_tracklet_id", changed_observation(tracklet_id=7), 422, "validation_failed", "tracklet_id"),
    ("timestamp_without_zone", changed_observation(observed_at="2026-01-10T10:00:00"), 422, "validation_failed", "observed_at"),
    ("timestamp_with_offset", changed_observation(observed_at="2026-01-10T10:00:00+05:30"), 422, "validation_failed", "observed_at"),
    ("timestamp_utc_offset_form", changed_observation(received_at="2026-01-10T10:00:00+00:00"), 422, "validation_failed", "received_at"),
    ("impossible_date", changed_observation(observed_at="2026-02-30T00:00:00.000Z"), 422, "validation_failed", "observed_at"),
    ("epoch_number", changed_observation(received_at=1767996000), 422, "validation_failed", "received_at"),
    ("latitude_too_high", changed_observation(latitude=90.5), 422, "validation_failed", "latitude"),
    ("latitude_too_low", changed_observation(latitude=-90.5), 422, "validation_failed", "latitude"),
    ("longitude_too_high", changed_observation(longitude=180.5), 422, "validation_failed", "longitude"),
    ("longitude_too_low", changed_observation(longitude=-180.5), 422, "validation_failed", "longitude"),
    ("latitude_text", changed_observation(latitude="north"), 422, "validation_failed", "latitude"),
    ("latitude_boolean", changed_observation(latitude=True), 422, "validation_failed", "latitude"),
    ("latitude_null", changed_observation(latitude=None), 422, "validation_failed", "latitude"),
    ("altitude_text", changed_observation(altitude_ft="high"), 422, "validation_failed", "altitude_ft"),
    ("empty_record_id", changed_supporting(record_id=""), 422, "validation_failed", "record_id"),
    ("supporting_timestamp", changed_supporting(received_at="yesterday"), 422, "validation_failed", "received_at"),
    ("supporting_attributes_null", changed_supporting(attributes=None), 422, "validation_failed", "attributes"),
    ("bad_cutoff", replaced(receipt_cutoff="never"), 422, "validation_failed", "receipt_cutoff"),
] + [
    (f"missing_{name}", without_observation_field(name), 422, "validation_failed", name)
    for name in ("observation_id", "tracklet_id", "observed_at", "received_at", "latitude", "longitude", "source")
]


@pytest.mark.parametrize("make, status, code, field", [case[1:] for case in ERROR_MATRIX], ids=[case[0] for case in ERROR_MATRIX])
def test_error_matrix(api, make, status, code, field):
    response = api.http.post(ASSOCIATE, **make(api.payload("validation/scenario_001")))
    error = assert_error(response, status, code)
    if field is not None:
        assert [issue["field"] for issue in error["issues"]] == [field], error
    assert "secret-value" not in response.text


def test_conflicting_observation_is_rejected_and_named(api):
    payload = api.payload("validation/scenario_001")
    twin = {**payload["observations"][0], "latitude": payload["observations"][0]["latitude"] + 0.5}
    payload["observations"].append(twin)
    error = assert_error(api.http.post(ASSOCIATE, json=payload), 422, "validation_failed")
    assert [(i["code"], i["record_id"]) for i in error["issues"]] == [("conflicting_id", twin["observation_id"])]
    assert twin["observation_id"] in error["message"]


def test_conflicting_supporting_record_is_rejected_and_named(api):
    payload = api.payload("validation/scenario_001")
    twin = {**payload["supporting_records"][0], "attributes": {"changed": 1}}
    payload["supporting_records"].append(twin)
    error = assert_error(api.http.post(ASSOCIATE, json=payload), 422, "validation_failed")
    assert [(i["code"], i["record_id"]) for i in error["issues"]] == [("conflicting_id", twin["record_id"])]
    assert twin["record_id"] in error["message"]


def test_rows_received_after_the_cutoff_are_rejected(api):
    payload = api.payload("validation/scenario_001")
    payload["receipt_cutoff"] = "2000-01-01T00:00:00.000Z"
    error = assert_error(api.http.post(ASSOCIATE, json=payload), 422, "validation_failed")
    assert {issue["code"] for issue in error["issues"]} == {"after_cutoff"}


@pytest.mark.parametrize("method, path, status, code", [
    ("GET", "/nowhere", 404, "not_found"),
    ("GET", "/health/", 404, "not_found"),
    ("GET", ASSOCIATE, 405, "method_not_allowed"),
    ("POST", "/health", 405, "method_not_allowed"),
    ("DELETE", "/scenarios", 405, "method_not_allowed"),
    ("GET", "/scenarios/validation/scenario_999/records", 404, "not_found"),
    ("GET", "/scenarios/test/scenario_001/records", 404, "not_found"),
    ("GET", "/scenarios/TEST/scenario_001/records", 404, "not_found"),
    ("GET", "/scenarios/..%2Ftest/scenario_001/records", 404, "not_found"),
    ("GET", "/scenarios/validation/..%2F..%2Ftest%2Fscenario_001/records", 404, "not_found"),
    ("GET", "/scenarios/staged_demo/snapshot_3/records", 404, "not_found"),
])
def test_routes_answer_with_the_error_shape(api, method, path, status, code):
    assert_error(api.http.request(method, path), status, code)


# Restart, missing models, snapshot independence, concurrency, what the process touches


def test_restart_gives_byte_identical_responses(tmp_path):
    port = free_port()
    keys = ["validation/scenario_001", "train/scenario_001", *STAGED]

    def capture(server):
        answers = {"health": server.http.get("/health").content}
        for key in keys:
            records = server.http.get(f"/scenarios/{key}/records").json()
            answers[key] = server.http.post(ASSOCIATE, json=payload_from(records)).content
        return answers

    with api_server(tmp_path / "first.log", port) as server:
        first = capture(server)
        first_pid = server.process.pid
    with pytest.raises(httpx.TransportError):
        httpx.get(f"http://127.0.0.1:{port}/health", timeout=2, trust_env=False)
    with api_server(tmp_path / "second.log", port) as server:
        assert server.process.pid != first_pid
        second = capture(server)
    assert second == first


def test_missing_models_report_503_without_any_path(no_model_api, no_model_root):
    server = no_model_api
    health = server.http.get("/health")
    assert health.status_code == 503
    body = strict_json(health.text)
    assert body["ready"] is False and body["status"] == "unavailable"
    assert body["learned"] == {"available": False, "version": None, "feature_names": None}
    assert body["error"]["code"] == "model_unavailable" and body["problems"]
    assert_no_paths(health.text, no_model_root)

    records = server.http.get("/scenarios/validation/scenario_001/records")
    assert records.status_code == 200
    inference = server.http.post(ASSOCIATE, json=payload_from(records.json()))
    assert_error(inference, 503, "model_unavailable")
    assert_no_paths(inference.text, no_model_root)


def test_each_snapshot_is_analysed_on_its_own(api):
    first_records, second_records = (api.records(key) for key in STAGED)
    first, second = (api.body(key) for key in STAGED)
    first_ids = {row["tracklet_id"] for row in first_records["observations"]}
    second_ids = {row["tracklet_id"] for row in second_records["observations"]}
    assert second_ids - first_ids, "snapshot 2 is expected to hold a tracklet that snapshot 1 doesn't"

    # Every tracklet ID anywhere in snapshot 1's response, including in free text, was in snapshot 1's input.
    assert set(TRACKLET_ID.findall(api.response(STAGED[0]).text)) <= first_ids
    assert set(TRACKLET_ID.findall(api.response(STAGED[1]).text)) <= second_ids

    assert first["analysis"]["input_fingerprint"] != second["analysis"]["input_fingerprint"]
    assert first["analysis"]["snapshot_id"] == manifest(STAGED[0])["snapshot_id"]
    assert second["analysis"]["snapshot_id"] == manifest(STAGED[1])["snapshot_id"]
    assert first["analysis"]["snapshot_id"] != second["analysis"]["snapshot_id"]
    for body in (first, second):
        for method in METHODS:
            assert body[method]["input_fingerprint"] == body["analysis"]["input_fingerprint"]
            assert body[method]["snapshot_id"] == body["analysis"]["snapshot_id"]

    # No state carries between requests: snapshot 1 answers alike after snapshot 2 has been analysed.
    assert api.http.post(ASSOCIATE, json=api.payload(STAGED[1])).status_code == 200
    assert api.http.post(ASSOCIATE, json=api.payload(STAGED[0])).content == api.response(STAGED[0]).content


def test_concurrent_requests_get_the_same_bytes_as_one_at_a_time(api):
    keys = ["validation/scenario_001", "validation/scenario_010", "train/scenario_014", *STAGED]
    expected = {key: api.response(key).content for key in keys}
    payloads = {key: api.payload(key) for key in keys}

    def ask(key):
        with httpx.Client(base_url=api.url, timeout=120, trust_env=False) as http:
            return key, http.post(ASSOCIATE, json=payloads[key]).content

    with ThreadPoolExecutor(12) as pool:
        answers = list(pool.map(ask, keys * 8))
    assert len(answers) == 40
    assert all(content == expected[key] for key, content in answers)


AUDITED_SERVER = """
import os, sys
log = open(sys.argv[1], "a", encoding="utf-8", buffering=1)
def watch(event, args):
    if event in ("open", "os.scandir", "os.listdir") and args and isinstance(args[0], (str, bytes, os.PathLike)):
        log.write(os.fsdecode(args[0]).replace(chr(92), "/") + chr(10))
sys.addaudithook(watch)
import uvicorn
uvicorn.run("govia_tracks.api:app", host="127.0.0.1", port=int(sys.argv[2]), log_level="warning")
"""


def test_the_serving_process_never_touches_labels_or_the_test_partition(tmp_path):
    touched_log = tmp_path / "touched.txt"
    port = free_port()
    command = [sys.executable, "-c", AUDITED_SERVER, str(touched_log), str(port)]
    with running(command, port, tmp_path / "audited.log", process_env(), "/health") as server:
        assert server.http.get("/scenarios").status_code == 200
        for key in ("validation/scenario_001", "train/scenario_001", *STAGED):
            records = server.http.get(f"/scenarios/{key}/records").json()
            assert server.http.post(ASSOCIATE, json=payload_from(records)).status_code == 200
        assert server.http.get("/scenarios/test/scenario_001/records").status_code == 404
    touched = touched_log.read_text(encoding="utf-8").splitlines()
    assert any(p.endswith("validation/scenario_001/observations.jsonl") for p in touched), "the audit hook saw nothing"
    assert not [p for p in touched if p.endswith("labels.jsonl")]
    assert not [p for p in touched if "/data/test" in p]
    for module in ("labels", "training", "evaluation"):
        assert not [p for p in touched if p.endswith(f"govia_tracks/{module}.py")], f"{module} was loaded"


# Dashboard: a real Streamlit server


def drive_session(port):
    """Be a browser, minus the rendering: open Streamlit's websocket, run the page, then press Run association."""
    websockets = pytest.importorskip("websockets.sync.client")
    from streamlit.proto.BackMsg_pb2 import BackMsg
    from streamlit.proto.ForwardMsg_pb2 import ForwardMsg

    def until_finished(socket_):
        elements, deadline = [], time.monotonic() + APP_TIMEOUT_S
        while time.monotonic() < deadline:
            message = ForwardMsg()
            message.ParseFromString(socket_.recv(timeout=APP_TIMEOUT_S))
            kind = message.WhichOneof("type")
            if kind == "script_finished":
                return elements
            if kind == "delta" and message.delta.WhichOneof("type") == "new_element":
                element = message.delta.new_element
                elements.append((element.WhichOneof("type"), element))
            elif kind == "delta" and message.delta.WhichOneof("type") == "new_transient":  # how a spinner arrives
                elements.append(("transient", message.delta.new_transient))
        raise AssertionError("the script did not finish in time")

    with websockets.connect(f"ws://127.0.0.1:{port}/_stcore/stream", subprotocols=["streamlit"], origin=f"http://127.0.0.1:{port}", max_size=None) as ws:
        request = BackMsg()
        request.rerun_script.SetInParent()
        ws.send(request.SerializeToString())
        first = until_finished(ws)
        run = next(element for kind, element in first if kind == "button" and element.button.label == "Run association")
        request = BackMsg()
        widget = request.rerun_script.widget_states.widgets.add()
        widget.id, widget.trigger_value = run.button.id, True
        ws.send(request.SerializeToString())
        return first, until_finished(ws)


def streamlit_command(port):
    return [
        sys.executable, "-m", "streamlit", "run", str(APP), "--server.port", str(port), "--server.address", "127.0.0.1",
        "--server.headless", "true", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false",
    ]


def test_streamlit_serves_and_runs_a_real_session_with_a_clean_log(api, tmp_path):
    port = free_port()
    with running(streamlit_command(port), port, tmp_path / "streamlit.log", process_env(GOVIA_API_URL=api.url), "/_stcore/health") as server:
        health = server.http.get("/_stcore/health")
        assert health.status_code == 200 and health.text.strip() == "ok"
        page = server.http.get("/")
        assert page.status_code == 200 and "<title" in page.text
        before, after = drive_session(port)
        log = server.log()

    assert not [e for kind, e in before + after if kind == "exception"]
    assert {"button", "selectbox", "radio"} <= {kind for kind, _ in before}
    kinds = {kind for kind, _ in after}
    assert {"metric", "plotly_chart", "dataframe"} <= kinds
    body = api.body("validation/scenario_001")
    shown = {e.metric.label: e.metric.body for kind, e in after if kind == "metric"}
    assert shown["Tracklets"] == str(body["analysis"]["tracklet_count"])
    assert shown["Accepted (baseline)"] == str(len(body["baseline"]["links"]))
    for failure in ("Traceback", "Exception", "Unhandled dashboard error"):
        assert failure not in log, log[-2000:]


# Dashboard: AppTest, with the real client against a real server


FORBIDDEN_WORDING = re.compile(r"\b(precision|recall|f1|ground truth|true link|accuracy)\b", re.IGNORECASE)
ALLOWED_CONFIDENCE = "Neither is a confidence value."


def page_text(at):
    parts = [str(e.value) for group in (at.title, at.subheader, at.markdown, at.caption, at.info, at.warning, at.error, at.success) for e in group]
    parts += [m.label for m in at.metric]
    parts += [df.value.to_csv(index=False) for df in at.dataframe]
    return "\n".join(parts)


def assert_clean(at):
    """The page catches its own exceptions, so `at.exception` alone would miss a failure."""
    assert not at.exception, at.exception
    text = page_text(at)
    assert "unexpected problem" not in text, text[:800]
    assert not FORBIDDEN_WORDING.search(text), FORBIDDEN_WORDING.search(text)
    assert "confidence" not in text.replace(ALLOWED_CONFIDENCE, "").lower()
    assert not re.search(r"\bnan\b", text, re.IGNORECASE)
    assert_no_paths(text)
    return text


def start_app(monkeypatch, url):
    monkeypatch.setenv("GOVIA_API_URL", url)
    at = AppTest.from_file(str(APP), default_timeout=APP_TIMEOUT_S)
    at.run()
    assert not at.exception
    return at


def metrics(at):
    return {m.label: m.value for m in at.metric}


def set_view(at, name):
    at.segmented_control(key="view").set_value(name).run()
    return at


def test_dashboard_runs_a_scenario_through_every_view(api, monkeypatch):
    key = "validation/scenario_001"
    body, records = api.body(key), api.records(key)
    at = start_app(monkeypatch, api.url)
    assert_clean(at)
    assert at.sidebar.success[0].value == "API ready"
    assert "Run association" in page_text(at), "the empty state should say what to do"
    assert not at.metric

    at.button(key="run").click().run()
    text = assert_clean(at)
    assert body["analysis"]["input_fingerprint"] in text
    assert body["baseline"]["version"] in text and body["learned"]["version"] in text
    assert len(at.get("plotly_chart")) == 1

    for method in METHODS:
        at.radio(key="method").set_value(method).run()
        text = assert_clean(at)
        tracklets, links = body[method]["tracklets"], body[method]["links"]
        shown = metrics(at)
        assert shown["Tracklets"] == str(len(tracklets))
        assert shown[f"Accepted ({method})"] == str(len(links))
        assert shown[f"Unresolved ({method})"] == str(len(tracklets) - len(links))
        assert shown[f"Candidates ({method})"] == str(sum(len(t["candidates"]) for t in tracklets))
        assert ("no lower bound" in text) if method == "baseline" else ("uncalibrated" in text)

        selected = at.selectbox(key=f"tracklet::{key}").value
        tracklet = next(t for t in tracklets if t["tracklet_id"] == selected)
        assert selected in text, "the inspector should give the full ID"
        table = next(df.value for df in at.dataframe if "Rank" in df.value.columns) if tracklet["candidates"] else None
        if table is not None:
            assert table["Rank"].tolist() == [str(c["rank"]) for c in tracklet["candidates"]]
        reports = next(df.value for df in at.dataframe if "Observation ID" in df.value.columns and "Copies received" in df.value.columns)
        raw_ids = {row["observation_id"] for row in records["observations"] if row["tracklet_id"] == selected}
        assert set(reports["Observation ID"]) == raw_ids

    set_view(at, "Baseline vs learned")
    text = assert_clean(at)
    assert "different scales" in text
    comparison = next(df.value for df in at.dataframe if "Agreement" in df.value.columns)
    agreeing = {t["tracklet_id"] for t in body["baseline"]["tracklets"] if body["baseline"]["links"].get(t["tracklet_id"]) == body["learned"]["links"].get(t["tracklet_id"])}
    assert comparison["Agreement"].tolist().count("Agree") == len(agreeing)
    assert metrics(at)["Methods agree"] == str(len(agreeing))

    set_view(at, "Unresolved")
    text = assert_clean(at)
    assert "does not mean the aircraft stopped" in text
    assert "aircraft stopped" not in text.replace("Unresolved does not mean the aircraft stopped", "")


@pytest.mark.parametrize("key", ["train/scenario_001", "staged_demo/snapshot_2"])
def test_dashboard_runs_other_partitions(api, monkeypatch, key):
    partition, name = key.split("/")
    at = start_app(monkeypatch, api.url)
    at.selectbox(key="partition").select(partition).run()
    at.selectbox(key=f"scenario::{partition}").select(key).run()
    at.button(key="run").click().run()
    assert_clean(at)
    assert metrics(at)["Tracklets"] == str(api.body(key)["analysis"]["tracklet_count"])


def link_changes(before, after):
    """Added, withdrawn, replaced and unchanged, worked out here from nothing but the API's own links."""
    keys = set(before) | set(after)
    return {
        "added": {k for k in keys if k not in before},
        "withdrawn": {k for k in keys if k not in after},
        "replaced": {k for k in keys if k in before and k in after and before[k] != after[k]},
        "unchanged": {k for k in keys if before.get(k) == after.get(k)},
    }


def test_dashboard_staged_comparison_matches_the_api(api, monkeypatch):
    first, second = (api.body(key) for key in STAGED)
    at = start_app(monkeypatch, api.url)
    set_view(at, "Additional evidence")
    assert_clean(at)
    assert "Analyse both snapshots" in page_text(at)
    at.button(key="run_staged").click().run()
    text = assert_clean(at)

    shown = metrics(at)
    new = {t["tracklet_id"] for t in second["baseline"]["tracklets"]} - {t["tracklet_id"] for t in first["baseline"]["tracklets"]}
    assert new
    for method in METHODS:
        changes = link_changes(first[method]["links"], second[method]["links"])
        assert shown[f"New tracklets ({method})"] == str(len(new))
        assert shown[f"Links added ({method})"] == str(len(changes["added"]))
        assert shown[f"Withdrawn ({method})"] == str(len(changes["withdrawn"]))
        assert shown[f"Replaced ({method})"] == str(len(changes["replaced"]))
        assert shown[f"Unchanged ({method})"] == str(len(changes["unchanged"]))

    snapshots = next(df.value for df in at.dataframe if "Fingerprint" in df.value.columns)
    assert snapshots["Fingerprint"].tolist() == [first["analysis"]["input_fingerprint"], second["analysis"]["input_fingerprint"]]
    assert snapshots["Snapshot ID"].tolist() == [first["analysis"]["snapshot_id"], second["analysis"]["snapshot_id"]]
    assert snapshots["Receipt cutoff"].tolist() == [first["analysis"]["receipt_cutoff"], second["analysis"]["receipt_cutoff"]]
    assert "not a change of identity" in text

    # A link replaced through the new fragment, A -> C becoming A -> N and N -> C, is named as exactly that.
    change_tables = [df.value for df in at.dataframe if "Change" in df.value.columns]
    assert len(change_tables) == len(METHODS)
    for method, table in zip(METHODS, change_tables):
        before, after = first[method]["links"], second[method]["links"]
        through_new = [k for k in link_changes(before, after)["replaced"] if after[k] in new and after.get(after[k]) == before[k]]
        between = table["Note"].str.contains("sits between").sum()
        assert between == 2 * len(through_new), table.to_csv()

    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    highlighted = {t["meta"]["tracklet_id"] for t in spec["data"] if t.get("name") == "New tracklet"}
    assert highlighted == new


def test_dashboard_shows_missing_altitude_as_not_available(api, monkeypatch):
    """The page only posts supplied records, and those keep some altitude on each tracklet. So this strips it, gets
    the real API's answer, and hands that to the page as a finished analysis, to read what the page says about nulls."""
    key = "validation/scenario_001"
    payload = api.payload(key)
    for row in payload["observations"]:
        row.pop("altitude_ft", None)
    response = api.http.post(ASSOCIATE, json=payload)
    assert response.status_code == 200, response.text
    at = start_app(monkeypatch, api.url)
    at.session_state["results"] = {key: {"records": {**payload, "key": key}, "response": response.json()}}
    at.run()
    selected = next(t["tracklet_id"] for t in response.json()["baseline"]["tracklets"] if t["candidates"])
    at.selectbox(key=f"tracklet::{key}").select(selected).run()
    assert_clean(at)
    candidates = next(df.value for df in at.dataframe if "Rank" in df.value.columns)
    for column in ("Alt diff (ft)", "Alt miss (ft)"):
        assert set(candidates[column]) == {"not available"}
    reports = next(df.value for df in at.dataframe if "Copies received" in df.value.columns)
    assert set(reports["Altitude (ft)"]) == {"not available"}
    hover = {row[3] for trace in json.loads(at.get("plotly_chart")[0].proto.spec)["data"] for row in trace.get("customdata") or [] if len(row) == 5}
    assert hover == {"not available"}


SLOW_RECORDS_SERVER = """
import asyncio, sys, uvicorn
from govia_tracks.api import app
@app.middleware("http")
async def slow_records(request, call_next):
    if request.url.path.endswith("/records"):
        await asyncio.sleep(1.5)
    return await call_next(request)
uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
"""


def test_loading_state_reaches_a_real_session_when_the_api_is_slow(tmp_path):
    api_port, port = free_port(), free_port()
    with running([sys.executable, "-c", SLOW_RECORDS_SERVER, str(api_port)], api_port, tmp_path / "slow.log", process_env(), "/health") as slow:
        with running(streamlit_command(port), port, tmp_path / "streamlit.log", process_env(GOVIA_API_URL=slow.url), "/_stcore/health"):
            _, after = drive_session(port)
    assert any("Running association" in str(element) for kind, element in after if kind == "transient")
    assert {"metric", "plotly_chart"} <= {kind for kind, _ in after}


def test_dashboard_reports_a_lost_backend_and_recovers(monkeypatch, tmp_path):
    port = free_port()
    with api_server(tmp_path / "before.log", port) as server:
        at = start_app(monkeypatch, server.url)
        at.button(key="run").click().run()
        assert_clean(at)
        assert metrics(at)["Tracklets"]

    at.button(key="run").click().run()
    text = assert_clean(at)
    assert "Backend unavailable" in text and at.sidebar.error[0].value == "API unavailable"
    assert not at.metric

    with api_server(tmp_path / "after.log", port):
        at.button(key="check_api").click().run()
        text = assert_clean(at)
        assert "Backend unavailable" not in text and at.sidebar.success[0].value == "API ready"
        at.button(key="run").click().run()
        assert_clean(at)
        assert metrics(at)["Tracklets"]


def test_dashboard_with_no_backend_at_all_says_so(monkeypatch):
    at = start_app(monkeypatch, f"http://127.0.0.1:{free_port()}")
    text = assert_clean(at)
    assert "Backend unavailable" in text
    assert at.button(key="run").disabled


def test_dashboard_with_missing_models_says_so(no_model_api, no_model_root, monkeypatch):
    at = start_app(monkeypatch, no_model_api.url)
    text = assert_clean(at)
    assert_no_paths(text, no_model_root)
    assert at.sidebar.warning[0].value == "API not ready"
    assert "Model unavailable" in text and "learned model files are missing" in text
    at.button(key="run").click().run()
    text = assert_clean(at)
    assert_no_paths(text, no_model_root)
    assert "Model unavailable" in text
    assert not at.metric


# Confirmed defects. Strict: when one is fixed its test fails here, and the marker should then be removed.


@pytest.mark.parametrize("field", ["scenario_id", "snapshot_id"])
@pytest.mark.parametrize("value", ["", "   "])
def test_defect_blank_snapshot_identifiers_should_be_rejected(api, field, value):
    payload = api.payload("validation/scenario_001")
    payload[field] = value
    assert api.http.post(ASSOCIATE, json=payload).status_code == 422


def test_defect_non_ascii_digits_in_a_timestamp_should_be_rejected(api):
    payload = api.payload("validation/scenario_001")
    index = lone_row(payload["observations"], "observation_id")
    payload["observations"][index]["observed_at"] = "\u0662\u0660\u0662\u0666" + payload["observations"][index]["observed_at"][4:]
    assert api.http.post(ASSOCIATE, json=payload).status_code == 422


def test_defect_evidence_view_should_describe_the_score_scales(api, monkeypatch):
    at = start_app(monkeypatch, api.url)
    set_view(at, "Additional evidence")
    at.button(key="run_staged").click().run()
    text = page_text(at)
    assert any("Score" in df.value.columns for df in at.dataframe)
    assert "uncalibrated" in text and "no lower bound" in text


def test_defect_run_on_the_evidence_view_should_show_an_outcome(api, monkeypatch):
    at = start_app(monkeypatch, api.url)
    set_view(at, "Additional evidence")
    before = page_text(at)
    at.button(key="run").click().run()
    assert at.session_state["results"], "the analysis itself did run"
    assert page_text(at) != before


def test_defect_evidence_view_should_not_blame_the_scenario_list_when_the_backend_is_down(monkeypatch):
    at = start_app(monkeypatch, f"http://127.0.0.1:{free_port()}")
    set_view(at, "Additional evidence")
    assert "does not list" not in page_text(at)
