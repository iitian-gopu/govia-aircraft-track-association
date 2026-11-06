"""The HTTP API, through TestClient, on small synthetic flights and the real saved artifacts.

Expected links come from the flights' construction: fragments on one straight line belong together, and flights a
long way apart can't. Nothing here opens a labels file, and the test partition is only ever asked for to see it refused.
"""
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pytest
import sklearn
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier

from govia_tracks import baseline as baseline_module
from govia_tracks import learned as learned_module
from govia_tracks.api import create_app

from conftest import observation, supporting

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "candidate-packet" / "data"
EPOCH = datetime(2026, 1, 10, tzinfo=timezone.utc)
SPEED = 0.0012  # degrees of latitude per second: about 480 km/h
ASSOCIATE = "/tracklets/associate"
REAL_RECORDS = "/scenarios/validation/scenario_001/records"
ISSUE_FIELDS = {"code", "record_kind", "message", "index", "record_id", "field"}
GENERIC_500 = "An internal error occurred."


def stamp(seconds):
    return (EPOCH + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def flight(tracklet, start_s, count=6, step_s=10, lon=-108.0, altitude=15000.0):
    """Reports on a northbound line defined from time zero, so fragments of one flight line up across gaps."""
    return [
        observation(
            observation_id=f"{tracklet}-{i}", tracklet_id=tracklet, observed_at=stamp(start_s + i * step_s), received_at=stamp(start_s + i * step_s + 1),
            latitude=35.0 + SPEED * (start_s + i * step_s), longitude=lon, altitude_ft=altitude,
        )
        for i in range(count)
    ]


def two_aircraft():
    """Two aircraft, each split once with a 70 s gap, about 700 km apart. The only links that can be right are A1 -> A2 and B1 -> B2."""
    return flight("A1", 0) + flight("A2", 120) + flight("B1", 0, lon=-100.0) + flight("B2", 120, lon=-100.0)


EXPECTED_LINKS = {"A1": "A2", "B1": "B2"}


def strict_json(text):
    """Refuses NaN and Infinity, which `response.json()` would quietly accept."""
    def refuse(name):
        raise AssertionError(f"{name} in a response")
    return json.loads(text, parse_constant=refuse)


def assert_error(response, status, code):
    assert response.status_code == status, response.text
    body = strict_json(response.text)
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "issues"}
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    assert all(set(issue) == ISSUE_FIELDS for issue in body["error"]["issues"])
    return body["error"]


def assert_no_leak(text, *paths):
    assert "Traceback" not in text
    assert 'File "' not in text
    assert str(ROOT) not in text and json.dumps(str(ROOT))[1:-1] not in text
    for path in paths:
        assert str(path) not in text and json.dumps(str(path))[1:-1] not in text


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(ROOT)) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def lenient_client(client):
    """The same started app, but a server error comes back as a response, as it would over a real connection."""
    return TestClient(client.app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def real_records(client):
    if not DATA.exists():
        pytest.skip("data package not present")
    return client.get(REAL_RECORDS).json()


def payload_from(records):
    return {key: records[key] for key in ("observations", "supporting_records", "scenario_id", "snapshot_id", "receipt_cutoff")}


@pytest.fixture(scope="module")
def real_result(client, real_records):
    response = client.post(ASSOCIATE, json=payload_from(real_records))
    assert response.status_code == 200, response.text
    return response


def copied_root(tmp_path):
    """A root holding only copies of the artifacts, with no data package, for damaging safely."""
    root = tmp_path / "root"
    (root / "configs").mkdir(parents=True)
    (root / "models").mkdir()
    for name in ("baseline.json", "learned.json"):
        shutil.copy(ROOT / "configs" / name, root / "configs" / name)
    for name in (learned_module.MODEL_FILE, learned_module.METADATA_FILE):
        shutil.copy(ROOT / "models" / name, root / "models" / name)
    return root


def edit_json(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def rewrite_model_with_matching_hash(root, write):
    """Replace the artifact and update its recorded hash, so only the load itself can notice."""
    model_path = root / "models" / learned_module.MODEL_FILE
    write(model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    edit_json(root / "models" / learned_module.METADATA_FILE, lambda d: d.update(artifact_sha256=digest))


# Valid requests


def test_valid_request_gives_both_methods(client):
    response = client.post(ASSOCIATE, json={"observations": two_aircraft(), "supporting_records": [], "scenario_id": "s", "snapshot_id": "n", "receipt_cutoff": None})
    assert response.status_code == 200, response.text
    body = strict_json(response.text)
    assert set(body) == {"analysis", "baseline", "learned"}
    assert body["baseline"]["method"] == "baseline" and body["learned"]["method"] == "learned"
    assert body["baseline"]["links"] == EXPECTED_LINKS
    assert body["learned"]["links"] == EXPECTED_LINKS
    for result in (body["baseline"], body["learned"]):
        assert {"method", "version", "config", "policy", "links", "tracklets"} <= set(result)
        statuses = {t["tracklet_id"]: t["status"] for t in result["tracklets"]}
        assert statuses == {"A1": "accepted", "A2": "unresolved", "B1": "accepted", "B2": "unresolved"}
    analysis = body["analysis"]
    assert (analysis["scenario_id"], analysis["snapshot_id"], analysis["receipt_cutoff"]) == ("s", "n", None)
    assert (analysis["tracklet_count"], analysis["observation_count"], analysis["supporting_count"]) == (4, 24, 0)
    assert analysis["input_fingerprint"] == body["baseline"]["input_fingerprint"] == body["learned"]["input_fingerprint"]
    assert analysis["quality"]["observation_rows"] == 24
    assert analysis["quality"]["altitude_null"] == 0


def test_supporting_records_are_optional_and_used_when_given(client):
    rows = two_aircraft()
    without = client.post(ASSOCIATE, json={"observations": rows})
    assert without.status_code == 200, without.text
    assert without.json()["analysis"]["supporting_count"] == 0
    assert without.json()["baseline"]["links"] == EXPECTED_LINKS

    extra = [supporting(record_id="sup-a", observation_id="A1-0", received_at=stamp(5)), supporting(record_id="sup-x", observation_id="nobody", received_at=stamp(5))]
    given = client.post(ASSOCIATE, json={"observations": rows, "supporting_records": extra})
    assert given.status_code == 200, given.text
    analysis = given.json()["analysis"]
    assert analysis["supporting_count"] == 2
    assert analysis["quality"]["supporting_without_observation"] == 1
    assert analysis["input_fingerprint"] != without.json()["analysis"]["input_fingerprint"]


def test_missing_and_null_altitude_work_and_stay_distinct(client):
    def no_altitude(rows):
        return [{k: v for k, v in row.items() if k != "altitude_ft"} for row in rows]

    # A-fragments have null altitude, B-fragments have none at all.
    rows = flight("A1", 0, altitude=None) + flight("A2", 120, altitude=None) + no_altitude(flight("B1", 0, lon=-100.0)) + no_altitude(flight("B2", 120, lon=-100.0))
    response = client.post(ASSOCIATE, json={"observations": rows})
    assert response.status_code == 200, response.text
    body = strict_json(response.text)
    assert body["analysis"]["quality"]["altitude_null"] == 12
    assert body["analysis"]["quality"]["altitude_absent"] == 12
    for result in (body["baseline"], body["learned"]):
        assert result["links"] == EXPECTED_LINKS
        pair = next(t for t in result["tracklets"] if t["tracklet_id"] == "A1")["candidates"][0]
        assert pair["successor"] == "A2"
        assert pair["measurements"]["altitude_miss_ft"] is None  # never zero, never NaN


def test_identical_duplicates_and_reordering_change_nothing(client):
    rows = two_aircraft()
    plain = client.post(ASSOCIATE, json={"observations": rows}).json()
    shuffled = rows + rows[:5]
    random.Random(4).shuffle(shuffled)
    again = client.post(ASSOCIATE, json={"observations": shuffled}).json()
    assert again["analysis"]["input_fingerprint"] == plain["analysis"]["input_fingerprint"]
    assert again["baseline"] == plain["baseline"]
    assert again["learned"] == plain["learned"]
    assert again["analysis"]["quality"]["observation_rows"] == 29
    assert again["analysis"]["observation_count"] == 24


# Invalid JSON and request shape


@pytest.mark.parametrize("body", [
    b"{not json",
    b"",
    b'{"observations": [NaN]}',
    b'{"observations": [Infinity]}',
    b'{"observations": [-Infinity]}',
    b'{"observations": [{"latitude": NaN}]}',
    b'{"observations": [{"attributes": {"spd": Infinity}}]}',
    b'{"observations": [1e999]}',
    b'{"observations": [{"attributes": {"spd": -1e999}}]}',
    b'{"observations": [1' + b"0" * 400 + b"]}",  # finite as an integer, but beyond any double
    b'{"observations": [1' + b"0" * 5000 + b"]}",  # beyond Python's integer parsing limit
    b'{"observations": [], "observations": []}',
    b'{"observations": [{"\\ud800": 1, "\\ud800": 2}]}',
    b'{"observations": ["\\ud800"]}',
    b"\xff\xfe\xfa not utf-8",
    b'{"observations": ' + b"[" * 100_000 + b"]" * 100_000 + b"}",
], ids=lambda body: body[:32].decode("latin-1"))
def test_unusable_json_is_rejected(client, body):
    assert_error(client.post(ASSOCIATE, content=body, headers={"content-type": "application/json"}), 400, "invalid_json")


def test_record_nested_too_deep_for_processing_is_rejected_not_a_server_error(lenient_client):
    # Shallow enough for the parser, deep enough for the Python fingerprint code.
    nested = "[" * 600 + "]" * 600
    row = json.dumps(observation()).replace('"attributes": {"spd": 290.0, "q": 1}', f'"attributes": {{"deep": {nested}}}')
    response = lenient_client.post(ASSOCIATE, content=f'{{"observations": [{row}]}}'.encode(), headers={"content-type": "application/json"})
    assert_error(response, 422, "invalid_request")


@pytest.mark.parametrize("payload, field", [
    ({}, "observations"),
    ({"supporting_records": []}, "observations"),
    ({"observations": "nope"}, "observations"),
    ({"observations": {"a": 1}}, "observations"),
    ({"observations": None}, "observations"),
    ({"observations": [{}], "supporting_records": None}, "supporting_records"),
    ({"observations": [{}], "scenario_id": 5}, "scenario_id"),
    ({"observations": [{}], "receipt_cutoff": 5}, "receipt_cutoff"),
])
def test_wrong_request_shape(client, payload, field):
    error = assert_error(client.post(ASSOCIATE, json=payload), 422, "invalid_request")
    assert [issue["field"] for issue in error["issues"]] == [field]


@pytest.mark.parametrize("body", [b"[]", b"5", b"null", b'"text"'])
def test_request_must_be_an_object(client, body):
    assert_error(client.post(ASSOCIATE, content=body), 422, "invalid_request")


def test_labels_field_is_rejected_and_not_echoed(client):
    secret = {"tracklet_id": "A1", "next_tracklet_id": "secret-value"}
    response = client.post(ASSOCIATE, json={"observations": two_aircraft(), "labels": [secret]})
    error = assert_error(response, 422, "invalid_request")
    assert [(issue["code"], issue["field"]) for issue in error["issues"]] == [("unknown_field", "labels")]
    assert "secret-value" not in response.text


def test_empty_input(client):
    assert_error(client.post(ASSOCIATE, json={"observations": []}), 422, "empty_input")
    only_supporting = {"observations": [], "supporting_records": [supporting()]}
    assert_error(client.post(ASSOCIATE, json=only_supporting), 422, "empty_input")


# Record validation, by the existing validators


@pytest.mark.parametrize("change, field, issue_code", [
    ({"observation_id": ""}, "observation_id", "invalid_field"),
    ({"tracklet_id": "   "}, "tracklet_id", "invalid_field"),
    ({"tracklet_id": None}, "tracklet_id", "invalid_field"),
    ({"observed_at": "2026-13-45T00:00:00Z"}, "observed_at", "invalid_field"),
    ({"observed_at": "2026-01-10T10:00:00"}, "observed_at", "invalid_field"),
    ({"received_at": 1767996000}, "received_at", "invalid_field"),
    ({"latitude": 91.0}, "latitude", "invalid_field"),
    ({"longitude": -180.5}, "longitude", "invalid_field"),
    ({"latitude": None}, "latitude", "invalid_field"),
    ({"longitude": "east"}, "longitude", "invalid_field"),
    ({"altitude_ft": "high"}, "altitude_ft", "invalid_field"),
])
def test_invalid_observation_field(client, change, field, issue_code):
    rows = two_aircraft()
    rows[3] = {**rows[3], **change}
    error = assert_error(client.post(ASSOCIATE, json={"observations": rows}), 422, "validation_failed")
    assert [(i["code"], i["field"], i["index"], i["record_kind"]) for i in error["issues"]] == [(issue_code, field, 3, "observation")]


@pytest.mark.parametrize("field", ["observation_id", "tracklet_id", "observed_at", "received_at", "latitude", "longitude", "source"])
def test_missing_required_observation_field(client, field):
    rows = flight("A1", 0)
    del rows[0][field]
    error = assert_error(client.post(ASSOCIATE, json={"observations": rows}), 422, "validation_failed")
    assert [(i["code"], i["field"]) for i in error["issues"]] == [("missing_field", field)]


def test_non_object_row_and_bad_supporting_record(client):
    error = assert_error(client.post(ASSOCIATE, json={"observations": [5]}), 422, "validation_failed")
    assert error["issues"][0]["record_kind"] == "observation" and error["issues"][0]["index"] == 0

    bad = {"observations": flight("A1", 0), "supporting_records": [supporting(record_id="")]}
    error = assert_error(client.post(ASSOCIATE, json=bad), 422, "validation_failed")
    assert [(i["record_kind"], i["field"]) for i in error["issues"]] == [("supporting", "record_id")]


def test_every_problem_is_reported_together(client):
    rows = flight("A1", 0)
    rows[0]["latitude"] = 500
    rows[2]["observed_at"] = "yesterday"
    error = assert_error(client.post(ASSOCIATE, json={"observations": rows}), 422, "validation_failed")
    assert sorted(i["field"] for i in error["issues"]) == ["latitude", "observed_at"]


def test_conflicting_ids_name_the_id(client):
    rows = two_aircraft()
    rows.append({**rows[0], "latitude": rows[0]["latitude"] + 0.5})
    error = assert_error(client.post(ASSOCIATE, json={"observations": rows}), 422, "validation_failed")
    assert [(i["code"], i["record_id"]) for i in error["issues"]] == [("conflicting_id", "A1-0")]
    assert "A1-0" in error["issues"][0]["message"] and "A1-0" in error["message"]


def test_receipt_cutoff_is_applied(client):
    rows = flight("A1", 0)
    error = assert_error(client.post(ASSOCIATE, json={"observations": rows, "receipt_cutoff": "never"}), 422, "validation_failed")
    assert [i["field"] for i in error["issues"]] == ["receipt_cutoff"]
    # The last report arrives at 51 s, so a 20 s cutoff catches the reports received at 21, 31, 41 and 51 s.
    error = assert_error(client.post(ASSOCIATE, json={"observations": rows, "receipt_cutoff": stamp(20)}), 422, "validation_failed")
    assert sorted(i["record_id"] for i in error["issues"]) == ["A1-2", "A1-3", "A1-4", "A1-5"]
    assert {i["code"] for i in error["issues"]} == {"after_cutoff"}


# The error shape everywhere


def test_unknown_route_and_wrong_method_use_the_error_shape(client):
    assert_error(client.get("/nowhere"), 404, "not_found")
    assert_error(client.get("/health/"), 404, "not_found")  # no bare redirect
    assert_error(client.get("/scenarios/", follow_redirects=False), 404, "not_found")
    response = client.get(ASSOCIATE)
    assert_error(response, 405, "method_not_allowed")
    assert response.headers["allow"] == "POST"
    assert_error(client.post("/health"), 405, "method_not_allowed")
    assert_error(client.delete("/scenarios"), 405, "method_not_allowed")


def test_internal_error_is_generic(lenient_client, monkeypatch, caplog):
    def explode(*args, **kwargs):
        raise RuntimeError(rf"cannot read {ROOT}\models\secret.joblib")

    monkeypatch.setattr(baseline_module, "analyze", explode)
    response = lenient_client.post(ASSOCIATE, json={"observations": two_aircraft()})
    error = assert_error(response, 500, "internal_error")
    assert error["message"] == GENERIC_500 and error["issues"] == []
    assert_no_leak(response.text)
    assert "secret.joblib" not in response.text and "RuntimeError" not in response.text
    assert "secret.joblib" in caplog.text  # kept for the server's log


def test_non_finite_result_is_an_error_not_invalid_json(lenient_client, monkeypatch):
    monkeypatch.setattr(learned_module, "analyze", lambda *args: {"links": {}, "score": float("inf")})
    response = lenient_client.post(ASSOCIATE, json={"observations": two_aircraft()})
    error = assert_error(response, 500, "internal_error")
    assert error["message"] == GENERIC_500


@pytest.mark.parametrize("altitude", [1e200, 1e308])
def test_extreme_finite_altitude_never_yields_non_compliant_json(lenient_client, altitude):
    """The validators accept any finite altitude. Whatever follows, the response must parse as strict JSON."""
    rows = flight("A1", 0, altitude=altitude) + flight("A2", 120, altitude=-altitude)
    response = lenient_client.post(ASSOCIATE, json={"observations": rows})
    strict_json(response.text)
    if response.status_code != 200:
        assert_error(response, response.status_code, "validation_failed" if response.status_code == 422 else "internal_error")


# Health and loading


def test_health_ready_on_the_real_artifacts(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = strict_json(response.text)
    assert set(body) == {"status", "ready", "baseline", "learned", "problems"}
    assert body["status"] == "ready" and body["ready"] is True and body["problems"] == []
    assert body["baseline"]["available"] is True and body["baseline"]["version"].startswith("baseline-")
    assert body["learned"]["available"] is True and body["learned"]["version"] == json.loads((ROOT / "models" / "learned_model.json").read_text(encoding="utf-8"))["model_version"]
    assert body["learned"]["feature_names"] == list(learned_module.FEATURES)


def garbage_with_matching_hash(root):
    rewrite_model_with_matching_hash(root, lambda path: path.write_bytes(b"not a pickle"))


def wrong_object_with_matching_hash(root):
    rewrite_model_with_matching_hash(root, lambda path: joblib.dump({"not": "a model"}, path))


LEARNED_DAMAGE = {
    "model_file_missing": lambda root: (root / "models" / learned_module.MODEL_FILE).unlink(),
    "metadata_missing": lambda root: (root / "models" / learned_module.METADATA_FILE).unlink(),
    "model_file_changed": lambda root: (root / "models" / learned_module.MODEL_FILE).write_bytes(b"corrupted"),
    "metadata_corrupted": lambda root: (root / "models" / learned_module.METADATA_FILE).write_text("{oops", encoding="utf-8"),
    "garbage_with_matching_hash": garbage_with_matching_hash,
    "wrong_object_with_matching_hash": wrong_object_with_matching_hash,
    "feature_schema_changed": lambda root: edit_json(root / "models" / learned_module.METADATA_FILE, lambda d: d.update(feature_names=["gap_s"])),
    "policy_for_another_model": lambda root: edit_json(root / "configs" / "learned.json", lambda d: d.update(model_version="learned-other")),
    "policy_missing": lambda root: (root / "configs" / "learned.json").unlink(),
    "policy_corrupted": lambda root: (root / "configs" / "learned.json").write_text("[]", encoding="utf-8"),
    "other_scikit_learn": lambda root: edit_json(root / "models" / learned_module.METADATA_FILE, lambda d: d["library_versions"].update({"scikit-learn": "0.0.1"})),
    "no_recorded_scikit_learn": lambda root: edit_json(root / "models" / learned_module.METADATA_FILE, lambda d: d["library_versions"].pop("scikit-learn")),
}


def assert_not_ready(client, tmp_path):
    response = client.get("/health")
    assert response.status_code == 503
    body = strict_json(response.text)
    assert body["status"] == "unavailable" and body["ready"] is False
    assert body["problems"] and all(isinstance(problem, str) for problem in body["problems"])
    assert body["error"]["code"] == "model_unavailable" and body["error"]["issues"] == []
    assert_no_leak(response.text, tmp_path)

    inference = client.post(ASSOCIATE, json={"observations": two_aircraft()})
    assert_error(inference, 503, "model_unavailable")
    assert_no_leak(inference.text, tmp_path)
    return body


@pytest.mark.parametrize("damage", sorted(LEARNED_DAMAGE))
def test_learned_failure_is_reported_and_startup_survives(tmp_path, damage):
    root = copied_root(tmp_path)
    LEARNED_DAMAGE[damage](root)
    with TestClient(create_app(root)) as client:
        body = assert_not_ready(client, tmp_path)
        assert body["learned"] == {"available": False, "version": None, "feature_names": None}
        assert body["baseline"]["available"] is True and body["baseline"]["version"].startswith("baseline-")
        assert client.get("/scenarios").json() == {"scenarios": []}


def test_other_scikit_learn_is_named_in_the_problem(tmp_path):
    root = copied_root(tmp_path)
    LEARNED_DAMAGE["other_scikit_learn"](root)
    with TestClient(create_app(root)) as client:
        problems = client.get("/health").json()["problems"]
    assert len(problems) == 1 and "0.0.1" in problems[0] and sklearn.__version__ in problems[0]


def test_other_scikit_learn_is_refused_without_unpickling(tmp_path, monkeypatch):
    root = copied_root(tmp_path)
    LEARNED_DAMAGE["other_scikit_learn"](root)

    def refuse(*args, **kwargs):
        raise AssertionError("an incompatible artifact was unpickled")

    monkeypatch.setattr(joblib, "load", refuse)
    with TestClient(create_app(root)) as client:
        assert "scikit-learn" in client.get("/health").json()["problems"][0]


@pytest.mark.parametrize("damage", [lambda p: p.unlink(), lambda p: p.write_text("{oops", encoding="utf-8"), lambda p: p.write_text("{}", encoding="utf-8")], ids=["missing", "corrupted", "incomplete"])
def test_baseline_failure_is_reported_and_startup_survives(tmp_path, damage):
    root = copied_root(tmp_path)
    damage(root / "configs" / "baseline.json")
    with TestClient(create_app(root)) as client:
        body = assert_not_ready(client, tmp_path)
        assert body["baseline"] == {"available": False, "version": None}
        assert body["learned"]["available"] is True


def test_empty_root_reports_everything_missing(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        body = assert_not_ready(client, tmp_path)
    assert body["baseline"]["available"] is False and body["learned"]["available"] is False
    assert body["problems"] == ["baseline configuration is missing", "learned model files are missing"]


def test_before_startup_nothing_is_claimed_ready(tmp_path):
    client = TestClient(create_app(tmp_path))  # no `with`, so the lifespan hasn't run
    assert_not_ready(client, tmp_path)


def test_validation_of_shape_still_works_while_unavailable(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert_error(client.post(ASSOCIATE, content=b"{bad"), 400, "invalid_json")
        assert_error(client.post(ASSOCIATE, json={"labels": []}), 422, "invalid_request")


def test_govia_root_variable_is_used_and_an_argument_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("GOVIA_ROOT", str(tmp_path))
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 503
    with TestClient(create_app(ROOT)) as client:
        assert client.get("/health").status_code == 200


def test_default_root_is_the_repository(monkeypatch):
    monkeypatch.delenv("GOVIA_ROOT", raising=False)
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200


def test_openapi_document_is_served(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "labels" not in response.text


# Scenarios (a synthetic root keeps the odd cases fast and independent of the package)


def write_scenario(root, partition, name, observations=("{}",), supporting_rows=(), **manifest):
    folder = root / "candidate-packet" / "data" / partition / name
    folder.mkdir(parents=True)
    (folder / "observations.jsonl").write_text("\n".join(observations) + "\n", encoding="utf-8")
    (folder / "supporting.jsonl").write_text("\n".join(supporting_rows) + ("\n" if supporting_rows else ""), encoding="utf-8")
    (folder / "snapshot.json").write_text(json.dumps({
        "scenario_id": f"id-{name}", "snapshot_id": f"snap-{name}", "receipt_cutoff": None, "tracklet_count": 1, "unique_observation_count": 1,
        "observation_row_count": len(observations), "supporting_row_count": len(supporting_rows), "labeled_link_count": 7,
        "partition": partition, "files": {"labels": "labels.jsonl"}, **manifest,
    }), encoding="utf-8")
    return folder


def test_discovery_refuses_other_partitions_and_ignores_bad_folders(tmp_path):
    write_scenario(tmp_path, "validation", "scenario_001")
    write_scenario(tmp_path, "train", "scenario_002")
    write_scenario(tmp_path, "staged_demo", "snapshot_1")
    write_scenario(tmp_path, "staged_demo", "snapshot_3")  # not one of the staged folders
    write_scenario(tmp_path, "test", "scenario_009")
    write_scenario(tmp_path, "other", "scenario_010")
    write_scenario(tmp_path, "validation", "not_a_scenario")
    (write_scenario(tmp_path, "validation", "scenario_broken") / "snapshot.json").write_text("{oops", encoding="utf-8")
    (write_scenario(tmp_path, "validation", "scenario_partial") / "snapshot.json").write_text('{"scenario_id": "x"}', encoding="utf-8")
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/scenarios")
        assert [s["key"] for s in response.json()["scenarios"]] == ["train/scenario_002", "validation/scenario_001", "staged_demo/snapshot_1"]
        assert "labeled_link_count" not in response.text and "labels" not in response.text
        assert_no_leak(response.text, tmp_path)
        for path in ("test/scenario_009", "other/scenario_010", "staged_demo/snapshot_3", "validation/not_a_scenario", "validation/scenario_broken"):
            assert_error(client.get(f"/scenarios/{path}/records"), 404, "not_found")


def test_damaged_data_file_is_a_generic_server_error(tmp_path):
    write_scenario(tmp_path, "validation", "scenario_001", observations=("{broken",))
    write_scenario(tmp_path, "validation", "scenario_002")
    (tmp_path / "candidate-packet" / "data" / "validation" / "scenario_002" / "supporting.jsonl").unlink()
    with TestClient(create_app(tmp_path), raise_server_exceptions=False) as client:
        for name in ("scenario_001", "scenario_002"):
            response = client.get(f"/scenarios/validation/{name}/records")
            assert assert_error(response, 500, "internal_error")["message"] == GENERIC_500
            assert_no_leak(response.text, tmp_path)


@pytest.mark.data
def test_scenarios_list_exactly_the_allowed_partitions(client, packet):
    response = client.get("/scenarios")
    assert response.status_code == 200
    assert "labeled_link_count" not in response.text
    scenarios = response.json()["scenarios"]
    partitions = [s["partition"] for s in scenarios]
    assert {p: partitions.count(p) for p in set(partitions)} == {"train": 80, "validation": 20, "staged_demo": 2}
    keys = {s["key"] for s in scenarios}
    assert {"validation/scenario_001", "staged_demo/snapshot_1", "staged_demo/snapshot_2"} <= keys
    assert not [key for key in keys if key.startswith("test")]
    fields = {"key", "partition", "name", "scenario_id", "snapshot_id", "receipt_cutoff", "tracklet_count", "unique_observation_count", "observation_row_count", "supporting_row_count"}
    assert all(set(s) == fields for s in scenarios)
    assert_no_leak(response.text)


@pytest.mark.parametrize("path", [
    "/scenarios/validation/scenario_999/records",
    "/scenarios/nope/scenario_001/records",
    "/scenarios/test/scenario_001/records",
    "/scenarios/TEST/scenario_001/records",
    "/scenarios/Validation/scenario_001/records",  # Windows would find these on disk; an exact match doesn't
    "/scenarios/validation/SCENARIO_001/records",
    "/scenarios/validation/scenario_001./records",
    "/scenarios/validation/../test/scenario_001/records",
    "/scenarios/validation/..%2F..%2Ftest%2Fscenario_001/records",
    "/scenarios/..%2Ftest/scenario_001/records",
    "/scenarios/..%5Ctest/scenario_001/records",
    "/scenarios/validation/%2E%2E/records",
    "/scenarios/validation/scenario_001%00/records",
    "/scenarios/validation/C:%5CWindows/records",
])
def test_unknown_and_traversal_style_names_are_404(client, path):
    response = client.get(path)
    assert_error(response, 404, "not_found")
    assert "observed_at" not in response.text


@pytest.mark.data
@pytest.mark.parametrize("partition, name", [("validation", "scenario_001"), ("train", "scenario_001"), ("staged_demo", "snapshot_1"), ("staged_demo", "snapshot_2")])
def test_records_match_the_files(client, packet, partition, name):
    folder = DATA / partition / name
    response = client.get(f"/scenarios/{partition}/{name}/records")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"key", "scenario_id", "snapshot_id", "receipt_cutoff", "observations", "supporting_records"}
    assert body["key"] == f"{partition}/{name}"
    summary = next(s for s in client.get("/scenarios").json()["scenarios"] if s["key"] == body["key"])
    observation_lines = (folder / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    supporting_lines = (folder / "supporting.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(body["observations"]) == len(observation_lines) == summary["observation_row_count"]
    assert len(body["supporting_records"]) == len(supporting_lines) == summary["supporting_row_count"]
    assert body["observations"] == [json.loads(line) for line in observation_lines]  # raw rows, in file order
    assert body["supporting_records"] == [json.loads(line) for line in supporting_lines]
    assert (body["scenario_id"], body["snapshot_id"], body["receipt_cutoff"]) == (summary["scenario_id"], summary["snapshot_id"], summary["receipt_cutoff"])
    assert "labeled_link_count" not in response.text and "next_tracklet" not in response.text


@pytest.mark.data
@pytest.mark.parametrize("snapshot", ["snapshot_1", "snapshot_2"])
def test_staged_snapshot_runs_with_its_cutoff(client, packet, snapshot):
    records = client.get(f"/scenarios/staged_demo/{snapshot}/records").json()
    response = client.post(ASSOCIATE, json=payload_from(records))
    assert response.status_code == 200, response.text
    assert response.json()["analysis"]["receipt_cutoff"] == records["receipt_cutoff"]


# A real scenario, checked from its raw observations


def raw_tracklet_spans(observations):
    def when(text):
        return datetime.fromisoformat(text.replace("Z", "+00:00"))

    spans = {}
    for row in observations:
        start, end = spans.get(row["tracklet_id"], (None, None))
        moment = when(row["observed_at"])
        spans[row["tracklet_id"]] = (min(moment, start or moment), max(moment, end or moment))
    return spans


@pytest.mark.data
def test_real_scenario_invariants(real_records, real_result):
    spans = raw_tracklet_spans(real_records["observations"])
    body = real_result.json()
    assert body["analysis"]["tracklet_count"] == len(spans)
    assert body["analysis"]["observation_count"] == len({row["observation_id"] for row in real_records["observations"]})
    for method in ("baseline", "learned"):
        result = body[method]
        links = result["links"]
        assert links, f"{method} linked nothing"
        assert len(set(links.values())) == len(links)  # one predecessor each; the dict already gives one successor each
        for a, b in links.items():
            assert a != b
            assert a in spans and b in spans
            assert spans[b][0] > spans[a][1], f"{method}: {a} -> {b} doesn't move forward"
        assert {t["tracklet_id"] for t in result["tracklets"]} == set(spans)
        for tracklet in result["tracklets"]:
            assert tracklet["accepted_successor"] == links.get(tracklet["tracklet_id"])
            assert tracklet["status"] == ("accepted" if tracklet["tracklet_id"] in links else "unresolved")
        assert result["scenario_id"] == real_records["scenario_id"] and result["snapshot_id"] == real_records["snapshot_id"]


def finite_everywhere(value):
    if isinstance(value, dict):
        return all(finite_everywhere(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_everywhere(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


@pytest.mark.data
def test_real_result_is_strict_json_and_round_trips(real_result):
    parsed = strict_json(real_result.text)
    assert finite_everywhere(parsed)
    assert json.loads(json.dumps(parsed, allow_nan=False)) == parsed
    assert real_result.headers["content-type"] == "application/json"


@pytest.mark.data
def test_real_records_round_trip_gives_the_same_result(client, real_records, real_result):
    shuffled = dict(real_records)
    shuffled["observations"] = random.Random(1).sample(real_records["observations"], len(real_records["observations"]))
    again = client.post(ASSOCIATE, json=payload_from(shuffled)).json()
    original = real_result.json()
    assert again["analysis"]["input_fingerprint"] == original["analysis"]["input_fingerprint"]
    assert again["baseline"] == original["baseline"]
    assert again["learned"] == original["learned"]


@pytest.mark.data
def test_restart_gives_identical_decisions(real_records, real_result):
    with TestClient(create_app(ROOT)) as second:
        again = second.post(ASSOCIATE, json=payload_from(real_records))
    assert again.status_code == 200
    assert again.json() == real_result.json()


def test_synthetic_restart_gives_identical_decisions(client):
    first = client.post(ASSOCIATE, json={"observations": two_aircraft()}).json()
    with TestClient(create_app(ROOT)) as second:
        assert second.post(ASSOCIATE, json={"observations": two_aircraft()}).json() == first


# What the API path must not do


def test_inference_never_trains(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("training was invoked")

    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", refuse)
    with TestClient(create_app(ROOT)) as client:
        assert client.get("/health").status_code == 200
        response = client.post(ASSOCIATE, json={"observations": two_aircraft()})
    assert response.status_code == 200
    assert response.json()["learned"]["links"] == EXPECTED_LINKS


_FRESH_INTERPRETER = """
import json, os, sys
touched = []
def watch(event, args):
    if event in ("open", "os.scandir", "os.listdir") and args and isinstance(args[0], (str, bytes, os.PathLike)):
        touched.append(os.fsdecode(args[0]).replace(chr(92), "/"))
sys.addaudithook(watch)
from fastapi.testclient import TestClient
from govia_tracks.api import app
loaded_on_import = [m for m in ("govia_tracks.labels", "govia_tracks.evaluation", "govia_tracks.training") if m in sys.modules]
with TestClient(app) as client:
    assert client.get("/health").status_code == 200
    listed = client.get("/scenarios").json()["scenarios"]
    results = 0
    for scenario in [s for s in listed if s["key"] in ("validation/scenario_001", "staged_demo/snapshot_2", "train/scenario_001")]:
        records = client.get("/scenarios/" + scenario["key"] + "/records").json()
        records.pop("key")
        results += client.post("/tracklets/associate", json=records).status_code == 200
print(json.dumps({
    "on_import": loaded_on_import,
    "after_use": [m for m in ("govia_tracks.labels", "govia_tracks.evaluation", "govia_tracks.training") if m in sys.modules],
    "labels_touched": [p for p in touched if "labels" in p],
    "test_partition_touched": [p for p in touched if "/data/test" in p],
    "records_seen": any(p.endswith("validation/scenario_001/observations.jsonl") for p in touched),  # the watch itself works
    "results": results,
    "listed": len(listed),
}))
"""


@pytest.mark.data
def test_fresh_interpreter_never_loads_or_opens_labels(packet):
    environment = {**os.environ, "GOVIA_ROOT": str(ROOT)}
    done = subprocess.run([sys.executable, "-c", _FRESH_INTERPRETER], capture_output=True, text=True, cwd=ROOT, env=environment, timeout=120)
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report == {"on_import": [], "after_use": [], "labels_touched": [], "test_partition_touched": [], "records_seen": True, "results": 3, "listed": 102}


def test_scenario_and_snapshot_ids_must_not_be_blank_in_unit_scope():
    from govia_tracks.api import AssociateRequest
    with pytest.raises(ValueError, match="blank"):
        AssociateRequest(observations=[], scenario_id="  ")
    assert AssociateRequest(observations=[], scenario_id=None, snapshot_id="s").snapshot_id == "s"


@pytest.mark.data
@pytest.mark.parametrize("partition,name", [("validation", "scenario_001"), ("staged_demo", "snapshot_1"), ("staged_demo", "snapshot_2")])
def test_labels_appear_in_no_response_for_real_scenarios(client, packet, partition, name):
    records = client.get(f"/scenarios/{partition}/{name}/records")
    body = {k: records.json()[k] for k in ("observations", "supporting_records", "scenario_id", "snapshot_id", "receipt_cutoff")}
    associated = client.post("/tracklets/associate", json=body)
    assert records.status_code == 200 and associated.status_code == 200
    for text in (records.text, associated.text):
        assert "successor_tracklet_id" not in text and "labeled_link" not in text and "labels.jsonl" not in text
