"""Record validation. Expected outcomes are written out by hand, not computed by the code under test."""
import copy

import pytest

from govia_tracks.errors import InputValidationError
from govia_tracks.preprocess import prepare_scenario
from govia_tracks.schema import json_equal, parse_utc_timestamp

from conftest import observation, supporting


def codes(rows, support=()):
    with pytest.raises(InputValidationError) as caught:
        prepare_scenario(rows, support)
    return [(i.code, i.field) for i in caught.value.issues]


@pytest.mark.parametrize("field", ["observation_id", "tracklet_id", "observed_at", "received_at", "latitude", "longitude", "source"])
def test_each_required_field_is_enforced(field):
    row = observation()
    del row[field]
    assert codes([row]) == [("missing_field", field)]


def test_altitude_and_attributes_are_optional():
    row = observation()
    del row["altitude_ft"], row["attributes"]
    assert prepare_scenario([row]).tracklets[0].observations[0].altitude_ft is None


@pytest.mark.parametrize("field", ["observation_id", "tracklet_id", "source"])
@pytest.mark.parametrize("value", ["", "   ", 7, None])
def test_ids_must_be_nonempty_strings(field, value):
    assert codes([observation(**{field: value})]) == [("invalid_field", field)]


@pytest.mark.parametrize("value", [90.0001, -90.0001, 95, "39.6", None, True, float("nan"), float("inf")])
def test_invalid_latitude(value):
    assert codes([observation(latitude=value)]) == [("invalid_field", "latitude")]


@pytest.mark.parametrize("value", [180.0001, -180.5, "x"])
def test_invalid_longitude(value):
    assert codes([observation(longitude=value)]) == [("invalid_field", "longitude")]


@pytest.mark.parametrize("lat,lon", [(90, 180), (-90, -180), (0, 0)])
def test_coordinate_boundaries_are_valid(lat, lon):
    assert prepare_scenario([observation(latitude=lat, longitude=lon)]).quality.unique_observations == 1


@pytest.mark.parametrize("value", [
    "2026-01-10T10:00:00",  # no zone
    "2026-01-10T10:00:00+00:00",  # UTC, but not the documented Z form
    "2026-01-10T10:00:00+05:30",
    "2026-02-30T10:00:00Z",  # impossible date
    "2026-01-10 10:00:00Z",
    "not-a-time",
    1736503200,
    None,
])
def test_invalid_timestamps(value):
    assert codes([observation(observed_at=value)]) == [("invalid_field", "observed_at")]


def test_timestamp_value_against_a_known_epoch():
    # 2021-01-01T00:00:00Z is 1609459200 seconds since the epoch.
    assert parse_utc_timestamp("2021-01-01T00:00:00Z") == 1_609_459_200_000_000
    assert parse_utc_timestamp("2021-01-01T00:00:00.5Z") == 1_609_459_200_500_000
    assert parse_utc_timestamp("2021-01-01T00:00:00.123456789Z") == 1_609_459_200_123_456


@pytest.mark.parametrize("value", [1e160, -1e160, 1e13])
def test_altitude_too_large_to_calculate_with_is_rejected(value):
    assert codes([observation(altitude_ft=value)]) == [("invalid_field", "altitude_ft")]


def test_surprising_altitude_is_kept_as_a_data_quality_matter():
    assert prepare_scenario([observation(altitude_ft=250000.0)]).quality.unique_observations == 1


@pytest.mark.parametrize("value", ["18000", True, [1], float("nan")])
def test_invalid_altitude_is_rejected_not_dropped(value):
    assert codes([observation(altitude_ft=value)]) == [("invalid_field", "altitude_ft")]


def test_all_problems_are_reported_together():
    rows = [observation(observation_id="a", latitude=95), observation(observation_id="b", observed_at="nope")]
    assert codes(rows) == [("invalid_field", "latitude"), ("invalid_field", "observed_at")]


def test_error_message_names_the_record_and_problem():
    with pytest.raises(InputValidationError, match=r"obs-1.*latitude.*95"):
        prepare_scenario([observation(latitude=95)])


def test_non_object_rows_are_rejected():
    assert codes(["not an object"]) == [("invalid_field", None)]


@pytest.mark.parametrize("field", ["record_id", "observation_id", "source", "received_at", "attributes"])
def test_supporting_required_fields(field):
    row = supporting()
    del row[field]
    assert codes([observation()], [row]) == [("missing_field", field)]


@pytest.mark.parametrize("a,b,equal", [
    ({"x": 1, "y": 2}, {"y": 2, "x": 1}, True),  # key order is irrelevant
    ({"x": None}, {}, False),  # null is not missing
    ([1, 2], [2, 1], False),  # array order matters
    ({"x": {"p": 1, "q": 2}}, {"x": {"q": 2, "p": 1}}, True),  # nested key order
    ({"x": True}, {"x": 1}, False),  # Python's == would say True here
    ({"x": False}, {"x": 0}, False),
    ({"x": 1}, {"x": 1.0}, True),  # JSON has one number type
    ({"x": "1"}, {"x": 1}, False),
])
def test_json_equal(a, b, equal):
    assert json_equal(a, b) is equal


def test_validation_does_not_mutate_input():
    rows, support = [observation()], [supporting()]
    before = copy.deepcopy((rows, support))
    prepare_scenario(rows, support)
    assert (rows, support) == before


def test_timestamps_in_other_scripts_digits_are_rejected():
    assert codes([observation(observed_at="\u0662\u0660\u0662\u0666-01-10T10:00:00Z")]) == [("invalid_field", "observed_at")]
