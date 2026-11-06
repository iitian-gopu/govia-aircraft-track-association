"""Record validation. Original parsed records are kept untouched on `raw`."""
import calendar
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .errors import Issue

_TIMESTAMP = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?Z", re.ASCII)
_MISSING = object()
_MAX_ALTITUDE_FT = 1e12  # not a claim about aircraft: surprising values are kept, but squaring anything near 1e160 overflows


def parse_utc_timestamp(value: Any) -> int:
    """Return integer microseconds since the epoch. Only the documented `...Z` form is accepted."""
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    match = _TIMESTAMP.fullmatch(value)
    if match is None:
        raise ValueError("timestamp must be ISO 8601 UTC ending in Z")
    parsed = datetime.strptime(match[1], "%Y-%m-%dT%H:%M:%S")  # also rejects impossible dates
    fraction = (match[2] or "").ljust(6, "0")[:6]
    return calendar.timegm(parsed.timetuple()) * 1_000_000 + int(fraction)


def json_equal(a: Any, b: Any) -> bool:
    """Compare parsed JSON. Key order is ignored; missing differs from null; array order matters.

    Python's `==` treats True as 1, which JSON does not, so booleans are compared strictly.
    JSON has one number type, so 1 and 1.0 are equal.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return type(a) is type(b) and a == b


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class Observation:
    observation_id: str
    tracklet_id: str
    observed_us: int
    received_us: int
    latitude: float
    longitude: float
    altitude_ft: float | None  # None for both missing and null; `raw` keeps the distinction
    source: str
    attributes: dict | None
    raw: dict


@dataclass(frozen=True)
class SupportingRecord:
    record_id: str
    observation_id: str
    received_us: int
    source: str
    attributes: dict
    raw: dict


class _Checker:
    def __init__(self, kind, index, raw, id_field, issues):
        self.kind, self.index, self.raw, self.issues = kind, index, raw, issues
        value = raw.get(id_field)
        self.record_id = value if isinstance(value, str) and value else None
        self.ok = True

    def fail(self, code, field, message):
        self.ok = False
        label = f"{self.kind} row {self.index}"
        if self.record_id:
            label += f" ({self.record_id})"
        self.issues.append(Issue(code, self.kind, f"{label}: {message}", self.index, self.record_id, field))

    def get(self, field):
        value = self.raw.get(field, _MISSING)
        if value is _MISSING:
            self.fail("missing_field", field, f"required field '{field}' is missing")
            return None
        return value

    def text(self, field):
        value = self.get(field)
        if value is None and field not in self.raw:
            return None
        if not isinstance(value, str) or not value.strip():
            self.fail("invalid_field", field, f"'{field}' must be a nonempty string")
            return None
        return value

    def timestamp(self, field):
        value = self.get(field)
        if value is None and field not in self.raw:
            return None
        try:
            return parse_utc_timestamp(value)
        except ValueError as exc:
            self.fail("invalid_field", field, f"'{field}' is invalid ({value!r}): {exc}")
            return None

    def coordinate(self, field, limit):
        value = self.get(field)
        if value is None and field not in self.raw:
            return None
        if not _is_number(value):
            self.fail("invalid_field", field, f"'{field}' must be a finite number, got {value!r}")
            return None
        if not -limit <= value <= limit:
            self.fail("invalid_field", field, f"'{field}' {value} is outside [-{limit}, {limit}]")
            return None
        return float(value)


def parse_observation(raw: Any, index: int, issues: list[Issue]) -> Observation | None:
    if not isinstance(raw, dict):
        issues.append(Issue("invalid_field", "observation", f"observation row {index}: not a JSON object", index))
        return None
    check = _Checker("observation", index, raw, "observation_id", issues)
    observation_id = check.text("observation_id")
    tracklet_id = check.text("tracklet_id")
    observed = check.timestamp("observed_at")
    received = check.timestamp("received_at")
    latitude = check.coordinate("latitude", 90)
    longitude = check.coordinate("longitude", 180)
    source = check.text("source")

    altitude = raw.get("altitude_ft")
    if altitude is not None and not _is_number(altitude):
        check.fail("invalid_field", "altitude_ft", f"'altitude_ft' must be a number, null or absent, got {altitude!r}")
    elif altitude is not None and abs(altitude) > _MAX_ALTITUDE_FT:
        check.fail("invalid_field", "altitude_ft", f"'altitude_ft' {altitude} is too large to calculate with")
    attributes = raw.get("attributes")
    if attributes is not None and not isinstance(attributes, dict):
        check.fail("invalid_field", "attributes", "'attributes' must be an object, null or absent")

    if not check.ok:
        return None
    return Observation(
        observation_id, tracklet_id, observed, received, latitude, longitude,
        None if altitude is None else float(altitude), source, attributes, raw,
    )


def parse_supporting(raw: Any, index: int, issues: list[Issue]) -> SupportingRecord | None:
    if not isinstance(raw, dict):
        issues.append(Issue("invalid_field", "supporting", f"supporting row {index}: not a JSON object", index))
        return None
    check = _Checker("supporting", index, raw, "record_id", issues)
    record_id = check.text("record_id")
    observation_id = check.text("observation_id")
    source = check.text("source")
    received = check.timestamp("received_at")
    attributes = check.get("attributes")
    if "attributes" in raw and not isinstance(attributes, dict):
        check.fail("invalid_field", "attributes", "'attributes' must be an object")
    if not check.ok:
        return None
    return SupportingRecord(record_id, observation_id, received, source, attributes, raw)
