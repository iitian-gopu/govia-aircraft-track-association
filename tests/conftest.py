import copy
from pathlib import Path

import pytest

PACKET = Path(__file__).resolve().parent.parent / "candidate-packet"
_OBSERVATION = {
    "observation_id": "obs-1",
    "tracklet_id": "t-1",
    "observed_at": "2026-01-10T10:00:00.000Z",
    "received_at": "2026-01-10T10:00:05.000Z",
    "latitude": 35.0,
    "longitude": -108.0,
    "altitude_ft": 18000.0,
    "source": "feed-a",
    "attributes": {"spd": 290.0, "q": 1},
}
_SUPPORTING = {
    "record_id": "sup-1",
    "observation_id": "obs-1",
    "source": "telemetry-x",
    "received_at": "2026-01-10T10:00:06.000Z",
    "attributes": {"a1": 5486.4},
}


def observation(**changes):
    return {**copy.deepcopy(_OBSERVATION), **changes}


def supporting(**changes):
    return {**copy.deepcopy(_SUPPORTING), **changes}


@pytest.fixture
def packet():
    if not PACKET.exists():
        pytest.skip("data package not present")
    return PACKET
