"""Geometry and motion against independently known values, not against the same formulas."""
import math

import pytest

from govia_tracks.movement import angle_difference_deg, bearing_deg, haversine_km, summarize_tracklet
from govia_tracks.preprocess import prepare_scenario

from conftest import observation

R = 6371.0088


def track(points, tracklet_id="t"):
    """points: (seconds, lat, lon)."""
    rows = [
        observation(
            observation_id=f"o{i}", tracklet_id=tracklet_id,
            observed_at=f"2026-01-10T10:{int(s) // 60:02d}:{int(s) % 60:02d}.000Z",
            received_at="2026-01-10T11:00:00.000Z", latitude=lat, longitude=lon,
        )
        for i, (s, lat, lon) in enumerate(points)
    ]
    return prepare_scenario(rows).tracklets[0]


def test_haversine_known_distances():
    assert haversine_km(0, 0, 90, 0) == pytest.approx(math.pi / 2 * R)  # equator to pole
    assert haversine_km(0, 0, 0, 180) == pytest.approx(math.pi * R)  # half the equator
    assert haversine_km(10, 20, 10, 20) == 0
    assert haversine_km(0, 179.5, 0, -179.5) == pytest.approx(math.radians(1) * R)  # across the antimeridian
    # Paris to London by published coordinates; the widely quoted figure is about 344 km.
    assert haversine_km(48.8566, 2.3522, 51.5074, -0.1278) == pytest.approx(343.5, abs=1.5)


@pytest.mark.parametrize("to,expected", [((1, 0), 0), ((0, 1), 90), ((-1, 0), 180), ((0, -1), 270)])
def test_bearing_cardinal_directions_from_the_origin(to, expected):
    assert bearing_deg(0, 0, *to) == pytest.approx(expected)


@pytest.mark.parametrize("a,b,expected", [(350, 10, 20), (10, 350, 20), (0, 180, 180), (90, 90, 0)])
def test_angle_difference(a, b, expected):
    assert angle_difference_deg(a, b) == pytest.approx(expected)


def north_track(count=9, step_s=10, deg_per_s=0.002, lat0=35.0, lon0=-108.0):
    return [(i * step_s, lat0 + i * step_s * deg_per_s, lon0) for i in range(count)]


def test_constant_northbound_track_recovers_known_speed_and_course():
    # 0.002 deg/s of latitude is 0.002 * (pi/180) * R km/s.
    expected_kmh = 0.002 * math.pi / 180 * R * 3600
    motion = summarize_tracklet(track(north_track()))
    assert motion.overall.speed_kmh == pytest.approx(expected_kmh, rel=1e-6)
    assert motion.overall.course_deg == pytest.approx(0, abs=1e-6)
    assert motion.overall.median_residual_km == pytest.approx(0, abs=1e-6)
    assert motion.duration_s == 80
    assert motion.report_count == 9


def test_eastbound_track_at_latitude_accounts_for_longitude_shrinkage():
    # At 60N a degree of longitude is half as long as at the equator.
    points = [(i * 10, 60.0, -100 + i * 0.02) for i in range(9)]
    motion = summarize_tracklet(track(points))
    assert motion.overall.course_deg == pytest.approx(90, abs=1e-6)
    assert motion.overall.speed_kmh == pytest.approx(0.002 * math.pi / 180 * R * 0.5 * 3600, rel=1e-6)


def test_irregular_sampling_does_not_bias_speed():
    regular = north_track()
    irregular = [regular[i] for i in (0, 1, 2, 5, 6, 8)]
    assert summarize_tracklet(track(irregular)).overall.speed_kmh == pytest.approx(
        summarize_tracklet(track(regular)).overall.speed_kmh, rel=1e-6)


def test_single_bad_report_barely_moves_the_fit():
    clean = north_track(count=11)
    corrupted = list(clean)
    corrupted[5] = (clean[5][0], clean[5][1] + 1.0, clean[5][2])  # roughly 111 km off
    good, bad = summarize_tracklet(track(clean)), summarize_tracklet(track(corrupted))
    assert bad.overall.speed_kmh == pytest.approx(good.overall.speed_kmh, rel=0.02)
    assert bad.overall.course_deg == pytest.approx(0, abs=1)


def test_bad_endpoint_is_visible_and_the_fitted_endpoint_resists_it():
    clean = north_track(count=11)
    corrupted = clean[:-1] + [(clean[-1][0], clean[-1][1] + 0.5, clean[-1][2])]
    motion = summarize_tracklet(track(corrupted))
    assert motion.end_position[0] == corrupted[-1][1]  # the observed value is preserved
    assert haversine_km(*motion.fitted_end_position, clean[-1][1], clean[-1][2]) < 1


def test_track_crossing_the_antimeridian():
    points = [(i * 10, 0.0, 179.96 + i * 0.02) for i in range(5)]
    points = [(s, lat, lon - 360 if lon > 180 else lon) for s, lat, lon in points]
    motion = summarize_tracklet(track(points))
    assert motion.overall.course_deg == pytest.approx(90, abs=1e-6)
    assert motion.overall.speed_kmh == pytest.approx(0.002 * math.pi / 180 * R * 3600, rel=1e-6)
    assert -180 <= motion.fitted_end_position[1] <= 180


def test_single_report_has_no_motion_rather_than_zero_motion():
    motion = summarize_tracklet(track([(0, 35.0, -108.0)]))
    assert motion.report_count == 1
    assert motion.overall is None and motion.fitted_end_position is None and motion.head is None


def test_reports_at_one_instant_have_no_motion():
    motion = summarize_tracklet(track([(0, 35.0, -108.0), (0, 35.1, -108.0)]))
    assert motion.overall is None


def test_stationary_track_reports_no_course_rather_than_an_arbitrary_one():
    motion = summarize_tracklet(track([(i * 10, 35.0, -108.0) for i in range(4)]))
    assert motion.overall.speed_kmh == 0 and motion.overall.course_deg is None


def test_head_and_tail_see_a_turn():
    west = [(i * 10, 35.0, -108 - i * 0.01) for i in range(6)]
    north = [(60 + i * 10, 35.0 + (i + 1) * 0.01, -108.05) for i in range(6)]
    motion = summarize_tracklet(track(west + north))
    assert motion.head.course_deg == pytest.approx(270, abs=1)
    assert motion.tail.course_deg == pytest.approx(0, abs=1)
