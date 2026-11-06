"""Geometry and per-tracklet motion from documented time and position fields only.

Attributes are not read here. Their meanings are partly inferred, so any use of them belongs
in a later, separately justified feature step.

Motion uses a Theil-Sen line in a local equirectangular plane: tracklets span tens of kilometres,
sampling is irregular, and isolated position errors occur, so an ordinary least-squares fit
would be pulled by a single bad report.
"""
import math
from dataclasses import dataclass

import numpy as np

from .preprocess import Tracklet

EARTH_RADIUS_KM = 6371.0088  # IUGG mean radius
SECONDS_PER_MICROSECOND = 1e-6


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((phi2 - phi1) / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Initial great-circle bearing, clockwise from north, in [0, 360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta = math.radians(lon2 - lon1)
    y = math.sin(delta) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta)
    return math.degrees(math.atan2(y, x)) % 360


def angle_difference_deg(a, b) -> float:
    """Smallest absolute difference between two directions, in [0, 180]."""
    return abs((a - b + 180) % 360 - 180)


def project_position(lat, lon, east_km_s, north_km_s, seconds) -> tuple[float, float]:
    """Move a position along a constant velocity, using the local plane at its latitude."""
    north_per_deg = math.radians(1) * EARTH_RADIUS_KM
    east_per_deg = north_per_deg * math.cos(math.radians(lat))
    return (
        lat + north_km_s * seconds / north_per_deg,
        (lon + east_km_s * seconds / east_per_deg + 180) % 360 - 180,
    )


@dataclass(frozen=True)
class LineFit:
    east_km_s: float
    north_km_s: float
    speed_kmh: float
    course_deg: float | None  # None when the fitted speed is exactly zero
    median_residual_km: float


@dataclass(frozen=True)
class TrackletMotion:
    tracklet_id: str
    report_count: int
    start_us: int
    end_us: int
    start_position: tuple[float, float]  # observed (lat, lon) of the first report
    end_position: tuple[float, float]  # observed (lat, lon) of the last report
    fitted_start_position: tuple[float, float] | None  # derived from the line fit, robust to a bad endpoint
    fitted_end_position: tuple[float, float] | None
    overall: LineFit | None  # None when the reports don't span any time
    head: LineFit | None  # first third of reports, minimum three
    tail: LineFit | None  # last third of reports, minimum three
    altitude_reports: int
    median_altitude_ft: float | None  # None when no report has an altitude; never zero-filled
    head_altitude_ft: float | None  # median over the first third of reports with an altitude
    tail_altitude_ft: float | None  # median over the last third of reports with an altitude
    altitude_rate_ft_s: float | None = None  # Theil-Sen slope; None without two altitudes at different times
    fitted_start_altitude_ft: float | None = None  # derived from that slope, at the tracklet's first report
    fitted_end_altitude_ft: float | None = None  # derived from that slope, at the tracklet's last report
    altitude_median_residual_ft: float | None = None  # how far reports typically sit from that line: altitude noise

    @property
    def duration_s(self) -> float:
        return (self.end_us - self.start_us) * SECONDS_PER_MICROSECOND


def _fit(seconds, east, north) -> tuple[LineFit, float, float] | None:
    """Return the fit plus its east and north intercepts, or None without a usable time span."""
    first, second = np.triu_indices(len(seconds), 1)
    dt = seconds[second] - seconds[first]
    usable = dt > 0
    if not usable.any():
        return None
    ve = float(np.median((east[second] - east[first])[usable] / dt[usable]))
    vn = float(np.median((north[second] - north[first])[usable] / dt[usable]))
    intercept_e = float(np.median(east - ve * seconds))
    intercept_n = float(np.median(north - vn * seconds))
    residual = float(np.median(np.hypot(east - intercept_e - ve * seconds, north - intercept_n - vn * seconds)))
    speed = math.hypot(ve, vn)
    course = math.degrees(math.atan2(ve, vn)) % 360 if speed > 0 else None
    return LineFit(ve, vn, speed * 3600, course, residual), intercept_e, intercept_n


def summarize_tracklet(tracklet: Tracklet) -> TrackletMotion:
    reports = tracklet.observations
    times = np.array([o.observed_us for o in reports], dtype=np.int64)
    seconds = (times - times[0]) * SECONDS_PER_MICROSECOND
    lat = np.array([o.latitude for o in reports])
    lon = np.array([o.longitude for o in reports])

    lat0, lon0 = float(np.median(lat)), float(lon[0])
    east_per_deg = math.radians(1) * EARTH_RADIUS_KM * math.cos(math.radians(lat0))
    north_per_deg = math.radians(1) * EARTH_RADIUS_KM
    east = ((lon - lon0 + 180) % 360 - 180) * east_per_deg  # wrapped so the antimeridian stays continuous
    north = (lat - lat0) * north_per_deg

    def position(e, n):
        return (lat0 + n / north_per_deg, (lon0 + e / east_per_deg + 180) % 360 - 180)

    window = max(3, len(reports) // 3)
    overall = _fit(seconds, east, north)
    head = _fit(seconds[:window], east[:window], north[:window])
    tail = _fit(seconds[-window:], east[-window:], north[-window:])

    fitted_start = fitted_end = None
    if overall is not None:
        line, intercept_e, intercept_n = overall
        fitted_start = position(intercept_e, intercept_n)
        fitted_end = position(intercept_e + line.east_km_s * seconds[-1], intercept_n + line.north_km_s * seconds[-1])

    altitudes = [o.altitude_ft for o in reports if o.altitude_ft is not None]
    third = max(1, len(altitudes) // 3)
    altitude_rate = fitted_start_altitude = fitted_end_altitude = altitude_residual = None
    if len(altitudes) >= 2:
        known = np.array([o.altitude_ft is not None for o in reports])
        pairs = np.triu_indices(int(known.sum()), 1)
        known_seconds, known_altitudes = seconds[known], np.array(altitudes)
        dt = known_seconds[pairs[1]] - known_seconds[pairs[0]]
        if (dt > 0).any():
            altitude_rate = float(np.median((known_altitudes[pairs[1]] - known_altitudes[pairs[0]])[dt > 0] / dt[dt > 0]))
            fitted_start_altitude = float(np.median(known_altitudes - altitude_rate * known_seconds))
            fitted_end_altitude = fitted_start_altitude + altitude_rate * float(seconds[-1])
            altitude_residual = float(np.median(np.abs(known_altitudes - fitted_start_altitude - altitude_rate * known_seconds)))
    return TrackletMotion(
        tracklet_id=tracklet.tracklet_id,
        report_count=len(reports),
        start_us=int(times[0]),
        end_us=int(times[-1]),
        start_position=(reports[0].latitude, reports[0].longitude),
        end_position=(reports[-1].latitude, reports[-1].longitude),
        fitted_start_position=fitted_start,
        fitted_end_position=fitted_end,
        overall=overall[0] if overall else None,
        head=head[0] if head else None,
        tail=tail[0] if tail else None,
        altitude_rate_ft_s=altitude_rate,
        fitted_start_altitude_ft=fitted_start_altitude,
        fitted_end_altitude_ft=fitted_end_altitude,
        altitude_median_residual_ft=altitude_residual,
        altitude_reports=len(altitudes),
        median_altitude_ft=float(np.median(altitudes)) if altitudes else None,
        head_altitude_ft=float(np.median(altitudes[:third])) if altitudes else None,
        tail_altitude_ft=float(np.median(altitudes[-third:])) if altitudes else None,
    )
