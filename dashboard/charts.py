"""Lat/lon track plot, built only from raw observations and an API result.

Dedupe and ordering here are for display only; no decision depends on them. Color encodes a tracklet's
status, not its identity, and every color is paired with a dash style or marker so it never stands alone.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import plotly.graph_objects as go

NOT_AVAILABLE = "not available"

INK = "#1a1d21"
MUTED = "#5b6168"
GRID = "#e6e8eb"
# Passed the palette validator's color-vision checks in this order on a light surface.
ACCEPTED = "#2a6fb0"
UNRESOLVED = "#c2570c"
LINK = "#0f8a6d"
CANDIDATE = "#a3339a"
FADED = 0.55


@dataclass(frozen=True)
class PlotOptions:
    labels: str = "all"  # "all", "focus" (selected and related) or "none"
    show_links: bool = True
    show_candidates: bool = True
    max_candidates: int = 6
    dim_others: bool = True


def short_id(tracklet_id: str) -> str:
    """First block of the ID, for labels. Full IDs stay in hovers and tables."""
    text = str(tracklet_id).removeprefix("tracklet-")
    return text.split("-", 1)[0] or text


def short_labels(tracklet_ids) -> dict[str, str]:
    """A short label per ID, lengthened only if two IDs would otherwise share one."""
    labels = {tracklet_id: short_id(tracklet_id) for tracklet_id in tracklet_ids}
    for tracklet_id, label in list(labels.items()):
        if sum(other == label for other in labels.values()) > 1:
            labels[tracklet_id] = str(tracklet_id).removeprefix("tracklet-")
    return labels


def time_key(text) -> tuple:
    """Sort key for a timestamp. An unparseable one sorts last rather than failing the page."""
    try:
        parsed = datetime.fromisoformat(str(text))
    except ValueError:
        return (1, 0.0, str(text))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (0, parsed.timestamp(), str(text))


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def altitude_text(observation: dict) -> str:
    altitude = observation.get("altitude_ft")
    return f"{altitude:,.0f} ft" if _is_number(altitude) else NOT_AVAILABLE


def tracks_from_observations(observations) -> dict[str, list[dict]]:
    """{tracklet_id: reports}, deduped by observation_id and in observed_at order, sorted by tracklet ID."""
    seen: dict[str, dict[str, dict]] = {}
    for row in observations:
        if _is_number(row.get("latitude")) and _is_number(row.get("longitude")):
            seen.setdefault(row.get("tracklet_id"), {}).setdefault(row.get("observation_id"), row)
    return {
        tracklet_id: sorted(rows.values(), key=lambda r: (time_key(r.get("observed_at")), str(r.get("observation_id"))))
        for tracklet_id, rows in sorted(seen.items(), key=lambda item: str(item[0]))
    }


def _report_customdata(tracklet_id: str, report: dict) -> list:
    return [
        tracklet_id, report.get("observation_id") or NOT_AVAILABLE, report.get("observed_at") or NOT_AVAILABLE,
        altitude_text(report), report.get("source") or NOT_AVAILABLE,
    ]


REPORT_HOVER = (
    "<b>%{customdata[0]}</b><br>Observation: %{customdata[1]}<br>Time: %{customdata[2]}<br>"
    "Altitude: %{customdata[3]}<br>Source: %{customdata[4]}<br>Lat, lon: %{y:.4f}, %{x:.4f}<extra>%{fullData.name}</extra>"
)
# Plotly dims unselected points after a click. The selector and thick line already show what's selected.
_NO_DIM = dict(selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)))


def _legend(group: str, title: str, seen: set, rank: int) -> dict:
    first = group not in seen
    seen.add(group)
    return dict(legendgroup=group, showlegend=first, legendrank=rank, name=title)


def _candidate_hover(tracklet_id, candidate, method) -> str:
    m = candidate.get("measurements") or {}
    gap = m.get("gap_s")
    score = candidate.get("score")
    return (
        f"<b>Candidate rank {candidate.get('rank')}</b><br>{tracklet_id}<br>to {candidate.get('successor')}<br>"
        f"Decision: {candidate.get('decision')}<br>Gap: {f'{gap:.1f} s' if _is_number(gap) else NOT_AVAILABLE}<br>"
        f"Score ({method} scale): {f'{score:.3f}' if _is_number(score) else NOT_AVAILABLE}"
    )


def _arrow(a, b, color, width, dash, customdata, hover, **legend) -> go.Scatter:
    return go.Scatter(
        x=[a["longitude"], b["longitude"]], y=[a["latitude"], b["latitude"]], mode="lines+markers",
        line=dict(color=color, width=width, dash=dash),
        marker=dict(symbol="arrow", angleref="previous", size=[0, 11], color=color),
        customdata=[customdata, customdata], hovertext=[hover, hover], hovertemplate="%{hovertext}<extra></extra>",
        **_NO_DIM, **legend,
    )


def _label_anchors(tracks: dict[str, list[dict]], shown: list[str]) -> list[dict]:
    """For each tracklet, the report farthest from labels already placed, so close tracks don't stack their labels."""
    placed: list[dict] = []
    for tracklet_id in shown:
        reports = tracks[tracklet_id]
        best, best_gap = reports[len(reports) // 2], -1.0
        for report in reports if placed else ():  # the first label just takes its midpoint
            shrink = math.cos(math.radians(report["latitude"]))
            gap = min(((report["longitude"] - p["longitude"]) * shrink) ** 2 + (report["latitude"] - p["latitude"]) ** 2 for p in placed)
            if gap > best_gap:
                best, best_gap = report, gap
        placed.append(best)
    return placed


def build_track_figure(observations, result: dict, selected: str | None = None, highlight=(), options: PlotOptions = PlotOptions(),
                       revision: str | None = None, height: int = 640) -> go.Figure:
    """One method's view of one scenario. `highlight` marks tracklets of special interest, such as a new fragment."""
    tracks = tracks_from_observations(observations)
    tracklets = {t.get("tracklet_id"): t for t in result.get("tracklets", [])}
    labels = short_labels(tracks)
    links = {a: b for a, b in (result.get("links") or {}).items() if a in tracks and b in tracks}
    method = result.get("method", "selected")
    highlight = {h for h in highlight if h in tracks}
    selected = selected if selected in tracks else None

    candidates = []
    if selected is not None and options.show_candidates:
        ranked = sorted(tracklets.get(selected, {}).get("candidates") or [], key=lambda c: c.get("rank", 0))
        candidates = [c for c in ranked if c.get("successor") in tracks and links.get(selected) != c.get("successor")][:options.max_candidates]
    related = set(highlight)
    if selected is not None:
        related |= {selected} | {c["successor"] for c in candidates}
        related |= {b for a, b in links.items() if a == selected} | {a for a, b in links.items() if b == selected}
    dimmed = options.dim_others and bool(related)

    fig = go.Figure()
    seen: set = set()

    def path_trace(tracklet_id, color, width, dash, opacity, group, title, rank):
        reports = tracks[tracklet_id]
        return go.Scatter(
            x=[r["longitude"] for r in reports], y=[r["latitude"] for r in reports], mode="lines+markers",
            line=dict(color=color, width=width, dash=dash), marker=dict(color=color, size=5), opacity=opacity,
            customdata=[_report_customdata(tracklet_id, r) for r in reports], hovertemplate=REPORT_HOVER,
            meta=dict(role="path", tracklet_id=tracklet_id), **_NO_DIM, **_legend(group, title, seen, rank),
        )

    emphasised = []
    for tracklet_id in tracks:
        if tracklet_id == selected or tracklet_id in highlight:
            emphasised.append(tracklet_id)
            continue
        opacity = FADED if dimmed and tracklet_id not in related else 1.0
        if tracklets.get(tracklet_id, {}).get("status") == "accepted":
            fig.add_trace(path_trace(tracklet_id, ACCEPTED, 1.6, "solid", opacity, "accepted", "Tracklet with accepted successor", 10))
        else:
            fig.add_trace(path_trace(tracklet_id, UNRESOLVED, 1.6, "dot", opacity, "unresolved", "Tracklet unresolved", 20))

    if options.show_links:
        for a, b in sorted(links.items()):
            hover = f"<b>Accepted link</b><br>{a}<br>to {b}"
            trace = _arrow(tracks[a][-1], tracks[b][0], LINK, 2.5 if a == selected else 2, "solid", [a, b], hover,
                           **_legend("link", "Accepted link", seen, 30))
            trace.meta = dict(role="link", predecessor=a, successor=b)
            trace.opacity = FADED if dimmed and a not in related and b not in related else 1.0
            fig.add_trace(trace)
    for candidate in candidates:
        successor = candidate["successor"]
        trace = _arrow(tracks[selected][-1], tracks[successor][0], CANDIDATE, 1.6, "dash", [selected, successor],
                       _candidate_hover(selected, candidate, method), **_legend("candidate", "Competing candidate of selected", seen, 40))
        trace.meta = dict(role="candidate", predecessor=selected, successor=successor, rank=candidate.get("rank"))
        fig.add_trace(trace)

    for tracklet_id in emphasised:
        if tracklet_id == selected:
            fig.add_trace(path_trace(tracklet_id, INK, 4, "solid", 1.0, "selected", "Selected tracklet", 5))
        else:
            fig.add_trace(path_trace(tracklet_id, CANDIDATE, 4, "solid", 1.0, "new", "New tracklet", 6))

    def end_markers(pick, symbol, title, role, rank):
        ids = list(tracks)
        reports = [tracks[i][pick] for i in ids]
        fig.add_trace(go.Scatter(
            x=[r["longitude"] for r in reports], y=[r["latitude"] for r in reports], mode="markers",
            marker=dict(symbol=symbol, size=9, color=INK, line=dict(color=INK, width=1.5)),
            opacity=1.0, customdata=[_report_customdata(i, r) for i, r in zip(ids, reports)], hovertemplate=REPORT_HOVER,
            meta=dict(role=role), name=title, legendgroup=role, legendrank=rank, **_NO_DIM,
        ))

    if tracks:
        end_markers(0, "circle-open", "Tracklet start", "start", 50)
        end_markers(-1, "square", "Tracklet end", "end", 51)

    if options.labels != "none":
        shown = list(tracks) if options.labels == "all" else [t for t in tracks if t in related]
        anchors = _label_anchors(tracks, shown)
        if shown:
            fig.add_trace(go.Scatter(
                x=[a["longitude"] for a in anchors], y=[a["latitude"] for a in anchors], mode="text",
                text=[labels[t] for t in shown], textposition="top center",
                textfont=dict(size=11, color=INK, shadow="0 0 3px #ffffff, 0 0 3px #ffffff"),
                hoverinfo="skip", showlegend=False, meta=dict(role="labels"), name="Labels",
            ))

    latitudes = [r["latitude"] for reports in tracks.values() for r in reports]
    mean_latitude = sum(latitudes) / len(latitudes) if latitudes else 0.0
    # One degree of longitude spans cos(latitude) of a degree of latitude, so latitude gets more pixels per degree.
    scale_ratio = 1 / max(math.cos(math.radians(mean_latitude)), 0.1)
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=60, r=16, t=28, b=112), hovermode="closest", dragmode="pan",
        uirevision=revision, font=dict(color=INK, size=12), hoverlabel=dict(bgcolor="#ffffff", font=dict(color=INK, size=12)),
        legend=dict(orientation="h", x=0, y=-0.1, xanchor="left", yanchor="top", font=dict(size=12), groupclick="togglegroup"),
    )
    fig.update_xaxes(title_text="Longitude (degrees)", tickformat=".2f", gridcolor=GRID, zeroline=False, color=MUTED)
    fig.update_yaxes(title_text="Latitude (degrees)", tickformat=".2f", gridcolor=GRID, zeroline=False, color=MUTED,
                     scaleanchor="x", scaleratio=scale_ratio)
    return fig
