"""Aircraft track association dashboard. Displays what the API returns; all inference goes through HTTP.

Run from the project root: streamlit run dashboard/app.py
"""
import logging
import sys
from functools import partial
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.charts import PlotOptions, build_track_figure, short_id, short_labels  # noqa: E402
from dashboard.client import BACKEND_UNAVAILABLE, TIMEOUT, ApiClient, ApiError  # noqa: E402
from dashboard.compare import (  # noqa: E402
    REPLACEMENT_NOTE,
    MEASUREMENT_NOTE,
    SCALE_NOTES,
    SCALES_DIFFER,
    TRAIN_NOTE,
    UNRESOLVED_CATEGORIES,
    UNRESOLVED_NOTE,
    agreement_counts,
    by_id,
    candidate_rows,
    change_rows,
    compare_snapshots,
    explain_decision,
    gated_rows,
    method_comparison,
    observation_rows,
    policy_text,
    reason_meaning,
    status_line,
    summarize,
    supporting_rows,
    unresolved_groups,
    unresolved_rows,
)

logger = logging.getLogger("govia.dashboard")

TITLE = "Aircraft Track Association — Govia"
METHODS = ("baseline", "learned")
VIEWS = ("Tracks and inspector", "Baseline vs learned", "Unresolved", "Additional evidence")
PARTITION_ORDER = {"validation": 0, "train": 1, "staged_demo": 2}
STAGED = ("staged_demo/snapshot_1", "staged_demo/snapshot_2")
PLOT_CONFIG = {"displaylogo": False, "scrollZoom": False}  # scroll zoom would trap page scrolling over a full-width plot
INVALID_INPUT_CODES = {"invalid_json", "invalid_request", "empty_input", "validation_failed"}
ISSUE_FIELDS = ("code", "record_kind", "message", "index", "record_id", "field")

state = st.session_state


def api_client() -> ApiClient:
    if "api_client" not in state:
        state["api_client"] = ApiClient.from_env()
    return state["api_client"]


def init_state() -> None:
    for name in ("results", "errors"):
        if name not in state:
            state[name] = {}


def table(rows: list[dict], key: str | None = None, height: int | str = "auto") -> None:
    st.dataframe(rows, hide_index=True, width="stretch", height=height, key=key)


def render_error(error: ApiError) -> None:
    """Every failure state, named by what it means to the person looking at it."""
    if error.code == BACKEND_UNAVAILABLE:
        st.error(f"**Backend unavailable.** {error.message}")
        st.caption("Start the API with `uvicorn govia_tracks.api:app --port 8000`. This page recovers on the next interaction once it is back.")
    elif error.code == TIMEOUT:
        st.error(f"**The API timed out.** {error.message}")
    elif error.code == "model_unavailable":
        st.error(f"**Model unavailable.** {error.message}")
    elif error.code in INVALID_INPUT_CODES:
        st.error(f"**Invalid input** (`{error.code}`). {error.message}")
    elif error.code in ("not_found", "method_not_allowed"):
        st.error(f"**Request rejected** (`{error.code}`). {error.message}")
    else:
        st.error(f"**Unexpected server error** (`{error.code}`). {error.message}")
    if error.issues:
        st.caption(f"{len(error.issues)} issue(s) reported by the API")
        rows = [{field: "not available" if issue.get(field) is None else str(issue.get(field)) for field in ISSUE_FIELDS} for issue in error.issues]
        table(rows)


def run_analysis(client: ApiClient, key: str) -> None:
    """Fetch and POST one scenario on its own. Clears that scenario's cached result first."""
    state["results"].pop(key, None)
    state["errors"].pop(key, None)
    partition, name = key.split("/", 1)
    try:
        with st.spinner(f"Running association for {key} (gives up after {client.timeout:.0f} s per request)"):
            records = client.records(partition, name)
            response = client.associate(records)
    except ApiError as error:
        state["errors"][key] = error
    else:
        state["results"][key] = {"records": records, "response": response}


def clear_recovered_errors(health: dict | None) -> None:
    """A connection failure, or a model outage once health is ready, shouldn't outlive its cause."""
    if health is None:
        return
    for key, error in list(state["errors"].items()):
        if error.code in (BACKEND_UNAVAILABLE, TIMEOUT) or (error.code == "model_unavailable" and health.get("ready")):
            del state["errors"][key]


def fetch_health(client: ApiClient) -> tuple[dict | None, ApiError | None]:
    try:
        return client.health(), None
    except ApiError as error:
        return None, error


def fetch_scenarios(client: ApiClient) -> tuple[list[dict], ApiError | None]:
    cached = state.get("scenarios")
    if cached:
        return cached, None
    try:
        scenarios = client.scenarios()
    except ApiError as error:
        return [], error
    state["scenarios"] = scenarios
    return scenarios, None


def render_status(client: ApiClient, health: dict | None) -> None:
    box = st.sidebar
    if health is None:
        box.error("API unavailable")
    elif health.get("ready"):
        box.success("API ready")
    else:
        box.warning("API not ready")
    box.caption(f"API: {client.base_url}")
    box.button("Check again", key="check_api", help="Reruns the page, including the API health check.")


def scenario_picker(scenarios: list[dict]) -> str | None:
    if not scenarios:
        st.sidebar.selectbox("Scenario", ["No scenarios available"], disabled=True, key="scenario_none")
        return None
    groups: dict[str, list[dict]] = {}
    for scenario in scenarios:
        groups.setdefault(scenario.get("partition") or "unknown", []).append(scenario)
    partition = st.sidebar.selectbox(
        "Partition", sorted(groups, key=lambda p: (PARTITION_ORDER.get(p, 9), p)), key="partition",
    )
    members = sorted(groups[partition], key=lambda s: str(s.get("key")))
    by_key = {s["key"]: s for s in members}
    key = st.sidebar.selectbox(
        "Scenario", list(by_key), format_func=lambda k: by_key[k].get("name") or k, key=f"scenario::{partition}",
    )
    meta = by_key[key]
    st.sidebar.caption(
        f"Scenario ID: {meta.get('scenario_id', 'not available')}  \n"
        f"Snapshot: {meta.get('snapshot_id', 'not available')}  \n"
        f"Receipt cutoff: {meta.get('receipt_cutoff') or 'none (complete batch)'}  \n"
        f"Tracklets: {meta.get('tracklet_count', 'not available')}  \n"
        f"Observations: {meta.get('unique_observation_count', 'not available')} unique, {meta.get('observation_row_count', 'not available')} rows  \n"
        f"Supporting records: {meta.get('supporting_row_count', 'not available')} rows"
    )
    return key


def tracklet_picker(entry: dict | None, method: str, key: str | None) -> str | None:
    tracklets = entry["response"][method]["tracklets"] if entry else []
    if not tracklets:
        st.sidebar.selectbox("Tracklet", ["Run association first" if not entry else "No tracklets"], disabled=True, key="tracklet_none")
        return None
    ids = [t["tracklet_id"] for t in tracklets]
    labels = short_labels(ids)
    statuses = {t["tracklet_id"]: t.get("status", "not available") for t in tracklets}
    widget_key = f"tracklet::{key}"
    if widget_key in state and state[widget_key] not in ids:
        del state[widget_key]
    return st.sidebar.selectbox(
        "Tracklet", ids, format_func=lambda i: f"{labels[i]} ({statuses[i]})", key=widget_key,
        help="Clicking a tracklet's report in the plot also selects it. Status is for the selected method.",
    )


def plot_options() -> PlotOptions:
    with st.sidebar.expander("Plot options", expanded=False):
        labels = st.selectbox(
            "Tracklet labels", ("focus", "all", "none"), key="plot_labels",  # labels on every tracklet collide in dense scenarios
            format_func={"all": "All tracklets", "focus": "Selected and related", "none": "None"}.get,
        )
        links = st.checkbox("Accepted links", value=True, key="plot_links")
        candidates = st.checkbox("Competing candidates of selected", value=True, key="plot_candidates")
        dim = st.checkbox("Fade unrelated tracklets", value=True, key="plot_dim")
        count = st.slider("Most candidates drawn", 1, 20, 6, key="plot_max_candidates")
    return PlotOptions(labels=labels, show_links=links, show_candidates=candidates, dim_others=dim, max_candidates=count)


def render_summary(response: dict, method: str) -> None:
    counts = summarize(response[method])
    agreement = agreement_counts(response["baseline"], response["learned"])
    cells = st.columns(6)
    cells[0].metric("Tracklets", counts["tracklets"])
    cells[1].metric(f"Accepted ({method})", counts["accepted"])
    cells[2].metric(f"Unresolved ({method})", counts["unresolved"])
    cells[3].metric(f"Candidates ({method})", counts["candidates"])
    cells[4].metric("Methods agree", agreement["agree"])
    cells[5].metric("Methods disagree", agreement["disagree"])


def on_plot_select(plot_key: str, tracklet_key: str, known: list[str]) -> None:
    """A click reports the point's customdata, whose first entry is always a tracklet ID."""
    points = state[plot_key].selection.points if plot_key in state else []
    for point in reversed(points):
        customdata = point.get("customdata") if isinstance(point, dict) else None
        if customdata and customdata[0] in known:
            state[tracklet_key] = customdata[0]
            return


def render_plot(entry: dict, method: str, selected: str | None, options: PlotOptions, key: str) -> None:
    result = entry["response"][method]
    figure = build_track_figure(entry["records"]["observations"], result, selected, options=options, revision=key)
    plot_key = f"plot::{key}"
    known = [t["tracklet_id"] for t in result["tracklets"]]
    st.plotly_chart(
        figure, key=plot_key, theme=None, selection_mode="points",
        on_select=partial(on_plot_select, plot_key, f"tracklet::{key}", known),
        config=PLOT_CONFIG,
    )
    st.caption(
        "Longitude across, latitude up, with aspect corrected for latitude; no background map. Arrows run from a tracklet's end to its "
        "successor's start. Drag to pan, zoom with the plot toolbar, double-click to reset. Click a report to select its tracklet."
    )


def render_inspector(entry: dict, method: str, tracklet: dict) -> None:
    st.subheader("Association inspector")
    st.markdown(f"**{tracklet['tracklet_id']}**")
    if tracklet.get("status") == "accepted":
        st.success(status_line(tracklet))
    else:
        st.warning(status_line(tracklet))
    st.write(explain_decision(tracklet))
    rows = candidate_rows(tracklet)
    st.caption(f"{SCALE_NOTES[method]} {policy_text(entry['response'][method])}")
    if rows:
        table(rows, key=f"candidates::{method}")
        st.caption(MEASUREMENT_NOTE)
    else:
        st.info("No candidates were scored for this tracklet.")
    gated = gated_rows(tracklet)
    if gated:
        st.caption("Forward pairs removed by physical gates before scoring.")
        table(gated, key="gated")
    else:
        st.caption("No forward pairs were removed by physical gates for this tracklet.")


def render_records(entry: dict, tracklet_id: str) -> None:
    observations = entry["records"]["observations"]
    supporting = entry["records"]["supporting_records"]
    obs_rows = observation_rows(observations, supporting, tracklet_id)
    sup_rows = supporting_rows(observations, supporting, tracklet_id)
    st.subheader("Original records")
    with st.expander(f"Observations ({len(obs_rows)} unique)", expanded=False):
        st.caption(
            "As supplied, in time order; identical retransmissions are collapsed into a copy count. Attribute names are source-specific "
            "and unverified. ref and tag values are not aircraft identities: different aircraft can share them."
        )
        if obs_rows:
            table(obs_rows, key="observations")
        else:
            st.info("No observations were found for this tracklet.")
    with st.expander(f"Supporting records ({len(sup_rows)})", expanded=False):
        if sup_rows:
            without = sum(row["Supporting records"] == "0" for row in obs_rows)
            st.caption(f"{without} of {len(obs_rows)} observations have no supporting record.")
            table(sup_rows, key="supporting")
        else:
            st.info("No supporting record exists for any observation of this tracklet.")


def render_tracks_view(entry: dict, method: str, selected: str | None, options: PlotOptions, key: str) -> None:
    render_plot(entry, method, selected, options, key)
    tracklet = by_id(entry["response"][method]).get(selected)
    if tracklet is None:
        st.info("Select a tracklet to inspect it.")
        return
    render_inspector(entry, method, tracklet)
    render_records(entry, selected)


def render_method_panel(entry: dict, method: str, tracklet: dict | None) -> None:
    st.markdown(f"**{method.capitalize()}**")
    if tracklet is None:
        st.info("This method returned nothing for this tracklet.")
        return
    st.write(status_line(tracklet))
    st.write(explain_decision(tracklet))
    st.caption(SCALE_NOTES[method])
    rows = candidate_rows(tracklet, compact=True)
    if rows:
        table(rows, key=f"compare_candidates::{method}")
    else:
        st.info("No candidates were scored.")


def render_compare_view(entry: dict, key: str, selected: str | None) -> None:
    response = entry["response"]
    st.subheader("Baseline vs learned")
    st.warning(SCALES_DIFFER)
    choice = st.radio("Show tracklets", ("All", "Methods agree", "Methods disagree"), horizontal=True, key="agree_filter")
    rows = method_comparison(response["baseline"], response["learned"])
    if choice != "All":
        rows = [row for row in rows if row["agree"] == (choice == "Methods agree")]
    st.caption("Agree means the same successor, or both unresolved (their reasons may differ).")
    if not rows:
        st.info(f"No tracklets match the filter: {choice.lower()}.")
        return
    table([
        {"Tracklet": short_id(r["tracklet_id"]), "Baseline decision": r["baseline"], "Learned decision": r["learned"], "Agreement": "Agree" if r["agree"] else "Disagree"}
        for r in rows
    ], key="comparison_table")

    ids = [row["tracklet_id"] for row in rows]
    labels = short_labels(ids)
    detail = st.selectbox(
        "Tracklet detail", ids, index=ids.index(selected) if selected in ids else 0, format_func=labels.get, key=f"compare_tracklet::{key}::{choice}",
    )
    left, right = st.columns(2)
    with left:
        render_method_panel(entry, "baseline", by_id(response["baseline"]).get(detail))
    with right:
        render_method_panel(entry, "learned", by_id(response["learned"]).get(detail))


def render_unresolved_view(entry: dict, method: str) -> None:
    result = entry["response"][method]
    st.subheader(f"Unresolved tracklets ({method})")
    st.info(UNRESOLVED_NOTE)
    groups = unresolved_groups(result)
    options = ["All reasons", *UNRESOLVED_CATEGORIES, *[g for g in groups if g not in UNRESOLVED_CATEGORIES]]
    counted = {name: len(groups.get(name, [])) for name in options[1:]}
    choice = st.selectbox(
        "Reason", options, key="unresolved_filter",
        format_func=lambda name: f"{name} ({sum(counted.values()) if name == options[0] else counted[name]})",
    )
    if not groups:
        st.success(f"Every tracklet has an accepted successor under the {method} method.")
        return
    shown = groups if choice == options[0] else {choice: groups.get(choice, [])}
    for name, tracklets in shown.items():
        st.markdown(f"**{name}** ({len(tracklets)})")
        if tracklets:
            codes = sorted({str(t.get("reason_code")) for t in tracklets})
            st.caption(" ".join(f"`{code}`: {reason_meaning(code)}" for code in codes))
            table(unresolved_rows(tracklets), key=f"unresolved::{name}")
        else:
            st.info(f"No unresolved tracklets have this reason under the {method} method.")


def render_snapshot_table(first: dict, second: dict) -> None:
    rows = []
    for name, entry in (("Snapshot 1", first), ("Snapshot 2", second)):
        analysis = entry["response"]["analysis"]
        rows.append({
            "Snapshot": name, "Tracklets": str(analysis.get("tracklet_count", "not available")),
            "Receipt cutoff": str(analysis.get("receipt_cutoff") or "none"),
            "Snapshot ID": str(analysis.get("snapshot_id", "not available")), "Fingerprint": str(analysis.get("input_fingerprint", "not available")),
            "Baseline version": str(entry["response"]["baseline"].get("version", "not available")),
            "Learned version": str(entry["response"]["learned"].get("version", "not available")),
        })
    table(rows, key="snapshot_table")
    st.caption("  \n".join(f"{row['Snapshot']}: {row['Snapshot ID']}, fingerprint {row['Fingerprint']}" for row in rows))


def render_method_changes(method: str, first: dict, second: dict) -> dict:
    changes = compare_snapshots(first["response"][method], second["response"][method])
    st.markdown(f"**{method.capitalize()}**")
    cells = st.columns(5)
    cells[0].metric(f"New tracklets ({method})", len(changes["new_tracklets"]))
    cells[1].metric(f"Links added ({method})", len(changes["added"]))
    cells[2].metric(f"Withdrawn ({method})", len(changes["withdrawn"]))
    cells[3].metric(f"Replaced ({method})", len(changes["replaced"]))
    cells[4].metric(f"Unchanged ({method})", len(changes["unchanged"]))
    if changes["new_tracklets"]:
        st.caption(f"New tracklets: {', '.join(changes['new_tracklets'])}")
    if changes["removed_tracklets"]:
        st.warning(f"Tracklets missing from snapshot 2: {', '.join(changes['removed_tracklets'])}")
    rows = change_rows(changes)
    if rows:
        st.caption(SCALE_NOTES[method])
        table(rows, key=f"changes::{method}")
    else:
        st.info(f"No link changed under the {method} method between these snapshots.")
    return changes


def render_evidence_view(client: ApiClient, scenarios: list[dict], method: str, options: PlotOptions) -> None:
    st.subheader("Additional evidence")
    st.caption("Each snapshot is fetched and analysed on its own, then only the outputs are compared. Nothing is combined across snapshots.")
    known = {s.get("key") for s in scenarios}
    missing = [key for key in STAGED if key not in known]
    if missing:
        if scenarios:  # with no list at all, the error above has already said why
            st.warning(f"The API does not list {', '.join(missing)}, so there is nothing to compare.")
        return
    if st.button("Analyse both snapshots", key="run_staged", type="primary"):
        for key in STAGED:
            run_analysis(client, key)
    first, second = (state["results"].get(key) for key in STAGED)
    failed = False
    for key in STAGED:
        if key in state["errors"]:
            st.markdown(f"**{key}**")
            render_error(state["errors"][key])
            failed = True
    if failed or first is None or second is None:
        if not failed:
            st.info("Press Analyse both snapshots to compare them.")
        return

    render_snapshot_table(first, second)
    if first["response"]["analysis"].get("scenario_id") != second["response"]["analysis"].get("scenario_id"):
        st.warning("These snapshots report different scenario IDs, so their links may not be comparable.")
    st.info(REPLACEMENT_NOTE)
    new_tracklets: list[str] = []
    for each in METHODS:
        changes = render_method_changes(each, first, second)
        if each == method:
            new_tracklets = changes["new_tracklets"]
    st.markdown(f"**Snapshot 2, {method} links**")
    st.caption("Uses the method selected in the sidebar. New tracklets are highlighted.")
    figure = build_track_figure(second["records"]["observations"], second["response"][method], highlight=new_tracklets, options=options, revision="staged")
    st.plotly_chart(figure, key="staged_plot", theme=None, config=PLOT_CONFIG)


def render_footer(entry: dict) -> None:
    response = entry["response"]
    fingerprint = response["analysis"].get("input_fingerprint") or response["baseline"].get("input_fingerprint") or "not available"
    st.divider()
    st.caption(
        f"Input fingerprint: {fingerprint}  \nBaseline version: {response['baseline'].get('version', 'not available')}  \n"
        f"Learned version: {response['learned'].get('version', 'not available')}"
    )


def main() -> None:
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    init_state()
    client = api_client()

    health, health_error = fetch_health(client)
    clear_recovered_errors(health)
    scenarios, scenario_error = fetch_scenarios(client)
    render_status(client, health)

    key = scenario_picker(scenarios)
    run_clicked = st.sidebar.button("Run association", type="primary", key="run", disabled=key is None, width="stretch")
    method = st.sidebar.radio("Method view", METHODS, horizontal=True, key="method", format_func=str.capitalize)

    if health_error is not None:
        render_error(health_error)
    elif not health.get("ready"):
        st.error("**Model unavailable.** The API reports that it is not ready.")
        for problem in health.get("problems") or ["The API gave no details."]:
            st.write(f"- {problem}")
    if scenario_error is not None and health_error is None:
        render_error(scenario_error)
    elif scenario_error is None and not scenarios:
        st.info("The API lists no scenarios.")

    if run_clicked and key is not None:
        run_analysis(client, key)
    entry = state["results"].get(key) if key else None
    error = state["errors"].get(key) if key else None
    selected = tracklet_picker(entry, method, key)
    options = plot_options()

    view = st.segmented_control("View", VIEWS, default=VIEWS[0], required=True, key="view", label_visibility="collapsed") or VIEWS[0]

    backend_down = health_error is not None and health_error.code in (BACKEND_UNAVAILABLE, TIMEOUT)
    if view == VIEWS[3]:
        if run_clicked and error is not None and not backend_down:
            render_error(error)
        elif run_clicked and entry is not None:
            st.success(f"Analysed {key}. Its results are under Tracks and inspector; this view compares the staged snapshots.")
        render_evidence_view(client, scenarios, method, options)
        return
    if error is not None and not (backend_down and error.code in (BACKEND_UNAVAILABLE, TIMEOUT)):  # already shown above
        render_error(error)
        return
    if entry is None:
        if key is not None and not backend_down:
            st.info(f"Press **Run association** in the sidebar to analyse {key}.")
        return
    response = entry["response"]
    if not response[method]["tracklets"]:
        st.info(f"The API returned no tracklets for {key}.")
        render_footer(entry)
        return
    if health_error is not None or not health.get("ready"):
        st.warning("These results are from an earlier successful analysis. The API can't run a new one right now.")
    render_summary(response, method)
    if key.startswith("train/"):
        st.caption(TRAIN_NOTE)
    if view == VIEWS[0]:
        render_tracks_view(entry, method, selected, options, key)
    elif view == VIEWS[1]:
        render_compare_view(entry, key, selected)
    else:
        render_unresolved_view(entry, method)
    render_footer(entry)


try:
    main()
except Exception:  # never show a raw traceback; the full error goes to the server log
    logger.exception("Unhandled dashboard error")
    st.error("**The dashboard hit an unexpected problem while rendering.** The details are in the server log. Rerun the analysis, or restart the dashboard if it persists.")
