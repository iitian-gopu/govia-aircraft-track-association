"""Start the API and dashboard together, using the current Python environment. Ctrl+C stops both.

  python scripts/start_app.py
"""
import argparse
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = ("configs/baseline.json", "configs/learned.json", "models/learned_model.joblib", "models/learned_model.json")
REQUIRED_MODULES = ("fastapi", "uvicorn", "streamlit", "sklearn")
START_TIMEOUT_S = 90


def missing_requirements(root: Path = ROOT) -> list[str]:
    """Everything that would stop the application from working, in plain words."""
    problems = [f"missing file: {name}" for name in REQUIRED_FILES if not (root / name).is_file()]
    data = root / "candidate-packet" / "data"
    if not any(data.glob("*/scenario_*")) or not any(data.glob("staged_demo/snapshot_*")):
        problems.append("missing dataset: candidate-packet/data has no scenarios or staged snapshots")
    modules = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if modules:
        problems.append(f"missing packages: {', '.join(modules)}. Install them with: {Path(sys.executable).name} -m pip install -e \".[dev]\"")
    return problems


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


def fetch_json(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def stop(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_for_api(api: subprocess.Popen, port: int) -> str | None:
    """Wait until the API answers. Returns what's wrong, or None once it reports ready."""
    deadline = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < deadline:
        if api.poll() is not None:
            return f"the API exited during startup with code {api.returncode}; its output is above"
        try:
            status, body = fetch_json(f"http://127.0.0.1:{port}/health")
        except (OSError, ValueError):
            time.sleep(0.5)
            continue
        if status == 200:
            return None
        return "the API started, but isn't ready: " + "; ".join(body.get("problems") or ["no details given"])
    return f"the API didn't answer within {START_TIMEOUT_S} s"


def wait_for_dashboard(dashboard: subprocess.Popen, port: int) -> str | None:
    deadline = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < deadline:
        if dashboard.poll() is not None:
            return f"the dashboard exited during startup with code {dashboard.returncode}; its output is above"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=3) as response:
                if response.status == 200:
                    return None
        except OSError:
            time.sleep(0.5)
    return f"the dashboard didn't answer within {START_TIMEOUT_S} s"


def _interrupt(*_) -> None:
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the API and dashboard. Ctrl+C stops both.")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--dashboard-port", type=int, default=8501)
    arguments = parser.parse_args()

    problems = missing_requirements()
    problems += [f"port {port} is already in use: stop whatever is using it, or choose another with {flag}"
                 for port, flag in ((arguments.api_port, "--api-port"), (arguments.dashboard_port, "--dashboard-port")) if port_in_use(port)]
    if problems:
        print("Can't start:", *(f"  - {problem}" for problem in problems), sep="\n", file=sys.stderr)
        return 1

    signal.signal(signal.SIGTERM, _interrupt)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _interrupt)

    print("Starting the API...", flush=True)
    processes = [subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "govia_tracks.api:app", "--port", str(arguments.api_port), "--log-level", "warning"], cwd=ROOT,
    )]
    try:
        failure = wait_for_api(processes[0], arguments.api_port)
        if failure is None:
            print("Starting the dashboard...", flush=True)
            processes.append(subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.port", str(arguments.dashboard_port), "--logger.level", "warning"],
                cwd=ROOT, env={**os.environ, "GOVIA_API_URL": f"http://127.0.0.1:{arguments.api_port}"},
            ))
            failure = wait_for_dashboard(processes[1], arguments.dashboard_port)
        if failure is not None:
            print(f"Can't start: {failure}", file=sys.stderr)
            return 1

        print(f"\nDashboard: http://localhost:{arguments.dashboard_port}", f"API docs:  http://localhost:{arguments.api_port}/docs", "Press Ctrl+C to stop.\n", sep="\n", flush=True)
        while True:
            for process, name in zip(processes, ("API", "dashboard")):
                if process.poll() is not None:
                    print(f"The {name} stopped unexpectedly with code {process.returncode}; stopping the rest.", file=sys.stderr)
                    return 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
        return 0
    finally:
        stop(processes)


if __name__ == "__main__":
    sys.exit(main())
