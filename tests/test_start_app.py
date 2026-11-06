"""The launcher: what it refuses to start without, and a real start and stop of both services."""
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "scripts" / "start_app.py"
specification = importlib.util.spec_from_file_location("start_app", LAUNCHER)
start_app = importlib.util.module_from_spec(specification)
specification.loader.exec_module(start_app)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def complete_root(tmp_path):
    for name in start_app.REQUIRED_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("{}", encoding="utf-8")
    for folder in ("train/scenario_001", "staged_demo/snapshot_1"):
        (tmp_path / "candidate-packet" / "data" / folder).mkdir(parents=True)
    return tmp_path


def test_a_complete_project_has_no_problems(tmp_path):
    assert start_app.missing_requirements(complete_root(tmp_path)) == []


@pytest.mark.parametrize("name", start_app.REQUIRED_FILES)
def test_each_missing_artifact_is_named(tmp_path, name):
    root = complete_root(tmp_path)
    (root / name).unlink()
    assert start_app.missing_requirements(root) == [f"missing file: {name}"]


@pytest.mark.parametrize("folder", ["train/scenario_001", "staged_demo/snapshot_1"])
def test_missing_data_is_reported(tmp_path, folder):
    root = complete_root(tmp_path)
    shutil.rmtree(root / "candidate-packet" / "data" / folder)
    assert "missing dataset" in start_app.missing_requirements(root)[0]


def test_missing_package_says_how_to_install(tmp_path, monkeypatch):
    monkeypatch.setattr(start_app.importlib.util, "find_spec", lambda name: None if name == "streamlit" else object())
    problems = start_app.missing_requirements(complete_root(tmp_path))
    assert len(problems) == 1 and "streamlit" in problems[0] and "pip install" in problems[0]


def test_port_check_sees_a_listener():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        assert start_app.port_in_use(listener.getsockname()[1])
    assert not start_app.port_in_use(free_port())


def test_a_busy_port_stops_startup_with_a_clear_message():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        done = subprocess.run([sys.executable, str(LAUNCHER), "--api-port", str(port), "--dashboard-port", str(free_port())], capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 1
    assert f"port {port} is already in use" in done.stderr and "--api-port" in done.stderr


@pytest.mark.data
def test_starts_both_services_and_stops_them_on_interrupt(packet):
    api, dashboard = free_port(), free_port()
    flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen([sys.executable, str(LAUNCHER), "--api-port", str(api), "--dashboard-port", str(dashboard)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", **flags)
    lines = []
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            lines.append(line)
            if "Press Ctrl+C" in line or not line:
                break
        assert any(f"Dashboard: http://localhost:{dashboard}" in line for line in lines), lines
        with urllib.request.urlopen(f"http://127.0.0.1:{api}/health", timeout=5) as response:
            assert json.load(response)["ready"] is True
        with urllib.request.urlopen(f"http://127.0.0.1:{dashboard}/_stcore/health", timeout=5) as response:
            assert response.status == 200
        os.kill(process.pid, signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        process.communicate(timeout=30)
        assert process.returncode in (0, 3221225786)  # 0xC000013A: Windows reports the break itself on some consoles
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and (start_app.port_in_use(api) or start_app.port_in_use(dashboard)):
            time.sleep(0.5)
        assert not start_app.port_in_use(api) and not start_app.port_in_use(dashboard)
    finally:
        if process.poll() is None:
            process.kill()
