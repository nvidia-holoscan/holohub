#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_TIMEOUT_SECONDS = 120


def _timeout_output(exc: subprocess.TimeoutExpired) -> str:
    streams = []
    for stream in (exc.stdout, exc.stderr):
        if isinstance(stream, bytes):
            streams.append(stream.decode(errors="replace"))
        elif stream:
            streams.append(stream)
    return "\n".join(streams)


def _holoscan_cli_env() -> Dict[str, str]:
    env = os.environ.copy()
    source = env.get("HOLOSCAN_CLI_SOURCE")
    if source:
        src_path = str(Path(source).expanduser().resolve() / "src")
        env["PYTHONPATH"] = f"{src_path}:{env.get('PYTHONPATH', '')}".rstrip(":")
    # holoscan-cli#177 decoupled the container base image from the CLI version:
    # `run`/`build`/`package` fatal unless a base SDK version (or --base-img) is
    # configured. The ./holohub wrapper sets this; the direct `python -m
    # holoscan_cli` invocations below must supply it too. Mirror the HoloHub
    # default SDK version so the dryrun resolves a base image.
    env.setdefault("HOLOSCAN_CLI_BASE_SDK_VERSION", "4.3.0")
    return env


def _run_holoscan_cli(*args: str) -> subprocess.CompletedProcess:
    env = _holoscan_cli_env()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "holoscan_cli", *args],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=CLI_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"holoscan_cli timed out after {CLI_TIMEOUT_SECONDS}s: {exc.cmd}\n"
            f"{_timeout_output(exc)}"
        )
    # `python -m holoscan_cli` reports the missing package as
    # `No module named holoscan_cli` (not a ModuleNotFoundError traceback)
    # and the wrapper bootstrap can surface the same string via pip's
    # output, so match on that phrase rather than the exception name.
    if (
        result.returncode != 0
        and "No module named" in result.stderr
        and "holoscan_cli" in result.stderr
    ):
        pytest.skip("holoscan_cli is not installed; set HOLOSCAN_CLI_SOURCE to a checkout")
    return result


def test_unified_cli_lists_holohub_projects():
    result = _run_holoscan_cli("list")

    assert result.returncode == 0, result.stderr
    assert "== APPLICATIONS" in result.stdout
    assert "endoscopy_tool_tracking" in result.stdout


def test_unified_cli_dryruns_holohub_project_run():
    # `endoscopy_tool_tracking` ships both cpp and python; pass --language
    # explicitly so the smoke succeeds rather than asking the user to
    # disambiguate (the latter behavior is covered by the CTest entry
    # `test_holohub_run_multi_language_project`).
    result = _run_holoscan_cli("run", "endoscopy_tool_tracking", "--language", "python", "--dryrun")
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "Running endoscopy_tool_tracking" in output
    assert "docker run" in output
    assert "holoscan run endoscopy_tool_tracking" in output


def _run_holohub_wrapper(*args: str) -> subprocess.CompletedProcess:
    env = _holoscan_cli_env()
    try:
        result = subprocess.run(
            [str(REPO_ROOT / "holohub"), *args],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=CLI_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"holohub wrapper timed out after {CLI_TIMEOUT_SECONDS}s: {exc.cmd}\n"
            f"{_timeout_output(exc)}"
        )
    # `python -m holoscan_cli` reports the missing package as
    # `No module named holoscan_cli` (not a ModuleNotFoundError traceback)
    # and the wrapper bootstrap can surface the same string via pip's
    # output, so match on that phrase rather than the exception name.
    if (
        result.returncode != 0
        and "No module named" in result.stderr
        and "holoscan_cli" in result.stderr
    ):
        pytest.skip("holoscan_cli is not installed; set HOLOSCAN_CLI_SOURCE to a checkout")
    return result


def test_wrapper_delegates_list_to_holoscan_cli():
    """./holohub list should hit the consolidated CLI's source-project dispatch."""
    result = _run_holohub_wrapper("list")

    assert result.returncode == 0, result.stderr
    assert "== APPLICATIONS" in result.stdout
    assert "endoscopy_tool_tracking" in result.stdout


def test_wrapper_dryruns_project_run_through_consolidated_cli():
    """./holohub run renders an in-container `holoscan run` recursion."""
    # Same multi-language note as `test_unified_cli_dryruns_holohub_project_run`.
    result = _run_holohub_wrapper(
        "run", "endoscopy_tool_tracking", "--language", "python", "--dryrun"
    )
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "Running endoscopy_tool_tracking" in output
    assert "docker run" in output
    assert "holoscan run endoscopy_tool_tracking" in output
