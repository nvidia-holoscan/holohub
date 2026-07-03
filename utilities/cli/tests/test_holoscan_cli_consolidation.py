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
    env.setdefault("HOLOSCAN_CLI_BASE_SDK_VERSION", "4.4.0")
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


def _run_holohub_wrapper(
    *args: str,
    extra_env: Dict[str, str] | None = None,
    env: Dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    if env is None:
        env = _holoscan_cli_env()
    if extra_env:
        env = {**env, **extra_env}
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


def test_wrapper_runs_source_override_from_managed_venv(tmp_path):
    """A source override still runs inside the wrapper-managed venv."""
    source = os.environ.get("HOLOSCAN_CLI_SOURCE")
    if not source:
        pytest.skip("set HOLOSCAN_CLI_SOURCE to test managed-venv bootstrap")

    env = os.environ.copy()
    # Keep this test focused on the wrapper's source handling: an inherited
    # PYTHONPATH could make the system interpreter appear ready, and an
    # inherited VIRTUAL_ENV / explicit interpreter override would take the
    # wrapper's explicit-choice branch instead of creating the managed venv
    # (e.g. when pytest itself runs inside an activated venv).
    for var in ("PYTHONPATH", "VIRTUAL_ENV", "HOLOSCAN_CLI_PYTHON_BIN", "HOLOHUB_PYTHON_BIN"):
        env.pop(var, None)
    env["HOLOSCAN_CLI_SOURCE"] = source
    env["HOLOSCAN_CLI_VENV"] = str(tmp_path / "holoscan-cli-venv")
    result = _run_holohub_wrapper("env-info", env=env)

    assert result.returncode == 0, result.stderr
    assert f"Executable: {env['HOLOSCAN_CLI_VENV']}/bin/python" in result.stdout


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


def test_wrapper_quotes_default_docker_build_args_with_single_quotes():
    """./holohub forwards docker build args that shlex can split."""
    install_args = "--pre --extra-index-url https://example.invalid/simple pkg'name>1.0"
    result = _run_holohub_wrapper(
        "run-container",
        "--dryrun",
        "--verbose",
        extra_env={"HOLOSCAN_CLI_INSTALL_ARGS": install_args},
    )
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert f"HOLOSCAN_CLI_INSTALL_ARGS={install_args}" in output


def test_wrapper_test_forwards_holohub_ctest_script_to_container():
    """./holohub test forwards HoloHub's CTest script for container resolution."""
    result = _run_holohub_wrapper(
        "test",
        "endoscopy_tool_tracking",
        "--language",
        "cpp",
        "--dryrun",
        "--no-docker-build",
    )
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "docker run" in output
    assert "-e HOLOSCAN_CLI_CTEST_SCRIPT=utilities/testing/holohub.container.ctest" in output
    assert (
        "-S utilities/testing/holohub.container.ctest" in output
        or "from holoscan_cli.cli import HoloscanCLI" in output
    )
