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


def _holoscan_cli_env() -> Dict[str, str]:
    env = os.environ.copy()
    source = env.get("HOLOSCAN_CLI_SOURCE")
    if source:
        src_path = str(Path(source).expanduser().resolve() / "src")
        env["PYTHONPATH"] = f"{src_path}:{env.get('PYTHONPATH', '')}".rstrip(":")
    return env


def _run_holoscan_cli(*args: str) -> subprocess.CompletedProcess:
    env = _holoscan_cli_env()
    result = subprocess.run(
        [sys.executable, "-m", "holoscan_cli", *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 and "No module named holoscan_cli" in result.stderr:
        pytest.skip("holoscan_cli is not installed; set HOLOSCAN_CLI_SOURCE to a checkout")
    return result


def test_unified_cli_lists_holohub_projects():
    result = _run_holoscan_cli("list")

    assert result.returncode == 0, result.stderr
    assert "== APPLICATIONS" in result.stdout
    assert "endoscopy_tool_tracking" in result.stdout


def test_unified_cli_dryruns_holohub_project_run():
    result = _run_holoscan_cli("run", "endoscopy_tool_tracking", "--dryrun")
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "Running endoscopy_tool_tracking" in output
    assert "docker run" in output
    assert "holoscan run endoscopy_tool_tracking" in output


def _run_holohub_wrapper(*args: str) -> subprocess.CompletedProcess:
    env = _holoscan_cli_env()
    result = subprocess.run(
        [str(REPO_ROOT / "holohub"), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 and "No module named holoscan_cli" in result.stderr:
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
    result = _run_holohub_wrapper("run", "endoscopy_tool_tracking", "--dryrun")
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "Running endoscopy_tool_tracking" in output
    assert "docker run" in output
    assert "holoscan run endoscopy_tool_tracking" in output
