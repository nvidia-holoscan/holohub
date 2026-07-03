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
# Root in a container resolves to system Python before the branches these
# tests exercise.
requires_nonroot = pytest.mark.skipif(
    os.geteuid() == 0, reason="resolver branch tests assume a non-root invocation"
)


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
    cwd: Path = REPO_ROOT,
) -> subprocess.CompletedProcess:
    if env is None:
        env = _holoscan_cli_env()
    if extra_env:
        env = {**env, **extra_env}
    try:
        result = subprocess.run(
            [str(REPO_ROOT / "holohub"), *args],
            cwd=cwd,
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


def _fake_python_for_wrapper(
    tmp_path: Path, installed_version: str, executable_name: str = "python3"
):
    """Return a fake interpreter and its mutable package-version/log files.

    The intercepted argv shapes must mirror the wrapper's `_holoscan_cli_ok`
    probe and `_install_cli`; update both together. Setting FAKE_PYTHON_PIP_NOOP
    makes `pip install` succeed without changing the installed version, which
    models an unsatisfiable requirement (e.g. a pin no index can provide).
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    version_file = tmp_path / "installed-version.txt"
    log_file = tmp_path / "fake-python.log"
    version_file.write_text(installed_version, encoding="utf-8")
    fake_python = tmp_path / executable_name
    fake_python.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "${FAKE_PYTHON_LOG}"

if [[ "${1:-}" == "-c" ]]; then
    cat "${FAKE_PYTHON_VERSION_FILE}"
elif [[ "${1:-}" == "-m" && "${2:-}" == "holoscan_cli" && "${3:-}" == "build" && "${4:-}" == "--help" ]]; then
    :
elif [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "--version" ]]; then
    echo "pip 1.0"
elif [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "install" ]]; then
    if [[ -z "${FAKE_PYTHON_PIP_NOOP:-}" ]]; then
        for arg in "$@"; do
            if [[ "$arg" == holoscan-cli==* ]]; then
                printf '%s' "${arg#holoscan-cli==}" > "${FAKE_PYTHON_VERSION_FILE}"
            fi
        done
    fi
elif [[ "${1:-}" == "-m" && "${2:-}" == "holoscan_cli" ]]; then
    :
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return fake_python, version_file, log_file


def _fake_wrapper_env(fake_python: Path, version_file: Path, log_file: Path):
    env = os.environ.copy()
    for var in (
        "HOLOSCAN_CLI_SOURCE",
        "HOLOSCAN_CLI_PINNED_VERSION",
        "PYTHONPATH",
        "PIP_BREAK_SYSTEM_PACKAGES",
        "VIRTUAL_ENV",
        "HOLOHUB_PYTHON_BIN",
    ):
        env.pop(var, None)
    env["HOLOSCAN_CLI_PYTHON_BIN"] = str(fake_python)
    env["FAKE_PYTHON_VERSION_FILE"] = str(version_file)
    env["FAKE_PYTHON_LOG"] = str(log_file)
    env["HOLOSCAN_CLI_INSTALL_ARGS"] = (
        "--pre --extra-index-url https://example.invalid/simple holoscan-cli>4.2.0"
    )
    return env


def _fake_system_cli_env(tmp_path: Path, version="4.3.0", managed_python=None):
    system_bin = tmp_path / "system-bin"
    system_bin.mkdir()
    system_log = tmp_path / "system-python.log"
    system_pip_log = tmp_path / "system-pip.log"
    managed_venv = tmp_path / "managed-venv"
    fake_python = system_bin / "python3"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "${FAKE_SYSTEM_LOG}"

if [[ "${1:-}" == "-c" ]]; then
    printf '%s' "${FAKE_SYSTEM_VERSION}"
elif [[ "${1:-}" == "-m" && "${2:-}" == "holoscan_cli" && "${3:-}" == "build" && "${4:-}" == "--help" ]]; then
    :
elif [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "--version" ]]; then
    echo "pip 1.0"
elif [[ "${1:-}" == "-m" && "${2:-}" == "pip" && "${3:-}" == "install" ]]; then
    touch "${FAKE_SYSTEM_PIP_LOG}"
elif [[ "${1:-}" == "-m" && "${2:-}" == "venv" ]]; then
    if [[ -n "${FAKE_VENV_FAIL_ONCE_FILE:-}" && ! -e "${FAKE_VENV_FAIL_ONCE_FILE}" ]]; then
        touch "${FAKE_VENV_FAIL_ONCE_FILE}"
        mkdir -p "${3}/bin"
        printf '#!/usr/bin/env bash\nexit 1\n' > "${3}/bin/python"
        chmod +x "${3}/bin/python"
        exit 1
    fi
    mkdir -p "${3}/bin"
    cp "${FAKE_MANAGED_PYTHON_SOURCE}" "${3}/bin/python"
elif [[ "${1:-}" == "-m" && "${2:-}" == "holoscan_cli" ]]; then
    :
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    for var in (
        "HOLOSCAN_CLI_SOURCE",
        "HOLOSCAN_CLI_PINNED_VERSION",
        "HOLOSCAN_CLI_PYTHON_BIN",
        "HOLOHUB_PYTHON_BIN",
        "PIP_BREAK_SYSTEM_PACKAGES",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    ):
        env.pop(var, None)
    env["PATH"] = f"{system_bin}:{env['PATH']}"
    env["SUDO_USER"] = "host-user"
    env["HOLOSCAN_CLI_VENV"] = str(managed_venv)
    env["HOLOSCAN_CLI_INSTALL_ARGS"] = "holoscan-cli>4.2.0"
    env["FAKE_SYSTEM_LOG"] = str(system_log)
    env["FAKE_SYSTEM_PIP_LOG"] = str(system_pip_log)
    env["FAKE_SYSTEM_VERSION"] = version
    env["FAKE_MANAGED_PYTHON_SOURCE"] = str(managed_python or tmp_path / "unused")
    return env, managed_venv, system_log, system_pip_log


def test_wrapper_delegates_list_to_holoscan_cli():
    """./holohub list should hit the consolidated CLI's source-project dispatch."""
    result = _run_holohub_wrapper("list")

    assert result.returncode == 0, result.stderr
    assert "== APPLICATIONS" in result.stdout
    assert "endoscopy_tool_tracking" in result.stdout


def test_wrapper_reconciles_an_explicit_cli_version_pin(tmp_path):
    fake_python, version_file, log_file = _fake_python_for_wrapper(tmp_path, "4.3.0")
    env = _fake_wrapper_env(fake_python, version_file, log_file)
    env["HOLOSCAN_CLI_PINNED_VERSION"] = "4.4.1"

    result = _run_holohub_wrapper("version", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert version_file.read_text(encoding="utf-8") == "4.4.1"
    log = log_file.read_text(encoding="utf-8")
    assert "-m pip install" in log
    assert "holoscan-cli>4.2.0 holoscan-cli==4.4.1" in log


def test_wrapper_fails_clearly_when_a_pin_cannot_be_satisfied(tmp_path):
    fake_python, version_file, log_file = _fake_python_for_wrapper(tmp_path, "4.3.0")
    env = _fake_wrapper_env(fake_python, version_file, log_file)
    env["HOLOSCAN_CLI_PINNED_VERSION"] = "4.4.1"
    env["FAKE_PYTHON_PIP_NOOP"] = "1"

    result = _run_holohub_wrapper("version", env=env)

    assert result.returncode == 1
    assert "does not match" in result.stderr
    assert "Could not install holoscan-cli" in result.stderr
    # pip ran ("already satisfied" model) but must not have changed anything.
    assert version_file.read_text(encoding="utf-8") == "4.3.0"


def test_wrapper_does_not_expand_install_argument_globs(tmp_path):
    fake_python, version_file, log_file = _fake_python_for_wrapper(tmp_path / "fake-bin", "4.3.0")
    env = _fake_wrapper_env(fake_python, version_file, log_file)
    env["HOLOSCAN_CLI_PINNED_VERSION"] = "4.4.1"
    env["HOLOSCAN_CLI_INSTALL_ARGS"] = "[abc] holoscan-cli>4.2.0"
    (tmp_path / "a").touch()

    result = _run_holohub_wrapper("version", env=env, cwd=tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    log = log_file.read_text(encoding="utf-8")
    assert "-m pip install [abc] holoscan-cli>4.2.0" in log
    assert "-m pip install a holoscan-cli>4.2.0" not in log


@pytest.mark.parametrize("pin", [None, "4.3.0"])
def test_wrapper_does_not_reinstall_a_compatible_cli(tmp_path, pin):
    fake_python, version_file, log_file = _fake_python_for_wrapper(tmp_path, "4.3.0")
    env = _fake_wrapper_env(fake_python, version_file, log_file)
    if pin is None:
        env.pop("HOLOSCAN_CLI_PINNED_VERSION", None)
    else:
        env["HOLOSCAN_CLI_PINNED_VERSION"] = pin

    result = _run_holohub_wrapper("version", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "-m pip install" not in log_file.read_text(encoding="utf-8")


@requires_nonroot
def test_wrapper_prefers_an_existing_managed_venv(tmp_path):
    managed_venv = tmp_path / "managed-venv"
    managed_python, version_file, log_file = _fake_python_for_wrapper(
        managed_venv / "bin", "4.3.0", executable_name="python"
    )
    global_dir = tmp_path / "global-bin"
    global_dir.mkdir()
    global_log = tmp_path / "global-python.log"
    global_python = global_dir / "python3"
    global_python.write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "${FAKE_GLOBAL_PYTHON_LOG}"
[[ "${1:-}" == "-m" && "${2:-}" == "holoscan_cli" ]] && echo build
""",
        encoding="utf-8",
    )
    global_python.chmod(0o755)

    env = _fake_wrapper_env(managed_python, version_file, log_file)
    env.pop("HOLOSCAN_CLI_PYTHON_BIN")
    env["HOLOSCAN_CLI_VENV"] = str(managed_venv)
    env["FAKE_GLOBAL_PYTHON_LOG"] = str(global_log)
    env["PATH"] = f"{global_dir}:{env['PATH']}"
    # Also makes this model the isolated sudo-host path when CI itself is root.
    env["SUDO_USER"] = "host-user"

    result = _run_holohub_wrapper("version", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "-m holoscan_cli version" in log_file.read_text(encoding="utf-8")
    assert not global_log.exists()


@requires_nonroot
@pytest.mark.parametrize("pin", [None, "4.3.0"])
def test_wrapper_reuses_a_compatible_system_cli_readonly(tmp_path, pin):
    env, managed_venv, system_log, system_pip_log = _fake_system_cli_env(tmp_path)
    if pin:
        env["HOLOSCAN_CLI_PINNED_VERSION"] = pin

    result = _run_holohub_wrapper("version", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "-m holoscan_cli version" in system_log.read_text(encoding="utf-8")
    assert not managed_venv.exists()
    assert not system_pip_log.exists()


@requires_nonroot
def test_wrapper_uses_managed_venv_for_a_mismatched_system_pin(tmp_path):
    managed_python, version_file, managed_log = _fake_python_for_wrapper(
        tmp_path / "managed-template", "4.3.0", executable_name="python"
    )
    env, managed_venv, _, system_pip_log = _fake_system_cli_env(
        tmp_path, managed_python=managed_python
    )
    env["HOLOSCAN_CLI_PINNED_VERSION"] = "4.4.1"
    env["FAKE_PYTHON_VERSION_FILE"] = str(version_file)
    env["FAKE_PYTHON_LOG"] = str(managed_log)

    result = _run_holohub_wrapper("version", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (managed_venv / "bin/python").is_file()
    assert version_file.read_text(encoding="utf-8") == "4.4.1"
    assert "-m pip install" in managed_log.read_text(encoding="utf-8")
    assert not system_pip_log.exists()


@requires_nonroot
def test_wrapper_retries_after_a_partial_venv_failure(tmp_path):
    managed_python, version_file, managed_log = _fake_python_for_wrapper(
        tmp_path / "managed-template", "4.3.0", executable_name="python"
    )
    env, managed_venv, system_log, system_pip_log = _fake_system_cli_env(
        tmp_path, managed_python=managed_python
    )
    failure_marker = tmp_path / "failed-once"
    env["HOLOSCAN_CLI_PINNED_VERSION"] = "4.4.1"
    env["FAKE_PYTHON_VERSION_FILE"] = str(version_file)
    env["FAKE_PYTHON_LOG"] = str(managed_log)
    env["FAKE_VENV_FAIL_ONCE_FILE"] = str(failure_marker)

    first = _run_holohub_wrapper("version", env=env)
    second = _run_holohub_wrapper("version", env=env)

    assert first.returncode == 1
    assert "Could not create" in first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert system_log.read_text(encoding="utf-8").count("-m venv") == 2
    assert (managed_venv / "bin/python").is_file()
    assert version_file.read_text(encoding="utf-8") == "4.4.1"
    assert not system_pip_log.exists()


@requires_nonroot
def test_wrapper_works_without_home_when_system_cli_is_compatible(tmp_path):
    """Personas without HOME (minimal images, USER stages) must not die on the
    managed-venv default expansion when they never need the venv."""
    env, managed_venv, system_log, system_pip_log = _fake_system_cli_env(tmp_path)
    for var in ("HOME", "XDG_DATA_HOME", "HOLOSCAN_CLI_VENV"):
        env.pop(var, None)

    result = _run_holohub_wrapper("version", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "-m holoscan_cli version" in system_log.read_text(encoding="utf-8")
    assert not managed_venv.exists()
    assert not system_pip_log.exists()


def test_wrapper_rejects_an_invalid_source_override(tmp_path):
    env = os.environ.copy()
    env["HOLOSCAN_CLI_SOURCE"] = str(tmp_path / "missing-checkout")

    result = _run_holohub_wrapper("version", env=env)

    assert result.returncode == 1
    assert "Invalid HOLOSCAN_CLI_SOURCE" in result.stderr


def test_wrapper_runs_host_setup_from_managed_venv(tmp_path):
    """A source override and setup both run inside the managed host venv."""
    source = os.environ.get("HOLOSCAN_CLI_SOURCE")
    if not source:
        pytest.skip("set HOLOSCAN_CLI_SOURCE to test managed-venv bootstrap")

    env = os.environ.copy()
    # Keep this test focused on the wrapper's source handling: an inherited
    # PYTHONPATH could resolve a different checkout, and an inherited
    # VIRTUAL_ENV / explicit interpreter override would take the wrapper's
    # caller-owned branch instead of creating the managed venv.
    for var in ("PYTHONPATH", "VIRTUAL_ENV", "HOLOSCAN_CLI_PYTHON_BIN", "HOLOHUB_PYTHON_BIN"):
        env.pop(var, None)
    env["HOLOSCAN_CLI_SOURCE"] = source
    env["HOLOSCAN_CLI_VENV"] = str(tmp_path / "holoscan-cli-venv")
    result = _run_holohub_wrapper("env-info", env=env)

    assert result.returncode == 0, result.stderr
    assert f"Executable: {env['HOLOSCAN_CLI_VENV']}/bin/python" in result.stdout

    # Assert only the wrapper contract: the default setup dry-runs through the
    # managed venv. Per-step behavior (apt, ngc/sccache destinations) belongs
    # to holoscan-cli's own test suites -- asserting it here would couple this
    # repo's CI to unreleased holoscan-cli changes.
    setup_result = _run_holohub_wrapper("setup", "--dryrun", env=env)
    assert setup_result.returncode == 0, setup_result.stdout + setup_result.stderr


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
