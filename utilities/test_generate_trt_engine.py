# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the conversion launcher against a small executable instead of TensorRT."""

import json
import os
import signal
import sys

import pytest

from utilities.generate_trt_engine import convert_onnx


@pytest.fixture
def trtexec(tmp_path, monkeypatch):
    executable = tmp_path / "trtexec"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['TRTEXEC_ARGUMENTS']).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.exit(int(os.environ.get('TRTEXEC_EXIT_CODE', '0')))\n"
    )
    executable.chmod(0o700)
    arguments = tmp_path / "arguments.json"
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("TRTEXEC_ARGUMENTS", str(arguments))
    monkeypatch.chdir(tmp_path)
    return arguments


@pytest.mark.parametrize("field", ["input", "output"])
def test_paths_cannot_execute_shell_commands(field, trtexec):
    payload = "model'; printf injected > INJECTION_MARKER; #"
    input_file = payload if field == "input" else "normal input.onnx"
    output_file = payload if field == "output" else "normal output.engine"

    assert convert_onnx(input_file, output_file, False) == 0
    assert not trtexec.with_name("INJECTION_MARKER").exists()
    assert json.loads(trtexec.read_text()) == [
        f"--onnx={input_file}",
        f"--saveEngine={output_file}",
    ]


@pytest.mark.parametrize("fp16", [False, True])
def test_paths_and_fp16_are_forwarded_as_arguments(fp16, trtexec):
    input_file = "model's $(echo ignored); input.onnx"
    output_file = "output's name.engine"
    assert convert_onnx(input_file, output_file, fp16) == 0
    expected = [f"--onnx={input_file}", f"--saveEngine={output_file}"]
    if fp16:
        expected.append("--fp16")
    assert json.loads(trtexec.read_text()) == expected


def test_conversion_failure_preserves_exit_code(trtexec, monkeypatch):
    monkeypatch.setenv("TRTEXEC_EXIT_CODE", "19")
    assert convert_onnx("input.onnx", "output.engine", False) == 19


def test_missing_trtexec_preserves_command_not_found_status(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert convert_onnx("input.onnx", "output.engine", False) == 127


def test_signal_failure_preserves_shell_exit_status(trtexec):
    executable = trtexec.with_name("trtexec")
    executable.write_text(
        f"#!{sys.executable}\nimport os, signal\nos.kill(os.getpid(), signal.SIGTERM)\n"
    )
    assert convert_onnx("input.onnx", "output.engine", False) == 128 + signal.SIGTERM


def test_nonexecutable_converter_returns_failure(trtexec, monkeypatch):
    trtexec.with_name("trtexec").chmod(0o600)
    monkeypatch.setenv("PATH", str(trtexec.parent))
    assert convert_onnx("input.onnx", "output.engine", False) == 126


def test_invalid_executable_format_returns_failure(trtexec):
    trtexec.with_name("trtexec").write_bytes(b"\x00not an executable")
    assert convert_onnx("input.onnx", "output.engine", False) == 126
