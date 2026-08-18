#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Export yolo11l-pose.pt to ONNX without loading SAM/FastSAM/NAS/RTDETR. Those
# subpackages import torchvision; on stacks where torch and torchvision are ABI-
# mismatched, import fails with: RuntimeError: operator torchvision::nms does not exist.
# Only the YOLO pose stack is needed for this export.

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import types


def _ensure_onnx_export_dependencies() -> None:
    """Install ONNX export deps before importing ultralytics (avoids pip AutoUpdate warnings)."""
    missing: list[str] = []
    try:
        import onnxslim  # noqa: F401
    except ImportError:
        missing.append("onnxslim>=0.1.71")
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        missing.append("onnxruntime")
    if not missing:
        return
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            "--disable-pip-version-check",
            *missing,
        ],
        check=True,
        env={
            **os.environ,
            "PIP_ROOT_USER_ACTION": "ignore",
            "PIP_NO_CACHE_DIR": "1",
        },
    )


def _stub_ultralytics_optional_models() -> None:
    """Register minimal modules before ultralytics.models is imported."""
    stubs: list[tuple[str, str]] = [
        ("ultralytics.models.fastsam", "FastSAM"),
        ("ultralytics.models.nas", "NAS"),
        ("ultralytics.models.rtdetr", "RTDETR"),
        ("ultralytics.models.sam", "SAM"),
    ]
    for mod_name, cls_name in stubs:
        if mod_name in sys.modules:
            continue
        mod = types.ModuleType(mod_name)
        setattr(mod, cls_name, type(cls_name, (), {}))
        sys.modules[mod_name] = mod


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Ultralytics YOLO pose weights to ONNX.")
    parser.add_argument(
        "--model",
        default="yolo11l-pose.pt",
        help="Checkpoint name or path (default: yolo11l-pose.pt)",
    )
    parser.add_argument("--opset", type=int, default=20, help="ONNX opset (default: 20)")
    parser.add_argument(
        "--ultralytics-config-dir",
        default=None,
        help="Writable parent directory for Ultralytics settings (default: ./ultralytics_user under cwd).",
    )
    args = parser.parse_args()

    # ONNX Runtime: suppress GPU device discovery warnings in headless / DRM-less environments.
    os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
    # Pip: avoid unwritable default cache (e.g. repo .cache/pip in containers).
    os.environ.setdefault("PIP_NO_CACHE_DIR", "1")

    config_parent = (
        os.path.abspath(args.ultralytics_config_dir)
        if args.ultralytics_config_dir
        else os.path.join(os.getcwd(), "ultralytics_user")
    )
    os.makedirs(config_parent, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = config_parent

    _ensure_onnx_export_dependencies()

    _stub_ultralytics_optional_models()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.export(format="onnx", opset=args.opset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
