#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal test that application modules can be imported (no GPU or data required).
# If holoscan/cupy are not available (e.g. host without container), skip and exit 0.
"""Test that gsplat_scene_recon and dependencies can be imported."""

import importlib.util
import sys
from pathlib import Path

# Ensure app root is on path when run from application directory
app_dir = Path(__file__).resolve().parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))


def main():
    try:
        import holoscan  # noqa: F401
    except ModuleNotFoundError as e:
        if "holoscan" in str(e).lower():
            print("SKIP: holoscan not available (run inside container)")
            return 0
        raise
    if importlib.util.find_spec("cupy") is None:
        print("SKIP: cupy not available (run inside container)")
        return 0

    # Import application modules
    import gsplat_scene_recon  # noqa: F401
    from utils import progress_monitor  # noqa: F401
    import run_gsharp  # noqa: F401

    print("SUCCESS: imports OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
