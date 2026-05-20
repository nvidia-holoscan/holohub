# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shallow git clone helper for fetching external Holoscan Module content at website build time.

Distinct from utilities/cli/external_resolver.py, which parses module dependency declarations
into CMake FetchContent records without performing any network operations. This module does
the actual git checkout so the website generator can read remote README and metadata files.
"""

import subprocess
import tempfile
from pathlib import Path


def clone_external_module(url: str, ref: str) -> tuple[Path, tempfile.TemporaryDirectory]:
    """Shallow-clone url@ref into a temp dir.

    Returns (clone_path, tmp_dir). Caller must call tmp_dir.cleanup() when done.
    Raises subprocess.CalledProcessError on failure — caller is responsible for
    catching and skipping the module gracefully.
    """
    tmp = tempfile.TemporaryDirectory(prefix="holohub_module_")
    subprocess.run(
        ["git", "clone", "--depth=1", "--branch", ref, url, tmp.name],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return Path(tmp.name), tmp
