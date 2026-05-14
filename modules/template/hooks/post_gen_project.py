#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Post-generation hook: clean up language-specific files and initialise git."""

import os
import pathlib
import shutil
import subprocess
import warnings

LANGUAGE = "{{ cookiecutter.language }}"
MODULE_SLUG = "{{ cookiecutter.module_slug }}"
MODULE_REPO_NAME = "{{ cookiecutter.module_repo_name }}"
OPERATOR_SLUG = "{{ cookiecutter.operator_slug }}"


def remove_paths(*paths: str) -> None:
    for p in paths:
        if os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)


def remove_empty_dirs(root: str = ".") -> None:
    """Bottom-up removal of directories left empty by conditional filenames."""
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        if dirpath == root:
            continue
        if not os.listdir(dirpath):
            os.rmdir(dirpath)


# For Python-only modules, remove directories that only make sense for C++.
if LANGUAGE == "python":
    remove_paths("tests/cpp", ".clang-format")

# Remove any directories that became empty (from Jinja2 conditional filenames).
remove_empty_dirs()

# Locate holohub-internal so we can copy in companion files. The cookiecutter
# runs this hook from a temp file, so __file__ is unreliable. The HoloHub CLI
# sets HOLOHUB_ROOT before invoking cookiecutter; standalone cookiecutter use
# falls back to a sibling-directory search.
_holohub_root = None
_env_root = os.environ.get("HOLOHUB_ROOT")
_candidates = (
    pathlib.Path.cwd().parent / "holohub-internal",
    pathlib.Path.cwd().parent.parent / "holohub-internal",
    pathlib.Path.cwd().parent / "holohub",
    pathlib.Path.cwd().parent.parent / "holohub",
)
if _env_root and (pathlib.Path(_env_root) / "utilities" / "metadata").is_dir():
    _holohub_root = pathlib.Path(_env_root)
else:
    for candidate in _candidates:
        if (candidate / "utilities" / "metadata").is_dir():
            _holohub_root = candidate.resolve()
            break

if _holohub_root is None:
    _candidate_list = ", ".join(str(c) for c in _candidates)
    warnings.warn(
        f"HoloHub root not found. CMake helpers were not copied into cmake/.\n"
        f"  HOLOHUB_ROOT env var: {_env_root!r}\n"
        f"  cwd: {pathlib.Path.cwd()}\n"
        f"  Candidates checked: {_candidate_list}\n"
        f"To fix: set HOLOHUB_ROOT=/path/to/holohub before running cookiecutter,\n"
        f"  or rename your HoloHub clone to 'holohub' or 'holohub-internal' as a\n"
        f"  sibling of the generated module directory.",
        stacklevel=1,
    )

# Copy CMake helpers from the HoloHub clone so the module builds standalone:
#   - HoloHubConfigHelpers.cmake: add_holohub_application/operator/package gating
#   - holohub_configure_deb.cmake: deb packaging helper used by the root CMakeLists
#   - pybind11_add_holohub_module.cmake: fetches pybind11 at the HSDK-pinned
#     version + ABI-aligned target. Brings two companion assets:
#       - pybind11/__init__.py: per-operator __init__ template configured by
#         the helper (provides ABI error diagnostics)
#       - pydoc/macros.hpp: docstring macros referenced by pybind sources
if _holohub_root:
    _cmake_dst = pathlib.Path("cmake")
    _cmake_dst.mkdir(exist_ok=True)
    # Single-file helpers
    for _rel in (
        ("cmake", "HoloHubConfigHelpers.cmake"),
        ("cmake", "modules", "holohub_configure_deb.cmake"),
        ("cmake", "pybind11_add_holohub_module.cmake"),
    ):
        _src = _holohub_root.joinpath(*_rel)
        if _src.exists():
            shutil.copy2(_src, _cmake_dst / _src.name)
    # Companion directories the pybind11 helper expects alongside itself
    for _subdir in ("pybind11", "pydoc"):
        _src_dir = _holohub_root / "cmake" / _subdir
        if _src_dir.is_dir():
            _dst_dir = _cmake_dst / _subdir
            if _dst_dir.exists():
                shutil.rmtree(_dst_dir)
            shutil.copytree(_src_dir, _dst_dir)

# Make the module CLI wrapper executable.
wrapper = "./holohub"
if os.path.isfile(wrapper):
    os.chmod(wrapper, 0o755)

# Initialise a git repository so the module is ready to push. Pin the
# initial branch to `main` independently of the user's global
# init.defaultBranch — the next-steps message below assumes `main`,
# and `git symbolic-ref` works on any git version (including pre-2.28
# where `git init -b` isn't supported) by relabelling HEAD before any
# commit creates the branch ref.
git_ok = False
try:
    subprocess.run(["git", "init", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "symbolic-ref", "HEAD", "refs/heads/main"],
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], check=True, capture_output=True)
    git_ok = True
except (subprocess.CalledProcessError, FileNotFoundError):
    pass

# ── Next-steps message ────────────────────────────────────────────────────────
op_parts = OPERATOR_SLUG.split("_")
OPERATOR_CLASS = "".join(p.capitalize() for p in op_parts)
pipeline = f"{MODULE_SLUG}_pipeline"

print(f"\n\033[32mHoloscan Module '{MODULE_SLUG}' created successfully!\033[0m\n")
print(f"Implement your operator ({OPERATOR_CLASS}) in:")
if LANGUAGE == "cpp":
    print(f"  operators/{OPERATOR_SLUG}/{OPERATOR_SLUG}.cpp\n")
else:
    print(f"  operators/{OPERATOR_SLUG}/{OPERATOR_SLUG}.py\n")

print("Build and run:")
print(f"  ./holohub run-container")
print("  # Inside the container:")
print(f"  ./holohub build {pipeline}")
print(f"  ./holohub run   {pipeline} --language python\n")

if git_ok:
    print("Git repository initialised. Push to a remote when ready:")
    print("  git remote add origin <your-repo-url>")
    print("  git push -u origin main\n")
else:
    print("Note: could not run git init — initialise the repository manually.\n")
print("Register your module at https://nvidia-holoscan.github.io/ when ready.")
