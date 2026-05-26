# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Parser for external Holoscan Module dependencies declared in metadata.json.
# Produces ModuleDep records consumed by cmake_manifest.py (CMake backend) and
# any future resolver backends.
#
# Layered architecture:
#   - Package identity / dependencies: metadata.json:dependencies[] (schema v2).
#   - Workspace materialization: cmake_manifest.py emits FetchContent_Declare
#     entries; CMake does the actual fetch.
#   - Build resolution: CMake. Reads the manifest's FetchContent_Declare entries
#     + HOLOHUB_EXT_OP_<op>_PROVIDER lookup table, and the HoloHub root
#     CMakeLists.txt's post-step calls FetchContent_MakeAvailable for any
#     module whose operators ended up enabled (OP_<x>=ON).

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _ref_is_immutable(ref: str) -> bool:
    """True if ref looks like a full commit SHA (40 hex chars)."""
    return bool(_SHA_RE.match(ref or ""))


@dataclass
class ModuleDep:
    """A parsed module dependency from a consumer's metadata.json.

    `is_internal=True` means the module lives inside the HoloHub tree (under
    modules/<name>/) and requires no FetchContent fetch — its operators are
    already included by HoloHub's normal operators/CMakeLists.txt.
    """

    name: str
    git_url: Optional[str] = None
    ref: Optional[str] = None
    provides_operators: list[str] = field(default_factory=list)
    override_path: Optional[Path] = None
    is_internal: bool = False


def _override_env_name(module_name: str) -> str:
    """Translate a module name into the local-override env-var key.

    holoscan-example-utils -> HOLOHUB_LOCAL_HOLOSCAN_EXAMPLE_UTILS
    """
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", module_name).strip("_").upper()
    return f"HOLOHUB_LOCAL_{sanitized}"


def _read_metadata(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _module_dependencies_raw(metadata: dict) -> list[dict]:
    """Pull the raw module dependency dicts out of either an app or module
    metadata.json. Returns [] if no external deps are declared."""
    if "module" in metadata:
        return list(metadata["module"].get("dependencies", []) or [])
    for key in ("application", "workflow", "benchmark"):
        if key in metadata:
            deps = metadata[key].get("dependencies") or {}
            if isinstance(deps, dict) and "modules" in deps:
                return list(deps["modules"] or [])
    return []


def parse_module_dependencies(
    metadata_path: Path,
    holohub_root: Optional[Path] = None,
) -> list[ModuleDep]:
    """Parse a metadata.json's module dependency list into ModuleDep records.

    Honors HOLOHUB_LOCAL_<NAME> env-var overrides by populating `override_path`.
    When `holohub_root` is provided, dependencies with no `source` block are
    checked against `holohub_root/modules/<name>/` — if a metadata.json exists
    there, the dep is treated as an in-tree module (`is_internal=True`) rather
    than raising an error.

    Does not fetch anything. A missing metadata.json is treated as "no deps"
    rather than an error — unifies the file-doesn't-exist path with the
    file-vanished-between-exists-and-open race window.
    """
    try:
        metadata = _read_metadata(metadata_path)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in {metadata_path}: {e}") from e
    raw = _module_dependencies_raw(metadata)
    out: list[ModuleDep] = []
    for entry in raw:
        name = entry.get("name")
        if not name:
            continue
        source = entry.get("source") or {}
        provides = list(entry.get("provides_operators") or [])

        override = os.environ.get(_override_env_name(name))
        override_path: Optional[Path] = None
        if override:
            p = Path(override).expanduser().resolve()
            if not (p / "metadata.json").exists():
                raise FileNotFoundError(
                    f"{_override_env_name(name)}={override} does not contain a "
                    "metadata.json — point it at the root of a Holoscan Module "
                    "project."
                )
            override_path = p

        ref = source.get("ref")
        git_url = source.get("git_url")
        if override_path is None:
            if not (git_url and ref):
                # Check if this is an in-tree module before raising.
                if holohub_root is not None:
                    in_tree_path = holohub_root / "modules" / name
                    if (in_tree_path / "metadata.json").exists():
                        out.append(
                            ModuleDep(
                                name=name,
                                provides_operators=provides,
                                is_internal=True,
                                override_path=in_tree_path,
                            )
                        )
                        continue
                raise ValueError(
                    f"Dependency '{name}' missing source.git_url or source.ref. "
                    f"Declare a complete source block, set "
                    f"{_override_env_name(name)}=<path>, or add a module descriptor "
                    f"at modules/{name}/metadata.json for in-tree modules."
                )
            if not _ref_is_immutable(ref):
                import sys as _sys

                print(
                    f"WARNING: dependency '{name}' pinned to ref '{ref}', which "
                    "is not a 40-char commit SHA. Tags and branches are mutable; "
                    "consider pinning to an immutable SHA for reproducible builds.",
                    file=_sys.stderr,
                )

        out.append(
            ModuleDep(
                name=name,
                git_url=git_url,
                ref=ref,
                provides_operators=provides,
                override_path=override_path,
            )
        )
    return out
