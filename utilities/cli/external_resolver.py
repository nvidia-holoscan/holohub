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
import sys
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
    env: Optional[dict] = None,
) -> list[ModuleDep]:
    """Parse a metadata.json's module dependency list into ModuleDep records.

    Honors HOLOHUB_LOCAL_<NAME> env-var overrides by populating `override_path`.
    Pass `env` to override the process environment for override lookups (defaults
    to os.environ).  When `holohub_root` is provided, dependencies with no
    `source` block are checked against `holohub_root/modules/<name>/` — if a
    metadata.json exists there, the dep is treated as an in-tree module
    (`is_internal=True`) rather than raising an error.

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
    _env = env if env is not None else os.environ
    raw = _module_dependencies_raw(metadata)
    out: list[ModuleDep] = []
    for entry in raw:
        name = entry.get("name")
        if not name:
            continue
        source = entry.get("source") or {}
        provides = list(entry.get("provides_operators") or [])

        override = _env.get(_override_env_name(name))
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
                print(
                    f"WARNING: dependency '{name}' pinned to ref '{ref}', which "
                    "is not a 40-char commit SHA. Tags and branches are mutable; "
                    "consider pinning to an immutable SHA for reproducible builds.",
                    file=sys.stderr,
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


def parse_module_sites(
    sites_path: Path,
    holohub_root: Optional[Path] = None,
    env: Optional[dict] = None,
) -> list[ModuleDep]:
    """Parse modules/module-sites.json into ModuleDep records.

    External entries (url + ref present) become fetchable deps; `provides_operators`
    is read directly from the site entry and is authoritative. Project metadata serves
    only as a fallback via merge_deps().  In-tree entries (no url) resolve to
    is_internal=True when holohub_root/modules/<name>/metadata.json exists;
    entries with neither a url nor an in-tree path are silently skipped.

    An entry with url but no ref (or ref but no url) raises ValueError — partial
    source specs hide typos and should fail loudly.

    Honors HOLOHUB_LOCAL_<NAME> overrides with the same semantics as
    parse_module_dependencies.  Pass `env` to override the process environment
    for override lookups (defaults to os.environ).  A missing sites_path is
    treated as no module sites rather than an error.
    """
    try:
        with sites_path.open() as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in {sites_path}: {e}") from e

    _env = env if env is not None else os.environ
    out: list[ModuleDep] = []
    for entry in data.get("modules") or []:
        name = entry.get("name")
        if not name:
            continue

        override = _env.get(_override_env_name(name))
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

        url = entry.get("url")
        ref = entry.get("ref")

        provides = list(entry.get("provides_operators") or [])

        if bool(url) != bool(ref):
            raise ValueError(
                f"module-sites entry '{name}' must declare both 'url' and 'ref' "
                "together, or neither for an in-tree/local-only module."
            )

        if url and ref:
            if not _ref_is_immutable(ref):
                print(
                    f"WARNING: module-sites entry '{name}' pinned to ref '{ref}', which "
                    "is not a 40-char commit SHA. Tags and branches are mutable; "
                    "consider pinning to an immutable SHA for reproducible builds.",
                    file=sys.stderr,
                )
            out.append(
                ModuleDep(
                    name=name,
                    git_url=url,
                    ref=ref,
                    provides_operators=provides,
                    override_path=override_path,
                )
            )
        elif override_path is not None:
            # No canonical url but a local override is active — treat as external.
            out.append(
                ModuleDep(name=name, provides_operators=provides, override_path=override_path)
            )
        elif holohub_root is not None:
            in_tree_path = holohub_root / "modules" / name
            if (in_tree_path / "metadata.json").exists():
                out.append(
                    ModuleDep(
                        name=name,
                        provides_operators=provides,
                        override_path=in_tree_path,
                        is_internal=True,
                    )
                )
            # else: not external and not in-tree — skip

    return out


def merge_deps(
    sites_deps: list[ModuleDep],
    project_deps: list[ModuleDep],
) -> list[ModuleDep]:
    """Merge module-sites deps with project-specific deps.

    Sites supply canonical git coordinates and own `provides_operators`
    authoritatively; project deps provide `override_path` and serve as a
    fallback source for `provides_operators` when the site entry has none.
    For a module present in both lists the merged record takes the site's
    git_url/ref and is_internal classification, but the project dep's
    override_path.  Modules only in project_deps are appended after all site
    entries (preserving sites order first).
    """
    project_by_name = {d.name: d for d in project_deps}
    seen: set[str] = set()
    result: list[ModuleDep] = []

    for sd in sites_deps:
        pd = project_by_name.get(sd.name)
        if pd is not None:
            # Sites owns the canonical git coords and is_internal classification.
            # Sites also owns provides_operators (authoritative module metadata);
            # project ops are used only as a fallback when sites has none.
            ops = sd.provides_operators if sd.provides_operators else pd.provides_operators
            result.append(
                ModuleDep(
                    name=sd.name,
                    git_url=sd.git_url,
                    ref=sd.ref,
                    provides_operators=ops,
                    override_path=pd.override_path,
                    is_internal=sd.is_internal,
                )
            )
        else:
            result.append(sd)
        seen.add(sd.name)

    for pd in project_deps:
        if pd.name not in seen:
            result.append(pd)

    return result
