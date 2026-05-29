#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unit tests for parse_module_sites() and merge_deps() in external_resolver.py.
#
# Pure I/O via temp files; no docker, no network, no GPU. Runs in milliseconds.

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(os.getcwd())))

from utilities.cli.external_resolver import ModuleDep, merge_deps, parse_module_sites

SHA = "0" * 40


def _write_sites(tmpdir: Path, payload: dict) -> Path:
    p = tmpdir / "module-sites.json"
    p.write_text(json.dumps(payload))
    return p


def _make_in_tree(holohub_root: Path, name: str) -> Path:
    d = holohub_root / "modules" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text("{}")
    return d


class TestParseModuleSites(unittest.TestCase):
    def setUp(self):
        self._tmpdir = Path(tempfile.mkdtemp(prefix="holohub_test_sites_"))
        self._saved = {k: v for k, v in os.environ.items() if k.startswith("HOLOHUB_LOCAL_")}
        for k in list(self._saved):
            del os.environ[k]

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ[k] = v
        for child in sorted(self._tmpdir.rglob("*"), reverse=True):
            try:
                child.rmdir() if child.is_dir() else child.unlink()
            except OSError:
                pass
        try:
            self._tmpdir.rmdir()
        except OSError:
            pass

    # ── Basic external entry ────────────────────────────────────────────

    def test_external_entry_parsed(self):
        p = _write_sites(
            self._tmpdir,
            {"modules": [{"name": "holoscan-deltacast", "url": "https://x.git", "ref": SHA,
                          "provides_operators": ["deltacast_videomaster"], "nvidia_quality_score": 2}]},
        )
        deps = parse_module_sites(p)
        self.assertEqual(len(deps), 1)
        self.assertEqual(deps[0].name, "holoscan-deltacast")
        self.assertEqual(deps[0].git_url, "https://x.git")
        self.assertEqual(deps[0].ref, SHA)
        self.assertFalse(deps[0].is_internal)
        self.assertEqual(deps[0].provides_operators, ["deltacast_videomaster"])

    def test_external_entry_no_provides_operators(self):
        p = _write_sites(
            self._tmpdir,
            {"modules": [{"name": "mymod", "url": "https://x.git", "ref": SHA, "nvidia_quality_score": 1}]},
        )
        deps = parse_module_sites(p)
        self.assertEqual(deps[0].provides_operators, [])

    def test_mutable_ref_warns(self):
        p = _write_sites(
            self._tmpdir,
            {"modules": [{"name": "mymod", "url": "https://x.git", "ref": "main", "nvidia_quality_score": 1}]},
        )
        buf = io.StringIO()
        with redirect_stderr(buf):
            deps = parse_module_sites(p)
        self.assertEqual(len(deps), 1)
        self.assertIn("not a 40-char commit SHA", buf.getvalue())

    def test_immutable_sha_no_warn(self):
        p = _write_sites(
            self._tmpdir,
            {"modules": [{"name": "mymod", "url": "https://x.git", "ref": SHA, "nvidia_quality_score": 1}]},
        )
        buf = io.StringIO()
        with redirect_stderr(buf):
            parse_module_sites(p)
        self.assertNotIn("not a 40-char commit SHA", buf.getvalue())

    # ── In-tree entry ───────────────────────────────────────────────────

    def test_in_tree_entry_recognized(self):
        holohub_root = self._tmpdir / "holohub"
        _make_in_tree(holohub_root, "holoscan-gstreamer")
        p = _write_sites(
            self._tmpdir,
            {"modules": [{"name": "holoscan-gstreamer", "nvidia_quality_score": 3}]},
        )
        deps = parse_module_sites(p, holohub_root=holohub_root)
        self.assertEqual(len(deps), 1)
        self.assertTrue(deps[0].is_internal)
        self.assertIsNone(deps[0].git_url)

    def test_no_url_not_in_tree_skipped(self):
        holohub_root = self._tmpdir / "holohub"
        holohub_root.mkdir()
        p = _write_sites(
            self._tmpdir,
            {"modules": [{"name": "ghost-module", "nvidia_quality_score": 1}]},
        )
        deps = parse_module_sites(p, holohub_root=holohub_root)
        self.assertEqual(deps, [])

    def test_no_url_no_holohub_root_skipped(self):
        p = _write_sites(
            self._tmpdir,
            {"modules": [{"name": "holoscan-gstreamer", "nvidia_quality_score": 3}]},
        )
        deps = parse_module_sites(p)  # holohub_root=None
        self.assertEqual(deps, [])

    # ── Missing / malformed file ────────────────────────────────────────

    def test_missing_file_returns_empty(self):
        deps = parse_module_sites(self._tmpdir / "no_such_file.json")
        self.assertEqual(deps, [])

    def test_malformed_json_raises(self):
        bad = self._tmpdir / "bad.json"
        bad.write_text("{not valid json")
        with self.assertRaises(ValueError) as cm:
            parse_module_sites(bad)
        self.assertIn("Malformed JSON", str(cm.exception))

    # ── HOLOHUB_LOCAL_* override ────────────────────────────────────────

    def test_local_override_for_external_entry(self):
        override_dir = self._tmpdir / "local_mod"
        override_dir.mkdir()
        (override_dir / "metadata.json").write_text("{}")
        p = _write_sites(
            self._tmpdir,
            {"modules": [{"name": "mymod", "url": "https://x.git", "ref": SHA, "nvidia_quality_score": 1}]},
        )
        with patch.dict(os.environ, {"HOLOHUB_LOCAL_MYMOD": str(override_dir)}):
            deps = parse_module_sites(p)
        self.assertEqual(deps[0].override_path, override_dir.resolve())

    def test_local_override_for_in_tree_entry_makes_external(self):
        holohub_root = self._tmpdir / "holohub"
        _make_in_tree(holohub_root, "holoscan-gstreamer")
        override_dir = self._tmpdir / "local_gs"
        override_dir.mkdir()
        (override_dir / "metadata.json").write_text("{}")
        p = _write_sites(
            self._tmpdir,
            {"modules": [{"name": "holoscan-gstreamer", "nvidia_quality_score": 3}]},
        )
        with patch.dict(os.environ, {"HOLOHUB_LOCAL_HOLOSCAN_GSTREAMER": str(override_dir)}):
            deps = parse_module_sites(p, holohub_root=holohub_root)
        self.assertEqual(len(deps), 1)
        self.assertFalse(deps[0].is_internal)
        self.assertEqual(deps[0].override_path, override_dir.resolve())

    def test_local_override_missing_metadata_raises(self):
        bad_dir = self._tmpdir / "no_meta"
        bad_dir.mkdir()
        p = _write_sites(
            self._tmpdir,
            {"modules": [{"name": "mymod", "url": "https://x.git", "ref": SHA, "nvidia_quality_score": 1}]},
        )
        with patch.dict(os.environ, {"HOLOHUB_LOCAL_MYMOD": str(bad_dir)}):
            with self.assertRaises(FileNotFoundError):
                parse_module_sites(p)


class TestMergeDeps(unittest.TestCase):
    # ── Sites-only and project-only ─────────────────────────────────────

    def test_sites_only_passthrough(self):
        sites = [ModuleDep(name="modA", git_url="/x", ref=SHA)]
        result = merge_deps(sites, [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "modA")
        self.assertEqual(result[0].git_url, "/x")

    def test_project_only_appended(self):
        project = [ModuleDep(name="modB", git_url="/y", ref=SHA, provides_operators=["op_b"])]
        result = merge_deps([], project)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "modB")

    def test_project_only_goes_after_sites(self):
        sites = [ModuleDep(name="modA", git_url="/x", ref=SHA)]
        project = [ModuleDep(name="modB", git_url="/y", ref=SHA)]
        result = merge_deps(sites, project)
        self.assertEqual([d.name for d in result], ["modA", "modB"])

    # ── Merge: both lists contain the same module ───────────────────────

    def test_merged_takes_site_coords(self):
        sites = [ModuleDep(name="mod", git_url="https://canonical.git", ref=SHA)]
        project = [ModuleDep(name="mod", git_url="https://fork.git", ref="a" * 40, provides_operators=["op"])]
        result = merge_deps(sites, project)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].git_url, "https://canonical.git")
        self.assertEqual(result[0].ref, SHA)

    def test_merged_sites_provides_operators_wins(self):
        sites = [ModuleDep(name="mod", git_url="/x", ref=SHA, provides_operators=["sites_op"])]
        project = [ModuleDep(name="mod", git_url="/x", ref=SHA, provides_operators=["project_op"])]
        result = merge_deps(sites, project)
        self.assertEqual(result[0].provides_operators, ["sites_op"])

    def test_merged_project_provides_operators_fallback_when_sites_empty(self):
        sites = [ModuleDep(name="mod", git_url="/x", ref=SHA)]  # no provides_operators
        project = [ModuleDep(name="mod", git_url="/x", ref=SHA, provides_operators=["my_op"])]
        result = merge_deps(sites, project)
        self.assertEqual(result[0].provides_operators, ["my_op"])

    def test_merged_takes_project_override_path(self):
        override = Path("/local/mod")
        sites = [ModuleDep(name="mod", git_url="/x", ref=SHA)]
        project = [ModuleDep(name="mod", git_url="/x", ref=SHA, override_path=override)]
        result = merge_deps(sites, project)
        self.assertEqual(result[0].override_path, override)

    def test_merged_takes_site_is_internal(self):
        sites = [ModuleDep(name="mod", is_internal=True, override_path=Path("/tree/mod"))]
        project = [ModuleDep(name="mod", provides_operators=["op"])]
        result = merge_deps(sites, project)
        self.assertTrue(result[0].is_internal)

    # ── Order preservation ───────────────────────────────────────────────

    def test_sites_order_preserved_extras_at_end(self):
        sites = [
            ModuleDep(name="alpha", git_url="/a", ref=SHA),
            ModuleDep(name="beta", git_url="/b", ref=SHA),
        ]
        project = [
            ModuleDep(name="beta", git_url="/b", ref=SHA, provides_operators=["b_op"]),
            ModuleDep(name="gamma", git_url="/g", ref=SHA),
        ]
        result = merge_deps(sites, project)
        self.assertEqual([d.name for d in result], ["alpha", "beta", "gamma"])

    # ── Empty inputs ─────────────────────────────────────────────────────

    def test_both_empty(self):
        self.assertEqual(merge_deps([], []), [])


if __name__ == "__main__":
    unittest.main()
