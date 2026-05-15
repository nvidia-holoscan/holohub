#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unit tests for utilities/cli/external_resolver.py.
#
# Covers:
#   - parse_module_dependencies: reads metadata.json:dependencies.modules from
#     application/workflow/benchmark/module shapes; honors HOLOHUB_LOCAL_<NAME>;
#     skips malformed entries cleanly.
#   - String helpers: _override_env_name, _ref_is_immutable.
#
# CMake manifest emission is tested in test_cmake_manifest.py.
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

# Add the holohub root to sys.path so we can import as utilities.cli.…
sys.path.insert(0, str(Path(os.getcwd())))

from utilities.cli.external_resolver import (
    _override_env_name,
    _ref_is_immutable,
    parse_module_dependencies,
)


def _write_metadata(tmpdir: Path, payload: dict) -> Path:
    """Write payload as metadata.json in tmpdir, return the path."""
    p = tmpdir / "metadata.json"
    p.write_text(json.dumps(payload))
    return p


class TestStringHelpers(unittest.TestCase):
    """_override_env_name / _ref_is_immutable: pure transforms."""

    def test_override_env_name_uppercases_and_underscores(self):
        self.assertEqual(
            _override_env_name("holoscan-example-utils"),
            "HOLOHUB_LOCAL_HOLOSCAN_EXAMPLE_UTILS",
        )

    def test_override_env_name_strips_non_alnum(self):
        # Funky characters collapse to one underscore each.
        self.assertEqual(_override_env_name("foo.bar/baz"), "HOLOHUB_LOCAL_FOO_BAR_BAZ")

    def test_ref_immutable_recognises_full_sha(self):
        self.assertTrue(_ref_is_immutable("abcdef0123456789abcdef0123456789abcdef01"))

    def test_ref_immutable_rejects_short_or_invalid(self):
        self.assertFalse(_ref_is_immutable("abcdef0"))  # short hex
        self.assertFalse(_ref_is_immutable("v1.0.0"))  # tag
        self.assertFalse(_ref_is_immutable("main"))  # branch
        self.assertFalse(_ref_is_immutable(""))  # empty
        self.assertFalse(_ref_is_immutable("X" * 40))  # right length, wrong charset


class TestParseModuleDependencies(unittest.TestCase):
    """parse_module_dependencies: metadata.json -> [ModuleDep]."""

    def setUp(self):
        self._tmpdir = Path(tempfile.mkdtemp(prefix="holohub_test_resolver_"))
        # Strip any pre-existing HOLOHUB_LOCAL_* env vars that could mask test
        # behaviour (the suite is meant to be idempotent across runs).
        self._saved_env = {k: v for k, v in os.environ.items() if k.startswith("HOLOHUB_LOCAL_")}
        for k in list(self._saved_env):
            del os.environ[k]

    def tearDown(self):
        for k, v in self._saved_env.items():
            os.environ[k] = v
        # Best-effort cleanup; tests use tempfile so leftovers self-collect.
        for child in sorted(self._tmpdir.rglob("*"), reverse=True):
            try:
                if child.is_dir():
                    child.rmdir()
                else:
                    child.unlink()
            except OSError:
                pass
        try:
            self._tmpdir.rmdir()
        except OSError:
            pass

    # ── Shape-detection: app / workflow / benchmark / module ────────────

    def test_application_dependencies_modules_shape(self):
        meta = _write_metadata(
            self._tmpdir,
            {
                "application": {
                    "dependencies": {
                        "operators": ["mymod8_op"],
                        "modules": [
                            {
                                "name": "mymod8",
                                "source": {"git_url": "/tmp/x", "ref": "0" * 40},
                                "provides_operators": ["mymod8_op"],
                            }
                        ],
                    }
                }
            },
        )
        deps = parse_module_dependencies(meta)
        self.assertEqual(len(deps), 1)
        self.assertEqual(deps[0].name, "mymod8")
        self.assertEqual(deps[0].provides_operators, ["mymod8_op"])

    def test_module_metadata_dependencies_flat_array(self):
        meta = _write_metadata(
            self._tmpdir,
            {
                "module": {
                    "name": "holoscan-mymod",
                    "authors": [{"name": "X", "affiliation": "Y"}],
                    "version": "1.0.0",
                    "language": ["C++"],
                    "platforms": ["x86_64"],
                    "tags": [],
                    "holoscan_sdk": {"minimum_required_version": "4.0", "tested_versions": ["4.0"]},
                    "dependencies": [
                        {
                            "name": "transitive-dep",
                            "source": {"git_url": "/tmp/x", "ref": "0" * 40},
                        }
                    ],
                }
            },
        )
        deps = parse_module_dependencies(meta)
        self.assertEqual([d.name for d in deps], ["transitive-dep"])

    def test_workflow_and_benchmark_shapes(self):
        for outer in ("workflow", "benchmark"):
            with self.subTest(outer=outer):
                meta = _write_metadata(
                    self._tmpdir,
                    {
                        outer: {
                            "dependencies": {
                                "modules": [
                                    {
                                        "name": f"{outer}_mod",
                                        "source": {"git_url": "/tmp/x", "ref": "0" * 40},
                                    }
                                ]
                            }
                        }
                    },
                )
                deps = parse_module_dependencies(meta)
                self.assertEqual([d.name for d in deps], [f"{outer}_mod"])

    # ── Empty / missing handling ────────────────────────────────────────

    def test_missing_metadata_file_returns_empty(self):
        deps = parse_module_dependencies(self._tmpdir / "does_not_exist.json")
        self.assertEqual(deps, [])

    def test_no_dependencies_field(self):
        meta = _write_metadata(self._tmpdir, {"application": {}})
        self.assertEqual(parse_module_dependencies(meta), [])

    def test_dependencies_no_modules_subfield(self):
        meta = _write_metadata(
            self._tmpdir, {"application": {"dependencies": {"operators": ["foo"]}}}
        )
        self.assertEqual(parse_module_dependencies(meta), [])

    def test_unnamed_module_skipped(self):
        meta = _write_metadata(
            self._tmpdir,
            {
                "application": {
                    "dependencies": {
                        "modules": [
                            {"source": {"git_url": "/tmp/x", "ref": "0" * 40}},  # missing name
                            {"name": "ok", "source": {"git_url": "/tmp/y", "ref": "0" * 40}},
                        ]
                    }
                }
            },
        )
        deps = parse_module_dependencies(meta)
        self.assertEqual([d.name for d in deps], ["ok"])

    # ── Source / ref validation ─────────────────────────────────────────

    def test_missing_source_raises_when_no_override(self):
        meta = _write_metadata(
            self._tmpdir,
            {"application": {"dependencies": {"modules": [{"name": "mod_no_source"}]}}},
        )
        with self.assertRaises(ValueError) as cm:
            parse_module_dependencies(meta)
        self.assertIn("missing source.git_url or source.ref", str(cm.exception))

    def test_branch_ref_warns_but_succeeds(self):
        meta = _write_metadata(
            self._tmpdir,
            {
                "application": {
                    "dependencies": {
                        "modules": [
                            {
                                "name": "mod_branch",
                                "source": {"git_url": "/tmp/x", "ref": "main"},
                            }
                        ]
                    }
                }
            },
        )
        buf = io.StringIO()
        with redirect_stderr(buf):
            deps = parse_module_dependencies(meta)
        self.assertEqual(len(deps), 1)
        self.assertIn("not a 40-char commit SHA", buf.getvalue())

    def test_full_sha_does_not_warn(self):
        meta = _write_metadata(
            self._tmpdir,
            {
                "application": {
                    "dependencies": {
                        "modules": [
                            {
                                "name": "mod_sha",
                                "source": {"git_url": "/tmp/x", "ref": "0" * 40},
                            }
                        ]
                    }
                }
            },
        )
        buf = io.StringIO()
        with redirect_stderr(buf):
            parse_module_dependencies(meta)
        self.assertNotIn("not a 40-char commit SHA", buf.getvalue())

    # ── HOLOHUB_LOCAL_<NAME> override ───────────────────────────────────

    def test_local_override_populates_override_path(self):
        # Stand up a minimal "module" directory the override will point at.
        override_dir = self._tmpdir / "local_mod"
        override_dir.mkdir()
        (override_dir / "metadata.json").write_text("{}")

        meta = _write_metadata(
            self._tmpdir,
            {
                "application": {
                    "dependencies": {
                        "modules": [
                            {
                                "name": "mymod",
                                "source": {"git_url": "/should/not/be/used", "ref": "0" * 40},
                            }
                        ]
                    }
                }
            },
        )
        with patch.dict(os.environ, {"HOLOHUB_LOCAL_MYMOD": str(override_dir)}):
            deps = parse_module_dependencies(meta)
        self.assertEqual(len(deps), 1)
        self.assertEqual(deps[0].override_path, override_dir.resolve())

    def test_local_override_without_metadata_raises(self):
        bad_dir = self._tmpdir / "no_metadata"
        bad_dir.mkdir()
        meta = _write_metadata(
            self._tmpdir,
            {
                "application": {
                    "dependencies": {
                        "modules": [
                            {
                                "name": "mymod",
                                "source": {"git_url": "/x", "ref": "0" * 40},
                            }
                        ]
                    }
                }
            },
        )
        with patch.dict(os.environ, {"HOLOHUB_LOCAL_MYMOD": str(bad_dir)}):
            with self.assertRaises(FileNotFoundError) as cm:
                parse_module_dependencies(meta)
        self.assertIn("metadata.json", str(cm.exception))

    def test_local_override_skips_source_validation(self):
        # No source block at all is OK when the override is set, since the
        # CLI doesn't need to fetch.
        override_dir = self._tmpdir / "ok_override"
        override_dir.mkdir()
        (override_dir / "metadata.json").write_text("{}")

        meta = _write_metadata(
            self._tmpdir, {"application": {"dependencies": {"modules": [{"name": "mymod"}]}}}
        )
        with patch.dict(os.environ, {"HOLOHUB_LOCAL_MYMOD": str(override_dir)}):
            deps = parse_module_dependencies(meta)
        self.assertEqual(deps[0].override_path, override_dir.resolve())
        self.assertIsNone(deps[0].git_url)


if __name__ == "__main__":
    unittest.main()
