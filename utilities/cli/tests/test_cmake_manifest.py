#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unit tests for utilities/cli/cmake_manifest.py.
#
# Covers:
#   - write_external_operators_manifest: emits FetchContent_Declare entries,
#     HOLOHUB_EXT_OP_<op>_PROVIDER lookup variables, and the
#     FETCHCONTENT_SOURCE_DIR_<UPPER> override line. Warns on operator
#     collisions. Idempotent on identical input.
#   - _provider_id string helper.
#
# Pure I/O via temp files; no docker, no network, no GPU. Runs in milliseconds.

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

# Add the holohub root to sys.path so we can import as utilities.cli.…
sys.path.insert(0, str(Path(os.getcwd())))

from utilities.cli.cmake_manifest import _provider_id, write_external_operators_manifest
from utilities.cli.external_resolver import ModuleDep


class TestProviderIdHelper(unittest.TestCase):
    """_provider_id: module name -> CMake-safe identifier."""

    def test_sanitises_hyphens(self):
        self.assertEqual(_provider_id("holoscan-example-utils"), "holoscan_example_utils")

    def test_keeps_underscores_and_alnum(self):
        self.assertEqual(_provider_id("mymod8_op"), "mymod8_op")

    def test_rejects_special_chars(self):
        self.assertEqual(_provider_id("Foo.Bar/Baz-Qux"), "Foo_Bar_Baz_Qux")


class TestWriteManifest(unittest.TestCase):
    """write_external_operators_manifest: ModuleDep[] -> CMake text."""

    def setUp(self):
        self._tmpdir = Path(tempfile.mkdtemp(prefix="holohub_test_manifest_"))

    def tearDown(self):
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

    def _write(self, deps):
        out = self._tmpdir / "external_operators_manifest.cmake"
        write_external_operators_manifest(deps, out)
        return out.read_text()

    # ── holohub_declare_external_module emission ────────────────────────

    def test_emits_declare_function_call(self):
        text = self._write(
            [
                ModuleDep(name="mymod", git_url="/tmp/x", ref="0" * 40),
            ]
        )
        self.assertIn("holohub_declare_external_module(mymod", text)

    def test_emits_git_repository_and_tag(self):
        text = self._write(
            [
                ModuleDep(
                    name="mymod", git_url="https://example.com/foo.git", ref="abc" + "0" * 37
                ),
            ]
        )
        self.assertIn("holohub_declare_external_module(mymod", text)
        self.assertIn('GIT_REPOSITORY  "https://example.com/foo.git"', text)
        self.assertIn('GIT_TAG         "' + "abc" + "0" * 37 + '"', text)

    def test_provider_id_sanitised_in_declare(self):
        text = self._write(
            [
                ModuleDep(name="holoscan-example-utils", git_url="/x", ref="0" * 40),
            ]
        )
        # Declared name uses the provider_id (underscores), not the original.
        self.assertIn("holohub_declare_external_module(holoscan_example_utils", text)
        self.assertNotIn("holohub_declare_external_module(holoscan-example-utils", text)

    # ── PROVIDES_OPERATORS forwarding ───────────────────────────────────

    def test_emits_provides_operators_in_function_call(self):
        text = self._write(
            [
                ModuleDep(
                    name="bigmod",
                    git_url="/x",
                    ref="0" * 40,
                    provides_operators=["bigmod_signal_op", "bigmod_render_op"],
                ),
            ]
        )
        # PROVIDES_OPERATORS are forwarded inside holohub_declare_external_module;
        # the function sets HOLOHUB_EXT_OP_<op>_PROVIDER as normal variables.
        # No raw set() calls for PROVIDER appear in the manifest.
        self.assertIn("PROVIDES_OPERATORS bigmod_signal_op bigmod_render_op", text)
        self.assertNotRegex(text, r"set\(HOLOHUB_EXT_OP_\S+_PROVIDER\b")

    def test_no_provides_operators_when_empty(self):
        text = self._write([ModuleDep(name="mymod", git_url="/x", ref="0" * 40)])
        self.assertNotIn("PROVIDES_OPERATORS", text)
        self.assertNotRegex(text, r"set\(HOLOHUB_EXT_OP_\S+_PROVIDER\b")

    # ── Local override redirection ──────────────────────────────────────

    def test_local_override_emits_source_dir_var(self):
        text = self._write(
            [
                ModuleDep(
                    name="mymod",
                    override_path=Path("/abs/path/to/mymod"),
                    provides_operators=["mymod_op"],
                ),
            ]
        )
        # FETCHCONTENT_SOURCE_DIR_<UPPER> (cache var) goes BEFORE the function
        # call so it's visible to readers; CMake honors it at MakeAvailable time.
        idx_src = text.find("FETCHCONTENT_SOURCE_DIR_MYMOD")
        idx_decl = text.find("holohub_declare_external_module(mymod")
        self.assertGreater(idx_src, -1, "expected FETCHCONTENT_SOURCE_DIR_MYMOD line")
        self.assertGreater(idx_decl, -1, "expected holohub_declare_external_module line")
        self.assertLess(idx_src, idx_decl, "override line must precede function call")
        self.assertIn('"/abs/path/to/mymod"', text)
        self.assertIn("FORCE", text)  # cache override needs FORCE to take effect

    def test_local_override_only_no_source_block(self):
        # Override-only: SOURCE_DIR forwarded to FetchContent_Declare inside the function.
        text = self._write(
            [
                ModuleDep(name="mymod", override_path=Path("/abs/local")),
            ]
        )
        self.assertIn("holohub_declare_external_module(mymod", text)
        self.assertIn('SOURCE_DIR  "/abs/local"', text)

    # ── Operator collision warning ──────────────────────────────────────

    def test_collision_warns_and_keeps_latter(self):
        deps = [
            ModuleDep(name="modA", git_url="/x", ref="0" * 40, provides_operators=["shared_op"]),
            ModuleDep(name="modB", git_url="/y", ref="0" * 40, provides_operators=["shared_op"]),
        ]
        out = self._tmpdir / "manifest.cmake"
        buf = io.StringIO()
        with redirect_stderr(buf):
            write_external_operators_manifest(deps, out)
        text = out.read_text()
        # Both function calls emit; warning fires for the collision.
        self.assertIn("holohub_declare_external_module(modA", text)
        self.assertIn("holohub_declare_external_module(modB", text)
        self.assertIn("WARNING", buf.getvalue())
        self.assertIn("shared_op", buf.getvalue())
        self.assertIn("modA", buf.getvalue())
        self.assertIn("modB", buf.getvalue())

    # ── Idempotence ─────────────────────────────────────────────────────

    def test_writing_twice_produces_same_text(self):
        deps = [
            ModuleDep(
                name="mymod",
                git_url="/x",
                ref="0" * 40,
                provides_operators=["mymod_op"],
            ),
        ]
        first = self._write(deps)
        second = self._write(deps)
        self.assertEqual(first, second)

    def test_empty_deps_writes_minimal_skeleton(self):
        text = self._write([])
        # Header only — no function calls, no FetchContent_Declare, no PROVIDER set()s.
        self.assertNotRegex(text, r"holohub_declare_external_module\s*\(")
        self.assertNotRegex(text, r"FetchContent_Declare\s*\(")
        self.assertNotRegex(text, r"set\(HOLOHUB_EXT_OP_\S+_PROVIDER\b")
        self.assertNotIn("CACHE", text)

    def test_provider_lookup_not_emitted_as_raw_set(self):
        # HOLOHUB_EXT_OP_<op>_PROVIDER variables are set inside
        # holohub_declare_external_module (PARENT_SCOPE → normal variable, not cache).
        # The manifest must not emit them as raw set() calls — the contract is
        # enforced by the function definition in HoloHubConfigHelpers.cmake.
        text = self._write(
            [
                ModuleDep(
                    name="mymod",
                    git_url="/x",
                    ref="0" * 40,
                    provides_operators=["mymod_op"],
                ),
            ]
        )
        self.assertIn("PROVIDES_OPERATORS mymod_op", text)
        self.assertNotRegex(text, r"set\(HOLOHUB_EXT_OP_")


if __name__ == "__main__":
    unittest.main()
