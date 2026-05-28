# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pkg/ layout and HOLOHUB_SEARCH_PATH fixes in modules/template.

Covers the two bugs fixed in the latest template commit:

  1. HOLOHUB_SEARCH_PATH defaulted to ``${SCRIPT_DIR}/applications,…`` which
     caused ``holohub list`` to skip the module's own metadata.json and
     ``holohub package <name>`` to fail to resolve the project.

  2. Packaging used an inline ``option(PKG_…)/holohub_configure_deb()`` block
     that did not cascade through OP_/APP_ options.  Fixed by adopting the
     canonical ``pkg/<name>/`` layout driven by ``add_holohub_package()``.

These tests run without cmake or the Holoscan SDK; they only need cookiecutter.
See ``test_module_package_integration.py`` for the cmake configure + cpack path.
"""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HOLOHUB_ROOT = Path(__file__).resolve().parents[3]
HOLOHUB_SCRIPT = HOLOHUB_ROOT / "holohub"

_PROJECT_NAME = "Pkg Layout Test"
_MODULE_SLUG = "pkg_layout_test"
_OPERATOR_SLUG = f"{_MODULE_SLUG}_op"
_MODULE_REPO_NAME = f"holoscan-{_MODULE_SLUG.replace('_', '-')}"
_APP_NAME = f"{_MODULE_SLUG}_pipeline"


def _cookiecutter_available() -> bool:
    try:
        import cookiecutter  # noqa: F401

        return True
    except ImportError:
        return False


def _run(cmd: list, *, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(HOLOHUB_ROOT),
        env=env or os.environ.copy(),
    )


def _create_module(
    project_name: str, language: str, output_dir: Path
) -> subprocess.CompletedProcess:
    return _run(
        [
            str(HOLOHUB_SCRIPT),
            "create",
            project_name,
            "--template",
            "modules/template",
            "-i",
            "False",
            "--language",
            language,
            "--directory",
            str(output_dir),
        ]
    )


@unittest.skipIf(not _cookiecutter_available(), "cookiecutter not installed")
class TestPkgLayoutGenerated(unittest.TestCase):
    """Generated module must include the canonical pkg/ directory structure."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        result = _create_module(_PROJECT_NAME, "python", Path(cls._tmp.name))
        if result.returncode != 0:
            cls._tmp.cleanup()
            raise unittest.SkipTest(
                f"holohub create failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
        cls.module_dir = Path(cls._tmp.name) / _MODULE_REPO_NAME

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_tmp"):
            cls._tmp.cleanup()

    # ------------------------------------------------------------------
    # pkg/ presence
    # ------------------------------------------------------------------

    def test_pkg_directory_exists(self):
        self.assertTrue((self.module_dir / "pkg").is_dir())

    def test_pkg_root_cmake_exists(self):
        self.assertTrue((self.module_dir / "pkg" / "CMakeLists.txt").exists())

    def test_pkg_inner_cmake_exists(self):
        self.assertTrue((self.module_dir / "pkg" / _MODULE_REPO_NAME / "CMakeLists.txt").exists())

    def test_pkg_metadata_json_exists(self):
        self.assertTrue((self.module_dir / "pkg" / _MODULE_REPO_NAME / "metadata.json").exists())

    # ------------------------------------------------------------------
    # pkg/CMakeLists.txt content
    # ------------------------------------------------------------------

    def test_pkg_root_cmake_calls_add_holohub_package(self):
        content = (self.module_dir / "pkg" / "CMakeLists.txt").read_text()
        self.assertIn("add_holohub_package", content)

    def test_pkg_root_cmake_has_module_repo_name(self):
        content = (self.module_dir / "pkg" / "CMakeLists.txt").read_text()
        self.assertIn(_MODULE_REPO_NAME, content)

    def test_pkg_root_cmake_has_operator_slug(self):
        content = (self.module_dir / "pkg" / "CMakeLists.txt").read_text()
        self.assertIn(_OPERATOR_SLUG, content)

    def test_pkg_root_cmake_has_app_name(self):
        content = (self.module_dir / "pkg" / "CMakeLists.txt").read_text()
        self.assertIn(_APP_NAME, content)

    # ------------------------------------------------------------------
    # pkg/<name>/CMakeLists.txt content
    # ------------------------------------------------------------------

    def test_pkg_inner_cmake_calls_holohub_configure_deb(self):
        content = (self.module_dir / "pkg" / _MODULE_REPO_NAME / "CMakeLists.txt").read_text()
        self.assertIn("holohub_configure_deb", content)

    def test_pkg_inner_cmake_has_correct_name(self):
        content = (self.module_dir / "pkg" / _MODULE_REPO_NAME / "CMakeLists.txt").read_text()
        self.assertIn(f'NAME        "{_MODULE_REPO_NAME}"', content)

    # ------------------------------------------------------------------
    # pkg/<name>/metadata.json content
    # ------------------------------------------------------------------

    def test_pkg_metadata_json_has_dockerfile(self):
        meta = json.loads(
            (self.module_dir / "pkg" / _MODULE_REPO_NAME / "metadata.json").read_text()
        )
        self.assertIn("package", meta)
        self.assertEqual(meta["package"]["dockerfile"], "Dockerfile")

    # ------------------------------------------------------------------
    # Root CMakeLists.txt ordering
    # ------------------------------------------------------------------

    def test_cmakelists_pkg_before_operators(self):
        """pkg/ must be add_subdirectory()'d before operators/ so OP_ FORCE takes effect."""
        content = (self.module_dir / "CMakeLists.txt").read_text()
        pkg_pos = content.find("add_subdirectory(pkg)")
        ops_pos = content.find("add_subdirectory(operators)")
        self.assertGreater(pkg_pos, -1, "add_subdirectory(pkg) not found in CMakeLists.txt")
        self.assertGreater(ops_pos, -1, "add_subdirectory(operators) not found")
        self.assertLess(pkg_pos, ops_pos, "add_subdirectory(pkg) must precede operators/")

    def test_cmakelists_pkg_before_applications(self):
        """pkg/ must be add_subdirectory()'d before applications/ so APP_ FORCE takes effect."""
        content = (self.module_dir / "CMakeLists.txt").read_text()
        pkg_pos = content.find("add_subdirectory(pkg)")
        app_pos = content.find("add_subdirectory(applications)")
        self.assertGreater(pkg_pos, -1, "add_subdirectory(pkg) not found in CMakeLists.txt")
        self.assertGreater(app_pos, -1, "add_subdirectory(applications) not found")
        self.assertLess(pkg_pos, app_pos, "add_subdirectory(pkg) must precede applications/")

    # ------------------------------------------------------------------
    # holohub wrapper HOLOHUB_SEARCH_PATH
    # ------------------------------------------------------------------

    def test_holohub_wrapper_search_path_defaults_to_script_dir(self):
        """Wrapper must default HOLOHUB_SEARCH_PATH to ${SCRIPT_DIR} (the module root)."""
        content = (self.module_dir / "holohub").read_text()
        self.assertIn(
            "HOLOHUB_SEARCH_PATH:-${SCRIPT_DIR}",
            content,
            "HOLOHUB_SEARCH_PATH must default to ${SCRIPT_DIR}",
        )

    def test_holohub_wrapper_search_path_not_applications_subtree(self):
        """Old default '${SCRIPT_DIR}/applications,…' caused holohub list to miss the module."""
        content = (self.module_dir / "holohub").read_text()
        self.assertNotIn(
            "${SCRIPT_DIR}/applications",
            content,
            "HOLOHUB_SEARCH_PATH must not default to the applications/ subtree only",
        )


@unittest.skipIf(not _cookiecutter_available(), "cookiecutter not installed")
class TestModuleListWithRootSearchPath(unittest.TestCase):
    """holohub list must show the module entry when HOLOHUB_SEARCH_PATH = module root."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        result = _create_module(_PROJECT_NAME, "python", Path(cls._tmp.name))
        if result.returncode != 0:
            cls._tmp.cleanup()
            raise unittest.SkipTest(
                f"holohub create failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
        cls.module_dir = Path(cls._tmp.name) / _MODULE_REPO_NAME

        env = {**os.environ, "HOLOHUB_SEARCH_PATH": str(cls.module_dir)}
        cls.list_result = _run([str(HOLOHUB_SCRIPT), "list"], env=env)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_tmp"):
            cls._tmp.cleanup()

    def test_list_succeeds(self):
        self.assertEqual(
            self.list_result.returncode,
            0,
            f"holohub list failed:\n{self.list_result.stdout}\n{self.list_result.stderr}",
        )

    def test_list_shows_modules_section(self):
        """The MODULES section must appear — this was missing with the old narrower path."""
        self.assertIn(
            "MODULES",
            self.list_result.stdout,
            "Expected '== MODULES =' section in holohub list output with root search path",
        )

    def test_list_shows_module_name(self):
        self.assertIn(
            _MODULE_REPO_NAME,
            self.list_result.stdout,
            f"Expected '{_MODULE_REPO_NAME}' in holohub list output",
        )

    def test_list_shows_application_too(self):
        """Root search path must also surface applications (not just the module descriptor)."""
        self.assertIn(
            _APP_NAME,
            self.list_result.stdout,
            f"Expected '{_APP_NAME}' in holohub list output",
        )


if __name__ == "__main__":
    unittest.main()
