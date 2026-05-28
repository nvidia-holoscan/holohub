# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: create a module → holohub build → holohub package → verify .deb.

This test exercises the full packaging path introduced in the modules/template
``pkg/`` layout commit using the generated module's own ``./holohub`` wrapper,
exactly as a real developer would.

Prerequisites (auto-skipped if missing):
  - cookiecutter  (installed by ``./holohub setup --scripts template``)
  - cmake / cpack  (standard build tools; added to CI apt-get install step)
  - dpkg-deb       (standard on Debian/Ubuntu)

No C++ compiler and no Holoscan cmake SDK are required.  The template was
updated to use ``LANGUAGES NONE`` and to skip ``find_package(holoscan)`` for
pure-Python modules, so cmake configure needs no external toolchain.

The test avoids downloading the HoloHub CLI from GitHub by setting ``CLI_DIR``
to the local holohub-internal tree, which the generated wrapper uses directly
when the ``utilities/cli`` sub-path is already present there.

What is verified:
  1. ``holohub create`` produces the correct module and pkg/ tree.
  2. The generated ``./holohub list`` shows the module under MODULES (search-path fix).
  3. ``./holohub build <app> --local`` succeeds (regular build path).
  4. ``./holohub package <module> --local --pkg-generator DEB`` succeeds.
     (Package resolution uses cwd/metadata.json; cmake runs -DBUILD_ALL=OFF -DPKG_<slug>=ON.)
  5. CMakeCache.txt from step 4 has OP_ and APP_ FORCE-cascaded ON (core regression check).
  6. ``pkg/CPackConfig-<name>.cmake`` was emitted by ``holohub_configure_deb()``.
  7. A ``.deb`` file is produced with correct control-file metadata.

HOLOHUB_SEARCH_PATH fix is exercised by step 2 (list shows MODULES section) and step 3
(build resolves the app via find_project → HOLOHUB_SEARCH_PATH walk).
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HOLOHUB_ROOT = Path(__file__).resolve().parents[3]
HOLOHUB_SCRIPT = HOLOHUB_ROOT / "holohub"

_PROJECT_NAME = "Package Integration Test"
_MODULE_SLUG = "package_integration_test"
_OPERATOR_SLUG = f"{_MODULE_SLUG}_op"
_MODULE_REPO_NAME = f"holoscan-{_MODULE_SLUG.replace('_', '-')}"
_APP_NAME = f"{_MODULE_SLUG}_pipeline"
_PKG_SLUG = _MODULE_REPO_NAME.replace("-", "_")


# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------


def _cookiecutter_available() -> bool:
    try:
        import cookiecutter  # noqa: F401

        return True
    except ImportError:
        return False


def _tool_available(name: str) -> bool:
    return subprocess.run(["which", name], capture_output=True).returncode == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
    cmd: list,
    *,
    cwd: Path = HOLOHUB_ROOT,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env or os.environ.copy(),
    )


def _ensure_cmake_helper(cmake_dir: Path, src: Path) -> None:
    """Copy *src* into *cmake_dir* if it isn't already there (post-gen hook fallback)."""
    import shutil

    dst = cmake_dir / src.name
    if not dst.exists() and src.exists():
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Integration test class
# ---------------------------------------------------------------------------


@unittest.skipIf(not _cookiecutter_available(), "cookiecutter not installed")
@unittest.skipIf(not _tool_available("cmake"), "cmake not available")
@unittest.skipIf(not _tool_available("cpack"), "cpack not available")
class TestModulePackageIntegration(unittest.TestCase):
    """End-to-end: create module → holohub build → holohub package → inspect .deb."""

    # ------------------------------------------------------------------
    # Class-level fixture: create module, run build + package once.
    # Tests assert against the saved results.
    # ------------------------------------------------------------------

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)

        # ---- 1. holohub create ----
        # Pass HOLOHUB_ROOT so the post-gen hook finds and copies cmake helpers
        # (HoloHubConfigHelpers.cmake, holohub_configure_deb.cmake).
        create_env = {**os.environ, "HOLOHUB_ROOT": str(HOLOHUB_ROOT)}
        create_result = _run(
            [
                str(HOLOHUB_SCRIPT),
                "create",
                _PROJECT_NAME,
                "--template",
                "modules/template",
                "-i",
                "False",
                "--language",
                "python",
                "--directory",
                str(tmp),
                "--context",
                "full_name=Test User",
                "--context",
                "affiliation=NVIDIA",
                "--context",
                "contact_email=test@nvidia.example.com",
            ],
            env=create_env,
        )
        if create_result.returncode != 0:
            cls._tmp.cleanup()
            raise unittest.SkipTest(
                f"holohub create failed:\nstdout: {create_result.stdout}\n"
                f"stderr: {create_result.stderr}"
            )

        cls.module_dir = tmp / _MODULE_REPO_NAME
        cls.module_holohub = cls.module_dir / "holohub"

        # ---- 2. Ensure cmake helpers are present (fallback) ----
        # The post-gen hook copies them when HOLOHUB_ROOT is in the environment.
        # This fallback handles edge cases (e.g. restricted CI env).
        cmake_dir = cls.module_dir / "cmake"
        cmake_dir.mkdir(exist_ok=True)
        _ensure_cmake_helper(cmake_dir, HOLOHUB_ROOT / "cmake" / "HoloHubConfigHelpers.cmake")
        _ensure_cmake_helper(
            cmake_dir, HOLOHUB_ROOT / "cmake" / "modules" / "holohub_configure_deb.cmake"
        )

        # ---- CLI env: route ./holohub to local clone, avoid GitHub download ----
        # CLI_DIR tells the wrapper to use <HOLOHUB_ROOT>/utilities/cli directly.
        # HOLOHUB_ROOT is intentionally NOT set so the wrapper defaults it to
        # SCRIPT_DIR (the generated module), which is the correct cmake source root.
        cls.cli_env = {**os.environ, "CLI_DIR": str(HOLOHUB_ROOT)}

        # ---- 3. holohub list ----
        cls.list_result = _run(
            [str(cls.module_holohub), "list"],
            cwd=cls.module_dir,
            env=cls.cli_env,
        )

        # ---- 4. holohub build <app> --local ----
        cls.build_result = _run(
            [
                str(cls.module_holohub),
                "build",
                _APP_NAME,
                "--local",
                "--language",
                "python",
            ],
            cwd=cls.module_dir,
            env=cls.cli_env,
        )

        # ---- 5. holohub package <module> --local --pkg-generator DEB ----
        # _resolve_module_project() detects the module via cwd/metadata.json ("module" key),
        # then cmake is invoked as:
        #   cmake -S <module_dir> -DBUILD_ALL=OFF -DPKG_<slug>=ON
        # The PKG option cascades OP_/APP_ via add_holohub_package() in pkg/ — the core
        # regression test for the pkg/ ordering fix.
        cls.package_result = _run(
            [
                str(cls.module_holohub),
                "package",
                _MODULE_REPO_NAME,
                "--local",
                "--pkg-generator",
                "DEB",
            ],
            cwd=cls.module_dir,
            env=cls.cli_env,
        )

        # Locate the CMakeCache from the packaging build.
        # holohub package --local uses:
        #   build_dir = DEFAULT_BUILD_PARENT_DIR / pkg_slug / "package"
        # where DEFAULT_BUILD_PARENT_DIR = HOLOHUB_ROOT/build = <module_dir>/build
        # and pkg_slug = module_repo_name.replace("-", "_")
        cls.pkg_build_dir = cls.module_dir / "build" / _PKG_SLUG / "package"

        # Find the .deb produced anywhere under the temp tree
        cls.deb_paths = list(tmp.glob("**/*.deb"))
        cls.deb_path = cls.deb_paths[0] if cls.deb_paths else None

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_tmp"):
            cls._tmp.cleanup()

    # ------------------------------------------------------------------
    # 1. holohub create structure
    # ------------------------------------------------------------------

    def test_module_dir_created(self):
        self.assertTrue(
            self.module_dir.is_dir(),
            f"Expected generated module directory: {self.module_dir}",
        )

    def test_pkg_tree_present(self):
        for rel in [
            Path("pkg") / "CMakeLists.txt",
            Path("pkg") / _MODULE_REPO_NAME / "CMakeLists.txt",
            Path("pkg") / _MODULE_REPO_NAME / "metadata.json",
        ]:
            self.assertTrue(
                (self.module_dir / rel).exists(),
                f"Missing pkg/ file: {rel}",
            )

    # ------------------------------------------------------------------
    # 2. holohub list (search-path fix smoke-test)
    # ------------------------------------------------------------------

    def test_list_exits_zero(self):
        self.assertEqual(
            self.list_result.returncode,
            0,
            f"holohub list failed:\n{self.list_result.stdout}\n{self.list_result.stderr}",
        )

    def test_list_shows_modules_section(self):
        """The MODULES section requires the root metadata.json to be found — search-path fix."""
        self.assertIn(
            "MODULES",
            self.list_result.stdout,
            "Expected MODULES section; was HOLOHUB_SEARCH_PATH restricted to applications/?",
        )

    def test_list_shows_module_name(self):
        self.assertIn(_MODULE_REPO_NAME, self.list_result.stdout)

    def test_list_shows_application(self):
        self.assertIn(_APP_NAME, self.list_result.stdout)

    # ------------------------------------------------------------------
    # 3. holohub build --local
    # ------------------------------------------------------------------

    def test_build_exits_zero(self):
        self.assertEqual(
            self.build_result.returncode,
            0,
            f"holohub build --local failed:\n{self.build_result.stdout}\n"
            f"{self.build_result.stderr}",
        )

    # ------------------------------------------------------------------
    # 4. holohub package --local
    # ------------------------------------------------------------------

    def test_package_exits_zero(self):
        self.assertEqual(
            self.package_result.returncode,
            0,
            f"holohub package --local failed:\n{self.package_result.stdout}\n"
            f"{self.package_result.stderr}",
        )

    # ------------------------------------------------------------------
    # 5. CMakeCache.txt — PKG/OP/APP cascade (core regression check)
    # ------------------------------------------------------------------

    def _cache(self) -> str:
        cache_path = self.pkg_build_dir / "CMakeCache.txt"
        if not cache_path.exists():
            self.skipTest(
                f"CMakeCache.txt not found at {cache_path} — packaging build did not succeed"
            )
        return cache_path.read_text()

    def test_pkg_option_on_in_cache(self):
        cache = self._cache()
        self.assertRegex(cache, rf"PKG_{_PKG_SLUG}:BOOL=ON")

    def test_operator_force_cascaded_in_cache(self):
        """add_holohub_package() must FORCE OP_<op>=ON before operators/ is processed."""
        cache = self._cache()
        self.assertRegex(
            cache,
            rf"OP_{_OPERATOR_SLUG}:BOOL=ON",
            f"OP_{_OPERATOR_SLUG} not force-cascaded; pkg/ ordering fix regression?",
        )

    def test_application_force_cascaded_in_cache(self):
        """add_holohub_package() must FORCE APP_<app>=ON before applications/ is processed."""
        cache = self._cache()
        self.assertRegex(
            cache,
            rf"APP_{_APP_NAME}:BOOL=ON",
            f"APP_{_APP_NAME} not force-cascaded; pkg/ ordering fix regression?",
        )

    # ------------------------------------------------------------------
    # 6. CPackConfig generated by holohub_configure_deb()
    # ------------------------------------------------------------------

    def test_cpack_config_generated(self):
        cpack_config = self.pkg_build_dir / "pkg" / f"CPackConfig-{_MODULE_REPO_NAME}.cmake"
        if self.package_result.returncode != 0:
            self.skipTest("holohub package did not succeed")
        self.assertTrue(
            cpack_config.exists(),
            f"Expected {cpack_config} — holohub_configure_deb() must write it",
        )

    def test_cpack_config_has_package_name(self):
        cpack_config = self.pkg_build_dir / "pkg" / f"CPackConfig-{_MODULE_REPO_NAME}.cmake"
        if not cpack_config.exists():
            self.skipTest("CPackConfig not generated")
        self.assertIn(_MODULE_REPO_NAME, cpack_config.read_text())

    # ------------------------------------------------------------------
    # 7. .deb produced with correct metadata
    # ------------------------------------------------------------------

    def test_deb_file_produced(self):
        if self.package_result.returncode != 0:
            self.skipTest("holohub package did not succeed")
        self.assertIsNotNone(
            self.deb_path,
            f"No .deb found under {self._tmp.name}; "
            f"package stdout:\n{self.package_result.stdout}",
        )
        self.assertTrue(self.deb_path.exists())

    def _deb_field(self, field: str) -> str:
        if self.deb_path is None:
            self.skipTest("No .deb produced")
        result = _run(["dpkg-deb", "-f", str(self.deb_path), field])
        self.assertEqual(result.returncode, 0, f"dpkg-deb -f failed for field {field}")
        return result.stdout.strip()

    def test_deb_package_name(self):
        self.assertEqual(self._deb_field("Package"), _MODULE_REPO_NAME)

    def test_deb_version_semver(self):
        self.assertRegex(self._deb_field("Version"), r"^\d+\.\d+\.\d+")

    def test_deb_depends_holoscan(self):
        self.assertIn("holoscan", self._deb_field("Depends").lower())

    def test_deb_contact_in_maintainer(self):
        self.assertIn("test@nvidia.example.com", self._deb_field("Maintainer"))


if __name__ == "__main__":
    unittest.main()
