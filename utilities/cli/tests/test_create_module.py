# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke tests for `holohub create --template modules/template`.

Each test:
  1. Runs `./holohub create` in a temporary directory (non-interactively).
  2. Inspects the generated directory structure.
  3. Runs `./holohub list` with HOLOHUB_SEARCH_PATH pointing at the generated
     module's applications/ directory and asserts the scaffolded application
     appears in stdout.

These tests require cookiecutter (installed by `./holohub setup --scripts template`)
and are skipped automatically when it is not available.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HOLOHUB_ROOT = Path(__file__).resolve().parents[3]
HOLOHUB_SCRIPT = HOLOHUB_ROOT / "holohub"


def _cookiecutter_available() -> bool:
    try:
        import cookiecutter  # noqa: F401

        return True
    except ImportError:
        return False


def _run(
    cmd: list, *, env: dict | None = None, cwd: Path = HOLOHUB_ROOT
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env or os.environ.copy(),
    )


@unittest.skipIf(not _cookiecutter_available(), "cookiecutter not installed")
class TestCreateModule(unittest.TestCase):
    """Smoke tests for holohub create + holohub list with the modules template."""

    def _create_module(
        self, project_name: str, language: str, output_dir: Path
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

    def _list_with_search_path(self, search_path: Path) -> subprocess.CompletedProcess:
        env = {**os.environ, "HOLOHUB_SEARCH_PATH": str(search_path)}
        return _run([str(HOLOHUB_SCRIPT), "list"], env=env)

    # ------------------------------------------------------------------
    # Python module
    # ------------------------------------------------------------------

    def test_create_and_list_python_module(self):
        """Scaffold a pure-Python module and verify it appears in holohub list."""
        project_name = "Smoke Test Module"
        # Derived by cookiecutter: module_slug=smoke_test_module,
        # module_repo_name=holoscan-smoke-test-module, operator_slug=smoke_test_module_op
        module_slug = "smoke_test_module"
        operator_slug = f"{module_slug}_op"
        module_repo_name = f"holoscan-{module_slug.replace('_', '-')}"
        app_name = f"{module_slug}_pipeline"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # 1. Create
            result = self._create_module(project_name, "python", tmp_path)
            self.assertEqual(
                result.returncode,
                0,
                f"holohub create failed:\nstdout: {result.stdout}\nstderr: {result.stderr}",
            )

            module_dir = tmp_path / module_repo_name
            self.assertTrue(module_dir.is_dir(), f"Expected module dir {module_dir} not found")

            # 2. Structure checks
            expected_paths = [
                Path("holohub"),
                Path("metadata.json"),
                Path("README.md"),
                Path("DEVELOPER.md"),
                Path(f"operators/{operator_slug}/{operator_slug}.py"),
                Path(f"applications/{app_name}/metadata.json"),
                Path("python") / "holoscan" / module_slug / "__init__.py",
            ]
            for rel in expected_paths:
                self.assertTrue(
                    (module_dir / rel).exists(),
                    f"Expected generated file missing: {rel}",
                )

            # Python module must NOT have C++-only files
            self.assertFalse(
                (module_dir / ".clang-format").exists(), ".clang-format present in Python module"
            )
            self.assertFalse(
                (module_dir / "tests" / "cpp").exists(), "tests/cpp present in Python module"
            )

            # 3. Wrapper must be executable
            self.assertTrue(
                os.access(module_dir / "holohub", os.X_OK),
                "holohub wrapper is not executable",
            )

            # 4. holohub list
            list_result = self._list_with_search_path(module_dir / "applications")
            self.assertEqual(
                list_result.returncode,
                0,
                f"holohub list failed:\nstdout: {list_result.stdout}\nstderr: {list_result.stderr}",
            )
            self.assertIn(
                app_name,
                list_result.stdout,
                f"Expected '{app_name}' in holohub list output:\n{list_result.stdout}",
            )

    # ------------------------------------------------------------------
    # C++ module
    # ------------------------------------------------------------------

    def test_create_and_list_cpp_module(self):
        """Scaffold a C++ module and verify it appears in holohub list."""
        project_name = "Smoke Test Cpp Module"
        module_slug = "smoke_test_cpp_module"
        operator_slug = f"{module_slug}_op"
        module_repo_name = f"holoscan-{module_slug.replace('_', '-')}"
        app_name = f"{module_slug}_pipeline"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # 1. Create
            result = self._create_module(project_name, "cpp", tmp_path)
            self.assertEqual(
                result.returncode,
                0,
                f"holohub create failed:\nstdout: {result.stdout}\nstderr: {result.stderr}",
            )

            module_dir = tmp_path / module_repo_name
            self.assertTrue(module_dir.is_dir(), f"Expected module dir {module_dir} not found")

            # 2. Structure checks — C++ specific files
            expected_paths = [
                Path("holohub"),
                Path("metadata.json"),
                Path("README.md"),
                Path("DEVELOPER.md"),
                Path(f"operators/{operator_slug}/{operator_slug}.cpp"),
                Path(f"operators/{operator_slug}/{operator_slug}.hpp"),
                Path(f"operators/{operator_slug}/python/_{operator_slug}_bindings.cpp"),
                Path(f"applications/{app_name}/metadata.json"),
                Path(f"applications/{app_name}/{app_name}.cpp"),
                Path("tests/cpp/test_operators.cpp"),
                Path(".clang-format"),
                Path("python") / "holoscan" / module_slug / "__init__.py",
            ]
            for rel in expected_paths:
                self.assertTrue(
                    (module_dir / rel).exists(),
                    f"Expected generated file missing: {rel}",
                )

            # 3. Wrapper must be executable
            self.assertTrue(
                os.access(module_dir / "holohub", os.X_OK),
                "holohub wrapper is not executable",
            )

            # 4. holohub list
            list_result = self._list_with_search_path(module_dir / "applications")
            self.assertEqual(
                list_result.returncode,
                0,
                f"holohub list failed:\nstdout: {list_result.stdout}\nstderr: {list_result.stderr}",
            )
            self.assertIn(
                app_name,
                list_result.stdout,
                f"Expected '{app_name}' in holohub list output:\n{list_result.stdout}",
            )


if __name__ == "__main__":
    unittest.main()
