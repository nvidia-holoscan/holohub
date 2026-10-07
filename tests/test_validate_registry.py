"""Exercise the offline CLI against temporary registries, including failures."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from test_source_schema import example, KINDS, ROOT, SCHEMA_PATH


SCRIPT = ROOT / "utilities" / "validate_registry.py"


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.schema_path = self.root / "utilities" / "source.schema.json"
        self.schema_path.parent.mkdir()
        shutil.copyfile(SCHEMA_PATH, self.schema_path)
        for kind in KINDS:
            (self.root / f"{kind}s").mkdir()

    def write_record(self, kind="module", name=None, record=None, directory=None):
        record = example(kind) if record is None else record
        if name is not None:
            record["name"] = name
        path = self.root / (directory or f"{kind}s") / f"{record['name']}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def run_validator(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.root), *map(str, args)],
            capture_output=True, text=True, check=False,
        )

    def assert_rejected(self, expected, *args):
        result = self.run_validator(*args)
        self.assertEqual(1, result.returncode, result.stdout + result.stderr)
        self.assertIn(expected, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        return result

    def test_empty_registry_is_valid(self):
        result = self.run_validator()
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("Validated 0 source records", result.stdout)

    def test_all_kinds_are_validated_and_index_is_sorted_and_reproducible(self):
        for kind in reversed(KINDS):
            self.write_record(kind)
        self.write_record("module", name="aaa-module", record={
            "name": "aaa-module", "kind": "module",
            "source": {"type": "git", "url": "https://example.org/another", "ref": "main"},
        })
        index = self.root / "dist" / "sources.json"
        result = self.run_validator("--index", index)
        self.assertEqual(0, result.returncode, result.stderr)
        content = index.read_bytes()
        records = json.loads(content)
        self.assertEqual(6, len(records))
        keys = [(record["kind"], record["name"]) for record in records]
        self.assertEqual(sorted(keys), keys)
        self.assertTrue(all("$schema" not in record for record in records))
        self.assertEqual(0, self.run_validator("--index", index).returncode)
        self.assertEqual(content, index.read_bytes())

    def test_issues_url_is_preserved_in_the_source_index(self):
        record = example("application")
        record["issues_url"] = "https://support.example.org/report?project=example#bug"
        self.write_record("application", record=record)
        index = self.root / "sources.json"
        result = self.run_validator("--index", index)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual([record], json.loads(index.read_text()))

    def test_readmes_in_all_kinds_are_allowed_and_excluded_from_index(self):
        for kind in KINDS:
            self.write_record(kind)
            (self.root / f"{kind}s" / "README.md").write_text(
                f"# Register a {kind}\n", encoding="utf-8",
            )
        index = self.root / "sources.json"
        result = self.run_validator("--index", index)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(
            [example(kind) for kind in sorted(KINDS)],
            json.loads(index.read_text()),
        )

    def test_issues_url_does_not_distinguish_duplicate_sources(self):
        self.write_record()
        record = example("module")
        record["issues_url"] = "https://support.example.org/issues"
        self.write_record(name="alias", record=record)
        self.assert_rejected("duplicate source")

    def test_wrong_directory_and_filename_are_rejected(self):
        path = self.write_record("module", directory="applications")
        path.rename(path.with_name("wrong-name.json"))
        result = self.assert_rejected("belongs in modules/")
        self.assertIn("filename must be example-module.json", result.stderr)

    def test_invalid_schema_is_rejected_even_for_an_empty_registry(self):
        self.schema_path.write_text('{"type": "not-a-type"}')
        self.assert_rejected("utilities/source.schema.json")

    def test_missing_schema_and_missing_entity_directory_are_rejected(self):
        self.schema_path.unlink()
        self.assert_rejected("utilities/source.schema.json")
        shutil.copyfile(SCHEMA_PATH, self.schema_path)
        (self.root / "skills").rmdir()
        self.assert_rejected("skills: missing directory")

    def test_remote_schema_references_are_rejected_without_retrieval(self):
        schema = json.loads(self.schema_path.read_text())
        schema["$ref"] = "https://example.invalid/remote.schema.json"
        self.schema_path.write_text(json.dumps(schema))
        self.assert_rejected("schema references must be local")

    def test_record_schema_hint_is_rejected(self):
        record = example("module")
        record["$schema"] = "https://example.invalid/untrusted.schema.json"
        self.write_record(record=record)
        self.assert_rejected("$schema")

    def test_malformed_json_and_duplicate_keys_are_rejected(self):
        for content, message in (
            ('{"name":', "invalid JSON"),
            ('{"name": "first", "name": "second"}', "duplicate JSON key"),
            ('{"source": {"ref": "main", "ref": "v1"}}', "duplicate JSON key"),
            ('{"value": NaN}', "non-finite JSON number"),
            ('{"value": Infinity}', "non-finite JSON number"),
        ):
            with self.subTest(content=content):
                (self.root / "modules" / "broken.json").write_text(content)
                self.assert_rejected(message)

    def test_non_utf8_record_is_rejected(self):
        (self.root / "modules" / "broken.json").write_bytes(b"\xff")
        self.assert_rejected("modules/broken.json")

    def test_legacy_lists_and_non_objects_are_rejected(self):
        for record in ([], None, "a string", {"modules": [example("module")]}):
            with self.subTest(record=record):
                (self.root / "modules" / "legacy.json").write_text(json.dumps(record))
                self.assert_rejected("modules/legacy.json")

    def test_source_code_nested_directories_and_symlinks_are_rejected(self):
        directory = self.root / "modules"
        (directory / "implementation.py").write_text("print('not executed')")
        (directory / "nested").mkdir()
        (directory / "linked.json").symlink_to(self.schema_path)
        result = self.assert_rejected("only source record JSON files")
        self.assertIn("nested", result.stderr)
        self.assertIn("symlinks are not allowed", result.stderr)

    def test_entity_directory_cannot_itself_be_a_symlink(self):
        (self.root / "modules").rmdir()
        (self.root / "modules").symlink_to(self.root / "applications", target_is_directory=True)
        self.assert_rejected("modules: symlinks are not allowed")

    def test_readme_directories_are_rejected(self):
        (self.root / "modules" / "README.md").mkdir()
        self.assert_rejected("modules/README.md: only source record JSON files")

    def test_readme_symlinks_are_rejected(self):
        path = self.root / "modules" / "README.md"
        for target in (self.schema_path, self.root / "tutorials", self.root / "missing"):
            with self.subTest(target=target):
                path.symlink_to(target)
                self.assert_rejected("modules/README.md: symlinks are not allowed")
                path.unlink()

    def test_other_markdown_files_are_rejected(self):
        for name in ("CONTRIBUTING.md", "readme.md", "README.MD"):
            with self.subTest(name=name):
                path = self.root / "modules" / name
                path.write_text("# Other documentation\n", encoding="utf-8")
                self.assert_rejected(f"modules/{name}: only source record JSON files")
                path.unlink()

    def test_only_empty_gitkeep_is_allowed(self):
        path = self.root / "modules" / ".gitkeep"
        path.touch()
        self.assertEqual(0, self.run_validator().returncode)
        path.write_text("not a source record")
        self.assert_rejected(".gitkeep must be empty")

    def test_duplicate_git_sources_ignore_revision_and_url_aliases(self):
        first = example("module")
        first["source"]["url"] = "https://github.com/example/library"
        self.write_record(record=first)
        second = example("module")
        second["source"].update(url="https://GITHUB.com:443/EXAMPLE/LIBRARY.git/", ref="other", path=".")
        self.write_record(name="alias", record=second)
        self.assert_rejected("duplicate source")

    def test_one_repository_can_supply_multiple_kinds_and_distinct_paths(self):
        first = example("module")
        self.write_record(record=first)
        second = example("module")
        second["source"]["path"] = "another-module"
        self.write_record(name="another-module", record=second)
        application = example("application")
        application["source"] = first["source"].copy()
        self.write_record("application", record=application)
        result = self.run_validator()
        self.assertEqual(0, result.returncode, result.stderr)

    def test_duplicate_git_sources_normalize_equivalent_url_paths(self):
        first = example("module")
        first["source"]["url"] = "https://github.com/example/library"
        self.write_record(record=first)
        for alias in (
            "/example/./library", "/example/%6cibrary", "/example/unused/../library",
            "/example/%2e/library", "/example/unused/%2E%2e/library",
            "/example/library%2egit/", "/../example/library",
        ):
            with self.subTest(alias=alias):
                second = example("module")
                second["source"]["url"] = "https://github.com" + alias
                self.write_record(name="alias", record=second)
                self.assert_rejected("duplicate source")

    def test_git_url_normalization_preserves_reserved_characters(self):
        for name, url in (("encoded", "https://example.org/space%2Frepo"), ("literal", "https://example.org/space/repo")):
            record = example("module")
            record["source"]["url"] = url
            self.write_record(name=name, record=record)
        result = self.run_validator()
        self.assertEqual(0, result.returncode, result.stderr)

    def test_invalid_source_url_does_not_overwrite_index(self):
        record = example("benchmark")
        record["source"]["url"] = "https://example.org:99999/repo"
        self.write_record("benchmark", record=record)
        index = self.root / "sources.json"
        index.write_text("previous index")
        self.assert_rejected("source", "--index", index)
        self.assertEqual("previous index", index.read_text())

    def test_entity_metadata_does_not_enter_source_index(self):
        record = example("module")
        record["details"] = {"provides_operators": ["capture"]}
        self.write_record(record=record)
        index = self.root / "sources.json"
        self.assert_rejected("details", "--index", index)
        self.assertFalse(index.exists())

    def test_duplicate_hosted_urls_normalize_host_and_default_port(self):
        first = example("tutorial")
        self.write_record("tutorial", record=first)
        second = example("tutorial")
        second["source"]["url"] = first["source"]["url"].replace("example.org", "EXAMPLE.org:443")
        self.write_record("tutorial", name="alias", record=second)
        self.assert_rejected("duplicate source")

    def test_hosted_url_fragments_queries_and_paths_remain_distinct(self):
        for name, url in (
            ("overview", "https://example.org/tutorial#overview"),
            ("next", "https://example.org/tutorial#next"),
            ("version", "https://example.org/tutorial?version=2#overview"),
            ("case", "https://example.org/Tutorial#overview"),
        ):
            record = example("tutorial")
            record["source"]["url"] = url
            self.write_record("tutorial", name=name, record=record)
        result = self.run_validator()
        self.assertEqual(0, result.returncode, result.stderr)

    def test_errors_from_all_records_are_reported(self):
        (self.root / "modules" / "first.json").write_text("{")
        (self.root / "skills" / "second.json").write_text("{")
        result = self.assert_rejected("modules/first.json")
        self.assertIn("skills/second.json", result.stderr)

    def test_invalid_registry_does_not_create_or_overwrite_index(self):
        (self.root / "modules" / "bad.json").write_text("{")
        index = self.root / "sources.json"
        self.assert_rejected("invalid JSON", "--index", index)
        self.assertFalse(index.exists())
        index.write_text("previous index")
        self.assert_rejected("invalid JSON", "--index", index)
        self.assertEqual("previous index", index.read_text())

    def test_index_cannot_overwrite_registry_inputs(self):
        record_path = self.write_record()
        original = record_path.read_bytes()
        self.assert_rejected("index output cannot overwrite registry inputs", "--index", record_path)
        self.assertEqual(original, record_path.read_bytes())
        self.assert_rejected("index output cannot overwrite registry inputs", "--index", self.schema_path)


if __name__ == "__main__":
    unittest.main()
