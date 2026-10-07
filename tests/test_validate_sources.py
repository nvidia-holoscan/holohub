"""Source validation with local Git repositories and the installed Holoscan CLI."""

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from utilities import validate_sources as sources


def registration(kind="module", name="example", **source):
    return {"name": name, "kind": kind, "source": {
        "type": "git", "url": "https://example.org/repo.git", "ref": "main",
        **source,
    }}


def metadata(kind="module", name="example"):
    return {kind: {
        "name": name, "description": "Example project", "version": "1.0.0",
        "authors": [{"name": "Example", "affiliation": "Example"}],
        "language": "Python", "platforms": ["x86_64"], "tags": [],
        "holoscan_sdk": {"minimum_required_version": "4.0.0", "tested_versions": ["4.0.0"]},
        "changelog": {}, "ranking": 1, "requirements": {},
    }}


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "example"
        self.project.mkdir()

    def write_metadata(self, relative="metadata.json", kind="module", name="example"):
        path = self.project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata(kind, name)), encoding="utf-8")
        return path

    def write_skill(self, relative="SKILL.md", name="example", extra=""):
        path = self.project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"---\nname: {name}\ndescription: Example workflow.\n{extra}---\n\n# Instructions\nRead the project.\n",
            encoding="utf-8",
        )
        return path

    def test_real_cli_lists_each_supported_kind(self):
        for kind in ("application", "module", "tutorial", "benchmark"):
            with self.subTest(kind=kind):
                self.write_metadata(kind=kind)
                self.assertEqual([], sources.validate_content(registration(kind), self.project))

    def test_collections_use_entity_selectors_not_display_or_registration_names(self):
        self.write_metadata("camera/python/metadata.json", "application", "Camera Display Name")
        self.write_metadata("viewer/metadata.json", "application", "Viewer Display Name")
        self.assertEqual([], sources.validate_content(
            registration("application", "collection"), self.project,
        ))

    def test_direct_language_subdirectory_keeps_parent_application_selector(self):
        for language in ("python", "cpp", "py"):
            with self.subTest(language=language):
                path = self.write_metadata(f"camera/{language}/metadata.json", "application", "Camera Display Name")
                self.assertEqual([], sources.validate_content(
                    registration("application", "camera"), path.parent,
                ))

    def test_collection_without_registered_kind_is_rejected(self):
        self.write_metadata("camera/metadata.json", "application")
        self.assertIn("registered kind 'module'", "\n".join(sources.validate_content(registration(), self.project)))

    def test_invalid_member_does_not_hide_behind_valid_collection_member(self):
        self.write_metadata("first/metadata.json")
        path = self.write_metadata("second/metadata.json", name="second")
        path.write_text('{"module": {"name": "second"}}')
        self.assertIn("second/metadata.json", "\n".join(sources.validate_content(registration(), self.project)))

    def test_standalone_module_ignores_its_nested_application_metadata(self):
        self.write_metadata()
        self.write_metadata("applications/camera/metadata.json", "application", "Camera")
        self.assertEqual([], sources.validate_content(registration(), self.project))

    def test_wrong_kind_and_ambiguous_envelopes_are_rejected(self):
        path = self.write_metadata(kind="application")
        errors = sources.validate_content(registration(), self.project)
        self.assertIn("kind", "\n".join(errors))
        data = metadata()
        data.update(metadata("application"))
        path.write_text(json.dumps(data))
        self.assertIn("envelope", "\n".join(sources.validate_content(registration(), self.project)))

    def test_metadata_must_pass_upstream_schema(self):
        path = self.write_metadata()
        data = metadata()
        del data["module"]["version"]
        path.write_text(json.dumps(data))
        self.assertIn("version", "\n".join(sources.validate_content(registration(), self.project)))

    def test_direct_entity_name_must_match_registration(self):
        self.write_metadata(name="different")
        self.assertIn("registration name", "\n".join(sources.validate_content(registration(), self.project)))

    def test_absent_malformed_duplicate_and_oversized_metadata_are_rejected(self):
        path = self.project / "metadata.json"
        self.assertIn("metadata.json", "\n".join(sources.validate_content(registration(), self.project)))
        for value in ("{", '{"module": {}, "module": {}}', " " * (sources.MAX_FILE_BYTES + 1)):
            with self.subTest(value=value[:40]):
                path.write_text(value)
                self.assertTrue(sources.validate_content(registration(), self.project))

    def test_symlink_metadata_is_rejected(self):
        target = self.root / "metadata.json"
        target.write_text(json.dumps(metadata()))
        (self.project / "metadata.json").symlink_to(target)
        self.assertIn("symlink", "\n".join(sources.validate_content(registration(), self.project)))

    def test_cli_must_report_same_name_kind_and_source_folder(self):
        self.write_metadata()
        expected = {"name": "example", "project_type": "module", "source_folder": str(self.project)}
        for projects in ([], [{**expected, "name": "other"}],
                         [{**expected, "project_type": "application"}],
                         [{**expected, "source_folder": str(self.root)}]):
            with self.subTest(projects=projects), patch.object(sources, "list_projects", return_value=projects):
                self.assertIn("holoscan list", "\n".join(sources.validate_content(registration(), self.project)))

    def test_cli_failure_is_reported(self):
        self.write_metadata()
        with patch.object(sources, "list_projects", side_effect=ValueError("holoscan list failed")):
            self.assertIn("holoscan list failed", "\n".join(sources.validate_content(registration(), self.project)))

    def test_cli_environment_cannot_override_discovery(self):
        self.write_metadata()
        with patch.dict(os.environ, {"HOLOSCAN_CLI_SEARCH_PATH": "/missing", "HOLOSCAN_CLI_APP_NAME": "wrong"}):
            self.assertEqual([], sources.validate_content(registration(), self.project))

    def test_malformed_cli_output_is_rejected(self):
        path = self.write_metadata()
        for output in (b"not json", b"[]", b'{"projects": {}}', b'{"projects": [{}]}'):
            with self.subTest(output=output), patch.object(sources, "run_command", return_value=output):
                with self.assertRaises(ValueError):
                    sources.list_projects(self.project, [path])

    def test_skills_need_no_metadata_or_cli(self):
        self.write_skill()
        with patch.object(sources, "list_projects", side_effect=AssertionError("Skills must not invoke CLI")):
            self.assertEqual([], sources.validate_content(registration("skill"), self.project))

    def test_skill_collection_checks_every_skill(self):
        self.write_skill("first/SKILL.md", "first")
        self.write_skill("second/SKILL.md", "second")
        self.assertEqual([], sources.validate_content(registration("skill", "collection"), self.project))
        self.write_skill("second/SKILL.md", "wrong")
        self.assertIn("second/SKILL.md", "\n".join(sources.validate_content(registration("skill"), self.project)))

    def test_skill_name_matches_direct_registration(self):
        self.write_skill()
        self.assertIn("registration name", "\n".join(sources.validate_content(
            registration("skill", "other"), self.project,
        )))

    def test_invalid_skill_files_are_rejected(self):
        path = self.project / "SKILL.md"
        self.assertTrue(sources.validate_content(registration("skill"), self.project))
        for content in (
            "# No frontmatter", "---\nname: example\n", "---\n[]\n---\nBody",
            "---\nname: example\ndescription: [\n---\nBody",
            "---\nname: example\nname: example\ndescription: Example\n---\nBody",
            "---\nname: example\ndescription: ' '\n---\nBody",
            "---\nname: example\ndescription: Example\n---\n",
            "---\nname: Wrong_Name\ndescription: Example\n---\nBody",
            "---\nname: example\ndescription: Example\nmetadata: wrong\n---\nBody",
            "---\nname: example\ndescription: Example\nmetadata:\n  kind: module\n---\nBody",
        ):
            with self.subTest(content=content):
                path.write_text(content)
                self.assertTrue(sources.validate_content(registration("skill"), self.project))

    def test_skill_extensions_are_allowed_but_identity_must_agree(self):
        self.write_skill(extra='version: "1.0"\nmetadata:\n  tags: [holoscan]\n  name: example\n  kind: skill\n')
        self.assertEqual([], sources.validate_content(registration("skill"), self.project))
        self.write_skill(extra="metadata:\n  name: wrong\n")
        self.assertIn("name", "\n".join(sources.validate_content(registration("skill"), self.project)))

    def test_hosted_files_and_directory_urls(self):
        for kind, filename, body in (
            ("module", "metadata.json", json.dumps(metadata()).encode()),
            ("skill", "SKILL.md", b"---\nname: example\ndescription: Example\n---\nInstructions"),
        ):
            for url in (f"https://example.org/example/{filename}?v=1#section", "https://example.org/example/?v=1"):
                with self.subTest(kind=kind, url=url), patch.object(sources, "urlopen", return_value=io.BytesIO(body)):
                    record = {"name": "example", "kind": kind, "source": {"type": "url", "url": url}}
                    with tempfile.TemporaryDirectory() as tmp:
                        content = sources.fetch_source(record, Path(tmp))
                        self.assertEqual([], sources.validate_content(record, content))
        self.assertEqual("https://example.org/example/metadata.json?v=1", sources.descriptor_url(
            "https://example.org/example?v=1#section", "metadata.json",
        ))

    def test_hosted_errors_and_excessive_content_fail(self):
        record = {"name": "example", "kind": "module", "source": {"type": "url", "url": "https://example.org/metadata.json"}}
        for response in (OSError("not found"), io.BytesIO(b" " * (sources.MAX_FILE_BYTES + 1))):
            kwargs = {"side_effect": response} if isinstance(response, Exception) else {"return_value": response}
            with self.subTest(response=response), patch.object(sources, "urlopen", **kwargs):
                with self.assertRaises((OSError, ValueError)):
                    sources.fetch_source(record, self.root / "download")

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.project), *args], check=True, capture_output=True, text=True).stdout.strip()

    def make_repository(self):
        self.git("init", "--quiet")
        self.git("config", "user.name", "Source Test")
        self.git("config", "user.email", "source-test@example.org")
        self.write_metadata("modules/example/metadata.json")
        self.git("add", ".")
        self.git("commit", "--quiet", "-m", "Valid source")
        return self.git("rev-parse", "HEAD")

    def test_git_uses_exact_revision_and_subdirectory(self):
        revision = self.make_repository()
        self.write_metadata("modules/example/metadata.json", name="later")
        self.git("commit", "--quiet", "-am", "Later change")
        record = registration(url=str(self.project), ref=revision, path="modules/example")
        content = sources.fetch_source(record, self.root / "fetched")
        self.assertEqual([], sources.validate_content(record, content))
        self.assertEqual("example", json.loads((content / "metadata.json").read_text())["module"]["name"])

    def test_git_branch_tag_and_root_default_are_supported(self):
        self.make_repository()
        self.write_metadata()
        self.git("add", ".")
        self.git("commit", "--quiet", "-m", "Root module")
        self.git("branch", "source-test")
        self.git("tag", "v1")
        for ref in ("source-test", "v1"):
            with self.subTest(ref=ref), tempfile.TemporaryDirectory() as tmp:
                record = registration(url=str(self.project), ref=ref)
                content = sources.fetch_source(record, Path(tmp))
                self.assertEqual([], sources.validate_content(record, content))

    def test_git_refspecs_cannot_select_or_rewrite_revisions(self):
        self.make_repository()
        self.write_metadata()
        self.git("add", ".")
        self.git("commit", "--quiet", "-m", "Root module")
        for ref in ("HEAD:refs/heads/copied", "refs/heads/*:refs/heads/*"):
            with self.subTest(ref=ref), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "ref"):
                    sources.fetch_source(registration(url=str(self.project), ref=ref), Path(tmp))

    def test_git_rejects_missing_ref_path_and_symlink_descriptor(self):
        revision = self.make_repository()
        for ref, path in (("missing-ref", "."), (revision, "missing-path")):
            with self.subTest(ref=ref, path=path), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    sources.fetch_source(registration(url=str(self.project), ref=ref, path=path), Path(tmp))
        (self.project / "metadata.json").symlink_to("modules/example/metadata.json")
        self.git("add", ".")
        self.git("commit", "--quiet", "-m", "Symlink")
        with self.assertRaisesRegex(ValueError, "symlink"):
            sources.fetch_source(registration(url=str(self.project), ref="HEAD"), self.root / "symlink")

    def make_registry(self):
        schema = self.root / "utilities/source.schema.json"
        schema.parent.mkdir()
        shutil.copyfile(Path(sources.__file__).with_name("source.schema.json"), schema)
        for directory in sources.KIND_DIRECTORIES.values():
            (self.root / directory).mkdir()

    def test_main_reports_all_bad_sources_and_exits_nonzero(self):
        self.make_registry()
        for name in ("first", "second"):
            record = {"name": name, "kind": "module", "source": {
                "type": "url", "url": f"https://example.org/{name}/metadata.json",
            }}
            (self.root / "modules" / f"{name}.json").write_text(json.dumps(record))
        errors = io.StringIO()
        with patch.object(sources, "urlopen", side_effect=OSError("unreachable")), redirect_stdout(io.StringIO()), redirect_stderr(errors):
            self.assertEqual(1, sources.main(["--root", str(self.root)]))
        self.assertIn("modules/first.json: unreachable", errors.getvalue())
        self.assertIn("modules/second.json: unreachable", errors.getvalue())

    def test_main_validates_registry_before_fetching(self):
        self.make_registry()
        (self.root / "modules" / "bad.json").write_text("{")
        errors = io.StringIO()
        with patch.object(sources, "fetch_source", side_effect=AssertionError("must not fetch")), redirect_stderr(errors):
            self.assertEqual(1, sources.main(["--root", str(self.root)]))
        self.assertIn("invalid JSON", errors.getvalue())

    def test_main_success_for_hosted_skill(self):
        self.make_registry()
        record = {"name": "example", "kind": "skill", "source": {
            "type": "url", "url": "https://example.org/example/SKILL.md",
        }}
        (self.root / "skills/example.json").write_text(json.dumps(record))
        body = self.write_skill().read_bytes()
        output = io.StringIO()
        with patch.object(sources, "urlopen", return_value=io.BytesIO(body)), redirect_stdout(output):
            self.assertEqual(0, sources.main(["--root", str(self.root)]))
        self.assertIn("Validated content for 1 sources", output.getvalue())


if __name__ == "__main__":
    unittest.main()
