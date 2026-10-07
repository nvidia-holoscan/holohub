"""Contract tests for source registrations, without fetching source content."""

import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "utilities" / "source.schema.json"
KINDS = ("application", "module", "tutorial", "skill", "benchmark")


def example(kind):
    return json.loads((ROOT / "tests" / "fixtures" / f"{kind}.json").read_text())


class SourceSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert SCHEMA_PATH.is_file(), "utilities/source.schema.json must define the registry contract"
        cls.schema = json.loads(SCHEMA_PATH.read_text())
        cls.validator = Draft202012Validator(cls.schema, format_checker=FormatChecker())

    def assert_valid(self, record):
        self.assertEqual([], list(self.validator.iter_errors(record)))

    def assert_invalid(self, record):
        self.assertTrue(list(self.validator.iter_errors(record)), record)

    def test_schema_is_valid_draft_2020_12(self):
        self.assertEqual("https://json-schema.org/draft/2020-12/schema", self.schema["$schema"])
        Draft202012Validator.check_schema(self.schema)

    def test_every_kind_has_a_valid_example(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.assert_valid(example(kind))

    def test_same_location_fields_work_for_every_kind(self):
        for kind in KINDS:
            for source in (
                {"type": "git", "url": "https://example.org/repo", "ref": "main"},
                {"type": "git", "url": "https://example.org/repo", "ref": "v1", "path": "projects/example"},
                {"type": "url", "url": "https://example.org/entity/metadata.json"},
            ):
                with self.subTest(kind=kind, source=source):
                    self.assert_valid({
                        "name": "minimal", "kind": kind,
                        "source": source,
                    })

    def test_required_fields_cannot_be_omitted(self):
        for field in ("name", "kind", "source"):
            record = example("module")
            del record[field]
            with self.subTest(field=field):
                self.assert_invalid(record)

    def test_issues_url_works_for_every_kind_and_source_type(self):
        for kind in KINDS:
            for source in (
                {"type": "git", "url": "https://example.org/repo", "ref": "main"},
                {"type": "url", "url": "https://example.org/content"},
            ):
                for url in (
                    "https://github.com/example/project/issues",
                    "https://support.example.org/report?project=example#bug",
                    "https://support.example.org:8443/issues",
                ):
                    with self.subTest(kind=kind, source=source, url=url):
                        self.assert_valid({
                            "name": "example", "kind": kind,
                            "source": source, "issues_url": url,
                        })

    def test_issues_url_requires_an_absolute_https_url_without_credentials(self):
        for value in (
            None, 42, {}, [], "", "not-a-url", "./issues",
            "http://example.org/issues", "mailto:support@example.org",
            "javascript:alert(1)", "https:///issues", "https://:443/issues",
            "https://user:password@example.org/issues", "https://example.org/a b",
            "https://example.org/issues\n", "https://example.org:0/issues",
            "https://example.org:65536/issues",
        ):
            with self.subTest(value=value):
                record = example("application")
                record["issues_url"] = value
                self.assert_invalid(record)

    def test_unknown_kinds_fields_and_names_are_rejected(self):
        for field, value in (
            ("kind", "operator"),
            ("name", "Example"), ("name", "../outside"), ("name", ""),
            ("build", {"command": "make"}),
        ):
            with self.subTest(field=field, value=value):
                record = example("module")
                record[field] = value
                self.assert_invalid(record)

    def test_git_sources_require_url_and_revision(self):
        for field in ("url", "ref", "type"):
            record = example("module")
            del record["source"][field]
            with self.subTest(field=field):
                self.assert_invalid(record)
        for revision in ("", " ", "main\n", "-option"):
            record = example("module")
            record["source"]["ref"] = revision
            with self.subTest(revision=revision):
                self.assert_invalid(record)

    def test_urls_must_be_absolute_https_without_credentials(self):
        for url in (
            "not-a-url", "./repo", "file:///tmp/repo", "http://example.org/repo",
            "ssh://example.org/repo", "https:///missing-host", "https://example.org/a b",
            "https://user:password@example.org/repo", "https://example.org/\n",
        ):
            for kind in ("module", "tutorial"):
                with self.subTest(url=url, kind=kind):
                    record = example(kind)
                    record["source"]["url"] = url
                    self.assert_invalid(record)

    def test_git_urls_do_not_embed_query_or_fragment_revisions(self):
        for suffix in ("?ref=main", "#main"):
            record = example("module")
            record["source"]["url"] += suffix
            with self.subTest(suffix=suffix):
                self.assert_invalid(record)

    def test_content_urls_require_a_host_and_valid_port(self):
        for url in ("https://:443/path", "https://example.org:0/path", "https://example.org:65536/path", "https://example.org:99999/path"):
            with self.subTest(url=url):
                record = example("tutorial")
                record["source"]["url"] = url
                self.assert_invalid(record)
        for url in ("https://example.org:1/path", "https://example.org:65535/path", "https://example.org:00443/path", "https://[2001:db8::1]:443/path"):
            with self.subTest(url=url):
                record = example("tutorial")
                record["source"]["url"] = url
                self.assert_valid(record)

    def test_git_paths_are_relative_and_canonical(self):
        for path in ("../outside", "/absolute", "a/../b", "a/./b", "a//b", "a/", "", "C:/temp", "a\\b"):
            with self.subTest(path=path):
                record = example("application")
                record["source"]["path"] = path
                self.assert_invalid(record)
        record = example("application")
        record["source"]["path"] = "."
        self.assert_valid(record)

    def test_hosted_sources_cannot_have_git_fields(self):
        for field, value in (("ref", "main"), ("path", "."), ("command", "sh")):
            record = example("tutorial")
            record["source"][field] = value
            with self.subTest(field=field):
                self.assert_invalid(record)

    def test_entity_metadata_is_rejected_for_every_kind(self):
        for kind in KINDS:
            for field, value in (
                ("description", "Entity description"), ("tags", ["video"]),
                ("nvidia_quality_score", 3), ("details", {"provides_operators": ["capture"]}),
                ("provides_operators", ["capture"]), ("entrypoint", "README.md"),
                ("results_url", "https://example.org/results"),
            ):
                with self.subTest(kind=kind, field=field):
                    record = {
                        "name": "example", "kind": kind,
                        "source": {"type": "git", "url": "https://example.org/repo", "ref": "main"},
                        field: value,
                    }
                    self.assert_invalid(record)


if __name__ == "__main__":
    unittest.main()
