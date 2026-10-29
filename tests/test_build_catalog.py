"""Catalog collection with local Git repositories and patched hosted content."""

import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch
from urllib.error import HTTPError

from utilities import build_catalog as catalog
from utilities import validate_sources as sources
from test_validate_sources import metadata


def skill(name):
    return f"---\nname: {name}\ndescription: Example workflow.\n---\n\n# Instructions\nRead the project.\n"


def hosted(files, missing=404):
    """Serve bytes by URL; every other URL fails with the given HTTP status."""
    def fake_urlopen(url, timeout=None):
        if url in files:
            return io.BytesIO(files[url])
        raise HTTPError(url, missing, "Error", {}, None)
    return fake_urlopen


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.registry = self.root / "registry"
        self.output = self.root / "catalog"
        schema = self.registry / "utilities" / "source.schema.json"
        schema.parent.mkdir(parents=True)
        shutil.copyfile(Path(sources.__file__).with_name("source.schema.json"), schema)
        for directory in sources.KIND_DIRECTORIES.values():
            (self.registry / directory).mkdir()

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True, text=True).stdout.strip()

    def commit(self, files):
        """Write files (text, bytes, or JSON objects) into the repository and return the commit."""
        if not (self.repo / ".git").exists():
            self.repo.mkdir(exist_ok=True)
            self.git("init", "--quiet")
            self.git("config", "user.name", "Catalog Test")
            self.git("config", "user.email", "catalog-test@example.org")
        for relative, content in files.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
        self.git("add", "--all")
        self.git("commit", "--quiet", "--allow-empty", "-m", "Catalog fixture")
        return self.git("rev-parse", "HEAD")

    def record(self, kind, name, **source):
        return {"name": name, "kind": kind, "source": {"type": "git", "url": str(self.repo), **source}}

    def collect(self, record):
        """Collect one Git source into the output directory the way main() does."""
        with tempfile.TemporaryDirectory() as directory:
            source = sources.open_source(record, Path(directory))
            return catalog.collect_source(record, source, self.output / catalog.source_root(record))

    def register(self, kind, name, url):
        path = self.registry / sources.KIND_DIRECTORIES[kind] / f"{name}.json"
        path.write_text(json.dumps({"name": name, "kind": kind, "source": {"type": "url", "url": url}}), encoding="utf-8")

    def build(self, expected_code=0):
        shutil.rmtree(self.output, ignore_errors=True)
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = catalog.main(["--root", str(self.registry), "--output", str(self.output)])
        self.assertEqual(expected_code, code, out.getvalue() + err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())
        return json.loads((self.output / "catalog.json").read_text()), err.getvalue()

    def mirrored(self, root):
        return sorted(str(path.relative_to(self.output / root)) for path in (self.output / root).rglob("*") if path.is_file())

    def test_module_catalogs_contained_entities_with_variants_and_parents(self):
        commit = self.commit({
            "metadata.json": metadata("module", "holoscan-example"),
            "README.md": "# Module\n",
            "operators/capture/metadata.json": metadata("operator", "Capture Display Name"),
            "operators/capture/README.md": "# Capture\n",
            "applications/viewer/cpp/metadata.json": metadata("application", "Viewer"),
            "applications/viewer/py/metadata.json": metadata("application", "Viewer"),
            "applications/viewer/README.md": "# Viewer\n",
            "benchmarks/latency/metadata.json": metadata("benchmark", "Latency"),
            "tutorials/intro/metadata.json": metadata("tutorial", "Intro"),
            "tutorials/intro/README.md": "# Intro\n",
        })
        record = self.record("module", "holoscan-example", ref=commit)
        source = self.collect(record)
        self.assertEqual(record, source["record"])
        self.assertEqual("modules/holoscan-example", source["root"])
        self.assertEqual(commit, source["commit"])
        self.assertEqual("README.md", source["readme"])
        entities = {entity["path"]: entity for entity in source["entities"]}
        self.assertEqual([".", "applications/viewer", "benchmarks/latency", "operators/capture", "tutorials/intro"], list(entities))
        self.assertEqual({
            "kind": "module", "name": "holoscan-example", "path": ".", "parent": None,
            "descriptors": [{"language": None, "path": "metadata.json"}], "documents": {"readme": "README.md"},
        }, entities["."])
        self.assertEqual({
            "kind": "application", "name": "viewer", "path": "applications/viewer", "parent": ".",
            "descriptors": [
                {"language": "cpp", "path": "applications/viewer/cpp/metadata.json"},
                {"language": "python", "path": "applications/viewer/py/metadata.json"},
            ],
            "documents": {"readme": "applications/viewer/README.md"},
        }, entities["applications/viewer"])
        self.assertEqual(("operator", "capture", "."), tuple(entities["operators/capture"][key] for key in ("kind", "name", "parent")))
        self.assertEqual({}, entities["benchmarks/latency"]["documents"])
        self.assertEqual(sorted(
            [descriptor["path"] for entity in entities.values() for descriptor in entity["descriptors"]]
            + [path for entity in entities.values() for path in entity["documents"].values()]
        ), self.mirrored(source["root"]))
        self.assertEqual("# Viewer\n", (self.output / source["root"] / "applications/viewer/README.md").read_text())

    def test_collection_members_have_no_parent_and_only_entity_files_are_mirrored(self):
        commit = self.commit({
            "README.md": "# Repository\n",
            "tutorials/README.md": "# Tutorials\n",
            "tutorials/basics/metadata.json": metadata("tutorial"),
            "tutorials/basics/README.md": "# Basics\n",
            "tutorials/debugging/README.md": "# Section without metadata\n",
            "tutorials/debugging/cli/metadata.json": metadata("tutorial"),
            "tutorials/debugging/cli/README.md": "# CLI\n",
            "tutorials/debugging/cli/src/README.md": "# Nested\n",
        })
        source = self.collect(self.record("tutorial", "collection", ref=commit, path="tutorials"))
        self.assertEqual("README.md", source["readme"])
        self.assertEqual([("tutorial", "basics", "basics", None), ("tutorial", "cli", "debugging/cli", None)], [
            (entity["kind"], entity["name"], entity["path"], entity["parent"]) for entity in source["entities"]
        ])
        self.assertEqual(
            ["README.md", "basics/README.md", "basics/metadata.json", "debugging/cli/README.md", "debugging/cli/metadata.json"],
            self.mirrored(source["root"]),
        )

    def test_direct_entity_readme_falls_back_to_language_directory(self):
        commit = self.commit({
            "apps/camera/cpp/metadata.json": metadata("application", "Camera"),
            "apps/camera/cpp/README.md": "# C++\n",
            "apps/camera/python/metadata.json": metadata("application", "Camera"),
            "apps/camera/python/README.md": "# Python\n",
        })
        source = self.collect(self.record("application", "camera", ref=commit, path="apps/camera"))
        [entity] = source["entities"]
        self.assertEqual(("application", "camera", "."), (entity["kind"], entity["name"], entity["path"]))
        self.assertEqual({"readme": "cpp/README.md"}, entity["documents"])
        self.assertIsNone(source["readme"])
        self.assertEqual(["cpp/README.md", "cpp/metadata.json", "python/metadata.json"], self.mirrored(source["root"]))

    def test_skills_include_cards_when_present(self):
        commit = self.commit({
            "skills/README.md": "# Skills\n",
            "skills/first/SKILL.md": skill("first") + "Run [the check](scripts/check.sh).\n",
            "skills/first/skill-card.md": "## Description\n![Results](evals/results.png)\n",
            "skills/first/README.md": "# Not collected\n",
            "skills/second/SKILL.md": skill("second"),
        })
        bases = {"raw": "https://raw.example.org/r/abc/", "browse": "https://example.org/r/blob/abc/", "path": "skills/"}
        with patch.object(catalog, "upstream", return_value=bases):
            collection = self.collect(self.record("skill", "collection", ref=commit, path="skills"))
        self.assertEqual("README.md", collection["readme"])
        self.assertEqual({"raw": "https://raw.example.org/r/abc/skills/", "browse": "https://example.org/r/blob/abc/skills/"}, collection["upstream"])
        self.assertEqual("## Description\n![Results](https://raw.example.org/r/abc/skills/first/evals/results.png)\n",
                         (self.output / collection["root"] / "first/skill-card.md").read_text())
        self.assertEqual(skill("first") + "Run [the check](scripts/check.sh).\n", (self.output / collection["root"] / "first/SKILL.md").read_text())
        self.assertEqual([
            {"kind": "skill", "name": "first", "path": "first", "parent": None,
             "descriptors": [{"language": None, "path": "first/SKILL.md"}], "documents": {"skill_card": "first/skill-card.md"}},
            {"kind": "skill", "name": "second", "path": "second", "parent": None,
             "descriptors": [{"language": None, "path": "second/SKILL.md"}], "documents": {}},
        ], collection["entities"])
        self.assertEqual(["README.md", "first/SKILL.md", "first/skill-card.md", "second/SKILL.md"], self.mirrored(collection["root"]))
        direct = self.collect(self.record("skill", "first", ref=commit, path="skills/first"))
        [entity] = direct["entities"]
        self.assertEqual(("first", ".", {"skill_card": "skill-card.md"}), (entity["name"], entity["path"], entity["documents"]))
        with self.assertRaisesRegex(ValueError, "registration name"):
            self.collect(self.record("skill", "other", ref=commit, path="skills/first"))

    def test_declared_module_readme_is_required_and_stays_inside_the_source(self):
        data = metadata("module", "holoscan-example")
        data["module"]["documentation"] = {"readme": "docs/overview.md"}
        commit = self.commit({"metadata.json": data, "docs/overview.md": "# Overview\n", "README.md": "# Root\n"})
        source = self.collect(self.record("module", "holoscan-example", ref=commit))
        self.assertEqual({"readme": "docs/overview.md"}, source["entities"][0]["documents"])
        self.assertEqual("README.md", source["readme"])
        for declared, message in (("docs/missing.md", "declared documentation"), ("../outside.md", "stay inside"), ("/etc/passwd", "stay inside")):
            with self.subTest(declared=declared):
                data["module"]["documentation"]["readme"] = declared
                with self.assertRaisesRegex(ValueError, message):
                    self.collect(self.record("module", "holoscan-example", ref=self.commit({"metadata.json": data})))

    def test_relative_references_in_documents_resolve_to_upstream_urls(self):
        raw, browse = "https://raw.example.org/repo/abc123/", "https://example.org/repo/blob/abc123/"
        lines = [
            ("![Pipeline](images/pipeline.png)", f"![Pipeline]({raw}module/images/pipeline.png)"),
            ('![Spaced](<images/with space.png> "title")', f'![Spaced](<{raw}module/images/with%20space.png> "title")'),
            ("![Encoded](images/encoded%20name.png) ![Parens](images/latency%20(ms).png)",
             f"![Encoded]({raw}module/images/encoded%20name.png) ![Parens]({raw}module/images/latency%20(ms).png)"),
            ("[![Build](badges/ci.svg)](https://ci.example.org/run)", f"[![Build]({raw}module/badges/ci.svg)](https://ci.example.org/run)"),
            ("[Guide](docs/GUIDE.md#setup) [License](../LICENSE) [Above](../../outside.md)",
             f"[Guide]({browse}module/docs/GUIDE.md#setup) [License]({browse}LICENSE) [Above](../../outside.md)"),
            ("![Ref][Diagram] [Docs][docs] ![Collapsed][]", "![Ref][Diagram] [Docs][docs] ![Collapsed][]"),
            ("[diagram]: docs/diagram.svg", f"[diagram]: {raw}module/docs/diagram.svg"),
            ("[docs]: <docs/index.md>", f"[docs]: <{browse}module/docs/index.md>"),
            ("[collapsed]: docs/collapsed.svg 'Title'", f"[collapsed]: {raw}module/docs/collapsed.svg 'Title'"),
            ('<img src="images/html.png" width="200"> <a href="docs/API.md">API</a>',
             f'<img src="{raw}module/images/html.png" width="200"> <a href="{browse}module/docs/API.md">API</a>'),
            ("<picture><source srcset=\"images/dark.png 1x, images/dark@2x.png 2x\"><img src='images/light.png'></picture>",
             f"<picture><source srcset=\"{raw}module/images/dark.png 1x, {raw}module/images/dark@2x.png 2x\"><img src='{raw}module/images/light.png'></picture>"),
            ('<video src=images/demo.mp4 poster="images/poster.jpg"></video>',
             f'<video src={raw}module/images/demo.mp4 poster="{raw}module/images/poster.jpg"></video>'),
            ("![Remote](https://example.org/remote.png) [Top](#top) [Mail](mailto:a@example.org) ![Data](data:image/png;base64,AAAA) ![Site](/images/site.png)",
             "![Remote](https://example.org/remote.png) [Top](#top) [Mail](mailto:a@example.org) ![Data](data:image/png;base64,AAAA) ![Site](/images/site.png)"),
            ("```python", "```python"),
            ("callbacks[0](event)  # see [Guide](docs/GUIDE.md)", "callbacks[0](event)  # see [Guide](docs/GUIDE.md)"),
            ("```", "```"),
            ("Use `items[0](arg)` then read [Guide](docs/GUIDE.md).", f"Use `items[0](arg)` then read [Guide]({browse}module/docs/GUIDE.md)."),
        ]
        commit = self.commit({
            "module/metadata.json": metadata("module", "holoscan-example"),
            "module/README.md": "\n".join(written for written, _ in lines) + "\n",
            "module/operators/capture/metadata.json": metadata("operator", "Capture"),
            "module/operators/capture/README.md": "![Shared](../../images/pipeline.png)\n",
        })
        with patch.object(catalog, "upstream", return_value={"raw": raw, "browse": browse, "path": "module/"}):
            source = self.collect(self.record("module", "holoscan-example", ref=commit, path="module"))
        self.assertEqual({"raw": raw + "module/", "browse": browse + "module/"}, source["upstream"])
        self.assertEqual("\n".join(rendered for _, rendered in lines) + "\n", (self.output / source["root"] / "README.md").read_text())
        self.assertEqual(f"![Shared]({raw}module/images/pipeline.png)\n", (self.output / source["root"] / "operators/capture/README.md").read_text())
        self.assertEqual(json.dumps(metadata("module", "holoscan-example")), (self.output / source["root"] / "metadata.json").read_text())

    def test_upstream_urls_for_github_other_hosts_and_hosted_content(self):
        github = {"name": "x", "kind": "module", "source": {"type": "git", "url": "https://github.com/example/repo.git", "ref": "main", "path": "modules/my x"}}
        self.assertEqual({
            "raw": "https://raw.githubusercontent.com/example/repo/abc/",
            "browse": "https://github.com/example/repo/blob/abc/",
            "path": "modules/my%20x/",
        }, catalog.upstream(github, "abc"))
        other = {"name": "x", "kind": "module", "source": {"type": "git", "url": "https://git.example.org:8443/group/sub/repo/", "ref": "v1"}}
        self.assertEqual({
            "raw": "https://git.example.org:8443/group/sub/repo/raw/abc/",
            "browse": "https://git.example.org:8443/group/sub/repo/blob/abc/",
            "path": "",
        }, catalog.upstream(other, "abc"))
        hosted = {"name": "x", "kind": "tutorial", "source": {"type": "url", "url": "https://example.org/tutorials/basics/metadata.json?v=1#x"}}
        self.assertEqual({"raw": "https://example.org/", "browse": "https://example.org/", "path": "tutorials/basics/"}, catalog.upstream(hosted, None))

    def test_conflicting_entities_in_one_directory_are_rejected(self):
        commit = self.commit({"metadata.json": metadata("module", "holoscan-example"), "cpp/metadata.json": metadata("application")})
        with self.assertRaisesRegex(ValueError, "module 'holoscan-example' conflicts with application 'holoscan-example'"):
            self.collect(self.record("module", "holoscan-example", ref=commit))

    def test_collection_templates_are_neither_validated_nor_mirrored(self):
        commit = self.commit({
            "applications/camera-template/metadata.json": metadata("application"),
            "applications/template/{{project}}/metadata.json": "{{ unrendered metadata }}",
            "applications/holoviz/template/{{project}}/metadata.json": "{{ unrendered metadata }}",
        })
        record = self.record("application", "collection", ref=commit, path="applications")
        result = self.collect(record)
        self.assertEqual(["camera-template"], [entity["name"] for entity in result["entities"]])
        self.assertEqual(["camera-template/metadata.json"], self.mirrored(result["root"]))

    def test_module_keeps_benchmark_named_template_outside_modules_directory(self):
        commit = self.commit({
            "metadata.json": metadata(),
            "benchmarks/template/metadata.json": metadata("benchmark"),
        })
        record = self.record("module", "holoscan-example", ref=commit)
        result = self.collect(record)
        self.assertEqual([("module", "."), ("benchmark", "benchmarks/template")], [
            (entity["kind"], entity["path"]) for entity in result["entities"]
        ])

    def test_missing_descriptors_symlinks_and_bad_refs_are_rejected(self):
        commit = self.commit({"docs/README.md": "# No descriptor\n"})
        for message, record in (
            ("no metadata.json found", self.record("module", "empty", ref=commit)),
            ("source path 'nowhere'", self.record("module", "missing-path", ref=commit, path="nowhere")),
            ("git exited", self.record("module", "missing-ref", ref="no-such-ref")),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.collect(record)
        (self.repo / "README.md").symlink_to("docs/README.md")
        commit = self.commit({"metadata.json": metadata("module", "holoscan-linked")})
        with self.assertRaisesRegex(ValueError, "README.md: must be a regular file, not a symlink"):
            self.collect(self.record("module", "holoscan-linked", ref=commit))

    def test_hosted_sources_fetch_optional_documents_in_sorted_order(self):
        files = {
            "https://example.org/mod/metadata.json": json.dumps(metadata("module", "holoscan-hosted-module")).encode(),
            "https://example.org/mod/README.md": b"# Hosted module\n![Flow](images/flow.png) [Shared](../shared/README.md#top)\n",
            "https://example.org/tutorials/basics/metadata.json": json.dumps(metadata("tutorial")).encode(),
            "https://example.org/skills/hosted-skill/SKILL.md": skill("hosted-skill").encode(),
            "https://example.org/skills/hosted-skill/skill-card.md": b"## Card\n",
        }
        self.register("tutorial", "basics", "https://example.org/tutorials/basics/#overview")
        self.register("skill", "hosted-skill", "https://example.org/skills/hosted-skill/SKILL.md")
        self.register("module", "holoscan-hosted-module", "https://example.org/mod/metadata.json")
        with patch.object(sources, "urlopen", side_effect=hosted(files)):
            index, _ = self.build()
        self.assertEqual(catalog.FORMAT, index["format"])
        module, skill_source, tutorial = index["sources"]
        self.assertEqual(["module", "skill", "tutorial"], [source["record"]["kind"] for source in index["sources"]])
        self.assertEqual([None, None, None], [source["commit"] for source in index["sources"]])
        self.assertEqual(("README.md", "holoscan-hosted-module", {"readme": "README.md"}), (
            module["readme"], module["entities"][0]["name"], module["entities"][0]["documents"],
        ))
        self.assertEqual({"raw": "https://example.org/mod/", "browse": "https://example.org/mod/"}, module["upstream"])
        self.assertEqual(
            "# Hosted module\n![Flow](https://example.org/mod/images/flow.png) [Shared](https://example.org/shared/README.md#top)\n",
            (self.output / module["root"] / "README.md").read_text(),
        )
        self.assertEqual(["README.md", "metadata.json"], self.mirrored(module["root"]))
        self.assertEqual((None, "basics", {}), (tutorial["readme"], tutorial["entities"][0]["name"], tutorial["entities"][0]["documents"]))
        self.assertEqual({"skill_card": "skill-card.md"}, skill_source["entities"][0]["documents"])
        self.assertEqual(["SKILL.md", "skill-card.md"], self.mirrored(skill_source["root"]))

    def test_hosted_document_errors_other_than_missing_fail_the_source(self):
        files = {"https://example.org/mod/metadata.json": json.dumps(metadata("module", "holoscan-hosted-module")).encode()}
        self.register("module", "holoscan-hosted-module", "https://example.org/mod/metadata.json")
        with patch.object(sources, "urlopen", side_effect=hosted(files, missing=500)):
            index, errors = self.build(expected_code=1)
        self.assertEqual([], index["sources"])
        self.assertIn("modules/holoscan-hosted-module.json: HTTP Error 500", errors)
        self.assertFalse((self.output / "modules").exists())

    def test_invalid_descriptor_omits_only_its_source(self):
        self.register("application", "good", "https://example.org/good/metadata.json")
        self.register("application", "bad", "https://example.org/bad/metadata.json")
        for content in (b"{", b'{"module": {}, "application": {}}', b'{"application": {"name": "Missing fields"}}', b'{"other": {}}'):
            files = {"https://example.org/good/metadata.json": json.dumps(metadata("application", "Good")).encode(),
                     "https://example.org/bad/metadata.json": content}
            with self.subTest(content=content), patch.object(sources, "urlopen", side_effect=hosted(files)):
                index, errors = self.build(expected_code=1)
                self.assertEqual(["good"], [source["record"]["name"] for source in index["sources"]])
                self.assertIn("applications/bad.json:", errors)
                self.assertFalse((self.output / "applications/bad").exists())
                self.assertTrue((self.output / "applications/good/metadata.json").is_file())

    def test_output_must_be_empty_and_defaults_to_a_temporary_directory(self):
        files = {"https://example.org/mod/metadata.json": json.dumps(metadata("module", "holoscan-hosted-module")).encode()}
        self.register("module", "holoscan-hosted-module", "https://example.org/mod/metadata.json")
        self.output.mkdir()
        (self.output / "stale").write_text("previous build")
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            self.assertEqual(1, catalog.main(["--root", str(self.registry), "--output", str(self.output)]))
        self.assertIn("must be empty", err.getvalue())
        self.assertFalse((self.output / "catalog.json").exists())
        out = io.StringIO()
        with patch.object(sources, "urlopen", side_effect=hosted(files)), redirect_stdout(out):
            self.assertEqual(0, catalog.main(["--root", str(self.registry)]))
        output = Path(out.getvalue().rsplit(" to ", 1)[1].strip())
        self.addCleanup(shutil.rmtree, output, True)
        self.assertEqual(Path(tempfile.gettempdir()), output.parent)
        self.assertTrue(output.name.startswith("holohub-catalog-"))
        self.assertTrue((output / "catalog.json").is_file())

    def test_catalog_is_reproducible(self):
        files = {
            "https://example.org/tutorials/basics/metadata.json": json.dumps(metadata("tutorial")).encode(),
            "https://example.org/apps/viewer/metadata.json": json.dumps(metadata("application", "Viewer")).encode(),
            "https://example.org/apps/viewer/README.md": b"# Viewer\n",
        }
        self.register("tutorial", "basics", "https://example.org/tutorials/basics/metadata.json")
        self.register("application", "viewer", "https://example.org/apps/viewer/")
        with patch.object(sources, "urlopen", side_effect=hosted(files)):
            self.build()
            first = (self.output / "catalog.json").read_bytes()
            self.build()
        self.assertEqual(first, (self.output / "catalog.json").read_bytes())
        self.assertEqual(["application", "tutorial"], [source["record"]["kind"] for source in json.loads(first)["sources"]])

    def test_registry_is_validated_before_fetching(self):
        (self.registry / "modules" / "bad.json").write_text("{")
        err = io.StringIO()
        with patch.object(catalog, "open_source", side_effect=AssertionError("must not fetch")), redirect_stderr(err):
            self.assertEqual(1, catalog.main(["--root", str(self.registry), "--output", str(self.output)]))
        self.assertIn("invalid JSON", err.getvalue())
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
