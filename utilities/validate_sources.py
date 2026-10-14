#!/usr/bin/env python3
"""Retrieve registered descriptors and check metadata, skills, and CLI discovery."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

from holoscan_cli.metadata.metadata_validator import KNOWN_ENVELOPES, validate_json
import yaml

if __package__:
    from .validate_registry import KIND_DIRECTORIES, reject_constant, unique_object, validate_registry
else:
    from validate_registry import KIND_DIRECTORIES, reject_constant, unique_object, validate_registry


MAX_FILE_BYTES = 1024 * 1024  # Match the CLI's metadata discovery limit.
COMMAND_TIMEOUT = 120
HTTP_TIMEOUT = 30
DESCRIPTOR_FILES = ("metadata.json", "SKILL.md")
# A language subdirectory belongs to its parent project, as in the CLI.
LANGUAGE_DIRECTORIES = {"cpp": "cpp", "python": "python", "py": "python"}


def descriptor_filename(kind):
    return "SKILL.md" if kind == "skill" else "metadata.json"


def run_command(command, *, cwd, env=None):
    try:
        result = subprocess.run(
            command, cwd=cwd, env=env, capture_output=True, timeout=COMMAND_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"{command[0]} failed: {error}") from error
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"{command[0]} exited {result.returncode}: {detail}")
    return result.stdout


def hosted_url(url, filename):
    """Locate a file beside a hosted descriptor or inside a hosted directory, preserving the query."""
    parts = urlsplit(url)
    path = parts.path
    for descriptor in DESCRIPTOR_FILES:
        path = path.removesuffix("/" + descriptor)
    path = path.rstrip("/") + "/" + filename
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def read_bounded(stream):
    content = stream.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise ValueError(f"content exceeds {MAX_FILE_BYTES} bytes")
    return content


class GitSource:
    """One directory of a repository at one revision, read from Git objects without a checkout."""

    def __init__(self, source, store):
        # No checkout, hooks, submodules, or credential prompts. A fresh bare store
        # also keeps source repository configuration out of the validation process.
        self._store = store
        self._env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        self._env.update(GIT_TERMINAL_PROMPT="0", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        store.mkdir()
        # fetch accepts refspecs as well as exact revisions. Do not let a wildcard
        # or source:destination mapping silently select a different FETCH_HEAD.
        try:
            self._run("check-ref-format", "--allow-onelevel", source["ref"])
        except ValueError as error:
            raise ValueError(f"source ref {source['ref']!r} must name a branch, tag, or commit, not a refspec") from error
        self._run("init", "--bare", "--quiet")
        self._run("remote", "add", "origin", source["url"])
        self._run("config", "remote.origin.promisor", "true")
        self._run("config", "remote.origin.partialclonefilter", "blob:none")
        self._run("fetch", "--quiet", "--depth=1", "--filter=blob:none", "--no-tags",
                  "--no-recurse-submodules", "origin", source["ref"])
        path = source.get("path", ".")
        tree = "FETCH_HEAD^{tree}" if path == "." else f"FETCH_HEAD:{path}"
        try:
            is_tree = self._run("cat-file", "-t", tree).strip() == b"tree"
        except ValueError:
            is_tree = False
        if not is_tree:
            raise ValueError(f"source path {path!r} must be a directory at ref {source['ref']!r}")
        self.commit = self._run("rev-parse", "FETCH_HEAD").decode("ascii").strip()
        self.entries = {}
        for entry in self._run("ls-tree", "-r", "-z", tree).split(b"\0"):
            if not entry:
                continue
            header, raw_path = entry.split(b"\t", 1)
            relative = PurePosixPath(raw_path.decode("utf-8"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"invalid path in source tree: {relative}")
            self.entries[str(relative)] = header.decode("ascii").split()

    def _run(self, *args):
        return run_command(["git", "-c", "core.hooksPath=/dev/null", *args], cwd=self._store, env=self._env)

    def find(self, filename):
        """Sorted paths of every file with this name under the source path."""
        return sorted(path for path in self.entries if PurePosixPath(path).name == filename)

    def read(self, relative):
        try:
            mode, object_type, oid = self.entries[relative]
        except KeyError:
            raise ValueError(f"{relative}: not found under source path") from None
        if mode not in ("100644", "100755") or object_type != "blob":
            raise ValueError(f"{relative}: must be a regular file, not a symlink or submodule")
        if int(self._run("cat-file", "-s", oid)) > MAX_FILE_BYTES:
            raise ValueError(f"{relative}: exceeds {MAX_FILE_BYTES} bytes")
        return self._run("cat-file", "blob", oid)

    def read_optional(self, relative):
        return self.read(relative) if relative in self.entries else None


class HostedSource:
    """A single hosted entity whose files sit beside the registered descriptor URL."""

    commit = None

    def __init__(self, source):
        self.url = source["url"]

    def find(self, filename):
        return [filename]

    def read(self, relative):
        with urlopen(hosted_url(self.url, relative), timeout=HTTP_TIMEOUT) as response:
            return read_bounded(response)

    def read_optional(self, relative):
        try:
            return self.read(relative)
        except HTTPError as error:
            if error.code in (404, 410):
                return None
            raise


def open_source(record, destination):
    """Open a registered source for reading; never check out or execute source code."""
    source = record["source"]
    if source["type"] == "url":
        return HostedSource(source)
    destination.mkdir(parents=True, exist_ok=True)
    return GitSource(source, destination / "repository.git")


def fetch_source(record, destination):
    """Materialize only descriptors, never check out or execute source code."""
    source = open_source(record, destination)
    if isinstance(source, HostedSource):
        content_root = destination / record["name"]
    else:
        # Preserve actual directory names for CLI selectors. For a repository-root
        # source, the repository name is the normal clone directory name.
        repo_name = PurePosixPath(urlsplit(record["source"]["url"]).path).name.removesuffix(".git")
        path = record["source"].get("path", ".")
        content_root = destination / "content" / repo_name
        if path != ".":
            content_root = content_root / path
    content_root.mkdir(parents=True, exist_ok=True)
    for relative in source.find(descriptor_filename(record["kind"])):
        target = content_root.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read(relative))
    return content_root


def read_descriptor(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("descriptor must be a regular file, not a symlink")
    with path.open("rb") as stream:
        return read_bounded(stream).decode("utf-8")


def parse_json(content):
    return json.loads(content, object_pairs_hook=unique_object, parse_constant=reject_constant)


class UniqueSafeLoader(yaml.SafeLoader):
    """Safe YAML with duplicate keys rejected instead of silently overwritten."""

    def construct_mapping(self, node, deep=False):
        pairs = [(self.construct_object(key, deep=deep), self.construct_object(value, deep=deep))
                 for key, value in node.value]
        return unique_object(pairs)


def metadata_kind(data):
    """The single recognized kind envelope of a metadata object."""
    envelopes = [key for key in KNOWN_ENVELOPES if isinstance(data, dict) and key in data]
    if len(envelopes) != 1:
        raise ValueError("metadata must have exactly one recognized kind envelope")
    return envelopes[0]


def validate_metadata(data, kind):
    """Check metadata against the schema bundled with the pinned Holoscan CLI."""
    valid, detail = validate_json(data, kind)
    if not valid:
        raise ValueError(f"invalid {kind} metadata: {detail.message}")


def validate_skill(path, registration_name=None):
    """Check SKILL.md frontmatter and instructions; return the skill name."""
    text = read_descriptor(path)
    lines = text.splitlines()
    if not lines or lines[0] != "---" or "---" not in lines[1:]:
        raise ValueError("SKILL.md must start with delimited YAML frontmatter")
    end = lines.index("---", 1)
    frontmatter = yaml.load("\n".join(lines[1:end]), Loader=UniqueSafeLoader)
    if not isinstance(frontmatter, dict):
        raise ValueError("SKILL.md frontmatter must be a mapping")
    name = frontmatter.get("name")
    if not isinstance(name, str) or len(name) > 64 or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ValueError("skill name must use 1-64 lowercase letters, digits, and single hyphens")
    if registration_name is not None and name != registration_name:
        raise ValueError(f"skill name {name!r} does not match registration name {registration_name!r}")
    if name != path.parent.name:
        raise ValueError(f"skill name {name!r} does not match directory {path.parent.name!r}")
    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
        raise ValueError("skill description must be a nonempty string of at most 1024 characters")
    if not "\n".join(lines[end + 1:]).strip():
        raise ValueError("SKILL.md must contain instructions after its frontmatter")
    for key in ("license", "compatibility", "allowed-tools"):
        if key in frontmatter and (not isinstance(frontmatter[key], str) or not frontmatter[key].strip()):
            raise ValueError(f"skill {key} must be a nonempty string")
    if len(frontmatter.get("compatibility", "")) > 500:
        raise ValueError("skill compatibility must be at most 500 characters")
    metadata = frontmatter.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("skill metadata must be a mapping")
    # Allow agent-specific extensions, but reject contradictory identity fields.
    for mapping in (frontmatter, metadata):
        if mapping.get("name", name) != name:
            raise ValueError("skill metadata name must agree with the frontmatter name")
        if mapping.get("kind", "skill") != "skill":
            raise ValueError("skill metadata kind must be 'skill'")
    return name


def list_projects(root, descriptors):
    """Run the installed CLI with only this source's metadata in its search path."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("HOLOSCAN_")}
    # Explicit descriptor paths support collections outside conventional folders,
    # tutorials/benchmarks at the root, and both application language variants.
    if any("," in str(path) for path in descriptors):
        raise ValueError("holoscan list search paths cannot contain commas")
    env["HOLOSCAN_CLI_SEARCH_PATH"] = ",".join(str(path.resolve()) for path in descriptors)
    # A language directory belongs to its parent application. Selecting the
    # language as a standalone project would rename its selector to 'python'.
    project_root = root.parent if root.name in LANGUAGE_DIRECTORIES else root
    output = run_command(
        [sys.executable, "-I", "-m", "holoscan_cli", "--project-root", str(project_root), "list", "--json"],
        cwd=root, env=env,
    )
    result = parse_json(output)
    if not isinstance(result, dict) or not isinstance(result.get("projects"), list):
        raise ValueError("holoscan list --json did not return a projects array")
    for project in result["projects"]:
        if not isinstance(project, dict) or any(
            not isinstance(project.get(key), str) for key in ("name", "project_type", "source_folder")
        ):
            raise ValueError("holoscan list --json returned an invalid project")
    return result["projects"]


def validate_content(record, root):
    """Check a direct entity or every matching entity in a source collection."""
    errors = []
    kind = record["kind"]
    filename = descriptor_filename(kind)
    direct = root / filename
    is_direct = direct.exists() or direct.is_symlink()
    paths = [direct] if is_direct else sorted(root.rglob(filename))
    if not paths:
        return [f"no {filename} found under source path"]
    expected = []
    descriptors = []
    for path in paths:
        label = path.relative_to(root).as_posix()
        try:
            if kind == "skill":
                validate_skill(path, record["name"] if is_direct else None)
                continue
            data = parse_json(read_descriptor(path))
            actual_kind = metadata_kind(data)
            if actual_kind != kind:
                if is_direct:
                    raise ValueError(f"metadata kind {actual_kind!r} does not match registered kind {kind!r}")
                # A collection may contain operators and other supporting entities.
                continue
            validate_metadata(data, kind)
            name = data[kind]["name"].strip() if kind == "module" else (
                path.parent.parent.name if path.parent.name in LANGUAGE_DIRECTORIES else path.parent.name
            )
            if is_direct and name != record["name"]:
                raise ValueError(f"entity name {name!r} does not match registration name {record['name']!r}")
            descriptors.append(path)
            expected.append((name, kind, str(path.parent.resolve())))
        except (OSError, ValueError, TypeError, RecursionError, yaml.YAMLError) as error:
            errors.append(f"{label}: {error}")
    if kind == "skill":
        return errors
    if not expected:
        if not errors:
            errors.append(f"no metadata.json with registered kind {kind!r} found under source path")
        return errors
    try:
        projects = list_projects(root, descriptors)
        actual = {(project["name"], project["project_type"], project["source_folder"]) for project in projects}
        for name, project_kind, folder in expected:
            if (name, project_kind, folder) not in actual:
                errors.append(f"holoscan list did not return {name!r} under kind {project_kind!r} from {folder}")
    except (OSError, ValueError, TypeError, RecursionError) as error:
        errors.append(f"holoscan list failed: {error}")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="registry root")
    args = parser.parse_args(argv)
    records, errors = validate_registry(args.root.resolve())
    if not errors:
        for record in records:
            label = f"{KIND_DIRECTORIES[record['kind']]}/{record['name']}.json"
            print(f"Checking {label} ...", flush=True)
            try:
                with tempfile.TemporaryDirectory(prefix="holohub-source-") as directory:
                    content = fetch_source(record, Path(directory))
                    problems = validate_content(record, content)
            except (OSError, ValueError, TypeError, RecursionError) as error:
                problems = [str(error)]
            errors.extend(f"{label}: {problem}" for problem in problems)
            if not problems:
                print(f"Validated {label}", flush=True)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"Source validation failed with {len(errors)} error(s).", file=sys.stderr)
        return 1
    print(f"Validated content for {len(records)} sources.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
