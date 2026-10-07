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


def descriptor_url(url, filename):
    """Accept a descriptor URL or a hosted directory, preserving its query."""
    parts = urlsplit(url)
    path = parts.path
    if not path.endswith("/" + filename):
        path = path.rstrip("/") + "/" + filename
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def read_bounded(stream):
    content = stream.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise ValueError(f"descriptor exceeds {MAX_FILE_BYTES} bytes")
    return content


def fetch_source(record, destination):
    """Materialize only descriptors, never check out or execute source code."""
    destination.mkdir(parents=True, exist_ok=True)
    source = record["source"]
    filename = "SKILL.md" if record["kind"] == "skill" else "metadata.json"
    if source["type"] == "url":
        content_root = destination / record["name"]
        content_root.mkdir(exist_ok=True)
        with urlopen(descriptor_url(source["url"], filename), timeout=HTTP_TIMEOUT) as response:
            content = read_bounded(response)
        (content_root / filename).write_bytes(content)
        return content_root

    store = destination / "repository.git"
    store.mkdir()
    # No checkout, hooks, submodules, or credential prompts. A fresh bare store
    # also keeps source repository configuration out of the validation process.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_TERMINAL_PROMPT="0", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)

    def git(*args):
        return run_command(["git", "-c", "core.hooksPath=/dev/null", *args], cwd=store, env=env)

    # fetch accepts refspecs as well as exact revisions. Do not let a wildcard
    # or source:destination mapping silently select a different FETCH_HEAD.
    try:
        git("check-ref-format", "--allow-onelevel", source["ref"])
    except ValueError as error:
        raise ValueError(f"source ref {source['ref']!r} must name a branch, tag, or commit, not a refspec") from error
    git("init", "--bare", "--quiet")
    git("remote", "add", "origin", source["url"])
    git("config", "remote.origin.promisor", "true")
    git("config", "remote.origin.partialclonefilter", "blob:none")
    git("fetch", "--quiet", "--depth=1", "--filter=blob:none", "--no-tags",
        "--no-recurse-submodules", "origin", source["ref"])
    path = source.get("path", ".")
    tree = "FETCH_HEAD^{tree}" if path == "." else f"FETCH_HEAD:{path}"
    if git("cat-file", "-t", tree).strip() != b"tree":
        raise ValueError(f"source path {path!r} must be a directory at ref {source['ref']!r}")

    # Preserve actual directory names for CLI selectors. For a repository-root
    # source, the repository name is the normal clone directory name.
    repo_name = PurePosixPath(urlsplit(source["url"]).path).name.removesuffix(".git")
    checkout = destination / "content" / repo_name
    content_root = checkout if path == "." else checkout / path
    content_root.mkdir(parents=True)
    for entry in git("ls-tree", "-r", "-z", tree).split(b"\0"):
        if not entry:
            continue
        header, raw_path = entry.split(b"\t", 1)
        mode, object_type, oid = header.decode("ascii").split()
        relative = PurePosixPath(raw_path.decode("utf-8"))
        if relative.name != filename:
            continue
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid descriptor path: {relative}")
        if mode not in ("100644", "100755") or object_type != "blob":
            raise ValueError(f"{relative}: descriptor must be a regular file, not a symlink")
        if int(git("cat-file", "-s", oid)) > MAX_FILE_BYTES:
            raise ValueError(f"{relative}: descriptor exceeds {MAX_FILE_BYTES} bytes")
        target = content_root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git("cat-file", "blob", oid))
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


def validate_skill(path, registration_name=None):
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
    if name != path.parent.name:
        raise ValueError(f"skill name {name!r} does not match directory {path.parent.name!r}")
    if registration_name is not None and name != registration_name:
        raise ValueError(f"skill name {name!r} does not match registration name {registration_name!r}")
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
    project_root = root.parent if root.name in ("cpp", "python", "py") else root
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
    filename = "SKILL.md" if kind == "skill" else "metadata.json"
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
            envelopes = [key for key in KNOWN_ENVELOPES if isinstance(data, dict) and key in data]
            if len(envelopes) != 1:
                raise ValueError("metadata must have exactly one recognized kind envelope")
            actual_kind = envelopes[0]
            if actual_kind != kind:
                if is_direct:
                    raise ValueError(f"metadata kind {actual_kind!r} does not match registered kind {kind!r}")
                # A collection may contain operators and other supporting entities.
                continue
            valid, detail = validate_json(data, KIND_DIRECTORIES[kind])
            if not valid:
                raise ValueError(f"invalid {kind} metadata: {detail.message}")
            name = data[kind]["name"].strip() if kind == "module" else (
                path.parent.parent.name if path.parent.name in ("cpp", "python", "py") else path.parent.name
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
