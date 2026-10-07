#!/usr/bin/env python3
"""Validate HoloHub source registrations and optionally write an offline index."""

import argparse
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit, urlunsplit

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError
from referencing import Registry


KIND_DIRECTORIES = {
    "application": "applications",
    "module": "modules",
    "tutorial": "tutorials",
    "skill": "skills",
    "benchmark": "benchmarks",
}
SCHEMA_PATH = Path("utilities") / "source.schema.json"


def unique_object(pairs):
    """Reject duplicate object keys instead of silently taking the last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )


def check_local_references(value):
    """Keep the single source schema self-contained, with no remote resolution."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ("$ref", "$dynamicRef") and not child.startswith("#"):
                raise ValueError("schema references must be local fragments")
            check_local_references(child)
    elif isinstance(value, list):
        for child in value:
            check_local_references(child)


def canonical_git_path(path):
    """Normalize URL dot segments and percent-encoded unreserved characters."""
    def normalize_escape(match):
        character = chr(int(match[1], 16))
        if character in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~":
            return character
        return match[0].upper()

    path = re.sub(r"%([0-9a-fA-F]{2})", normalize_escape, path)
    segments = []
    # An HTTP URL path is empty or absolute. Preserve empty segments so that
    # distinct paths such as /group//repo and /group/repo are not conflated.
    for segment in path.split("/")[1:]:
        if segment == "..":
            if segments:
                segments.pop()
        elif segment != ".":
            segments.append(segment)
    return "/" + "/".join(segments)


def source_identity(record):
    """A ref update changes the existing registration, not its source identity."""
    source = record["source"]
    parts = urlsplit(source["url"])
    host = parts.hostname
    if not host:
        raise ValueError("source URL must have a hostname")
    # Accessing port also checks that it is numeric and in range.
    port = parts.port
    authority = f"[{host}]" if ":" in host else host
    if port is not None and port != 443:
        authority += f":{port}"
    path = parts.path
    if source["type"] == "git":
        path = canonical_git_path(path).rstrip("/")
        # GitHub repository names are case insensitive; other hosts may differ.
        if host == "github.com":
            path = path.lower()
        path = path.removesuffix(".git")
        url = urlunsplit(("https", authority, path, "", ""))
        return record["kind"], "git", url, source.get("path", ".")
    url = urlunsplit(("https", authority, path or "/", parts.query, parts.fragment))
    return record["kind"], "url", url


def validate_registry(root):
    """Return valid records and all diagnostics; never retrieve source content."""
    errors = []
    records = []
    schema_path = root / SCHEMA_PATH
    try:
        if schema_path.is_symlink():
            raise ValueError("symlinks are not allowed")
        schema = read_json(schema_path)
        Draft202012Validator.check_schema(schema)
        if not isinstance(schema, dict) or schema.get("$schema") != Draft202012Validator.META_SCHEMA["$id"]:
            raise ValueError("expected a Draft 2020-12 source schema")
        check_local_references(schema)
    except (OSError, ValueError, SchemaError) as error:
        return [], [f"{SCHEMA_PATH}: {error}"]

    validator = Draft202012Validator(schema, format_checker=FormatChecker(), registry=Registry())
    seen_sources = {}
    for kind, directory_name in KIND_DIRECTORIES.items():
        directory = root / directory_name
        if directory.is_symlink():
            errors.append(f"{directory_name}: symlinks are not allowed")
            continue
        if not directory.is_dir():
            errors.append(f"{directory_name}: missing directory")
            continue
        for path in sorted(directory.iterdir()):
            label = path.relative_to(root).as_posix()
            if path.is_symlink():
                errors.append(f"{label}: symlinks are not allowed")
                continue
            if path.is_file() and path.name == "README.md":
                continue
            if path.is_file() and path.name == ".gitkeep":
                if path.stat().st_size:
                    errors.append(f"{label}: .gitkeep must be empty")
                continue
            if not path.is_file() or path.suffix != ".json":
                errors.append(f"{label}: only source record JSON files, README.md, and an empty .gitkeep are allowed")
                continue
            try:
                record = read_json(path)
            except (OSError, ValueError) as error:
                errors.append(f"{label}: invalid JSON: {error}")
                continue
            record_errors = list(validator.iter_errors(record))
            if record_errors:
                for error in record_errors:
                    pointer = "/" + "/".join(str(part) for part in error.absolute_path)
                    errors.append(f"{label}:{pointer}: {error.message}")
                continue
            if record["kind"] != kind:
                errors.append(f"{label}: kind {record['kind']!r} belongs in {KIND_DIRECTORIES[record['kind']]}/")
            expected_name = f"{record['name']}.json"
            if path.name != expected_name:
                errors.append(f"{label}: filename must be {expected_name}")
            try:
                identity = source_identity(record)
            except ValueError as error:
                errors.append(f"{label}: {error}")
                continue
            if identity in seen_sources:
                errors.append(f"{label}: duplicate source also registered in {seen_sources[identity]}; update that record instead")
            else:
                seen_sources[identity] = label
            records.append(record)
    return records, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="registry root (default: this repository)")
    parser.add_argument("--index", type=Path, help="write a sorted JSON array after successful validation")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    records, errors = validate_registry(root)
    if args.index:
        output = args.index.resolve()
        if output == root / SCHEMA_PATH or any(output.is_relative_to(root / directory) for directory in KIND_DIRECTORIES.values()):
            errors.append("index output cannot overwrite registry inputs")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"Registry validation failed with {len(errors)} error(s).", file=sys.stderr)
        return 1
    if args.index:
        index = sorted(records, key=lambda record: (record["kind"], record["name"]))
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        except OSError as error:
            print(f"Cannot write index: {error}", file=sys.stderr)
            return 1
    print(f"Validated {len(records)} source records across {len(KIND_DIRECTORIES)} kinds.")
    if args.index:
        print(f"Wrote source index to {args.index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
