#!/usr/bin/env python3
"""Fetch registered descriptors and documentation into a catalog directory.

A catalog is a directory that a web catalog can render directly:

    catalog.json                index of every source and entity, described below
    <kind directory>/<name>/    files from one registered source at their upstream paths

Only metadata.json, README.md, SKILL.md, and skill-card.md files are retrieved,
from Git objects or hosted URLs; no source code or media is fetched and nothing
is executed. Every entity under a source path is indexed: a module lists the
operators, applications, benchmarks, and tutorials it contains, and a
collection lists each member. Language subdirectories (cpp, python, py) are
descriptors of one entity.

Entity and file paths are relative to the registered source.path and mirrored
under root, so <catalog>/<root>/<path> is the local copy and <upstream>/<path>
is the original. Sources are sorted by kind and name, entities by path, and
the index is reproducible for pinned sources:

    {
      "format": "urn:holohub:catalog:v1",
      "sources": [{
        "record": {...},                    registration record, unchanged
        "root": "modules/holoscan-deltacast",
        "commit": "36deda4f...",            fetched Git revision; null for hosted content
        "upstream": {                       source directory at that revision
          "raw": "https://raw.githubusercontent.com/deltacasttv/holoscan-modules/36deda4f.../",
          "browse": "https://github.com/deltacasttv/holoscan-modules/blob/36deda4f.../"
        },
        "readme": "README.md",              README at the source root, or null
        "entities": [{
          "kind": "operator",               metadata envelope kind, or "skill"
          "name": "deltacast_videomaster",  CLI selector, or skill name
          "path": "operators/deltacast_videomaster",   entity directory; "." is the source root
          "parent": ".",                    closest enclosing entity, or null
          "descriptors": [{"language": null, "path": "operators/deltacast_videomaster/metadata.json"}],
          "documents": {"readme": "operators/deltacast_videomaster/README.md"}
        }]
      }]
    }

documents holds a readme for metadata entities (the entity directory's
README.md, a language directory's as fallback, or a module's declared
documentation.readme) and a skill_card for skills, only when present.

Descriptors are copied verbatim. Documents keep their content except that
relative Markdown and HTML references are resolved to absolute URLs so they
render without fetching anything else: images and media to raw file content,
links to browsable pages, both at the fetched revision. GitHub raw content is
served from raw.githubusercontent.com; other Git hosts use <repository>/raw/
and <repository>/blob/, and hosted content resolves beside its descriptor URL.
Absolute URLs, anchors, references inside code, and references that leave the
repository are left unchanged.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

import yaml

if __package__:
    from .validate_registry import KIND_DIRECTORIES, validate_registry
    from .validate_sources import (
        LANGUAGE_DIRECTORIES, descriptor_filename, hosted_url, is_ignored_descriptor, metadata_kind, open_source, parse_json,
        source_directory, validate_metadata, validate_skill,
    )
else:
    from validate_registry import KIND_DIRECTORIES, validate_registry
    from validate_sources import (
        LANGUAGE_DIRECTORIES, descriptor_filename, hosted_url, is_ignored_descriptor, metadata_kind, open_source, parse_json,
        source_directory, validate_metadata, validate_skill,
    )


FORMAT = "urn:holohub:catalog:v1"
INDEX = "catalog.json"
README = "README.md"
SKILL_CARD = "skill-card.md"

# Markdown destinations after "](", which may contain balanced parentheses as
# in "latency (ms).png"; [label]: definitions; and the ![alt][label] images
# that make a definition an image rather than a link.
DESTINATION = re.compile(r"\]\(\s*(<[^>]*>|(?:[^\s()]|\([^\s()]*\))+)")
DEFINITION = re.compile(r"^ {0,3}\[([^\]]+)\]:\s*(<[^>]*>|\S+)", re.MULTILINE)
REFERENCE_IMAGE = re.compile(r"!\[([^\]]*)\]\[([^\]]*)\]")
# Inline HTML media and anchors, including responsive srcset candidates.
HTML_TAG = re.compile(r"<(img|source|video|audio|embed|a)\b[^>]*>", re.IGNORECASE)
HTML_ATTRIBUTE = re.compile(r"""\b(src|poster|srcset|href)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))""", re.IGNORECASE)
SRCSET_URL = re.compile(r"(?:^|,)\s*([^\s,]+)")
# Code never holds references: fenced blocks and inline spans.
CODE = re.compile(r"^ {0,3}(`{3,}|~{3,}).*\n[\s\S]*?^ {0,3}\1[`~]*[ \t]*$|(`+)[\s\S]*?\2", re.MULTILINE)
# RFC 3986 path characters that stay literal when a reference is re-encoded.
PATH_SAFE = "/!$&'()*+,;=:@"


def upstream(record, commit):
    """Root URLs for raw content and browsable pages, and the source directory under them."""
    source = record["source"]
    if source["type"] == "url":
        parts = urlsplit(hosted_url(source["url"], ""))
        root = urlunsplit((parts.scheme, parts.netloc, "/", "", ""))
        return {"raw": root, "browse": root, "path": parts.path.lstrip("/")}
    parts = urlsplit(source["url"])
    repository = parts.path.rstrip("/").removesuffix(".git")
    if parts.hostname == "github.com":
        raw = f"https://raw.githubusercontent.com{repository}/{commit}/"
    else:
        raw = f"https://{parts.netloc}{repository}/raw/{commit}/"
    path = source.get("path", ".")
    return {
        "raw": raw,
        "browse": f"https://{parts.netloc}{repository}/blob/{commit}/",
        "path": "" if path == "." else quote(path, safe=PATH_SAFE) + "/",
    }


def resolve_reference(bases, directory, reference, media):
    """Absolute upstream URL for a relative reference from a file in directory, or None."""
    parts = urlsplit(reference.strip())
    if parts.scheme or parts.netloc or not parts.path or parts.path.startswith("/"):
        return None
    root = bases["raw" if media else "browse"]
    base = root + bases["path"] + ("" if directory == PurePosixPath() else quote(str(directory), safe=PATH_SAFE) + "/")
    path = quote(unquote(parts.path), safe=PATH_SAFE)
    url = urljoin(base, urlunsplit(("", "", path, parts.query, parts.fragment)))
    return url if url.startswith(root) else None


def is_image(text, close):
    """Whether the bracket pair closing at close is an image, with '!' before its opener."""
    depth = 0
    for index in range(close, -1, -1):
        if text[index] == "]":
            depth += 1
        elif text[index] == "[":
            depth -= 1
            if depth == 0:
                return index > 0 and text[index - 1] == "!"
    return False


def normalize_label(label):
    return " ".join(label.split()).casefold()


def rewrite_references(text, resolve):
    """Replace relative Markdown and HTML destinations where resolve(reference, media) returns a URL."""
    protected = [match.span() for match in CODE.finditer(text)]
    edits = []

    def edit(start, end, media):
        value = text[start:end]
        if value.startswith("<") and value.endswith(">"):
            start, end, value = start + 1, end - 1, value[1:-1]
        if any(begin <= start < stop for begin, stop in protected) or any(
            begin < end and start < stop for begin, stop, _ in edits
        ):
            return
        url = resolve(value, media)
        if url is not None:
            edits.append((start, end, url))

    for match in DESTINATION.finditer(text):
        edit(match.start(1), match.end(1), is_image(text, match.start()))
    image_labels = {normalize_label(match[2] or match[1]) for match in REFERENCE_IMAGE.finditer(text)}
    for match in DEFINITION.finditer(text):
        edit(match.start(2), match.end(2), normalize_label(match[1]) in image_labels)
    for tag in HTML_TAG.finditer(text):
        for attribute in HTML_ATTRIBUTE.finditer(tag[0]):
            start = tag.start() + attribute.start(attribute.lastindex)
            if attribute[1].lower() == "srcset":
                for candidate in SRCSET_URL.finditer(attribute[attribute.lastindex]):
                    edit(start + candidate.start(1), start + candidate.end(1), True)
            else:
                edit(start, tag.start() + attribute.end(attribute.lastindex), tag[1].lower() != "a")
    for start, end, url in sorted(edits, reverse=True):
        text = text[:start] + url + text[end:]
    return text


class Mirror:
    """Copy files from one source under its catalog root, fetching each path once."""

    def __init__(self, source, root):
        self.source = source
        self.root = root
        self.present = {}

    def path(self, relative):
        return self.root.joinpath(*PurePosixPath(relative).parts)

    def _write(self, relative, content):
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        self.present[relative] = True

    def require(self, relative):
        content = self.source.read(relative)
        self._write(relative, content)
        return content

    def optional(self, relative):
        if relative not in self.present:
            content = self.source.read_optional(relative)
            if content is not None:
                self._write(relative, content)
            self.present[relative] = content is not None
        return self.present[relative]

    def resolve(self, relative, bases):
        """Rewrite a mirrored Markdown document's relative references to upstream URLs."""
        target = self.path(relative)
        text = target.read_text(encoding="utf-8", errors="surrogateescape")
        directory = PurePosixPath(relative).parent
        resolved = rewrite_references(text, lambda reference, media: resolve_reference(bases, directory, reference, media))
        if resolved != text:
            target.write_text(resolved, encoding="utf-8", errors="surrogateescape")


def source_root(record):
    return f"{KIND_DIRECTORIES[record['kind']]}/{record['name']}"


def document_path(directory, declared):
    """Resolve a declared document path relative to its descriptor, inside the source."""
    path = PurePosixPath(declared)
    if not declared or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"documentation path {declared!r} must be relative and stay inside the source")
    return directory / path


def parent_of(path, paths):
    """The closest other entity directory enclosing this one, or None."""
    ancestors = [candidate for candidate in paths if candidate != path and candidate in path.parents]
    return max(ancestors, key=lambda candidate: len(candidate.parts), default=None)


def collect_source(record, source, root):
    """Mirror one source's descriptors and documents under root and return its index entry."""
    kind = record["kind"]
    filename = descriptor_filename(kind)
    paths = [path for path in source.find(filename)
             if kind == "skill" or not is_ignored_descriptor(source_directory(record) / path)]
    if not paths:
        raise ValueError(f"no {filename} found under source path")
    mirror = Mirror(source, root)
    entities = {}
    pending = []
    for relative in paths:
        content = mirror.require(relative)
        descriptor = PurePosixPath(relative)
        directory = descriptor.parent
        language = None
        if kind != "skill" and directory.name in LANGUAGE_DIRECTORIES:
            language, directory = LANGUAGE_DIRECTORIES[directory.name], directory.parent
        is_root = directory == PurePosixPath()
        name = record["name"] if is_root else directory.name
        if kind == "skill":
            entity_kind = "skill"
            name = validate_skill(root.joinpath(*descriptor.parts), record["name"] if is_root else None)
            wanted = [("skill_card", directory / SKILL_CARD, False)]
        else:
            data = parse_json(content.decode("utf-8"))
            entity_kind = metadata_kind(data)
            validate_metadata(data, entity_kind)
            wanted = [("readme", directory / README, False), ("readme", descriptor.parent / README, False)]
            if entity_kind == "module":
                name = data["module"]["name"].strip() or name
                declared = data["module"].get("documentation", {}).get("readme")
                if declared is not None:
                    wanted = [("readme", document_path(descriptor.parent, declared), True)]
        entity = entities.setdefault(str(directory), {
            "kind": entity_kind, "name": name, "path": str(directory), "descriptors": [], "documents": {},
        })
        if (entity["kind"], entity["name"]) != (entity_kind, name):
            raise ValueError(f"{relative}: {entity_kind} {name!r} conflicts with {entity['kind']} {entity['name']!r}")
        entity["descriptors"].append({"language": language, "path": str(descriptor)})
        pending.extend((entity, key, str(path), required) for key, path, required in wanted)
    for entity, key, relative, required in pending:
        if key in entity["documents"]:
            continue
        if mirror.optional(relative):
            entity["documents"][key] = relative
        elif required:
            raise ValueError(f"{entity['path']}: declared documentation {relative!r} not found")
    readme = README if mirror.optional(README) else None
    bases = upstream(record, source.commit)
    documents = {path for entity in entities.values() for path in entity["documents"].values()}
    for relative in sorted(documents | ({readme} - {None})):
        mirror.resolve(relative, bases)
    ordered = sorted(entities.values(), key=lambda entity: PurePosixPath(entity["path"]).parts)
    directories = [PurePosixPath(entity["path"]) for entity in ordered]
    for entity in ordered:
        parent = parent_of(PurePosixPath(entity["path"]), directories)
        entity["parent"] = None if parent is None else str(parent)
    return {
        "record": record,
        "root": source_root(record),
        "commit": source.commit,
        "upstream": {key: bases[key] + bases["path"] for key in ("raw", "browse")},
        "readme": readme,
        "entities": ordered,
    }


def prepare_output(path):
    if path is None:
        return Path(tempfile.mkdtemp(prefix="holohub-catalog-"))
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f"output directory {path} must be empty")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="registry root")
    parser.add_argument("--output", type=Path, help="new or empty catalog directory (default: a new temporary directory)")
    args = parser.parse_args(argv)
    records, errors = validate_registry(args.root.resolve())
    if not errors:
        try:
            output = prepare_output(args.output)
        except (OSError, ValueError) as error:
            errors.append(str(error))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"Catalog build failed with {len(errors)} error(s).", file=sys.stderr)
        return 1
    catalog = {"format": FORMAT, "sources": []}
    for record in sorted(records, key=lambda record: (record["kind"], record["name"])):
        label = f"{KIND_DIRECTORIES[record['kind']]}/{record['name']}.json"
        print(f"Fetching {label} ...", flush=True)
        root = output / source_root(record)
        try:
            with tempfile.TemporaryDirectory(prefix="holohub-source-") as directory:
                entry = collect_source(record, open_source(record, Path(directory)), root)
        except (OSError, ValueError, TypeError, RecursionError, yaml.YAMLError) as error:
            shutil.rmtree(root, ignore_errors=True)
            if root.parent.is_dir() and not any(root.parent.iterdir()):
                root.parent.rmdir()
            errors.append(f"{label}: {error}")
            continue
        catalog["sources"].append(entry)
        print(f"Cataloged {len(entry['entities'])} entities from {label}", flush=True)
    try:
        (output / INDEX).write_text(json.dumps(catalog, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    except OSError as error:
        errors.append(f"cannot write {INDEX}: {error}")
    else:
        total = sum(len(entry["entities"]) for entry in catalog["sources"])
        print(f"Wrote catalog of {total} entities from {len(catalog['sources'])} sources to {output}")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"Catalog build failed with {len(errors)} error(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
