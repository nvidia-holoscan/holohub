# HoloHub

HoloHub is a registry and index of externally maintained Holoscan applications,
modules, tutorials, skills, and benchmarks. Each registration tells consumers
where to find a source. Entity implementations, metadata, dependencies, tests,
packaging, and CI live with their owners, outside this repository.

The only schema is [`utilities/source.schema.json`](utilities/source.schema.json).
Every kind uses the same fields to identify a source and locate it. Entity metadata is read
from the target and validated against its own schema, such as those maintained
in [`holoscan-cli/metadata`](https://github.com/nvidia-holoscan/holoscan-cli/tree/main/src/holoscan_cli/metadata).
Repository tooling validates registrations and produces an index; it does not
clone, build, install, or execute registered sources.

## Registry layout

```text
applications/<name>.json
modules/<name>.json
tutorials/<name>.json
skills/<name>.json
benchmarks/<name>.json
utilities/source.schema.json
utilities/validate_registry.py
.github/workflows/validate-registry.yml
```

Each entity directory contains **one JSON object per source**, stored directly
as `<name>.json`. A source may provide multiple entities, such as the operators
exported by a module. There are no aggregate registration files, entity
subdirectories, or entity metadata schemas. An empty `.gitkeep` preserves each
directory in Git before sources are registered.

Registered sources live in the entity directories. Illustrative records for all five kinds live in
[`tests/fixtures/`](tests/fixtures/) and are excluded from the index.

## Source records

For example, `modules/example-module.json` could contain this illustrative record:

```json
{
  "name": "example-module",
  "kind": "module",
  "issues_url": "https://github.com/example/holoscan-modules/issues",
  "source": {
    "type": "git",
    "url": "https://github.com/example/holoscan-modules",
    "ref": "0123456789abcdef0123456789abcdef01234567",
    "path": "."
  }
}
```

`name`, `kind`, and `source` are required. Names use lowercase
letters, numbers, hyphens, and underscores, with no leading, trailing, or adjacent
separators. The name must match its filename and is unique within its kind.
All records are validated against `utilities/source.schema.json`.
Unknown fields are rejected. Descriptions, tags, ratings, operator inventories,
entrypoints, and results belong in the entity's own metadata at the target.

Optional `issues_url` specifies where users should report issues with the
registered source. It is a top-level HTTPS URL to an issue tracker or support
form, available for every kind and both source forms. The destination may use a
different host from the source; query strings and fragments are supported.
It is included unchanged in the generated index and does not affect source
identity. Omit it when no reporting destination is specified.

Two source forms are supported:

- **Git:** `type: "git"`, an HTTPS clone `url`, and a required `ref` naming a
  branch, tag, or commit. A full commit SHA is recommended for reproducibility.
  Optional `path` selects a directory inside the repository; omission or `"."`
  means the root. This is where consumers find the entity and its own metadata
  (commonly `metadata.json`). Put the revision and path in their fields, not in
  the URL.
- **Hosted content:** `type: "url"` and an HTTPS `url` pointing directly to
  content such as a tutorial, download, or results page. Query strings and
  fragments are supported. Prefer versioned URLs when available. Git fields
  do not apply to this form.

All URLs must be absolute HTTPS URLs with a hostname, no credentials, and an
optional port from 1 through 65535. Relative paths use
POSIX separators and cannot contain `.`/`..` components, empty components,
absolute paths, backslashes, or control characters. The Git source root `"."`
is the sole dot-path exception.

`kind` identifies the registry category; it does not change the record's fields.
There are no kind-specific fields or copies of upstream entity schemas here.

## One record per source

A Git source is identified by **kind + repository URL + path**. Updating its
revision edits the existing record. Hostname case, default HTTPS port, trailing
repository slash, and an optional `.git` suffix are normalized for duplicate
checks. Equivalent URL dot segments and percent-encoded unreserved characters
are normalized, and GitHub repository names are compared without case
sensitivity. Repository-relative paths remain case sensitive.

A hosted source is identified by **kind + URL**. Hostname case and default
HTTPS port are normalized; paths, queries, and fragments distinguish content.
The same repository can supply multiple kinds or distinct source directories.
Names are registration identities, not release numbers. Upstream release
history and detailed entity inventories stay upstream.

## Validation and indexing

Use Python 3.12 or later:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirement.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python utilities/validate_registry.py --index dist/sources.json
```

The validator checks the schema itself, strict JSON parsing (including duplicate
keys), all records, directory/kind agreement, matching filenames, source
uniqueness, and the source-only entity directory layout. It reports failures
with filenames and exits nonzero. Symlinks, nested directories, implementation
files, and nonempty `.gitkeep` files are rejected in entity directories.
An empty registry is valid.

The optional index is a JSON array sorted by kind and name. Each item is an
unchanged, validated source record. There is no index-specific schema or
timestamp. The generated `dist/` directory is not
committed. Invalid registries do not create or overwrite the index.

The **Registry / Validate registry** workflow runs on pull requests, pushes,
merge-queue checks, and manual dispatches. It tests the registry tooling,
validates the entire registry, and uploads `sources.json` as the `source-index`
artifact. Configure this job as a required branch-protection check when the
workflow is enabled in GitHub. The workflow has read-only repository permissions
and uses actions pinned to commit SHAs.

Validation runs offline after dependencies are installed. It checks source
declarations, not remote availability, revision existence, file existence,
entity metadata, entity correctness, or benchmark results. Those responsibilities
belong to source owners and consumers of the upstream metadata. See
[CONTRIBUTING.md](CONTRIBUTING.md) for registration changes.
