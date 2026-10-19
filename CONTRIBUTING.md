# Contributing source registrations

HoloHub accepts pointers to external entities. Keep implementations, notebooks,
skill definitions, build scripts, entity metadata, and entity CI upstream.

## Request a registration change

The easiest way to register, update, or remove a source is to
[open a registration request](https://github.com/nvidia-holoscan/holohub/issues/new?template=registry.yml).
You do not need to edit JSON or open a pull request yourself.

- **Register:** enter the kind (`application`, `module`, `tutorial`, `skill`,
  or `benchmark`), a new name, and the source URL. Add a Git revision and
  repository path if needed.
- **Update:** enter the registered kind and name, plus only the fields you want
  to change. Blank fields keep their current values; enter `none` to clear an
  optional value.
- **Remove:** enter the registered kind and name, then tick **Remove**. Only
  the HoloHub record is removed; your project is not changed.

Tips for the source fields:

- For a Git repository, use its HTTPS clone URL (or GitHub repository page).
  A full commit SHA is the most reproducible revision; leave the revision blank
  to use the repository's default branch.
- For hosted content, link directly to `metadata.json` (or `SKILL.md` for a
  skill) and leave the revision blank.
- If you leave the issue tracker blank, a public GitHub or GitLab repository's
  issue tracker is used.
- To rename a registration, remove it first, then register the new name after
  the removal is merged.

After you submit, a bot checks the request, renames the issue (for example,
`🗂️ [NEW] application: my-application`), and opens a pull request. It posts a
comment on the issue with the result, including any values it filled in or
what to correct. To fix or change the request, edit the issue; the pull request
is updated automatically, so do not push to it. Closing the issue withdraws the
request, and reopening it tries again. A maintainer reviews and merges the pull
request, which closes the issue.

## Register a source

The [module](modules/README.md), [application](applications/README.md),
[tutorial](tutorials/README.md), and [skill](skills/README.md) guides start with
registration examples and then cover development with the Holoscan CLI.
To edit the JSON yourself instead of
[using the form](#request-a-registration-change), follow the steps below. The
CLI's `list` command discovers local source projects and does not register them
in HoloHub.

1. Choose `application`, `module`, `tutorial`, `skill`, or `benchmark`.
2. Check that the source is not already registered under that kind. A new Git
   revision does not create a new source; edit the existing record instead.
3. Create `<kind-directory>/<name>.json` directly in the relevant directory.
   Start from the corresponding illustrative record in `tests/fixtures/`
   and replace its example values.
4. Provide an HTTPS Git clone URL and revision, or a direct hosted-content URL.
   Use `source.path` for a repository subdirectory. Prefer immutable commits or
   versioned URLs. Confirm the upstream location is publicly accessible and
   contains `metadata.json`, or `SKILL.md` for skills. A Git path may also
   select a collection; hosted URLs must serve the descriptor directly or
   from the specified directory.
   Optionally add a top-level `issues_url` pointing to the source's HTTPS issue
   tracker or support form so users know where to report problems.
5. Keep entity descriptions, tags, operator lists, ratings, and execution details
   in the target's own metadata. All kinds use the same registration fields.
6. Run the checks below and submit the registration for review.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirement.txt
.venv/bin/python utilities/validate_registry.py
.venv/bin/python utilities/validate_sources.py
```

Keep the stable registration name when changing `source.ref`. If an entity
moves, update its source location in place. Remove its record when retiring the
registration. Documentation and entity files belong upstream.

Entity metadata is validated using its own schema at the target. The
[`holoscan-cli` metadata schemas](https://github.com/nvidia-holoscan/holoscan-cli/tree/main/src/holoscan_cli/metadata)
define those contracts independently of HoloHub's source registrations.
The source check fetches the exact revision and path, validates descriptors,
and verifies that `holoscan list --json` discovers each entity with the expected
name and kind. A direct entity's selector must match its registration name;
collection registrations retain their own names. Skills instead validate
`SKILL.md` frontmatter and its agreement with the skill directory and, for a
direct skill, the registration name. See the
[validation rules](#validation) for details.

## Registry layout

```text
applications/<name>.json
modules/<name>.json
tutorials/<name>.json
skills/<name>.json
benchmarks/<name>.json
```

Each entity directory contains **one JSON object per source**, stored directly
as `<name>.json`. A source may provide multiple entities, such as the operators
exported by a module. Illustrative records for all five kinds live in
[`tests/fixtures/`](tests/fixtures/).

For registration instructions followed by guidance on creating and building an
entity, see the directory guides:

| Entity | Guide |
| --- | --- |
| Reusable Holoscan modules | [Modules](modules/README.md) |
| End-to-end applications | [Applications](applications/README.md) |
| Educational walkthroughs | [Tutorials](tutorials/README.md) |
| Agent instructions and supporting resources | [Skills](skills/README.md) |

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
All records are validated against
[`utilities/source.schema.json`](utilities/source.schema.json).
Unknown fields are rejected. Descriptions, tags, ratings, operator inventories,
entrypoints, and results belong in the entity's own metadata at the target.

Optional `issues_url` specifies where users should report issues with the
registered source. It is a top-level HTTPS URL to an issue tracker or support
form, available for every kind and both source forms. The destination may use a
different host from the source; query strings and fragments are supported.
Omit it when no reporting destination is specified.

Two source forms are supported:

- **Git:** `type: "git"`, an HTTPS clone `url`, and a required `ref` naming a
  branch, tag, or commit. A full commit SHA is recommended for reproducibility.
  Optional `path` selects a directory inside the repository; omission or `"."`
  means the root. Select an entity directory with `metadata.json` (or `SKILL.md`
  for a skill), or a collection containing those descriptors. Put the revision
  and path in their fields, not in the URL.
- **Hosted content:** `type: "url"` and an HTTPS `url` pointing directly to
  `metadata.json` or `SKILL.md`, or to a directory serving that file. The source
  validator appends the descriptor filename to directory URLs, preserves query
  strings, and ignores fragments when fetching. Prefer versioned URLs when
  available. Git fields do not apply to this form. HTML pages and archives alone
  do not satisfy source validation.

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

## Validation

Use Python 3.12 or later. `requirement.txt` installs the Holoscan CLI and its
metadata schemas from NVIDIA's package index, `https://pypi.nvidia.com`, and
other dependencies from PyPI. Validation does not require the Holoscan SDK or a
GPU. Pull requests run the same checks in CI.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirement.txt
.venv/bin/python utilities/validate_registry.py
.venv/bin/python utilities/validate_sources.py
```

`validate_registry.py` works offline. It checks strict JSON parsing (including
duplicate keys), each record against the schema, directory/kind agreement,
matching filenames, and source uniqueness. Only source record JSON files belong
in entity directories; symlinks, nested directories, and implementation files
are rejected.

`validate_sources.py` requires network access and checks every source:

- Git sources must be publicly fetchable at the exact `ref` and `path`. Only
  descriptor files are materialized from Git objects; no checkout, submodules,
  builds, or source scripts run. Missing revisions and paths fail validation.
- A direct entity's `metadata.json` must have exactly one recognized kind
  envelope, match the registration kind, and pass that kind's schema bundled
  with the pinned Holoscan CLI. Its CLI selector must match the registration
  name. Modules use their declared name; other kinds use the project directory
  name, with `cpp`, `python`, and `py` treated as language subdirectories.
  Display names in application metadata may differ from these selectors.
- Collections must contain at least one entity of the registered kind. Each
  matching descriptor is validated; accompanying entities of other kinds are
  ignored. A source with a root descriptor is treated as a direct entity.
  `holoscan list --json` must return every validated entity with the expected
  selector, kind, and source directory. The collection's registration name is
  not an entity selector.
- Skills require `SKILL.md`, without `metadata.json` or CLI discovery. Each
  file needs YAML frontmatter with a valid `name`, a nonempty `description`,
  and a nonempty Markdown instruction body. Names must match skill directories
  and, for a direct skill, the registration name. Any frontmatter or nested
  `metadata` identity fields must agree (`name` and `kind: skill`). Optional
  agent-specific fields are allowed. The core naming and frontmatter rules
  follow the [Agent Skills specification](https://agentskills.io/specification).

Descriptors are limited to 1 MiB; duplicate JSON/YAML keys and symlink
descriptors are rejected. Failures name the registration file. Entity
behavior, builds, and benchmark results remain the source owner's
responsibility. Follow
[Register a source](#register-a-source) for registration changes.
