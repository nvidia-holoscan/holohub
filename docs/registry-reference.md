# Source registration reference

Use this reference when editing registration JSON or diagnosing validation
failures. For the submission workflow, start with
[Contributing](../CONTRIBUTING.md).

## Source records

For example, `modules/holoscan-example.json` could contain this illustrative
record. Replace the repository and revision with your published source:

```json
{
  "name": "holoscan-example",
  "kind": "module",
  "issues_url": "https://github.com/example/holoscan-example/issues",
  "source": {
    "type": "git",
    "url": "https://github.com/example/holoscan-example",
    "ref": "0123456789abcdef0123456789abcdef01234567",
    "path": "."
  }
}
```

`name`, `kind`, and `source` are required. Names use lowercase
letters, numbers, hyphens, and underscores, with no leading, trailing, or adjacent
separators. The name must match its filename and is unique within its kind.
All records are validated against
[`utilities/source.schema.json`](../utilities/source.schema.json).
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

## Discovery names

For a source selecting one project, the registration name must match the name
used to discover that project:

| Kind | Name to use | Example |
| --- | --- | --- |
| Module | The module's declared metadata name | `holoscan-my-module` |
| Application, tutorial, benchmark | The project directory name | `my-application` |
| Skill | The `SKILL.md` frontmatter name and skill directory name | `my-skill` |

For applications, tutorials, and benchmarks with descriptors in `cpp`, `python`,
or `py` language directories, use the parent project directory name. Confirm
non-skill project names with `holoscan list --json` in the source checkout;
a metadata display name can differ.

A **collection** selects a directory without its own root descriptor and
contains one or more projects of the registered kind. Give it a stable name,
such as `holoscan-applications`; this name need not match any member's project
name. If you add a project inside an already registered collection, only its
registered revision may need updating.

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

Use Python 3.12–3.14. [`requirement.txt`](../requirement.txt) installs the
Holoscan CLI, its bundled metadata schemas, and the other dependencies from
PyPI. Validation does not require the Holoscan SDK or a GPU. Pull requests run
the same checks in CI.

Run the commands in [Contributing: validation](../CONTRIBUTING.md#validation)
from the HoloHub repository root.

`validate_registry.py` works offline. It checks strict JSON parsing (including
duplicate keys), each record against the schema, directory/kind agreement,
matching filenames, and source uniqueness. Each category directory accepts
source record JSON files, `README.md`, and an empty `.gitkeep`. Symlinks, nested
directories, and implementation files are rejected.

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
  ignored. Scaffolding directories such as `template/` are skipped according
  to the pinned CLI's directory rules. A source with a root descriptor is
  treated as a direct entity.
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
[Register a source](../CONTRIBUTING.md#register-a-source) for registration changes.
