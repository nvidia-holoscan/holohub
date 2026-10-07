# Contributing source registrations

HoloHub accepts pointers to external entities. Keep implementations, notebooks,
skill definitions, build scripts, entity metadata, and entity CI upstream.
The Python utilities and tests here serve the registry itself.

## Register a source

The [module](modules/README.md), [application](applications/README.md),
[tutorial](tutorials/README.md), and [skill](skills/README.md) guides start with
registration examples and then cover development with the Holoscan CLI.
Registration is a JSON change in this repository; the CLI's `list` command
discovers local source projects and does not register them in HoloHub.

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
6. Run the commands below and submit the registration for review.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirement.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python utilities/validate_registry.py --index dist/sources.json
.venv/bin/python utilities/validate_sources.py
```

Keep the stable registration name when changing `source.ref`. If an entity
moves, update its source location in place. Remove its record when retiring the
registration. Keep an empty `.gitkeep` in each entity directory so removing its
last registration does not remove the directory from a fresh checkout.
Directory-level `README.md` guides are allowed but do not become source records
or index entries. Other documentation and entity files belong upstream.

Entity metadata is validated using its own schema at the target. The
[`holoscan-cli` metadata schemas](https://github.com/nvidia-holoscan/holoscan-cli/tree/main/src/holoscan_cli/metadata)
define those contracts independently of HoloHub's source registrations.
The source check fetches the exact revision and path, validates descriptors,
and verifies that `holoscan list --json` discovers each entity with the expected
name and kind. A direct entity's selector must match its registration name;
collection registrations retain their own names. Skills instead validate
`SKILL.md` frontmatter and its agreement with the skill directory and, for a
direct skill, the registration name. See the
[validation rules](README.md#validation-and-indexing) for details.

## Extending the registry

Add new kinds to the enum in `utilities/source.schema.json`, with the corresponding
directory, validator kind mapping, tests, and documentation. Reuse the same
location fields for every kind. Do not create schemas for upstream entity
metadata. The schema's `$id` identifies its version. For incompatible changes,
update the schema and all registry records together.

Registry CI validates declarations, source descriptors, and CLI discovery, and
builds a derived index. The source check requires network access and fails for
unavailable repositories, revisions, paths, or hosted descriptors. It retrieves
descriptors without executing registered source code. Do not add entity
build, test, container, deployment, GPU, or benchmark-execution jobs here.
