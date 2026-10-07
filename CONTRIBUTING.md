# Contributing source registrations

HoloHub accepts pointers to external entities. Keep implementations, notebooks,
skill definitions, build scripts, entity metadata, and entity CI upstream.
The Python utilities and tests here serve the registry itself.

## Register a source

1. Choose `application`, `module`, `tutorial`, `skill`, or `benchmark`.
2. Check that the source is not already registered under that kind. A new Git
   revision does not create a new source; edit the existing record instead.
3. Create `<kind-directory>/<name>.json` directly in the relevant directory.
   Start from the corresponding illustrative record in `tests/fixtures/`
   and replace its example values.
4. Provide an HTTPS Git clone URL and revision, or a direct hosted-content URL.
   Use `source.path` for a repository subdirectory. Prefer immutable commits or
   versioned URLs. Confirm the upstream location exists.
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
```

Keep the stable registration name when changing `source.ref`. If an entity
moves, update its source location in place. Remove its record when retiring the
registration. Keep an empty `.gitkeep` in each entity directory so removing its
last registration does not remove the directory from a fresh checkout.

## Migrating module source records

The earlier
[`module-sites.schema.json`](https://github.com/nvidia-holoscan/holohub/blob/main/utilities/metadata/module-sites.schema.json)
described a shared `modules` array. Each external entry now becomes a separate
`modules/<name>.json` record:

| Earlier field | Source registration |
| --- | --- |
| `name` | `name`, with `kind: "module"` |
| `url` | `source.url`, with `source.type: "git"` |
| `ref` | `source.ref` |
| `provides_operators` | Keep operator inventories in upstream entity metadata |
| `nvidia_quality_score` | Keep assessments upstream; there is no registry rating field |
| `source_url` | Resolve to the authoritative clone URL and optional `source.path`; do not copy an ambiguous alternate URL |

Old entries with no URL represented in-tree modules. Move their implementations
to an external repository before registering them; the new schema requires an
explicit external source. Do not fabricate locations during migration.

Entity metadata is validated using its own schema at the target. The
[`holoscan-cli` metadata schemas](https://github.com/nvidia-holoscan/holoscan-cli/tree/main/src/holoscan_cli/metadata)
define those contracts independently of HoloHub's source registrations.

## Extending the registry

Add new kinds to the enum in `utilities/source.schema.json`, with the corresponding
directory, validator kind mapping, tests, and documentation. Reuse the same
location fields for every kind. Do not create schemas for upstream entity
metadata. The schema's `$id` identifies its version. For incompatible changes,
update the schema and migrate the registry records together.

Registry CI validates declarations and builds a derived index. Do not add entity
build, test, container, deployment, GPU, or benchmark-execution jobs here.
