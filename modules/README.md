# Holoscan Modules

A **Holoscan Module** is a library project that extends the Holoscan SDK API.

## Creating a New External Module

Run the following command to initialize a new module adhering to HoloHub conventions
from the provided cookiecutter template. The template will prompt for several project
details (name, project languages, description) before creating the new project directory
on your local system.

```bash
pip install cookiecutter

./holohub create <my_module> --template modules/template
```

The generated project is a self-contained git repository with its own operators/,
applications/, tests/, and packaging files. It is intended to be hosted outside of
HoloHub (e.g. a dedicated GitHub repo) and declared as an external dependency in
consuming applications' metadata.json.

## Declaring an In-Tree Module

Some Holoscan Modules are best maintained directly inside the HoloHub monorepo —
either because their operator libraries are already there or because tight integration
with HoloHub's CI and tooling is desirable. These are called **in-tree modules**.

An in-tree module uses a **descriptor-only** layout: the operator sources stay in
`operators/<name>/` (and applications in `applications/<name>/`) while a thin
descriptor directory under `modules/<module-name>/` holds the module-level metadata
and packaging files.

```text
modules/
└── holoscan-gstreamer/          ← module descriptor (this directory)
    ├── metadata.json            ← module schema v2: identity, namespace, operators list
    ├── pyproject.toml           ← wheel packaging (drives HoloHub's CMake selectively)
    └── Dockerfile               ← dev container for module-focused development

operators/gstreamer/             ← operator sources (unchanged, in-tree as always)
applications/gstreamer/          ← application sources (unchanged)
```

### Creating an In-Tree Module Descriptor

1. Create `modules/<holoscan-name>/` (e.g. `modules/holoscan-gstreamer/`).
2. Add `metadata.json` using the `holohub/module/v2` schema. Omit `source_repository`
   (the module lives in HoloHub). Set `operators` to the list of HoloHub `OP_*` names
   the module provides.
3. Add `pyproject.toml` with `cmake.source-dir = "../.."` pointing at HoloHub root,
   and `-DOP_<name>=ON -DBUILD_ALL=OFF` to build only the module's operators.
4. Add a `Dockerfile` extending the Holoscan SDK base image with module-specific
   system dependencies.

See `modules/holoscan-gstreamer/` for a complete reference example.

### Dependency Resolution for In-Tree Modules

The HoloHub CLI resolver (`utilities/cli/external_resolver.py`) automatically
recognizes in-tree modules: a dependency with no `source` block is looked up in
`modules/<name>/metadata.json`. If found, the dep is marked `is_internal=True` and
the CMake manifest emits a comment instead of a FetchContent_Declare — the operators
are already present in HoloHub's tree and are built when `OP_<name>=ON`.
