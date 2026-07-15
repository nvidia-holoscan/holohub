# AGENTS.md

Guidance for AI agents working in this repository.

## Repository Structure

Projects live under `applications/`, `benchmarks/`, `operators/`, `gxf_extensions/`, `tutorials/`, and `pkg/`. Each project has a `metadata.json` (configuration, modes, dependencies) and a `CMakeLists.txt` (build registration). Check a project's README for hardware requirements and data downloads before building.

## Boundaries

- **Always** run `./holohub run-container -- "./holohub lint --install-dependencies; ./holohub lint"` before committing
- **Always** preview mutating CLI commands with `--dryrun` before running them for real; add `--verbose` where supported. Read-only diagnostics (`list`, `modes`, `env-info`, `env-check`, `status`, `version`) need no preview. See the [CLI Reference](utilities/cli/cli_reference.md#preview-support) for per-command support.
- **Ask first** before changing `metadata.json` schemas, shared Dockerfiles, or CMake registration macros (`add_holohub_application`, `add_holohub_operator`, etc.)
- **Never** delete `build/`, `data/`, or `install/` directories without asking

## References

- [Main README](README.md) — overview, building, running, contributing
- [Contributing Guide](CONTRIBUTING.md) — how to contribute to the repository
- [CLI Reference](utilities/cli/cli_reference.md) — commands, flags, modes, environment variables
- [CLI Developer Guide](utilities/cli/cli_dev_guide.md) — workflow tips, implementation invariants, and extension guide
- [Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/introduction/overview)
