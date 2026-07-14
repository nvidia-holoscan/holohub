# AGENTS.md

Guidance for AI agents working in this repository.

## Repository Structure

Projects live under `applications/`, `benchmarks/`, `operators/`, `gxf_extensions/`, `tutorials/`, and `pkg/`. Each project has a `metadata.json` (configuration, modes, dependencies) and a `CMakeLists.txt` (build registration). Check a project's README for hardware requirements and data downloads before building.

## Boundaries

- **Always** run `./holohub run-container -- "./holohub lint --install-dependencies; ./holohub lint"` before committing
- **Always** preview mutating CLI commands before running them for real:
  - Use `--dryrun --verbose` with `build`, `run`, `build-container`,
    `run-container`, `install`, `test`, and `package`
  - Use `--dryrun` with `create`, `lint`, `setup`, and `clear-cache`; these
    commands do not accept `--verbose`
  - `list`, `modes`, `env-info`, `env-check`, `status`, and `version` are
    read-only diagnostics and accept neither preview flag
- **Ask first** before changing `metadata.json` schemas, shared Dockerfiles, or CMake registration macros (`add_holohub_application`, `add_holohub_operator`, etc.)
- **Never** delete `build/`, `data/`, or `install/` directories without asking

## References

- [Main README](README.md) — overview, building, running, contributing
- [Contributing Guide](CONTRIBUTING.md) — how to contribute to the repository
- [CLI Reference](utilities/cli/cli_reference.md) — commands, flags, modes, environment variables
- [CLI Developer Guide](utilities/cli/cli_dev_guide.md) — workflow tips, implementation invariants, and extension guide
- [Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/introduction/overview)
