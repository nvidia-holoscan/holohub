# AGENTS.md

Guidance for AI agents working in this repository. For build prerequisites, see [README.md](README.md).

## Repository Structure

Projects live under `applications/`, `benchmarks/`, `operators/`, `gxf_extensions/`, `tutorials/`, `workflows/`, and `pkg/`. Each project has a `metadata.json` (configuration, modes, dependencies) and a `CMakeLists.txt` (build registration). Check a project's README for hardware requirements and data downloads before building.

## Boundaries

- **Always** run `./holohub run-container -- "./holohub lint --install-dependencies; ./holohub lint"` before committing
- **Always** use `--dryrun --verbose` to inspect a CLI command before running it for real
- **Ask first** before changing `metadata.json` schemas, shared Dockerfiles, or CMake registration macros (`add_holohub_application`, `add_holohub_operator`, etc.)
- **Never** delete `build/`, `data/`, or `install/` directories without asking

## References

- [CLI Reference](utilities/cli/README.md) — commands, flags, modes, environment variables
- [CLI Developer Guide](utilities/cli/CLI_DEV_GUIDE.md) — workflow tips, implementation invariants, and extension guide (portable across repos)
- [Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/overview.html)
