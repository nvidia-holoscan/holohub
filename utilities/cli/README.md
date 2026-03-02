# HoloHub CLI

A command-line interface for managing Holoscan-based applications and workflows. Single tool for the full development lifecycle: setup, build, run, test, package, and maintain.

## Design Goals

- **Simplicity**: complex workflows transformed into single intuitive commands
- **Developer experience**: fast iteration cycles with granular build control
- **Consistency**: predictable behavior across project types and deployment targets
- **Extensibility**: easy to add commands; portable across repositories via env var customization
- **Reliability**: comprehensive error handling, fuzzy suggestions, and dry-run support

## Quick Start

```bash
./holohub --help                          # List all commands
./holohub run <project> --language cpp    # Build + run in container (default)
./holohub run <project> --local           # Build + run on host
./holohub list                            # List available projects
./holohub run <project> --dryrun --verbose  # Preview without executing
```

## System Overview

The CLI covers the full lifecycle of a Holoscan application:

| Stage | Commands | Description |
|-------|----------|-------------|
| **Setup** | `setup`, `create` | Install dependencies, scaffold new projects from templates |
| **Build** | `build`, `build-container` | Compile applications/operators in containers or on host; auto-detect hardware and CUDA version |
| **Run** | `run`, `run-container` | Execute applications with mode-based configuration; launch interactive dev containers |
| **Package** | `install` | Install built artifacts for deployment |
| **Test** | `test` | Run CTest suites with optional coverage reporting |
| **Maintain** | `lint`, `env-info`, `clear-cache`, `vscode` | Linting, diagnostics, cache management, VS Code devcontainer |

By default, `build` and `run` use a containerized workflow (build image → build app inside → run inside). Use `--local` to build and run directly on the host.

## Core Components

Commands flow through a pipeline of six components:

1. **Input Resolution** — Parses commands, routes to handlers. Features fuzzy matching for typos, dash-prefix ambiguity detection, and `--` forwarding for `run-container`.

2. **Project Resolution** — Scans `metadata.json` files across configured search paths to build a project index. Handles multi-language projects and suggests similar names on not-found.

3. **Mode Resolution & Config Merge** — Resolves execution modes from `metadata.json` and merges with CLI flags. CLI flags always override mode settings. Modes define pre-configured setups for different hardware, data sources, or deployment scenarios.

4. **Container Build & Run** — Builds Docker images with auto-detected hardware configuration and CUDA version. Composes docker run arguments from multiple sources (security, display, devices, user groups, UCX).

5. **Application Build & Run** — Configures and executes CMake builds with optimizations (Ninja, parallel jobs, sccache). Manages path placeholder expansion and environment setup.

6. **Execution & Error Handling** — Transparent command execution with dry-run support, sudo detection, progressive validation, and actionable error messages.

## Metadata System

Each project is described by a `metadata.json` file that enables automatic discovery, language-specific configuration, dependency tracking, custom Dockerfiles, run command templates, and mode definitions.

Projects can define **modes** — named configurations for different use cases (e.g., `replayer` vs `production` vs `aja_hardware`). Each mode specifies build dependencies, Docker arguments, CMake options, run commands, and environment variables.

For the complete mode schema and field reference, see the [CLI Reference](https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/doc/cli_reference.md).

## On-Demand Dependencies

Setup scripts under `utilities/setup/` provide composable, reusable dependency recipes:

```bash
./holohub setup --list-scripts                    # List available scripts
./holohub setup --scripts benchmark               # Install on host
./holohub run <project> --extra-scripts benchmark  # Add as Docker layer
```

Each `--extra-scripts` invocation adds an independently-cached Docker layer, so dev dependencies (profilers, benchmarking tools) don't require rebuilding the base image. Multiple scripts can be composed in sequence.

## Reusable Tooling for External Codebases

The CLI is designed to be reused by external repositories through a lightweight bootstrapping process:

1. **Bootstrapping** — A wrapper script sparse-checkouts `utilities/cli/`, `utilities/metadata/`, and `utilities/testing/` from HoloHub. Branch selection is supported via `HOLOHUB_BRANCH`.

2. **Tool-Level Configuration** — The wrapper exports env vars to customize identity (`HOLOHUB_CMD_NAME`, `HOLOHUB_ROOT`), build paths (`HOLOHUB_BUILD_PARENT_DIR`, `HOLOHUB_WORKSPACE_NAME`), container images (`HOLOHUB_BASE_IMAGE`, `HOLOHUB_DEFAULT_IMAGE_FORMAT`), path prefixes (`HOLOHUB_PATH_PREFIX`), and test scripts (`HOLOHUB_CTEST_SCRIPT`).

3. **System-Specific Setup** — The wrapper can detect hardware and configure build/runtime arguments as env vars for the CLI to consume.

This ensures the CLI tools are downloaded once and configured per-project through environment variables, preventing code duplication while maintaining project-specific behavior.

## Documentation

- [CLI Reference](https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/doc/cli_reference.md) — commands, flags, modes, environment variables
- [CLI Developer Guide](CLI_DEV_GUIDE.md) — workflow tips, implementation invariants, extension guide
