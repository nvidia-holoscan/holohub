# HoloHub CLI Migration Guide

## Overview

As of June 2025, HoloHub has been refactored from bash-based CLI scripts (`./run` and `./dev_container`) to a modern, unified Python-based CLI tool (`./holohub`). This upgrade provides a cleaner, more powerful, and more intuitive development experience.

### Quick Start

```bash
./holohub list                    # See available projects
./holohub run <project>           # Build and run in container (default)
./holohub --help                  # List all commands and options
```

### Table of Contents

- [Overview](#overview)
  - [Quick Start](#quick-start)
- [Key Benefits of the Refactoring](#key-benefits-of-the-refactoring)
  - [Simplified Workflows](#simplified-workflows)
  - [Enhanced Developer Experience](#enhanced-developer-experience)
  - [Quick Command Reference](#quick-command-reference)
- [Migration Examples](#migration-examples)
  - [Build and Run Examples](#build-and-run-examples)
  - [Container Operation Examples](#container-operation-examples)
- [New Capabilities and Enhancements](#new-capabilities-and-enhancements)
- [Build and Run Configuration](#build-and-run-configuration)
- [Useful Commands](#useful-commands)

## Key Benefits of the Refactoring

### Simplified Workflows

Transform complex multi-command workflows into single commands.

| Development | Previous | Updated |
| ----------- | -------- | ------- |
| Container | `./dev_container build_and_run myapp --language cpp` | `./holohub run myapp --language cpp` |
| Local | `./run build myapp --type debug`<br>`./run launch myapp cpp` | `./holohub run myapp --language=cpp --local --build-type=debug` |

### Enhanced Developer Experience

- **Unified Interface**: One tool for both local and containerized development instead of two: `./run` and `./dev_container`
- **Intelligent Defaults**: Container-first development for consistency and reproducibility
- **Better Error Messages**: Helpful suggestions when commands or project names don't match
- **Modern Python Implementation**: More reliable, maintainable, and extensible

### Quick Command Reference


| Workflow                  | Previous                   | Updated                               |
| ------------------------- | ------------------------------- | ------------------------------------------- |
| **Container Build and Run** | `./dev_container build_and_run` | `./holohub run`                             |
| **Local Build and Run**     | `./run build` + `./run launch`  | `./holohub run --local`                     |
| **Container Build Only**  | `./dev_container build`         | `./holohub build-container`                 |
| **Local Build Only**      | `./run build`                   | `./holohub build --local`                   |
| **Run Without Building**  | `./run launch`                  | `./holohub run --local --no-local-build`    |
| **Container Run Only**    | `./dev_container launch`        | `./holohub run-container --no-docker-build` |


## Migration Examples

The following examples expand on the [Quick Command Reference](#quick-command-reference) table above.

### Build and Run Examples

1. Build locally with optional operators and build type:

| Previous | Updated |
| -------- | ------- |
| `./run build myapp --type debug --with "op1;op2"` | `./holohub build myapp --local --build-type debug --build-with "op1;op2"` |

2. Build in container.

| Previous | Updated |
| -------- | ------- |
| `./dev_container build --docker_file path/to/myapp/dockerfile --img holohub:myapp`<br>`./dev_container launch --img holohub:myapp`<br>`./run build myapp --type debug` | `./holohub build myapp --build-type debug` |


3. Run locally.

| Previous | Updated |
| -------- | ------- |
| `./run build myapp && ./run launch myapp cpp` | `./holohub run myapp --local --language cpp --build-type debug` |

4 Run in container. Use `=` for options that look like arguments, for example `--run-args=`.

| Previous | Updated |
| -------- | ------- |
| `./dev_container build_and_run myapp cpp --run_args "--config=config.json"` | `./holohub run myapp --language=cpp --run-args="--config=config.json"` |


5. Install built files.

| Previous | Updated |
| -------- | ------- |
| `./dev_container build_and_install myapp` | `./holohub install myapp` |

### Container Operation Examples

1. Container build.

| Previous | Updated |
| -------- | ------- |
| `./dev_container build --docker_file <path/to/myapp/Dockerfile> --img holohub:myapp` | `./holohub build-container <my_project> [options]` |

2. Container run.

| Previous | Updated |
| -------- | ------- |
| `./dev_container launch --img holohub:myapp` | `./holohub run-container <my_project> --no-docker-build [options]` |

3. Development environment

| Previous | Updated |
| -------- | ------- |
| `./dev_container vscode myapp` | `./holohub vscode myapp --language cpp` |



## New Capabilities and Enhancements

For project creation, testing, application modes (including mode structure, CLI overrides, metadata.json, and using different Docker options for build and run), see [New Capabilities and Application Modes](NEW_CAPABILITIES_AND_MODES.md).

## Build and Run Configuration

For details on default build and run behavior, command-line options (for example `--local`, `--no-docker-build`), and all environment variables (`HOLOHUB_*`, paths, container configuration, Docker image selection and tagging, and related settings), see the [Build and Run Configuration Reference](BUILD_AND_RUN_REFERENCE.md).


## Useful Commands

For tips on run-args syntax, Docker cache, argument naming, and running as root, see [Useful Commands](USEFUL_COMMANDS.md).

