# HoloHub CLI Reference

Quick reference for the HoloHub Python CLI (`./holohub`). For migration from the legacy bash scripts and detailed concepts, see [README.md](README.md).

---

## Quick Start

Run the CLI from the HoloHub repository root:

```bash
./holohub <command> [options] [arguments]
```

**Getting help:**

```bash
./holohub --help                    # List all commands
./holohub <command> --help          # Options for a specific command
./holohub run myapp --verbose --dryrun   # Debug: print without executing
./holohub list                      # List available projects
```

---

## Command Index

| Command | Description |
|---------|-------------|
| [create](#create) | Create a new Holoscan application from a template |
| [build-container](#build-container) | Build the development container image |
| [run-container](#run-container) | Build and launch the development container |
| [build](#build) | Build a project (container or local) |
| [run](#run) | Build and run a project (container or local) |
| [list](#list) | List all available targets (applications, workflows, operators, and so on) |
| [modes](#modes) | List available modes for an application |
| [lint](#lint) | Run linting tools |
| [setup](#setup) | Install HoloHub recommended packages for development |
| [env-info](#env-info) | Display environment debugging information |
| [install](#install) | Install a built project |
| [test](#test) | Run tests for a project |
| [clear-cache](#clear-cache) | Clear cache folders (build, data, install) |
| [vscode](#vscode) | Launch VS Code in Dev Container |

---

## Shared Options

Many commands inherit **container build** and/or **container run** options. They are listed here once; see each command for which apply.

### Container Build Options

Used by: `build-container`, `run-container`, `build`, `run`, `install`, `test`, `vscode`.

| Option | Description |
|--------|-------------|
| `--base-img` | Fully qualified base image name for the container build |
| `--docker-file` | Path to Dockerfile to use |
| `--img` | Fully qualified output container image name |
| `--no-cache` | Do not use cache when building the image |
| `--cuda <version>` | CUDA major version (for example `12`, `13`). Default: based on host driver |
| `--build-args` | Extra arguments to `docker build` (for example `--build-args '--network=host'`) |
| `--extra-scripts <name>` | Run named setup scripts as Docker layers (search in `HOLOHUB_SETUP_SCRIPTS_DIR`). Can be repeated. Use `./holohub setup --list-scripts` to list. |

### Container Run Options

Used by: `run-container`, `build`, `run`, `install`.

| Option | Description |
|--------|-------------|
| `--docker-opts <opts>` | Additional options to `docker run` (for example `--docker-opts='--gpus=all'`) |
| `--ssh-x11` | Enable X11 forwarding over SSH |
| `--nsys-profile` | Support Nsight Systems profiling in container |
| `--local-sdk-root <path>` | Path to Holoscan SDK on host for local SDK container |
| `--init` | Use tini entry point |
| `--persistent` | Do not delete container after it exits |
| `--add-volume <path>` | Mount path at `/workspace/volumes`. Can be repeated |
| `--as-root` | Run container as root |
| `--nsys-location <path>` | Nsight Systems installation path on host |
| `--mps` | Mount CUDA MPS host directories into container (if MPS enabled on host) |
| `--enable-x11` | Enable X11 forwarding (default: true) |

---

## Commands

### Create

Create a new Holoscan application from a template (for example, cookiecutter).

**Usage:**

```bash
./holohub create <project> [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `project` | Name of the project to create |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--template <path>` | `applications/template` | Path to template directory |
| `--language` | `cpp` | `cpp` or `python` |
| `--directory <path>` | `applications` | Directory to create the project in |
| `--context key=value` | — | Cookiecutter context; can be repeated |
| `-i`, `--interactive` | true | Interactive mode; use `-i False` to disable |
| `--dryrun` | — | Print commands without executing |

**Examples:**

```bash
./holohub create my_new_app --language cpp
./holohub create my_new_app --language python --directory applications
```

---

### Build Container

Build the development container image for a project (or the default image if no project).

**Usage:**

```bash
./holohub build-container [project] [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `project` | (Optional) Project to build container for |

**Options:** All [container build](#container-build-options) options, plus:

| Option | Description |
|--------|-------------|
| `--language` | `cpp` or `python` (when project has multiple implementations) |
| `--verbose` | Print variables passed to docker build |
| `--dryrun` | Print commands without executing |

**Examples:**

```bash
./holohub build-container myapp
./holohub build-container myapp --extra-scripts sccache --no-cache
```

---

### Run Container

Build (unless skipped) and launch the development container. Trailing args after ` -- ` are passed to `docker run`.

**Usage:**

```bash
./holohub run-container [project] [options] [-- docker run args...]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `project` | (Optional) Project to run container for |

**Options:** All [container build](#container-build-options) and [container run](#container-run-options) options, plus:

| Option | Description |
|--------|-------------|
| `--no-docker-build` | Skip building the container; use existing image |
| `--language` | `cpp` or `python` |
| `--verbose` | Print variables passed to docker run |
| `--dryrun` | Print commands without executing |

**Examples:**

```bash
./holohub run-container myapp --no-docker-build
./holohub run-container --local-sdk-root /path/to/holoscan-sdk --img holohub:sdk-dev-latest
```

---

### Build

Build a project. By default builds inside the container; use `--local` to build on the host.

**Usage:**

```bash
./holohub build <project> [mode] [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `project` | Project to build |
| `mode` | (Optional) Mode from `metadata.json` |

**Options:** All [container build](#container-build-options) and [container run](#container-run-options) options, plus:

| Option | Description |
|--------|-------------|
| `--local` | Build locally instead of in container |
| `--build-type` | `debug`, `release`, or `rel-debug`. Default: `CMAKE_BUILD_TYPE` or `release` |
| `--build-with <list>` | Semicolon-separated operators to build with |
| `--configure-args <arg>` | Extra CMake options; can be repeated (for example `--configure-args='-DCUSTOM=ON'`) |
| `--parallel <n>` | Number of parallel build jobs |
| `--pkg-generator` | `DEB` (default) or other cpack generator |
| `--language` | `cpp` or `python` |
| `--benchmark` | Build for Holoscan Flow Benchmarking (applications/workflows only) |
| `--no-docker-build` | Skip building the container |
| `--verbose` | Extra output |
| `--dryrun` | Print commands without executing |

**Examples:**

```bash
./holohub build myapp --build-type debug
./holohub build myapp --local --build-with "op1;op2" --configure-args='-DCUSTOM=ON'
./holohub build myapp --benchmark
```

---

### Run

Build and run a project. By default: build container (if needed), build app in container, run in container. Use `--local` for host build/run.

**Usage:**

```bash
./holohub run <project> [mode] [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `project` | Project to run |
| `mode` | (Optional) Mode from `metadata.json` (for example `replayer`, `standalone`) |

**Options:** All [container build](#container-build-options) and [container run](#container-run-options) options, plus:

| Option | Description |
|--------|-------------|
| `--local` | Run locally instead of in container |
| `--language` | `cpp` or `python` |
| `--build-type` | `debug`, `release`, or `rel-debug` |
| `--run-args <args>` | Arguments passed to the application (use `=` for example `--run-args="--config=config.json"`) |
| `--build-with <list>` | Semicolon-separated operators to build with |
| `--configure-args <arg>` | Extra CMake options; can be repeated |
| `--parallel <n>` | Parallel build jobs |
| `--pkg-generator` | `DEB` (default) or other |
| `--no-local-build` | Skip build; only run the application |
| `--no-docker-build` | Skip building the container |
| `--verbose` | Extra output |
| `--dryrun` | Print commands without executing |

**Examples:**

```bash
./holohub run myapp --language cpp
./holohub run myapp --local --language cpp --build-type debug
./holohub run myapp --language cpp --run-args="--config=config.json"
./holohub run body_pose_estimation replayer
./holohub run holochat standalone
```

---

### List

List all available targets (applications, workflows, operators, packages, and so on).

**Usage:**

```bash
./holohub list
```

No arguments or options.

---

### Modes

List available modes for an application (from `metadata.json`).

**Usage:**

```bash
./holohub modes <project> [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `project` | Project to list modes for |

**Options:**

| Option | Description |
|--------|-------------|
| `--language` | `cpp` or `python` |

**Examples:**

```bash
./holohub modes body_pose_estimation
```

---

### Lint

Run linting tools on the codebase or a path.

**Usage:**

```bash
./holohub lint [path] [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `path` | Path to lint (default: current directory) |

**Options:**

| Option | Description |
|--------|-------------|
| `--fix` | Auto-fix lint issues where possible |
| `--install-dependencies` | Install lint dependencies (may require `sudo`) |
| `--dryrun` | Print commands without executing |

**Examples:**

```bash
./holohub lint
./holohub lint --fix
./holohub lint --install-dependencies
```

---

### Setup

Install HoloHub recommended packages and run setup scripts (for example for Holoscan SDK development). May require `sudo`.

**Usage:**

```bash
./holohub setup [options]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--list-scripts` | List available scripts in `HOLOHUB_SETUP_SCRIPTS_DIR` |
| `--scripts <name>` | Run named script(s); can be repeated. Omit to run default recommended setup |
| `--dryrun` | Print commands without executing |

**Examples:**

```bash
./holohub setup
./holohub setup --list-scripts
./holohub setup --scripts my_script
```

---

### Env-Info

Print environment and path information for debugging.

**Usage:**

```bash
./holohub env-info
```

No arguments or options.

---

### Install

Build and install a project (container or local). Installs built artifacts (for example, using cpack).

**Usage:**

```bash
./holohub install <project> [mode] [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `project` | Project to install |
| `mode` | (Optional) Mode from `metadata.json` |

**Options:** All [container build](#container-build-options) and [container run](#container-run-options) options, plus:

| Option | Description |
|--------|-------------|
| `--local` | Install from local build instead of container |
| `--build-type` | `debug`, `release`, or `rel-debug` |
| `--build-with <list>` | Semicolon-separated operators |
| `--configure-args <arg>` | Extra CMake options; can be repeated |
| `--parallel <n>` | Parallel build jobs |
| `--language` | `cpp` or `python` |
| `--no-docker-build` | Skip building the container |
| `--verbose` | Extra output |
| `--dryrun` | Print commands without executing |

**Examples:**

```bash
./holohub install myapp
./holohub install myapp --local --build-type release
```

---

### Test

Run tests for a project (for example CTest in container or locally).

**Usage:**

```bash
./holohub test [project] [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `project` | (Optional) Project to test |

**Options:** All [container build](#container-build-options) options, plus:

| Option | Description |
|--------|-------------|
| `--local` | Test locally instead of in container |
| `--coverage` | Enable code coverage (adds coverage flags, runs ctest_coverage) |
| `--language` | `cpp` or `python` |
| `--clear-cache` | Clear cache before running |
| `--ctest-script <path>` | CTest script to use |
| `--cmake-options <opt>` | CMake options; can be repeated |
| `--ctest-options <opt>` | CTest options; can be repeated |
| `--site-name`, `--cdash-url`, `--platform-name` | CDash/reporting options |
| `--no-xvfb` | Do not use xvfb |
| `--build-name-suffix` | Suffix for CTest build name (default: image tag) |
| `--no-docker-build` | Skip building the container |
| `--verbose` | Extra output |
| `--dryrun` | Print commands without executing |

**Examples:**

```bash
./holohub test myapp
./holohub test myapp --coverage
./holohub test myapp --language python
```

---

### Clear Cache

Clear cache directories (build, data, install).

**Usage:**

```bash
./holohub clear-cache [options]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--build` | Clear build folders only |
| `--data` | Clear data folders only |
| `--install` | Clear install folders only |
| `--dryrun` | Print commands without executing |

If none of `--build`, `--data`, `--install` are given, all are cleared.

**Examples:**

```bash
./holohub clear-cache
./holohub clear-cache --build
```

---

### VSCode

Launch VS Code in a Dev Container for the given project.

**Usage:**

```bash
./holohub vscode [project] [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `project` | (Optional) Project to open in Dev Container |

**Options:** All [container build](#container-build-options) options, plus:

| Option | Description |
|--------|-------------|
| `--language` | `cpp` or `python` |
| `--docker-opts <opts>` | Extra options for Docker launch |
| `--no-docker-build` | Skip building the container |
| `--verbose` | Print variables passed to docker run |
| `--dryrun` | Print commands without executing |

**Examples:**

```bash
./holohub vscode myapp --language cpp
```

---

## Concepts

### Modes

Applications can define **modes** in `metadata.json`: named configurations for different hardware, data sources, or deployment scenarios. When a mode is set, it supplies default run command, build options, Docker options, and env vars; CLI options override mode settings when provided.

- Discover modes: `./holohub modes <project>`
- Run with a mode: `./holohub run <project> <mode>`
- Build with a mode: `./holohub build <project> <mode>`

CLI parameters (for example `--run-args`, `--build-with`, `--docker-opts`) always override the corresponding mode settings. For full mode structure, field reference, and examples, see [README.md — Application Modes](README.md#application-modes).

### Local Versus Container

- **Default:** `./holohub run` and `./holohub build` use the **container**: build image (if needed), build app in container, run in container.
- **`--local`:** Build and/or run on the **host** (no container for app build/run).
- **`--no-docker-build`:** Use an existing container image (skip image build).
- **`--no-local-build`:** (run only) Skip app build; run existing binaries.

Environment variable `HOLOHUB_BUILD_LOCAL` forces local mode (same as `--local`). See [README.md — Granular Build Control](README.md#granular-build-control) for details.

---

## Environment Variables

The CLI respects these variables. Defaults and behavior are summarized below; see [README.md — Granular Build Control](README.md#granular-build-control) for full detail.

### Build and Execution

| Variable | Purpose |
|----------|---------|
| `HOLOHUB_BUILD_LOCAL` | Force local mode (like `--local`) |
| `HOLOHUB_ALWAYS_BUILD` | Set to `false` to skip builds when using `--no-local-build` / `--no-docker-build` |
| `HOLOHUB_ENABLE_SCCACHE` | Set to `true` to enable sccache for builds; use with `--extra-scripts sccache` in container |

### Paths and Directories

| Variable | Default / purpose |
|----------|-------------------|
| `HOLOHUB_ROOT` | HoloHub repo root |
| `HOLOHUB_BUILD_PARENT_DIR` | `<HOLOHUB_ROOT>/build` |
| `HOLOHUB_DATA_DIR` | `<HOLOHUB_ROOT>/data` |
| `HOLOHUB_SETUP_SCRIPTS_DIR` | `<HOLOHUB_ROOT>/utilities/setup` |
| `HOLOHUB_PATH_PREFIX` | `holohub_` (prefix for path placeholders in metadata) |
| `HOLOHUB_DEFAULT_HSDK_DIR` | `/opt/nvidia/holoscan` |
| `HOLOSCAN_SDK_ROOT` | Local Holoscan SDK path (for mounting into containers) |
| `HOLOHUB_SEARCH_PATH` | Comma-separated dirs to scan for metadata (for example `applications,workflows,...`) |

### Container and Docker

| Variable | Default / purpose |
|----------|-------------------|
| `HOLOHUB_REPO_PREFIX` | `holohub`; base for naming |
| `HOLOHUB_CONTAINER_PREFIX` | Same as repo prefix; container name prefix |
| `HOLOHUB_WORKSPACE_NAME` | Workspace dir name in container |
| `HOLOHUB_HOSTNAME_PREFIX` | Container hostname prefix (for example for VSCode) |
| `HOLOHUB_DOCKER_EXE` | `docker` |
| `HOLOHUB_BASE_IMAGE`, `HOLOHUB_BASE_SDK_VERSION`, `HOLOHUB_BASE_IMAGE_FORMAT` | Base image for Dockerfiles |
| `HOLOHUB_DEFAULT_IMAGE_FORMAT` | Default output image tag format |
| `HOLOHUB_DEFAULT_DOCKER_BUILD_ARGS` | Extra default args for `docker build` |
| `HOLOHUB_DEFAULT_DOCKER_RUN_ARGS` | Extra default args for `docker run` |
| `HOLOHUB_DEFAULT_DOCKERFILE` | Default Dockerfile path |
| `HOLOHUB_BENCHMARKING_SUBDIR` | Benchmarking subdir (for example for flow benchmarking) |

### Other

| Variable | Purpose |
|----------|---------|
| `HOLOHUB_CTEST_SCRIPT` | CTest script used by `./holohub test` |
| `HOLOHUB_CMD_NAME` | Command name in help (default: `./holohub`) |
| `HOLOHUB_CLI_DOCS_URL` | URL for CLI docs (for example for external forks) |
| `CMAKE_BUILD_TYPE` | Default CMake build type when not set on CLI |
| `CMAKE_BUILD_PARALLEL_LEVEL` | Default parallel build jobs |

---

## Appendix

### Bash Autocompletion

Autocompletion is usually installed during `./holohub setup`. It completes project names and command names.

```bash
./holohub <TAB><TAB>       # List commands
./holohub run ultra<TAB>   # Complete project name
./holohub build vid<TAB>   # Complete project name
```

**Manual install:**

```bash
sudo cp utilities/holohub_autocomplete /etc/bash_completion.d/
echo ". /etc/bash_completion.d/holohub_autocomplete" >> ~/.bashrc
source ~/.bashrc
```

### Useful Tips

- For options that look like arguments, use `=` to avoid ambiguity:  
  `--run-args="--verbose"` instead of `--run-args "--verbose"`.
- All CLI options use **hyphens** (`-`), not underscores (for example `--base-img`, not `--base_img`).
- `sudo ./holohub` may not work correctly due to environment filtering (for example `PATH`).
- To free disk during development: `docker image prune`, `docker buildx prune`, `docker system prune` (see [Docker docs](https://docs.docker.com/reference/cli/)).

### See Also

- **[README.md](README.md)** — Migration guide from legacy scripts, application modes and metadata reference, and detailed environment variable and Docker image behavior.
