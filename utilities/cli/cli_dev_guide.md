# CLI Developer Guide

Reference for AI agents and developers working with or extending the CLI.
For command/flag docs, see the [CLI Reference](cli_reference.md).

> [!NOTE] Portability
> This guide applies to any repo built on the HoloHub CLI.
> Examples use `./holohub`; downstream repos have their own entry point
> (e.g. `./i4h`, `./isaac_os`), configured via `HOLOHUB_CMD_NAME`.
> Currently `HOLOHUB_`-prefixed env vars are shared CLI infrastructure, not
> HoloHub-specific. Substitute your repo's entry point wherever you
> see `./holohub`.

## Agent Safety Rules

1. Always `./holohub <cmd> --dryrun --verbose` before running any command for real.
2. Never `./holohub build --local` or `./holohub run --local` without explicit user approval — they run directly in the host workspace, may modify files there, and require host dependencies to already be installed (no automatic package installation).
3. Never hardcode repo names, command names, or path prefixes — the CLI is reused across repos via env vars.
4. Check the project's README and `metadata.json` for architecture/platform (e.g. x86_64, aarch64, IGX, Thor, DGX Spark) requirements before building.  The project may not support all platforms, do not suggest building on platforms that are not supported unless the user explicitly asks for it.
5. By default, do not suggest running cmake, python, or application binaries directly on the host. The CLI runs everything inside a container by default. If a user needs to run commands manually (custom cmake flags, flat build directory, debugging), the answer is `./holohub run-container <app>` first, then run commands inside. Raw host commands require the same explicit user approval as `--local`.

## Mental Model

The CLI is a **container-first** build system. Every `build`, `run`, `install`, and `test` command builds a Docker image and then re-invokes itself with `--local` inside that container. The only host-side commands are `docker build` and `docker run`; cmake, python, and application processes execute inside the container where dependencies are guaranteed to exist.

When a user asks how to customize a build or run — change the build directory, add cmake flags, use a different build type — the answer is `./holohub run-container <app>` first, then run commands inside (see [Why this matters](#why-this-matters) for examples).

## Implementation Invariants

| #   | Rule                                   | Detail                                                                                                                                                                                                                                                                                                   |
| --- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Container re-enters local              | `build`/`run`/`install` in container mode build an image, then run the CLI with `--local` inside it. Local-mode changes affect both host and container execution.                                                                                                                                        |
| 2   | CLI flags override modes               | Resolution: `resolve_mode` → `validate_mode` → `get_effective_build_config`/`get_effective_run_config`. Most flags override (do not merge with) mode values; `--run-args` is **appended** to the argument list passed to `docker run` or the application process (not inserted into the command string). |
| 3   | Expand placeholders first              | Mode commands may contain `<{prefix}_app_source>`, `<{prefix}_data_dir>`, etc. (prefix = `HOLOHUB_PATH_PREFIX`, default `holohub`). Run through `build_holohub_path_mapping()` + `replace_placeholders()`.                                                                                               |
| 4   | Use `run_command()`                    | Handles dry-run, sudo detection, fail-fast (`check=True`). Use `run_info_command()` for best-effort probes.                                                                                                                                                                                              |
| 5   | Use `self.script_name` / `self.prefix` | `self.script_name` (from `HOLOHUB_CMD_NAME`) for user messages. `self.prefix` (from `HOLOHUB_PATH_PREFIX`) for placeholder resolution.                                                                                                                                                                   |

## Execution Model

`build`, `run`, `install`, and `test` follow a two-phase container pattern by default (without `--local`):

### Phase 1 — Container setup (runs on host)

1. Build a Docker image with project dependencies (`docker build`). Skip with `--no-docker-build`.
2. Launch a container (`docker run`) with the repo bind-mounted at `/workspace/<name>` (`HOLOHUB_WORKSPACE_NAME`, default `holohub`) and `HOLOHUB_BUILD_LOCAL=1` set in the environment.

### Phase 2 — Build / run (runs inside the container)

The container re-invokes the CLI with `--local` appended. For example, `./holohub build myapp` on the host becomes `./holohub build myapp --local` inside the container. The local handler runs the underlying tools:

| Command   | What runs inside the container                                                 |
| --------- | ------------------------------------------------------------------------------ |
| `build`   | `cmake -B build/<app> -S . -DAPP_<app>=ON [opts] && cmake --build build/<app>` |
| `run`     | Same cmake build, then the mode's `run.command` with placeholders expanded     |
| `install` | Same cmake build, then `cmake --install build/<app>`                           |
| `test`    | CTest via a CTest script                                                       |

The cmake flag `-DAPP_<app>=ON` selects a single project (operators use `-DOP_<name>=ON`, extensions use `-DEXT_<name>=ON`). The build directory `build/<app>/` isolates per-project artifacts; this path is `HOLOHUB_BUILD_PARENT_DIR/<app>` (default parent varies by repo; `<repo_root>/build` in HoloHub). Because the repo is bind-mounted, artifacts persist on the host after the container exits.

`--local` bypasses both phases and runs cmake / the app directly on the host.

### Why this matters

By default (without `--local` or relevant environment variables) every cmake, python, and application command executes **inside the container**, not on the host. To run those commands manually — customize the build directory, add cmake flags, or debug a failure — enter the container first, then run whatever you need:

```bash
# 1. Enter the container (builds the image if needed, then opens an interactive shell):
./holohub run-container <app> [mode]

# 2. Inside the container, HOLOHUB_BUILD_LOCAL=1 is already set, so all
#    ./holohub commands behave as --local. Run the standard CLI build:
./holohub build <app> [mode]

# Or run cmake directly with custom options — e.g. a flat build directory:
cmake -B build -S . -DAPP_<app>=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/nvidia/holoscan/lib
cmake --build build
```

Use `--dryrun --local` on the host to see the exact cmake commands the CLI would generate, then adapt them inside the container:

```bash
./holohub build <app> [mode] --dryrun --local    # prints: cmake -B build/<app> -S . -DAPP_<app>=ON ...
```

## Workflow

> Examples use `./holohub` — substitute your repo's entry point (see portability note at top).

### Inspect before running

`--dryrun` prints every command (prefixed `[dryrun]`) without executing anything. It works with every CLI command.

Add `--local` to bypass the Docker layer and see the underlying cmake / app commands directly. Without `--local`, you see the `docker build` + `docker run` wrapper instead.

```bash
./holohub build <app> [mode] --dryrun --local      # cmake configure + build commands
./holohub run <app> [mode] --dryrun --local # cmake + build + app run (with resolved placeholders)
./holohub build <app> [mode] --dryrun              # docker build + docker run wrapping the local build
```

`--verbose` adds env variable dumps, path mappings, and Docker launch details. In local mode, `--dryrun` already implies the same env output as `--verbose`.

### Test

```bash
./holohub test <app>                        # build + CTest in container (default)
./holohub test <app> --language python      # select language for multi-lang projects
./holohub test <app> --coverage             # enable coverage collection
./holohub test <app> --ctest-options "-R unit_tests"   # filter tests by regex
```

Default CTest script: `utilities/testing/holohub.container.ctest` (relative to the directory containing `utilities/`). Override with `--ctest-script` or `HOLOHUB_CTEST_SCRIPT`. Downstream repos typically override this in their entry-point script — check the value of `HOLOHUB_CTEST_SCRIPT` there.

### Iterate fast

Check for existing images first: `docker images | grep <project>`.

| Flag / Env                                                | Effect                                                                                                            |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--no-docker-build`                                       | Reuse existing image — default during dev. Rebuild only when Dockerfile, base image, or `--extra-scripts` change. |
| `--no-local-build`                                        | Skip cmake — for re-runs with different args or Python-only changes.                                              |
| `HOLOHUB_ALWAYS_BUILD=false`                              | Disable both build phases globally.                                                                               |
| `HOLOHUB_ENABLE_SCCACHE=true` + `--extra-scripts sccache` | Compiler cache at `~/.cache/sccache`.                                                                             |

### Debug inside the container

Prefer container over `--local` to avoid modifying the host. See [Execution Model](#execution-model) for how `build`/`run` re-enter the container.

```bash
./holohub run-container <app>               # interactive shell, repo at /workspace/<name>
./holohub run-container <app> -- <cmd>      # run a one-off command (passed to bash -c)
./holohub env-info                          # GPU, CUDA, Docker, Python diagnostics
```

Everything after `--` is joined into a single string and executed via `bash -c`, so standard shell syntax works inside quotes. Quote the command to use shell operators (`&&`, `|`, `;`), since the host shell would consume them otherwise:

```bash
./holohub run-container <app> -- "./holohub build <app> && ./holohub run <app>"
./holohub run-container <app> -- "cmake -B build -S . -DAPP_<app>=ON && cmake --build build"
```

### Resource management

- `CMAKE_BUILD_PARALLEL_LEVEL=N` — cap parallel jobs to prevent OOM.
- `./holohub clear-cache` removes cache directories matching `build*/`, `data*/`, and `install*/` at the repo root (e.g., `build/`, `build-*/`, `data/`, `data-*/`, `install/`, `install-*/`), plus any overridden build/data parent directories configured via `HOLOHUB_BUILD_PARENT_DIR` and `HOLOHUB_DATA_DIR` (which may live outside the repo). Ask for approval before running; use `--dryrun` first to preview.
  - `--build` — most common. Use when builds are broken, stale, or after switching branches, SDK versions, or build types.
  - `--data` — use when downloaded models or datasets are corrupt/incomplete, or to reclaim disk space. Re-downloading can be slow.
  - `--install` — use when installed artifacts are stale or from a different build configuration.
  - No flags — clears everything for a fresh start (e.g., switching SDK or CUDA versions).
- `docker image prune` / `docker system prune` — reclaim disk (ask for approval before running).
- Don't use `sudo ./holohub` — sudo filters `PATH` and other env vars.

### Gotchas

- Dash-prefixed values need `=`: `--run-args="--verbose"`, not `--run-args --verbose`.
- `--local` + `--docker-opts` is a fatal error.
- `--build-with` replaces `build.depends` entirely (does not append).
- `--benchmark` is a `build`-only flag and only applies to applications, workflows, and benchmarks.
- Root in container: `--docker-opts="-u root"`.

### Project configuration

- Each project's `metadata.json` drives automatic discovery, language selection, dependency tracking, Dockerfile resolution, run command templates, and mode definitions.
- `metadata.json` modes must be self-contained (Docker opts, run opts, env vars).
- `CMakeLists.txt` must guard inclusion behind `APP_<name>=ON` (or `OP_`/`EXT_`/`PKG_`).
- Dockerfiles resolve by walking up parent dirs — siblings share a common-ancestor Dockerfile.

**Metadata references** (for downstream repos without local access to HoloHub examples):

- [Base project schema](../metadata/project.schema.json) — all valid fields, types, and constraints
- [CLI Reference > Modes](cli_reference.md#modes) — mode fields, path placeholders, build/run config
- Examples: [simple app](https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/applications/endoscopy_tool_tracking/python/metadata.json), [multi-mode with docker opts](https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/applications/isaac_sim_holoscan_bridge/metadata.json), [multi-mode with build deps](https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/workflows/ai_surgical_video/python/metadata.json)

## Extending the CLI

1. Add parser entry in `_create_parser()` with `func=self.handle_<cmd>`.
2. Implement `handle_<cmd>()` using `run_command()`, honoring `args.dryrun`.
3. Wire config through `get_effective_build_config`/`get_effective_run_config`.
4. Add command to `handle_autocompletion_list()`.
5. Add tests under `utilities/cli/tests/` — mock `subprocess.run` and hardware detection; must pass on CPU-only nodes.
6. Lazy-import heavy modules inside handlers.

## Running CLI Tests

Tests mock all hardware and Docker calls —no GPU or Docker required. CI runs them across Python 3.10–3.13. Module paths below assume the working directory contains the `utilities/` package directly (the repo root in HoloHub). Downstream repos that vendor the CLI under a subdirectory (e.g. `tools/`) should set `PYTHONPATH` to the parent of `utilities/`:

```bash
# HoloHub (utilities/ at repo root):
python -m unittest utilities.cli.tests.test_cli
python -m unittest utilities.cli.tests.test_container
python -m unittest utilities.cli.tests.test_util

# Downstream repos where utilities/ lives under tools/:
PYTHONPATH=tools python -m unittest utilities.cli.tests.test_cli
PYTHONPATH=tools python -m unittest utilities.cli.tests.test_container
PYTHONPATH=tools python -m unittest utilities.cli.tests.test_util
```

If a container image is available, prefer running inside it for a clean environment:

```bash
./holohub run-container --no-docker-build -- python -m unittest utilities.cli.tests.test_cli
./holohub run-container --no-docker-build -- python -m unittest utilities.cli.tests.test_container
./holohub run-container --no-docker-build -- python -m unittest utilities.cli.tests.test_util
```
