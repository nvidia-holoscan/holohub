# CLI Developer Guide

Reference for AI agents and developers working with or extending the CLI.
For command/flag docs, see the [CLI Reference](cli_reference.md).

> [!NOTE] Portability
> This guide applies to any repo built on the HoloHub CLI.
> Examples use `./holohub`; downstream repos have their own entry point
> (e.g. `./i4h`, `./isaac_os`), configured via `HOLOSCAN_CLI_CMD_NAME`.
> Currently `HOLOSCAN_CLI_`-prefixed env vars are shared CLI infrastructure, not
> HoloHub-specific. Substitute your repo's entry point wherever you
> see `./holohub`.

## Agent Safety Rules

1. Preview mutating commands with the flags they support:
   - `build`, `run`, `build-container`, `run-container`, `install`, `test`,
     and `package`: use `--dryrun --verbose`.
   - `create`, `lint`, `setup`, and `clear-cache`: use `--dryrun`; they do not
     accept `--verbose`.
   - `list`, `modes`, `env-info`, `env-check`, `status`, and `version` are
     read-only diagnostics and accept neither preview flag.
2. Never `./holohub build --local` or `./holohub run --local` without explicit user approval — they run directly in the host workspace, may modify files there, and require host dependencies to already be installed (no automatic package installation).
3. Never hardcode repo names, command names, or path prefixes — the CLI is reused across repos via env vars.
4. Check the project's README and `metadata.json` for architecture/platform (e.g. x86_64, aarch64, IGX, Thor, DGX Spark) requirements before building.  The project may not support all platforms, do not suggest building on platforms that are not supported unless the user explicitly asks for it.
5. By default, do not suggest running cmake, python, or application binaries directly on the host. The CLI runs everything inside a container by default. If a user needs to run commands manually (custom cmake flags, flat build directory, debugging), the answer is `./holohub run-container <app>` first, then run commands inside. Raw host commands require the same explicit user approval as `--local`.

## Mental Model

The CLI is a **container-first** build system. In normal container mode,
`build`, `run`, `install`, and `package` resolve the project on the host, build
or reuse its Docker image, and then re-invoke the CLI with `--local` inside
that container. `test` uses the same image setup but launches CTest directly.
The host handles wrapper bootstrap, metadata resolution, and Docker
orchestration; cmake, python, packaging, testing, and application processes
execute inside the container. `--local`, the `--no-*-build` flags, and
`HOLOSCAN_CLI_ALWAYS_BUILD=false` intentionally alter that flow.

When a user asks how to customize a build or run — change the build directory, add cmake flags, use a different build type — the answer is `./holohub run-container <app>` first, then run commands inside (see [Why this matters](#why-this-matters) for examples).

## Implementation Invariants

1. Container re-entry: `build`/`run`/`install`/`package` in container mode
   build or reuse an image, then run the CLI with `--local` inside it. `test`
   launches its CTest command directly in the selected image. Local-mode
   changes affect both host and re-entered container execution.
2. CLI flags override modes: resolution is `resolve_mode` -> `validate_mode` ->
   `get_effective_build_config`/`get_effective_run_config`. Most flags override
   mode values; `--run-args` is appended to the argument list passed to
   `docker run` or the application process.
3. Expand placeholders first: mode commands may contain
   `<{prefix}app_source>`, `<{prefix}data_dir>`, etc. The prefix comes from
   `HOLOSCAN_CLI_PATH_PREFIX` and defaults to `holohub_`. Run through
   `build_holohub_path_mapping()` and `replace_placeholders()`.
4. Use `run_command()`: it handles dry-run, sudo detection, and fail-fast
   behavior. Use `run_info_command()` for best-effort probes.
5. Use `self.script_name` / `self.prefix`: `self.script_name` comes from
   `HOLOSCAN_CLI_CMD_NAME`; `self.prefix` comes from
   `HOLOSCAN_CLI_PATH_PREFIX` for placeholder resolution.

## Execution Model

`build`, `run`, `install`, `test`, and `package` follow a two-phase container
pattern by default (without `--local`):

### Phase 1 — Container setup (runs on host)

1. Build a Docker image with project dependencies (`docker build`), unless
   `--no-docker-build` or `HOLOSCAN_CLI_ALWAYS_BUILD=false` skips that step.
2. Launch a container (`docker run`) with the repo bind-mounted at `/workspace/<name>` (`HOLOSCAN_CLI_WORKSPACE_NAME`, default `holohub`) and `HOLOSCAN_CLI_BUILD_LOCAL=1` set in the environment.

### Phase 2 — Build / run (runs inside the container)

For `build`, `run`, `install`, and `package`, the container re-invokes the CLI
with `--local` appended. For example, `./holohub build myapp` on the host
becomes `./holohub build myapp --local` inside the container. `test` launches
its CTest command directly in the container.

| Command   | What runs inside the container                                                 |
| --------- | ------------------------------------------------------------------------------ |
| `build`   | `cmake -B build/<app> -S . -DAPP_<app>=ON [opts] && cmake --build build/<app>` |
| `run`     | Same cmake build, then the mode's `run.command` with placeholders expanded     |
| `install` | Same cmake build, then `cmake --install build/<app>`                           |
| `package` | Module configure/build plus requested CPack and/or wheel generation            |
| `test`    | CTest via the configured CTest script                                          |

The cmake flag `-DAPP_<app>=ON` selects a single project (operators use `-DOP_<name>=ON`, extensions use `-DEXT_<name>=ON`). The build directory `build/<app>/` isolates per-project artifacts; this path is `HOLOSCAN_CLI_BUILD_PARENT_DIR/<app>` (default parent varies by repo; `<repo_root>/build` in HoloHub). Because the repo is bind-mounted, artifacts persist on the host after the container exits.

`--local` bypasses both phases and runs cmake / the app directly on the host.

### Why this matters

By default (without `--local`) cmake, python, packaging, and application
commands execute **inside the container**, not on the host. Build-skip controls
can reuse existing image or project artifacts, but do not move those commands
to the host. To run commands manually — customize the build directory, add
cmake flags, or debug a failure — enter the container first, then run whatever
you need:

```bash
# 1. Enter the container (builds the image if needed, then opens an interactive shell):
./holohub run-container <app> [mode]

# 2. Inside the container, HOLOSCAN_CLI_BUILD_LOCAL=1 is already set, so all
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

For commands that support it, `--dryrun` prints external commands (prefixed
`[dryrun]`) without executing them. Read-only diagnostics do not need a
preview mode.

Add `--local` to bypass the Docker layer and see the underlying cmake / app commands directly. Without `--local`, you see the `docker build` + `docker run` wrapper instead.

```bash
./holohub build <app> [mode] --dryrun --local      # cmake configure + build commands
./holohub run <app> [mode] --dryrun --local # cmake + build + app run (with resolved placeholders)
./holohub build <app> [mode] --dryrun              # docker build + docker run wrapping the local build
```

For container/build commands that support it, `--verbose` adds environment
variables, path mappings, and Docker launch details. In local mode,
`--dryrun` already implies the same environment output as `--verbose`.

### Test

```bash
./holohub test <app>                        # build + CTest in container (default)
./holohub test <app> --language python      # select language for multi-lang projects
./holohub test <app> --coverage             # enable coverage collection
./holohub test <app> --ctest-options "-R unit_tests"   # filter tests by regex
```

Default CTest script: `utilities/testing/holohub.container.ctest` (relative to the directory containing `utilities/`). Override with `--ctest-script` or `HOLOSCAN_CLI_CTEST_SCRIPT`. Downstream repos typically override this in their entry-point script - check the value of `HOLOSCAN_CLI_CTEST_SCRIPT` there.

### Lint

```bash
./holohub lint                              # pre-commit run --all-files
./holohub lint applications/<app>           # pre-commit run --files <files under app>
./holohub lint --install-dependencies       # explicit setup/prefetch
```

`./holohub lint` is a thin wrapper around [pre-commit](https://pre-commit.com/) and uses the hooks declared in `.pre-commit-config.yaml`. If `pre-commit` is not already available, the wrapper installs it before running lint. `--fix` is kept as a no-op compatibility alias because pre-commit hooks already auto-fix where possible.

In a fresh container, `./holohub lint` installs `pre-commit`
automatically if needed:

```bash
./holohub run-container -- ./holohub lint
```

Because `run-container` sets `HOME` to the mounted workspace, the generated
`.local/` install and `.cache/pre-commit/` hook environments persist across
later container runs and are already ignored by git. Use
`./holohub lint --install-dependencies` when you want to prefetch hooks or
install dependencies without running lint.

Downstream projects may override the upstream lint handling (or simply ship their own `.pre-commit-config.yaml`) to point lint at different tooling. If no `.pre-commit-config.yaml` is present at the project root, the command exits zero with a recommendation so downstream wrappers without a config don't break their CI.

### Iterate fast

Check for existing images first: `docker images | grep <project>`.

| Flag / Env                                                     | Effect                                                                                                            |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--no-docker-build`                                            | Reuse an existing image during iteration. Rebuild when the Dockerfile, base image, or `--extra-scripts` change.   |
| `--no-local-build`                                             | Skip cmake — for re-runs with different args or Python-only changes.                                              |
| `HOLOSCAN_CLI_ALWAYS_BUILD=false`                              | With `run`, reuse both image and project artifacts; other container commands skip only the image build.           |
| `HOLOSCAN_CLI_ENABLE_SCCACHE=true` + `--extra-scripts sccache` | Compiler cache at `~/.cache/sccache`.                                                                             |

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
- `./holohub clear-cache` removes cache directories matching `build*/`, `data*/`, and `install*/` at the repo root (e.g., `build/`, `build-*/`, `data/`, `data-*/`, `install/`, `install-*/`), plus any overridden build/data parent directories configured via `HOLOSCAN_CLI_BUILD_PARENT_DIR` and `HOLOSCAN_CLI_DATA_DIR` (which may live outside the repo). Ask for approval before running; use `--dryrun` first to preview.
  - `--build` — most common. Use when builds are broken, stale, or after switching branches, SDK versions, or build types.
  - `--data` — use when downloaded models or datasets are corrupt/incomplete, or to reclaim disk space. Re-downloading can be slow.
  - `--install` — use when installed artifacts are stale or from a different build configuration.
  - No flags — clears everything for a fresh start (e.g., switching SDK or CUDA versions).
- `docker image prune` / `docker system prune` — reclaim disk (ask for approval before running).
- Don't use `sudo ./holohub` — sudo filters `PATH` and other env vars. Use
  `./holohub run <app> --as-root` instead: the build stays user-owned and only
  the application phase runs as root.

### Gotchas

- Dash-prefixed values need `=`: `--run-args="--verbose"`, not `--run-args --verbose`.
- `--local` + `--docker-opts` is a fatal error.
- `--build-with` replaces `build.depends` entirely (does not append).
- `--benchmark` is a `build`-only flag and only applies to applications and benchmarks.
- Root application: `run --as-root`. Root whole container: `run-container --as-root`.

### Project configuration

- Each project's `metadata.json` drives automatic discovery, language selection, dependency tracking, Dockerfile resolution, run command templates, and mode definitions.
- Application `metadata.json` files may define either a top-level `run` or
  `modes`, but not both. Every mode requires `description` and `run`; two or
  more modes also require `default_mode`.
- `CMakeLists.txt` must guard inclusion behind `APP_<name>=ON` (or `OP_`/`EXT_`/`PKG_`).
- Dockerfiles resolve by walking up parent dirs — siblings share a common-ancestor Dockerfile.

**Metadata references** (for downstream repos without local access to HoloHub examples):

- [Base project schema](../metadata/project.schema.json) — all valid fields, types, and constraints
- [CLI Reference > Modes](cli_reference.md#application-modes) — mode fields,
  path placeholders, and build/run configuration
- Examples: [simple app](https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/applications/endoscopy_tool_tracking/python/metadata.json), [multi-mode with docker opts](https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/applications/isaac_sim_holoscan_bridge/metadata.json), [multi-mode with build deps](https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/applications/ai_surgical_video/python/metadata.json)

## Extending the CLI

The CLI implementation now lives in the standalone
[`holoscan-cli`](https://github.com/nvidia-holoscan/holoscan-cli) package.
To add or modify a subcommand, parser entry, mode-resolution rule, or
container-build behavior, open a PR there — the changes flow to HoloHub
on the next `holoscan-cli` release. The contribution workflow is documented
in [holoscan-cli/CONTRIBUTING.md](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/CONTRIBUTING.md).

The HoloHub-side tests that exercise the CLI against this repo's
project tree live under [`utilities/cli/tests/`](tests/) and are
registered in [`utilities/cli/tests/CMakeLists.txt`](tests/CMakeLists.txt).
Add a new `add_test(COMMAND ${CMAKE_SOURCE_DIR}/holohub <subcmd> ...)`
entry there if your change needs an integration test against the real
HoloHub layout (real `metadata.json`, real applications, etc.). Pure
CLI-internal unit tests belong in the holoscan-cli repo under
`tests/unit/`.

## Running CLI Tests

CLI-internal unit tests run from the upstream `holoscan-cli` checkout:

```bash
git clone https://github.com/nvidia-holoscan/holoscan-cli.git
cd holoscan-cli
python -m pytest tests/unit/
```

The HoloHub-integration smoke tests run via CTest from this repo:

```bash
cmake -B build -DBUILD_TESTING=ON -DBUILD_HOLOHUB_TESTING=ON .
ctest --test-dir build -V
```

This registers ~10 black-box `add_test(COMMAND ./holohub <subcmd>)`
checks (project listing, autocompletion, mode resolution, env-var
forwarding, generated-project smoke) plus the pytest-based
`holoscan_cli_consolidation` and `holohub_create_module` smokes.
