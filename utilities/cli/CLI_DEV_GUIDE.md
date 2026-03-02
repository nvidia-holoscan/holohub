# CLI Developer Guide

Reference for AI agents and developers working with or extending the CLI.
For command/flag docs, see the [CLI Reference](https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/doc/cli_reference.md).

## Agent Safety Rules

1. Always `--dryrun --verbose` before running any command for real.
2. Never `build --local` or `run --local` without explicit user approval — they run directly in the host workspace, may modify files there, and require host dependencies to already be installed (no automatic package installation).
3. Never hardcode repo names, command names, or path prefixes — the CLI is reused across repos via env vars.
4. Check the project's README and `metadata.json` for architecture/platform requirements before building.

## Implementation Invariants

| # | Rule | Detail |
|---|------|--------|
| 1 | Container re-enters local | `build`/`run`/`install` in container mode build an image, then run the CLI with `--local` inside it. Local-mode changes affect both host and container execution. |
| 2 | CLI flags override modes | Resolution: `resolve_mode` → `validate_mode` → `get_effective_build_config`/`get_effective_run_config`. Most flags override (do not merge with) mode values; `--run-args` is **appended** to the mode `run.command`. |
| 3 | Expand placeholders first | Mode commands may contain `<holohub_app_source>`, `<holohub_data_dir>`, etc. Run through `build_holohub_path_mapping()` + `replace_placeholders()`. |
| 4 | Use `run_command()` | Handles dry-run, sudo detection, fail-fast (`check=True`). Use `run_info_command()` for best-effort probes. |
| 5 | Use `self.script_name` / `self.prefix` | `self.script_name` (from `HOLOHUB_CMD_NAME`) for user messages. `self.prefix` (from `HOLOHUB_PATH_PREFIX`) for placeholder resolution. |

## Workflow

### Inspect before running

```bash
build <app> --dryrun              # full Docker build command
run <app> --dryrun --local        # cmake + app run with resolved placeholders
test <app> --dryrun --verbose     # full test pipeline (docker, cmake, ctest)
```

`--verbose` adds variable dumps (build-args, docker-run flags, PYTHONPATH, data paths).

### Test

```bash
test <app>                        # build + CTest in container (default)
test <app> --language python      # select language for multi-lang projects
test <app> --coverage             # enable coverage collection
test <app> --ctest-options "-R unit_tests"   # filter tests by regex
```

Default CTest script: `utilities/testing/holohub.container.ctest`. Override with `--ctest-script` or `HOLOHUB_CTEST_SCRIPT`.

### Iterate fast

Check for existing images first: `docker images | grep <project>`.

| Flag / Env | Effect |
|------------|--------|
| `--no-docker-build` | Reuse existing image — default during dev. Rebuild only when Dockerfile, base image, or `--extra-scripts` change. |
| `--no-local-build` | Skip cmake — for re-runs with different args or Python-only changes. |
| `HOLOHUB_ALWAYS_BUILD=false` | Disable both build phases globally. |
| `HOLOHUB_ENABLE_SCCACHE=true` + `--extra-scripts sccache` | Compiler cache at `~/.cache/sccache`. |

### Debug inside the container

Prefer container over `--local` to avoid modifying the host.

```bash
run-container <app>               # interactive shell, repo at /workspace/<name>
run-container <app> -- <cmd>      # run a specific command (e.g. ./holohub env-info)
env-info                          # GPU, CUDA, Docker, Python diagnostics
```

Inside the shell: run cmake, ctest, or `./holohub <cmd> --local` directly.

### Resource management

- `CMAKE_BUILD_PARALLEL_LEVEL=N` — cap parallel jobs to prevent OOM.
- `clear-cache --build` — reset build dir only. Bare `clear-cache` also deletes `data/` (models, datasets).
- `docker image prune` / `docker system prune` — reclaim disk.
- Don't use `sudo ./holohub` — sudo filters `PATH` and other env vars.

### Gotchas

- Dash-prefixed values need `=`: `--run-args="--verbose"`, not `--run-args --verbose`.
- `--local` + `--docker-opts` is a fatal error.
- `--build-with` replaces `build.depends` entirely (does not append).
- `--benchmark` is a `build`-only flag and only applies to applications, workflows, and benchmarks.
- Root in container: `--docker-opts="-u root"`.

### Project configuration

- `metadata.json` modes must be self-contained (Docker opts, run opts, env vars).
- `CMakeLists.txt` must guard inclusion behind `APP_<name>=ON` (or `OP_`/`EXT_`/`PKG_`).
- Dockerfiles resolve by walking up parent dirs — siblings share a common-ancestor Dockerfile.

**Metadata references** (for downstream repos without local access to HoloHub examples):

- [Base project schema](https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/utilities/metadata/project.schema.json) — all valid fields, types, and constraints
- [CLI Reference > Modes](https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/doc/cli_reference.md) — mode fields, path placeholders, build/run config
- Examples: [simple app](https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/applications/endoscopy_tool_tracking/python/metadata.json), [multi-mode with docker opts](https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/applications/isaac_sim_holoscan_bridge/metadata.json), [multi-mode with build deps](https://raw.githubusercontent.com/nvidia-holoscan/holohub/main/workflows/ai_surgical_video/python/metadata.json)

## Source Layout

| File | Purpose |
|------|---------|
| `holohub.py` | Parser, metadata collection, mode resolution, command handlers |
| `container.py` | Image naming/build/run, docker arg composition, device mounts |
| `util.py` | Subprocess wrappers, env/path helpers, host capability detection |
| `tests/` | `unittest` + `@patch` behavior tests |

## Extending the CLI

1. Add parser entry in `_create_parser()` with `func=self.handle_<cmd>`.
2. Implement `handle_<cmd>()` using `run_command()`, honoring `args.dryrun`.
3. Wire config through `get_effective_build_config`/`get_effective_run_config`.
4. Add command to `handle_autocompletion_list()`.
5. Add tests under `utilities/cli/tests/` — mock `subprocess.run` and hardware detection; must pass on CPU-only nodes.
6. Lazy-import heavy modules inside handlers.

## Running CLI Tests

Tests mock all hardware and Docker calls — no GPU or Docker required. CI runs them across Python 3.10–3.13.

```bash
python -m unittest utilities.cli.tests.test_cli
python -m unittest utilities.cli.tests.test_container
python -m unittest utilities.cli.tests.test_util
```

If a container image is available, prefer running inside it for a clean environment:

```bash
./holohub run-container --no-docker-build -- python -m unittest utilities.cli.tests.test_cli
```
