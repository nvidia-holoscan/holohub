# HoloHub CLI

The `./holohub` command is provided by the standalone
[**holoscan-cli**](https://github.com/nvidia-holoscan/holoscan-cli) package,
which HoloHub installs (or detects) the first time the wrapper is invoked.
The Python implementation that used to live under this directory
(`holohub.py`, `container.py`, `util.py`, `status.py`, `system_check.py`,
`version_check.py`) has moved to the holoscan-cli package; HoloHub is now
one of its downstream consumers alongside Isaac OS and the I4H Workflows
repos.

For the canonical command, option, and environment-variable reference, run
`holoscan --help` (or `holoscan env-info` for the full env-var surface), or
visit the [holoscan-cli](https://github.com/nvidia-holoscan/holoscan-cli)
repository — that is now the single source of truth and stays current as
the CLI evolves. The historical `cli_reference.md` in this directory
documents the old in-tree CLI and is no longer authoritative.

This directory keeps the user-facing CLI documentation:

| Document                                | Description                                                                                                                  |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| [CLI Developer Guide](cli_dev_guide.md) | Workflow tips, implementation invariants, and extension guide. For the canonical source layout, see the holoscan-cli repo.   |

## Quick Start

```bash
./holohub <command> [options] [arguments]
./holohub -h | --help                    # List all commands
./holohub <command> -h | --help          # Options for a specific command
./holohub list                           # List available projects
```

## How the wrapper resolves the CLI

The `./holohub` script selects one Python environment in this order:

1. `HOLOSCAN_CLI_PYTHON_BIN` (or the legacy override) and an active virtual
   environment are caller-owned.
2. An existing wrapper-managed venv is reused.
3. Root **in a container** (detected by container markers, or by an
   environment that already opts into system installation via
   `PIP_BREAK_SYSTEM_PACKAGES=1`, as Holoscan SDK base images do) uses the
   container's system Python. Root on a host is not, by itself, this signal:
   a direct root login falls through to the next steps.
4. A compatible system CLI is reused without wrapper pip installation. This
   supports existing `sudo pip` installations and non-root runtime containers.
5. Otherwise the wrapper creates and provisions its managed venv (concurrent
   first runs are serialized by a lock). If venv creation fails while running
   as root — for example a minimal image without `python3-venv` — the wrapper
   falls back to the system interpreter loudly instead of failing the build.

A venv isolates Python packages, not operating-system resources; commands still
have normal access to devices, CUDA, files, networking, and system libraries.
On a host — ordinary, sudo, or direct root — wrapper bootstrap never installs
into system Python unless the caller explicitly selects that interpreter (an
explicit `PIP_BREAK_SYSTEM_PACKAGES=0` opt-out is also respected in
containers). Under `sudo` that preserves the caller's `HOME`/`XDG_DATA_HOME`,
the wrapper refuses to create or modify a managed venv inside directories the
caller owns — re-run without sudo, or use `sudo -H` for a root-owned HOME.

It then does three things in order:

1. Probes the selected environment's command surface with
   `python -m holoscan_cli build --help` rather than just checking
   importability. This ensures a
   source-project-capable CLI is present, so a stale legacy `holoscan-cli`
   (<= 4.2.0, which lacks `build`/`list`) does not satisfy the check. When
   `HOLOSCAN_CLI_PINNED_VERSION` is set, package metadata must also match that
   exact version.
2. If the probe fails, it bootstraps pip when missing and `pip install`s the
   package. The install spec is chosen in this precedence order:

   | Env var                       | Behavior                                                                                                 |
   | ----------------------------- | -------------------------------------------------------------------------------------------------------- |
   | `HOLOSCAN_CLI_SOURCE`         | Valid local checkout. Wins over package installation and bypasses package-version matching.              |
   | `HOLOSCAN_CLI_INSTALL_ARGS`   | Pip arguments; defaults to `--pre --extra-index-url https://pypi.nvidia.com holoscan-cli>4.2.0`.         |
   | `HOLOSCAN_CLI_PINNED_VERSION` | Exact canonical package version; mismatch is reconciled and verified. Empty keeps floating behavior.     |

   The exact requirement is appended to the effective pip arguments. It is
   also forwarded through the existing Docker build argument; Dockerfiles that
   declare and consume `ARG HOLOSCAN_CLI_INSTALL_ARGS` receive the constraint.
   Floating mode keeps an already-compatible CLI; it does not upgrade on every
   invocation.

3. Exports the `HOLOSCAN_CLI_*` environment variables that configure the CLI
   for this repo, then delegates the user's argv to
   the selected environment's `python -m holoscan_cli`.

The legacy `HOLOHUB_*` env-var spellings are still honored for one release
with a one-line deprecation warning; this wrapper sets the new names
proactively so the warning never fires.

## holoscan-cli smoke test

```bash
HOLOSCAN_CLI_SOURCE=/path/to/holoscan-cli \
  pytest -q utilities/cli/tests/test_holoscan_cli_consolidation.py
```

That single test exercises the wrapper -> install -> delegate path and
asserts that `holoscan list` / `modes` / `run --dryrun` work against this
repo.
