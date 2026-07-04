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

1. An explicit `HOLOSCAN_CLI_PYTHON_BIN` (or legacy
   `HOLOHUB_PYTHON_BIN`) and an active `VIRTUAL_ENV` are caller-owned and win.
2. Root uses system Python inside a container (container markers, or the
   `/tmp/scripts` Dockerfile bootstrap location).
   This check intentionally precedes an implicit `/root` managed venv so a CLI
   installed while building a container remains available after `USER` changes.
3. A healthy wrapper-managed venv is reused.
4. A compatible system CLI is reused read-only. This supports previous global
   installations and containers built as root but run as a regular user.
5. Otherwise the wrapper creates and provisions its managed venv. A failed
   venv creation is reported with the `python3-venv` installation hint; it
   never falls back to modifying system Python.

A venv isolates Python packages, not operating-system resources; commands still
have normal access to devices, CUDA, files, networking, and system libraries.
On a host, wrapper bootstrap does not modify system Python: root installs
system-wide only with a positive container marker or when the wrapper's
physical directory is exactly `/tmp/scripts`, the controlled Dockerfile
bootstrap location. A compatible system CLI may still be reused read-only, and
an explicit caller-owned interpreter always has highest precedence.

`PIP_BREAK_SYSTEM_PACKAGES` controls pip only after system scope has been
selected; it is not a host/container signal. The wrapper defaults it to `1` in
root system scope and respects an explicit `0`. Under `sudo` that preserves a
caller's `HOME`/`XDG_DATA_HOME`, the wrapper refuses to use or create a managed
venv below directories that are not root-owned. Re-run without sudo, or
use `sudo -H` for a root-owned home. Explicit interpreter and active-venv
overrides are trusted and bypass this managed-path check; do not preserve them
through `sudo` unless that is intentional.

It then does three things in order:

1. Probes the selected environment's command surface with
   `python -m holoscan_cli build --help` rather than just checking
   importability. This ensures a
   source-project-capable CLI is present, so a stale legacy `holoscan-cli`
   (<= 4.2.0, which lacks `build`/`list`) does not satisfy the check. When
   `HOLOSCAN_CLI_PINNED_VERSION` is set, package metadata must also match that
   exact version.
2. If the probe fails, it bootstraps pip when missing and `pip install`s the
   package. The source/package choice has this precedence:

   - `HOLOSCAN_CLI_SOURCE` selects a valid local checkout, wins over package
     installation, and bypasses package-version matching.
   - `HOLOSCAN_CLI_INSTALL_ARGS` supplies single-line, whitespace-separated
     pip tokens. It defaults to `--pre --extra-index-url
     https://pypi.nvidia.com holoscan-cli>4.2.0`; shell quotes are not parsed.
   - `HOLOSCAN_CLI_PINNED_VERSION` supplies one exact package version. A
     mismatch is installed and verified; an empty value keeps floating mode.

   The exact requirement is appended to the effective pip arguments. It is
   also forwarded through the existing Docker build argument; Dockerfiles that
   declare and consume `ARG HOLOSCAN_CLI_INSTALL_ARGS` receive the constraint.
   Floating mode keeps an already-compatible CLI; it does not upgrade on every
   invocation.

3. Exports the `HOLOSCAN_CLI_*` environment variables that configure the CLI
   for this repo, then delegates the user's argv to
   the selected environment's `python -m holoscan_cli`.

### Wrapper configuration

| Variable | Default | Effect |
| --- | --- | --- |
| `HOLOSCAN_CLI_PYTHON_BIN` | unset | Explicit caller-owned Python interpreter; highest precedence. |
| `HOLOHUB_PYTHON_BIN` | unset | Legacy alias for `HOLOSCAN_CLI_PYTHON_BIN`. |
| `VIRTUAL_ENV` | unset | Its `bin/python` is used when no explicit interpreter is set. |
| `HOLOSCAN_CLI_VENV` | `${XDG_DATA_HOME:-$HOME/.local/share}/holoscan-cli/venv` | Location of the wrapper-managed environment. Set it explicitly when neither `HOME` nor `XDG_DATA_HOME` exists and a managed environment is needed. |
| `HOLOSCAN_CLI_SOURCE` | unset | Local holoscan-cli checkout; prepends `<checkout>/src` to `PYTHONPATH` and installs from that checkout if needed. |
| `HOLOSCAN_CLI_INSTALL_ARGS` | `--pre --extra-index-url https://pypi.nvidia.com holoscan-cli>4.2.0` | Floating package and index arguments, parsed as single-line whitespace-separated tokens without shell evaluation. |
| `HOLOSCAN_CLI_PINNED_VERSION` | unset | One exact package version; appends `holoscan-cli==<version>` to the effective pip/Docker arguments. Ignored with `HOLOSCAN_CLI_SOURCE`. |
| `PIP_BREAK_SYSTEM_PACKAGES` | `1` in root system scope | Passed to pip for PEP 668 behavior. It does not authorize or select system installation. |
| `HOLOSCAN_CLI_BASE_SDK_VERSION` | `4.4.0` | Default Holoscan SDK base-image version used by CLI container commands. |
| `HOLOSCAN_CLI_CTEST_SCRIPT` | `utilities/testing/holohub.container.ctest` | CTest driver used by `./holohub test`. |
| `HOLOSCAN_CLI_DEFAULT_DOCKER_BUILD_ARGS` | unset | Existing default Docker build arguments; the wrapper appends its effective CLI install constraint. |

Floating mode keeps any compatible CLI and does not upgrade it on every run.
To force a version, set `HOLOSCAN_CLI_PINNED_VERSION`; the wrapper reconciles
and verifies it. To inspect or manage the default venv directly:

```bash
cli_venv="${XDG_DATA_HOME:-$HOME/.local/share}/holoscan-cli/venv"
"${cli_venv}/bin/python" -m pip show holoscan-cli
"${cli_venv}/bin/python" -m pip install --upgrade \
  --pre --extra-index-url https://pypi.nvidia.com 'holoscan-cli>4.2.0'
rm -rf "${cli_venv}"  # remove only the wrapper-managed CLI environment
```

After removal, the next invocation resolves again: it may reuse a compatible
system CLI read-only or create a new managed environment.

## holoscan-cli smoke test

```bash
HOLOSCAN_CLI_SOURCE=/path/to/holoscan-cli \
  pytest -q utilities/cli/tests/test_holoscan_cli_consolidation.py
```

That single test exercises the wrapper -> install -> delegate path and
asserts that `holoscan list` / `modes` / `run --dryrun` work against this
repo.
