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

`./holohub` delegates to the standalone `holoscan-cli` package through one
Python environment. On a host, it does not implicitly install into or reuse
system Python.

The wrapper selects an environment in this order:

1. An explicit `HOLOSCAN_CLI_PYTHON_BIN` or active `VIRTUAL_ENV`.
2. System Python while provisioning a container as root.
3. An existing wrapper-managed virtual environment.
4. A compatible system installation in a non-root container.
5. A new wrapper-managed virtual environment.

The managed environment defaults to
`${XDG_DATA_HOME:-$HOME/.local/share}/holoscan-cli/venv`. After selecting an
environment, the wrapper verifies the required CLI version and commands,
installs or repairs the package when allowed, and runs
`python -m holoscan_cli` with the original arguments.

Container images install the CLI into system Python so it remains available
after the Dockerfile changes users. At runtime, a non-root container reuses
that installation. If the image does not contain the required version, the
default behavior is to request an image rebuild instead of creating a virtual
environment in the mounted source tree.

The main overrides are:

| Variable | Effect |
| --- | --- |
| `HOLOSCAN_CLI_PYTHON_BIN` | Use a specific caller-managed Python interpreter. |
| `HOLOSCAN_CLI_VENV` | Change the managed environment location. |
| `HOLOSCAN_CLI_SOURCE` | Use a local holoscan-cli checkout for development. |
| `HOLOSCAN_CLI_PINNED_VERSION` | Override the required package version; an empty value enables floating mode. |

Host source and version overrides are not forwarded into Docker; containers
continue to use the version pinned in the wrapper. To force a clean reinstall,
remove only the managed environment; the next host invocation recreates it:

```bash
rm -rf "${XDG_DATA_HOME:-$HOME/.local/share}/holoscan-cli/venv"
```

Never run the wrapper with `sudo`. When an application needs root, use
`./holohub run <app> --as-root` (with or without `--local`): the build stays
user-owned and only the application phase runs as root.

## Using the wrapper pattern in another source project

Any holoscan-cli source project can adopt the same contract; a CLI upgrade
is then a one-line pin bump in that repo's wrapper:

1. Vendor one wrapper script that exports the project's `HOLOSCAN_CLI_*`
   identity variables, sets `HOLOSCAN_CLI_PINNED_VERSION`, and bootstraps
   holoscan-cli before delegating to `python -m holoscan_cli`. See the
   [module template](../../modules/template) wrapper.
2. Provision containers by running that wrapper in the Dockerfile, copied
   before anything else so the layer stays cached until the pin changes:

   ```dockerfile
   COPY --chmod=755 <wrapper> /tmp/scripts/
   RUN /tmp/scripts/<wrapper> env-info
   ```

3. Export `HOLOSCAN_CLI_IN_CONTAINER_CMD=./<wrapper>` so in-container
   recursion re-enters through the mounted wrapper and verifies the pin.

The wrapper file is the only version carrier; do not forward the CLI version
through Docker build args or environment variables.

## holoscan-cli smoke test

```bash
HOLOSCAN_CLI_SOURCE=/path/to/holoscan-cli \
  pytest -q utilities/cli/tests/test_holoscan_cli_consolidation.py
```

That single test exercises the wrapper -> install -> delegate path and
asserts that `holoscan list` / `modes` / `run --dryrun` work against this
repo.
