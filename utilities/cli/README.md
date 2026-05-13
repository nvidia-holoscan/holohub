# HoloHub CLI

The `./holohub` command is provided by the standalone
[**holoscan-cli**](https://github.com/nvidia-holoscan/holoscan-cli) package,
which HoloHub installs (or detects) the first time the wrapper is invoked.
The Python implementation that used to live under this directory
(`holohub.py`, `container.py`, `util.py`, `status.py`, `system_check.py`,
`version_check.py`) has moved to the consolidated package; HoloHub is now
one of its downstream consumers alongside Isaac OS and the I4H Workflows
repos.

This directory keeps the user-facing CLI documentation:

| Document                                | Description                                                                                                                  |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| [CLI Reference](cli_reference.md)       | Complete command reference: all commands, options, modes, environment variables, and examples.                               |
| [CLI Developer Guide](cli_dev_guide.md) | Workflow tips, implementation invariants, and extension guide. For the canonical source layout, see the holoscan-cli repo.   |

## Quick Start

```bash
./holohub <command> [options] [arguments]
./holohub -h | --help                    # List all commands
./holohub <command> -h | --help          # Options for a specific command
./holohub list                           # List available projects
```

## How the wrapper resolves the CLI

The `./holohub` script at the repo root does three things in order:

1. Imports `holoscan_cli` and checks that the installed version exposes the
   source-project dispatch table (`PROJECT_COMMANDS`). Pre-consolidation
   releases (e.g. `holoscan-cli==4.0.0`) import cleanly but do not satisfy
   this check.
2. If the check fails, `pip install`s the package. The install spec is
   chosen in this precedence order:

   | Env var                | Behavior                                                                       |
   | ---------------------- | ------------------------------------------------------------------------------ |
   | `HOLOSCAN_CLI_SOURCE`  | Path to a local checkout. Wins over `HOLOSCAN_CLI_VERSION`.                    |
   | `HOLOSCAN_CLI_VERSION` | Pip spec — `4.3.0`, `>=4.3`, `git+https://...`. Defaults to `holoscan-cli`.    |

3. Exports the `HOLOSCAN_CLI_*` environment variables that configure the CLI
   for this repo, then delegates the user's argv to
   `python3 -m holoscan_cli`.

The legacy `HOLOHUB_*` env-var spellings are still honored for one release
with a one-line deprecation warning; this wrapper sets the new names
proactively so the warning never fires.

## Consolidation smoke test

```bash
HOLOSCAN_CLI_SOURCE=/path/to/holoscan-cli \
  pytest -q utilities/cli/tests/test_holoscan_cli_consolidation.py
```

That single test exercises the wrapper -> install -> delegate path and
asserts that `holoscan list` / `modes` / `run --dryrun` work against this
repo.
