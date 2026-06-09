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

The `./holohub` script at the repo root does three things in order:

1. Probes the installed command surface — `python3 -m holoscan_cli --help |
   grep -qw build` — rather than just checking importability. This ensures a
   source-project-capable CLI is present, so a stale legacy `holoscan-cli`
   (<= 4.2.0, which lacks `build`/`list`) does not satisfy the check.
2. If the probe fails, it bootstraps pip when missing and `pip install`s the
   package. The install spec is chosen in this precedence order:

   | Env var                | Behavior                                                    |
   | ---------------------- | ----------------------------------------------------------- |
   | `HOLOSCAN_CLI_SOURCE`  | Path to a local checkout. Wins over `HOLOSCAN_CLI_VERSION`. |
   | `HOLOSCAN_CLI_VERSION` | Pip spec; defaults to the pinned `holoscan-cli==4.3.0rc2`.  |

   A bare version or comparator (`4.3.0`, `>=4.3`) given as
   `HOLOSCAN_CLI_VERSION` is normalized to a valid `holoscan-cli` requirement;
   full specs, URLs, and `git+https://...` pass through unchanged.

3. Exports the `HOLOSCAN_CLI_*` environment variables that configure the CLI
   for this repo, then delegates the user's argv to
   `python3 -m holoscan_cli`.

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
