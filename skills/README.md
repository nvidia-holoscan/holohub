# HoloHub agent skills

## Purpose

These skills guide coding agents through public HoloHub workflows using the
repository's `./holohub` command:

- [`holohub-app-lifecycle`](holohub-app-lifecycle/SKILL.md) covers non-failing
  application work from checkout selection through build, run, test, visual
  evidence, lint, and optional flow benchmarking.
- [`holohub-debug-build-run`](holohub-debug-build-run/SKILL.md) diagnoses one
  concrete failing, hanging, regressed, or incorrect `./holohub` command.
- [`holohub-module-lifecycle`](holohub-module-lifecycle/SKILL.md) covers
  reusable Holoscan Module development, consumer proof, and DEB/WHEEL
  packaging.

## Requirements

Use a valid HoloHub checkout and its checked-in `./holohub` wrapper. Read the
checkout's `AGENTS.md`, contribution guidance, schemas, and local command help
before acting. Docker, a supported GPU, display access, devices, data, or
credentials are required only when the selected project and requested proof
need them; each skill documents its own safety and approval boundaries.

## Usage

Select the skill that matches the request, open its `SKILL.md`, and load only
the references named by that workflow. Keep effect-bearing options identical
between each preview and real command, and require finite behavioral evidence
in addition to process success.

Examples:

- “Add a finite smoke mode to this working application” uses
  [`holohub-app-lifecycle`](holohub-app-lifecycle/SKILL.md).
- “Diagnose this exact failing `./holohub run` command” uses
  [`holohub-debug-build-run`](holohub-debug-build-run/SKILL.md).
- “Package this working Module as DEB and WHEEL and test clean consumers” uses
  [`holohub-module-lifecycle`](holohub-module-lifecycle/SKILL.md).

## Architecture

The three directories are independently installable skill packages. Each root
`SKILL.md` routes the task and defines the high-level contract; its
`references/` directory supplies focused operational detail, and `evals/`
captures positive and negative behavior cases. All execution stays on the
public `./holohub` command surface. The selected checkout, local help, schemas,
and repository guidance remain the live technical authority.

## Publication artifacts

Each skill directory is self-contained: start with its `SKILL.md` and read only
the linked references needed for the task.

Keep the publication artifacts in each directory together. The skill card,
external evaluation dataset, benchmark, and detached `skill.oms.sig` allow the
package to be verified and mirrored into the NVIDIA skills catalog. Generated
Markdown remains byte-identical to the signed source; verify its detached
signature instead of auto-formatting it after publication.
