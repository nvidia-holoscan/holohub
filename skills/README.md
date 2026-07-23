# HoloHub agent skills

These skills guide coding agents through public HoloHub workflows using the
repository's `./holohub` command.

- [`holohub-app-lifecycle`](holohub-app-lifecycle/SKILL.md) covers non-failing
  application work from checkout selection through build, run, test, visual
  evidence, lint, and optional flow benchmarking.
- [`holohub-debug-build-run`](holohub-debug-build-run/SKILL.md) diagnoses one
  concrete failing, hanging, regressed, or incorrect `./holohub` command.
- [`holohub-module-lifecycle`](holohub-module-lifecycle/SKILL.md) covers
  reusable Holoscan Module development, consumer proof, and DEB/WHEEL
  packaging.

Each skill directory is self-contained: start with its `SKILL.md` and read only
the linked references needed for the task. The checked-out HoloHub revision,
local command help, schemas, and repository guidance remain the live authority.

Keep the publication artifacts in each directory together. The skill card,
external evaluation dataset, benchmark, and detached `skill.oms.sig` allow the
package to be verified and mirrored into the NVIDIA skills catalog.
