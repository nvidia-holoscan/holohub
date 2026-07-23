# `./holohub` debugging workflow

Use this after capturing one exact failing command.

## Baseline and reproduction

Read applicable `AGENTS.md`, project and CLI documentation, metadata, schemas,
and local help. Record project, mode, language, build type, image, input,
display/headless state, devices, output path, full HEAD, and concise status.

The launcher selects a command environment before parsing the verb, so even
help or dry-run output can include environment setup or repair. Use `version`
and `env-info` to identify that environment, `env-check` for host/GPU/Docker/
display/device health, and `status` for image/cache facts. Compare outer and
container `env-info` when their behavior differs.

Preview the original command without changing its effect-bearing arguments.
Inspect selected project/mode/language, image, mounts, devices, environment,
workdir, child commands, and output. Reproduce once without edits and capture
the first causal error rather than only shutdown noise.

If the failure does not reproduce, compare revision, dirty state, input hashes,
image, cache, display/devices, and environment. Report the mismatch instead of
manufacturing a change.

## Layer matrix

| Layer | Check first |
| --- | --- |
| Launcher | Selected environment, active virtual environment, ownership, package/index access, checkout pin |
| Create/setup/lint | Complete unattended inputs, parent, template dependencies, privileges, auto-fixes |
| Discovery/metadata | Project/language/mode, schema, registration, `list`, `modes` |
| Host | `env-check`, platform, driver, Docker, disk, display, devices/permissions |
| Image/container | Dockerfile/base image, image age, registry, mounts, workdir, user, display |
| Configure/build | First CMake/compiler error, targets, include/link paths, bindings, cache |
| Test | Selected driver/CTest target, language, xvfb, registration, first failure |
| Module install/package | Selected Python, build dir/hook, metadata, generator, artifact/consumer |
| Data/model | Origin/license, exact path/hash, permissions, format, engine compatibility |
| Application | Ports, tensors, conditions, resources, scheduler, termination, CUDA streams |
| Visual/recording | Mode, HoloViz layers, render path, display/headless, decoded pixels |
| Benchmark | Healthy normal app, instrumentation, workload, logs, analyzer, restoration |

Test the boundary between the two most plausible layers before editing. For
example, inspect mounts to separate host-data absence from an application path
bug, or decode a recording to separate successful execution from blank output.

## Observe, hypothesize, and fix

- Prefer existing logs, finite verdicts, focused tests, and artifacts.
- Increase Holoscan log level only for the smallest reproduction and remove the
  temporary verbosity.
- Enter the project environment through a previewed `run-container`; do not run
  project tools or debuggers directly on the host by default.
- Obtain approval for host execution, debugger capabilities, core dumps, or
  device/permission changes.
- Decode representative frames or a short clip for visual failures.

State one falsifiable hypothesis tied to the selected layer. Choose the least
invasive observation that distinguishes it, change one variable, record the
result, and revert diagnostic-only edits. Read source only after ownership is
narrowed. Fix the cause without unrelated refactoring, blanket suppression,
broad dependency upgrades, or public-contract changes; add a focused
deterministic regression test when possible.

## Cleanup and proof

Treat cleanup as a separate hypothesis. If evidence proves stale build state,
preview only the required scope:

```bash
./holohub clear-cache --build --dryrun
```

Review resolved paths and obtain explicit approval before the action. Never
manually delete build, data, or install trees, and never prefix the wrapper with
`sudo`.

Re-run the identical original command and nearest focused test. Inspect
corrected frames, consumed packages, or benchmark restoration as appropriate.
Remove diagnostic-only changes, run `git diff --check`, compare final status
with the baseline, and follow the current checkout's lint workflow before a
requested commit.

In the tested revision, `test <module>` is not module-scoped; validate declared
operators, demo apps, and consumer behavior instead.
