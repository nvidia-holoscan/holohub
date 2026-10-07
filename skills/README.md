# Holoscan skills

## Register an existing skill

A skill supplies instructions and supporting resources for an agent performing
a Holoscan workflow. Register its external source or a collection of skills;
keep `SKILL.md`, references, scripts, evaluations, and publication metadata in
the owner's repository.

1. Check this directory's JSON files for the same repository and source path.
   Update an existing registration for a new revision instead of adding one
   per release. A collection may use a single source record.
2. Add `skills/my-skill.json` in your HoloHub checkout:

   ```json
   {
     "name": "my-skill",
     "kind": "skill",
     "issues_url": "https://github.com/example/holoscan-skills/issues",
     "source": {
       "type": "git",
       "url": "https://github.com/example/holoscan-skills.git",
       "ref": "v0.1.0",
       "path": "skills/my-skill"
     }
   }
   ```

   Replace the example repository and revision with your published source.
   Prefer a full commit SHA for `ref`. Set `path` to the skill directory, a
   collection such as `skills`, or `"."` for a repository root. The lowercase
   `name` must match the JSON filename. `issues_url` is optional.
3. Confirm the target contains the skill instructions and all referenced
   resources. Each `SKILL.md` needs YAML frontmatter with a name matching its
   directory, a nonempty description, and a Markdown instruction body. For a
   direct skill, its name must also match the registration name; collection
   names may differ. Any declared `kind` must be `skill`, and nested metadata
   identity fields must agree with the frontmatter. Keep skill metadata upstream.
4. Validate from the HoloHub root:

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirement.txt
   .venv/bin/python utilities/validate_registry.py --index dist/sources.json
   .venv/bin/python utilities/validate_sources.py
   ```

5. Submit the JSON change in a pull request to the HoloHub registry branch you
   are contributing to. Follow [CONTRIBUTING.md](../CONTRIBUTING.md) for the
   full checks. Do not include generated indexes or the skill implementation.

The Holoscan CLI does not submit registrations or install skills into agents.
Registration makes the source discoverable; users still follow the source's
agent-specific installation instructions. CI fetches and validates the skill
definitions without executing or evaluating their instructions. Skills require
neither `metadata.json` nor `holoscan list`. See the
[source record reference](../README.md#source-records) for all fields and
hosted-content sources. [`holoscan-sdk.json`](holoscan-sdk.json) is a current
registration for a skill collection.

## Create a new skill

Build the skill package in an external repository. The Holoscan CLI has no
built-in skill scaffold or skill build command. A typical package is:

```text
skills/my-skill/
├── SKILL.md            # Trigger description and workflow instructions
├── README.md           # Purpose, requirements, installation, and usage
├── references/         # Focused documentation loaded when needed
├── scripts/            # Optional helpers
└── evals/              # Example requests and expected behavior
```

Start `SKILL.md` with the frontmatter required by the intended agent or skill
format. For agents using `name` and `description`, a starting point is:

```markdown
---
name: my-skill
description: Build and verify an existing Holoscan application using its documented configuration.
---

# Build and verify a Holoscan application

1. Read the application's README and project guidance to identify requirements.
2. Use `holoscan list` and `holoscan modes <application>` to select the project.
3. Build and run with the documented mode, environment, and inputs.
4. Run the project's tests and compare output with its expected results.
5. Report the commands, observed results, and any unmet requirements.
```

Replace this starting point with the workflow your skill actually supports.
Define when to use it, required inputs, prerequisites, decision points, expected
results, and what to do when a step fails. Use relative links to bundled
resources and keep large reference material outside the main instructions.
Document which agents can load the skill and how users install or enable it.

Prefer `holoscan` commands wherever the CLI supports the operation. For example,
application discovery, builds, runs, tests, and module packaging should use
`holoscan list`, `holoscan build`, `holoscan run`, `holoscan test`, and
`holoscan package` with the appropriate project and flags. Require the agent
to consult `holoscan <command> --help` for its installed version. Link to the
[CLI reference](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/CLI_REFERENCE.md)
and the source project's instructions instead of duplicating a build system
inside the skill.

## Build and verify the skill

The instruction files do not compile. Build any bundled helper tools using
their documented toolchain; Holoscan application or module examples can use
the [application](../applications/README.md#build-run-and-test) and
[module](../modules/README.md#build-run-test-and-package) workflows.

Before publishing:

- Check frontmatter, local links, required files, and referenced commands using
  the source repository's validator and the target agent's skill requirements.
- Load the package in a supported agent and try representative requests that
  should trigger it, plus unrelated requests that should not.
- Exercise its workflow on a small Holoscan example. Check successful behavior
  and failure cases such as a missing SDK, unknown project, or unavailable data.
  Record expected outcomes and evaluation results upstream.
- Test bundled scripts independently and document their dependencies. Include
  licensing and any publication artifacts required by your chosen host.

There is no `holoscan test` target for agent instructions; that command tests
source projects through CTest. Once the skill and its resources are published,
register the source above or update the existing record's revision.
