# Holoscan agent skills

A skill gives a coding agent instructions and supporting resources for a
Holoscan workflow. This directory registers skill sources; instructions,
helper scripts, and evaluations stay in the owner's repository.

## Find and use a skill

Start with the [Holoscan SDK skills](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/skills),
registered in [holoscan-sdk.json](holoscan-sdk.json).

Read the selected skill's purpose, prerequisites, supported agents, and
installation instructions at the registered revision. Install or enable it
using the method documented by its owner and try the provided example request.
The Holoscan CLI does not install skills into agents. Report problems through
the feedback channel the skill owner documents.

## Create a new skill

Check the project's contribution policy before submitting a skill. The SDK's
[skill contribution policy](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/skills#contributions)
currently directs feedback to community channels and does not accept external
skill pull requests. You can maintain a skill in your own repository and
register it here. Where an owner accepts contributions, follow its documented
process. Keep the instructions and supporting resources together:

```text
skills/my-skill/
├── SKILL.md            # Required name, description, and instructions
├── README.md           # Requirements, supported agents, installation, examples
├── references/         # Optional supporting documentation
├── scripts/            # Optional helpers
└── evals/              # Example requests and expected behavior
```

Start `SKILL.md` with YAML frontmatter and a nonempty instruction body:

```markdown
---
name: my-skill
description: Build and verify an existing Holoscan application using its documented configuration.
---

# Build and verify an application

Read the application's README, prepare its documented environment, run its
supported build and test commands, and report the observed results.
```

Replace the example with your workflow's triggers, required inputs, steps,
expected results, and failure handling. Follow the
[Agent Skills specification](https://agentskills.io/specification) and your
target agent's requirements. Link to project documentation and the
[CLI reference](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/CLI_REFERENCE.md)
for supported commands instead of duplicating a build system. Include ownership
and licensing. The Holoscan CLI has no built-in skill scaffold.

## Build and verify the skill

Before publishing, check frontmatter, links, and referenced files. Load the skill
in a supported agent and evaluate requests that should trigger it and requests
that should not. Exercise both successful workflows and expected failure cases,
such as missing dependencies, and record results upstream. Test bundled scripts
with their own tools.

Skill instructions do not compile, and `holoscan test` tests source projects
through CTest rather than evaluating agent behavior. Registry checks validate
skill structure and identity without executing its instructions.

## Register an existing skill

Publish the skill and check this directory for an existing source registration.
A skill already covered by a registered collection needs no separate record;
update that record's revision if needed. Otherwise,
[request registration with kind `skill`](https://github.com/nvidia-holoscan/holohub/issues/new?template=registry.yml&kind=skill).

Point to the skill directory, a collection of skill directories, a hosted
`SKILL.md` URL, or the directory serving it. Skills require no `metadata.json`
or CLI discovery. The
frontmatter `name` must match the skill directory and, for a direct skill, the
registration name. Any declared identity fields must agree. Follow
[Contributing](../CONTRIBUTING.md) for submission and review, and the
[validation reference](../docs/registry-reference.md#validation) for exact rules.

[Back to HoloHub](../README.md)
