# Holoscan tutorials

Tutorials teach a Holoscan concept or workflow through a walkthrough, notebook,
or runnable example. This directory registers tutorial sources; the learning
material and its supporting files stay in the owning repository.

## Find and follow a tutorial

| Source | What you will find |
| --- | --- |
| [holoscan-tutorials](https://github.com/nvidia-holoscan/holoscan-tutorials) | Development walkthroughs, including [creating and using modules](https://github.com/nvidia-holoscan/holoscan-tutorials/tree/main/tutorials/holoscan-modules). [Registration](holoscan-tutorials.json). |
| [Archived HoloHub tutorials](https://github.com/nvidia-holoscan/holohub/tree/archive-4x/tutorials) | Earlier walkthroughs with their original SDK requirements. [Registration](holohub-tutorials-archive.json). |

Choose a walkthrough and read its learning goal, requirements, and expected
result. For the dedicated repository, start with Git:

```bash
git clone https://github.com/nvidia-holoscan/holoscan-tutorials.git
cd holoscan-tutorials
```

Follow the selected tutorial's README from the working directories it specifies.
Install its required SDK, notebook runtime, packages, and data. Tutorials have
different execution workflows; use CLI commands only where the tutorial supports
them. Report errors to the source repository's issue tracker.

## Create a new tutorial

1. Follow [holoscan-tutorials' contribution guide](https://github.com/nvidia-holoscan/holoscan-tutorials/blob/main/CONTRIBUTING.md)
   or author the tutorial in your own repository. Improve an existing tutorial
   through a pull request to its owner.
2. State the audience, learning goal, prerequisites, and outcome. Include
   commands with working directories, expected output, and troubleshooting.
   Keep notebooks, example code, images, and data preparation beside the text.
3. Include a maintainer contact, licensing, and `metadata.json` following the
   [tutorial schema](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/src/holoscan_cli/metadata/tutorial.schema.json).
   Record the SDK versions you tested. If needed, build the example using the
   [application](../applications/README.md#create-a-new-application) or
   [module](../modules/README.md#create-a-new-module) guide.

The CLI has no built-in tutorial scaffold. Use the tutorial repository's layout
and tools for Markdown, notebooks, or documentation builds.

## Build and verify the tutorial

Follow every step from a clean environment and execute notebook cells in order.
Compare results with the documented outcome and check links and downloads.
Run accompanying application tests using the source project's instructions.
A metadata file alone does not supply a runnable application or build system.

Keep validation in the source repository's CI where practical, then publish
the tested tutorial revision.

## Register an existing tutorial

Check the records here first. A tutorial already covered by a registered
collection needs no separate record; update the collection's revision when
needed. Otherwise, [request registration with kind `tutorial`](https://github.com/nvidia-holoscan/holohub/issues/new?template=registry.yml&kind=tutorial).

For Git, select the tutorial directory or collection containing `metadata.json`
descriptors. A direct tutorial's registration name matches its directory name.
For metadata at a Git repository root, use the repository name without `.git`;
see [discovery names](../docs/registry-reference.md#discovery-names).
For hosted content, provide an HTTPS `metadata.json` URL or the directory serving
it; an HTML tutorial page alone is insufficient. See
[Contributing](../CONTRIBUTING.md) for submission steps and the
[source reference](../docs/registry-reference.md) for hosted-content records.

[Back to HoloHub](../README.md)
