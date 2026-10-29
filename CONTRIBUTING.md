# Contributing to HoloHub

Develop and publish your project in its owning repository, then register its
source in HoloHub so others can find it. HoloHub stores source locations;
project owners maintain the code, metadata, documentation, tests, and releases.

## Choose where to contribute

| Contribution | Where to develop it | Guide |
| --- | --- | --- |
| Application | [holoscan-applications](https://github.com/nvidia-holoscan/holoscan-applications/blob/main/CONTRIBUTING.md) or your own repository | [Applications](applications/README.md) |
| Reusable operators or components | The existing module's repository or a dedicated module repository | [Modules](modules/README.md) |
| Tutorial | [holoscan-tutorials](https://github.com/nvidia-holoscan/holoscan-tutorials/blob/main/CONTRIBUTING.md) or your own repository | [Tutorials](tutorials/README.md) |
| Benchmark or measurement tool | [holoscan-benchmarks](https://github.com/nvidia-holoscan/holoscan-benchmarks/blob/main/CONTRIBUTING.md) or your own repository | [Benchmarks](benchmarks/README.md) |
| Agent skill | The project it supports or a dedicated skill repository | [Skills](skills/README.md) |
| Registration, HoloHub documentation, or registry tooling | This repository | [Submit a pull request](#register-a-source) |

For an existing project, follow that repository's contribution guide. Keep
application-specific operators with the application; use a module when other
projects should reuse them as a dependency. SDK and CLI changes belong in
[holoscan-sdk](https://github.com/nvidia-holoscan/holoscan-sdk) and
[holoscan-cli](https://github.com/nvidia-holoscan/holoscan-cli), respectively.

## Prepare your source

Before requesting registration:

- Publish a publicly accessible source with a README, license, maintainer
  contact, and issue tracker or support route.
- Document the supported SDK and platforms, hardware, dependencies, data,
  setup commands, working directories, and expected results. Verify the
  workflow and keep tests in the source repository.
- Include `metadata.json` for applications, modules, tutorials, and benchmarks,
  following the [CLI metadata schemas](https://github.com/nvidia-holoscan/holoscan-cli/tree/main/src/holoscan_cli/metadata).
  Skills use `SKILL.md` instead; see the [skill guide](skills/README.md).
- Check the relevant category directory for an existing source. One record can
  cover a collection of projects or a module's operators. Add a project covered
  by an existing collection upstream; update the collection's `ref` here only
  if its registered revision needs to change.

## Request a registration change

The easiest path is the [registration request form](https://github.com/nvidia-holoscan/holohub/issues/new?template=registry.yml).
You need a GitHub account, but no local tools or JSON editing.

| Form field | What to enter |
| --- | --- |
| **Kind** | `application`, `module`, `tutorial`, `benchmark`, or `skill`. |
| **Name** | A stable lowercase name using letters, numbers, and single hyphens or underscores between words. For a direct project, use its [discovery name](docs/registry-reference.md#discovery-names); a collection has its own name. |
| **Source URL** | An HTTPS Git repository URL, or a direct hosted `metadata.json` / `SKILL.md` URL. Required for new registrations. |
| **Git revision** | Prefer a full commit SHA; a branch or tag also works. A blank field uses the default branch for GitHub repository URLs or URLs ending in `.git`. For other Git URLs, including GitLab URLs without `.git`, enter a revision. Leave blank for hosted content. |
| **Repository path** | The project or collection directory within the Git repository; `.` means the root. Leave blank for hosted content. |
| **Issue tracker URL** | An HTTPS issue tracker or support form. If omitted for a new GitHub or GitLab source, automation tries its public issue tracker. |
| **Reason** | What the source provides or why the registration should change. |

To **update**, enter the existing Kind and Name and only the fields to change.
Blank fields keep their current values; `none` clears an optional value.
To **remove**, enter Kind and Name and tick Remove. Removal only deletes the
HoloHub record. To rename, remove the old registration first, then register the
new name after the removal merges.

After submission, automation comments on the issue and opens a pull request
when the request is valid. Correct errors by editing the issue; automation
updates the pull request, so make changes through the issue. Closing the issue
withdraws the request; reopening retries it. A maintainer reviews and merges
the pull request, which closes the issue. **A merged record is a completed
registration.** Website publication is managed separately.

## Register a source

To submit JSON yourself instead of using the form:

1. Fork HoloHub, clone your fork, and create a branch from `main`.
2. Add or edit `<kind-directory>/<name>.json`. Start with a
   [fixture](tests/fixtures/) or a record in the relevant directory, replace
   its example values, and follow the [source record reference](docs/registry-reference.md).
3. Run the [validation checks](#validation) from your HoloHub checkout.
4. Sign off your commits with `git commit -s` and open a pull request against
   HoloHub's `main` branch. Include the source's
   purpose, maintainer, and validation results. Commit the registration only;
   project files stay upstream and generated `dist/` output stays local.

The same fork-and-pull-request workflow applies to HoloHub documentation and
tooling changes. For tooling changes, also run
`.venv/bin/python -m unittest discover -s tests -v`.

## Registry layout

| Kind | Record location |
| --- | --- |
| `application` | `applications/<name>.json` |
| `module` | `modules/<name>.json` |
| `tutorial` | `tutorials/<name>.json` |
| `benchmark` | `benchmarks/<name>.json` |
| `skill` | `skills/<name>.json` |

## Source records

Records require `name`, `kind`, and `source`, with an optional `issues_url`.
Git sources require a repository URL and `ref`, with an optional `path`.
Names match filenames and remain stable across releases. See the
[reference](docs/registry-reference.md) for a complete example, hosted sources,
identity rules, and validation details.

## Validation

Use Python 3.12–3.14 and Git. From the HoloHub repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirement.txt
.venv/bin/python utilities/validate_registry.py
.venv/bin/python utilities/validate_sources.py
```

The `pre-commit.ci` check runs formatting hooks on pull requests. To run the
same checks locally:

```bash
.venv/bin/python -m pip install pre-commit
.venv/bin/pre-commit run --all-files
```

The first validator checks record structure and duplicates offline. The second
requires network access: it fetches every registered source, validates its
descriptors, and checks CLI discovery (or skill frontmatter). Success means
both commands exit with no validation errors. Failures identify the record to
correct; see the [validation reference](docs/registry-reference.md#validation).

If a check fails on an unchanged registration, include the output in your pull
request or a [HoloHub issue](https://github.com/nvidia-holoscan/holohub/issues) so
maintainers can investigate. Identify which checks passed and which failed.

These checks require neither the Holoscan SDK nor a GPU, and do not build or
run projects. Project owners verify behavior and compatibility in their own
tests. `holoscan list` discovers projects in a local checkout; it does not
install them or submit a HoloHub registration.
