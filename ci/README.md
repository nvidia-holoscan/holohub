# Pre-merge continuous integration

The `pre-merge-pipeline` Jenkins job validates a synthetic merge of every GitLab merge request
against its current target. It runs repository linting and builds and tests the sample against a
pinned private Holoscan SDK revision on native x86_64 and SBSA Blossom workers, plus the latest
`main-5x` SDK branch on native x86_64.

The production job must load [`pre-merge-pipeline.groovy`](pre-merge-pipeline.groovy) from the
protected `main` branch. Do not configure Pipeline from SCM to load the Groovy file from the merge
request ref: merge-request code is untrusted, while the pipeline can access the private SDK.

## Pipeline structure

The CI implementation follows the Holoscan SDK pipeline layout:

- `pre-merge-pipeline.groovy` loads trusted helpers, defines flows, and orders high-level stages;
- `utils.groovy` owns reusable SCM, Blossom pod, driver, credential, and flow setup; and
- `stage.groovy` owns reusable lint, checkout, build, test, and artifact stages.

Add future pipelines as thin orchestration scripts that load the same helpers from protected
`main`. Put infrastructure or flow mechanics in `utils.groovy` and reusable user-visible Jenkins
stages in `stage.groovy`. Pipeline-specific flow selection and stage ordering remain in the
pipeline entry point.

## Jenkins project configuration

Use the Jenkins project at <https://prod.blsm.nvidia.com/clara-holoscan-module/>.

### Credentials

Create these credentials in the Jenkins project:

| Jenkins ID | Type | Minimum access |
| --- | --- | --- |
| `HOLOSCAN_5_0_EA_SAMPLES_GITLAB_READ_TOKEN` | Username with password/token | `read_repository` for this repository and supported internal forks |
| `HOLOSCAN_SDK_GITLAB_READ_TOKEN` | Username with password/token | `read_repository` for `holoscan/holoscan-sdk` |
| `holoscan-samples-gitlab-api` | GitLab API token | Commit status updates |

The pipeline passes the repository tokens only to Jenkins SCM operations. It does not inject them
into build shells.

Configure a Jenkins GitLab connection named `holoscan-module-connection` with
`holoscan-samples-gitlab-api`, then use the connection for the job.

### Blossom Kubernetes access

The pipeline uses:

- Kubernetes cloud `sc-ipp-blossom-prod`;
- namespace `clara`;
- private multi-architecture image
  `gitlab-master.nvidia.com:5005/holoscan/holoscan-sdk/ci-dind:26.02.20`;
- version-pinned, public, multi-architecture Java 21 Jenkins inbound agent
  `jenkins/inbound-agent:3345.v03dee9b_f88fc-1-jdk21`;
- NVIDIA drivers from the R580, R595, or R610 families; and
- native `amd64` and `arm64` GPU nodes.

Use the existing `clara-holoscan-sdk-read-registry` image-pull secret in the `clara` namespace. It
must contain credentials with `read_registry` access to the Holoscan SDK registry. The Jenkins
agent image does not require a private-registry pull secret.

The image-pull secret is a Kubernetes credential, not a Jenkins credential ID. Adding
`HOLOSCAN_SDK_GITLAB_READ_TOKEN` to Jenkins does not make it available while Kubernetes is pulling
the pod image.

The scheduler affinity limits GPU flows to the certified driver versions copied from the pinned
`main-5x` CI configuration. Each GPU flow also checks the driver reported by `nvidia-smi` and
rejects driver majors below 580.

The pod runs its inbound agent through a bounded retry loop with increasing backoff. This keeps the
pod alive through transient Blossom DNS or TCP-listener failures and applies identically to
`amd64` and `arm64`. After 12 failed attempts it preserves the agent's nonzero status so Jenkins
reports a real infrastructure failure instead of a successfully completed container.

#### Registry 401 recovery

An `ErrImagePull` message ending in `failed to fetch oauth token: ... 401 Unauthorized` means
`clara-holoscan-sdk-read-registry` is missing from the namespace or contains an expired or
insufficient token. Update it through the Blossom image-pull-secret UI at
<https://prod.blsm.nvidia.com/jenkins/#/projects/clara/imagepullsecret> with:

| Field | Value |
| --- | --- |
| Secret name | `clara-holoscan-sdk-read-registry` |
| Registry server | `gitlab-master.nvidia.com:5005` |
| Username | Username associated with the GitLab deploy/project token |
| Password | GitLab token with `read_registry` for `holoscan/holoscan-sdk` |

If the UI project or secret is unavailable, request access through the Blossom administrators. A
user who can update the namespace directly may recreate the same `kubernetes.io/dockerconfigjson`
secret instead.

### Pipeline job

Create a Pipeline job named `pre-merge-pipeline` with these settings:

| Setting | Value |
| --- | --- |
| Definition | Pipeline script from SCM |
| SCM | Git |
| Repository URL | HTTPS URL for this samples repository |
| Credentials | `HOLOSCAN_5_0_EA_SAMPLES_GITLAB_READ_TOKEN` |
| Refspec | `+refs/heads/main:refs/remotes/origin/main` |
| Branch specifier | `origin/main` |
| Script path | `ci/pre-merge-pipeline.groovy` |
| Lightweight checkout | Disabled |

The job intentionally gets only its pipeline definition from `main`. At runtime, the Groovy script
uses the GitLab webhook environment to fetch the source and target repositories and performs a
Jenkins `PreBuildMerge`.

Set a build retention policy of 30 days or 50 builds. Pipeline changes should first be exercised in
a temporary job without production credentials, then merged to `main`.

Enable **Build periodically** with the Jenkins cron schedule chosen for the nightly coverage run.
The pipeline identifies these builds from Jenkins' timer cause. It also identifies a manual
**Build Now** run from Jenkins' direct-user cause; both submit results to CDash.

### Manual Build Now coverage

Launching the job with Jenkins **Build Now** does not provide GitLab merge-request environment
values. The pipeline detects that context, checks out the latest `main` revision from the job's
trusted SCM configuration, and runs the same lint, x86_64, and SBSA flows.

Manual runs submit results using CDash's `Nightly` dashboard model, but do not call the GitLab
commit-status steps because there is no merge-request commit to update. If a webhook supplies a
merge-request IID but omits another required GitLab value, the pipeline fails before scheduling
Blossom workers and reports the missing fields.

### GitLab trigger

Enable the Jenkins GitLab trigger for:

- merge-request open;
- merge-request update, including source-branch pushes;
- merge-request reopen; and
- note/comment events whose first word is `rebuild`, case-insensitively.

Set the Jenkins GitLab trigger's **Comment (regex) for triggering a build** to:

```text
(?i)^rebuild(?:\s.*)?$
```

The expression deliberately accepts trailing comment options. Jenkins decides whether to create
the build from the `rebuild` prefix alone; options such as `[push]` are interpreted later by the
pipeline and do not affect triggering.
The pipeline self-checks these comment helper cases before applying the draft skip gate.
For local validation, run `groovy ci/test_utils.groovy`.

Add the Jenkins-generated webhook URL and secret token to the GitLab project. Enable SSL
verification. Restrict automatic execution to internal GitLab merge requests because the build
checks out the private SDK source.

After an end-to-end validation, protect `main` in GitLab and require the aggregate external commit
status `pre-merge`. The pipeline also publishes `lint`, `x86_64-cuda13`, `sbsa-cuda13`, and
the configured flow statuses for diagnosis. The moving `main-5x` flow is advisory for merge
requests: it is shown as a failed Jenkins stage on failure, but does not publish a GitLab status or
affect the aggregate `pre-merge` result.

Draft merge requests are skipped unless a comment beginning with `rebuild`, in any capitalization,
triggers them explicitly.

Add `[push]` to a rebuild comment, for example `Rebuild [push]`, to submit that run's configure,
build, and test results to the
[Holoscan Modules CDash project](http://cdash.nvidia.com/index.php?project=Holoscan-Modules).
Jenkins can expose the brackets with backslashes; the pipeline accepts both forms. The `[push]`
marker is case-sensitive.

## Pipeline coverage

The lint flow runs:

```bash
./holohub list
./holohub modes v4l2_depth
./holohub lint
python3 -m unittest ci.test_report_cdash_test_failure
git diff --exit-code
```

Each GPU architecture flow:

1. verifies its native architecture and R580-or-newer driver;
2. checks out either the SDK SHA from [`holoscan-sdk.version`](holoscan-sdk.version) or the latest
   `main-5x` branch tip;
3. builds the CUDA 13 SDK without Python or benchmark targets;
4. resolves and validates the generated SDK installation;
5. builds `v4l2_depth` through Holoscan CLI; and
6. tests `v4l2_depth` through the project CTest driver.

The pinned x86_64 and SBSA flows, and the moving `x86_64-main-5x-cuda13` flow, run in parallel and
do not use fail-fast, so Jenkins reports every result. JUnit output, CTest failure logs, the exact
SDK revision and source, runner details, and available sccache statistics are retained as build
results. The `main-5x` flow resolves the branch again for every MR, nightly, and manual build; it
is drift coverage and does not replace the pinned validation baseline. For merge requests, failures
in this flow are retained as Jenkins stage failures without being propagated to `pre-merge` or
reported as GitLab statuses.

CTest submits each nightly cron run and manual Jenkins **Build Now** run using the `Nightly`
dashboard model. An MR comment containing `[push]` uses the `Experimental` model. Configure,
build, and test results are submitted before any corresponding failure is returned to Jenkins, so
failed outcomes remain visible in CDash. Other merge-request, source-push, and reopen runs omit
the CDash URL entirely; their build and test output stays in the Jenkins log and archived Jenkins
results.

If the `v4l2_depth` application container fails to build during a CDash-enabled run, the test
container does not exist and normal CTest cannot start. The pipeline submits a synthetic failed test
named `v4l2_depth.container_build` directly from the Jenkins worker, including the tail of the
container-build log. It then preserves the original build failure. This fallback applies only to the
application-container build; earlier SDK, checkout, and runner failures do not create synthetic test
results. Jenkins-only runs archive the same build log without contacting CDash.

## Updating the SDK pin

The version file must contain one lowercase, full-length Git commit SHA. To advance the baseline:

1. update `ci/holoscan-sdk.version` to a reviewed commit on `main-5x`;
2. update the driver allowlist and DinD image in `utils.groovy` if the same SDK revision changes
   either value;
3. run the pre-merge job; and
4. require both x86_64 and SBSA results to pass before merging.

At the pinned SDK revision, `ci/scripts/utils.groovy` in the full SDK checkout is authoritative for
driver and DinD-image conventions. Do not copy values from an unpinned or stale worktree.
