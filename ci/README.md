# Pre-Merge Continuous Integration

The `pre-merge-pipeline` Jenkins job validates a synthetic merge of every
GitLab merge request against its current target. It runs linting, builds a
pinned Holoscan SDK 5.0 EA revision, then builds, tests, and packages
`holoscan-camera` on native x86_64 and SBSA Blossom workers. It also covers
the latest `main-5x` SDK branch on both architectures.

The production job must load [`pre-merge-pipeline.groovy`](pre-merge-pipeline.groovy)
from protected `main`. Do not load pipeline code from the merge-request ref:
merge-request code is untrusted, while the pipeline can access the private SDK.

## Pipeline Structure

- `pre-merge-pipeline.groovy` loads trusted helpers, defines flows, and orders stages.
- `utils.groovy` owns SCM, Blossom pod, driver, credential, and flow setup.
- `stage.groovy` owns checkout, lint, SDK build, Camera build/test/package, and artifact stages.

## Jenkins Project Configuration

Use the Jenkins project at <https://prod.blsm.nvidia.com/clara-holoscan-module/>.

### Credentials

Create these credentials in the Jenkins project:

| Jenkins ID | Type | Minimum access |
| --- | --- | --- |
| `HOLOSCAN_CAMERA_GITLAB_READ_TOKEN` | Username with password/token | `read_repository` for this repository and supported internal forks |
| `HOLOSCAN_SDK_GITLAB_READ_TOKEN` | Username with password/token | `read_repository` and `read_registry` for `holoscan/holoscan-sdk` |
| `holoscan-camera-gitlab-api` | GitLab API token | Commit status updates and merge-request comments |

Configure a Jenkins GitLab connection named `holoscan-module-connection` with
the API token, then use that connection for the job.

### Blossom Kubernetes Access

The pipeline uses:

- Kubernetes cloud `sc-ipp-blossom-prod`;
- namespace `clara`;
- private multi-architecture image
  `gitlab-master.nvidia.com:5005/holoscan/holoscan-sdk/ci-dind:26.02.20`;
- Java 21 inbound agent `jenkins/inbound-agent:3345.v03dee9b_f88fc-1-jdk21`;
- NVIDIA R580, R595, or R610 driver families; and
- native `amd64` and `arm64` GPU nodes.

Use the existing `clara-holoscan-sdk-read-registry` image-pull secret in the
`clara` namespace. It must contain credentials with `read_registry` access to
the Holoscan SDK registry. This Kubernetes secret authenticates pod image pulls;
the Docker daemon inside the DinD container authenticates separately with
`HOLOSCAN_SDK_GITLAB_READ_TOKEN` before importing the SDK build-image cache.

### Pipeline Job

Create a Pipeline job named `pre-merge-pipeline` with these settings:

| Setting | Value |
| --- | --- |
| Definition | Pipeline script from SCM |
| SCM | Git |
| Repository URL | HTTPS URL for this repository |
| Credentials | `HOLOSCAN_CAMERA_GITLAB_READ_TOKEN` |
| Refspec | `+refs/heads/main:refs/remotes/origin/main` |
| Branch specifier | `origin/main` |
| Script path | `ci/pre-merge-pipeline.groovy` |
| Lightweight checkout | Disabled |

The job intentionally gets only its pipeline definition from `main`. At
runtime, the Groovy script uses GitLab webhook environment values to fetch the
source and target repositories and performs a Jenkins `PreBuildMerge`.

Enable **Build periodically** for nightly coverage. Nightly and `[push]`
comment-triggered builds submit to CDash; regular merge-request and manual
Build Now runs keep results in Jenkins only.

### GitLab Trigger

Enable the Jenkins GitLab trigger for:

- merge-request open;
- merge-request update, including source-branch pushes;
- merge-request reopen; and
- note/comment events whose first word is `rebuild`, case-insensitively.

Set the trigger comment regex to:

```text
(?i)^rebuild(?:\s.*)?$
```

Add `[push]` to a rebuild comment, for example `Rebuild [push]`, to submit
that run's configure, build, and test results to the
[Holoscan Modules CDash project](http://cdash.nvidia.com/index.php?project=Holoscan-Modules).
The `[push]` marker is case-sensitive.

For local validation of comment parsing, run:

```bash
groovy ci/test_utils.groovy
```

## Pipeline Coverage

The lint flow runs:

```bash
./holoscan_camera list
./holoscan_camera modes holoscan_camera_v4l2 --language cpp
./holoscan_camera lint
python3 -m unittest ci.test_report_cdash_test_failure
git diff --exit-code
```

Each GPU architecture flow:

1. verifies native architecture and an R580-or-newer NVIDIA driver;
2. checks out either the SDK SHA from [`holoscan-sdk.version`](holoscan-sdk.version) or the latest
   `main-5x` branch tip;
3. authenticates to the Holoscan SDK registry and attempts to reuse the build-container cache for
   the checked-out SDK revision, falling back to a local container build if authentication or the
   cache-enabled build fails;
4. builds the CUDA 13 SDK with Python, benchmarks, examples, and tests explicitly disabled;
5. resolves and validates `HOLOSCAN_SDK_INSTALL_DIR`;
6. builds `holoscan_camera_v4l2` through Holoscan CLI;
7. tests `holoscan_camera_v4l2` through `cmake/container.ctest`; and
8. packages `holoscan-camera` as a Debian package.

The pinned x86_64 and SBSA flows and the moving `x86_64-main-5x-cuda13` and
`sbsa-main-5x-cuda13` flows run in parallel. The `main-5x` flows resolve the branch again for every
MR, nightly, and manual build; they are drift coverage and do not replace the pinned baseline. For
merge requests, their failures remain visible as failed Jenkins stages but do not publish GitLab
statuses or affect the aggregate `pre-merge` result.

JUnit output, CTest logs, package artifacts, the exact SDK revision and source, runner details, and
available sccache statistics are archived as Jenkins artifacts.

If the application container fails to build during a CDash-enabled run, normal
CTest cannot start. The pipeline submits a synthetic failed test named
`holoscan_camera_v4l2.container_build` with the tail of the build log, then
preserves the original build failure.

CDash build names identify the architecture, SDK coverage role, and source branch:
`holoscan-camera_<arch>_sdk-<pinned|latest>_<branch>`. `latest` is the moving
`main-5x` SDK flow; `pinned` is the reviewed SDK revision. The default C++ test
submission covers the module's enabled tests, including SIPL unit tests on SBSA;
SIPL hardware tests remain individually reported as not run when no camera is configured.

## Updating The SDK Pin

`ci/holoscan-sdk.version` must contain one lowercase, full-length Git commit
SHA. To advance the baseline:

1. update `ci/holoscan-sdk.version` to a reviewed commit on `main-5x`, preferring
   the commit behind a release tag over a bare branch tip where one is available;
2. update the driver allowlist and DinD image in `utils.groovy` if that SDK
   revision changes either value;
3. run the pre-merge job; and
4. require both x86_64 and SBSA results to pass before merging.

The pin currently names `16cfacb2`, a reviewed `main-5x` tip. A release cut is
preferred over a tip that happens to be newer: the two are the same input to this
repo whenever they do not differ in `public/include` or `public/src`, and the tag
additionally says which SDK release this repo was qualified against. Verify that
equivalence rather than assume it — `git diff --name-only <tag>..<tip> --
public/include public/src` should be empty. Between `v5.0.0.2` and this pin it is
not, which is why the tip is named here and not the tag: the newer revision is a
real change, so it belongs to its own pin bump with its own pre-merge run rather
than being folded into an unrelated one.

> **The pin has a floor, not just a ceiling.** `V4l2CaptureOp` requires `holoscan::sensor_io`,
> and it publishes `ImageEncoding_YUYV` against the frame extent the schema derives for that
> encoding. Both arrived with the sensor I/O foundation and are present from `v5.0.0.2` onward.
> A pin below that revision does not fail at runtime or merely describe the output poorly — the
> operator does not compile, so the GPU flows above cannot get past step 5. Treat moving the pin
> backwards as the same class of change as moving it forwards, and run the pre-merge job either way.
