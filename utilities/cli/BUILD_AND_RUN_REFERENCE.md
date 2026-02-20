# Build and Run Configuration Reference

This document is a **reference** for customizing how the HoloHub CLI builds and runs applications. Use it when you need to change default paths, switch between local and container workflows, control Docker image selection and naming, or tune build and runtime behavior with environment variables.

- [Default Behavior](#default-behavior)
- [Command-line Options](#command-line-options)
- [Environment Variables](#environment-variables)
  - [Build and Execution Control](#build-and-execution-control)
  - [Paths and Directories](#paths-and-directories)
  - [Container Configuration](#container-configuration)
- [Docker Images](#docker-images)
  - [Docker Runtime](#docker-runtime)
  - [Testing](#testing)
  - [Other](#other)
- [Use Different Docker Options for Build and Run](#use-different-docker-options-for-build-and-run)

This reference does not cover migrating from the older bash-based CLI; for that, see the [HoloHub CLI Migration Guide](README.md).

## Default Behavior

By default, `./holohub run` operates in a containerized mode, which means it will:

1. Build the container image (unless skipped, for example using `--no-docker-build`)
2. Build the application inside the container
3. Run the application inside the container

This container-first approach ensures consistency and reproducibility across different development environments.

## Command-line Options

- `--local`: Explicit local development mode
- `--no-local-build`: Skip application rebuild for faster iteration
- `--no-docker-build`: Use existing container images
- Dedicated Commands: Separate `build`, `run`, `build-container`, and `run-container` commands for clear workflow control

## Environment Variables

The CLI supports the following environment variables for customization:

- [Default Behavior](#default-behavior)
- [Command-line Options](#command-line-options)
- [Environment Variables](#environment-variables)

### Build and Execution Control


| Variable                 | Description                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `HOLOHUB_BUILD_LOCAL`    | Forces local mode (equivalent to `--local`), skips the container build steps and runs on the host directly.                                                                                                                                                                                                                                                            |
| `HOLOHUB_ALWAYS_BUILD`   | Set to `false` to skip builds with `--no-local-build` and `--no-docker-build`.                                                                                                                                                                                                                                                                                         |
| `HOLOHUB_ENABLE_SCCACHE` | Defaults to `false`. Set to `true` to enable rapids-sccache for the build. Configure sccache with `SCCACHE_`* environment variables per the [sccache documentation](https://github.com/rapidsai/sccache/tree/rapids/docs). Use `--extra-scripts sccache` to install sccache in the container image (for example, `./holohub build-container --extra-scripts sccache`). |


### Paths and Directories


| Variable                    | Description                                                                                                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `HOLOHUB_ROOT`              | HoloHub repository root directory, used to resolve relative paths for components, build artifacts, data, and other resources.                                           |
| `HOLOHUB_BUILD_PARENT_DIR`  | Root directory for all builds (default: `<HOLOHUB_ROOT>/build`).                                                                                                        |
| `HOLOHUB_DATA_DIR`          | Root data (such as models, datasets downloading during component building) directory (default: `<HOLOHUB_ROOT>/data`).                                                  |
| `HOLOHUB_SETUP_SCRIPTS_DIR` | Directory containing setup scripts (default: `<HOLOHUB_ROOT>/utilities/setup`), typically used to install dependencies and setup the environment via `--extra-scripts`. |
| `HOLOHUB_PATH_PREFIX`       | Prefix for path placeholder variables (default: `holohub_`). Path placeholders are used in `metadata.json` files, for example, `"workdir": "holohub_app_source"`.       |
| `HOLOHUB_DEFAULT_HSDK_DIR`  | Holoscan SDK root directory (default: `/opt/nvidia/holoscan`). Used for both local builds and container operations to locate SDK libraries and Python bindings.         |
| `HOLOSCAN_SDK_ROOT`         | Path to local Holoscan SDK root directory (source or build tree), used for mounting local Holoscan SDK development trees into containers.                               |
| `HOLOHUB_SEARCH_PATH`       | Comma-separated directories to scan for metadata (default `applications,benchmarks,gxf_extensions,operators,pkg,tutorials,workflows`).                                  |


### Container Configuration


| Variable                   | Description                                                                                                                                                                  |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `HOLOHUB_REPO_PREFIX`      | Repository prefix for naming defaults (default: `holohub`). Used as the default value for container-related variables below.                                                 |
| `HOLOHUB_CONTAINER_PREFIX` | Docker container name prefix (default: `HOLOHUB_REPO_PREFIX`).                                                                                                               |
| `HOLOHUB_WORKSPACE_NAME`   | Workspace directory name in container (default: `HOLOHUB_REPO_PREFIX`). The `<HOLOHUB_ROOT>` directory is mounted in the container as `/workspace/<HOLOHUB_WORKSPACE_NAME>`. |
| `HOLOHUB_HOSTNAME_PREFIX`  | Container hostname prefix (default: `HOLOHUB_REPO_PREFIX`). Used when building VSCode development containers.                                                                |
| `HOLOHUB_DOCKER_EXE`       | Docker executable command (default: `docker`).                                                                                                                               |


## Docker Images

By default, the Dockerfile used to build a container image for your project is chosen in that order:

1. Specified in your project's `metadata.json` file.
2. `<app_source>/<language>/Dockerfile`
3. `<app_source>/Dockerfile`
4. `<app_source>/../Dockerfile` (traverse up to `HOLOHUB_ROOT`)
5. `HOLOHUB_DEFAULT_DOCKERFILE` environment variable
6. `<HOLOHUB_ROOT>/Dockerfile`

This can be overridden by the `--docker-file` option.

If the selected Dockerfile has `ARG BASE_IMAGE`, the value of `BASE_IMAGE` will be automatically populated by the `./holohub build-container`, defaulting to `{base_image}:v{sdk_version}-{cuda_tag}` where:

| Placeholder | Description |
| ----------- | ----------- |
| `{base_image}` | Defaults to `nvcr.io/nvidia/clara-holoscan/holoscan`; overridden by `HOLOHUB_BASE_IMAGE`. |
| `{sdk_version}` | Defaults to the latest available Holoscan SDK version (for example `3.11.0`); overridden by `HOLOHUB_BASE_SDK_VERSION`. |
| `{cuda_tag}` | Holoscan SDK container CUDA tag. For Holoscan 3.7+, default CUDA major version is based on host driver; overridden with `--cuda <major_version>`. |
| `{base_image}:v{sdk_version}-{cuda_tag}` | Format can be overridden globally by `HOLOHUB_BASE_IMAGE_FORMAT`. |

The whole base image (`<repo:tag>`) can be overridden by the `--base-img` option.

The name of the output image generated by `./holohub build-container [project]` varies based on the following factors:

- Legacy tags:
  - If the project is using the default Dockerfile, the name will default to `{container_prefix}:ngc-v{sdk_version}-{cuda_tag}` where:
- `{container_prefix}` defaults to `HOLOHUB_REPO_PREFIX` and can be overridden by the `HOLOHUB_CONTAINER_PREFIX` environment variable.
- `{sdk_version}` defaults to the latest available Holoscan SDK version (for example `3.11.0`) and can be overridden by the `HOLOHUB_BASE_SDK_VERSION` environment variable.
  - `{cuda_tag}` is the Holoscan SDK container CUDA tag. For Holoscan 3.7+, the default CUDA major version is based on your host driver version, and can be overridden with the `--cuda <major_version>` option.
  - `{container_prefix}:ngc-v{sdk_version}-{cuda_tag}` as a format can be overridden by the `HOLOHUB_DEFAULT_IMAGE_FORMAT` environment variable globally.
  - If the project is using a custom Dockerfile, the name will default to `{container_prefix}:{project_name}` (see above for `container_prefix` value).
- New tags (see above for `container_prefix` value):
  - If building for a specific project, two additional tags will be created:
    - `{container_prefix}-{project_name}:{short_git_sha}`
    - `{container_prefix}-{project_name}:{git_branch_slug}`
  - If building the default container with no project specified, the tags will be:
    - `{container_prefix}:{short_git_sha}`
    - `{container_prefix}:{git_branch_slug}`

These output tags can be overridden by the `--img` option.

### Docker Runtime

| Variable | Description |
| -------- | ----------- |
| `HOLOHUB_DEFAULT_DOCKER_BUILD_ARGS` | Additional default arguments passed to `docker build` commands. Typically used to set global build arguments for all applications in the codebase; applications and commands can override these arguments. |
| `HOLOHUB_DEFAULT_DOCKER_RUN_ARGS` | Additional default arguments passed to `docker run` commands. Typically used to set global runtime arguments for all applications in the codebase; applications and commands can override these arguments. |
| `HOLOHUB_BENCHMARKING_SUBDIR` | Benchmarking subdirectory (default: `benchmarks/holoscan_flow_benchmarking`). |

### Testing

- `HOLOHUB_CTEST_SCRIPT`: CTest script path used by `./holohub test` command (default: `<HOLOHUB_ROOT>/utilities/testing/holohub.container.ctest`).

### Other

| Variable | Description |
| -------- | ----------- |
| `HOLOHUB_CMD_NAME` | Command name displayed in help messages (default: `./holohub`). Allows customizing the command name for external codebases. |
| `HOLOHUB_CLI_DOCS_URL` | CLI documentation URL. Allows customizing the documentation URL for external codebases. |
| `CMAKE_BUILD_TYPE` | Default CMake build type (`debug`, `release`, or `relwithdebinfo`) when not specified in build commands. |
| `CMAKE_BUILD_PARALLEL_LEVEL` | Number of parallel CMake build jobs. |

## Use Different Docker Options for Build and Run

When build needs one set of Docker options (for example network access) and run needs another (for example GPU), use two modes that share the same image.

1. Add a build-focused mode: set `build.depends`, `build.docker_build_args`, and a minimal `run`. For example, use `command`: `"echo 'Build complete'"`, `run.docker_run_args` for the build container.

2. To add a run-focused mode, set only `run.command` and `run.docker_run_args` for the app container. Do not specify build overrides.

3. Build, then run:  
   Build with the build mode; run with the run mode and `--no-docker-build` so the run mode reuses that image.

```bash
./holohub build myapp production_build
./holohub run myapp production_run --no-docker-build
```

Both modes use the same image name (for example `holohub:myapp`). For a full JSON example, see [Example `metadata.json` File](#example-metadatajson-file) above.

