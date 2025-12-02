# HoloHub CLI Migration Guide

## Overview

As of June 2025, HoloHub has been refactored from bash-based CLI scripts (`./run` and `./dev_container`) to a modern, unified Python-based CLI tool (`./holohub`). This upgrade provides a cleaner, more powerful, and more intuitive development experience.

## Key Benefits of the Refactoring

### **Simplified Workflows**

Transform complex multi-command workflows into single, intuitive commands:

```bash
# OLD: Container development
./dev_container build_and_run myapp --language cpp

# NEW: Container development
./holohub run myapp --language cpp
```

```bash
# OLD: Local development
./run build myapp --type debug
./run launch myapp cpp

# NEW: Local development (single command)
./holohub run myapp --language=cpp --local --build-type=debug
```

### **Enhanced Developer Experience**

- **Unified Interface**: One tool for both local and containerized development instead of juggling `./run` and `./dev_container`
- **Intelligent Defaults**: Container-first development for consistency and reproducibility
- **Better Error Messages**: Helpful suggestions when commands or project names don't match
- **Modern Python Implementation**: More reliable, maintainable, and extensible

### **Quick Command Reference**

| Workflow | Old Commands | New Command |
|----------|-------------|-------------|
| **Container Build & Run** | `./dev_container build_and_run` | `./holohub run` |
| **Local Build & Run** | `./run build` + `./run launch` | `./holohub run --local` |
| **Container Build Only** | `./dev_container build` | `./holohub build-container` |
| **Local Build Only** | `./run build` | `./holohub build --local` |
| **Run Without Building** | `./run launch` | `./holohub run --local --no-local-build` |
| **Container Run Only** | `./dev_container launch` | `./holohub run-container --no-docker-build` |

## Migration Examples

### **Build & Run**

```bash
# Build locally with optional operators and build type
# previous ./run build myapp --type debug --with "op1;op2"
./holohub build myapp --local --build-type debug --build-with "op1;op2"

# Build in container
# Previous:
# $ ./dev_container build --docker_file path/to/myapp/dockerfile --img holohub:myapp
# $ ./dev_container launch --img holohub:myapp
# >>> ./run build myapp --type debug
# Updated:
./holohub build myapp --build-type debug

# Run locally
# previous ./run build myapp && ./run launch myapp cpp
./holohub run myapp --local --language cpp --build-type debug

# Run in container
# previous ./dev_container build_and_run myapp cpp --run_args "--config=config.json"
./holohub run myapp --language=cpp --run-args="--config=config.json"

# Install built files
# previous ./dev_container build_and_install myapp
./holohub install myapp
```

### **Container Operations**

```bash
# Container build
# previous ./dev_container build --docker_file <path/to/myapp/Dockerfile> --img holohub:myapp
./holohub build-container <my_project> [options]

# Container run
# previous ./dev_container launch --img holohub:myapp
./holohub run-container <my_project> --no-docker-build [options]

# Development environment
# previous ./dev_container vscode myapp
./holohub vscode myapp --language cpp
```

### **Advanced Features**

Use `-h` or `--help` to view a complete list of commands or subcommand options.

```bash
# Custom build options
./holohub build myapp --build-with "op1;op2" --configure-args='-DCUSTOM=ON'

# Container run with path to a local Holoscan SDK installation from the host system
./holohub run-container --local-sdk-root <path/to/holoscan-sdk> --img holohub:sdk-dev-latest [additional-options]
# Example of additional options: --no-docker-build --env VAR=value

# Benchmarking an application (only for applications/workflows)
./holohub build myapp --benchmark
```

### **Development Environment**

Development commands remain familiar:

```bash
./holohub setup  # sudo privileges may be required

# Enhanced linting with integrated dependency management
./holohub lint --install-dependencies  # Previously ./run install_lint_deps
./holohub lint                         # Previously ./run lint
./holohub lint --fix                   # Previously ./run lint --fix

# Consistent naming conventions
./holohub clear-cache                  # Previously ./run clear_cache
./holohub list                         # Previously ./run list
```

## New Capabilities & Enhancements

### **Project Creation and Testing**

The new CLI introduces development lifecycle features:

```bash
# Scaffold new projects with templates
./holohub create my_new_app --language cpp

# Integrated testing framework
./holohub test myapp

# With code coverage (requires coverage tools in image)
./holohub test myapp --coverage

# Specify language when a project has multiple implementations
./holohub test myapp --language python

# Show environment information
./holohub env-info
```

### Application Modes

HoloHub applications with a 'modes' field in their `metadata.json` now support multiple **modes** - pre-configured setups for different use cases, hardware configurations, or deployment scenarios. This eliminates the need to remember complex command-line arguments for different use cases.

#### **Understanding Modes**

Modes potentially provide typical configurations for:

- **Different hardware**
- **Deployment scenarios** (e.g. development vs. production vs. cloud inference)
- **Data sources** (e.g. live video vs. recorded files vs. distributed streaming)

#### **Using Modes**

Examples:

```bash
# Discover available modes for an application
./holohub modes body_pose_estimation

# Run an application in a specific mode
./holohub run body_pose_estimation replayer               # Replayer mode
./holohub run holochat standalone                         # Local LLM inference mode

# Default behavior (no mode specified uses default_mode)
./holohub run body_pose_estimation                        # Uses 'default_mode' if specified
```

#### **Mode vs. CLI Arguments**

**CLI Parameters Always Override Mode Settings:**
CLI parameters can be used in addition to modes. When provided, CLI parameters will override the corresponding mode settings. The CLI shows warnings to help you understand what's being overridden:

```bash
# CLI parameters override mode settings (works with both implicit and explicit modes)
./holohub run holochat --run-args="--debug"            # appends to default_mode.run.command
./holohub run myapp --build-with="ops"                 # overrides default_mode.build.depends
./holohub build myapp standard --build-with="ops"      # overrides standard.build.depends
./holohub run holochat cloud --run-args="--test=case"  # appends to cloud.run.command
```

**Mode Priority Rules:**
When CLI parameters are **not** provided, mode settings are used:

- If mode defines `run.docker_run_args` and no `--docker-opts` → Mode's docker options are used
- If mode defines `build.depends` and no `--build-with` → Mode's dependencies are used
- If mode defines `build.docker_build_args` and no `--build-args` → Mode's build args are used
- If mode defines `build.cmake_options` and no `--configure-args` → Mode's cmake options are used

When CLI parameters **are** provided, they always override mode settings.

#### **For Application Developers**

Applications can define modes in their `metadata.json`. Here's the complete field reference:

##### **Mode Structure**

Each mode is defined as a named object under the `modes` key:

```json
{
  "metadata": {
    "default_mode": "mode_name",
    "modes": {
      "mode_name": {
        // Mode configuration fields
      }
    }
  }
}
```

##### **Application-Level Fields**

**Optional Fields:**

- **`default_mode`** *(string)*: Specifies which mode to use when no mode is explicitly provided. Only needed when there are multiple modes (2 or more).

##### **Supported Fields for Each Mode**

**Required Fields:**

- **`description`** *(string)*: Human-readable description of what this mode does
- **`run`** *(object)*: Run configuration (see run command fields below)

**Optional Fields:**

- **`requirements`** *(array of strings)*: List of dependency IDs required for this mode
- **`build`** *(object)*: Build configuration (see build configuration fields below)
- **`env`** *(object)*: Environment variables to set for both build and run operations
  - Key-value pairs of environment variable names and values
  - These environment variables are available during both the build phase and runtime
  - Example: `{"CUDA_VISIBLE_DEVICES": "0", "BUILD_ENV": "production"}`

##### **Build Configuration Fields (`build` object):**

- **`depends`** *(array of strings)*: List of operators/dependencies to build with this mode
- **`docker_build_args`** *(string or array)*: Docker **build** arguments (equivalent to CLI `--build-args`)
  - Can be a single string: `"--build-arg CUSTOM=value"`
  - Or an array: `["--build-arg", "CUSTOM=value"]`
- **`cmake_options`** *(array of strings)*: Additional CMake configure arguments to pass to the local build step
- **`env`** *(object)*: Environment variables to set **only for build operations** (local build and install)
  - Key-value pairs of environment variable names and values
  - Only applied during build, not during application execution
  - **Note**: For environment variables needed during both build and run, use the top-level `env` field instead

##### **Run Command Fields (`run` object):**

- **`command`** *(string)*: Complete command to execute including all arguments
- **`workdir`** *(string)*: Working directory for command execution
- **`docker_run_args`** *(string or array)*: Docker **run** arguments used for both build and application containers (equivalent to CLI `--docker-opts`)
  - Can be a single string: `"--privileged --net=host"`
  - Or an array: `["--privileged", "--net=host"]`
  - **Note**: These arguments apply to both the container that builds your application and the container that runs it
  - **Common use cases**: GPU access (`--gpus=all`), device access (`--device=/dev/video0`), privileged operations (`--privileged`)
- **`env`** *(object)*: Environment variables to set **only for runtime** (local runs only)
  - Key-value pairs of environment variable names and values
  - Only applied during application execution, not during build
  - Example: `{"LOG_LEVEL": "debug", "RUNTIME_CONFIG": "optimized"}`
  - **Note**: For environment variables needed during both build and run, use the top-level `env` field instead

##### **Complete Example**

```json
{
  "metadata": {
    "default_mode": "standard",
    "modes": {

      "standard": {
        "description": "Standard camera input for development and testing",
        "requirements": ["camera", "model"],
        "run": {
          "command": "python3 <holohub_app_source>/app.py --source camera",
          "workdir": "holohub_bin"
        }
      },

      "production": {
        "description": "High-performance mode with GPU acceleration",
        "requirements": ["gpu", "model", "tensorrt"],
        "env": {
          "TEST_VAR": "0",
          "BUILD_ENV": "production"
        },
        "build": {
          "depends": ["tensorrt_backend", "gpu_ops"],
          "docker_build_args": ["--build-arg", "TENSORRT_VERSION=8.6", "--network=host"],
          "cmake_options": ["-DUSE_TENSORRT=ON", "-DCUDA_ARCH=sm_86"]
        },
        "run": {
          "command": "python3 <holohub_app_source>/app.py --backend tensorrt --optimization-level 3",
          "workdir": "holohub_bin",
          "docker_run_args": ["--gpus=all", "--shm-size=1g"],
          "env": {
            "LOG_LEVEL": "info",
            "RUNTIME_OPTIMIZATION": "max"
          }
        }
      },

      "aja_hardware": {
        "description": "AJA capture card with hardware overlay",
        "requirements": ["aja_card_with_overlay", "model"],
        "build": {
          "depends": ["aja_source"]
        },
        "run": {
          "command": "python3 <holohub_app_source>/app.py --source=aja --config=overlay.yaml",
          "workdir": "holohub_bin",
          "docker_run_args": [
          "--privileged",
          "--device=/dev/ajantv2",
          "-v", "/dev:/dev"
        ]
        }
      }
    }
  }
}
```

##### **Advanced Pattern: Separate Build and Run Modes**

For cases where build and run containers need different Docker configurations, you can create separate modes that share the same Docker image:

```json
{
  "modes": {
    "default_mode": "production_build",

    "production_build": {
      "description": "Build production image with network access",
      "build": {
        "depends": ["tensorrt_backend"],
        "docker_build_args": ["--build-arg", "REGISTRY_TOKEN=xyz"]
      },
      "run": {
        "docker_run_args": ["--network=host"],
        "command": "echo 'Build complete'"
      }
    },

    "production_run": {
      "description": "Run production app with GPU access",
      "run": {
        "docker_run_args": ["--gpus=all", "--shm-size=1g"],
        "command": "python3 <holohub_app_source>/app.py --gpu"
      }
    }
  }
}
```

**Usage:**

```bash
# Phase 1: Build with network access
./holohub build myapp production_build

# Phase 2: Run with GPU access (reuses image from phase 1)
./holohub run myapp production_run --no-docker-build
```

Both modes automatically share the same Docker image name (`holohub:myapp`), so the run mode can use the image built by the build mode.

##### **Key Points for Mode Development**

- **`default_mode`** is required only if your project defines two or more modes; with a single mode, it is selected automatically. The default mode is used when no mode is explicitly specified on the command line.
- **Mode names** must match pattern `^[a-zA-Z_][a-zA-Z0-9_]*$` (alphanumeric + underscore, can't start with number)
- **Environment variables** can be specified at two levels:
  - Top-level `env`: Environment variables for **both build and run**
  - `mode_name.build.env`: Set environment variables **only for build and install**
  - `mode_name.run.env`: Set environment variables **only for run**
  - They can affect CLI behavior (e.g., `HOLOHUB_BUILD_LOCAL` to force local builds)
  - You can append to the existing environment variables by using `:`, like

    ```json
    "modes": {
      "mode_name": {
        "run": {
          "env": {
            "PATH": "<PATH>:<holohub_app_bin>/bin"  # append to the updated PATH by mode_name.env
            "CMAKE_BUILD_TYPE": "Release"  # override the existing CMAKE_BUILD_TYPE
          }
        },
        "env": {
          "PATH": "<holohub_bin>/bin:<PATH>"  # prepend to the existing PATH in the environment
        }
      }
    }
    ```

  - The inner `env` object takes the precedence over the top-level `env` object, and the top-level `env` object takes precedence over the CLI environment variables.
- **Docker arguments** can be specified in two places for different purposes:
  - `build.docker_build_args`: Docker **build** arguments for container image building (equivalent to CLI `--build-args`)
  - `run.docker_run_args`: Docker **run** arguments for both build and application containers (equivalent to CLI `--docker-opts`)
  - `run.docker_run_args` apply to both build containers (during compilation) and application containers (during execution)
  - if `no-docker-build` is specified, `build.docker_build_args` is ignored
- **Path placeholders** like `<holohub_app_source>`, `<holohub_data_dir>` are supported in commands
- **CLI parameter behavior**:
  - CLI parameters always override mode settings when provided
  - This flexible approach allows users to customize mode behavior without modifying `metadata.json`
  - When CLI parameters are not provided, mode settings are used as defaults, this is preferred to ensure consistency and reproducibility.
- **Requirements** reference dependency IDs defined elsewhere in the metadata
- **Modes provide complete control** over both build and runtime behavior for different deployment scenarios

### **Granular Build Control**

The new architecture provides precise control over your development workflow.

**Default Behavior:**
By default, `./holohub run` operates in a 'containerized mode', which means it will:

1. Build the container image (unless skipped, e.g. using `--no-docker-build`)
2. Build the application inside the container
3. Run the application inside the container

This container-first approach ensures consistency and reproducibility across different development environments.

**Command-line Options:**

- **`--local`**: Explicit local development mode
- **`--no-local-build`**: Skip application rebuild for faster iteration
- **`--no-docker-build`**: Use existing container images
- **Dedicated Commands**: Separate `build`, `run`, `build-container`, and `run-container` commands for clear workflow control

**Environment Variables:**

The CLI supports the following environment variables for customization:

**Build and Execution Control:**

- **`HOLOHUB_BUILD_LOCAL`**: Forces local mode (equivalent to `--local`), skips the container build steps and runs on the host directly.
- **`HOLOHUB_ALWAYS_BUILD`**: Set to `false` to skip builds with `--no-local-build` and `--no-docker-build`.
- **`HOLOHUB_ENABLE_SCCACHE`**: Defaults to `false`. Set to `true` to enable rapids-sccache for the build. You can configure sccache with `SCCACHE_*` environment variables per the [sccache documentation](https://github.com/rapidsai/sccache/tree/rapids/docs).

**Paths and Directories:**

- **`HOLOHUB_ROOT`**: HoloHub repository root directory, used to resolve relative paths for components, build artifacts, data, and other resources.
  - **`HOLOHUB_BUILD_PARENT_DIR`**: Root directory for all builds (default: `<HOLOHUB_ROOT>/build`).
  - **`HOLOHUB_DATA_DIR`**: Root data (such as models, datasets downloading during component building) directory (default: `<HOLOHUB_ROOT>/data`).
  - **`HOLOHUB_SETUP_SCRIPTS_DIR`**: Directory containing setup scripts (default: `<HOLOHUB_ROOT>/utilities/setup`), typically used to install dependencies and setup the environment via `--extra-scripts`.
- **`HOLOHUB_PATH_PREFIX`**: Prefix for path placeholder variables (default: `holohub_`). Path placeholders are used in `metadata.json` files, e.g., `"workdir": "holohub_app_source"`.
- **`HOLOHUB_DEFAULT_HSDK_DIR`**: Holoscan SDK root directory (default: `/opt/nvidia/holoscan`). Used for both local builds and container operations to locate SDK libraries and Python bindings.
- **`HOLOSCAN_SDK_ROOT`**: Path to local Holoscan SDK root directory (source or build tree), used for mounting local Holoscan SDK development trees into containers.
- **`HOLOHUB_SEARCH_PATH`**: Comma-separated directories to scan for metadata (default `applications,benchmarks,gxf_extensions,operators,pkg,tutorials,workflows`).

**Container Configuration:**

- **`HOLOHUB_REPO_PREFIX`**: Repository prefix for naming defaults (default: `holohub`). Used as the default value for container-related variables below.
- **`HOLOHUB_CONTAINER_PREFIX`**: Docker container name prefix (default: `HOLOHUB_REPO_PREFIX`).
- **`HOLOHUB_WORKSPACE_NAME`**: Workspace directory name in container (default: `HOLOHUB_REPO_PREFIX`). The `<HOLOHUB_ROOT>` directory is mounted in the container as `/workspace/<HOLOHUB_WORKSPACE_NAME>`.
- **`HOLOHUB_HOSTNAME_PREFIX`**: Container hostname prefix (default: `HOLOHUB_REPO_PREFIX`). Used when building VSCode development containers.
- **`HOLOHUB_DOCKER_EXE`**: Docker executable command (default: `docker`).

**Docker Images:**

By default, the Dockerfile used to build a container image for your project is chosen in that order:

1. Specified in your project's `metadata.json` file.
2. `<app_source>/Dockerfile`
3. `<app_source>/<language>/Dockerfile`
4. `HOLOHUB_DEFAULT_DOCKERFILE` env variable
5. `<HOLOHUB_ROOT>/Dockerfile`

This can be overridden by the `--docker-file` option.

If the selected Dockerfile has `ARG BASE_IMAGE`, the value of `BASE_IMAGE` will be automatically populated by the `./holohub build-container`, defaulting to `{base_image}:v{sdk_version}-{cuda_tag}` where:

- `{base_image}` defaults to `nvcr.io/nvidia/clara-holoscan/holoscan` and can be overridden by the **`HOLOHUB_BASE_IMAGE`** env variable.
- `{sdk_version}` defaults to the latest available Holoscan SDK version (e.g. `3.9.0`) and can be overridden by the **`HOLOHUB_BASE_SDK_VERSION`** env variable.
- `{cuda_tag}` is the Holoscan SDK container cuda tag. For Holoscan 3.7+, the default cuda major version is based on your host driver version, and can be overridden with the `--cuda <major_version>` option.
- `{base_image}:v{sdk_version}-{cuda_tag}` as a format can be overridden by the **`HOLOHUB_BASE_IMAGE_FORMAT`** env variable globally.

The whole base image (`<repo:tag>`) can be overridden by the `--base-img` option.

The name of the output image generated by `./holohub build-container [project]` varies based on the following factors:

- Legacy tags:
  - If the project is using the default Dockerfile, the name will default to `{container_prefix}:ngc-v{sdk_version}-{cuda_tag}` where:
    - `{container_prefix}` defaults to **`HOLOHUB_REPO_PREFIX`** and can be overridden by the **`HOLOHUB_CONTAINER_PREFIX`** env variable.
    - `{sdk_version}` defaults to the latest available Holoscan SDK version (e.g. `3.9.0`) and can be overridden by the **`HOLOHUB_BASE_SDK_VERSION`** env variable.
    - `{cuda_tag}` is the Holoscan SDK container cuda tag. For Holoscan 3.7+, the default cuda major version is based on your host driver version, and can be overridden with the `--cuda <major_version>` option.
    - `{container_prefix}:ngc-v{sdk_version}-{cuda_tag}` as a format can be overridden by the **`HOLOHUB_DEFAULT_IMAGE_FORMAT`** env variable globally.
  - If the project is using a custom Dockerfile, the name will default to `{container_prefix}:{project_name}` (see above for `container_prefix` value).
- New tags (see above for `container_prefix` value):
  - If building for a specific project, two additional tags will be created:
    - `{container_prefix}-{project_name}:{short_git_sha}`
    - `{container_prefix}-{project_name}:{git_branch_slug}`
  - If building the default container with no project specified, the tags will be:
    - `{container_prefix}:{short_git_sha}`
    - `{container_prefix}:{git_branch_slug}`

These output tags can be overridden by the `--img` option.

**Docker Runtime:**

- **`HOLOHUB_DEFAULT_DOCKER_BUILD_ARGS`**: Additional default arguments passed to `docker build` commands. Typically used to set global build arguments for all applications in the codebase, applications and commands can override these arguments.
- **`HOLOHUB_DEFAULT_DOCKER_RUN_ARGS`**: Additional default arguments passed to `docker run` commands. Typically used to set global runtime arguments for all applications in the codebase, applications and commands can override these arguments.
- **`HOLOHUB_BENCHMARKING_SUBDIR`**: Benchmarking subdirectory (default: `benchmarks/holoscan_flow_benchmarking`).

**Testing:**

- **`HOLOHUB_CTEST_SCRIPT`**: CTest script path used by `./holohub test` command (default: `<HOLOHUB_ROOT>/utilities/testing/holohub.container.ctest`).

**Other:**

- **`HOLOHUB_CMD_NAME`**: Command name displayed in help messages (default: `./holohub`). Allows customizing the command name for external codebases.
- **`HOLOHUB_CLI_DOCS_URL`**: CLI documentation URL. Allows customizing the documentation URL for external codebases.
- **`CMAKE_BUILD_TYPE`**: Default CMake build type (`debug`, `release`, or `relwithdebinfo`) when not specified in build commands.
- **`CMAKE_BUILD_PARALLEL_LEVEL`**: Number of parallel CMake build jobs.

## Getting Help

```bash
./holohub --help              # General help
./holohub run --help          # Command-specific help
./holohub run myapp --verbose --dryrun  # Debug mode
./holohub list                # Check available projects
```

## Bash Autocompletion

The autocompletion is automatically installed during setup and provides:

- **Project names**: All available applications, workflows, operators, and packages
- **Commands**: All available CLI commands (`build`, `run`, `list`, etc.)
- **Dynamic discovery**: Automatically finds new projects without manual configuration

### **Usage**

```bash
./holohub <TAB><TAB>          # Show all available options
./holohub run ultra<TAB>      # Complete to "ultrasound_segmentation"
./holohub build vid<TAB>      # Complete to "video_deidentification"
```

### **Manual Installation**

If autocompletion isn't working, you can manually enable it:

```bash
# Copy the autocomplete script
sudo cp utilities/holohub_autocomplete /etc/bash_completion.d/
# Add to your shell profile
echo ". /etc/bash_completion.d/holohub_autocomplete" >> ~/.bashrc
# Reload your shell
source ~/.bashrc
```

The autocompletion uses `./holohub autocompletion_list` command internally.

## Useful Commands

- When adding option that looks like an argument, use `=` instead of whitespace ` ` (because of [Python argparse design choice](https://github.com/python/cpython/issues/53580)):

```bash
--run-args="--verbose"   # instead of --run-args "--verbose"
```

- To clear unused docker cache during development:
  - Clear Docker images ([doc](https://docs.docker.com/reference/cli/docker/image/prune/)): `docker image prune`
  - Clear Docker buildx cache ([doc](https://docs.docker.com/reference/cli/docker/buildx/prune/)): `docker buildx prune`
  - Clear Docker containers, networks, images ([doc](https://docs.docker.com/reference/cli/docker/system/prune/): `docker system prune`

- All previous arguments containing `_` are changed to `-` consistently (e.g. `--base_img` to `--base-img`).

- `sudo ./holohub` may not work due the filtering of environment variable such as `PATH`.

- Running Docker container as a root user `--docker-opts="-u root"` or `--docker-opts="-u root --privileged"`:
  Running a Docker container as a root user may be necessary in scenarios where the containerized application requires elevated privileges to access certain system resources, modify system configurations, or perform administrative tasks. For example, this might be needed for debugging, testing, or running development tools that require root access.
  However, running containers as root poses significant security risks, as it can expose the host system to potential vulnerabilities if the container is compromised. It is recommended to avoid this practice in production environments and to use non-root users whenever possible. If root access is required, ensure that the container is run in a controlled and secure environment.
