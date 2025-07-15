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

# Container run with path to Holoscan SDK
./holohub run-container --local-sdk-root <path/to/holoscan-sdk> --img holohub:sdk-dev-latest [additional-options]
# Example of additional options: --no-docker-build --env VAR=value

# Benchmarking an application (only for applications/workflows)
./holohub build myapp --benchmark
```

###  **Development Environment**

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

# Show environment information
./holohub env-info
```


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
- **`HOLOHUB_BUILD_LOCAL`**: When set to any value, forces local mode (equivalent to always passing `--local`)
- **`HOLOHUB_ALWAYS_BUILD`**: Controls whether builds should be executed (defaults to `true`)
  - Set to `false` to skip both local and container builds
  - Useful for development iterations where you only want to run existing builds


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
