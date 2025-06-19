# HoloHub CLI Migration Guide

## Overview

HoloHub has been refactored from bash-based CLI scripts (`./run` and `./dev_container`) to a modern, unified Python-based CLI tool (`./holohub`). This upgrade provides a cleaner, more powerful, and more intuitive development experience.

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
./holohub run myapp --language cpp --local
```

### **Enhanced Developer Experience**
- **Unified Interface**: One tool instead of juggling `./run` and `./dev_container`
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
# previous ./run build myapp ...
./holohub build myapp --local --build-type debug --build-with "op1;op2"

# Build in container
# previous ./dev_container build ...
./holohub build myapp --build-type debug --verbose

# Run locally
# previous ./run build myapp && ./run launch myapp cpp
./holohub run myapp --local --language cpp --build-type debug

# Run in container
# previous ./dev_container build_and_run ...
./holohub run myapp --language=cpp --run-args="--config=config.json" # NEW

# Install packages
# previous ./run build myapp --install
./holohub install myapp
```

### **Container Operations**
```bash
# Container build
# previous ./dev_container build ...
./holohub build-container --base-img nvidia/holoscan:latest

# Container run
# previous ./dev_container launch ...
./holohub run-container --img myapp:dev --ssh-x11 --docker-opts="--entrypoint=bash" --no-docker-build

# Development environment
# previous ./dev_container vscode ...
./holohub vscode myapp --language cpp
```

### **Advanced Features**
```bash
# Custom build options
./holohub build myapp --build-with "op1;op2" --configure-args='-DCUSTOM=ON'

# Benchmarking an application (only for applications/workflows)
./holohub build myapp --benchmark
```

###  **Development Environment**

Development commands remain familiar:

```bash
# Environment setup (unchanged)
[sudo] ./holohub setup

# Enhanced linting with integrated dependency management
./holohub lint --install-dependencies  # Previously ./run install_lint_deps
./holohub lint --fix                   # Same clean interface

# Consistent naming conventions
./holohub clear-cache                  # Previously ./run clear_cache
./holohub list                         # Unchanged
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

### **Intelligent Development Experience**
- **Smart Error Messages**: Automatic project name suggestions for typos and better validation
- **Modern Argument Parsing**: Consistent interface across all commands with improved help text
- **Enhanced Container Integration**: Better argument forwarding and container lifecycle management

### **Granular Build Control**
The new architecture provides precise control over your development workflow:
- **`--local`**: Explicit local development mode
- **`--no-local-build`**: Skip application rebuild for faster iteration
- **`--no-docker-build`**: Use existing container images
- **Dedicated Commands**: Separate `build`, `run`, `build-container`, and `run-container` commands for clear workflow control

## Getting Help

```bash
./holohub --help              # General help
./holohub run --help          # Command-specific help
./holohub run myapp --verbose --dryrun  # Debug mode
./holohub list                # Check available projects
```


## Useful Commands

- When adding option that looks like an argument, use `=` instead of whitespace ` `:
```bash
--run-args="--verbose"   # instead of --run-args "--verbose"
```

- To clear unused docker cache during development:
```
docker image prune -f && docker buildx prune --filter type=regular -f
```
