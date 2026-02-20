# New Capabilities and Enhancements

- [Project Creation and Testing](#project-creation-and-testing)
- [Development Environment](#development-environment)
- [Application Modes](#application-modes)
  - [Mode and CLI Arguments](#mode-and-cli-arguments)
  - [Application Development](#application-development)
  - [Example `metadata.json` File](#example-metadatajson-file)
- [Use Different Docker Options for Build and Run](#use-different-docker-options-for-build-and-run)
- [Advanced Feature Examples](#advanced-feature-examples)

## Project Creation and Testing

The new CLI introduces development lifecycle features:

| Description | Command |
| ----------- | ------- |
| Scaffold new projects with templates | `./holohub create my_new_app --language cpp` |
| Integrated testing framework | `./holohub test myapp` |
| With code coverage (requires coverage tools in image) | `./holohub test myapp --coverage` |
| Specify language when a project has multiple implementations | `./holohub test myapp --language python` |
| Show environment information | `./holohub env-info` |


## Development Environment

* Setup Holohub stays the same and sudo privileges may be required.

  ```bash
  ./holohub setup  

* Enhanced linting with integrated dependency management.

  ```bash
  ./holohub lint --install-dependencies  # Previously ./run install_lint_deps
  ```

  The following lint commands stay the same:

    ```bash
    ./holohub lint                         
    ./holohub lint --fix                   
    ```

* More consistent naming conventions for the following commands:
  
  ```bash
  ./holohub clear-cache                 
  ./holohub list                        
  ```
  
  These commands were previously prefaced with `./run`.


## Application Modes

HoloHub applications with a 'modes' field in their `metadata.json` now support multiple **modes**. The modes are pre-configured setups for different use cases, hardware configurations, or deployment scenarios.

Modes can be used to provide typical configurations for:

- Different hardware
- Deployment scenarios (for example, development, production, cloud inference)
- Data sources (for example, live video, recorded files, distributed streaming)

If no mode is specified, then the command is run in `default_mode`.

To use modes:

1. Discover available modes for an application

   ```bash
   ./holohub modes body_pose_estimation
   ```

  * To run an application in Replayer mode:

    ```bash
    ./holohub run body_pose_estimation replayer  
    ```

  * To run an application in Local LLM inference mode:

    ```bash
    ./holohub run holochat standalone                        
    ```


### Mode and CLI Arguments

CLI flags interact with mode settings. When you:

* Pass a CLI parameter, it overrides the corresponding mode setting (the CLI shows warnings for overrides).  
* Do not pass a parameter, the mode setting is used.

CLI parameters override mode settings for both implicit and explicit modes:

| Description | Command |
| ----------- | ------- |
| Append to mode's run command | `./holohub run holochat --run-args="--debug"` |
| Override mode's build dependencies | `./holohub run myapp --build-with="ops"` |
| Override build dependencies for a specific mode | `./holohub build myapp standard --build-with="ops"` |
| Append to a non-default mode's run command | `./holohub run holochat cloud --run-args="--test=case"` |

If provided, a CLI parameter always overrides the mode setting.

### Application Development

Define modes in `metadata.json` so users can run your app with different configurations. For example `./holohub run myapp production`. Each mode is a named object under `metadata.modes`. Use `metadata.default_mode` only when you have two or more modes.

Mode names must match `^[a-zA-Z_][a-zA-Z0-9_]*$`.  

* Env: `run.env` overrides mode `env`; mode `env` overrides CLI.
* CLI parameters override mode settings when provided. 
* `build.docker_build_args` is ignored if the user passes `--no-docker-build`.

Structure of `metadata.json`:

```json
{
  "metadata": {
    "default_mode": "mode_name",
    "modes": {
      "mode_name": {
        "description": "...",
        "run": { "command": "...", "workdir": "..." }
      }
    }
  }
}
```

Each mode needs the following fields:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `description` | string | **Required.** What this mode does. |
| `run` | object | **Required.** Run config: `command`, `workdir`, optional `docker_run_args`, `env`. |
| `requirements` | array of strings | Dependency IDs for this mode. |
| `build` | object | Optional. `depends`, `docker_build_args`, `cmake_options`, `env`. |
| `env` | object | Environmental variables for **both** build and run (key-value). |

If needed, the environment variables provide for three scopes, only use the one that fits:

| Where | When applied, result in: |
| ----- | ------------- |
| `mode.env` | Build and run |
| `mode.build.env` | Build and install only |
| `mode.run.env` | Run only (local runs) |

* Build fields map to the following CLI commands: 

  * `depends` → build deps 
  * `docker_build_args` → `--build-args`
  *  `cmake_options` → CMake args

* Run field `docker_run_args` maps to the CLI `--docker-opts` (applies to both build and app containers). 

* String or array allowed for `docker_build_args` and `docker_run_args`.

### Example `metadata.json` File

Path placeholders such as `<holohub_app_source>` and `<holohub_data_dir>` are supported in commands.

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
        "env": { "BUILD_ENV": "production" },
        "build": {
          "depends": ["tensorrt_backend", "gpu_ops"],
          "docker_build_args": ["--build-arg", "TENSORRT_VERSION=8.6", "--network=host"],
          "cmake_options": ["-DUSE_TENSORRT=ON", "-DCUDA_ARCH=sm_86"]
        },
        "run": {
          "command": "python3 <holohub_app_source>/app.py --backend tensorrt --optimization-level 3",
          "workdir": "holohub_bin",
          "docker_run_args": ["--gpus=all", "--shm-size=1g"],
          "env": { "LOG_LEVEL": "info" }
        }
      }
    }
  }
}
```



## Advanced Feature Examples

Use `-h` or `--help` to view a complete list of commands or subcommand options.

* Custom build options

  ```bash
  ./holohub build myapp --build-with "op1;op2" --configure-args='-DCUSTOM=ON'
  ```

* Container run with a path to a local Holoscan SDK installation from the host system:

  ```bash
  ./holohub run-container --local-sdk-root <path/to/holoscan-sdk> --img holohub:sdk-dev-latest [additional-options]
  ```

  Additional options: `--no-docker-build` `--env VAR=value`

* Benchmarking an application, which is only for applications and workflows.

  ```bash
  ./holohub build myapp --benchmark
  ```
