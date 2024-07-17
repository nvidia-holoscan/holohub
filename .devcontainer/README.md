# Holohub Dev Containers

Holohub uses [Development Containers](https://containers.dev/) to provide consistent and convenient development environments for Holoscan & Holohub. This guide covers the usage of Holohub Dev Container using [Visual Studio Code](https://code.visualstudio.com/).

> 💡: Note: this guide is specific to the Linux development environment and is tested on Ubuntu 22.04 LTS.

## Prerequisites

- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [VS Code](https://code.visualstudio.com/) with the [Dev Container Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
  - Install [Dev Container Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) via command line
    ```bash
    code --install-extension ms-vscode-remote.remote-containers
    ```

### Steps

1. Clone the Repository
    ```bash
    git clone git@github.com:nvidia-holoscan/holohub.git
    ```
2. Open the cloned directory in terminal.

3. Launch a Dev Container with the `./dev_container` script:

The following command starts a new Dev Container for Holohub using the base [Dockerfile](../Dockerfile).

```bash
./dev_container vscode
```

4. VS Code will build and initialize the selected Dev Container. This can take a few minutes the first time.

5. Once initialized, a new VS Code window will open with a prompt, click **Trust Folder & Continue** to continue the Dev Container build process.

6. When ready, the Holohub directory is mirrored into the container under `/workspace/holohub` to ensure any changes are persistent.


### Debugging Holohub Application

Most of the Holohub application are pre-configured with one or more launch profiles. Click the **Run & Debug** tab and find the application that you want to run and debug from the dropdown.



## Advanced Options


For Holohub applications that bundles with a Dockerfile with additional dependencies and tools, pass the name of the application to the `./dev_container` script.
Take the [endoscopy_depth_estimation](../applications/endoscopy_depth_estimation) application as an example, the command will launch a Dev Container using the [Dockerfile](../applications/endoscopy_depth_estimation/Dockerfile) as the base image that builds `OpenCV` 

```bash
./dev_container vscode endoscopy_depth_estimation
```

# Get help on the available vscode command:
./dev_container vscode -h
