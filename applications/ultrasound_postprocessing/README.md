# Ultrasound Post-Processing Filter Design

A collection of tools for developing and deploying realtime ultrasound post-processing filters.

## Mission

Enable the ultrasound community (academia and industry) to have reproducible, real-time post-processing.

## Overview

This project includes an Ultrasound Post-Processing Filter Designer and light-weight library that enables you to design and run your ultrasound post-processing filter collection in Holoscan, all from a small and simple YAML config file.

The Filter Designer enables you to create the YAML configs working on Ultrasound File Format ([UFF](https://ieeexplore.ieee.org/document/8579642)) files (beamformed but not log compressed). You can load and filter your own data locally.

The included Holoscan Runner tool enables you to run those configurations in real-time using the Holoscan SDK.

## Prerequisites

### Filter Designer Application

- Linux workstation with NVIDIA GPU
- Streamlit
- CUDA Driver R580 or later

![docs/assets/designer.png]

### Holoscan Runner Application

- Holoscan SDK 3.9.0
- CUDA (if using GPU acceleration)
- Docker (for containerized deployment)

## Quick Start with Holoscan CLI

First, clone this repository:
```bash
git clone https://github.com/nvidia-holoscan/holohub.git
```

Then, run the containerized application:
```bash
./holohub run ultrasound_postprocessing designer
```
```bash
./holohub run ultrasound_postprocessing realtime
```

## Install and Run with Python Packaging

As an alternative to HoloHub CLI, you can instead design and run filters with project Python entrypoints.

Install this project with your Python package manager:
```bash
uv -m pip install git+ssh://git@gitlab-master.nvidia.com:12051/holoscan/holohub-internal.git@tbirdsong/us-postprocessing#subdirectory=applications/ultrasound_postprocessing
```

Then run your application of choice:
```bash
uv run streamlit run applications/ultrasound_postprocessing/ultra_post/app/streamlit_app.py
```

```bash
uv run python -m ultra_post.app.holoscan_app [--uff path/to/myfile.uff] [--fps 10]
```

## Key Concepts

### Glossary

- **Filters** are stateful, CuPy-based Python functions that operate on ultrasound images.
- The **Filter Registry** is a collection of pre-defined filters for use in apps such as the Filter Designer and in Holoscan SDK pipelines.
- **Pipelines** are a set of one or more filter objects assembled into a serialized graph or "chain".
- **Presets** are instructions for configuring a pipeline out of specific filters.
- **Operators** are Holoscan SDK "nodes" wrapping filter definitions for real-time serial execution.

## Development

### Project Structure

```
ultrasound_postprocessing
├── CMakeLists.txt
├── Dockerfile
├── docs
├── external
├── metadata.json
├── plans
├── presets
├── pyproject.toml
├── README.md
├── tests
├── tools
├── ultra_post (Python module)
```

### Review Existing Operators

```bash
./holohub run ultrasound_postprocessing list-filters
```

### Validate a preset

```bash
./holohub run ultrasound_postprocessing validate <presets/my-preset.yml>
```

### Run Tests

```bash
./holohub test ultrasound_postprocessing
```

## Contributing

We welcome contributions that align with our mission to enable the ultrasound community to have reproducible, real-time post-processing.

## Ways to Contribute

1.  **Presets**: Create a processing pipeline preset (YAML) and save it to `presets/`.
2.  **Filters**:
    -   Add your filter implementation to `ultra_post/filters/`.
    -   Register it in `ultra_post/filters/registry.py` (add to `FILTERS` and `DEFAULT_PARAMS`).
3.  **Improvements**: Bug fixes and performance optimizations are highly encouraged.
    -   Our goal is to have code that is easy to understand and highly performant.

Please see the [HoloHub Contributing Guide](/CONTRIBUTING.md) for developer guidance.

## Code Style

-   **Concise & Approachable**: We prioritize readability so new users can quickly understand the post-processing logic.
-   **Performant**: We aim for code that is both expressive and performant, enabling real-time processing.


## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## Authors

- Walter Simson - NVIDIA Holoscan Team

## Acknowledgments

- NVIDIA Holoscan Team
- Open source community contributors
