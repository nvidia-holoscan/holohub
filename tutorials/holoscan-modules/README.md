# Holoscan Modules Tutorials

## Purpose

Learn to create, package, and consume Holoscan Modules — reusable, versioned SDK extensions
that bundle operators, assets, and metadata for distribution and discovery across projects.

## Usage

Select a tutorial from the list below to get started:

- **[Create a Holoscan Module](./create-a-module)** — Learn to set up a new module from scratch,
  implement operators, build, test, package, and publish it to your team or community.
- **[Use a Holoscan Module](./use-a-module)** — Learn to declare a module dependency, install
  or build it, and import its operators in your own application or framework project.

Browse published modules in the [Modules tab](https://nvidia-holoscan.github.io/holohub/modules/).

## Requirements

- Holoscan SDK 0.4.2 or later
- C++ compiler with C++17 support or Python 3.8+
- CMake 3.20 or later (for C++ modules)

## Examples

### Running a Tutorial

Navigate to the tutorial directory and follow the instructions in its README:

```bash
cd create-a-module
# Follow the walkthrough in README.md
```

Each tutorial includes step-by-step guidance and runnable examples.

## Architecture

The holoscan-modules tutorials are structured as standalone walkthroughs:

- **create-a-module/** — End-to-end guide for module scaffolding, implementation, and publication
- **use-a-module/** — Guide for consuming modules from HoloHub, external CMake projects, and pip packages

Each tutorial includes:

- Conceptual overview and rationale
- Step-by-step walkthrough with code examples
- Troubleshooting and reference sections
- Links to relevant Holoscan SDK documentation and the Modules directory
