# Contributing to HoloHub

Welcome to HoloHub! We're excited that you're interested in contributing to the NVIDIA Holoscan developer community. This guide will help you understand how to make meaningful contributions to our collection of applications, operators, workflows, and tutorials.

## Quick Start

New to HoloHub? Follow these steps:

1. **Understand HoloHub**: Read our [README](./README.md) to learn about the project
2. **Determine your contribution type**: Use our [decision guide](#types-of-contributions) below
3. **Set up your development environment**: Follow the [developer process](#developer-process)
4. **Prepare your submission**: Use our [submission guidelines](#preparing-your-submission)
5. **Review the checklist**: Complete our [contribution checklist](#contribution-checklist)
6. **Submit for review**: Create a pull request following our [process](#developer-process)

## Table of Contents

- [Quick Start](#quick-start)
- [Introduction](#introduction)
- [Types of Contributions](#types-of-contributions)
- [Readiness Assessment](#readiness-assessment)
- [Developer Process](#developer-process)
- [Preparing Your Submission](#preparing-your-submission)
- [Build System Integration](#build-system-integration)
- [Contribution Checklist](#contribution-checklist)
- [Code Quality and Standards](#code-quality-and-standards)
  - [Linting](#linting-and-code-quality)
  - [Testing](#testing)
  - [Unit Testing Python Operators](#unit-testing-python-operators)
- [Development Tools](#development-tools)
  - [Debugging](#debugging-and-performance)
  - [Performance](#performance)
- [Getting Help](#getting-help)
  - [Troubleshooting](#troubleshooting)
  - [Reporting Issues](#reporting-issues)

## Introduction

HoloHub is a collaborative ecosystem for the NVIDIA Holoscan SDK, featuring community-contributed applications, reusable operators, end-to-end workflows, and educational tutorials. Your contributions help expand the capabilities available to developers working on real-time AI applications across healthcare, industrial inspection, and other domains.

Whether you're fixing a bug, adding a new feature, or sharing a complete application, this guide will help you contribute effectively to the HoloHub community.

## Types of Contributions

Choose the right contribution type based on what you want to share:

### Decision Tree

```text
What are you contributing?
├── 🔄 Complete end-to-end pipeline (sensor → insight)?
│   └── → Submit as a "Workflow"
├── 🎯 Focused application for specific use case?
│   └── → Submit as an "Application"
├── 🧩 Reusable component for multiple use cases?
│   └── → Submit as an "Operator" + demo Application
├── 📚 Educational content or tutorial?
│   └── → Submit as a "Tutorial"
└── 🔧 Bug fix or enhancement to existing code?
    └── → Submit a "Pull Request"
```

> **Important**: Workflows are _end-to-end_ reference applications demonstrating complete "sensor-to-insight" pipelines. They integrate multiple components to solve entire use cases, while applications may focus on specific functionality.

## Developer Process

### Prerequisites

Before getting started:

1. Review [HoloHub prerequisites](./README.md#prerequisites)
2. Ensure you have Git and required development tools installed
3. Familiarize yourself with GitHub's [starting documentation](https://docs.github.com/en/get-started/start-your-journey) if you're new to GitHub

### Step-by-Step Process

1. **Fork the Repository**

   [Fork](https://help.github.com/en/articles/fork-a-repo) the [upstream HoloHub repository](https://github.com/nvidia-holoscan/holohub).

2. **Clone and Set Up Local Development**

   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_FORK.git holohub
   cd holohub
   
   # Add upstream remote for staying in sync
   git remote add upstream https://github.com/nvidia-holoscan/holohub.git
   
   # Create a feature branch
   git checkout -b feature/your-feature-name
   ```

3. **Develop Your Contribution**

   - Follow the specific guidelines for your [contribution type](#types-of-contributions)
   - If you are developing a new application, you can use the `./holohub create <application_name>` command to generate the initial project scaffolding. This command sets up a new project with the appropriate directory structure and necessary files.
   - Ensure your code meets [HoloHub standards](#preparing-your-submission)
   - Test your changes locally

4. **Commit and Push Changes**

   ```bash
   # Stage your changes
   git add .
   
   # Commit with sign-off (required)
   git commit -s -m "Add your descriptive commit message"
   
   # Push to your fork
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**

   - Navigate to your fork on GitHub
   - [Create a Pull Request](https://help.github.com/en/articles/creating-a-pull-request) to merge your branch into the upstream repository
   - Ensure you select the correct source and target branches
   - Fill out the PR template completely

6. **Review Process**

   - HoloHub maintainers will review your PR
   - Address any feedback or requested changes
   - Once approved, your contribution will be merged

Thanks in advance for your patience as we review your contributions. We do appreciate them!

## Preparing Your Submission

We request that members follow the guidelines in this document to make sure new submissions can be easily used by others.

### Required Components

A typical submission consists of:

- **Code**: Application, workflow, operator, and/or tutorial code using the Holoscan SDK
- **Metadata**: A [`metadata.json`](#metadata-description) file
- **Documentation**: A [README](#readme-file) file describing the contribution

### Readiness Assessment

**✅ Ready to submit:**

- Feature-complete and tested code
- Documentation included
- Follows HoloHub standards
- Submit a PR and request review from `@nvidia-holoscan/holohub`

**🚧 Work in progress:**

- Fork HoloHub for development
- Submit a "Draft PR" as early as possible
- Call `@nvidia-holoscan/holohub` in the comments if you need to discuss anything.

### Acceptance Criteria

For a submission to be accepted into HoloHub it must meet these criteria:

- ✅ Clearly demonstrates added value to the Holoscan community
- ✅ Receives approval from at least one HoloHub maintainer
- ✅ [Code linting](#linting-and-code-quality) tests pass
- ✅ Any new [code tests](#testing) pass
- ✅ Includes proper documentation and metadata
- ✅ Follows naming conventions and coding standards

### Metadata description

Every application and operator should have an associated `metadata.json` file describing features and dependencies.

Schemas are available for different contribution types:

- [Workflows](./workflows/metadata.schema.json)
- [Applications](./applications/metadata.schema.json)
- [GXF Extensions](./gxf_extensions/metadata.schema.json)
- [Operators](./operators/metadata.schema.json)
- [Tutorials](./tutorials/metadata.schema.json)

#### Example metadata.json Structure

```json
// Main json definition for application or operator
"application|operator": {
    // Explicit name of the contribution
    "name": "explicit name of the application/operator",
    // Author(s) of the contribution
    "authors": [
      {
        "name": "Your Name",
        "affiliation": "Your Organization"
      }
    ],
    // Supported language
    // If multiple languages are supported, create a directory per language and a json file accordingly
    "language": "C++|Python|GXF",
    // Version of the contribution
    "version": "Version of the contribution in the form: major.minor.patch",
    // Change log
    "changelog": {
        "X.X": "Short description of the changes"
    },
    "holoscan_sdk": {
        // Minimum supported holoscan version
        "minimum_required_version": "0.6.0",
        // All versions of Holoscan SDK tested for this operator/application
        "tested_versions": [
            "0.6.0"
        ]
    },
    // Supported platforms
    "platforms": ["x86_64", "aarch64"],
    // Free-form tags for referencing the contribution
    "tags": ["Endoscopy", "Video Encoding"],
    // Ranking of the contribution. See below for ranking meaning
    "ranking": 4,
    // Dependencies for the current contribution
    "dependencies": {
        "operators": [{
            "name": "mydependency",
            "version": "x.x.x"
        }
        ]
    },
    // Command to run/test the contribution. This is valid for applications.
    // This command is used by the main run script to test the application/
    // Use the <holohub_data_dir> for referencing the data directory
    // "workdir" specifies the working directory and can be holohub_app_bin, holohub_app_source or holohub_bin
    "run": {
        "command": "./myapplication --data <holohub_data_dir>/mydata",
        "workdir": "holohub_app_bin|holohub_app_source|holohub_bin"
    }
}
```

In the `metadata.json` file, the following directories can be referenced:

- `holohub_app_bin`: The directory containing the built application binary (e.g. `<holohub_root>/build/myapp/applications/myapp/cpp`)
- `holohub_app_source`: The directory containing the source code of the application (e.g. `<holohub_root>/applications/myapp/cpp/`)
- `holohub_bin`: The root build directory containing built binaries (e.g. `<holohub_root>/build/`)
- `holohub_data_dir`: The directory containing the data for the application (e.g. `<holohub_root>/data/`)

#### Ranking Levels for metadata.json

Please provide a self-assessment of your HoloHub contribution according to these levels:

| Level | Description | Requirements |
|-------|-------------|--------------|
| **0** | Production-ready, SDK-level quality | • Widespread community dependence<br>• Above 90% code coverage<br>• Nightly testing monitored<br>• All Level 1 requirements |
| **1** | Very high-quality code | • Meets all Holoscan SDK coding standards<br>• Builds on all platforms within 1 month of releases<br>• 75% code coverage<br>• Continuous integration testing<br>• All Level 2 requirements |
| **2** | Quality code | • Compiles on community platforms<br>• May have specific external dependencies<br>• Tests pass on supported platforms<br>• All Level 3 requirements |
| **3** | Features under development | • Builds on specific platforms/configurations<br>• Some tests passing on supported platforms<br>• All Level 4 requirements |
| **4** | Code of unknown quality | • Builds on specific platforms/configurations<br>• Minimal test coverage |
| **5** | Deprecated | • Known to be of limited utility<br>• May have known bugs |

### README File

Include a comprehensive `README.md` file with:

- **Purpose**: Clear description of what your contribution does
- **Usage**: How to build, run, and use your contribution
- **Requirements**: Dependencies and system requirements
- **Examples**: Code samples or usage examples where applicable
- **Architecture**: High-level design overview (for complex contributions)

Use the [terms defined in the glossary](README.md#Glossary) when referring to HoloHub-specific locations.

### Directory Structure

All contributions should follow consistent directory structures based on their type:

#### Operators

```text
holohub/operators/your_operator_name/
├── metadata.json                   # Required: follows operator schema
├── README.md                       # Required: describes purpose and usage
├── your_operator_name.py|.cpp|.hpp # Main operator implementation
├── test_your_operator_name.py      # Required for Python operators
└── CMakeLists.txt                  # If needed for C++ operators
```

#### Applications

```text
holohub/applications/your_app_name/
├── metadata.json                   # Required: follows application schema
├── README.md                       # Required: describes purpose and architecture
├── your_app_name.py|.cpp           # Main application code
└── CMakeLists.txt                  # For build system integration
```

#### Workflows

```text
holohub/workflows/your_workflow_name/
├── metadata.json                   # Required: follows workflow schema
├── README.md                       # Required: describes workflow purpose
├── your_workflow_name.py|.cpp      # Main application code
└── CMakeLists.txt                  # For build system integration
```

#### GXF Extensions

```text
holohub/gxf_extensions/your_extension_name/
├── metadata.json                   # Required: follows extension schema
├── README.md                       # Required: describes extension purpose
├── your_extension.cpp              # Main extension implementation
├── your_extension.hpp              # Header files
└── CMakeLists.txt                  # Required for build system
```

#### Tutorials

```text
holohub/tutorials/your_tutorial_name/
├── README.md                       # Required: tutorial content and objectives
├── metadata.json                   # Optional: follows tutorial schema
├── tutorial_code.py|.cpp           # Tutorial implementation
└── assets/                         # Optional: images, diagrams, etc.
```

#### Packages

```text
holohub/pkg/your_package_name/
├── CMakeLists.txt                  # Required: package configuration
└── README.md                       # Optional: package description
```

### Naming Conventions

For an operator named "Adaptive Thresholding":

| Component | Convention | Example |
|-----------|------------|---------|
| Class Name | TitleCase + "Op" suffix | `AdaptiveThresholdingOp` |
| metadata.json "name" | Same as class name | `AdaptiveThresholdingOp` |
| Directory | snake_case | `adaptive_thresholding` |
| Filename | Same as directory + extension | `adaptive_thresholding.py` |
| README Title | Title Case + "Operator" | "Adaptive Thresholding Operator" |
| Unit Test | "test_" + directory name | `test_adaptive_thresholding.py` |

### Build System Integration

All contributions that include code need to be integrated with HoloHub's build system using CMake. Edit the appropriate `CMakeLists.txt` to add your contribution:

**For Operators:**

```cmake
# In ./operators/CMakeLists.txt
add_holohub_operator(my_operator DEPENDS EXTENSIONS my_extension)
```

If the operator wraps a GXF extension then the optional `DEPENDS EXTENSIONS` should be added to tell the build system to build the dependent extension(s).

**For Extensions:**

```cmake
# In ./gxf_extensions/CMakeLists.txt
add_holohub_extension(my_extension)
```

**For Applications:**

```cmake
# In ./applications/CMakeLists.txt
add_holohub_application(my_application DEPENDS
                        OPERATORS my_operator1 my_operator2)
```

If the application relies on one or more operators then the optional `DEPENDS OPERATORS` should be added so that
the build system knows to build the dependent operator(s).

**For Workflows:**

```cmake
# In ./workflow/CMakeLists.txt
add_holohub_application(my_workflow DEPENDS
                        OPERATORS my_operator1 my_operator2)
```

If the workflow relies on one or more operators then the optional `DEPENDS OPERATORS` should be added so that
the build system knows to build the dependent operator(s).

**For Packages:**

**CMake Configuration:**

```cmake
# In ./pkg/my_package/CMakeLists.txt
holohub_configure_deb(
  NAME "my-package-dev"
  COMPONENTS "my-headers" "my-libs"  # optional
  DESCRIPTION "My project description"
  VERSION "1.0.0"
  VENDOR "Your Organization"
  CONTACT "Your Name <your.email@example.com>"
  DEPENDS "libc6 (>= 2.34), libstdc++6 (>= 11)"
  SECTION "devel"      # optional
  PRIORITY "optional"  # optional
)
```

**Package Registration:**

```cmake
# In ./pkg/CMakeLists.txt
add_holohub_package(my_packager
                    APPLICATIONS my_app1
                    OPERATORS my_op1 my_op2)
```

**Prerequisites:**
Ensure your CMake targets have `install` rules defined. Use `COMPONENT` to control packaging granularity.

### License and Legal Guidelines

- **Open Source Compatibility**: Ensure you have rights to contribute your work
- **License Compliance**: All contributions inherit the Apache 2.0 license
- **Patent Considerations**: Verify no patent conflicts are introduced
- **Contribution Signing**: All commits must be signed-off (see [signing requirements](#signing-your-contribution))

> **Note**: NVIDIA is not responsible for conflicts resulting from community contributions.

### Coding Guidelines

- **Style Compliance**: All code must adhere to Holoscan SDK coding standards
- **Descriptive Naming**: Use clear, English descriptive names for functionality
- **Avoid Abbreviations**: Minimize use of acronyms, brand names, or team names
- **Code Documentation**: Include inline comments for complex logic
- **Error Handling**: Implement appropriate error handling and validation

### Signing Your Contribution

**Why Sign-Off is Required:**
We require all contributors to "sign-off" their commits to certify the contribution is their original work or they have rights to submit it under the same license.

**How to Sign-Off:**

```bash
# Sign-off when committing
git commit -s -m "Add cool feature."

# This appends to your commit message:
# Signed-off-by: Your Name <your@email.com>
```

**Developer Certificate of Origin (DCO):**

By signing-off, you certify that:

- (a) The contribution was created by you with rights to submit under the open source license
- (b) The contribution is based on appropriately licensed previous work
- (c) The contribution was provided by someone who certified (a), (b), or (c)
- (d) You understand this project and contribution are public with permanent record

Full DCO text is available in the [Linux Foundation DCO](https://developercertificate.org/).

> **Important**: Contributions without proper sign-off will not be accepted.

## Contribution Checklist

Before submitting your contribution, ensure you've completed:

### Pre-Submission Checklist

- [ ] **Code Quality**
  - [ ] Code follows Holoscan SDK coding standards
  - [ ] All linting checks pass (`./holohub lint`)
  - [ ] Code is properly documented with clear comments
  - [ ] Error handling is implemented appropriately

- [ ] **Testing**
  - [ ] Code builds successfully on target platforms
  - [ ] All existing tests still pass
  - [ ] New functionality includes appropriate tests
  - [ ] Python operators include unit tests (if applicable)

- [ ] **Documentation**
  - [ ] `README.md` is comprehensive and well-written
  - [ ] `metadata.json` is complete and follows the correct schema
  - [ ] Code examples and usage instructions are included
  - [ ] Architecture or design decisions are documented

- [ ] **Legal and Compliance**
  - [ ] All commits are signed-off (`git commit -s`)
  - [ ] No license or patent conflicts introduced
  - [ ] Code is original work or properly attributed

- [ ] **Integration**
  - [ ] Follows HoloHub naming conventions
  - [ ] Properly integrated with build system (CMakeLists.txt updated)
  - [ ] Dependencies are correctly specified
  - [ ] Can be built and run following provided instructions

### Submission Checklist

- [ ] **GitHub Workflow**
  - [ ] Forked the upstream repository
  - [ ] Created a descriptive feature branch
  - [ ] Pull request targets the correct base branch
  - [ ] PR description clearly explains the contribution

- [ ] **Review Readiness**
  - [ ] Code is ready for review (not work-in-progress)
  - [ ] All CI/CD checks pass
  - [ ] Requested review from appropriate maintainers (@nvidia-holoscan/holohub)

## Code Quality and Standards

### Linting and Code Quality

HoloHub enforces code quality through automated linting checks that run in CI/CD pipelines.

#### Installing Lint Tools

```bash
./holohub lint --install-dependencies
```

#### Running Lint Checks

```bash
# Lint entire repository
./holohub lint

# Lint specific path
./holohub lint path/to/your/code
```

#### Fixing Common Lint Issues

```bash
./holohub lint --fix
```

### Testing

#### Integration tests

Each operator should have at least one associated [application](./applications/) demonstrating its capabilities.

#### Writing Tests

Applications should include a testing section in their `CMakeLists.txt` for functional testing. HoloHub uses [CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html) for automated testing.

#### Running Tests

```bash
./holohub test <project>
```

### Unit Testing Python Operators

HoloHub strongly encourages unit tests for Python operators.

#### Testing Framework and Structure

- **Framework**: Use `pytest`
- **File Location**: Same directory as operator: `test_<operator_name>.py`
- **Fixtures**: Reuse common fixtures from `conftest.py`

#### Required Test Categories

1. **Initialization Tests**: Verify operator creation and properties
2. **Port Setup Tests**: Ensure input/output ports are configured correctly
3. **Error Handling Tests**: Test invalid arguments using `pytest.raises`
4. **Compute Logic Tests**: Test main functionality with various inputs
5. **Edge Case Tests**: Cover boundary conditions and error scenarios

#### Example Test Structure

```python
import pytest
from .my_operator import MyOperatorOp
from holoscan.core import Operator, _Operator as BaseOperator


def test_my_operator_init(fragment):
    name = "myoperator_op"
    op = MyOperatorOp(fragment=fragment, name=name, tensor_name="image")
    assert isinstance(op, BaseOperator), "MyOperator should be a Holoscan operator"
    assert op.operator_type == Operator.OperatorType.NATIVE, "Operator type should be NATIVE"
    assert f"name: {name}" in repr(op), "Operator name should appear in repr()"

@pytest.mark.parametrize("shape", [(32, 32, 3), (16, 16, 1)])
def test_my_operator_compute(fragment, op_input_factory, op_output, execution_context, mock_image, shape):
    image = mock_image(shape)
    op_input = op_input_factory(image, tensor_name="image", port="in")
    op = MyOperatorOp(fragment=fragment, tensor_name="image")
    op.compute(op_input, op_output, execution_context)
    out_msg, out_port = op_output.emitted
    assert out_port == "out"
    assert out_msg["image"].shape == shape
    # the rest of compute logic that covers the main functionality of the operator

def test_my_operator_invalid_param(fragment):
    with pytest.raises(ValueError):
        MyOperatorOp(fragment=fragment, tensor_name="image", invalid_param=-1)

# Add as many test cases as needed to cover all the functionality of the operator and the edge cases.
```

#### Running Python Unit Tests

```bash
# From repository root
pytest operators/<your_operator_dir>/

# Run with coverage
pytest --cov=operators/<your_operator_dir>/ operators/<your_operator_dir>/

# Run with verbose output
pytest -v operators/<your_operator_dir>/
```

#### Best Practices

- **Test Isolation**: Keep tests independent and isolated
- **Descriptive Names**: Use clear, descriptive test function names
- **Assertion Messages**: Include helpful assertion messages
- **Parameterized Tests**: Use `@pytest.mark.parametrize` for multiple scenarios
- **Fixture Reuse**: Leverage common fixtures from `conftest.py`
- **Edge Cases**: Test boundary conditions and error scenarios
- **Documentation**: Add docstrings explaining test purpose

For examples, see existing test files like:

- `operators/deidentification/pixelator/test_pixelator.py`
- `conftest.py` for available fixtures

## Development Tools

### Debugging and Performance

#### Debugging Resources

**Holoscan SDK Documentation:**

- [Debugging Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_debugging.html) - Common debugging scenarios, crashes, profiling
- [Logging Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_logging.html) - Runtime logging setup

**Development Environment:**

- VSCode Dev Container support available in Holoscan SDK
- `holohub` CLI tool with debugging options:
  - `--as_root`: Launch as root for expanded debugging permissions
  - `--local_sdk_root`: Mount local SDK for debug symbol access

**Debugging Tools:**

- **C++ Applications**: Use `gdb` for tracing and debugging
- **Python Applications**: Use `pdb` for interactive debugging
- **Application Profiling**: Various profiling tools discussed in SDK guide
- **Code Coverage**: Tools for inspecting test coverage

> **Note**: HoloHub doesn't provide a single debugging container due to the variety of methods across applications. Open an issue if you need additional debugging tools.

> **Note**: Refer to [Holoscan SDK debugging documentation](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_debugging.html) and [HoloHub debugging tutorials](https://github.com/nvidia-holoscan/holohub/tree/main/tutorials/debugging)
### Performance

Low latency is crucial for many HoloHub applications. Use these resources for performance analysis:

#### Performance Analysis Tools

- **HoloHub Benchmarks**: Projects in [`benchmarks/`](./benchmarks/) folder
- **Flow Tracking**: [`holoscan_flow_benchmarking/`](./benchmarks/holoscan_flow_benchmarking/) for data flow analysis
- **SDK Profiling**: General profiling tools in the [Holoscan SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_debugging.html)

#### Performance Best Practices

- Include performance insights in your application README
- Document any performance considerations or optimizations
- Share benchmarking results when relevant for community benefit
- Consider latency implications in your design decisions

## Getting Help

### Troubleshooting

**Common Issues:**

1. **Build Failures**
   - Verify all dependencies are installed
   - Check CMakeLists.txt configuration
   - Ensure correct build flags are used

2. **Linting Errors**
   - Run `./holohub lint` locally before submitting
   - Use automated fix commands when available
   - Check code formatting against standards

3. **Test Failures**
   - Verify test environment setup
   - Check for missing test data or dependencies
   - Review test output for specific error messages

4. **PR Review Issues**
   - Address all reviewer feedback promptly
   - Ensure all CI/CD checks pass
   - Update documentation if requested

**Getting Additional Help:**

- Check existing [GitHub Issues](https://github.com/nvidia-holoscan/holohub/issues)
- Review similar contributions for reference
- Ask questions in your PR comments for specific guidance and call `@nvidia-holoscan/holohub`.

### Reporting Issues

Found a bug or need a feature? Please open a [HoloHub Issue](https://github.com/nvidia-holoscan/holohub/issues).

**When reporting issues, include:**

- Clear description of the problem or enhancement request
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Environment details (OS, SDK version, etc.)
- Relevant logs or error messages

**For enhancement requests:**

- Describe the use case and benefits
- Propose potential implementation approach
- Consider if it fits HoloHub's scope and goals

### Advanced Developer Guide

Please refer to the [HoloHub Developer Reference](./doc/developer.md) for more advanced developer guidance.

---

Thank you for contributing to HoloHub! Your contributions help build a stronger ecosystem for the Holoscan community. 🚀
