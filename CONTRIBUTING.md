# Contributing to HoloHub

## Table of Contents

- [Introduction](#introduction)
- [Types of Contributions](#types-of-contributions)
- [Developer Process](#developer-process)
- [Preparing your submission](#preparing-your-submission)
- [Unit Testing Python Operators](#unit-testing-python-operators)
- [Reporting issues](#reporting-issues)

## Introduction

Welcome to HoloHub! Please read our [README](./README.md) document for an overview of the project.

HoloHub is a collection of applications and reusable operators available to the NVIDIA Holoscan developer community.
Polished contributions from community members like you help us augment the Holoscan open source ecosystem with new features
and demonstrations.

Please read this guide if you are interested in contributing open source code to HoloHub.

## Types of Contributions

Before getting started, assess how your idea or project may best benefit the Holoscan community.

If your idea is:

- _an end-to-end pipeline that address the entire use case:_ Consider submitting to HoloHub as a [workflow](./workflows/).
- _specific to a narrow practical application or use case:_ Consider submitting to HoloHub as an [application](./applications/).
- _widely applicable across a domain of interests:_ Consider submitting to HoloHub as an [operator](./operators/) and an accompanying [application](./applications/).
- _neither a new operator, application, nor workflow_: Consider submitting a [tutorial](./tutorials/) to HoloHub.

If your code is:

- _feature-complete and tested_: Submit a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) to contribute your work to HoloHub.
- _a work in progress:_ We recommend to [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) HoloHub and track your local development there, then submit to HoloHub when ready. Alternatively, open pull request and indicate that it is a "work-in-progress" with the prefix "WIP".
- _a patch for an existing application, workflow, or operator_: Submit a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) and request a review from the original author of the contribution you are patching.

We recommend referring to contributing guidelines for testing and styling goals throughout your development process.

## Developer Process

### Requirements

Review [HoloHub prerequisites](./README.md#prerequisites) before getting started.

We recommend that new developers review GitHub's [starting documentation](https://docs.github.com/en/get-started/start-your-journey) before making their first contribution.

### Process

1. Developers must first [fork](https://help.github.com/en/articles/fork-a-repo) the [upstream](https://github.com/nvidia-holoscan/holohub) HoloHub repository.

2. Git clone the forked repository and push changes to the personal fork.

    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_FORK.git HoloHub
    # Checkout the targeted branch and commit changes
    # Push the commits to a branch on the fork (remote).
    git push -u origin <local-branch>:<remote-branch>
    ```

3. Once the code changes are staged on the fork and ready for review, please [submit](https://help.github.com/en/articles/creating-a-pull-request) a [Pull Request](https://help.github.com/en/articles/about-pull-requests) (PR) to merge the changes from a branch of the fork into a selected branch of upstream.

    - Exercise caution when selecting the source and target branches for the PR.
    - Creation of a PR kicks off the [code review](#preparing-your-submission) process.

4. HoloHub maintainers will review the PR and accept the proposal if changes meet HoloHub standards.

Thanks in advance for your patience as we review your contributions. We do appreciate them!

## Preparing your submission

We request that members follow the guidelines in this document to make sure new submissions can be easily used by others.

A typical submission consists of:

- Application, workflow, operator, and/or tutorial code making use of the Holoscan SDK;
- A [`metadata.json`](#metadata-description) file;
- A [README](#readme-file) file describing the application, workflow, operator, and/or tutorial;

For a submission to be accepted into HoloHub it must meet at least these criteria:

- Clearly demonstrates added value to the Holoscan community;
- Receives approval from at least one HoloHub maintainer;
- [Code linting](#linting) tests pass;
- Any new [code tests](#testing) pass.

We do not require that community members conduct formal Software Quality Assurance (SQA) to submit to HoloHub.

### Metadata description

Every application and operator should have an associated _metadata.json_ file which describes the features
and dependencies.

`metadata.json` schemas differ slightly for [workflows](./workflows/metadata.schema.json), [applications](./applications/metadata.schema.json), [GXF extensions](./gxf_extensions/metadata.schema.json), [operators](./operators/metadata.schema.json), and [tutorials](./tutorials/metadata.schema.json), but generally follow the convention below:

```json
// Main json definition for application or operator
"application|operator": {
    // Explicit name of the contribution
    "name": "explicit name of the application/operator",
    // Author(s) of the contribution
    "authors": [
        {
            "name": "firstname lastname",
            "affiliation": "affiliation"
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
    // Holoscan SDK
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

### Ranking Levels for `metadata.json`

Please provide a self-assessment of your HoloHub contribution in your `metadata.json` file(s) according to the levels below:

#### Level 0 - In par with Main SDK modules

- Widespread community dependence
- Above 90% code coverage
- Nightly dashboards and testing monitored rigorously
- All requirements below

#### Level 1 - Very high-quality code

- Meets all Holoscan SDK code style standards
- No external requirements beyond those needed by Holoscan SDK proper
- Builds and passes tests on all supported platforms within 1 month of each core tagged release
- Active developer community dedicated to maintaining code-base
- 75% code coverage demonstrated for testing suite
- Continuous integration testing performed
- All requirements below

#### Level 2 - Quality code

- Compiles on niche community platforms
- May depend on specific external tools or specific external libraries
- Tests passing on all supported platforms
- All requirements below

#### Level 3 - Features under development

- Code build on specific platforms/configuration
- Some tests are passing on supported platforms

#### Level 4 - Code of unknown quality

- Code builds on specific platforms/configuration
- Minimal set of test exists

#### Level 5 - Deprecated

- Deprecated code, known to be of limited utility, perhaps has known bugs

### README File

Adding a `README.md` file with clarification on the intent and usage of the application or operator helps developers and users get started quickly with your contribution.

We recommend writing README files in the Markdown format (`.md`).

Please use the [terms defined in the glossary](README.md#Glossary) to refer to specific location of files for HoloHub.

### Adding an Operator or GXF Extension

Add each operator or extension in its own directory under the [```operators```](./operators/) or [```gxf_extensions```](./gxf_extensions) directory. The subdirectory should contain:

- A _metadata.json_ file which describes its specifications and requirements in accordance with the [operator metadata.json schema](./operators/metadata.schema.json).
- A README file summarizing the operator's purpose.
- A separate unit test file (for Python operators).

Additionally, each operator should have at least one associated [application](./applications/) to demonstrate the capabilities of the operator.

Edit the [```CMakeLists.txt```](./operators/CMakeLists.txt) to add the new operator as part of the build system using the ```add_holohub_operator```
CMake function, passing the new operator folder name as the first argument. If the operator wraps a GXF extension then the optional ```DEPENDS EXTENSIONS``` should be added to tell the build
system to build the dependent extension(s).

```cmake
add_holohub_operator(my_operator DEPENDS EXTENSIONS my_extension)
```

Note that extensions do not have a ```DEPENDS``` option.

Refer to the [HoloHub operator template folder](./operators/template/) for stub `metadata.json` and `README` files to copy
and update for your new operator.

You can then build that operator by configuring your build with `-D OP_my_operator:BOOL=1` (or `-D EXT_my_extension:BOOL=1` for GXF extensions):

```bash
cmake -S . -B ./build -D OP_my_operator:BOOL=1
cmake --build ./build -j
```

#### Operator Naming Convention

For an operator named "Adaptive Thresholding", follow these naming conventions:

1. **Class Name**: `AdaptiveThresholdingOp` (title case without spaces and with "Op" at the end)
2. **metadata.json ("name")**:  Same as class name.
3. **Directory**: `adaptive_thresholding` (snake case)
4. **Filename**: same as directory name with extension.
5. **README (Title)**: "`Adaptive Thresholding Operator`" (title case with spaces and Operator at the end).
6. **Unit testing**: `test_adaptive_thresholding.py` (for Python operators)

### Adding an Application

Add each application in its own subdirectory under the [`applications`](./applications/) directory. The subdirectory should contain:

- A _metadata.json_ file which describes its specifications and requirements in accordance with the [application metadata.json schema](./applications/metadata.schema.json).
- A README file summarizing the application's purpose and architecture;

Edit [```CMakeLists.txt```](./applications/CMakeLists.txt) to add the new application as part of the build system using the ```add_holohub_application```
CMake function, passing the new application folder name as the first argument. If the application relies on one or more operators then the optional ```DEPENDS OPERATORS``` should be added so that
the build system knows to build the dependent operator(s).

```cmake
add_holohub_application(my_application DEPENDS
                        OPERATORS my_operator1
                                  my_operator2
                        )
```

Refer to the [HoloHub application template folder](./applications/template/) for stub `metadata.json` and `README` files to copy
and update for your new application.

You can then build that application by configuring your build with `-D APP_my_application:BOOL=1` :

```bash
cmake -S . -B ./build -D APP_my_application:BOOL=1
cmake --build ./build -j
```

### Adding a Workflow

> NOTE: Workflows are _end-to-end_ reference applications that demonstrate complete pipelines from sensors to results. To determine if your contribution should be a workflow rather than an application, consider whether it represents a complete "from sensor to insight" pipeline. Workflows typically integrate multiple components (operators, models, etc.) to solve entire use case, while applications may focus on demonstrating specific functionality or techniques.

Add each workflow in its own subdirectory under the [`workflows`](./workflows/) directory. The subdirectory should contain:

- A _metadata.json_ file which describes its specifications and requirements in accordance with the [workflow metadata.json schema](./workflows/metadata.schema.json).
- A README file summarizing the workflow's purpose and architecture;

Workflows should follow the organization conventions described in the [workflows README](./workflows/README.md), which includes:

- Required conventions:
  - Each workflow must provide a `metadata.json` file
  - Each workflow must provide a `README.md` file
  - Each workflow must be organized in its own subfolder under `holohub/workflows/`

- Recommended conventions:
  - Language-specific code should be organized in language subdirectories (`cpp/`, `python/`)
  - Container environment can be customized or use the default HoloHub environment
  - Build and run instructions should be provided in the `metadata.json` file

If your workflow does not adhere to these conventions, set the field `manual_setup` to `true` in your project `metadata.json` file.

Refer to the workflows directory for examples of existing workflows that you can use as templates for your submission.

You can run your workflow using the `./run launch` command:

```bash
./run launch workflows/my_workflow
```

### Adding a Package Configuration

1. Ensure the applications/operator CMake targets and/or files you want to package have an `install` rule define in your application/operator CMakeLists.txt files (see [CMake docs](https://cmake.org/cmake/help/latest/command/install.html)). Optionally, pass a `COMPONENT` argument to the `install` rule to control precisely which of your targets/files will get packaged. This can be useful if - for example - you want to create dev vs run packages, not include python bindings, or split your libraries in separate packages for different backends.
2. Create a folder for your package (or group of packages) under the [`pkg`](./pkg/) directory.
3. Create a `CMakeLists.txt` file in that new folder with the following contents:

   ```cmake
   holohub_configure_deb(
     NAME "my-package-dev"
     COMPONENTS "my-headers" "my-libs" # (optional) list of installation components to package, Default: all targets/files configured with an install rule will be packaged.
     DESCRIPTION "My project (dev)"
     VERSION "X.Y.Z"
     VENDOR "My org name"
     CONTACT "John Doe <john@doe.org>"
     DEPENDS "libc6 (>= 2.34), libstdc++6 (>= 11)" # list of your package debian dependencies - https://www.debian.org/doc/debian-policy/ch-relationships.html
     SECTION "devel" # (optional)
     PRIORITY "optional" # (optional)
   )
   ```

4. Update the `CMakeLists.txt` file under [`pkg`](./pkg/CMakeLists.txt) to include this new package directory and define the app and operators to build and package using the `add_holohub_package` CMake function:

   ```bash
   add_holohub_package(my_packager
                       APPLICATIONS my_app1
                       OPERATORS my_op1 my_op2)
   ```

5. You can then generate the package(s) by configuring your build with `-D PKG_my_packager:BOOL=1` :

    ```bash
    cmake -S . -B ./build -D PKG_my_packager:BOOL=1 # NOTE: avoid adding any other -D PKG_, -D APP_ or -D OP_ as their built targets would currently pollute the content of your package, unless you used COMPONENTS in holohub_configure_deb above
    cmake --build ./build -j
    cpack --config ./build/pkg/CPackConfig-*.cmake
    ```

### Adding a Tutorial

Add each tutorial in its own subdirectory under the [```tutorials```](./tutorials) directory. The subdirectory should contain:

- A README file summarizing the application's purpose and architecture;
- A _metadata.json_ file which describes its specifications and requirements in accordance with the [tutorial metadata.json schema](./tutorials/metadata.schema.json) (optional);

There are no project-wide build requirements for tutorials.

### License Guidelines

- Make sure that you can contribute your work to open source.  Verify that no license and/or patent conflict is introduced by your code. NVIDIA is not responsible for conflicts resulting from community contributions.
- Holohub license is the Apache 2.0 permissive open source license, so any contributions will inherit this license.
- We require that members [sign](#signing-your-contribution) their contributions to certify their work.

### Coding Guidelines

- All source code contributions must strictly adhere to the Holoscan SDK coding style.

- Every application, workflow, and operator should be named with an english descriptive name of the functionality
provided. Please avoid using acronyms, brand, or team names.

### Signing Your Contribution

- We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

- Any contribution which contains commits that are not Signed-Off will not be accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```txt
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

  ```txt
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```txt
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

## Linting

The code submitted to HoloHub needs to pass linting checks to demonstrate compliance with minimum style guidelines.

HoloHub runs linting checks in CI/CD pipelines when new changes are proposed. You will see a linting check result when you create a new pull request. You can also install and run linting tools to aid in local development.

### Running lint

To install the necessary linting tools, run:

```bash
./run install_lint_deps
```

The following command can then be used to run various linting tools on the repository.
You may optionally pass a path argument to limit the linting to a specific subdirectory.

```bash
./run lint [path]
```

### Fixing lint issues

```bash
# To fix python ruff issues which can be automatically fixed, run:
ruff --fix --ignore E712 [path]
# To fix python isort issues, run:
isort [path]
# To fix python black code formatting issues, run:
black [path]
# To fix C++ lint issues, run:
clang-format --style=file --sort-includes=0 --lines=20:10000 -i <filename>
# To fix codespell issues, run:
codespell -w -i 3 [path]
```

## Testing

### Writing tests

Ideally applications should have a testing section in their CMakeLists.txt allowing to run the application for functional testing.
HoloHub uses [CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html) to drive the automated testing.

### Running tests

To run the suite of HoloHub tests, run CMake from the top of the HoloHub directory and compile the binary tree. Once the compilation
succeeds you can run all the tests using the following command from the top of the binary tree:

```bash
cd <holoscan_binary_directory>
# To run all the tests
ctest
# To run a specific test
ctest -R MyTest
# To run with verbose mode
ctest -V
# To run with extra verbose mode
ctest -VV
```

### Unit Testing Python Operators

HoloHub encourages contributors to provide unit tests for Python operators. Please follow these conventions and guidelines:

- **Test Framework**: Use `pytest` for writing tests.
- **Test File Location**: Place test files in the same directory as the operator, naming them as `test_<operator>.py`.
- **Fixtures**: Reuse common fixtures defined in `conftest.py` (e.g., `fragment`, `mock_image`, `op_input_factory`, `op_output`, `execution_context`).
- **Test Structure**:
  - Test operator initialization and properties.
  - Test setup of input/output ports.
  - Test error handling for invalid arguments using `pytest.raises`.
  - Test the operator’s compute logic, including parameterized tests for different input shapes and values.
  - Use mock inputs and outputs to isolate operator logic.

**Example**:

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

**Running Python Unit Tests**: From the repository root or operator directory, run:

```bash
pytest operators/<your_operator_dir>/
```

**Best Practices**:

- Cover both positive and negative cases.
- Use descriptive assertion messages.
- Keep tests isolated and independent.
- Add any generic/reusable fixtures in `conftest.py`.

For more examples, see existing operator test files such as `operators/deidentification/pixelator/test_pixelator.py` and `conftest.py` in the repository.

## Debugging

We recommend exploring the following tools below for debugging your application:

- The [Holoscan SDK User Guide Debugging section](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_debugging.html) discusses several common debugging scenarios. Review the user guide for help on investigating application crashes and segfaults, application profiling, inspecting code coverage, and tracing with `gdb` for C++ applications or `pdb` for Python applications.
- The [Holoscan SDK User Guide Logging section](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_logging.html) describes how to set up for runtime logging from Holoscan SDK applications. Logging is a non-disruptive way to understand application runtime behavior.
- [Holoscan SDK]() provides a VSCode Dev Container for development, described in the [Debugging section](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_debugging.html).
- The HoloHub `dev_container` script provides several options that can be useful for debugging:
  - The `--as_root` option launches an application container as the root user, giving you expanded permissions to install and run debugging programs like `gdb`.
  - The `--local_sdk_root` option mounts a local SDK installation into your application container. You can build a local Holoscan SDK installation in Debug mode and then mount it into your container to inspect more complete debug information with `gdb`.

Note that there is no single debugging process nor VSCode Dev Container in HoloHub due to the variety of methods and libraries used across HoloHub applications. If you feel that tools or processes are missing, please open an issue on GitHub to let us know.

## Performance

Low latency is a key feature of many HoloHub applications. We recommend exploring the following tools to analyze and report application performance:

- Projects in the [`benchmarks`](/benchmarks/) folder are focused on HoloHub benchmarking. The [`holoscan_flow_tracking`](/benchmarks/holoscan_flow_benchmarking/) project in particular provides common tooling for Holoscan SDK data flow analysis.
- The [Holoscan SDK User Guide Debugging section](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_debugging.html) discusses general tools for application profiling.

We encourage contributors to provide performance insights in application README discussions for community knowledge.

## Reporting issues

Please open a [HoloHub Issue Request](https://github.com/nvidia-holoscan/holohub/issues) to request an enhancement, bug fix, or other change in HoloHub.
