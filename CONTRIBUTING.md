# Contributing to HoloHub

## Table of Contents
- [Introduction](#introduction)
- [Preparing your submission](#preparing-your-submission)
- [Submitting a new application/operator](#submitting-a-new-application-and-operator)
- [Reporting issues](#reporting-issues)
- [Fixing issues](#fixing-issues)
- [Testing](#testing)

## Introduction
HoloHub is a collection of applications and reusable operators that can be shared across engineering teams.
While formal SQA is not required to submit to HoloHub, it is recommended to follow the guidelines
in this document to make sure new submission can be easily used by others.

Each submission (application and/or operator) requires a *metadata.json* file which describes its
specifications and requirements.

**Every operator should have at least one associated application to demonstrate the capabilities of the operator.**

## Preparing your submission

### Naming convention
Every application and operator should be named with an english descriptive name of the functionality
provided. Please avoid using acronyms, brand, or team names.

### Metadata description
Every application and operator should have an associated *metadata.json* file which describes the features
and dependencies.

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
		"platforms": ["amd64", "arm64"],
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

### Ranking levels
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

### Readme file
While it is not required, adding a README.md file with clarification on the intent and usage of the application or operator is a plus and helps developers and users get started quickly with your application.

### Build System
#### Adding an Operator or GXF extension
Each operator should be added in its own directory under the ```operators``` or ```gxf_extensions``` directories and should
contain a ```metadata.json``` file as well as a README file.

Edit the ```CMakeLists.txt``` file to add the new operator as part of the build system using the ```add_holohub_operator```
CMake function. If the operator wraps a GXF extension then the optional ```DEPENDS EXTENSIONS``` should be added to tell the build
system to build the dependent extension(s).

```cmake
add_holohub_operator(my_operator DEPENDS EXTENSIONS my_extension)
```

Note that extensions do not have a ```DEPENDS``` option.

#### Adding an application
Each application should be added in its own directory under the ```applications``` directory and should
contain a ```metadata.json``` file as well as a README file.

Edit the ```CMakeLists.txt``` file to add the new application  as part of the build system using the ```add_holohub_application```
CMake function. If the application relies on one or more operators then the optional ```DEPENDS OPERATORS``` should be added so that
the build system knows to build the dependent operator(s).

```cmake
add_holohub_application(my_application DEPENDS
                        OPERATORS my_operator1
                                  my_operator2
                        )
```

Note that some applications have the optional ```HOLOSCAN_SAMPLE_APP``` keywords at the end of the ```add_holohub_application```
function. This keyword should only be used for sample applications that are maintained by the Holoscan team.

## Submitting a new application and operator

### Coding Guidelines

- All source code contributions must strictly adhere to the Holoscan SDK coding style.

- Make sure that you can contribute your work to open source (no license and/or patent conflict is introduced by your code). You will need to [`sign`](#signing-your-work) your commit.

- Thanks in advance for your patience as we review your contributions; we do appreciate them!

### Developer Workflow

1. Developers must first [fork](https://help.github.com/en/articles/fork-a-repo) the [upstream](https://github.com/nvidia-holoscan/holohub) HoloHub repository.

2. Git clone the forked repository and push changes to the personal fork.

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_FORK.git HoloHub
# Checkout the targeted branch and commit changes
# Push the commits to a branch on the fork (remote).
git push -u origin <local-branch>:<remote-branch>
```

3. Once the code changes are staged on the fork and ready for review, a [Pull Request](https://help.github.com/en/articles/about-pull-requests) (PR) can be [requested](https://help.github.com/en/articles/creating-a-pull-request) to merge the changes from a branch of the fork into a selected branch of upstream.
  * Exercise caution when selecting the source and target branches for the PR.
  * Creation of a PR creation kicks off the code review process.
  * While under review, mark your PRs as work-in-progress by prefixing the PR title with [WIP].

4. Upon review the PR will be accepted only if it meets the standards for HoloHub.


### Signing Your Contribution

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

* Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1
    
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129
    
    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1
    
    By making a contribution to this project, I certify that:
    
    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
    
    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
    
    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
    
    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

## Linting

The code submitted to Holohub needs to pass linting.

To install the necessary linting tools, run:

```sh
./run install_lint_deps
```

The following command can then be used to run various linting tools on the repository.
You may optionally pass a path argument to limit the linting to a specific subdirectory.

```sh
./run lint [path]
```


## Reporting issues

 All enhancement, bugfix, or change requests must begin with the creation of a [HoloHub Issue Request](https://github.com/nvidia-holoscan/holohub/issues).

## Fixing issues

Patches to existing applications and operators are welcome and should follow the same workflow as
[submitting new contributions](#Submitting-a-new-application-and-operator). Make sure you assign the original author of the contribution as the reviewer.

## Testing

### Writing tests
Ideally applications should have a testing section in their CMakeLists.txt allowing to run the application for functional testing.
HoloHub uses [CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html) to drive the automated testing.

### Running tests
To run the suite of HoloHub tests, run CMake from the top of the HoloHub directory and compile the binary tree. Once the compilation
succeeds you can run all the tests using the following command from the top of the binary tree:
```sh
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
