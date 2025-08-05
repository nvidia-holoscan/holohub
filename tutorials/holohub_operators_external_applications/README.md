# Using Holohub Operators in External Applications

This tutorial demonstrates how to import and use Holohub operators in your own external applications. You'll learn how to fetch specific operators from the Holohub repository and integrate them into your Holoscan-based applications.

## Overview

Holohub provides a collection of pre-built operators that you can easily integrate into your applications. This tutorial shows you how to:

1. Set up a CMake project that uses Holohub operators
2. Fetch specific operators using the `FetchHolohubOperator.cmake` utility
3. Link against the required Holohub libraries
4. Use the operators in your application code

## Prerequisites

- CMake 3.18 or higher
- Holoscan SDK installed
- Git
- C++ compiler (GCC, Clang, or MSVC)

## Project Structure

```
your_external_app/
├── CMakeLists.txt
├── main.cpp
└── build/
```

## Step-by-Step Guide

### 1. Create Your CMakeLists.txt

Create a `CMakeLists.txt` file in your project root. The file includes two different approaches for including the `FetchHolohubOperator.cmake` utility (plus an internal-only option):

```cmake
cmake_minimum_required(VERSION 3.18)
project(your_app_name)

# Find the Holoscan package
find_package(holoscan REQUIRED)

# =============================================================================
# FETCH HOLOHUB OPERATOR UTILITY - TWO APPROACHES AVAILABLE
# =============================================================================
# Choose one of the following two approaches by uncommenting the desired option:

# INTERNAL ONLY: Repository Include (for Holohub repository internal use only)
# This approach is only available when the application is within the Holohub repository structure
# For external applications, use one of the two approaches below
include(../../cmake/FetchHolohubOperator.cmake)

# APPROACH 1: Download from GitHub (for external applications with internet access)
# Pros: Independent of repository structure, automatic updates, no manual copying
# Cons: Requires internet connection, depends on GitHub availability
# Uncomment the following lines to use this approach (and comment out the internal include above):
# set(FETCH_HOLOHUB_OPERATOR_URL "https://raw.githubusercontent.com/nvidia-holoscan/holohub/refs/heads/main/cmake/FetchHolohubOperator.cmake")
# set(FETCH_HOLOHUB_OPERATOR_LOCAL_PATH "${CMAKE_CURRENT_BINARY_DIR}/FetchHolohubOperator.cmake")
# if(NOT EXISTS ${FETCH_HOLOHUB_OPERATOR_LOCAL_PATH})
#      file(DOWNLOAD
#          ${FETCH_HOLOHUB_OPERATOR_URL}
#          ${FETCH_HOLOHUB_OPERATOR_LOCAL_PATH}
#          SHOW_PROGRESS
#          TLS_VERIFY ON
#      )
#      if(NOT EXISTS ${FETCH_HOLOHUB_OPERATOR_LOCAL_PATH})
#          message(FATAL_ERROR "Failed to download FetchHolohubOperator.cmake from ${FETCH_HOLOHUB_OPERATOR_URL}")
#      endif()
# endif()
# include(${FETCH_HOLOHUB_OPERATOR_LOCAL_PATH})

# APPROACH 2: Local Copy (for offline environments or version control)
# Pros: Complete independence, no internet dependency, full version control
# Cons: Requires manual file copying, need to manually update
# Uncomment the following line to use this approach (requires FetchHolohubOperator.cmake in same directory):
# include(${CMAKE_CURRENT_SOURCE_DIR}/FetchHolohubOperator.cmake)

# Fetch the specific operator you need
fetch_holohub_operator(aja_source)

# Add your executable
add_executable(${PROJECT_NAME} main.cpp)

# Link against Holohub libraries
target_link_libraries(${PROJECT_NAME} 
   PRIVATE 
   holoscan::core
   holoscan::aja
   )
```

### 2. Understanding the CMakeLists.txt

Let's break down each section:

#### Project Setup
```cmake
cmake_minimum_required(VERSION 3.18)
project(your_app_name)
```

- Sets the minimum CMake version required
- Defines your project name

#### Holoscan Integration
```cmake
find_package(holoscan REQUIRED)
```

- Locates and configures the Holoscan SDK
- Makes Holoscan targets available for linking

#### Operator Fetching - Two Approaches Available

The CMakeLists.txt provides two different approaches for including the `FetchHolohubOperator.cmake` utility:

**Internal Only: Repository Include**
```cmake
include(../../cmake/FetchHolohubOperator.cmake)
```

- ✅ Simple and straightforward
- ✅ Always uses the latest version from the repository
- ✅ No additional files needed
- ❌ **INTERNAL USE ONLY** - Requires the application to be within the Holohub repository structure
- ❌ Not suitable for external applications

**Approach 1: Download from GitHub**
```cmake
 set(FETCH_HOLOHUB_OPERATOR_URL "https://raw.githubusercontent.com/nvidia-holoscan/holohub/refs/heads/main/cmake/FetchHolohubOperator.cmake")
 set(FETCH_HOLOHUB_OPERATOR_LOCAL_PATH "${CMAKE_CURRENT_BINARY_DIR}/FetchHolohubOperator.cmake")
 if(NOT EXISTS ${FETCH_HOLOHUB_OPERATOR_LOCAL_PATH})
      file(DOWNLOAD
          ${FETCH_HOLOHUB_OPERATOR_URL}
          ${FETCH_HOLOHUB_OPERATOR_LOCAL_PATH}
          SHOW_PROGRESS
          TLS_VERIFY ON
      )
      if(NOT EXISTS ${FETCH_HOLOHUB_OPERATOR_LOCAL_PATH})
          message(FATAL_ERROR "Failed to download FetchHolohubOperator.cmake from ${FETCH_HOLOHUB_OPERATOR_URL}")
      endif()
 endif()
 include(${FETCH_HOLOHUB_OPERATOR_LOCAL_PATH})
```

- ✅ Completely independent of repository structure
- ✅ Works for truly external applications
- ✅ Always gets the latest version from the main branch
- ❌ Requires internet connection during build
- ❌ Depends on GitHub availability

**Approach 2: Local Copy**
```cmake
 include(${CMAKE_CURRENT_SOURCE_DIR}/FetchHolohubOperator.cmake)
```

- ✅ Complete independence from repository structure
- ✅ No internet dependency during build
- ✅ Full version control over the utility file
- ❌ Requires manual file copying
- ❌ Need to manually update when new versions are released

**Usage:**
```cmake
fetch_holohub_operator(aja_source)
```

- Downloads the `aja_source` operator from Holohub using sparse checkout

#### Application Building
```cmake
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} 
   PRIVATE 
   holoscan::core
   holoscan::aja
   )
```

- Creates your executable from `main.cpp`
- Links against the required Holohub libraries

### 3. The FetchHolohubOperator.cmake Utility

The `FetchHolohubOperator.cmake` file provides a convenient way to fetch specific operators from the Holohub repository. It uses Git sparse checkout to download only the required operator, making the process efficient.

#### Function Signature
```cmake
fetch_holohub_operator(OPERATOR_NAME [PATH path] [REPO_URL url] [BRANCH branch])
```

#### Parameters
- `OPERATOR_NAME`: The name of the operator to fetch
- `PATH` (optional): The path to the operator within the Holohub repository (defaults to OPERATOR_NAME)
- `REPO_URL` (optional): The URL of the Holohub repository (defaults to the official Holohub repo)
- `BRANCH` (optional): The branch to checkout (defaults to "main")

#### Examples
```cmake
# Fetch the aja_source operator
fetch_holohub_operator(aja_source)

# Fetch an operator with a custom path
fetch_holohub_operator(dds_operator_base PATH dds/base)

# Fetch from a custom repository
fetch_holohub_operator(custom_operator REPO_URL "https://github.com/custom/holohub.git")

# Fetch from a specific branch
fetch_holohub_operator(custom_operator BRANCH "dev")
```

### 4. Choosing the Right Approach

When deciding which approach to use for including `FetchHolohubOperator.cmake`, consider your specific requirements:

**Use Repository Include (Internal Only) when:**

- Your application is part of the Holohub repository
- You want to always use the latest version
- You're developing within the Holohub ecosystem
- **Note:** This approach is only available for internal Holohub repository use

**Use Approach 1 (Download from GitHub) when:**

- Your application is external to the Holohub repository
- You have reliable internet connectivity during builds
- You want automatic updates from the main branch
- You need independence from the repository structure

**Use Approach 2 (Local Copy) when:**

- Your application needs to work offline
- You require version stability and reproducibility
- You need complete control over the utility file
- You're building for production environments

### 5. Create Your Application Code

Create a `main.cpp` file that uses the fetched operator:

```cpp
#include "holoscan/holoscan.hpp"
#include "aja_source.hpp"

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Create an instance of the AJA source operator
    auto aja_source = make_operator<ops::AJASourceOp>("aja");
    
    // Add the operator to your application
    add_operator(aja_source);
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();
  app->run();

  return 0;
}
```

### 6. Building Your Application

```bash
# Create a build directory
mkdir build && cd build

# Configure the project
cmake ..

# Build the project
make -j$(nproc)
```

## Available Operators

Holohub provides many operators that you can fetch and use. Some popular ones include:

- `aja_source` - AJA video capture
- `aja_sink` - AJA video output
- `realsense_camera` - Intel RealSense camera
- `dds_operator_base` - DDS communication
- `tensor_rt_inference` - TensorRT inference
- `format_converter` - Format conversion utilities

To find more operators, check the [Holohub operators directory](https://github.com/nvidia-holoscan/holohub/tree/main/operators).

## Advanced Usage

### Fetching Multiple Operators

You can fetch multiple operators in the same project:

```cmake
# Fetch multiple operators
fetch_holohub_operator(aja_source)
fetch_holohub_operator(format_converter)
fetch_holohub_operator(tensor_rt_inference)

# Link against all required libraries
target_link_libraries(${PROJECT_NAME} 
   PRIVATE 
   holoscan::core
   holoscan::aja
   holoscan::format_converter
   holoscan::tensor_rt_inference
   )
```

### Custom Operator Paths

If an operator is located in a subdirectory within the Holohub repository:

```cmake
fetch_holohub_operator(dds_operator_base PATH dds/base)
```

### Using Different Branches

To use operators from a specific branch:

```cmake
fetch_holohub_operator(experimental_operator BRANCH "experimental")
```

## Troubleshooting

### Common Issues

1. **CMake can't find Holoscan**
- Ensure Holoscan SDK is properly installed
- Set `CMAKE_PREFIX_PATH` to point to your Holoscan installation

2. **Operator not found**
- Verify the operator name exists in the Holohub repository
- Check the correct path if the operator is in a subdirectory

3. **Linking errors**
- Ensure you're linking against the correct Holohub libraries
- Check that the operator dependencies are satisfied

### Debug Information

To see what's being fetched, you can enable CMake verbose output:

```bash
cmake -DCMAKE_VERBOSE_MAKEFILE=ON ..
```

## Best Practices

1. **Version Pinning**: Consider using specific branches or tags for production applications
2. **Dependency Management**: Only fetch the operators you actually need
3. **Error Handling**: Always check if the `find_package(holoscan REQUIRED)` succeeds
4. **Documentation**: Document which operators your application depends on

## Example Complete Project

See the `main.cpp` and `CMakeLists.txt` files in this directory for a complete working example that demonstrates how to use the AJA source operator from Holohub.

## Additional Resources

- [Holohub Repository](https://github.com/nvidia-holoscan/holohub)
- [Holoscan Documentation](https://docs.nvidia.com/holoscan/)
- [Holohub Operators Documentation](https://github.com/nvidia-holoscan/holohub/tree/main/operators)

## License

This tutorial is part of Holohub and is licensed under the Apache 2.0 License. 