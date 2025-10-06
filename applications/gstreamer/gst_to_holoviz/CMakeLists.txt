cmake_minimum_required(VERSION 3.16)
project(gstreamer-applications C CXX)

# Set C and C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Find Holoscan
find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Find GStreamer
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_BASE REQUIRED gstreamer-base-1.0)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)

# Include directories
include_directories(${GSTREAMER_INCLUDE_DIRS})
include_directories(${GSTREAMER_BASE_INCLUDE_DIRS})
include_directories(${GSTREAMER_VIDEO_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../operators/gstreamer)

# Add the operators/gstreamer subdirectory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../operators/gstreamer ${CMAKE_CURRENT_BINARY_DIR}/operators_gstreamer)

# Add compiler definitions
add_definitions(${GSTREAMER_CFLAGS_OTHER})
add_definitions(${GSTREAMER_BASE_CFLAGS_OTHER})
add_definitions(${GSTREAMER_VIDEO_CFLAGS_OTHER})

# Create executable
add_executable(gst-to-holoviz gst_to_holoviz.cpp)

# Link libraries
target_link_libraries(gst-to-holoviz
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_BASE_LIBRARIES}
    ${GSTREAMER_VIDEO_LIBRARIES}
    holoscan_gstreamer_sink
    holoscan::core
    holoscan::ops::holoviz)

# Set library directories
link_directories(${GSTREAMER_LIBRARY_DIRS})
link_directories(${GSTREAMER_BASE_LIBRARY_DIRS})
link_directories(${GSTREAMER_VIDEO_LIBRARY_DIRS})

# Set compile flags
target_compile_options(gst-to-holoviz PRIVATE 
    ${GSTREAMER_CFLAGS_OTHER} 
    ${GSTREAMER_BASE_CFLAGS_OTHER} 
    ${GSTREAMER_VIDEO_CFLAGS_OTHER})

# Platform-specific settings
if(APPLE)
    # For macOS, we might need additional frameworks
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COCOA_FRAMEWORK Cocoa)
    if(FOUNDATION_FRAMEWORK AND COCOA_FRAMEWORK)
        target_link_libraries(gst-to-holoviz ${FOUNDATION_FRAMEWORK} ${COCOA_FRAMEWORK})
    endif()
endif()

# Install target
install(TARGETS gst-to-holoviz DESTINATION bin)

# Print some information
message(STATUS "GStreamer include dirs: ${GSTREAMER_INCLUDE_DIRS}")
message(STATUS "GStreamer libraries: ${GSTREAMER_LIBRARIES}")
message(STATUS "GStreamer library dirs: ${GSTREAMER_LIBRARY_DIRS}")

