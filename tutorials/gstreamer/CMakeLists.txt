cmake_minimum_required(VERSION 3.16)
project(gstreamer-tutorial CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Find GStreamer
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)

# Include directories
include_directories(${GSTREAMER_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../operators/gstreamer/core)

# Add compiler definitions
add_definitions(${GSTREAMER_CFLAGS_OTHER})

# Create executables
add_executable(basic-tutorial-1 basic-tutorial-1.cpp)
add_executable(basic-tutorial-2 basic-tutorial-2.cpp)
add_executable(basic-tutorial-3 basic-tutorial-3.cpp)
add_executable(playback-tutorial-1 playback-tutorial-1.cpp)

# Link libraries
target_link_libraries(basic-tutorial-1 ${GSTREAMER_LIBRARIES})
target_link_libraries(basic-tutorial-2 ${GSTREAMER_LIBRARIES})
target_link_libraries(basic-tutorial-3 ${GSTREAMER_LIBRARIES})
target_link_libraries(playback-tutorial-1 ${GSTREAMER_LIBRARIES})

# Set library directories
link_directories(${GSTREAMER_LIBRARY_DIRS})

# Set compile flags
target_compile_options(basic-tutorial-1 PRIVATE ${GSTREAMER_CFLAGS_OTHER})
target_compile_options(basic-tutorial-2 PRIVATE ${GSTREAMER_CFLAGS_OTHER})
target_compile_options(basic-tutorial-3 PRIVATE ${GSTREAMER_CFLAGS_OTHER})
target_compile_options(playback-tutorial-1 PRIVATE ${GSTREAMER_CFLAGS_OTHER})

# Platform-specific settings
if(APPLE)
    # For macOS, we might need additional frameworks
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COCOA_FRAMEWORK Cocoa)
    if(FOUNDATION_FRAMEWORK AND COCOA_FRAMEWORK)
        target_link_libraries(basic-tutorial-1 ${FOUNDATION_FRAMEWORK} ${COCOA_FRAMEWORK})
        target_link_libraries(basic-tutorial-2 ${FOUNDATION_FRAMEWORK} ${COCOA_FRAMEWORK})
        target_link_libraries(basic-tutorial-3 ${FOUNDATION_FRAMEWORK} ${COCOA_FRAMEWORK})
        target_link_libraries(playback-tutorial-1 ${FOUNDATION_FRAMEWORK} ${COCOA_FRAMEWORK})
    endif()
endif()

# Install targets (optional)
install(TARGETS basic-tutorial-1 basic-tutorial-2 basic-tutorial-3 playback-tutorial-1 DESTINATION bin)

# Print some information
message(STATUS "GStreamer include dirs: ${GSTREAMER_INCLUDE_DIRS}")
message(STATUS "GStreamer libraries: ${GSTREAMER_LIBRARIES}")
message(STATUS "GStreamer library dirs: ${GSTREAMER_LIBRARY_DIRS}")
