#
# Place the license header here
#

cmake_minimum_required(VERSION 3.20)

project({{ cookiecutter.project_slug }})

find_package(holoscan {{ cookiecutter.holoscan_version }} REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

{% if cookiecutter.language == "cpp" %}
add_executable({{ cookiecutter.project_slug }}
    src/main.cpp
)

target_link_libraries({{ cookiecutter.project_slug }}
    PRIVATE
    holoscan::core
)

if(BUILD_TESTING)
  add_test({{ cookiecutter.project_slug }}-test
    COMMAND echo "Add test coverage for {{ cookiecutter.project_slug }} here"
  )
endif()
{% endif %}
