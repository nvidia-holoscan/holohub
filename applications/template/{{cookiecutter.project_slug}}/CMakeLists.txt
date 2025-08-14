# SPDX-FileCopyrightText: Copyright (c) {{ cookiecutter.year }} {{ cookiecutter.author }}
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
