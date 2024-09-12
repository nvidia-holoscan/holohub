# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(ehr_query_llm/domain_specific/fhir CXX)

find_package(holoscan 2.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Install application and dependencies into the install/ directory for packaging
install(
  FILES __main__.py
        fhir_client.py
        metadata.json
        README.md
        requirements.txt
        run_cmd_template.sh
        test_fhir_client.py
  DESTINATION fhir
  COMPONENT "holohub-apps"
)

install(
  DIRECTORY ${CMAKE_HOME_DIRECTORY}/operators/ehr_query_llm
  DESTINATION fhir/operators)
