#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Error out if a command fails or a variable is not defined
set -eu

# Get the directory containing this script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Add the path to the generated flatbuffers files to the PYTHONPATH
FLATBUFFERS_PATH=${SCRIPT_DIR}/../../../build/pipeline_visualization/applications/pipeline_visualization/flatbuffers/
export PYTHONPATH=${PYTHONPATH:-""}:$FLATBUFFERS_PATH

python3 ${SCRIPT_DIR}/visualizer_${1:-dynamic}.py
