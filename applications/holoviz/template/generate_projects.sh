#!/bin/bash
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

set -x

PATH=$PATH:$HOME/.local/bin

if ! command -v cookiecutter &> /dev/null; then
    python3 -m pip install cookiecutter
fi
if ! command -v clang-format &> /dev/null; then
    python3 -m pip install clang-format
fi

generate() {
    local project_dir="$1"
    local example="$2"
    local project_name="$3"

    rm -rf ../${project_dir}
    cookiecutter --no-input cookiecutter-holoviz "example=${example}" "project_name=${project_name}" ${@:4} -o ..
    clang-format -i ../${project_dir}/*.cpp
}

generate "holoviz_hdr" "HDR" "Holoviz HDR" "tags=,\"BT.2020\",\"ST.2084\",\"EOTF\"" "holoscan_version=2.5"
generate "holoviz_srgb" "sRGB" "Holoviz sRGB"
generate "holoviz_vsync" "vsync" "Holoviz vsync"
generate "holoviz_yuv" "YUV" "Holoviz YUV" "tags=,\"YCbCr\"" "holoscan_version=2.4"
