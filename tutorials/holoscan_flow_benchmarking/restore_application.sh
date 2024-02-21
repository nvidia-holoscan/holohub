#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e

# Get the first argument as the directory to search
dir=$1
# find all the cpp and py files in the directory and the subdirectories
all_files=$(find "$dir" -type f \( -name "*.cpp" -o -name "*.py" \))

# Iterate over each found file
for file in $all_files; do
    # Find whether the file has "BenchmarkedApplication" in it and skip it if it does
    if grep -q "BenchmarkedApplication" "$file"; then
        # move the backup file to the original file
        mv "$file.bak" "$file"
    # else skip the file as it may not contain any benchmarking code
    else
        # Show a message that this file is skipped from being restored
        echo "File $file is probably not patched. Skipping restoration."
        continue
    fi
    echo "Restored $file"
done
