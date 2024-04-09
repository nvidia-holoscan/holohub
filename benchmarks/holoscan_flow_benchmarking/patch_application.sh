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
# find all the cpp files in the directory and the subdirectories
cpp_files=$(find "$dir" -type f -name "*.cpp")

# Iterate over each found file
for file in $cpp_files; do
    # Find whether the file has "BenchmarkedApplication" in it and skip it if it does
    if grep -q "BenchmarkedApplication" "$file"; then
        # Show error message for the file
        echo "Found \"BenchmarkedApplication\" in file $file. It is probably patched already. Skipping."
        continue
    fi
    # Find the "holoscan::Application" line in the cpp file and include benchmark header file before it
    include_line=$(grep -nE -- "holoscan::Application" "$file" | tail -n 1 | awk -F ':' '{print $1}')
    if [ -z "$include_line" ]; then
        continue
    fi

    # If we detect a holoscan application in this file
    cp "$file" "$file.bak"
    include_line=$((include_line-1))
    sed -i "${include_line} a #include \"benchmark.hpp\"\n" "$file"

    # Modify holoscan::Application to "BenchmarkedApplication"
    sed -i 's/holoscan::Application/BenchmarkedApplication/g' "$file"

    echo "Patched $file. Original file is backed up in $file.bak.  Run command below to see the differences:"
    echo "  diff -u $file.bak $file"
done


# find all the py files in the directory and the subdirectories
py_files=$(find "$dir" -type f -name "*.py")

for file in $py_files; do
    # Find whether the file has "BenchmarkedApplication" in it and skip it if it does
    if grep -q "BenchmarkedApplication" "$file"; then
        # Show error message for the file
        echo "Found \"BenchmarkedApplication\" in file $file. It is probably patched already. Skipping."
        continue
    fi

    # Find the first import statement and add "import sys newline sys.path.append(absolute path of
    # the directory of this bash script)
    # newline import benchmarked_application before it"
    cp "$file" "$file.bak"
    import_line=$(grep -nE -- "import" "$file" | head -n 1 | awk -F ':' '{print $1}')
    sed -i "$import_line i import sys\nsys.path.append(\"$(dirname "$(readlink -f "$0")")\")\nfrom benchmarked_application import BenchmarkedApplication\n" "$file"

    # Find "(Application)" and change it to "(BenchmarkedApplication)"
    sed -i 's/(Application)/(BenchmarkedApplication)/g' "$file"

    echo "Patched $file. Original file is backed up in $file.bak. diff -u $file $file.bak to see the changes."
done
