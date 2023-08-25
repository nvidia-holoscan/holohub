#!/bin/bash
# set -o xtrace
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
        echo "File $file is probably already patched. Skipping."
        continue
    fi
    # Find the last #include line and add "#include "benchmark.hpp""" header after it
    cp "$file" "$file.bak"
    include_line=$(grep -nE -- "#include" "$file" | tail -n 1 | awk -F ':' '{print $1}')
    sed -i "$include_line a #include \"benchmark.hpp\"" "$file"

    # Find the "holoscan::Application" line in the cpp file and replace it with
    # "BenchmarkedApplication"
    sed -i 's/holoscan::Application/BenchmarkedApplication/g' "$file"

    echo "Patched $file"
done

