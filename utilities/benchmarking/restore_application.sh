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
        # move the backup file to the original file
        mv "$file.bak" "$file"
    # else skip the file as it does not contain any benchmarking code
    else
        # Show a message that this file is skipped from being restored
        echo "File $file is probably not patched. Skipping restoration."
        continue
    fi
    echo "Restored $file"
done
