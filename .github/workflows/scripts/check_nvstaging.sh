#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Script to check for internal artifacts (nvstaging) in files
# Usage: check_nvstaging.sh [--exclude-config <file>] [path]

set -e

EXCLUDE_CONFIG=""
SEARCH_PATH="."

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --exclude-config) EXCLUDE_CONFIG="$2"; shift 2 ;;
    *) SEARCH_PATH="$1"; shift ;;
  esac
done

# Check if file should be excluded
should_exclude() {
  [[ -f "$EXCLUDE_CONFIG" ]] || return 1
  while IFS= read -r pattern; do
    [[ -z "$pattern" || "$pattern" =~ ^[[:space:]]*# ]] && continue
    [[ "$1" == $pattern ]] && return 0
  done < "$EXCLUDE_CONFIG"
  return 1
}

echo "Checking for internal artifacts (nvstaging)..."

# Find and check files
internal_refs=()
while IFS= read -r -d '' file; do
  should_exclude "$file" || {
    grep -l "nvstaging" "$file" >/dev/null 2>&1 && internal_refs+=("$file")
  }
done < <(find "$SEARCH_PATH" -type f \( -name "*.cpp" -o -name "*.c" -o -name "*.h" -o -name "*.hpp" -o -name "*.py" -o -name "*.cu" -o -name "*.cuh" -o -name "*.md" -o -name "*.rst" -o -name "*.txt" -o -name "*.yml" -o -name "*.yaml" -o -name "*.json" \) -print0 2>/dev/null)

# Report results
if [[ ${#internal_refs[@]} -gt 0 ]]; then
  echo "Files containing internal artifacts (nvstaging):"
  printf '%s\n' "${internal_refs[@]}"
  exit 1
else
  echo "No internal artifacts found"
fi
