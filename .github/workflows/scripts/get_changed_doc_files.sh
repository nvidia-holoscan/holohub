#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -euo pipefail

: "${BASE_REF:?BASE_REF must be set}"
: "${WORKSPACE:?WORKSPACE must be set}"
: "${RUNNER_TEMP:?RUNNER_TEMP must be set}"
: "${GITHUB_OUTPUT:?GITHUB_OUTPUT must be set}"

manifest="${RUNNER_TEMP}/changed_doc_files.txt"
expected_paths="${RUNNER_TEMP}/changed_doc_files.expected.txt"
diff_list="${RUNNER_TEMP}/changed_files.z"
: > "${manifest}"
: > "${expected_paths}"

if ! git diff -z --name-only --diff-filter=d "origin/${BASE_REF}...HEAD" -- > "${diff_list}"; then
  printf -v quoted_ref '%q' "${BASE_REF}"
  echo "::error::Could not diff against origin/${quoted_ref}; refusing to skip the link check."
  exit 1
fi

changed_file_count=0
while IFS= read -r -d '' file; do
  case "${file##*.}" in
    [mM][dD]|[hH][tT][mM][lL]|[rR][sS][tT]) ;;
    *) continue ;;
  esac

  # Lychee's --files-from format is line-delimited, and its Markdown report
  # renders input names verbatim. Reject line breaks before writing or logging
  # the filename so it cannot create a forged workflow command.
  if [[ "${file}" == *$'\n'* || "${file}" == *$'\r'* ]]; then
    printf -v quoted_file '%q' "${file}"
    echo "::error::Documentation filename contains an unsupported line break: ${quoted_file}"
    exit 1
  fi

  # --files-from still treats each line as a glob pattern. Encode Lychee's
  # glob metacharacters as literal character classes, matching Rust glob's
  # Pattern::escape behavior while leaving spaces unchanged.
  path="${WORKSPACE}/${file}"
  glob_escaped=""
  for ((index = 0; index < ${#path}; index++)); do
    character="${path:index:1}"
    case "${character}" in
      '?'|'*'|'['|']') glob_escaped+="[${character}]" ;;
      *) glob_escaped+="${character}" ;;
    esac
  done
  printf '%s\n' "${glob_escaped}" >> "${manifest}"
  printf '%s\n' "${path}" >> "${expected_paths}"
  changed_file_count=$((changed_file_count + 1))
done < "${diff_list}"

if [[ "${changed_file_count}" -gt 0 ]]; then
  echo "any_changed=true" >> "${GITHUB_OUTPUT}"
else
  echo "any_changed=false" >> "${GITHUB_OUTPUT}"
fi
echo "changed_file_count=${changed_file_count}" >> "${GITHUB_OUTPUT}"
