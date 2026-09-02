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

collector="${1:-.github/workflows/scripts/get_changed_doc_files.sh}"
lychee_config="${2:-.github/workflows/lychee.toml}"
lychee_bin="$(command -v "${LYCHEE:-lychee}" || true)"
test_tmp="$(mktemp -d)"
trap 'rm -rf -- "${test_tmp}"' EXIT

if [[ ! -r "${collector}" ]]; then
  echo "not ok - collector is readable: ${collector}" >&2
  exit 1
fi

failures=0

fail() {
  echo "not ok - $1" >&2
  failures=$((failures + 1))
}

pass() {
  echo "ok - $1"
}

# The collector runs as its own Bash process. Exporting this function gives it
# a deterministic git stub while still exercising its real argv construction.
git() {
  printf '%s\0' "$@" > "${GIT_ARGS_FILE}"
  if [[ "${GIT_EXIT_STATUS}" -ne 0 ]]; then
    return "${GIT_EXIT_STATUS}"
  fi
  command cat -- "${GIT_PATHS_FILE}"
}
export -f git

run_collector() {
  local case_name="$1"
  local git_status="$2"
  shift 2

  LAST_CASE_DIR="${test_tmp}/${case_name}"
  mkdir -p "${LAST_CASE_DIR}"
  printf '%s\0' "$@" > "${LAST_CASE_DIR}/git-paths"
  : > "${LAST_CASE_DIR}/github-output"

  set +e
  BASE_REF="main" \
    WORKSPACE="/workspace" \
    RUNNER_TEMP="${LAST_CASE_DIR}" \
    CHANGED_DOCS_MANIFEST="${LAST_CASE_DIR}/ambient-manifest" \
    CHANGED_DOCS_EXPECTED="${LAST_CASE_DIR}/ambient-expected-paths" \
    GITHUB_OUTPUT="${LAST_CASE_DIR}/github-output" \
    GIT_ARGS_FILE="${LAST_CASE_DIR}/git-args" \
    GIT_PATHS_FILE="${LAST_CASE_DIR}/git-paths" \
    GIT_EXIT_STATUS="${git_status}" \
    bash "${collector}" > "${LAST_CASE_DIR}/log" 2>&1
  LAST_STATUS=$?
  set -e
}

assert_git_argv() {
  local label="$1"
  local -a actual=()
  local -a expected=(
    diff
    -z
    --name-only
    --diff-filter=d
    origin/main...HEAD
    --
  )
  local index

  mapfile -d '' -t actual < "${LAST_CASE_DIR}/git-args"
  if [[ "${#actual[@]}" -ne "${#expected[@]}" ]]; then
    fail "${label}"
    return
  fi
  for index in "${!expected[@]}"; do
    if [[ "${actual[index]}" != "${expected[index]}" ]]; then
      fail "${label}"
      return
    fi
  done
  pass "${label}"
}

test_selects_and_escapes_documentation_paths() {
  local malicious_path="docs/evil\$(printf injected > ${test_tmp}/INJECTED).md"
  local -a changed_paths=(
    "applications/template/{{cookiecutter.project_slug}}/README.md"
    "docs/with space.html"
    "docs/ümlaut.rst"
    "docs/uppercase.MD"
    "docs/uppercase.HTML"
    "docs/uppercase.RsT"
    "${malicious_path}"
    "docs/.hidden[1].md"
    "docs/report*.md"
    "docs/report?.rst"
    "docs/literal]end.html"
    "docs/argument--mode-task.md"
    "docs/not-documentation.txt"
  )
  local expected_manifest="${test_tmp}/expected-manifest"
  local expected_paths="${test_tmp}/expected-paths"

  run_collector selection 0 "${changed_paths[@]}"

  printf '%s\n' \
    "/workspace/applications/template/{{cookiecutter.project_slug}}/README.md" \
    "/workspace/docs/with space.html" \
    "/workspace/docs/ümlaut.rst" \
    "/workspace/docs/uppercase.MD" \
    "/workspace/docs/uppercase.HTML" \
    "/workspace/docs/uppercase.RsT" \
    "/workspace/${malicious_path}" \
    "/workspace/docs/.hidden[[]1[]].md" \
    "/workspace/docs/report[*].md" \
    "/workspace/docs/report[?].rst" \
    "/workspace/docs/literal[]]end.html" \
    "/workspace/docs/argument--mode-task.md" > "${expected_manifest}"
  printf '%s\n' \
    "/workspace/applications/template/{{cookiecutter.project_slug}}/README.md" \
    "/workspace/docs/with space.html" \
    "/workspace/docs/ümlaut.rst" \
    "/workspace/docs/uppercase.MD" \
    "/workspace/docs/uppercase.HTML" \
    "/workspace/docs/uppercase.RsT" \
    "/workspace/${malicious_path}" \
    "/workspace/docs/.hidden[1].md" \
    "/workspace/docs/report*.md" \
    "/workspace/docs/report?.rst" \
    "/workspace/docs/literal]end.html" \
    "/workspace/docs/argument--mode-task.md" > "${expected_paths}"

  if [[ "${LAST_STATUS}" -ne 0 ]] ||
    ! cmp -s "${expected_manifest}" "${LAST_CASE_DIR}/changed_doc_files.txt" ||
    ! cmp -s "${expected_paths}" "${LAST_CASE_DIR}/changed_doc_files.expected.txt" ||
    [[ -e "${LAST_CASE_DIR}/ambient-manifest" ]] ||
    [[ -e "${LAST_CASE_DIR}/ambient-expected-paths" ]] ||
    ! grep -qx 'any_changed=true' "${LAST_CASE_DIR}/github-output" ||
    ! grep -qx 'changed_file_count=12' "${LAST_CASE_DIR}/github-output" ||
    grep -q '^changed_files_list=' "${LAST_CASE_DIR}/github-output" ||
    [[ -e "${test_tmp}/INJECTED" ]]; then
    fail "selects each documentation extension without evaluating filename bytes"
  else
    pass "selects each documentation extension without evaluating filename bytes"
  fi

  assert_git_argv "uses a NUL diff, excludes deletions, and diffs from the merge base"
}

test_fails_when_git_diff_fails() {
  run_collector diff-failure 42

  if [[ "${LAST_STATUS}" -eq 0 ]] || [[ -s "${LAST_CASE_DIR}/github-output" ]]; then
    fail "fails closed when git diff fails"
  else
    pass "fails closed when git diff fails"
  fi
}

test_rejects_line_breaks_before_emitting_paths() {
  local newline_path=$'docs/x\n::error file=README.md,line=1::pwned.md'
  local carriage_return_path=$'docs/x\r::error file=README.md,line=1::pwned.rst'
  local case_name
  local path

  for case_name in newline carriage-return; do
    if [[ "${case_name}" == newline ]]; then
      path="${newline_path}"
    else
      path="${carriage_return_path}"
    fi
    run_collector "${case_name}" 0 "${path}"

    if [[ "${LAST_STATUS}" -eq 0 ]] ||
      [[ ! -f "${LAST_CASE_DIR}/changed_doc_files.txt" ]] ||
      [[ -s "${LAST_CASE_DIR}/changed_doc_files.txt" ]] ||
      [[ ! -f "${LAST_CASE_DIR}/changed_doc_files.expected.txt" ]] ||
      [[ -s "${LAST_CASE_DIR}/changed_doc_files.expected.txt" ]] ||
      [[ -s "${LAST_CASE_DIR}/github-output" ]] ||
      grep -q '^::error file=README.md,line=1::' "${LAST_CASE_DIR}/log"; then
      fail "rejects ${case_name} filenames before emitting their bytes"
    else
      pass "rejects ${case_name} filenames before emitting their bytes"
    fi
  done
}

test_handles_large_changed_file_sets_outside_action_inputs() {
  local -a changed_paths=()
  local index
  local name

  for ((index = 0; index < 600; index++)); do
    printf -v name 'docs/%04d-%0235d.md' "${index}" 0
    changed_paths+=("${name}")
  done
  run_collector large-list 0 "${changed_paths[@]}"

  if [[ "${LAST_STATUS}" -ne 0 ]] ||
    [[ "$(wc -l < "${LAST_CASE_DIR}/changed_doc_files.txt")" -ne 600 ]] ||
    [[ ! -f "${LAST_CASE_DIR}/changed_doc_files.expected.txt" ]] ||
    [[ "$(wc -l < "${LAST_CASE_DIR}/changed_doc_files.expected.txt")" -ne 600 ]] ||
    [[ "$(wc -c < "${LAST_CASE_DIR}/github-output")" -ge 1024 ]] ||
    grep -q '^changed_files_list=' "${LAST_CASE_DIR}/github-output"; then
    fail "keeps large filename lists in the manifest"
  else
    pass "keeps large filename lists in the manifest"
  fi
}

test_reports_no_changed_documents() {
  run_collector no-docs 0 "src/code.cpp" "images/logo.png" "notes.txt"

  if [[ "${LAST_STATUS}" -ne 0 ]] ||
    [[ -s "${LAST_CASE_DIR}/changed_doc_files.txt" ]] ||
    [[ ! -f "${LAST_CASE_DIR}/changed_doc_files.expected.txt" ]] ||
    [[ -s "${LAST_CASE_DIR}/changed_doc_files.expected.txt" ]] ||
    ! grep -qx 'any_changed=false' "${LAST_CASE_DIR}/github-output" ||
    ! grep -qx 'changed_file_count=0' "${LAST_CASE_DIR}/github-output"; then
    fail "reports an empty documentation manifest"
  else
    pass "reports an empty documentation manifest"
  fi
}

test_config_preserves_manifest_case_and_expands_hidden_paths() {
  local case_dir="${test_tmp}/lychee-config"
  local actual="${case_dir}/actual"
  local config_dir
  local expected="${case_dir}/expected"
  local glob_actual="${case_dir}/glob-actual"
  local glob_expected="${case_dir}/glob-expected"
  local manifest="${case_dir}/manifest"

  if [[ -z "${lychee_bin}" ]] || [[ ! -r "${lychee_config}" ]]; then
    fail "Lychee config preserves manifest case and expands hidden paths"
    return
  fi
  config_dir="$(cd "$(dirname "${lychee_config}")" && pwd)"
  mkdir -p "${case_dir}/.github" "${case_dir}/docs"
  : > "${case_dir}/.github/guide.md"
  : > "${case_dir}/docs/.HIDDEN[1].md"
  : > "${case_dir}/docs/.hidden[1].md"
  : > "${case_dir}/docs/normal.md"
  : > "${case_dir}/docs/UPPER.MD"
  printf '%s\n' \
    "${case_dir}/docs/.hidden[[]1[]].md" \
    "${case_dir}/docs/normal.md" > "${manifest}"
  printf '%s\n' \
    "${case_dir}/docs/.hidden[1].md" \
    "${case_dir}/docs/normal.md" > "${expected}"
  printf '%s\n' \
    "${case_dir}/.github/guide.md" \
    "${case_dir}/docs/.HIDDEN[1].md" \
    "${case_dir}/docs/.hidden[1].md" \
    "${case_dir}/docs/normal.md" \
    "${case_dir}/docs/UPPER.MD" > "${glob_expected}"

  if ! (cd "${config_dir}" && "${lychee_bin}" --dump-inputs \
      --files-from "${manifest}") > "${actual}" 2> "${case_dir}/diagnostics" ||
    ! cmp -s <(LC_ALL=C sort "${expected}") <(LC_ALL=C sort "${actual}") ||
    ! (cd "${config_dir}" && "${lychee_bin}" --dump-inputs \
      "${case_dir}/**/*.[mM][dD]") > "${glob_actual}" 2>> "${case_dir}/diagnostics" ||
    ! cmp -s <(LC_ALL=C sort "${glob_expected}") <(LC_ALL=C sort "${glob_actual}"); then
    fail "Lychee config preserves manifest case and expands hidden paths"
  else
    pass "Lychee config preserves manifest case and expands hidden paths"
  fi
}

test_config_excludes_only_local_override_files() {
  local case_dir="${test_tmp}/lychee-excludes"
  local config_dir
  local output="${case_dir}/output"

  if [[ -z "${lychee_bin}" ]] || [[ ! -r "${lychee_config}" ]]; then
    fail "Lychee config excludes only local website overrides"
    return
  fi
  config_dir="$(cd "$(dirname "${lychee_config}")" && pwd)"
  mkdir -p "${case_dir}"
  printf '%s\n' \
    '[local](file:///workspace/holohub/doc/website/overrides/missing.html)' \
    '[remote](https://openai.com/doc/website/overrides/page)' > "${case_dir}/links.md"

  if ! (cd "${config_dir}" && "${lychee_bin}" --dump \
      "${case_dir}/links.md") > "${output}" 2> "${case_dir}/diagnostics" ||
    ! grep -Eq '^file:///workspace/holohub/doc/website/overrides/missing\.html .* \[excluded\]$' "${output}" ||
    grep -Eq '^https://openai\.com/doc/website/overrides/page .* \[excluded\]$' "${output}"; then
    fail "Lychee config excludes only local website overrides"
  else
    pass "Lychee config excludes only local website overrides"
  fi
}

test_selects_and_escapes_documentation_paths
test_fails_when_git_diff_fails
test_rejects_line_breaks_before_emitting_paths
test_handles_large_changed_file_sets_outside_action_inputs
test_reports_no_changed_documents
test_config_preserves_manifest_case_and_expands_hidden_paths
test_config_excludes_only_local_override_files

exit "${failures}"
