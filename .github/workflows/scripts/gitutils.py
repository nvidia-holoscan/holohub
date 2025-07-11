"""
SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

import os
import re
import subprocess


def is_file_empty(f):
    return os.stat(f).st_size == 0


def __git(*opts):
    """Runs a git command and returns its output"""
    cmd = "git " + " ".join(list(opts))
    ret = subprocess.check_output(cmd, shell=True)
    return ret.decode("UTF-8").rstrip("\n")


def __gitdiff(*opts):
    """Runs a git diff command with no pager set"""
    return __git("--no-pager", "diff", *opts)


def branch():
    """Returns the name of the current branch"""
    name = __git("rev-parse", "--abbrev-ref", "HEAD")
    name = name.rstrip()
    return name


def dir_():
    """Returns the top level directory of the repository"""
    git_dir = __git("rev-parse", "--show-toplevel")
    git_dir = git_dir.rstrip()
    return git_dir


def repo_version():
    """
    Determines the version of the repo by using `git describe`

    Returns
    -------
    str
        The full version of the repo in the format 'v#.#.#{a|b|rc}'
    """
    return __git("describe", "--tags", "--abbrev=0")


def repo_version_major_minor():
    """
    Determines the version of the repo using `git describe` and returns only
    the major and minor portion

    Returns
    -------
    str
        The partial version of the repo in the format '{major}.{minor}'
    """

    full_repo_version = repo_version()

    match = re.match(r"^v?(?P<major>[0-9]+)(?:\.(?P<minor>[0-9]+))?", full_repo_version)

    if match is None:
        print(
            "   [DEBUG] Could not determine repo major minor version. "
            f"Full repo version: {full_repo_version}."
        )
        return None

    out_version = match.group("major")

    if match.group("minor"):
        out_version += "." + match.group("minor")

    return out_version


def uncommitted_files():
    """
    Returns a list of all changed files that are not yet committed. This
    means both untracked/unstaged as well as uncommitted files too.
    """
    files = __git("status", "-u", "-s")
    ret = []
    for f in files.splitlines():
        f = f.strip(" ")
        f = re.sub(r"\s+", " ", f)  # noqa: W605
        tmp = f.split(" ", 1)
        # only consider staged files or uncommitted files
        # in other words, ignore untracked files
        if tmp[0] == "M" or tmp[0] == "A":
            ret.append(tmp[1])
    return ret


def changed_files_between(base_ref, new_ref):
    """
    Returns a list of files changed between base_ref and new_ref
    """
    files = __gitdiff("--name-only", "--ignore-submodules", f"{base_ref}..{new_ref}")
    return files.splitlines()


def changes_in_file_between(file, b1, b2, filter=None):  # noqa: A002
    """Filters the changed lines to a file between the branches b1 and b2"""
    current = branch()
    __git("checkout", "--quiet", b1)
    __git("checkout", "--quiet", b2)
    diffs = __gitdiff("--ignore-submodules", "-w", "--minimal", "-U0", f"{b1}...{b2}", "--", file)
    __git("checkout", "--quiet", current)
    return [line for line in diffs.splitlines() if (filter is None or filter(line))]


def modified_files(target=None, absolute_path=False):
    """
    If target is passed, then lists out all files modified between that git
    reference and HEAD. If this fails, this function will list out all
    the uncommitted files in the current branch.
    """
    all_files = changed_files_between(target, "HEAD") if target else uncommitted_files()

    if absolute_path:
        git_dir = dir_()
        return list(map(lambda fn: os.path.join(git_dir, fn), all_files))
    else:
        return all_files
