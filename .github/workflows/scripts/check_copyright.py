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

# This file is modified from the RAPIDS RAFT project which is under the
# Apache 2.0 license.
# (https://github.com/rapidsai/raft/blob/branch-22.08/ci/checks/copyright.py)

import argparse
import datetime
import itertools
import os
import re
import sys

import gitutils

FilesToCheck = [
    re.compile(r"[.](cmake|cpp|cu|cuh|h|hpp|sh|pxd|py|pyx|yaml)$"),
    re.compile(r"CMakeLists[.]txt$"),
    re.compile(r"Dockerfile$"),
    re.compile(r"[.]dockerfile$"),
    re.compile(r"CMakeLists_standalone[.]txt$"),
    re.compile(r"setup[.]cfg$"),
    re.compile(r"[.]flake8[.]cython$"),
    re.compile(r"meta[.]yaml$"),
]
ExemptFiles = []

# this will break starting at year 10000, which is probably OK :)
CheckSimple = re.compile(
    r"(^|\s*[#*/*]\s*)?SPDX-FileCopyrightText: Copyright \(c\) *(\d{4}),? NVIDIA CORPORATION & AFFILIATES\.(\s*[#*/*]?\s*)"
    r"(?:All rights|ALL RIGHTS)(\s*[#*/*]?\s*)(?:reserved\.|RESERVED\.)",
    re.MULTILINE | re.DOTALL,
)
CheckDouble = re.compile(
    r"(^|\s*[#*/*]\s*)?SPDX-FileCopyrightText: Copyright \(c\) *(\d{4})-(\d{4}),? NVIDIA CORPORATION & AFFILIATES\.(\s*[#*/*]?\s*)"
    r"(?:All rights|ALL RIGHTS)(\s*[#*/*]?\s*)(?:reserved\.|RESERVED\.)",
    re.MULTILINE | re.DOTALL,
)


def check_this_file(f):
    # This check covers things like symlinks which point to files that DNE
    if not (os.path.exists(f)):
        return False
    if gitutils and gitutils.is_file_empty(f):
        return False
    for exempt in ExemptFiles:
        if exempt.search(f):
            return False
    return any(checker.search(f) for checker in FilesToCheck)


def get_copyright_years(line):
    res = CheckSimple.search(line)
    if res:
        return (int(res.group(2)), int(res.group(2)))
    res = CheckDouble.search(line)
    if res:
        return (int(res.group(2)), int(res.group(3)))
    return (None, None)


def replace_current_year(line, start, end):
    # Determine the case format from the original text
    is_all_caps = "ALL RIGHTS" in line

    if is_all_caps:
        rights_text = "ALL RIGHTS"
        reserved_text = "RESERVED."
    else:
        rights_text = "All rights"
        reserved_text = "reserved."

    # first turn a simple regex into double (if applicable). then update years
    res = CheckSimple.sub(
        f"\\1SPDX-FileCopyrightText: Copyright (c) \\2-\\2 NVIDIA CORPORATION & AFFILIATES.\\3"
        f"{rights_text}\\4{reserved_text}",
        line,
    )
    res = CheckDouble.sub(
        f"\\1SPDX-FileCopyrightText: Copyright (c) {start}-{end} NVIDIA CORPORATION & AFFILIATES.\\4"
        f"{rights_text}\\5{reserved_text}",
        res,
    )
    return res


def check_copyright(f, update_current_year):
    """
    Checks for copyright headers and their years
    """
    errs = []
    this_year = datetime.datetime.now().year
    line_num = 0
    cr_found = False
    year_matched = False
    with open(f, encoding="utf-8") as fp:
        lines = fp.readlines()
        content = "".join(lines)

    # Check the entire file content for copyright headers (handles both single-line and multi-line headers)
    start, end = get_copyright_years(content)
    if start is not None:
        cr_found = True
        if start > end:
            e = [
                f,
                1,
                "First year after second year in the copyright header (manual fix required)",
                None,
            ]
            errs.append(e)
        if this_year < start or this_year > end:
            e = [f, 1, "Current year not included in the copyright header", None]
            if this_year < start:
                e[-1] = replace_current_year(content, this_year, end)
            if this_year > end:
                e[-1] = replace_current_year(content, start, this_year)
            errs.append(e)
        else:
            year_matched = True
    fp.close()
    # copyright header itself not found
    if not cr_found:
        e = [
            f,
            0,
            "Copyright header missing or formatted incorrectly (manual fix required)",
            None,
        ]
        errs.append(e)
    # even if the year matches a copyright header, make the check pass
    if year_matched:
        errs = []

    if update_current_year:
        errs_update = [x for x in errs if x[-1] is not None]
        if len(errs_update) > 0:
            print(
                "File: {}. Changing line(s) {}".format(
                    f, ", ".join(str(x[1]) for x in errs if x[-1] is not None)
                )
            )
            # Check if we're updating the entire file content (line_num == 1 and replacement is entire content)
            if len(errs_update) == 1 and errs_update[0][1] == 1 and "\n" in errs_update[0][3]:
                # This is a full file content replacement
                with open(f, "w", encoding="utf-8") as out_file:
                    out_file.write(errs_update[0][3])
            else:
                # This is line-by-line replacement
                for _, line_num, __, replacement in errs_update:
                    lines[line_num - 1] = replacement
                with open(f, "w", encoding="utf-8") as out_file:
                    for new_line in lines:
                        out_file.write(new_line)
        errs = [x for x in errs if x[-1] is None]

    return errs


def get_all_files_under_dir(root):
    ret_list = []
    for dirpath, _, filenames in os.walk(root):
        ret_list.extend([os.path.join(dirpath, fn) for fn in filenames])
    return ret_list


def check_copyright_main():
    """
    Checks for copyright headers in all the modified files. In case of local
    repo, this script will just look for uncommitted files and in case of CI
    it compares between branches "$PR_TARGET_BRANCH" and "current-pr-branch"
    """
    ret_val = 0
    global ExemptFiles

    argparser = argparse.ArgumentParser(
        "Checks for a consistent copyright header in git's modified files"
    )
    argparser.add_argument(
        "--update-current-year",
        dest="update_current_year",
        action="store_true",
        required=False,
        help="If set, update the current year if a header is already present and well formatted.",
    )
    argparser.add_argument(
        "--git-modified-only",
        dest="git_modified_only",
        action="store",
        type=str,
        nargs="?",
        default=None,
        const="no-target",
        required=False,
        help="If set, "
        "only files seen as modified by git will be "
        "processed. It will look for local modifications"
        "(unstaged, untracked) if no git reference is provided.",
    )
    argparser.add_argument(
        "--exclude",
        dest="exclude",
        action="append",
        required=False,
        default=[],
        help=("Exclude the paths specified (regexp). Can be specified multiple times."),
    )
    argparser.add_argument(
        "--exclude-config",
        dest="exclude_config",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to a file containing exclude patterns (one per line). "
            "Lines starting with # are treated as comments."
        ),
    )

    (args, dirs) = argparser.parse_known_args()

    # Read excludes from config file if specified
    config_excludes = []
    if args.exclude_config:
        config_path = args.exclude_config
        if not os.path.isabs(config_path):
            # If relative path, try current working directory first
            if os.path.exists(config_path):
                config_path = os.path.abspath(config_path)
            else:
                # If not found in current directory, try relative to script directory
                script_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(script_dir, os.path.basename(config_path))

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith("#"):
                        config_excludes.append(line)
        else:
            print(f"Warning: Config file not found at {config_path}")

    try:
        # Combine config file excludes with command line excludes
        all_excludes = config_excludes + args.exclude
        ExemptFiles = ExemptFiles + [pathName for pathName in all_excludes]
        ExemptFiles = [re.compile(file) for file in ExemptFiles]
    except re.error as reException:
        print("Regular expression error:")
        print(reException)
        return 1

    all_files = []
    if dirs:
        for d in [os.path.abspath(d) for d in dirs]:
            if not (os.path.isdir(d)):
                raise ValueError(f"{d} is not a directory.")
            all_files += get_all_files_under_dir(d)

    if args.git_modified_only:
        target_branch = None
        if args.git_modified_only != "no-target":
            target_branch = args.git_modified_only
        modified_files = gitutils.modified_files(target_branch, True)
        all_files = list(set(all_files).intersection(modified_files)) if dirs else modified_files

    files = [f for f in all_files if check_this_file(f)]

    # Print progress information
    print(f"Checking copyright headers in {len(files)} files out of {len(all_files)} total files")
    if len(files) > 0:
        print(f"Example files being checked: {', '.join(files[:3])}")
        if len(files) > 3:
            print(f"... and {len(files) - 3} more files")

    errors = tuple(itertools.chain(*[check_copyright(f, args.update_current_year) for f in files]))
    if errors:
        print("Copyright headers incomplete in some of the files!")
        for file_name, line_no, err_msg, _ in errors:
            print(f"  {file_name}:{line_no} Issue: {err_msg}")
        print("")
        n_fixable = sum(1 for e in errors if e[-1] is not None)
        if n_fixable > 0:
            print(
                f"You can run `python3 {' '.join(sys.argv)} --update-current-year` to fix "
                f"{n_fixable} of these errors."
            )
        ret_val = 1
    else:
        print("Copyright check passed")

    return ret_val


if __name__ == "__main__":
    import sys

    sys.exit(check_copyright_main())
