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

import re

# From https://www.python.org/dev/peps/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions
VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""

VERSION_REGEX = re.compile(
    r"^\s*" + VERSION_PATTERN + r"\s*$",
    re.VERBOSE | re.IGNORECASE,
)

# From https://semver.org/#is-there-a-suggested-regular-expression-regex-to-check-a-semver-string
SEMVER_PATTERN = (
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

SEMVER_REGEX = re.compile(
    r"^\s*" + SEMVER_PATTERN + r"\s*$",
    re.VERBOSE | re.IGNORECASE,
)


def get_sdk_semver():
    """Convert a version string to a semver string.

    MONAI Deploy App SDK is a python package and its version is based on PEP-0440
    (https://www.python.org/dev/peps/pep-0440/).

    e.g., 0.1.0a1, 1.0.0a2.dev456, 1.0+abc.5, 0.1.0a1+0.g8444606.dirty

    The the canonical public version identifiers are like below:

        [N!]N(.N)*[{a|b|rc}N][.postN][.devN]

    MONAI Application Package (MAP) requires a semver string to be used in.
    The semver string is used to identify the version of the MONAI Deploy App SDK package.

    This method converts the MONAI Deploy App SDK package version string to a semver string.

    This uses a regular expression from the following link to parse the version string.

    https://www.python.org/dev/peps/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions

    The semver string would be returned with the following format:
        <release>(-<pre_l>.<pre_n>)?(+<local>)?

    Example)
        0.1.0a1 -> 0.1.0-a.1
        1.0.0a.2 -> 1.0.0-a.2
        1.0.0a2.dev456 -> 1.0.0-a.2
        1.0.0+abc.5 -> 1.0.0+abc.5
        0.1.0a1+0.g8444606.dirty -> 0.1.0-a.1+g.8444606.dirty

    Assumption:
        <release> is always X.Y.Z format.
        <pre_l> is 'a|b|rc'.
        <pre_n> is a positive number.
        <post> and <dev> are ignored.
        <local> is always "dot-separated build identifiers" (e.g., 0.g8444606.dirty).
    """
    import operators.medical_imaging

    version_str = operators.medical_imaging.__version__

    match = VERSION_REGEX.match(version_str)
    if match:
        release = match.group("release")
        pre_l = match.group("pre_l")
        pre_n = match.group("pre_n")
        local = match.group("local")
        if pre_l and pre_n:
            pre_release = f"-{pre_l}.{pre_n}"
        else:
            pre_release = ""
        if local:
            build = f"+{local}"
        else:
            build = ""

        semver_str = f"{release}{pre_release}{build}"

        if SEMVER_REGEX.match(semver_str):
            return semver_str
        else:
            raise ValueError(f"Invalid semver string: {semver_str!r} (from {version_str!r})")
    else:
        raise ValueError(f"Invalid version string: {version_str!r}")


if __name__ == "__main__":
    print(get_sdk_semver())
