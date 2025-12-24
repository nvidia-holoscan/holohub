# -*- coding: utf-8 -*-
#
# (C) Copyright 2022 Karellen, Inc. (https://www.karellen.co/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import site
import sys
from email.message import Message
from email.parser import Parser
from os.path import dirname, join as jp, commonpath
from typing import Tuple

from pip._internal.exceptions import UnsupportedWheel
from pip._internal.locations import get_scheme
from pip._vendor import pkg_resources
from pip._vendor.pkg_resources import Distribution

__all__ = ["get_current_scheme", "get_dist_meta", "wheel_root_is_purelib"]


def _get_dist(metadata_directory: str) -> Distribution:
    """Return a pkg_resources.Distribution for the provided
    metadata directory.
    """
    dist_dir = metadata_directory.rstrip(os.sep)

    # Build a PathMetadata object, from path to metadata. :wink:
    base_dir, dist_dir_name = os.path.split(dist_dir)
    metadata = pkg_resources.PathMetadata(base_dir, dist_dir)

    # Determine the correct Distribution object type.
    if dist_dir.endswith(".egg-info"):
        dist_cls = pkg_resources.Distribution
        dist_name = os.path.splitext(dist_dir_name)[0]
    else:
        assert dist_dir.endswith(".dist-info")
        dist_cls = pkg_resources.DistInfoDistribution
        dist_name = os.path.splitext(dist_dir_name)[0].split("-")[0]

    return dist_cls(
        base_dir,
        project_name=dist_name,
        metadata=metadata,
    )


def get_current_scheme(dist_info_dir, dist_name, wheel_meta):
    install_dir = dirname(dist_info_dir)

    if not sys.flags.no_site and site.check_enableusersite():
        user_site_path = site.getusersitepackages()
        if commonpath((install_dir, user_site_path)) == user_site_path:
            scheme = get_scheme(dist_name, user=True)
        else:
            scheme = get_scheme(dist_name)
    else:
        scheme = get_scheme(dist_name)

    def check_correct_scheme(scheme, install_dir):
        if wheel_root_is_purelib(wheel_meta):
            scheme_dir = scheme.purelib
        else:
            scheme_dir = scheme.platlib

        if os.path.samefile(install_dir, scheme_dir):
            return True

        return False

    if check_correct_scheme(scheme, install_dir):
        return scheme

    raise RuntimeError("unable to determine current installation scheme")


def wheel_root_is_purelib(wheel_meta: Message) -> bool:
    return wheel_meta.get("Root-Is-Purelib", "").lower() == "true"


def get_dist_meta(dist_info_dir: str) -> Tuple[Distribution, Message]:
    return _get_dist(dist_info_dir), wheel_metadata(dist_info_dir)


def wheel_metadata(dist_info_dir) -> Message:
    """Return the WHEEL metadata of an extracted wheel, if possible.
    Otherwise, raise UnsupportedWheel.
    """
    path = jp(dist_info_dir, "WHEEL")
    with open(path, "rb") as f:
        wheel_contents = f.read()

    try:
        wheel_text = wheel_contents.decode()
    except UnicodeDecodeError as e:
        raise UnsupportedWheel(f"error decoding {path!r}: {e!r}")

    # FeedParser (used by Parser) does not raise any exceptions. The returned
    # message may have .defects populated, but for backwards-compatibility we
    # currently ignore them.
    return Parser().parsestr(wheel_text)
