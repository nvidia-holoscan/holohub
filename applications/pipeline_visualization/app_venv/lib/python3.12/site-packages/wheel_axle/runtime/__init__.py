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
import contextlib
import sys
import site
from os.path import exists, join as jp, commonpath
from threading import RLock

from wheel_axle.runtime.constants import AXLE_DONE_FILE, AXLE_LOCK_FILE

__version__ = "0.0.7"

_DIST_INFO = "dist-info"

inter_thread_lock = RLock()


def _run_installers(dist_info_dir):
    # Get metadata
    from wheel_axle.runtime._libpython import LibPythonInstaller
    from wheel_axle.runtime._symlinks import SymlinksInstaller
    from wheel_axle.runtime._axle import AxleFinalizer

    installers = [LibPythonInstaller, SymlinksInstaller, AxleFinalizer]  # AxleFinalizer is always last!
    for installer in installers:
        installer(dist_info_dir).run()


@contextlib.contextmanager
def user_site_handler(pth_path):
    if sys.flags.no_site or (not site.check_enableusersite()):
        yield
        return

    user_site_path = site.getusersitepackages()
    in_user_site_path = False
    try:
        if commonpath((user_site_path, pth_path)) == user_site_path:
            in_user_site_path = True

        if in_user_site_path:
            current_sys_path = sys.path[:]
            sys.path[:] = site.addsitepackages(set(sys.path))
        yield
    finally:
        if in_user_site_path:
            sys.path[:] = current_sys_path


def finalize(pth_path):
    dist_info_dir = pth_path[:-3] + _DIST_INFO
    axle_done_path = jp(dist_info_dir, AXLE_DONE_FILE)

    # Double lock-check for performance
    if exists(axle_done_path):
        return

    lock_path = jp(dist_info_dir, AXLE_LOCK_FILE)

    # Lock in-process for thread race
    with inter_thread_lock:
        from filelock import FileLock  # Local import for speed

        # Lock inter-process for process race
        with FileLock(lock_path):
            # Double lock-check for performance
            if exists(axle_done_path):
                return

            with user_site_handler(pth_path):
                _run_installers(dist_info_dir)

            # Always the last step
            try:
                os.unlink(pth_path)
            except OSError:
                # This will probably fail on Windows
                pass
