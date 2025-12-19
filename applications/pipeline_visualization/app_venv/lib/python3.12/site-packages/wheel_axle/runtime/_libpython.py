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
import sysconfig

from pip._internal.exceptions import InstallationError

from wheel_axle.runtime._common import Installer, PLATLIBDIR, LIBDIR
from wheel_axle.runtime.constants import REQUIRE_LIBPYTHON_FILE


class LibPythonInstaller(Installer):
    def install(self) -> None:
        require_libpython_path = os.path.join(self.dist_info_dir, REQUIRE_LIBPYTHON_FILE)

        if not os.path.exists(require_libpython_path):
            return

        self._install_libpython()

    def _install_libpython(self):
        enable_shared = sysconfig.get_config_var("PY_ENABLE_SHARED") or sysconfig.get_config_var("Py_ENABLE_SHARED")
        if not enable_shared or not int(enable_shared):
            message = (
                "The distribution {!r} requires dynamic linking to the `libpython` "
                "but current instance of CPython was built without `--enable-shared`."
            )
            raise InstallationError(
                message.format(self.dist_meta.project_name)
            )

        in_venv = sys.base_exec_prefix != sys.exec_prefix
        is_user_site = self.lib_dir.startswith(site.USER_SITE)

        # Find libpython library names and locations
        shared_library_path = LIBDIR
        all_ld_library_names = list(set(n for n in (sysconfig.get_config_var("LDLIBRARY"),
                                                    sysconfig.get_config_var("INSTSONAME")) if n))

        # There are no libraries to link to
        if not all_ld_library_names:
            message = (
                "The distribution {!r} requires dynamic linking to the `libpython` "
                "but was unable to find any libraries declared available in the current installation of CPython, "
                "even though it was compiled with '--enable-shared'"
            )
            raise InstallationError(
                message.format(self.dist_meta.project_name)
            )

        all_ld_library_paths = list(os.path.join(shared_library_path, n) for n in all_ld_library_names)
        for p in all_ld_library_paths:
            if not os.path.exists(p) or not os.access(p, os.R_OK):
                message = (
                    "The distribution {!r} requires dynamic linking to the `libpython` "
                    "but a library {!r} declared available in the current installation of CPython "
                    "cannot be accessed or doesn't exist"
                )
                raise InstallationError(
                    message.format(self.dist_meta.project_name, p)
                )

        lib_path = None
        if is_user_site:
            lib_path = os.path.join(site.USER_BASE, PLATLIBDIR)
        elif in_venv:
            lib_path = os.path.join(sys.exec_prefix, PLATLIBDIR)

        # We're neither in a venv, nor in a user site, i.e. it's an install into Python proper
        if not lib_path:
            return

        all_ld_library_links = list(os.path.join(lib_path, p) for p in all_ld_library_names)

        # Check is symlinks already exist
        for idx, ld_library_link in enumerate(all_ld_library_links):
            if os.path.islink(ld_library_link) or os.path.exists(ld_library_link):
                # Link or file already exists, so skip to next
                continue
            os.symlink(all_ld_library_paths[idx], ld_library_link, False)
