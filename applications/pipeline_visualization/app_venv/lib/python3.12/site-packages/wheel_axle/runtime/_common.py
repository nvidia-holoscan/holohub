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

import csv
import os
import sysconfig
from email.message import Message
from typing import Dict, Set, List, cast, IO

from pip._internal.models.scheme import Scheme
from pip._internal.operations.install.wheel import (
    get_csv_rows_for_installed,
    csv_io_kwargs,
    _normalized_outrows,
    _fs_to_record_path
)
from pip._vendor.pkg_resources import Distribution

from wheel_axle.runtime._wheel import get_dist_meta, get_current_scheme, wheel_root_is_purelib

IO = IO

LIBDIR = sysconfig.get_config_var("LIBDIR")
PLATLIBDIR = os.path.basename(LIBDIR)


class Installer:
    def __init__(self, dist_info_dir: str):
        self.dist_info_dir = dist_info_dir

        self.installed: Dict[str, str] = {}
        self.changed: Set[str] = set()
        self.generated: List[str] = []

        self.dist_meta: Distribution = None
        self.wheel_meta: Message = None
        self.scheme: Scheme = None
        self.lib_dir: str = None

    def run(self) -> None:
        dist_info_dir = self.dist_info_dir
        self.dist_meta, self.wheel_meta = get_dist_meta(dist_info_dir)
        self.scheme = get_current_scheme(dist_info_dir, self.dist_meta.project_name, self.wheel_meta)

        if wheel_root_is_purelib(self.wheel_meta):
            self.lib_dir = self.scheme.purelib
        else:
            self.lib_dir = self.scheme.platlib

        self.install()

        self.finalize()

    def install(self) -> None:
        pass

    def finalize(self) -> None:
        self.update_records()

    def update_records(self):
        if not (self.installed or self.changed or self.generated):
            return

        record_text = self.dist_meta.get_metadata_lines("RECORD")
        record_rows = list(csv.reader(record_text))

        rows = get_csv_rows_for_installed(
            record_rows,
            installed=self.installed,
            changed=self.changed,
            generated=self.generated,
            lib_dir=self.lib_dir,
        )

        # Record details of all files installed
        record_path = os.path.join(self.dist_info_dir, "RECORD")

        with open(record_path, **csv_io_kwargs("w")) as record_file:
            # Explicitly cast to typing.IO[str] as a workaround for the mypy error:
            # "writer" has incompatible type "BinaryIO"; expected "_Writer"
            writer = csv.writer(cast("IO[str]", record_file))
            writer.writerows(_normalized_outrows(rows))

    def record_installed(self,
                         srcfile: str,
                         destfile: str,
                         modified: bool = False
                         ) -> None:
        """Map archive RECORD paths to installation RECORD paths."""
        newpath = _fs_to_record_path(destfile, self.lib_dir)
        self.installed[srcfile] = newpath
        # if modified:
        #    self.changed.add(_fs_to_record_path(destfile))
