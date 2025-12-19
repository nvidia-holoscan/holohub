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

from os.path import join as jp

from wheel_axle.runtime import AXLE_DONE_FILE
from wheel_axle.runtime._common import Installer


class AxleFinalizer(Installer):
    def install(self) -> None:
        self.axle_done_path = jp(self.dist_info_dir, AXLE_DONE_FILE)

        self.record_installed(self.axle_done_path, self.axle_done_path, False)
        self.update_records()

    def finalize(self) -> None:
        with open(self.axle_done_path, "wb"):
            pass
