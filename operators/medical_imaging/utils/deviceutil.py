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

import subprocess


def has_rocm():
    """Return True if ROCm is installed and GPU device is detected.

    Args:

    Returns:
        True if ROCm is installed and GPU device is detected, otherwise False.
    """
    cmd = "rocminfo"
    try:
        process = subprocess.run([cmd], stdout=subprocess.PIPE)
        for line_in in process.stdout.decode().splitlines():
            if "Device Type" in line_in and "GPU" in line_in:
                return True
    except Exception:
        pass

    return False


if __name__ == "__main__":
    print(has_rocm())
