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

import os
from abc import ABC
from typing import Dict, Optional, Tuple


class RuntimeEnv(ABC):
    """Class responsible for managing run time settings.

    The expected environment variables are the keys in the defaults dictionary,
    and they can be set to override the defaults.
    """

    ENV_DEFAULT: Dict[str, Tuple[str, ...]] = {
        "input": ("HOLOSCAN_INPUT_PATH", "input"),
        "output": ("HOLOSCAN_OUTPUT_PATH", "output"),
        "model": ("HOLOSCAN_MODEL_PATH", "models"),
        "workdir": ("HOLOSCAN_WORKDIR", ""),
    }

    input: str = ""
    output: str = ""
    model: str = ""
    workdir: str = ""

    def __init__(self, defaults: Optional[Dict[str, Tuple[str, ...]]] = None):
        if defaults is None:
            defaults = self.ENV_DEFAULT
        for key, (env, default) in defaults.items():
            self.__dict__[key] = os.environ.get(env, default)
