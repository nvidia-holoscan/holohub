# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.core import Operator as HoloOperator
from .bridge import Bridge


class Operator(HoloOperator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def ros2_bridge(self):
        """Get the ROS2 bridge resource."""
        # Look for bridge resource by type
        resources = self.resources
        for resource in resources.values():
            if isinstance(resource, Bridge):
                return resource
        raise RuntimeError(
            "No ROS2 Bridge resource found. Make sure to pass a Bridge resource to the operator."
        )
