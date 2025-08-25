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

from holoscan_ros2.operator import Operator


class PublisherOp(Operator):
    def __init__(
        self, fragment, *args, topic_name=None, qos=None, message_type=None, **kwargs
    ):
        self.topic_name_ = topic_name
        self.qos_ = qos
        self.message_type_ = message_type
        super().__init__(fragment, *args, **kwargs)
        self.publisher_ = None

    def initialize(self):
        super().initialize()
        self.publisher_ = self.ros2_bridge().create_publisher(
            self.message_type_, self.topic_name_, self.qos_
        )

    def publish(self, message):
        """Publish a message.

        Args:
            message: The message to publish
        """
        self.publisher_.publish(message)
