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

import rclpy
from holoscan.conditions import PeriodicCondition
from holoscan.core import Application
from holoscan_ros2.bridge import Bridge
from holoscan_ros2.operators.publisher import PublisherOp
from rclpy.node import Node
from std_msgs.msg import String


class SimplePublisherOp(PublisherOp):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, message_type=String, **kwargs)
        self.count = 0

    def compute(self, op_input, op_output, context):
        msg = String()
        msg.data = f"Hello, world! {self.count}"
        print(f"Publishing: '{msg.data}'")
        self.publish(msg)
        self.count += 1

    def stop(self):
        """Override stop method to handle KeyboardInterrupt gracefully."""
        try:
            super().stop()
        except KeyboardInterrupt:
            # Gracefully handle KeyboardInterrupt during operator shutdown
            pass


class HoloscanSimplePublisherApp(Application):
    def __init__(self):
        super().__init__()
        self.node = Node("holoscan_publisher_node")

    def compose(self):
        bridge = Bridge(self, self.node, name="holoscan_publisher_resource")
        simple_publisher_op = SimplePublisherOp(
            self,
            PeriodicCondition(self, name="period", recess_period=0.5),
            bridge,
            topic_name="topic",
            qos=10,  # Replace with a QoSProfile or your QoS class if needed
        )
        self.add_operator(simple_publisher_op)


def main():
    # Initialize ROS2
    rclpy.init()
    app = HoloscanSimplePublisherApp()
    app.run()


if __name__ == "__main__":
    main()
