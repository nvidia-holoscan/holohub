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
from rclpy.node import Node
from std_msgs.msg import String
import concurrent.futures

from holoscan.core import Application
from holoscan_ros2.operators.subscriber import SubscriberOp
from holoscan_ros2.bridge import Bridge


class MySubscriberOp(SubscriberOp):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(
            fragment, *args, message_type=String, topic_name="topic", qos=10, **kwargs
        )

    def compute(self, op_input, op_output, context):
        while True:
            future = self.receive()
            try:
                # Wait 1 second for a message
                message = future.result(timeout=1.0)
                print(f"I heard: '{message.data}'")
                return  # Exit after processing one message
            except concurrent.futures.TimeoutError:
                # Check if we should terminate
                if not rclpy.ok():
                    print("ROS2 shutdown detected, exiting...")
                    return
                # Otherwise, continue waiting for the next message
                continue

    def stop(self):
        """Override stop method to handle KeyboardInterrupt gracefully."""
        try:
            super().stop()
        except KeyboardInterrupt:
            # Gracefully handle KeyboardInterrupt during operator shutdown
            pass


class HoloscanSubscriberApp(Application):
    def __init__(self):
        super().__init__()
        self.node = Node("holoscan_subscriber_node")

    def compose(self):
        bridge = Bridge(self, self.node, name="holoscan_subscriber_resource")
        subscriber_op = MySubscriberOp(
            self,
            bridge,
        )
        self.add_operator(subscriber_op)


def main():
    rclpy.init()
    app = HoloscanSubscriberApp()
    app.run()


if __name__ == "__main__":
    main()
