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

"""
Bridge class for ROS2-Holoscan interoperability.

This module provides the Bridge class that handles ROS2 communication patterns including
publishers, subscribers.
"""

import threading
from queue import Queue
from concurrent.futures import Future

import rclpy
import rclpy.executors
from rclpy.node import Node

import holoscan.core


class Bridge(holoscan.core.Resource):
    """Bridge class for ROS2 communication."""

    def __init__(self, fragment, node=None, name=None):
        """Initialize the Bridge with a ROS2 node.

        Args:
            fragment: The fragment that owns this resource
            node: The ROS2 node to use for communication (optional)
            name: Name for this resource
        """
        super().__init__(fragment, name=name)
        self.node = node
        self._spin_thread = None
        self._shutdown_requested = False

    @classmethod
    def from_node(cls, fragment, node, name=None):
        """Create a Bridge from an existing ROS2 node.

        Args:
            fragment: The fragment that owns this resource
            node: The ROS2 node to use for communication
            name: Name for this resource

        Returns:
            Bridge instance
        """
        return cls(fragment, node, name)

    @classmethod
    def from_node_name(cls, fragment, node_name, name=None):
        """Create a Bridge with a new ROS2 node using the given name.

        Args:
            fragment: The fragment that owns this resource
            node_name: Name for the new ROS2 node
            name: Name for this resource

        Returns:
            Bridge instance
        """
        node = Node(node_name)
        return cls(fragment, node, name)

    @classmethod
    def from_node_name_and_namespace(cls, fragment, node_name, namespace, name=None):
        """Create a Bridge with a new ROS2 node using the given name and namespace.

        Args:
            fragment: The fragment that owns this resource
            node_name: Name for the new ROS2 node
            namespace: Namespace for the node
            name: Name for this resource

        Returns:
            Bridge instance
        """
        node = Node(node_name, namespace=namespace)
        return cls(fragment, node, name)

    def _spin_node(self):
        """Spin the ROS2 node with proper exception handling."""
        try:
            while not self._shutdown_requested and rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.1)
        except rclpy.executors.ExternalShutdownException:
            # Expected when ROS2 shuts down externally (e.g., Ctrl+C)
            pass
        except Exception as e:
            # Log other unexpected exceptions but don't crash
            print(f"ROS2 spinning thread encountered an error: {e}")

    def initialize(self):
        """Initialize the ROS2 bridge."""
        print("Initializing ROS2 bridge")
        if self._spin_thread is None:
            self._shutdown_requested = False
            self._spin_thread = threading.Thread(target=self._spin_node, daemon=True)
            self._spin_thread.start()

    def shutdown(self):
        """Shutdown the ROS2 bridge cleanly."""
        self._shutdown_requested = True
        if self._spin_thread is not None and self._spin_thread.is_alive():
            # Give the thread a moment to exit gracefully
            self._spin_thread.join(timeout=1.0)

    def valid(self):
        """Check if the bridge has a valid node."""
        return self.node is not None

    class Publisher:
        """Publisher class for ROS2 messages."""

        def __init__(self, node, topic_name=None, qos=None, message_type=None):
            """Initialize the Publisher.

            Args:
                node: ROS2 node
                topic_name: Topic name to publish to
                qos: Quality of Service settings
                message_type: Type of messages to publish (e.g., std_msgs.msg.String)
            """
            self.topic_name = topic_name
            self.qos = qos
            self.message_type = message_type
            self.publisher = None

            # Create the publisher immediately
            assert node is not None, "ROS2 node must be provided for Publisher"
            assert message_type is not None, (
                "message_type must be provided for Publisher"
            )
            assert topic_name is not None, "topic_name must be provided for Publisher"
            assert qos is not None, "qos must be provided for Publisher"

            self.publisher = node.create_publisher(message_type, topic_name, qos)
            assert self.publisher is not None, "Failed to create publisher"

        def publish(self, message):
            """Publish a message.

            Args:
                message: Message to publish (must match message_type)
            """
            self.publisher.publish(message)

    def create_publisher(self, message_type, topic_name, qos):
        """Create a new publisher.

        Args:
            message_type: Type of messages to publish
            topic_name: Topic to publish to
            qos: Quality of Service settings

        Returns:
            A new Publisher instance
        """
        return self.Publisher(self.node, topic_name, qos, message_type)

    class Subscriber:
        """Subscriber class for ROS2 messages."""

        def __init__(
            self,
            node,
            topic_name=None,
            qos=None,
            message_type=None,
            message_queue_max_size=0,
        ):
            """Initialize the Subscriber.

            Args:
                node: ROS2 node
                topic_name: Topic to subscribe to
                qos: Quality of Service settings
                message_type: Type of messages to receive (e.g., std_msgs.msg.String)
                message_queue_max_size: Maximum size of message queue (0 for unlimited)
            """
            self.message_queue = Queue(
                maxsize=message_queue_max_size if message_queue_max_size > 0 else 0
            )
            self.promise_queue = Queue()
            self.lock = threading.Lock()

            # Create the subscriber immediately
            assert node is not None, "ROS2 node must be provided for Subscriber"
            assert message_type is not None, (
                "message_type must be provided for Subscriber"
            )
            assert topic_name is not None, "topic_name must be provided for Subscriber"
            assert qos is not None, "qos must be provided for Subscriber"

            self.subscriber = node.create_subscription(
                message_type, topic_name, self._on_receive, qos
            )
            assert self.subscriber is not None, "Failed to create subscriber"

        def _on_receive(self, message):
            """Handle received messages."""
            with self.lock:
                if not self.promise_queue.empty():
                    future = self.promise_queue.get()
                    future.set_result(message)
                elif (
                    self.message_queue_max_size == 0
                    or self.message_queue.qsize() < self.message_queue_max_size
                ):
                    self.message_queue.put(message)
                else:
                    holoscan.core.log_warn("Message queue is full, dropping message")

        def receive(self):
            """Receive a message.

            Returns:
                A future that will contain the received message
            """
            with self.lock:
                if not self.message_queue.empty():
                    msg = self.message_queue.get()
                    future = Future()
                    future.set_result(msg)
                    return future
                else:
                    future = Future()
                    self.promise_queue.put(future)
                    return future

    def create_subscription(
        self, message_type, topic_name, qos, message_queue_max_size=0
    ):
        """Create a new subscriber.

        Args:
            message_type: Type of messages to receive
            topic_name: Topic to subscribe to
            qos: Quality of Service settings
            message_queue_max_size: Maximum size of message queue

        Returns:
            A new Subscriber instance
        """
        return self.Subscriber(
            self.node, topic_name, qos, message_type, message_queue_max_size
        )
