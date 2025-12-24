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
"""This module provides a Python interface to the Holoscan SDK logger.

.. autosummary::

    holoscan.pose_tree.Pose2
    holoscan.pose_tree.Pose3
    holoscan.pose_tree.PoseTree
    holoscan.pose_tree.PoseTreeAccessMethod
    holoscan.pose_tree.SO2
    holoscan.pose_tree.SO3
    holoscan.pose_tree.PoseTreeUCXClient
    holoscan.pose_tree.PoseTreeUCXClientConfig
    holoscan.pose_tree.PoseTreeUCXServer
    holoscan.pose_tree.PoseTreeUCXServerConfig
"""

import logging
import threading

from ._pose_tree import (
    SO2,
    SO3,
    Pose2,
    Pose3,
    PoseTree,
    PoseTreeAccessMethod,
    PoseTreeUCXClient,
    PoseTreeUCXClientConfig,
    PoseTreeUCXServer,
    PoseTreeUCXServerConfig,
)

__all__ = [
    "Pose2",
    "Pose3",
    "PoseTree",
    "PoseTreeAccessMethod",
    "PoseTreeManager",
    "SO2",
    "SO3",
    "PoseTreeUCXClient",
    "PoseTreeUCXClientConfig",
    "PoseTreeUCXServer",
    "PoseTreeUCXServerConfig",
]

from holoscan.core import (
    DistributedAppService,
    Resource,
)

# Create module logger
logger = logging.getLogger(__name__)


class PoseTreeManager(Resource, DistributedAppService):
    """Manage a shared PoseTree instance as a FragmentService, with optional distribution.

    This resource creates and holds a `holoscan.pose_tree.PoseTree` instance, making it
    accessible to multiple components within the same fragment. When used in a distributed
    application, it can synchronize the pose tree across a driver and worker processes.

    To use it, register an instance of `PoseTreeManager` with the fragment in your
    application's `compose` method:

        # In Application.compose()
        pose_tree_manager = PoseTreeManager(
            self, name="pose_tree", **self.kwargs("my_pose_tree_config")
        )
        self.register_service(pose_tree_manager)

    Operators can then access the underlying `PoseTree` instance via the `service()` method:

        # In Operator.__init__()
        self.pose_tree = self.service(PoseTreeManager, "pose_tree").tree

    Parameters
    ----------
    fragment : holoscan.core.Fragment
        The fragment that the resource belongs to.
    port : int, optional
        Port for the UCX server to listen on in a distributed setup. Defaults to 13337.
    number_frames : int, optional
        Maximum number of coordinate frames. Defaults to 1024.
    number_edges : int, optional
        Maximum number of edges (direct transformations). Defaults to 16384.
    history_length : int, optional
        Total capacity for historical pose data. Defaults to 1048576.
    default_number_edges : int, optional
        Default edges allocated per new frame. Defaults to 16.
    default_history_length : int, optional
        Default history capacity per new edge. Defaults to 1024.
    edges_chunk_size : int, optional
        Allocation chunk size for a frame's edge list. Defaults to 4.
    history_chunk_size : int, optional
        Allocation chunk size for an edge's history buffer. Defaults to 64.
    request_timeout_ms : int, optional
        UCX client request timeout in milliseconds. Defaults to 5000.
    request_poll_sleep_us : int, optional
        UCX client polling sleep interval in microseconds. Defaults to 10.
    worker_progress_sleep_us : int, optional
        UCX progress loop sleep interval in microseconds. Defaults to 100.
    server_shutdown_timeout_ms : int, optional
        UCX server shutdown timeout in milliseconds. Defaults to 1000.
    server_shutdown_poll_sleep_ms : int, optional
        UCX server shutdown polling interval in milliseconds. Defaults to 10.
    name : str, optional
        The name of the resource.

    Notes
    -----
    **Experimental Feature**
    The Pose Tree feature is experimental. The API may change in future releases.
    """

    def __init__(
        self,
        fragment,
        *args,
        port=13337,
        number_frames=1024,
        number_edges=16384,
        history_length=1048576,
        default_number_edges=16,
        default_history_length=1024,
        edges_chunk_size=4,
        history_chunk_size=64,
        request_timeout_ms=5000,
        request_poll_sleep_us=10,
        worker_progress_sleep_us=100,
        server_shutdown_timeout_ms=1000,
        server_shutdown_poll_sleep_ms=10,
        **kwargs,
    ):
        Resource.__init__(self, fragment, *args, **kwargs)
        DistributedAppService.__init__(self)

        self.port = port
        self.request_timeout_ms = request_timeout_ms
        self.request_poll_sleep_us = request_poll_sleep_us
        self.worker_progress_sleep_us = worker_progress_sleep_us
        self.server_shutdown_timeout_ms = server_shutdown_timeout_ms
        self.server_shutdown_poll_sleep_ms = server_shutdown_poll_sleep_ms

        self._server = None
        self._client = None
        self._lock = threading.Lock()  # Thread synchronization lock

        self._tree = PoseTree()
        self._tree.init(
            number_frames=number_frames,
            number_edges=number_edges,
            history_length=history_length,
            default_number_edges=default_number_edges,
            default_history_length=default_history_length,
            edges_chunk_size=edges_chunk_size,
            history_chunk_size=history_chunk_size,
        )

    # FragmentService interface – return ourself as the resource
    def resource(self, _new_resource=None):
        if _new_resource:
            raise ValueError("Setting resource on PoseTreeManager is not supported.")
        return self

    @property
    def tree(self):
        """Get the managed `PoseTree` instance.

        This is the primary method for accessing the pose tree from other components.

        Returns
        -------
        holoscan.pose_tree.PoseTree
            The underlying pose tree instance managed by this resource.
        """
        return self._tree

    def driver_start(self, _driver_ip):
        """Start the PoseTreeUCXServer on the driver."""
        # First check if server already exists
        with self._lock:
            if self._server is not None:
                return  # Server already started

            # Create server instance while holding lock
            config = PoseTreeUCXServerConfig()
            config.worker_progress_sleep_us = self.worker_progress_sleep_us
            config.shutdown_timeout_ms = self.server_shutdown_timeout_ms
            config.shutdown_poll_sleep_ms = self.server_shutdown_poll_sleep_ms
            server = PoseTreeUCXServer(self._tree, config)

        # Release lock before calling potentially blocking start() method
        try:
            server.start(self.port)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to start PoseTreeUCXServer on port {self.port}: {e}") from e

        # Only set _server after successful start
        with self._lock:
            self._server = server

    def driver_shutdown(self):
        """Stop the PoseTreeUCXServer on the driver."""
        # Get server reference and clear it atomically
        with self._lock:
            server = self._server
            self._server = None

        # Stop server outside of lock
        if server:
            try:
                server.stop()
            except RuntimeError as e:
                # Log but continue with cleanup
                logger.warning("Error stopping PoseTreeUCXServer: %s", e)

    def worker_connect(self, driver_ip):
        """Connect the worker's PoseTreeUCXClient to the driver."""
        # First check if client already exists
        with self._lock:
            if self._client is not None:
                return  # Client already connected

            # Create client instance while holding lock
            config = PoseTreeUCXClientConfig()
            config.request_timeout_ms = self.request_timeout_ms
            config.request_poll_sleep_us = self.request_poll_sleep_us
            config.worker_progress_sleep_us = self.worker_progress_sleep_us
            client = PoseTreeUCXClient(self._tree, config)

        # Release lock before calling potentially blocking connect() method
        try:
            client.connect(driver_ip, self.port, True)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to connect to PoseTreeUCXServer at {driver_ip}:{self.port}: {e}"
            ) from e

        # Only set _client after successful connection
        with self._lock:
            self._client = client

    def worker_disconnect(self):
        """Disconnect the worker's PoseTreeUCXClient."""
        # Get client reference and clear it atomically
        with self._lock:
            client = self._client
            self._client = None

        # Disconnect client outside of lock
        if client:
            try:
                client.disconnect()
            except RuntimeError as e:
                # Log but continue with cleanup
                logger.warning("Error disconnecting PoseTreeUCXClient: %s", e)
