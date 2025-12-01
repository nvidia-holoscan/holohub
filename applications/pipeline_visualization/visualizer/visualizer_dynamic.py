#!/usr/bin/python3
"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging

import pipeline_visualization.flatbuffers.Message
import pipeline_visualization.flatbuffers.Payload
import tensor_to_numpy
from dash import ALL, Dash, Input, Output, Patch, State, ctx, set_props
from dash.exceptions import PreventUpdate
from graph_components import create_app_layout, create_graph
from nats_async import NatsAsync

# Set up logger for debugging and info messages
logger = logging.getLogger(__name__)


class Visualizer:
    """
    A Dash-based web application for dynamically visualizing pipeline data from NATS streams.

    This visualizer automatically discovers and displays real-time data from Holoscan data streams,
    dynamically creating graphs as new data sources are detected. Unlike the static visualizer,
    this version does not require predefined stream names.
    """

    def __init__(self):
        """Initialize the visualizer with dynamic graph discovery and Dash layout."""
        # Create the Dash web application
        self._app = Dash(__name__)

        # Build the application layout (dynamic mode - no initial graphs)
        self._app.layout = create_app_layout()

        # NATS client instance (initialized when run() is called)
        self._nats_inst = None

        @self._app.callback(
            Input("subject", "value"),
            Input("connect", "n_clicks"),
        )
        def update_subject(subject, n_clicks):
            """
            Handle NATS subscription connection/disconnection.

            This callback manages the connect/disconnect button behavior:
            - Odd clicks: Subscribe to NATS subject and disable input
            - Even clicks: Unsubscribe from NATS subject and enable input
            """
            if ctx.triggered_id == "connect":
                if n_clicks % 2 == 1:
                    # Connect: subscribe to the NATS subject with ".data" suffix
                    self._nats_inst.subscribe(subject + ".data")
                    self._subject = subject
                    set_props("subject", {"disabled": True})
                    set_props("connect", {"children": "Disconnect"})
                else:
                    # Disconnect: unsubscribe from the NATS subject
                    self._nats_inst.unsubscribe(subject + ".data")
                    self._subject = None
                    set_props("subject", {"disabled": False})
                    set_props("connect", {"children": "Connect"})
            # Disable connect button if subject field is empty
            set_props("connect", {"disabled": len(subject) == 0})

        @self._app.callback(
            Output("data-container", "children"),
            Input("interval-component", "n_intervals"),
            State("subject", "value"),
            State({"type": "graph-container", "id": ALL}, "id"),
            prevent_initial_call=True,
        )
        def update_source(n_intervals, subject, data_container_ids):  # noqa: ARG001
            """
            Periodically fetch and update data from NATS, dynamically creating graphs as needed.

            This callback is triggered by the interval component (every 200ms) and:
            1. Retrieves all pending messages from the NATS queue
            2. Parses Holoscan flatbuffer messages
            3. Extracts tensor data and metadata
            4. Dynamically creates new graphs for previously unseen data streams
            5. Updates existing graphs with new data

            Args:
                n_intervals: Number of intervals elapsed (from interval component)
                subject: The NATS subject name
                data_container_ids: List of existing graph container IDs

            Returns:
                Patch object for updating the data container with new/updated graphs
            """
            # Skip update if NATS client is not initialized
            if self._nats_inst is None:
                raise PreventUpdate

            done = False
            # Create a Patch object for efficient partial updates
            patch = Patch()

            # Process all available messages in the queue
            while not done:
                # Retrieve next message from NATS
                obj = self._nats_inst.get_message(subject + ".data")
                if obj is None:
                    done = True
                    continue

                # Parse the Holoscan flatbuffer message
                message = pipeline_visualization.flatbuffers.Message.Message.GetRootAs(obj, 0)

                # Verify the payload is a tensor type
                if (
                    message.PayloadType()
                    != pipeline_visualization.flatbuffers.Payload.Payload().pipeline_visualization_flatbuffers_Tensor
                ):
                    logger.warning(f"Unknown PayloadType: {message.PayloadType()}")
                    continue

                # Get the unique identifier for this data stream
                unique_id = message.UniqueId().decode()
                # Replace dots with dashes for Dash component ID compatibility
                dash_id = unique_id.replace(".", "-")

                # Convert tensor data to numpy array
                try:
                    data = tensor_to_numpy.tensor_to_numpy(message.Payload())
                except ValueError:
                    logger.exception(f"Error converting tensor at {unique_id} to numpy")
                    continue

                # Check if a graph already exists for this data stream
                exists = False
                for index, container_id in enumerate(data_container_ids):
                    if container_id["id"] == dash_id:
                        # Update metadata fields: IO type (Input/Output)
                        patch[index]["props"]["children"][0]["props"]["children"][1]["props"][
                            "children"
                        ][1]["props"]["children"] = (
                            "Output"
                            if message.IoType()
                            == pipeline_visualization.flatbuffers.IOType.IOType().kOutput
                            else "Input"
                        )
                        # Update metadata fields: acquisition timestamp
                        patch[index]["props"]["children"][0]["props"]["children"][2]["props"][
                            "children"
                        ][1]["props"]["children"] = str(message.AcquisitionTimestampNs())
                        # Update metadata fields: publish timestamp
                        patch[index]["props"]["children"][0]["props"]["children"][3]["props"][
                            "children"
                        ][1]["props"]["children"] = str(message.TimestampNs())
                        # Update the graph's y-axis data with the new tensor values
                        patch[index]["props"]["children"][1]["props"]["figure"]["data"][0][
                            "y"
                        ] = data
                        exists = True
                        break

                # If this is a new data stream, create a new graph for it
                if not exists:
                    new_graph = create_graph(
                        name=unique_id,
                        message=message,
                        data=data,
                        id_type="graph-container",
                    )
                    data_container_ids.append({"id": dash_id})
                    patch.append(new_graph)

            return patch

    def run(self):
        """
        Start the NATS client and run the Dash web application.

        The application will be accessible at http://0.0.0.0:8050
        NATS connection is established at 0.0.0.0:4222
        """
        # Initialize NATS client for receiving messages
        self._nats_inst = NatsAsync(host="0.0.0.0:4222")

        # Start the Dash web server (blocking call)
        self._app.run(debug=False, host="0.0.0.0", port=8050)

        # Clean up NATS connection when the app shuts down
        self._nats_inst.shutdown()


if __name__ == "__main__":
    # Configure logging to display info-level messages
    logging.basicConfig(level=logging.INFO)

    # Create and run the visualizer application
    visualizer = Visualizer()
    visualizer.run()
