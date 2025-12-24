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
from dash import Dash, Input, Output, Patch, ctx, set_props
from dash.exceptions import PreventUpdate
from graph_components import create_app_layout
from nats_async import NatsAsync

# Set up logger for debugging and info messages
logger = logging.getLogger(__name__)


class Visualizer:
    """
    A Dash-based web application for visualizing pipeline data from NATS streams.

    This visualizer displays real-time data from multiple Holoscan data streams,
    showing both metadata (timestamps, IO types) and data plots.
    """

    def __init__(self, use_ising=False):
        """Initialize the visualizer with predefined data streams and Dash layout.
        
        Args:
            use_ising: If True, configure for Ising model pipeline. If False, use sine wave pipeline.
        """
        # Define the data streams to visualize (operator.port format)
        if use_ising:
            # Ising model pipeline: ising_source -> sink
            self._unique_ids = ["ising_source.out", "sink.in"]
        else:
            # Sine wave pipeline: source -> modulate -> sink
            self._unique_ids = ["source.out", "modulate.in", "modulate.out", "sink.in"]

        # Create the Dash web application
        self._app = Dash(__name__)

        # Build the application layout
        self._app.layout = create_app_layout(unique_ids=self._unique_ids)

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

        # Create dynamic outputs based on unique_ids
        outputs = [Output(uid.replace(".", "-"), "children") for uid in self._unique_ids]
        
        @self._app.callback(
            outputs,
            Input("interval-component", "n_intervals"),
            Input("subject", "value"),
            prevent_initial_call=True,
        )
        def update_source(n_intervals, subject):  # noqa: ARG001
            """
            Periodically fetch and update data from NATS for all data streams.

            This callback is triggered by the interval component (every 200ms) and:
            1. Retrieves all pending messages from the NATS queue
            2. Parses Holoscan flatbuffer messages
            3. Extracts tensor data and metadata
            4. Updates the corresponding graphs with new data

            Args:
                n_intervals: Number of intervals elapsed (from interval component)
                subject: The NATS subject name

            Returns:
                List of Patch objects for updating each graph component
            """
            # Skip update if NATS client is not initialized
            if self._nats_inst is None:
                raise PreventUpdate

            done = False
            # Create Patch objects for efficient partial updates of each graph
            patches = [Patch() for _ in self._unique_ids]

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

                # Identify which graph this data belongs to
                unique_id = message.UniqueId().decode()
                for i, id in enumerate(self._unique_ids):
                    # the python application is using tensor maps where the unique_id also contains
                    # the tensor name, so we need to check if the unique_id starts with the id
                    if unique_id.startswith(id):
                        index = i
                        break
                else:
                    logger.warning(f"Unknown unique_id: {unique_id}")
                    continue

                # Convert tensor data to numpy array
                try:
                    data = tensor_to_numpy.tensor_to_numpy(message.Payload())
                    print(f"[DEBUG VISUALIZER] Received data for {unique_id}: shape={data.shape}, ndim={data.ndim}, dtype={data.dtype}")
                except ValueError:
                    logger.exception(f"Error converting tensor at {unique_id} to numpy")
                    continue

                # Update metadata fields: IO type (Input/Output)
                patches[index][0]["props"]["children"][1]["props"]["children"][1]["props"][
                    "children"
                ] = (
                    "Output"
                    if message.IoType()
                    == pipeline_visualization.flatbuffers.IOType.IOType().kOutput
                    else "Input"
                )
                # Update metadata fields: acquisition timestamp
                patches[index][0]["props"]["children"][2]["props"]["children"][1]["props"][
                    "children"
                ] = str(message.AcquisitionTimestampNs())
                # Update metadata fields: publish timestamp
                patches[index][0]["props"]["children"][3]["props"]["children"][1]["props"][
                    "children"
                ] = str(message.TimestampNs())
                
                # Update the graph based on data dimensionality
                if data.ndim >= 2 and data.shape[0] > 1 and data.shape[1] > 1:
                    # 2D data: need to replace entire figure with heatmap
                    import plotly.express as px
                    # Squeeze if needed
                    if data.ndim == 3 and data.shape[2] == 1:
                        data = data.squeeze(axis=2)
                    new_figure = px.imshow(
                        data,
                        labels={"x": "x", "y": "y", "color": "spin"},
                        color_continuous_scale="RdBu_r",
                        aspect="equal",
                        origin="upper",
                    )
                    new_figure.update_layout(
                        coloraxis_colorbar=dict(
                            title="Spin",
                            tickvals=[0, 0.5, 1],
                            ticktext=["↓", "~", "↑"],
                        )
                    )
                    patches[index][1]["props"]["figure"] = new_figure
                else:
                    # 1D data: update y-axis data of line plot
                    patches[index][1]["props"]["figure"]["data"][0]["y"] = data.flatten()

            return patches

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
    import argparse
    
    # Configure logging to display info-level messages
    logging.basicConfig(level=logging.INFO)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Pipeline Visualization - Static Web Dashboard")
    parser.add_argument(
        "--ising",
        action="store_true",
        help="Configure for Ising model visualization (2D heatmap)",
    )
    args = parser.parse_args()

    # Create and run the visualizer application
    visualizer = Visualizer(use_ising=args.ising)
    visualizer.run()
