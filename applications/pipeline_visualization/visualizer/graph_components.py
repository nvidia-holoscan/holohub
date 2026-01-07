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

from typing import List, Optional

import numpy as np
import pipeline_visualization.flatbuffers.IOType
import plotly.express as px
from dash import dcc, html
from styles import (
    BUTTON_STYLE,
    CONTROLS_CONTAINER_STYLE,
    INPUT_STYLE,
    LABEL_STYLE,
    METADATA_CONTAINER_STYLE,
    METADATA_FIELD_STYLE,
    VALUE_STYLE,
)


def create_graph(
    name: str,
    message: Optional[object] = None,
    data: Optional[np.ndarray] = None,
    id_type: Optional[str] = None,
):
    """
    Create a graph component with metadata display for a data stream.

    This function can be used in two modes:
    1. Static mode (message=None): Creates a graph with placeholder metadata values
    2. Dynamic mode (message provided): Creates a graph with actual metadata from the message

    Args:
        name: The unique identifier for the data stream (e.g., "source.out")
        message: Optional Holoscan flatbuffer Message object containing metadata
        data: Optional numpy array containing the tensor data to plot
        id_type: Optional type for creating pattern-matching IDs (e.g., "graph-container")
                 If provided, creates a dict ID instead of string ID

    Returns:
        A Dash HTML Div containing metadata fields and a Plotly line graph
    """
    # Replace dots with dashes for Dash component ID compatibility
    dash_name = name.replace(".", "-")

    # Determine metadata values based on whether a message is provided
    if message is not None:
        io_type = (
            "Output"
            if message.IoType() == pipeline_visualization.flatbuffers.IOType.IOType().kOutput
            else "Input"
        )
        acquisition_timestamp = str(message.AcquisitionTimestampNs())
        publish_timestamp = str(message.TimestampNs())
    else:
        # Use placeholder values for static mode
        io_type = ""
        acquisition_timestamp = ""
        publish_timestamp = ""

    # Create the appropriate ID format
    if id_type is not None:
        component_id = {"type": id_type, "id": dash_name}
    else:
        component_id = dash_name

    # Create the figure
    if data is not None:
        # Create graph with data
        figure = px.line(
            x=np.arange(len(data)),
            y=data,
            labels={"x": "t", "y": "amplitude"},
        )
    else:
        # Create empty graph
        figure = px.line(
            labels={"x": "t", "y": "amplitude"},
        )

    return html.Div(
        id=component_id,
        children=[
            # Metadata display section with labeled fields
            html.Div(
                children=[
                    # Stream name field
                    html.Div(
                        children=[
                            html.Span("Stream name: ", style=LABEL_STYLE),
                            html.Span(name, style=VALUE_STYLE),
                        ],
                        style=METADATA_FIELD_STYLE,
                    ),
                    # IO Type field
                    html.Div(
                        children=[
                            html.Span("IO Type: ", style=LABEL_STYLE),
                            html.Span(io_type, style=VALUE_STYLE),
                        ],
                        style=METADATA_FIELD_STYLE,
                    ),
                    # Acquisition timestamp field
                    html.Div(
                        children=[
                            html.Span("Acquisition Timestamp (ns): ", style=LABEL_STYLE),
                            html.Span(acquisition_timestamp, style=VALUE_STYLE),
                        ],
                        style=METADATA_FIELD_STYLE,
                    ),
                    # Publish timestamp field
                    html.Div(
                        children=[
                            html.Span("Publish Timestamp (ns): ", style=LABEL_STYLE),
                            html.Span(publish_timestamp, style=VALUE_STYLE),
                        ],
                        style=METADATA_FIELD_STYLE,
                    ),
                ],
                style=METADATA_CONTAINER_STYLE,
            ),
            # Create a line graph
            dcc.Graph(figure=figure),
        ],
    )


def create_app_layout(unique_ids: Optional[List[str]] = None):
    """
    Create the application layout for the visualizer.

    This function creates a standard Dash layout with:
    - An interval component for periodic updates
    - Connection controls (NATS subject input and connect/disconnect button)
    - A data container for displaying graphs

    Args:
        unique_ids: Optional list of unique identifiers for initial graphs
                    If provided, creates initial graphs for each ID (static mode)
                    If None, starts with an empty container (dynamic mode)

    Returns:
        A Dash HTML Div containing the complete application layout
    """
    # Determine initial children for the data container
    if unique_ids is not None:
        # Static mode: create initial graphs
        initial_children = [create_graph(unique_id) for unique_id in unique_ids]
    else:
        # Dynamic mode: start with empty container
        initial_children = []

    return html.Div(
        [
            # Interval component triggers periodic updates (every 200ms)
            dcc.Interval(
                id="interval-component",
                interval=200,  # in milliseconds
            ),
            # Connection controls: subject input field and connect/disconnect button
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "NATS Subject:",
                                style={
                                    **LABEL_STYLE,
                                    "marginRight": "10px",
                                    "alignSelf": "center",
                                },
                            ),
                            dcc.Input(
                                id="subject",
                                type="text",
                                value="nats_demo",
                                style=INPUT_STYLE,
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                        },
                    ),
                    html.Button(
                        children="Connect",
                        id="connect",
                        disabled=False,
                        style=BUTTON_STYLE,
                    ),
                ],
                style=CONTROLS_CONTAINER_STYLE,
            ),
            # Container for data stream graphs
            html.Div(
                id="data-container",
                children=initial_children,
            ),
        ]
    )
