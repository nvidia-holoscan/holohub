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

"""
Shared style definitions for pipeline visualization applications.

This module contains reusable CSS-like style dictionaries for Dash components,
ensuring consistent styling across all visualizer applications.
"""

# Label style for bold text labels
LABEL_STYLE = {"fontWeight": "bold", "color": "#333"}

# Value style for data values displayed in blue
VALUE_STYLE = {"color": "#0066cc"}

# Style for individual metadata field containers
METADATA_FIELD_STYLE = {"marginRight": "20px"}

# Style for the main metadata container with background and border
METADATA_CONTAINER_STYLE = {
    "display": "flex",
    "flex-direction": "row",
    "flexWrap": "wrap",
    "padding": "15px",
    "backgroundColor": "#f5f5f5",
    "borderRadius": "8px",
    "marginBottom": "10px",
    "border": "1px solid #ddd",
}

# Style for input fields (NATS subject)
INPUT_STYLE = {
    "padding": "10px 15px",
    "fontSize": "14px",
    "border": "2px solid #ddd",
    "borderRadius": "6px",
    "outline": "none",
    "transition": "border-color 0.3s",
    "minWidth": "250px",
}

# Style for buttons (Connect/Disconnect)
BUTTON_STYLE = {
    "padding": "10px 25px",
    "fontSize": "14px",
    "fontWeight": "bold",
    "color": "white",
    "backgroundColor": "#0066cc",
    "border": "none",
    "borderRadius": "6px",
    "cursor": "pointer",
    "transition": "background-color 0.3s",
}

# Style for the connection controls container
CONTROLS_CONTAINER_STYLE = {
    "display": "flex",
    "flexDirection": "row",
    "gap": "20px",
    "padding": "20px",
    "backgroundColor": "#f5f5f5",
    "borderRadius": "8px",
    "marginBottom": "20px",
    "border": "1px solid #ddd",
    "alignItems": "center",
}
