#!/usr/bin/env python3
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

import logging
import os
from typing import Any, Dict

import yaml

# Configure logging
logger = logging.getLogger(__name__)


def read_yaml_file(yaml_path: str) -> Dict[str, Any]:
    """
    Read a YAML file and return its contents as a dictionary.

    Args:
        yaml_path: Path to the YAML file

    Returns:
        Dictionary representation of the YAML file
    """
    try:
        with open(yaml_path, "r") as file:
            # Parse YAML preserving the order of items
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to read YAML file {yaml_path}: {e}")
        return {}


def write_yaml_file(yaml_path: str, data: Dict[str, Any]) -> bool:
    """
    Write a dictionary to a YAML file.

    Args:
        yaml_path: Path to the YAML file
        data: Dictionary to write

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

        with open(yaml_path, "w") as file:
            # Write YAML preserving the original style if possible
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        logger.error(f"Failed to write YAML file {yaml_path}: {e}")
        return False


def update_yaml_file(yaml_path: str, output_path: str, updates: Dict[str, Any]) -> bool:
    """
    Update multiple fields in a YAML file using dot-notation paths.

    The field_path in the updates dictionary uses dot notation to navigate the YAML structure:
    - Use dots (.) to navigate through nested objects
    - Use array notation [index] to access list items

    Example:
        update_yaml_file(yaml_path, output_path, {
            "scheduler.max_duration_ms": 5000,
            "advanced_network.cfg.interfaces[0].address": "0000:3b:00.0",
            "bench_tx.eth_dst_addr": "00:11:22:33:44:55"
        })

    Args:
        yaml_path: Path to the YAML file to update
        output_path: Path where the updated YAML should be saved
        updates: Dictionary mapping field paths to values

    Returns:
        True if successful, False otherwise

    Raises:
        ValueError: If a field path is not found in the YAML file
        IOError: If there's an error reading or writing the YAML file
    """
    # Load YAML data
    yaml_data = read_yaml_file(yaml_path)
    if not yaml_data:
        raise IOError(f"Failed to read YAML file: {yaml_path}")

    # Process all updates
    for field_path, value in updates.items():
        _update_yaml_field(yaml_data, field_path, value)

    # Write the updated YAML
    if not write_yaml_file(output_path, yaml_data):
        raise IOError(f"Failed to write YAML file: {output_path}")

    return True


def _update_yaml_field(yaml_data: Dict[str, Any], field_path: str, value: Any) -> bool:
    """
    Update a single field in a YAML data structure using a dot-notation path.
    If path components don't exist, they will be created.

    Args:
        yaml_data: The YAML data structure to update
        field_path: Path to the field to update in dot notation
        value: Value to set

    Returns:
        True if successful

    Raises:
        ValueError: If there's an invalid structure in the path (e.g., trying to index a non-list)
    """
    # Split the path into components
    components = field_path.split(".")

    # Start at the root
    current = yaml_data
    parent = None
    last_key = None

    # Navigate to the target field
    for i, component in enumerate(components):
        is_last = i == len(components) - 1
        parent = current

        # Handle array indexing
        if "[" in component and component.endswith("]"):
            # Split "interfaces[0]" into "interfaces" and "0"
            base_name, index_str = component.split("[", 1)
            index = int(index_str.rstrip("]"))

            # Create the list if it doesn't exist
            if base_name not in current:
                logger.debug(f"Creating list '{base_name}' at {field_path}")
                current[base_name] = []

            # Check if it's a list
            if not isinstance(current[base_name], list):
                error_msg = f"'{base_name}' is not a list in YAML at {field_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Extend the list if needed to accommodate the index
            if index >= len(current[base_name]):
                # Fill with None values up to the index
                current[base_name].extend([None] * (index + 1 - len(current[base_name])))
                logger.debug(f"Extended list '{base_name}' to accommodate index {index}")

            # For the last component, we want to update the parent[base_name][index]
            if is_last:
                parent = current[base_name]
                last_key = index
            else:
                # If the indexed item is None, initialize it as a dict
                if current[base_name][index] is None:
                    logger.debug(f"Initializing dict at '{base_name}[{index}]'")
                    current[base_name][index] = {}
                # Navigate to the indexed item
                current = current[base_name][index]
        else:
            # Regular object property
            if component not in current:
                # Create the component if it doesn't exist
                if is_last:
                    # For the last component, we'll set the value directly later
                    logger.debug(f"Creating field '{component}' at {field_path}")
                    current[component] = None
                    last_key = component
                else:
                    # For intermediate components, create a new dictionary
                    logger.debug(f"Creating dict '{component}' at {field_path}")
                    current[component] = {}
                    current = current[component]
                    continue

            if is_last:
                last_key = component
            else:
                # If the current value isn't a dict, make it one
                if not isinstance(current[component], dict):
                    logger.debug(f"Converting '{component}' to dict at {field_path}")
                    current[component] = {}
                current = current[component]

    # Update the value
    logger.info(f"Setting {field_path} to {value}")
    parent[last_key] = value
    return True


if __name__ == "__main__":
    # Set up console logging if run directly
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("This module is designed to be imported, not run directly.")
