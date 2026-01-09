# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Mapping

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from ultra_post.core.pipeline import Pipeline, create_node
from ultra_post.filters.registry import DEFAULT_PARAMS


def render_filter_controls(
    container: DeltaGenerator, pipeline: Pipeline, filters: Mapping[str, Any]
) -> None:
    """Render active filters with edit/move/remove controls."""

    container.markdown("**Filters**")
    if not pipeline:
        container.info("Pipeline is empty. Add a filter below.")
        return

    for i, node in enumerate(list(pipeline)):
        op_name = str(node.get("op", ""))
        label = f"{i + 1}. {op_name.replace('_', ' ').title()}"
        key_prefix = f"op_{op_name}_{i}"

        with container.expander(label, expanded=True):
            defaults = DEFAULT_PARAMS.get(op_name, {})
            if not filters.get(op_name):
                st.error(f"Operator '{op_name}' is missing.")
            else:
                node["params"] = _render_params(st, node.get("params", {}), defaults, key_prefix)

            cols = st.columns([2, 1, 1, 1])
            node["enabled"] = cols[0].checkbox(
                "Enable", value=node.get("enabled", True), key=f"{key_prefix}__enabled"
            )

            if cols[1].button("↑", key=f"{key_prefix}_up", disabled=i == 0):
                pipeline.insert(i - 1, pipeline.pop(i))
                st.rerun()
            if cols[2].button("↓", key=f"{key_prefix}_down", disabled=i == len(pipeline) - 1):
                pipeline.insert(i + 1, pipeline.pop(i))
                st.rerun()
            if cols[3].button("✖", key=f"{key_prefix}_del"):
                pipeline.pop(i)
                st.rerun()


def render_add_filter_controls(
    container: DeltaGenerator, pipeline: Pipeline, filters: Mapping[str, Any]
) -> None:
    """Render selector to append new filters."""

    container.markdown("**Add Filter**")
    names = sorted(filters.keys())
    if not names:
        return

    cols = container.columns([3, 1])
    selected = cols[0].selectbox(
        "Type", options=names, label_visibility="collapsed", key="add_op_select"
    )
    if cols[1].button("Add", key="add_op_btn"):
        pipeline.append(create_node(selected))
        st.rerun()


def _render_params(
    container: DeltaGenerator,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key_prefix: str,
) -> Dict[str, Any]:
    """Render controls for parameters based on default values."""

    updated: Dict[str, Any] = {}
    editable_params = {k: v for k, v in params.items() if k != "enable"}

    for name, value in editable_params.items():
        default = defaults.get(name, value)
        label = name.replace("_", " ").title()
        key = f"{key_prefix}_{name}"

        if isinstance(default, bool):
            updated[name] = container.checkbox(label, value=bool(value), key=key)
        elif isinstance(default, int):
            updated[name] = container.number_input(label, value=int(value), step=1, key=key)
        elif isinstance(default, float):
            max_val = max(float(default) * 2.0, 1.0) if default != 0 else 1.0
            updated[name] = container.slider(label, 0.0, max_val, float(value), key=key)
        elif isinstance(default, Enum):
            options = list(type(default))
            current_val = value
            if isinstance(value, str):
                for opt in options:
                    if opt.value == value or opt.name == value:
                        current_val = opt
                        break
            try:
                idx = (
                    options.index(current_val) if current_val in options else options.index(default)
                )
            except ValueError:
                idx = 0
            selected = container.selectbox(
                label,
                options=options,
                index=idx,
                format_func=lambda x: x.value,
                key=key,
            )
            updated[name] = selected
        elif (
            isinstance(default, tuple)
            and len(default) == 2
            and all(isinstance(x, int) for x in default)
        ):
            c1, c2 = container.columns(2)
            v1 = c1.number_input(f"{label} X", value=value[0], key=f"{key}_0")
            v2 = c2.number_input(f"{label} Y", value=value[1], key=f"{key}_1")
            updated[name] = (v1, v2)
        else:
            updated[name] = container.text_input(label, value=str(value), key=key)

    return updated
