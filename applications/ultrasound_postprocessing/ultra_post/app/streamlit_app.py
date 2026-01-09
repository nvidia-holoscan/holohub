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

import sys
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cupy as cp
import streamlit as st

from ultra_post.app.components import render_add_filter_controls, render_filter_controls
from ultra_post.core.display import (
    DisplayCompressionSettings,
    run_pipeline_colormap_last,
    tensor_to_display,
)
from ultra_post.core.loader import load_uff_frame
from ultra_post.core.pipeline import Pipeline, create_node, pipeline_to_yaml
from ultra_post.filters.registry import FILTERS

DEFAULT_UFF = PROJECT_ROOT / "ultra_post/examples/demo.uff"


def init_state() -> None:
    """Initialize session state variables."""

    if "pipeline" not in st.session_state:
        pipe: Pipeline = []
        if "gamma_compression" in FILTERS:
            pipe.append(create_node("gamma_compression"))
        st.session_state["pipeline"] = pipe

    if "frame" not in st.session_state:
        try:
            st.session_state["frame"] = load_uff_frame(DEFAULT_UFF)
            st.session_state["source"] = DEFAULT_UFF.name
        except Exception:
            st.session_state["frame"] = None
            st.session_state["source"] = "None"


def main() -> None:
    st.set_page_config(page_title="US Post-Process", layout="wide")
    init_state()

    st.title("Ultrasound Post-Processing")

    with st.sidebar:
        st.header("Pipeline")
        pipeline = st.session_state["pipeline"]
        render_filter_controls(st, pipeline, FILTERS)
        st.divider()
        render_add_filter_controls(st, pipeline, FILTERS)
        st.divider()

        st.header("Display Settings")
        dyn_range = st.slider("Dynamic Range (dB)", 20, 80, 60)
        mode = st.selectbox("Compression", ["power", "partial_log", "gamma"])

        gamma = 1.0
        mix = 0.5
        if mode == "gamma":
            gamma = st.slider("Gamma", 0.2, 3.0, 1.0)
        elif mode == "partial_log":
            mix = st.slider("Log Mix", 0.0, 1.0, 0.5)

        apply_orig = st.checkbox("Compress Original", True)

        st.divider()
        display_config = {
            "mode": mode,
            "dynamic_range_db": float(dyn_range),
            "gamma": float(gamma),
            "partial_log_mix": float(mix),
        }
        st.download_button(
            "Save Pipeline",
            data=pipeline_to_yaml(pipeline, display=display_config),
            file_name="pipeline.yaml",
            mime="application/x-yaml",
        )

        st.header("Data")
        uploaded = st.file_uploader("Upload UFF", type=["uff"])
        if uploaded:
            with tempfile.NamedTemporaryFile(suffix=".uff") as tmp:
                tmp.write(uploaded.getvalue())
                tmp.flush()
                st.session_state["frame"] = load_uff_frame(Path(tmp.name))
                st.session_state["source"] = uploaded.name

    frame = st.session_state.get("frame")
    if not frame:
        st.warning("No data loaded.")
        return

    source = frame["data"]
    pipeline = st.session_state["pipeline"]
    processed = run_pipeline_colormap_last(pipeline, source)

    settings = DisplayCompressionSettings(
        mode=mode,
        dynamic_range_db=float(dyn_range),
        gamma=float(gamma),
        partial_log_mix=float(mix),
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Original ({st.session_state['source']})")
        disp = tensor_to_display(source, settings, apply_compression=apply_orig)
        st.image(cp.asnumpy(disp), use_container_width=True, clamp=True)
    with col2:
        st.subheader("Processed")
        disp = tensor_to_display(processed, settings, apply_compression=True)
        st.image(cp.asnumpy(disp), use_container_width=True, clamp=True)


if __name__ == "__main__":
    main()
