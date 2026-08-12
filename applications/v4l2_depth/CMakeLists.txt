# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

enable_language(CUDA)
add_subdirectory(cpp)

# Inference execution requires the model, while the default validate mode
# deliberately compiles the graph without downloading model sources or weights.
# Application metadata selects ON for live/loopback and OFF for validate; users
# normally do not need to set this option directly.
option(V4L2_DEPTH_PREPARE_MODEL
    "Prepare the pinned Depth Anything V2 ONNX model in the project data directory"
    OFF
)

if(V4L2_DEPTH_PREPARE_MODEL)
    if(NOT HOLOHUB_DATA_DIR)
        set(HOLOHUB_DATA_DIR "${PROJECT_SOURCE_DIR}/data")
    endif()
    set(v4l2_depth_model_dir "${HOLOHUB_DATA_DIR}/v4l2_depth/models")
    set(v4l2_depth_model
        "${v4l2_depth_model_dir}/depth_anything_v2_vits.onnx"
    )
    set(v4l2_depth_external_data
        "${v4l2_depth_model_dir}/depth_anything_v2_vits.onnx.data"
    )
    set(v4l2_depth_license_dir "${HOLOHUB_DATA_DIR}/v4l2_depth/licenses")
    set(v4l2_depth_licenses
        "${v4l2_depth_license_dir}/depth-anything-v2-apache-2.0.txt"
        "${v4l2_depth_license_dir}/depth-anything-tensorrt-mit.txt"
    )
    add_custom_command(
        OUTPUT
            "${v4l2_depth_model}"
            "${v4l2_depth_external_data}"
            ${v4l2_depth_licenses}
        COMMAND
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/prepare_model.sh"
            "${HOLOHUB_DATA_DIR}/v4l2_depth"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/prepare_model.sh"
        COMMENT "Preparing the pinned Depth Anything V2 Small ONNX model"
        VERBATIM
    )
    add_custom_target(v4l2_depth_model ALL
        DEPENDS
            "${v4l2_depth_model}"
            "${v4l2_depth_external_data}"
            ${v4l2_depth_licenses}
    )
    add_dependencies(v4l2_depth v4l2_depth_model)
endif()
