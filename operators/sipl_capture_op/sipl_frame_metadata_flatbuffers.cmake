# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generates the FlatBuffers C++ native table and Holoscan schema traits for
# SIPLFrameMetadata.  Included by operators/sipl_capture_op/CMakeLists.txt.

# Generated files land in generated/sipl_capture_op/ so that the consuming header
# can use the directory-qualified form #include "sipl_capture_op/<file>" which
# satisfies cpplint's build/include_subdir rule.
set(SIPL_METADATA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

holoscan_add_flatbuffer_schema(
    TARGET sipl_frame_metadata_generated
    SCHEMA sipl_frame_metadata.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${SIPL_METADATA_GENERATED_DIR}/sipl_capture_op"
    BFBS_EMBED
    TRAITS_HEADER sipl_frame_metadata_schema_traits.hpp
    COMPATIBILITY_EPOCH 1
    MAX_SERIALIZED_SIZE 450000
)
