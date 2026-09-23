# SPDX-FileCopyrightText: Copyright (c) 2026, TECNALIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(nv_video_decoder_packetized_framing_test
  test_packetized_framing.cpp
)

add_executable(nv_video_decoder_display_crop_test
  test_display_crop.cpp
)

target_link_libraries(nv_video_decoder_packetized_framing_test PRIVATE
  holoscan::core
  holoscan::ops::nv_video_decoder
)

target_link_libraries(nv_video_decoder_display_crop_test PRIVATE
  holoscan::core
  holoscan::ops::nv_video_decoder
)

target_include_directories(nv_video_decoder_packetized_framing_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(nv_video_decoder_display_crop_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

set(NV_VIDEO_DECODER_TEST_FIXTURE
    ${CMAKE_CURRENT_SOURCE_DIR}/data/three_access_units.h265)
set(NV_VIDEO_DECODER_BFRAME_TEST_FIXTURE
    ${CMAKE_CURRENT_SOURCE_DIR}/data/bframes.h264)
set(NV_VIDEO_DECODER_CROP250_TEST_FIXTURE
    ${CMAKE_CURRENT_SOURCE_DIR}/data/crop250.h264)
set(NV_VIDEO_DECODER_CROP256_TEST_FIXTURE
    ${CMAKE_CURRENT_SOURCE_DIR}/data/crop256.h264)

add_test(NAME nv_video_decoder_access_unit_framing_test
         COMMAND nv_video_decoder_packetized_framing_test
                 ${NV_VIDEO_DECODER_TEST_FIXTURE} access_unit 3 separate no_eos uint8)

# With normal display latency, a decoded frame can be returned by a later access-unit
# submission. Verify that it retains the metadata of its originating access unit.
add_test(NAME nv_video_decoder_access_unit_metadata_test
         COMMAND nv_video_decoder_packetized_framing_test
                 ${NV_VIDEO_DECODER_TEST_FIXTURE} access_unit 3 separate eos uint8)

# B-frame input is submitted in decode order with the real presentation timestamp
# of each picture. Verify that display-order pixels retain their originating AU metadata.
add_test(NAME nv_video_decoder_bframe_metadata_test
         COMMAND nv_video_decoder_packetized_framing_test
                 ${NV_VIDEO_DECODER_BFRAME_TEST_FIXTURE}
                 access_unit 12 separate eos uint8 bframes)

# Characterize the current CUVID parser lookahead for arbitrary stream chunks:
# without EOP, the final picture remains pending because the test sends no EOS,
# flush, or fourth access unit.
add_test(NAME nv_video_decoder_stream_lookahead_test
         COMMAND nv_video_decoder_packetized_framing_test
                 ${NV_VIDEO_DECODER_TEST_FIXTURE} stream 2 separate no_eos uint8)

# A single arbitrary stream chunk may contain several complete pictures. Verify
# that every frame returned by that one decoder submission is emitted.
add_test(NAME nv_video_decoder_stream_multi_picture_chunk_test
         COMMAND nv_video_decoder_packetized_framing_test
                 ${NV_VIDEO_DECODER_TEST_FIXTURE} stream 2 combined no_eos uint8)

# The final tensor carries an explicit EOS marker. With normal display reordering,
# the EOS submission must release and emit the final delayed frame.
add_test(NAME nv_video_decoder_stream_eos_drain_test
         COMMAND nv_video_decoder_packetized_framing_test
                 ${NV_VIDEO_DECODER_TEST_FIXTURE} stream 3 combined eos uint8)

# Encoded bytes may be stored in a tensor with elements wider than one byte.
# Verify that the decoder submits nbytes(), rather than the element count, to NVDEC.
add_test(NAME nv_video_decoder_wide_element_tensor_test
         COMMAND nv_video_decoder_packetized_framing_test
                 ${NV_VIDEO_DECODER_TEST_FIXTURE} stream 3 combined eos uint16)

# A crop-only sequence change with unchanged coded dimensions must be rejected before
# NvDecoder updates its global layout and queues frames with incompatible allocations.
add_test(NAME nv_video_decoder_display_crop_change_test
         COMMAND nv_video_decoder_display_crop_test
                 ${NV_VIDEO_DECODER_CROP250_TEST_FIXTURE}
                 ${NV_VIDEO_DECODER_CROP256_TEST_FIXTURE}
                 reject)

add_test(NAME nv_video_decoder_display_crop_250_control_test
         COMMAND nv_video_decoder_display_crop_test
                 ${NV_VIDEO_DECODER_CROP250_TEST_FIXTURE}
                 ${NV_VIDEO_DECODER_CROP256_TEST_FIXTURE}
                 crop250)

add_test(NAME nv_video_decoder_display_crop_256_control_test
         COMMAND nv_video_decoder_display_crop_test
                 ${NV_VIDEO_DECODER_CROP250_TEST_FIXTURE}
                 ${NV_VIDEO_DECODER_CROP256_TEST_FIXTURE}
                 crop256)

add_test(NAME nv_video_decoder_display_crop_reset_control_test
         COMMAND nv_video_decoder_display_crop_test
                 ${NV_VIDEO_DECODER_CROP250_TEST_FIXTURE}
                 ${NV_VIDEO_DECODER_CROP256_TEST_FIXTURE}
                 reset)

set_tests_properties(
  nv_video_decoder_access_unit_framing_test
  nv_video_decoder_access_unit_metadata_test
  nv_video_decoder_bframe_metadata_test
  nv_video_decoder_stream_lookahead_test
  nv_video_decoder_stream_multi_picture_chunk_test
  nv_video_decoder_stream_eos_drain_test
  nv_video_decoder_wide_element_tensor_test
  nv_video_decoder_display_crop_change_test
  nv_video_decoder_display_crop_250_control_test
  nv_video_decoder_display_crop_256_control_test
  nv_video_decoder_display_crop_reset_control_test
  PROPERTIES
    LABELS "gpu"
    TIMEOUT 30
)
