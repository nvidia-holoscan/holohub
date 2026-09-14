# SPDX-FileCopyrightText: Copyright (c) 2026, TECNALIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(nv_video_decoder_packetized_framing_test
  test_packetized_framing.cpp
)

target_link_libraries(nv_video_decoder_packetized_framing_test PRIVATE
  holoscan::core
  holoscan::ops::nv_video_decoder
)

target_include_directories(nv_video_decoder_packetized_framing_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

set(NV_VIDEO_DECODER_TEST_FIXTURE
    ${CMAKE_CURRENT_SOURCE_DIR}/data/three_access_units.h265)

add_test(NAME nv_video_decoder_access_unit_framing_test
         COMMAND nv_video_decoder_packetized_framing_test
                 ${NV_VIDEO_DECODER_TEST_FIXTURE} access_unit 3)

# Characterize the current CUVID parser lookahead for arbitrary stream chunks:
# without EOP, the final picture remains pending because the test sends no EOS,
# flush, or fourth access unit.
add_test(NAME nv_video_decoder_stream_lookahead_test
         COMMAND nv_video_decoder_packetized_framing_test
                 ${NV_VIDEO_DECODER_TEST_FIXTURE} stream 2)

set_tests_properties(
  nv_video_decoder_access_unit_framing_test
  nv_video_decoder_stream_lookahead_test
  PROPERTIES
    LABELS "gpu"
    TIMEOUT 30
)
