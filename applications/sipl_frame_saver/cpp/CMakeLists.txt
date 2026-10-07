# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(sipl_frame_saver sipl_frame_saver.cpp)

target_compile_features(sipl_frame_saver PRIVATE cxx_std_20)

target_link_libraries(sipl_frame_saver PRIVATE
    holoscan::sipl_capture_op
    holoscan::core)

# sipl_capture_op.so carries undefined references to L4T BSP libs (nvsipl, nvscibuf, nvscisync)
# that are mounted at runtime via the holohub wrapper.
target_link_options(sipl_frame_saver PRIVATE "LINKER:--allow-shlib-undefined")

include(GNUInstallDirs)
install(TARGETS sipl_frame_saver
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
