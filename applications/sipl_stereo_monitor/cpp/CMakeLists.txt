# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(sipl_stereo_monitor sipl_stereo_monitor.cpp)

target_compile_features(sipl_stereo_monitor PRIVATE cxx_std_20)

target_link_libraries(sipl_stereo_monitor PRIVATE
    holoscan::sipl_capture_op
    holoscan::core)

# sipl_capture_op.so carries undefined references to L4T BSP libs (nvsipl, nvscibuf, nvscisync)
# that are mounted at runtime via the holohub wrapper.
target_link_options(sipl_stereo_monitor PRIVATE "LINKER:--allow-shlib-undefined")

include(GNUInstallDirs)
install(TARGETS sipl_stereo_monitor
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
