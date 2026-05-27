NOTICE - holoscan-gstreamer
===========================
Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This software is licensed under the Apache License, Version 2.0.
See the LICENSE file or https://www.apache.org/licenses/LICENSE-2.0 for details.

-------------------------------------------------------------------------------
Third-Party Components
-------------------------------------------------------------------------------

The holoscan-gstreamer binary links the following third-party libraries at
runtime. These libraries are not bundled in this package and must be installed
separately (e.g., via the system package manager). Their licenses govern your
use of those libraries independently of this package.

1. GStreamer
   Description : Open-source multimedia framework (core, base, video, app,
                 and CUDA libraries)
   Version     : 1.0 or later (1.24+ recommended for CUDA support)
   License     : GNU Lesser General Public License v2.1 (LGPL-2.1)
   Source      : https://gitlab.freedesktop.org/gstreamer/gstreamer
                 (subprojects/gstreamer/, gst-plugins-base/)
   apt packages: libgstreamer1.0-0, libgstreamer-plugins-base1.0-0,
                 libgstreamer-plugins-bad1.0-0, gstreamer1.0-tools

2. GStreamer Good Plugins (gst-plugins-good)
   Description : GStreamer plugins with good licensing and code quality;
                 provides h264parse, qtdemux, rtspsrc, udpsrc/udpsink,
                 rtph264pay/rtph264depay
   License     : GNU Lesser General Public License v2.1 (LGPL-2.1)
   Source      : https://gitlab.freedesktop.org/gstreamer/gstreamer
                 (subprojects/gst-plugins-good/)
   apt package : gstreamer1.0-plugins-good

3. GStreamer Bad Plugins (gst-plugins-bad)
   Description : GStreamer plugins without guaranteed API/ABI stability or
                 complete documentation; provides h265parse, CUDA elements
                 (cudaconvert, cudadownload), and the nvcodec plugin
                 (nvh264enc, nvh265enc, nvh264dec)
   License     : GNU Lesser General Public License v2.1 (LGPL-2.1)
   Source      : https://gitlab.freedesktop.org/gstreamer/gstreamer
                 (subprojects/gst-plugins-bad/)
   apt package : gstreamer1.0-plugins-bad

4. GStreamer Ugly Plugins (gst-plugins-ugly)
   Description : GStreamer plugins that may have distribution restrictions;
                 provides additional codec elements
   License     : GNU Lesser General Public License v2.1 (LGPL-2.1)
                 Note: the x264enc element links libx264 (GPL-2.0-or-later);
                 use of that element makes the resulting binary subject to
                 GPL-2.0-or-later terms.
   Source      : https://gitlab.freedesktop.org/gstreamer/gstreamer
                 (subprojects/gst-plugins-ugly/)
   apt package : gstreamer1.0-plugins-ugly

5. GStreamer libav Plugin (gst-libav)
   Description : GStreamer plugin wrapping FFmpeg/libav; provides avdec_h264
                 and other FFmpeg-backed codec elements
   License     : GNU Lesser General Public License v2.1 (LGPL-2.1)
                 Note: depending on how the FFmpeg libraries are built by the
                 distribution, some components may be GPL-2.0-or-later.
   Source      : https://gitlab.freedesktop.org/gstreamer/gstreamer
                 (subprojects/gst-libav/)
   Upstream    : https://github.com/FFmpeg/FFmpeg
   apt package : gstreamer1.0-libav

6. NVIDIA Video Codec SDK (NVENC / NVDEC)
   Description : Proprietary NVIDIA hardware encoder and decoder APIs used
                 by the nvcodec GStreamer plugin elements nvh264enc, nvh265enc,
                 and nvh264dec. These elements are only functional on NVIDIA
                 GPU hardware.
   License     : NVIDIA Video Codec SDK Software License Agreement (proprietary)
   Source      : https://developer.nvidia.com/video-codec-sdk
   Note        : The SDK libraries are not redistributed by this package.
                 They are provided as part of the NVIDIA GPU driver.

-------------------------------------------------------------------------------
Additional Components
-------------------------------------------------------------------------------

The following components are compiled into or dynamically linked by the
holoscan-gstreamer binaries and are not part of the GStreamer ecosystem.

7. pybind11
   Description : C++/Python interoperability library. Headers are compiled
                 directly into the Python binding module
                 (_holoscan_gstreamer_bridge.so).
   Version     : v2.13.6
   License     : BSD 3-Clause License
   Copyright   : Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>
                 and contributors
   Source      : https://github.com/pybind/pybind11

8. DLPack
   Description : Tensor data structure interchange protocol. DLDeviceType
                 and DLDevice are used directly in gst_src_bridge.cpp and
                 gst_sink_bridge.cpp via the Holoscan tensor API headers.
   License     : Apache License, Version 2.0
   Source      : https://github.com/dmlc/dlpack

9. {fmt}
   Description : Text formatting library. fmt::format() is called directly
                 in gst_src_bridge.cpp and gst_sink_bridge.cpp. Provided as
                 a shared library dependency of the Holoscan SDK.
   License     : MIT License
   Copyright   : Copyright (c) 2012-present Victor Zverovich and {fmt}
                 contributors
   Source      : https://github.com/fmtlib/fmt

10. Holoscan SDK (NVIDIA)
    Description : NVIDIA Holoscan sensor processing framework. Provides the
                  operator base class, tensor types, GXF runtime, and logging
                  macros used throughout this package. Dynamically linked.
    License     : NVIDIA Holoscan SDK License Agreement (proprietary)
    Source      : https://github.com/nvidia-holoscan/holoscan-sdk

11. NVIDIA CUDA Toolkit (CUDA Runtime)
    Description : cuda_runtime_api.h is included and libcudart is linked for
                  GPU memory management. Provided by the holoscan-cuda-13
                  package dependency.
    License     : NVIDIA Software License Agreement (proprietary)
    Source      : https://developer.nvidia.com/cuda-toolkit

-------------------------------------------------------------------------------
Full license texts for open-source components:
  Apache-2.0  : https://www.apache.org/licenses/LICENSE-2.0
  BSD-3-Clause: https://opensource.org/licenses/BSD-3-Clause
  MIT         : https://opensource.org/licenses/MIT
  LGPL-2.1   : https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html
  GPL-2.0    : https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
