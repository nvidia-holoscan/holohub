/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <cuda.h>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include "cuDisp.h"

static inline CUresult create_cuda_context(CUcontext* ctx, unsigned int flags, CUdevice dev) {
#if CUDA_VERSION >= 13000
  return cuCtxCreate(ctx, nullptr, flags, dev);
#else
  return cuCtxCreate(ctx, flags, dev);
#endif
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void make_default_attribs(cuDispCreateAttribute* attrs) {
  memset(attrs, 0, sizeof(cuDispCreateAttribute) * 2);

  attrs[0].id = CUDISP_CREATE_ATTRIBUTE_MODE_INFO;
  attrs[0].value.modeInfo.modeWidth = 1920;
  attrs[0].value.modeInfo.modeHeight = 1080;
  attrs[0].value.modeInfo.refreshRateMilliHz = 60000;
  attrs[0].value.modeInfo.enableVrr = 0;
  attrs[0].value.modeInfo.maxBpc = CUDISP_MAX_BPC_DEFAULT;

  attrs[1].id = CUDISP_CREATE_ATTRIBUTE_BUFFER_INFO;
  attrs[1].value.bufferInfo.layerIndex = 0;
  attrs[1].value.bufferInfo.numBuffers = 2;
  attrs[1].value.bufferInfo.format = CUDISP_SURFACE_FORMAT_ARGB8888;
  attrs[1].value.bufferInfo.width = 1920;
  attrs[1].value.bufferInfo.height = 1080;
  attrs[1].value.bufferInfo.posX = 0;
  attrs[1].value.bufferInfo.posY = 0;
  attrs[1].value.bufferInfo.scaleWidth = 0;
  attrs[1].value.bufferInfo.scaleHeight = 0;
  attrs[1].value.bufferInfo.alpha = 0xFFFF;
  attrs[1].value.bufferInfo.blendMode = CUDISP_BLEND_MODE_DEFAULT;
  attrs[1].value.bufferInfo.rotation = CUDISP_ROTATE_0;
  attrs[1].value.bufferInfo.colorEncoding = CUDISP_COLOR_ENCODING_DEFAULT;
  attrs[1].value.bufferInfo.colorRange = CUDISP_COLOR_RANGE_DEFAULT;
}

// ===========================================================================
// cuDispGetVersion
// ===========================================================================

TEST(GetVersion, ValidCall) {
  uint64_t ver = 0;
  EXPECT_EQ(cuDispGetVersion(&ver), cuDispSuccess);
  uint32_t major = (ver >> 32) & 0xFFFF;
  uint32_t minor = (ver >> 16) & 0xFFFF;
  uint32_t patch = ver & 0xFFFF;
  EXPECT_EQ(major, CUDISP_VER_MAJOR);
  EXPECT_EQ(minor, CUDISP_VER_MINOR);
  EXPECT_EQ(patch, CUDISP_VER_PATCH);
}

TEST(GetVersion, NullPointerReturnsInvalidParam) {
  EXPECT_EQ(cuDispGetVersion(nullptr), cuDispErrorInvalidParam);
}

// ===========================================================================
// cuDispCreateSwapchain — parameter validation
// ===========================================================================

TEST(CreateSwapchain, NullSwapchainReturnsInvalidParam) {
  cuDispCreateAttribute attrs[2];
  make_default_attribs(attrs);
  EXPECT_EQ(cuDispCreateSwapchain(nullptr, attrs, 2, 0), cuDispErrorInvalidParam);
}

TEST(CreateSwapchain, NullAttributesReturnsInvalidParam) {
  cuDispSwapchain sc = nullptr;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, nullptr, 2, 0), cuDispErrorInvalidParam);
}

TEST(CreateSwapchain, NonZeroFlagsReturnsInvalidParam) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[2];
  make_default_attribs(attrs);
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 2, 1), cuDispErrorInvalidParam);
}

TEST(CreateSwapchain, MissingBufferInfoReturnsInvalidParam) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[1];
  memset(attrs, 0, sizeof(attrs));
  attrs[0].id = CUDISP_CREATE_ATTRIBUTE_MODE_INFO;
  attrs[0].value.modeInfo.modeWidth = 1920;
  attrs[0].value.modeInfo.modeHeight = 1080;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 1, 0), cuDispErrorInvalidParam);
}

TEST(CreateSwapchain, ZeroNumBuffersReturnsInvalidParam) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[2];
  make_default_attribs(attrs);
  attrs[1].value.bufferInfo.numBuffers = 0;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 2, 0), cuDispErrorInvalidParam);
}

TEST(CreateSwapchain, ExcessiveNumBuffersReturnsInvalidParam) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[2];
  make_default_attribs(attrs);
  attrs[1].value.bufferInfo.numBuffers = 100;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 2, 0), cuDispErrorInvalidParam);
}

// --- Unsupported features returning cuDispErrorNotSupported ---

TEST(CreateSwapchain, NonZeroLayerIndexReturnsNotSupported) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[2];
  make_default_attribs(attrs);
  attrs[1].value.bufferInfo.layerIndex = 1;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 2, 0), cuDispErrorNotSupported);
}

TEST(CreateSwapchain, UnsupportedFormatReturnsNotSupported) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[2];
  make_default_attribs(attrs);
  attrs[1].value.bufferInfo.format = CUDISP_SURFACE_FORMAT_NV12;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 2, 0), cuDispErrorNotSupported);
}

TEST(CreateSwapchain, NonZeroScalingReturnsNotSupported) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[2];
  make_default_attribs(attrs);
  attrs[1].value.bufferInfo.scaleWidth = 1280;
  attrs[1].value.bufferInfo.scaleHeight = 720;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 2, 0), cuDispErrorNotSupported);
}

TEST(CreateSwapchain, NonZeroPositionReturnsNotSupported) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[2];
  make_default_attribs(attrs);
  attrs[1].value.bufferInfo.posX = 100;
  attrs[1].value.bufferInfo.posY = 50;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 2, 0), cuDispErrorNotSupported);
}

TEST(CreateSwapchain, NonDefaultAlphaReturnsNotSupported) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[2];
  make_default_attribs(attrs);
  attrs[1].value.bufferInfo.alpha = 0x8000;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 2, 0), cuDispErrorNotSupported);
}

TEST(CreateSwapchain, NonDefaultBlendModeReturnsNotSupported) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[2];
  make_default_attribs(attrs);
  attrs[1].value.bufferInfo.blendMode = CUDISP_BLEND_MODE_PREMULTIPLIED;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 2, 0), cuDispErrorNotSupported);
}

TEST(CreateSwapchain, NonDefaultRotationReturnsNotSupported) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[2];
  make_default_attribs(attrs);
  attrs[1].value.bufferInfo.rotation = CUDISP_ROTATE_90;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 2, 0), cuDispErrorNotSupported);
}

TEST(CreateSwapchain, NonDefaultColorEncodingReturnsNotSupported) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[2];
  make_default_attribs(attrs);
  attrs[1].value.bufferInfo.colorEncoding = CUDISP_COLOR_ENCODING_BT709;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 2, 0), cuDispErrorNotSupported);
}

TEST(CreateSwapchain, NonDefaultColorRangeReturnsNotSupported) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[2];
  make_default_attribs(attrs);
  attrs[1].value.bufferInfo.colorRange = CUDISP_COLOR_RANGE_FULL;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 2, 0), cuDispErrorNotSupported);
}

TEST(CreateSwapchain, NonDefaultMaxBpcReturnsNotSupported) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[2];
  make_default_attribs(attrs);
  attrs[0].value.modeInfo.maxBpc = CUDISP_MAX_BPC_10;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 2, 0), cuDispErrorNotSupported);
}

TEST(CreateSwapchain, HdrMetadataReturnsNotSupported) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[3];
  make_default_attribs(attrs);
  memset(&attrs[2], 0, sizeof(attrs[2]));
  attrs[2].id = CUDISP_CREATE_ATTRIBUTE_HDR_METADATA;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 3, 0), cuDispErrorNotSupported);
}

TEST(CreateSwapchain, ColorspaceReturnsNotSupported) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[3];
  make_default_attribs(attrs);
  memset(&attrs[2], 0, sizeof(attrs[2]));
  attrs[2].id = CUDISP_CREATE_ATTRIBUTE_COLORSPACE;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 3, 0), cuDispErrorNotSupported);
}

TEST(CreateSwapchain, DegammaLutReturnsNotSupported) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[3];
  make_default_attribs(attrs);
  memset(&attrs[2], 0, sizeof(attrs[2]));
  attrs[2].id = CUDISP_CREATE_ATTRIBUTE_DEGAMMA_LUT;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 3, 0), cuDispErrorNotSupported);
}

TEST(CreateSwapchain, GammaLutReturnsNotSupported) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[3];
  make_default_attribs(attrs);
  memset(&attrs[2], 0, sizeof(attrs[2]));
  attrs[2].id = CUDISP_CREATE_ATTRIBUTE_GAMMA_LUT;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 3, 0), cuDispErrorNotSupported);
}

TEST(CreateSwapchain, CtmReturnsNotSupported) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[3];
  make_default_attribs(attrs);
  memset(&attrs[2], 0, sizeof(attrs[2]));
  attrs[2].id = CUDISP_CREATE_ATTRIBUTE_CTM;
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 3, 0), cuDispErrorNotSupported);
}

TEST(CreateSwapchain, DisplaySelectReturnsNotSupported) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[3];
  make_default_attribs(attrs);
  memset(&attrs[2], 0, sizeof(attrs[2]));
  attrs[2].id = CUDISP_CREATE_ATTRIBUTE_DISPLAY_SELECT;
  strncpy(attrs[2].value.displaySelect.name, "DP-1",
          sizeof(attrs[2].value.displaySelect.name) - 1);
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 3, 0), cuDispErrorNotSupported);
}

TEST(CreateSwapchain, UnknownAttributeIdReturnsInvalidParam) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[3];
  make_default_attribs(attrs);
  memset(&attrs[2], 0, sizeof(attrs[2]));
  attrs[2].id = static_cast<cuDispCreateAttributeID>(999);
  EXPECT_EQ(cuDispCreateSwapchain(&sc, attrs, 3, 0), cuDispErrorInvalidParam);
}

TEST(CreateSwapchain, IgnoreAttributeIsSkipped) {
  cuDispSwapchain sc = nullptr;
  cuDispCreateAttribute attrs[3];
  memset(attrs, 0, sizeof(attrs));

  attrs[0].id = CUDISP_CREATE_ATTRIBUTE_IGNORE;

  attrs[1].id = CUDISP_CREATE_ATTRIBUTE_BUFFER_INFO;
  attrs[1].value.bufferInfo.layerIndex = 0;
  attrs[1].value.bufferInfo.numBuffers = 2;
  attrs[1].value.bufferInfo.format = CUDISP_SURFACE_FORMAT_ARGB8888;
  attrs[1].value.bufferInfo.width = 1920;
  attrs[1].value.bufferInfo.height = 1080;
  attrs[1].value.bufferInfo.alpha = 0xFFFF;

  attrs[2].id = CUDISP_CREATE_ATTRIBUTE_IGNORE;

  cuDispStatus st = cuDispCreateSwapchain(&sc, attrs, 3, 0);
  // Should pass validation (may fail later during HW init, but not at param check)
  EXPECT_NE(st, cuDispErrorInvalidParam);
  EXPECT_NE(st, cuDispErrorNotSupported);
  if (st == cuDispSuccess && sc) {
    cuDispDestroySwapchain(sc);
  }
}

// ===========================================================================
// cuDispGetBuffer — parameter validation
// ===========================================================================

TEST(GetBuffer, NullSwapchainReturnsInvalidParam) {
  cuDispBufferMemory mem;
  memset(&mem, 0, sizeof(mem));
  CUdeviceptr ptr = 0;
  mem.devicePtr = &ptr;
  EXPECT_EQ(cuDispGetBuffer(nullptr, 0, 0, &mem, 0), cuDispErrorInvalidParam);
}

TEST(GetBuffer, NullOutBufferReturnsInvalidParam) {
  EXPECT_EQ(cuDispGetBuffer(reinterpret_cast<cuDispSwapchain>(0x1), 0, 0, nullptr, 0),
            cuDispErrorInvalidParam);
}

TEST(GetBuffer, NonZeroLayerReturnsNotSupported) {
  cuDispBufferMemory mem;
  memset(&mem, 0, sizeof(mem));
  CUdeviceptr ptr = 0;
  mem.devicePtr = &ptr;
  EXPECT_EQ(cuDispGetBuffer(reinterpret_cast<cuDispSwapchain>(0x1), 1, 0, &mem, 0),
            cuDispErrorNotSupported);
}

TEST(GetBuffer, NonZeroFlagsReturnsInvalidParam) {
  cuDispBufferMemory mem;
  memset(&mem, 0, sizeof(mem));
  CUdeviceptr ptr = 0;
  mem.devicePtr = &ptr;
  EXPECT_EQ(cuDispGetBuffer(reinterpret_cast<cuDispSwapchain>(0x1), 0, 0, &mem, 1),
            cuDispErrorInvalidParam);
}

// ===========================================================================
// cuDispPresent — parameter validation
// ===========================================================================

TEST(Present, NullSwapchainReturnsInvalidParam) {
  cuDispBufferMemory mem;
  memset(&mem, 0, sizeof(mem));
  CUdeviceptr ptr = 0;
  mem.devicePtr = &ptr;
  EXPECT_EQ(cuDispPresent(nullptr, nullptr, &mem, 1, 0), cuDispErrorInvalidParam);
}

TEST(Present, NullBufferMemoryReturnsInvalidParam) {
  EXPECT_EQ(cuDispPresent(reinterpret_cast<cuDispSwapchain>(0x1),
                          nullptr, nullptr, 1, 0),
            cuDispErrorInvalidParam);
}

TEST(Present, NumLayersNotOneReturnsNotSupported) {
  cuDispBufferMemory mem[2];
  memset(mem, 0, sizeof(mem));
  CUdeviceptr p0 = 0, p1 = 0;
  mem[0].devicePtr = &p0;
  mem[1].devicePtr = &p1;
  EXPECT_EQ(cuDispPresent(reinterpret_cast<cuDispSwapchain>(0x1),
                          nullptr, mem, 2, 0),
            cuDispErrorNotSupported);
}

TEST(Present, VsyncOffReturnsNotSupported) {
  cuDispBufferMemory mem;
  memset(&mem, 0, sizeof(mem));
  CUdeviceptr ptr = 0;
  mem.devicePtr = &ptr;
  EXPECT_EQ(cuDispPresent(reinterpret_cast<cuDispSwapchain>(0x1),
                          nullptr, &mem, 1, CUDISP_PRESENT_FLAG_VSYNC_OFF),
            cuDispErrorNotSupported);
}

TEST(Present, UnknownFlagsReturnsInvalidParam) {
  cuDispBufferMemory mem;
  memset(&mem, 0, sizeof(mem));
  CUdeviceptr ptr = 0;
  mem.devicePtr = &ptr;
  // 0x02 has VSYNC_OFF bit clear but is still non-zero
  EXPECT_EQ(cuDispPresent(reinterpret_cast<cuDispSwapchain>(0x1),
                          nullptr, &mem, 1, 0x02),
            cuDispErrorInvalidParam);
}

// ===========================================================================
// cuDispDestroySwapchain — parameter validation
// ===========================================================================

TEST(DestroySwapchain, NullReturnsInvalidParam) {
  EXPECT_EQ(cuDispDestroySwapchain(nullptr), cuDispErrorInvalidParam);
}

// ===========================================================================
// Display-dependent tests — require a real DRM device and CUDA context.
// Skipped automatically if the environment is not available.
// ===========================================================================

class DisplayTest : public ::testing::Test {
 protected:
  CUcontext ctx_ = nullptr;
  uint32_t native_w_ = 0;
  uint32_t native_h_ = 0;

  void SetUp() override {
    if (!QueryPreferredMode()) {
      GTEST_SKIP() << "No connected DRM display found; skipping display test";
    }

    if (cuInit(0) != CUDA_SUCCESS) {
      GTEST_SKIP() << "cuInit failed; skipping display test";
    }
    int count = 0;
    cuDeviceGetCount(&count);
    if (count == 0) {
      GTEST_SKIP() << "No CUDA GPU found; skipping display test";
    }
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    if (create_cuda_context(&ctx_, 0, dev) != CUDA_SUCCESS) {
      GTEST_SKIP() << "Failed to create CUDA context; skipping display test";
    }
  }

  void TearDown() override {
    if (ctx_) {
      cuCtxDestroy(ctx_);
      ctx_ = nullptr;
    }
  }

  bool QueryPreferredMode() {
    int fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (fd < 0) return false;

    drmModeRes* res = drmModeGetResources(fd);
    if (!res) { close(fd); return false; }

    for (int c = 0; c < res->count_connectors; c++) {
      drmModeConnector* conn = drmModeGetConnector(fd, res->connectors[c]);
      if (!conn) continue;
      if (conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
        for (int m = 0; m < conn->count_modes; m++) {
          if (conn->modes[m].type & DRM_MODE_TYPE_PREFERRED) {
            native_w_ = conn->modes[m].hdisplay;
            native_h_ = conn->modes[m].vdisplay;
            break;
          }
        }
        if (native_w_ == 0) {
          native_w_ = conn->modes[0].hdisplay;
          native_h_ = conn->modes[0].vdisplay;
        }
        drmModeFreeConnector(conn);
        break;
      }
      drmModeFreeConnector(conn);
    }
    drmModeFreeResources(res);
    close(fd);
    return (native_w_ > 0 && native_h_ > 0);
  }

  cuDispSwapchain CreateValidSwapchain() {
    cuDispSwapchain sc = nullptr;
    cuDispCreateAttribute attrs[2];
    memset(attrs, 0, sizeof(attrs));

    attrs[0].id = CUDISP_CREATE_ATTRIBUTE_MODE_INFO;
    attrs[0].value.modeInfo.modeWidth = native_w_;
    attrs[0].value.modeInfo.modeHeight = native_h_;
    attrs[0].value.modeInfo.maxBpc = CUDISP_MAX_BPC_DEFAULT;

    attrs[1].id = CUDISP_CREATE_ATTRIBUTE_BUFFER_INFO;
    attrs[1].value.bufferInfo.layerIndex = 0;
    attrs[1].value.bufferInfo.numBuffers = 2;
    attrs[1].value.bufferInfo.format = CUDISP_SURFACE_FORMAT_ARGB8888;
    attrs[1].value.bufferInfo.width = native_w_;
    attrs[1].value.bufferInfo.height = native_h_;
    attrs[1].value.bufferInfo.alpha = 0xFFFF;

    cuDispStatus st = cuDispCreateSwapchain(&sc, attrs, 2, 0);
    if (st != cuDispSuccess) return nullptr;
    return sc;
  }
};

TEST_F(DisplayTest, ValidCreateDestroy) {
  cuDispSwapchain sc = CreateValidSwapchain();
  ASSERT_NE(sc, nullptr);
  EXPECT_EQ(cuDispDestroySwapchain(sc), cuDispSuccess);
}

TEST_F(DisplayTest, CreateDestroyLoop) {
  for (int i = 0; i < 5; i++) {
    cuDispSwapchain sc = CreateValidSwapchain();
    ASSERT_NE(sc, nullptr) << "iteration " << i;
    EXPECT_EQ(cuDispDestroySwapchain(sc), cuDispSuccess) << "iteration " << i;
  }
}

TEST_F(DisplayTest, GetBufferValidOutputs) {
  cuDispSwapchain sc = CreateValidSwapchain();
  ASSERT_NE(sc, nullptr);

  CUdeviceptr ptr = 0;
  uint64_t size = 0;
  uint32_t stride = 0;

  cuDispBufferMemory mem;
  memset(&mem, 0, sizeof(mem));
  mem.devicePtr = &ptr;
  mem.size = &size;
  mem.stride = &stride;
  mem.pHDRMetadata = nullptr;

  EXPECT_EQ(cuDispGetBuffer(sc, 0, 0, &mem, 0), cuDispSuccess);
  EXPECT_NE(ptr, static_cast<CUdeviceptr>(0));
  EXPECT_GT(size, 0u);
  EXPECT_GT(stride, 0u);

  EXPECT_EQ(cuDispDestroySwapchain(sc), cuDispSuccess);
}

TEST_F(DisplayTest, GetBufferOutOfRange) {
  cuDispSwapchain sc = CreateValidSwapchain();
  ASSERT_NE(sc, nullptr);

  CUdeviceptr ptr = 0;
  cuDispBufferMemory mem;
  memset(&mem, 0, sizeof(mem));
  mem.devicePtr = &ptr;

  EXPECT_EQ(cuDispGetBuffer(sc, 0, 999, &mem, 0), cuDispErrorInvalidParam);

  EXPECT_EQ(cuDispDestroySwapchain(sc), cuDispSuccess);
}

TEST_F(DisplayTest, GetBufferNullDevicePtrReturnsInvalidParam) {
  cuDispSwapchain sc = CreateValidSwapchain();
  ASSERT_NE(sc, nullptr);

  cuDispBufferMemory mem;
  memset(&mem, 0, sizeof(mem));

  EXPECT_EQ(cuDispGetBuffer(sc, 0, 0, &mem, 0), cuDispErrorInvalidParam);

  EXPECT_EQ(cuDispDestroySwapchain(sc), cuDispSuccess);
}

TEST_F(DisplayTest, HdrMetadataOutputIsZeroed) {
  cuDispSwapchain sc = CreateValidSwapchain();
  ASSERT_NE(sc, nullptr);

  CUdeviceptr ptr = 0;
  cuDispHDRMetadata hdr;
  memset(&hdr, 0xFF, sizeof(hdr));

  cuDispBufferMemory mem;
  memset(&mem, 0, sizeof(mem));
  mem.devicePtr = &ptr;
  mem.pHDRMetadata = &hdr;

  EXPECT_EQ(cuDispGetBuffer(sc, 0, 0, &mem, 0), cuDispSuccess);

  cuDispHDRMetadata zeroed;
  memset(&zeroed, 0, sizeof(zeroed));
  EXPECT_EQ(memcmp(&hdr, &zeroed, sizeof(cuDispHDRMetadata)), 0);

  EXPECT_EQ(cuDispDestroySwapchain(sc), cuDispSuccess);
}

TEST_F(DisplayTest, ValidPresentReturnsSuccess) {
  cuDispSwapchain sc = CreateValidSwapchain();
  ASSERT_NE(sc, nullptr);

  CUdeviceptr ptr = 0;
  cuDispBufferMemory mem;
  memset(&mem, 0, sizeof(mem));
  mem.devicePtr = &ptr;

  ASSERT_EQ(cuDispGetBuffer(sc, 0, 0, &mem, 0), cuDispSuccess);
  EXPECT_EQ(cuDispPresent(sc, nullptr, &mem, 1, 0), cuDispSuccess);

  EXPECT_EQ(cuDispDestroySwapchain(sc), cuDispSuccess);
}

// ===========================================================================
// Struct size / ABI sanity checks
// ===========================================================================

TEST(ABI, AttributeValueIs64Bytes) {
  EXPECT_EQ(sizeof(cuDispCreateAttributeValue), 64u);
}

TEST(ABI, DisplaySelectFitsInUnion) {
  EXPECT_LE(sizeof(cuDispDisplaySelect), sizeof(cuDispCreateAttributeValue));
}

TEST(ABI, HdrMetadataFitsInUnion) {
  EXPECT_LE(sizeof(cuDispHDRMetadata), sizeof(cuDispCreateAttributeValue));
}

TEST(ABI, ModeInfoFitsInUnion) {
  EXPECT_LE(sizeof(cuDispModeInfo), sizeof(cuDispCreateAttributeValue));
}

TEST(ABI, BufferInfoFitsInUnion) {
  EXPECT_LE(sizeof(cuDispBufferInfo), sizeof(cuDispCreateAttributeValue));
}

TEST(ABI, LutConfigFitsInUnion) {
  EXPECT_LE(sizeof(cuDispLutConfig), sizeof(cuDispCreateAttributeValue));
}

TEST(ABI, CtmConfigFitsInUnion) {
  EXPECT_LE(sizeof(cuDispCtmConfig), sizeof(cuDispCreateAttributeValue));
}

TEST(ABI, ColorspaceConfigFitsInUnion) {
  EXPECT_LE(sizeof(cuDispColorspaceConfig), sizeof(cuDispCreateAttributeValue));
}

// ===========================================================================
// main
// ===========================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
