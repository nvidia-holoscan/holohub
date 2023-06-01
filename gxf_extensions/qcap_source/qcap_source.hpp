/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 YUAN High-Tech Development Co., Ltd. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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
#pragma once

#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "qcap_queue.hpp"

#include "gxf/std/codelet.hpp"
#include "gxf/std/transmitter.hpp"

//namespace gxf = nvidia::gxf;

namespace yuan {
namespace holoscan {

// PIXELFORMAT TYPE
enum {
  PIXELFORMAT_RGB24 = 0,          //   0xBBGGRR -> R0 G0 B0 R1 G1 B1 R2 G2 B2 ... >>
  PIXELFORMAT_BGR24 = 1,          //   0xRRGGBB -> B0 G0 R0 B1 G1 R1 B2 G2 R2 ... >>
  PIXELFORMAT_ARGB32 = 2,         // 0xAABBGGRR -> R0 G0 B0 A0 R1 G1 B1 A1 R2 G2 B2 A2 ... >>
  PIXELFORMAT_ABGR32 = 3,         // 0xAARRGGBB -> B0 G0 R0 A0 B1 G1 R1 A1 B2 G2 R2 A2 ... >>
  PIXELFORMAT_Y416 = 0x36313459,  // 0x36313459 -> MAKEFOURCC('Y', '4', '1', '6') (4:4:4 | 10 BITS)
  PIXELFORMAT_P210 = 0x30313250,  // 0x30313250 -> MAKEFOURCC('P', '2', '1', '0') (4:2:2 | 10 BITS)
  PIXELFORMAT_P010 = 0x30313050,  // 0x30313050 -> MAKEFOURCC('P', '0', '1', '0') (4:2:0 | 10 BITS)
  PIXELFORMAT_YUY2 = 0x32595559,  // 0x32595559 -> MAKEFOURCC('Y', 'U', 'Y', '2') (4:2:2 | 08 BITS)
  PIXELFORMAT_UYVY = 0x59565955,  // 0x59565955 -> MAKEFOURCC('U', 'Y', 'V', 'Y') (4:2:2 | 08 BITS)
  PIXELFORMAT_YV12 = 0x32315659,  // 0x32315659 -> MAKEFOURCC('Y', 'V', '1', '2') (4:2:0 | 08 BITS) (Y V U)
  PIXELFORMAT_I420 = 0x30323449,  // 0x30323449 -> MAKEFOURCC('I', '4', '2', '0') (4:2:0 | 08 BITS) (Y U V)
  PIXELFORMAT_NV12 = 0x3231564E,  // 0x3231564E -> MAKEFOURCC('N', 'V', '1', '2') (4:2:0 | 08 BITS) (Y C)
  PIXELFORMAT_Y800 = 0x30303859,  // 0x30303859 -> MAKEFOURCC('Y', '8', '0', '0') (4:0:0 | 08 BITS) (Y)
  PIXELFORMAT_MJPG = 0x47504A4D,  // 0x47504A4D -> MAKEFOURCC('M', 'J', 'P', 'G')
  PIXELFORMAT_H264 = 0x34363248,  // 0x34363248 -> MAKEFOURCC('H', '2', '6', '4')
  PIXELFORMAT_H265 = 0x35363248,  // 0x35363248 -> MAKEFOURCC('H', '2', '6', '5')
  PIXELFORMAT_MPG2 = 0x3247504D,  // 0x3247504D -> MAKEFOURCC('M', 'P', 'G', '2')
} ePIXELFORMAT;

enum {
  DISPLAYPORT_SST_MODE = 0,
  DISPLAYPORT_MST_MODE = 1,
} eDISPLAYPORT_MST_MODE;

enum {
  SDI12G_DEFAULT_MODE  = 0,
  SDI12G_QUADLINK_MODE = 1,
  SDI12G_SI_MODE       = 2,
} eSDI12G_MODE;

enum {
  INPUTTYPE_COMPOSITE	    = 0,
  INPUTTYPE_SVIDEO	    = 1,
  INPUTTYPE_HDMI	    = 2,
  INPUTTYPE_DVI_D	    = 3,
  INPUTTYPE_COMPONENTS	    = 4,
  INPUTTYPE_YCBCR	    = 4,
  INPUTTYPE_DVI_A	    = 5,
  INPUTTYPE_RGB	            = 5,
  INPUTTYPE_VGA	            = 5,
  INPUTTYPE_SDI	            = 6,
  INPUTTYPE_DISPLAY_PORT    = 8,
  INPUTTYPE_AUTO	    = 7,
} eINPUT_TYPE;

constexpr char kDefaultDevice[] = "SC0710 PCI";
constexpr uint32_t kDefaultChannel = 0;
constexpr uint32_t kDefaultWidth = 3840;
constexpr uint32_t kDefaultHeight = 2160;
constexpr uint32_t kDefaultFramerate = 60;
constexpr uint32_t kDefaultPreviewSize = kDefaultWidth * kDefaultHeight * 4;
constexpr uint32_t kDefaultGPUDirectRingQueueSize = 6;
constexpr uint32_t kDefaultColorConvertBufferSize = 3;
constexpr bool kDefaultRDMA = true;
constexpr char kDefaultPixelFormatStr[] = "bgr24";
constexpr uint32_t kDefaultPixelFormat = PIXELFORMAT_BGR24;
//constexpr uint32_t kDefaultPixelFormat = PIXELFORMAT_YUY2;
//constexpr uint32_t kDefaultPixelFormat = PIXELFORMAT_NV12;
constexpr uint32_t kDefaultOutputPixelFormat = PIXELFORMAT_RGB24;
constexpr uint32_t kDefaultDisplayPortMstMode = DISPLAYPORT_SST_MODE;
constexpr char kDefaultInputTypeStr[] = "auto";
constexpr uint32_t kDefaultInputType = INPUTTYPE_AUTO;
constexpr uint32_t kDefaultSDI12GMode = SDI12G_DEFAULT_MODE;

struct PreviewFrame {
  unsigned char* pFrameBuffer;
  unsigned long nFrameBufferLen;
};

struct Image {
  int width;
  int height;
  int components;
  unsigned char* data;
  CUdeviceptr cu_src;
  CUdeviceptr cu_dst;
};

enum DeviceStatus {
  STATUS_NO_SDK,
  STATUS_NO_DEVICE,
  STATUS_NO_SIGNAL,
  STATUS_SIGNAL_REMOVED,
  STATUS_SIGNAL_LOCKED,
};

/// @brief Video input codelet for use with capture cards.
///
/// Provides a codelet for supporting capture card as a source.
/// It offers support for GPUDirect-RDMA on Quadro GPUs.
/// The output is a VideoBuffer object.
class QCAPSource : public nvidia::gxf::Codelet {
 public:
  QCAPSource();

  gxf_result_t registerInterface(nvidia::gxf::Registrar* registrar) override;

  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

  void initCuda();
  void cleanupCuda();

  void loadImage(const char* filename, const unsigned char* buffer, const size_t size,
                 struct Image* image);
  void destroyImage(struct Image* image);

  nvidia::gxf::Parameter<nvidia::gxf::Handle<nvidia::gxf::Transmitter>> video_buffer_output_;
  nvidia::gxf::Parameter<std::string> device_specifier_;
  nvidia::gxf::Parameter<uint32_t> channel_;
  nvidia::gxf::Parameter<uint32_t> width_;
  nvidia::gxf::Parameter<uint32_t> height_;
  nvidia::gxf::Parameter<uint32_t> framerate_;
  nvidia::gxf::Parameter<bool> use_rdma_;
  nvidia::gxf::Parameter<std::string> pixel_format_str_;
  uint32_t pixel_format_;
  uint32_t output_pixel_format_;
  nvidia::gxf::Parameter<uint32_t>  mst_mode_;
  nvidia::gxf::Parameter<std::string> input_type_str_;
  uint32_t input_type_;
  nvidia::gxf::Parameter<uint32_t> sdi12g_mode_;

  volatile DeviceStatus m_status = STATUS_NO_SDK;
  void* m_hDevice = nullptr;
  unsigned long m_nVideoWidth = 0;
  unsigned long m_nVideoHeight = 0;
  bool m_bVideoIsInterleaved = false;
  double m_dVideoFrameRate = 0.0f;
  unsigned long m_nAudioChannels = 0;
  unsigned long m_nAudioBitsPerSample = 0;
  unsigned long m_nAudioSampleFrequency = 0;
  unsigned long m_nVideoInput = 0;
  unsigned long m_nAudioInput = 0;
  unsigned char* m_pGPUDirectBuffer[kDefaultGPUDirectRingQueueSize] = {};

  unsigned char* m_pRGBBUffer[kDefaultColorConvertBufferSize] = {};
  unsigned long m_nRGBBufferIndex = 0;

  CUcontext m_CudaContext = nullptr;

  struct Image m_iNoDeviceImage;
  struct Image m_iNoSignalImage;
  struct Image m_iSignalRemovedImage;
  struct Image m_iNoSdkImage;

  threadsafe_queue_t<PreviewFrame> m_queue;
};

}  // namespace holoscan
}  // namespace yuan
