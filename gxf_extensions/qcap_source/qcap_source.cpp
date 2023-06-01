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
#include "qcap_source.hpp"

#ifdef BUILD_WITH_QCAP_SDK
#include <qcap.common.h>
#include <qcap.h>
#include <qcap.linux.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <npp.h>

#include <sstream>
#include <string>
#include <utility>

#include "gxf/multimedia/video.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define EXPAND_FILE(filename)                                                 \
  extern const char _binary_##filename##_start[], _binary_##filename##_end[]; \
  const char* filename##_ptr = _binary_##filename##_start;                    \
  const char* filename##_end = _binary_##filename##_end;                      \
  size_t filename##_size = filename##_end - filename##_ptr;

EXPAND_FILE(no_device_png)
EXPAND_FILE(no_signal_png)
EXPAND_FILE(no_sdk_png)

namespace yuan {
namespace holoscan {

#ifdef BUILD_WITH_QCAP_SDK
QRETURN on_process_signal_removed(PVOID pDevice, ULONG nVideoInput, ULONG nAudioInput,
                                  PVOID pUserData) {
  struct QCAPSource* qcap = (struct QCAPSource*)pUserData;

  GXF_LOG_INFO("QCAP Source: signal removed \n");

  qcap->m_status = STATUS_SIGNAL_REMOVED;
  qcap->m_queue.signal(false);
  return QCAP_RT_OK;
}

QRETURN on_process_no_signal_detected(PVOID pDevice, ULONG nVideoInput, ULONG nAudioInput,
                                      PVOID pUserData) {
  struct QCAPSource* qcap = (struct QCAPSource*)pUserData;

  GXF_LOG_INFO("QCAP Source: no signal Detected \n");

  qcap->m_status = STATUS_NO_SIGNAL;
  qcap->m_queue.signal(false);
  return QCAP_RT_OK;
}

QRETURN on_process_format_changed(PVOID pDevice, ULONG nVideoInput, ULONG nAudioInput,
                                  ULONG nVideoWidth, ULONG nVideoHeight, BOOL bVideoIsInterleaved,
                                  double dVideoFrameRate, ULONG nAudioChannels,
                                  ULONG nAudioBitsPerSample, ULONG nAudioSampleFrequency,
                                  PVOID pUserData) {
  struct QCAPSource* qcap = (struct QCAPSource*)pUserData;

  //GXF_LOG_INFO("QCAP Source: format changed Detected");

  CHAR strVideoInput[64] = {0};
  CHAR strAudioInput[64] = {0};
  CHAR strFrameType[64] = {0};

  qcap->m_nVideoWidth = nVideoWidth;
  qcap->m_nVideoHeight = nVideoHeight;
  qcap->m_bVideoIsInterleaved = bVideoIsInterleaved;
  qcap->m_dVideoFrameRate = dVideoFrameRate;
  qcap->m_nAudioChannels = nAudioChannels;
  qcap->m_nAudioBitsPerSample = nAudioBitsPerSample;
  qcap->m_nAudioSampleFrequency = nAudioSampleFrequency;
  qcap->m_nVideoInput = nVideoInput;
  qcap->m_nAudioInput = nAudioInput;

  if (nVideoInput == 0) { sprintf(strVideoInput, "COMPOSITE"); }
  if (nVideoInput == 1) { sprintf(strVideoInput, "SVIDEO"); }
  if (nVideoInput == 2) { sprintf(strVideoInput, "HDMI"); }
  if (nVideoInput == 3) { sprintf(strVideoInput, "DVI_D"); }
  if (nVideoInput == 4) { sprintf(strVideoInput, "COMPONENTS (YCBCR)"); }
  if (nVideoInput == 5) { sprintf(strVideoInput, "DVI_A (RGB / VGA)"); }
  if (nVideoInput == 6) { sprintf(strVideoInput, "SDI"); }
  if (nVideoInput == 7) { sprintf(strVideoInput, "AUTO"); }
  if (nAudioInput == 0) { sprintf(strAudioInput, "EMBEDDED_AUDIO"); }
  if (nAudioInput == 1) { sprintf(strAudioInput, "LINE_IN"); }
  if (nAudioInput == 2) { sprintf(strAudioInput, "SOUNDCARD_MICROPHONE"); }
  if (nAudioInput == 3) { sprintf(strAudioInput, "SOUNDCARD_LINE_IN"); }

  ULONG nVH = bVideoIsInterleaved == TRUE ? nVideoHeight / 2 : nVideoHeight;
  sprintf(strFrameType, bVideoIsInterleaved == TRUE ? " I " : " P ");

  GXF_LOG_INFO(
      "QCAP Source: INFO %ld x %ld%s @%2.3f FPS,"
      " %ld CH x %ld BITS x %ld HZ, VIDEO INPUT: %s, AUDIO INPUT: %s",
      nVideoWidth,
      nVH,
      strFrameType,
      dVideoFrameRate,
      nAudioChannels,
      nAudioBitsPerSample,
      nAudioSampleFrequency,
      strVideoInput,
      strAudioInput);

  qcap->m_status = STATUS_SIGNAL_LOCKED;
  qcap->m_queue.signal(true);

  return QCAP_RT_OK;
}

QRETURN on_process_video_preview(PVOID pDevice, double dSampleTime, BYTE* pFrameBuffer,
                                 ULONG nFrameBufferLen, PVOID pUserData) {
  struct QCAPSource* qcap = (struct QCAPSource*)pUserData;

  PVOID pRCBuffer = QCAP_BUFFER_GET_RCBUFFER(pFrameBuffer, nFrameBufferLen);
  QCAP_RCBUFFER_ADD_REF(pRCBuffer);

  PreviewFrame preview;
  preview.pFrameBuffer = pFrameBuffer;
  preview.nFrameBufferLen = nFrameBufferLen;

  qcap->m_queue.push_and_drop(preview, [](PreviewFrame drop) {
    PVOID pRCBuffer = QCAP_BUFFER_GET_RCBUFFER(drop.pFrameBuffer, drop.nFrameBufferLen);
    QCAP_RCBUFFER_RELEASE(pRCBuffer);
  });

  return QCAP_RT_OK;
}

QRETURN on_process_audio_preview(PVOID pDevice, double dSampleTime, BYTE* pFrameBuffer,
                                 ULONG nFrameBufferLen, PVOID pUserData) {
  return QCAP_RT_OK;
}
#endif

QCAPSource::QCAPSource()
    : pixel_format_(kDefaultPixelFormat),
      output_pixel_format_(kDefaultOutputPixelFormat),
      input_type_(kDefaultInputType) {}

gxf_result_t QCAPSource::registerInterface(nvidia::gxf::Registrar* registrar) {
  nvidia::gxf::Expected<void> result;
  result &= registrar->parameter(video_buffer_output_,
                                 "video_buffer_output",
                                 "VideoBufferOutput",
                                 "Output for the video buffer.");
  result &= registrar->parameter(
      device_specifier_, "device", "Device", "Device specifier.", std::string(kDefaultDevice));
  result &=
      registrar->parameter(channel_, "channel", "Channel", "Channel to use.", kDefaultChannel);
  result &= registrar->parameter(width_, "width", "Width", "Width of the stream.", kDefaultWidth);
  result &=
      registrar->parameter(height_, "height", "Height", "Height of the stream.", kDefaultHeight);
  result &= registrar->parameter(
      framerate_, "framerate", "Framerate", "Framerate of the stream.", kDefaultFramerate);
  result &= registrar->parameter(use_rdma_, "rdma", "RDMA", "Enable RDMA.", kDefaultRDMA);

  result &= registrar->parameter(
      pixel_format_str_, "pixel_format", "PixelFormat", "Pixel Format.", std::string(kDefaultPixelFormatStr));

  result &= registrar->parameter(
      input_type_str_, "input_type", "InputType", "Input Type.", std::string(kDefaultInputTypeStr));

  result &= registrar->parameter(mst_mode_, "mst_mode", "DisplayPortMSTMode", "Display port MST mode.", kDefaultDisplayPortMstMode);

  result &= registrar->parameter(sdi12g_mode_, "sdi12g_mode", "SDI12GMode", "SDI 12G Mode.", kDefaultSDI12GMode);

#ifdef BUILD_WITH_QCAP_SDK
  m_status = STATUS_NO_DEVICE;
#else
  GXF_LOG_WARNING("QCAP Source: build without QCAP sdk.\n");
  m_status = STATUS_NO_SDK;
#endif

  return nvidia::gxf::ToResultCode(result);
}

void QCAPSource::loadImage(const char* filename, const unsigned char* buffer, const size_t size,
                           struct Image* image) {
  if (image == nullptr) {
    GXF_LOG_INFO("QCAP Source: invalid parameter, image is null\n");
    return;
  }

  // Init
  image->width = 0;
  image->height = 0;
  image->components = 0;
  image->data = nullptr;
  image->cu_src = 0;
  image->cu_dst = 0;

  // Loading
  image->data = stbi_load_from_memory(buffer,
                                      size,
                                      reinterpret_cast<int*>(&image->width),
                                      reinterpret_cast<int*>(&image->height),
                                      &image->components,
                                      0);

  if (image->data == nullptr) {
    GXF_LOG_INFO("QCAP Source: load image %s fail", filename);
    return;
  }

  GXF_LOG_INFO("QCAP Source: load image %s %dx%d %d",
               filename,
               image->width,
               image->height,
               image->components);

  // memset(image->data, 128, image->width * image->height * image->components);

  if (image->components == 4) {
    int width = image->width;
    int height = image->height;

    if (cuMemAlloc(&image->cu_src, width * height * 4) != CUDA_SUCCESS) {
      throw std::runtime_error("cuMemAlloc failed.");
    }
    if (cuMemAlloc(&image->cu_dst, width * height * 3) != CUDA_SUCCESS) {
      throw std::runtime_error("cuMemAlloc failed.");
    }
    if (cuMemcpyHtoD(image->cu_src, image->data, width * height * 4) != CUDA_SUCCESS) {
      throw std::runtime_error("cuMemcpyHtoD failed.");
    }

    if (output_pixel_format_ == PIXELFORMAT_RGB24) {  // RGBA to RGB
      NppStatus status;
      NppiSize oSizeROI;
      int video_width = width;
      int video_height = height;
      oSizeROI.width = video_width;
      oSizeROI.height = video_height;
      const int aDstOrder[3] = {0, 1, 2};
      status = nppiSwapChannels_8u_C4C3R((Npp8u*)image->cu_src,
                                         video_width * 4,
                                         (Npp8u*)image->cu_dst,
                                         video_width * 3,
                                         oSizeROI,
                                         aDstOrder);

      if (status != 0) {
        GXF_LOG_INFO(
            "QCAP Source: image convert error %d %dx%d", status, video_width, video_height);
      }
    }
  }
}

void QCAPSource::destroyImage(struct Image* image) {
  if (image->cu_src && cuMemFree(image->cu_src) != CUDA_SUCCESS) {
    throw std::runtime_error("cuMemFree failed.");
  }
  if (image->cu_dst && cuMemFree(image->cu_dst) != CUDA_SUCCESS) {
    throw std::runtime_error("cuMemFree failed.");
  }
}

void QCAPSource::initCuda() {
  if (cuInit(0) != CUDA_SUCCESS) { throw std::runtime_error("cuInit failed."); }

  if (cuDevicePrimaryCtxRetain(&m_CudaContext, 0) != CUDA_SUCCESS) {
    throw std::runtime_error("cuDevicePrimaryCtxRetain failed.");
  }

  if (cuCtxPushCurrent(m_CudaContext) != CUDA_SUCCESS) {
    throw std::runtime_error("cuDevicePrimaryCtxRetain failed.");
  }
}

void QCAPSource::cleanupCuda() {
  if (m_CudaContext) {
    if (cuCtxPopCurrent(&m_CudaContext) != CUDA_SUCCESS) {
      throw std::runtime_error("cuCtxPopCurrent failed.");
    }
    m_CudaContext = nullptr;

    if (cuDevicePrimaryCtxRelease(0) != CUDA_SUCCESS) {
      throw std::runtime_error("cuDevicePrimaryCtxRelease failed.");
    }
  }
}

gxf_result_t QCAPSource::start() {

  if (pixel_format_str_.get().compare("yuy2") == 0) {
      pixel_format_ = PIXELFORMAT_YUY2;
  } else if (pixel_format_str_.get().compare("nv12") == 0) {
      pixel_format_ = PIXELFORMAT_NV12;
  } else {
      pixel_format_ = PIXELFORMAT_BGR24;
  }

  if (input_type_str_.get().compare("dvi_d") == 0) {
      input_type_ = INPUTTYPE_DVI_D;
  } else if (input_type_str_.get().compare("dp") == 0) {
      input_type_ = INPUTTYPE_DISPLAY_PORT;
  } else if (input_type_str_.get().compare("sdi") == 0) {
      input_type_ = INPUTTYPE_SDI;
  } else if (input_type_str_.get().compare("hdmi") == 0) {
      input_type_ = INPUTTYPE_HDMI;
  } else {
      input_type_ = INPUTTYPE_AUTO;
  }

  GXF_LOG_INFO("QCAP Source: Using channel %d", (channel_.get() + 1));
  GXF_LOG_INFO("QCAP Source: RDMA is %s", use_rdma_ ? "enabled" : "disabled");
  GXF_LOG_INFO("QCAP Source: Resolution %dx%d", width_.get(), height_.get());
  GXF_LOG_INFO("QCAP Source: Pixel format is %s (%d)", pixel_format_str_.get().c_str(), pixel_format_);
  GXF_LOG_INFO("QCAP Source: Input type is %s (%d)", input_type_str_.get().c_str(), input_type_);

  initCuda();

  loadImage(
      "no_device.png", (unsigned char*)no_device_png_ptr, no_device_png_size, &m_iNoDeviceImage);
  loadImage(
      "no_signal.png", (unsigned char*)no_signal_png_ptr, no_signal_png_size, &m_iNoSignalImage);
  loadImage(
      "no_sdk.png", (unsigned char*)no_sdk_png_ptr, no_sdk_png_size, &m_iNoSdkImage);
  //loadImage("signal_remove.png",
  //          (unsigned char*)signal_remove_png_ptr,
  //          signal_remove_png_size,
  //          &m_iSignalRemovedImage);

  for (int i = 0; i < kDefaultColorConvertBufferSize; i++) {
    cudaMalloc((void**)&m_pRGBBUffer[i], kDefaultPreviewSize);
  }

#ifdef BUILD_WITH_QCAP_SDK
  QCAP_CREATE((char*)device_specifier_.get().c_str(), 0, nullptr, &m_hDevice, TRUE);

  QCAP_SET_DEVICE_CUSTOM_PROPERTY(m_hDevice, QCAP_DEVPROP_IO_METHOD, 1);
  QCAP_SET_DEVICE_CUSTOM_PROPERTY(m_hDevice, QCAP_DEVPROP_VO_BACKEND, 2);

  // QCAP_SET_AUDIO_SOUND_RENDERER(m_hDevice, 0);

  QCAP_REGISTER_NO_SIGNAL_DETECTED_CALLBACK(m_hDevice, on_process_no_signal_detected, this);
  QCAP_REGISTER_SIGNAL_REMOVED_CALLBACK(m_hDevice, on_process_signal_removed, this);
  QCAP_REGISTER_FORMAT_CHANGED_CALLBACK(m_hDevice, on_process_format_changed, this);
  QCAP_REGISTER_VIDEO_PREVIEW_CALLBACK(m_hDevice, on_process_video_preview, this);
  QCAP_REGISTER_AUDIO_PREVIEW_CALLBACK(m_hDevice, on_process_audio_preview, this);

  if (input_type_ == INPUTTYPE_DVI_D) {
    QCAP_SET_VIDEO_INPUT(m_hDevice, QCAP_INPUT_TYPE_DVI_D);
  } else if (input_type_ == INPUTTYPE_DISPLAY_PORT) {
    GXF_LOG_INFO("QCAP Source: DP MST mode is %d", mst_mode_.get());
    if (mst_mode_ == DISPLAYPORT_MST_MODE) {
      QCAP_SET_VIDEO_INPUT(m_hDevice, QCAP_INPUT_TYPE_DISPLAY_PORT_MST);
    } else {
      QCAP_SET_VIDEO_INPUT(m_hDevice, QCAP_INPUT_TYPE_DISPLAY_PORT_SST);
    }
  } else if (input_type_ == INPUTTYPE_SDI) {
    QCAP_SET_VIDEO_INPUT(m_hDevice, QCAP_INPUT_TYPE_SDI);
  } else if (input_type_ == INPUTTYPE_HDMI) {
    QCAP_SET_VIDEO_INPUT(m_hDevice, QCAP_INPUT_TYPE_HDMI);
  } else { // INPUTTYPE_AUTO or Default
    /* do nothing, we don't change input type. Let driver select it. */
  }

  ULONG nVideoInput = 0;
  QCAP_GET_VIDEO_INPUT(m_hDevice, &nVideoInput);
  GXF_LOG_INFO("QCAP Source: Use input %ld", nVideoInput);
  if (nVideoInput == QCAP_INPUT_TYPE_SDI) {
      GXF_LOG_INFO("QCAP Source: SDI 12G mode is %d", sdi12g_mode_.get());

      if (sdi12g_mode_.get() != SDI12G_DEFAULT_MODE) {
          int qcap_sdi_mode = (sdi12g_mode_.get() == SDI12G_QUADLINK_MODE ? 0 : 1);
          QCAP_SET_DEVICE_CUSTOM_PROPERTY(m_hDevice, QCAP_DEVPROP_SDI12G_MODE, qcap_sdi_mode);
          QCAP_SET_VIDEO_INPUT(m_hDevice, QCAP_INPUT_TYPE_SDI);
      }

      // Workaround
      if (pixel_format_ == PIXELFORMAT_BGR24 ) {
          GXF_LOG_INFO("QCAP Source: SDI only support YUY2 or NV12, switch to yuy2");
          pixel_format_ = PIXELFORMAT_YUY2;
      }
  }
  //QCAP_SET_VIDEO_DEFAULT_OUTPUT_FORMAT(m_hDevice, pixel_format_, width_.get(), height_.get(), 0, 0);
  QCAP_SET_VIDEO_DEFAULT_OUTPUT_FORMAT(m_hDevice, pixel_format_, 0, 0, 0, 0);

  if (use_rdma_) {
    for (int i = 0; i < kDefaultGPUDirectRingQueueSize; i++) {
      cudaMalloc((void**)&m_pGPUDirectBuffer[i], kDefaultPreviewSize);
      // QCAP_ALLOC_VIDEO_GPUDIRECT_PREVIEW_BUFFER(m_hDevice, &m_pGPUDirectBuffer[i],
      // kDefaultPreviewSize);
      QCAP_BIND_VIDEO_GPUDIRECT_PREVIEW_BUFFER(
          m_hDevice, i, m_pGPUDirectBuffer[i], kDefaultPreviewSize);
      GXF_LOG_INFO("QCAP Source: Allocate gpu buffer id:%d, pointer:%p size:%d",
                   i,
                   m_pGPUDirectBuffer[i],
                   kDefaultPreviewSize);
    }
  }

  QCAP_RUN(m_hDevice);
#endif

  return GXF_SUCCESS;
}

gxf_result_t QCAPSource::stop() {
#ifdef BUILD_WITH_QCAP_SDK
  if (m_hDevice) {
    QCAP_STOP(m_hDevice);

    m_queue.quit();

    if (use_rdma_) {
      for (int i = 0; i < kDefaultGPUDirectRingQueueSize; i++) {
        QCAP_UNBIND_VIDEO_GPUDIRECT_PREVIEW_BUFFER(
            m_hDevice, i, m_pGPUDirectBuffer[i], kDefaultPreviewSize);
        // QCAP_FREE_VIDEO_GPUDIRECT_PREVIEW_BUFFER(m_hDevice, m_pGPUDirectBuffer[i],
        // kDefaultPreviewSize);
        cudaFree((void**)&m_pGPUDirectBuffer[i]);
      }
    }

    for (int i = 0; i < kDefaultColorConvertBufferSize; i++) { cudaFree((void**)&m_pRGBBUffer[i]); }

    destroyImage(&m_iNoDeviceImage);
    destroyImage(&m_iNoSignalImage);
    destroyImage(&m_iSignalRemovedImage);

    cleanupCuda();

    QCAP_DESTROY(m_hDevice);
    m_hDevice = nullptr;
  }
#endif
  return GXF_SUCCESS;
}

gxf_result_t QCAPSource::tick() {
  PreviewFrame preview;

  // Pass the frame downstream.
  auto message = nvidia::gxf::Entity::New(context());
  if (!message) {
    GXF_LOG_ERROR("QCAP Source: Failed to allocate message; terminating.");
    return GXF_FAILURE;
  }

  auto buffer = message.value().add<nvidia::gxf::VideoBuffer>();
  if (!buffer) {
    GXF_LOG_ERROR("QCAP Source: Failed to allocate video buffer; terminating.");
    return GXF_FAILURE;
  }

  // GXF_LOG_ERROR("QCAP Source: status %d in tick", m_status);
  // Show error image
  if (m_status != STATUS_SIGNAL_LOCKED) {
    struct Image* image = nullptr;
    switch (m_status) {
      case STATUS_NO_SDK:
        image = &m_iNoSdkImage;
        break;
      case STATUS_NO_DEVICE:
        image = &m_iNoDeviceImage;
        break;
      case STATUS_NO_SIGNAL:
        image = &m_iNoSignalImage;
        break;
      case STATUS_SIGNAL_REMOVED:
        image = &m_iNoSignalImage;
        break;
    }
    int out_width = image->width;
    int out_height = image->height;
    int out_size = out_width * out_height * (output_pixel_format_ == PIXELFORMAT_RGB24 ? 3 : 4);

    // GXF_LOG_ERROR("QCAP Source: show %d image %dx%d", m_status, out_width, out_height);
    nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB> video_type;
    nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB> color_format;
    auto color_planes = color_format.getDefaultColorPlanes(out_width, out_height);
    //GXF_LOG_ERROR("QCAP Source: stride %dx%d %d", out_width, out_height, color_planes[0].stride);
    color_planes[0].stride = out_width * 3;
    nvidia::gxf::VideoBufferInfo info{(uint32_t)out_width,
                              (uint32_t)out_height,
                              video_type.value,
                              color_planes,
                              nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};
    auto storage_type = nvidia::gxf::MemoryStorageType::kDevice;
    // auto storage_type = nvidia::gxf::MemoryStorageType::kHost;
    buffer.value()->wrapMemory(info, out_size, storage_type, (void*)image->cu_dst, nullptr);
    // buffer.value()->wrapMemory(info, out_size, storage_type, (void*) image->data, nullptr);
    const auto result = video_buffer_output_->publish(std::move(message.value()));
    return nvidia::gxf::ToResultCode(message);
  }

#ifdef BUILD_WITH_QCAP_SDK
  // GXF_LOG_ERROR("QCAP Source: status %d block >>", m_status);
  if (m_queue.pop_block(preview) == false) {
    // GXF_LOG_ERROR("QCAP Source: status %d block <<<", m_status);
    return GXF_SUCCESS;
  }
  // GXF_LOG_ERROR("QCAP Source: status %d block <<", m_status);

  PVOID pRCBuffer = QCAP_BUFFER_GET_RCBUFFER(preview.pFrameBuffer, preview.nFrameBufferLen);
  qcap_av_frame_t* pAVFrame = (qcap_av_frame_t*)QCAP_RCBUFFER_LOCK_DATA(pRCBuffer);

  PVOID frame = pAVFrame->pData[0];
  NppStatus status;
  NppiSize oSizeROI;
  int video_width = m_nVideoWidth;
  int video_height = m_nVideoHeight;
  oSizeROI.width = video_width;
  oSizeROI.height = video_height;
  m_nRGBBufferIndex = (m_nRGBBufferIndex + 1) % kDefaultColorConvertBufferSize;
  frame = m_pRGBBUffer[m_nRGBBufferIndex];
  auto storage_type = use_rdma_ ? nvidia::gxf::MemoryStorageType::kDevice : nvidia::gxf::MemoryStorageType::kHost;

#if 0  // for debug
  struct cudaPointerAttributes attributes;
  if (cudaPointerGetAttributes(&attributes, pAVFrame->pData[0]) != cudaSuccess)
  {
      throw std::runtime_error("cudaPointerGetAttributes failed.");
  }
  GXF_LOG_INFO("video preview cb frame: %p type: %d\n", pAVFrame->pData[0], attributes.type);
#endif

  storage_type = nvidia::gxf::MemoryStorageType::kDevice;
  if (pixel_format_ == PIXELFORMAT_YUY2 &&
      output_pixel_format_ == PIXELFORMAT_RGB24) {  // YUY2 to RGB
    status = nppiYCbCr422ToRGB_8u_C2C3R(
        pAVFrame->pData[0], video_width * 2, (Npp8u*)frame, video_width * 3, oSizeROI);
  } else if (pixel_format_ == PIXELFORMAT_BGR24 &&
             output_pixel_format_ == PIXELFORMAT_RGB24) {  // Default is BGR. BGR to RGB
    const int aDstOrder[3] = {2, 1, 0};
    status = nppiSwapChannels_8u_C3R(
        pAVFrame->pData[0], video_width * 3, (Npp8u*)frame, video_width * 3, oSizeROI, aDstOrder);
  } else if (pixel_format_ == PIXELFORMAT_NV12 &&
             output_pixel_format_ == PIXELFORMAT_RGB24) {  // NV12 to RGB
    const int aDstOrder[3] = {2, 1, 0};
    Npp8u* input[2];
    input[0] = (Npp8u*) pAVFrame->pData[0];
    input[1] = (Npp8u*) pAVFrame->pData[1];
    status = nppiNV12ToRGB_8u_P2C3R(
        input, video_width, (Npp8u*)frame, video_width*3, oSizeROI);
  } else {
    status = NPP_ERROR;
  }

  QCAP_RCBUFFER_UNLOCK_DATA(pRCBuffer);
  QCAP_RCBUFFER_RELEASE(pRCBuffer);

  if (status != 0) {
    GXF_LOG_INFO("QCAP Source: convert error %d buffer %p(%08x) to %p(%08x) %dx%d\n",
                 status,
                 pAVFrame->pData[0],
                 pixel_format_,
                 frame,
                 output_pixel_format_,
                 video_width,
                 video_height);
    return GXF_FAILURE;
  }

  int out_width = m_nVideoWidth;
  int out_height = m_nVideoHeight;
  int out_size = out_width * out_height * 3;

  nvidia::gxf::VideoTypeTraits<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB> video_type;
  nvidia::gxf::VideoFormatSize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB> color_format;
  auto color_planes = color_format.getDefaultColorPlanes(out_width, out_height);
  //GXF_LOG_ERROR("QCAP Source: stride %dx%d %d", out_width, out_height, color_planes[0].stride);
  color_planes[0].stride = out_width * 3;
  nvidia::gxf::VideoBufferInfo info{(uint32_t)out_width,
                            (uint32_t)out_height,
                            video_type.value,
                            color_planes,
                            nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};
  buffer.value()->wrapMemory(info, out_size, storage_type, frame, nullptr);
  const auto result = video_buffer_output_->publish(std::move(message.value()));

  return nvidia::gxf::ToResultCode(message);
#else
  return GXF_SUCCESS;
#endif
}

}  // namespace holoscan
}  // namespace yuan
