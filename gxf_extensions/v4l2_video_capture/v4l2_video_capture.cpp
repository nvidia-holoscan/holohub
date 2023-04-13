/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "v4l2_video_capture.hpp"
#include <fcntl.h>
#include <libv4l2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <string>
#include <utility>

#include "gxf/core/handle.hpp"
#include "gxf/multimedia/video.hpp"

#define CLEAR(x) memset(&(x), 0, sizeof(x))

namespace nvidia {
namespace holoscan {

static constexpr char kDefaultDevice[] = "/dev/video0";
static constexpr char kDefaultPixelFormat[] = "RGBA32";
static constexpr uint32_t kDefaultWidth = 1920;
static constexpr uint32_t kDefaultHeight = 1080;
static constexpr uint32_t kDefaultNumBuffers = 2;

gxf_result_t V4L2VideoCapture::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(signal_, "signal", "Output", "Output channel");
  result &= registrar->parameter(allocator_, "allocator", "Allocator", "Output Allocator");
  result &= registrar->parameter(
      device_, "device", "VideoDevice", "Path to the V4L2 device", std::string(kDefaultDevice));
  result &=
      registrar->parameter(width_, "width", "Width", "Width of the V4L2 image", kDefaultWidth);
  result &=
      registrar->parameter(height_, "height", "Height", "Height of the V4L2 image", kDefaultHeight);
  result &= registrar->parameter(num_buffers_,
                                 "numBuffers",
                                 "NumBuffers",
                                 "Number of V4L2 buffers to use",
                                 kDefaultNumBuffers);
  result &= registrar->parameter(pixel_format_,
                                 "pixel_format",
                                 "Pixel Format",
                                 "Pixel format of capture stream (RGBA32 or YUYV)",
                                 std::string(kDefaultPixelFormat));
  return gxf::ToResultCode(result);
}

gxf_result_t V4L2VideoCapture::start() {
  gxf_result_t result = v4l2_initialize();
  if (result == GXF_SUCCESS) { result = v4l2_set_mode(); }
  if (result == GXF_SUCCESS) { result = v4l2_requestbuffers(); }
  if (result == GXF_SUCCESS) { result = v4l2_start(); }

  return result;
}

gxf_result_t V4L2VideoCapture::stop() {
  // stream off
  enum v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (-1 == ioctl(fd_, VIDIOC_STREAMOFF, &buf_type)) {
    GXF_LOG_ERROR("StreamOFF Ioctl Failed");
    return GXF_FAILURE;
  }

  // free buffers
  for (uint32_t i = 0; i < num_buffers_.get(); ++i)
    if (-1 == munmap(buffers_[i].ptr, buffers_[i].length)) {
      GXF_LOG_ERROR("munmap Failed for index %d", i);
      return GXF_FAILURE;
    }
  free(buffers_);

  // close FD
  if (-1 == v4l2_close(fd_)) {
    GXF_LOG_ERROR("Close failed");
    return GXF_FAILURE;
  }

  fd_ = -1;

  return GXF_SUCCESS;
}

gxf_result_t V4L2VideoCapture::tick() {
  auto message = gxf::Entity::New(context());
  if (!message) {
    GXF_LOG_ERROR("Failed to allocate message");
    return GXF_FAILURE;
  }

  auto rgba_buf = message.value().add<gxf::VideoBuffer>();
  if (!rgba_buf) {
    GXF_LOG_ERROR("Failed to allocate RGBA buffer");
    return GXF_FAILURE;
  }

  // Read buffer.
  struct v4l2_buffer buf;
  CLEAR(buf);
  gxf_result_t results_read = v4l2_read_buffer(buf);
  if (results_read == GXF_FAILURE) {
    GXF_LOG_ERROR("Failed to read buffer");
    return GXF_FAILURE;
  }

  // Allocate RGBA output buffer.
  rgba_buf.value()->resize<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
      width_.get(),
      height_.get(),
      gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
      gxf::MemoryStorageType::kHost,
      allocator_);
  if (!rgba_buf.value()->pointer()) {
    GXF_LOG_ERROR("Failed to allocate RGBA buffer.");
    return GXF_FAILURE;
  }
  Buffer& read_buf = buffers_[buf.index];
  if (pixel_format_.get().compare("YUYV") == 0) {
    // Convert YUYV to RGBA output buffer
    YUYVToRGBA(read_buf.ptr, rgba_buf.value()->pointer(), width_.get(), height_.get());
  } else {
    // Wrap memory into output buffer
    rgba_buf.value()->wrapMemory(rgba_buf.value()->video_frame_info(),
                                 4 * width_.get() * height_.get(),
                                 gxf::MemoryStorageType::kHost,
                                 read_buf.ptr,
                                 nullptr);
  }

  // Return (queue) the buffer.
  if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
    GXF_LOG_ERROR("Failed to queue buffer %d on %s", buf.index, device_.get().c_str());
    return GXF_FAILURE;
  }

  const auto result = signal_->publish(std::move(message.value()));

  return gxf::ToResultCode(message);
}

gxf_result_t V4L2VideoCapture::v4l2_initialize() {
  // Initialise V4L2 device
  fd_ = v4l2_open(device_.get().c_str(), O_RDWR);
  if (fd_ < 0) {
    GXF_LOG_ERROR("Failed to open device, OPEN");
    return GXF_FAILURE;
  }

  struct v4l2_capability caps;
  ioctl(fd_, VIDIOC_QUERYCAP, &caps);
  if (!(caps.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
    GXF_LOG_ERROR("No V4l2 Video capture node");
    return GXF_FAILURE;
  }
  if (!(caps.capabilities & V4L2_CAP_STREAMING)) {
    GXF_LOG_ERROR("Does not support streaming i/o");
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t V4L2VideoCapture::v4l2_requestbuffers() {
  // Request V4L2 buffers
  struct v4l2_requestbuffers req;
  CLEAR(req);
  req.count = num_buffers_.get();
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;

  if (-1 == ioctl(fd_, VIDIOC_REQBUFS, &req)) {
    if (errno == EINVAL)
      GXF_LOG_ERROR(
          "Video capturing or DMABUF streaming is not supported type %d memory %d count %d",
          req.type,
          req.memory,
          req.count);
    else
      GXF_LOG_ERROR("Request buffers Ioctl failed");
    return GXF_FAILURE;
  }

  buffers_ = (Buffer*)calloc(req.count, sizeof(*buffers_));
  if (!buffers_) {
    GXF_LOG_ERROR("Allocate buffers failed");
    return GXF_FAILURE;
  }

  for (uint32_t i = 0; i < req.count; ++i) {
    struct v4l2_buffer buf;
    CLEAR(buf);
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;

    if (-1 == ioctl(fd_, VIDIOC_QUERYBUF, &buf)) {
      GXF_LOG_ERROR("VIDIOC_QUERYBUF Ioctl failed");
      return GXF_FAILURE;
    }

    buffers_[i].length = buf.length;
    buffers_[i].ptr = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
    if (MAP_FAILED == buffers_[i].ptr) {
      GXF_LOG_ERROR("MMAP failed");
      return GXF_FAILURE;
    }
  }
  return GXF_SUCCESS;
}

gxf_result_t V4L2VideoCapture::v4l2_set_mode() {
  // Set V4L2 device mode
  struct v4l2_format vfmt;
  memset(&vfmt, 0, sizeof(vfmt));
  vfmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(fd_, VIDIOC_G_FMT, &vfmt) == -1) {
    GXF_LOG_ERROR("Get format Ioctl failed");
    return GXF_FAILURE;
  }

  // Get pixel format
  uint32_t pixel_format;
  if (pixel_format_.get().compare("RGBA32") == 0) {
    pixel_format = V4L2_PIX_FMT_RGBA32;
  } else if (pixel_format_.get().compare("YUYV") == 0) {
    pixel_format = V4L2_PIX_FMT_YUYV;
  } else {
    GXF_LOG_ERROR("Unsupported pixel format %s, supported formats are RGBA32 and YUYV",
                  pixel_format_.get());
    return GXF_FAILURE;
  }
  vfmt.fmt.pix_mp.width = width_.get();
  vfmt.fmt.pix_mp.height = height_.get();
  vfmt.fmt.pix.pixelformat = pixel_format;

  if (ioctl(fd_, VIDIOC_S_FMT, &vfmt) == -1) {
    if (errno == EINVAL) {
      GXF_LOG_ERROR("Requested buffer type not supported in Set FMT");
    } else {
      GXF_LOG_ERROR("Set FMT Ioctl failed with %d", errno);
    }
    return GXF_FAILURE;
  }
  if (vfmt.fmt.pix.width != width_.get() || vfmt.fmt.pix.height != height_.get()) {
    GXF_LOG_ERROR("Format not supported by device");
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t V4L2VideoCapture::v4l2_start() {
  // Start streaming on V4L2 device
  // queue capture plane into device
  for (uint32_t i = 0; i < num_buffers_.get(); i++) {
    struct v4l2_buffer buf;
    CLEAR(buf);

    buf.index = i;
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == ioctl(fd_, VIDIOC_QBUF, &buf)) {
      GXF_LOG_ERROR("Failed to queue buf, Ioctl failed");
      return GXF_FAILURE;
    }
  }

  enum v4l2_buf_type buf_type;
  buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (-1 == ioctl(fd_, VIDIOC_STREAMON, &buf_type)) {
    GXF_LOG_ERROR(" StreamOn Ioctl failed");
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t V4L2VideoCapture::v4l2_read_buffer(v4l2_buffer& buf) {
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(fd_, &fds);

  struct timeval tv;
  tv.tv_sec = 2;
  tv.tv_usec = 0;

  int r;
  r = select(fd_ + 1, &fds, NULL, NULL, &tv);

  if (-1 == r) {
    GXF_LOG_ERROR("Error in querying file descriptor");
    return GXF_FAILURE;
  }
  if (0 == r) {
    GXF_LOG_ERROR("Querying file descriptor timed out");
    return GXF_FAILURE;
  }

  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  if (-1 == ioctl(fd_, VIDIOC_DQBUF, &buf)) {
    GXF_LOG_ERROR("Failed to deque buffer");
    return GXF_FAILURE;
  }

  if (buf.index >= num_buffers_.get()) {
    GXF_LOG_ERROR("Buf index is %d more than the queue size %d", buf.index, num_buffers_.get());
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

void V4L2VideoCapture::YUYVToRGBA(const void* yuyv, void* rgba, size_t width, size_t height) {
  auto r_convert = [](int y, int cr) {
    double r = y + (1.4065 * (cr - 128));
    return static_cast<unsigned int>(std::max(0, std::min(255, static_cast<int>(r))));
  };
  auto g_convert = [](int y, int cb, int cr) {
    double g = y - (0.3455 * (cb - 128)) - (0.7169 * (cr - 128));
    return static_cast<unsigned int>(std::max(0, std::min(255, static_cast<int>(g))));
  };
  auto b_convert = [](int y, int cb) {
    double b = y + (1.7790 * (cb - 128));
    return static_cast<unsigned int>(std::max(0, std::min(255, static_cast<int>(b))));
  };

  const unsigned char* yuyv_buf = static_cast<const unsigned char*>(yuyv);
  unsigned char* rgba_buf = static_cast<unsigned char*>(rgba);

  for (unsigned int i = 0, j = 0; i < width * height * 4; i += 8, j += 4) {
    int cb = yuyv_buf[j + 1];
    int cr = yuyv_buf[j + 3];

    // First pixel
    int y = yuyv_buf[j];
    rgba_buf[i] = r_convert(y, cr);
    rgba_buf[i + 1] = g_convert(y, cb, cr);
    rgba_buf[i + 2] = b_convert(y, cb);
    rgba_buf[i + 3] = 255;

    // Second pixel
    y = yuyv_buf[j + 2];
    rgba_buf[i + 4] = r_convert(y, cr);
    rgba_buf[i + 5] = g_convert(y, cb, cr);
    rgba_buf[i + 6] = b_convert(y, cb);
    rgba_buf[i + 7] = 255;
  }
}

}  // namespace holoscan
}  // namespace nvidia
