// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "v4l2_source/v4l2_source.hpp"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "v4l2_depth_common/tensor_utils.hpp"

namespace holoscan::examples::v4l2_depth {
namespace {

// Request four MMAP slots so the driver can keep capturing while one slot is
// dequeued and copied. Drivers may return fewer slots; any positive count is
// accepted below.
constexpr std::uint32_t kRequestedBufferCount = 4U;
constexpr std::uint32_t kMaximumReasonableBufferCount = 64U;
constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;
constexpr std::int64_t kNanosecondsPerMicrosecond = 1'000LL;

[[nodiscard]] holoscan::expected<void, holoscan::Error> failure(
    holoscan::ErrorCode code, std::string message) {
  return holoscan::make_unexpected(holoscan::Error{code, std::move(message)});
}

[[nodiscard]] std::string errno_message(std::string_view operation, int error_number) {
  std::ostringstream message;
  message << operation << " failed: " << std::strerror(error_number);
  return message.str();
}

[[nodiscard]] std::string cuda_message(const char* operation, cudaError_t error) {
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorString(error);
  return message.str();
}

[[nodiscard]] std::string fourcc_string(std::uint32_t fourcc) {
  std::array<char, 5U> text{
      static_cast<char>(fourcc & 0xffU),
      static_cast<char>((fourcc >> 8U) & 0xffU),
      static_cast<char>((fourcc >> 16U) & 0xffU),
      static_cast<char>((fourcc >> 24U) & 0xffU),
      '\0',
  };
  for (std::size_t index = 0; index < 4U; ++index) {
    if (text[index] < 0x20 || text[index] > 0x7e) {
      text[index] = '?';
    }
  }
  return std::string{text.data()};
}

[[nodiscard]] std::uint32_t resolved_colorspace(const v4l2_pix_format& format) {
  if (format.colorspace != V4L2_COLORSPACE_DEFAULT) {
    return format.colorspace;
  }
  const bool is_sdtv = format.width <= 720U && format.height <= 576U;
  const bool is_hdtv = format.width >= 1280U || format.height >= 720U;
  return V4L2_MAP_COLORSPACE_DEFAULT(is_sdtv, is_hdtv);
}

void require_supported_colorimetry(const v4l2_pix_format& format) {
  const std::uint32_t colorspace = resolved_colorspace(format);
  const std::uint32_t ycbcr_encoding =
      format.ycbcr_enc == V4L2_YCBCR_ENC_DEFAULT
          ? V4L2_MAP_YCBCR_ENC_DEFAULT(colorspace)
          : format.ycbcr_enc;
  const std::uint32_t quantization =
      format.quantization == V4L2_QUANTIZATION_DEFAULT
          ? V4L2_MAP_QUANTIZATION_DEFAULT(false, colorspace, ycbcr_encoding)
          : format.quantization;
  const bool is_601 =
      ycbcr_encoding == V4L2_YCBCR_ENC_601 ||
      ycbcr_encoding == V4L2_YCBCR_ENC_SYCC;
  if (!is_601 || quantization != V4L2_QUANTIZATION_LIM_RANGE) {
    std::ostringstream message;
    message << "V4L2 driver selected unsupported YUYV colorimetry: colorspace="
            << colorspace << " ycbcr_enc=" << ycbcr_encoding
            << " quantization=" << quantization
            << "; this example requires limited-range BT.601-compatible YCbCr";
    throw holoscan::RuntimeError(holoscan::ErrorCode::kIncompatible, message.str());
  }
}

}  // namespace

V4L2SourceOp::V4L2SourceOp(std::string device, int width, int height, int fps)
    : device_(std::move(device)), width_(width), height_(height), fps_(fps) {
  if (device_.empty()) {
    throw std::invalid_argument("V4L2 device path must not be empty");
  }
  if (width_ <= 0 || height_ <= 0 || fps_ <= 0) {
    throw std::invalid_argument("V4L2 width, height, and fps must all be positive");
  }
  if (fps_ > kNanosecondsPerSecond) {
    throw std::invalid_argument("V4L2 fps exceeds the nanosecond clock resolution");
  }
  if ((width_ & 1) != 0) {
    throw std::invalid_argument("YUYV capture width must be even");
  }

  const auto unsigned_width = static_cast<std::size_t>(width_);
  const auto unsigned_height = static_cast<std::size_t>(height_);
  if (unsigned_width > std::numeric_limits<std::size_t>::max() / 2U) {
    throw std::invalid_argument("V4L2 frame row byte count overflows size_t");
  }
  row_bytes_ = unsigned_width * 2U;
  if (unsigned_height > std::numeric_limits<std::size_t>::max() / row_bytes_) {
    throw std::invalid_argument("V4L2 frame byte count overflows size_t");
  }
  frame_bytes_ = row_bytes_ * unsigned_height;
  poll_period_ = std::chrono::nanoseconds{kNanosecondsPerSecond / fps_};
}

V4L2SourceOp::~V4L2SourceOp() { cleanup(); }

void V4L2SourceOp::setup(holoscan::OperatorSpec& spec) {
  spec.output(frame, "frame")
      .max_emits_per_compute(1U)
      .produces_tensor(holoscan::TensorPortLayout{
          .memory_kind = holoscan::MemoryKind::kCudaDevice,
          .dtype = kUInt8Dtype,
          .rank = 3U,
      })
      .tensor_allocation(
          holoscan::TensorAllocationBounds{.max_rank = 3U, .max_byte_span = frame_bytes_});
}

holoscan::Contract V4L2SourceOp::contract() const {
  holoscan::Contract result;
  // The EA1 runtime has no pollable-fd temporal trigger. Poll the nonblocking
  // descriptor at the negotiated cadence; EAGAIN is an ordinary clock tick
  // with no publication.
  result.trigger(holoscan::OnClock{.period = poll_period_});
  return result;
}

void V4L2SourceOp::start() {
  // Lifecycle restart and rollback can invoke start after a prior partial
  // attempt. Always begin from a clean, idempotent state.
  cleanup();

  try {
    fd_ = ::open(device_.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd_ < 0) {
      const int error_number = errno;
      throw holoscan::RuntimeError(
          holoscan::ErrorCode::kNotFound, errno_message("open(" + device_ + ")", error_number));
    }

    v4l2_capability capability{};
    if (xioctl(VIDIOC_QUERYCAP, &capability) < 0) {
      const int error_number = errno;
      throw holoscan::RuntimeError(
          holoscan::ErrorCode::kFailure, errno_message("VIDIOC_QUERYCAP", error_number));
    }
    const std::uint32_t capabilities =
        (capability.capabilities & V4L2_CAP_DEVICE_CAPS) != 0U
            ? capability.device_caps
            : capability.capabilities;
    if ((capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0U ||
        (capabilities & V4L2_CAP_STREAMING) == 0U) {
      throw holoscan::RuntimeError(
          holoscan::ErrorCode::kNotSupported,
          "V4L2 device must support single-plane VIDEO_CAPTURE and STREAMING");
    }

    v4l2_format format{};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = static_cast<__u32>(width_);
    format.fmt.pix.height = static_cast<__u32>(height_);
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    format.fmt.pix.field = V4L2_FIELD_NONE;
    if (xioctl(VIDIOC_S_FMT, &format) < 0) {
      const int error_number = errno;
      throw holoscan::RuntimeError(
          holoscan::ErrorCode::kFailure, errno_message("VIDIOC_S_FMT", error_number));
    }
    if (format.type != V4L2_BUF_TYPE_VIDEO_CAPTURE ||
        format.fmt.pix.width != static_cast<__u32>(width_) ||
        format.fmt.pix.height != static_cast<__u32>(height_) ||
        format.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV ||
        format.fmt.pix.field != V4L2_FIELD_NONE) {
      std::ostringstream message;
      message << "requested exact " << width_ << 'x' << height_
              << " progressive YUYV, but the driver selected " << format.fmt.pix.width << 'x'
              << format.fmt.pix.height << ' ' << fourcc_string(format.fmt.pix.pixelformat)
              << " field=" << format.fmt.pix.field;
      throw holoscan::RuntimeError(holoscan::ErrorCode::kIncompatible, message.str());
    }
    require_supported_colorimetry(format.fmt.pix);

    bytes_per_line_ = format.fmt.pix.bytesperline;
    if (bytes_per_line_ == 0U) {
      bytes_per_line_ = row_bytes_;
    }
    if (bytes_per_line_ < row_bytes_ ||
        static_cast<std::size_t>(height_) >
            std::numeric_limits<std::size_t>::max() / bytes_per_line_) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kIncompatible,
                                   "V4L2 driver returned an invalid bytesperline");
    }

    v4l2_streamparm stream_parameters{};
    stream_parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    auto& capture_parameters = stream_parameters.parm.capture;  // codespell:ignore parm
    capture_parameters.timeperframe.numerator = 1U;
    capture_parameters.timeperframe.denominator = static_cast<__u32>(fps_);
    if (xioctl(VIDIOC_S_PARM, &stream_parameters) < 0) {
      const int error_number = errno;
      throw holoscan::RuntimeError(
          holoscan::ErrorCode::kFailure, errno_message("VIDIOC_S_PARM", error_number));
    }
    const std::uint32_t numerator = capture_parameters.timeperframe.numerator;
    const std::uint32_t denominator = capture_parameters.timeperframe.denominator;
    if (numerator == 0U || denominator == 0U ||
        static_cast<std::uint64_t>(denominator) !=
            static_cast<std::uint64_t>(fps_) * numerator) {
      std::ostringstream message;
      message << "requested exactly " << fps_ << " fps, but the driver selected " << numerator
              << '/' << denominator << " seconds per frame";
      throw holoscan::RuntimeError(holoscan::ErrorCode::kIncompatible, message.str());
    }

    v4l2_requestbuffers request{};
    request.count = kRequestedBufferCount;
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    if (xioctl(VIDIOC_REQBUFS, &request) < 0) {
      const int error_number = errno;
      throw holoscan::RuntimeError(
          holoscan::ErrorCode::kFailure, errno_message("VIDIOC_REQBUFS", error_number));
    }
    if (request.count == 0U || request.count > kMaximumReasonableBufferCount) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kResourceExhausted,
                                   "V4L2 driver returned an invalid MMAP buffer count");
    }

    buffers_.reserve(request.count);
    const std::size_t minimum_buffer_span =
        (static_cast<std::size_t>(height_) - 1U) * bytes_per_line_ + row_bytes_;
    for (std::uint32_t index = 0U; index < request.count; ++index) {
      v4l2_buffer buffer{};
      buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buffer.memory = V4L2_MEMORY_MMAP;
      buffer.index = index;
      if (xioctl(VIDIOC_QUERYBUF, &buffer) < 0) {
        const int error_number = errno;
        throw holoscan::RuntimeError(
            holoscan::ErrorCode::kFailure, errno_message("VIDIOC_QUERYBUF", error_number));
      }
      if (buffer.length < minimum_buffer_span) {
        throw holoscan::RuntimeError(
            holoscan::ErrorCode::kIncompatible,
            "V4L2 MMAP buffer is too small for the negotiated row stride");
      }

      void* address = ::mmap(
          nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buffer.m.offset);
      if (address == MAP_FAILED) {
        const int error_number = errno;
        throw holoscan::RuntimeError(
            holoscan::ErrorCode::kFailure, errno_message("mmap", error_number));
      }
      buffers_.push_back(MappedBuffer{
          .address = address,
          .length = buffer.length,
      });

      if (xioctl(VIDIOC_QBUF, &buffer) < 0) {
        const int error_number = errno;
        throw holoscan::RuntimeError(
            holoscan::ErrorCode::kFailure, errno_message("VIDIOC_QBUF", error_number));
      }
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(VIDIOC_STREAMON, &type) < 0) {
      const int error_number = errno;
      throw holoscan::RuntimeError(
          holoscan::ErrorCode::kFailure, errno_message("VIDIOC_STREAMON", error_number));
    }
    streaming_ = true;

    std::fprintf(stdout,
                 "[v4l2_source] streaming %dx%d YUYV at %d fps from %s "
                 "(stride=%zu, buffers=%zu)\n",
                 width_,
                 height_,
                 fps_,
                 device_.c_str(),
                 bytes_per_line_,
                 buffers_.size());
  } catch (const std::exception& error) {
    std::fprintf(stderr,
                 "[v4l2_source] startup failed for %s: %s\n",
                 device_.c_str(),
                 error.what());
    cleanup();
    throw;
  } catch (...) {
    std::fprintf(
        stderr, "[v4l2_source] startup failed for %s: unknown error\n", device_.c_str());
    cleanup();
    throw;
  }
}

void V4L2SourceOp::stop() { cleanup(); }

holoscan::expected<void, holoscan::Error> V4L2SourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (fd_ < 0 || !streaming_) {
    return failure(holoscan::ErrorCode::kNotReady, "V4L2 source is not streaming");
  }

  v4l2_buffer buffer{};
  buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buffer.memory = V4L2_MEMORY_MMAP;
  if (xioctl(VIDIOC_DQBUF, &buffer) < 0) {
    const int error_number = errno;
    if (error_number == EAGAIN || error_number == EWOULDBLOCK) {
      return {};
    }
    return failure(
        holoscan::ErrorCode::kFailure, errno_message("VIDIOC_DQBUF", error_number));
  }

  bool requeued = false;
  auto finish_buffer =
      [this, &buffer, &requeued](std::optional<holoscan::Error> prior = std::nullopt)
      -> holoscan::expected<void, holoscan::Error> {
    if (!requeued) {
      if (xioctl(VIDIOC_QBUF, &buffer) < 0) {
        const int error_number = errno;
        return failure(
            holoscan::ErrorCode::kFailure, errno_message("VIDIOC_QBUF", error_number));
      }
      requeued = true;
    }
    if (prior.has_value()) {
      return holoscan::make_unexpected(std::move(*prior));
    }
    return {};
  };

  if (buffer.index >= buffers_.size()) {
    return finish_buffer(holoscan::Error{
        holoscan::ErrorCode::kProtocolError, "V4L2 returned an out-of-range buffer index"});
  }
  if ((buffer.flags & V4L2_BUF_FLAG_ERROR) != 0U) {
    // The driver explicitly marked this capture as damaged. Recycle it and
    // wait for the next clock tick without publishing an invalid image.
    return finish_buffer();
  }

  const MappedBuffer& mapped = buffers_[buffer.index];
  const std::size_t required_source_span =
      (static_cast<std::size_t>(height_) - 1U) * bytes_per_line_ + row_bytes_;
  if (mapped.address == nullptr || mapped.length < required_source_span ||
      buffer.bytesused < required_source_span ||
      static_cast<std::size_t>(buffer.bytesused) > mapped.length) {
    return finish_buffer(holoscan::Error{
        holoscan::ErrorCode::kProtocolError,
        "V4L2 frame payload does not cover the negotiated row-strided image"});
  }

  const std::array<std::int64_t, 3U> shape{
      static_cast<std::int64_t>(height_),
      static_cast<std::int64_t>(width_),
      2,
  };
  auto loan =
      frame.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8Dtype});
  if (!loan) {
    holoscan::Error error = std::move(loan).error();
    auto requeue_result = finish_buffer();
    if (!requeue_result) {
      return requeue_result;
    }
    // A camera cannot be backpressured after dequeue. Dropping is the bounded,
    // real-time response when every output pool slot is still in flight.
    if (error.code == holoscan::ErrorCode::kResourceExhausted ||
        error.code == holoscan::ErrorCode::kBackpressured) {
      return {};
    }
    return holoscan::make_unexpected(std::move(error));
  }

  const auto runtime_stream = context.cuda_stream();
  cudaStream_t stream = runtime_stream.get();
  if (stream == nullptr) {
    return finish_buffer(holoscan::Error{
        holoscan::ErrorCode::kNotSupported, "V4L2 source has no runtime CUDA stream"});
  }

  auto writer = loan->write(runtime_stream);
  if (!writer) {
    return finish_buffer(std::move(writer).error());
  }
  if (writer->data() == nullptr || writer->byte_size() != frame_bytes_) {
    return finish_buffer(holoscan::Error{
        holoscan::ErrorCode::kFailure, "V4L2 output tensor pool returned an invalid allocation"});
  }

  const cudaError_t copied = cudaMemcpy2DAsync(writer->data(),
                                               row_bytes_,
                                               mapped.address,
                                               bytes_per_line_,
                                               row_bytes_,
                                               static_cast<std::size_t>(height_),
                                               cudaMemcpyHostToDevice,
                                               stream);
  if (copied != cudaSuccess) {
    return finish_buffer(
        holoscan::Error{holoscan::ErrorCode::kFailure,
                        cuda_message("cudaMemcpy2DAsync(YUYV H2D)", copied)});
  }

  // The driver is allowed to overwrite an MMAP buffer as soon as QBUF
  // succeeds. Complete the transfer before returning that buffer.
  const cudaError_t synchronized = cudaStreamSynchronize(stream);
  if (synchronized != cudaSuccess) {
    // A failed synchronization does not prove that the source read has
    // completed. Keep this buffer dequeued; the terminal compute error causes
    // STREAMOFF cleanup to reclaim it without letting the driver overwrite
    // memory that CUDA could still be reading.
    return failure(holoscan::ErrorCode::kFailure,
                   cuda_message("cudaStreamSynchronize(YUYV H2D)", synchronized));
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return finish_buffer(std::move(committed).error());
  }

  const holoscan::TimePoint captured = capture_time(buffer, context.activation_time());
  const std::uint64_t current_frame_id = ++frame_id_;
  auto requeue_result = finish_buffer();
  if (!requeue_result) {
    return requeue_result;
  }
  return frame.emit(
      std::move(*loan),
      holoscan::EmitOptions{.capture_time = captured, .frame_id = current_frame_id});
}

int V4L2SourceOp::xioctl(unsigned long request,  // NOLINT(runtime/int)
                         void* argument) const noexcept {
  int result = -1;
  do {
    result = ::ioctl(fd_, request, argument);
  } while (result < 0 && errno == EINTR);
  return result;
}

void V4L2SourceOp::cleanup() noexcept {
  if (fd_ < 0) {
    streaming_ = false;
    buffers_.clear();
    return;
  }

  if (streaming_) {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(VIDIOC_STREAMOFF, &type) < 0) {
      std::fprintf(stderr,
                   "[v4l2_source] VIDIOC_STREAMOFF failed during cleanup: %s\n",
                   std::strerror(errno));
    }
    streaming_ = false;
  }

  for (MappedBuffer& buffer : buffers_) {
    if (buffer.address != nullptr && buffer.address != MAP_FAILED) {
      if (::munmap(buffer.address, buffer.length) < 0) {
        std::fprintf(stderr,
                     "[v4l2_source] munmap failed during cleanup: %s\n",
                     std::strerror(errno));
      }
    }
    buffer = {};
  }
  buffers_.clear();

  v4l2_requestbuffers release{};
  release.count = 0U;
  release.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  release.memory = V4L2_MEMORY_MMAP;
  static_cast<void>(xioctl(VIDIOC_REQBUFS, &release));

  if (::close(fd_) < 0) {
    std::fprintf(
        stderr, "[v4l2_source] close failed during cleanup: %s\n", std::strerror(errno));
  }
  fd_ = -1;
  bytes_per_line_ = 0U;
}

holoscan::TimePoint V4L2SourceOp::capture_time(
    const v4l2_buffer& buffer, holoscan::TimePoint activation_time) const noexcept {
  if (!activation_time.valid()) {
    return {};
  }

  // V4L2 monotonic timestamps and the SDK activation clock have different
  // numeric origins. Preserve the driver's capture age by translating its
  // CLOCK_MONOTONIC coordinate into the activation clock at this callback.
  if ((buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) ==
      V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC) {
    timespec now{};
    if (::clock_gettime(CLOCK_MONOTONIC, &now) == 0 && buffer.timestamp.tv_sec >= 0 &&
        buffer.timestamp.tv_usec >= 0 && buffer.timestamp.tv_usec < 1'000'000) {
      const std::int64_t now_ns =
          static_cast<std::int64_t>(now.tv_sec) * kNanosecondsPerSecond + now.tv_nsec;
      const std::int64_t captured_ns =
          static_cast<std::int64_t>(buffer.timestamp.tv_sec) * kNanosecondsPerSecond +
          static_cast<std::int64_t>(buffer.timestamp.tv_usec) * kNanosecondsPerMicrosecond;
      if (captured_ns <= now_ns) {
        const std::int64_t age_ns = now_ns - captured_ns;
        return holoscan::TimePoint{
            .timestamp_ns = activation_time.timestamp_ns - age_ns,
            .clock_id = activation_time.clock_id,
        };
      }
    }
  }

  // The kernel did not identify a compatible clock domain. Activation time is
  // the earliest truthful SDK-domain timestamp available in that case.
  return activation_time;
}

}  // namespace holoscan::examples::v4l2_depth
