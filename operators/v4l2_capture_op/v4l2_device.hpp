// SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/// @file
/// @brief A V4L2 MMAP capture device split along the lifecycle stages that own each step.
///
/// The V4L2 streaming sequence is usually written as one `start()` that opens the descriptor,
/// negotiates a format, reserves and maps buffers, and turns the stream on, plus one `stop()` that
/// undoes all four. That collapse is convenient and wrong in a way that only shows up under
/// restart and failure: mapping buffers is a resource reservation, turning the stream on is a
/// readiness transition, and the two belong to different lifecycle barriers. A graph that has to
/// reconfigure one operator should not have to re-open every descriptor to do it.
///
/// Each method here is one step of that sequence, so an operator can bind it to the stage that
/// actually owns it. Nothing in this file knows about Holoscan, and nothing in it knows about
/// Holoscan Sensor Bridge; it is the kernel interface and the C++ standard library only.
///
/// Errors are reported as `false` plus a \ref Device::last_error record rather than exceptions,
/// because the caller is a `noexcept` lifecycle callback.
///
/// This header is an implementation detail of \ref holoscan::holoscan_camera::V4l2CaptureOp and is
/// deliberately not installed: it would otherwise pull the Linux kernel headers into the include
/// path of every consumer of the operator.

#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string_view>
#include <vector>

namespace holoscan::holoscan_camera::v4l2 {

/// @brief Build a V4L2 pixel format code, matching the kernel's `v4l2_fourcc()`.
[[nodiscard]] constexpr std::uint32_t fourcc(char a, char b, char c, char d) noexcept {
  return static_cast<std::uint32_t>(static_cast<unsigned char>(a)) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(b)) << 8U) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(c)) << 16U) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(d)) << 24U);
}

inline constexpr std::uint32_t kFourccYuyv = fourcc('Y', 'U', 'Y', 'V');

/// @brief Render a pixel format code as its four characters.
[[nodiscard]] inline std::array<char, 5U> fourcc_name(std::uint32_t code) noexcept {
  return {static_cast<char>(code & 0xFFU), static_cast<char>((code >> 8U) & 0xFFU),
          static_cast<char>((code >> 16U) & 0xFFU), static_cast<char>((code >> 24U) & 0xFFU), '\0'};
}

/// @brief Describe one capture geometry.
struct Format {
  std::uint32_t code{kFourccYuyv};  ///< Pixel format.
  std::uint32_t width{};            ///< Pixels per line.
  std::uint32_t height{};           ///< Lines per frame.
  std::uint32_t bytes_per_line{};   ///< Driver-reported stride.
  std::uint32_t size_image{};       ///< Driver-reported maximum frame size in bytes.
};

/// @brief One frame interval as the driver states it, in seconds.
///
/// Kept as the rational V4L2 reports rather than a frames-per-second integer. The rates a camera
/// actually runs at are not all integers -- 30000/1001 is the common one -- and dividing them out
/// turns a rate the driver substituted into the rate that was asked for, which is the substitution
/// \ref Device::negotiate exists to refuse.
struct Interval {
  std::uint32_t numerator{};    ///< Seconds numerator, e.g. 1001.
  std::uint32_t denominator{};  ///< Seconds denominator, e.g. 30000.

  /// @brief Return whether this interval is exactly \p fps frames per second.
  /// @param fps Frames per second to compare against.
  /// @note Cross-multiplied in 64-bit rather than divided, so a rate that is not a whole number of
  /// frames per second compares unequal instead of truncating to one that is.
  [[nodiscard]] constexpr bool equals_rate(std::uint32_t fps) const noexcept {
    return numerator != 0U && denominator != 0U &&
           std::uint64_t{denominator} == std::uint64_t{fps} * numerator;
  }
};

/// @brief Name the clock the driver stamps buffers with.
///
/// A capture timestamp is meaningless without knowing which clock produced it, and V4L2 is one of
/// the few interfaces that says so explicitly. Reading this flag instead of assuming is the
/// difference between a timestamp that can be projected into another domain and one that can only
/// be guessed at.
enum class TimestampDomain : std::uint8_t {
  kUnknown = 0,    ///< The driver does not state a clock; the coordinate is not projectable.
  kMonotonic = 1,  ///< `CLOCK_MONOTONIC` on this host.
  kCopy = 2,       ///< Copied from an output buffer; meaningful only for mem-to-mem devices.
};

/// @brief Name the event within the frame that the driver's timestamp refers to.
///
/// Orthogonal to \ref TimestampDomain and carried separately because the two answer different
/// questions. The domain says which clock the number is expressed in, and so whether it can be
/// projected at all; the source says which instant of the frame it marks, and so what the number
/// means once projected. A stamp can be perfectly projectable and still refer to the wrong event.
///
/// V4L2 encodes the source in its own flag field and defaults it to end-of-frame, which is the
/// distinction that matters here: end-of-frame trails the start of integration by the exposure
/// plus the readout, so publishing one as the other is not a rounding error but a systematic bias
/// that grows with exposure time.
enum class TimestampSource : std::uint8_t {
  kUnknown = 0,          ///< The driver names a source this build does not recognize.
  kEndOfFrame = 1,       ///< V4L2's default: the last row of the frame was received.
  kStartOfExposure = 2,  ///< Integration began; the only source the SDK's capture time admits.
};

/// @brief Read which clock the driver stamped a buffer against.
/// @param flags `v4l2_buffer::flags` as the driver returned them.
/// @note A free function rather than a device member because it is a pure decoding of one word and
/// depends on nothing the device holds. That also makes the mapping testable without a capture
/// device, which is the only part of the dequeue path that can be checked without one.
[[nodiscard]] constexpr TimestampDomain domain_of(std::uint32_t flags) noexcept {
  switch (flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) {
    case V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC:
      return TimestampDomain::kMonotonic;
    case V4L2_BUF_FLAG_TIMESTAMP_COPY:
      return TimestampDomain::kCopy;
    default:
      return TimestampDomain::kUnknown;
  }
}

/// @brief Read which instant of the frame the driver's timestamp marks.
/// @param flags `v4l2_buffer::flags` as the driver returned them.
/// @note A separate mask from \ref domain_of over the same word. End-of-frame is the zero value of
/// this field, so a driver that says nothing is reported as end-of-frame rather than as unknown:
/// V4L2 defines that default, and treating a stated default as an absence would discard the one
/// thing the kernel does guarantee here.
[[nodiscard]] constexpr TimestampSource source_of(std::uint32_t flags) noexcept {
  switch (flags & V4L2_BUF_FLAG_TSTAMP_SRC_MASK) {
    case V4L2_BUF_FLAG_TSTAMP_SRC_EOF:
      return TimestampSource::kEndOfFrame;
    case V4L2_BUF_FLAG_TSTAMP_SRC_SOE:
      return TimestampSource::kStartOfExposure;
    default:
      return TimestampSource::kUnknown;
  }
}

/// @brief Whether a granted field order describes a whole progressive frame.
/// @param field `v4l2_pix_format::field` as the driver returned it.
/// @note Only `V4L2_FIELD_NONE` qualifies, and the rejection is deliberately not a list of the
/// layouts V4L2 defines today: anything this build does not recognize is refused rather than
/// assumed progressive, so a kernel that adds a layout cannot have it silently published as one.
/// `V4L2_FIELD_ANY` is refused for the same reason -- it is a request value, and a driver
/// returning it has answered the question with the question.
[[nodiscard]] constexpr bool is_progressive(std::uint32_t field) noexcept {
  return field == static_cast<std::uint32_t>(V4L2_FIELD_NONE);
}

/// @brief Report which call failed and why.
struct Failure {
  std::string_view op{};  ///< Failing operation, e.g. "VIDIOC_STREAMON".
  int error{};            ///< `errno` captured immediately after the failure.
};

/// @brief One dequeued capture buffer, valid until it is requeued.
struct Frame {
  const std::uint8_t* data{};   ///< Mapped buffer contents.
  std::size_t bytes{};          ///< Bytes the driver actually wrote.
  std::int64_t timestamp_ns{};  ///< Capture coordinate in the device's timestamp domain.
  /// @brief Instant of the frame \ref timestamp_ns marks.
  /// @note Per-frame rather than per-device, because V4L2 states it per buffer and a driver is
  /// entitled to change it mid-stream. Reading it from the device would report the last frame's
  /// source for the current frame's timestamp.
  TimestampSource timestamp_source{TimestampSource::kUnknown};
  std::uint32_t sequence{};  ///< Driver frame counter; gaps in it are dropped frames.
  std::uint32_t index{};     ///< Buffer index, required by \ref Device::requeue.
  bool driver_error{};       ///< Driver flagged the frame as corrupt but still delivered it.
};

/// @brief What ended a \ref Device::wait_readable.
enum class Readiness : std::uint8_t {
  kReadable,  ///< A completed buffer is waiting to be dequeued.
  kTimedOut,  ///< Nothing became ready before the timeout expired.
  kFailed,    ///< The descriptor reported an error condition instead.
};

/// @brief The outcome of one \ref Device::wait_readable, with what to report when it failed.
/// @note Carried in the return value rather than recorded in \ref Device::last_error, because the
/// wait runs on the reader thread while the owning thread may be inside another call. Writing the
/// shared failure record from here would race with the owner reading it.
struct Wait {
  Readiness readiness{Readiness::kTimedOut};  ///< What ended the wait.
  int error{};    ///< `errno` when `poll` itself failed, zero otherwise.
  int revents{};  ///< Conditions `poll` reported on the descriptor, zero otherwise.
};

/// @brief The outcome of one \ref Device::dequeue.
struct Dequeued {
  std::optional<Frame> frame{};  ///< The completed buffer, absent when none was taken.
  /// @brief Whether the driver refused the request, as against having nothing ready.
  /// @note An empty queue is ordinary; a spurious wake produces one. Only a refusal is worth
  /// reporting, and \ref Device::last_error describes it.
  bool failed{};
};

/// @brief Whether a driver-reported byte count fits the mapping it refers to.
/// @param bytesused `v4l2_buffer::bytesused` as the driver returned it.
/// @param length Mapping capacity this process recorded at `VIDIOC_QUERYBUF`.
/// @note Equality is admitted, and that is the case that matters rather than an edge: a full frame
/// reports exactly the mapping's length, so a bound that excluded equality would reject every
/// ordinary frame. Only a count *exceeding* what was mapped is a value no copy may be sized by.
[[nodiscard]] constexpr bool fits_mapping(std::uint32_t bytesused, std::size_t length) noexcept {
  return static_cast<std::size_t>(bytesused) <= length;
}

/// @brief Drive one V4L2 MMAP capture device one lifecycle step at a time.
///
/// The object is not thread-safe with one exception that the design depends on: \ref
/// wait_readable only reads the descriptor and may be called from a separate reader thread while
/// the owning thread is not inside another call.
class Device {
 public:
  Device() = default;
  ~Device() { close(); }

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(Device&&) = delete;

  /// @brief Open the device node and confirm it can stream captured video.
  /// @param path Device node, typically `/dev/video0`.
  /// @return Whether the node was opened and advertises capture plus streaming.
  /// @note Suitable for a `kConfigure` body. A missing or unsupported node fails here, which is
  /// early enough for the graph to refuse to start rather than to fail mid-stream.
  /// @note All-or-nothing: a node that opens and is then rejected leaves no descriptor behind, so
  /// \ref is_open always agrees with what this returned. That is what lets an owner treat \ref
  /// is_open as the record of whether the `kConfigure` step still holds anything to release.
  [[nodiscard]] bool open(const char* path) noexcept {
    close();
    fd_ = ::open(path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd_ < 0) {
      return fail("open");
    }
    v4l2_capability capability{};
    if (ioctl_retry(VIDIOC_QUERYCAP, &capability) < 0) {
      return fail_open("VIDIOC_QUERYCAP");
    }
    const std::uint32_t caps = (capability.capabilities & V4L2_CAP_DEVICE_CAPS) != 0U
                                   ? capability.device_caps
                                   : capability.capabilities;
    if ((caps & V4L2_CAP_VIDEO_CAPTURE) == 0U || (caps & V4L2_CAP_STREAMING) == 0U) {
      errno = ENOTSUP;
      return fail_open("V4L2_CAP_VIDEO_CAPTURE|STREAMING");
    }
    return true;
  }

  /// @brief Negotiate a capture geometry and frame rate, refusing silent substitution.
  /// @param desired Requested pixel format and geometry.
  /// @param fps Requested frames per second.
  /// @return Whether the driver granted exactly what was asked for.
  /// @note Suitable for a `kDiscover` body. V4L2's `VIDIOC_S_FMT` and `VIDIOC_S_PARM` both succeed
  /// while quietly rounding what they were given -- the geometry or format for the first, the
  /// frame interval for the second -- so both are read back and compared, and either substitution
  /// fails the call. A pipeline sized for one geometry that silently receives another produces
  /// corrupt output that looks like a bug in whatever consumes it, and a run that believes it is
  /// capturing at a rate the driver never granted misreports every measurement taken against it.
  [[nodiscard]] bool negotiate(Format desired, std::uint32_t fps) noexcept {
    v4l2_format format{};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = desired.width;
    format.fmt.pix.height = desired.height;
    format.fmt.pix.pixelformat = desired.code;
    format.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl_retry(VIDIOC_S_FMT, &format) < 0) {
      return fail("VIDIOC_S_FMT");
    }
    if (format.fmt.pix.pixelformat != desired.code || format.fmt.pix.width != desired.width ||
        format.fmt.pix.height != desired.height) {
      errno = EINVAL;
      return fail("VIDIOC_S_FMT substituted a different mode");
    }
    // The field order is requested above and so has to be read back too, on exactly the reasoning
    // that applies to the geometry. It is the substitution with the quietest failure of the three:
    // an interlaced or field-sequential frame has the byte count a progressive one does, so it
    // passes every size check here and downstream, and is then published as one progressive image
    // with no field metadata and no deinterlacing. Alternate-field capture is worse than
    // mislabelled
    // -- a buffer may hold a single field rather than the frame the descriptor claims, so half the
    // declared rows are not the scene at the rows they occupy. Nothing about that is detectable
    // from the payload; a consumer receives a structurally valid image whose spatial and temporal
    // interpretation is wrong. Representing the other layouts would mean deinterlacing or field
    // pairing, neither of which belongs in a capture operator, so the substitution is refused.
    if (!is_progressive(format.fmt.pix.field)) {
      errno = EINVAL;
      return fail("VIDIOC_S_FMT substituted a different field order");
    }

    v4l2_streamparm parameters{};
    parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parameters.parm.capture.timeperframe.numerator = 1U;
    parameters.parm.capture.timeperframe.denominator = fps;
    if (ioctl_retry(VIDIOC_S_PARM, &parameters) < 0) {
      return fail("VIDIOC_S_PARM");
    }
    // Read back for the same reason the geometry is, and it is the weaker of the two guarantees:
    // VIDIOC_S_PARM is advisory, so a driver may return the nearest interval it supports, or its
    // own fixed one if it has no rate control at all, and report success either way. An
    // unverified rate is worse than an unverified geometry, because nothing downstream sees it:
    // wrong geometry corrupts the image, while a run that was granted 15 fps and believes it
    // asked for and received 30 looks correct and quietly invalidates every timing expectation
    // measured against it.
    const Interval granted{.numerator = parameters.parm.capture.timeperframe.numerator,
                           .denominator = parameters.parm.capture.timeperframe.denominator};
    if (!granted.equals_rate(fps)) {
      errno = EINVAL;
      return fail("VIDIOC_S_PARM substituted a different frame interval");
    }
    granted_ = Format{.code = format.fmt.pix.pixelformat,
                      .width = format.fmt.pix.width,
                      .height = format.fmt.pix.height,
                      .bytes_per_line = format.fmt.pix.bytesperline,
                      .size_image = format.fmt.pix.sizeimage};
    granted_interval_ = granted;
    return true;
  }

  /// @brief Reserve driver buffers, map them, and queue them for capture.
  /// @param count Requested buffer count; the driver may grant fewer.
  /// @return Whether every granted buffer was mapped and queued.
  /// @note Suitable for a `kAllocate` body. This is the step that actually commits memory, which
  /// is why it is separable from turning the stream on. It is all-or-nothing: a failure part way
  /// through leaves nothing mapped and no reservation held, so a caller that retries starts from
  /// the same state as the first attempt.
  [[nodiscard]] bool map_buffers(std::uint32_t count) noexcept {
    v4l2_requestbuffers request{};
    request.count = count;
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    if (ioctl_retry(VIDIOC_REQBUFS, &request) < 0) {
      return fail("VIDIOC_REQBUFS");
    }
    if (request.count == 0U) {
      errno = ENOMEM;
      return fail("VIDIOC_REQBUFS granted no buffers");
    }

    buffers_.resize(request.count);
    for (std::uint32_t index = 0U; index < request.count; ++index) {
      v4l2_buffer buffer{};
      buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buffer.memory = V4L2_MEMORY_MMAP;
      buffer.index = index;
      if (ioctl_retry(VIDIOC_QUERYBUF, &buffer) < 0) {
        return fail_mapping("VIDIOC_QUERYBUF");
      }
      void* mapped =
          ::mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buffer.m.offset);
      if (mapped == MAP_FAILED) {
        return fail_mapping("mmap");
      }
      buffers_[index] =
          MappedBuffer{.start = static_cast<std::uint8_t*>(mapped), .length = buffer.length};
      if (ioctl_retry(VIDIOC_QBUF, &buffer) < 0) {
        return fail_mapping("VIDIOC_QBUF");
      }
    }
    return true;
  }

  /// @brief Start streaming.
  /// @return Whether the driver accepted `VIDIOC_STREAMON`.
  /// @note Belongs in a `kStart` body, paired with the \ref stream_off in `kStop`. A restart
  /// requested at `kStart` replays only that pair, so activating the stream any earlier leaves it
  /// off for the resumed epoch.
  [[nodiscard]] bool stream_on() noexcept {
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl_retry(VIDIOC_STREAMON, &type) < 0) {
      return fail("VIDIOC_STREAMON");
    }
    streaming_ = true;
    return true;
  }

  /// @brief Stop streaming and return every queued buffer to the application.
  /// @return Whether the driver accepted `VIDIOC_STREAMOFF`, or `true` if not streaming.
  /// @note Suitable for a `kStop` body. The buffers stay mapped, so a later \ref stream_on resumes
  /// capture without reallocating.
  [[nodiscard]] bool stream_off() noexcept {
    if (!streaming_) {
      return true;
    }
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl_retry(VIDIOC_STREAMOFF, &type) < 0) {
      return fail("VIDIOC_STREAMOFF");
    }
    // Recorded after the driver accepts, as \ref stream_on records its own transition, so the flag
    // never claims a stop the driver rejected. Clearing it first would also spend the retry that
    // the \ref close call site exists to make: the guard above would short-circuit the second
    // attempt into a success, which is the one case that call is there for.
    streaming_ = false;
    return true;
  }

  /// @brief Unmap every buffer, release the driver's reservation, and close the descriptor.
  /// @note Suitable for a `kRelease` body, and safe to call more than once.
  void close() noexcept {
    if (fd_ < 0) {
      buffers_.clear();
      return;
    }
    static_cast<void>(stream_off());
    release_buffers();
    static_cast<void>(::close(fd_));
    fd_ = -1;
    streaming_ = false;
  }

  /// @brief Block until the device has a frame ready, the timeout expires, or the wait fails.
  /// @param timeout_ms Maximum wait in milliseconds.
  /// @return What ended the wait, with the `errno` or poll conditions behind a failure.
  /// @note This is the call that belongs on a reader thread rather than inside `compute()`. It
  /// only reads the descriptor, so it is safe to run concurrently with the owning thread as long
  /// as that thread is not inside another member.
  [[nodiscard]] Wait wait_readable(int timeout_ms) const noexcept {
    pollfd waitable{.fd = fd_, .events = POLLIN, .revents = 0};
    const int ready = ::poll(&waitable, 1U, timeout_ms);
    if (ready < 0) {
      // A signal is not a device condition, and the caller comes straight back here.
      if (errno == EINTR) {
        return Wait{.readiness = Readiness::kTimedOut};
      }
      return Wait{.readiness = Readiness::kFailed, .error = errno};
    }
    if (ready == 0) {
      return Wait{.readiness = Readiness::kTimedOut};
    }
    if ((waitable.revents & POLLIN) != 0) {
      return Wait{.readiness = Readiness::kReadable};
    }
    // POLLERR, POLLHUP, or POLLNVAL. Reported rather than folded into the timeout above, because
    // an unplugged device returns from poll immediately and keeps doing so: read as "nothing ready
    // yet", it turns the reader into a hot loop that says nothing about a device that is gone.
    return Wait{.readiness = Readiness::kFailed, .revents = waitable.revents};
  }

  /// @brief Take the next completed buffer without waiting.
  /// @return The frame when one was ready, and whether the driver refused the request or answered
  /// it with a buffer this device never mapped.
  /// @note The returned pointer stays valid until \ref requeue is called for the same index.
  [[nodiscard]] Dequeued dequeue() noexcept {
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    if (ioctl_retry(VIDIOC_DQBUF, &buffer) < 0) {
      // An empty queue is ordinary, and a spurious wake produces one. A refusal is not, and the
      // two leave here distinguishable so that a caller can report one without reporting both.
      if (errno == EAGAIN) {
        return Dequeued{};
      }
      static_cast<void>(fail("VIDIOC_DQBUF"));
      return Dequeued{.failed = true};
    }
    // The index selects a mapped buffer below, and it arrives from the kernel rather than from
    // this process. videobuf2 keeps it inside the count REQBUFS granted, so a conforming driver
    // cannot fail this; the check is here because the cost of being wrong is not proportionate to
    // the cost of asking. std::vector::operator[] does not range-check, the operator opens
    // whatever node it was pointed at including out-of-tree and virtual drivers, and an
    // out-of-range read here does not fault at the point of the mistake: it yields a garbage
    // pointer that is dereferenced later as the frame payload.
    if (buffer.index >= buffers_.size()) {
      errno = ERANGE;
      static_cast<void>(fail("VIDIOC_DQBUF returned an out-of-range buffer index"));
      return Dequeued{.failed = true};
    }
    // The byte count arrives from the same place the index did, and validating one without the
    // other leaves the read unbounded for the same reason: `bytesused` is what every copy out of
    // this mapping is sized against, so a value larger than the mapping reads past its end. The
    // consumer-side clamp is not this check. That one bounds the copy by the *destination* extent
    // and by this very number, which is exactly the pair that fails together -- a destination as
    // large as a full frame and a `bytesused` claiming more than a short mapping holds is an
    // in-bounds write fed by an out-of-bounds read. Capacity this process recorded at QUERYBUF is
    // the only bound the kernel does not get to choose.
    const MappedBuffer& mapped = buffers_[buffer.index];
    if (!fits_mapping(buffer.bytesused, mapped.length)) {
      errno = EOVERFLOW;
      static_cast<void>(fail("VIDIOC_DQBUF reported more bytes than the buffer maps"));
      return Dequeued{.failed = true};
    }
    timestamp_domain_ = domain_of(buffer.flags);
    return Dequeued{
        .frame = Frame{
            .data = mapped.start,
            .bytes = buffer.bytesused,
            .timestamp_ns = static_cast<std::int64_t>(buffer.timestamp.tv_sec) * 1'000'000'000LL +
                            static_cast<std::int64_t>(buffer.timestamp.tv_usec) * 1'000LL,
            .timestamp_source = source_of(buffer.flags),
            .sequence = buffer.sequence,
            .index = buffer.index,
            .driver_error = (buffer.flags & V4L2_BUF_FLAG_ERROR) != 0U,
        }};
  }

  /// @brief Give a dequeued buffer back to the driver.
  /// @param index Buffer index from the frame being retired.
  /// @return Whether the driver accepted the buffer.
  /// @note A buffer that is never requeued is permanently lost to the capture pool, which starves
  /// the device a few frames later rather than at the point of the mistake.
  [[nodiscard]] bool requeue(std::uint32_t index) noexcept {
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = index;
    if (ioctl_retry(VIDIOC_QBUF, &buffer) < 0) {
      return fail("VIDIOC_QBUF");
    }
    return true;
  }

  /// @brief Read `CLOCK_MONOTONIC`, the domain a V4L2 monotonic buffer timestamp belongs to.
  /// @return Current coordinate in nanoseconds, or absence if the clock could not be read.
  [[nodiscard]] static std::optional<std::int64_t> monotonic_now_ns() noexcept {
    timespec now{};
    if (::clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
      return std::nullopt;
    }
    return static_cast<std::int64_t>(now.tv_sec) * 1'000'000'000LL +
           static_cast<std::int64_t>(now.tv_nsec);
  }

  /// @brief Return the geometry the driver granted.
  [[nodiscard]] const Format& format() const noexcept { return granted_; }

  /// @brief Return the frame interval the driver granted.
  /// @note Exactly the interval that was asked for, since \ref negotiate refuses any other, and
  /// carried as the driver's own rational so a caller reporting it cannot round it back into a
  /// rate the driver never granted.
  [[nodiscard]] const Interval& interval() const noexcept { return granted_interval_; }

  /// @brief Return the number of mapped buffers.
  [[nodiscard]] std::size_t buffer_count() const noexcept { return buffers_.size(); }

  /// @brief Return the clock the most recent frame's timestamp belongs to.
  /// @note Absent until the first frame is dequeued, because V4L2 reports the domain per buffer.
  [[nodiscard]] TimestampDomain timestamp_domain() const noexcept { return timestamp_domain_; }

  /// @brief Return whether a device node is currently open.
  [[nodiscard]] bool is_open() const noexcept { return fd_ >= 0; }

  /// @brief Return the operation and `errno` of the most recent failure.
  [[nodiscard]] const Failure& last_error() const noexcept { return failure_; }

 private:
  struct MappedBuffer {
    std::uint8_t* start{};
    std::size_t length{};
  };

  /// @brief Unmap every buffer and hand the reservation back to the driver.
  /// @note Kept separate from \ref close so a failed \ref map_buffers can undo its partial work
  /// without tearing the descriptor down. The driver refuses a later `VIDIOC_REQBUFS` with `EBUSY`
  /// while any buffer from the previous request is still mapped, so a retry depends on this having
  /// run. Returning the reservation before closing also keeps a driver that outlives this
  /// descriptor from holding the buffers until the last reference goes away.
  void release_buffers() noexcept {
    for (MappedBuffer& buffer : buffers_) {
      if (buffer.start != nullptr) {
        static_cast<void>(::munmap(buffer.start, buffer.length));
      }
    }
    buffers_.clear();
    v4l2_requestbuffers release{};
    release.count = 0U;
    release.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    release.memory = V4L2_MEMORY_MMAP;
    static_cast<void>(ioctl_retry(VIDIOC_REQBUFS, &release));
  }

  /// @brief Record a mapping failure and drop the partial reservation it would otherwise leave.
  /// @note The failure is recorded first because \ref release_buffers overwrites `errno`.
  [[nodiscard]] bool fail_mapping(std::string_view op) noexcept {
    const bool failed = fail(op);
    release_buffers();
    return failed;
  }

  /// @brief Record an open failure and drop the descriptor it would otherwise leave behind.
  /// @note \ref open is all-or-nothing for the same reason \ref map_buffers is. A node that opens
  /// but fails `VIDIOC_QUERYCAP`, or that opens and turns out not to be a streaming capture device,
  /// left a live descriptor behind a call that reported failure, so \ref is_open contradicted the
  /// return value and nothing but `~Device` would ever close it. A caller that retries or tears down
  /// now sees the same state it saw before the attempt. The failure is recorded first because \ref
  /// close overwrites `errno`.
  [[nodiscard]] bool fail_open(std::string_view op) noexcept {
    const bool failed = fail(op);
    close();
    return failed;
  }

  int ioctl_retry(unsigned long request, void* argument) noexcept {  // NOLINT(runtime/int)
    int result = 0;
    do {
      result = ::ioctl(fd_, request, argument);
    } while (result < 0 && errno == EINTR);
    return result;
  }

  [[nodiscard]] bool fail(std::string_view op) noexcept {
    failure_ = Failure{.op = op, .error = errno};
    return false;
  }

  int fd_{-1};
  bool streaming_{};
  Format granted_{};
  Interval granted_interval_{};
  TimestampDomain timestamp_domain_{TimestampDomain::kUnknown};
  std::vector<MappedBuffer> buffers_;
  Failure failure_{};
};

}  // namespace holoscan::holoscan_camera::v4l2
