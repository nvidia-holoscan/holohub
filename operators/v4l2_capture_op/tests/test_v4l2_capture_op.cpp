// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// GTest coverage for the construction contract of V4l2CaptureOp, and for the restart contract its
// lifecycle stage placement depends on.
//
// These tests cover what is checkable without a capture device: the argument validation, the
// geometry arithmetic, and the guard sequencing a start-only restart relies on. Streaming
// behaviour is covered by the application's --capture mode,
// which needs hardware and therefore cannot run on every CI runner.

#include <gtest/gtest.h>

// clang-format off: see the note on the include block in v4l2_capture_op.hpp. IncludeBlocks:
// Regroup would sort <holoscan/...> in among the standard library headers below, which is the order
// cpplint's build/include_order check rejects.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <holoscan/core/lifecycle.hpp>
#include <holoscan/sensor_io/lifecycle_guards.hpp>
#include <v4l2_capture_op/v4l2_capture_op.hpp>
#include <v4l2_capture_op/v4l2_device.hpp>
// clang-format on

namespace mm = holoscan::holoscan_camera;

TEST(V4l2CaptureOpTest, DefaultContractParameters) {
  const mm::V4l2CaptureOp op;

  EXPECT_EQ(op.device(), "/dev/video0");
  EXPECT_EQ(op.width(), 640);
  EXPECT_EQ(op.height(), 480);
  EXPECT_EQ(op.fps(), 30);
  EXPECT_EQ(op.frame_id(), "camera_optical_frame");
  // The driver delivers YUYV 4:2:2 and the operator publishes it whole, so one figure bounds both
  // the driver buffer accepted at kDiscover and the tensor declared at setup(). It is also what the
  // schema requires of a 640x480 YUYV frame, since an even width makes 4 * ceil(width / 2) equal
  // 2 * width; a Tensor short of it would be refused on emit.
  EXPECT_EQ(op.capture_bytes(), 640U * 480U * 2U);
}

TEST(V4l2CaptureOpTest, AcceptsExplicitContractParameters) {
  const mm::V4l2CaptureOp op{"/dev/video2", 1280, 720, 60, "left_camera_optical_frame"};

  EXPECT_EQ(op.device(), "/dev/video2");
  EXPECT_EQ(op.width(), 1280);
  EXPECT_EQ(op.height(), 720);
  EXPECT_EQ(op.fps(), 60);
  EXPECT_EQ(op.frame_id(), "left_camera_optical_frame");
  EXPECT_EQ(op.capture_bytes(), 1280U * 720U * 2U);
}

TEST(V4l2CaptureOpTest, RejectsInvalidContractParameters) {
  EXPECT_THROW(mm::V4l2CaptureOp("", 640, 480, 30), std::invalid_argument);
  EXPECT_THROW(mm::V4l2CaptureOp("/dev/video0", 0, 480, 30), std::invalid_argument);
  EXPECT_THROW(mm::V4l2CaptureOp("/dev/video0", 640, 0, 30), std::invalid_argument);
  EXPECT_THROW(mm::V4l2CaptureOp("/dev/video0", 640, 480, 0), std::invalid_argument);
  EXPECT_THROW(mm::V4l2CaptureOp("/dev/video0", 641, 480, 30), std::invalid_argument);
  EXPECT_THROW(mm::V4l2CaptureOp("/dev/video0", 640, 480, 1'000'000'001), std::invalid_argument);
}

TEST(V4l2CaptureOpTest, RejectsAnUnnamedCoordinateFrame) {
  // An image whose coordinate frame is the empty string is indistinguishable from one whose frame
  // was never set, and a consumer cannot place it relative to any other sensor.
  EXPECT_THROW(mm::V4l2CaptureOp("/dev/video0", 640, 480, 30, ""), std::invalid_argument);
}

TEST(V4l2CaptureOpTest, DefaultsToHostPlacement) {
  // The placement is frozen into the port contract at setup() and cannot be renegotiated, so the
  // default is the one that leaves a graph which never asked for a placement where it already was.
  EXPECT_EQ(mm::V4l2CaptureOp().memory_kind(), holoscan::MemoryKind::kHost);
}

TEST(V4l2CaptureOpTest, CarriesTheRequestedPlacement) {
  for (const holoscan::MemoryKind kind :
       {holoscan::MemoryKind::kHost, holoscan::MemoryKind::kPinnedHost,
        holoscan::MemoryKind::kCudaDevice}) {
    const mm::V4l2CaptureOp op("/dev/video0", 640, 480, 30, std::string{mm::kDefaultCameraFrameId},
                               kind);
    EXPECT_EQ(op.memory_kind(), kind);
  }
}

TEST(V4l2CaptureOpTest, RejectsAPlacementItCannotWrite) {
  // kUnknown never describes a published tensor, and kCudaManaged would appear to work through the
  // host write path while moving the cost of the frame's first device touch somewhere this operator
  // cannot account for it. Refusing both here reports the cause rather than the later symptom.
  for (const holoscan::MemoryKind kind :
       {holoscan::MemoryKind::kUnknown, holoscan::MemoryKind::kCudaManaged}) {
    EXPECT_THROW(mm::V4l2CaptureOp("/dev/video0", 640, 480, 30,
                                   std::string{mm::kDefaultCameraFrameId}, kind),
                 std::invalid_argument);
  }
}

// A start-only restart has to replay the stage that owns VIDIOC_STREAMON and clear what the
// previous stream left behind, or the reader thread resumes against a device that is no longer
// streaming, or against a raised post interlock that no activation will ever lower, and the
// operator reports a healthy restart while producing no frames. V4l2CaptureOp answers that by
// keeping stream_on() and the sequence and interlock resets in on_start, and leaving on_arm holding
// only the notification sender, which kStop does not release.
//
// This asserts the sequencing that placement depends on -- the guard every stage of the operator is
// routed through -- and not the operator's own bodies. Reaching on_start from a test needs two
// things this repository does not have. on_arm takes a holoscan::LifecycleContext to acquire its
// notification sender, and LifecycleContext's only constructor is private to
// holoscan::detail::LifecycleContextTestAccess, which the SDK defines under public/src and does not
// install; and on_configure opens the device node through a concrete v4l2::Device held by
// unique_ptr with no injection point, so on a runner without a camera kConfigure fails and every
// later stage guard-skips. Asserting the placement of stream_on(), or that a restart lowers an
// interlock the previous stream left raised, therefore needs an installed lifecycle test harness,
// or a device seam plus a fake that records STREAMON/STREAMOFF.
//
// The subset that does run is not a restatement of the guard's own unit tests. It is the one
// property this operator would be broken by, pinned in the repository that would have to fix it: a
// StageGuards that invalidated its arm record on a successful kStop made exactly this kStart
// guard-skip, and Core reads a guard skip as success, so the regression is silent at the operator
// and visible only as a camera that stopped producing frames.
TEST(V4l2CaptureRestartContractTest, StartOnlyRestartReplaysTheStageOwningStreamOn) {
  holoscan::sensor_io::StageGuards guards;
  int arm_bodies = 0;
  int start_bodies = 0;
  const auto succeed = []() noexcept { return holoscan::LifecycleStatus::kOk; };
  const auto arm_body = [&arm_bodies]() noexcept {
    ++arm_bodies;
    return holoscan::LifecycleStatus::kOk;
  };
  const auto start_body = [&start_bodies]() noexcept {
    ++start_bodies;
    return holoscan::LifecycleStatus::kOk;
  };

  // Cold bring-up, in the order Core drives the forward stages.
  EXPECT_EQ(guards.run<holoscan::LifecycleStage::kConfigure>(succeed),
            holoscan::LifecycleStatus::kOk);
  EXPECT_EQ(guards.run<holoscan::LifecycleStage::kDiscover>(succeed),
            holoscan::LifecycleStatus::kOk);
  EXPECT_EQ(guards.run<holoscan::LifecycleStage::kAllocate>(succeed),
            holoscan::LifecycleStatus::kOk);
  EXPECT_EQ(guards.run<holoscan::LifecycleStage::kArm>(arm_body), holoscan::LifecycleStatus::kOk);
  EXPECT_EQ(guards.run<holoscan::LifecycleStage::kStart>(start_body),
            holoscan::LifecycleStatus::kOk);

  // The entirety of a restart requested at kStart: Core retains the completed kArm checkpoint and
  // replays this pair and nothing else.
  EXPECT_EQ(guards.run<holoscan::LifecycleStage::kStop>(succeed), holoscan::LifecycleStatus::kOk);
  EXPECT_EQ(guards.run<holoscan::LifecycleStage::kStart>(start_body),
            holoscan::LifecycleStatus::kOk);

  // The kStart body ran a second time, so a stream_on() living there is reissued.
  EXPECT_EQ(start_bodies, 2);
  EXPECT_FALSE(guards.guard_skipped(holoscan::LifecycleStage::kStart));
  EXPECT_TRUE(guards.started());
  // kArm was not revisited, which is the other half of the contract: whatever on_arm establishes is
  // never re-established by this suffix, so it must not be anything on_stop takes away.
  EXPECT_EQ(arm_bodies, 1);
}

// The rate half of negotiation, which is the half a driver is most likely to substitute quietly:
// VIDIOC_S_PARM is advisory, so it reports success while returning whatever interval it can
// actually run. Only the comparison is reachable without a device, but it is the part that decides
// whether a substitution is caught, and the fractional case below is the one an frames-per-second
// integer silently accepts.
// A full frame reports exactly the mapping's length, which is what these pin. The boundary is the
// ordinary case rather than an edge, so a bound that excluded equality would reject every frame a
// working camera delivers; that is the regression worth a test, and it is not one a review of the
// comparison operator would obviously catch.
TEST(V4l2MappingBoundTest, AFullFrameFitsItsMapping) {
  constexpr std::size_t kFrame = 640U * 480U * 2U;

  EXPECT_TRUE(mm::v4l2::fits_mapping(kFrame, kFrame));
}

TEST(V4l2MappingBoundTest, AShortFrameFitsItsMapping) {
  // Ordinary: a driver reports the bytes it wrote, not the capacity it was given.
  EXPECT_TRUE(mm::v4l2::fits_mapping(1U, 640U * 480U * 2U));
  EXPECT_TRUE(mm::v4l2::fits_mapping(0U, 640U * 480U * 2U));
}

TEST(V4l2MappingBoundTest, AnOversizedCountDoesNotFitItsMapping) {
  constexpr std::size_t kFrame = 640U * 480U * 2U;

  // One byte past is the case a clamp against the destination extent cannot catch, because the
  // destination is a full frame and so is this claim plus one.
  EXPECT_FALSE(mm::v4l2::fits_mapping(kFrame + 1U, kFrame));
  EXPECT_FALSE(mm::v4l2::fits_mapping(0xFFFFFFFFU, kFrame));
  // A mapping the driver never granted admits no bytes at all.
  EXPECT_FALSE(mm::v4l2::fits_mapping(1U, 0U));
}

// The operator publishes one progressive image and has no field metadata to carry anything else,
// so only V4L2_FIELD_NONE may be accepted. This is the substitution with the quietest failure of
// the three the negotiation checks: an interlaced frame has the byte count a progressive one does,
// so no size check anywhere catches it.
TEST(V4l2FieldOrderTest, ProgressiveIsAccepted) {
  EXPECT_TRUE(mm::v4l2::is_progressive(V4L2_FIELD_NONE));
}

TEST(V4l2FieldOrderTest, InterlacedAndFieldSequentialAreRejected) {
  EXPECT_FALSE(mm::v4l2::is_progressive(V4L2_FIELD_INTERLACED));
  EXPECT_FALSE(mm::v4l2::is_progressive(V4L2_FIELD_INTERLACED_TB));
  EXPECT_FALSE(mm::v4l2::is_progressive(V4L2_FIELD_INTERLACED_BT));
  EXPECT_FALSE(mm::v4l2::is_progressive(V4L2_FIELD_SEQ_TB));
  EXPECT_FALSE(mm::v4l2::is_progressive(V4L2_FIELD_SEQ_BT));
}

TEST(V4l2FieldOrderTest, AlternateAndSingleFieldAreRejected) {
  // Alternate is the layout that would be more than mislabelled: a buffer may hold one field
  // rather than the frame the descriptor claims, so half the declared rows would not be the scene
  // at the rows they occupy.
  EXPECT_FALSE(mm::v4l2::is_progressive(V4L2_FIELD_ALTERNATE));
  EXPECT_FALSE(mm::v4l2::is_progressive(V4L2_FIELD_TOP));
  EXPECT_FALSE(mm::v4l2::is_progressive(V4L2_FIELD_BOTTOM));
}

TEST(V4l2FieldOrderTest, AnUnansweredOrUnknownFieldOrderIsRejected) {
  // ANY is a request value: a driver returning it has answered the question with the question.
  EXPECT_FALSE(mm::v4l2::is_progressive(V4L2_FIELD_ANY));
  // A layout this build does not know is refused rather than assumed progressive, so a kernel that
  // adds one cannot have it published as a whole frame by default.
  EXPECT_FALSE(mm::v4l2::is_progressive(0xFFFFFFFFU));
}

TEST(V4l2IntervalTest, ExactRateIsAccepted) {
  EXPECT_TRUE((mm::v4l2::Interval{.numerator = 1U, .denominator = 30U}.equals_rate(30U)));
  // Unreduced but equal. V4L2 does not promise a reduced rational, so comparing the pair rather
  // than the printed form is what makes this one pass.
  EXPECT_TRUE((mm::v4l2::Interval{.numerator = 2U, .denominator = 60U}.equals_rate(30U)));
}

TEST(V4l2IntervalTest, SubstitutedRateIsRejected) {
  EXPECT_FALSE((mm::v4l2::Interval{.numerator = 1U, .denominator = 15U}.equals_rate(30U)));
  EXPECT_FALSE((mm::v4l2::Interval{.numerator = 1U, .denominator = 60U}.equals_rate(30U)));
}

TEST(V4l2IntervalTest, FractionalRateIsRejected) {
  // 30000/1001 is 29.97, the rate a camera asked for 30 most often lands on. Dividing the pair out
  // in integers yields 29 or 30 depending on which way it truncates, so this is precisely the case
  // that a frames-per-second comparison lets through and a cross-multiplied one catches.
  EXPECT_FALSE((mm::v4l2::Interval{.numerator = 1001U, .denominator = 30000U}.equals_rate(30U)));
}

TEST(V4l2IntervalTest, DegenerateIntervalNamesNoRate) {
  // A driver with no rate control may leave the struct zeroed and still report success. Neither
  // half of a rational may be zero, and a zeroed interval must not compare equal to a zero rate.
  EXPECT_FALSE((mm::v4l2::Interval{.numerator = 0U, .denominator = 30U}.equals_rate(30U)));
  EXPECT_FALSE((mm::v4l2::Interval{.numerator = 1U, .denominator = 0U}.equals_rate(30U)));
  EXPECT_FALSE((mm::v4l2::Interval{}.equals_rate(0U)));
}

// The two flag fields are read from one word with different masks, and a timestamp needs the right
// answer from both before it may be published as a capture time. These pin the decoding; that a
// capture time is withheld unless both agree is the operator's rule, which needs the device seam
// tracked under the review's separate testing finding.
TEST(V4l2TimestampFlagsTest, RealDeviceFlagsDecodeToMonotonicStartOfExposure) {
  // Observed from a UVC camera: V4L2_BUF_FLAG_MAPPED | TIMESTAMP_MONOTONIC | TSTAMP_SRC_SOE. Held
  // as a literal because it is evidence rather than construction -- it is what a working device
  // actually sets, so it fails if either mask is later widened over the other's bits.
  constexpr std::uint32_t kObserved = 0x00012001U;

  EXPECT_EQ(mm::v4l2::domain_of(kObserved), mm::v4l2::TimestampDomain::kMonotonic);
  EXPECT_EQ(mm::v4l2::source_of(kObserved), mm::v4l2::TimestampSource::kStartOfExposure);
}

TEST(V4l2TimestampFlagsTest, AnUnstatedSourceIsEndOfFrame) {
  // V4L2 defines end-of-frame as the zero value of the source field, so a driver that sets nothing
  // has stated end-of-frame rather than said nothing. That is the default case, and it is the one
  // whose timestamp must not be published as a start-of-integration capture time.
  EXPECT_EQ(mm::v4l2::source_of(0U), mm::v4l2::TimestampSource::kEndOfFrame);
  EXPECT_EQ(mm::v4l2::source_of(V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC),
            mm::v4l2::TimestampSource::kEndOfFrame);
}

TEST(V4l2TimestampFlagsTest, TheSourceFieldIsReadIndependentlyOfTheClockField) {
  // The point of two masks. A monotonic clock says the coordinate can be projected; it says nothing
  // about which instant it marks, and reading one field for the other is the confusion that
  // publishes an end-of-frame stamp as the start of integration.
  EXPECT_EQ(mm::v4l2::domain_of(V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC | V4L2_BUF_FLAG_TSTAMP_SRC_EOF),
            mm::v4l2::TimestampDomain::kMonotonic);
  EXPECT_EQ(mm::v4l2::source_of(V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC | V4L2_BUF_FLAG_TSTAMP_SRC_EOF),
            mm::v4l2::TimestampSource::kEndOfFrame);
  // Start of exposure against a clock this host cannot project is the converse: the right instant
  // in the wrong coordinate system, and equally unpublishable.
  EXPECT_EQ(mm::v4l2::domain_of(V4L2_BUF_FLAG_TIMESTAMP_COPY | V4L2_BUF_FLAG_TSTAMP_SRC_SOE),
            mm::v4l2::TimestampDomain::kCopy);
  EXPECT_EQ(mm::v4l2::source_of(V4L2_BUF_FLAG_TIMESTAMP_COPY | V4L2_BUF_FLAG_TSTAMP_SRC_SOE),
            mm::v4l2::TimestampSource::kStartOfExposure);
}

TEST(V4l2TimestampFlagsTest, AnUnrecognizedSourceIsNotTreatedAsAStart) {
  // The source field is three bits and V4L2 defines two values, so a kernel that adds a third must
  // not have it read as either of the ones this build knows. Unknown is the safe reading: it
  // withholds the capture time rather than publishing a stamp of unstated meaning.
  constexpr std::uint32_t kUndefinedSource = 0x00070000U;

  EXPECT_EQ(mm::v4l2::source_of(kUndefinedSource), mm::v4l2::TimestampSource::kUnknown);
}
