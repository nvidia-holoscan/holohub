// SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Interface unit tests for SIPLCaptureOp.
//
// These tests exercise the C++ interface (construction, accessors, initial state) without
// requiring camera hardware or an active SIPL pipeline. No SIPL functions are called; the
// SIPL shared libraries must be loadable but the hardware does not need to be present.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/schema/binary_schema.hpp>
#include <holoscan/schema/flatbuffer_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <sipl_capture_op/sipl_capture_op.hpp>
#include <sipl_capture_op/sipl_capture_service.hpp>
#include <sipl_capture_op/tsc_to_ns.hpp>

namespace mm = holoscan::holoscan_camera;

namespace {
std::shared_ptr<mm::SIPLCaptureService> make_test_service() {
  return std::make_shared<mm::SIPLCaptureService>(
      "ov2311_raw", /*json_config=*/"", /*raw_output=*/true,
      /*capture_queue_depth=*/4U, /*nito_base_path=*/"/var/nvidia/nvcam/settings/sipl",
      /*timeout_us=*/1'000'000U, /*cuda_device=*/0);
}
}  // namespace

// ---------------------------------------------------------------------------
// Frame placement
// ---------------------------------------------------------------------------
//
// memory_kind is a constructor argument because setup() freezes it into the frame port contract.

TEST(SIPLCaptureOpTest, DefaultMemoryKindIsCudaDevice) {
  // A caller that never names a placement must keep publishing to the device, unchanged.
  const mm::SIPLCaptureOp op(make_test_service(), 0U);
  EXPECT_EQ(op.memory_kind(), holoscan::MemoryKind::kCudaDevice);
}

TEST(SIPLCaptureOpTest, AcceptsHostAndPinnedHostPlacement) {
  const mm::SIPLCaptureOp host_op(make_test_service(), 0U, holoscan::MemoryKind::kHost);
  EXPECT_EQ(host_op.memory_kind(), holoscan::MemoryKind::kHost);

  const mm::SIPLCaptureOp pinned_op(make_test_service(), 0U, holoscan::MemoryKind::kPinnedHost);
  EXPECT_EQ(pinned_op.memory_kind(), holoscan::MemoryKind::kPinnedHost);
}

TEST(SIPLCaptureOpTest, RejectsUnknownAndCudaManagedPlacement) {
  // kUnknown is a diagnostic value the schema never admits for a published tensor. kCudaManaged
  // is host-writable and would appear to work, but a captured frame that migrates on first device
  // touch trades a copy this operator can see for a fault it cannot.
  EXPECT_THROW((mm::SIPLCaptureOp{ make_test_service(), 0U, holoscan::MemoryKind::kUnknown }),
               std::invalid_argument);
  EXPECT_THROW((mm::SIPLCaptureOp{ make_test_service(), 0U, holoscan::MemoryKind::kCudaManaged }),
               std::invalid_argument);
}

TEST(SIPLCaptureOpTest, RejectsNullService) {
  EXPECT_THROW((mm::SIPLCaptureOp{ nullptr, 0U }), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Generated sensor metadata schema
// ---------------------------------------------------------------------------

TEST(SIPLFrameMetadataSchemaTest, PublishesCurrentGeneratedIdentityAndPackage) {
  constexpr holoscan::SchemaIdentity identity = holoscan::schema_identity<mm::SIPLFrameMetadataT>();
  EXPECT_EQ(identity.canonical_name, "holoscan.holoscan_camera.SIPLFrameMetadata");
  EXPECT_EQ(identity.compatibility_epoch, 1U);
  EXPECT_EQ(identity.file_identifier.value, (std::array<char, 4>{ 'H', 'C', 'S', 'M' }));

  const auto& package = holoscan::schema_package<mm::SIPLFrameMetadataT>();
  EXPECT_EQ(package.max_serialized_size(), 450000U);
  EXPECT_TRUE(holoscan::validate_schema_package(package));

  static_assert(holoscan::HasBinarySchema<mm::SIPLFrameMetadataT>);
  EXPECT_FALSE(holoscan::binary_schema<mm::SIPLFrameMetadataT>().empty());
}

// ---------------------------------------------------------------------------
// Graph compilation — kCudaDevice + bind_tensor_output_device path
// ---------------------------------------------------------------------------
//
// Ported from the applications/holoscan_camera_sipl example, which had no operational use of its
// own (it never streamed a frame) -- its only job was proving this compile path still works.
// Every downstream app now uses kHost (sipl_frame_saver, sipl_stereo_monitor), so without this,
// nothing would exercise SIPLCaptureOp's default kCudaDevice placement or the device-binding
// contract that comes with it at compile time.
//
// This runs without hardware: SIPLCaptureService's constructor does not touch NvSIPL (init is
// lazy, per its own doc comment), and holoscan::compile() validates topology and tensor contracts
// without running any lifecycle stage, so no camera needs to be present.

namespace {
class NullFrameSinkOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{ input });
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> input;
};
}  // namespace

TEST(SIPLCaptureOpCompileTest, CudaDevicePlacementCompilesWithDeviceBinding) {
  holoscan::Graph graph{ "sipl_capture_op_compile_test" };

  // sensor_data is deliberately left unconnected: compile() does not require every declared
  // output to have a consumer, and this test's only concern is the frame port's placement.
  auto service = make_test_service();

  auto camera = graph.op<mm::SIPLCaptureOp>("camera", service, 0U);
  const auto sink = graph.op<NullFrameSinkOp>("sink");
  graph.add_flow(camera->frame, sink->input, holoscan::ConnectionOptions{ .queue_depth = 1U });
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions compile_options;
  compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "camera", .output_port = "frame", .device = holoscan::DeviceId{ 0 } });
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(compile_options));
  EXPECT_TRUE(plan.ok()) << plan.json();
}

// ---------------------------------------------------------------------------
// tsc_to_ns — the arithmetic feeding holoscan::sensor_io::ClockDiscipline
// ---------------------------------------------------------------------------
//
// Pins down the exact conversion so a future edit to this arithmetic (e.g. reverting the
// overflow-safe split back to a single tsc * 1e9 / freq_hz expression) fails a test instead of
// silently corrupting every published capture_timestamp_ns.

TEST(TscToNsTest, OneSecondAtTegraFrequency) {
  // 31,250,000 Hz is cntfrq_el0's value on Tegra/IGX Orin; a tick count equal to the
  // frequency is exactly one second by definition.
  EXPECT_EQ(mm::tsc_to_ns(31'250'000ULL, 31'250'000ULL), 1'000'000'000);
}

TEST(TscToNsTest, ZeroTicksIsZeroNanoseconds) {
  EXPECT_EQ(mm::tsc_to_ns(0ULL, 31'250'000ULL), 0);
}

TEST(TscToNsTest, NonTegraFrequencyIsHandledGenerically) {
  // 24 MHz is a common ARM Generic Timer frequency on non-Tegra platforms; this function
  // makes no Tegra-specific assumption, so it must convert correctly for any frequency.
  EXPECT_EQ(mm::tsc_to_ns(7ULL, 24'000'000ULL), 291);
}

TEST(TscToNsTest, LargeTickCountDoesNotOverflowUnlikeTheNaiveFormula) {
  // At 31.25 MHz this tick count is ~5h20m of uptime -- unremarkable for a running device.
  // The naive tsc * 1'000'000'000 / freq_hz formula overflows uint64 here (600e9 * 1e9 =
  // 6e20, versus a uint64 max of ~1.8e19); the sec/rem split this function uses must not.
  constexpr std::uint64_t kTsc = 600'000'000'000ULL;
  constexpr std::uint64_t kFreq = 31'250'000ULL;
  EXPECT_EQ(mm::tsc_to_ns(kTsc, kFreq), 19'200'000'000'000);
}
