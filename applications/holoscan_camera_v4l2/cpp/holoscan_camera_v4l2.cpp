// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Reference graph for the Holoscan Camera V4L2 source.
//
// Two modes, because the two questions are different. `--validate` compiles the plan and opens no
// device, which is what CI can check on a runner with no camera attached. `--capture` streams from
// a real device and then reports how much of the acquisition contract actually survived the trip:
// how many frames carried a capture timestamp, how many were flagged, and what the spread in
// end-to-end latency was.
//
// The reporting is the point. A pipeline that prints "it ran" cannot tell the difference between a
// source that stamps every frame against a disciplined clock and one that publishes unstamped
// buffers at the same rate, and those two differ in every way that matters downstream.

// clang-format off: see the note on the include block in v4l2_capture_op.hpp.
#include <algorithm>
#include <charconv>
#include <chrono>  // NOLINT(build/c++11)
#include <condition_variable>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>  // NOLINT(build/c++11)
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_io.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <v4l2_capture_op/v4l2_capture_op.hpp>
// clang-format on

namespace camera = holoscan::holoscan_camera;

namespace {

struct Options {
  std::string device{"/dev/video0"};
  std::string frame_id{std::string{camera::kDefaultCameraFrameId}};
  int width{640};
  int height{480};
  int fps{30};
  std::size_t frames{60U};
  std::chrono::seconds timeout{10};
  holoscan::MemoryKind memory_kind{holoscan::MemoryKind::kHost};
  std::int32_t cuda_device{};
  bool capture{};
  bool show_help{};
};

/// @brief Names of the source operator and its port, as compilation needs them spelled.
///
/// A device placement is requested by path rather than through the operator, so these two strings
/// are the join between what `build_graph` authored and what `run` binds. Naming them once keeps a
/// rename from silently producing an unresolved-device rejection.
constexpr std::string_view kCameraOperatorPath = "camera";
constexpr std::string_view kCameraFramePort = "frame";

/// @brief Name a placement for the usage text and the report.
[[nodiscard]] std::string_view memory_kind_name(holoscan::MemoryKind kind) noexcept {
  switch (kind) {
    case holoscan::MemoryKind::kHost:
      return "host";
    case holoscan::MemoryKind::kPinnedHost:
      return "pinned";
    case holoscan::MemoryKind::kCudaDevice:
      return "device";
    default:
      return "unsupported";
  }
}

/// @brief Describe the placement for output, naming the device only when one was bound.
[[nodiscard]] std::string placement_label(const Options& options) {
  std::string label{memory_kind_name(options.memory_kind)};
  if (options.memory_kind == holoscan::MemoryKind::kCudaDevice) {
    label += ':';
    label += std::to_string(options.cuda_device);
  }
  return label;
}

/// @brief What the far end of the pipeline could still say about the samples it received.
struct Survey {
  std::mutex mutex;
  std::condition_variable ready;

  std::size_t frames{};
  std::size_t stamped{};        ///< Frames carrying a capture timestamp from a named clock.
  std::size_t degraded{};       ///< Frames the source marked as lossy, short, or gap-following.
  std::size_t invalid{};        ///< Frames the driver itself reported as corrupt.
  std::size_t unnamed_clock{};  ///< Stamped frames naming a clock the plan's catalog does not hold.
  /// @brief Stamped frames whose latency could actually be measured.
  ///
  /// Distinct from \ref stamped because carrying a capture time is necessary but not sufficient: the
  /// activation clock also has to admit current-time reads. Counting the two separately keeps
  /// "the source published no capture time" from being reported as "the pipeline has no latency".
  std::size_t timed{};
  std::int64_t min_latency_ns{std::numeric_limits<std::int64_t>::max()};
  std::int64_t max_latency_ns{std::numeric_limits<std::int64_t>::min()};

  /// @brief Core's plan-local clock table, copied once after the graph is compiled.
  ///
  /// The view owns the plan state it reads, so it stays resolvable for the report after the run.
  holoscan::ClockCatalogView clocks;
  holoscan::ClockId observed_clock{};  ///< Domain the first stamped frame arrived from.
  bool clock_domain_conflict{};        ///< Whether a later frame named a different domain.

  void wait_for_frames(std::size_t wanted, std::chrono::seconds limit) {
    std::unique_lock lock(mutex);
    static_cast<void>(ready.wait_for(lock, limit, [this, wanted]() { return frames >= wanted; }));
  }
};

/// @brief The one survey both sinks report into and `main` reads after the run.
///
/// Process-scoped rather than passed to `graph.op<>()`, because Core requires the graph to *own*
/// every capture: `normalize_graph_capture` decays an argument to a value and `GraphOwnedCapture`
/// then demands it be copy-constructible, while separately excluding pointers, `shared_ptr`, and
/// `reference_wrapper`. A `Survey` holds a mutex and is not copyable, so there is no spelling of
/// "share this object with two operators" that satisfies the concept. That is the rule working as
/// intended -- an operator's observable state should be its ports, not ambient memory it was handed
/// -- and this accessor is the deliberate exception a two-sink diagnostic app needs, matching how
/// the SDK's own multi-operator tests share observation state.
[[nodiscard]] Survey& survey() {
  static Survey instance;
  return instance;
}

/// @brief Name the clock a sample's identity refers to, against the plan's own catalog.
///
/// An empty result means the identity resolves to nothing in this plan. That is worth counting
/// rather than papering over: a `ClockId` is a dense index into one plan generation, so an
/// unresolvable one is either a stale identity or evidence that the sample was labelled somewhere
/// this catalog does not describe.
[[nodiscard]] std::string_view clock_name(const holoscan::ClockCatalogView& catalog,
                                          holoscan::ClockId id) noexcept {
  for (const holoscan::ClockPlanRecord& record : catalog.clocks) {
    if (record.id == id) {
      return record.key.name();
    }
  }
  return {};
}

/// @brief Consume frames and measure what the source's metadata discipline delivered.
class FrameSurveyOp final : public holoscan::Operator<> {
 public:
  /// @param memory_kind Placement the source was constructed with, which this port must name for
  ///                    the two contracts to agree.
  /// @note The placement is a constructor argument rather than a fixed `kHost` because a port
  /// contract is checked for equality at compile time, not coerced: a source publishing to pinned
  /// or device memory against a sink demanding host memory is a plan that Core refuses, and
  /// rightly. `holoscan::MemoryKind` is a `std::uint8_t` enum, so it satisfies the copyable-capture
  /// rule that `Survey` above cannot.
  explicit FrameSurveyOp(holoscan::MemoryKind memory_kind) noexcept
      : survey_(survey()), memory_kind_(memory_kind) {}

  void setup(holoscan::OperatorSpec& spec) override {
    // Rank 2, one byte per element. This survey reads only metadata, so it constrains the carriage
    // and not the encoding; the rank is 2 because a packed YUYV row is a byte extent rather than a
    // pixel axis. Reading no pixels is also what lets this sink accept a device placement without
    // a transfer: it never dereferences the tensor, so where the bytes live does not reach it.
    spec.input(frame_in, "frame_in")
        .queue_depth(4U)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = {.memory_kind = memory_kind_,
                               .dtype = camera::kFrameElementDtype,
                               .rank = 2U},
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{frame_in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto received = frame_in.receive();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    const holoscan::SampleMetadata& metadata = received->metadata;

    const std::lock_guard lock(survey_.mutex);
    ++survey_.frames;
    if (metadata.degraded()) {
      ++survey_.degraded;
    }
    if (!metadata.valid()) {
      ++survey_.invalid;
    }
    if (metadata.source_clock_id.valid()) {
      ++survey_.stamped;
      // The runtime clock identity is only knowable from a sample; the *name* is only knowable
      // from the compiled plan. Joining them is what lets a stored capture timestamp be attributed
      // to a clock later, which is the whole reason the identity is worth carrying.
      if (clock_name(survey_.clocks, metadata.source_clock_id).empty()) {
        ++survey_.unnamed_clock;
      }
      // One plan assigns one identity per clock domain, so a second distinct identity arriving on
      // this port means the timestamps counted below are not all in the same domain, and the
      // latency spread they produce is a comparison of unrelated coordinates.
      if (!survey_.observed_clock.valid()) {
        survey_.observed_clock = metadata.source_clock_id;
      } else if (survey_.observed_clock != metadata.source_clock_id) {
        survey_.clock_domain_conflict = true;
      }
      // Read the clock here rather than taking ExecutionContext::activation_time(). That accessor
      // returns the *logical* activation coordinate, which Core freezes from the witness that
      // caused the activation -- and for a data-triggered operator like this one the witness is the
      // arriving sample, whose reference time is the capture timestamp being subtracted. The
      // difference is therefore exactly zero for every frame, which reads like a suspiciously fast
      // pipeline rather than like a measurement that was never taken. Determinism is the reason
      // Core defines it that way and it is the right default for triggering and replay; it is just
      // not a wall clock.
      //
      // Both coordinates are still in the same domain, which is the part that had to be earned: the
      // source projected the driver's CLOCK_MONOTONIC stamp into this clock. Without that
      // projection this subtraction would be two unrelated numbers and the result would be
      // meaningless rather than merely wrong.
      if (const auto observed = context.clock().now()) {
        const std::int64_t latency = observed->timestamp_ns - metadata.capture_timestamp_ns;
        survey_.min_latency_ns = std::min(survey_.min_latency_ns, latency);
        survey_.max_latency_ns = std::max(survey_.max_latency_ns, latency);
        ++survey_.timed;
      }
    }
    survey_.ready.notify_all();
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> frame_in;

 private:
  Survey& survey_;
  holoscan::MemoryKind memory_kind_;
};

void build_graph(holoscan::Graph& graph,  // NOLINT(runtime/references)
                 const Options& options) {
  const auto camera_op = graph.op<camera::V4l2CaptureOp>(
      std::string{kCameraOperatorPath}, options.device, options.width, options.height, options.fps,
      options.frame_id, options.memory_kind);
  // The same placement on both ends. Passing it twice rather than deriving the sink's from the
  // source is deliberate: the contract is what the operators independently declare, so a run that
  // compiles is evidence the two agree, which is the property --memory-kind is here to demonstrate.
  //
  // The survey takes no other capture argument; it reaches the process-scoped survey() itself. See
  // that function's comment for why the graph cannot be handed a reference to it.
  const auto frames = graph.op<FrameSurveyOp>("frame-survey", options.memory_kind);

  graph.add_flow(camera_op->frame, frames->frame_in,
                 holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));
}

/// @brief Print the session's stable failure tokens, or say that it published none.
/// @param session Session to read diagnostics from, after it has been waited on.
/// @note The empty case is printed rather than skipped. A failed run that names nothing is a
/// different and more awkward situation than a failed run that names something, and silence here
/// reads as though the failure was never reported at all.
void report_diagnostics(const holoscan::RunSession& session) {
  bool any = false;
  for (const auto& diagnostic : session.diagnostics()) {
    std::cerr << "  " << diagnostic.token << " @vertex=" << diagnostic.source.vertex.value << '\n';
    any = true;
  }
  if (!any) {
    std::cerr << "  (the session published no diagnostics)\n";
  }
}

void report(const Options& options, const Survey& survey) {
  std::cout << "frames: received=" << survey.frames << " with-capture-time=" << survey.stamped
            << " degraded=" << survey.degraded << " corrupt=" << survey.invalid
            << " unnamed-clock=" << survey.unnamed_clock << '\n';
  if (survey.timed != 0U) {
    std::cout << "latency: [" << (survey.min_latency_ns / 1000) << ','
              << (survey.max_latency_ns / 1000) << "]us over " << survey.timed << " frames\n";
  } else if (survey.stamped != 0U) {
    // The frames arrived stamped, so the source held up its end; the graph's clock simply does not
    // admit current-time reads. Worth distinguishing from the case below, because the remedy is a
    // clock capability and not a change to the sensor.
    std::cout << "latency: unmeasurable (activation clock does not admit current-time reads)\n";
  } else {
    // Not a formatting detail. A source that publishes without a capture time leaves every
    // downstream latency, alignment, and freshness question unanswerable.
    std::cout << "latency: unmeasurable (no frame carried a capture timestamp)\n";
  }

  // Core's table, not a copy this application maintains. The generation is printed with the
  // identities because a bare ClockId means nothing without it: the numbering is dense within one
  // plan, so the same value names a different clock in the next one.
  std::cout << "clock catalog: generation=" << survey.clocks.plan_generation;
  for (const holoscan::ClockPlanRecord& record : survey.clocks.clocks) {
    std::cout << ' ' << record.key.name() << '=' << record.id.value;
    if (record.is_default) {
      std::cout << "(default)";
    }
  }
  std::cout << '\n';
  std::cout << "device: " << options.device << ' ' << options.width << 'x' << options.height << '@'
            << options.fps << " frame_id=" << options.frame_id
            << " memory=" << placement_label(options) << '\n';
}

void print_usage(std::ostream& output, std::string_view program) {
  output << "Usage: " << program << " [options]\n\n"
         << "Reference graph for the Holoscan Camera V4L2 source.\n\n"
         << "Options:\n"
         << "  --validate            Compile the graph without opening a camera (default)\n"
         << "  --capture             Stream from the device and report what survived\n"
         << "  --device PATH         V4L2 capture device path (default: /dev/video0)\n"
         << "  --width PIXELS        Capture width; must be even for YUYV (default: 640)\n"
         << "  --height PIXELS       Capture height (default: 480)\n"
         << "  --fps RATE            Requested frames per second (default: 30)\n"
         << "  --frames COUNT        Frames to collect in --capture mode (default: 60)\n"
         << "  --timeout SECONDS     Give up waiting for those frames (default: 10)\n"
         << "  --frame-id NAME       Coordinate frame carried in Header::frame_id\n"
         << "  --memory-kind KIND    Frame placement: host, pinned, or device (default: host)\n"
         << "  --cuda-device ORDINAL CUDA device for --memory-kind device (default: 0)\n"
         << "  -h, --help            Show this help text\n";
}

template <typename Integer>
[[nodiscard]] Integer parse_integer(std::string_view text, std::string_view option) {
  Integer value{};
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto [next, error] = std::from_chars(begin, end, value);
  if (error != std::errc() || next != end) {
    throw std::invalid_argument(std::string(option) + " requires an integer");
  }
  return value;
}

[[nodiscard]] holoscan::MemoryKind parse_memory_kind(std::string_view text) {
  if (text == "host") {
    return holoscan::MemoryKind::kHost;
  }
  if (text == "pinned") {
    return holoscan::MemoryKind::kPinnedHost;
  }
  if (text == "device") {
    return holoscan::MemoryKind::kCudaDevice;
  }
  throw std::invalid_argument("--memory-kind must be host, pinned, or device");
}

[[nodiscard]] std::string_view option_value(int& index, int argc,  // NOLINT(runtime/references)
                                            char** argv, std::string_view option) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string(option) + " requires a value");
  }
  return argv[++index];
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "-h" || argument == "--help") {
      options.show_help = true;
    } else if (argument == "--validate") {
      options.capture = false;
    } else if (argument == "--capture") {
      options.capture = true;
    } else if (argument == "--device") {
      options.device = option_value(index, argc, argv, argument);
    } else if (argument == "--frame-id") {
      options.frame_id = option_value(index, argc, argv, argument);
    } else if (argument == "--memory-kind") {
      options.memory_kind = parse_memory_kind(option_value(index, argc, argv, argument));
    } else if (argument == "--cuda-device") {
      options.cuda_device =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--width") {
      options.width = parse_integer<int>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--height") {
      options.height = parse_integer<int>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--fps") {
      options.fps = parse_integer<int>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--frames") {
      options.frames =
          parse_integer<std::size_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--timeout") {
      options.timeout = std::chrono::seconds{
          parse_integer<std::int64_t>(option_value(index, argc, argv, argument), argument)};
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  if (options.frames == 0U) {
    throw std::invalid_argument("--frames must be positive");
  }
  if (options.timeout <= std::chrono::seconds::zero()) {
    throw std::invalid_argument("--timeout must be positive");
  }
  if (options.cuda_device < 0) {
    // A negative ordinal is what DeviceId uses to mean "unresolved", so passing one through would
    // reach compilation as a device binding that names no device.
    throw std::invalid_argument("--cuda-device must be nonnegative");
  }
  return options;
}

[[nodiscard]] int run(const Options& options) {
  Survey& state = survey();
  holoscan::Graph graph{"holoscan-camera-v4l2"};
  build_graph(graph, options);

  // A device-resident Tensor pool needs a device, and compilation will not pick one: it rejects the
  // plan with DEVICE_UNRESOLVED rather than defaulting to ordinal zero, because an implicitly
  // chosen device is an implicitly chosen cross-device transfer later. The placement is requested
  // here rather than by the operator because which GPU to use is a deployment fact the application
  // owns, while the operator only says that the frame belongs on one.
  //
  // Bound only for kCudaDevice. Host and process-local pinned pools must have no device binding at
  // all -- supplying one there is rejected as TENSOR_DEVICE_BINDING_UNEXPECTED -- so this is not a
  // binding that is harmless to set unconditionally.
  holoscan::CompileOptions compile_options{};
  if (options.memory_kind == holoscan::MemoryKind::kCudaDevice) {
    compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = kCameraOperatorPath,
        .output_port = kCameraFramePort,
        .device = holoscan::DeviceId{options.cuda_device},
    });
  }

  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(compile_options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  // Core owns clock cataloguing, and this is the only place the two halves of a name-to-identity
  // pairing are both in scope: the graph knows the names it declared and nothing of the identities
  // compilation assigns, while an operator sees an identity and never a name. Copied before the
  // run starts so the sinks read it without synchronization, and it outlives `plan` because the
  // view retains the plan state it points into.
  state.clocks = plan.clock_catalog();

  if (!options.capture) {
    // Everything checkable without hardware has now been checked: the port types agree, the
    // schemas the ports name are registered, the trigger is satisfiable, and the tensor bounds are
    // consistent with the declared geometry. The placement is part of that -- reaching this line
    // with --memory-kind device means the source's declared placement and the sink's expectation
    // were compiled and found to agree, which needs no camera to establish.
    std::cout << "holoscan_camera_v4l2 graph validation complete (memory="
              << placement_label(options) << ")\n";
    return 0;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  state.wait_for_frames(options.frames, options.timeout);
  session.request_stop();
  session.wait();

  const std::lock_guard lock(state.mutex);
  report(options, state);

  // Asked first, and not only when the frame count came up short. A failed session is a failed
  // run whatever the sink managed to collect on the way, and the failures this ordering catches
  // are exactly the ones a frame count cannot: kFailed covers mandatory cleanup, so a source that
  // delivered every requested frame and then failed to quiesce its device reaches every check
  // below in a state where they all pass. Reporting success there would make this the wrong tool
  // for the one job it has, since teardown is where a capture source is most likely to break.
  if (session.termination() == holoscan::RunTermination::kFailed) {
    std::cerr << "holoscan_camera_v4l2: the run session failed\n";
    report_diagnostics(session);
    return 2;
  }
  if (state.clock_domain_conflict) {
    std::cerr << "holoscan_camera_v4l2: frames arrived from more than one clock domain\n";
    return 2;
  }
  if (state.frames < options.frames) {
    std::cerr << "holoscan_camera_v4l2: collected " << state.frames << " of " << options.frames
              << " frames before the timeout; termination="
              << static_cast<int>(session.termination()) << '\n';
    report_diagnostics(session);
    return 2;
  }
  // A source that streams but stamps nothing is a failure of the thing this reference exists to
  // demonstrate, so it is an error rather than a remark in the output.
  if (state.stamped != state.frames) {
    std::cerr << "holoscan_camera_v4l2: " << (state.frames - state.stamped)
              << " frames carried no capture timestamp\n";
    return 2;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.show_help) {
      print_usage(std::cout, argc > 0 ? argv[0] : "holoscan_camera_v4l2");
      return 0;
    }
    return run(options);
  } catch (const std::exception& exception) {
    std::cerr << "holoscan_camera_v4l2: " << exception.what() << '\n';
    print_usage(std::cerr, argc > 0 ? argv[0] : "holoscan_camera_v4l2");
    return 64;
  }
}
