// SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// sipl_stereo_monitor — drives two SIPLCaptureOp instances against one shared SIPLCaptureService
// and validates stereo hardware sync: per-camera frame counts, cross-camera sync skew (paired by
// device_sequence, see --debug-pairing), and liveness (declares a stall, rather than hanging
// silently or waiting for a fixed --frames count that a stall would prevent from ever being
// reached, when either camera goes quiet for --stall-timeout-s).
//
// Originally written as a one-off repro harness for a stereo-sync stall report against
// holoscan-camera; kept and maintained as the standing tool for validating stereo rigs, since the
// class of bug it was built to catch (sync stalls, cross-camera skew regressions) recurs.
//
// Usage:
//   sipl_stereo_monitor --camera-config VB1940 --json-config applications/config/sipl/vb1940_stereo.json

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <sipl_capture_op/sipl_capture_op.hpp>
#include <sipl_capture_op/sipl_capture_service.hpp>

namespace mm = holoscan::holoscan_camera;

namespace {

// ---------------------------------------------------------------------------
// Shared per-camera tracking, read by the stall-watchdog loop in main().
// ---------------------------------------------------------------------------

struct CameraTrack {
  std::atomic<std::uint64_t> frame_count{0};
  std::atomic<std::int64_t> last_capture_time_ns{0};
  // Set on the first record(); before that, idle_for() reflects time-since-construction rather
  // than time-since-last-frame, which is meaningless on a rig that hasn't started streaming yet.
  std::atomic<bool> received_first_frame{false};
  std::mutex mutex;
  std::chrono::steady_clock::time_point last_frame_wall_time{std::chrono::steady_clock::now()};

  void record(std::int64_t capture_time_ns) {
    frame_count.fetch_add(1, std::memory_order_relaxed);
    last_capture_time_ns.store(capture_time_ns, std::memory_order_relaxed);
    received_first_frame.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex);
    last_frame_wall_time = std::chrono::steady_clock::now();
  }

  [[nodiscard]] std::chrono::steady_clock::duration idle_for() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex));
    return std::chrono::steady_clock::now() - last_frame_wall_time;
  }
};

struct StereoResults {
  std::array<CameraTrack, 2> cameras;

  // Cross-camera skew, paired by device_sequence rather than by "whichever frame happened to be
  // last when the watchdog polled" -- the latter compares different frame indices on the two
  // cameras and produces a number that looks stable but is not a real sync measurement. Pairing
  // relies on sync_sensors:true giving both cameras a shared frame sequence counter.
  std::mutex pairing_mutex;
  std::map<std::uint64_t, std::int64_t> pending_seq[2];
  std::atomic<std::int64_t> last_paired_skew_ns{0};
  std::atomic<bool> skew_valid{false};
  // When set, record_pair logs every match/insert/eviction to stderr -- used to root-cause
  // whether an anomalous skew reading is a real sync hiccup or a seq-pairing mismatch in this
  // harness (e.g. an evicted/stale entry getting matched against the wrong frame).
  std::atomic<bool> debug_pairing{false};

  void record_pair(std::uint32_t camera_index, std::uint64_t seq, std::int64_t capture_ns) {
    static constexpr std::size_t kMaxPending = 64;
    const std::uint32_t other = camera_index == 0U ? 1U : 0U;
    const bool debug = debug_pairing.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(pairing_mutex);
    auto it = pending_seq[other].find(seq);
    if (it != pending_seq[other].end()) {
      const std::int64_t skew =
          camera_index == 0U ? (it->second - capture_ns) : (capture_ns - it->second);
      last_paired_skew_ns.store(skew, std::memory_order_relaxed);
      skew_valid.store(true, std::memory_order_relaxed);
      if (debug) {
        std::cerr << "[pair] seq=" << seq << " cam" << camera_index << "_ts=" << capture_ns
                   << " cam" << other << "_ts=" << it->second << " skew_ns=" << skew
                   << " pending0=" << pending_seq[0].size() << " pending1=" << pending_seq[1].size()
                   << "\n";
      }
      pending_seq[other].erase(it);
      return;
    }
    auto& own = pending_seq[camera_index];
    own[seq] = capture_ns;
    if (debug) {
      std::cerr << "[insert] cam" << camera_index << " seq=" << seq << " ts=" << capture_ns
                 << " pending0=" << pending_seq[0].size() << " pending1=" << pending_seq[1].size()
                 << "\n";
    }
    if (own.size() > kMaxPending) {
      if (debug) {
        std::cerr << "[evict] cam" << camera_index << " seq=" << own.begin()->first
                   << " ts=" << own.begin()->second << " (map exceeded " << kMaxPending << ")\n";
      }
      own.erase(own.begin());
    }
  }
};

// ---------------------------------------------------------------------------
// One sink per camera: counts frames, pairs capture_timestamp_ns by device_sequence for skew.
// ---------------------------------------------------------------------------

struct FrameSinkParams {
  std::shared_ptr<StereoResults> results;
  std::uint32_t camera_index{0};
};

class FrameSinkOp final : public holoscan::Operator<FrameSinkParams> {
 public:
  holoscan::Input<holoscan::schema::ImageT> input;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(4U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnEach{input});
    return c;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const auto& p = params();
    std::int64_t capture_ns = 0;
    std::uint64_t seq = 0;
    if (sample->data.header) {
      capture_ns = sample->data.header->capture_timestamp_ns;
      seq = sample->data.header->device_sequence;
    }
    p.results->cameras.at(p.camera_index).record(capture_ns);
    p.results->record_pair(p.camera_index, seq, capture_ns);
    return {};
  }
};

// ---------------------------------------------------------------------------
// Discard sink for sensor_data: SIPLCaptureOp emits it every compute(); leaving it unconnected
// produces an "unconnected output port" warning per frame per camera.
// ---------------------------------------------------------------------------

class MetadataSinkOp final : public holoscan::Operator<> {
 public:
  holoscan::Input<mm::SIPLFrameMetadataT> input;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(4U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnEach{input});
    return c;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    return {};
  }
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Options {
  std::string camera_config{"VB1940"};
  std::string json_config;
  std::int32_t cuda_device{0};
  std::uint32_t max_frames{500};
  double stall_timeout_s{5.0};
  double startup_timeout_s{15.0};
  double overall_timeout_s{120.0};
  bool isp_output{false};
  bool debug_pairing{false};
  bool show_help{false};
};

void print_usage(std::ostream& out, std::string_view prog) {
  out << "Usage: " << prog << " [options]\n\n"
      << "Drive two SIPLCaptureOp instances against one shared SIPLCaptureService and\n"
      << "report a stall (rather than hanging) if either camera goes quiet.\n\n"
      << "Options:\n"
      << "  --camera-config NAME     SIPL camera configuration name (default: VB1940)\n"
      << "  --json-config PATH      Vendor JSON platform config (required for this rig)\n"
      << "  --cuda-device N         CUDA device ordinal (default: 0)\n"
      << "  --frames N              Frames per camera before stopping cleanly (default: 500)\n"
      << "  --stall-timeout-s SEC   Idle seconds on either camera (after its first frame) before\n"
      << "                          declaring a stall (default: 5)\n"
      << "  --startup-timeout-s SEC Seconds to wait for each camera's first frame before\n"
      << "                          declaring a stall (default: 15)\n"
      << "  --overall-timeout-s SEC Hard cap on total run time (default: 120)\n"
      << "  --isp                   Request NV12/ISP output instead of RAW10 (raw_output=false)\n"
      << "  --debug-pairing         Log every skew pairing match/insert/eviction to stderr\n"
      << "  -h, --help              Show this help text\n";
}

std::string_view next_arg(int& i, int argc, char** argv, std::string_view opt) {
  if (i + 1 >= argc) {
    throw std::invalid_argument(std::string(opt) + " requires a value");
  }
  return argv[++i];
}

Options parse_options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg{argv[i]};
    if (arg == "-h" || arg == "--help") {
      o.show_help = true;
    } else if (arg == "--camera-config") {
      o.camera_config = next_arg(i, argc, argv, arg);
    } else if (arg == "--json-config") {
      o.json_config = next_arg(i, argc, argv, arg);
    } else if (arg == "--cuda-device") {
      o.cuda_device = std::atoi(std::string(next_arg(i, argc, argv, arg)).c_str());
    } else if (arg == "--frames") {
      o.max_frames =
          static_cast<std::uint32_t>(std::atoi(std::string(next_arg(i, argc, argv, arg)).c_str()));
    } else if (arg == "--stall-timeout-s") {
      o.stall_timeout_s = std::atof(std::string(next_arg(i, argc, argv, arg)).c_str());
    } else if (arg == "--startup-timeout-s") {
      o.startup_timeout_s = std::atof(std::string(next_arg(i, argc, argv, arg)).c_str());
    } else if (arg == "--overall-timeout-s") {
      o.overall_timeout_s = std::atof(std::string(next_arg(i, argc, argv, arg)).c_str());
    } else if (arg == "--isp") {
      o.isp_output = true;
    } else if (arg == "--debug-pairing") {
      o.debug_pairing = true;
    } else {
      throw std::invalid_argument("unknown option: " + std::string(arg));
    }
  }
  return o;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int run(const Options& opts) {
  auto results = std::make_shared<StereoResults>();
  results->debug_pairing.store(opts.debug_pairing, std::memory_order_relaxed);

  // One shared service for both sensors -- this is the architecture the stereo-sync work in
  // SIPLCaptureService (start_buffers()'s sensor-group handling) assumes.
  auto service = std::make_shared<mm::SIPLCaptureService>(
      opts.camera_config,
      opts.json_config,
      /*raw_output=*/!opts.isp_output,
      /*capture_queue_depth=*/4U,
      /*nito_base_path=*/"/var/nvidia/nvcam/settings/sipl",
      /*timeout_us=*/1'000'000U,
      opts.cuda_device);

  holoscan::Graph graph{"sipl_stereo_monitor"};

  // OperatorHandle has no default constructor, so this is unrolled rather than looped into a
  // std::array of handles.
  //
  // Both cameras publish to kHost: FrameSinkOp below reads only sample->data.header (timestamp,
  // sequence) and never touches the pixel tensor, so there is no consumer here to justify the
  // device copy and device binding that kCudaDevice (the operator's default) would require.
  auto camera0 = graph.op<mm::SIPLCaptureOp>(
      "camera0", service, /*camera_index=*/0U, holoscan::MemoryKind::kHost);
  auto sink0 = graph.op<FrameSinkOp>("sink0");
  sink0.set_params(FrameSinkParams{.results = results, .camera_index = 0U});
  graph.add_flow(camera0->frame, sink0->input);
  auto metadata_sink0 = graph.op<MetadataSinkOp>("metadata_sink0");
  graph.add_flow(camera0->sensor_data, metadata_sink0->input);

  auto camera1 = graph.op<mm::SIPLCaptureOp>(
      "camera1", service, /*camera_index=*/1U, holoscan::MemoryKind::kHost);
  auto sink1 = graph.op<FrameSinkOp>("sink1");
  sink1.set_params(FrameSinkParams{.results = results, .camera_index = 1U});
  graph.add_flow(camera1->frame, sink1->input);
  auto metadata_sink1 = graph.op<MetadataSinkOp>("metadata_sink1");
  graph.add_flow(camera1->sensor_data, metadata_sink1->input);

  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // No device binding: both cameras publish to kHost above, and a binding is refused (as
  // TENSOR_DEVICE_BINDING_UNEXPECTED) for a host-accessible placement.
  const auto plan = holoscan::compile(graph, holoscan::CompileOptions{});
  if (!plan.ok()) {
    std::cerr << "Graph compilation failed:\n" << plan.json() << "\n";
    return 1;
  }

  std::cout << "Streaming from '" << opts.camera_config << "' (json_config=" << opts.json_config
            << "), target " << opts.max_frames << " frames/camera, stall timeout "
            << opts.stall_timeout_s << "s\n";

  auto session = holoscan::run_async(plan);

  const auto start = std::chrono::steady_clock::now();
  const auto stall_timeout = std::chrono::duration<double>(opts.stall_timeout_s);
  const auto startup_timeout = std::chrono::duration<double>(opts.startup_timeout_s);
  const auto overall_timeout = std::chrono::duration<double>(opts.overall_timeout_s);

  std::string stop_reason;
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    const auto& c0 = results->cameras[0];
    const auto& c1 = results->cameras[1];
    const std::uint64_t n0 = c0.frame_count.load();
    const std::uint64_t n1 = c1.frame_count.load();

    std::cout << "\rcamera0=" << n0 << " camera1=" << n1 << " skew_ns=";
    if (results->skew_valid.load(std::memory_order_relaxed)) {
      std::cout << results->last_paired_skew_ns.load(std::memory_order_relaxed);
    } else {
      std::cout << "n/a";
    }
    std::cout << "        " << std::flush;

    if (n0 >= opts.max_frames && n1 >= opts.max_frames) {
      stop_reason = "reached --frames target on both cameras";
      break;
    }
    const bool c0_started = c0.received_first_frame.load(std::memory_order_relaxed);
    const bool c1_started = c1.received_first_frame.load(std::memory_order_relaxed);
    if (!c0_started || !c1_started) {
      // Before a camera's first frame, idle_for() measures time since CameraTrack was
      // constructed (before streaming even starts), not time since a frame stopped arriving --
      // that isn't the same failure mode as a mid-stream stall, so it gets its own timeout.
      if (std::chrono::steady_clock::now() - start > startup_timeout) {
        stop_reason = "STALL: no first frame within " + std::to_string(opts.startup_timeout_s) +
                      "s on camera" + std::string(!c0_started ? "0" : "1");
        break;
      }
    } else if (c0.idle_for() > stall_timeout || c1.idle_for() > stall_timeout) {
      stop_reason = "STALL: no frame for over " + std::to_string(opts.stall_timeout_s) +
                    "s on camera" + std::string(c0.idle_for() > stall_timeout ? "0" : "1");
      break;
    }
    if (std::chrono::steady_clock::now() - start > overall_timeout) {
      stop_reason = "overall timeout reached";
      break;
    }
  }

  session.request_stop();
  session.wait();

  std::cout << "\n" << stop_reason << "\n"
            << "Final: camera0=" << results->cameras[0].frame_count.load()
            << " camera1=" << results->cameras[1].frame_count.load() << "\n";

  return stop_reason.rfind("STALL", 0) == 0 ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opts = parse_options(argc, argv);
    if (opts.show_help) {
      print_usage(std::cout, argc > 0 ? argv[0] : "sipl_stereo_monitor");
      return 0;
    }
    if (opts.json_config.empty()) {
      std::cerr << "sipl_stereo_monitor: --json-config is required for a two-sensor rig\n";
      print_usage(std::cerr, argc > 0 ? argv[0] : "sipl_stereo_monitor");
      return 64;
    }
    return run(opts);
  } catch (const std::exception& e) {
    std::cerr << "sipl_stereo_monitor: " << e.what() << "\n";
    return 1;
  }
}
