// SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SIPL frame saver — capture N frames from a SIPL camera and write them to disk.
//
// Each frame is saved as three files:
//   frame_<seq>.raw          — raw tensor bytes (packed 10-bit RAW10 or packed NV12)
//   frame_<seq>.txt          — human-readable metadata from the ImageT descriptor
//   frame_<seq>_sensor.txt   — AE/AWB control-loop metadata from the 'sensor_data' port
//
// SIPLCaptureOp is constructed with MemoryKind::kHost here: this app's only consumer is the
// filesystem, so publishing straight to host memory avoids the device round trip a GPU-resident
// default would force every consumer -- this one included -- to buy back with its own copy.
//
// Usage:
//   sipl_frame_saver [OPTIONS]
//
// Options:
//   --camera-config NAME   SIPL camera configuration name (default: ov2311_raw)
//   --json-config PATH     Path to a vendor JSON platform config file (optional)
//   --camera-index N       Zero-based camera index within the rig (default: 0)
//   --cuda-device N        CUDA device ordinal for NvSci buffer import (default: 0)
//   --frames N             Number of frames to save then stop (default: 5)
//   --output-dir DIR       Output directory; created if absent (default: /tmp/sipl_frames)
//   --isp                  Request NV12/ISP output instead of RAW10 (raw_output=false)
//   --timeout-s SEC        Seconds to wait for --frames to complete (default: 60)
//   -h, --help             Show this help text
//
// The raw file contains exactly the bytes produced by SIPLCaptureOp: packed 10-bit
// Bayer for RAW10 sensors (pitch × height bytes), or planar Y + interleaved UV for
// NV12 sensors.  The sidecar .txt carries enough descriptor fields for a consumer to
// decode the bytes without any out-of-band knowledge -- RAW10's Bayer phase included,
// as bayer_phase, even though SIPLCaptureOp reports ImageEncoding_CUSTOM for that
// format and the phase is therefore not in the encoding string itself.
//
// Quick decode with Python:
//   import numpy as np
//   raw = np.fromfile("frame_00000001.raw", dtype=np.uint8)
//   # For RAW10: width=W height=H pitch=P
//   # raw.reshape(H, P) gives packed 10-bit rows

#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <sipl_capture_op/sipl_capture_op.hpp>
#include <sipl_capture_op/sipl_capture_service.hpp>

namespace mm = holoscan::holoscan_camera;

namespace {

// ---------------------------------------------------------------------------
// Shared completion signal
// ---------------------------------------------------------------------------

struct SaveStatus {
  std::atomic<std::uint32_t> frames_saved{0};
  std::atomic<std::uint32_t> sensor_frames_saved{0};
  std::atomic<bool> done{false};
  std::mutex mutex;
  std::condition_variable cv;

  bool wait(std::chrono::duration<double> timeout) {
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_for(lock, timeout, [this] { return done.load(); });
  }

  // Both FrameSaverOp and SensorDataSaverOp call this after updating their own counter. 'frame'
  // and 'sensor_data' are emitted together per-frame by SIPLCaptureOp but travel to these two
  // operators over independently-scheduled queues, so one can lag the other; signaling done only
  // once both have reached max_frames avoids stopping the session while the slower one still has
  // a trailing sidecar to write for the last frame(s).
  void maybe_finish(std::uint32_t max_frames) {
    if (max_frames == 0) return;
    if (frames_saved.load() < max_frames || sensor_frames_saved.load() < max_frames) return;
    std::lock_guard<std::mutex> lock(mutex);
    done.store(true);
    cv.notify_all();
  }
};

std::string pad8(std::uint64_t n) {
  auto s = std::to_string(n);
  if (s.size() < 8) s.insert(0, 8 - s.size(), '0');
  return s;
}

// SIPLCaptureOp reports RAW10 frames as ImageEncoding_CUSTOM: NvSci's X2Rc10Rb10Ra10 packing (three
// 10-bit samples per 4-byte dword) does not satisfy a BAYER_* encoding's contract that one Tensor
// element is one sample, so the descriptor itself no longer names the phase. This is the only place
// that still has it -- CameraInfo::bayer_format, resolved once from the live SIPL session -- so the
// sidecar carries it separately for decode_sipl_frame.py to read back.
std::string_view bayer_phase_name(mm::SIPLCaptureService::BayerFormat format) {
  switch (format) {
    case mm::SIPLCaptureService::BayerFormat::kRggb: return "RGGB";
    case mm::SIPLCaptureService::BayerFormat::kBggr: return "BGGR";
    case mm::SIPLCaptureService::BayerFormat::kGbrg: return "GBRG";
    case mm::SIPLCaptureService::BayerFormat::kGrbg: return "GRBG";
  }
  return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// FrameSaverOp
// ---------------------------------------------------------------------------

struct FrameSaverParams {
  std::string output_dir{"/tmp/sipl_frames"};
  std::uint32_t max_frames{5};
  std::shared_ptr<SaveStatus> status;
  // RGGB/BGGR/GBRG/GRBG, resolved once in run() from CameraInfo::bayer_format. Only meaningful --
  // and only written to the sidecar -- for a CUSTOM-encoded (RAW10) frame.
  std::string bayer_phase;
};

class FrameSaverOp final : public holoscan::Operator<FrameSaverParams> {
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

  void start() override {
    const auto& p = params();
    std::filesystem::create_directories(p.output_dir);
    HOLOSCAN_LOG_INFO("FrameSaverOp: saving up to {} frames to {}",
                      p.max_frames, p.output_dir);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& /*context*/) override {
    const auto& p = params();

    auto sample = input.receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());

    // Keep consuming after max_frames to avoid back-pressure on the camera pipeline.
    const std::uint32_t saved = p.status ? p.status->frames_saved.load() : frames_saved_;
    if (p.max_frames > 0 && saved >= p.max_frames) return {};

    const holoscan::schema::ImageT& desc = sample->data;
    const holoscan::Tensor* tensor = desc.data.get();
    if (!tensor) {
      HOLOSCAN_LOG_WARN("FrameSaverOp: received frame with null tensor; skipping");
      return {};
    }

    // SIPLCaptureOp publishes to host memory here (see the file header), so the bytes are already
    // readable without a copy.
    const std::size_t nbytes = tensor->nbytes();
    const auto* host_data = static_cast<const std::uint8_t*>(tensor->data());

    const std::uint64_t seq =
        (desc.header ? desc.header->device_sequence : frames_saved_);
    const std::string base =
        p.output_dir + "/frame_" + pad8(seq);

    // Write raw bytes.
    {
      std::ofstream f(base + ".raw", std::ios::binary);
      if (!f) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure,
                            "FrameSaverOp: cannot open " + base + ".raw"});
      }
      f.write(reinterpret_cast<const char*>(host_data),
              static_cast<std::streamsize>(nbytes));
    }

    // Write text metadata sidecar.
    {
      std::ofstream f(base + ".txt");
      if (f) {
        f << "encoding=" << holoscan::schema::EnumNameImageEncoding(desc.encoding) << "\n"
          << "width=" << desc.width << "\n"
          << "height=" << desc.height << "\n"
          << "significant_bits=" << static_cast<int>(desc.significant_bits) << "\n"
          << "roi_offset_y=" << desc.roi_offset_y << "\n"
          << "bytes=" << nbytes << "\n";
        if (!desc.plane_layouts.empty()) {
          f << "plane_layouts=";
          for (std::size_t i = 0; i < desc.plane_layouts.size(); ++i) {
            if (i) f << ",";
            f << desc.plane_layouts[i].row_stride_bytes();
          }
          f << "\n";
        }
        if (desc.encoding == holoscan::schema::ImageEncoding_CUSTOM && !p.bayer_phase.empty()) {
          f << "bayer_phase=" << p.bayer_phase << "\n";
        }
        f << "exposure_time_ns=" << desc.exposure_time_ns << "\n"
          << "gain=" << desc.gain << "\n";
        if (desc.header) {
          f << "device_sequence=" << desc.header->device_sequence << "\n"
            << "capture_timestamp_ns=" << desc.header->capture_timestamp_ns << "\n"
            << "frame_id=" << desc.header->frame_id << "\n";
        }
      }
    }

    ++frames_saved_;
    HOLOSCAN_LOG_INFO("FrameSaverOp: saved {}.raw ({} bytes, {}x{} {})",
                      base, nbytes, desc.width, desc.height,
                      holoscan::schema::EnumNameImageEncoding(desc.encoding));

    if (p.status) {
      p.status->frames_saved.store(frames_saved_);
      p.status->maybe_finish(p.max_frames);
    }

    return {};
  }

 private:
  std::uint32_t frames_saved_{0};
};

// ---------------------------------------------------------------------------
// SensorDataSaverOp
// ---------------------------------------------------------------------------
//
// Saves the AE/AWB control-loop metadata SIPLCaptureOp emits on 'sensor_data' in lock-step
// with 'frame', as frame_<seq>_sensor.txt. The sequence number comes from SampleMetadata::frame_id
// (the same EmitOptions::frame_id SIPLCaptureOp stamped on the correlated 'frame' sample), so
// these sidecars line up with the .raw/.txt pair FrameSaverOp writes for the same frame.

struct SensorDataSaverParams {
  std::string output_dir{"/tmp/sipl_frames"};
  std::uint32_t max_frames{5};
  std::shared_ptr<SaveStatus> status;
};

class SensorDataSaverOp final : public holoscan::Operator<SensorDataSaverParams> {
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
    const auto& p = params();

    auto sample = input.receive();
    if (!sample) return holoscan::make_unexpected(std::move(sample).error());

    // Keep consuming after max_frames to avoid back-pressure on the camera pipeline, matching
    // FrameSaverOp's policy for the same reason.
    if (p.max_frames > 0 && saved_ >= p.max_frames) return {};

    const std::uint64_t seq = sample->metadata.frame_id;
    const mm::SIPLFrameMetadataT& m = sample->data;
    const std::string path = p.output_dir + "/frame_" + pad8(seq) + "_sensor.txt";

    std::ofstream f(path);
    if (f) {
      f << "num_exposures=" << static_cast<int>(m.num_exposures) << "\n"
        << "exp_time_valid=" << m.exp_time_valid << "\n";
      for (std::size_t i = 0; i < m.exposure_time.size(); ++i) {
        f << "exposure_time[" << i << "]=" << m.exposure_time[i] << "\n";
      }
      f << "gain_valid=" << m.gain_valid << "\n";
      for (std::size_t i = 0; i < m.sensor_gain.size(); ++i) {
        f << "sensor_gain[" << i << "]=" << m.sensor_gain[i] << "\n";
      }
      f << "wb_valid=" << m.wb_valid << "\n";
      for (std::size_t i = 0; i < m.wb_gain.size(); ++i) {
        const auto& wb = m.wb_gain[i];
        f << "wb_gain[" << i << "]=" << wb.r() << "," << wb.gr() << "," << wb.gb() << ","
          << wb.b() << "\n";
      }
      f << "temp_valid=" << m.temp_valid << "\n";
      for (std::size_t i = 0; i < m.sensor_temp_celsius.size(); ++i) {
        f << "sensor_temp_celsius[" << i << "]=" << m.sensor_temp_celsius[i] << "\n";
      }
      f << "error_flag=" << static_cast<int>(m.error_flag) << "\n"
        << "hsb_valid=" << m.hsb_valid << "\n";
      if (m.hsb_valid) {
        f << "hsb_frame_number=" << m.hsb_frame_number << "\n"
          << "hsb_crc=" << m.hsb_crc << "\n";
      }
    }

    ++saved_;
    if (p.status) {
      p.status->sensor_frames_saved.store(saved_);
      p.status->maybe_finish(p.max_frames);
    }
    return {};
  }

 private:
  std::uint32_t saved_{0};
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Options {
  std::string camera_config{"ov2311_raw"};
  std::string json_config{};
  std::uint32_t camera_index{0};
  std::int32_t cuda_device{0};
  std::uint32_t frames{5};
  std::string output_dir{"/tmp/sipl_frames"};
  bool isp_output{false};
  double timeout_s{60.0};
  bool show_help{false};
};

void print_usage(std::ostream& out, std::string_view prog) {
  out << "Usage: " << prog << " [options]\n\n"
      << "Capture frames from a SIPL camera and save them to disk.\n\n"
      << "Options:\n"
      << "  --camera-config NAME   SIPL camera configuration name (default: ov2311_raw)\n"
      << "  --json-config PATH     Path to vendor JSON platform config file (optional)\n"
      << "  --camera-index N       Zero-based camera index within the rig (default: 0)\n"
      << "  --cuda-device N        CUDA device ordinal (default: 0)\n"
      << "  --frames N             Number of frames to save then stop (default: 5)\n"
      << "  --output-dir DIR       Output directory (default: /tmp/sipl_frames)\n"
      << "  --isp                  Request NV12/ISP output instead of RAW10 (raw_output=false)\n"
      << "  --timeout-s SEC        Seconds to wait for --frames to complete before giving up\n"
      << "                         (default: 60; raise this for large --frames captures)\n"
      << "  -h, --help             Show this help text\n\n"
      << "Output files per frame:\n"
      << "  frame_<seq>.raw          — raw device bytes (packed RAW10 or NV12)\n"
      << "  frame_<seq>.txt          — descriptor metadata (encoding, dimensions, timestamps)\n"
      << "  frame_<seq>_sensor.txt   — AE/AWB control-loop metadata (exposure, gain, WB, temp)\n";
}

template <typename T>
T parse_int(std::string_view text, std::string_view opt) {
  T v{};
  const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), v);
  if (ec != std::errc() || end != text.data() + text.size())
    throw std::invalid_argument(std::string(opt) + " requires an integer");
  return v;
}

std::string_view next_arg(int& i, int argc, char** argv, std::string_view opt) {
  if (i + 1 >= argc)
    throw std::invalid_argument(std::string(opt) + " requires a value");
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
    } else if (arg == "--camera-index") {
      o.camera_index = parse_int<std::uint32_t>(next_arg(i, argc, argv, arg), arg);
    } else if (arg == "--cuda-device") {
      o.cuda_device = parse_int<std::int32_t>(next_arg(i, argc, argv, arg), arg);
    } else if (arg == "--frames") {
      o.frames = parse_int<std::uint32_t>(next_arg(i, argc, argv, arg), arg);
    } else if (arg == "--output-dir") {
      o.output_dir = next_arg(i, argc, argv, arg);
    } else if (arg == "--isp") {
      o.isp_output = true;
    } else if (arg == "--timeout-s") {
      o.timeout_s = parse_int<double>(next_arg(i, argc, argv, arg), arg);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(arg));
    }
  }
  if (o.cuda_device < 0)
    throw std::invalid_argument("--cuda-device must be non-negative");
  return o;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int run(const Options& opts) {
  auto status = std::make_shared<SaveStatus>();

  auto service = std::make_shared<mm::SIPLCaptureService>(
      opts.camera_config,
      opts.json_config,
      /*raw_output=*/!opts.isp_output,
      /*capture_queue_depth=*/4U,
      /*nito_base_path=*/"/var/nvidia/nvcam/settings/sipl",
      /*timeout_us=*/1'000'000U,
      opts.cuda_device);

  // Resolve this once before graph execution so FrameSaverOp does not need the service only to
  // write one sidecar field.
  const std::string bayer_phase = std::string(
      bayer_phase_name(service->get_camera_info().at(opts.camera_index).bayer_format));

  holoscan::Graph graph{"sipl_frame_saver"};

  auto camera = graph.op<mm::SIPLCaptureOp>(
      "camera", service, opts.camera_index, holoscan::MemoryKind::kHost);

  auto saver = graph.op<FrameSaverOp>("saver");
  saver.set_params(FrameSaverParams{
      .output_dir  = opts.output_dir,
      .max_frames  = opts.frames,
      .status      = status,
      .bayer_phase = bayer_phase});

  auto sensor_data_saver = graph.op<SensorDataSaverOp>("sensor_data_saver");
  sensor_data_saver.set_params(SensorDataSaverParams{
      .output_dir = opts.output_dir,
      .max_frames = opts.frames,
      .status     = status});

  graph.add_flow(camera->frame, saver->input,
                 holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(camera->sensor_data, sensor_data_saver->input,
                 holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // No device binding here: the camera now publishes to kHost, and a binding is refused (as
  // TENSOR_DEVICE_BINDING_UNEXPECTED) for a host-accessible placement.
  const auto plan = holoscan::compile(graph, holoscan::CompileOptions{});
  if (!plan.ok()) {
    std::cerr << "Graph compilation failed:\n" << plan.json() << "\n";
    return 1;
  }

  std::cout << "Capturing " << opts.frames << " frames"
            << " (camera=" << opts.camera_config
            << " index=" << opts.camera_index << ")"
            << " → " << opts.output_dir << "/\n";

  auto session = holoscan::run_async(plan);
  const bool done = status->wait(std::chrono::duration<double>(opts.timeout_s));
  if (!done) {
    std::cerr << "Timed out after " << opts.timeout_s << "s — only "
              << status->frames_saved.load() << "/" << opts.frames << " frames saved, "
              << status->sensor_frames_saved.load() << "/" << opts.frames
              << " sensor sidecars saved. Try a larger --timeout-s for bigger --frames captures.\n";
  }
  session.request_stop();
  session.wait();

  std::cout << "Saved " << status->frames_saved.load()
            << " frames to " << opts.output_dir << "/\n";
  return done ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opts = parse_options(argc, argv);
    if (opts.show_help) {
      print_usage(std::cout, argc > 0 ? argv[0] : "sipl_frame_saver");
      return 0;
    }
    return run(opts);
  } catch (const std::exception& e) {
    std::cerr << "sipl_frame_saver: " << e.what() << "\n";
    print_usage(std::cerr, argc > 0 ? argv[0] : "sipl_frame_saver");
    return 64;
  }
}
