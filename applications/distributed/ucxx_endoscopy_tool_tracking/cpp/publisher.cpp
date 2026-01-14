/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "publisher.h"

#include <algorithm>
#include <atomic>
#include <vector>

#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <holoviz/holoviz.hpp>
#include <lstm_tensor_rt_inference.hpp>
#include <tool_tracking_postprocessor.hpp>

#include "gxf/std/timestamp.hpp"

#include <operators/ucxx_send_receive/ucxx_endpoint.hpp>
#include <operators/ucxx_send_receive/receiver_op/ucxx_receiver_op.hpp>
#include <operators/ucxx_send_receive/sender_op/ucxx_sender_op.hpp>

namespace holoscan::apps {

namespace {

// Shared state for the most recent frame counter value.
class FrameCounterState : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS(FrameCounterState)

  FrameCounterState() = default;

  void set(uint64_t v) {
    // Use an atomic to ensure counter starts from 1. Discard previous UCXX in-flight
    // values when UCXX endpoint reconnects.
    if (waiting_for_reset_.load(std::memory_order_acquire)) {
      if (v != 1) { return; }
      waiting_for_reset_.store(false, std::memory_order_release);
    }
    value_.store(v, std::memory_order_release);
    has_value_.store(true, std::memory_order_release);
  }

  bool has_value() const { return has_value_.load(std::memory_order_acquire); }
  uint64_t get() const { return value_.load(std::memory_order_acquire); }

  void clear() {
    has_value_.store(false, std::memory_order_release);
    waiting_for_reset_.store(true, std::memory_order_release);
  }

 private:
  std::atomic<uint64_t> value_{0};
  std::atomic<bool> has_value_{false};
  std::atomic<bool> waiting_for_reset_{true};
};

// Consumes frame-counter overlay updates from UCXX and updates FrameCounterState.
class FrameCounterUpdateOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FrameCounterUpdateOp)

  FrameCounterUpdateOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("in");
    spec.param(state_, "state", "State", "Shared overlay state");
  }

  void compute(holoscan::InputContext& input, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    auto in_entity = input.receive<holoscan::gxf::Entity>("in").value();
    auto maybe_in_tensor =
        static_cast<nvidia::gxf::Entity&>(in_entity).get<nvidia::gxf::Tensor>("");
    if (!maybe_in_tensor) {
      HOLOSCAN_LOG_WARN("FrameCounterUpdateOp: missing input tensor \"\" (counter)");
      return;
    }

    auto& in_tensor = *maybe_in_tensor.value();
    if (in_tensor.element_type() != nvidia::gxf::PrimitiveType::kUnsigned64 ||
        in_tensor.rank() != 1 || in_tensor.shape().dimension(0) != 1) {
      HOLOSCAN_LOG_WARN("FrameCounterUpdateOp: unexpected tensor shape/type for counter");
      return;
    }
    const uint64_t v = *reinterpret_cast<const uint64_t*>(in_tensor.pointer());
    state_.get()->set(v);
  }

 private:
  holoscan::Parameter<std::shared_ptr<FrameCounterState>> state_;
};

// Attaches the most recent counter value to each incoming frame entity using a non-tensor GXF
// component (Timestamp) named "counter_value".
class FrameCounterAttachValueOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FrameCounterAttachValueOp)

  FrameCounterAttachValueOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("in");
    spec.output<holoscan::gxf::Entity>("out");
    spec.param(state_, "state", "State", "Shared counter state updated by UCXX receiver");
    spec.param(
        endpoint_, "endpoint", "Endpoint", "Overlay UCXX endpoint (for connectivity gating)");
  }

  void compute(holoscan::InputContext& input, holoscan::OutputContext& output,
               holoscan::ExecutionContext& context) override {
    (void)context;
    auto frame_entity = input.receive<holoscan::gxf::Entity>("in").value();

    // Attach frame counter tensor only if: a) the overlay subscriber is connected and
    // b) the frame counter state has a value (i.e. the overlay subscriber is sending updates).
    //
    // Attaching to frame entity using GXF Timestamp to bypass Holoviz's render type
    // auto-detection for tensors.
    if (endpoint_.get() && endpoint_.get()->endpoint()) {
      if (state_.get() && state_.get()->has_value()) {
        auto& gxf_frame_entity = static_cast<nvidia::gxf::Entity&>(frame_entity);
        auto maybe_counter_ts = gxf_frame_entity.get<nvidia::gxf::Timestamp>("counter_value");
        if (!maybe_counter_ts) {
          maybe_counter_ts = gxf_frame_entity.add<nvidia::gxf::Timestamp>("counter_value");
        }
        if (maybe_counter_ts) {
          const auto v = static_cast<int64_t>(state_.get()->get());
          maybe_counter_ts.value()->acqtime = v;
          maybe_counter_ts.value()->pubtime = v;
        }
      }
    }

    output.emit(frame_entity, "out");
  }

 private:
  holoscan::Parameter<std::shared_ptr<FrameCounterState>> state_;
  holoscan::Parameter<std::shared_ptr<holoscan::ops::UcxxEndpoint>> endpoint_;
};

}  // namespace

void UcxxEndoscopyPublisherApp::compose() {
  using namespace holoscan;

  HOLOSCAN_LOG_INFO("Composing PUBLISHER with full processing pipeline");

  // Constants for video dimensions
  const uint32_t width = 854;
  const uint32_t height = 480;
  const uint64_t source_block_size = width * height * 3 * 4;
  const uint64_t source_num_blocks = 2;

  // Create allocators
  auto replayer_allocator =
      make_resource<RMMAllocator>("video_replayer_allocator",
                                  Arg("device_memory_max_size") = std::string("256MB"),
                                  Arg("device_memory_initial_size") = std::string("256MB"));

  auto cuda_stream_pool = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

  // 2 UCXX endpoints for 2 subscribers: holoviz and overlay.
  // Note: the current UCXX endpoint implementation supports a single connection per listening
  // port. To support two parallel subscribers, we listen on two ports:
  // - port_      : subscriber_holoviz
  // - port_ + 1  : "overlay" application subscriber
  auto ucxx_endpoint_holoviz = make_resource<holoscan::ops::UcxxEndpoint>(
      "ucxx_endpoint_holoviz", Arg("hostname", hostname_), Arg("port", port_), Arg("listen", true));
  auto ucxx_endpoint_overlay =
      make_resource<holoscan::ops::UcxxEndpoint>("ucxx_endpoint_overlay",
                                                 Arg("hostname", hostname_),
                                                 Arg("port", port_ + 1),
                                                 Arg("listen", true));

  // -------------------------------------------------------------------------------------------
  // ---------------------------- Main Display: Tool Tracking ----------------------------------
  // -------------------------------------------------------------------------------------------

  // Video replayer source
  auto replayer = make_operator<ops::VideoStreamReplayerOp>("replayer",
                                                            Arg("allocator") = replayer_allocator,
                                                            Arg("directory", datapath_),
                                                            from_config("publisher.replayer"));

  // Format converter for inference input
  auto format_converter = make_operator<ops::FormatConverterOp>(
      "format_converter",
      from_config("publisher.format_converter"),
      Arg("pool") = make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks),
      Arg("cuda_stream_pool") = cuda_stream_pool);

  // LSTM inference
  const std::string model_file_path = datapath_ + "/tool_loc_convlstm.onnx";
  const std::string engine_cache_dir = datapath_ + "/engines";
  const uint64_t lstm_inferer_block_size = 107 * 60 * 7 * 4;
  const uint64_t lstm_inferer_num_blocks = 2 + 5 * 2;

  auto lstm_inferer = make_operator<ops::LSTMTensorRTInferenceOp>(
      "lstm_inferer",
      from_config("publisher.lstm_inference"),
      Arg("model_file_path", model_file_path),
      Arg("engine_cache_dir", engine_cache_dir),
      Arg("pool") = make_resource<BlockMemoryPool>(
          "pool", 1, lstm_inferer_block_size, lstm_inferer_num_blocks),
      Arg("cuda_stream_pool") = cuda_stream_pool);

  // Tool tracking postprocessor
  const uint64_t tool_tracking_postprocessor_block_size =
      std::max(107 * 60 * 7 * 4 * sizeof(float), 7 * 3 * sizeof(float));
  const uint64_t tool_tracking_postprocessor_num_blocks = 2 * 2;

  auto tool_tracking_postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
      "tool_tracking_postprocessor",
      Arg("device_allocator") =
          make_resource<BlockMemoryPool>("device_allocator",
                                         1,
                                         tool_tracking_postprocessor_block_size,
                                         tool_tracking_postprocessor_num_blocks));

  // Holoviz for visualization with render buffer output enabled for UCXX transmission
  const uint64_t render_buffer_size = width * height * 4 * 4;  // RGBA, 4 bytes per channel
  auto visualizer_allocator =
      make_resource<BlockMemoryPool>("allocator", 1, render_buffer_size, source_num_blocks);

  // Use Holoviz layer_callback to draw the frame counter value on the frame.
  auto holoviz = make_operator<ops::HolovizOp>(
      "holoviz",
      from_config("publisher.holoviz"),
      Arg("width") = width,
      Arg("height") = height,
      Arg("enable_render_buffer_output") = true,
      Arg("allocator") = visualizer_allocator,
      Arg("cuda_stream_pool") = cuda_stream_pool,
      Arg("layer_callback",
          ops::HolovizOp::LayerCallbackFunction(
              [](const std::vector<holoscan::gxf::Entity>& inputs) {
                // Draw the "counter_value" Timestamp component on the current frame entity.
                for (const auto& e : inputs) {
                  // Use const_cast since GXF get() is non-const. Read-only here.
                  auto& gxf_e =
                      static_cast<nvidia::gxf::Entity&>(const_cast<holoscan::gxf::Entity&>(e));
                  auto maybe_counter_ts = gxf_e.get<nvidia::gxf::Timestamp>("counter_value");
                  if (!maybe_counter_ts) {
                    continue;
                  }
                  const uint64_t v = static_cast<uint64_t>(maybe_counter_ts.value()->acqtime);
                  const std::string s = std::string("Frame Counter: ") + std::to_string(v);
                  holoscan::viz::BeginGeometryLayer();
                  holoscan::viz::Color(0.f, 1.f, 0.f, 1.f);  // green
                  holoscan::viz::Text(0.55f, 0.05f, 0.1f, s.c_str());
                  holoscan::viz::EndLayer();
                  return;
                }
              })));

  // -------------------------------------------------------------------------------------------
  // ---------------------------- Holoviz Subscriber -------------------------------------------
  // -------------------------------------------------------------------------------------------

  // Format converter to convert HolovizOp's VideoBuffer output to Tensor for UCXX
  auto render_buffer_converter =
      make_operator<ops::FormatConverterOp>("render_buffer_converter",
                                            Arg("in_dtype") = std::string("rgba8888"),
                                            Arg("out_dtype") = std::string("rgba8888"),
                                            Arg("out_tensor_name") = std::string(""),
                                            Arg("pool") = make_resource<BlockMemoryPool>(
                                                "pool", 1, render_buffer_size, source_num_blocks),
                                            Arg("cuda_stream_pool") = cuda_stream_pool);

  // UCXX sender to broadcast rendered frames for holoviz subscriber.
  auto ucxx_sender_holoviz = make_operator<holoscan::ops::UcxxSenderOp>(
      "ucxx_sender_holoviz",
      Arg("tag", 1ul),
      Arg("endpoint") = ucxx_endpoint_holoviz,
      Arg("allocator") = replayer_allocator,
      Arg("blocking") = from_config("publisher.blocking").as<bool>());

  // -------------------------------------------------------------------------------------------
  // ---------------------------- Overlay Subscriber -------------------------------------------
  // -------------------------------------------------------------------------------------------

  // Frame counter value state.
  auto frame_counter_state = make_resource<FrameCounterState>("frame_counter_state");

  // Convert original frame for UCXX transmission for overlay subscriber.
  auto source_to_ucxx = make_operator<ops::FormatConverterOp>(
      "source_rgba_to_ucxx",
      Arg("out_dtype") = std::string("rgba8888"),
      Arg("out_tensor_name") = std::string(""),
      Arg("pool") = make_resource<BlockMemoryPool>(
          "source_rgba_pool", 1, width * height * 4, source_num_blocks),
      Arg("cuda_stream_pool") = cuda_stream_pool);

  // UCXX sender to broadcast format-converted original frames for overlay subscriber.
  auto ucxx_sender_overlay = make_operator<holoscan::ops::UcxxSenderOp>(
      "ucxx_sender_overlay",
      Arg("tag", 1ul),
      Arg("endpoint") = ucxx_endpoint_overlay,
      Arg("allocator") = replayer_allocator,
      Arg("blocking") = from_config("publisher.blocking").as<bool>());

  // UCXX receiver to receive frame-counter overlay from overlay subscriber.
  auto overlay_rx_allocator = make_resource<UnboundedAllocator>("overlay_rx_allocator");
  auto ucxx_receiver_overlay = make_operator<holoscan::ops::UcxxReceiverOp>(
      "ucxx_receiver_overlay",
      Arg("tag", 2ul),
      Arg("buffer_size", 4 << 10),  // 4 KiB for scalar uint64 counter + header.
      Arg("endpoint") = ucxx_endpoint_overlay,
      Arg("allocator") = overlay_rx_allocator);

  // Update the frame counter state with the received frame-counter overlay.
  auto frame_counter_update = make_operator<FrameCounterUpdateOp>(
      "frame_counter_update", Arg("state") = frame_counter_state);

  // Attach the current frame counter value to source frame for display.
  auto frame_counter_attach =
      make_operator<FrameCounterAttachValueOp>("frame_counter_attach",
                                               Arg("state") = frame_counter_state,
                                               Arg("endpoint") = ucxx_endpoint_overlay);

  // Clear the overlay immediately when the overlay subscriber disconnects.
  ucxx_endpoint_overlay->add_close_callback(
      [frame_counter_state](ucs_status_t) { frame_counter_state->clear(); });

  // -------------------------------------------------------------------------------------------
  // ---------------------------- Build the Pipeline -------------------------------------------
  // -------------------------------------------------------------------------------------------

  // Path 1: LSTM inference and tool tracking display. Parallel to Path 2.
  add_flow(replayer, format_converter, {{"output", "source_video"}});
  add_flow(format_converter, lstm_inferer);
  add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});
  add_flow(tool_tracking_postprocessor, holoviz, {{"out", "receivers"}});

  // Path 2: Frame + counter overlay display. Parallel to Path 1.
  add_flow(replayer, frame_counter_attach, {{"output", "in"}});
  add_flow(frame_counter_attach, holoviz, {{"out", "receivers"}});

  // Path 3: UCXX transmission for Holoviz subscriber. Follows Path 1 and Path 2.
  add_flow(holoviz, render_buffer_converter, {{"render_buffer_output", "source_video"}});
  add_flow(render_buffer_converter, ucxx_sender_holoviz, {{"", "in"}});

  // Path 4: UCXX transmission for overlay subscriber. From source directly.
  add_flow(replayer, source_to_ucxx, {{"output", "source_video"}});
  add_flow(source_to_ucxx, ucxx_sender_overlay, {{"", "in"}});

  // Path 5: UCXX receiver for frame-counter overlay update. Independent of other paths.
  add_flow(ucxx_receiver_overlay, frame_counter_update, {{"out", "in"}});

  HOLOSCAN_LOG_INFO(
      "Publisher pipeline: Replayer → "
      "(Format→LSTM→Postprocess→Holoviz→Convert→Send(tag=1,port=50008)) "
      "+ (Source→Convert→Send(tag=1,port=50009)) "
      "+ (Recv(tag=2,port=50009)→UpdateCounterState→HolovizTextLayer)");
}

}  // namespace holoscan::apps
