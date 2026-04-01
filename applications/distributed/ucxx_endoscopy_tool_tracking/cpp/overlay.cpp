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

#include "overlay.h"

#include <cstdint>
#include <string>

#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoviz/holoviz.hpp>

#include "gxf/std/timestamp.hpp"

#include <operators/ucxx_send_receive/ucxx_endpoint.hpp>
#include <operators/ucxx_send_receive/receiver_op/ucxx_receiver_op.hpp>
#include <operators/ucxx_send_receive/sender_op/ucxx_sender_op.hpp>

namespace holoscan::apps {

namespace {

// Get incoming frame entity and emits a frame counter for UCXX send-back.
// Optionally emits the incoming frame entity for local visualization.
//
// This is analogous to a per-frame processing algorithm. In real-world,
// this can be for instance an AI application that segments organs from
// each frame.
class FrameCounterOverlayOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FrameCounterOverlayOp)

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("in");
    // `frame_out`: the incoming frame entity (for optional local visualization)
    // `overlay_out`: frame counter overlay to send via UCXX
    spec.output<holoscan::gxf::Entity>("frame_out");
    spec.output<holoscan::gxf::Entity>("overlay_out");
    spec.param(allocator_,
               "allocator",
               "Allocator",
               "Host allocator for the outgoing scalar counter tensor",
               std::shared_ptr<holoscan::Allocator>{});
  }

  void compute(holoscan::InputContext& input, holoscan::OutputContext& output,
               holoscan::ExecutionContext& context) override {
    auto frame_entity = input.receive<holoscan::gxf::Entity>("in").value();

    frame_idx_++;

    if (!allocator_.get()) {
      HOLOSCAN_LOG_ERROR("FrameCounterOverlayOp: allocator is not set");
      return;
    }

    auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), allocator_.get()->gxf_cid());
    if (!gxf_allocator) {
      HOLOSCAN_LOG_ERROR("FrameCounterOverlayOp: failed to create GXF allocator handle");
      return;
    }

    // Create a standalone overlay_message containing the frame counter for UCXX send-back.
    auto overlay_message = holoscan::gxf::Entity::New(&context);
    auto maybe_overlay_tensor =
        static_cast<nvidia::gxf::Entity&>(overlay_message).add<nvidia::gxf::Tensor>("");
    if (!maybe_overlay_tensor) {
      HOLOSCAN_LOG_ERROR("FrameCounterOverlayOp: failed to add output overlay tensor \"\"");
      return;
    }
    auto overlay_tensor = maybe_overlay_tensor.value();
    const nvidia::gxf::Shape scalar_shape{1};
    const auto ok = overlay_tensor->reshape<uint64_t>(
        scalar_shape, nvidia::gxf::MemoryStorageType::kHost, gxf_allocator.value());
    if (!ok) {
      HOLOSCAN_LOG_ERROR("FrameCounterOverlayOp: failed to reshape output counter tensor");
      return;
    }

    *reinterpret_cast<uint64_t*>(overlay_tensor->pointer()) = frame_idx_;

    // Local visualization: attach the frame counter value to the frame entity.
    // Use GXF Timestamp to bypass Holoviz's render type auto-detection for tensors.
    auto& gxf_frame_entity = static_cast<nvidia::gxf::Entity&>(frame_entity);
    auto maybe_counter_ts = gxf_frame_entity.get<nvidia::gxf::Timestamp>("counter_value");
    if (!maybe_counter_ts) {
      maybe_counter_ts = gxf_frame_entity.add<nvidia::gxf::Timestamp>("counter_value");
    }
    if (maybe_counter_ts) {
      const auto v = static_cast<int64_t>(frame_idx_);
      maybe_counter_ts.value()->acqtime = v;
      maybe_counter_ts.value()->pubtime = v;
    }

    // Emit both outputs.
    output.emit(frame_entity, "frame_out");
    output.emit(overlay_message, "overlay_out");
  }

 private:
  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  uint64_t frame_idx_{0};
};

// Helper operator to drop frame entity when visualization is disabled.
class SinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SinkOp)

  SinkOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.input<holoscan::gxf::Entity>("in"); }

  void compute(holoscan::InputContext& input, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    (void)input.receive<holoscan::gxf::Entity>("in");
  }
};

}  // namespace

void UcxxEndoscopyOverlayApp::compose() {
  using namespace holoscan;

  const uint32_t width = 854;
  const uint32_t height = 480;
  const bool visualize = from_config("overlay.visualize").as<bool>();

  HOLOSCAN_LOG_INFO(
      "Composing overlay - receiving frames and sending frame-counter overlay "
      "back");

  auto allocator =
      make_resource<RMMAllocator>("video_replayer_allocator",
                                  Arg("device_memory_max_size") = std::string("256MB"),
                                  Arg("device_memory_initial_size") = std::string("256MB"));

  auto ucxx_endpoint = make_resource<holoscan::ops::UcxxEndpoint>(
      "ucxx_endpoint", Arg("hostname", hostname_), Arg("port", port_), Arg("listen", false));

  // Buffer size: RGB frame (width * height * 3 channels)
  const int buffer_size = width * height * 3;
  auto ucxx_receiver = make_operator<holoscan::ops::UcxxReceiverOp>(
      "ucxx_receiver",
      Arg("tag", 1ul),
      Arg("buffer_size", buffer_size),
      Arg("endpoint") = ucxx_endpoint,
      Arg("allocator") = allocator);

  auto frame_counter_overlay = make_operator<FrameCounterOverlayOp>(
      "frame_counter_overlay",
      Arg("allocator") = make_resource<UnboundedAllocator>("frame_counter_allocator"));

  auto ucxx_sender_back = make_operator<holoscan::ops::UcxxSenderOp>(
      "ucxx_sender_back",
      Arg("tag", 2ul),
      Arg("endpoint") = ucxx_endpoint,
      Arg("blocking") = false);

  // Receives frames from publisher and sends frame-counter overlay (overlay_out) back to publisher.
  add_flow(ucxx_receiver, frame_counter_overlay, {{"out", "in"}});
  add_flow(frame_counter_overlay, ucxx_sender_back, {{"overlay_out", "in"}});

  // Use Holoviz layer_callback to draw the frame counter value on the frame.
  if (visualize) {
    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
        from_config("overlay.holoviz"),
        Arg("width") = width,
        Arg("height") = height,
        Arg("allocator") = allocator,
        Arg("layer_callback",
            ops::HolovizOp::LayerCallbackFunction(
                [](const std::vector<holoscan::gxf::Entity>& inputs) {
                  // Draw the "counter_value" Timestamp component on the current frame entity.
                  for (const auto& e : inputs) {
                    auto maybe_counter_ts =
                        static_cast<nvidia::gxf::Entity&>(const_cast<holoscan::gxf::Entity&>(e))
                            .get<nvidia::gxf::Timestamp>("counter_value");
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
    add_flow(frame_counter_overlay, holoviz, {{"frame_out", "receivers"}});
  } else {
    // Drop the frame_out stream so both output entities of
    // `frame_counter_overlay` are always fully connected.
    auto drop = make_operator<SinkOp>("sink_frame_out");
    add_flow(frame_counter_overlay, drop, {{"frame_out", "in"}});
  }

  HOLOSCAN_LOG_INFO(
      "Overlay pipeline: Receive(tag=1) → CountFrames → SendBack(tag=2) (+ optional "
      "local text)");
}

}  // namespace holoscan::apps
