/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, Real-Time Innovations, Inc. All rights reserved.
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

#include "dds_video_subscriber.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

#include "dds/topic/find.hpp"

namespace holoscan::ops {

void DDSVideoSubscriberOp::setup(OperatorSpec& spec) {
  DDSOperatorBase::setup(spec);

  spec.output<gxf::Entity>("output");

  spec.param(allocator_, "allocator", "Allocator", "Allocator for output buffers.");
  spec.param(reader_qos_, "reader_qos", "Reader QoS", "Data Reader QoS Profile", std::string());
  spec.param(stream_id_, "stream_id", "Stream ID for the video stream");
}

void DDSVideoSubscriberOp::initialize() {
  DDSOperatorBase::initialize();

  // Create the subscriber
  dds::sub::Subscriber subscriber(participant_);

  // Create the VideoFrame topic
  const std::string topic_name(VIDEO_FRAME_TOPIC);
  auto topic = dds::topic::find<dds::topic::Topic<VideoFrame>>(participant_, topic_name);
  if (topic == dds::core::null) {
    topic = dds::topic::Topic<VideoFrame>(participant_, topic_name);
  }

  // Create the filtered topic for the requested stream id.
  dds::topic::ContentFilteredTopic<VideoFrame> filtered_topic(topic,
      "FilteredVideoFrame",
      dds::topic::Filter("stream_id = %0", {std::to_string(stream_id_.get())}));

  // Create the reader for the VideoFrame
  reader_ = std::make_unique<dds::sub::DataReader<VideoFrame>>(
      subscriber, filtered_topic, qos_provider_.datareader_qos(reader_qos_.get()));
}

void DDSVideoSubscriberOp::compute(InputContext& op_input,
                                   OutputContext& op_output,
                                   ExecutionContext& context) {
  dds::sub::LoanedSamples<VideoFrame> frames = reader_->take();
  const auto frame = std::find_if(frames.begin(), frames.end(), [](const auto& sample) {
    return sample.info().valid();
  });
  if (frame == frames.end()) {
    // Return control to the scheduler regularly so the application can stop
    // cleanly while no DDS samples are available.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return;
  }

  auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      context.context(), allocator_->gxf_cid());

  auto output = nvidia::gxf::Entity::New(context.context());
  if (!output) {
    throw std::runtime_error("Failed to allocate message for output");
  }

  auto video_buffer = output.value().add<nvidia::gxf::VideoBuffer>();
  if (!video_buffer) {
    throw std::runtime_error("Failed to allocate video buffer");
  }

  // Copy the frame to the output buffer.
  video_buffer.value()->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
      frame->data().width, frame->data().height,
      nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
      nvidia::gxf::MemoryStorageType::kHost, allocator.value());
  memcpy(video_buffer.value()->pointer(), frame->data().data.data(), frame->data().data.size());

  // Output the buffer
  auto result = gxf::Entity(std::move(output.value()));
  op_output.emit(result, "output");
}

void DDSVideoSubscriberOp::stop() {
  if (reader_) {
    reader_->close();
    reader_.reset();
  }
}

}  // namespace holoscan::ops
