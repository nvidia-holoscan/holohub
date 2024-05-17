/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dds_video_publisher.hpp"

#include <dds/topic/find.hpp>

namespace holoscan::ops {

void DDSVideoPublisherOp::setup(OperatorSpec& spec) {
  DDSOperatorBase::setup(spec);

  spec.input<gxf::Entity>("input");

  spec.param(writer_qos_, "writer_qos", "Writer QoS", "Data Writer QoS Profile", std::string());
  spec.param(stream_id_, "stream_id", "Stream ID", "Stream ID for the DDS Video Stream", 0u);
}

void DDSVideoPublisherOp::initialize() {
  DDSOperatorBase::initialize();

  // Create the publisher
  dds::pub::Publisher publisher(participant_);

  // Create the VideoFrame topic
  auto topic = dds::topic::find<dds::topic::Topic<VideoFrame>>(participant_, VIDEO_FRAME_TOPIC);
  if (topic == dds::core::null) {
    topic = dds::topic::Topic<VideoFrame>(participant_, VIDEO_FRAME_TOPIC);
  }

  // Create the writer for the VideoFrame
  writer_ = dds::pub::DataWriter<VideoFrame>(publisher, topic,
                                             qos_provider_.datawriter_qos(writer_qos_.get()));
}

void DDSVideoPublisherOp::compute(InputContext& op_input,
                                  OutputContext& op_output,
                                  ExecutionContext& context) {
  auto input = op_input.receive<gxf::Entity>("input").value();
  if (!input) {
    throw std::runtime_error("No input available");
  }

  const auto& buffer = static_cast<nvidia::gxf::Entity>(input).get<nvidia::gxf::VideoBuffer>();
  if (!buffer) {
    throw std::runtime_error("No video buffer attached to input");
  }

  const auto& info = buffer.value()->video_frame_info();
  if (info.color_format != nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA) {
    throw std::runtime_error("Invalid buffer format; Only RGBA is supported");
  }

  // Create the VideoFrame sample from the input buffer
  std::vector<uint8_t> data(buffer.value()->size());
  if (buffer.value()->storage_type() == nvidia::gxf::MemoryStorageType::kHost) {
    memcpy(data.data(), buffer.value()->pointer(), data.size());
  } else {
    cudaMemcpy(data.data(), buffer.value()->pointer(), data.size(), cudaMemcpyDeviceToHost);
  }
  VideoFrame frame(stream_id_.get(), frame_num_++, info.width, info.height, data);

  // Write the VideoFrame to the writer
  writer_.write(frame);
}

}  // namespace holoscan::ops
