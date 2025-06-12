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

#include "dds_video_subscriber.hpp"

#include "dds/topic/find.hpp"

namespace holoscan::ops {

void DDSVideoSubscriberOp::setup(OperatorSpec& spec) {
  DDSOperatorBase::setup(spec);

  spec.output<gxf::Entity>("output");

  spec.param(allocator_, "allocator", "Allocator", "Allocator for output buffers.");
  spec.param(reader_qos_, "reader_qos", "Reader QoS", "Data Reader QoS Profile", std::string());
  spec.param(stream_id_, "stream_id", "Stream ID for the video stream");
  spec.param(fps_report_interval_,
             "fps_report_interval",
             "FPS Report Interval",
             "Interval in seconds to report FPS statistics",
             1.0);
}

void DDSVideoSubscriberOp::initialize() {
  DDSOperatorBase::initialize();

  // Create the subscriber
  dds::sub::Subscriber subscriber(participant_);

  // Create the VideoFrame topic
  auto topic = dds::topic::find<dds::topic::Topic<VideoFrame>>(participant_, VIDEO_FRAME_TOPIC);
  if (topic == dds::core::null) {
    topic = dds::topic::Topic<VideoFrame>(participant_, VIDEO_FRAME_TOPIC);
  }

  // Create the filtered topic for the requested stream id.
  dds::topic::ContentFilteredTopic<VideoFrame> filtered_topic(
      topic,
      "FilteredVideoFrame",
      dds::topic::Filter("stream_id = %0", {std::to_string(stream_id_.get())}));

  // Create the reader for the VideoFrame
  reader_ = dds::sub::DataReader<VideoFrame>(
      subscriber, filtered_topic, qos_provider_.datareader_qos(reader_qos_.get()));

  // Obtain the reader's status condition
  status_condition_ = dds::core::cond::StatusCondition(reader_);

  // Enable the 'data available' status
  status_condition_.enabled_statuses(dds::core::status::StatusMask::data_available());

  // Attach the status condition to the waitset
  waitset_ += status_condition_;

  // Initialize FPS tracking
  frame_count_ = 0;
  timing_initialized_ = false;
}

void DDSVideoSubscriberOp::compute(InputContext& op_input, OutputContext& op_output,
                                   ExecutionContext& context) {
  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

  auto output = nvidia::gxf::Entity::New(context.context());
  if (!output) {
    throw std::runtime_error("Failed to allocate message for output");
  }

  bool output_written = false;
  while (!output_written) {
    // Wait for a new frame
    dds::core::cond::WaitSet::ConditionSeq active_conditions =
        waitset_.wait(dds::core::Duration::from_secs(1));

    for (const auto& cond : active_conditions) {
      if (cond == status_condition_) {
        // Take the available frame
        dds::sub::LoanedSamples<VideoFrame> frames = reader_.take();
        auto current_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count();
        for (const auto& frame : frames) {
          if (frame.info().valid()) {
            auto selected_frame = frame.data();
            // Initialize timing on first frame
            auto transfer_time = current_time - selected_frame.transfer_start_time();
            if (!timing_initialized_) {
              start_time_ = std::chrono::steady_clock::now();
              last_fps_report_time_ = start_time_;
              timing_initialized_ = true;
            } else {
              transfer_times_.push_back(transfer_time);
              // Keep only the last 1000 samples
              if (transfer_times_.size() > 1000) {
                transfer_times_.erase(transfer_times_.begin());
              }

              frame_sizes.push_back(selected_frame.data().size());
              if (frame_sizes.size() > 1000) {
                frame_sizes.erase(frame_sizes.begin());
              }

              // Print transfer time if it's greater than 10ms
              if (transfer_time > 10000000) {
                HOLOSCAN_LOG_WARN("Transfer time: {}ns/{}ms, Frame Number: {}",
                                  transfer_time,
                                  transfer_time / 1000000.0,
                                  selected_frame.frame_num());
              }
            }

            // Increment frame count
            frame_count_++;

            // Calculate and report FPS at specified intervals
            auto current_time = std::chrono::steady_clock::now();
            auto duration_since_last_report =
                std::chrono::duration_cast<std::chrono::duration<double>>(current_time -
                                                                          last_fps_report_time_)
                    .count();

            if (duration_since_last_report >= fps_report_interval_.get()) {
              auto total_duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                                        current_time - start_time_)
                                        .count();

              double average_fps = frame_count_ / total_duration;

              if (fps_report_interval_.get() > 0.0) {
                double average_transfer_time = 0.0;
                double jitter_time = 0.0;
                if (transfer_times_.size() > 0) {
                  // Calculate average
                  auto sum = std::accumulate(transfer_times_.begin(), transfer_times_.end(), 0LL);
                  double avg_ns =
                      static_cast<double>(sum) / static_cast<double>(transfer_times_.size());
                  average_transfer_time = avg_ns / 1000000.0;

                  // Calculate jitter (standard deviation)
                  if (transfer_times_.size() > 1) {
                    double variance_sum = 0.0;
                    for (const auto& time : transfer_times_) {
                      double diff = static_cast<double>(time) - avg_ns;
                      variance_sum += diff * diff;
                    }
                    double variance =
                        variance_sum / static_cast<double>(transfer_times_.size() - 1);
                    jitter_time = std::sqrt(variance) / 1000000.0;  // Convert to ms
                  }
                }

                auto sum_frame_sizes = std::accumulate(frame_sizes.begin(), frame_sizes.end(), 0LL);
                double avg_frame_size =
                    static_cast<double>(sum_frame_sizes) / static_cast<double>(frame_sizes.size());

                HOLOSCAN_LOG_INFO(
                    "DDS Video Subscriber - Stream ID: {} | Total Frames: {} | "
                    "Average FPS: {:.3f} | Total Time: {:.3f}s | Width: {} | Height: {} | Avg "
                    "Size: {} "
                    "| Codec: {} | Avg Transfer Time: {:.3f}ms | Jitter: {:.3f}ms",
                    stream_id_.get(),
                    frame_count_,
                    average_fps,
                    total_duration,
                    selected_frame.width(),
                    selected_frame.height(),
                    avg_frame_size,
                    static_cast<int>(selected_frame.codec()),
                    average_transfer_time,
                    jitter_time);
              }

              last_fps_report_time_ = current_time;
            }

            if (selected_frame.codec() == Codec::H264) {
              auto tensor = output.value().add<nvidia::gxf::Tensor>();
              if (!tensor) {
                throw std::runtime_error("Failed to allocate tensor");
              }
              tensor.value()->reshape<uint8_t>(nvidia::gxf::Shape({selected_frame.data().size()}),
                                               nvidia::gxf::MemoryStorageType::kHost,
                                               allocator.value());
              memcpy(tensor.value()->pointer(),
                     selected_frame.data().data(),
                     selected_frame.data().size());
              output_written = true;
            } else {
              auto video_buffer = output.value().add<nvidia::gxf::VideoBuffer>();
              if (!video_buffer) {
                throw std::runtime_error("Failed to allocate video buffer");
              }
              // Copy the frame to the output buffer
              video_buffer.value()->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
                  selected_frame.width(),
                  selected_frame.height(),
                  nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
                  nvidia::gxf::MemoryStorageType::kHost,
                  allocator.value());
              memcpy(video_buffer.value()->pointer(),
                     selected_frame.data().data(),
                     selected_frame.data().size());
              output_written = true;
            }
          }
        }
      }
    }
  }

  // Output the buffer
  auto result = gxf::Entity(std::move(output.value()));
  op_output.emit(result, "output");
}

}  // namespace holoscan::ops
