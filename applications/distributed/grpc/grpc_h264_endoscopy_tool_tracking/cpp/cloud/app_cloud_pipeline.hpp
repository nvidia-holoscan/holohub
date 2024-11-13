/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_CLOUD_APP_CLOUD_PIPELINE_HPP
#define GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_CLOUD_APP_CLOUD_PIPELINE_HPP

#include <gxf/core/entity.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <grpc_server.hpp>
#include <lstm_tensor_rt_inference.hpp>
#include <tool_tracking_postprocessor.hpp>

#include "../gxf_imports.hpp"

namespace holohub::grpc_h264_endoscopy_tool_tracking {

using namespace holoscan;
using namespace holoscan::ops;

using holoscan::entity::EntityResponse;

/**
 * @class AppCloudPipeline
 * @brief A class that represents the application pipeline for H264 endoscopy tool tracking.
 *
 * This class inherits from HoloscanGrpcApplication and is responsible for composing the pipeline
 * for processing video frames and performing tool tracking using a series of operators.
 *
 * @note the `HoloscanGrpcApplication` base class composes the `GrpcServerRequestOp` and
 * `GrpcServerResponseOp` operators to handle incoming requests and outgoing responses.
 * It also configures the queues for handling requests and responses.
 */
class AppCloudPipeline : public HoloscanGrpcApplication {
 public:
  AppCloudPipeline(std::queue<std::shared_ptr<nvidia::gxf::Entity>> incoming_request_queue,
                   std::queue<std::shared_ptr<EntityResponse>> outgoing_response_queue)
      : HoloscanGrpcApplication(incoming_request_queue, outgoing_response_queue) {}

  void compose() override {
    // Call base class compose to initialize the queues.
    HoloscanGrpcApplication::compose();

    // Create the Endoscopy Tool Tracking (ETT) Pipeline similar to the regular ETT application.
    auto response_condition = make_condition<AsynchronousCondition>("response_condition");
    auto video_decoder_context = make_resource<VideoDecoderContext>(
        "decoder-context", Arg("async_scheduling_term") = response_condition);

    auto request_condition = make_condition<AsynchronousCondition>("request_condition");
    auto video_decoder_request =
        make_operator<VideoDecoderRequestOp>("video_decoder_request",
                                             from_config("video_decoder_request"),
                                             Arg("async_scheduling_term") = request_condition,
                                             Arg("videodecoder_context") = video_decoder_context);

    auto video_decoder_response = make_operator<VideoDecoderResponseOp>(
        "video_decoder_response",
        streaming_enabled,
        from_config("video_decoder_response"),
        Arg("pool") = make_resource<RMMAllocator>(
            "pool", Arg("device_memory_max_size") = std::string("256MB")),
        Arg("videodecoder_context") = video_decoder_context);

    auto cuda_stream_pool = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    auto decoder_output_format_converter = make_operator<ops::FormatConverterOp>(
        "decoder_output_format_converter",
        streaming_enabled,
        from_config("decoder_output_format_converter"),
        Arg("pool") = make_resource<RMMAllocator>(
            "pool", Arg("device_memory_max_size") = std::string("256MB")),
        Arg("cuda_stream_pool") = cuda_stream_pool);

    auto rgb_float_format_converter = make_operator<ops::FormatConverterOp>(
        "rgb_float_format_converter",
        streaming_enabled,
        from_config("rgb_float_format_converter"),
        Arg("pool") = make_resource<RMMAllocator>(
            "pool", Arg("device_memory_max_size") = std::string("256MB")),
        Arg("cuda_stream_pool") = cuda_stream_pool);

    const std::string model_file_path = data_path + "/tool_loc_convlstm.onnx";
    const std::string engine_cache_dir = data_path + "/engines";

    auto lstm_inferer = make_operator<ops::LSTMTensorRTInferenceOp>(
        "lstm_inferer",
        streaming_enabled,
        from_config("lstm_inference"),
        Arg("model_file_path", model_file_path),
        Arg("engine_cache_dir", engine_cache_dir),
        Arg("pool") = make_resource<RMMAllocator>(
            "pool", Arg("device_memory_max_size") = std::string("256MB")),
        Arg("cuda_stream_pool") = cuda_stream_pool);

    auto tool_tracking_postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
        "tool_tracking_postprocessor",
        streaming_enabled,
        from_config("tool_tracking_postprocessor"),
        Arg("cuda_stream_pool") = cuda_stream_pool,
        Arg("device_allocator") = make_resource<RMMAllocator>(
            "device_allocator", Arg("device_memory_max_size") = std::string("256MB")));

    // Here we connect the GrpcServerRequestOp to the VideoDecoderRequestOp to send the received
    // video frames for decoding.
    add_flow(grpc_request_op, video_decoder_request, {{"output", "input_frame"}});

    add_flow(video_decoder_response,
             decoder_output_format_converter,
             {{"output_transmitter", "source_video"}});
    add_flow(
        decoder_output_format_converter, rgb_float_format_converter, {{"tensor", "source_video"}});
    add_flow(rgb_float_format_converter, lstm_inferer);
    add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});

    // Lastly, we connect the results from the tool tracking postprocessor to the
    // GrpcServerResponseOp so the pipeline can return the results back to the client
    add_flow(tool_tracking_postprocessor, grpc_response_op, {{"out", "input"}});
  }
};
}  // namespace holohub::grpc_h264_endoscopy_tool_tracking
#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_CLOUD_APP_CLOUD_PIPELINE_HPP */
