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

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

#include "matlab_utils.h"
#include "matlab_image_processing.h"
#include "matlab_image_processing_terminate.h"

namespace holoscan::ops {
class MatlabImageProcessingOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MatlabImageProcessingOp)

  MatlabImageProcessingOp() = default;

  void setup(OperatorSpec& spec) override {
    auto& in_tensor = spec.input<gxf::Entity>("in_tensor");
    auto& out_tensor = spec.output<gxf::Entity>("out_tensor");

    spec.param(in_, "in", "Input", "Input channel.", &in_tensor);
    spec.param(out_, "out", "Output", "Output channel.", &out_tensor);
    spec.param(in_tensor_name_,
               "in_tensor_name",
               "InputTensorName",
               "Name of the input tensor.",
               std::string(""));
    spec.param(out_tensor_name_,
               "out_tensor_name",
               "OutputTensorName",
               "Name of the output tensor.",
               std::string(""));
    // MATLAB function output specifications
    spec.param(out_width_, "out_width", "OutWidth", "Output width.", 0U);
    spec.param(out_height_, "out_height", "OutHeight", "Output height.", 0U);
    spec.param(out_channels_, "out_channels", "OutChannels", "Output number of channels.", 0U);
    // Matlab function arguments
    spec.param(sigma_, "sigma", "Sigma", "Smoothing kernel standard deviation.", 4.0f);
    // Allocator/CUDA
    spec.param(allocator_, "allocator", "Allocator", "Output Allocator");
    cuda_stream_handler_.defineParams(spec);
  }

  void start() {
    if (out_width_.get() == 0 || out_height_.get() == 0 || out_channels_.get() == 0) {
      throw std::runtime_error(
          "Parameters out_width and/or out_height and/or out_channels_ not specified!");
    }
    // Set output tensor shapes
    out_shape_.push_back(out_height_.get());
    out_shape_.push_back(out_width_.get());
    out_shape_.push_back(out_channels_.get());
  }

  void stop() {
    matlab_image_processing_terminate();
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    // Get input message
    auto in_message = op_input.receive<gxf::Entity>("in_tensor").value();

    // Get CUDA stream
    gxf_result_t stream_handler_result =
        cuda_stream_handler_.from_message(context.context(), in_message);
    if (stream_handler_result != GXF_SUCCESS) {
      throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
    }
    auto cuda_stream = cuda_stream_handler_.get_cuda_stream(context.context());

    // Get input tensor
    const std::string in_tensor_name = in_tensor_name_.get();
    auto maybe_tensor = in_message.get<Tensor>(in_tensor_name.c_str());
    if (!maybe_tensor) {
      maybe_tensor = in_message.get<Tensor>();
      if (!maybe_tensor) {
        throw std::runtime_error(fmt::format("Tensor '{}' not found in message", in_tensor_name));
      }
    }
    auto in_tensor = maybe_tensor;
    auto in_tensor_shape = in_tensor->shape();
    std::vector<int32_t> in_shape = {
      static_cast<int32_t>(in_tensor_shape[0]),
      static_cast<int32_t>(in_tensor_shape[1]),
      static_cast<int32_t>(in_tensor_shape[2])
    };
    nvidia::gxf::Shape out_tensor_shape = {out_shape_[0], out_shape_[1], out_shape_[2]};

    // Get input tensor data
    uint8_t* in_tensor_data = static_cast<uint8_t*>(in_tensor->data());
    if (!in_tensor_data) { throw std::runtime_error("Failed to get in tensor data!"); }

    // Get allocator
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      context.context(), allocator_->gxf_cid());

    // Allocate output buffer on the device.
    auto out_message = nvidia::gxf::Entity::New(context.context());
    auto out_tensor = out_message.value().add<nvidia::gxf::Tensor>(
      out_tensor_name_.get().c_str());
    if (!out_tensor) { throw std::runtime_error("Failed to allocate output tensor"); }
    out_tensor.value()->reshape<uint8_t>(
        out_tensor_shape, nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
    if (!out_tensor.value()->pointer()) {
      throw std::runtime_error("Failed to allocate output tensor buffer.");
    }
    // Get output data
    nvidia::gxf::Expected<uint8_t*> out_tensor_data = out_tensor.value()->data<uint8_t>();
    if (!out_tensor_data) { throw std::runtime_error("Failed to get out tensor data!"); }

    // Allocate temporary input tensor (for storing row-to-column converted data)
    auto tmp_tensor_in = make_tensor(in_shape, nvidia::gxf::PrimitiveType::kUnsigned8,
                                     sizeof(uint8_t), allocator.value());
    nvidia::gxf::Expected<uint8_t*> tmp_tensor_in_data = tmp_tensor_in->data<uint8_t>();
    if (!tmp_tensor_in_data) {
      throw std::runtime_error("Failed to get temporary tensor input data!");
    }
    // Allocate temporary output tensor (for storing column-to-row converted data)
    auto tmp_tensor_out = make_tensor(out_shape_, nvidia::gxf::PrimitiveType::kUnsigned8,
                                      sizeof(uint8_t), allocator.value());
    nvidia::gxf::Expected<uint8_t*> tmp_tensor_out_data = tmp_tensor_out->data<uint8_t>();
    if (!tmp_tensor_out_data) {
      throw std::runtime_error("Failed to get temporary tensor output data!");
    }

    // Convert output from row- to column-major ordering
    cuda_hard_transpose<uint8_t>(in_tensor_data, tmp_tensor_in_data.value(), in_shape,
                                 cuda_stream, Flip::DoNot);

    // Call MATLAB CUDA function to do image processing
    matlab_image_processing(
      tmp_tensor_in_data.value(), sigma_.get(), tmp_tensor_out_data.value());
    delete tmp_tensor_in;

    // Convert output from column- to row-major ordering
    cuda_hard_transpose<uint8_t>(tmp_tensor_out_data.value(), out_tensor_data.value(), out_shape_,
                                 cuda_stream, Flip::Do);
    delete tmp_tensor_out;

    // Pass the CUDA stream to the output message
    stream_handler_result = cuda_stream_handler_.to_message(out_message);
    if (stream_handler_result != GXF_SUCCESS) {
      throw std::runtime_error("Failed to add the CUDA stream to the outgoing messages");
    }

    // Create output message
    auto result = gxf::Entity(std::move(out_message.value()));
    op_output.emit(result);
  }

 private:
  Parameter<holoscan::IOSpec*> in_;
  Parameter<holoscan::IOSpec*> out_;
  Parameter<std::string> in_tensor_name_;
  Parameter<std::string> out_tensor_name_;
  Parameter<uint32_t> out_width_;
  Parameter<uint32_t> out_height_;
  Parameter<uint32_t> out_channels_;
  Parameter<float> sigma_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  CudaStreamHandler cuda_stream_handler_;
  std::vector<int32_t> out_shape_;
};

}  // namespace holoscan::ops

class MatlabImageProcessingApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    // Define operators and configure using yaml configuration
    auto replayer = make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"));
    auto matlab = make_operator<ops::MatlabImageProcessingOp>(
        "matlab",
        from_config("matlab"),
        Arg("allocator") = make_resource<BlockMemoryPool>(
            "pool", 1, 854 * 480 * 3 * 4, 4), /* width * height * channels * bpp */
        Arg("cuda_stream_pool") = cuda_stream_pool);
    auto visualizer =
        make_operator<ops::HolovizOp>("holoviz",
                                      from_config("holoviz"),
                                      Arg("allocator") = make_resource<UnboundedAllocator>("pool"),
                                      Arg("cuda_stream_pool") = cuda_stream_pool);

    // Define the workflow
    add_flow(replayer, matlab, {{"output", "in_tensor"}});
    add_flow(matlab, visualizer, {{"out_tensor", "receivers"}});
  }
};

int main(int argc, char** argv) {
  // Get the yaml configuration file
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path /= std::filesystem::path("matlab_image_processing.yaml");
  if (argc >= 2) { config_path = argv[1]; }

  auto app = holoscan::make_application<MatlabImageProcessingApp>();
  app->config(config_path);
  app->run();

  return 0;
}
