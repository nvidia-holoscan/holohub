/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <getopt.h>

#include <cstdint>
#include <memory>
#include <string>

#include "cvcuda_utils.hpp"
#include "holoscan/holoscan.hpp"
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <cvcuda/OpFlip.hpp>     // cvcuda::Flip
#include <nvcv/DataType.hpp>     // nvcv::DataType
#include <nvcv/Tensor.hpp>       // nvcv::Tensor
#include <nvcv/TensorData.hpp>   // nvcv::TensorDataStridedCuda

namespace holoscan::ops {

/**
 * @brief Find a tensor in the tensormap and validate that it has the expected shape and dtype
 *
 * @param tensormap A holoscan::TensorMap containing a single RGB tensor.
 * @return A shared pointer to the holoscan tensor.
 */
std::shared_ptr<holoscan::Tensor> validate_and_get_rgb_image(holoscan::TensorMap tensormap,
                                                             bool has_batch_dimension = false) {
  std::shared_ptr<holoscan::Tensor> in_tensor;
  auto n_tensors = tensormap.size();
  if (n_tensors != 1) {
    throw std::runtime_error(
        fmt::format("expected exactly 1 tensor in input_tensor map, found {}", n_tensors));
  } else {
    // get the tensor without needing to know the key name
    for (auto& [key, tensor] : tensormap) { in_tensor = tensor; }
  }

  int expected_ndim = has_batch_dimension ? 4 : 3;

  // assume 2D + channels without batch dimension
  if (in_tensor->ndim() != expected_ndim) {
    if (expected_ndim == 4) {
      throw std::runtime_error(
          "expected input tensor with 4 dimensions: (batch, height, width, channels)");
    } else {
      throw std::runtime_error(
          "expected input tensor with 3 dimensions: (height, width, channels)");
    }
  }

  // raise error if tensor data is not on the device
  DLDevice dev = in_tensor->device();
  if (dev.device_type != kDLCUDA) {
    throw std::runtime_error("expected input tensor to be on a CUDA device");
  }

  auto ndim_in = in_tensor->ndim();
  HOLOSCAN_LOG_DEBUG("in_tensor.ndim() = {}", ndim_in);
  HOLOSCAN_LOG_DEBUG("in_tensor.itemsize() = {}", in_tensor->itemsize());
  HOLOSCAN_LOG_DEBUG("in_tensor.shape()[0] = {}", in_tensor->shape()[0]);
  HOLOSCAN_LOG_DEBUG("in_tensor.shape()[1] = {}", in_tensor->shape()[1]);
  HOLOSCAN_LOG_DEBUG("in_tensor.shape()[2] = {}", in_tensor->shape()[2]);
  if (ndim_in > 3) { HOLOSCAN_LOG_DEBUG("in_tensor.shape()[3] = {}", in_tensor->shape()[3]); }
  HOLOSCAN_LOG_DEBUG("in_tensor.strides()[0] = {}", in_tensor->strides()[0]);
  HOLOSCAN_LOG_DEBUG("in_tensor.strides()[1] = {}", in_tensor->strides()[1]);
  HOLOSCAN_LOG_DEBUG("in_tensor.strides()[2] = {}", in_tensor->strides()[2]);
  if (ndim_in > 3) { HOLOSCAN_LOG_DEBUG("in_tensor.strides()[3] = {}", in_tensor->strides()[3]); }

  auto itemsize = in_tensor->itemsize();

  // Sanity checks that the input tensor is 8-bit RGB data in HWC format
  DLDataType dtype = in_tensor->dtype();
  if ((dtype.code != kDLUInt) || (dtype.bits != 8)) {
    throw std::runtime_error(
        fmt::format("expected 8-bit unsigned integer data, found DLDataTypeCode: {}, bits: {}",
                    static_cast<int>(dtype.code),
                    dtype.bits));
  }

  if (in_tensor->shape()[ndim_in - 1] != 3) {
    throw std::runtime_error("expected in_tensor with 3 channels on the last dimension (RGB)");
  }

  return in_tensor;
}

// Apply basic CV-CUDA processing to the input frame

class ImageProcessingOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ImageProcessingOp);
  ImageProcessingOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("input_tensor");
    spec.output<holoscan::TensorMap>("output_tensor");
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("Converting incoming packet data to Complex Tensor data");
    holoscan::Operator::initialize();
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    // The type of `in_message` is 'holoscan::TensorMap'.
    auto maybe_tensormap = op_input.receive<holoscan::TensorMap>("input_tensor");
    if (!maybe_tensormap) { HOLOSCAN_LOG_ERROR("no input tensor found"); }
    auto in_tensor = validate_and_get_rgb_image(maybe_tensormap.value());
    const auto& [cv_in_tensor, cv_in_buffer] = holoscan_tensor_to_cvcuda_NHWC(in_tensor);

    // cv_in_tensor will be in NHWC format
    int num_batch = cv_in_tensor.shape()[0];
    int image_height = cv_in_tensor.shape()[1];
    int image_width = cv_in_tensor.shape()[2];
    int num_channels = cv_in_tensor.shape()[3];
    if (num_batch > 1) {
      HOLOSCAN_LOG_ERROR("batch_size > 1 is not supported (HolovizOp needs a single image)");
    }

    // Create an out_message entity containing a single GXF tensor corresponding to the output.
    const auto& [out_message, tensor_data_pointer] =
        create_out_message_with_tensor_like(context.context(), cv_in_tensor);

    // Create a CV-CUDA cv_out_tensor pointing to the same CUDA memory (`tensor_data_pointer`)
    // as the tensor in `out_message`.
    // Note: If we allocated the memory in CV-CUDA, it would instead be deallocated once the
    // CV-CUDA tensor goes out of scope (at the end of this compute call).

    // copy strides from cv_in_buffer
    nvcv::TensorDataStridedCuda::Buffer cv_out_buffer = cv_in_buffer;
    cv_out_buffer.basePtr = static_cast<NVCVByte*>(*tensor_data_pointer);
    nvcv::TensorDataStridedCuda out_data(cv_in_tensor.shape(), cv_in_tensor.dtype(), cv_out_buffer);
    nvcv::Tensor cv_out_tensor = nvcv::TensorWrapData(out_data);

    // apply the Flip operator
    cvcuda::Flip flipOp;
    int32_t flipCode = 0;
    flipOp(0, cv_in_tensor, cv_out_tensor, flipCode);  // Using default stream (0) here

    // Emit the tensor.
    op_output.emit(out_message, "output_tensor");
  }
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  void set_datapath(const std::string& path) { datapath = path; }

  void compose() override {
    using namespace holoscan;

    uint32_t width = 854;
    uint32_t height = 480;
    auto source = make_operator<ops::VideoStreamReplayerOp>(
        "replayer", from_config("replayer"), Arg("directory", datapath));

    auto image_processing = make_operator<ops::ImageProcessingOp>("image_processing");

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    std::shared_ptr<ops::HolovizOp> visualizer =
        make_operator<ops::HolovizOp>("holoviz",
                                      from_config("holoviz"),
                                      Arg("width") = width,
                                      Arg("height") = height,
                                      Arg("cuda_stream_pool") = cuda_stream_pool);

    // Flow definition
    add_flow(source, image_processing);
    add_flow(image_processing, visualizer, {{"output_tensor", "receivers"}});
  }

 private:
  std::string datapath = "data/endoscopy";
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path) {
  static struct option long_options[] = {{"data", required_argument, 0, 'd'}, {0, 0, 0, 0}};

  while (int c = getopt_long(argc, argv, "d", long_options, NULL)) {
    if (c == -1 || c == '?') break;

    switch (c) {
      case 'd':
        data_path = optarg;
        break;
      default:
        std::cout << "Unknown arguments returned: " << c << std::endl;
        return false;
    }
  }

  if (optind < argc) { config_name = argv[optind++]; }
  return true;
}

/** Main function */
int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Parse the arguments
  std::string data_path = "";
  std::string config_name = "";
  if (!parse_arguments(argc, argv, config_name, data_path)) { return 1; }

  if (config_name != "") {
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/cvcuda_basic.yaml";
    app->config(config_path);
  }
  if (data_path != "") app->set_datapath(data_path);

  app->run();

  return 0;
}
