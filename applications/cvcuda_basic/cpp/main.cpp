/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include "cvcuda_to_holoscan.hpp"
#include "holoscan/holoscan.hpp"
#include "holoscan_to_cvcuda.hpp"
#include "message_logger.hpp"

#include <cvcuda/OpFlip.hpp>     // cvcuda::Flip
#include <nvcv/DataType.hpp>     // nvcv::DataType
#include <nvcv/Tensor.hpp>       // nvcv::Tensor
#include <nvcv/TensorData.hpp>   // nvcv::TensorDataStridedCuda

namespace holoscan::ops {

// Apply basic CV-CUDA processing to the input frame
class ImageProcessingOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ImageProcessingOp);
  ImageProcessingOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<nvcv::Tensor>("input_tensor");
    spec.output<nvcv::Tensor>("output_tensor");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    auto cv_in_tensor = op_input.receive<nvcv::Tensor>("input_tensor").value();

    // cv_in_tensor will be in NHWC format
    int num_batch = cv_in_tensor.shape()[0];
    if (num_batch > 1) { HOLOSCAN_LOG_ERROR("batch_size > 1 is not supported"); }
    if (cv_in_tensor.rank() > 4) { HOLOSCAN_LOG_ERROR("rank > 4 is not supported"); }

    // this has two components: strides and basePtr
    nvcv::TensorDataStridedCuda::Buffer cv_out_buffer;

    // Copy the strides from the input tensor
    auto cv_in_buffer = cv_in_tensor.exportData<nvcv::TensorDataStridedCuda>();
    for (int i = 0; i < cv_in_tensor.rank(); i++) {
      cv_out_buffer.strides[i] = std::move(cv_in_buffer->stride(i));
    }

    // calculate the required device memory to allocate it
    auto reqs = nvcv::Tensor::CalcRequirements(cv_in_tensor.shape(), cv_in_tensor.dtype());
    auto nbytes = nvcv::CalcTotalSizeBytes(nvcv::Requirements{reqs.mem}.cudaMem());

    // allocate the memory in the out buffer's basePtr
    cudaMalloc(&cv_out_buffer.basePtr, nbytes);
    nvcv::TensorDataStridedCuda out_data(cv_in_tensor.shape(), cv_in_tensor.dtype(), cv_out_buffer);

    HOLOSCAN_LOG_DEBUG("cv_out_buffer created");

    // output tensor is now properly formatted and allocated
    nvcv::Tensor cv_out_tensor = nvcv::TensorWrapData(out_data);

    // apply the Flip operator
    cvcuda::Flip flipOp;
    int32_t flipCode = 0;
    flipOp(0, cv_in_tensor, cv_out_tensor, flipCode);  // Using default stream (0) here
    HOLOSCAN_LOG_DEBUG("Flipped done");

    // Emit the tensor.
    op_output.emit(cv_out_tensor, "output_tensor");
  }
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  void set_datapath(const std::string& path) { datapath = path; }

  void compose() override {
    using namespace holoscan;

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    uint32_t width = 854;
    uint32_t height = 480;
    auto source = make_operator<ops::VideoStreamReplayerOp>(
        "replayer", from_config("replayer"), Arg("directory", datapath));

    auto holoscan_to_cvcuda = make_operator<ops::HoloscanToCvCuda>("holoscan_to_cvcuda");

    auto image_processing = make_operator<ops::ImageProcessingOp>("image_processing");

    auto cvcuda_to_holoscan = make_operator<ops::CvCudaToHoloscan>("cvcuda_to_holoscan");

    std::shared_ptr<ops::HolovizOp> visualizer1 =
        make_operator<ops::HolovizOp>("holoviz1",
                                      from_config("holoviz1"),
                                      Arg("window_title") = std::string("Original Stream"),
                                      Arg("width") = width,
                                      Arg("height") = height,
                                      Arg("cuda_stream_pool") = cuda_stream_pool);

    std::shared_ptr<ops::HolovizOp> visualizer2 =
        make_operator<ops::HolovizOp>("holoviz2",
                                      from_config("holoviz2"),
                                      Arg("window_title") = std::string("Flipped Stream"),
                                      Arg("width") = width,
                                      Arg("height") = height,
                                      Arg("cuda_stream_pool") = cuda_stream_pool);

    // Flow definition
    // add_flow(source, visualizer1, {{"output", "receivers"}}); // optional to watch the original
    // stream
    add_flow(source, holoscan_to_cvcuda);
    add_flow(holoscan_to_cvcuda, image_processing);
    add_flow(image_processing, cvcuda_to_holoscan);
    add_flow(cvcuda_to_holoscan, visualizer2, {{"output", "receivers"}});

    // configure this application to use Holohub's example MessageLogger
    auto message_logger = make_resource<data_loggers::MessageLogger>("message_logger");
    add_data_logger(message_logger);
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
