/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <getopt.h>

#include "holoscan/holoscan.hpp"
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <cvcuda/OpFlip.hpp>     // cvcuda::Flip
#include <nvcv/DataType.hpp>     // nvcv::DataType
#include <nvcv/ImageFormat.hpp>  // nvcv::FMT_RGB8, etc.
#include <nvcv/Tensor.hpp>       // nvcv::Tensor
#include <nvcv/TensorData.hpp>   // nvcv::TensorDataStridedCuda
#include <nvcv/TensorShape.hpp>  // nvcv::TensorShape

namespace holoscan::ops {

// Apply basic CV-CUDA processing to the input frame

class ImageProcessingOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ImageProcessingOp);
  ImageProcessingOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("input_tensor");
    spec.output<holoscan::TensorMap>("output_tensor");

    cuda_stream_handler_.defineParams(spec);
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("Converting incoming packet data to Complex Tensor data");
    holoscan::Operator::initialize();
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    // The type of `in_message` is 'holoscan::TensorMap'.
    auto in_message = op_input.receive<holoscan::TensorMap>("input_tensor").value();

    // get the CUDA stream from the input message
    gxf_result_t stream_handler_result =
        cuda_stream_handler_.fromMessage(context.context(), in_message);
    if (stream_handler_result != GXF_SUCCESS) {
      throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
    }

    auto& [inReqs, cv_inTensor] = tensormap_2d_rgb_tensor(in_message);

    // // // tensor output of VideoReplayerOp is just an empty string ""
    // // std::shared_ptr<holoscan::Tensor> in_tensor = in_message[""];

    // std::shared_ptr<holoscan::Tensor> in_tensor;
    // auto n_tensors = in_message.size();
    // if (n_tensors != 1) {
    //   throw std::runtime_error(
    //       fmt::format("expected exactly 1 tensor in input_tensor map, found {}", n_tensors));
    // } else {
    //   // get the tensor without needing to know the key name
    //   for (auto& [key, tensor] : in_message) { in_tensor = tensor; }
    // }

    // // Sanity checks that the input tensor is C-contiguous RGB data in HWC format
    // if (in_tensor->itemsize() != sizeof(uint8_t)) {
    //   HOLOSCAN_LOG_ERROR("expected in_tensor with itemsize of 1 byte");
    // }
    // if (in_tensor->ndim() != 3) {
    //   HOLOSCAN_LOG_ERROR("expected in_tensor with 3 dimensions: (height, width, channels)");
    //   return;
    // }
    // if (in_tensor->shape()[2] != 3) {
    //   HOLOSCAN_LOG_ERROR("expected in_tensor with 3 channels (RGB)");
    // }
    // if (in_tensor->strides()[0] != (in_tensor->shape()[1] * in_tensor->shape()[2])) {
    //   HOLOSCAN_LOG_ERROR("expected C-contiguous in_tensor");
    // }

    // int batchSize = 1;
    int ImageWidth = in_tensor->shape()[1];   // 854
    int ImageHeight = in_tensor->shape()[0];  // 480
    int num_channels = 3;                     // RGB

    // // buffer with strides defined for NHWC format (For cvcuda::Flip could also just use HWC)
    // nvcv::TensorDataStridedCuda::Buffer inBuf;
    // inBuf.strides[3] = sizeof(uint8_t);
    // inBuf.strides[2] = num_channels * inBuf.strides[3];
    // inBuf.strides[1] = ImageWidth * inBuf.strides[2];
    // inBuf.strides[0] = ImageHeight * inBuf.strides[1];
    // inBuf.basePtr = static_cast<NVCVByte*>(in_tensor->data());

    // // Calculate the requirements for the RGBI uint8_t Tensor which include
    // // pitch bytes, alignment, shape  and tensor layout
    // nvcv::Tensor::Requirements inReqs =
    //     nvcv::Tensor::CalcRequirements(batchSize, {ImageWidth, ImageHeight}, nvcv::FMT_RGB8);

    // // Create a tensor buffer to store the data pointer and pitch bytes for each plane
    // nvcv::TensorDataStridedCuda inData(nvcv::TensorShape{inReqs.shape, inReqs.rank, inReqs.layout},
    //                                    nvcv::DataType{inReqs.dtype},
    //                                    inBuf);

    // // TensorWrapData allows for interoperation of external tensor representations with CVCUDA
    // // Tensor.
    // nvcv::Tensor cv_inTensor = nvcv::TensorWrapData(inData);


    // Create a GXF tensor for the output (we will then reuse the same data pointer to intialize
    // the CV-CUDA output tensor)
    nvidia::gxf::PrimitiveType element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
    int element_size = nvidia::gxf::PrimitiveTypeSize(element_type);
    nvidia::gxf::Shape shape = nvidia::gxf::Shape{ImageHeight, ImageWidth, num_channels};
    size_t nbytes = ImageHeight * ImageWidth * num_channels;
    // Create a shared pointer for the CUDA memory with a custom deleter.
    auto pointer = std::shared_ptr<void*>(new void*, [](void** pointer) {
      if (pointer != nullptr) {
        if (*pointer != nullptr) { cudaFree(*pointer); }
        delete pointer;
      }
    });
    // Allocate the CUDA memory (don't need to explicitly intialize)
    cudaError_t err = cudaMalloc(pointer.get(), nbytes);
    // Holoscan Tensor doesn't support direct memory allocation.
    // Thus, create an Entity and use GXF tensor to wrap the CUDA memory.
    auto out_message = nvidia::gxf::Entity::New(context.context());
    auto gxf_tensor = out_message.value().add<nvidia::gxf::Tensor>("image");
    gxf_tensor.value()->wrapMemory(shape,
                                   element_type,
                                   element_size,
                                   nvidia::gxf::ComputeTrivialStrides(shape, element_size),
                                   nvidia::gxf::MemoryStorageType::kDevice,
                                   *pointer,
                                   [orig_pointer = pointer](void*) mutable {
                                     orig_pointer.reset();  // decrement ref count
                                     return nvidia::gxf::Success;
                                   });

    // Create a CV-CUDA cv_outTensor pointing to the same CUDA memory as gxf_tensor.
    // Note: If we allocated the memory in CV-CUDA, it would be deallocated once the CV-CUDA
    // tensor goes out of scope (at the end of this compute call). To avoid this, we allocated a
    // GXF Tensor above and then create cv_outTensor using nvcv::TensorWrapData on its data.
    nvcv::TensorDataStridedCuda::Buffer outBuf = inBuf;
    outBuf.basePtr = static_cast<NVCVByte*>(*pointer);
    nvcv::TensorDataStridedCuda outData(nvcv::TensorShape{inReqs.shape, inReqs.rank, inReqs.layout},
                                        nvcv::DataType{inReqs.dtype},
                                        outBuf);
    nvcv::Tensor cv_outTensor = nvcv::TensorWrapData(outData);

    // apply the Flip operator
    cvcuda::Flip flipOp;
    int32_t flipCode = 0;
    // Using default stream (0) here for simplicity, but could add cuda_stream_handler_ support
    cudaStream_t stream = cuda_stream_handler_.getCudaStream(context.context());
    flipOp(stream, cv_inTensor, cv_outTensor, flipCode);  // first argument is the CUDA stream

    // Emit the tensor.
    op_output.emit(out_message.value(), "output_tensor");
  }
 private:
  CudaStreamHandler cuda_stream_handler_;
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

    // TODO: Could use a BlockMemoryPool with ImageProcessingOp
    // TODO: Could use CudaStreamPool with ImageProcessingOp
    auto image_processing = make_operator<ops::ImageProcessingOp>("image_processing");

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    std::shared_ptr<ops::HolovizOp> visualizer = make_operator<ops::HolovizOp>(
        "holoviz",
        from_config("holoviz"),
        // Modify width to account for row padding added by CV-CUDA (TODO: should not be necessary?)
        // Arg("width") = width % 32 == 0 ? width : (width / 32 + 1) * 32,
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
