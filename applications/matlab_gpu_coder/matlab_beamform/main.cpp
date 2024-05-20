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

#include "matlab_utils.h"
#include "matlab_beamform.h"
#include "matlab_beamform_terminate.h"
#include "matlab_beamform_types.h"

namespace holoscan::ops {
class MatlabBeamformOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MatlabBeamformOp)

  MatlabBeamformOp() = default;

  // Image field-of-view and windowing parameters
  const int depth = 6494;
  const int length = 10000;
  const int length_window = 4000;  // window size
  const int step_window = 40;  // how much to move the window at each update

  // Beamforming parameters
  void init_params(struct0_T *p) {
      p->c = 1540;
      p->fc = 3000000;
      p->rangeRes = 0.003;
      p->alongTrackRes = 0.003;
      p->Bw = 2.566666666666667e+05;
      p->prf = 500;
      p->speed = 0.100;
      p->aperture = 0.004;
      p->Tpd = 3.000000000000000e-06;
      p->fs = 10000000;
  }

  void setup(OperatorSpec& spec) override {
    auto& out_tensor = spec.output<gxf::Entity>("tensor");
    spec.param(out_, "out", "Output", "Output channel.", &out_tensor);
    spec.param(out_tensor_name_,
             "out_tensor_name",
             "OutputTensorName",
             "Name of the output tensor.",
             std::string(""));
    spec.param(path_data_, "path_data", "PathData", "Path to binary data on disk.",
               std::string(""));
    spec.param(allocator_, "allocator", "Allocator", "Output Allocator");
    cuda_stream_handler_.define_params(spec);
  }

  void start() {
    // Init beamforming parameters
    init_params(&params_);

    // Define output shape
    shape_.push_back(depth);
    shape_.push_back(length_window);
    shape_.push_back(3);

    // Open binary file reader
    std::ifstream is_(path_data_.get(), std::ios::binary);
    if (!is_) { throw std::runtime_error("Error opening binary file"); }

    // Read binary data to CUDA buffers
    cudaMalloc(&fast_time_, sizeof(float) * depth);
    disk2cuda_fbuffer(is_, fast_time_, depth);
    cudaMalloc(&x_axis_, sizeof(float) * length);
    disk2cuda_fbuffer(is_, x_axis_, length);
    cudaMalloc(&rdata_tmp_, sizeof(float) * depth * length);
    disk2cuda_fbuffer(is_, rdata_tmp_, depth * length);
    cudaMalloc(&idata_tmp_, sizeof(float) * depth * length);
    disk2cuda_fbuffer(is_, idata_tmp_, depth * length);

    // Allocate window buffers
    cudaMalloc(&x_axis_window_, sizeof(float) * length_window);
    cudaMalloc(&data_window_, sizeof(creal32_T) * depth * length_window);

    // Close file reader
    is_.close();
  }

  void stop() {
    // Free CUDA memory
    cudaFree(fast_time_);
    cudaFree(x_axis_);
    cudaFree(data_);
    cudaFree(x_axis_window_);
    cudaFree(data_window_);

    // Terminate MATLAB lib
    matlab_beamform_terminate();
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    // Get CUDA stream
    auto cuda_stream = cuda_stream_handler_.get_cuda_stream(context.context());

    if (complex_data_populated_ == false) {
      // Once: Populate complex MATLAB struct and free temporary CUDA buffers
      cudaMalloc(&data_, sizeof(creal32_T) * depth * length);
      cuda_populate_complex(rdata_tmp_, idata_tmp_, (void*)data_, depth * length, cuda_stream);
      cudaStreamSynchronize(cuda_stream);
      cudaFree(rdata_tmp_);
      cudaFree(idata_tmp_);
      complex_data_populated_ = true;
    }

    // Create output message
    auto out_message = nvidia::gxf::Entity::New(context.context());

    // Get allocator
    auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

    // Allocate output buffer on the device.
    auto out_tensor = out_message.value().add<nvidia::gxf::Tensor>(out_tensor_name_.get().c_str());
    if (!out_tensor) { throw std::runtime_error("Failed to allocate output tensor"); }
    out_tensor.value()->reshape<uint8_t>(
        nvidia::gxf::Shape{shape_}, nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
    if (!out_tensor.value()->pointer()) {
      throw std::runtime_error("Failed to allocate output tensor buffer.");
    }
    // Get output data
    nvidia::gxf::Expected<uint8_t*> out_tensor_data = out_tensor.value()->data<uint8_t>();
    if (!out_tensor_data) { throw std::runtime_error("Failed to get out tensor data!"); }

    // Get windows of data_ and x_axis_
    cudaMemcpy(
      data_window_,
      data_ + refresh_counter_ * step_window * depth,
      depth * length_window * sizeof(creal32_T),
      cudaMemcpyDeviceToDevice);

    cudaMemcpy(
      x_axis_window_,
      x_axis_ + refresh_counter_ * step_window,
      length_window * sizeof(float),
      cudaMemcpyDeviceToDevice);

    refresh_counter_ += 1;
    if ((refresh_counter_ * step_window * depth) >= (length * depth - length_window * depth)) {
      refresh_counter_ = 0;}

    // Allocate temp output buffer on device
    auto tmp_tensor = make_tensor(shape_, nvidia::gxf::PrimitiveType::kUnsigned8, sizeof(uint8_t),
                                  allocator.value());
    // Get temp data
    nvidia::gxf::Expected<uint8_t*> tmp_tensor_data = tmp_tensor->data<uint8_t>();
    if (!tmp_tensor_data) { throw std::runtime_error("Failed to get temporary tensor data!"); }

    // Call MATLAB CUDA function to do beamforming
    matlab_beamform(data_window_, &params_, x_axis_window_, fast_time_, tmp_tensor_data.value());
    cudaDeviceSynchronize();

    // Convert output from column- to row-major ordering
    cuda_hard_transpose<uint8_t>(tmp_tensor_data.value(), out_tensor_data.value(), shape_,
                                 cuda_stream, Flip::Do);
    cudaStreamSynchronize(cuda_stream);
    delete tmp_tensor;

    // Create output message
    auto result = gxf::Entity(std::move(out_message.value()));
    op_output.emit(result);
  }

  void disk2cuda_fbuffer(std::ifstream& is, float* dbuffer, int numel) {
    // Read binary data from disk and copy to device buffer
    float* temp = new float[numel];
    is.read(reinterpret_cast<char*>(temp), sizeof(float) * numel);
     if (is.fail()) {
        throw std::runtime_error("Error reading binary file");
    } else if (is.eof()) {
      throw std::runtime_error("Reached end of file unexpectedly");
    }
    cudaMemcpy(dbuffer, temp, sizeof(float) * numel, cudaMemcpyHostToDevice);
    delete[] temp;
  }

 private:
  Parameter<holoscan::IOSpec*> out_;
  Parameter<std::string> path_data_;
  Parameter<std::string> out_tensor_name_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  CudaStreamHandler cuda_stream_handler_;
  float* fast_time_;
  float* x_axis_;
  creal32_T* data_;
  float* x_axis_window_;
  creal32_T* data_window_;
  struct0_T params_;
  bool complex_data_populated_ = false;
  int refresh_counter_ = 0;
  std::ifstream is_;
  float* rdata_tmp_;
  float* idata_tmp_;
  std::vector<int32_t> shape_;
};

}  // namespace holoscan::ops

class MatlabBeamformApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
      make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    // Define operators and configure using yaml configuration
    auto matlab = make_operator<ops::MatlabBeamformOp>(
      "matlab",
      from_config("matlab"),
      Arg("allocator") = make_resource<BlockMemoryPool>(
        "pool", 1, 3 * 6496 * 4000, 2),
      Arg("cuda_stream_pool") = cuda_stream_pool);
    auto holoviz = make_operator<ops::HolovizOp>(
      "holoviz",
      from_config("holoviz"),
      Arg("allocator") = make_resource<UnboundedAllocator>("pool"),
      Arg("cuda_stream_pool") = cuda_stream_pool);

    // Define the workflow
    add_flow(matlab, holoviz, {{"tensor", "receivers"}});
  }
};

int main(int argc, char** argv) {
  // Get the yaml configuration file
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path /= std::filesystem::path("matlab_beamform.yaml");
  if (argc >= 2) { config_path = argv[1]; }

  auto app = holoscan::make_application<MatlabBeamformApp>();
  app->config(config_path);
  app->run();

  return 0;
}
