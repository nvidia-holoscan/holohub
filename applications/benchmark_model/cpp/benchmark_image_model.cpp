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

#include <getopt.h>

#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/segmentation_postprocessor/segmentation_postprocessor.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include "holoscan/holoscan.hpp"

#include <opencv2/opencv.hpp>

#include "cvcuda_utils.hpp"
#include "gxf_utils.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <nvcv/NvDecoder.h>
#include <cvcuda/OpConvertTo.hpp>
#include <cvcuda/OpCustomCrop.hpp>
#include <cvcuda/OpNormalize.hpp>
#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpResize.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#define N_CLASSES 1000

namespace holoscan::ops {

/**
 * @brief This Operator takes an input tensor and does nothing with it. It works as a sink
 * without any function.
 *
 */
class SinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SinkOp)

  SinkOp() = default;

  void setup(OperatorSpec& spec) { spec.input<std::any>("in"); }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
    // auto value = op_input.receive<std::any>("in");
    auto value = op_input.receive<holoscan::TensorMap>("in");
    auto maybe_tensormap = value.value();
    // iterate key value of this map, printing all keys
    for (auto& [key, tensor] : maybe_tensormap) { std::cout << "Key: " << key << std::endl; }
  }
};

class ReadImagesOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ReadImagesOp)

  ReadImagesOp() = default;

  void setup(OperatorSpec& spec) { spec.output<holoscan::TensorMap>("image_tensor"); }

  void start() override {
    // Check if the directory exists
    if (!std::filesystem::exists(image_directory)) {
      std::cerr << "Image directory " << image_directory << " does not exist." << std::endl;
      return;
    }
    for (const auto& entry : std::filesystem::directory_iterator(image_directory)) {
      files.push_back(entry.path());
    }
    cudaStreamCreate(&cudastream);
  }

  void stop() override { cudaStreamDestroy(cudastream); }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) {
    using namespace cv;
    // list all the files in the directory
    std::string current_file = files[count++];

    HOLOSCAN_LOG_INFO("Current file {}", current_file);

    Mat image = imread(current_file, IMREAD_COLOR);
    if (image.empty()) {
      std::cout << "Could not open or find the image" << std::endl;
      return;
    }

    // Convert the image to RGB8 array
    // OpenCV stores images in BGR format, so we need to convert it to RGB
    Mat rgbImage;
    cvtColor(image, rgbImage, COLOR_BGR2RGB);

    // Access the RGB8 array
    uchar* rgbArray = rgbImage.data;
    int width = rgbImage.cols;
    int height = rgbImage.rows;

    uint32_t batchSize = 1;
    int totalImages = 1;
    int maxImageWidth = width;
    int maxImageHeight = height;
    int maxChannels = 3;

    nvcv::TensorDataStridedCuda::Buffer inBuf;
    inBuf.strides[3] = sizeof(uint8_t);
    inBuf.strides[2] = maxChannels * inBuf.strides[3];
    inBuf.strides[1] = maxImageWidth * inBuf.strides[2];
    inBuf.strides[0] = maxImageHeight * inBuf.strides[1];
    cudaMallocAsync(&inBuf.basePtr, batchSize * inBuf.strides[0], cudastream);

    nvcv::Tensor::Requirements inReqs =
        nvcv::Tensor::CalcRequirements(batchSize, {width, height}, nvcv::FMT_RGB8);

    nvcv::TensorDataStridedCuda inData(nvcv::TensorShape{inReqs.shape, inReqs.rank, inReqs.layout},
                                       nvcv::DataType{inReqs.dtype},
                                       inBuf);

    nvcv::Tensor inTensor = TensorWrapData(inData);

    // NvJpeg is used to load the images to create a batched input device buffer.
    uint8_t* gpuInput = reinterpret_cast<uint8_t*>(inBuf.basePtr);
    cudaMemcpy(inBuf.basePtr, rgbArray, width * height * 3 * sizeof(uchar), cudaMemcpyHostToDevice);

    // Apply preprocessing
    int resize = 256;

    nvcv::Tensor::Requirements outReqs =
        nvcv::Tensor::CalcRequirements(batchSize, {resize, resize}, nvcv::FMT_RGB8);

    nvcv::TensorDataStridedCuda::Buffer outBuf;
    std::copy(outReqs.strides, outReqs.strides + NVCV_TENSOR_MAX_RANK, outBuf.strides);

    int64_t outLayerSize = nvcv::CalcTotalSizeBytes(nvcv::Requirements{outReqs.mem}.cudaMem());
    cudaMalloc(&outBuf.basePtr, outLayerSize);

    nvcv::TensorDataStridedCuda outData(
        nvcv::TensorShape{outReqs.shape, outReqs.rank, outReqs.layout},
        nvcv::DataType{outReqs.dtype},
        outBuf);
    nvcv::Tensor outTensor = TensorWrapData(outData);

    // Apply resize
    cvcuda::Resize resizeOp;
    HOLOSCAN_LOG_INFO("Will resize");
    resizeOp(cudastream, inTensor, outTensor, NVCV_INTERP_NEAREST);

    HOLOSCAN_LOG_INFO("Resize succesful");

    int crop_size = 224;
    int xy_start = (resize - crop_size) / 2;

    NVCVRectI crpRect = {xy_start, xy_start, crop_size, crop_size};

    nvcv::Tensor cropTensor(batchSize, {crop_size, crop_size}, nvcv::FMT_RGB8);
    cvcuda::CustomCrop cropOp;
    cropOp(cudastream, outTensor, cropTensor, crpRect);

    // Convert to FP32
    nvcv::Tensor floatTensor(batchSize, {resize, resize}, nvcv::FMT_RGBf32);
    cvcuda::ConvertTo convertOp;
    convertOp(cudastream, cropTensor, floatTensor, 1.0f / 255.f, 0.0f);

#if 0
    // Normalize the pixel values
    nvcv::Tensor::Requirements reqsScale =
        nvcv::Tensor::CalcRequirements(1, {1, 1}, nvcv::FMT_RGBf32);
    int64_t scaleBufferSize = CalcTotalSizeBytes(nvcv::Requirements{reqsScale.mem}.cudaMem());
    nvcv::TensorDataStridedCuda::Buffer bufScale;
    std::copy(reqsScale.strides, reqsScale.strides + NVCV_TENSOR_MAX_RANK, bufScale.strides);
    cudaMalloc(&bufScale.basePtr, scaleBufferSize);
    nvcv::TensorDataStridedCuda scaleIn(
        nvcv::TensorShape{reqsScale.shape, reqsScale.rank, reqsScale.layout},
        nvcv::DataType{reqsScale.dtype},
        bufScale);
    nvcv::Tensor stddevTensor = TensorWrapData(scaleIn);

    nvcv::TensorDataStridedCuda::Buffer bufBase;
    nvcv::Tensor::Requirements reqsBase =
        nvcv::Tensor::CalcRequirements(1, {1, 1}, nvcv::FMT_RGBf32);
    int64_t baseBufferSize = CalcTotalSizeBytes(nvcv::Requirements{reqsBase.mem}.cudaMem());
    std::copy(reqsBase.strides, reqsBase.strides + NVCV_TENSOR_MAX_RANK, bufBase.strides);
    cudaMalloc(&bufBase.basePtr, baseBufferSize);
    nvcv::TensorDataStridedCuda baseIn(
        nvcv::TensorShape{reqsBase.shape, reqsBase.rank, reqsBase.layout},
        nvcv::DataType{reqsBase.dtype},
        bufBase);
    nvcv::Tensor meanTensor = TensorWrapData(baseIn);

    float stddev[3] = {0.229, 0.224, 0.225};
    float mean[3] = {0.485f, 0.456f, 0.406f};
    auto meanData = meanTensor.exportData<nvcv::TensorDataStridedCuda>();
    auto stddevData = stddevTensor.exportData<nvcv::TensorDataStridedCuda>();

    uint32_t flags = CVCUDA_NORMALIZE_SCALE_IS_STDDEV;
    cudaMemcpyAsync(
        stddevData->basePtr(), stddev, 3 * sizeof(float), cudaMemcpyHostToDevice, cudastream);
    cudaMemcpyAsync(
        meanData->basePtr(), mean, 3 * sizeof(float), cudaMemcpyHostToDevice, cudastream);

    nvcv::Tensor normTensor(batchSize, {crop_size, crop_size}, nvcv::FMT_RGBf32);

    cvcuda::Normalize normalizeOp;
    normalizeOp(
        cudastream, cropTensor, meanTensor, stddevTensor, normTensor, 1.0f, 0.0f, 0.0f, flags);
#endif

    nvcv::Tensor::Requirements reqsOutputLayer =
        nvcv::Tensor::CalcRequirements(batchSize, {crop_size, crop_size}, nvcv::FMT_RGBf32p);

    nvcv::Tensor outputLayerTensor(
        nvcv::TensorShape{reqsOutputLayer.shape, reqsOutputLayer.rank, reqsOutputLayer.layout},
        nvcv::DataType{reqsOutputLayer.dtype});

    const auto& [out_message, tensor_data_pointer] =
        create_out_message_with_tensor_like(context.context(), outputLayerTensor);

    nvcv::TensorDataStridedCuda::Buffer cv_out_buffer;
    std::copy(reqsOutputLayer.strides,
              reqsOutputLayer.strides + NVCV_TENSOR_MAX_RANK,
              cv_out_buffer.strides);
    cv_out_buffer.basePtr = static_cast<NVCVByte*>(*tensor_data_pointer);
    nvcv::TensorDataStridedCuda out_data(
        nvcv::TensorShape{reqsOutputLayer.shape, reqsOutputLayer.rank, reqsOutputLayer.layout},
        nvcv::DataType{reqsOutputLayer.dtype},
        cv_out_buffer);
    nvcv::Tensor cv_out_tensor = nvcv::TensorWrapData(out_data);

    cvcuda::Reformat reformatOp;
    // reformatOp(cudastream, normTensor, cv_out_tensor);
    // reformatOp(cudastream, cropTensor, cv_out_tensor);
    reformatOp(cudastream, floatTensor, cv_out_tensor);

    std::cout << "cv_out_tensor shape: " << cv_out_tensor.shape() << std::endl;

    cudaStreamSynchronize(cudastream);

    op_output.emit(out_message, "image_tensor");
  }

 private:
  std::string image_directory = "../images/";
  std::vector<std::string> files;
  int count = 0;
  // read the image from the current_file using cvcuda
  cudaStream_t cudastream;
};

class PrintTextOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PrintTextOp)

  PrintTextOp() = default;

  void start() override {
    std::ifstream file(image_classes_file);
    std::string line;
    while (std::getline(file, line)) { labels.push_back(line); }
  }

  void setup(OperatorSpec& spec) { spec.input<holoscan::TensorMap>("in"); }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
    auto value = op_input.receive<holoscan::TensorMap>("in");
    auto maybe_tensor = value.value().at("output");

    auto predictions = maybe_tensor.get()->data();
    auto bytes = maybe_tensor.get()->nbytes();

    std::cout << "bytes: " << bytes << std::endl;
    // allocate host memory
    float* host_predictions = (float*)malloc(bytes);
    cudaMemcpy(host_predictions, predictions, bytes, cudaMemcpyDeviceToHost);

    // sync default stream
    cudaStreamSynchronize(0);

    HOLOSCAN_LOG_DEBUG("Memcpy Successful");

    // convert host_predictions[0]...[N_CLASSES] to a vector
    std::vector<float> predictions_vec(host_predictions, host_predictions + N_CLASSES);

    HOLOSCAN_LOG_DEBUG("Vector created");

    // sort the vector and return the top 5 indices
    std::vector<int> indices(N_CLASSES);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(), indices.begin() + 5, indices.end(), [&predictions_vec](int i1, int i2) {
          return predictions_vec[i1] > predictions_vec[i2];
        });

    HOLOSCAN_LOG_INFO("Top 5 predictions: ");
    for (int i = 0; i < 5; i++) {
      // .2f decimal points
      std::cout << "Class " << indices[i] << ": " << labels[indices[i]]
                << " - Probability: " << std::fixed << std::setprecision(2)
                << predictions_vec[indices[i]] * 100 << "%" << std::endl;
    }
  }

 private:
  std::string image_classes_file = "../data/imagenet_classes.txt";
  std::vector<std::string> labels;
};
}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  App() { holoscan::Application(); }

  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Resource> pool_resource = make_resource<UnboundedAllocator>("pool");
    // Create the operators
    auto read_images_op = make_operator<holoscan::ops::ReadImagesOp>(
        "read_images_op", make_condition<holoscan::CountCondition>(3));
    auto sink_op = make_operator<holoscan::ops::SinkOp>("sink_op");

    ops::InferenceOp::DataMap model_path_map;
    ops::InferenceOp::DataVecMap pre_processor_map;
    ops::InferenceOp::DataVecMap inference_map;

    std::string model_index_str = "own_model";

    model_path_map.insert(model_index_str, "../data/resnet50/model.onnx");
    pre_processor_map.insert(model_index_str, {"image"});

    std::string output_name = "output";
    inference_map.insert(model_index_str, {output_name});

    auto inference = make_operator<ops::InferenceOp>("inference",
                                                     from_config("inference"),
                                                     Arg("allocator") = pool_resource,
                                                     Arg("model_path_map", model_path_map),
                                                     Arg("pre_processor_map", pre_processor_map),
                                                     Arg("inference_map", inference_map));

    // Connect the operators
    // add_flow(read_images_op, sink_op);
    add_flow(read_images_op, inference, {{"", "receivers"}});

    auto print_text_op = make_operator<ops::PrintTextOp>("print_text_op");
    add_flow(inference, print_text_op);
  }

 private:
  int num_inferences = 1;
  bool only_inference = false, inference_postprocessing = false;
  std::string datapath, model_name, video_name;
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();
  std::string config_name = "";
  if (config_name != "") {
    // Check if config_name is a valid path
    if (!std::filesystem::exists(config_name)) {
      std::cerr << "Config file " << config_name << " does not exist." << std::endl;
      return 0;
    }
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/benchmark_model.yaml";
    app->config(config_path);
  }

  app->run();
  return 0;
}