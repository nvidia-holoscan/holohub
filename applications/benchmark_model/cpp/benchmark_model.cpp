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

#include <cstdint>
#include <cvcuda/OpConvertTo.hpp>
#include <cvcuda/OpCustomCrop.hpp>
#include <cvcuda/OpCvtColor.hpp>
#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpResize.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/segmentation_postprocessor/segmentation_postprocessor.hpp>
#include <holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include "cvcuda_utils.hpp"
#include "holoscan/holoscan.hpp"

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
    auto value = op_input.receive<std::any>("in");
  }
};

class HoloscanToCvCudaTensorOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(HoloscanToCvCudaTensorOp)

  HoloscanToCvCudaTensorOp() = default;

  void setup(OperatorSpec& spec) {
    spec.input<gxf::Entity>("input");
    spec.output<nvcv::Tensor>("output");
  }

  static nvcv::Tensor to_cvcuda_NHWC_tensor(std::shared_ptr<holoscan::Tensor> in_tensor) {
    // The output tensor will always be created in NHWC format even if no batch dimension existed
    // on the GXF tensor.
    int ndim = in_tensor->ndim();
    int batch_size, image_height, image_width, num_channels;
    auto in_shape = in_tensor->shape();
    if (ndim == 4) {
      batch_size = in_shape[0];
      image_height = in_shape[1];
      image_width = in_shape[2];
      num_channels = in_shape[3];
    } else if (ndim == 3) {
      batch_size = 1;
      image_height = in_shape[0];
      image_width = in_shape[1];
      num_channels = in_shape[2];
    } else if (ndim == 2) {
      batch_size = 1;
      image_height = in_shape[0];
      image_width = in_shape[1];
      num_channels = 1;
    } else {
      throw std::runtime_error(
          "expected a tensor with (height, width) or (height, width, channels) or "
          "(batch, height, width, channels) dimensions");
    }

    // buffer with strides defined for NHWC format (For cvcuda::Flip could also just use HWC)
    auto in_buffer = holoscan::nhwc_buffer_from_holoscan_tensor(in_tensor);
    nvcv::TensorShape cv_tensor_shape{{batch_size, image_height, image_width, num_channels},
                                      NVCV_TENSOR_NHWC};
    nvcv::DataType cv_dtype = dldatatype_to_nvcvdatatype(in_tensor->dtype());

    // Create a tensor buffer to store the data pointer and pitch bytes for each plane
    nvcv::TensorDataStridedCuda in_data(cv_tensor_shape, cv_dtype, in_buffer);

    // TensorWrapData allows for interoperation of external tensor representations with CVCUDA
    // Tensor.
    nvcv::Tensor cv_in_tensor = nvcv::TensorWrapData(in_data);
    return cv_in_tensor;
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
    auto input_message = op_input.receive<gxf::Entity>("input").value();

    // auto tensor = input_message.get<holoscan::Tensor>();
    auto holoscan_tensor = input_message.get<
        holoscan::Tensor>();  // holoscan::gxf::GXFTensor(*(tensor.value().get())).as_tensor();

    DLDevice dev = holoscan_tensor->device();
    if (dev.device_type != kDLCUDA) {
      throw std::runtime_error("expected input tensor to be on a CUDA device");
    }

    auto ndim_in = holoscan_tensor->ndim();
    HOLOSCAN_LOG_INFO("in_tensor.ndim() = {}", ndim_in);
    for (int i = 0; i < ndim_in; i++) {
      HOLOSCAN_LOG_INFO("in_tensor.shape()[{}] = {}", i, holoscan_tensor->shape()[i]);
    }
    DLDataType dtype = holoscan_tensor->dtype();
    HOLOSCAN_LOG_INFO("in_tensor.dtype().code = {}, dtype().bits: {}", dtype.code, dtype.bits);
    if (holoscan_tensor->shape()[ndim_in - 1] != 3) {
      throw std::runtime_error(
          "expected holoscan_tensor with 3 channels on the last dimension (RGB)");
    }

    // nvcv::Tensor cv_tensor;
    const auto& cv_tensor = to_cvcuda_NHWC_tensor(holoscan_tensor);
    op_output.emit(cv_tensor, "output");
  }
};

class VideoBufferToCvCuda : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VideoBufferToCvCuda)

  VideoBufferToCvCuda() = default;

  void setup(OperatorSpec& spec) {
    spec.input<gxf::Entity>("input");
    spec.output<nvcv::Tensor>("output");
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) {
    auto in_message = op_input.receive<gxf::Entity>("input").value();
    auto video_buffer = holoscan::gxf::get_videobuffer(in_message);
    auto width = video_buffer.get()->video_frame_info().width;
    auto height = video_buffer.get()->video_frame_info().height;

    HOLOSCAN_LOG_INFO("new image going to be declared");
    // nvidia::gxf::Tensor;
    // nvcv::Tensor out_tensor = nvcv::TensorWrapData(outData);

    // nvcv::Image image({width, height}, nvcv::FMT_RGBA8);
    // nvcv::Image dummy_image({width, height}, nvcv::FMT_RGB8);
    // HOLOSCAN_LOG_INFO("new image declared");

    // void* image_data;
    // cudaMallocAsync(&image_data, video_buffer.get()->size(), 0);
    // cudaMemcpy(image_data,
    //            video_buffer.get()->pointer(),
    //            video_buffer.get()->size(),
    //            cudaMemcpyHostToDevice);
    // image.setUserPointer(image_data);

    // nvcv::ImageBatchVarShape image_batch_in(1), image_batch_out(1);
    // image_batch_in.pushBack(image);
    // image_batch_out.pushBack(dummy_image);

    // HOLOSCAN_LOG_INFO("in num images: {}, out num images: {}",
    //                   image_batch_in.numImages(),
    //                   image_batch_out.numImages());

    // cvcuda::CvtColor cvtcolorOp;

    // nvcv::ImageBatch image_batch_in_fixed = image_batch_in, image_batch_out_fixed =
    // image_batch_out; cvtcolorOp(0, image_batch_in_fixed, image_batch_out_fixed,
    // NVCV_COLOR_YUV2RGB);

    // HOLOSCAN_LOG_INFO("setuserpointer");
    // nvcv::ImageBatchVarShape out_image_batch = std::move(image_batch_out_fixed);
    // auto out_image = out_image_batch[0];
    // nvcv::Tensor in_tensor = nvcv::TensorWrapImage(image);
    // nvcv::Tensor out_tensor = nvcv::TensorWrapImage(out_image);
    // HOLOSCAN_LOG_INFO("wrap succeeded");
    // op_output.emit(out_tensor, "output");
  }
};

class CvCudaToHoloscan : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CvCudaToHoloscan)

  CvCudaToHoloscan() = default;

  void setup(OperatorSpec& spec) {
    spec.input<nvcv::Tensor>("input");
    spec.output<holoscan::TensorMap>("output");
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) {
    auto cv_in_tensor = op_input.receive<nvcv::Tensor>("input").value();

    HOLOSCAN_LOG_INFO("cv_in_tensor retrieved");

    nvcv::Tensor::Requirements in_tensor_req =
        nvcv::Tensor::CalcRequirements(1, {224, 224}, nvcv::FMT_RGBf32p);

    nvcv::Tensor out_tensor_like(
        nvcv::TensorShape{in_tensor_req.shape, in_tensor_req.rank, in_tensor_req.layout},
        nvcv::DataType{in_tensor_req.dtype});

    HOLOSCAN_LOG_INFO("before create_out_message_with_tensor_like");
    const auto& [out_message, tensor_data_pointer] =
        create_out_message_with_tensor_like(context.context(), out_tensor_like);
    HOLOSCAN_LOG_INFO("create_out_message_with_tensor_like success");

    // auto cv_in_strided = cv_in_tensor.exportData<nvcv::TensorDataStridedCuda>();

    // int64_t tot_size = nvcv::CalcTotalSizeBytes(nvcv::Requirements{in_tensor_req.mem}.cudaMem());
    // cudaMemcpy(*tensor_data_pointer, cv_in_strided->basePtr(), tot_size,
    // cudaMemcpyDeviceToDevice); HOLOSCAN_LOG_INFO("Before cudaMemcpy"); cudaStreamSynchronize(0);

    nvcv::TensorDataStridedCuda::Buffer cv_out_buffer;
    std::copy(
        in_tensor_req.strides, in_tensor_req.strides + NVCV_TENSOR_MAX_RANK, cv_out_buffer.strides);

    cv_out_buffer.basePtr = static_cast<NVCVByte*>(*tensor_data_pointer);
    nvcv::TensorDataStridedCuda out_data(
        nvcv::TensorShape{in_tensor_req.shape, in_tensor_req.rank, in_tensor_req.layout},
        cv_in_tensor.dtype(),
        cv_out_buffer);

    nvcv::Tensor cv_out_tensor = nvcv::TensorWrapData(out_data);

    cvcuda::Reformat reformatOp;
    reformatOp(0, cv_in_tensor, cv_out_tensor);

    std::cout << "cv_out_tensor shape: " << cv_out_tensor.shape() << std::endl;

    HOLOSCAN_LOG_INFO("After last");

    cudaStreamSynchronize(0);

    op_output.emit(out_message, "output");
  }
};

class PreprocessImage : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PreprocessImage)

  PreprocessImage() = default;

  void setup(OperatorSpec& spec) {
    spec.input<nvcv::Tensor>("input");
    spec.output<nvcv::Tensor>("output");
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) {
    auto in_tensor = op_input.receive<nvcv::Tensor>("input").value();

    int resize = 256;

    nvcv::Tensor::Requirements outReqs =
        nvcv::Tensor::CalcRequirements(1, {resize, resize}, nvcv::FMT_RGB8);

    nvcv::TensorDataStridedCuda::Buffer outBuf;
    std::copy(outReqs.strides, outReqs.strides + NVCV_TENSOR_MAX_RANK, outBuf.strides);

    int64_t outLayerSize = nvcv::CalcTotalSizeBytes(nvcv::Requirements{outReqs.mem}.cudaMem());
    cudaMalloc(&outBuf.basePtr, outLayerSize);

    nvcv::TensorDataStridedCuda outData(
        nvcv::TensorShape{outReqs.shape, outReqs.rank, outReqs.layout},
        nvcv::DataType{outReqs.dtype},
        outBuf);
    nvcv::Tensor resized_tensor = TensorWrapData(outData);

    cvcuda::Resize resizeOp;
    HOLOSCAN_LOG_INFO("Will resize");
    resizeOp(0, in_tensor, resized_tensor, NVCV_INTERP_NEAREST);

    int crop_size = 224;
    int xy_start = (resize - crop_size) / 2;

    NVCVRectI crpRect = {xy_start, xy_start, crop_size, crop_size};

    nvcv::Tensor cropTensor(1, {crop_size, crop_size}, nvcv::FMT_RGB8);
    cvcuda::CustomCrop cropOp;
    cropOp(0, resized_tensor, cropTensor, crpRect);
    // cropOp(0, resized_tensor, cropTensor, crpRect);

    // Convert to FP32
    nvcv::Tensor floatTensor(1, {crop_size, crop_size}, nvcv::FMT_RGBf32);
    cvcuda::ConvertTo convertOp;
    convertOp(0, cropTensor, floatTensor, 1.0f / 255.f, 0.0f);

    cudaStreamSynchronize(0);

    // op_output.emit(cropTensor, "output");
    op_output.emit(floatTensor, "output");
  }
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

    // convert host_predictions[0]...[num_classes] to a vector
    std::vector<float> predictions_vec(host_predictions, host_predictions + num_classes);

    HOLOSCAN_LOG_DEBUG("Vector created");

    // sort the vector and return the top 5 indices
    std::vector<int> indices(num_classes);
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
  int num_classes = 1000;
  std::string image_classes_file = "/workspace/holohub/data/imagenet_classes.txt";
  std::vector<std::string> labels;
};
}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  App() = default;
  App(const std::string& datapath, const std::string& model_name, const std::string& video_name,
      int num_inferences, bool only_inference, bool inference_postprocessing)
      : datapath(datapath),
        model_name(model_name),
        video_name(video_name),
        num_inferences(num_inferences),
        only_inference(only_inference),
        inference_postprocessing(inference_postprocessing) {
    holoscan::Application();
    if (!std::filesystem::exists(datapath)) {
      std::cerr << "Data path " << datapath << " does not exist." << std::endl;
      exit(1);
    }

    std::string model_path =
        datapath.back() == '/' ? (datapath + model_name) : (datapath + "/" + model_name);
    if (!std::filesystem::exists(model_path)) {
      std::cerr << "Model path " << model_path << " does not exist." << std::endl;
      exit(1);
    }
  }

  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Resource> pool_resource = make_resource<UnboundedAllocator>("pool");
    auto source = make_operator<ops::V4L2VideoCaptureOp>(
        "source", from_config("source"), Arg("allocator") = pool_resource);

    auto preprocessor = make_operator<ops::FormatConverterOp>(
        "preprocessor", from_config("preprocessor"), Arg("pool") = pool_resource);

    auto holoscantocvcuda = make_operator<ops::HoloscanToCvCudaTensorOp>("holoscantocvcuda");

    auto cvcudatoholoscan = make_operator<ops::CvCudaToHoloscan>("cvcudatoholoscan");

    auto preprocess = make_operator<ops::PreprocessImage>("preprocess");

    auto videobuffertocvcuda = make_operator<ops::VideoBufferToCvCuda>("videobuffertocvcuda");

    auto holoviz = make_operator<ops::HolovizOp>("viz", from_config("viz"));

    auto preinference = make_operator<ops::FormatConverterOp>(
        "preinference", from_config("preinference"), Arg("pool") = pool_resource);

    // add_flow(source, videobuffertocvcuda, {{"signal", "input"}});
    // add_flow(videobuffertocvcuda, cvcudatoholoscan, {{"output", "input"}});
    // add_flow(cvcudatoholoscan, holoviz, {{"output", "receivers"}});

    // add_flow(source, holoviz, {{"signal", "receivers"}});

    add_flow(source, preprocessor, {{"signal", "source_video"}});
    // add_flow(preprocessor, holoviz, {{"tensor", "receivers"}});

    add_flow(preprocessor, holoscantocvcuda, {{"tensor", "input"}});
    add_flow(holoscantocvcuda, preprocess);
    add_flow(preprocess, cvcudatoholoscan);
    // add_flow(holoscantocvcuda, cvcudatoholoscan);
    // add_flow(cvcudatoholoscan, holoviz, {{"output", "receivers"}});

    ops::InferenceOp::DataMap model_path_map;
    ops::InferenceOp::DataVecMap pre_processor_map;
    ops::InferenceOp::DataVecMap inference_map;

    std::string model_index_str = "own_model";

    model_path_map.insert(model_index_str, "/workspace/holohub/data/resnet50/model.onnx");
    // pre_processor_map.insert(model_index_str, {"source_video"});
    pre_processor_map.insert(model_index_str, {"image"});

    std::string output_name = "output";
    inference_map.insert(model_index_str, {output_name});

    auto inference = make_operator<ops::InferenceOp>("inference",
                                                     from_config("inference"),
                                                     Arg("allocator") = pool_resource,
                                                     Arg("model_path_map", model_path_map),
                                                     Arg("pre_processor_map", pre_processor_map),
                                                     Arg("inference_map", inference_map));

    // add_flow(cvcudatoholoscan, preinference, {{"output", "source_video"}});
    // add_flow(preinference, inference, {{"tensor", "receivers"}});
    add_flow(cvcudatoholoscan, inference, {{"output", "receivers"}});
    auto print_text_op = make_operator<ops::PrintTextOp>("print_text_op");
    add_flow(inference, print_text_op);

    // auto source = make_operator<ops::VideoStreamReplayerOp>("replayer",
    //                                                         from_config("replayer"),
    //                                                         Arg("directory") = datapath,
    //                                                         Arg("basename") = video_name);

    // ops::InferenceOp::DataMap model_path_map;
    // ops::InferenceOp::DataVecMap pre_processor_map;
    // ops::InferenceOp::DataVecMap inference_map;

    // for (int i = 0; i < num_inferences; i++) {
    //   std::string model_index_str = "own_model_" + std::to_string(i);

    //   model_path_map.insert(model_index_str, datapath + "/" + model_name);
    //   pre_processor_map.insert(model_index_str, {"source_video"});

    //   std::string output_name = "output" + std::to_string(i);
    //   inference_map.insert(model_index_str, {output_name});
    // }
    // auto inference = make_operator<ops::InferenceOp>("inference",
    //                                                  from_config("inference"),
    //                                                  Arg("allocator") = pool_resource,
    //                                                  Arg("model_path_map", model_path_map),
    //                                                  Arg("pre_processor_map", pre_processor_map),
    //                                                  Arg("inference_map", inference_map));

    // std::vector<std::shared_ptr<Operator>> holovizs;
    // holovizs.reserve(num_inferences);
    // // Flow definition

    // if (!only_inference && !inference_postprocessing) {
    //   for (int i = 0; i < num_inferences; i++) {
    //     std::string holoviz_name = "holoviz" + std::to_string(i);
    //     auto holoviz = make_operator<ops::HolovizOp, std::string>(holoviz_name,
    //     from_config("viz")); holovizs.push_back(holoviz);
    //     // Passthrough to Visualization
    //     // add_flow(source, holoviz, {{"output", "receivers"}});
    //     add_flow(source, holoviz, {{"signal", "receivers"}});
    //   }
    // }

    // // Inference Path
    // // add_flow(source, preprocessor, {{"output", "source_video"}});
    // add_flow(source, preprocessor, {{"signal", "source_video"}});
    // add_flow(preprocessor, inference, {{"tensor", "receivers"}});
    // if (only_inference) {
    //   HOLOSCAN_LOG_INFO(
    //       "Only inference mode is on, no post-processing and visualization will be done.");
    //   auto sink = make_operator<ops::SinkOp>("sink");
    //   add_flow(inference, sink);
    //   return;
    // }

    // std::vector<std::shared_ptr<Operator>> postprocessors;
    // postprocessors.reserve(num_inferences);

    // for (int i = 0; i < num_inferences; i++) {
    //   std::string postprocessor_name = "postprocessor" + std::to_string(i);
    //   std::string in_tensor_name = "output" + std::to_string(i);
    //   auto postprocessor = make_operator<ops::SegmentationPostprocessorOp, std::string>(
    //       postprocessor_name,
    //       from_config("postprocessor"),
    //       Arg("allocator") = pool_resource,
    //       Arg("in_tensor_name") = in_tensor_name);
    //   postprocessors.push_back(postprocessor);
    //   add_flow(inference, postprocessor, {{"transmitter", "in_tensor"}});
    // }

    // if (inference_postprocessing) {
    //   HOLOSCAN_LOG_INFO("Inference and Post-processing mode is on. No visualization will be
    //   done."); for (int i = 0; i < num_inferences; i++) {
    //     std::string sink_name = "sink" + std::to_string(i);
    //     auto sink = make_operator<ops::SinkOp, std::string>(sink_name);
    //     add_flow(postprocessors[i], sink);
    //   }
    //   return;
    // }

    // for (int i = 0; i < num_inferences; i++) {
    //   add_flow(postprocessors[i], holovizs[i], {{"out_tensor", "receivers"}});
    // }
  }

 private:
  int num_inferences = 1;
  bool only_inference = false, inference_postprocessing = false;
  std::string datapath, model_name, video_name;
};

void print_help() {
  std::cout << "Usage: benchmark_model [OPTIONS] [ConfigPath]" << std::endl;
  std::cout << "ConfigPath                    Path to the config file (default: "
               "<current directory>/benchmark_model.yaml)"
            << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -d, --data <path>               Path to the data directory (default: "
               "$HOLOSCAN_INPUT_PATH/../data)"
            << std::endl;
  std::cout << "  -m, --model-name <path>              Path to the model directory (default: "
               "identity_model.onnx)"
            << std::endl;
  std::cout << "  -v, --video-name <path>         Path to the video file (default: video)"
            << std::endl;
  std::cout << "  -i, --only-inference            Only run inference, no post-processing or "
               "visualization"
            << std::endl;
  std::cout << "  -p, --inference-postprocessing  Run inference and post-processing, no "
               "visualization"
            << std::endl;
  std::cout
      << "  -l, --multi-inference <num>     Number of inferences to run in parallel (default: 1)"
      << std::endl;
  std::cout << "  -h, --help                      Print this help" << std::endl;
}

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path,
                     std::string& model_name, std::string& video_name, bool& only_inference,
                     bool& inference_postprocessing, int& num_inferences) {
  static struct option long_options[] = {{"help", required_argument, 0, 'h'},
                                         {"data", required_argument, 0, 'd'},
                                         {"model-name", required_argument, 0, 'm'},
                                         {"video-name", required_argument, 0, 'v'},
                                         {"only-inference", optional_argument, 0, 'i'},
                                         {"inference-postprocessing", optional_argument, 0, 'p'},
                                         {"multi-inference", required_argument, 0, 'l'},
                                         {0, 0, 0, 0}};

  while (int c = getopt_long(argc, argv, "hd:m:v:ipl:", long_options, NULL)) {
    if (c == -1 || c == '?') break;

    switch (c) {
      case 'h':
        print_help();
        return false;
      case 'd':
        data_path = optarg;
        break;
      case 'm':
        model_name = optarg;
        break;
      case 'v':
        video_name = optarg;
        break;
      case 'i':
        only_inference = true;
        break;
      case 'p':
        inference_postprocessing = true;
        break;
      case 'l':
        num_inferences = std::stoi(optarg);
        break;
      default:
        std::cerr << "Unknown arguments returned: " << c << std::endl;
        print_help();
        return false;
    }
  }

  if (optind < argc) { config_name = argv[optind++]; }
  return true;
}

int main(int argc, char** argv) {
  // Parse the arguments
  std::string config_name = "";
  auto env_var = std::getenv("HOLOSCAN_INPUT_PATH");
  auto data_path = std::string((env_var == nullptr ? "" : env_var)) + "../data";
  std::string model_name = "identity_model.onnx";
  std::string video_name = "video";
  bool only_inference = false, inference_postprocessing = false;
  int num_inferences = 1;
  if (!parse_arguments(argc,
                       argv,
                       config_name,
                       data_path,
                       model_name,
                       video_name,
                       only_inference,
                       inference_postprocessing,
                       num_inferences)) {
    return 1;
  }

  auto app = holoscan::make_application<App>(
      data_path, model_name, video_name, num_inferences, only_inference, inference_postprocessing);
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