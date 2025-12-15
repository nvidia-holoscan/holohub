/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/segmentation_postprocessor/segmentation_postprocessor.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#ifdef AJA_SOURCE
#include <aja_source.hpp>
#endif

#include "gxf/std/tensor.hpp"
#include "matx.h"

#define CUDA_TRY(stmt)                                                                  \
  {                                                                                     \
    cudaError_t cuda_status = stmt;                                                     \
    if (cudaSuccess != cuda_status) {                                                   \
      HOLOSCAN_LOG_ERROR("Runtime call {} in line {} of file {} failed with '{}' ({})", \
                         #stmt,                                                         \
                         __LINE__,                                                      \
                         __FILE__,                                                      \
                         cudaGetErrorString(cuda_status),                               \
                         static_cast<int>(cuda_status));                                \
      throw std::runtime_error("Unable to copy device to host");                        \
    }                                                                                   \
  }

// Debug prints tensor content.
void print_tensor(const std::shared_ptr<holoscan::Tensor>& tensor, size_t n_printouts = 20) {
  HOLOSCAN_LOG_INFO("==========TENSOR INFO=============");

  auto device = tensor->device();
  auto dtype = tensor->dtype();
  auto shape = tensor->shape();
  auto strides = tensor->strides();
  size_t size = tensor->size();
  auto ndim = tensor->ndim();
  auto itemsize = tensor->itemsize();

  HOLOSCAN_LOG_INFO("device.device_type={}, device.device_id={}",
                    static_cast<int>(device.device_type),
                    device.device_id);
  HOLOSCAN_LOG_INFO(
      "dtype.code={}, dtype.bits={}, dtype.lanes={}", dtype.code, dtype.bits, dtype.lanes);
  for (int i = 0; i < shape.size(); ++i) {
    HOLOSCAN_LOG_INFO("shape[{}]={}", i, shape[i]);
  }
  for (int i = 0; i < strides.size(); ++i) {
    HOLOSCAN_LOG_INFO("strides[{}]={}", i, strides[i]);
  }
  HOLOSCAN_LOG_INFO("size={}", size);
  HOLOSCAN_LOG_INFO("ndim={}", ndim);
  HOLOSCAN_LOG_INFO("itemsize={}", itemsize);

  size_t nbytes = tensor->nbytes();
  HOLOSCAN_LOG_INFO("nbytes={}", nbytes);
  std::vector<float> data(nbytes);
  CUDA_TRY(cudaMemcpy(data.data(), tensor->data(), nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < std::min(size, n_printouts); i++) {
    HOLOSCAN_LOG_INFO("data[{}]={:f}", i, data[i]);
  }

  CUDA_TRY(cudaMemcpy(tensor->data(), data.data(), nbytes, cudaMemcpyHostToDevice));

  HOLOSCAN_LOG_INFO("==================================");
}

// Copies device Holoscan tensor data to host vector
template <typename T>
std::vector<T> copy_device2vec(const std::shared_ptr<holoscan::Tensor>& in) {
  size_t nbytes = in->nbytes();
  std::vector<T> out = std::vector<T>(nbytes/sizeof(T));

  CUDA_TRY(cudaMemcpy(out.data(), in->data(), nbytes, cudaMemcpyDeviceToHost));

  return out;
}

namespace holoscan::ops {

// Operator for post-processesing inference output
class DetectionPostprocessorOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DetectionPostprocessorOp)

  DetectionPostprocessorOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<gxf::Entity>("in");
    spec.output<gxf::Entity>("out");
    spec.param(
      scores_threshold_,
      "scores_threshold",
      "Scores Threshold",
      "Threshold NMS scores by this value",
      0.3f);
    spec.param(
      label_names_,
      "label_names",
      "Label Names",
      "List of label names",
      std::vector<std::string>());
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    // Get input message and make output message
    auto in_message = op_input.receive<gxf::Entity>("in").value();
    auto out_message = gxf::Entity::New(&context);

    // Get total number of detections
    auto Nd = in_message.get<Tensor>("inference_output_num_detections");
    // Copy to host
    auto Ndh = copy_device2vec<int>(Nd);

    if (Ndh[0] == 0) {
      // Check if zero detections, in which case we return empty messages

      for (int k = 0; k < label_names_.get().size(); k++) {
        // Create empty messages for each label
        auto label = label_names_.get()[k];

        auto boxes_mx = matx::make_tensor<float>({1, 2, 2});
        auto coords_mx = matx::make_tensor<float>({1, 1, 2});
        (coords_mx = -1.0).run(matx::SingleThreadHostExecutor());

        // MatX to Holoscan tensor
        auto boxes_hs = std::make_shared<holoscan::Tensor>(boxes_mx.GetDLPackTensor());
        auto coords_hs = std::make_shared<holoscan::Tensor>(coords_mx.GetDLPackTensor());

        out_message.add(boxes_hs, (label + "_rectangle").c_str());
        out_message.add(coords_hs, (label + "_label").c_str());
      }

      op_output.emit(out_message, "out");

      return;
    }

    // Boxes detected, get boxes, scores and labels
    auto boxesh = in_message.get<Tensor>("inference_output_detection_boxes");  // (1, num_boxes, 4)
    auto scoresh = in_message.get<Tensor>("inference_output_detection_scores");  // (1, num_boxes)
    auto labelsh = in_message.get<Tensor>("inference_output_detection_classes");  // (1, num_boxes)
    int32_t Nb = scoresh->shape()[1];  // Number of boxes

    // Copy to host
    auto boxes = copy_device2vec<float>(boxesh);
    auto scores = copy_device2vec<float>(scoresh);
    auto labels = copy_device2vec<int>(labelsh);

    // Holoscan tensors to MatX tensors
    auto scores_mx = matx::make_tensor<float>(scores.data(), {1, Nb});
    auto boxes_mx = matx::make_tensor<float>(boxes.data(), {1, Nb, 4});
    auto labels_mx = matx::make_tensor<int>(labels.data(), {1, Nb});

    // Find box indices with a score larger than threshold
    auto ix_mx = matx::make_tensor<int>({Nb});
    auto Ns = matx::make_tensor<int>({});  // Number of boxes above threshold
    (matx::mtie(ix_mx, Ns) = matx::find_idx(scores_mx, matx::GTE{scores_threshold_.get()})).run(
                                         matx::SingleThreadHostExecutor());

    // Get boxes and labels corresponding to indices
    auto boxes_ix_mx = matx::make_tensor<float>({1, Ns(), 4});
    auto labels_ix_mx = matx::make_tensor<int>({1, Ns()});
    auto ixs_mx = ix_mx.Slice({0}, {Ns()});
    (boxes_ix_mx = matx::remap<1>(boxes_mx, ixs_mx)).run(matx::SingleThreadHostExecutor());
    (labels_ix_mx = matx::remap<1>(labels_mx, ixs_mx)).run(matx::SingleThreadHostExecutor());

    // Get x0,y0 coordinates from boxes
    auto coords_mx = matx::make_tensor<float>({1, Ns(), 2});
    (coords_mx = boxes_ix_mx.Slice({0, 0, 0}, {matx::matxEnd, matx::matxEnd, 2})).run(
                                                                matx::SingleThreadHostExecutor());

    // Create messages for each label
    auto Nl = matx::make_tensor<int>({});  // Number of label boxes
    for (int k = 0; k < label_names_.get().size(); k++) {
      // Loop over label name and index
      auto label = label_names_.get()[k];

      // Find boxes equal to label index
      (matx::mtie(ixs_mx, Nl) =
          matx::find_idx(labels_ix_mx, matx::EQ{k + 1})).run(matx::SingleThreadHostExecutor());

      if (Nl() > 0) {
        // Label has boxes, create output messages
        auto ixl_mx = ixs_mx.Slice({0}, {Nl()});

        // Get boxes and labels corresponding to indices
        auto boxesl_mx = matx::make_tensor<float>({1, Nl(), 4});
        auto coordsl_mx = matx::make_tensor<float>({1, Nl(), 2});
        (boxesl_mx = matx::remap<1>(boxes_ix_mx, ixl_mx)).run(matx::SingleThreadHostExecutor());
        (coordsl_mx = matx::remap<1>(coords_mx, ixl_mx)).run(matx::SingleThreadHostExecutor());

        // Reshape boxes to Holoviz shape
        auto boxesls_mx = boxesl_mx.View({1, 2 * Nl(), 2});

        // MatX to Holoscan tensor
        auto boxes_hs = std::make_shared<holoscan::Tensor>(boxesls_mx.GetDLPackTensor());
        auto coords_hs = std::make_shared<holoscan::Tensor>(coordsl_mx.GetDLPackTensor());

        out_message.add(boxes_hs, (label + "_rectangle").c_str());
        out_message.add(coords_hs, (label + "_label").c_str());

      } else {
        // Label has no boxes, create empty output messages
        auto boxesl_mx = matx::make_tensor<float>({1, 2, 2});
        auto coordsl_mx = matx::make_tensor<float>({1, 1, 2});
        (coordsl_mx = -1.0).run(matx::SingleThreadHostExecutor());

        // MatX to Holoscan tensor
        auto boxes_hs = std::make_shared<holoscan::Tensor>(boxesl_mx.GetDLPackTensor());
        auto coords_hs = std::make_shared<holoscan::Tensor>(coordsl_mx.GetDLPackTensor());

        out_message.add(boxes_hs, (label + "_rectangle").c_str());
        out_message.add(coords_hs, (label + "_label").c_str());
      }
    }

    // Emit output message
    op_output.emit(out_message, "out");
  };

 private:
  Parameter<float> scores_threshold_;
  Parameter<std::vector<std::string>> label_names_;
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  void set_source(const std::string& source) {
    if (source == "aja") {
      is_aja_source_ = true;
    }
  }

  void set_datapath(const std::string& path) {
     datapath = path;
  }

  void compose() override {
    using namespace holoscan;

    /* Allocator */
    std::shared_ptr<Resource> pool = make_resource<UnboundedAllocator>("pool");

    /* Source */
    std::shared_ptr<Operator> source;
    if (is_aja_source_) {
#ifdef AJA_SOURCE
      source = make_operator<ops::AJASourceOp>("aja", from_config("aja"));
#else
      throw std::runtime_error(
          "AJA is requested but not available. Please enable AJA at build time.");
#endif
    } else {
      source = make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"),
                                                  Arg("directory", datapath + "/endoscopy"));
    }

    /* preprocessor segmentations */
    auto in_dtype = is_aja_source_ ? std::string("rgba8888") : std::string("rgb888");
    auto segmentation_preprocessor = make_operator<ops::FormatConverterOp>(
                                                          "segmentation_preprocessor",
                                                          from_config("segmentation_preprocessor"),
                                                          Arg("in_dtype") = in_dtype,
                                                          Arg("pool") = pool);

    /* preprocessor boxes */
    auto detection_preprocessor = make_operator<ops::FormatConverterOp>(
        "detection_preprocessor",
        from_config("detection_preprocessor"),
        Arg("in_tensor_name", std::string(is_aja_source_ ? "source_video" : "")),
        Arg("in_dtype") = in_dtype,
        Arg("pool") = pool);

    /* inference */
    ops::InferenceOp::DataMap model_path_map;
    model_path_map.insert("ssd", datapath + "/ssd_model/epoch24_nms.onnx");
    model_path_map.insert("tool_seg", datapath +
     "/monai_tool_seg_model/model_endoscopic_tool_seg_sanitized_nhwc_in_nchw_out.onnx");
    auto inference = make_operator<ops::InferenceOp>(
        "inference", from_config("inference"),
        Arg("model_path_map", model_path_map),
        Arg("allocator") = pool);

    /* segmentation postprocessor */
    auto segmentation_postprocessor = make_operator<ops::SegmentationPostprocessorOp>(
        "segmentation_postprocessor", from_config("segmentation_postprocessor"),
                                                        Arg("allocator") = pool);

    /* detection postprocessor */
    auto detection_postprocessor = make_operator<ops::DetectionPostprocessorOp>(
        "detection_postprocessor", from_config("detection_postprocessor"),
                                               Arg("allocator") = pool);

    /* visualizer */
    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz", from_config("holoviz"), Arg("allocator") = pool);

    /* Define flow */

    // Source
    if (is_aja_source_) {
      add_flow(source, holoviz, {{"video_buffer_output", "receivers"}});
      add_flow(source, detection_preprocessor, {{"video_buffer_output", ""}});
      add_flow(source, segmentation_preprocessor, {{"video_buffer_output", ""}});
    } else {
      add_flow(source, holoviz, {{"", "receivers"}});
      add_flow(source, detection_preprocessor);
      add_flow(source, segmentation_preprocessor);
    }

    // Detection
    add_flow(detection_preprocessor, inference, {{"", "receivers"}});
    add_flow(inference, detection_postprocessor, {{"transmitter", "in"}});
    add_flow(detection_postprocessor, holoviz, {{"out", "receivers"}});

    // Segmentation
    add_flow(segmentation_preprocessor, inference, {{"", "receivers"}});
    add_flow(inference, segmentation_postprocessor, {{"transmitter", ""}});
    add_flow(segmentation_postprocessor, holoviz, {{"", "receivers"}});
  }

 private:
  bool is_aja_source_ = false;
  std::string datapath = "data";
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path) {
  static struct option long_options[] = {
      {"data",    required_argument, 0,  'd' },
      {0,         0,                 0,  0 }
  };

  while (int c = getopt_long(argc, argv, "d",
                   long_options, NULL))  {
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

  if (optind < argc) {
    config_name = argv[optind++];
  }
  return true;
}

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Parse the arguments
  std::string data_path = "";
  std::string config_name = "";
  if (!parse_arguments(argc, argv, config_name, data_path)) {
    return 1;
  }

  if (config_name != "") {
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/app_config.yaml";
    app->config(config_path);
  }

  auto source = app->from_config("source").as<std::string>();
  app->set_source(source);
  if (data_path != "") app->set_datapath(data_path);
  app->run();

  return 0;
}
