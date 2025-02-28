/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <gxf/std/tensor.hpp>

#ifdef AJA_SOURCE
#include <aja_source.hpp>
#endif

#if __has_include("gxf/std/dlpack_utils.hpp")
  #define GXF_HAS_DLPACK_SUPPORT 1
#else
  #define GXF_HAS_DLPACK_SUPPORT 0
  // Holoscan 1.0 used GXF without DLPack so gxf_tensor.hpp was needed to add it
  #include <holoscan/core/gxf/gxf_tensor.hpp>
#endif

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

// Copies device Holoscan tensor data to host vector
template <typename T>
std::vector<T> copy_device2vec(const std::shared_ptr<holoscan::Tensor>& in) {
  size_t nbytes = in->nbytes();
  std::vector<T> out = std::vector<T>(nbytes/sizeof(T));

  CUDA_TRY(cudaMemcpy(out.data(), in->data(), nbytes, cudaMemcpyDeviceToHost));

  return out;
}

// Copies host Holoscan tensor data to host vector
template <typename T>
std::vector<T> copy_host2vec(const std::shared_ptr<holoscan::Tensor>& in) {
  size_t nbytes = in->nbytes();
  const T *ptr = (T *) in->data();
  std::vector<T> out(ptr, ptr + nbytes);

  return out;
}

// Debug prints Holoscan tensor
void print_tensor(const std::shared_ptr<holoscan::Tensor>& tensor,
                  size_t n_printouts = 20, std::string device_type = "cpu") {
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
  for (int i = 0; i < shape.size(); ++i) { HOLOSCAN_LOG_INFO("shape[{}]={}", i, shape[i]); }
  for (int i = 0; i < strides.size(); ++i) { HOLOSCAN_LOG_INFO("strides[{}]={}", i, strides[i]); }
  HOLOSCAN_LOG_INFO("size={}", size);
  HOLOSCAN_LOG_INFO("ndim={}", ndim);
  HOLOSCAN_LOG_INFO("itemsize={}", itemsize);
  size_t nbytes = tensor->nbytes();
  HOLOSCAN_LOG_INFO("nbytes={}", nbytes);

  if (device_type == "cpu") {
    auto data = copy_host2vec<float>(tensor);
    for (size_t i = 0; i < std::min(size, n_printouts); i++) {
      HOLOSCAN_LOG_INFO("data[{}]={:f}", i, data[i]);
    }
  } else {
    auto data = copy_device2vec<float>(tensor);
    for (size_t i = 0; i < std::min(size, n_printouts); i++) {
      HOLOSCAN_LOG_INFO("data[{}]={:f}", i, data[i]);
    }
    CUDA_TRY(cudaMemcpy(tensor->data(), data.data(), nbytes, cudaMemcpyHostToDevice));
  }

  HOLOSCAN_LOG_INFO("==================================");
}

// Creates a host Holoscan tensor from a vector of floats (via GXF tensor)
std::shared_ptr<holoscan::Tensor> fvec2tensor(
  const std::shared_ptr<std::vector<float>>& vec, std::initializer_list<int32_t> dim) {
    // Create GXF tensor
    auto tg = std::make_shared<nvidia::gxf::Tensor>();
    tg->wrapMemory(
      nvidia::gxf::Shape(dim),
      nvidia::gxf::PrimitiveType::kFloat32,
      4,
      nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
      nvidia::gxf::MemoryStorageType::kSystem,
      (void*) vec->data(),
      [buffer = vec](void*) mutable {
        buffer.reset();
        return nvidia::gxf::Success;
      });

#if GXF_HAS_DLPACK_SUPPORT
    // Export DLPack context corresponding to the nvidia::gxf::Tensor
    auto maybe_dl_ctx = tg->toDLManagedTensorContext();
    if (!maybe_dl_ctx) {
      throw std::runtime_error(
          "failed to get std::shared_ptr<DLManagedTensorContext> from nvidia::gxf::Tensor");
    }
    // zero-copy creation of holoscan::Tensor from the DLPack context
    return std::make_shared<holoscan::Tensor>(maybe_dl_ctx.value());
#else
    // Create Holoscan GXF tensor
    auto thg = holoscan::gxf::GXFTensor(*tg);

    // Create Holoscan tensor
    auto th = thg.as_tensor();

    return th;
#endif
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
    // Get input/create output message
    auto in = op_input.receive<gxf::Entity>("in").value();
    auto out = gxf::Entity::New(&context);

    // Get boxes tensor info
    auto boxes_hs = in.get<Tensor>("inference_output_detection_boxes");
    auto strides = boxes_hs->strides();
    auto shape = boxes_hs->shape();

    // Copy Holoscan tensor to host vector
    auto boxes_in = copy_device2vec<float>(boxes_hs);
    auto scores_in = copy_device2vec<float>(in.get<Tensor>("inference_output_detection_scores"));
    auto labels_in = copy_device2vec<int>(in.get<Tensor>("inference_output_detection_classes"));

    // Allocate vectors that hold Holoviz tensor data
    int num_labels = label_names_.get().size();
    std::vector<std::shared_ptr<std::vector<float>>> boxes_out(num_labels);
    std::vector<std::shared_ptr<std::vector<float>>> coords_out(num_labels);
    std::vector<int> count_found(num_labels);
    for (int k = 0; k < num_labels; k++) {
      // Default Holoviz rectangle values
      auto vb = std::make_shared<std::vector<float>>(4);
      *vb = {0.0, 0.0, 0.0, 0.0};
      boxes_out[k] = vb;

      // Default Holoviz text coord values
      auto vc = std::make_shared<std::vector<float>>(2);
      *vc = {-1.0, -1.0};
      coords_out[k] = vc;

      // For keeping count of number of rectangles of each label that passes thresholding
      count_found[k] = 0;
    }

    // Loop over boxes and only keep boxes that passes thresholding
    int64_t ii, jj;
    int count_ij = 0;
    for (int64_t i = 0; i < shape[0]; i++) {
      for (int64_t j = 0; j < shape[1]; j++) {
        if (scores_in[count_ij] > scores_threshold_.get() && labels_in[count_ij] > 0) {
          ii = i*strides[0]/4;
          jj = j*strides[1]/4;

          int k = labels_in[count_ij] - 1;
          float x0 = boxes_in[ii + jj];
          float y0 = boxes_in[ii + jj +   strides[2]/4];
          float x1 = boxes_in[ii + jj + 2*strides[2]/4];
          float y1 = boxes_in[ii + jj + 3*strides[2]/4];

          if (count_found[k] == 0) {
            (*boxes_out[k])[0] = x0;
            (*boxes_out[k])[1] = y0;
            (*boxes_out[k])[2] = x1;
            (*boxes_out[k])[3] = y1;

            (*coords_out[k])[0] = x0;
            (*coords_out[k])[1] = y0;
          } else {
            (*boxes_out[k]).push_back(x0);
            (*boxes_out[k]).push_back(y0);
            (*boxes_out[k]).push_back(x1);
            (*boxes_out[k]).push_back(y1);

            (*coords_out[k]).push_back(x0);
            (*coords_out[k]).push_back(y0);
          }
          ++count_found[k];
        }

        ++count_ij;
      }
    }

    // Convert vector to Holoscan tensor and add to output message
    for (int k = 0; k < num_labels; k++) {
      auto label = label_names_.get()[k];
      int32_t n_bbox = (*boxes_out[k]).size()/2;
      int32_t n_coords = (*coords_out[k]).size()/2;

      auto boxes_k = fvec2tensor(
        boxes_out[k],
        {1, n_bbox, 2});

      auto coords_k = fvec2tensor(
        coords_out[k],
        {1, n_coords, 2});

      out.add(boxes_k, (label + "_rectangle").c_str());
      out.add(coords_k, (label + "_label").c_str());
    }

    op_output.emit(out, "out");
  };

 private:
  Parameter<float> scores_threshold_;
  Parameter<std::vector<std::string>> label_names_;
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  void set_source(const std::string& source) {
    if (source == "aja") { is_aja_source_ = true; }
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
      source = make_operator<ops::AJASourceOp>("aja", from_config("aja"));
    } else {
      source = make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"),
                                                  Arg("directory", datapath + "/endoscopy"));
    }

    /* preprocessor segmentations */
    auto in_dtype = is_aja_source_ ? std::string("rgba8888") : std::string("rgb888");
    auto segmentation_preprocessor =
                              make_operator<ops::FormatConverterOp>("segmentation_preprocessor",
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
        "detection_postprocessor", from_config("detection_postprocessor"), Arg("allocator") = pool);

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

