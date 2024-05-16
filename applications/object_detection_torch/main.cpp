/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <getopt.h>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/aja_source/aja_source.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/inference_processor/inference_processor.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <holoscan/operators/video_stream_recorder/video_stream_recorder.hpp>

class App : public holoscan::Application {
 public:
  void set_source(const std::string &source) {
    if (source == "aja") {
      is_aja_source_ = true;
    }
  }

  void set_datapath(const std::string &path) {
    datapath = path;
  }

  // Specifies if the output of the visualizer should be recorded.
  enum class Record { NONE, VISUALIZER };

  void set_record(const std::string& record) {
    if (record == "visualizer") {
      record_type_ = Record::VISUALIZER;
    } else {
      record_type_ = Record::NONE;
    }
  }

  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Operator> source;
    std::shared_ptr<Operator> recorder;
    std::shared_ptr<Operator> recorder_format_converter;

    std::shared_ptr<Resource> pool_resource = make_resource<UnboundedAllocator>("pool");
    if (is_aja_source_) {
      source = make_operator<ops::AJASourceOp>("aja", from_config("aja"));
    } else {
      source = make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"),
                                                         Arg("directory", datapath));
    }

    auto in_dtype = is_aja_source_ ? std::string("rgba8888") : std::string("rgb888");

    auto detect_preprocessor =
        make_operator<ops::FormatConverterOp>("detect_preprocessor",
                                              from_config("detect_preprocessor"),
                                              Arg("in_dtype") = in_dtype,
                                              Arg("pool") = pool_resource);

    ops::InferenceOp::DataMap model_path_map;
    std::string model_file_name = "frcnn_resnet50_t.pt";

    model_path_map.insert("detect", datapath + "/" + model_file_name);

    auto detect_inference = make_operator<ops::InferenceOp>(
        "detect_inference", from_config("detect_inference"),
        Arg("model_path_map", model_path_map), Arg("allocator") = pool_resource);

    std::string processing_config_path = datapath+"/postprocessing.yaml";

    auto detect_postprocessor =
        make_operator<ops::InferenceProcessorOp>("detect_postprocessor",
                                                   from_config("detect_postprocessor"),
                                                   Arg("config_path", processing_config_path),
                                                   Arg("allocator") = pool_resource);

    std::map<std::string, int> label_count;
    std::map<std::string, std::vector<float>> color_map;

    if (processing_config_path.length() == 0) {
      HOLOSCAN_LOG_ERROR("Config path cannot be empty");
      std::exit(1);
    }

    if (!std::filesystem::exists(processing_config_path)) {
      HOLOSCAN_LOG_ERROR("Config path not found. Please make sure the path exists {}",
                          processing_config_path);
      HOLOSCAN_LOG_ERROR("Did you forget to download the datasets manually?");
      std::exit(1);
    }

    YAML::Node config = YAML::LoadFile(processing_config_path);
    if (!config["generate_boxes"]) {
      HOLOSCAN_LOG_ERROR("generate_boxes key not present in config file.");
      std::exit(1);
    }
    auto configuration = config["generate_boxes"].as<holoscan::inference::node_type>();

    if (configuration.find("objects") != configuration.end()) {
      for (const auto &[current_object, count] : configuration["objects"]) {
        label_count.insert({current_object, std::stoi(count)});
      }
    } else {
      HOLOSCAN_LOG_WARN("No object settings found in config.");
    }

    if (configuration.find("color") != configuration.end()) {
      for (const auto &[current_object, current_color] : configuration["color"]) {
        std::vector<std::string> tokens;
        std::vector<float> col;

        if (current_color.length() != 0) {
          holoscan::inference::string_split(current_color, tokens, ' ');
          if (tokens.size() == 4) {
            for (const auto &t : tokens) {
              col.push_back(std::stof(t));
            }
          } else {
            color_map.insert({current_object, {0, 0, 0, 1}});
          }
        } else {
          color_map.insert({current_object, {0, 0, 0, 1}});
        }
        color_map.insert({current_object, col});
      }
    }

    size_t num_of_objects = 0;
    for (const auto &item : label_count) {
      num_of_objects += item.second;
    }
    std::cout << "Maximum number of items displayed in Holoviz: " << num_of_objects << std::endl;
    std::vector<ops::HolovizOp::InputSpec> input_object_specs(1 + 2 * num_of_objects);

    int object_populated_so_far = 0;
    for (const auto &item : label_count) {
      auto key = item.first;
      auto max_objects = item.second;

      for (int u = 0; u < max_objects; u++) {
        int index = 2 * u + object_populated_so_far;
        input_object_specs[index].tensor_name_ = key + std::to_string(u);
        input_object_specs[index].priority_ = 1;
        input_object_specs[index].line_width_ = 4;
        input_object_specs[index].depth_map_render_mode_ =
            ops::HolovizOp::DepthMapRenderMode::POINTS;
        input_object_specs[index].type_ = ops::HolovizOp::InputType::RECTANGLES;

        input_object_specs[index + 1].tensor_name_ = key + "text" + std::to_string(u);
        input_object_specs[index + 1].priority_ = 1;
        input_object_specs[index + 1].depth_map_render_mode_ =
            ops::HolovizOp::DepthMapRenderMode::POINTS;
        input_object_specs[index + 1].type_ = ops::HolovizOp::InputType::TEXT;
        input_object_specs[index + 1].text_ = {key};

        if (color_map.find(key) != color_map.end()) {
          input_object_specs[index].color_ = color_map.at(key);
          input_object_specs[index + 1].color_ = color_map.at(key);
        } else {
          input_object_specs[index].color_ = {0.0, 0.0, 0.0, 1.0};  // black is default
          input_object_specs[index + 1].color_ = {0.0, 0.0, 0.0, 1.0};
        }
      }
      object_populated_so_far += 2 * max_objects;
    }
    auto image_index = input_object_specs.size() - 1;
    input_object_specs[image_index].tensor_name_ = "";
    input_object_specs[image_index].type_ = ops::HolovizOp::InputType::COLOR;
    input_object_specs[image_index].priority_ = 0;

    auto holoviz = make_operator<ops::HolovizOp>("holoviz",
                                                 from_config("holoviz"),
                                                 Arg("allocator") = pool_resource,
                                                 Arg("enable_render_buffer_output")
                                                    = (record_type_ == Record::VISUALIZER),
                                                 Arg("tensors") = input_object_specs);

    if (record_type_ == Record::VISUALIZER) {
      recorder_format_converter = make_operator<ops::FormatConverterOp>(
           "recorder_format_converter",
           from_config("recorder_format_converter"),
           Arg("pool") = pool_resource);

      recorder = make_operator<ops::VideoStreamRecorderOp>("recorder", from_config("recorder"));
    }

    // Flow definition
    if (is_aja_source_) {
      const std::set<std::pair<std::string, std::string>> aja_ports =
                                          {{"video_buffer_output", ""}};
      add_flow(source, detect_preprocessor, aja_ports);
    } else {
      add_flow(source, detect_preprocessor);
      add_flow(source, holoviz, {{"", "receivers"}});
    }

    if (record_type_ == Record::VISUALIZER) {
      add_flow(holoviz,
               recorder_format_converter,
               {{"render_buffer_output", "source_video"}});
      add_flow(recorder_format_converter, recorder);
    }

    add_flow(detect_preprocessor, detect_inference, {{"", "receivers"}});
    add_flow(detect_inference, detect_postprocessor, {{"transmitter", "receivers"}});
    add_flow(detect_postprocessor, holoviz, {{"", "receivers"}});
  }

 private:
  bool is_aja_source_ = false;
  Record record_type_ = Record::NONE;
  std::string datapath = "data/object_detection_torch";
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char **argv, std::string &config_name, std::string &data_path) {
  static struct option long_options[] = {
      {"data", required_argument, 0, 'd'},
      {0, 0, 0, 0}};

  while (int c = getopt_long(argc, argv, "d",
                             long_options, NULL)) {
    if (c == -1 || c == '?')
      break;

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

int main(int argc, char **argv) {
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
    config_path += "/object_detection_torch.yaml";
    app->config(config_path);
  }

  auto record_type = app->from_config("record_type").as<std::string>();
  app->set_record(record_type);

  auto source = app->from_config("source").as<std::string>();
  app->set_source(source);
  if (data_path != "")
    app->set_datapath(data_path);
  app->run();

  return 0;
}
