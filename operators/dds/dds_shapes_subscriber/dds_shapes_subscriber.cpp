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

#include "dds_shapes_subscriber.hpp"

namespace holoscan::ops {

void DDSShapesSubscriberOp::setup(OperatorSpec& spec) {
  DDSOperatorBase::setup(spec);

  spec.output<std::vector<Shape>>("output");

  spec.param(reader_qos_, "reader_qos", "Reader QoS", "Data Reader QoS Profile", std::string());
}

void DDSShapesSubscriberOp::initialize() {
  DDSOperatorBase::initialize();

  // Create the subscriber
  dds::sub::Subscriber subscriber(participant_);

  // Create the shape topics
  dds::topic::Topic<ShapeTypeExtended> square_topic(participant_, "Square");
  dds::topic::Topic<ShapeTypeExtended> circle_topic(participant_, "Circle");
  dds::topic::Topic<ShapeTypeExtended> triangle_topic(participant_, "Triangle");

  // Create the readers
  auto qos = qos_provider_.datareader_qos(reader_qos_.get());
  square_reader_ = dds::sub::DataReader<ShapeTypeExtended>(subscriber, square_topic, qos);
  circle_reader_ = dds::sub::DataReader<ShapeTypeExtended>(subscriber, circle_topic, qos);
  triangle_reader_ = dds::sub::DataReader<ShapeTypeExtended>(subscriber, triangle_topic, qos);
}

void DDSShapesSubscriberOp::compute(InputContext& op_input,
                                    OutputContext& op_output,
                                    ExecutionContext& context) {
  std::vector<Shape> shapes;

  // Read the shapes from the readers and add them to the output.
  add_shapes_to_output(shapes, square_reader_, Shape::Type::SQUARE);
  add_shapes_to_output(shapes, circle_reader_, Shape::Type::CIRCLE);
  add_shapes_to_output(shapes, triangle_reader_, Shape::Type::TRIANGLE);

  op_output.emit(shapes, "output");
}

void DDSShapesSubscriberOp::add_shapes_to_output(std::vector<Shape>& shapes,
                                                 dds::sub::DataReader<ShapeTypeExtended>& reader,
                                                 Shape::Type shape_type) {
  dds::sub::LoanedSamples<ShapeTypeExtended> read_shapes = reader.take();
  for (const auto& shape : read_shapes) {
    if (shape.info().valid()) {
      shapes.push_back(Shape(shape_type,
                             shape.data().color(),
                             shape.data().x() / publisher_width_,
                             shape.data().y() / publisher_height_,
                             shape.data().shapesize() / publisher_width_,
                             shape.data().shapesize() / publisher_height_));
    }
  }
}

std::vector<float> DDSShapesSubscriberOp::Shape::str_to_color(const std::string& color) {
  if (color == "PURPLE")  return std::vector<float>({0.5f, 0.0f, 1.0f, 1.0f});
  if (color == "BLUE")    return std::vector<float>({0.0f, 0.0f, 1.0f, 1.0f});
  if (color == "RED")     return std::vector<float>({1.0f, 0.0f, 0.0f, 1.0f});
  if (color == "GREEN")   return std::vector<float>({0.0f, 1.0f, 0.0f, 1.0f});
  if (color == "YELLOW")  return std::vector<float>({1.0f, 1.0f, 0.0f, 1.0f});
  if (color == "CYAN")    return std::vector<float>({0.0f, 1.0f, 1.0f, 1.0f});
  if (color == "MAGENTA") return std::vector<float>({1.0f, 0.0f, 1.0f, 1.0f});
  if (color == "ORANGE")  return std::vector<float>({1.0f, 0.5f, 0.0f, 1.0f});
  else                    return std::vector<float>({0.0f, 0.0f, 0.0f, 1.0f});
}

}  // namespace holoscan::ops
