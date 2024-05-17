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

#pragma once

#include <dds/sub/ddssub.hpp>

#include "dds_operator_base.hpp"
#include "ShapeType.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to subscribe to DDS shapes as published by the RTI Connext Shapes demo.
 *
 * Note that this does not currently support fill types or rotation.
 */
class DDSShapesSubscriberOp : public DDSOperatorBase {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(DDSShapesSubscriberOp, DDSOperatorBase)

  DDSShapesSubscriberOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

  /**
   * @brief Class used by the operator's output to describe a shape.
   */
  class Shape {
   public:
    enum Type {
      SQUARE,
      CIRCLE,
      TRIANGLE
    };

    Shape() = default;
    Shape(Type type, const std::string& color, float x, float y, float width, float height)
      : type_(type), color_(str_to_color(color)), x_(x), y_(y), width_(width), height_(height) {}

    Type type_;
    std::vector<float> color_;
    float x_;
    float y_;
    float width_;
    float height_;

   private:
    /**
     * @brief Converts a color name to its numerical values.
     */
    static std::vector<float> str_to_color(const std::string& color);
  };

 private:
  Parameter<std::string> reader_qos_;

  // Shape readers.
  dds::sub::DataReader<ShapeTypeExtended> square_reader_ = dds::core::null;
  dds::sub::DataReader<ShapeTypeExtended> circle_reader_ = dds::core::null;
  dds::sub::DataReader<ShapeTypeExtended> triangle_reader_ = dds::core::null;

  // Constants to scale the shapes relative to what the RTI Connext
  // shapes application uses for its window size.
  const float publisher_width_ = 235.0f;
  const float publisher_height_ = 265.0f;

  /**
   * @brief Takes shape samples from a reader and generates/appends the corresponding
   * Shape objects for the operator's output.
   */
  void add_shapes_to_output(std::vector<Shape>& shapes,
                            dds::sub::DataReader<ShapeTypeExtended>& reader,
                            Shape::Type shape_type);
};

}  // namespace holoscan::ops
