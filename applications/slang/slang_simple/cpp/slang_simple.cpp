/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/core/application.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/resources/gxf/unbounded_allocator.hpp>

#include "slang_shader_op.hpp"

namespace holoscan::ops {

/**
 * A simple source operator that generates incrementing integer values.
 *
 * This operator serves as a data source in the Holoscan pipeline, emitting
 * incrementing integer values starting from 1. Each compute cycle produces
 * a new value that gets passed to downstream operators.
 */
class SourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SourceOp)

  SourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(allocator_,
               "allocator",
               "Allocator for output buffers.",
               "Allocator for output buffers.",
               std::static_pointer_cast<Allocator>(
                   fragment()->make_resource<UnboundedAllocator>("allocator")));
    spec.output<gxf::Entity>("output");
  }

  void initialize() override {
    // Add the allocator to the operator so that it is initialized
    add_arg(allocator_.default_value());

    // Call the base class initialize function
    Operator::initialize();
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto entity = gxf::Entity::New(&context);
    auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>().value();
    // Get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), allocator_.get()->gxf_cid());
    tensor->reshape<int>(
        nvidia::gxf::Shape({1}), nvidia::gxf::MemoryStorageType::kHost, gxf_allocator.value());

    auto value = index_++;
    std::memcpy(tensor->pointer(), &value, sizeof(value));

    op_output.emit(entity, "output");
  };

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  int index_ = 1;
};

/**
 * A simple sink operator that receives and prints data from upstream operators.
 *
 * This operator acts as a data sink in the Holoscan pipeline, receiving
 * tensor data from upstream operators and printing the received values.
 * It's typically used for debugging or final data consumption in a pipeline.
 */
class SinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SinkOp)

  SinkOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<gxf::Entity>("input"); }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
      throw std::runtime_error("Failed to receive input");
    }

    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

    auto maybe_tensor = entity.get<nvidia::gxf::Tensor>();
    if (!maybe_tensor) {
      throw std::runtime_error("Failed to get tensor");
    }
    auto tensor = maybe_tensor.value();
    auto value = *reinterpret_cast<int*>(tensor->pointer());
    std::cout << "Received value: " << value << std::endl;
  };
};

}  // namespace holoscan::ops

/**
 * A Holoscan application demonstrating Slang shader integration.
 *
 * This application creates a simple pipeline that:
 * 1. Generates incrementing integer values using SourceOp
 * 2. Processes the data through a Slang shader using SlangShaderOp
 * 3. Receives and prints the processed results using SinkOp
 *
 * The pipeline demonstrates how to integrate Slang shaders into Holoscan
 * applications for GPU-accelerated data processing.
 *
 * Pipeline Flow:
 *     SourceOp -> SlangShaderOp -> SinkOp
 */
class SlangSimpleApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto source = make_operator<ops::SourceOp>("Source");
    auto sink = make_operator<ops::SinkOp>("Sink");

    auto slang = make_operator<ops::SlangShaderOp>("Slang",
                                                   Arg("shader_source_file", "simple.slang"),
                                                   Arg("parameter", 10),
                                                   make_condition<CountCondition>(10));

    // Define the workflow
    add_flow(source, slang, {{"output", "input_buffer"}});
    add_flow(slang, sink, {{"output_buffer", "input"}});
  }
};

int main() {
  auto app = holoscan::make_application<SlangSimpleApp>();
  app->run();

  return 0;
}
