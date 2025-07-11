/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

class HelloWorldOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(HelloWorldOp)

  HelloWorldOp() = default;

  void setup(OperatorSpec& spec) override {}

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    std::cout << std::endl;
    std::cout << "Hello World!" << std::endl;
    std::cout << std::endl;
  }
};

}  // namespace holoscan::ops

class HelloWorldApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the operators
    auto hello = make_operator<ops::HelloWorldOp>("hello", make_condition<CountCondition>(1));

    // Define the one-operator workflow
    add_operator(hello);
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<HelloWorldApp>();
  app->run();

  return 0;
}
