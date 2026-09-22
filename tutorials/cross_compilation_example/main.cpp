/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace holoscan::ops {

class HelloWorldOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(HelloWorldOp)

  void setup(OperatorSpec&) override {}

  void compute(InputContext&, OutputContext&, ExecutionContext&) override {
    HOLOSCAN_LOG_INFO("Hello Holoscan!");
  }
};

}  // namespace holoscan::ops

class HelloWorldApp : public holoscan::Application {
 public:
  void compose() override {
    auto hello = make_operator<holoscan::ops::HelloWorldOp>(
        "hello", make_condition<holoscan::CountCondition>(1));
    add_operator(hello);
  }
};

int main() {
  auto app = holoscan::make_application<HelloWorldApp>();
  app->run();
  return 0;
}
