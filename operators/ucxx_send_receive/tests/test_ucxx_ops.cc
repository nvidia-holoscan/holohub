/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <thread>

#include "gtest/gtest.h"
#include "holoscan/holoscan.hpp"
#include "torch/torch.h"

#include "common/utils/torch_utils.h"
#include "messages/schemas/image_generated.h"
#include "operators/ucxx/ucxx_endpoint.h"
#include "operators/ucxx/ucxx_receiver_op.h"
#include "operators/ucxx/ucxx_sender_op.h"

constexpr int kImageWidth = 1920;
constexpr int kImageHeight = 1080;

// Emits an ImageT.
class PingImageTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingImageTxOp)

  PingImageTxOp() {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<std::shared_ptr<isaac::ImageT>>("out");
  }

  void compute([[maybe_unused]] holoscan::InputContext& input, holoscan::OutputContext& output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto image = std::make_shared<isaac::ImageT>();
    image->width = kImageWidth;
    image->height = kImageHeight;
    image->encoding = isaac::ImageEncoding::ImageEncoding_RGB8;

    auto torch_image = torch::rand({image->height, image->width, 3}, torch::device(torch::kCUDA));
    image->data = std::make_shared<holoscan::Tensor>(isaac::utils::torch_to_holoscan(torch_image));

    output.emit(image, "out");
  }
};

// Receives and verifies ImageT.
class PingImageRxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingImageRxOp)

  PingImageRxOp() {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<std::shared_ptr<isaac::ImageT>>("in");
  }

  void compute(holoscan::InputContext& input, [[maybe_unused]] holoscan::OutputContext& output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto image = input.receive<std::shared_ptr<isaac::ImageT>>("in").value();
    EXPECT_EQ(image->width, kImageWidth);
    EXPECT_EQ(image->height, kImageHeight);
    ASSERT_NE(image->data, nullptr);
    EXPECT_EQ(image->data->shape()[1], kImageWidth);
    EXPECT_EQ(image->data->shape()[0], kImageHeight);
    EXPECT_EQ(image->data->shape()[2], 3);
    EXPECT_EQ(image->encoding, isaac::ImageEncoding::ImageEncoding_RGB8);
  }
};

class UcxxTestApp : public holoscan::Application {
 public:
  UcxxTestApp() {}

  void compose() override {
    auto ucxx_server_endpoint = make_resource<isaac::os::UcxxEndpoint>(
        "ucxx_server_endpoint", holoscan::Arg("port", 50009), holoscan::Arg("listen", true));

    auto ucxx_client_endpoint = make_resource<isaac::os::UcxxEndpoint>(
        "ucxx_client_endpoint", holoscan::Arg("port", 50009), holoscan::Arg("listen", false));

    auto image_tx =
        make_operator<PingImageTxOp>("image_tx", make_condition<holoscan::CountCondition>(10));

    auto ucxx_tx = make_operator<isaac::os::ops::UcxxSenderOp>(
        "ucxx_tx", holoscan::Arg("tag", 777ul), holoscan::Arg("endpoint", ucxx_client_endpoint));

    auto ucxx_rx = make_operator<isaac::os::ops::UcxxReceiverOp>(
        "ucxx_rx", holoscan::Arg("tag", 777ul), holoscan::Arg("schema_name", "isaac.Image"),
        holoscan::Arg("buffer_size", (4 << 10) + kImageWidth * kImageHeight * 3),
        holoscan::Arg("endpoint", ucxx_server_endpoint),
        make_condition<holoscan::CountCondition>(11));

    auto image_rx =
        make_operator<PingImageRxOp>("ping_image_rx", make_condition<holoscan::CountCondition>(10));

    add_flow(image_tx, ucxx_tx);
    add_flow(ucxx_rx, image_rx);
  }
};

TEST(TestUcxxOps, TestSendReceive) {
  auto app = holoscan::make_application<UcxxTestApp>();
  EXPECT_NO_THROW(app->run());
}
