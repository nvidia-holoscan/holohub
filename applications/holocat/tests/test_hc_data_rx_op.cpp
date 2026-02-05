/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

/**
 * @file test_hc_data_rx_op.cpp
 * @brief Unit Tests for HcDataRxOp
 *
 * Tests the HcDataRxOp operator that receives counter data.
 */

#include <gtest/gtest.h>
#include <memory>

#include "holoscan/holoscan.hpp"
#include "hc_data_rx_op.hpp"
#include "hc_data_tx_op.hpp"
#include "test_helpers.hpp"

namespace holocat {

/**
 * @brief Test fixture for HcDataRxOp unit tests
 *
 * Provides a Holoscan Fragment context for testing the HcDataRxOp operator
 */
class HcDataRxOpTest : public ::testing::Test {
 protected:
  void SetUp() override { fragment_ = std::make_shared<holoscan::Fragment>(); }

  void TearDown() override {
    rx_op_.reset();
    fragment_.reset();
  }

  std::shared_ptr<holoscan::Fragment> fragment_;
  std::shared_ptr<HcDataRxOp> rx_op_;
};

/**
 * @brief Test basic creation and setup
 */
TEST_F(HcDataRxOpTest, BasicCreationAndSetup) {
  rx_op_ = fragment_->make_operator<HcDataRxOp>("test_rx_op");

  ASSERT_NE(rx_op_, nullptr);
  EXPECT_EQ(rx_op_->name(), "test_rx_op");

  auto spec = std::make_shared<holoscan::OperatorSpec>(fragment_.get());
  EXPECT_NO_THROW(rx_op_->setup(*spec));
}

/**
 * @brief Test proper cleanup
 */
TEST_F(HcDataRxOpTest, ResourceCleanup) {
  rx_op_ = fragment_->make_operator<HcDataRxOp>("cleanup_test");
  ASSERT_NE(rx_op_, nullptr);

  EXPECT_NO_THROW(rx_op_.reset());
  EXPECT_EQ(rx_op_, nullptr);
}

}  // namespace holocat
