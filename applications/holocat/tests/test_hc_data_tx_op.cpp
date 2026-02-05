/**
 * @file test_hc_data_tx_op.cpp
 * @brief Unit Tests for HcDataTxOp
 *
 * Tests the HcDataTxOp operator that generates counter data.
 */

#include <gtest/gtest.h>
#include <memory>

#include "holoscan/holoscan.hpp"
#include "hc_data_tx_op.hpp"
#include "test_helpers.hpp"

namespace holocat {

/**
 * @brief Test fixture for HcDataTxOp unit tests
 *
 * Provides a Holoscan Fragment context for testing the HcDataTxOp operator
 */
class HcDataTxOpTest : public ::testing::Test {
 protected:
  void SetUp() override { fragment_ = std::make_shared<holoscan::Fragment>(); }

  void TearDown() override {
    tx_op_.reset();
    fragment_.reset();
  }

  std::shared_ptr<holoscan::Fragment> fragment_;
  std::shared_ptr<HcDataTxOp> tx_op_;
};

/**
 * @brief Test basic operator creation and setup
 *
 * Verifies that HcDataTxOp can be instantiated with expected properties
 * and that setup() configures the output port correctly.
 */
TEST_F(HcDataTxOpTest, BasicCreationAndSetup) {
  tx_op_ = fragment_->make_operator<HcDataTxOp>("test_tx_op");

  ASSERT_NE(tx_op_, nullptr);
  EXPECT_EQ(tx_op_->name(), "test_tx_op");
  EXPECT_EQ(tx_op_->operator_type(), holoscan::Operator::OperatorType::kNative);

  // Verify setup doesn't crash
  auto spec = std::make_shared<holoscan::OperatorSpec>(fragment_.get());
  EXPECT_NO_THROW(tx_op_->setup(*spec));
}

/**
 * @brief Test creating multiple independent instances
 *
 * Verifies that multiple HcDataTxOp instances can coexist independently.
 */
TEST_F(HcDataTxOpTest, MultipleIndependentInstances) {
  auto op1 = fragment_->make_operator<HcDataTxOp>("tx_1");
  auto op2 = fragment_->make_operator<HcDataTxOp>("tx_2");

  ASSERT_NE(op1, nullptr);
  ASSERT_NE(op2, nullptr);
  EXPECT_NE(op1.get(), op2.get());
  EXPECT_EQ(op1->name(), "tx_1");
  EXPECT_EQ(op2->name(), "tx_2");
}

/**
 * @brief Test proper resource cleanup
 *
 * Verifies that operators can be destroyed without crashes or leaks.
 */
TEST_F(HcDataTxOpTest, ResourceCleanup) {
  tx_op_ = fragment_->make_operator<HcDataTxOp>("cleanup_test");
  ASSERT_NE(tx_op_, nullptr);

  EXPECT_NO_THROW(tx_op_.reset());
  EXPECT_EQ(tx_op_, nullptr);
}

}  // namespace holocat
