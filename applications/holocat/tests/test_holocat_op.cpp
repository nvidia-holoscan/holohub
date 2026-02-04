/**
 * @file test_holocat_op.cpp
 * @brief Unit Tests for HolocatOp
 * 
 * Tests the HolocatOp operator configuration and initialization.
 */

#include <gtest/gtest.h>
#include <memory>

#include "holoscan/holoscan.hpp"
#include "holocat_op.hpp"
#include "test_helpers.hpp"

namespace holocat {

/**
 * @brief Test fixture for HolocatOp unit tests
 * 
 * These tests focus on configuration and initialization without
 * requiring actual EtherCAT hardware or complete EC-Master SDK functionality.
 */
class HolocatOpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fragment_ = std::make_shared<holoscan::Fragment>();
  }

  void TearDown() override {
    holocat_op_.reset();
    fragment_.reset();
  }

  std::shared_ptr<holoscan::Fragment> fragment_;
  std::shared_ptr<HolocatOp> holocat_op_;
};

/**
 * @brief Test basic creation and configuration
 */
TEST_F(HolocatOpTest, BasicCreationAndConfiguration) {
  holocat_op_ = fragment_->make_operator<HolocatOp>("test_holocat_op");

  ASSERT_NE(holocat_op_, nullptr);
  EXPECT_EQ(holocat_op_->name(), "test_holocat_op");

  auto config = test::CreateDefaultTestConfig();
  EXPECT_NO_THROW(holocat_op_->set_config(config));
}

/**
 * @brief Test operator setup
 */
TEST_F(HolocatOpTest, OperatorSetup) {
  holocat_op_ = fragment_->make_operator<HolocatOp>("test_op");
  ASSERT_NE(holocat_op_, nullptr);

  auto config = test::CreateDefaultTestConfig();
  holocat_op_->set_config(config);

  auto spec = std::make_shared<holoscan::OperatorSpec>(fragment_.get());
  EXPECT_NO_THROW(holocat_op_->setup(*spec));
}

/**
 * @brief Test proper cleanup
 */
TEST_F(HolocatOpTest, ResourceCleanup) {
  holocat_op_ = fragment_->make_operator<HolocatOp>("cleanup_test");
  ASSERT_NE(holocat_op_, nullptr);

  auto config = test::CreateDefaultTestConfig();
  holocat_op_->set_config(config);

  EXPECT_NO_THROW(holocat_op_.reset());
  EXPECT_EQ(holocat_op_, nullptr);
}

}  // namespace holocat
