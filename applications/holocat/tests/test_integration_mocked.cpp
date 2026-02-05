/**
 * @file test_integration_mocked.cpp
 * @brief Mocked Integration Tests for HoloCat
 *
 * Tests use mock EC-Master SDK interfaces to test the full HoloCat
 * data flow and state machine without requiring actual hardware.
 */

#include <gtest/gtest.h>
#include <memory>

#include "holoscan/holoscan.hpp"
// Note: holocat_op.hpp requires EC-Master SDK headers, so we don't include it here
// #include "holocat_op.hpp"
#include "hc_data_tx_op.hpp"
#include "hc_data_rx_op.hpp"
#include "test_helpers.hpp"
#include "mock_ec_master.hpp"

namespace holocat {
namespace integration_test {

/**
 * @brief Test fixture for mocked integration tests
 */
class MockedIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fragment_ = std::make_shared<holoscan::Fragment>();

    // Reset mock context before each test
    mock::GetMockContext().Reset();
  }

  void TearDown() override {
    fragment_.reset();

    // Clean up mock context after each test
    mock::GetMockContext().Reset();
  }

  std::shared_ptr<holoscan::Fragment> fragment_;
};

// ====================================================================================
// Mock EC-Master State Machine Tests
// ====================================================================================

/**
 * @brief Test mock EC-Master initialization
 */
TEST_F(MockedIntegrationTest, MockEcMasterInitialization) {
  auto& ctx = mock::GetMockContext();

  EXPECT_FALSE(ctx.IsInitialized());
  EXPECT_EQ(ctx.GetState(), mock::EcMasterState::UNKNOWN);

  // Simulate initialization
  auto result = mock::MockEcatInitMaster(nullptr);
  EXPECT_EQ(result, mock::EC_E_NOERROR);

  EXPECT_TRUE(ctx.IsInitialized());
  EXPECT_EQ(ctx.GetState(), mock::EcMasterState::INIT);
}

/**
 * @brief Test mock EC-Master configuration
 */
TEST_F(MockedIntegrationTest, MockEcMasterConfiguration) {
  auto& ctx = mock::GetMockContext();

  // Initialize first
  EXPECT_EQ(mock::MockEcatInitMaster(nullptr), mock::EC_E_NOERROR);
  EXPECT_FALSE(ctx.IsConfigured());

  // Configure
  auto result = mock::MockEcatConfigureMaster("/tmp/test_eni.xml");
  EXPECT_EQ(result, mock::EC_E_NOERROR);

  EXPECT_TRUE(ctx.IsConfigured());
}

/**
 * @brief Test mock EC-Master state transitions
 */
TEST_F(MockedIntegrationTest, MockEcMasterStateTransitions) {
  auto& ctx = mock::GetMockContext();

  // Initialize and configure
  EXPECT_EQ(mock::MockEcatInitMaster(nullptr), mock::EC_E_NOERROR);
  EXPECT_EQ(mock::MockEcatConfigureMaster("/tmp/test.xml"), mock::EC_E_NOERROR);

  // Transition to PREOP
  auto result =
      mock::MockEcatSetMasterState(3000, static_cast<uint32_t>(mock::EcMasterState::PREOP));
  EXPECT_EQ(result, mock::EC_E_NOERROR);
  EXPECT_EQ(ctx.GetState(), mock::EcMasterState::PREOP);

  // Transition to SAFEOP
  result = mock::MockEcatSetMasterState(3000, static_cast<uint32_t>(mock::EcMasterState::SAFEOP));
  EXPECT_EQ(result, mock::EC_E_NOERROR);
  EXPECT_EQ(ctx.GetState(), mock::EcMasterState::SAFEOP);

  // Transition to OP
  result = mock::MockEcatSetMasterState(3000, static_cast<uint32_t>(mock::EcMasterState::OP));
  EXPECT_EQ(result, mock::EC_E_NOERROR);
  EXPECT_EQ(ctx.GetState(), mock::EcMasterState::OP);
}

/**
 * @brief Test mock EC-Master full startup sequence
 */
TEST_F(MockedIntegrationTest, MockEcMasterFullStartupSequence) {
  auto& ctx = mock::GetMockContext();

  // Complete startup sequence: INIT -> CONFIGURE -> PREOP -> SAFEOP -> OP
  EXPECT_EQ(mock::MockEcatInitMaster(nullptr), mock::EC_E_NOERROR);
  EXPECT_EQ(ctx.GetState(), mock::EcMasterState::INIT);

  EXPECT_EQ(mock::MockEcatConfigureMaster("/tmp/test.xml"), mock::EC_E_NOERROR);
  EXPECT_TRUE(ctx.IsConfigured());

  EXPECT_EQ(mock::MockEcatSetMasterState(3000, static_cast<uint32_t>(mock::EcMasterState::PREOP)),
            mock::EC_E_NOERROR);
  EXPECT_EQ(ctx.GetState(), mock::EcMasterState::PREOP);

  EXPECT_EQ(mock::MockEcatSetMasterState(3000, static_cast<uint32_t>(mock::EcMasterState::SAFEOP)),
            mock::EC_E_NOERROR);
  EXPECT_EQ(ctx.GetState(), mock::EcMasterState::SAFEOP);

  EXPECT_EQ(mock::MockEcatSetMasterState(3000, static_cast<uint32_t>(mock::EcMasterState::OP)),
            mock::EC_E_NOERROR);
  EXPECT_EQ(ctx.GetState(), mock::EcMasterState::OP);
}

// ====================================================================================
// Mock Error Simulation Tests
// ====================================================================================

/**
 * @brief Test mock EC-Master initialization error
 */
TEST_F(MockedIntegrationTest, MockEcMasterInitializationError) {
  auto& ctx = mock::GetMockContext();

  // Enable error simulation
  ctx.SimulateInitError(true);

  auto result = mock::MockEcatInitMaster(nullptr);
  EXPECT_EQ(result, mock::EC_E_ERROR);
  EXPECT_FALSE(ctx.IsInitialized());
}

/**
 * @brief Test mock EC-Master timeout simulation
 */
TEST_F(MockedIntegrationTest, MockEcMasterTimeoutSimulation) {
  auto& ctx = mock::GetMockContext();

  // Initialize and configure normally
  EXPECT_EQ(mock::MockEcatInitMaster(nullptr), mock::EC_E_NOERROR);
  EXPECT_EQ(mock::MockEcatConfigureMaster("/tmp/test.xml"), mock::EC_E_NOERROR);

  // Enable timeout simulation
  ctx.SimulateTimeout(true);

  // State transition should timeout
  auto result =
      mock::MockEcatSetMasterState(3000, static_cast<uint32_t>(mock::EcMasterState::PREOP));
  EXPECT_EQ(result, mock::EC_E_TIMEOUT);
}

/**
 * @brief Test mock EC-Master invalid state error
 */
TEST_F(MockedIntegrationTest, MockEcMasterInvalidStateError) {
  auto& ctx = mock::GetMockContext();

  // Try to configure without initializing
  auto result = mock::MockEcatConfigureMaster("/tmp/test.xml");
  EXPECT_EQ(result, mock::EC_E_INVALIDSTATE);
  EXPECT_FALSE(ctx.IsConfigured());

  // Try to transition state without configuring
  EXPECT_EQ(mock::MockEcatInitMaster(nullptr), mock::EC_E_NOERROR);
  result = mock::MockEcatSetMasterState(3000, static_cast<uint32_t>(mock::EcMasterState::PREOP));
  EXPECT_EQ(result, mock::EC_E_INVALIDSTATE);
}

// ====================================================================================
// Mock Cyclic Job Tests
// ====================================================================================

/**
 * @brief Test mock EC-Master cyclic job execution
 */
TEST_F(MockedIntegrationTest, MockEcMasterCyclicJob) {
  auto& ctx = mock::GetMockContext();

  // Setup: Initialize, configure, and reach OP state
  EXPECT_EQ(mock::MockEcatInitMaster(nullptr), mock::EC_E_NOERROR);
  EXPECT_EQ(mock::MockEcatConfigureMaster("/tmp/test.xml"), mock::EC_E_NOERROR);
  EXPECT_EQ(mock::MockEcatSetMasterState(3000, static_cast<uint32_t>(mock::EcMasterState::PREOP)),
            mock::EC_E_NOERROR);
  EXPECT_EQ(mock::MockEcatSetMasterState(3000, static_cast<uint32_t>(mock::EcMasterState::SAFEOP)),
            mock::EC_E_NOERROR);
  EXPECT_EQ(mock::MockEcatSetMasterState(3000, static_cast<uint32_t>(mock::EcMasterState::OP)),
            mock::EC_E_NOERROR);

  EXPECT_EQ(ctx.GetCycleCount(), 0);

  // Execute cyclic jobs
  for (int i = 0; i < 10; ++i) {
    auto result = mock::MockEcatExecJob(0, nullptr);
    EXPECT_EQ(result, mock::EC_E_NOERROR);
  }

  EXPECT_EQ(ctx.GetCycleCount(), 10);
}

/**
 * @brief Test mock EC-Master cyclic job without OP state
 */
TEST_F(MockedIntegrationTest, MockEcMasterCyclicJobInvalidState) {
  auto& ctx = mock::GetMockContext();

  // Initialize and configure but don't reach OP
  EXPECT_EQ(mock::MockEcatInitMaster(nullptr), mock::EC_E_NOERROR);
  EXPECT_EQ(mock::MockEcatConfigureMaster("/tmp/test.xml"), mock::EC_E_NOERROR);

  // Try to execute cyclic job in INIT state
  auto result = mock::MockEcatExecJob(0, nullptr);
  EXPECT_EQ(result, mock::EC_E_INVALIDSTATE);
  EXPECT_EQ(ctx.GetCycleCount(), 0);
}

// ====================================================================================
// Mock Context Management Tests
// ====================================================================================

/**
 * @brief Test mock context reset
 */
TEST_F(MockedIntegrationTest, MockContextReset) {
  auto& ctx = mock::GetMockContext();

  // Setup some state
  EXPECT_EQ(mock::MockEcatInitMaster(nullptr), mock::EC_E_NOERROR);
  EXPECT_EQ(mock::MockEcatConfigureMaster("/tmp/test.xml"), mock::EC_E_NOERROR);
  ctx.SetAdapterName("eth0");
  ctx.SetEniFile("/tmp/test.xml");

  EXPECT_TRUE(ctx.IsInitialized());
  EXPECT_TRUE(ctx.IsConfigured());

  // Reset
  ctx.Reset();

  EXPECT_FALSE(ctx.IsInitialized());
  EXPECT_FALSE(ctx.IsConfigured());
  EXPECT_EQ(ctx.GetState(), mock::EcMasterState::UNKNOWN);
  EXPECT_EQ(ctx.GetCycleCount(), 0);
  EXPECT_TRUE(ctx.GetAdapterName().empty());
  EXPECT_TRUE(ctx.GetEniFile().empty());
}

/**
 * @brief Test mock EC-Master deinitialization
 */
TEST_F(MockedIntegrationTest, MockEcMasterDeinitialization) {
  auto& ctx = mock::GetMockContext();

  // Setup
  EXPECT_EQ(mock::MockEcatInitMaster(nullptr), mock::EC_E_NOERROR);
  EXPECT_EQ(mock::MockEcatConfigureMaster("/tmp/test.xml"), mock::EC_E_NOERROR);

  EXPECT_TRUE(ctx.IsInitialized());

  // Deinitialize
  auto result = mock::MockEcatDeinitMaster();
  EXPECT_EQ(result, mock::EC_E_NOERROR);

  EXPECT_FALSE(ctx.IsInitialized());
  EXPECT_FALSE(ctx.IsConfigured());
}

// ====================================================================================
// Operator Pipeline Integration Tests (Structure)
// ====================================================================================

/**
 * @brief Test creating full operator pipeline
 *
 * Note: Full data flow testing would require running the application
 * with proper operator connections and compute() execution.
 */
TEST_F(MockedIntegrationTest, CreateOperatorPipeline) {
  auto tx_op = fragment_->make_operator<HcDataTxOp>("tx_source");
  auto rx_op = fragment_->make_operator<HcDataRxOp>("rx_sink");

  ASSERT_NE(tx_op, nullptr);
  ASSERT_NE(rx_op, nullptr);

  EXPECT_EQ(tx_op->name(), "tx_source");
  EXPECT_EQ(rx_op->name(), "rx_sink");
}

/**
 * @brief Test operator setup in pipeline
 */
TEST_F(MockedIntegrationTest, SetupOperatorPipeline) {
  auto tx_op = fragment_->make_operator<HcDataTxOp>("tx_source");
  auto rx_op = fragment_->make_operator<HcDataRxOp>("rx_sink");

  auto tx_spec = std::make_shared<holoscan::OperatorSpec>(fragment_.get());
  auto rx_spec = std::make_shared<holoscan::OperatorSpec>(fragment_.get());

  EXPECT_NO_THROW({
    tx_op->setup(*tx_spec);
    rx_op->setup(*rx_spec);
  });
}

// ====================================================================================
// Configuration Integration Tests
// ====================================================================================

/**
 * @brief Test that configuration helpers work with mocked environment
 */
TEST_F(MockedIntegrationTest, ConfigurationHelpersIntegration) {
  auto config = test::CreateDefaultTestConfig();

  EXPECT_TRUE(test::IsConfigReasonable(config));
  EXPECT_FALSE(config.adapter_name.empty());
  EXPECT_FALSE(config.eni_file.empty());
  EXPECT_GT(config.cycle_time_us, 0);
}

/**
 * @brief Test configuration validation in integration context
 */
TEST_F(MockedIntegrationTest, ConfigurationValidation) {
  auto config = test::CreateDefaultTestConfig();
  EXPECT_TRUE(test::IsConfigReasonable(config));

  auto invalid_config = test::CreateInvalidTestConfig();
  EXPECT_FALSE(test::IsConfigReasonable(invalid_config));
}

// ====================================================================================
// Mock State Verification Tests
// ====================================================================================

/**
 * @brief Test getting mock master state
 */
TEST_F(MockedIntegrationTest, GetMockMasterState) {
  auto& ctx = mock::GetMockContext();

  EXPECT_EQ(mock::MockEcatGetMasterState(), static_cast<uint32_t>(mock::EcMasterState::UNKNOWN));

  EXPECT_EQ(mock::MockEcatInitMaster(nullptr), mock::EC_E_NOERROR);
  EXPECT_EQ(mock::MockEcatGetMasterState(), static_cast<uint32_t>(mock::EcMasterState::INIT));

  EXPECT_EQ(mock::MockEcatConfigureMaster("/tmp/test.xml"), mock::EC_E_NOERROR);
  EXPECT_EQ(mock::MockEcatSetMasterState(3000, static_cast<uint32_t>(mock::EcMasterState::PREOP)),
            mock::EC_E_NOERROR);
  EXPECT_EQ(mock::MockEcatGetMasterState(), static_cast<uint32_t>(mock::EcMasterState::PREOP));
}

/**
 * @brief Test mock context adapter and ENI storage
 */
TEST_F(MockedIntegrationTest, MockContextConfigurationStorage) {
  auto& ctx = mock::GetMockContext();

  ctx.SetAdapterName("eth0");
  ctx.SetEniFile("/tmp/test.xml");

  EXPECT_EQ(ctx.GetAdapterName(), "eth0");
  EXPECT_EQ(ctx.GetEniFile(), "/tmp/test.xml");
}

}  // namespace integration_test
}  // namespace holocat
