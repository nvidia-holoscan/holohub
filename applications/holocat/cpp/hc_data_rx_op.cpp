/**
 * @file hc_data_rx_op.cpp
 * @brief HoloCat Data Receive Operator
 * 
 * HoloCat Data Receive Operator
 * 
 * This file implements the HcDataRxOp operator that receives counter data
 * for testing and demonstration purposes.
 */

#include <holoscan/holoscan.hpp>

#include "hc_data_rx_op.hpp"


namespace holocat {

void HcDataRxOp::setup(holoscan::OperatorSpec& spec) {
  // Configure input port for receiving counter data
  spec.input<int>("count_in");
  
  HOLOSCAN_LOG_INFO("HcDataRxOp: Setup complete - configured input port 'count_in'");
}

void HcDataRxOp::compute(holoscan::InputContext& op_input, 
                         holoscan::OutputContext& op_output, 
                         holoscan::ExecutionContext& context) {

  // Receive count value from ECat bus
  auto maybe_count = op_input.receive<int>("count_in");
  if (!maybe_count) {
    HOLOSCAN_LOG_ERROR("HcDataRxOp: Failed to receive count from ECat bus");
    return;
  }
  last_count_ = maybe_count.value();
  HOLOSCAN_LOG_INFO("HcDataRxOp: Received count: {}", last_count_);
}

} // namespace holocat

/*-END OF SOURCE FILE--------------------------------------------------------*/
