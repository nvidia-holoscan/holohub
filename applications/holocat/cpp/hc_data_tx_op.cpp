/**
 * @file hc_data_tx_op.cpp
 * @brief HoloCat Data Transmit Operator Implementation
 * 
 * This file implements the HcDataTxOp operator that generates incrementing
 * counter data for testing and demonstration purposes.
 */


#include <holoscan/holoscan.hpp>

#include "hc_data_tx_op.hpp"


namespace holocat {

void HcDataTxOp::setup(holoscan::OperatorSpec& spec) {
  // Configure output port for emitting counter data
  spec.output<int>("count_out");
  
  HOLOSCAN_LOG_INFO("HcDataTxOp: Setup complete - configured output port 'count_out'");
}

void HcDataTxOp::compute(holoscan::InputContext& op_input, 
                         holoscan::OutputContext& op_output, 
                         holoscan::ExecutionContext& context) {

  // Increment counter
  counter_ = (counter_ + 1) % kMaxCount;

  // Emit current counter value
  op_output.emit<int>(counter_, "count_out");
  
  // Log every 50 counts to avoid spam
  if (counter_ % 50 == 0) {
    HOLOSCAN_LOG_DEBUG("HcDataTxOp: 50x Emitted count = {}", counter_);
  }
}

} // namespace holocat

/*-END OF SOURCE FILE--------------------------------------------------------*/
