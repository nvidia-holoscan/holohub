/**
 * @file hc_data_tx_op.hpp
 * @brief HoloCat Data Transmit Operator for Holoscan SDK
 * 
 * This file defines the HcDataTxOp operator that generates incrementing
 * counter data for testing and demonstration purposes.
 */

#ifndef INC_HC_DATA_TX_OP_H
#define INC_HC_DATA_TX_OP_H 1

#include <holoscan/holoscan.hpp>
#include <cstdint>

namespace holocat {

/**
 * @brief HoloCat Data Transmit Operator
 * 
 * A Holoscan operator that generates incrementing counter data from 0 to 255.
 * This operator acts as a data source, emitting integer values that increment
 * on each compute cycle. When the counter reaches 255, it wraps back to 0.
 * 
 * The operator runs as a source operator and can be used for testing data
 * flow in Holoscan applications or as a simple data generator.
 */
class HcDataTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(HcDataTxOp);
  
  /**
   * @brief Default constructor
   */
  HcDataTxOp() = default;
  
  /**
   * @brief Setup the operator's input/output specifications
   * @param spec The operator specification to configure
   */
  void setup(holoscan::OperatorSpec& spec) override;
  
  /**
   * @brief Compute method called on each execution cycle
   * @param op_input Input context (unused for this source operator)
   * @param op_output Output context for emitting counter data
   * @param context Execution context
   */
  void compute(holoscan::InputContext& op_input, 
               holoscan::OutputContext& op_output, 
               holoscan::ExecutionContext& context) override;

 private:
    // counter from 0 to kMaxCount-1 for output on EtherCAT bus
  int counter_ = 0;
  static constexpr int kMaxCount = 256;
};

} // namespace holocat

#endif /* INC_HC_DATA_TX_OP_H */

/*-END OF SOURCE FILE--------------------------------------------------------*/
