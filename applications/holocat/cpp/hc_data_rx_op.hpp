/**
 * @file hc_data_rx_op.hpp
 * @brief HoloCat Data Receive Operator for Holoscan SDK
 * 
 * This file defines the HcDataRxOp operator that receives integer data from the ECat bus.
 */

#ifndef INC_HC_DATA_RX_OP_H
#define INC_HC_DATA_RX_OP_H 1

#include <holoscan/holoscan.hpp>
#include <cstdint>

namespace holocat {

/**
 * @brief HoloCat Data Receive Operator
 * 
 * A Holoscan operator that receives integer data from the ECat bus.
 * This operator acts as a data receiver, receiving integer values from the ECat bus.
 * 
 * The operator runs as a receiver operator and can be used for testing data
 * flow in Holoscan applications or as a simple data receiver.
 */
class HcDataRxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(HcDataRxOp);
  
  /**
   * @brief Default constructor
   */
  HcDataRxOp() = default;
  
  /**
   * @brief Setup the operator's input/output specifications
   * @param spec The operator specification to configure
   */
  void setup(holoscan::OperatorSpec& spec) override;
  
  /**
   * @brief Compute method called on each execution cycle
   * @param op_input Input context for receiving process data from EtherCAT
   * @param op_output Output context (unused for this receiver operator)
   * @param context Execution context
   */
  void compute(holoscan::InputContext& op_input, 
               holoscan::OutputContext& op_output, 
               holoscan::ExecutionContext& context) override;

 private:
    // last received count value
  int last_count_ = 0;
};

} // namespace holocat

#endif /* INC_HC_DATA_RX_OP_H */

/*-END OF SOURCE FILE--------------------------------------------------------*/
