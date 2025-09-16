#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <fstream>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/io_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include <holoscan/core/execution_context.hpp>

namespace holoscan::ops {

/**
 * @brief Operator that saves received frames to disk
 */
class FrameSaverOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FrameSaverOp)

  FrameSaverOp() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  Parameter<std::string> output_dir_;
  Parameter<std::string> base_filename_;
  Parameter<bool> save_as_raw_;
  
  std::string current_file_;
  std::ofstream output_file_;
  uint64_t frame_count_ = 0;
};

}  // namespace holoscan::ops 