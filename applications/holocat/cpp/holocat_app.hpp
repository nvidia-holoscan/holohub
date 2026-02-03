/**
 * @file holocat_app.hpp
 * @brief HoloCat Application - EtherCAT integration with Holoscan
 */

#pragma once

#include <holoscan/holoscan.hpp>
#include "holocat_op.hpp"
#include "holocat_config.hpp"

namespace holocat {

/**
 * @brief HoloCat Application Class
 * 
 * Orchestrates EtherCAT integration with periodic HolocatOp scheduling.
 */
class HolocatApp : public holoscan::Application {
public:
  void compose() override;
  
  HolocatConfig extract_config();

private:
  bool validate_config(HolocatConfig& config);
};

} // namespace holocat
