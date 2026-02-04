/**
 * @file test_helpers.hpp
 * @brief Common Test Utilities for HoloCat
 * 
 * Provides test utilities and helpers for HoloCat tests.
 */

#ifndef HOLOCAT_TEST_HELPERS_HPP
#define HOLOCAT_TEST_HELPERS_HPP

#include <memory>
#include <string>
#include "holoscan/holoscan.hpp"
#include "holocat_config.hpp"

namespace holocat {
namespace test {

/**
 * @brief Create a default valid HolocatConfig for testing
 */
inline HolocatConfig CreateDefaultTestConfig() {
  HolocatConfig config;
  config.adapter_name = "eth0";
  config.eni_file = "/tmp/test_eni.xml";
  config.rt_priority = 39;
  config.job_thread_priority = 98;
  config.enable_rt = false;
  config.cycle_time_us = 1000;
  return config;
}

/**
 * @brief Create an invalid HolocatConfig for error testing
 */
inline HolocatConfig CreateInvalidTestConfig() {
  HolocatConfig config;
  config.adapter_name = "";
  config.cycle_time_us = 0;
  return config;
}

/**
 * @brief Validate that a HolocatConfig has reasonable values
 */
inline bool IsConfigReasonable(const HolocatConfig& config) {
  if (config.adapter_name.empty()) return false;
  if (config.adapter_name.length() > 255) return false;
  if (config.cycle_time_us == 0) return false;
  if (config.rt_priority > 99) return false;
  if (config.job_thread_priority > 99) return false;
  return true;
}

}  // namespace test
}  // namespace holocat

#endif  // HOLOCAT_TEST_HELPERS_HPP
