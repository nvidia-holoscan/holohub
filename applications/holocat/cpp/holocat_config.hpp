/**
 * @file holocat_config.hpp
 * @brief HoloCat application and EtherCAT configuration struct
 * 
 * Defines the HolocatConfig structure used for configuring EtherCAT
 * and HoloCat integration within the NVIDIA Holoscan framework.
 */

#ifndef INC_HOLOCAT_CONFIG_H
#define INC_HOLOCAT_CONFIG_H 1

#include <string>
#include <cstdint>

namespace holocat {

  // Configuration structure for Holoscan integration
struct HolocatConfig {
    std::string adapter_name;
    std::string eni_file;
    uint32_t rt_priority;
    uint32_t job_thread_priority;
    bool enable_rt;
    uint32_t dio_out_offset;
    uint32_t dio_in_offset;
    uint32_t max_acyc_frames;
    uint32_t job_thread_stack_size;
    std::string log_level;
    uint64_t cycle_time_us;
    // For validation error reporting
    std::string error_message;
};

}
#endif /* INC_HOLOCAT_CONFIG_H */

