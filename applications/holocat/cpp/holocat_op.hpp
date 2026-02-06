/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file holocat_op.hpp
 * @brief HoloCat EtherCAT Operator for Holoscan SDK
 *
 * Based on EC-Master demo from acontis technologies GmbH
 *
 * This file defines the HolocatOp operator that integrates EtherCAT real-time
 * communication with NVIDIA's Holoscan framework.
 */

#ifndef INC_HOLOCAT_H
#define INC_HOLOCAT_H 1

#define INCLUDE_EC_MASTER

#include <cstdint>
#include <future>
#include <holoscan/holoscan.hpp>
#include <string>
#include <time.h>

#include "EcMaster.h"
#include "holocat_config.hpp"

namespace holocat {

/**
 * @brief HoloCat EtherCAT Operator
 *
 * A Holoscan operator that provides real-time EtherCAT communication capabilities.
 * This operator manages the complete EtherCAT master lifecycle including:
 * - Network configuration and initialization
 * - State machine management (INIT -> PREOP -> SAFEOP -> OP)
 * - Cyclic process data exchange
 * - Asynchronous state transitions to maintain real-time performance
 *
 * The operator runs as a source operator with periodic scheduling, executing
 * EtherCAT communication cycles at regular intervals defined by PeriodicCondition.
 */
class HolocatOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(HolocatOp)

  // Constructor that accepts configuration
  HolocatOp() = default;

  // Set configuration
  void set_config(const HolocatConfig& config) { config_ = config; }

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
  void stop() override;

 private:
  // Configuration
  HolocatConfig config_;

  // State machine for functions that need to be called while bus communication is also running.
  void BusStartupStateMachine();

  // Master state change timeout in ms.
  static constexpr EC_T_DWORD kEthercatStateChangeTimeoutMs = 3000;

  // Job task member variables (moved from global)
  EC_T_MEMREQ_DESC oPdMemorySize_{0, 0};

  // EtherCAT constants
  static constexpr int kTwoBytes = 16;  // 2 bytes bit length

  // EtherCAT parameter structs (moved from global)
  EC_T_LOG_PARMS log_parms_;
  EC_T_LINK_PARMS_SOCKRAW sockraw_params_;
  EC_T_OS_PARMS os_parms_;
  EC_T_INIT_MASTER_PARMS ec_master_init_parms_;

  // EtherCAT state variables
  EC_T_BOOL bHasConfig_ = EC_FALSE;

  // Startup state machine variables
  enum class StartupState {
    INIT,
    CONFIGURE_NETWORK,
    GET_PD_MEMORY_SIZE,
    SET_MASTER_INIT,
    SET_MASTER_PREOP,
    SET_MASTER_OP,
    OPERATIONAL,
    STOP_REQUESTED
  };
  StartupState startup_state_ = StartupState::INIT;

  // Async state transition management
  std::future<EC_T_DWORD> pending_state_transition_;
  bool state_transition_in_progress_ = false;
  bool stop_requested_ = false;
  // Helper methods
  void InitializeEthercatParams();
  EC_T_DWORD ConfigureNetwork();
  static EC_T_DWORD LogWrapper(struct _EC_T_LOG_CONTEXT* pContext, EC_T_DWORD dwLogMsgSeverity,
                               const EC_T_CHAR* szFormat, ...);

  // Process data output value
  int outval_ = 0;

  // Performance tracking
  struct timespec t_last {
    0, 0
  };
};

}  // namespace holocat
#endif /* INC_HOLOCAT_H */

/*-END OF SOURCE FILE--------------------------------------------------------*/
