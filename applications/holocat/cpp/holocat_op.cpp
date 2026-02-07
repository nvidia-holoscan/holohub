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
 * @file holocat_op.cpp
 * @brief HoloCat EtherCAT Operator Implementation
 *
 * Based on EC-Master demo application from acontis technologies GmbH
 */

// System includes
#include <chrono>
#include <future>
#include <iostream>
#include <sched.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/utsname.h>
#include <time.h>

// Third-party includes
#include <holoscan/holoscan.hpp>

// EtherCAT SDK includes
#include "EcOs.h"
#include "EcOsPlatform.h"
#include "EcType.h"

// Local includes
#include "holocat_op.hpp"

namespace holocat {
// EtherCAT logging callback function - maps EC-Master log levels to Holoscan
EC_T_DWORD HolocatOp::LogWrapper(struct _EC_T_LOG_CONTEXT* pContext, EC_T_DWORD dwLogMsgSeverity,
                                 const EC_T_CHAR* szFormat, ...) {
  EC_T_VALIST vaArgs;
  EC_VASTART(vaArgs, szFormat);

  char buffer[1024];
  const char* prefix = "[EC-Master] ";
  size_t prefix_len = strlen(prefix);
  strncpy(buffer, prefix, sizeof(buffer) - 1);
  int written = vsnprintf(buffer + prefix_len, sizeof(buffer) - prefix_len - 1, szFormat, vaArgs);

  // Check for truncation
  if (written >= static_cast<int>(sizeof(buffer) - prefix_len - 1)) {
    // Message was truncated - append indicator
    const char* truncated = "...[truncated]";
    size_t trunc_len = strlen(truncated);
    if (sizeof(buffer) > trunc_len + 1) {
      strncpy(buffer + sizeof(buffer) - trunc_len - 1, truncated, trunc_len + 1);
    }
  }

  EC_VAEND(vaArgs);

  if (dwLogMsgSeverity == EC_LOG_LEVEL_SILENT) {
    // Don't log
  } else if (dwLogMsgSeverity == EC_LOG_LEVEL_CRITICAL) {
    HOLOSCAN_LOG_CRITICAL("{}", buffer);
  } else if (dwLogMsgSeverity == EC_LOG_LEVEL_ERROR) {
    HOLOSCAN_LOG_ERROR("{}", buffer);
  } else if (dwLogMsgSeverity == EC_LOG_LEVEL_WARNING) {
    HOLOSCAN_LOG_WARN("{}", buffer);
  } else if (dwLogMsgSeverity == EC_LOG_LEVEL_INFO) {
    HOLOSCAN_LOG_INFO("{}", buffer);
  } else {  // VERBOSE_CYC (8), ANY (1)
    HOLOSCAN_LOG_INFO("{}", buffer);
  }

  return EC_E_NOERROR;
}

// Initialize EtherCAT parameters
void HolocatOp::InitializeEthercatParams() {
  // Initialize logging parameters with LogWrapper callback
  log_parms_ = {EC_LOG_LEVEL_VERBOSE, LogWrapper, EC_NULL};

  // EtherCat link/interface parameters
  OsMemset(&sockraw_params_, 0, sizeof(EC_T_LINK_PARMS_SOCKRAW));
  sockraw_params_.linkParms.dwSignature = EC_LINK_PARMS_SIGNATURE_SOCKRAW;
  sockraw_params_.linkParms.dwSize = sizeof(EC_T_LINK_PARMS_SOCKRAW);
  sockraw_params_.linkParms.LogParms.dwLogLevel = log_parms_.dwLogLevel;
  sockraw_params_.linkParms.LogParms.pfLogMsg = log_parms_.pfLogMsg;
  sockraw_params_.linkParms.LogParms.pLogContext = log_parms_.pLogContext;
  OsStrncpy(sockraw_params_.linkParms.szDriverIdent,
            EC_LINK_PARMS_IDENT_SOCKRAW,
            EC_DRIVER_IDENT_MAXLEN);  // Driver identity
  sockraw_params_.linkParms.dwInstance = 1;
  sockraw_params_.linkParms.eLinkMode = EcLinkMode_POLLING;  // POLLING mode - no receiver thread
  sockraw_params_.linkParms.dwIstPriority = 0;               // Not used in polling mode
  // Get adapter name from config or use default

  if (config_.adapter_name.empty()) {
    throw std::runtime_error("Adapter name is empty");
  }
  if (size_t(config_.adapter_name.length()) >= EC_SOCKRAW_ADAPTER_NAME_MAXLEN - 1) {
    throw std::runtime_error("Adapter name is too long: " + config_.adapter_name);
  } else {
    // log message: "Adapter name: {}"
    HOLOSCAN_LOG_INFO("Adapter name: {}", config_.adapter_name);
  }

  OsStrncpy(
      sockraw_params_.szAdapterName, config_.adapter_name.c_str(), EC_SOCKRAW_ADAPTER_NAME_MAXLEN);
  sockraw_params_.bDisableForceBroadcast = EC_TRUE;
  sockraw_params_.bReplacePaddingWithNopCmd = EC_FALSE;
  sockraw_params_.bUsePacketMmapRx = EC_TRUE;
  sockraw_params_.bSetCoalescingParms = EC_FALSE;
  sockraw_params_.bSetPromiscuousMode = EC_FALSE;

  // OS parameters
  OsMemset(&os_parms_, 0, sizeof(EC_T_OS_PARMS));
  os_parms_.dwSignature = EC_OS_PARMS_SIGNATURE;
  os_parms_.dwSize = sizeof(EC_T_OS_PARMS);
  os_parms_.pLogParms = &(log_parms_);
  os_parms_.dwSupportedFeatures = 0xFFFFFFFF;
  // Don't configure mutex - avoids pthread conflicts
  os_parms_.PlatformParms.bConfigMutex = EC_FALSE;
  os_parms_.PlatformParms.nMutexType = PTHREAD_MUTEX_RECURSIVE;
  os_parms_.PlatformParms.nMutexProtocol = PTHREAD_PRIO_NONE;
  OsInit(&os_parms_);

  // EC-Master init parameters from config
  OsMemset(&ec_master_init_parms_, 0, sizeof(EC_T_INIT_MASTER_PARMS));
  ec_master_init_parms_.dwSignature = ATECAT_SIGNATURE;
  ec_master_init_parms_.dwSize = sizeof(EC_T_INIT_MASTER_PARMS);
  ec_master_init_parms_.pOsParms = &os_parms_;
  ec_master_init_parms_.pLinkParms = (EC_T_LINK_PARMS*)&sockraw_params_;
  ec_master_init_parms_.pLinkParmsRed = EC_NULL;
  ec_master_init_parms_.dwBusCycleTimeUsec = config_.cycle_time_us;
  ec_master_init_parms_.dwMaxAcycFramesQueued = config_.max_acyc_frames;
}

// Network configuration library call wrapper
EC_T_DWORD HolocatOp::ConfigureNetwork() {
  EC_T_DWORD dwRes = EC_E_NOERROR;
  // setup ENI configuration
  EC_T_CNF_TYPE eCnfType = eCnfType_Filename;
  constexpr size_t kMaxEniFileLength = 256;
  EC_T_CHAR szEniFilename[kMaxEniFileLength];
  // ENI file path from config or environment variable
  // Get ENI file path from config or use default
  std::string eni_file = config_.eni_file;
  if (size_t(eni_file.length()) >= kMaxEniFileLength - 1) {
    HOLOSCAN_LOG_ERROR(
        "ENI file path is too long: {} (max length: {})", eni_file, kMaxEniFileLength);
    return EC_E_INVALIDPARM;
  }
  OsSnprintf(szEniFilename, sizeof(szEniFilename), "%s", eni_file.c_str());
  EC_T_BYTE* pbyCnfData = (EC_T_BYTE*)szEniFilename;
  EC_T_DWORD dwCnfDataLen = (EC_T_DWORD)OsStrlen(szEniFilename);

  // configure network
  dwRes = ecatConfigureNetwork(eCnfType, pbyCnfData, dwCnfDataLen);
  if (dwRes != EC_E_NOERROR) {
    HOLOSCAN_LOG_ERROR("Cannot configure EtherCAT-Master: {} (0x{:x})", ecatGetText(dwRes), dwRes);
    return dwRes;
  }
  bHasConfig_ = EC_TRUE;
  return dwRes;
}

void HolocatOp::setup(holoscan::OperatorSpec& spec) {
  t_last.tv_sec = 0;
  t_last.tv_nsec = 0;
  spec.input<int>("count_in").condition(holoscan::ConditionType::kNone);    // Make input optional
  spec.output<int>("count_out").condition(holoscan::ConditionType::kNone);  // Make output optional
}

void HolocatOp::start() {
  HOLOSCAN_LOG_INFO("HolocatOp: Starting EtherCAT Master");
  InitializeEthercatParams();

  // Setup EtherCAT Master
  EC_T_DWORD dwRes = ecatInitMaster(&ec_master_init_parms_);
  if (dwRes != EC_E_NOERROR) {
    HOLOSCAN_LOG_ERROR("Cannot initialize EtherCAT-Master: {} (0x{:x})", ecatGetText(dwRes), dwRes);
    throw std::runtime_error("Cannot initialize EtherCAT-Master: " +
                             std::string(ecatGetText(dwRes)));
  }
  HOLOSCAN_LOG_INFO("Master initialized");
}

void HolocatOp::stop() {
  HOLOSCAN_LOG_INFO("HolocatOp: stopping EtherCAT Master");
  stop_requested_ = true;
  // Don't wait for state transition to complete, since that may never happen.
  // Just wait for 1 second for a best-effort approach.
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // deinitialize master
  ecatDeinitMaster();
}

void HolocatOp::BusStartupStateMachine() {
  EC_T_DWORD dwRes = EC_E_NOERROR;

  if (stop_requested_) {
    if (!state_transition_in_progress_) {
      // reset master to INIT
      startup_state_ = StartupState::STOP_REQUESTED;
    }
  }

  switch (startup_state_) {
    case StartupState::INIT:
      startup_state_ = StartupState::CONFIGURE_NETWORK;
      break;

    case StartupState::CONFIGURE_NETWORK:
      HOLOSCAN_LOG_INFO("EtherCAT: Configuring network");
      dwRes = ConfigureNetwork();
      if (dwRes != EC_E_NOERROR) {
        HOLOSCAN_LOG_ERROR(
            "EtherCAT: Failed to configure network: {} (0x{:x})", ecatGetText(dwRes), dwRes);
        return;
      }
      startup_state_ = StartupState::GET_PD_MEMORY_SIZE;
      break;

    case StartupState::GET_PD_MEMORY_SIZE:
      HOLOSCAN_LOG_INFO("EtherCAT: Getting process data memory size");
      dwRes = ecatIoCtl(EC_IOCTL_GET_PDMEMORYSIZE,
                        EC_NULL,
                        0,
                        &oPdMemorySize_,
                        sizeof(EC_T_MEMREQ_DESC),
                        EC_NULL);
      if (dwRes != EC_E_NOERROR) {
        HOLOSCAN_LOG_ERROR(
            "EtherCAT: Cannot get process data size: {} (0x{:x})", ecatGetText(dwRes), dwRes);
        return;
      }
      startup_state_ = StartupState::SET_MASTER_INIT;
      break;

    case StartupState::SET_MASTER_INIT:
      if (!state_transition_in_progress_) {
        HOLOSCAN_LOG_INFO("BusStartupStateMachine: Setting master state to INIT");
        pending_state_transition_ = std::async(std::launch::async, [this]() {
          return ecatSetMasterState(kEthercatStateChangeTimeoutMs, eEcatState_INIT);
        });
        state_transition_in_progress_ = true;
        return;
      }
      if (pending_state_transition_.wait_for(std::chrono::seconds(0)) ==
          std::future_status::ready) {
        state_transition_in_progress_ = false;
        dwRes = pending_state_transition_.get();
        if (dwRes != EC_E_NOERROR) {
          HOLOSCAN_LOG_ERROR(
              "Cannot set master state to INIT: {} (0x{:x})", ecatGetText(dwRes), dwRes);
          return;
        }
        startup_state_ = StartupState::SET_MASTER_PREOP;
      }
      break;

    case StartupState::SET_MASTER_PREOP:
      if (!state_transition_in_progress_) {
        HOLOSCAN_LOG_INFO("BusStartupStateMachine: Setting master state to PREOP");
        pending_state_transition_ = std::async(std::launch::async, [this]() {
          return ecatSetMasterState(kEthercatStateChangeTimeoutMs, eEcatState_PREOP);
        });
        state_transition_in_progress_ = true;
        return;
      }
      if (pending_state_transition_.wait_for(std::chrono::seconds(0)) ==
          std::future_status::ready) {
        state_transition_in_progress_ = false;
        dwRes = pending_state_transition_.get();
        if (dwRes != EC_E_NOERROR) {
          HOLOSCAN_LOG_ERROR(
              "Cannot set master state to PREOP: {} (0x{:x})", ecatGetText(dwRes), dwRes);
          return;
        }
        startup_state_ = StartupState::SET_MASTER_OP;
      }
      break;

    case StartupState::SET_MASTER_OP:
      if (bHasConfig_) {
        if (!state_transition_in_progress_) {
          HOLOSCAN_LOG_INFO("BusStartupStateMachine: Setting master state to OP");
          pending_state_transition_ = std::async(std::launch::async, [this]() {
            return ecatSetMasterState(kEthercatStateChangeTimeoutMs, eEcatState_OP);
          });
          state_transition_in_progress_ = true;
          return;
        }
        if (pending_state_transition_.wait_for(std::chrono::seconds(0)) ==
            std::future_status::ready) {
          state_transition_in_progress_ = false;
          dwRes = pending_state_transition_.get();
          if (dwRes != EC_E_NOERROR) {
            HOLOSCAN_LOG_ERROR(
                "Cannot set master state to OP: {} (0x{:x})", ecatGetText(dwRes), dwRes);
            return;
          }
          startup_state_ = StartupState::OPERATIONAL;
        }
      } else {
        HOLOSCAN_LOG_INFO("BusStartupStateMachine: No config available, skipping OP state");
        startup_state_ = StartupState::OPERATIONAL;
      }
      break;

    case StartupState::OPERATIONAL:
      // Startup complete, nothing to do
      break;

    case StartupState::STOP_REQUESTED:
      if (!state_transition_in_progress_) {
        HOLOSCAN_LOG_INFO("BusStartupStateMachine: STOP");
        pending_state_transition_ = std::async(std::launch::async, [this]() {
          return ecatSetMasterState(kEthercatStateChangeTimeoutMs, eEcatState_INIT);
        });
        state_transition_in_progress_ = true;
        return;
      }
      if (pending_state_transition_.wait_for(std::chrono::seconds(0)) ==
          std::future_status::ready) {
        state_transition_in_progress_ = false;
        dwRes = pending_state_transition_.get();
        if (dwRes != EC_E_NOERROR) {
          HOLOSCAN_LOG_ERROR(
              "Cannot set master state to INIT: {} (0x{:x})", ecatGetText(dwRes), dwRes);
          return;
        }
        startup_state_ = StartupState::OPERATIONAL;
      }
      break;
  }
}

void HolocatOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                        holoscan::ExecutionContext& context) {
  EC_T_STATE eMasterState = ecatGetMasterState();
  EC_T_DWORD dwRes = EC_E_NOERROR;

  // Read count_in input if available
  int count_in_value = 0;
  bool has_count_input = false;

  auto count_in_msg = op_input.receive<int>("count_in");
  if (count_in_msg) {
    count_in_value = count_in_msg.value();
    has_count_input = true;
    HOLOSCAN_LOG_DEBUG("HolocatOp: received count_in: {}", count_in_value);
  }

  // Step through asynchronous bus initialization states.
  BusStartupStateMachine();

  // PD output variables
  unsigned short int inval = 0;
  bool has_inval = false;

  // POLLING mode: Process received frames directly (no interrupt/event)
  dwRes = ecatExecJob(eUsrJob_ProcessAllRxFrames, EC_NULL);
  if (dwRes != EC_E_NOERROR) {
    HOLOSCAN_LOG_WARN("Processjobs failed: {} (0x{:x})", ecatGetText(dwRes), dwRes);
  }
  EC_UNREFPARM(dwRes);

  // read input values from process data
  EC_T_BYTE* pbyPdIn = ecatGetProcessImageInputPtr();
  if ((pbyPdIn != EC_NULL) &&
      (oPdMemorySize_.dwPDInSize * 8U >= config_.dio_in_offset + kTwoByteBitLength)) {
    EC_COPYBITS((EC_T_BYTE*)&inval, 0, pbyPdIn, config_.dio_in_offset, 16);
    has_inval = true;
  }

  // process data
  if ((eEcatState_SAFEOP == eMasterState) || (eEcatState_OP == eMasterState)) {
    if (has_count_input) {
      outval_ = count_in_value;
      EC_T_BYTE* pbyPdOut = ecatGetProcessImageOutputPtr();
      if ((pbyPdOut != EC_NULL) &&
          (oPdMemorySize_.dwPDOutSize * 8U >= config_.dio_out_offset + kTwoByteBitLength)) {
        EC_COPYBITS(pbyPdOut, config_.dio_out_offset, (EC_T_BYTE*)&outval_, 0, kTwoByteBitLength);
      }
    }
  } else {
    HOLOSCAN_LOG_DEBUG(".");
  }

  EC_T_USER_JOB_PARMS oJobParms;
  OsMemset(&oJobParms, 0, sizeof(EC_T_USER_JOB_PARMS));

  // write output values of current cycle, by sending all cyclic frames
  dwRes = ecatExecJob(eUsrJob_SendAllCycFrames, EC_NULL);
  EC_UNREFPARM(dwRes);

  // execute some administrative jobs. No bus traffic is performed by this function
  // Required for master state machine to run
  dwRes = ecatExecJob(eUsrJob_MasterTimer, EC_NULL);
  EC_UNREFPARM(dwRes);

  // send queued acyclic EtherCAT frames
  dwRes = ecatExecJob(eUsrJob_SendAcycFrames, EC_NULL);
  EC_UNREFPARM(dwRes);

  // Performance monitoring and diagnostics
  struct timespec t_now;
  clock_gettime(CLOCK_MONOTONIC, &t_now);

  // Only initialize t_last on first run (if it is zeroed)
  if (t_last.tv_sec == 0 && t_last.tv_nsec == 0) {
    t_last.tv_sec = t_now.tv_sec;
    t_last.tv_nsec = t_now.tv_nsec;
  }

  // Calculate sample delay with wraparound handling at 256
  // enforce inval in range 0-255
  constexpr int kMaxCount = 256;
  inval = inval % kMaxCount;
  int sample_delay = outval_ - inval;
  if (sample_delay < 0) {
    sample_delay += kMaxCount;
  }

  // Calculate elapsed time in milliseconds
  double elapsed_time_ms = (t_now.tv_sec - t_last.tv_sec) * 1000.0 +
                           (double)(t_now.tv_nsec - t_last.tv_nsec) / 1000000.0;
  t_last = t_now;

  // Log process data when in operational states
  if ((eEcatState_SAFEOP == eMasterState) || (eEcatState_OP == eMasterState)) {
    // emit count value read from ECat bus
    if (has_inval) {
      op_output.emit<int>(inval, "count_out");
    }
    HOLOSCAN_LOG_DEBUG(
        "Elapsed time: {:.2f} ms\t\t Process data out: 0x{:02x} \tin: 0x{:02x} \t"
        "sample delay: {}",
        elapsed_time_ms,
        outval_,
        inval,
        sample_delay);
  }
}

}  // namespace holocat
