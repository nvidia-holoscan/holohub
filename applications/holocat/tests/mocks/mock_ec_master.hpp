/**
 * @file mock_ec_master.hpp
 * @brief Mock EC-Master SDK Interface
 *
 * Provides lightweight mock implementation of EC-Master SDK
 * functions for testing without hardware.
 */

#ifndef MOCK_EC_MASTER_HPP
#define MOCK_EC_MASTER_HPP

#include <cstdint>
#include <map>
#include <string>

namespace holocat {
namespace mock {

// Mock EC-Master state machine states
enum class EcMasterState { UNKNOWN = 0, INIT = 1, PREOP = 2, SAFEOP = 4, OP = 8 };

// Mock EC-Master error codes
constexpr uint32_t EC_E_NOERROR = 0x00000000;
constexpr uint32_t EC_E_ERROR = 0x80000001;
constexpr uint32_t EC_E_TIMEOUT = 0x80000002;
constexpr uint32_t EC_E_INVALIDSTATE = 0x80000003;
constexpr uint32_t EC_E_NOTFOUND = 0x80000004;

/**
 * @brief Mock EC-Master context
 *
 * Simulates the state and behavior of an EC-Master instance
 */
class MockEcMasterContext {
 public:
  MockEcMasterContext()
      : state_(EcMasterState::UNKNOWN),
        is_initialized_(false),
        is_configured_(false),
        cycle_count_(0),
        simulate_timeout_(false),
        simulate_init_error_(false) {}

  // State management
  EcMasterState GetState() const { return state_; }
  void SetState(EcMasterState state) { state_ = state; }

  bool IsInitialized() const { return is_initialized_; }
  void SetInitialized(bool init) { is_initialized_ = init; }

  bool IsConfigured() const { return is_configured_; }
  void SetConfigured(bool config) { is_configured_ = config; }

  // Cycle management
  uint32_t GetCycleCount() const { return cycle_count_; }
  void IncrementCycle() { cycle_count_++; }
  void ResetCycle() { cycle_count_ = 0; }

  // Error simulation
  void SimulateTimeout(bool enable) { simulate_timeout_ = enable; }
  bool ShouldSimulateTimeout() const { return simulate_timeout_; }

  void SimulateInitError(bool enable) { simulate_init_error_ = enable; }
  bool ShouldSimulateInitError() const { return simulate_init_error_; }

  // Configuration
  void SetAdapterName(const std::string& name) { adapter_name_ = name; }
  std::string GetAdapterName() const { return adapter_name_; }

  void SetEniFile(const std::string& file) { eni_file_ = file; }
  std::string GetEniFile() const { return eni_file_; }

  // Reset to initial state
  void Reset() {
    state_ = EcMasterState::UNKNOWN;
    is_initialized_ = false;
    is_configured_ = false;
    cycle_count_ = 0;
    simulate_timeout_ = false;
    simulate_init_error_ = false;
    adapter_name_.clear();
    eni_file_.clear();
  }

 private:
  EcMasterState state_;
  bool is_initialized_;
  bool is_configured_;
  uint32_t cycle_count_;
  bool simulate_timeout_;
  bool simulate_init_error_;
  std::string adapter_name_;
  std::string eni_file_;
};

// Global mock context for testing
inline MockEcMasterContext& GetMockContext() {
  static MockEcMasterContext context;
  return context;
}

/**
 * @brief Mock ecatInitMaster - Initialize EC-Master
 */
inline uint32_t MockEcatInitMaster(void* /* pInitParams */) {
  auto& ctx = GetMockContext();

  if (ctx.ShouldSimulateInitError()) {
    return EC_E_ERROR;
  }

  ctx.SetInitialized(true);
  ctx.SetState(EcMasterState::INIT);
  return EC_E_NOERROR;
}

/**
 * @brief Mock ecatConfigureMaster - Configure EC-Master with ENI
 */
inline uint32_t MockEcatConfigureMaster(const char* /* eniFile */) {
  auto& ctx = GetMockContext();

  if (!ctx.IsInitialized()) {
    return EC_E_INVALIDSTATE;
  }

  ctx.SetConfigured(true);
  return EC_E_NOERROR;
}

/**
 * @brief Mock ecatSetMasterState - Request master state change
 */
inline uint32_t MockEcatSetMasterState(uint32_t timeout_ms, uint32_t target_state) {
  auto& ctx = GetMockContext();

  if (!ctx.IsConfigured()) {
    return EC_E_INVALIDSTATE;
  }

  if (ctx.ShouldSimulateTimeout()) {
    return EC_E_TIMEOUT;
  }

  // Simulate state transition based on target
  switch (target_state) {
    case static_cast<uint32_t>(EcMasterState::PREOP):
      ctx.SetState(EcMasterState::PREOP);
      break;
    case static_cast<uint32_t>(EcMasterState::SAFEOP):
      ctx.SetState(EcMasterState::SAFEOP);
      break;
    case static_cast<uint32_t>(EcMasterState::OP):
      ctx.SetState(EcMasterState::OP);
      break;
    default:
      return EC_E_ERROR;
  }

  return EC_E_NOERROR;
}

/**
 * @brief Mock ecatExecJob - Execute cyclic job
 */
inline uint32_t MockEcatExecJob(uint32_t /* jobType */, void* /* pJobParms */) {
  auto& ctx = GetMockContext();

  if (ctx.GetState() != EcMasterState::OP) {
    return EC_E_INVALIDSTATE;
  }

  ctx.IncrementCycle();
  return EC_E_NOERROR;
}

/**
 * @brief Mock ecatDeinitMaster - Deinitialize EC-Master
 */
inline uint32_t MockEcatDeinitMaster() {
  auto& ctx = GetMockContext();
  ctx.Reset();
  return EC_E_NOERROR;
}

/**
 * @brief Mock ecatGetMasterState - Get current master state
 */
inline uint32_t MockEcatGetMasterState() {
  auto& ctx = GetMockContext();
  return static_cast<uint32_t>(ctx.GetState());
}

}  // namespace mock
}  // namespace holocat

#endif  // MOCK_EC_MASTER_HPP
