/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 UNIVERSITY OF BRITISH COLUMBIA.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * This is a modified version of several files from the Holoscan SDK
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

#include <sys/time.h>

#include <holoscan/holoscan.hpp>

// Difference factor
#define ONE_SEC_TO_NS (uint64_t)1000000000

// Macro to convert seconds to nanoseconds
#define SEC_TO_NS(t) ((t)*ONE_SEC_TO_NS)

/*
 * Macro to subtract two timespec structures.
 * Parameters:
 * - tsp: Pointer to the minuend timespec structure.
 * - usp: Pointer to the subtrahend timespec structure.
 * - vsp: Pointer to the result timespec structure where the difference is
 * stored.
 *
 * The macro calculates the difference in seconds (tv_sec) and nanoseconds
 * (tv_nsec). If the nanoseconds difference is negative, it adjusts the seconds
 * and nanoseconds values accordingly to ensure the timespec remains normalized.
 */
#define timespecsub(tsp, usp, vsp)                    \
  do {                                                \
    (vsp)->tv_sec = (tsp)->tv_sec - (usp)->tv_sec;    \
    (vsp)->tv_nsec = (tsp)->tv_nsec - (usp)->tv_nsec; \
    if ((vsp)->tv_nsec < 0) {                         \
      (vsp)->tv_sec--;                                \
      (vsp)->tv_nsec += 1000000000L;                  \
    }                                                 \
  } while (0)

/*
 * Function to create a spinning delay for a given duration.
 *
 * Parameters:
 * - duration_ns: The duration to spin in nanoseconds.
 * - start_ts: The starting time as a timespec structure.
 *
 * This function uses a busy-wait loop to measure elapsed time from the provided
 * start_ts until the specified duration in nanoseconds is reached. It
 * calculates the elapsed time by repeatedly querying the current time and
 * subtracting it from the start time.
 */
void spin2(uint64_t duration_ns, timespec &start_ts) {
  uint64_t elapsed_ns = 0;

  timespec curr_ts, diff_ts;
  while (true) {
    clock_gettime(CLOCK_REALTIME, &curr_ts);  // Get the current time
    timespecsub(&curr_ts, &start_ts,
                &diff_ts);  // Calculate the time difference
    elapsed_ns =
        SEC_TO_NS(diff_ts.tv_sec) + diff_ts.tv_nsec;  // Convert to nanoseconds
    if (elapsed_ns >= duration_ns) {
      break;  // Exit the loop when the duration is met or exceeded
    }
  }
}

/*
 * Function to create a spinning delay for a given duration in nanoseconds.
 *
 * Parameters:
 * - duration_ns: The duration to spin in nanoseconds.
 *
 * This function records the start time, then delegates to spin2 to handle
 * the spinning logic using the recorded start time.
 */
void spin(uint64_t duration_ns) {
  timespec start_ts;
  clock_gettime(CLOCK_REALTIME, &start_ts);  // Record the start time
  spin2(duration_ns,
        start_ts);  // Delegate to spin2 with the recorded start time
}

namespace holoscan::ops {

class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec &spec) override { spec.output<int>("out"); }

  void compute(InputContext &op_input, OutputContext &op_output,
               ExecutionContext &) override {
    int value = index_++;
    op_output.emit(value, "out");
  };

 private:
  int index_ = 1;
};

class PingMxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxOp)

  PingMxOp() = default;

  void setup(OperatorSpec &spec) override {
    spec.input<int>("in");
    spec.output<int>("out").condition(ConditionType::kNone);
    spec.param(WCET_, "WCET", "WCET",
               "Worst case execution time in milliseconds", 0);
  }

  void compute(InputContext &op_input, OutputContext &op_output,
               ExecutionContext &) override {
    auto value = op_input.receive<int>("in").value();

    spin((WCET_ * static_cast<double>(1000000)));

    op_output.emit(value);
  };

 private:
  Parameter<int> WCET_;
};

class PingMxTwoOutputOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxTwoOutputOp)

  PingMxTwoOutputOp() = default;

  void setup(OperatorSpec &spec) override {
    spec.input<int>("in");
    spec.output<int>("out1").condition(ConditionType::kNone);
    spec.output<int>("out2").condition(ConditionType::kNone);
    spec.param(WCET_, "WCET", "WCET",
               "Worst case execution time in milliseconds", 0);
  }

  void compute(InputContext &op_input, OutputContext &op_output,
               ExecutionContext &) override {
    auto value = op_input.receive<int>("in").value();

    spin((WCET_ * static_cast<double>(1000000)));

    op_output.emit(value, "out1");
    op_output.emit(value, "out2");
  };

 private:
  Parameter<int> WCET_;
};

class PingMxTwoOutputDownstreamOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxTwoOutputDownstreamOp)

  PingMxTwoOutputDownstreamOp() = default;

  void setup(OperatorSpec &spec) override {
    spec.input<int>("in");
    spec.output<int>("out1");
    spec.output<int>("out2");
    spec.param(WCET_, "WCET", "WCET",
               "Worst case execution time in milliseconds", 0);
  }

  void compute(InputContext &op_input, OutputContext &op_output,
               ExecutionContext &) override {
    auto value = op_input.receive<int>("in").value();

    spin((WCET_ * static_cast<double>(1000000)));

    op_output.emit(value, "out1");
    op_output.emit(value, "out2");
  };

 private:
  Parameter<int> WCET_;
};

class PingMxThreeOutputDownstreamOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxThreeOutputDownstreamOp)

  PingMxThreeOutputDownstreamOp() = default;

  void setup(OperatorSpec &spec) override {
    spec.input<int>("in");
    spec.output<int>("out1");
    spec.output<int>("out2");
    spec.output<int>("out3");
    spec.param(WCET_, "WCET", "WCET",
               "Worst case execution time in milliseconds", 0);
  }

  void compute(InputContext &op_input, OutputContext &op_output,
               ExecutionContext &) override {
    auto value = op_input.receive<int>("in").value();

    spin((WCET_ * static_cast<double>(1000000)));

    op_output.emit(value, "out1");
    op_output.emit(value, "out2");
    op_output.emit(value, "out3");
  };

 private:
  Parameter<int> WCET_;
};

class PingMxFourOutputDownstreamOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxFourOutputDownstreamOp)

  PingMxFourOutputDownstreamOp() = default;

  void setup(OperatorSpec &spec) override {
    spec.input<int>("in");
    spec.output<int>("out1");
    spec.output<int>("out2");
    spec.output<int>("out3");
    spec.output<int>("out4");
    spec.param(WCET_, "WCET", "WCET",
               "Worst case execution time in milliseconds", 0);
  }

  void compute(InputContext &op_input, OutputContext &op_output,
               ExecutionContext &) override {
    auto value = op_input.receive<int>("in").value();

    spin((WCET_ * static_cast<double>(1000000)));

    op_output.emit(value, "out1");
    op_output.emit(value, "out2");
    op_output.emit(value, "out3");
    op_output.emit(value, "out4");
  };

 private:
  Parameter<int> WCET_;
};

class PingMultiInputMxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMultiInputMxOp)

  PingMultiInputMxOp() = default;

  void setup(OperatorSpec &spec) override {
    spec.param(receivers_, "receivers", "Input Receivers",
               "List of input receivers.", {});
    spec.output<int>("out");
    spec.param(WCET_, "WCET", "WCET",
               "Worst case execution time in milliseconds", 0);
  }

  void compute(InputContext &op_input, OutputContext &op_output,
               ExecutionContext &) override {
    auto value_vector = op_input.receive<std::vector<int>>("receivers").value();

    spin((WCET_ * static_cast<double>(1000000)));

    op_output.emit(value_vector[0]);
  };

 private:
  Parameter<std::vector<IOSpec *>> receivers_;
  Parameter<int> WCET_;
};

class PingMultiInputOutputMxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMultiInputOutputMxOp)

  PingMultiInputOutputMxOp() = default;

  void setup(OperatorSpec &spec) override {
    spec.param(receivers_, "receivers", "Input Receivers",
               "List of input receivers.", {});
    spec.output<int>("out1");
    spec.output<int>("out2");
    spec.param(WCET_, "WCET", "WCET",
               "Worst case execution time in milliseconds", 0);
  }

  void compute(InputContext &op_input, OutputContext &op_output,
               ExecutionContext &) override {
    auto value_vector = op_input.receive<std::vector<int>>("receivers").value();

    spin((WCET_ * static_cast<double>(1000000)));

    op_output.emit(value_vector[0], "out1");
    op_output.emit(value_vector[0], "out2");
  };

 private:
  Parameter<std::vector<IOSpec *>> receivers_;
  Parameter<int> WCET_;
};

class PingMxDownstreamOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxDownstreamOp)

  PingMxDownstreamOp() = default;

  void setup(OperatorSpec &spec) override {
    spec.input<int>("in");
    spec.output<int>("out");
    spec.param(WCET_, "WCET", "WCET",
               "Worst case execution time in milliseconds", 0);
  }

  void compute(InputContext &op_input, OutputContext &op_output,
               ExecutionContext &) override {
    auto value = op_input.receive<int>("in").value();

    spin((WCET_ * static_cast<double>(1000000)));

    op_output.emit(value);
  };

 private:
  Parameter<int> WCET_;
};

class PingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

  PingRxOp() = default;

  void setup(OperatorSpec &spec) override {
    spec.param(receivers_, "receivers", "Input Receivers",
               "List of input receivers.", {});
    spec.param(WCET_, "WCET", "WCET",
               "Worst case execution time in milliseconds", 0);
  }

  void compute(InputContext &op_input, OutputContext &op_output,
               ExecutionContext &) override {
    auto value_vector = op_input.receive<std::vector<int>>("receivers").value();

    spin((WCET_ * static_cast<double>(1000000)));

    std::cout << "Running... iteration " << value_vector[0] << " complete"
              << std::endl;
  };

 private:
  Parameter<std::vector<IOSpec *>> receivers_;
  Parameter<int> WCET_;
};

}  // namespace holoscan::ops
class MyPingApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;  // This is filled in by MakeVars.py when the
                               // scripts are run
  }
};

int main(int argc, char **argv) {
  holoscan::set_log_level(holoscan::LogLevel::ERROR);

  auto app = holoscan::make_application<MyPingApp>();

  // Get the configuration
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/experiment.yaml";
  app->config(config_path);

  auto &tracker = app->track();  // Enable Data Flow Tracking
  tracker.enable_logging("logger.log");

  // set customizable application parameters via the YAML

  bool multithreaded = app->from_config("multithreaded").as<bool>();
  if (multithreaded) {
    // use MultiThreadScheduler instead of the default GreedyScheduler
    app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
        "multithread-scheduler", app->from_config("scheduler")));
  }

  app->run();

  return 0;
}
