/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <holoscan/holoscan.hpp>
#include <sys/time.h>

#define ONE_SEC_TO_NS  (uint64_t)1000000000
#define SEC_TO_NS(t) ((t) * ONE_SEC_TO_NS)

#define timespecsub(tsp, usp, vsp)                                      \
        do {                                                            \
                (vsp)->tv_sec = (tsp)->tv_sec - (usp)->tv_sec;          \
                (vsp)->tv_nsec = (tsp)->tv_nsec - (usp)->tv_nsec;       \
                if ((vsp)->tv_nsec < 0) {                               \
                        (vsp)->tv_sec--;                                \
                        (vsp)->tv_nsec += 1000000000L;                  \
                }                                                       \
        } while (0)

void spin2(uint64_t duration_ns, timespec& start_ts) {
  uint64_t elapsed_ns = 0;

  timespec curr_ts, diff_ts;
  while(true) {
    clock_gettime(CLOCK_REALTIME, &curr_ts);
    timespecsub(&curr_ts, &start_ts, &diff_ts);
    elapsed_ns = SEC_TO_NS(diff_ts.tv_sec) + diff_ts.tv_nsec;
    if (elapsed_ns >= duration_ns) {
      break;
    }
  }
}

void spin(uint64_t duration_ns) {
  timespec start_ts;
  clock_gettime(CLOCK_REALTIME, &start_ts);
  spin2(duration_ns, start_ts);
}

namespace holoscan::ops {

class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<int>("out"); //THIS WAS SETTING FOR EXPERIMENTS .condition(ConditionType::kNone);
    //spec.output<int>("out2").condition(ConditionType::kNone);
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    int value = index_++;
    op_output.emit(value, "out");
    //op_output.emit(value, "out2");
  };

 private:
  int index_ = 1;
};

class PingMxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxOp)

  PingMxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out").condition(ConditionType::kNone);
    spec.param(WCET_, "WCET", "WCET", "Worst case execution time", 0);
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto value = op_input.receive<int>("in").value();

    //std::cout << "Middle message value: " << value << std::endl;

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

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out1").condition(ConditionType::kNone);
    spec.output<int>("out2").condition(ConditionType::kNone);
    spec.param(WCET_, "WCET", "WCET", "Worst case execution time", 0);
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto value = op_input.receive<int>("in").value();

    //std::cout << "Middle message value: " << value << std::endl;

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

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out1");  //.condition(ConditionType::kNone);
    spec.output<int>("out2");  //.condition(ConditionType::kNone);
    spec.param(WCET_, "WCET", "WCET", "Worst case execution time", 0);
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto value = op_input.receive<int>("in").value();

    //std::cout << "Middle message value: " << value << std::endl;

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

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out1"); //.condition(ConditionType::kNone);
    spec.output<int>("out2"); //.condition(ConditionType::kNone);
    spec.output<int>("out3"); //.condition(ConditionType::kNone);
    spec.param(WCET_, "WCET", "WCET", "Worst case execution time", 0);
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto value = op_input.receive<int>("in").value();

    //std::cout << "Middle message value: " << value << std::endl;

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

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out1"); //.condition(ConditionType::kNone);
    spec.output<int>("out2"); //.condition(ConditionType::kNone);
    spec.output<int>("out3"); //.condition(ConditionType::kNone);
    spec.output<int>("out4"); //.condition(ConditionType::kNone);
    spec.param(WCET_, "WCET", "WCET", "Worst case execution time", 0);
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto value = op_input.receive<int>("in").value();

    //std::cout << "Middle message value: " << value << std::endl;

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

    void setup(OperatorSpec& spec) override {
      spec.param(receivers_, "receivers", "Input Receivers", "List of input receivers.", {});
      spec.output<int>("out");
      spec.param(WCET_, "WCET", "WCET", "Worst case execution time", 0);
    }

    void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
      auto value_vector =
        op_input.receive<std::vector<int>>("receivers").value();

      //std::cout << "Middle message value: " << value << std::endl;

      spin((WCET_ * static_cast<double>(1000000)));

      op_output.emit(value_vector[0]);
    };

  private:
    Parameter<std::vector<IOSpec*>> receivers_;
    Parameter<int> WCET_;
};


class PingMultiInputOutputMxOp : public Operator {
  public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMultiInputOutputMxOp)

    PingMultiInputOutputMxOp() = default;

    void setup(OperatorSpec& spec) override {
      spec.param(receivers_, "receivers", "Input Receivers", "List of input receivers.", {});
      spec.output<int>("out1");
      spec.output<int>("out2");
      spec.param(WCET_, "WCET", "WCET", "Worst case execution time", 0);
    }

    void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
      auto value_vector =
        op_input.receive<std::vector<int>>("receivers").value();

      //std::cout << "Middle message value: " << value << std::endl;

      spin((WCET_ * static_cast<double>(1000000)));

      op_output.emit(value_vector[0], "out1");
      op_output.emit(value_vector[0], "out2");
    };

  private:
    Parameter<std::vector<IOSpec*>> receivers_;
    Parameter<int> WCET_;
};

class PingMxDownstreamOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxDownstreamOp)

  PingMxDownstreamOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out");
    spec.param(WCET_, "WCET", "WCET", "Worst case execution time", 0);
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto value = op_input.receive<int>("in").value();

    //std::cout << "Middle message value: " << value << std::endl;

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

  void setup(OperatorSpec& spec) override {
    spec.param(receivers_, "receivers", "Input Receivers", "List of input receivers.", {});
    spec.param(WCET_, "WCET", "WCET", "Worst case execution time", 0);
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto value_vector =
      op_input.receive<std::vector<int>>("receivers").value();

    spin((WCET_ * static_cast<double>(1000000)));


    std::cout << "Running... iteration " << value_vector[0] << " complete" <<  std::endl;
  };

 private:
  Parameter<std::vector<IOSpec*>> receivers_;
  Parameter<int> WCET_;
};

}  // namespace holoscan::ops

class MyPingApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;



  }
};

int main(int argc, char** argv) {
  holoscan::set_log_level(holoscan::LogLevel::ERROR);  
  
  auto app = holoscan::make_application<MyPingApp>();

  // Get the configuration
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/experiment.yaml";
  app->config(config_path);

  auto& tracker = app->track(); // Enable Data Flow Tracking
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

