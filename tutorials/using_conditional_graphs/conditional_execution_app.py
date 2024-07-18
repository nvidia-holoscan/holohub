# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import PingRxOp, PingTxOp


class PingMxOp(Operator):
    """Example of an operator modifying data.

    This operator has 1 input and 1 output port:
        input:  "in"
        output: "out"

    The data from each input is multiplied by a user-defined value.

    """

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")
        spec.param("multiplier", 2)

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")
        print(f"Middle message value: {value}")

        # Multiply the values by the multiplier parameter
        value *= self.multiplier

        op_output.emit(value, "out")


class ConditionGate(Operator):
    """
    An operator that performs a conditional check on the input data. 
    When the condition criteria is met, the data is passed to the next operator.
    Else, no data is passed to the next operator, effectively blocking the execution of the next operators
    in the directed acyclic graph.
    """
    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")
        print(f"Check received message value: {value}")
        if value < 10:
            print("condition not met")
        else:
            print(f"condition met, value is {value}")
            op_output.emit(value, "out")


class MyPingApp(Application):
    def compose(self):
        # Define the tx and rx operators, allowing tx to execute 20 times
        tx = PingTxOp(self, CountCondition(self, 20), name="tx")
        rx = PingRxOp(self, name="rx")
        check = ConditionGate(self, name="check")
        mx = PingMxOp(self, name="mx")
        rx2 = PingRxOp(self, name="rx2")

        # Define the workflow:  tx -> rx
        #                         \-> check -> mx -> rx2
        self.add_flow(tx, rx)
        self.add_flow(tx, check)
        self.add_flow(check, mx)
        self.add_flow(mx, rx2)


def main():
    app = MyPingApp()
    app.run()


if __name__ == "__main__":
    main()
