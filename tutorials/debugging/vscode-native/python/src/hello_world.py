"""
SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

# define custom Operators for use in the demo


class HelloWorldOp(Operator):
    """Simple hello world operator.
    This operator has no ports.
    On each tick, this operator prints out hello world messages.
    """

    def setup(self, spec: OperatorSpec):
        pass

    def compute(self, op_input, op_output, context):
        print("Hello World!")


# Now define a simple application using the operator defined above
class HelloWorldApp(Application):
    def compose(self):
        # Define the operators
        hello = HelloWorldOp(self, CountCondition(self, 1), name="hello")

        # Define the one-operator workflow
        self.add_operator(hello)


def main():
    app = HelloWorldApp()
    app.run()


if __name__ == "__main__":
    main()
