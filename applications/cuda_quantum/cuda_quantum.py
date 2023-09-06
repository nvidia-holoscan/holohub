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

import os

import cudaq
from cudaq import spin  # noqa: F401
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec


class ClassicalComputeOp(Operator):
    def __init__(self, *args, hamiltonian, **kwargs):
        self.count = 0
        self.hamiltonian = hamiltonian
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("Hamiltonian")
        spec.output("QuantumKernel")

    def compute(self, op_input, op_output, context):
        kernel, thetas = cudaq.make_kernel(list)
        qubits = kernel.qalloc(2)
        kernel.x(qubits[0])
        kernel.ry(thetas[0], qubits[1])
        kernel.cx(qubits[1], qubits[0])

        print("Printing Circuit: ", kernel)

        self.count += 1

        op_output.emit(self.hamiltonian, "Hamiltonian")
        op_output.emit(kernel, "QuantumKernel")


class QuantumComputeOp(Operator):
    def __init__(self, *args, backend_name, api_key=None, **kwargs):
        self.backend_name = backend_name
        self.api_key = api_key
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("Hamiltonian")
        spec.input("QuantumKernel")
        spec.output("Result")

    def compute(self, op_input, op_output, context):
        hamiltonian = eval(op_input.receive("Hamiltonian"))
        kernel = op_input.receive("QuantumKernel")

        optimizer = cudaq.optimizers.COBYLA()

        energy, parameter = cudaq.vqe(
            kernel=kernel, spin_operator=hamiltonian, optimizer=optimizer, parameter_count=1
        )

        result = "energy: " + str(energy) + "\n parameter: " + str(parameter)
        op_output.emit(result, "Result")


class PrintOp(Operator):
    def __init__(self, *args, prompt="Result: ", **kwargs):
        self.prompt = prompt
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("text")

    def compute(self, op_input, op_output, context):
        text = op_input.receive("text")
        print(self.prompt, text + "\n")


class QuantumVQEApp(Application):
    def compose(self):
        classical_computer_operator = ClassicalComputeOp(
            self,
            CountCondition(self, count=1),
            name="classical_op",
            **self.kwargs("ClassicalComputeOp"),
        )
        quantum_computer_operator = QuantumComputeOp(
            self, name="Quantum Ops", **self.kwargs("QuantumComputeOp")
        )
        print_result = PrintOp(self, name="print_result", prompt="VQE Result: ")

        self.add_flow(
            classical_computer_operator,
            quantum_computer_operator,
            {("Hamiltonian", "Hamiltonian"), ("QuantumKernel", "QuantumKernel")},
        )
        self.add_flow(quantum_computer_operator, print_result, {("Result", "text")})


if __name__ == "__main__":
    app = QuantumVQEApp()
    config_file = os.path.join(os.path.dirname(__file__), "cuda_quantum.yaml")
    app.config(config_file)
    app.run()
