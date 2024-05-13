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