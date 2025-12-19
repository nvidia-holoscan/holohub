import sys
import os
try:
    import holoscan
    print(f"Holoscan version: {holoscan.__version__}")
except ImportError:
    print("Holoscan not installed.")
    sys.exit(1)

from holoscan.core import Application, Operator
from holoscan.conditions import CountCondition

# Try to import Tracker
try:
    from holoscan.core import Tracker
    print("Imported Tracker from holoscan.core")
except ImportError:
    try:
        from holoscan.tracking import Tracker
        print("Imported Tracker from holoscan.tracking")
    except ImportError:
        print("Could not import Tracker.")
        sys.exit(1)

class PingOp(Operator):
    def setup(self, spec):
        spec.output("out")
    def compute(self, op_input, op_output, context):
        print("PingOp executing...")
        op_output.emit(1, "out")

class SinkOp(Operator):
    def setup(self, spec):
        spec.input("in")
    def compute(self, op_input, op_output, context):
        msg = op_input.receive("in")
        print(f"Sink received: {msg}")

class SimpleApp(Application):
    def compose(self):
        # Run 5 times
        ping = PingOp(self, CountCondition(self, 5), name="ping")
        sink = SinkOp(self, name="sink")
        self.add_flow(ping, sink)

def main():
    log_file = "debug_tracker.log"
    if os.path.exists(log_file):
        os.remove(log_file)
    
    app = SimpleApp()
    print(f"Running minimal app with Tracker -> {log_file}")
    
    try:
        with Tracker(app, filename=log_file) as tracker:
            app.run()
    except Exception as e:
        print(f"Error: {e}")
        
    if os.path.exists(log_file):
        size = os.path.getsize(log_file)
        print(f"Success! Log file created. Size: {size} bytes")
    else:
        print("Failure! Log file was NOT created.")

if __name__ == "__main__":
    main()

