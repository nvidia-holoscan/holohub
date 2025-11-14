#!/usr/bin/env python3
"""
Minimal Holoviz Test - No data needed
Just tests if Holoviz window can open and display
"""

import numpy as np
import holoscan as hs
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp
from holoscan.conditions import CountCondition

class DummySourceOp(Operator):
    """Generate a simple test pattern"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_count = 0
        
    def setup(self, spec: OperatorSpec):
        spec.output("output")
        
    def compute(self, op_input, op_output, context):
        # Create a simple gradient test pattern
        height, width = 512, 640
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create RGB gradient
        for i in range(height):
            for j in range(width):
                img[i, j, 0] = int((j / width) * 255)  # R gradient
                img[i, j, 1] = int((i / height) * 255)  # G gradient
                img[i, j, 2] = 128  # B constant
        
        # Emit
        out_message = hs.gxf.Entity(context)
        out_message.add(hs.as_tensor(img), "test_pattern")
        op_output.emit(out_message, "output")
        
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            print(f"Generated frame {self.frame_count}")

class MinimalVizApp(Application):
    def compose(self):
        print("Creating minimal Holoviz test app...")
        
        # Create count condition to limit frames
        count_condition = CountCondition(self, count=30)
        
        source = DummySourceOp(
            self,
            name="source",
            count=count_condition
        )
        
        viz = HolovizOp(
            self,
            name="viz",
            width=640,
            height=512,
            tensors=[dict(name="test_pattern", type="color")]
        )
        
        self.add_flow(source, viz, {("output", "receivers")})
        print("Pipeline created. Starting...")

if __name__ == "__main__":
    print("=" * 60)
    print("  Minimal Holoviz Test")
    print("=" * 60)
    print("This will test if Holoviz can open a window at all.")
    print("You should see a colored gradient pattern.")
    print("=" * 60)
    
    try:
        app = MinimalVizApp()
        app.run()
        print("\n✅ SUCCESS: Holoviz window opened and displayed!")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
