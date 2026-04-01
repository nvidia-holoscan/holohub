#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal test that a Holoscan app with HolovizOp runs headless (no display).
"""Test that minimal Holoviz app runs headless."""

import sys
from pathlib import Path

app_dir = Path(__file__).resolve().parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))


def main():
    try:
        import holoscan as hs
        from holoscan.conditions import CountCondition
        from holoscan.core import Application, Operator, OperatorSpec
        from holoscan.operators import HolovizOp
        from holoscan.resources import UnboundedAllocator
    except ImportError as e:
        print(f"SKIP: holoscan not available ({e})")
        return 0

    class MinimalSource(Operator):
        def __init__(self, fragment, *args, **kwargs):
            super().__init__(fragment, *args, **kwargs)

        def setup(self, spec: OperatorSpec):
            spec.output("out")

        def compute(self, op_input, op_output, context):
            import numpy as np

            # HolovizOp expects an Entity with a named tensor matching tensors=[dict(name="image", ...)]
            img = np.ones((4, 4, 4), dtype=np.float32) * 0.5
            out_message = hs.gxf.Entity(context)
            out_message.add(hs.as_tensor(img), "image")
            op_output.emit(out_message, "out")

    class MinimalHolovizApp(Application):
        def compose(self):
            pool = UnboundedAllocator(self, name="pool")
            count_condition = CountCondition(self, count=5)
            source = MinimalSource(self, name="source", count=count_condition)
            holoviz = HolovizOp(
                self,
                allocator=pool,
                name="holoviz",
                headless=True,
                width=4,
                height=4,
                tensors=[dict(name="image", type="color")],
            )
            self.add_flow(source, holoviz, {("out", "receivers")})

    try:
        app = MinimalHolovizApp()
        app.run()
    except Exception as e:
        print(f"FAIL: minimal Holoviz app raised {e}")
        return 1

    print("SUCCESS: Holoviz minimal OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
