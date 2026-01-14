#!/usr/bin/env python3
"""
ST2110 Multi-Output Test Application
=====================================
Comprehensive test to verify all three output formats work correctly:
- raw_output: YCbCr-4:2:2-10bit (always available)
- rgba_output: RGBA 8-bit (optional)
- nv12_output: NV12 8-bit (optional)

This test creates separate monitoring operators for each output to verify
they are all functioning correctly.
"""

import logging
import os
import sys

from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp

# Try to import the ST2110 operator
try:
    from holohub.st2110_source import ST2110SourceOp
except ImportError as e:
    print(f"Error: Could not import ST2110SourceOp: {e}")
    print("\nMake sure the operator is built:")
    print("  ./holohub build st2110_source --configure-args '-DHOLOHUB_BUILD_PYTHON=ON'")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutputMonitor(Operator):
    """Monitor operator to track and log frame reception on each output"""

    def __init__(self, fragment, output_name, *args, **kwargs):
        self.output_name = output_name
        self.frame_count = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        # Receive frame
        message = op_input.receive("input")
        self.frame_count += 1

        # Log every 50 frames
        if self.frame_count % 50 == 0:
            logger.info(f"[{self.output_name}] Received frame #{self.frame_count}")

            # Try to extract frame info
            try:
                if hasattr(message, "get"):
                    # Try to get tensor
                    tensor_names = ["raw_video", "video_tensor", "video"]
                    for name in tensor_names:
                        try:
                            tensor = message.get(name)
                            if tensor is not None:
                                logger.info(
                                    f"  [{self.output_name}] Tensor '{name}': shape={tensor.shape}, dtype={tensor.dtype}"
                                )
                                break
                        except Exception:
                            pass
            except Exception as e:
                logger.debug(f"  [{self.output_name}] Could not access tensor info: {e}")


class ST2110MultiOutputTest(Application):
    """
    Multi-output test application for ST 2110-20 video reception

    Pipeline:
        ST2110SourceOp → RGBA Holoviz (rgba_output)
                     ↓
                  NV12 Holoviz (nv12_output)
                     ↓
                  RawMonitor (raw_output)
                     ↓
                  RGBAMonitor (rgba_output)
                     ↓
                  NV12Monitor (nv12_output)

    This verifies all three outputs are functioning correctly:
    - raw_output: Always emits raw YCbCr-4:2:2-10bit frames
    - rgba_output: Emits RGBA frames if enable_rgba_output=true (visualized)
    - nv12_output: Emits NV12 frames if enable_nv12_output=true (visualized)
    """

    def compose(self):
        logger.info("=" * 70)
        logger.info("Composing ST2110 Multi-Output Test Application")
        logger.info("=" * 70)

        # Create ST2110 source operator
        st2110_source = ST2110SourceOp(self, name="st2110_source", **self.kwargs("st2110_source"))

        # Create monitoring operators for each output
        raw_monitor = OutputMonitor(self, output_name="RAW_OUTPUT", name="raw_monitor")

        rgba_monitor = OutputMonitor(self, output_name="RGBA_OUTPUT", name="rgba_monitor")

        nv12_monitor = OutputMonitor(self, output_name="NV12_OUTPUT", name="nv12_monitor")

        # Create Holoviz operator for RGBA visualization
        rgba_visualizer = HolovizOp(self, name="rgba_visualizer", **self.kwargs("rgba_visualizer"))

        # Create Holoviz operator for NV12 visualization
        nv12_visualizer = HolovizOp(self, name="nv12_visualizer", **self.kwargs("nv12_visualizer"))

        # Connect all outputs for monitoring
        # raw_output → RawMonitor
        self.add_flow(st2110_source, raw_monitor, {("raw_output", "input")})

        # rgba_output → RGBAMonitor + RGBA Holoviz
        self.add_flow(st2110_source, rgba_monitor, {("rgba_output", "input")})

        self.add_flow(st2110_source, rgba_visualizer, {("rgba_output", "receivers")})

        # nv12_output → NV12Monitor + NV12 Holoviz
        self.add_flow(st2110_source, nv12_monitor, {("nv12_output", "input")})

        self.add_flow(st2110_source, nv12_visualizer, {("nv12_output", "receivers")})

        logger.info("Pipeline composed with all three outputs:")
        logger.info("  - raw_output  → RawMonitor")
        logger.info("  - rgba_output → RGBAMonitor + RGBA Holoviz")
        logger.info("  - nv12_output → NV12Monitor + NV12 Holoviz")
        logger.info("=" * 70)


def main():
    """Main entry point"""

    import argparse

    parser = argparse.ArgumentParser(description="ST2110 Multi-Output Test")
    parser.add_argument(
        "--config",
        type=str,
        default="st2110_test_config.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("ST2110 Multi-Output Test")
    print("=" * 70)
    print()
    print("This test verifies all three output formats:")
    print()
    print("  1. raw_output  - YCbCr-4:2:2-10bit (always enabled)")
    print("  2. rgba_output - RGBA 8-bit (configurable)")
    print("  3. nv12_output - NV12 8-bit (configurable)")
    print()
    print("Each output has its own monitoring operator that logs frame reception.")
    print("You should see frame counters for all enabled outputs.")
    print()
    print(f"Configuration: {args.config}")
    print("  - Set enable_rgba_output: true/false")
    print("  - Set enable_nv12_output: true/false")
    print()
    print("Expected behavior:")
    print("  - Two Holoviz windows: one for RGBA, one for NV12 (if enabled)")
    print("  - Console logs showing frame reception on all enabled outputs")
    print("  - All outputs should receive frames at the same rate")
    print("  - Visual comparison of RGBA vs NV12 encoding quality")
    print()
    print("Press Ctrl+C or ESC to quit")
    print("=" * 70)
    print()

    # Check if config file exists
    config_file = (
        args.config
        if os.path.isabs(args.config)
        else os.path.join(os.path.dirname(__file__), args.config)
    )
    if not os.path.exists(config_file):
        print(f"ERROR: Configuration file not found: {config_file}")
        print()
        print(f"Create {args.config} with your network configuration.")
        sys.exit(1)

    # Create and run the application
    try:
        app = ST2110MultiOutputTest()
        app.config(config_file)
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
