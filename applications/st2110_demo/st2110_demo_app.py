# SPDX-FileCopyrightText: Copyright (c) 2025-2026, XRlabs. All rights reserved.
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

"""
ST2110 Demo Application
=======================
Demo application for ST 2110-20 video reception with visualization.
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


class NullSink(Operator):
    """Sink operator that receives and discards frames.

    Used to consume outputs that aren't connected to visualizers,
    preventing scheduler deadlock from unconnected ports.
    """

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, _op_output, _context):
        # Receive and discard
        op_input.receive("input")


class ST2110DemoApp(Application):
    """
    Demo application for ST 2110-20 video reception.

    Pipeline:
        ST2110SourceOp --raw_output--> NullSink (always, raw YCbCr not visualized)
                       --rgba_output--> HolovizOp (if enabled)
                       --nv12_output--> HolovizOp or NullSink (depending on config)
    """

    def compose(self):
        logger.info("=" * 70)
        logger.info("Composing ST2110 Demo Application")
        logger.info("=" * 70)

        # Get config to check which outputs are enabled
        st2110_config = self.kwargs("st2110_source")
        if not st2110_config:
            raise ValueError("Missing required 'st2110_source' configuration block")
        enable_rgba = st2110_config.get("enable_rgba_output", False)
        enable_nv12 = st2110_config.get("enable_nv12_output", False)

        logger.info(f"  enable_rgba_output: {enable_rgba}")
        logger.info(f"  enable_nv12_output: {enable_nv12}")

        # Create ST2110 source operator
        st2110_source = ST2110SourceOp(self, name="st2110_source", **st2110_config)

        # Always sink raw_output (raw YCbCr-4:2:2-10bit isn't directly visualizable)
        raw_sink = NullSink(self, name="raw_sink")
        self.add_flow(st2110_source, raw_sink, {("raw_output", "input")})
        logger.info("  raw_output -> NullSink")

        # RGBA output: visualize if enabled, otherwise sink
        if enable_rgba:
            rgba_visualizer = HolovizOp(
                self, name="rgba_visualizer", **self.kwargs("rgba_visualizer")
            )
            self.add_flow(st2110_source, rgba_visualizer, {("rgba_output", "receivers")})
            logger.info("  rgba_output -> HolovizOp (ENABLED)")
        else:
            rgba_sink = NullSink(self, name="rgba_sink")
            self.add_flow(st2110_source, rgba_sink, {("rgba_output", "input")})
            logger.info("  rgba_output -> NullSink (disabled)")

        # NV12 output: visualize if enabled, otherwise sink
        if enable_nv12:
            nv12_visualizer = HolovizOp(
                self, name="nv12_visualizer", **self.kwargs("nv12_visualizer")
            )
            self.add_flow(st2110_source, nv12_visualizer, {("nv12_output", "receivers")})
            logger.info("  nv12_output -> HolovizOp (ENABLED)")
        else:
            nv12_sink = NullSink(self, name="nv12_sink")
            self.add_flow(st2110_source, nv12_sink, {("nv12_output", "input")})
            logger.info("  nv12_output -> NullSink (disabled)")

        logger.info("=" * 70)


def main():
    """Main entry point"""

    import argparse

    parser = argparse.ArgumentParser(description="ST2110 Demo Application")
    parser.add_argument(
        "--config",
        type=str,
        default="st2110_demo_config.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("ST2110 Demo Application")
    print("=" * 70)
    print()
    print("Receiving ST 2110-20 video and displaying via Holoviz.")
    print()
    print(f"Configuration: {args.config}")
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
        app = ST2110DemoApp()
        app.config(config_file)
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
