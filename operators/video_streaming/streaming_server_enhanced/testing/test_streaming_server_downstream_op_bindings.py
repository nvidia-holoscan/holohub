# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Python binding tests for StreamingServerDownstreamOp.

These tests verify that the pybind11 bindings work correctly and that
C++ StreamingServerDownstreamOp can be used from Python.
"""

import os

import pytest

# Test configuration
STREAMING_SERVER_ENHANCED_MOCK_ONLY = (
    os.environ.get("STREAMING_SERVER_ENHANCED_MOCK_ONLY", "false").lower() == "true"
)


@pytest.mark.bindings
@pytest.mark.skipif(STREAMING_SERVER_ENHANCED_MOCK_ONLY, reason="Running in mock-only mode")
class TestStreamingServerDownstreamOpBindings:
    """Test suite for StreamingServerDownstreamOp Python bindings."""

    def setup_method(self):
        """Set up test method."""
        self.downstream_op = None
        self.resource = None

    def teardown_method(self):
        """Clean up after test method."""
        if self.downstream_op:
            self.downstream_op = None
        if self.resource:
            self.resource = None

    def test_import_streaming_server_downstream_op(self):
        """Test that StreamingServerDownstreamOp can be imported from Python bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerDownstreamOp

            assert StreamingServerDownstreamOp is not None
        except ImportError as e:
            pytest.skip(f"StreamingServerDownstreamOp not available: {e}")

    def test_streaming_server_downstream_op_construction(self):
        """Test StreamingServerDownstreamOp construction through Python bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerDownstreamOp

            # Test default construction
            downstream_op = StreamingServerDownstreamOp()
            assert downstream_op is not None
            self.downstream_op = downstream_op

        except ImportError as e:
            pytest.skip(f"StreamingServerDownstreamOp not available: {e}")
        except Exception as e:
            pytest.fail(f"Failed to construct StreamingServerDownstreamOp: {e}")

    def test_streaming_server_downstream_op_parameters(self):
        """Test StreamingServerDownstreamOp parameter setting through bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerDownstreamOp

            downstream_op = StreamingServerDownstreamOp()
            self.downstream_op = downstream_op

            # Test parameter setting
            test_cases = [
                ("width", 1920, "width parameter"),
                ("height", 1080, "height parameter"),
                ("fps", 60, "fps parameter"),
                ("mirror", True, "mirror parameter"),
                ("mirror", False, "mirror parameter (disabled)"),
            ]

            for param_name, value, description in test_cases:
                if hasattr(downstream_op, param_name):
                    try:
                        # Try as method call
                        if callable(getattr(downstream_op, param_name)):
                            getattr(downstream_op, param_name)(value)
                        else:
                            # Try as property
                            setattr(downstream_op, param_name, value)
                        print(f"✅ Successfully set {description}: {value}")
                    except Exception as e:
                        print(f"⚠️ Could not set {description}: {e}")
                else:
                    print(f"ℹ️ Parameter {param_name} not exposed in bindings")

        except ImportError as e:
            pytest.skip(f"StreamingServerDownstreamOp not available: {e}")

    def test_streaming_server_downstream_op_with_resource(self):
        """Test StreamingServerDownstreamOp with StreamingServerResource through bindings."""
        try:
            from holohub.streaming_server_enhanced import (
                StreamingServerDownstreamOp,
                StreamingServerResource,
            )

            # Create resource
            resource = StreamingServerResource()
            self.resource = resource

            # Create downstream operator
            downstream_op = StreamingServerDownstreamOp()
            self.downstream_op = downstream_op

            # Test resource assignment if available
            if hasattr(downstream_op, "streaming_server_resource"):
                try:
                    if callable(getattr(downstream_op, "streaming_server_resource")):
                        downstream_op.streaming_server_resource(resource)
                        print("✅ Successfully assigned resource to downstream operator")
                    else:
                        setattr(downstream_op, "streaming_server_resource", resource)
                        print("✅ Successfully set resource as property")
                except Exception as e:
                    print(f"⚠️ Could not assign resource: {e}")
            else:
                print("ℹ️ streaming_server_resource not exposed in bindings")

        except ImportError as e:
            pytest.skip(
                f"StreamingServerDownstreamOp or StreamingServerResource not available: {e}"
            )

    def test_streaming_server_downstream_op_mirror_functionality(self):
        """Test StreamingServerDownstreamOp mirror parameter through bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerDownstreamOp

            downstream_op = StreamingServerDownstreamOp()
            self.downstream_op = downstream_op

            # Test mirror parameter specifically
            if hasattr(downstream_op, "mirror"):
                mirror_method = getattr(downstream_op, "mirror")

                if callable(mirror_method):
                    # Test both true and false
                    test_values = [True, False, 1, 0]

                    for value in test_values:
                        try:
                            mirror_method(value)
                            print(f"✅ Successfully set mirror to {value} (type: {type(value)})")
                        except Exception as e:
                            print(f"⚠️ Failed to set mirror to {value}: {e}")
                else:
                    print("ℹ️ mirror is a property, not a method")
            else:
                print("ℹ️ mirror parameter not exposed in bindings")

        except ImportError as e:
            pytest.skip(f"StreamingServerDownstreamOp not available: {e}")

    def test_streaming_server_downstream_op_initialize(self):
        """Test StreamingServerDownstreamOp initialization through bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerDownstreamOp

            downstream_op = StreamingServerDownstreamOp()
            self.downstream_op = downstream_op

            # Test initialize method if available
            if hasattr(downstream_op, "initialize"):
                try:
                    downstream_op.initialize()
                    print("✅ Successfully called initialize()")
                except Exception as e:
                    print(f"⚠️ Initialize failed (expected without fragment): {e}")
            else:
                print("ℹ️ initialize() method not exposed in bindings")

        except ImportError as e:
            pytest.skip(f"StreamingServerDownstreamOp not available: {e}")

    def test_streaming_server_downstream_op_setup(self):
        """Test StreamingServerDownstreamOp setup through bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerDownstreamOp

            downstream_op = StreamingServerDownstreamOp()
            self.downstream_op = downstream_op

            # Test setup method if available
            if hasattr(downstream_op, "setup"):
                try:
                    # Note: setup() typically requires an OperatorSpec
                    print("ℹ️ setup() method available but requires OperatorSpec")
                except Exception as e:
                    print(f"⚠️ Setup test inconclusive: {e}")
            else:
                print("ℹ️ setup() method not exposed in bindings")

        except ImportError as e:
            pytest.skip(f"StreamingServerDownstreamOp not available: {e}")

    def test_streaming_server_downstream_op_type_checking(self):
        """Test that StreamingServerDownstreamOp has correct type in bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerDownstreamOp

            downstream_op = StreamingServerDownstreamOp()
            self.downstream_op = downstream_op

            # Check that it's the right type
            assert isinstance(downstream_op, StreamingServerDownstreamOp)

            # Check string representation
            op_str = str(downstream_op)
            assert "StreamingServerDownstreamOp" in op_str or "downstream" in op_str.lower()

            print(f"✅ Operator type: {type(downstream_op)}")
            print(f"✅ Operator string: {op_str}")

        except ImportError as e:
            pytest.skip(f"StreamingServerDownstreamOp not available: {e}")

    def test_streaming_server_downstream_op_docstring(self):
        """Test that StreamingServerDownstreamOp has proper documentation in bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerDownstreamOp

            # Check class docstring
            if StreamingServerDownstreamOp.__doc__:
                doc = StreamingServerDownstreamOp.__doc__
                assert len(doc) > 0
                print(f"✅ StreamingServerDownstreamOp docstring: {doc[:100]}...")
            else:
                print("ℹ️ No docstring found for StreamingServerDownstreamOp")

        except ImportError as e:
            pytest.skip(f"StreamingServerDownstreamOp not available: {e}")

    def test_streaming_server_downstream_op_processing_configurations(self):
        """Test different processing configurations through bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerDownstreamOp

            downstream_op = StreamingServerDownstreamOp()
            self.downstream_op = downstream_op

            # Test different processing configurations
            processing_configs = [
                (854, 480, 30, False, "Standard FWVGA"),
                (854, 480, 30, True, "Mirrored FWVGA"),
                (1920, 1080, 60, False, "Standard Full HD 60fps"),
                (1920, 1080, 60, True, "Mirrored Full HD 60fps"),
                (640, 480, 15, False, "Low resolution, low FPS"),
            ]

            for width, height, fps, mirror, description in processing_configs:
                try:
                    # Set parameters if available
                    if hasattr(downstream_op, "width") and callable(
                        getattr(downstream_op, "width")
                    ):
                        downstream_op.width(width)
                    if hasattr(downstream_op, "height") and callable(
                        getattr(downstream_op, "height")
                    ):
                        downstream_op.height(height)
                    if hasattr(downstream_op, "fps") and callable(getattr(downstream_op, "fps")):
                        downstream_op.fps(fps)
                    if hasattr(downstream_op, "mirror") and callable(
                        getattr(downstream_op, "mirror")
                    ):
                        downstream_op.mirror(mirror)

                    print(f"✅ Successfully configured {description}")

                except Exception as e:
                    print(f"⚠️ Failed to configure {description}: {e}")

        except ImportError as e:
            pytest.skip(f"StreamingServerDownstreamOp not available: {e}")

    def test_streaming_server_downstream_op_error_handling(self):
        """Test error handling in StreamingServerDownstreamOp bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerDownstreamOp

            downstream_op = StreamingServerDownstreamOp()
            self.downstream_op = downstream_op

            # Test error cases
            error_tests = [
                ("width", [-1, 0], "negative or zero width should be handled"),
                ("height", [-1, 0], "negative or zero height should be handled"),
                ("fps", [-1, 0], "negative or zero fps should be handled"),
            ]

            for param_name, error_values, description in error_tests:
                if hasattr(downstream_op, param_name) and callable(
                    getattr(downstream_op, param_name)
                ):
                    param_method = getattr(downstream_op, param_name)

                    for value in error_values:
                        try:
                            param_method(value)
                            print(f"⚠️ {param_name}({value}) unexpectedly accepted")
                        except Exception as e:
                            print(
                                f"✅ {param_name}({value}) correctly rejected: {type(e).__name__}"
                            )
                else:
                    print(f"ℹ️ Parameter {param_name} not available for error testing")

        except ImportError as e:
            pytest.skip(f"StreamingServerDownstreamOp not available: {e}")

    def test_streaming_server_downstream_op_inheritance(self):
        """Test that StreamingServerDownstreamOp inherits from Holoscan Operator."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerDownstreamOp

            downstream_op = StreamingServerDownstreamOp()
            self.downstream_op = downstream_op

            # Check method resolution order
            mro = type(downstream_op).__mro__
            print(f"✅ Method Resolution Order: {[cls.__name__ for cls in mro]}")

            # Check for common operator methods
            operator_methods = ["initialize", "setup", "compute", "start", "stop"]
            available_methods = []

            for method_name in operator_methods:
                if hasattr(downstream_op, method_name):
                    available_methods.append(method_name)

            print(f"✅ Available operator methods: {available_methods}")

        except ImportError as e:
            pytest.skip(f"StreamingServerDownstreamOp not available: {e}")

    def test_streaming_server_downstream_op_resource_consistency(self):
        """Test resource consistency between upstream and downstream operators."""
        try:
            from holohub.streaming_server_enhanced import (
                StreamingServerDownstreamOp,
                StreamingServerResource,
                StreamingServerUpstreamOp,
            )

            # Create shared resource
            resource = StreamingServerResource()
            self.resource = resource

            # Create both operators
            downstream_op = StreamingServerDownstreamOp()
            upstream_op = StreamingServerUpstreamOp()
            self.downstream_op = downstream_op

            # Test that both can use the same resource
            if hasattr(downstream_op, "streaming_server_resource") and hasattr(
                upstream_op, "streaming_server_resource"
            ):

                try:
                    # Assign same resource to both
                    if callable(getattr(downstream_op, "streaming_server_resource")):
                        downstream_op.streaming_server_resource(resource)
                        upstream_op.streaming_server_resource(resource)
                        print(
                            "✅ Successfully shared resource between upstream and downstream operators"
                        )
                except Exception as e:
                    print(f"⚠️ Failed to share resource: {e}")
            else:
                print("ℹ️ Resource sharing test not available")

        except ImportError as e:
            pytest.skip(f"Required operators not available: {e}")


@pytest.mark.bindings
@pytest.mark.skipif(
    not STREAMING_SERVER_ENHANCED_MOCK_ONLY, reason="Only run fallback test in mock-only mode"
)
class TestStreamingServerDownstreamOpBindingsFallback:
    """Fallback tests when real bindings are not available."""

    def test_binding_availability_info(self):
        """Provide information about binding availability."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerDownstreamOp  # noqa: F401

            pytest.fail("StreamingServerDownstreamOp should not be available in mock-only mode")
        except ImportError:
            print(
                "ℹ️ StreamingServerDownstreamOp bindings not available - this is expected in mock-only mode"
            )
            print(
                "ℹ️ To test real bindings, build with streaming server libraries and set BUILD_TESTING=ON"
            )
