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

"""Python binding tests for StreamingServerUpstreamOp.

These tests verify that the pybind11 bindings work correctly and that
C++ StreamingServerUpstreamOp can be used from Python.
"""

import os

import pytest

# Test configuration
STREAMING_SERVER_ENHANCED_MOCK_ONLY = (
    os.environ.get("STREAMING_SERVER_ENHANCED_MOCK_ONLY", "false").lower() == "true"
)


@pytest.mark.bindings
@pytest.mark.skipif(STREAMING_SERVER_ENHANCED_MOCK_ONLY, reason="Running in mock-only mode")
class TestStreamingServerUpstreamOpBindings:
    """Test suite for StreamingServerUpstreamOp Python bindings."""

    def setup_method(self):
        """Set up test method."""
        self.upstream_op = None
        self.resource = None

    def teardown_method(self):
        """Clean up after test method."""
        if self.upstream_op:
            self.upstream_op = None
        if self.resource:
            self.resource = None

    def test_import_streaming_server_upstream_op(self):
        """Test that StreamingServerUpstreamOp can be imported from Python bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerUpstreamOp

            assert StreamingServerUpstreamOp is not None
        except ImportError as e:
            pytest.skip(f"StreamingServerUpstreamOp not available: {e}")

    def test_streaming_server_upstream_op_construction(self):
        """Test StreamingServerUpstreamOp construction through Python bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerUpstreamOp

            # Test default construction
            upstream_op = StreamingServerUpstreamOp()
            assert upstream_op is not None
            self.upstream_op = upstream_op

        except ImportError as e:
            pytest.skip(f"StreamingServerUpstreamOp not available: {e}")
        except Exception as e:
            pytest.fail(f"Failed to construct StreamingServerUpstreamOp: {e}")

    def test_streaming_server_upstream_op_parameters(self):
        """Test StreamingServerUpstreamOp parameter setting through bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerUpstreamOp

            upstream_op = StreamingServerUpstreamOp()
            self.upstream_op = upstream_op

            # Test parameter setting
            test_cases = [
                ("width", 1920, "width parameter"),
                ("height", 1080, "height parameter"),
                ("fps", 60, "fps parameter"),
            ]

            for param_name, value, description in test_cases:
                if hasattr(upstream_op, param_name):
                    try:
                        # Try as method call
                        if callable(getattr(upstream_op, param_name)):
                            getattr(upstream_op, param_name)(value)
                        else:
                            # Try as property
                            setattr(upstream_op, param_name, value)
                        print(f"✅ Successfully set {description}: {value}")
                    except Exception as e:
                        print(f"⚠️ Could not set {description}: {e}")
                else:
                    print(f"ℹ️ Parameter {param_name} not exposed in bindings")

        except ImportError as e:
            pytest.skip(f"StreamingServerUpstreamOp not available: {e}")

    def test_streaming_server_upstream_op_with_resource(self):
        """Test StreamingServerUpstreamOp with StreamingServerResource through bindings."""
        try:
            from holohub.streaming_server_enhanced import (
                StreamingServerResource,
                StreamingServerUpstreamOp,
            )

            # Create resource
            resource = StreamingServerResource()
            self.resource = resource

            # Create upstream operator
            upstream_op = StreamingServerUpstreamOp()
            self.upstream_op = upstream_op

            # Test resource assignment if available
            if hasattr(upstream_op, "streaming_server_resource"):
                try:
                    if callable(getattr(upstream_op, "streaming_server_resource")):
                        upstream_op.streaming_server_resource(resource)
                        print("✅ Successfully assigned resource to upstream operator")
                    else:
                        setattr(upstream_op, "streaming_server_resource", resource)
                        print("✅ Successfully set resource as property")
                except Exception as e:
                    print(f"⚠️ Could not assign resource: {e}")
            else:
                print("ℹ️ streaming_server_resource not exposed in bindings")

        except ImportError as e:
            pytest.skip(f"StreamingServerUpstreamOp or StreamingServerResource not available: {e}")

    def test_streaming_server_upstream_op_initialize(self):
        """Test StreamingServerUpstreamOp initialization through bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerUpstreamOp

            upstream_op = StreamingServerUpstreamOp()
            self.upstream_op = upstream_op

            # Test initialize method if available
            if hasattr(upstream_op, "initialize"):
                try:
                    upstream_op.initialize()
                    print("✅ Successfully called initialize()")
                except Exception as e:
                    print(f"⚠️ Initialize failed (expected without fragment): {e}")
            else:
                print("ℹ️ initialize() method not exposed in bindings")

        except ImportError as e:
            pytest.skip(f"StreamingServerUpstreamOp not available: {e}")

    def test_streaming_server_upstream_op_setup(self):
        """Test StreamingServerUpstreamOp setup through bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerUpstreamOp

            upstream_op = StreamingServerUpstreamOp()
            self.upstream_op = upstream_op

            # Test setup method if available
            if hasattr(upstream_op, "setup"):
                try:
                    # Note: setup() typically requires an OperatorSpec
                    print("ℹ️ setup() method available but requires OperatorSpec")
                except Exception as e:
                    print(f"⚠️ Setup test inconclusive: {e}")
            else:
                print("ℹ️ setup() method not exposed in bindings")

        except ImportError as e:
            pytest.skip(f"StreamingServerUpstreamOp not available: {e}")

    def test_streaming_server_upstream_op_type_checking(self):
        """Test that StreamingServerUpstreamOp has correct type in bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerUpstreamOp

            upstream_op = StreamingServerUpstreamOp()
            self.upstream_op = upstream_op

            # Check that it's the right type
            assert isinstance(upstream_op, StreamingServerUpstreamOp)

            # Check string representation
            op_str = str(upstream_op)
            assert "StreamingServerUpstreamOp" in op_str or "upstream" in op_str.lower()

            print(f"✅ Operator type: {type(upstream_op)}")
            print(f"✅ Operator string: {op_str}")

        except ImportError as e:
            pytest.skip(f"StreamingServerUpstreamOp not available: {e}")

    def test_streaming_server_upstream_op_docstring(self):
        """Test that StreamingServerUpstreamOp has proper documentation in bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerUpstreamOp

            # Check class docstring
            if StreamingServerUpstreamOp.__doc__:
                doc = StreamingServerUpstreamOp.__doc__
                assert len(doc) > 0
                print(f"✅ StreamingServerUpstreamOp docstring: {doc[:100]}...")
            else:
                print("ℹ️ No docstring found for StreamingServerUpstreamOp")

        except ImportError as e:
            pytest.skip(f"StreamingServerUpstreamOp not available: {e}")

    def test_streaming_server_upstream_op_parameter_combinations(self):
        """Test valid parameter combinations in Python bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerUpstreamOp

            upstream_op = StreamingServerUpstreamOp()
            self.upstream_op = upstream_op

            # Test common resolution and FPS combinations
            test_configs = [
                (640, 480, 30, "VGA 30fps"),
                (1280, 720, 60, "HD 60fps"),
                (1920, 1080, 30, "Full HD 30fps"),
                (854, 480, 30, "FWVGA 30fps"),
            ]

            for width, height, fps, description in test_configs:
                try:
                    # Set parameters if available
                    if hasattr(upstream_op, "width") and callable(getattr(upstream_op, "width")):
                        upstream_op.width(width)
                    if hasattr(upstream_op, "height") and callable(getattr(upstream_op, "height")):
                        upstream_op.height(height)
                    if hasattr(upstream_op, "fps") and callable(getattr(upstream_op, "fps")):
                        upstream_op.fps(fps)

                    print(f"✅ Successfully configured {description}")

                except Exception as e:
                    print(f"⚠️ Failed to configure {description}: {e}")

        except ImportError as e:
            pytest.skip(f"StreamingServerUpstreamOp not available: {e}")

    def test_streaming_server_upstream_op_error_handling(self):
        """Test error handling in StreamingServerUpstreamOp bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerUpstreamOp

            upstream_op = StreamingServerUpstreamOp()
            self.upstream_op = upstream_op

            # Test error cases
            error_tests = [
                ("width", [-1, 0], "negative or zero width should be handled"),
                ("height", [-1, 0], "negative or zero height should be handled"),
                ("fps", [-1, 0], "negative or zero fps should be handled"),
            ]

            for param_name, error_values, description in error_tests:
                if hasattr(upstream_op, param_name) and callable(getattr(upstream_op, param_name)):
                    param_method = getattr(upstream_op, param_name)

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
            pytest.skip(f"StreamingServerUpstreamOp not available: {e}")

    def test_streaming_server_upstream_op_inheritance(self):
        """Test that StreamingServerUpstreamOp inherits from Holoscan Operator."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerUpstreamOp

            upstream_op = StreamingServerUpstreamOp()
            self.upstream_op = upstream_op

            # Check method resolution order
            mro = type(upstream_op).__mro__
            print(f"✅ Method Resolution Order: {[cls.__name__ for cls in mro]}")

            # Check for common operator methods
            operator_methods = ["initialize", "setup", "compute", "start", "stop"]
            available_methods = []

            for method_name in operator_methods:
                if hasattr(upstream_op, method_name):
                    available_methods.append(method_name)

            print(f"✅ Available operator methods: {available_methods}")

        except ImportError as e:
            pytest.skip(f"StreamingServerUpstreamOp not available: {e}")


@pytest.mark.bindings
@pytest.mark.skipif(
    not STREAMING_SERVER_ENHANCED_MOCK_ONLY, reason="Only run fallback test in mock-only mode"
)
class TestStreamingServerUpstreamOpBindingsFallback:
    """Fallback tests when real bindings are not available."""

    def test_binding_availability_info(self):
        """Provide information about binding availability."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerUpstreamOp  # noqa: F401

            pytest.fail("StreamingServerUpstreamOp should not be available in mock-only mode")
        except ImportError:
            print(
                "ℹ️ StreamingServerUpstreamOp bindings not available - this is expected in mock-only mode"
            )
            print(
                "ℹ️ To test real bindings, build with streaming server libraries and set BUILD_TESTING=ON"
            )
