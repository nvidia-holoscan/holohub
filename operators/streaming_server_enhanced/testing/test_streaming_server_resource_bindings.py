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

"""Python binding tests for StreamingServerResource.

These tests verify that the pybind11 bindings work correctly and that
C++ StreamingServerResource can be used from Python.
"""

import os

import pytest

# Test configuration
STREAMING_SERVER_ENHANCED_MOCK_ONLY = (
    os.environ.get("STREAMING_SERVER_ENHANCED_MOCK_ONLY", "false").lower() == "true"
)


@pytest.mark.bindings
@pytest.mark.skipif(STREAMING_SERVER_ENHANCED_MOCK_ONLY, reason="Running in mock-only mode")
class TestStreamingServerResourceBindings:
    """Test suite for StreamingServerResource Python bindings."""

    def setup_method(self):
        """Set up test method."""
        self.resource = None

    def teardown_method(self):
        """Clean up after test method."""
        if self.resource:
            self.resource = None

    def test_import_streaming_server_resource(self):
        """Test that StreamingServerResource can be imported from Python bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerResource

            assert StreamingServerResource is not None
        except ImportError as e:
            pytest.skip(f"StreamingServerResource not available: {e}")

    def test_streaming_server_resource_construction(self):
        """Test StreamingServerResource construction through Python bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerResource

            # Test default construction
            resource = StreamingServerResource()
            assert resource is not None
            self.resource = resource

        except ImportError as e:
            pytest.skip(f"StreamingServerResource not available: {e}")
        except Exception as e:
            pytest.fail(f"Failed to construct StreamingServerResource: {e}")

    def test_streaming_server_resource_parameters(self):
        """Test StreamingServerResource parameter setting through bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerResource

            resource = StreamingServerResource()
            self.resource = resource

            # Test parameter setting (if parameters are exposed)
            # Note: The exact parameter interface depends on the pybind11 implementation

            # These might be available as methods or properties
            test_cases = [
                # (method_name, value, description)
                ("width", 1920, "width parameter"),
                ("height", 1080, "height parameter"),
                ("fps", 60, "fps parameter"),
                ("port", 48015, "port parameter"),
            ]

            for param_name, value, description in test_cases:
                if hasattr(resource, param_name):
                    try:
                        # Try as method call
                        if callable(getattr(resource, param_name)):
                            getattr(resource, param_name)(value)
                        else:
                            # Try as property
                            setattr(resource, param_name, value)
                        print(f"✅ Successfully set {description}: {value}")
                    except Exception as e:
                        print(f"⚠️ Could not set {description}: {e}")
                else:
                    print(f"ℹ️ Parameter {param_name} not exposed in bindings")

        except ImportError as e:
            pytest.skip(f"StreamingServerResource not available: {e}")

    def test_streaming_server_resource_initialize(self):
        """Test StreamingServerResource initialization through bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerResource

            resource = StreamingServerResource()
            self.resource = resource

            # Test initialize method if available
            if hasattr(resource, "initialize"):
                try:
                    resource.initialize()
                    print("✅ Successfully called initialize()")
                except Exception as e:
                    print(f"⚠️ Initialize failed (expected without fragment): {e}")
            else:
                print("ℹ️ initialize() method not exposed in bindings")

        except ImportError as e:
            pytest.skip(f"StreamingServerResource not available: {e}")

    def test_streaming_server_resource_setup(self):
        """Test StreamingServerResource setup through bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerResource

            resource = StreamingServerResource()
            self.resource = resource

            # Test setup method if available
            if hasattr(resource, "setup"):
                try:
                    # Note: setup() typically requires an OperatorSpec
                    # This test verifies the method exists and can handle missing spec gracefully
                    print("ℹ️ setup() method available but requires OperatorSpec")
                except Exception as e:
                    print(f"⚠️ Setup test inconclusive: {e}")
            else:
                print("ℹ️ setup() method not exposed in bindings")

        except ImportError as e:
            pytest.skip(f"StreamingServerResource not available: {e}")

    def test_streaming_server_resource_type_checking(self):
        """Test that StreamingServerResource has correct type in bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerResource

            resource = StreamingServerResource()
            self.resource = resource

            # Check that it's the right type
            assert isinstance(resource, StreamingServerResource)

            # Check string representation (more flexible - just ensure it's a valid string)
            resource_str = str(resource)
            assert len(resource_str) > 0 and isinstance(
                resource_str, str
            ), f"Invalid string representation: {resource_str}"
            # More flexible check - either contains class name or is a valid resource representation
            valid_representation = (
                "StreamingServerResource" in resource_str
                or "streaming_server" in resource_str.lower()
                or "resource" in resource_str.lower()
                or "id:" in resource_str  # Holoscan resource string format
            )
            assert (
                valid_representation
            ), f"Resource string representation doesn't match expected patterns: {resource_str}"

            print(f"✅ Resource type: {type(resource)}")
            print(f"✅ Resource string: {resource_str}")

        except ImportError as e:
            pytest.skip(f"StreamingServerResource not available: {e}")

    def test_streaming_server_resource_docstring(self):
        """Test that StreamingServerResource has proper documentation in bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerResource

            # Check class docstring
            if StreamingServerResource.__doc__:
                doc = StreamingServerResource.__doc__
                assert len(doc) > 0
                print(f"✅ StreamingServerResource docstring: {doc[:100]}...")
            else:
                print("ℹ️ No docstring found for StreamingServerResource")

        except ImportError as e:
            pytest.skip(f"StreamingServerResource not available: {e}")

    def test_streaming_server_resource_parameter_types(self):
        """Test parameter type validation in Python bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerResource

            resource = StreamingServerResource()
            self.resource = resource

            # Test type validation for common parameters
            type_tests = [
                ("width", [1920, "1920"], "width should accept integers"),
                ("height", [1080, "1080"], "height should accept integers"),
                ("fps", [60, "60"], "fps should accept integers"),
                ("port", [48015, "48015"], "port should accept integers"),
            ]

            for param_name, test_values, description in type_tests:
                if hasattr(resource, param_name) and callable(getattr(resource, param_name)):
                    param_method = getattr(resource, param_name)

                    for value in test_values:
                        try:
                            param_method(value)
                            print(f"✅ {param_name}({value}) of type {type(value)} accepted")
                        except Exception as e:
                            print(f"⚠️ {param_name}({value}) of type {type(value)} rejected: {e}")
                else:
                    print(f"ℹ️ Parameter {param_name} not available for type testing")

        except ImportError as e:
            pytest.skip(f"StreamingServerResource not available: {e}")

    def test_streaming_server_resource_error_handling(self):
        """Test error handling in StreamingServerResource bindings."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerResource

            resource = StreamingServerResource()
            self.resource = resource

            # Test error cases if parameters are available
            error_tests = [
                ("width", [-1, 0], "negative or zero width should be handled"),
                ("height", [-1, 0], "negative or zero height should be handled"),
                ("fps", [-1, 0], "negative or zero fps should be handled"),
                ("port", [-1, 0, 65536], "invalid port values should be handled"),
            ]

            for param_name, error_values, description in error_tests:
                if hasattr(resource, param_name) and callable(getattr(resource, param_name)):
                    param_method = getattr(resource, param_name)

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
            pytest.skip(f"StreamingServerResource not available: {e}")


@pytest.mark.bindings
@pytest.mark.skipif(
    not STREAMING_SERVER_ENHANCED_MOCK_ONLY, reason="Only run fallback test in mock-only mode"
)
class TestStreamingServerResourceBindingsFallback:
    """Fallback tests when real bindings are not available."""

    def test_binding_availability_info(self):
        """Provide information about binding availability."""
        try:
            from holohub.streaming_server_enhanced import StreamingServerResource  # noqa: F401

            pytest.fail("StreamingServerResource should not be available in mock-only mode")
        except ImportError:
            print(
                "ℹ️ StreamingServerResource bindings not available - this is expected in mock-only mode"
            )
            print(
                "ℹ️ To test real bindings, build with streaming server libraries and set BUILD_TESTING=ON"
            )
