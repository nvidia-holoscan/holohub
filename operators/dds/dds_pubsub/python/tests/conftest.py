# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the DDS publish and subscribe operator tests."""

import pytest
from holoscan.core import Fragment


class MockOpOutput:
    def __init__(self):
        self.emitted = None

    def emit(self, value, port):
        self.emitted = (value, port)


@pytest.fixture
def fragment():
    return Fragment()


@pytest.fixture
def op_output():
    return MockOpOutput()


@pytest.fixture
def execution_context():
    return None
