# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(connext_dds_roundtrip LANGUAGES NONE)

include(../connext_dds_common/ConnextDDSExample.cmake)
add_connext_dds_example(connext_dds_roundtrip connext_dds_roundtrip.py)
