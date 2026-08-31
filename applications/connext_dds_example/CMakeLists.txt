# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(connext_dds_example LANGUAGES NONE)

add_custom_target(connext_dds_example ALL
  DEPENDS connext_dds_python
)

install(
  FILES connext_dds_example.py
  DESTINATION examples/connext_dds_example
  COMPONENT connext_dds_example-py
)
