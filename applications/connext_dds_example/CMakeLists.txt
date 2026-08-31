# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(connext_dds_example LANGUAGES NONE)

# Connext automatically loads this file from the application's working directory.
configure_file(USER_QOS_PROFILES.xml USER_QOS_PROFILES.xml COPYONLY)

add_custom_target(connext_dds_example ALL
  DEPENDS connext_dds_python
)

install(
  FILES connext_dds_example.py USER_QOS_PROFILES.xml
  DESTINATION examples/connext_dds_example
  COMPONENT connext_dds_example-py
)
