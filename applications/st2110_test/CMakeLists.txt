# SPDX-FileCopyrightText: Copyright (c) 2025 XRlabs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(st2110_test)

# This is a Python-only test application
# No C++ compilation required

# Install Python application files
install(FILES
  st2110_test_app.py
  st2110_test_config.yaml
  README.md
  DESTINATION examples/st2110_test
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
              GROUP_READ GROUP_EXECUTE
              WORLD_READ WORLD_EXECUTE
)
