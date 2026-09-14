#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

case "$(uname -m)" in
  aarch64 | arm64)
    export CONNEXTDDS_ARCH=armv8Linux4gcc8.5.0
    ;;
  x86_64 | amd64)
    export CONNEXTDDS_ARCH=x64Linux4gcc8.5.0
    ;;
  *)
    echo "Unsupported RTI Connext host architecture: $(uname -m)" >&2
    return 1
    ;;
esac
