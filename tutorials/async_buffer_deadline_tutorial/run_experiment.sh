#!/bin/bash

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

# Period Variation Experiment Runner
# This script runs a period variation experiment to compare default buffer vs async lock-free buffer

echo "=========================================="
echo "Period Variation Experiment"
echo "=========================================="
echo "This experiment will:"
echo "1. Test with fixed TX1 period (20ms) and varying TX2 period (20-100ms)"
echo "2. Test with fixed TX2 period (20ms) and varying TX1 period (20-100ms)"
echo "3. Compare default buffer vs async lock-free buffer"
echo "4. Generate plots for latency and message intervals"
echo "=========================================="

# Check if Python dependencies are available
python3 -c "import pandas, matplotlib, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required Python packages not found. Please install:"
    echo "  pip install pandas matplotlib numpy"
    exit 1
fi

# Figure out if we are in the current directory where the script is located
# if not, then change directory to where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(pwd)"

if [ "$SCRIPT_DIR" != "$CURRENT_DIR" ]; then
    echo "We are not in the script directory. Changing to script directory..."
    cd "$SCRIPT_DIR"
    echo "Changed to script directory: $SCRIPT_DIR"
fi

# Check if the binary exists
if [ ! -f "../../build/async_buffer_deadline/applications/async_buffer_deadline/async_buffer_deadline" ]; then
    echo "Error: Binary not found. Please build the project first:"
    echo "  ./holohub build async_buffer_deadline"
    exit 1
fi

# Run the experiment
echo "Starting experiment..."
python3 run_period_experiment.py

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Experiment completed successfully!"
    echo "Results saved in period_variation/ directory"
    echo "Generated plots:"
    echo "  - tx1_latency_vs_tx2_period.png"
    echo "  - tx2_latency_vs_tx1_period.png"
    echo "  - in1_period_vs_tx2_period.png"
    echo "  - in2_period_vs_tx1_period.png"
    echo "=========================================="
else
    echo "=========================================="
    echo "Experiment failed!"
    echo "Check the error messages above."
    echo "=========================================="
    exit 1
fi
