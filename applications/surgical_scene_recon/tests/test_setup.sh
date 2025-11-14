#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Test script to verify the training integration setup

set -e  # Exit on error

echo "========================================================================"
echo "  Testing Surgical Scene Reconstruction - Training Integration"
echo "========================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "run_surgical_recon.py" ]; then
    echo "ERROR: Must be run from surgical_scene_recon directory"
    exit 1
fi

echo "[1/4] Checking directory structure..."
echo ""

# Check training directory exists
if [ ! -d "training" ]; then
    echo "ERROR: training/ directory not found"
    exit 1
fi
echo "✓ training/ directory exists"

# Check training files exist
if [ ! -f "training/gsplat_train.py" ]; then
    echo "ERROR: training/gsplat_train.py not found"
    exit 1
fi
echo "✓ training/gsplat_train.py exists"

if [ ! -d "training/scene" ]; then
    echo "ERROR: training/scene/ directory not found"
    exit 1
fi
echo "✓ training/scene/ directory exists"

if [ ! -d "training/utils" ]; then
    echo "ERROR: training/utils/ directory not found"
    exit 1
fi
echo "✓ training/utils/ directory exists"

echo ""
echo "[2/4] Checking operator files..."
echo ""

# Check new operators exist
if [ ! -f "operators/data_accumulator_op.py" ]; then
    echo "ERROR: operators/data_accumulator_op.py not found"
    exit 1
fi
echo "✓ operators/data_accumulator_op.py exists"

if [ ! -f "operators/training_runner_op.py" ]; then
    echo "ERROR: operators/training_runner_op.py not found"
    exit 1
fi
echo "✓ operators/training_runner_op.py exists"

echo ""
echo "[3/4] Checking application scripts..."
echo ""

# Check application scripts
if [ ! -f "run_surgical_recon.py" ]; then
    echo "ERROR: run_surgical_recon.py not found"
    exit 1
fi
echo "✓ run_surgical_recon.py exists"

if [ ! -f "test_train_then_render.py" ]; then
    echo "ERROR: test_train_then_render.py not found"
    exit 1
fi
echo "✓ test_train_then_render.py exists"

# Check existing scripts still exist
if [ ! -f "test_dynamic_rendering_viz.py" ]; then
    echo "ERROR: test_dynamic_rendering_viz.py not found"
    exit 1
fi
echo "✓ test_dynamic_rendering_viz.py exists (inference mode)"

echo ""
echo "[4/4] Checking documentation..."
echo ""

if [ ! -f "TRAINING_MODE.md" ]; then
    echo "ERROR: TRAINING_MODE.md not found"
    exit 1
fi
echo "✓ TRAINING_MODE.md exists"

if [ ! -f "README.md" ]; then
    echo "ERROR: README.md not found"
    exit 1
fi
echo "✓ README.md exists"

echo ""
echo "========================================================================"
echo "  Setup Verification Complete! ✅"
echo "========================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Test inference-only mode:"
echo "   python run_surgical_recon.py \\"
echo "       --mode inference \\"
echo "       --data_dir <data_path> \\"
echo "       --checkpoint <checkpoint_path>"
echo ""
echo "2. Test training mode:"
echo "   python run_surgical_recon.py \\"
echo "       --mode train \\"
echo "       --data_dir <data_path> \\"
echo "       --output_dir output/test \\"
echo "       --training_iterations 500"
echo ""
echo "See TRAINING_MODE.md for detailed documentation."
echo ""
