#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Complete Training + Rendering Workflow - Single Command
#
# This script runs the complete workflow in production Docker:
# 1. Data Accumulation
# 2. Training  
# 3. Rendering with Trained Checkpoint
#
# All in one command!
#
# Usage:
#   ./run_complete_docker.sh [quick|production]
#
# Modes:
#   quick      - 30 frames, 500 iterations (~15 min)
#   production - all frames, 2000 iterations (~30 min)
#

set -e  # Exit on error

MODE="${1:-quick}"

echo "========================================================================"
echo "  Complete Workflow: Train + Render"
echo "========================================================================"
echo "Mode: $MODE"
echo "Started at: $(date)"
echo "========================================================================"
echo ""

# Configuration - Use environment variables
if [ -z "$HOLOHUB_PATH" ]; then
    echo "ERROR: HOLOHUB_PATH environment variable not set"
    echo "Please set it to your holohub-internal directory:"
    echo "  export HOLOHUB_PATH=/path/to/holohub-internal"
    exit 1
fi

if [ -z "$DATA_PATH" ]; then
    echo "ERROR: DATA_PATH environment variable not set"
    echo "Please set it to your dataset directory:"
    echo "  export DATA_PATH=/path/to/your/datasets"
    exit 1
fi

cd "$HOLOHUB_PATH"

# Set parameters based on mode
if [ "$MODE" == "quick" ]; then
    OUTPUT_DIR="output/docker_complete_quick"
    ITERATIONS=500
    COARSE_ITER=50
    NUM_FRAMES=30
    echo "Quick mode: 30 frames, 500 iterations (~15 minutes)"
elif [ "$MODE" == "production" ]; then
    OUTPUT_DIR="output/docker_complete_production"
    ITERATIONS=2000
    COARSE_ITER=200
    NUM_FRAMES=-1  # All frames
    echo "Production mode: all frames, 2000 iterations (~30 minutes)"
else
    echo "ERROR: Unknown mode '$MODE'. Use 'quick' or 'production'"
    exit 1
fi

echo ""

# Clean up old output (if exists) - try without sudo first
if [ -d "applications/surgical_scene_recon/$OUTPUT_DIR" ]; then
    echo "Cleaning up previous output..."
    rm -rf "applications/surgical_scene_recon/$OUTPUT_DIR" 2>/dev/null && echo "✓ Cleaned" || {
        echo "Note: Previous output owned by root. Run manually if needed:"
        echo "  sudo rm -rf applications/surgical_scene_recon/$OUTPUT_DIR"
        echo "Continuing anyway (Docker will overwrite)..."
    }
    echo ""
fi

# Stage 1 + 2: Data Accumulation & Training
echo "========================================================================"
echo "  STAGE 1 & 2: Data Accumulation + Training"
echo "========================================================================"
echo ""

docker run --rm --gpus all \
  -v "$(pwd)":/workspace/holohub \
  -v "$DATA_PATH":/workspace/data \
  -w /workspace/holohub/applications/surgical_scene_recon \
  surgical_scene_recon:latest \
  python train_standalone.py \
    --data_dir /workspace/data/EndoNeRF/pulling \
    --output_dir $OUTPUT_DIR \
    --training_iterations $ITERATIONS \
    --coarse_iterations $COARSE_ITER \
    --num_frames $NUM_FRAMES

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "✅ Stage 1 & 2 Complete: Training finished"
echo ""

# Note: Skip auto-fixing permissions to avoid blocking on sudo
# Users can fix permissions manually after if needed
echo "Note: Files created by Docker are owned by root"
echo "To fix permissions later, run:"
echo "  sudo chown -R \$USER:\$USER applications/surgical_scene_recon/$OUTPUT_DIR"
echo ""

# Find checkpoint
CHECKPOINT="applications/surgical_scene_recon/$OUTPUT_DIR/trained_model/ckpts/fine_best_psnr.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "✅ Checkpoint ready: $CHECKPOINT"
echo ""

# Stage 3: Rendering with Continuous Loop (1 minute auto-timeout)
echo "========================================================================"
echo "  STAGE 3: Rendering with Trained Checkpoint"
echo "========================================================================"
echo ""
echo "Rendering ALL frames in continuous loop..."
echo "Showing surgical tissue reconstruction with your trained model"
echo ""
echo "Auto-terminating after 3 minutes of visualization"
echo ""

timeout 180 docker run --rm --gpus all \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)":/workspace/holohub \
  -v "$DATA_PATH":/workspace/data \
  -w /workspace/holohub/applications/surgical_scene_recon \
  surgical_scene_recon:latest \
  python test_dynamic_rendering_viz.py \
    --data_dir /workspace/data/EndoNeRF/pulling \
    --checkpoint /workspace/holohub/$CHECKPOINT

RENDER_EXIT=$?
if [ $RENDER_EXIT -eq 124 ]; then
    echo ""
    echo "✅ Visualization completed (3 minute timeout)"
elif [ $RENDER_EXIT -ne 0 ]; then
    echo ""
    echo "⚠ Rendering may have failed (display issues expected in headless)"
fi

echo ""
echo "========================================================================"
echo "  COMPLETE WORKFLOW FINISHED! 🎉"
echo "========================================================================"
echo ""
echo "Results:"
echo "  Data: $OUTPUT_DIR/training_ingestion/"
echo "  Model: $OUTPUT_DIR/trained_model/"
echo "  Checkpoint: $CHECKPOINT"
echo "  Renders: $OUTPUT_DIR/../rendered_dynamic/"
echo ""
echo "To re-run rendering:"
echo "  python test_dynamic_rendering_viz.py \\"
echo "      --data_dir /workspace/data/EndoNeRF/pulling \\"
echo "      --checkpoint /workspace/holohub/$CHECKPOINT"
echo ""
echo "Completed at: $(date)"
echo "========================================================================"
