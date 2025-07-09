#!/bin/bash
GIT_ROOT=$(readlink -f ./$(git rev-parse --show-cdup))
export HOLOSCAN_INPUT_PATH="$GIT_ROOT/data/imaging_ai_segmentator/dicom"
export HOLOSCAN_MODEL_PATH="$GIT_ROOT/data/imaging_ai_segmentator/models"
export HOLOSCAN_OUTPUT_PATH="$GIT_ROOT/build/imaging_ai_segmentator/output"
