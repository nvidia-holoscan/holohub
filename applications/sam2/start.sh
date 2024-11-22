#!/bin/bash

# change the pythonpath
export PYTHONPATH="${PYTHONPATH}:/workspace/sam2"
export PYTHONPATH="${PYTHONPATH}:/opt/nvidia/holoscan/python/lib"

# launch the application
python /workspace/holohub/applications/sam2/segment_one_thing.py