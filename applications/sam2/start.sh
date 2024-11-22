#!/bin/bash

# change the pythonpath
export PYTHONPATH="/workspace/sam2"
export PYTHONPATH="/var/lib/holoscan/opt/nvidia/holoscan/python/lib":$PYTHONPATH

# launch the application
python /workspace/holohub/applications/sam2/segment_one_thing.py