#!/bin/bash

# Parse all arguments to check if xr_hello_holoscan is specified
USE_HELLO_HOLOSCAN=false
for arg in "$@"; do
    if [[ "$arg" == "xr_hello_holoscan" ]]; then
        USE_HELLO_HOLOSCAN=true
        break
    fi
done

if [ $USE_HELLO_HOLOSCAN = true ]; then
    # Run hello holoscan
    ./utils/xr_hello_holoscan/xr_hello_holoscan "$@"
else
    # Default: Run volume rendering
    ./volume_rendering_xr "$@"
fi