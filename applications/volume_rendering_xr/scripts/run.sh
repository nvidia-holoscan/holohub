#!/bin/bash

APP_NAME=$1
shift  # Remove first argument, leaving remaining args

if [ -z "$APP_NAME" ]; then
    APP_NAME="volume_rendering_xr"  # Default app
fi

case $APP_NAME in
    "volume_rendering_xr")
        # Pass all remaining arguments to the executable
        ./volume_rendering_xr "$@"
        ;;
    "xr_hello_holoscan")
        ./volume_rendering_xr/utils/xr_hello_holoscan/xr_hello_holoscan "$@"
        ;;
    *)
        echo "Usage: $0 [app_name] [arguments...]"
        echo "Available apps: volume_rendering_xr, xr_hello_holoscan"
        echo "Example:"
        echo "$0 volume_rendering_xr --config <holohub_app_source>/configs/ctnv_bb_er.json --density <holohub_data_dir>/volume_rendering_xr/highResCT.mhd --mask <holohub_data_dir>/volume_rendering_xr/smoothmasks.seg.mhd"
        exit 1
        ;;
esac