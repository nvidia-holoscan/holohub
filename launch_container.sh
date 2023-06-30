#!/bin/bash 
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Error out if a command fails
set -e


#===============================================================================
# Default values for environment variables.
#===============================================================================

init_globals() {
    if [ "$0" != "/bin/bash" ] && [ "$0" != "bash" ]; then
        SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
        export RUN_SCRIPT_FILE="$(readlink -f "$0")"
    else
        export RUN_SCRIPT_FILE="$(readlink -f "${BASH_SOURCE[0]}")"
    fi

    export TOP=$(dirname "${RUN_SCRIPT_FILE}")

    HOLOSCAN_PY_EXE=${HOLOSCAN_PY_EXE:-"python3"}
    export HOLOSCAN_PY_EXE
    HOLOSCAN_DOCKER_EXE=${HOLOSCAN_DOCKER_EXE:-"docker"}
    export HOLOSCAN_DOCKER_EXE

    export HOLOHUB_ROOT=${TOP}

    DO_DRY_RUN="false"  # print commands but do not execute them. Used by run_command
}


init_globals

c_echo() {
    # Select color/nocolor based on the first argument
    local mode="color"
    if [ "${1:-}" = "color" ]; then
        mode="color"
        shift
    elif [ "${1:-}" = "nocolor" ]; then
        mode="nocolor"
        shift
    else
        if [ ! -t 1 ]; then
            mode="nocolor"
        fi
    fi

    local old_opt="$(shopt -op xtrace)" # save old xtrace option
    set +x # unset xtrace

    if [ "${mode}" = "color" ]; then
        local text="$(c_str color "$@")"
        /bin/echo -e "$text\033[0m"
    else
        local text="$(c_str nocolor "$@")"
        /bin/echo -e "$text"
    fi
    eval "${old_opt}" # restore old xtrace option
}

c_str() {
    local old_color=39
    local old_attr=0
    local color=39
    local attr=0
    local text=""
    local mode="color"
    if [ "${1:-}" = "color" ]; then
        mode="color"
        shift
    elif [ "${1:-}" = "nocolor" ]; then
        mode="nocolor"
        shift
    fi

    for i in "$@"; do
        case "$i" in
            r|R)
                color=31
                ;;
            g|G)
                color=32
                ;;
            y|Y)
                color=33
                ;;
            b|B)
                color=34
                ;;
            p|P)
                color=35
                ;;
            c|C)
                color=36
                ;;
            w|W)
                color=37
                ;;

            z|Z)
                color=0
                ;;
        esac
        case "$i" in
            l|L|R|G|Y|B|P|C|W)
                attr=1
                ;;
            n|N|r|g|y|b|p|c|w)
                attr=0
                ;;
            z|Z)
                attr=0
                ;;
            *)
                text="${text}$i"
        esac
        if [ "${mode}" = "color" ]; then
            if [ ${old_color} -ne ${color} ] || [ ${old_attr} -ne ${attr} ]; then
                text="${text}\033[${attr};${color}m"
                old_color=$color
                old_attr=$attr
            fi
        fi
    done
    /bin/echo -en "$text"
}

c_echo_err() {
    >&2 c_echo "$@"
}

#######################################
# Check if current architecture is x86_64.
#
# Returns:
#   Exit code:
#     0 if $(uname -m) == "x86_64".
#     1 otherwise.
#######################################
checkif_x86_64() {
    if [ $(uname -m) == "x86_64" ]; then
        return 0
    else
        return 1
    fi
}

#######################################
# Check if current architecture is aarch64.
#
# Returns:
#   Exit code:
#     0 if $(uname -m) == "aarch64".
#     1 otherwise.
#######################################
checkif_aarch64() {
    if [ $(uname -m) == "aarch64" ]; then
        return 0
    else
        return 1
    fi
}


setup() {
    c_echo W "Setup development environment..."

    if ! command -v ${HOLOSCAN_DOCKER_EXE} > /dev/null; then
        fatal G "${HOLOSCAN_DOCKER_EXE}" W " doesn't exists. Please install NVIDIA Docker!"
    fi

    if ! groups | grep -q docker; then
        c_echo_err G "groups" W " doesn't contain 'docker' group. Please add 'docker' group to your user."
        fatal G "groups" W " doesn't contain 'docker' group. Please add 'docker' group to your user." B '
    # Create the docker group.
    sudo groupadd docker
    # Add your user to the docker group.
    sudo usermod -aG docker $USER
    newgrp docker
    docker run hello-world'
    fi

    if checkif_x86_64 && [ -n "${HOLOSCAN_BUILD_PLATFORM}" ] && [ ! -f /proc/sys/fs/binfmt_misc/qemu-aarch64 ]; then
        fatal G "qemu-aarch64" W " doesn't exists. Please install qemu with binfmt-support to run Docker container with aarch64 platform" B '
    # Install the qemu packages
    sudo apt-get install qemu binfmt-support qemu-user-static
    # Execute the registering scripts
    docker run --rm --privileged multiarch/qemu-user-static --reset -p yes'
    fi
}

run_command() {
    local status=0
    local cmd="$*"

    if [ "${DO_DRY_RUN}" != "true" ]; then
        c_echo_err B "$(date -u '+%Y-%m-%d %H:%M:%S') " W "\$ " G "${cmd}"
    else
        c_echo_err B "$(date -u '+%Y-%m-%d %H:%M:%S') " C "[dryrun] " W "\$ " G "${cmd}"
    fi

    [ "$(echo -n "$@")" = "" ] && return 1 # return 1 if there is no command available

    if [ "${DO_DRY_RUN}" != "true" ]; then
        "$@"
        status=$?
    fi

    return $status
}

#===============================================================================
# Section: Launch
#===============================================================================

launch_desc() { c_echo 'Launch Docker container
Export CMAKE_BUILD_PATH (default: "build") to change the build path.
  e.g.,
    export CMAKE_BUILD_PATH=build-arm64
Arguments:
    $1 - Working directory (e.g, "install" => "/workspace/holoscan-sdk/install")
         Default: "build"
'
}

launch() {
    local build_path="${CMAKE_BUILD_PATH:-build}"
    local working_dir=${1:-${build_path}}
    local mount_device_opt=""

    # Skip the first argument to pass the remaining arguments to the docker command.
    if [ -n "$1" ]; then
        shift
    fi

    setup

    # Allow connecting from docker. This is not needed for WSL2 (`SI:localuser:wslg` is added by default)
    run_command xhost +local:docker

    for i in 0 1 2 3; do
        if [ -e /dev/video${i} ]; then
            mount_device_opt+=" --device /dev/video${i}:/dev/video${i}"
        fi
        if [ -e /dev/ajantv2${i} ]; then
            mount_device_opt+=" --device /dev/ajantv2${i}:/dev/ajantv2${i}"
        fi
	# Deltacast SDI capture board
	if [ -e /dev/delta-x380${i} ]; then
            mount_device_opt+=" --device /dev/delta-x380${i}:/dev/delta-x380${i}"
        fi
	# Deltacast HDMI capture board
	if [ -e /dev/delta-x350${i} ]; then
            mount_device_opt+=" --device /dev/delta-x350${i}:/dev/delta-x350${i}"
        fi
    done

    c_echo W "Launching (mount_device_opt:" G "${mount_device_opt}" W ")..."

    # Find the nvidia_icd.json file which could reside at different paths
    # Needed due to https://github.com/NVIDIA/nvidia-container-toolkit/issues/16
    nvidia_icd_json=$(find /usr/share /etc -path '*/vulkan/icd.d/nvidia_icd.json' -type f -print -quit 2>/dev/null | grep .) || (echo "nvidia_icd.json not found" >&2 && false)

    # DOCKER PARAMETERS
    #
    # -it
    #   The container needs to be interactive to be able to interact with the X11 windows
    #
    # --rm
    #   Deletes the container after the command runs
    #
    # -u $(id -u):$(id -g)
    # -v /etc/group:/etc/group:ro
    # -v /etc/passwd:/etc/passwd:ro
    #   Ensures the generated files (build, install...) are owned by $USER and not root,
    #   and provide the configuration files to avoid warning for user and group names
    #
    # -v ${TOP}:/workspace/holoscan-sdk
    #   Mount the source directory
    #
    # -w /workspace/holoscan-sdk/${working_dir}
    #   Start in the build or install directory
    #
    # --runtime=nvidia \
    # -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display
    #   Enable GPU acceleration
    #
    # -v /tmp/.X11-unix:/tmp/.X11-unix
    # -e DISPLAY
    #   Enable graphical applications
    #
    # -v $nvidia_icd_json:$nvidia_icd_json:ro
    #   Bind NVIDIA's Vulkan installable client driver to run Vulkan
    #   Needed due to https://github.com/NVIDIA/nvidia-container-toolkit/issues/16
    #   The configurations files are installed to different locations when installing
    #   with deb packages or with run files, so we look at both places
    #
    # --device /dev/video${i}:/dev/video${i}
    #   Bind video capture devices for V4L2
    #
    # --device /dev/ajantv2${i}:/dev/ajantv2${i}
    #   Bind AJA capture cards for NTV2
    #
    # -e PYTHONPATH
    # -e HOLOSCAN_LIB_PATH
    # -e HOLOSCAN_SAMPLE_DATA_PATH
    #   Define paths needed by the python applications
    #
    # -e CUPY_CACHE_DIR
    #   Define path for cupy' kernel cache, needed since $HOME does
    #   not exist when running with `-u id:group`

    run_command ${HOLOSCAN_DOCKER_EXE} run -it --rm --net host \
        -u $(id -u):$(id -g) \
        -v /etc/group:/etc/group:ro \
        -v /etc/passwd:/etc/passwd:ro \
        -v ${HOLOHUB_ROOT}:/workspace/holohub \
        -v ${HOLOHUB_ROOT}/../holoscan-sdk:/workspace/holoscan-sdk \
        -v /opt/deltacast/videomaster/include:/usr/local/deltacast/Include \
        -v /usr/lib/libvideomasterhd.so:/usr/lib/libvideomasterhd.so \
        -w /workspace/holohub \
        --runtime=nvidia \
        -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -e DISPLAY \
        ${mount_device_opt} \
        -v $nvidia_icd_json:$nvidia_icd_json:ro \
        -e PYTHONPATH=/workspace/holoscan-sdk/${working_dir}/python/lib \
        -e HOLOSCAN_LIB_PATH=/workspace/holoscan-sdk/${working_dir}/lib \
        -e HOLOSCAN_SAMPLE_DATA_PATH=/workspace/holoscan-sdk/data \
        -e HOLOSCAN_TESTS_DATA_PATH=/workspace/holoscan-sdk/tests/data \
        -e CUPY_CACHE_DIR=/workspace/holoscan-sdk/.cupy/kernel_cache \
        holohub:sdk_0.5.1 "$@"
}

launch
