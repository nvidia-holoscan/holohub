#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
TOP=$(realpath "${SCRIPT_DIR}/../..")
FLOW_BENCHMARKING_DIR=${TOP}/benchmarks/holoscan_flow_benchmarking

run_command() {
    local status=0
    local cmd="$*"

    if [ "${DO_DRY_RUN}" != "true" ]; then
        echo -e "${YELLOW}[command]${NOCOLOR} ${cmd}"
    else
        echo -e "${YELLOW}[dryrun]${NOCOLOR} ${cmd}"
    fi

    [ "$(echo -n "$@")" = "" ] && return 1 # return 1 if there is no command available

    if [ "${DO_DRY_RUN}" != "true" ]; then
        eval "$@"
        status=$?
    fi
    return $status
}

print_platform_details() {
    echo "Platform details:"
    
    cpu_details=$(lscpu)
    echo "\"cpu\": \"$(echo "$cpu_details" | grep "Model name" | cut -d':' -f2 | xargs)\","

    echo "\"dgpu\": \"$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | xargs)\","
    echo "\"driver\": \"$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | xargs)\","
    echo "\"cuda\": \"$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')\","

    echo "\"os\": \"$(cat /etc/os-release | grep "PRETTY_NAME" | cut -d'"' -f2)\","
}

run_benchmark() {
    ARGS=("$@")
    local runs=3
    local instances=3
    local messages=1000
    local scheduler=greedy
    local output=""
    local headless="false"
    local realtime="false"
    local headless_str="display"
    local realtime_str="offline"
    local app_config=""
    local app=""

    for i in "${!ARGS[@]}"; do
        arg="${ARGS[i]}"
        if [[ $skipnext == "1" ]]; then
            skipnext=0
        elif [[ "$arg" == "--help" ]]; then
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --app <str>         Application to benchmark"
            echo "  --app_config <str>  Application configuration file"
            echo "  --runs <int>        Number of runs"
            echo "  --instances <int>   Number of instances"
            echo "  --messages <int>    Number of messages"
            echo "  --scheduler <str>   Scheduler to use"
            echo "  --output <str>      Output directory"
            echo "  --headless          Run in headless mode"
            echo "  --realtime          Run in real-time mode"
            exit 0
        elif [ "$arg" = "--app" ]; then
            app="${ARGS[i + 1]}"
            skipnext=1
        elif [ "$arg" = "--app_config" ]; then
            app_config="${ARGS[i + 1]}"
            skipnext=1
        elif [ "$arg" = "--runs" ]; then
            runs="${ARGS[i + 1]}"
            skipnext=1
        elif [ "$arg" = "--instances" ]; then
            instances="${ARGS[i + 1]}"
            skipnext=1
        elif [ "$arg" = "--messages" ]; then
            messages="${ARGS[i + 1]}"
            skipnext=1
        elif [ "$arg" = "--scheduler" ]; then
            scheduler="${ARGS[i + 1]}"
            skipnext=1
        elif [ "$arg" = "--output" ]; then
            output="${ARGS[i + 1]}"
            skipnext=1
        elif [ "$arg" = "--headless" ]; then
            headless="true"
        elif [ "$arg" = "--realtime" ]; then
            realtime="true"
        fi
    done

    if [[ -z "${app_config}" || ! -f "${app_config}" ]]; then
        echo "No application configuration filepath specified with '--app_config'"
        exit 1
    fi
    
    if [[ "${headless}" = "true" ]]; then
        headless_str="headless"
    fi
    if [[ "${realtime}" = "true" ]]; then
        realtime_str="realtime"
    fi
    output=${output:-"${SCRIPT_DIR}/output/${app}_${runs}_${instances}_${messages}_${scheduler}_${headless_str}_${realtime_str}"}
    mkdir -p $(dirname ${output})

    sed -i "s/^  headless: .*/  headless: ${headless}/" ${APP_CONFIG_PATH}
    sed -i "s/^  realtime: .*/  realtime: ${realtime}/" ${APP_CONFIG_PATH}

    run_command python \
        ${FLOW_BENCHMARKING_DIR}/benchmark.py \
        -a ${app} \
        -r ${runs} \
        -i ${instances} \
        -m ${messages} \
        --sched ${scheduler} \
        -d ${output}
}

plot_benchmark() {
    ARGS=("$@")
    local statistics_args="--max --avg --stddev --median --min --tail --flatness -p 90 95 99 99.9"
    local log_pattern=""
    local log_groups=""
    local log_dir="${SCRIPT_DIR}"

    for i in "${!ARGS[@]}"; do
        arg="${ARGS[i]}"
        if [[ $skipnext == "1" ]]; then
            skipnext=0
        elif [[ "$arg" == "--help" ]]; then
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --log_dir <str>     parent directory containing log \"output\" subdirectory"
            echo "  --log_pattern <str> log pattern to search for"
            echo "  --stats <str>       statistics arguments for analyze.py"
            echo "  --help              Display this help message"
            exit 0
        elif [ "$arg" = "--log_pattern" ]; then
            log_pattern="${ARGS[i + 1]}"
            skipnext=1
        elif [ "$arg" = "--log_dir" ]; then
            log_dir="${ARGS[i + 1]}"
            skipnext=1
        elif [ "$arg" = "--stats" ]; then
            statistics_args="${ARGS[i + 1]}"
            skipnext=1
        fi
    done

    if [[ -z "${log_pattern}" ]]; then
        echo "No log pattern specified with '--log_pattern'"
        exit 1
    fi
    app_name=$(echo "$log_pattern" | sed -E 's/_[0-9].*//')

    counter=1
    for log_directory in $(find ${log_dir}/output -name ${log_pattern} -type d | sort -d); do
        log_groups+="-g "
        for log_file in $(find ${log_directory} -name "logger_*"); do
            log_groups+="$(realpath ${log_file}) "
        done
        log_groups+="Group${counter} "
        counter=$((counter + 1))
    done
    if [[ -z "${log_groups}" ]]; then
        echo "No logs found for pattern ${log_pattern}"
        exit 1
    fi

    processing_dir="$(realpath ${log_dir})/processed/${log_pattern}"
    mkdir -p ${processing_dir}

    pushd ${processing_dir}
    run_command python \
        ${FLOW_BENCHMARKING_DIR}/analyze.py \
        --save-csv \
        --no-display-graphs \
        ${statistics_args} \
        ${log_groups}

    for statistics_csv in $(find ${processing_dir} -name "*.csv"); do
        run_command python \
            ${FLOW_BENCHMARKING_DIR}/generate_bar_graph.py \
            ${statistics_csv} \
            --app "${app_name}" \
            --title "${app_name}" \
            --quiet
    done
    popd
}

benchmark_endoscopy_tool_tracking() {
    APP_CONFIG_PATH=${TOP}/build/endoscopy_tool_tracking/applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking.yaml

    # Test fewer instances on IGX
    if [ $(uname -m) = "aarch64" ]; then
        instance_range=$(seq 1 3);
    else
        instance_range=$(seq 1 8);
    fi

    # Real time
    for instances in ${instance_range}; do
        run_benchmark \
            --app endoscopy_tool_tracking \
            --app_config ${APP_CONFIG_PATH} \
            --instances ${instances} \
            --realtime
    done

    # Real time disabled ("offline")
    for instances in ${instance_range}; do
        run_benchmark \
            --app endoscopy_tool_tracking \
            --app_config ${APP_CONFIG_PATH} \
            --instances ${instances}
    done

    # Real time disabled ("offline") + visualizations disabled ("headless")
    for instances in ${instance_range}; do
        run_benchmark \
            --app endoscopy_tool_tracking \
            --app_config ${APP_CONFIG_PATH} \
            --instances ${instances} \
            --headless
    done
}

benchmark_multiai_ultrasound() {
    APP_CONFIG_PATH=${TOP}/build/multiai_ultrasound/applications/multiai_ultrasound/cpp/multiai_ultrasound.yaml

    # Test fewer instances on IGX
    if [ $(uname -m) = "aarch64" ]; then
        instance_range=$(seq 1 2);
    else
        instance_range=$(seq 1 3);
    fi

    # Real time
    for instances in ${instance_range}; do
        run_benchmark \
            --app multiai_ultrasound \
            --app_config ${APP_CONFIG_PATH} \
            --instances ${instances} \
            --realtime
    done

    # Real time disabled + visualizations disabled
    for instances in ${instance_range}; do
        run_benchmark \
            --app multiai_ultrasound \
            --app_config ${APP_CONFIG_PATH} \
            --instances ${instances} \
            --headless
    done
}

main() {
    ARGS=("$@")
    local process_only=false
    local data_dirs=()

    for i in "${!ARGS[@]}"; do
        arg="${ARGS[i]}"
        if [[ $skipnext == "1" ]]; then
            skipnext=0;
        elif [[ "$arg" == "--help" ]]; then
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --dryrun             Print commands without running them"
            echo "  --print              Print platform details"
            echo "  --process <path(s)>  Process prior output without running new benchmarks. Multiple --process values can be specified."
            exit 0;
        elif [[ "$arg" == "--dryrun" ]]; then
            DO_DRY_RUN="true";
        elif [[ "$arg" == "--print" ]]; then
            set +x
            print_platform_details
            exit 0;
        elif [[ "$arg" == "--process" ]]; then
            data_dirs+=("${ARGS[i + 1]}")
            process_only=true
            skipnext=1;
        fi
    done

    if [[ -z "${data_dirs}" ]]; then
        data_dirs=("${SCRIPT_DIR}")
    fi

    if [[ ${process_only} = "false" ]]; then
        pushd ${TOP}
        benchmark_endoscopy_tool_tracking
        benchmark_multiai_ultrasound
        popd;
    fi
    
    for data_dir in ${data_dirs[@]}; do
        for pattern_suffix in "display_realtime" "display_offline" "headless_offline"; do
            plot_benchmark \
                --log_pattern "endoscopy_tool_tracking_3_[0-9]_1000_greedy_$pattern_suffix" \
                --log_dir ${data_dir}
        done

        for pattern_suffix in "display_realtime" "headless_offline"; do
            plot_benchmark \
                --log_pattern "multiai_ultrasound_3_[0-9]_1000_greedy_$pattern_suffix" \
                --log_dir ${data_dir}
        done
    done
}

main "$@"
