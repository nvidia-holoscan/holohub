# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os
import subprocess
import sys
import threading
import time

from nvitop import Device

# global variables for GPU monitoring
stop_gpu_monitoring = False
stop_gpu_monitoring_lock = threading.Lock()


def monitor_gpu(gpu_uuids, filename):
    print("Monitoring GPU utilization in a separate thread")
    devices = []
    if gpu_uuids == "all" or gpu_uuids == "" or gpu_uuids is None:
        devices = Device.all()
    else:
        devices = [Device(uuid=uuid) for uuid in gpu_uuids.split(",")]
    # run every 2 seconds
    global stop_gpu_monitoring
    stop_gpu_monitoring = False
    average_gpu_utilizations = []
    while True:
        stop_gpu_monitoring_lock.acquire()
        if stop_gpu_monitoring:
            stop_gpu_monitoring_lock.release()
            break
        stop_gpu_monitoring_lock.release()
        average_gpu_utilizations.append(
            sum(device.gpu_utilization() for device in devices) / len(devices)
        )
        time.sleep(2)
        # write average gpu utilization to a file and a new line

    with open(filename, "w") as f:
        # discard first 2 and last 2 values
        average_text = ",".join(map(str, average_gpu_utilizations[2:-2]))
        f.write(str(average_text) + "\n")


def run_command(app_launch_command, env):
    try:
        result = subprocess.run(
            [app_launch_command],
            shell=True,
            env=env,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"stdout:\n{e.stdout.decode('utf-8')}")
        print(f"stderr:\n{e.stderr.decode('utf-8')}")
        sys.exit(1)
    if result.returncode != 0:
        # The command returned an error
        print(f"Error: Command {app_launch_command} exited with code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run performance evaluation for a HoloHub application",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    requiredArgument = parser.add_argument_group("required arguments")
    requiredArgument.add_argument(
        "--sched",
        nargs="+",
        choices=["greedy", "multithread"],
        required=True,
        help="scheduler(s) to use",
    )

    parser.add_argument(
        "-a",
        "--holohub-application",
        type=str,
        required=False,
        help="name of HoloHub application to run. It runs the cpp version of an application\n\
To run a python application, use --run-command option.",
        default="endoscopy_tool_tracking",
    )

    parser.add_argument(
        "--run-command",
        type=str,
        required=False,
        help="command to run the application (this argument overwrites \
-a or --holohub-application parameter)",
        default="",
    )

    parser.add_argument(
        "-d",
        "--log-directory",
        type=str,
        required=False,
        help="directory where the log results will be stored",
    )

    parser.add_argument(
        "-g",
        "--gpu",
        type=str,
        required=False,
        help="comma-separated GPU UUIDs to run the application on.\
This option sets the CUDA_VISIBLE_DEVICES in the environment variable.\
(default: all)\nWarning: This option does not override any custom GPU\
assignment in Holoscan's Inference operator.",
        default="all",
    )

    parser.add_argument(
        "-i",
        "--instances",
        type=int,
        required=False,
        help="number of application instances to run in parallel (default: 1)",
        default=1,
    )

    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        default=1,
        help="number of times to repeat the experiment (default: 1)",
        required=False,
    )

    parser.add_argument(
        "-m",
        "--num_messages",
        type=int,
        default=100,
        help="number of messages or data frames to consider in benchmarking (default: 100)",
        required=False,
    )

    parser.add_argument(
        "-w",
        "--num_worker_threads",
        type=int,
        default=1,
        help="number of worker threads for multithread scheduler (default: 1)",
        required=False,
    )

    parser.add_argument(
        "-u", "--monitor_gpu", action="store_true", help="enable this to monitor GPU utilization"
    )

    args = parser.parse_args()

    if "multithread" not in args.sched and args.num_worker_threads != 1:
        print("Warning: num_worker_threads is ignored as multithread scheduler is not used")

    log_directory = None
    if args.log_directory is None:
        # create a timestamped directory: log_directory_<timestamp> in the current directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_directory = "log_directory_" + timestamp
        os.mkdir(log_directory)
    else:
        # check if the given directory is valid or not
        log_directory = args.log_directory
        if not os.path.isdir(log_directory):
            print(
                "Log directory is not found. Creating a new directory at",
                os.path.abspath(log_directory),
            )
            os.mkdir(os.path.abspath(log_directory))

    # if args.not_holohub or args.binary_path is not None:
    #     print ("Currently non-HoloHub applications are not supported")
    #     sys.exit(1)

    env = os.environ.copy()
    if args.gpu != "all":
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.num_messages != 100:
        env["HOLOSCAN_NUM_SOURCE_MESSAGES"] = str(args.num_messages)

    if args.run_command == "":
        app_launch_command = "./run launch " + args.holohub_application + " cpp"
    else:
        app_launch_command = args.run_command

    log_files = []
    gpu_utilization_log_files = []
    for scheduler in args.sched:
        if scheduler == "multithread":
            env["HOLOSCAN_SCHEDULER"] = scheduler
            env["HOLOSCAN_MULTITHREAD_WORKER_THREADS"] = str(args.num_worker_threads)
        elif scheduler != "greedy":
            print("Unsupported scheduler ", scheduler)
            sys.exit(1)
        # No need to set the scheduler for greedy scheduler

        for i in range(1, args.runs + 1):
            instance_threads = []
            if args.monitor_gpu:
                gpu_utilization_logfile_name = (
                    "gpu_utilization_" + scheduler + "_" + str(i) + ".csv"
                )
                fully_qualified_gpu_utilization_logfile_name = os.path.abspath(
                    os.path.join(log_directory, gpu_utilization_logfile_name)
                )
                gpu_monitoring_thread = threading.Thread(
                    target=monitor_gpu,
                    args=(args.gpu, fully_qualified_gpu_utilization_logfile_name),
                )
                gpu_monitoring_thread.start()
            for j in range(1, args.instances + 1):
                # prepend the full path of the log directory before log file name
                # log file name format: logger_<scheduler>_<run-id>_<instance-id>.log
                logfile_name = "logger_" + scheduler + "_" + str(i) + "_" + str(j) + ".log"
                fully_qualified_log_filename = os.path.abspath(
                    os.path.join(log_directory, logfile_name)
                )
                # make a copy of env before sending to the thread
                env_copy = env.copy()
                env_copy["HOLOSCAN_FLOW_TRACKING_LOG_FILE"] = fully_qualified_log_filename
                instance_thread = threading.Thread(
                    target=run_command, args=(app_launch_command, env_copy)
                )
                instance_thread.start()
                instance_threads.append(instance_thread)
                log_files.append(logfile_name)
            for each_thread in instance_threads:
                each_thread.join()
            if args.monitor_gpu:
                stop_gpu_monitoring_lock.acquire()
                global stop_gpu_monitoring
                stop_gpu_monitoring = True
                stop_gpu_monitoring_lock.release()
                gpu_monitoring_thread.join()
                gpu_utilization_log_files.append(gpu_utilization_logfile_name)
            print(f"Run {i} completed for {scheduler} scheduler.")
            time.sleep(1)  # cool down period

    # Just print comma-separate values of log_files
    print("\nEvaluation completed.")
    print("Log file directory: ", os.path.abspath(log_directory))
    print("All the data flow tracking log files are:", ", ".join(log_files))
    if args.monitor_gpu:
        print("All the GPU utilization log files are:", ", ".join(gpu_utilization_log_files))


if __name__ == "__main__":
    main()
