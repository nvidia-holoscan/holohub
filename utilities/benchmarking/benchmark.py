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

from nvitop import Device
import os, sys, subprocess, time
import threading
import argparse

devices = Device.all()

# if CUDA_VISIBLE_DEVICES is set to A4000
# a4000 = devices[0]
# a6000 = devices[1]
# devices = [a6000]

stop = False

def monitor_gpu():
    print ("Monitoring GPU utilization")
    # run every 1 seconds
    global stop
    stop = False
    average_gpu_utilizations = []
    while not stop:
        time.sleep(2)
        average_gpu_utilizations.append(sum(device.gpu_utilization() for device in devices) / len(devices))
        # write average gpu utilization to a file and a new line
    with open("gpu_utilization.txt", "a") as f:
        # discard first 2 and last 2 values
        average_text = ",".join(map(str, average_gpu_utilizations[2:-2]))
        f.write(str(average_text) + "\n")
        print ("Written GPU utilization to a file")
    stop = False

def run_command(app_launch_command, env):
    try:
        result = subprocess.run([app_launch_command], shell=True, env=env, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"stdout:\n{e.stdout.decode('utf-8')}")
        print(f"stderr:\n{e.stderr.decode('utf-8')}")
        sys.exit(1)
    if result.returncode != 0:
        # The command returned an error
        print(f"Error: Command exited with code {result.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run performance evaluation for an (HoloHub) application')

    requiredArgument = parser.add_argument_group('required arguments')
    requiredArgument.add_argument("--sched", nargs='+', choices=["greedy", "multithread"], required=True, help="scheduler(s) to use")

    parser.add_argument("-a", "--holohub_application", type=str, required=False, help="name of HoloHub application to run", default="endoscopy_tool_tracking")

    parser.add_argument("--not_holohub", action="store_true", help="enable this to indicate a non-HoloHub application")

    parser.add_argument("-p", "--binary_path", type=str, required=False, help="command to run the application if it is not a HoloHub application")

    parser.add_argument("-g", "--gpu", type=str, required=False, help="the GPU UUIDs to run the application on (default: all)", default="all")

    parser.add_argument("-i", "--instances", type=int, required=False, help="number of instances to run in parallel (default: 1)", default=1)

    parser.add_argument("-r", "--runs", type=int, default=1, help="number of times to run the application (default: 1)", required=False)

    parser.add_argument("-m", "--num_source_messages", type=int, default=100, help="number of source messages to send (default: 100)", required=False)

    parser.add_argument("-u", "--monitor_gpu", action="store_true", help="enable this to monitor GPU utilization")

    parser.add_argument("-w", "--num_worker_threads", type=int, default=1, help="number of worker threads for multithread scheduler (default: 1)", required=False)

    args = parser.parse_args()

    if args.not_holohub or args.binary_path is not None:
        print ("Currently non-HoloHub applications are not supported")
        sys.exit(1)

    if args.monitor_gpu:
        print ("Currently, collecting GPU utilization data is not supported")
        sys.exit(1)

    env = os.environ.copy()
    if args.gpu != "all":
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.num_source_messages != 100:
        env["HOLOSCAN_NUM_SOURCE_MESSAGES"] = str(args.num_source_messages)

    app_launch_command = "./run launch " + args.holohub_application + " cpp"

    log_files = []
    for scheduler in args.sched:
        if scheduler == "multithread":
            env["HOLOSCAN_SCHEDULER"] = scheduler
            if args.num_worker_threads != 1:
                env["HOLOSCAN_MULTITHREAD_WORKER_THREADS"] = args.num_worker_threads
        elif scheduler != "greedy":
            print ("Unsupported scheduler ", scheduler)
            sys.exit(1)
        # No need to set the scheduler for greedy scheduler

        for i in range(1, args.runs + 1):
            instance_threads = []
            for j in range(1, args.instances + 1):
                # log file name format: logger_<scheduler>_<run-id>_<instance-id>.log
                logfile_name = "logger_" + scheduler + "_" + str(i) + "_" + str(j) + ".log"
                env["HOLOSCAN_FLOW_TRACKING_LOG_FILE"] = logfile_name
                instance_thread = threading.Thread(target=run_command, args=(app_launch_command, env))
                instance_thread.start()
                instance_threads.append(instance_thread)
                log_files.append(logfile_name)
            for each_thread in instance_threads:
                each_thread.join()
            print (f"Run {i} completed for {scheduler} scheduler")
            time.sleep(1) # cool down period

    # Just print comma-separate values of log_files
    print ("Evaluation completed. All the log files are: ", ", ".join(log_files))
    # global stop
    # need to integrate with the collection of GPU utilization data

if __name__ == "__main__":
    main()
