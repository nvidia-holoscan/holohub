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
import atexit
import logging
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict

from nvitop import Device

# Ensure utility module is on import path
sys.path.append(os.path.dirname(__file__))
from patch_python_sources import patch_directory, restore_backups

# Add holohub root to path for importing utilities
script_dir = os.path.dirname(os.path.abspath(__file__))
holohub_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
if holohub_root not in sys.path:
    sys.path.insert(0, holohub_root)

from utilities.cli.holohub import HoloHubCLI  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.DEBUG)

# global variables for GPU monitoring
stop_gpu_monitoring = False
stop_gpu_monitoring_lock = threading.Lock()


def monitor_gpu(gpu_uuids, filename):
    logger.info("Monitoring GPU utilization in a separate thread")
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
        logger.error(f"Error running command: {e}")
        logger.error(f"stdout:\n{e.stdout.decode('utf-8')}")
        logger.error(f"stderr:\n{e.stderr.decode('utf-8')}")
        sys.exit(1)

    logger.debug(f"stdout:\n{result.stdout}")
    logger.debug(f"stderr:\n{result.stderr}")
    if result.returncode != 0:
        # The command returned an error
        logger.error(f'Command "{app_launch_command}" exited with code {result.returncode}')
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
        choices=["greedy", "multithread", "eventbased"],
        required=True,
        help="scheduler(s) to use",
    )

    parser.add_argument(
        "-a",
        "--holohub-application",
        type=str,
        required=False,
        help="name of HoloHub application to run",
        default="endoscopy_tool_tracking",
    )

    parser.add_argument(
        "--language",
        type=str,
        required=False,
        help="Application language to run. Runs cpp version of an application by default. "
        "Must also specify an application to run with an argument to '-a' or '--holohub-application'.",
        default="cpp",
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
        help="number of worker threads for multithread or eventbased scheduler (default: 1)",
        required=False,
    )

    parser.add_argument(
        "-u", "--monitor_gpu", action="store_true", help="enable this to monitor GPU utilization"
    )
    parser.add_argument("--level", type=str, default="INFO", help="Logging verbosity level")

    args = parser.parse_args()

    # Determine accurate application source directory using HoloHub metadata
    cli = HoloHubCLI()
    try:
        proj = cli._find_project(args.holohub_application, language=args.language)
        app_root = proj.get("source_folder", "")
    except Exception:
        app_root = os.path.abspath(
            os.path.join(holohub_root, "applications", args.holohub_application)
        )

    backups_list = []
    if app_root and os.path.isdir(app_root):
        backups_list = patch_directory(app_root, script_dir)

    atexit.register(lambda: restore_backups(backups_list))

    if (
        "multithread" not in args.sched
        and "eventbased" not in args.sched
        and args.num_worker_threads != 1
    ):
        logger.warning(
            "num_worker_threads is ignored as multithread or eventbased scheduler is not used"
        )

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
            logger.info(
                f"Log directory is not found. Creating a new directory at {os.path.abspath(log_directory)}",
            )
            os.mkdir(os.path.abspath(log_directory))

    # Set up detailed logging to file + custom verbosity logging to console
    fileHandler = logging.FileHandler(os.path.join(log_directory, "benchmark.log"))
    fileHandler.setLevel(level=logging.DEBUG)
    fileHandler.setFormatter(
        logging.Formatter("%(asctime)s %(threadName)s %(levelname)s: %(message)s")
    )
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(stream=sys.stdout)
    consoleHandler.setLevel(level=args.level.upper())
    consoleHandler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.root.removeHandler(logger.root.handlers[0])
    logger.addHandler(consoleHandler)

    env = os.environ.copy()
    if args.gpu != "all":
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.num_messages != 100:
        env["HOLOSCAN_NUM_SOURCE_MESSAGES"] = str(args.num_messages)

    if args.run_command == "":
        app_launch_command = "./run launch " + args.holohub_application + " " + args.language
    else:
        app_launch_command = args.run_command

    log_files = []
    gpu_utilization_log_files = []
    for scheduler in args.sched:
        if scheduler == "multithread":
            env["HOLOSCAN_SCHEDULER"] = scheduler
            env["HOLOSCAN_MULTITHREAD_WORKER_THREADS"] = str(args.num_worker_threads)
        elif scheduler == "eventbased":
            env["HOLOSCAN_SCHEDULER"] = scheduler
            env["HOLOSCAN_EVENTBASED_WORKER_THREADS"] = str(args.num_worker_threads)
        elif scheduler != "greedy":
            logger.error("Unsupported scheduler ", scheduler)
            sys.exit(1)
        # No need to set the scheduler for greedy scheduler
        for i in range(1, args.runs + 1):
            logger.info(f"Run {i} started for {scheduler} scheduler.")
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
            logger.info(f"Run {i} completed for {scheduler} scheduler.")
            time.sleep(1)  # cool down period
    logger.info("****************************************************************")
    logger.info("Evaluation completed.")
    logger.info("****************************************************************")

    logger.info(f"Log file directory: {os.path.abspath(log_directory)}")
    log_info = defaultdict(lambda: defaultdict(list))
    log_file_sets = [("log", log_files)] + (
        [("gpu", gpu_utilization_log_files)] if args.monitor_gpu else []
    )
    for log_type, log_list in log_file_sets:
        abs_filepaths = [
            os.path.abspath(os.path.join(log_directory, log_file)) for log_file in log_list
        ]
        log_info[log_type]["found"] = [file for file in abs_filepaths if os.path.exists(file)]
        log_info[log_type]["missing"] = [file for file in abs_filepaths if not os.path.exists(file)]
        logger.info(
            f'{log_type.capitalize()} files are available: {", ".join(log_info[log_type]["found"])}'
        )
        if log_info[log_type]["missing"]:
            logger.error(
                f'{log_type.capitalize()} files are missing: {", ".join(log_info[log_type]["missing"])}'
            )

    if any(log_info[log_type]["missing"] for log_type in ("log", "gpu")):
        logger.error("Some log files are missing. Please check the log directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
