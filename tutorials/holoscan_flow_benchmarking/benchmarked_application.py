import os

from holoscan.conditions import CountCondition
from holoscan.core import Application
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler


class BenchmarkedApplication(Application):
    conditioned_nodes = set()

    def add_flow(self, upstream_op, downstream_op, port_pairs=None):
        if port_pairs:
            super().add_flow(upstream_op, downstream_op, port_pairs)
        else:
            super().add_flow(upstream_op, downstream_op)

        if upstream_op not in self.conditioned_nodes:
            # Load the number of source messages from HOLOSCAN_NUM_SOURCE_MESSAGES
            num_source_messages = int(os.environ.get("HOLOSCAN_NUM_SOURCE_MESSAGES", 100))
            self.conditioned_nodes.add(upstream_op)
            upstream_op.add_arg(CountCondition(self, num_source_messages))

    def run(self):
        print("Running benchmarked application")
        tracker = self.track()

        # Get the data flow tracking log file from environment variable
        flow_tracking_log_file = os.environ.get("HOLOSCAN_FLOW_TRACKING_LOG_FILE", None)
        if flow_tracking_log_file:
            tracker.enable_logging(flow_tracking_log_file)
        else:
            tracker.enable_logging()

        # Load scheduler parameters from environment variables
        scheduler_str = os.environ.get("HOLOSCAN_SCHEDULER", None)
        if scheduler_str and scheduler_str == "multithread":
            num_threads = os.environ.get("HOLOSCAN_MULTITHREAD_WORKER_THREADS", None)
            scheduler = MultiThreadScheduler(
                self,
                name="multithread-scheduler",
                worker_thread_number=int(num_threads) if num_threads is not None else 1,
                stop_on_deadlock=True,
                check_recession_period_ms=0,
                max_duration_ms=100000,
            )
        else:
            scheduler = GreedyScheduler(self, name="greedy-scheduler")
        self.scheduler(scheduler)

        # Call the parent class' run()
        super().run()
