# Performance Evaluation with Holoscan SDK

This guide explains how to evaluate the latency performance of ultrasound post-processing presets using the Holoscan SDK's built-in profiling tools.

## Overview

The `ultra_post` application uses the Holoscan SDK's `Tracker` to log data flow execution times. This allows us to:
1.  Measure end-to-end latency (Source → Display).
2.  Identify bottlenecks (which operator takes the most time).
3.  Compare performance between different presets (e.g., `denoise.yml` vs `preset.yml`).

## generating Performance Logs

The Holoscan app (`ultra_post.app.holoscan_app`) is instrumented with a `Tracker` that records execution metrics by default.

To generate a log file:

1.  **Run the Holoscan App** with your desired preset and source.
    
    ```bash
    # Example: Run with the denoise preset and a UFF file
    uv run python -m ultra_post.app.holoscan_app \
      --preset presets/denoise.yml \
      --source uff \
      --uff ultra_post/examples/demo.uff \
      --log performance.log
    ```

    - `--log performance.log`: Specifies the output log file (default is `us_post_processing.log`).
    - `--headless`: Use this flag if you are running on a server without a display to avoid rendering overhead (though `HolovizOp` will still run in headless mode).

2.  **Let it run** for a few seconds to gather enough frames (data points). Use `Ctrl+C` to stop it.

## Analyzing the Logs

The log file contains timestamped events for every message sent and received between operators. To analyze this data, we provide a helper script.

### Using the Analysis Tool

We have provided a script in `tools/analyze_holoscan_log.py` to parse the log and generate a summary.

```bash
# Run the analysis script on the generated log
uv run python tools/analyze_holoscan_log.py performance.log
```

### metrics Explained

The analysis tool reports:

-   **End-to-End Latency**: The time difference between the Source emitting a frame and the Display receiving it.
-   **Operator Latency**: The time spent inside each operator (processing time).
-   **FPS**: Effective frames per second based on sink reception rate.

### Identifying Bottlenecks

1.  Look at the **Operator Latency** table.
2.  Identify operators with the highest **Avg Latency**.
3.  If an operator's latency is higher than the frame budget (e.g., >33ms for 30 FPS), it is a bottleneck.

## optimizing Presets

If a preset is too slow:
1.  **Disable heavy operators**: Edit the YAML to set `enable: false` for expensive ops (e.g., `non_local_means`).
2.  **Adjust parameters**: Reduce `patch_size` or search window sizes in the YAML preset.
3.  **Re-run the profile**: Generate a new log and compare.

