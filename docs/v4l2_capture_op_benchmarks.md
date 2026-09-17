# V4L2 Capture Benchmarks and Qualification Results

## Scope and setup

These results record capture qualification for Holoscan Camera 0.2.0 (tested tag
`v0.2.0.0`) with Holoscan SDK `v5.0.0.3`, as reported in issue 6744108.
They cover frame counts, quality flags, and timestamp availability. Throughput
and latency measurements were not recorded in the report.

The camera model attached to each platform was not recorded. Differences may
therefore be device-specific; their cause has not been established. Repeat runs
with the same camera model before attributing a difference to the platform.

Run from the repository root after completing [SDK setup](../README.md#holoscan-sdk-50-early-access-setup).
Check that the device supports YUYV, then capture 300 frames with an explicit
window (the example below uses the longer window that completed on x86 + Blackwell):

```bash
v4l2-ctl --list-formats-ext -d /dev/video0
./holoscan_camera run holoscan_camera_v4l2 capture --language cpp \
    --docker-opts '--device /dev/video0' \
    --run-args '--device /dev/video0 --frames 300 --timeout 30'
```

See the [operator guide](../operators/v4l2_capture_op/README.md) for device and
memory-placement options, and [capture counters](../operators/v4l2_capture_op/README.md#capture-counters)
for the meaning of each summary field.

## Recorded results

| Platform | Timeout (s) | `received` | `degraded` | `corrupt` | `unnamed-clock` |
| --- | --- | --- | --- | --- | --- |
| IGX Thor iGPU | 20 | 300 | 0 | 0 | 0 |
| AGX Thor | 20 | 300 | 0 | 0 | 0 |
| Concord | 20 | 300 | 0 | 0 | 0 |
| IGX Thor Blackwell | 20 | 300 | non-zero | 0 | 0 |
| DGX Spark | 20 | 300 | 0 | 1 | 0 |
| x86 + Blackwell | 20 | 281 | 0 | 1 | 0 |
| x86 + Blackwell | 30 | 300 | 0 | 1 | 0 |

The report records `with-capture-time=281` and `with-capture-time=300` for the
two x86 + Blackwell runs, respectively. It does not include that counter for
the other platforms. The exact IGX Thor Blackwell `degraded` count was not captured.

## Interpretation and follow-up

### Degraded frames on IGX Thor Blackwell

All 300 requested frames arrived, but some carried `SampleFlags::kDegraded`.
The operator sets this flag for a forward sequence gap, a sequence regression,
or a frame shorter than its declared byte extent. The recorded result does not
distinguish these causes or establish whether degradation was limited to startup.

Only sequence regression emits a throttled warning:

```text
v4l2 capture: device sequence went from <n> to <m>, so samples across the discontinuity
cannot be counted, <k> in a row
```

Forward gaps and short frames set the flag without a warning. Capture the exact
count and distinguish these paths in a follow-up run.

### Driver-flagged frames on DGX Spark and x86 + Blackwell

Each recorded run reported `corrupt=1`. This counter reflects
`V4L2_BUF_FLAG_ERROR` from the driver, which the operator publishes as
`SampleFlags::kInvalid`. Consumers must reject invalid samples.

The affected frame's position was not recorded, so a startup-only explanation
remains unconfirmed. Record its position and repeat the runs to establish whether
the observation is reproducible.

### Capture window on x86 + Blackwell

Increasing `--timeout` from 20 to 30 seconds raised `received` from 281 to 300
while the other recorded counters stayed unchanged. No sequence discontinuity
was reported in those runs (`degraded=0`); these counters alone do not establish
whether startup delay or effective capture rate accounts for the longer window.

At the application's default 30 fps, 300 frames require approximately ten seconds
of streaming, before allowing for startup and scheduling delays. The default
`--timeout 10` leaves little headroom. Choose an explicit window for the requested
frame count; 30 seconds completed the recorded run but is not a guarantee for
other devices. Check the frame count as well as the process exit status: the
current reference application reports an error and returns 2 when fewer than the
requested frames arrive.

### Capture timestamps

`unnamed-clock=0` on every reported platform means no unknown timestamp clock was
reported. On x86 + Blackwell, `with-capture-time` also equaled `received` in both
runs. Record both counters on the other platforms before claiming full timestamp
coverage. These results do not measure timestamp accuracy or end-to-end latency.

See [release notes](../RELEASE_NOTES.md) for the concise known-issue summary.
