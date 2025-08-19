# An Example of Async Lock-free Buffer with SCHED_DEADLINE

A simple application to measure the impact of async lock-free buffer communication between operators with earliest deadline first (`SCHED_DEADLINE`) scheduling policy of Linux.

## Overview

This application demonstrates how different kinds of buffer connectors can
impact the performance in terms of message latency. It uses Linux's
`SCHED_DEADLINE` scheduler to ensure predictable timing for operators.

The application consists of:
- Two transmitter operators (PingTxOp) that generate ping messages at different rates
- One receiver operator (PingRxOp) that processes messages from both transmitters
- Optional async buffer connectors between transmitters and receiver
- Earliest deadline first scheduling using Linux's `SCHED_DEADLINE` policy

The application measures and logs:
- Message latency from transmission to reception
- Observed periods between messages
- Performance impact of async buffer usage



## Requirements

This application requires:
1. Linux with `SCHED_DEADLINE` support
2. Root privileges in the container because of `SCHED_DEADLINE`
3. Holoscan SDK 3.5.0 or later

## Build and Run Instructions

**Note**: Please make sure the following command is run before running the
application:

```
sudo sysctl -w kernel.sched_rt_runtime_us=-1
```

To build and run the application:

```bash
# Basic run with default settings (100 messages, no async buffer)
./holohub run async_buffer_deadline

# Run with async buffer enabled
./holohub run async_buffer_deadline -- --async-buffer

# Run with custom message count and periods
./holohub run async_buffer_deadline -- --messages 50 --tx1-period 15 --tx2-period 25 --async-buffer
```

### Command Line Options

- `-h, --help`: Display help information
- `-m <COUNT>, --messages <COUNT>`: Number of messages to send (default: 100)
- `-a, --async-buffer`: Enable async buffer connector
- `-x <MS>, --tx1-period <MS>`: Set TX1 period in milliseconds (default: 20, min: 10)
- `-y <MS>, --tx2-period <MS>`: Set TX2 period in milliseconds (default: 20, min: 15)

## Output

The application generates several CSV files:
- `tx1.csv`: Latency measurements for TX1 messages
- `tx2.csv`: Latency measurements for TX2 messages
- `rx_in1_periods.csv`: Observed message intervals for RX input 1
- `rx_in2_periods.csv`: Observed message intervals for RX input 2

