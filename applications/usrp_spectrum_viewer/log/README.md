# Logging Utilities

## Overview

This folder contains the optional logging operator used for visibility into FFT throughput and sample contents.

## Role In The Pipeline

`LogOp` branches from the FFT stage and observes the transformed tensors without changing the main visualization path. It can:

- estimate throughput in MSps and Gbps,
- report tensor shape and channel metadata,
- optionally copy sample data to the host for debugging output.

## Key Files

- `log.hpp`: operator declaration and bookkeeping members.
- `log.cu`: throughput calculation and optional host-side sample dump.

## Notes

- This operator is for debugging and observability only.
- Disabling/Removing it does not change the main signal-processing result.
