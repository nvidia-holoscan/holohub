# Spectrum Magnitude Operator

## Overview

The operator that converts complex FFT output into a power spectrum in dB.

## Role In The Pipeline

`SpectrumMagnitudeOp` receives a batch of complex FFT results with shape `(num_bursts, burst_size)`. For each frequency bin, it:

1. computes the squared magnitude,
2. averages across all bursts in the batch,
3. converts the result to dB using `10 * log10(...)`.

The output is a single 1-D float tensor per channel, ready for display.

The values are **power in dBFS** (0 dB = a full-scale complex tone in a single bin). Because the input samples are normalized to +/-1.0 and the FFT is normalized by `1 / burst_size`, the per-bin value is `10 * log10(|X / N|^2)`, so the spectrum ranges from 0 dBFS at the top down to negative values (the noise floor). This matches the `db_min` / `db_max`
display band configured for the visualizer.

## Key Files

- `spectrum_magnitude.hpp`: operator declaration and buffer members.
- `spectrum_magnitude.cu`: magnitude-squared, averaging, and dB conversion logic.

## Inputs And Outputs

- Input: `std::tuple<tensor_t<complex, 2>, cudaStream_t>`
- Output: `std::tuple<tensor_t<float, 1>, cudaStream_t>`

## Notes

- The operator preserves the input CUDA stream so downstream visualization can stay synchronized without forcing a host-side wait.
