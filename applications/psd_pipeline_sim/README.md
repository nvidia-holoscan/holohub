<!--
SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.

SPDX-License-Identifier: Apache-2.0
-->
# PSD Pipeline with File Read

## Overview
The PSD pipeline simulator takes in a data stream from the file reader
operator, then performs an FFT, PSD, and averaging operation before
generating a VITA 49.2 spectral data packet which gets sent to a
destination UDP socket.

```
┌───────────┐    ┌─────┐    ┌─────┐    ┌─────────────┐    ┌──────────┐ VRT
│ File Read ├───►│ FFT ├───►│ PSD ├───►│PSD Averaging├───►│Packetizer├─────►
├───────────┤    └─────┘    └─────┘    └─────────────┘    └──────────┘
│Raw IQ Data│
└───────────┘
```

## Requirements

- [MatX](https://github.com/NVIDIA/MatX)
- [vrtgen](https://github.com/Geontech/vrtgen)

## Configuration

Each operator in the pipeline has its own configuration section. The specific options
and their meaning are defined in each operator's own README:

1. [`file_reader`](../../operators/file_reader/README.md)
2. [`fft`](../../operators/fft/README.md)
3. [`high_rate_psd`](../../operators/high_rate_psd/README.md)
4. [`low_rate_psd`](../../operators/low_rate_psd/README.md)
5. [`vita49_psd_packetizer`](../../operators/vita49_psd_packetizer/README.md)

The only config option specific to this application is the `num_psds` param
which defines how many PSDs to produce out of the pipeline before exiting.
Passing `-1` here will cause the pipeline to run indefinitely.

### Metadata

This pipeline leverages Holoscan's operator metadata dictionaries to pass
VITA 49-adjacent metadata through the pipeline.

Each operator in the pipeline adds its own metadata to the dictionary.
At the end of the pipeline, the packetizer operator uses the metadata
to construct VITA 49 context packets to send alongside the spectral data.

## Build & Run
1. **Launch** the development container with the command:
   ```bash
   ./dev_container launch
   ```

Once you are in the dev container:
1. **Build** the application using:
    ```bash
    ./run build psd_pipeline_sim
    ```
2. **Run** the application using:
    ```bash
     ./run launch psd_pipeline_sim --extra_args config.yaml
     ```

Note: the file reader will attempt to load the IQ data
file (`file_reader.file_name`) from its working directory
(`build/psd_pipeline_sim`), so you'll need to place the data file
there. If you just want to test the app, you can generate some random
data:

```bash
# Generate a 32MB random data file
dd if=/dev/urandom of=build/psd_pipeline_sim/<your_file_name> bs=32000 count=1000
```
