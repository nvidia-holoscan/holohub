# Simple Radar Pipeline

This demonstration walks the developer through building a simple radar signal processing pipeline, targeted towards detecting objects, with Holoscan. In this example, we generate random radar and waveform data, passing both through:
1. Pulse Compression
2. Moving Target Indication (MTI) Filtering
3. Range-Doppler Map
4. Constant False Alarm Rate (CFAR) Analysis

While this example generates 'offline' complex-valued data, it could be extended to accept streaming data from a phased array system or simulation via modification of the `SignalGeneratorOperator`.

The output of this demonstration is a measure of the number of pulses per second processed on GPU.

 The main objectives of this demonstration are to:
- Highlight developer productivity in building an end-to-end streaming application with Holoscan and existing GPU-Accelerated Python libraries
- Demonstrate how to construct and connect isolated units of work via Holoscan operators, particularly with handling multiple inputs and outputs into an Operator
- Emphasize that operators created for this application can be re-used in other ones doing similar tasks

# Running the Application

Prior to running the application, the user needs to install the necessary dependencies. This is most easily done in an Anaconda environment.

```
conda create --name holoscan-sdr-demo python=3.8
conda activate holoscan-sdr-demo
conda install -c conda-forge -c rapidsai -c nvidia cusignal
pip install holoscan
```

The simple radar signal processing pipeline example can then be run via
```
python applications/simple_radar_pipeline/simple_radar_pipeline.py
```
