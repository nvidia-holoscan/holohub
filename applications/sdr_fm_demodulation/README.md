# Software Defined Radio FM Demodulation

As the "Hello World" application of software defined radio developers, this demonstration highlights real-time FM demodulation, resampling, and playback on GPU with NVIDIA's Holoscan SDK. In this example, we are using an inexpensive USB-based [RTL-SDR](https://www.rtl-sdr.com/) dongle to feed complex valued Radio Frequency (RF) samples into GPU memory and use [cuSignal](https://github.com/rapidsai/cusignal) functions to perform the relevant signal processing. The main objectives of this demonstration are to:
- Highlight developer productivity in building an end-to-end streaming application with Holoscan and existing GPU-Accelerated Python libraries
- Demonstrate how to construct and connect isolated units of work via Holoscan operators
- Emphasize that operators created for this application can be re-used in other ones doing similar tasks

# Running the Application

Prior to running the application, the user needs to install the necessary dependencies (and, of course, plug in a SDR into your computer). This is most easily done in an Anaconda environment.

```
conda create --name holoscan-sdr-demo python=3.8
conda activate holoscan-sdr-demo
conda install -c conda-forge -c rapidsai -c nvidia cusignal soapysdr soapysdr-module-rtlsdr pyaudio
pip install holoscan
```

The FM demodulation example can then be run via
```
python applications/sdr_fm_demodulation/sdr_fm_demodulation.py
```
