#  Power Spectral Density with cuNumeric

[cuNumeric](https://github.com/nv-legate/cunumeric) is an drop-in replacement for NumPy that aims to provide a distributed and accelerated drop-in replacement for the NumPy API on top of the [Legion](https://legion.stanford.edu/) runtime. It works best for programs that have very large arrays of data that can't fit in the the memory of a single GPU or node.

In this example application, we are using the cuNumeric library within a Holoscan application graph to determine the Power Spectral Density (PSD) of an incoming signal waveform. Notably, this is simply achieved by taking the absolute value of the FFT of a data array.

 The main objectives of this demonstration are to:
- Highlight developer productivity in building an end-to-end streaming application with Holoscan and cuNumeric
- Demonstrate how to scale a given workload to multiple GPUs using cuNumeric

# Running the Application

Prior to running the application, the user needs to install the necessary dependencies. This is most easily done in an Anaconda environment.

```
conda create --name holoscan-cunumeric-demo python=3.9
conda activate holoscan-cunumeric-demo
conda install -c nvidia -c conda-forge -c legate cunumeric cupy
pip install holoscan
```

The cuNumeric PSD processing pipeline example can then be run via
```
legate --gpus 2 applications/cunumeric_integration/cunumeric_psd.py
```

While running the application, you can confirm multi GPU utilization via watching `nvidia-smi` or using another GPU utilization tool

To run the same application without cuNumeric, simply change `import cunumeric as np` to `import cupy as np` in the code and run
```
python applications/cunumeric_integration/cunumeric_psd.py
```
