# Benchmarking Package for HoloHub

This is a tool to evaluate the performance of HoloHub applications. The tool currently supports
benchmarking only a C++ HoloHub application.

## Pre-requisites
The following Python libraries are needed to be installed (`pip install <library name>` can be used):

```
numpy matplotlib nvitop argparse
```
## Steps for Benchmarking

1. Patch the application for benchmarking

```
$ ./utilities/benchmarking/patch_application.sh <application directory>
```

For example, to patch the endoscopy tool tracking application, you would run:

```
$ ./utilities/benchmarking/patch_application.sh applications/endoscopy_tool_tracking
```
This script saves the unmodified `cpp` files in a `*.cpp.bak` file.

2. Build the application

```
./run build <application name> <other options> --configure-args \
    -DCMAKE_CXX_FLAGS=-I$PWD/utilities/benchmarking
```

3. Run the performance evaluation

```
python utilities/benchmarking/benchmark.py -a <application name> <other options>
```

`python utilities/benchmarking/benchmark.py -h` shows all the possible evaluation options.

All the different log file names are printed out at the end of the evaulation. The data flow
tracking log files are located in the `build` directory, and the GPU utilization log files are in the
same directory as the script is running from.

4. Get performance results and insights

```
python utilities/benchmarking/analyze.py <options> -g <group of log files>
```
`python utilities/benchmarking/analyze.py -h` shows all the possible analysis options.

5. Restore the application

```
$ ./utilities/benchmarking/restore_application.sh <application directory>
```