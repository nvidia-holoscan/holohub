# A Benchmarking Tool for HoloHub

This is a tool to evaluate the performance of HoloHub applications. The tool currently supports
benchmarking of C++ HoloHub applications. We plan to have support for Python applications in the future.

## Pre-requisites
The following Python libraries need to be installed to run the benchmarking scripts (`pip install <library name>` can be used):

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
$ ./run build <application name> <other options> --configure-args \
    -DCMAKE_CXX_FLAGS=-I$PWD/utilities/benchmarking
```

3. Run the performance evaluation

```
$ python utilities/benchmarking/benchmark.py -a <application name> <other options>
```

`python utilities/benchmarking/benchmark.py -h` shows all the possible evaluation options.

All the log filenames are printed out at the end of the evaulation. The format of the flow tracking log filename is:
`logger_<scheduler>_<run_number>_<instance-id>.log`. The format of the GPU utilization log filename
is: `gpu_utilization_<scheduler>_<run_number>.csv`.

4. Get performance results and insights

```
$ python utilities/benchmarking/analyze.py -g <group of log files> <options>
```
`python utilities/benchmarking/analyze.py -h` shows all the possible options.

5. Restore the application

```
$ ./utilities/benchmarking/restore_application.sh <application directory>
```