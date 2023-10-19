# CUDA MPS Tutorial for Holoscan Applications

CUDA MPS is NVIDIA's [Multi-Process Service](https://docs.nvidia.com/deploy/mps/index.html) for CUDA
applications. It allows multiple CUDA applications to share a single GPU, which can be useful for
running more than one Holoscan application on a single machine featuring one or more GPUs. This
tutorial describes the steps to enable CUDA MPS and demonstrate few performance benefits of using it.

## Steps to enable CUDA MPS

Before enabling CUDA MPS, please [check](https://docs.nvidia.com/deploy/mps/index.html#topic_3_3)
whether your system supports CUDA MPS.

CUDA MPS can be enabled by running the `nvidia-cuda-mps-control -d` command and stopped by running 
`echo quit | nvidia-cuda-mps-control` command. More control commands are described 
[here](https://docs.nvidia.com/deploy/mps/index.html#topic_5_1). 

CUDA MPS does not require any changes to
an existing Holoscan application; even an already compiled application binary works as it is.
Therefore, a Holoscan application can work with CUDA MPS without any 
changes to its source code or binary.
However, a machine learning model like a TRT engine file may need to be recompiled 
for the first time after enabling CUDA MPS.

We have included a helper script in this tutorial `start_mps_daemon.sh` to enable 
CUDA MPS with necessary environment variables.

```bash
./start_mps_daemon.sh
```

## Customization

CUDA MPS provides many options to customize resource allocation for MPS clients. For example, it has
an option to limit the maximum number of GPU threads that can 
be used by every MPS client. 
The `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` environment variable can be used to control this limit
system-wide. This limit can also be set by communicating the active thread percentage to the control daemon with  
`echo "set_default_active_thread_percentage <Thread Percentage>" | nvidia-cuda-mps-control`.
Our `start_mps_daemon.sh` script also takes this percentage as the first argument.

```bash
./start_mps_daemon.sh <Active Thread Percentage>
```

For different applications, one may want to set different limits on the number of GPU threads
available to each of them. This can be done by setting the `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`
environment variable separately for each application. It is elaborated in details [here](https://docs.nvidia.com/deploy/mps/index.html#topic_5_2_5).

There are other customizations available in CUDA MPS as well. Please refer to the CUDA MPS
[documentation](https://docs.nvidia.com/deploy/mps/index.html#topic_5_1_1) to know more about them.

## Performance Benefits

CUDA MPS improves the performance for concurrently running Holoscan applications. 
Since multiple applications can simultaneously execute more than one CUDA compute tasks with CUDA
MPS, it can also improve the overall GPU utilization.

Suppose, we want to run the endoscopy tool tracking and ultrasound segmentation applications
concurrently on an x86 workstation with RTX A6000 GPU. The below table shows the maximum end-to-end latency performance
without and with CUDA MPS, where the active thread percentage is set to 40\% for each application.
It demonstrates 18% and 50% improvement in the maximum end-to-end latency for the
endoscopy tool tracking and ultrasound segmentation applications, respectively.

| Application | Without MPS (ms) | With MPS (ms) |
| ----------- | ---------------- | ------------- |
| Endoscopy Tool Tracking | 115.38 | 94.20 |
| Ultrasound Segmentation | 121.48 | 60.94 |

In another set of experiments, we concurrently run multiple instances of the endoscopy tool tracking
application in different processes. We set the active thread percentage to be 20\% for each MPS client. The below graph shows the maximum end-to-end latency with and
without CUDA MPS. The experiment demonstrates upto 36% improvement with CUDA MPS.

![Alt text](image.png)

Such experiments can easily be conducted with [Holoscan Flow Benchmarking](../holoscan_flow_benchmarking) to retrieve
various end-to-end latency performance metrics.