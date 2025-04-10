# Artifact 

## Platform Details

**Submitted Paper Evaluation Platform.** 
For the paper, we ran all experiments using an early version of 
NVIDIA Holoscan, [v0.6.0](https://github.com/nvidia-holoscan/holoscan-sdk/releases/tag/v0.6.0), built from source. The platform was an
NVIDIA [Jetson AGX Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) Developer Kit with a 12-core Arm 
Cortex-A78AE CPU, running Ubuntu-based Linux for Tegra (L4T) OS with kernel 5.15.

**Artifact Test Platform.** We have also reproduced our experiments with a more recent version of NVIDIA Holoscan, 
[v2.2.0](https://github.com/nvidia-holoscan/holoscan-sdk/releases/tag/v2.2.0), using a modified version of an open-source Docker container published by NVIDIA. 
We have tested this container on an Ubuntu 22.04.4 Linux server with a 64 core Intel Xeon Gold 6326 CPU.

**Recommendations.** To run the container, we recommend using a machine that has at least 12 cores that can be dedicated to the experiments, as that is the number of threads used in our applications. 
On a system with fewer cores, the main experimental results may 
no longer match what is presented in the paper, as the available cores may need to be split across all threads instead of running in parallel.
The experiments relying on simulation do not require multiple cores to work as intended.

Docker can run on Linux, Windows, or macOS, but we have only tested our container on Linux.

## Container Setup

The only prerequisite for running the container is an installation of [Docker](https://docs.docker.com/engine/install/). 
Depending on permissions, you may need to run all Docker commands with `sudo`.

The image can then be pulled using the following command: 

    docker pull pnschowitz/rtss2024_artifact_eval:CR

The process of downloading and extracting may take a while. After pulling the image, you can confirm with `docker images`, which should
show information similar to below:

    REPOSITORY                               TAG              IMAGE ID       CREATED        SIZE
    pnschowitz/rtss2024_artifact_eval        CR               46309e98ce88   46 hours ago   12.3GB

Before running the container, we create a directory to hold output we want to save from inside the container.

    mkdir saved_artifacts

Then we run the Docker image with the following command: 

    docker run -it --rm \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ${PWD}/saved_artifacts:/opt/nvidia/holoscan/artifact-eval/saved_artifacts \
    $"pnschowitz/rtss2024_artifact_eval:CR"

The meanings of the various options are as follows:

- `-it`
    - Allows for interaction with the container through a terminal interface.
- `-rm`
    - Automatically removes the container when exiting (the image itself will not be deleted).
- `--ulimit memlock=-1`
    - Removes the limit on locking in memory (Recommended by NVIDIA).
-  `--ulimit stack=67108864`
    - Sets the stack size limit to 64 MB (Recommended by NVIDIA).
- `-v /var/run/docker.sock:/var/run/docker.sock` 
    - Allows for Docker commands executed inside the container to communicate with the Docker daemon on the host.
- ` -v ${PWD}/saved_artifacts:/opt/nvidia/holoscan/artifact-eval/saved_artifacts` 
    - Allows for any files you move to the `saved_artifacts` folder within the container to be automatically transferred to the `saved_artifacts` folder on the local system.
- `$"pnschowitz/rtss2024_artifact_eval:CR"`
    - The actual image to run.

The only directory we modified is `artifact-eval`, which contains the code to reproduce our experimental results.

## Running experiments

### master.py

The script `master.py` reproduces our experimental results using the files in the `code` folder. The experiments fall into three categories, described here. More information on all output can be found in the `Output` section.

The main experiments compare the theoretical WCRT for eight HoloHub applications (listed in the paper) to the WCRT from both simulating their execution and running a dummy version of them (which busy waits instead of using the real operators). Creates `evalpessimism.pdf` and `evalsim.pdf`.  

The overhead experiment compares a theoretical WCRT that we know to be ideal to the actual response times in the system to estimate overheads. Creates `evaloverhead.pdf`.

The scalability experiments compare the theoretical WCRT for large graphs that we randomly generated  to the WCRT from simulating their execution. We also evaluate how long the theoretical analysis takes to run. Creates `evalscalability.pdf`, `evalscalabilitypess.pdf`, and `evalanalysis.pdf`.

`master.py` has the following arguments:

- `--iterations`
    - An integer number of iterations, determining how many inputs each graph will process. Default value is 10.
- `--numvars`
    - The number of variations to use for each graph. Default value is 10.
- `--build` or `-b`
    - Option to build executables.
- `--main-experiments` or `-m`
    - Option to run main experiments. 
- `--overhead` or `-o`
    - Option to run evaluation of system overheads.
- `--scalability` or `-s`
    - Option to run scalability experiments.

The experiments cover a user-defined number of variations, as determined by `numvars`. To exactly match the results in the paper,
`numvars` can be set to 0. In this case, each graph will have a number of variations equal to its node count.
In this case, the experiments cover a total of 51 variations, each corresponding to one of 8 graphs (see figure 9 in the paper).
The total number of iterations run over the course of the main experiments will thus be `51 * iterations`. Otherwise, 
the total number of iterations will be `numvars * iterations`.

As a result, the runtime of the experiments is almost entirely dependent on the values of these two arguments. With both `numvars` and `iterations` at 10,
execution of the main experiments will take around 20 minutes. This scales approximately linearly, so 100 will yield an execution time of around 3 hours,
and 1000 around 30 hours. Our experiments in the paper roughly correspond to an `iterations` value of 1000 with numvars at 0, but as few as 10 iteration
should still yield valid results for the majority of randomly generated graph variations.

Scalabilty experiments run a timing analysis of our algorithm averaging over multiple runs, so they may take around 15 minutes to complete with default settings.

#### Run all experiments

Before running the main or overhead experiments, it is necessary to first build the executables with the `--build` option. 
This uses the `--iterations` argument, so the executables will need to be rebuilt if a different `--iterations` value is desired. This is necessary because Holoscan requires some application properties to be hardcoded at build time.

    cd artifact-eval
    python master.py --iterations 10 -b
    python master.py --numvars 10 -m -o -s

While running the experiments, outputs will be periodically printed to the terminal. The completion of an iteration prints the following output:

    Running... iteration 1 complete

Furthermore, the completion of a graph's execution will print the following output:

    Graph 1 variation 0 complete

The simulations done in the main and scalability experiments do not print anything to the terminal, but these only take around a minute to complete.

When the experiments finish, the `data` folder will be populated with all the resulting raw data and figures, described in the Output section below.

### Viewing experiment results and cleanup

To transfer files from the container to the local system, copy the files to the `saved_artifacts` folder. For example:

    cp data/*.pdf saved_artifacts

This will make these files available even after the container is gone. When finished with the container, use `exit` to shut it down. 

## Output

After running `master.py` with the `-m`, `-o`, and `-s` options, the `data` folder will contain the following files, with the first six corresponding directly to figures from the paper:

- evalpessimism.pdf  
    - Graph showing observed versus predicted WCRTs for the experimental graphs. Corresponds to figure 10 in the paper.
- evaloverhead.pdf  
    - Graph showing observed versus predicted WCRTs for simple chains to measure overheads. Corresponds to figure 11 in the paper. 
- evalsim.pdf  
    - Graph showing observed versus simulated WCRTs for the experimental graphs. Corresponds to figure 12 in the paper.
- evalscalability.pdf  
    - Graph showing observed versus simulated WCRTs for the synthetic graphs to measure the scalability of our analysis. Corresponds to figure 13 in the paper.
- evalscalabilitypess.pdf  
    - Graph showing edge count vs relative pessimism for the synthetic graphs to measure the scalability of our analysis. Corresponds to figure 14 in the paper.
- evalanalysis.pdf  
    - Graph showing node count versus analysis time for the synthetic graphs to measure the scalability of our analysis. Corresponds to figure 15 in the paper.
- generatedexectimes.txt  
    - Sets of randomly generated execution times for each of the experimental graphs. These will be newly generated every time the experiments are run.
- observedoverheads.txt  
    - List of observed WCRTs for simple chains to measure overheads. Used to create evaloverhead.pdf.
- observedresponsetimes.txt  
    - List of observed WCRTs for the experimental graphs. Used to create evalpessimism.pdf.
- predictedresponsetimes.txt  
    - List of predicted WCRTs for the experimental graphs. Used to create evalpessimism.pdf and evalsim.pdf.  
- simulatedresponsetimes.txt
    - List of simulated WCRTs for the experimental graphs. Used to create evalsim.pdf.
- generatedexectimesscale.txt  
    - Sets of randomly generated execution times for each of the synthetic graphs. These will be newly generated every time the experiments are run.
- predictedresponsetimesscale.txt  
    - List of predicted WCRTs for the synthetic graphs. Used to create evalscalability.pdf and evalscalabilitypess.pdf.
- simulatedresponsetimesscale.txt
    - List of simulated WCRTs for the synthetic graphs. Used to create evalscalability.pdf and evalscalabilitypess.pdf.
- analysistimes.txt
    - List of averaged execution times for computing response time using our algorithm. Used to create evalanalysis.pdf.


## Code and inputs

The following documents the code and inputs used by `master.py` to populate `data`.

### rawgraph.txt

The file `rawgraph.txt` contains the raw data necessary to run our experiments. Graphs are represented as a set of tuples of two operators, 
representing an edge from the first operator to the second. For example, to represent a directed edge from an operator 'postprocessor', to 'visualizer':

    ('postprocessor', 'visualizer')

Each graph in the file is represented in three lines; the first line includes edges that are always in the graph, and the second and third 
are auxiliary inputs that can be used to include optional edges and nodes that are mutually exclusive to each other. If there are no such edges, 
the graph's information will be contained a single line with three newline characters at the end.

`rawscalabilitygraph.txt` contains similar raw data for large synthetic graphs used for scalability experiments.


### code/processDAGs.py

Code to directly read the contents of a file such as `rawgraph.txt` and construct usable graph data structures with the `networkx` package. 
This file also randomly generates execution times to be used in the experiments. As discussed in the evaluation section of the paper, we consider a 
total number of variations of each graph equal to how many operators it has. Each variation has a different execution time for each operator. This 
code also ensures that we are only using unique graphs by removing multiple copies of isomorphic graphs from consideration, along with unconnected 
graphs that violate our assumptions.


### code/DAGResponseTime.py

Uses the `networkx` graph representations created by `processDAGs.py` to find a worst-case end-to-end response bound for a given graph, using our
 bounds from the paper.

Outputs two files: `generatedexectimes.txt` and `predictedresponsetimes.txt`. The former contains the random execution times generated
by `processDAGs.py` for  each graph variation. These, alongside the graph structures, are used to compute the response time bounds. 
The latter file contains a WCRT bound for each graph variation. Can also perform profiling for timing measurement and output
`analysistimes.txt`.


### code/MakeVars.py

This file builds a Holoscan executable for each graph. The number of times each graph will execute is determined by the `iterations` argument given 
to `master.py`. The file `base.cpp` is a starting point, containing all the different types of operators necessary to build the programs (i.e., an 
operator that has two outgoing edges, an operator that has three incoming edges). For all graphs, `base.cpp` is modified to include code defining each 
of its operators, as well as code defining each of its edges, and then compiled. 

The graph structure information in `base.cpp` is hardcoded, as building Holoscan applications with arbitrary DAG structures may require new operators 
to be manually defined in `base.cpp`. Thus, if new graphs are added to `rawgraph.txt`, a matching definition of the structure will need to be added to 
`MakeVars.py`, along with any new operator types to `base.cpp`. Since the synthetic graphs are only simulated and not built and run, new graphs can be 
freely added to `rawscalabilitygraph.txt` without further modifications.


### code/RunExps.py

This file runs the main and overhead experiments using the executables built by `MakeVars.py`. It reads `generatedexectimes.txt` in order to modify 
`experimentbase.yaml` before each run of an executable. Then, when the application runs, it assigns each operator its execution time from the .yaml 
file. After the application finishes execution, the output log is parsed to find the WCRT, which is written to the file `observedresponsetimes.txt`


### code/visualize.py

Reads the output in the `data` folder in order to produce visualizations: the graphs `evaloverhead.pdf`, `evalpessimism.pdf`, `evalsim.pdf`, `evalscalability.pdf`, `evalscalabilitypess.pdf`, and `evalanalysis.pdf`.


### code/simulator.py

A discrete-event simulator that mimics the behavior of Holoscan applications, used for the main and scalability experiments. Takes the graph 
representations constructed by `processDAGs.py` and the execution times in `generatedexectimes.txt` as input. The simulation works by 
continuously emptying an event queue, with the possible events being either a new arrival to the system or an operator finishing execution. 
At each event, new events that are activated by the previous completion are enqueued, and the simulation continues until the time limit is 
reached (corresponding to approximately an hour of real execution time). The results are recorded in the file `simulatedresponsetimes.txt`.    


### base.cpp

The basic code defining each Holoscan application that we build, including code defining the operator classes and scheduler. The operators we 
use do not compute on data, instead busy-waiting for a predefined period of time on every activation. Each operator gets its execution time 
(the amount of ms it busy waits for) from the file `experiment.yaml`, which is generated for it by `RunExps.py` using `experimentbase.yaml`. 
`base.cpp` also defines the scheduler used by all applications. For our experiments, we use Holoscan's multithreaded scheduler running with 12 
cores to ensure no overhead due to contention, as all our apps use less than 12 cores.


### experimentbase.yaml

A base from which to build `experiment.yaml` files, used by Holoscan applications to configure themselves before running. This allows 
execution times and some scheduler properties to be changed without recompiling the executable.
