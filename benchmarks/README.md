# Holoscan benchmarks

Benchmarks provide workloads, measurement tools, and analysis resources for
understanding Holoscan performance. This directory registers their sources;
workload code, measurement procedures, and results stay with the source owner.

## Find and use a benchmark

| Source | What you will find |
| --- | --- |
| [holoscan-benchmarks](https://github.com/nvidia-holoscan/holoscan-benchmarks) | Benchmark workloads and measurement tools. [Registration](holoscan-benchmarks.json). |
| [Archived HoloHub benchmarks](https://github.com/nvidia-holoscan/holohub/tree/archive-4x/benchmarks) | Earlier workloads with their original SDK requirements. [Registration](holohub-benchmarks-archive.json). |

Clone the dedicated repository with Git:

```bash
git clone https://github.com/nvidia-holoscan/holoscan-benchmarks.git
cd holoscan-benchmarks
```

Start with its [measurement and analysis tools](https://github.com/nvidia-holoscan/holoscan-benchmarks/blob/main/benchmark_tools/README.md)
to measure an application, or its [workload overview](https://github.com/nvidia-holoscan/holoscan-benchmarks/blob/main/benchmarks/README.md)
to choose a benchmark. Follow the selected guide for hardware, SDK, data,
execution commands, and output locations. Analysis of existing logs may have
fewer requirements than running a workload.

Compare results only when hardware, software, input data, and measurement
settings are documented and comparable. Report tool or workload problems to
the source owner's issue tracker.

## Create a new benchmark

Follow [holoscan-benchmarks' contribution guide](https://github.com/nvidia-holoscan/holoscan-benchmarks/blob/main/CONTRIBUTING.md)
or maintain the benchmark in your own repository. Submit changes to existing
workloads and shared measurement tools to their owner.

Include:

- The question being measured, metric definitions and units, and workload
  configuration, including the application or module revision.
- Hardware, SDK and dependency versions, input data, and exact setup, run,
  and analysis commands with working directories and output locations.
- Warm-up handling, sample counts, repetitions, instrumentation overhead,
  and enough context to interpret and reproduce the results.
- A maintainer contact, license, README, and `metadata.json` following the
  [benchmark schema](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/src/holoscan_cli/metadata/benchmark.schema.json)
  for each registered benchmark project.

Use the source repository's documented measurement workflow. The CLI does not
provide a universal benchmark execution or analysis command.

## Verify and publish

Run the benchmark and analysis in the documented environment. Confirm that
outputs contain the intended measurements, validate the analysis, and record
the tested configuration. Run the source repository's checks before submitting
or publishing. HoloHub validates descriptors and discovery; project tests and
measurement evidence establish the validity of performance results.

## Register an existing benchmark

Check this directory's records first. A workload added to a registered collection
needs no separate record; update the collection's revision if required.
Otherwise, publish the source and metadata, then
[request registration with kind `benchmark`](https://github.com/nvidia-holoscan/holohub/issues/new?template=registry.yml&kind=benchmark).

Point to the benchmark directory or a collection of benchmark descriptors.
For a single benchmark, use its directory name as the registration name.
Follow [Contributing](../CONTRIBUTING.md) for form fields, manual submissions,
validation, and maintainer review.

[Back to HoloHub](../README.md)
