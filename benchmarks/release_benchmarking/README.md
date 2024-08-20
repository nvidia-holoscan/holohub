# Release Benchmarking Guide

This tutorial provides a simplified interface to evaluate Holoscan SDK performance on two
commonly visited HoloHub applications.

## Getting Started

Both the Endoscopy Tool Tracking and the Multi-AI Ultrasound applications use the
HoloHub base container.

```bash
# Build the container
./dev_container build \
    --img holohub:release_benchmarking \
    --docker_file benchmarks/release_benchmarking/Dockerfile

# Launch the dev environment
./dev_container launch --img holohub:release_benchmarking

# Inside the container, build the applications in benchmarking mode
./run build endoscopy_tool_tracking --benchmark
./run build multiai_ultrasound --benchmark

./run build release_benchmarking
```

Run the benchmarking script with no arguments to collect performance logs in the `./output` directory.
```bash
./run launch release_benchmarking
```

## Summarizing Data

After running benchmarks, inside the dev environment, use `run launch` to process data statistics and create bar plot PNGs: 
```bash
./dev_container launch --img holohub:release_benchmarking
./run launch release_benchmarking --extra_args "--process benchmarks/release_benchmarking"
```

Alternatively, collect results across platforms. On each machine:
1. Run benchmarks:
```bash
./run launch release_benchmarking
```
2. Include platform configuration information:
```bash
./run launch release_benchmarking --extra_args "--print" > benchmarks/release_benchmarking/output/platform.txt
```
2. Compress output contents:
```bash
pushd benchmarks/release_benchmarking
tar cvf benchmarks-<platform-name>.tar.gz output/*
```
3. Copy and extract contents to a platform subfolder on a single machine:
```bash
mkdir -p output/<release>/<platform-name>/
pushd output/<release>/<platform-name>
tar xvf benchmarks-<platform-name>
```
4. Use multiple `--process` flags to generate a batch of bar plots for multiple platform results:
```bash
./run launch release_benchmarking --extra_args "\
    --process benchmarks/release_benchmarking/2.4/x86_64 \
    --process benchmarks/release_benchmarking/2.4/IGX_iGPU \
    --process benchmarks/release/benchmarking/2.4/IGX_dGPU"
```

## Presenting Data

You can use the template markdown file in the [`template`](./template/) folder to generate a PDF
report with benchmark data with `pandoc` and `Jinja2`.

1. Edit `template/release.json` with information about the benchmarking configuration, including
the release version, platform configurations, and local paths to processed data. Run
`./run launch` to print JSON-formatted platform details to the console about the current system:
```bash
./dev_container launch --img holohub:release_benchmarking
./run launch release_benchmarking --extra_args "--print"
```
2. Render the document with the Jinja CLI tool:
```bash
pushd benchmarks/release_benchmarking
jinja2 template/results.md.tmpl template/release.json --format=json > output/<release-version>.md
```
3. Embed images as a PDF document:
```bash
pushd output
pandoc <release-version>.md -o <release-version>.pdf --toc
```
4. (Optional) Add the PDF document to the release folder:
```bash
pushd /workspace/holohub/benchmarks/release_benchmarking
mv output/<release-version>.pdf release/
```

## Cleanup
Benchmarking changes to application YAML files can be discarded after benchmarks complete.
```bash
git checkout applications/*.yaml
```

## Troubleshooting

__Why am I seeing high end-to-end latency spikes as outliers in my data?__

Latency spikes may occur in display-driven benchmarking if the display goes to sleep. Please configure your
display settings to prevent the display from going to sleep before running benchmarks.
