# Release Benchmarking Guide

This tutorial provides a simplified interface to evaluate Holoscan SDK performance on two
commonly visited HoloHub applications.

## Background

Holoscan SDK emphasizes low end-to-end latency in application pipelines. In addition to other
benchmarks, we can use HoloHub applications to evaluate Holoscan SDK performance over releases.

In this tutorial we provide a reusable workflow to evaluate end-to-end latency performance on
the Endoscopy Tool Tracking and Multi-AI Ultrasound HoloHub projects. These projects are generally
maintained by the NVIDIA Holoscan team and demonstrate baseline Holoscan SDK inference pipelines
with video replay and Holoviz rendering output.

Benchmark scenarios include:
- Running multiple Holoscan SDK pipelines concurrently on a single machine
- Running video replay input at real-time speeds or as fast as possible
- Running Holoviz output with either visual rendering or in headless mode

We plan to release HoloHub benchmarks in the [release subfolder](release) following Holoscan SDK general
releases. You can follow the tutorial below to similarly evaluate performance on your own machine.

Refer to related documents for more information:
- the [results report template file](template/results.md.tmpl) provides additional background on
definitions and background
- versioned releases are available for review in the [release subfolder](release)

## Getting Started

Data collection can be run in the HoloHub base container for both the Endoscopy Tool Tracking and the Multi-AI Ultrasound applications. We've provided a custom Dockerfile with tools to process collected data into a benchmark report.

```bash
# Build the container
./dev_container build \
    --img holohub:release_benchmarking \
    --docker_file benchmarks/release_benchmarking/Dockerfile \
    --base_img nvcr.io/nvidia/clara-holoscan/holoscan:<holoscan-sdk-version>-$(./dev_container get_host_gpu)

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
2. Add platform configuration information:
```bash
./run launch release_benchmarking --extra_args "--print" > benchmarks/release_benchmarking/output/platform.txt
```
3. Transfer output contents from each platform to a single machine:
```bash
# Compress information for transfer
pushd benchmarks/release_benchmarking
tar cvf benchmarks-<platform-name>.tar.gz output/*

# Migrate the results archive with your transfer tool of choice, such as SCP

# Extract results to a subfolder on the target machine
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

You can use the template markdown file in the [`template`](./template/) folder to generate a markdown
or PDF report with benchmark data with `pandoc` and `Jinja2`.

1. Copy and edit `template/release.json` with information about the benchmarking configuration, including
the release version, platform configurations, and local paths to processed data. Run
`./run launch` to print JSON-formatted platform details to the console about the current system:
```bash
./dev_container launch --img holohub:release_benchmarking
./run launch release_benchmarking --extra_args "--print"
```
2. Render the document with the Jinja CLI tool:
```bash
pushd benchmarks/release_benchmarking
jinja2 template/results.md.tmpl template/<release-version>.json --format=json > output/<release-version>.md
```

### (Optional) Generating a PDF report document

You can convert the report to PDF format as an easy way to share your report as a single file
with embedded plots.

1. In your copy of `template/release.json`, update the `"format"` string to `"pdf"`.
2. Follow the instructions above to generate your markdown report with Jinja2.
3. Use `pandoc` to convert the markdown file to PDF:
```bash
pushd output
pandoc <release-version>.md -o <release-version>.pdf --toc
```

### (Optional) Submitting Results to HoloHub

The Holoscan SDK team may submit release benchmarking reports to HoloHub git history for general visibility. We use Markdown formatting to make plot diagrams accessible for direct download.

1. Move `<release-version>.md` and accompanying plots to a new `release/<version>` folder.
2. Update image paths in `<release-version.md>` and verify locally with a markdown renderer such as VS Code.
3. Commit changes, push to GitHub, and open a Pull Request.

## Cleanup
Benchmarking changes to application YAML files can be discarded after benchmarks complete.
```bash
git checkout applications/*.yaml
```

## Troubleshooting

__Why am I seeing high end-to-end latency spikes as outliers in my data?__

Latency spikes may occur in display-driven benchmarking if the display goes to sleep. Please configure your
display settings to prevent the display from going to sleep before running benchmarks.

We have also infrequently observed latency spikes in cases where display drivers and CUDA Toolkit
versions are not matched.

__Benchmark applications are failing silently without writing log files.__

Silent failures may indicate an issue with the underlying applications undergoing benchmarking.
Try running the applications directly and verify execution is as expected:
- `./run launch endoscopy_tool_tracking cpp`
- `./run launch multiai_ultrasound cpp`

In some cases you may need to clear your HoloHub build or data folders to address errors:
- `./run clear_cache`
- `rm -rf ./data`
