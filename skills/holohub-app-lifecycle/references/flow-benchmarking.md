# HoloHub flow benchmarking

Use this reference only after the application passes its normal finite and
visual acceptance checks.

## Preconditions

1. Read the checked-in flow-benchmark README, tutorial, runner source, and
   analyzer help from the same HoloHub revision.
2. Reconcile examples with local `./holohub` help. The tested revision uses
   `--extra-scripts` (plural).
3. Run the normal finite application once so compilation or engine generation
   is outside measured trials.

## Instrumented build

Preview and perform the exact selected mode/language build:

```bash
./holohub build <app> <mode> --language <cpp-or-python> --benchmark \
  --extra-scripts benchmarking --dryrun --verbose
./holohub build <app> <mode> --language <cpp-or-python> --benchmark \
  --extra-scripts benchmarking --verbose
```

## Run and analyze

Run helpers inside the wrapper-managed container, not directly on the host.
Inspect each helper's `-h`, require the runner's scheduler, and preserve the
compound command as one quoted shell argument after `--`:

```bash
./holohub run-container <app> <mode> --language <cpp-or-python> \
  --extra-scripts benchmarking --dryrun --verbose -- \
  'python benchmarks/holoscan_flow_benchmarking/benchmark.py -a <app> --language <cpp-or-python> --sched greedy -r 3 -i 1 -m <messages> -d <output-dir>'
```

Run the same shape without `--dryrun`, then run `analyze.py` in the same
environment. Check each parser independently; runner and analyzer short options
do not necessarily mean the same thing.

Preserve raw logs and analyzer output. Report workload, scheduler, instances,
runs, input, retained samples, warm-up/trim rules, operator path, aggregation,
variability, hardware, image, SDK, driver, and exclusions.

## Restore and prove normal state

Before instrumentation, record concise status, affected source hashes, and the
normal build shape. After the run, search the application tree for `*.bak`,
compare status and hashes with that baseline, and restore only
benchmark-attributable changes without overwriting pre-existing work. Automatic
restoration is not guaranteed after failure or interruption.

Preview and run the normal build without `--benchmark`, confirm from its
configuration/build evidence that benchmark flags are absent, and rerun the
finite smoke case. If normal source, build state, or smoke behavior is not
restored after this single pass, stop and hand off the exact state to
`holohub-debug-build-run`; do not clear caches or loop speculatively. A clean
source diff alone does not remove instrumented binaries or cached flags.

Do not turn a benchmark into an accuracy, safety, clinical, regulatory, or
product-wide performance claim.
