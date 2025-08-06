import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

metric_names = {
    "max": "Maximum E2E Latency",
    "tail": "E2E Latency Distribution Tail",
    "flatness": "E2E Latency Distribution Flatness",
}


# Usage: python bar_plot_generic_datewise.py <filename1> <filename2> ...
def main():
    if len(sys.argv) < 2:
        print("Usage: python bar_plot_generic_datewise.py <filename1> <filename2> ...")
        return
    filenames = sys.argv[1:]
    # sort the filenames by date where filename format is <metric_name>_<date>.csv
    filenames.sort(key=lambda x: x.split("_")[-1].split(".")[0])
    # read all the single values from all files as arrays. All files have a value of "value,"
    all_values = []
    for filename in filenames:
        with open(filename) as f:
            all_values.append(float(f.readline().split(",")[0]))

    # extract all the dates
    dates = []
    for filename in filenames:
        dates.append(filename.split("_")[-1].split(".")[0])

    # Generate bar plots with dates
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.bar(dates, all_values, color="tab:blue", alpha=0.6)
    # Label the actual values just on top of the bars
    for i, v in enumerate(all_values):
        ax.text(i - 0.1, v * 1.05, str(round(v, 2)), fontsize=14, fontweight="bold")

    ax.set_ylabel("End-to-end Latency (ms)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=14, fontweight="bold")

    current_metric_name = filenames[0].split("_")[0]
    ax.set_title(f"{metric_names[current_metric_name]} vs Date", fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    max_value = max(all_values)
    ax.set_ylim([0, max_value * 1.2])

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{current_metric_name}_{dates[-1]}.png")
    print(
        f'<CTestMeasurementFile type="image/png" \
          name="{current_metric_name}">{current_metric_name}_{dates[-1]}.\
          png</CTestMeasurementFile>'
    )


if __name__ == "__main__":
    main()
