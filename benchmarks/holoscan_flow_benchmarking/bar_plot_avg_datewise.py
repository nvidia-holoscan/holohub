import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Usage: python bar_plot_avg_datewise <avg filename1> <avg filename2> ... <stddev filename1> <stddev
# filename2>
# Assumed that first half of the files are avg files and second half are stddev files
def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python bar_plot_avg_datewise <avg filename1> \
             <avg filename2> ... <stddev filename1> <stddev filename2>"
        )
        return
    filenames = sys.argv[1:]
    if len(filenames) % 2 != 0:
        print(
            "Usage: python bar_plot_avg_datewise <avg filename1> \
             <avg filename2> ... <stddev filename1> <stddev filename2>"
        )
        return
    filenames.sort(key=lambda x: x.split("_")[-1].split(".")[0])
    # read all the avg and stddev values
    all_avg_values = []
    all_stddev_values = []
    for filename in filenames:
        with open(filename) as f:
            if "avg" in filename:
                all_avg_values.append(float(f.readline().split(",")[0]))
            else:
                all_stddev_values.append(float(f.readline().split(",")[0]))

    # extract all the dates from only the avg files
    dates = []
    for filename in filenames:
        if "avg" in filename:
            dates.append(filename.split("_")[-1].split(".")[0])

    # Generate bar plots with dates
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.bar(dates, all_avg_values, yerr=all_stddev_values, color="tab:blue", alpha=0.6, capsize=5)
    # Label the actual values just on top of the bars
    for i, v in enumerate(all_avg_values):
        ax.text(i - 0.1, v * 1.05, str(round(v, 2)), fontsize=14, fontweight="bold")

    ax.set_ylabel("End-to-end Latency (ms)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=14, fontweight="bold")

    ax.set_title(
        "Average E2E Latency with Standard Deviation vs Date",
        fontsize=16,
        fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    max_value = max(all_avg_values)
    ax.set_ylim([0, max_value * 1.2])

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"avg_{dates[-1]}.png")
    print(
        f'<CTestMeasurementFile type="image/png" name="average">\
          avg_{dates[-1]}.png</CTestMeasurementFile>'
    )


if __name__ == "__main__":
    main()
