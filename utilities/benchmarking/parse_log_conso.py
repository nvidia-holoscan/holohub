# This python script parses the Data Frame Flow Tracking module's generated log file
# The format is the following:
#(replayer,1685129021110968,1685129021112852) -> (format_converter,1685129021113053,1685129021159460) -> (lstm_inferer,1685129021159626,1685129021161404) -> (tool_tracking_postprocessor,1685129021161568,1685129021194271) -> (holoviz,1685129021194404,1685129021265517)

# The format is (Operator1, receive timestamp, publish timestsamp) -> (Operator2, receive timestamp,
# publish timestsamp) -> ... -> (OperatorN, receive timestamp, publish timestsamp)

import sys
import matplotlib.pyplot as plt
import numpy as np

linestyles = ['--', '-.', '-', ':']
colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'brown']
index = 0

def parse_line(line):
    operators = line.split("->")
    # print (operators)
    op_timestamps = []
    for operator in operators:
        # trim whitespaces for left and right side
        # print ("op: ", operator)
        op_name_timestamp = operator.strip().rstrip()[1:-1]
        op_timestamps.append(op_name_timestamp.split(","))
    return op_timestamps

# return a path and latency pair where a path is a comma separate string of operators
def get_latency(op_timestamps):
    path = ""
    latency = 0
    for op_timestamp in op_timestamps:
        # print (op_timestamp)
        path += op_timestamp[0] + ","
    # convert the latency to ms
    latency = float(int(op_timestamps[-1][2]) - int(op_timestamps[0][1])) / 1000
    return path[:-1], latency

def parse_log(log_file):
    with open(log_file, "r") as f:
        paths_latencies = {}
        for line in f:
            # print ("line: ", line)
            if line[0] == "(":
                path_latency = get_latency(parse_line(line))
                if path_latency[0] in paths_latencies:
                    paths_latencies[path_latency[0]].append(path_latency[1])
                else:
                    paths_latencies[path_latency[0]] = [path_latency[1]]
        return paths_latencies

def get_avg_latencies(paths_latencies, skip_begin_messages = 10, discard_last_messages = 10):
    avg_latencies = {}
    for path in paths_latencies:
        # avg_latencies[path] =
        # sum(paths_latencies[path][skip_begin_messages:-discard_last_messages]) /
        # (len(paths_latencies[path]) - skip_begin_messages - discard_last_messages)
        avg_latencies[path] = np.mean(paths_latencies[path][skip_begin_messages:-discard_last_messages])
    return avg_latencies

def get_max_latencies(paths_latencies, skip_begin_messages = 10, discard_last_messages = 10):
    max_latencies = {}
    for path in paths_latencies:
        max_latencies[path] = max(paths_latencies[path][skip_begin_messages:-discard_last_messages])
    return max_latencies

# draw a CDF curve of the latencies using matplotlib where Y-Axis is the CDF and X-Axis is the
# latency. Show the image
def draw_cdf(ax, path_latencies, skip_begin_messages = 10, discard_last_messages = 10, label = None):
    global index

    data = sorted(path_latencies[1][skip_begin_messages:-discard_last_messages])
    data_max = max(data)
    data_avg = np.mean(data)
    data_stddev = np.std(data)
    # print (data)
    n = len(data)
    p = []
    ten_percentile = data[int(n * 0.1)]
    twenty_percentile = data[int(n * 0.2)]
    eighty_percentile = data[int(n * 0.8)]
    ninety_percentile = data[int(n * 0.9)]
    ninety_five_percentile = data[int(n * 0.95)]
    hundred_percentile = data[-1]
    flatness = int(ninety_percentile - ten_percentile)
    tail = int(hundred_percentile - ninety_five_percentile)
    print (f"label: {label}:: flatness: {flatness}, tail: {tail}")
    #print (f"label: {label}:: 10%: {ten_percentile}, 20%: {twenty_percentile}, 80%: {eighty_percentile}, 90%: {ninety_percentile}, 95%: {ninety_five_percentile}, 100%: {hundred_percentile}")
    for i in range(n):
        p.append(i/n)

    ax.plot(data, p, label=label, linewidth=2.0, color=colors[index], linestyle=linestyles[index])
    # ax.axvline(x=data_avg, color=colors[index], linestyle=linestyles[index], linewidth=1)
    # put a shaded area of stddev around average latency
    ax.axvspan(data_avg - data_stddev, data_avg + data_stddev, alpha=0.2, color=colors[index])

    ax.axvline(x=data_max, color=colors[index], linestyle=linestyles[index], linewidth=1.5)
    title_text = f"Avg: {data_avg:.2f}, Stddev: {data_stddev:.2f}, Max: {data_max:.2f}"
    # put an annotation of the max latency parallel to the max line
    # format float to 2 decimal places
    # ax.annotate("max: {:.2f}".format(data_max), xy=(data_max - 2, 0.5), xytext=(data_max - 2, 0.5), color=colors[index], rotation=90)
    # ax.annotate("avg: {:.2f}\nstddev: {:.2f}".format(data_avg, data_stddev), xy=(17, 0.5 - index / 5), xytext=(17, 0.5 - index / 5), color=colors[index])
    index += 1
    return title_text

def init_cdf_plot(numcharts=1):
    fig, axes = plt.subplots(ncols=3, nrows=(numcharts // 3), figsize=(16, 10))
    # fig.add_subplot(111, frameon=False)
    # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # fig.supxlabel("Latency (ms)", fontsize=12)
    # fig.supylabel("CDF", fontsize=12)
    fig.text(0.5, 0.02, 'End-to-end Latency (ms)', ha='center', va='center', size=12, weight='bold')
    fig.text(0.02, 0.55, 'Cumulative Distribution Function', ha='center', va='center', rotation='vertical', size=12, weight='bold')
    return fig, axes

def complete_cdf_plot(column, ax, title, morelines = None, labels=None):
    # convert the Y-axis ticks to percentage
    ax.set_ylim([-0.02, 1.03])
    ax.set_xlim(left=16)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    # ax.legend(prop={'size': 12})
    # ax.legend(prop={'size': 12}, loc="lower right", bbox_to_anchor=(0.9, 0.1))
    ax.set_title(title, fontsize=12, pad=48)
    if morelines:
        index = 0
        for line in morelines:
            xy = 18
            ax.annotate(labels[index] + ": " + line, xy=(xy, 1.18 - index/15), color=colors[index], fontsize=11, xycoords='data', annotation_clip=False)
            index += 1


# write a main function that takes a log file as argument and calls parse line
def main():
    global index
    if len(sys.argv) < 2:
        print("Usage: python parse_log.py <log file1> <log file2> <log file3>")
        sys.exit(1)

    log_files = sys.argv[1:]
    if len(log_files) % 3!= 0:
        print ("Please provide 3x number of log files")
        sys.exit(1)
    fig, axes = init_cdf_plot(len(log_files) // 3)
    labels = ["A4000", "A6000", "A4000+A6000"]
    # for every 2 elements in log files, create a chart
    handles = None
    for i in range(0, len(log_files), 3):
        chart_log_files = log_files[i:i+3]
        ax = axes[(i//3)//3][(i//3)%3]
        k = 0
        index = 0
        title_texts = []
        for log_file in chart_log_files:
            paths_latencies = parse_log(log_file)
            # print (get_avg_latencies(paths_latencies))
            # print (get_max_latencies(paths_latencies))
            for (path, latencies) in paths_latencies.items():
                txt = draw_cdf(ax, (path, latencies), label = labels[k])
                title_texts.append(txt)
                k += 1
                break # for now just log for one path because in endoscopy it's the same
        # get the last number before period (.) in chart_log_files[0]
        title = chart_log_files[0].split(".")[0][-1]
        complete_cdf_plot((i//3)%3, ax, title + " Instances", title_texts, labels)
        handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, prop={'size': 12, 'weight': 'bold'}, loc="upper center", ncols=3)
    fig.tight_layout()
    plt.tight_layout(pad=3.2)
    plt.subplots_adjust(bottom=0.07, left=0.06, hspace=0.32, wspace=0.15)
    # plt.show()
    # plt.savefig("endoscopy_cdf_consolidated_more.png", bbox_inches='tight')

# call the main function
if __name__ == "__main__":
    main()

#python3 parse_log_conso.py logger_single_4.log logger_single_a6000_4.log logger_multi_4.log logger_single_5.log logger_single_a6000_5.log logger_multi_5.log logger_single_6.log logger_single_a6000_6.log logger_multi_6.log logger_single_7.log logger_single_a6000_7.log logger_multi_7.log logger_single_8.log logger_single_a6000_8.log logger_multi_8.log logger_single_9.log logger_single_a6000_9.log logger_multi_9.log