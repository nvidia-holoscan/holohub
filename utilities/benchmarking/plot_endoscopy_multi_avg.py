import matplotlib.pyplot as plt

def get_execution_time_data(exec_file):
    with open(exec_file, 'r') as f:
        exec_lines = f.readlines()
    exec_data = []
    line_index = 0
    for num_instances in range(1, 11):
        avg_exec_time = sum(float(line.strip()) for line in exec_lines[line_index:line_index+num_instances]) / num_instances
        exec_data.append(avg_exec_time)
        line_index += num_instances
    return exec_data

def get_gpu_utilization_data(utilization_file):
    # Read GPU utilization from file
    with open(utilization_file, 'r') as f:
        utilization_lines = f.readlines()

    # Average GPU utilization for each number of instances
    utilization_data = []
    line_index = 0
    for num_instances in range(1, 11):
        utilizations = [float(value) for value in utilization_lines[line_index].strip().split(',')]
        avg_utilization = sum(utilizations) / len(utilizations)
        utilization_data.append(avg_utilization)
        line_index += 1
    return utilization_data

exec_data_multi = get_execution_time_data('build/endoscopy_average_multi.txt')
exec_data_single = get_execution_time_data('build/endoscopy_average_single.txt')
exec_data_a6000 = get_execution_time_data('build/endoscopy_average_a6000.txt')

gpu_util_multi = get_gpu_utilization_data('gpu_utilization_multi.txt')
gpu_util_single = get_gpu_utilization_data('gpu_utilization_single.txt')
gpu_util_a6000 = get_gpu_utilization_data('gpu_utilization_a6000.txt')

# Plotting
fig, ax1 = plt.subplots()
bar_width = 0.55

# Plot average execution time
xaxis = range(2, 2 * (len(exec_data_single) + 1), 2)
xaxis1 = [x - bar_width for x in xaxis]
ax1.bar(xaxis1, exec_data_single, color = 'tab:red', label='A4000 (Latency)', alpha=0.5, width=bar_width, hatch='//', edgecolor='black')

ax1.bar(xaxis, exec_data_a6000, color = 'tab:blue', label='A6000 (Latency)', alpha=0.5, width=bar_width, hatch='\\', edgecolor='black')

xaxis2 = [x + bar_width for x in xaxis]
ax1.bar(xaxis2, exec_data_multi, color = 'tab:green', label='A4000 + A6000 (Latency)', alpha=0.5, width=bar_width, hatch='.', edgecolor='black')
# set y-axis as log scale
# ax1.set_yscale('log')
# # but the yticks should be in normal format
# from matplotlib.ticker import ScalarFormatter
# formatter = ScalarFormatter(useOffset=False, useMathText=True)
# formatter.set_scientific(False)
# ax1.yaxis.set_major_formatter(formatter)
# ax1.yaxis.set_minor_formatter(formatter)

ax1.set_ylim(20, 130)
ax1.set_xlabel('Number of Instances')
ax1.set_ylabel('Average End-to-end Latency (ms)')
ax1.tick_params('y')

# set xticks and xticklabels to be the same as the number of instances
ax1.set_xticks(xaxis)
# ax1.set_xticklabels(range(1, len(exec_data_single) + 1))

# Create a second y-axis
ax2 = ax1.twinx()

# Plot GPU utilization
ax2.plot(xaxis, gpu_util_single, color = 'tab:red', linestyle='--', marker='o', alpha=0.8, label = "A4000 (GPU Util)")
ax2.plot(xaxis, gpu_util_a6000, color = 'tab:blue', linestyle='-.', marker='x', alpha=0.8, label = "A6000 (GPU Util)")
ax2.plot(xaxis, gpu_util_multi, 'g-', marker='.', label = "A4000 + A6000 (GPU Util)")
ax2.set_ylim(0, 104)
ax2.set_ylabel('Average GPU Utilization (%)')
ax2.tick_params('y')
# change xtick labels to be the half of their actual values
ax1.set_xticklabels([int(x/2) for x in xaxis])
# ax2.set_yticks(ax2.get_yticks())

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels, loc='upper left', fontsize=10, markerscale=1.3)

# Display the plot
# plt.title('Multi-GPU (A4000 + A6000) vs Single GPU (A4000, A6000) \nfor Endoscopy Tool Tracking Application')
plt.tight_layout()
# plt.show()
# save the figure
plt.savefig('endoscopy_avg_more.png', bbox_inches='tight')
