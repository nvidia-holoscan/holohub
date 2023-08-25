import matplotlib.pyplot as plt

def set_handle_height(legend, height):
    for handle in legend.legend_handles:
        handle.set_height(height)

def get_execution_time_data(exec_file):
    with open(exec_file, 'r') as f:
        exec_lines = f.readlines()
    exec_data = []
    line_index = 0
    for num_instances in range(1, 11):
        exec_time = sum(float(line.strip()) for line in exec_lines[line_index:line_index+num_instances]) / num_instances
        exec_data.append(exec_time)
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

exec_data_multi = get_execution_time_data('build/endoscopy_max_multi.txt')
exec_data_single = get_execution_time_data('build/endoscopy_max_single.txt')

gpu_util_multi = get_gpu_utilization_data('gpu_utilization_multi.txt')
gpu_util_single = get_gpu_utilization_data('gpu_utilization_single.txt')

# Plotting
fig, ax1 = plt.subplots()
bar_width = 0.4

# Plot max execution time
xaxis = range(1, len(exec_data_single) + 1)
xaxis1 = [x - bar_width / 2 for x in xaxis]
ax1.bar(xaxis1, exec_data_multi, color = 'tab:red', label='A4000 + A6000 (Latency)', alpha=0.5, width=bar_width, hatch='//', edgecolor='black')

xaxis2 = [x + bar_width / 2 for x in xaxis]
ax1.bar(xaxis2, exec_data_single, color = 'tab:blue', label='A4000  (Latency)', alpha=0.5, width=bar_width, hatch='\\', edgecolor='black')
# set y-axis as log scale
# ax1.set_yscale('log')
# # but the yticks should be in normal format
# from matplotlib.ticker import ScalarFormatter
# formatter = ScalarFormatter(useOffset=False, useMathText=True)
# formatter.set_scientific(False)
# ax1.yaxis.set_major_formatter(formatter)
# ax1.yaxis.set_minor_formatter(formatter)

ax1.set_ylim(20, 250)
ax1.set_xlim(0, 10.9)
ax1.set_xlabel('Number of Instances')
ax1.set_ylabel('Maximum End-to-end Latency (ms)')
ax1.tick_params('y')

# set xticks and xticklabels to be the same as the number of instances
ax1.set_xticks(xaxis)
# ax1.set_xticklabels(range(1, len(exec_data_single) + 1))
# set_handle_height(ax1.legend(), 5)

# Create a second y-axis
ax2 = ax1.twinx()

# Plot GPU utilization
ax2.plot(range(1, len(gpu_util_multi) + 1), gpu_util_multi, 'r-', marker='.', label = "A4000 + A6000 (GPU Util)")
ax2.plot(range(1, len(gpu_util_single) + 1), gpu_util_single, color = 'tab:blue', linestyle='--', marker='x', alpha=0.8, label = "A4000 (GPU Util)")
ax2.set_ylim(0, 104)
ax2.set_ylabel('Average GPU Utilization (%)')
ax2.tick_params('y')
# ax2.set_yticks(ax2.get_yticks())

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
legends = ax1.legend(lines, labels, loc='best', fontsize=11, markerscale=1.3)

# Display the plot
plt.title('Multi-GPU (A4000 + A6000) vs Single GPU (A4000) \nfor Endoscopy Tool Tracking Application')
plt.tight_layout()
# plt.show()
# save the figure
plt.savefig('endoscopy_max.png', bbox_inches='tight')
