# ##############################################################################
# Copyright (c)  - 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################


import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


# Graph style definitions
BAR_COLORS = [
    '#0078D4',  # Blue
    '#D83B01',  # Orange
    '#107C10',  # Green
    '#E81123',  # Red
    '#5C2D91',  # Purple
    '#00B7C3',  # Teal
    '#FFB900',  # Yellow
    '#7A7574',  # Gray
    '#E3008C',  # Magenta
    '#00AADE',  # Light Blue
    '#BAD80A',  # Lime Green
    '#B1560F',  # Brown
]
GRAPH_STYLE = "dark_background"


def plot_annotated_bars(ax, x, y, labels, format_string="%.2f"):
    num_benchmarks_in_group = len(y)
    x_indices = np.arange(len(x))  # Positions for the groups of bars

    total_width_for_group = 0.8  # Total width that bars for one batch size will occupy
    bar_width = total_width_for_group / num_benchmarks_in_group

    all_bar_containers_for_category = []  # To store bar containers for annotation

    for idx, data in enumerate(y):
        # Calculate offset for each bar within the group
        offset = (idx - num_benchmarks_in_group / 2.0 + 0.5) * bar_width
        current_bar_positions = x_indices + offset
        selected_bar_color = BAR_COLORS[idx % len(y)]
        bar_container = ax.bar(current_bar_positions, data,
                               width=bar_width, label=labels[idx], color=selected_bar_color)
        all_bar_containers_for_category.append(bar_container)

    # Annotate bar values
    for bar_container in all_bar_containers_for_category:
        for bar_patch in bar_container.patches:
            bar_height = bar_patch.get_height()
            # Get the center x-coordinate of the bar
            text_x = bar_patch.get_x() + bar_patch.get_width() / 2.0
            # Position text slightly above the bar
            text_y = bar_height

            ax.text(text_x, text_y, format_string % bar_height,  # Format to decimal places
                    ha='center', va='bottom', fontsize=6, color='lightgray', zorder=10)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graphs from benchmark results.")
    parser.add_argument("benchmark_results", help="Path to the benchmark results JSON file.")
    parser.add_argument("-o", "--output-dir", help="Directory to save the generated graphs.", default=".")
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    plt.style.use(GRAPH_STYLE)
    with open(args.benchmark_results, "r") as file:
        data = json.load(file)

    # Get device information from results file
    cpu_name = data["metadata"]["cpu"]
    cpu_threads = data["metadata"]["cpu_threads"]
    gpu_name = data["metadata"]["gpu"]

    graph_footnote = f"Benchmarks performed with {gpu_name} (GPU) and {cpu_name} (CPU) with {cpu_threads} threads. Execution time does not include data transfer latency from CPU to GPU and vice-versa."

    for category in data["results"]:
        fig, ax = plt.subplots(1, 3, dpi=200, figsize=(15, 6))

        # Setup execution time axis
        ex_time_ax = ax[0]
        ex_time_ax.set_xlabel("Batch Size")
        ex_time_ax.set_ylabel("Execution Time (s) [Log Scale]\n(Lower is better)")
        ex_time_ax.set_title("Execution Time")
        ex_time_ax.set_yscale('log')

        # Setup FPS axis
        fps_ax = ax[1]
        fps_ax.set_ylabel("Frames per Second [Log Scale]\n(Higher is better)")
        fps_ax.set_xlabel("Batch Size")
        fps_ax.set_title("Frames per Second")
        fps_ax.set_yscale('log')

        # Setup throughput axis
        throughput_ax = ax[2]
        throughput_ax.set_ylabel("Throughput (GB/s) [Log Scale]\n(Higher is better)")
        throughput_ax.set_xlabel("Batch Size")
        throughput_ax.set_title("Total Memory Throughput")
        throughput_ax.set_yscale('log')
        
        benchmarks_in_category = data["results"][category]
        
        # Shared x-axis: sample counts (taken from the first benchmark's runs)
        first_benchmark = next(iter(benchmarks_in_category.values()))
        samples = [run["samples"] for run in first_benchmark]

        # One entry per benchmark name (e.g. "GPU", and potentially "CPU" later)
        benchmark_names = []          # labels for the legend
        execution_time_data = []      # list of lists: one list of values per benchmark
        fps_data = []
        throughput_data = []

        for bench_name, runs in benchmarks_in_category.items():
            benchmark_names.append(bench_name)

            ex_times = []
            fps_vals = []
            tp_vals = []

            for run in runs:
                ex_time_ms = run["execution_time"]
                n_samples = run["samples"]
                total_bytes = run["read_memory_bytes"] + run["written_memory_bytes"]

                ex_times.append(run["execution_time"])
                fps_vals.append(n_samples / run["execution_time"])
                tp_vals.append(total_bytes / run["execution_time"] / 1e9)  # GB/s

            execution_time_data.append(ex_times)
            fps_data.append(fps_vals)
            throughput_data.append(tp_vals)

        plot_annotated_bars(ex_time_ax, samples, execution_time_data, benchmark_names, format_string="%.4g")
        plot_annotated_bars(fps_ax, samples, fps_data, benchmark_names, format_string="%.2f")
        plot_annotated_bars(throughput_ax, samples, throughput_data, benchmark_names, format_string="%.2f")

        
        handles, labels = ex_time_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 0.10))

        # Set bottom text for entire figure
        fig.subplots_adjust(bottom=0.25, wspace=0.35)
        fig.text(0.5, 0.02, graph_footnote, wrap=True, ha='center', fontsize=8, alpha=0.7)
        fig.suptitle(f"{category} Benchmarks")

        output_filename = os.path.join(args.output_dir, f"bench_{category}.png")

        fig.savefig(output_filename)
        print(f"Saved graph to {output_filename}")
