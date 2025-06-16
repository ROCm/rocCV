'''
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''

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


def plot_annotated_bars(ax, x, y, labels):
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

            ax.text(text_x, text_y, f'{bar_height:.2f}',  # Format to 2 decimal places
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
    cpu_name = data["device_info"]["cpu"]["name"]
    cpu_threads = data["device_info"]["cpu"]["threads"]
    gpu_name = data["device_info"]["gpu"]["name"]

    graph_footnote = f"Benchmarks performed with {gpu_name} (GPU) and {cpu_name} (CPU) with {cpu_threads} threads. Execution time does not include data transfer latency from CPU to GPU and vice-versa."

    for category in data["results"]:
        benchmarks_in_category = data["results"][category]

        fig, ax = plt.subplots(1, 2, dpi=200, figsize=(10, 6))

        # Setup execution time axis
        ex_time_ax = ax[0]
        ex_time_ax.set_xlabel("Batch Size")
        ex_time_ax.set_ylabel("Execution Time (ms) [Log Scale]")
        ex_time_ax.set_title("Execution Time")
        ex_time_ax.set_yscale('log')

        # Setup FPS axis
        fps_ax = ax[1]
        fps_ax.set_ylabel("Frames per Second [Log Scale]")
        fps_ax.set_xlabel("Batch Size")
        fps_ax.set_title("Frames per Second")
        fps_ax.set_yscale('log')

        # Gather data and plot results
        benchmark_names = []
        execution_time_data = []
        batches = data["results"][category][0]["batches"]
        image_height = data["results"][category][0]["height"][0]
        image_width = data["results"][category][0]["width"][0]
        fps_data = []

        for benchmark in benchmarks_in_category:
            benchmark_names.append(benchmark["name"])
            execution_time_data.append(benchmark["execution_time"])
            fps_data.append([1000 / (benchmark["execution_time"][i] / batches[i])
                            for i in range(len(benchmark["execution_time"]))])

        plot_annotated_bars(ex_time_ax, batches, execution_time_data, benchmark_names)
        plot_annotated_bars(fps_ax, batches, fps_data, benchmark_names)

        ex_time_ax.legend()
        fps_ax.legend()

        # Set bottom text for entire figure
        fig.subplots_adjust(bottom=0.2)  # Adjust bottom to make space for footnote
        fig.text(0.5, 0.04, graph_footnote, wrap=True, ha='center', fontsize=8, alpha=0.7)  # Centered footnote
        fig.suptitle(f"{category} Benchmarks (Batches of {image_width}x{image_height} 8-bit Images)")

        output_filename = os.path.join(args.output_dir, f"bench_{category}.png")

        fig.savefig(output_filename)
        print(f"Saved graph to {output_filename}")
