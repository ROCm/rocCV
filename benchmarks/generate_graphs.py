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


def generate_graphs():
    plt.style.use(GRAPH_STYLE)

    with open(sys.argv[1], "r") as file:
        data = json.load(file)

    # Get device information from results file
    cpu_name = data["device_info"]["cpu"]["name"]
    cpu_threads = data["device_info"]["cpu"]["threads"]
    gpu_name = data["device_info"]["gpu"]["name"]

    graph_footnote = f"Benchmarks performed with {gpu_name} (GPU) and {cpu_name} (CPU) with {cpu_threads} threads."

    for category in data["results"]:
        benchmarks_in_category = data["results"][category]

        if not benchmarks_in_category:
            print(f"No benchmarks found for category: {category}. Skipping.")
            continue

        # Assume all benchmarks in a category share the same "batches" array for the x-axis
        # and that "batches" is not empty.
        try:
            x_labels_text = benchmarks_in_category[0]["batches"]
            if not x_labels_text:
                print(f"No batch sizes found for benchmarks in category: {category}. Skipping.")
                continue
        except (KeyError, IndexError):
            print(f"Could not retrieve batch sizes for category: {category}. Skipping.")
            continue

        fig, ax = plt.subplots(dpi=150)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Execution Time (ms) [Log Scale]")
        ax.set_title(f"{category} Performance Comparison")
        ax.set_yscale('log')

        num_benchmarks_in_group = len(benchmarks_in_category)
        x_indices = np.arange(len(x_labels_text))  # Positions for the groups of bars

        total_width_for_group = 0.8  # Total width that bars for one batch size will occupy
        bar_width = total_width_for_group / num_benchmarks_in_group

        all_bar_containers_for_category = []  # To store bar containers for annotation

        for idx, benchmark_data in enumerate(benchmarks_in_category):
            execution_times = benchmark_data["execution_time"]
            # Calculate offset for each bar within the group
            offset = (idx - num_benchmarks_in_group / 2.0 + 0.5) * bar_width
            current_bar_positions = x_indices + offset
            selected_bar_color = BAR_COLORS[idx % len(benchmarks_in_category)]
            bar_container = ax.bar(current_bar_positions, execution_times,
                                   width=bar_width, label=benchmark_data["name"], color=selected_bar_color)
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
        ax.set_xticklabels(x_labels_text)

        fig.subplots_adjust(bottom=0.2)  # Adjust bottom to make space for footnote
        fig.text(0.5, 0.02, graph_footnote, wrap=True, ha='center', fontsize=8, alpha=0.7)  # Centered footnote
        ax.legend()

        output_filename = f"bench_{category}_comparison.png"
        fig.savefig(output_filename)
        print(f"Saved graph to {output_filename}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <benchmark_results>")
        quit()

    generate_graphs()
