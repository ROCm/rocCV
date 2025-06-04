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

import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

BENCH_DIR = f"bench_results_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
BENCH_RESULTS_FILE = os.path.join(BENCH_DIR, "benchmarks.csv")
DEVICE_INFO_FILE = os.path.join(BENCH_DIR, "device_info.csv")
GRAPH_OUTPUT_DIR = os.path.join(BENCH_DIR, "graphs")
NUM_RUNS = 5


def run_benchmarks():
    bench_cmd = sys.argv[1]
    subprocess.run([bench_cmd, str(NUM_RUNS), BENCH_DIR])


def generate_graphs():
    # Gather device information from device_info file
    device_info_df = pd.read_csv(DEVICE_INFO_FILE)
    gpu_name = device_info_df["GPU Name"][0]
    cpu_name = device_info_df["CPU Name"][0]
    cpu_threads = device_info_df["Thread Count"][0]
    experiment_info_text = f"Tests performed on a {gpu_name} GPU and a {cpu_name} CPU with {cpu_threads} threads. GPU metrics include only the kernel execution time."

    # Read benchmark results and graph each distinct category
    df = pd.read_csv(BENCH_RESULTS_FILE)
    for category, category_df in df.groupby("Category"):

        image_height = category_df["Height"][0]
        image_width = category_df["Width"][0]

        fig, ax = plt.subplots()

        for name, name_df in category_df.groupby("Name"):
            ax.plot(name_df["Batches"], name_df["Execution Time"], label=name)

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Execution Time (ms)")
        ax.set_title(f"Image Size: {image_height}x{image_width}", fontsize=10, color=(0, 0, 0, 0.6))
        fig.suptitle(f"{category} Performance")
        ax.legend()
        fig.subplots_adjust(bottom=0.2)
        fig.text(0.1, 0.05, experiment_info_text, wrap=True, fontsize=8, alpha=0.6)

        graph_filename = os.path.join(GRAPH_OUTPUT_DIR, f"bench_{category}.png")
        fig.savefig(graph_filename)
        print(f"Saved graph to {graph_filename}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <benchmark_executable>")
        quit()

    run_benchmarks()
    generate_graphs()
