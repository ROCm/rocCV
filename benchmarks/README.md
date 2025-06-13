# rocCV Benchmarking Suite

## Description
The rocCV benchmark tools compare rocCV operator performance against libraries with similar capabilities (such as OpenCV). Currently, the benchmarks compare rocCV CPU operator implementations against GPU implementations.

## Building and Running Benchmarks

### Building Benchmarks

To build the benchmarks, follow these steps:

1.  Navigate to the root of the rocCV project.
2.  Create a build directory if you haven't already:
    ```bash
    mkdir build
    cd build
    ```
3.  Run CMake with the `BENCHMARKS` option enabled:
    ```bash
    cmake -DBENCHMARKS=ON ..
    ```
4.  Build the project:
    ```bash
    make [-j <num_cores>]
    ```
    This will generate an executable file named `roccv_bench` in the `bin` directory within your build directory (e.g., `build/bin/roccv_bench`).

### Running Benchmarks

Once built, you can run the benchmarks from the build directory using the default configuration file provided in this directory (`benchmarks/config.json`):

```bash
./bin/roccv_bench --config ../benchmarks/config.json
```

This will generate a JSON file containing results of the benchmarks, `roccv_bench_results.json` by default.

You can also specify command-line arguments to list the available benchmarking categories, control which benchmarks are run, or to modify their parameters. For detailed options, run:

```bash
./bin/roccv_bench --help
```

## Graphing Benchmark Results
A python tool, `benchmarks/generate_graphs.py`, is provided to graph results output from `roccv_bench`. Run this tool from the `benchmarks` directory (assuming you've built the project in `build`):

```bash
python3 generate_graphs.py ../build/roccv_bench_results.json
```
This will output image files containing graphs for each benchmark category run in the benchmark.

You can also specify command-line arguments to control the output directory of the graphs. For detailed options, run:

```bash
python3 generate_graphs.py --help
```