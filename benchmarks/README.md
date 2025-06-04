# rocCV Benchmarking Suite

## Description
The rocCV benchmark tools compare rocCV operator performance against libraries with similar capabilities (such as OpenCV). Currently, the benchmarks compare rocCV CPU operator implementations against GPU implementations.

## Running Benchmarks
1. Before running benchmarks, ensure that the benchmarks have been built using the `-D BENCHMARKS=ON` flag in the cmake configuration setup.
2. Benchmarks should be run through the Python frontend:

    ```bash
    python3 run_bench.py <benchmark_executable>
    ``` 
    Where the benchmark executable is typically located in `{build_folder}/bin/bench`. This will generate separate graphs for each benchmarking category for performance comparison purposes.
3. Benchmark results and generated graphs will be written to a directory `bench_results_{timestamp}`.