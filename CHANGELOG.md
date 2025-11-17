# Changelog for rocCV

The full documentation for rocCV is available at [https://rocm.docs.amd.com/projects/rocCV/en/latest/index.html](https://rocm.docs.amd.com/projects/rocCV/en/latest/index.html)

## rocCV 0.4.0 for ROCm 7.2.0

### Changed

- `Pybind11` and `DLPack` requirements moved to git submodules.

### Known issues

- N/A

### Upcoming changes

- TBD

## rocCV 0.3.0 for ROCm 7.1.0

### Changed

- AMD Clang - Location update to `${ROCM_PATH}/lib/llvm/bin`

### Known issues

- Installation on CentOS/RedHat/SLES requires the manual installation of the `DLPack` and `OpenCV` packages.
- On Ubuntu 22.04 use `pip3 install numpy~=1.23`

### Upcoming changes

- TDB

## rocCV 0.2.0 for ROCm 7.0.0

### Changed

- AMD Clang is now the default CXX and C compiler.

### Known issues

- Installation on CentOS/RedHat/SLES requires the manual installation of the `DLPack` and `OpenCV` packages.
- On Ubuntu 22.04 use `pip3 install numpy~=1.23`

### Upcoming changes

- TDB
