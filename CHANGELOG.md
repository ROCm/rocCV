# Changelog for rocCV

The full documentation for rocCV is available at [https://rocm.docs.amd.com/projects/rocCV/en/latest/index.html](https://rocm.docs.amd.com/projects/rocCV/en/latest/index.html)

## (Unreleased) rocCV 0.5.0 for ROCm 7.12.0

### Added
- Added `Reformat` operator.
- Added `BrightnessContrast` operator.
- Added `Gaussian` operator.
- Added `Laplacian` operator.
- Dockerfile support for running rocCV in a container environment.

### Changed
- Switched from submodules to CMake's `FetchContent()` for DLPack and Pybind11 dependencies.
- Benchmarking suite now uses `rocRAND` for random data generation on the device.
- OpenMP moved from public -> private dependency for roccv.

### Known issues
- Python bindings are installed improperly, causing double import issues when importing it into a project.

### Upcoming changes
- TBD

## rocCV 0.4.0 for ROCm 7.2.0

### Added
- Reflect 101 border mode.
- Bicubic interpolation.
- Convert To operator.

### Changed

- `Pybind11` and `DLPack` requirements moved to git submodules.
- Border value parameters now use Saturate cast instead of Range cast when being converted to their internal types. This means passing in `1.0f` as a value for a `U8` image will result in a value of `1`, whereas previously it would have been converted to `255`.
- Updated border C-model.

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
