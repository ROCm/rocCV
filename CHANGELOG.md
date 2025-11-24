# Changelog for rocCV

The full documentation for rocCV is available at [https://rocm.docs.amd.com/projects/rocCV/en/latest/index.html](https://rocm.docs.amd.com/projects/rocCV/en/latest/index.html)

## rocCV 0.4.0 for ROCm 7.2.0

### Added
- Reflect 101 border mode.

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
