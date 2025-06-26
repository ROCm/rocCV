[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center"><img width="70%" src="docs/data/amd_roccv_logo.png" /></p>

rocCV is an efficient GPU-accelerated library for image pre and post-processing, powered by AMD's HIP platform.

---

>[!NOTE]
> **rocCV is currently only available as a source install.**
> 
> As rocCV is in early preview with ROCm 7.0.0, pre-built packages are not yet provided.
> Packages will be made available with the ROCm 7.1.x release.

## Prerequisites

* Linux distribution
  * Ubuntu - `22.04` / `24.04`
  * RHEL - `9`
  * SLES - `15 SP5`

* [ROCm-supported hardware](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
> [!IMPORTANT] 
> `gfx908` or higher GPU required

* Install ROCm `7.0.0` or later with [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html): Required usecase - rocm
> [!IMPORTANT]
> `sudo amdgpu-install --usecase=rocm`

* CMake Version `3.15` or later

  ```shell
  sudo apt install cmake
  ```

* AMD Clang++ Version `18.0.0` or later - installed with ROCm

* [HIP](https://github.com/ROCm/HIP)

  ```shell
  sudo apt install rocm-hip-runtime-dev
  ```

* [DLPack](https://pypi.org/project/dlpack/)

  ```shell
  sudo apt install libdlpack-dev   
  ```

* Python3 and Python3 PIP

  ```shell
  sudo apt install python3-dev python3-pip
  ```

* Python: [PyBind11](https://github.com/pybind/pybind11)

  ```shell
  pip3 install pybind11
  ```

* Python: Wheel
  
  ```shell
  pip3 install wheel
  ```

> [!IMPORTANT] 
> The following compiler features are required:
>   * C++20

### Samples and tests

To be able to build/run the samples and tests for both the rocCV C++ library and the python module, additional dependencies are required.

#### Python Samples/Tests

  ```shell
  sudo apt install python3-pytest python3-opencv python3-numpy
  ```

#### C++ Samples/Tests

```shell
sudo apt install 
```

>[!NOTE]
> All package installs are shown with the `apt` package manager. Use the appropriate package manager for your operating system.

### Prerequisites setup script

For your convenience, we provide the setup script,[roccv-setup.py](roccv-setup.py), which installs all required dependencies. Run this script only once.

```shell
python roccv-setup.py --rocm_path [ROCm Installation Path - optional (default:/opt/rocm)]
```

## Installation instructions

The installation process uses the following steps:

* [ROCm-supported hardware](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) install verification

* Install ROCm `7.0.0` or later with [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html) with `--usecase=rocm`

>[!IMPORTANT]
> Use **either** [package install](#package-install) **or** [source install](#source-install) as described below.

### Package install

Install rocCV runtime, development, and test packages.

* Runtime package - `roccv` only provides the dynamic libraries
* Development package - `roccv-dev`/`roccv-devel` provides the libraries, executables, header files, and samples
* Test package - `roccv-test` provides ctest to verify installation

#### `Ubuntu`

  ```shell
  sudo apt-get install roccv roccv-dev roccv-test
  ```

#### `CentOS`/`RedHat`

  ```shell
  sudo yum install roccv roccv-devel roccv-test
  ```

#### `SLES`

  ```shell
  sudo zypper install roccv roccv-devel roccv-test
  ```

>[!IMPORTANT]
> `RedHat`/`SLES` requires an additional manual installation of the `OpenCV` and `DLPack` dev packages.

>[!IMPORTANT]
> To use the rocCV python module, `PYTHONPATH` must be set appropriately to point to its install location in the ROCm directory:
> ```bash
> export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH`
> ```

### Source install

To build rocCV from source and install, follow the steps below:

* Clone rocCV source code

```shell
git clone https://github.com/ROCm/rocCV.git
```

>[!NOTE]
> To ensure all dependencies are installed, a python script for setup is included for your convenience.
>   ```shell
>   python3 roccv-setup.py
>   ```

#### Build the project using CMake

```shell
mkdir -p build
cd build
cmake ../
cmake --build . --parallel
```

#### Install the C++ and Python libraries

Both the C++ and Python libraries can be installed using the following make target:

```shell
sudo make install
```

#### Make package

```shell
sudo make package
```

#### Build Options

rocCV has a number of build options which may be set through cmake flags.
- `-D TESTS=ON`: Builds unit and integration tests.
- `-D SAMPLES=ON`: Builds sample applications.
- `-D BUILD_PYPACKAGE=ON:` Builds `rocpycv`, the Python wrapper for rocCV.

## Testing

After building, running `ctest` in the build directory will run tests for the C++ library.

## Verify installation

The installer copies:

* Libraries into `/opt/rocm/lib`
* Header files into `/opt/rocm/include/roccv`
* Samples folder into `/opt/rocm/share/roccv`
* Documents folder into `/opt/rocm/share/doc/roccv`

>[!NOTE]
> Ensure the required ROCm directories are added to your paths:
>   ```shell
>   export PATH=$PATH:/opt/rocm/bin
>   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
>   ```

### Verify rocCV PyBind with roccv-test package

Test package will install ctest module to test rocCV PyBindings. Follow below steps to test package install

```shell
mkdir roccv-pybind-test && cd roccv-pybind-test
cmake /opt/rocm/share/roccv/test/pybind
ctest -VV
```

>[!NOTE]
> Make sure all rocCV required libraries are in your PATH
> ```shell
> export PATH=$PATH:/opt/rocm/bin
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
> export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH
> ```