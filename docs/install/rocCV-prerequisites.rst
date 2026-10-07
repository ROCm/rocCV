.. meta::
  :description: rocCV install prerequisites
  :keywords: rocCV, ROCm, install, prerequisites

******************************************
rocCV prerequisites
******************************************

rocCV requires `ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_ running on `accelerators based on the CDNA architecture <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_.

rocCV can be installed on the following Linux environments:
  
* Ubuntu 22.04 or 24.04
* RedHat 9
* SLES 15-SP5

The following prerequisites are required when :doc:`building rocCV from source <./rocCV-build-and-install>`:

* `HIP runtime <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_ 
* `PyBind11 <https://github.com/pybind/pybind11/releases/tag/v2.11.1>`_ version 2.10.4
* `RapidJSON <https://github.com/Tencent/rapidjson>`_
* `CMake <https://cmake.org/>`_ version 3.15
* `DLPack <https://pypi.org/project/dlpack/>`_
* Python3, Python3 pip, and  Python3 wheel
* Python-opencv, Pytest, and numpy for testing

rocCV requires C++20.
