.. meta::
  :description: Installing rocCV with the package installer 
  :keywords: rocCV, ROCm, package install

*********************************************
Installing rocCV with the package installer
*********************************************

Three rocCV packages are available:

| ``roccv``: The rocCV runtime package. This is the basic rocCV package that only provides dynamic libraries. It must always be installed.
| ``roccv-dev``: The rocCV development package. This package installs a full suite of libraries, header files, and samples. This package needs to be installed to use samples.
* ``roccv-test``: A test package that provides CTests to verify the installation. 

All the required prerequisites except for OpenCV and DLPack on RHEL and SLES are installed with the rocCV packages. OpenCV and DLPack must be installed manually on RHEL and SLES.


Basic installation
========================================

Use the following commands to install only the rocCV runtime package:

.. tab-set::
 
  .. tab-item:: Ubuntu

    .. code:: shell

      sudo apt-get install roccv

  .. tab-item:: RHEL

    .. code:: shell
    
      sudo yum install roccv

  .. tab-item:: SLES

    .. code:: shell

      sudo zypper install roccv

Complete installation
========================================

Use the following commands to install ``roccv``, ``roccv-dev``, and ``roccv-test``:

... tab-set::
 
  .. tab-item:: Ubuntu

    .. code:: shell

      sudo apt-get install roccv roccv-dev roccv-test

  .. tab-item:: RHEL

    .. code:: shell

      sudo yum install roccv roccv-devel roccv-test

  .. tab-item:: SLES

    .. code:: shell

      sudo zypper install roccv roccv-devel roccv-test


Set ``PYTHONPATH`` to use the rocCV Python module:

.. code:: shell
  
  export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH

Add rocCV to your ``PATH`` and ``LD_LIBRARY_PATH``:

.. code:: shell

  export PATH=$PATH:/opt/rocm/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib

