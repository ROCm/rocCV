.. meta::
  :description: rocCV building and installing
  :keywords: rocCV, ROCm, API, documentation


********************************************************************
Building and installing rocCV from source code
********************************************************************

ROCm must be `installed <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_ before building and installing rocCV from source code.

The rocCV source code is available from `https://github.com/ROCm/rocCV <https://github.com/ROCm/rocCV>`_. Use the rocCV version that corresponds to the installed version of ROCm.

You can use the |setup| setup script to install most :doc:`prerequisites <./rocCV-prerequisites>`:

.. code:: shell

  python3 rocCV-setup.py [-h] [--rocm_path ROCM_PATH; default /opt/rocm]

.. note::
  
  OpenCv and DLPack must be installed manually on SLES and RHEL.

To build and install rocCV, create the ``build`` directory under the ``rocCV`` root directory:

.. code:: shell
 
  mkdir build

Change directory to ``build`` and use the ``cmake`` command to generate a makefile: 

.. code:: shell
  
  cd build
  cmake ../

Build rocCV using ``cmake``:

.. code:: shell

  cmake --build . --parallel


Use ``make`` to install rocCV:

.. code:: shell
  
  sudo make install

You can optionally create deb, rpm, and gzip packages for distribution:

.. code:: shell

  sudo make package

Run ``ctest`` in the ``build`` directory to verify the installation.

.. |setup| replace:: ``rocCV-setup.py``
.. _setup: https://github.com/ROCm/rocCV/blob/develop/rocCV-setup.py
