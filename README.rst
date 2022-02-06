Interface to the image correlation spectroscopy library ICSx64.dll
==================================================================

ICSdll is a Python ctypes interface to the Image Correlation Spectroscopy
Dynamic Link Library (ICSx64.dll) developed at the Laboratory for Fluorescence
Dynamics (LFD) for the Globals for Images SimFCS software.

ICSx64.dll is implemented in C++ using the Intel(r) oneAPI Math Kernel Library
and OpenMP. It provides functions and classes for the analysis of fluorescence
time series data:

* 1D, 2D, and 3D auto- and cross-correlation
* Image pair correlation function (ipCF)
* Airy detector pair correlation function (apCF)
* Image mean square displacement (iMSD)
* Line spatio-temporal image correlation spectroscopy (lSTICS)
* Fit 1D pair correlation functions to the results of ipCF analysis
* Subtract immobile fractions
* Correct photo-bleaching
* 1D DFTs of image stack
* Richardson Lucy deconvolution (WIP)

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2022.2.6

Requirements
------------
* `CPython >= 3.8, 64-bit <https://www.python.org>`_
* `Numpy 1.19.5 <https://pypi.org/project/numpy/>`_
* `Intel(r) oneAPI Math Kernel Library <https://software.intel.com/mkl>`_
  (build)
* `Visual Studio 2019 C++ compiler <https://visualstudio.microsoft.com/>`_
  (build)

Revisions
---------
2022.2.6
    Rebuild package with oneAPI 2022.
2021.3.2
    Rebuild package.
2019.11.22
    Wrap yxt_dft functions.
    Upgrade to ICSx64.DLL version 2019.11.22.
2019.7.10
    Pass 22 tests.
    Wrap apcf, imsd, and lstics functions.
    Raise IcsError in case of DLL function errors.
    Use ICSx64.DLL version 2019.7.10.
2019.5.22
    Initial release based on ICSx64.DLL version 2018.7.18.

Notes
-----
ICSdll is currently developed, built, and tested on 64-bit Windows only.

The API is not stable and might change between revisions.

Refer to the C++ header file and source code for function signatures.

References
----------
1. `ipcf.ipynb <https://github.com/cgohlke/ipcf.ipynb>`_
   Pair correlation function analysis of fluorescence fluctuations in
   big image time series using Python.
2. `Globals for Images SimFCS <https://www.lfd.uci.edu/globals/>`_,
   Software for fluorescence image acquisition, analysis, and simulation.
3. `Globals for Airyscan <https://www.lfd.uci.edu/globals/>`_,
   Image correlation analysis for the ZEISS(tm) LSM 880 Airyscan detector.
