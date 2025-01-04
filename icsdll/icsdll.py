# icsdll.py

# Copyright (c) 2016-2025, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Interface to the image correlation spectroscopy library ICSx64.dll.

ICSdll is a Python ctypes interface to the Image Correlation Spectroscopy
Dynamic Link Library (ICSx64.dll) developed at the Laboratory for Fluorescence
Dynamics (LFD) for the Globals for Images SimFCS software.

ICSx64.dll is implemented in C++ using the Intel(r) oneAPI Math Kernel Library
and OpenMP. It provides functions and classes for the analysis of fluorescence
time series data:

- 1D, 2D, and 3D auto- and cross-correlation
- Image pair correlation function (ipCF)
- Airy detector pair correlation function (apCF)
- Image mean square displacement (iMSD)
- Line spatio-temporal image correlation spectroscopy (lSTICS)
- Fit 1D pair correlation functions to the results of ipCF analysis
- Subtract immobile fractions
- Correct photo-bleaching
- 1D DFTs of image stack

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2025.1.6

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.10.11, 3.11.9, 3.12.8, 3.13.1 64-bit
- `Numpy <https://pypi.org/project/numpy>`_ 2.2.1
- `Intel(r) oneAPI Math Kernel Library <https://software.intel.com/mkl>`_
  2025.0 (build)
- `Visual Studio 2022 C++ compiler <https://visualstudio.microsoft.com/>`_
  (build)

Revisions
---------

2025.1.6

- Support Python 3.13 and numpy 2.
- Rebuild package with oneAPI MKL 2025.0.

2024.1.6

- Rebuild package with oneAPI MKL 2024.0.0.

2023.1.6

- Rebuild package with oneAPI MKL 2022.2.1 and Visual Studio 2022.
- Update metadata.

2022.2.6

- Rebuild package with oneAPI 2022.

2021.3.2

- Rebuild package.

2019.11.22

- Wrap yxt_dft functions.
- Upgrade to ICSx64.DLL version 2019.11.22.

2019.7.10

- Pass 22 tests.
- Wrap apcf, imsd, and lstics functions.
- Raise IcsError in case of DLL function errors.
- Use ICSx64.DLL version 2019.7.10.

2019.5.22

- Initial release based on ICSx64.DLL version 2018.7.18.

Notes
-----

ICSdll was developed, built, and tested on 64-bit Windows only.

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

"""

from __future__ import annotations

__version__ = '2025.1.6'

__all__ = [
    '__version__',
    'API',
    'IcsError',
    'rfftnd',
    'xyt',
    'nlsp',
    'yxt_ipcf',
    'yxt_apcf',
    'yxt_imsd',
    'yxt_lstics',
    'yxt_subtract_immobile',
    'yxt_correct_bleaching',
    'yxt_dft',
    'zyx_deconv',
    'ipcf_nlsp_1dpcf',
    'radial',
    'circle',
    'logbins',
    'bins2times',
    'points2distances',
    'nextpow2',
    'numpy_correlate',
]

import ctypes
import math
import os
import warnings

import numpy


def API(dllname=None):
    """Return ctypes interface to functions of ICSx64 DLL."""
    from ctypes import (
        POINTER,
        c_char_p,
        c_double,
        c_float,
        c_int,
        c_int32,
        c_int64,
        c_size_t,
        c_ssize_t,
    )

    c_ssize_t_p = POINTER(c_ssize_t)
    c_double_p = POINTER(c_double)
    handle_t = c_size_t

    if dllname is None:
        dllname = os.path.join(os.path.dirname(__file__), 'ICSx64.dll')

    api = ctypes.CDLL(dllname)

    version = c_char_p.in_dll(api, 'ICS_VERSION').value
    assert version is not None
    api.VERSION = version.decode('ascii')

    api.MODE_FCS = 2
    api.MODE_CC = 4
    api.AXIS0 = 1
    api.AXIS1 = 8
    api.AXIS2 = 16

    api.FALSE = 0
    api.TRUE = 1

    api.OK = 0
    api.ERROR = -1
    api.VALUE_ERROR = -2
    api.MEMORY_ERROR = -3
    api.NOTIMPLEMENTD_ERROR = -4

    api.VALUE_ERROR1 = -201
    api.VALUE_ERROR2 = -202
    api.VALUE_ERROR3 = -203
    api.VALUE_ERROR4 = -204
    api.VALUE_ERROR5 = -205
    api.VALUE_ERROR6 = -206
    api.VALUE_ERROR7 = -207
    api.VALUE_ERROR8 = -208
    api.VALUE_ERROR9 = -209

    api.MODE_DEFAULT = 0
    api.MODE_TIME = 1  # do not center correlation results in axis 0 (time)
    api.MODE_FCS = 2  # normalize correlation results according to FCS
    api.MODE_CC = 4  # allocate second buffer for cross correlation

    api.AXIS0 = 1  # do not center correlation results in axis 0
    api.AXIS1 = 8  # do not center correlation results in axis 1
    api.AXIS2 = 16  # do not center correlation results in axis 2

    api.MASK_DEFAULT = 0
    api.MASK_ANY = 0  # any one value must be True
    api.MASK_FIRST = 1  # first mask value must be True
    api.MASK_CENTER = 2  # center mask value must be True
    api.MASK_ALL = 4  # all mask values must be True
    api.MASK_CLEAR = 32  # clear output if not calculated

    api.RADIUS = 1
    api.DIAMETER = 2

    api.NLSP_ND = 1
    api.NLSP_1DPCF = 100

    api.ICS_DECONV_DEFAULT = 1
    api.ICS_DECONV_RICHARDSON_LUCY = 1
    api.ICS_DECONV_WIENER = 2
    api.ICS_DECONV_NOPAD = 256

    api.DTYPES = {'l': 'i', 'i': 'i', 'h': 'h', 'H': 'H', 'd': 'd', 'f': 'f'}

    def ndpointer(dtype=None, ndim=None, shape=None, flags=None, null=False):
        """Return numpy.ctypes.ndpointer type that also accepts None/NULL."""
        cls = numpy.ctypeslib.ndpointer(dtype, ndim, shape, flags)
        if not null:
            return cls
        from_param_ = cls.from_param

        def from_param(cls, param):
            if param is None:
                return param
            return from_param_(param)

        cls.from_param = classmethod(from_param)  # type: ignore
        return cls

    def outer(a, b, skip=tuple()):
        return ((x, y) for x in a for y in b if (x, y) not in skip)

    # rfft#d_ functions
    for nd in (1, 2, 3):
        rfftnd = f'rfft{nd}d_'

        func = getattr(api, rfftnd + 'new')
        setattr(func, 'argtypes', [c_ssize_t, c_int])
        setattr(func, 'restype', handle_t)

        func = getattr(api, rfftnd + 'del')
        setattr(func, 'argtypes', [handle_t])
        setattr(func, 'restype', None)

        func = getattr(api, rfftnd + 'mode')
        setattr(func, 'argtypes', [c_int])
        setattr(func, 'restype', None)

        for i, o in outer('dfihH', 'df'):
            ai = ndpointer(dtype=i, ndim=nd)
            ao = ndpointer(dtype=o, ndim=nd)

            func = getattr(api, rfftnd + f'autocorrelate_{i}{o}')
            setattr(func, 'argtypes', [handle_t, ai, ao, c_ssize_t_p])
            setattr(func, 'restype', c_int)

            func = getattr(api, rfftnd + f'crosscorrelate_{i}{o}')
            setattr(
                func,
                'argtypes',
                [handle_t, ai, ai, ao, c_ssize_t_p, c_ssize_t_p],
            )
            setattr(func, 'restype', c_int)

    # nlsp class
    # TODO: test nlsp_ functions
    api.nlsp_new.restype = handle_t
    api.nlsp_new.argtypes = [c_int, c_ssize_t_p]

    api.nlsp_del.restype = None
    api.nlsp_del.argtypes = [handle_t]

    api.nlsp_get.restype = c_int
    api.nlsp_get.argtypes = [
        handle_t,
        c_ssize_t_p,
        c_ssize_t_p,
        c_double_p,
        c_double_p,
    ]

    api.nlsp_set.restype = c_int
    api.nlsp_set.argtypes = [
        handle_t,
        c_ssize_t,
        c_ssize_t,
        ndpointer(dtype='float64', shape=(6,), null=True),
        c_double,
        c_double,
    ]

    for dt in 'fd':
        func = getattr(api, f'nlsp_eval_{dt}')
        setattr(func, 'restype', c_int)
        setattr(func, 'argtypes', [handle_t, ndpointer(dtype=dt), c_ssize_t_p])

        func = getattr(api, f'nlsp_solve_{dt}')
        setattr(func, 'restype', c_int)
        setattr(
            func,
            'argtypes',
            [
                handle_t,
                ndpointer(dtype=dt),  # data
                c_ssize_t_p,  # strides
                ndpointer(dtype='float64', ndim=1),  # extra
                ndpointer(dtype='float64', ndim=1, null=True),  # guess
                ndpointer(dtype='float64', ndim=1, null=True),  # bounds
                ndpointer(dtype='float64', ndim=1, null=True),  # datasolution
            ],
        )

    # xyt class
    api.yxt_new.restype = handle_t
    api.yxt_new.argtypes = [c_ssize_t_p]

    api.yxt_del.restype = None
    api.yxt_del.argtypes = [handle_t]

    api.yxt_get_buffer.restype = c_double_p
    api.yxt_get_buffer.argtypes = [handle_t, c_ssize_t_p, c_ssize_t_p]

    for ti, to in outer('dfihH', 'df'):
        # yxt_ipcf_*
        func = getattr(api, f'yxt_ipcf_{ti}{to}')
        setattr(func, 'restype', c_int)
        setattr(
            func,
            'argtypes',
            [
                handle_t,
                ndpointer(dtype=ti, ndim=3, null=True),  # data
                ndpointer(dtype=ti, ndim=3, null=True),  # channel
                c_ssize_t_p,  # strides
                ndpointer(dtype=to, ndim=4),  # out
                c_ssize_t_p,  # outstrides
                ndpointer(dtype='intp', ndim=1),  # points
                c_ssize_t,  # npoints
                ndpointer(dtype='intp', ndim=1),  # bins
                c_ssize_t,  # nbins
                c_double,  # threshold
                c_double,  # filter
                c_int,  # nthreads
            ],
        )

        # yxt_apcf_*
        func = getattr(api, f'yxt_apcf_{ti}{to}')
        setattr(func, 'restype', c_int)
        setattr(
            func,
            'argtypes',
            [
                handle_t,
                ndpointer(dtype=ti, ndim=2, null=True),  # data
                c_ssize_t_p,  # strides
                ndpointer(dtype=to, ndim=3),  # out
                c_ssize_t_p,  # outstrides
                ndpointer(dtype='intp', ndim=1),  # bins
                c_ssize_t,  # nbins
                c_int,  # autocorr
                c_double,  # filter
                c_int,  # nthreads
            ],
        )

        # yxt_imsd_*
        func = getattr(api, f'yxt_imsd_{ti}{to}')
        setattr(func, 'restype', c_int)
        setattr(
            func,
            'argtypes',
            [
                handle_t,
                ndpointer(dtype=ti, ndim=3, null=True),  # data
                c_ssize_t_p,  # strides
                ndpointer(dtype=ti, ndim=3, null=True),  # data1
                c_ssize_t_p,  # strides1
                ndpointer(dtype='int32', ndim=2, null=True),  # mask
                c_ssize_t_p,  # maskstrides
                c_int,  # maskmode
                ndpointer(dtype=to, ndim=5),  # out
                c_ssize_t_p,  # outstrides
                c_ssize_t_p,  # blocks
                ndpointer(dtype='intp', ndim=1, null=True),  # bins
                c_ssize_t,  # nbins
                c_double,  # filter
                c_int,  # nthreads
            ],
        )

        # yxt_lstics_*
        func = getattr(api, f'yxt_lstics_{ti}{to}')
        setattr(func, 'restype', c_int)
        setattr(
            func,
            'argtypes',
            [
                handle_t,
                ndpointer(dtype=ti, ndim=3, null=True),  # data
                c_ssize_t_p,  # strides
                ndpointer(dtype=ti, ndim=3, null=True),  # data1
                c_ssize_t_p,  # strides1
                ndpointer(dtype='int32', ndim=2, null=True),  # mask
                c_ssize_t_p,  # maskstrides
                c_int,  # maskmode
                ndpointer(dtype=to, ndim=5),  # out
                c_ssize_t_p,  # outstrides
                ndpointer(dtype='intp', ndim=3),  # lines
                c_ssize_t_p,  # lineshape
                c_ssize_t_p,  # blocks
                ndpointer(dtype='intp', ndim=1),  # bins
                c_ssize_t,  # nbins
                c_double,  # filter
                c_int,  # nthreads
            ],
        )

    # ipcf_nlsp_1dpcf
    for dt in 'f':
        func = getattr(api, f'ipcf_nlsp_1dpcf_{dt}')
        setattr(func, 'restype', c_int)
        setattr(
            func,
            'argtypes',
            [
                ndpointer(dtype=dt, ndim=4),  # ipcf
                c_ssize_t_p,  # shape
                c_ssize_t_p,  # strides
                ndpointer(dtype=dt, ndim=1),  # times
                ndpointer(dtype=dt, ndim=1),  # distances
                ndpointer(dtype=dt, ndim=1),  # args
                ndpointer(dtype=dt, ndim=1, null=True),  # bounds
                ndpointer(dtype=dt, ndim=4, null=True),  # ix
                c_ssize_t_p,  # stridesx
                ndpointer(dtype=dt, ndim=4, null=True),  # ifx
                c_ssize_t_p,  # stridesfx
                ndpointer(dtype=dt, ndim=4, null=True),  # status
                c_ssize_t_p,  # stridestatus
                ndpointer(dtype=dt, ndim=1, null=True),  # settings
                c_int,  # average (bool)
                c_int,  # nthreads
            ],
        )

    # subtract_immobile, yxt_correct_bleaching
    for dt in 'ihH':
        ai = ndpointer(dtype=dt, ndim=3)

        func = getattr(api, f'yxt_subtract_immobile_{dt}')
        setattr(func, 'argtypes', [ai, c_ssize_t_p, c_ssize_t_p, c_int])
        setattr(func, 'restype', c_int)

        func = getattr(api, f'yxt_correct_bleaching_{dt}')
        setattr(
            func,
            'argtypes',
            [
                ai,
                c_ssize_t_p,
                c_ssize_t_p,
                ndpointer(dtype='double', ndim=2),
                c_ssize_t_p,
                c_double,
                c_int,
            ],
        )
        setattr(func, 'restype', c_int)

    # yxt_dft
    for ti, to in outer('dfihH', 'df'):
        if ti + to in 'dfd':
            continue
        func = getattr(api, f'yxt_dft_{ti}{to}')
        setattr(func, 'restype', c_int)
        setattr(
            func,
            'argtypes',
            [
                ndpointer(dtype=ti, ndim=3),  # data
                c_ssize_t_p,  # shape
                c_ssize_t_p,  # strides
                ndpointer(dtype=to, ndim=3),  # out
                c_ssize_t_p,  # outshape
                c_ssize_t_p,  # outstrides
                c_int,  # nthreads
            ],
        )

    # zyx_deconv
    try:
        for ti, to in (('f', 'f'), ('d', 'd'), ('H', 'f')):  # outer('fH', 'f')
            func = getattr(api, f'zyx_deconv_{ti}{to}')
            setattr(func, 'restype', c_int)
            setattr(
                func,
                'argtypes',
                [
                    ndpointer(dtype=ti, ndim=3),  # image
                    c_ssize_t_p,  # shape
                    c_ssize_t_p,  # strides
                    ndpointer(dtype=ti, ndim=3),  # psf
                    c_ssize_t_p,  # shape
                    c_ssize_t_p,  # strides
                    ndpointer(dtype=to, ndim=3),  # out
                    c_ssize_t_p,  # outshape
                    c_ssize_t_p,  # outstrides
                    c_int,  # niter
                    c_int,  # mode
                    c_int,  # nthreads
                ],
            )
    except AttributeError:
        pass

    # helper functions
    api.radial.restype = c_ssize_t
    api.radial.argtypes = [
        ndpointer(dtype='intp', ndim=3),
        c_ssize_t,
        c_ssize_t,
        ndpointer(dtype='float64', ndim=1),
        c_int,
    ]

    api.circle.restype = c_ssize_t
    api.circle.argtypes = [
        c_ssize_t,
        ndpointer(dtype='intp', ndim=2, null=True),
        c_ssize_t,
    ]

    api.logbins.restype = c_ssize_t
    api.logbins.argtypes = [
        c_ssize_t,
        c_ssize_t,
        ndpointer(dtype='intp', ndim=1),
    ]

    api.points2distances_f.restype = c_float
    api.points2distances_f.argtypes = [
        ndpointer(dtype='intp', ndim=2),
        c_ssize_t,
        c_float,
        ndpointer(dtype='float32', ndim=1),
    ]

    api.points2distances_d.restype = c_double
    api.points2distances_d.argtypes = [
        ndpointer(dtype='intp', ndim=1),
        c_ssize_t,
        c_double,
        ndpointer(dtype='float64', ndim=1),
    ]

    api.bins2times_f.restype = c_float
    api.bins2times_f.argtypes = [
        ndpointer(dtype='intp', ndim=1),
        c_ssize_t,
        c_float,
        ndpointer(dtype='float32', ndim=1),
    ]

    api.bins2times_d.restype = c_double
    api.bins2times_d.argtypes = [
        ndpointer(dtype='intp', ndim=1),
        c_ssize_t,
        c_double,
        ndpointer(dtype='float64', ndim=1),
    ]

    api.nextpow2_i.restype = c_int32
    api.nextpow2_i.argtypes = [c_int32]

    api.nextpow2_q.restype = c_int64
    api.nextpow2_q.argtypes = [c_int64]

    return api


API = API()


class IcsError(RuntimeError):
    """ICS DLL Exceptions."""

    def __init__(self, func, err):
        msg = {
            None: 'NULL',
            API.OK: 'OK',
            API.ERROR: 'ERROR',
            API.VALUE_ERROR: 'VALUE_ERROR',
            API.MEMORY_ERROR: 'MEMORY_ERROR',
            API.NOTIMPLEMENTD_ERROR: 'NOTIMPLEMENTD_ERROR',
            API.VALUE_ERROR1: 'VALUE_ERROR1',
            API.VALUE_ERROR2: 'VALUE_ERROR2',
            API.VALUE_ERROR3: 'VALUE_ERROR3',
            API.VALUE_ERROR4: 'VALUE_ERROR4',
            API.VALUE_ERROR5: 'VALUE_ERROR5',
            API.VALUE_ERROR6: 'VALUE_ERROR6',
            API.VALUE_ERROR7: 'VALUE_ERROR7',
            API.VALUE_ERROR8: 'VALUE_ERROR8',
            API.VALUE_ERROR9: 'VALUE_ERROR9',
        }.get(err, f'unknown error {err}')
        RuntimeError.__init__(self, f'{func.__name__} returned {msg}')


class rfftnd:
    """Wrapper for rfft#d_ functions."""

    def __init__(self, shape, mode):
        self._rfftnd = f'rfft{len(shape)}d_'
        func = getattr(API, self._rfftnd + 'new')
        self._handle = func(*shape, mode)
        if self._handle == 0:
            raise IcsError(func, None)
        self._mode = mode

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value
        func = getattr(API, self._rfftnd + 'mode')
        func(value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._handle:
            func = getattr(API, self._rfftnd + 'del')
            func(self._handle)
        self._handle = None

    def autocorrelate(self, a, out):
        if out is None:
            out = a
        func = getattr(
            API,
            self._rfftnd
            + 'autocorrelate_{}{}'.format(
                API.DTYPES[a.dtype.char], API.DTYPES[out.dtype.char]
            ),
        )
        status = func(self._handle, a, out, a.ctypes.strides)
        if status:
            raise IcsError(func, status)

    def crosscorrelate(self, a, b, out):
        if out is None:
            out = a
        func = getattr(
            API,
            self._rfftnd
            + 'crosscorrelate_{}{}'.format(
                API.DTYPES[a.dtype.char], API.DTYPES[out.dtype.char]
            ),
        )
        status = func(
            self._handle, a, b, out, a.ctypes.strides, b.ctypes.strides
        )
        if status:
            raise IcsError(func, status)


class nlsp:
    """Wrapper class for nlsp_ functions.

    Solver of non-linear least squares problem with linear boundary
    constraints using RCI and the Trust-Region algorithm.

    Only the "1D pair correlation function" diffusion model is currently
    supported.

    """

    def __init__(self, shape, model='1dpcf'):
        warnings.warn('the nlsp class is untested')
        model = {API.NLSP_1DPCF: API.NLSP_1DPCF, '1dpcf': API.NLSP_1DPCF}[
            model
        ]
        shape = (ctypes.c_ssize_t * len(shape))(*shape)
        self._handle = API.nlsp_new(model, shape)
        if self._handle == 0:
            raise IcsError(API.nlsp_new, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._handle:
            API.nlsp_del(self._handle)
        self._handle = None

    def solve(self, data, extra, guess=None, bounds=None, solution=None):
        """Solve nonlinear least squares problem.

        For the 1dpcf model, the 'extra' argument contains the xaxis values,
        the w2 parameter, and the squared distance.

        """
        func = getattr(API, f'nlsp_solve_{API.DTYPES[data.dtype.char]}')
        status = func(self._handle, data, data.strides)
        if status:
            raise IcsError(func, status)

    def eval(self, data):
        """Evaluate function using current solution vector."""
        func = getattr(API, f'nlsp_eval_{API.DTYPES[data.dtype.char]}')
        status = func(self._handle, data, data.strides)
        if status:
            raise IcsError(func, status)

    def get(self):
        """Return solution statuses.

        Return number of iterations, stop criterion, initial residual, and
        final residual.

        """
        it = ctypes.c_ssize_t()
        st_cr = ctypes.c_ssize_t()
        r1 = ctypes.c_double()
        r2 = ctypes.c_double()
        status = API.nlsp_get(self._handle, it, st_cr, r1, r2)
        if status:
            raise IcsError(API.nlsp_get, status)
        return it.value, st_cr.value, r1.value, r2.value

    def set(self, iter1=0, iter2=0, eps=None, eps_jac=0.0, rs=0.0):
        """Set solver parameters."""
        status = API.nlsp_set(self._handle, iter1, iter2, eps, eps_jac, rs)
        if status:
            raise IcsError(API.nlsp_set, status)


class xyt:
    """Wrapper class for xyt_ functions."""

    def __init__(self, shape):
        shape = (ctypes.c_ssize_t * len(shape))(*shape)
        self._handle = API.yxt_new(shape)
        if self._handle == 0:
            raise IcsError(API.yxt_new, None)
        # retrieve trunated shape
        API.yxt_get_buffer(self._handle, shape, None)
        self.shape = int(shape[0]), int(shape[1]), int(shape[2])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._handle:
            API.yxt_del(self._handle)
        self._handle = None

    def ipcf(
        self,
        data,
        points,
        bins,
        channel=None,
        out=None,
        threshold=0.0,
        smooth=0.0,
        nthreads=0,
        verbose=False,
    ):
        """Image pair correlation function."""
        if (
            data.ndim != 3
            or data.shape[0] != self.shape[0]
            or data.shape[1] != self.shape[1]
        ):
            raise ValueError('invalid data shape')

        if channel is not None and (
            channel.strides != data.strides or channel.shape != data.shape
        ):
            raise ValueError('invalid channel shape')

        npoints = points.shape[0]
        nbins = len(bins)

        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        outshape = (
            data.shape[0] - y1 + x0,
            data.shape[1] - x1 + x0,
            npoints,
            nbins,
        )
        if out is None:
            out = numpy.zeros(shape=outshape, dtype='float32')
        if out.ndim != 4 or out.size < product(outshape):
            raise ValueError()

        func = getattr(
            API,
            'yxt_ipcf_{}{}'.format(
                API.DTYPES[data.dtype.char], API.DTYPES[out.dtype.char]
            ),
        )

        if verbose:
            print('data shape =', data.shape)
            print('data strides =', data.strides)
            print('out shape =', out.shape)
            print('out strides =', out.strides)

        status = func(
            self._handle,
            data,
            channel,
            data.ctypes.strides,
            out,
            out.ctypes.strides,
            points.flatten(),
            npoints,
            bins,
            nbins,
            threshold,
            smooth,
            nthreads,
        )
        if status != 0:
            raise IcsError(func, status)

        return out

    def apcf(
        self,
        data,
        bins,
        out=None,
        autocorr=True,
        smooth=0.0,
        nthreads=0,
        verbose=False,
    ):
        """Airy detector pair correlation."""
        if (
            data.ndim != 2
            or data.shape[0] != self.shape[1]
            or self.shape[0] != 1
        ):
            raise ValueError('invalid data shape')

        nbins = len(bins)
        if autocorr:
            outshape = data.shape[0], data.shape[0], nbins
        else:
            outshape = data.shape[0], data.shape[0] - 1, nbins
        if out is None:
            out = numpy.zeros(shape=outshape, dtype='float32')
        if out.ndim != 3 or out.size < product(outshape):
            raise ValueError()

        func = getattr(
            API,
            'yxt_apcf_{}{}'.format(
                API.DTYPES[data.dtype.char], API.DTYPES[out.dtype.char]
            ),
        )

        if verbose:
            print('data shape =', data.shape)
            print('data strides =', data.strides)
            print('out shape =', out.shape)
            print('out strides =', out.strides)

        status = func(
            self._handle,
            data,
            data.ctypes.strides,
            out,
            out.ctypes.strides,
            bins,
            nbins,
            autocorr,
            smooth,
            nthreads,
        )
        if status != 0:
            raise IcsError(func, status)

        return out

    def imsd(
        self,
        data,
        block,
        bins,
        channel=None,
        out=None,
        mask=None,
        mask_mode=None,
        smooth=0.0,
        nthreads=0,
        verbose=False,
    ):
        """Image mean square displacement."""
        if (
            data.ndim != 3
            or data.shape[0] != self.shape[0]
            or data.shape[1] != self.shape[1]
        ):
            raise ValueError('invalid data shape')

        if channel is not None:
            if channel.shape != data.shape:
                raise ValueError('invalid channel shape')
            channel_strides = channel.strides
        else:
            channel_strides = None

        if mask is not None:
            if mask.shape[:2] != data.shape[:2]:
                raise ValueError('invalid mask shape')
            mask_strides = mask.strides
        else:
            mask_strides = None

        if mask_mode is None:
            mask_mode = API.MASK_ANY | API.MASK_CLEAR

        if len(block) != 4:
            raise ValueError()

        try:
            nbins = int(bins)
            bins = None
        except Exception:
            nbins = len(bins)

        outshape = (
            (data.shape[0] - block[0]) // block[2] + 1,
            (data.shape[1] - block[1]) // block[3] + 1,
            block[0],
            block[1],
            nbins,
        )
        if out is None:
            out = numpy.zeros(shape=outshape, dtype='float32')
        elif out.ndim != 5 or out.size < product(outshape):
            raise ValueError('invalid out shape')

        block = (ctypes.c_ssize_t * 4)(*block)

        func = getattr(
            API,
            'yxt_imsd_{}{}'.format(
                API.DTYPES[data.dtype.char], API.DTYPES[out.dtype.char]
            ),
        )

        if verbose:
            print('data shape =', data.shape)
            print('data strides =', data.strides)
            print('out shape =', out.shape)
            print('out strides =', out.strides)

        status = func(
            self._handle,
            data,
            data.ctypes.strides,
            channel,
            channel_strides,
            mask,
            mask_strides,
            mask_mode,
            out,
            out.ctypes.strides,
            block,
            bins,
            nbins,
            smooth,
            nthreads,
        )
        if status != 0:
            raise IcsError(func, status)

        return out

    def lstics(
        self,
        data,
        block,
        lines,
        bins,
        channel=None,
        out=None,
        mask=None,
        mask_mode=None,
        smooth=0.0,
        nthreads=0,
        verbose=False,
    ):
        """Line spatio temporal image correlation spectroscopy."""
        if (
            data.ndim != 3
            or data.shape[0] != self.shape[0]
            or data.shape[1] != self.shape[1]
        ):
            raise ValueError('invalid data shape')

        if channel is not None:
            if channel.shape != data.shape:
                raise ValueError('invalid channel shape')
            channel_strides = channel.strides
        else:
            channel_strides = None

        if mask is not None:
            if mask.shape[:2] != data.shape[:2]:
                raise ValueError('invalid mask shape')
            mask_strides = mask.strides
        else:
            mask_strides = None

        if mask_mode is None:
            mask_mode = API.MASK_ANY | API.MASK_CLEAR

        if len(block) != 4:
            raise ValueError()

        nbins = len(bins)

        outshape = (
            (data.shape[0] - block[0]) // block[2] + 1,
            (data.shape[1] - block[1]) // block[3] + 1,
            lines.shape[0],
            lines.shape[1],
            nbins,
        )
        if out is None:
            out = numpy.zeros(shape=outshape, dtype='float32')
        elif out.ndim != 5 or out.size < product(outshape):
            raise ValueError('invalid out shape')

        block = (ctypes.c_ssize_t * 4)(*block)
        lines_shape = (ctypes.c_ssize_t * 3)(*lines.shape)

        func = getattr(
            API,
            'yxt_lstics_{}{}'.format(
                API.DTYPES[data.dtype.char], API.DTYPES[out.dtype.char]
            ),
        )

        if verbose:
            print('data shape =', data.shape)
            print('data strides =', data.strides)
            print('out shape =', out.shape)
            print('out strides =', out.strides)
            print('lines shape =', lines.shape)

        status = func(
            self._handle,
            data,
            data.ctypes.strides,
            channel,
            channel_strides,
            mask,
            mask_strides,
            mask_mode,
            out,
            out.ctypes.strides,
            lines,
            lines_shape,
            block,
            bins,
            nbins,
            smooth,
            nthreads,
        )
        if status != 0:
            raise IcsError(func, status)

        return out


def yxt_ipcf(data, radius=4, nbins=32, smooth=0.7, threshold=0.0, nthreads=0):
    """Simplified image pair correlation function."""
    # make time axis last dimension
    # if data.shape[0] > 8 * data.shape[2]:
    #     data = numpy.moveaxis(data, 0, -1)
    height, width, ntimes = data.shape

    # truncate time axis to power of two
    ntimes = 2 ** int(math.log(ntimes, 2))
    data = data[..., :ntimes]

    bins = logbins(ntimes // 2, nbins)
    # nbins = bins.shape[0]

    points = circle(radius)
    # npoints = points.shape[0]

    with xyt(data.shape) as handle:
        result = handle.ipcf(
            data,
            points,
            bins,
            smooth=smooth,
            threshold=threshold,
            nthreads=nthreads,
        )
    return result, bins, points


def yxt_apcf(data, nbins=256, autocorr=True, smooth=0.7, nthreads=0):
    """Simplified airy detector pair correlation."""
    width, ntimes = data.shape

    # truncate time axis to power of two
    ntimes = 2 ** int(math.log(ntimes, 2))
    data = data[..., :ntimes]

    bins = logbins(ntimes // 2, nbins)
    # nbins = bins.shape[0]

    with xyt((1, width, ntimes)) as handle:
        result = handle.apcf(
            data, bins, autocorr=autocorr, smooth=smooth, nthreads=nthreads
        )
    return result, bins


def yxt_imsd(data, block=(32, 32, 4, 4), bins=16, smooth=0.0, nthreads=0):
    """Simplified image mean square displacement."""
    with xyt(data.shape) as handle:
        result = handle.imsd(
            data, block, bins, smooth=smooth, nthreads=nthreads
        )
    return result


def yxt_lstics(
    data,
    block=(16, 16, 1, 1),
    nlines=16,
    linelength=8,
    nbins=16,
    smooth=0.0,
    nthreads=0,
):
    """Simplified line spatio temporal image correlation spectroscopy."""
    height, width, ntimes = data.shape

    # truncate time axis to power of two
    ntimes = 2 ** int(math.log(ntimes, 2))
    data = data[..., :ntimes]

    bins = logbins(ntimes // 2, nbins)
    # nbins = bins.shape[0]

    lines = radial(nlines, linelength)

    with xyt(data.shape) as handle:
        result = handle.lstics(
            data, block, lines, bins, smooth=smooth, nthreads=nthreads
        )
    return result, bins, lines


def yxt_subtract_immobile(a, nthreads=0):
    """Wrapper for yxt_subtract_immobile_ functions."""
    if a.ndim != 3:
        raise ValueError('input must be three dimensional')
    func = getattr(API, f'yxt_subtract_immobile_{API.DTYPES[a.dtype.char]}')
    status = func(a, a.ctypes.shape, a.ctypes.strides, nthreads)
    if status:
        raise IcsError(func, status)


def yxt_correct_bleaching(a, smooth, nthreads=0):
    """Wrapper for yxt_correct_bleaching_ functions."""
    if a.ndim != 3:
        raise ValueError('input must be three dimensional')
    func = getattr(API, f'yxt_correct_bleaching_{API.DTYPES[a.dtype.char]}')
    out = numpy.empty(a.shape[:2], 'float64')
    status = func(
        a,
        a.ctypes.shape,
        a.ctypes.strides,
        out,
        out.ctypes.strides,
        smooth,
        nthreads,
    )
    if status:
        raise IcsError(func, status)
    return out


def yxt_dft(data, samples=5, nthreads=0, asimages=True):
    """Wrapper for yxt_dft_ functions."""
    if data.ndim != 3:
        raise ValueError('input must be three dimensional')

    shape = data.shape
    if asimages:
        outshape = samples, shape[0], shape[1]
    else:
        outshape = shape[0], shape[1], samples
    result = numpy.zeros(outshape, dtype='float32')
    out = numpy.moveaxis(result, 0, -1) if asimages else result

    func = getattr(API, f'yxt_dft_{API.DTYPES[data.dtype.char]}f')

    status = func(
        data,
        data.ctypes.shape,
        data.ctypes.strides,
        out,
        out.ctypes.shape,
        out.ctypes.strides,
        nthreads,
    )
    if status:
        raise IcsError(func, status)
    return result


def zyx_deconv(image, psf, iterations=10, dtype='f4', mode=0, nthreads=0):
    """Wrapper for zyx_deconv_ functions."""
    if image.ndim != 3 or psf.ndim != 3:
        raise ValueError('input must be three dimensional')

    out = numpy.zeros(image.shape, dtype=dtype)

    func = getattr(
        API,
        'zyx_deconv_{}{}'.format(
            API.DTYPES[image.dtype.char], API.DTYPES[out.dtype.char]
        ),
    )

    status = func(
        image,
        image.ctypes.shape,
        image.ctypes.strides,
        psf,
        psf.ctypes.shape,
        psf.ctypes.strides,
        out,
        out.ctypes.shape,
        out.ctypes.strides,
        iterations,
        mode,
        nthreads,
    )
    if status:
        raise IcsError(func, status)
    return out


def ipcf_nlsp_1dpcf(
    ipcf,
    times,
    distances,
    args,
    bounds=None,
    ix=None,
    ifx=None,
    status=None,
    settings=None,
    average=False,
    nthreads=0,
):
    """Fit diffusion to results of ipcf analysis."""
    if ipcf.ndim != 4:
        raise ValueError()

    dtype = ipcf.dtype

    func = getattr(API, f'ipcf_nlsp_1dpcf_{API.DTYPES[dtype.char]}')

    args = numpy.array(args, dtype=dtype)
    if bounds is not None:
        bounds = numpy.array(bounds, dtype=dtype)

    if settings is not None:
        settings = numpy.array(settings, dtype=dtype)

    npoints = 1 if average else ipcf.shape[2]
    ifxshape = ipcf.shape[0], ipcf.shape[1], npoints, ipcf.shape[3]
    ixshape = ipcf.shape[0], ipcf.shape[1], npoints, 2
    statusshape = ipcf.shape[0], ipcf.shape[1], npoints, 4

    if ix is None:
        ix = numpy.zeros(shape=ixshape, dtype=dtype)
    elif ix is False:
        ix = None
    elif ix.ndim != 4 or ix.size < product(ixshape):
        raise ValueError()

    if ifx is None:
        ifx = numpy.zeros(shape=ifxshape, dtype=dtype)
    elif ifx is False:
        ifx = None
    elif ifx.ndim != len(ifxshape) or ifx.size < product(ifxshape):
        raise ValueError()

    if status is None:
        status = numpy.zeros(shape=statusshape, dtype=dtype)
    elif status is False:
        status = None
    elif status.ndim != len(statusshape) or status.size < product(statusshape):
        raise ValueError()

    distances = numpy.array(distances, dtype=dtype).reshape(-1)
    distances = distances * distances

    ret = func(
        ipcf,
        ipcf.ctypes.shape,
        ipcf.ctypes.strides,
        times,
        distances,
        args,
        bounds,
        ix,
        None if ix is None else ix.ctypes.strides,
        ifx,
        None if ifx is None else ifx.ctypes.strides,
        status,
        None if status is None else status.ctypes.strides,
        settings,
        average,
        nthreads,
    )
    if ret != 0:
        raise IcsError(func, ret)

    return ix, ifx, status


def points2distances(points, pixelsize, dtype='float32'):
    """Return distances from points on integer grid."""
    func = getattr(
        API, f'points2distances_{API.DTYPES[numpy.dtype(dtype).char]}'
    )
    distances = numpy.zeros(points.shape[0], dtype)
    ret = func(points, points.shape[0], pixelsize, distances)
    if ret == -1.0:
        raise IcsError(func, ret)
    return distances


def radial(nlines, length, mode='radius', offset=None):
    """Return integer coordinates of line segments through center of circle."""
    if nlines < 1 or length < 1:
        raise ValueError('invalid nbins or length')
    points = numpy.zeros((nlines, length, 2), dtype='intp')
    mode = {
        API.RADIUS: API.RADIUS,
        API.DIAMETER: API.DIAMETER,
        'radius': API.RADIUS,
        'diameter': API.DIAMETER,
    }[mode]
    if offset is None:
        offset = numpy.zeros(2, 'double')
    else:
        offset = numpy.array(offset).astype('double')
        if offset.size < 2:
            raise ValueError('invalid offset')
    status = API.radial(points, nlines, length, offset, mode)
    if status <= 0:
        raise IcsError(API.radial, status)
    return points


def circle(radius, npoints=0):
    """Return x, y integer coordinates of circle of radius."""
    if npoints <= 0:
        npoints = API.circle(radius, None, 0)
    points = numpy.zeros((npoints, 2), 'intp')
    if npoints == 0:
        return points  # TODO: passing empty array to API causes random crash
    ret = API.circle(radius, points, npoints)
    if ret < 0:
        raise IcsError(API.circle, ret)
    return points


def logbins(size, nbins):
    """Return exponentially increasing integers up to size."""
    if nbins < 1:
        raise ValueError('invalid nbins')
    bins = numpy.zeros(nbins, 'intp')
    nbins = API.logbins(size, nbins, bins)
    if nbins < 0:
        raise IcsError(API.logbins, nbins)
    if nbins != bins.size:
        bins = bins[:nbins].copy()
    return bins


def bins2times(bins, frametime, dtype='float32'):
    """Return times from bins."""
    func = getattr(API, f'bins2times_{API.DTYPES[numpy.dtype(dtype).char]}')
    times = numpy.zeros(bins.shape[0], dtype)
    status = func(bins, bins.shape[0], frametime, times)
    if status < 0:
        raise IcsError(func, status)
    return times


def nextpow2(n):
    """Return next power of 2."""
    return API.nextpow2_q(n)


def numpy_correlate(a, b, mode=0, axes=None):
    """Return cross-correlation using numpy.fft.fftn"""
    a = numpy.fft.fftn(a)
    b = numpy.fft.fftn(b)
    index = (0,) * a.ndim
    scale = a[index].real * b[index].real / a.size
    a *= b.conj()
    a = numpy.fft.ifftn(a).real
    if axes or axes is None:
        a = numpy.fft.fftshift(a, axes)
    if mode & API.MODE_FCS:
        a /= scale
        a -= 1.0
    return a


def numpy_correlate1d(a, b, mode=0, axes=None):
    """Return cross-correlation using numpy.fft.rfft"""
    a = numpy.fft.rfft(a)
    b = numpy.fft.rfft(b)
    scale = a[0].real * b[0].real / a.size
    a *= b.conj()
    a = numpy.fft.irfft(a).real
    if axes or axes is None:
        a = numpy.fft.fftshift(a, axes)
    if mode & API.MODE_FCS:
        a /= scale
        a -= 1.0
    return a


def product(iterable):
    """Return product of sequence of numbers."""
    prod = 1
    for i in iterable:
        prod *= i
    return prod


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="attr-defined"
