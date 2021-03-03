# test_icsdll.py

# Copyright (c) 2016-2021, Christoph Gohlke
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

"""Unit tests for the image correlation spectroscopy library ICSx64.dll.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2021.3.2

"""

import sys
import os
import math
import time
from contextlib import contextmanager

import pytest
import numpy

from numpy.testing import assert_array_equal, assert_allclose

HERE = os.path.dirname(__file__) + '/'


def test_versions(version='2021.3.2', apiversion='2021.3.2'):
    """Test versions match."""
    assert icsdll.__version__ == version
    assert API.VERSION == apiversion


def test_logbins():
    """Test logbins function."""
    assert_array_equal(
        logbins(32, 32),
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            16,
            18,
            20,
            22,
            25,
            28,
            32,
        ],
    )
    assert_array_equal(
        logbins(8192, 32),
        [
            1,
            2,
            3,
            4,
            5,
            7,
            10,
            13,
            18,
            24,
            32,
            43,
            58,
            78,
            104,
            139,
            187,
            250,
            334,
            447,
            598,
            800,
            1070,
            1432,
            1915,
            2561,
            3425,
            4580,
            6125,
            8192,
        ],
    )
    with pytest.raises(icsdll.IcsError):
        logbins(2, 2)


def test_circle():
    """Test circle function."""
    assert_array_equal(circle(0), numpy.empty((0, 2), dtype='int64'))
    assert_array_equal(circle(1), [[1, 0], [0, 1], [-1, 0], [0, -1]])
    assert_array_equal(
        circle(6),
        [
            [6, 0],
            [6, 1],
            [6, 2],
            [5, 3],
            [4, 4],
            [3, 5],
            [2, 6],
            [1, 6],
            [0, 6],
            [-1, 6],
            [-2, 6],
            [-3, 5],
            [-4, 4],
            [-5, 3],
            [-6, 2],
            [-6, 1],
            [-6, 0],
            [-6, -1],
            [-6, -2],
            [-5, -3],
            [-4, -4],
            [-3, -5],
            [-2, -6],
            [-1, -6],
            [0, -6],
            [1, -6],
            [2, -6],
            [3, -5],
            [4, -4],
            [5, -3],
            [6, -2],
            [6, -1],
        ],
    )


def test_radial():
    """Test radial function."""
    assert_array_equal(
        radial(8, 4),
        numpy.array(
            [
                0,
                0,
                0,
                1,
                0,
                2,
                0,
                3,
                0,
                0,
                1,
                1,
                1,
                1,
                2,
                2,
                0,
                0,
                1,
                0,
                2,
                0,
                3,
                0,
                0,
                0,
                1,
                -1,
                1,
                -1,
                2,
                -2,
                0,
                0,
                0,
                -1,
                0,
                -2,
                0,
                -3,
                0,
                0,
                -1,
                -1,
                -1,
                -1,
                -2,
                -2,
                0,
                0,
                -1,
                0,
                -2,
                0,
                -3,
                0,
                0,
                0,
                -1,
                1,
                -1,
                1,
                -2,
                2,
            ],
            dtype='intp',
        ).reshape((8, 4, 2))[:, :, ::-1],
    )


def test_rfftnd_autocorrelate444():
    """Test rfft3d autocorrelate with Enrico's data."""
    test = numpy.zeros((4, 4, 4))
    for k in range(4):
        for j in range(4):
            for i in range(4):
                test[k, j, i] = k * 4 * 4 + j * 4 + i

    a = test.copy()
    t = numpy.zeros((17, 29, 51), dtype='float64')
    t -= 1.0
    t[3:11:2, 19:15:-1, 37:49:3] = a
    b = t[3:11:2, 19:15:-1, 37:49:3]  # a view

    test_out = numpy.array(
        [
            [
                [0.30964979, 0.31015369, 0.31166541, 0.31015369],
                [0.31771227, 0.31821618, 0.31972789, 0.31821618],
                [0.34189972, 0.34240363, 0.34391534, 0.34240363],
                [0.31771227, 0.31821618, 0.31972789, 0.31821618],
            ],
            [
                [-0.07734946, -0.07684555, -0.07533384, -0.07684555],
                [-0.06928697, -0.06878307, -0.06727135, -0.06878307],
                [-0.04509952, -0.04459562, -0.0430839, -0.04459562],
                [-0.06928697, -0.06878307, -0.06727135, -0.06878307],
            ],
            [
                [-0.20634921, -0.2058453, -0.20433359, -0.2058453],
                [-0.19828672, -0.19778282, -0.1962711, -0.19778282],
                [-0.17409927, -0.17359536, -0.17208365, -0.17359536],
                [-0.19828672, -0.19778282, -0.1962711, -0.19778282],
            ],
            [
                [-0.07734946, -0.07684555, -0.07533384, -0.07684555],
                [-0.06928697, -0.06878307, -0.06727135, -0.06878307],
                [-0.04509952, -0.04459562, -0.0430839, -0.04459562],
                [-0.06928697, -0.06878307, -0.06727135, -0.06878307],
            ],
        ]
    )

    with rfftnd((4, 4, 4), mode=3) as fft:
        fft.autocorrelate(a, None)
        assert numpy.allclose(a, test_out)

        out = numpy.empty_like(b)
        fft.autocorrelate(b, out)
        assert numpy.allclose(out, test_out)

        a = test.copy()
        fft.crosscorrelate(a, a.copy(), None)
        assert numpy.allclose(a, test_out)

        out = numpy.empty_like(b)
        fft.crosscorrelate(b, b.copy(), out)
        assert numpy.allclose(out, test_out)


def test_rfftnd_autocorrelate1d():
    """Test rfft1d autocorrelate function."""
    shape = (256,)
    test = numpy.random.rand(*shape)
    result = numpy.empty_like(test)

    # default scaling and shift
    mode = 0
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.autocorrelate(a, result)
    a = test.copy()
    c = numpy_correlate(a, a, mode=mode)
    assert numpy.allclose(c, result)

    # fcs scaling
    mode = API.MODE_FCS
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.autocorrelate(a, result)
    a = test.copy()
    c = numpy_correlate(a, a, mode=mode)
    assert numpy.allclose(c, result)

    # do not shift first axis
    mode = API.AXIS0
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.autocorrelate(a, result)
    a = test.copy()
    c = numpy_correlate(a, a, mode=0, axes=False)
    assert numpy.allclose(c, result)

    # inplace
    mode = 0
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.autocorrelate(a, None)
    result = a
    a = test.copy()
    c = numpy_correlate(a, a, mode=mode)
    assert numpy.allclose(c, result)


def test_rfftnd_autocorrelate2d():
    """Test rfft2d autocorrelate function."""
    shape = 256, 128
    test = numpy.random.rand(*shape)
    result = numpy.empty_like(test)

    # default scaling and shift
    mode = 0
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.autocorrelate(a, result)
    a = test.copy()
    c = numpy_correlate(a, a, mode=mode)
    assert numpy.allclose(c, result)

    # fcs scaling
    mode = API.MODE_FCS
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.autocorrelate(a, result)
    a = test.copy()
    c = numpy_correlate(a, a, mode=mode)
    assert numpy.allclose(c, result)

    # do not shift first axis
    mode = API.AXIS0
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.autocorrelate(a, result)
    a = test.copy()
    c = numpy_correlate(a, a, mode=0, axes=(1,))
    assert numpy.allclose(c, result)

    # do not shift last axis
    mode = API.AXIS1
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.autocorrelate(a, result)
    a = test.copy()
    c = numpy_correlate(a, a, mode=0, axes=(0,))
    assert numpy.allclose(c, result)

    # do not shift any axis
    mode = API.AXIS0 | API.AXIS1
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.autocorrelate(a, result)
    a = test.copy()
    c = numpy_correlate(a, a, mode=0, axes=False)
    assert numpy.allclose(c, result)

    # inplace
    mode = 0
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.autocorrelate(a, None)
    result = a
    a = test.copy()
    c = numpy_correlate(a, a, mode=mode)
    assert numpy.allclose(c, result)


def test_rfftnd_autocorrelate3d():
    """Test rfft3d autocorrelate function."""
    shape = 64, 256, 128
    test = numpy.random.rand(*shape)
    result = numpy.empty_like(test)

    # default scaling and shift
    mode = 0
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.autocorrelate(a, result)
    a = test.copy()
    c = numpy_correlate(a, a, mode=mode)
    assert numpy.allclose(c, result)

    # fcs scaling
    mode = API.MODE_FCS
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.autocorrelate(a, result)
    a = test.copy()
    c = numpy_correlate(a, a, mode=mode)
    assert numpy.allclose(c, result)

    # do not shift first axis
    mode = API.AXIS0
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.autocorrelate(a, result)
    a = test.copy()
    c = numpy_correlate(a, a, mode=0, axes=(1, 2))
    assert numpy.allclose(c, result)

    # do not shift last axis
    mode = API.AXIS2
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.autocorrelate(a, result)
    a = test.copy()
    c = numpy_correlate(a, a, mode=0, axes=(0, 1))
    assert numpy.allclose(c, result)

    # do not shift any axis
    mode = API.AXIS0 | API.AXIS1 | API.AXIS2
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.autocorrelate(a, result)
    a = test.copy()
    c = numpy_correlate(a, a, mode=0, axes=False)
    assert numpy.allclose(c, result)

    # inplace
    mode = 0
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.autocorrelate(a, None)
    result = a
    a = test.copy()
    c = numpy_correlate(a, a, mode=mode)
    assert numpy.allclose(c, result)


def test_rfftnd_crosscorrelate1d():
    """Test rfft1d crosscorrelate function."""
    shape = (256,)
    test = numpy.random.rand(*shape)
    result = numpy.empty_like(test)
    b = numpy.random.rand(*shape)

    # default scaling and shift
    mode = 0 | API.MODE_CC
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, b, result)
    a = test.copy()
    c = numpy_correlate(a, b, mode=mode)
    assert numpy.allclose(c, result)

    # fcs scaling
    mode = API.MODE_FCS | API.MODE_CC
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, b, result)
    a = test.copy()
    c = numpy_correlate(a, b, mode=mode)
    assert numpy.allclose(c, result)

    # do not shift first axis
    mode = API.AXIS0 | API.MODE_CC
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, b, result)
    a = test.copy()
    c = numpy_correlate(a, b, mode=0, axes=False)
    assert numpy.allclose(c, result)

    # inplace
    mode = 0 | API.MODE_CC
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, b, None)
    result = a
    a = test.copy()
    c = numpy_correlate(a, b, mode=mode)
    assert numpy.allclose(c, result)


def test_rfftnd_crosscorrelate2d():
    """Test rfft2d crosscorrelate function."""
    shape = 256, 128
    test = numpy.random.rand(*shape)
    result = numpy.empty_like(test)
    b = numpy.random.rand(*shape)

    # default scaling and shift
    mode = 0 | API.MODE_CC
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, b, result)
    a = test.copy()
    c = numpy_correlate(a, b, mode=mode)
    assert numpy.allclose(c, result)

    # fcs scaling
    mode = API.MODE_FCS | API.MODE_CC
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, b, result)
    a = test.copy()
    c = numpy_correlate(a, b, mode=mode)
    assert numpy.allclose(c, result)

    # do not shift first axis
    mode = API.AXIS0 | API.MODE_CC
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, b, result)
    a = test.copy()
    c = numpy_correlate(a, b, mode=0, axes=(1,))
    assert numpy.allclose(c, result)

    # do not shift last axis
    mode = API.AXIS1 | API.MODE_CC
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, b, result)
    a = test.copy()
    c = numpy_correlate(a, b, mode=0, axes=(0,))
    assert numpy.allclose(c, result)

    # do not shift any axis
    mode = API.AXIS0 | API.AXIS1 | API.MODE_CC
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, b, result)
    a = test.copy()
    c = numpy_correlate(a, b, mode=0, axes=False)
    assert numpy.allclose(c, result)

    # inplace
    mode = 0 | API.MODE_CC
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, b, None)
    result = a
    a = test.copy()
    c = numpy_correlate(a, b, mode=mode)
    assert numpy.allclose(c, result)


def test_rfftnd_crosscorrelate3d():
    """Test rfft3d crosscorrelate function."""
    shape = 64, 256, 128
    test = numpy.random.rand(*shape)
    result = numpy.empty_like(test)
    b = numpy.random.rand(*shape)

    # default scaling and shift
    mode = 0 | API.MODE_CC
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, b, result)
    a = test.copy()
    c = numpy_correlate(a, b, mode=mode)
    assert numpy.allclose(c, result)

    # fcs scaling
    mode = API.MODE_FCS | API.MODE_CC
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, b, result)
    a = test.copy()
    c = numpy_correlate(a, b, mode=mode)
    assert numpy.allclose(c, result)

    # do not shift first axis
    mode = API.AXIS0 | API.MODE_CC
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, b, result)
    a = test.copy()
    c = numpy_correlate(a, b, mode=0, axes=(1, 2))
    assert numpy.allclose(c, result)

    # do not shift last axis
    mode = API.AXIS2 | API.MODE_CC
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, b, result)
    a = test.copy()
    c = numpy_correlate(a, b, mode=0, axes=(0, 1))
    assert numpy.allclose(c, result)

    # do not shift any axis
    mode = API.AXIS0 | API.AXIS1 | API.AXIS2 | API.MODE_CC
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, b, result)
    a = test.copy()
    c = numpy_correlate(a, b, mode=0, axes=False)
    assert numpy.allclose(c, result)

    # inplace
    mode = 0
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, b, None)
    result = a
    a = test.copy()
    c = numpy_correlate(a, b, mode=mode)
    assert numpy.allclose(c, result)

    # test cross- gives same result as auto-correlate
    # do not shift first axis
    mode = API.AXIS0 | API.MODE_FCS
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.autocorrelate(a, result)
    a = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, a, None)
    assert numpy.allclose(a, result)
    a = test.copy()
    b = test.copy()
    with rfftnd(shape, mode=mode) as fft:
        fft.crosscorrelate(a, b, None)
    assert numpy.allclose(a, result)


def test_rfftnd_huge3d():
    """Test rfft3d crosscorrelate with large data."""
    shape = 1024, 512, 1024
    a = numpy.random.rand(*shape)
    b = numpy.random.rand(*shape)
    with timer(f'rfftnd.autocorrelate {shape}'):
        with rfftnd(shape, mode=0) as fft:
            fft.crosscorrelate(a, b, a)


def test_rfftnd_shapes3d():
    """Test rfft3d crosscorrelate with different shapes."""
    shape = 32, 32, 2 ** 16
    a = numpy.random.rand(*shape)
    with timer(f'rfftnd.autocorrelate {shape}'):
        with rfftnd(shape, mode=0) as fft:
            fft.autocorrelate(a, a)

    shape = 2 ** 16, 32, 32
    a = numpy.random.rand(*shape)

    with timer(f'rfftnd.autocorrelate {shape}'):
        with rfftnd(shape, mode=0) as fft:
            fft.autocorrelate(a, a)


def test_rfftnd_threads():
    """Test rfft3d autocorrelate with threads."""
    from concurrent.futures import ThreadPoolExecutor
    from multiprocessing import cpu_count

    cpu_count = cpu_count() // 2

    shape = 256, 256, 256
    a = numpy.random.rand(*shape)

    def function(a):
        result = numpy.empty_like(a)
        with rfftnd(a.shape, mode=0) as fft:
            fft.autocorrelate(a, result)
        return result

    c = numpy_correlate(a, a, mode=0)
    with timer('rfftnd.autocorrelate threads'):
        with ThreadPoolExecutor(cpu_count) as executor:
            result = list(executor.map(function, [a] * cpu_count))
    for r in result:
        assert numpy.allclose(c, r)


def test_rfftnd_simulation_channel():
    """Test rfft3d correlate with simulation data."""
    data = numpy.fromfile(HERE + 'Simulation_Channel.bin', dtype='uint16')
    data.shape = -1, 64, 64
    tsize = 2 ** int(math.log(data.shape[0], 2))
    data = data[:tsize]
    # data = numpy.moveaxis(data, 0, -1)
    a = data[:, 10:42, 10:42]
    b = data[:, 10:42, 10:42].copy()
    c = numpy.empty(a.shape, 'float32')
    d = numpy.empty(a.shape, 'float32')
    with timer('rfftnd_simulation_channel'):
        with rfftnd(a.shape, mode=API.AXIS0 | API.MODE_FCS) as fft:
            fft.crosscorrelate(a, b, c)
            fft.autocorrelate(a, d)
    assert numpy.allclose(c, d)


def test_yxt_ipcf():
    """Test yxt_ipcf function."""
    data = numpy.fromfile(HERE + 'Simulation_Channel.bin', dtype='uint16')
    data.shape = -1, 64, 64  # txy
    data = numpy.moveaxis(data, 0, -1)  # xyt
    data //= 2
    data = data.astype('int16')

    with timer('yxt_ipcf'):
        result, bins, points = yxt_ipcf(
            data, radius=4, nbins=32, smooth=0.7, nthreads=0
        )

    assert_array_equal(
        bins,
        [
            1,
            2,
            3,
            4,
            5,
            7,
            10,
            13,
            18,
            24,
            32,
            43,
            58,
            78,
            104,
            139,
            187,
            250,
            334,
            447,
            598,
            800,
            1070,
            1432,
            1915,
            2561,
            3425,
            4580,
            6125,
            8192,
        ],
    )
    assert_array_equal(
        points.flat,
        [
            4,
            0,
            4,
            1,
            3,
            2,
            3,
            3,
            2,
            3,
            1,
            4,
            0,
            4,
            -1,
            4,
            -2,
            3,
            -3,
            3,
            -3,
            2,
            -4,
            1,
            -4,
            0,
            -4,
            -1,
            -3,
            -2,
            -3,
            -3,
            -2,
            -3,
            -1,
            -4,
            0,
            -4,
            1,
            -4,
            2,
            -3,
            3,
            -3,
            3,
            -2,
            4,
            -1,
        ],
    )

    known = numpy.fromfile(
        HERE + 'Simulation_Channel.ipcf.bin', dtype='float32'
    )
    known.shape = result.shape
    assert_allclose(result, known, atol=1e-6)


def test_yxt_apcf():
    """Test yxt_apcf function."""
    data = numpy.fromfile(HERE + 'Airy_Detectors.bin', dtype='uint16')

    # correct_bleaching needs 3D
    data.shape = 1, 32, 10000000
    with timer('yxt_correct_bleaching'):
        mean = yxt_correct_bleaching(data, smooth=0.99, nthreads=0)
    known = numpy.fromfile(HERE + 'Airy_Detectors.mean.bin', dtype='float64')
    known.shape = mean.shape
    assert_allclose(mean, known, atol=1e-6)

    # apcf needs 2D
    data.shape = data.shape[1:]
    with timer('yxt_apcf'):
        result, bins = yxt_apcf(data, nbins=256, smooth=0.7)
    known = numpy.fromfile(HERE + 'Airy_Detectors.apcf.bin', dtype='float32')
    known.shape = result.shape
    assert_allclose(result, known, atol=1e-6)
    assert_array_equal(
        bins,
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            21,
            22,
            23,
            25,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            43,
            45,
            48,
            51,
            54,
            58,
            61,
            65,
            69,
            74,
            78,
            83,
            88,
            94,
            99,
            106,
            112,
            119,
            126,
            134,
            143,
            151,
            161,
            171,
            181,
            192,
            204,
            217,
            230,
            245,
            260,
            276,
            293,
            311,
            330,
            350,
            372,
            395,
            419,
            445,
            473,
            502,
            533,
            566,
            601,
            638,
            677,
            719,
            763,
            810,
            860,
            913,
            969,
            1029,
            1093,
            1160,
            1231,
            1307,
            1388,
            1473,
            1564,
            1661,
            1763,
            1872,
            1987,
            2110,
            2240,
            2378,
            2524,
            2680,
            2845,
            3020,
            3207,
            3404,
            3614,
            3837,
            4073,
            4324,
            4591,
            4874,
            5174,
            5493,
            5832,
            6191,
            6573,
            6978,
            7408,
            7864,
            8349,
            8863,
            9410,
            9990,
            10605,
            11259,
            11953,
            12689,
            13471,
            14301,
            15183,
            16118,
            17112,
            18166,
            19286,
            20474,
            21736,
            23076,
            24498,
            26007,
            27610,
            29312,
            31118,
            33036,
            35072,
            37233,
            39528,
            41963,
            44549,
            47295,
            50209,
            53304,
            56589,
            60076,
            63778,
            67708,
            71881,
            76311,
            81013,
            86006,
            91306,
            96933,
            102906,
            109248,
            115981,
            123128,
            130716,
            138771,
            147323,
            156402,
            166040,
            176272,
            187135,
            198668,
            210910,
            223908,
            237706,
            252355,
            267906,
            284416,
            301944,
            320551,
            340305,
            361276,
            383540,
            407176,
            432268,
            458907,
            487187,
            517210,
            549083,
            582921,
            618843,
            656980,
            697466,
            740448,
            786078,
            834520,
            885948,
            940544,
            998506,
            1060039,
            1125364,
            1194715,
            1268339,
            1346501,
            1429479,
            1517571,
            1611092,
            1710376,
            1815778,
            1927676,
            2046469,
            2172583,
            2306469,
            2448606,
            2599502,
            2759696,
            2929763,
            3110311,
            3301984,
            3505470,
            3721495,
            3950833,
            4194304,
        ],
    )


def test_yxt_lstics():
    """Test yxt_lstics function."""
    data = numpy.fromfile(HERE + 'Simulation_Channel.bin', dtype='uint16')
    data.shape = -1, 64, 64  # txy
    data = numpy.moveaxis(data, 0, -1)  # xyt

    with timer('yxt_subtract_immobile'):
        yxt_subtract_immobile(data)

    with timer('yxt_lstics'):
        result, bins, lines = yxt_lstics(
            data,
            block=(16, 16, 2, 2),
            nlines=16,
            linelength=8,
            nbins=16,
            smooth=0.0,
        )

    known = numpy.fromfile(
        HERE + 'Simulation_Channel.lstics.bin', dtype='float32'
    )
    known.shape = result.shape
    assert_allclose(result, known, atol=1e-6)
    assert_array_equal(
        bins,
        [1, 3, 6, 11, 20, 36, 67, 122, 222, 406, 741, 1351, 2463, 4492, 8192],
    )
    assert_array_equal(
        lines,
        [
            [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0]],
            [[0, 0], [1, 0], [2, 1], [3, 1], [4, 2], [5, 2], [6, 2], [6, 3]],
            [[0, 0], [1, 1], [1, 1], [2, 2], [3, 3], [4, 4], [4, 4], [5, 5]],
            [[0, 0], [0, 1], [1, 2], [1, 3], [2, 4], [2, 5], [2, 6], [3, 6]],
            [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]],
            [
                [0, 0],
                [0, 1],
                [-1, 2],
                [-1, 3],
                [-2, 4],
                [-2, 5],
                [-2, 6],
                [-3, 6],
            ],
            [
                [0, 0],
                [-1, 1],
                [-1, 1],
                [-2, 2],
                [-3, 3],
                [-4, 4],
                [-4, 4],
                [-5, 5],
            ],
            [
                [0, 0],
                [-1, 0],
                [-2, 1],
                [-3, 1],
                [-4, 2],
                [-5, 2],
                [-6, 2],
                [-6, 3],
            ],
            [
                [0, 0],
                [-1, 0],
                [-2, 0],
                [-3, 0],
                [-4, 0],
                [-5, 0],
                [-6, 0],
                [-7, 0],
            ],
            [
                [0, 0],
                [-1, 0],
                [-2, -1],
                [-3, -1],
                [-4, -2],
                [-5, -2],
                [-6, -2],
                [-6, -3],
            ],
            [
                [0, 0],
                [-1, -1],
                [-1, -1],
                [-2, -2],
                [-3, -3],
                [-4, -4],
                [-4, -4],
                [-5, -5],
            ],
            [
                [0, 0],
                [0, -1],
                [-1, -2],
                [-1, -3],
                [-2, -4],
                [-2, -5],
                [-2, -6],
                [-3, -6],
            ],
            [
                [0, 0],
                [0, -1],
                [0, -2],
                [0, -3],
                [0, -4],
                [0, -5],
                [0, -6],
                [0, -7],
            ],
            [
                [0, 0],
                [0, -1],
                [1, -2],
                [1, -3],
                [2, -4],
                [2, -5],
                [2, -6],
                [3, -6],
            ],
            [
                [0, 0],
                [1, -1],
                [1, -1],
                [2, -2],
                [3, -3],
                [4, -4],
                [4, -4],
                [5, -5],
            ],
            [
                [0, 0],
                [1, 0],
                [2, -1],
                [3, -1],
                [4, -2],
                [5, -2],
                [6, -2],
                [6, -3],
            ],
        ],
    )


def test_yxt_imsd():
    """Test yxt_imsd function."""
    data = numpy.fromfile(HERE + 'Simulation_Channel.bin', dtype='uint16')
    data.shape = -1, 64, 64  # txy
    data = numpy.moveaxis(data, 0, -1)  # xyt

    with timer('yxt_subtract_immobile'):
        yxt_subtract_immobile(data)

    with timer('yxt_imsd'):
        result = yxt_imsd(data, block=(32, 32, 4, 4), bins=16, smooth=0.0)

    result = numpy.moveaxis(result, -1, -3)  # SimFCS axes order
    known = numpy.fromfile(
        HERE + 'Simulation_Channel.imsd.bin', dtype='float32'
    )
    known.shape = result.shape
    assert_allclose(result, known, atol=1e-6)


def test_yxt_subtract_immobile(dtype='int16', nthreads=0):
    """Test yxt_subtract_immobile function."""
    data = numpy.fromfile(HERE + 'Simulation_Channel.bin', dtype='uint16')
    data.shape = -1, 64, 64  # txy
    if dtype == 'int16':
        data //= 2
        data = data.astype(dtype)

    def subtract_immobile_np(a):
        dt = a.dtype
        a = numpy.ascontiguousarray(a).astype('float64')
        im = numpy.mean(a, axis=2)
        im -= numpy.mean(im)
        a -= im[..., numpy.newaxis]
        a = numpy.round(a)
        a = numpy.clip(a, numpy.iinfo(dt).min, numpy.iinfo(dt).max)
        return a.astype(dt)

    with timer('subtract_immobile_np'):
        corrected_np = subtract_immobile_np(numpy.moveaxis(data, 0, -1))

    # non-contiguous
    corrected = data.copy()
    corrected = numpy.moveaxis(corrected, 0, -1)  # need yxt
    # print(corrected.shape, corrected.strides)
    with timer('yxt_subtract_immobile tyx'):
        yxt_subtract_immobile(corrected, nthreads)
    assert numpy.allclose(corrected, corrected_np)

    # contiguous
    corrected = data.copy()
    corrected = numpy.moveaxis(corrected, 0, -1)  # need yxt
    corrected = numpy.ascontiguousarray(corrected)
    # print(corrected.shape, corrected.strides)
    with timer('yxt_subtract_immobile yxt'):
        yxt_subtract_immobile(corrected, nthreads)

    assert numpy.allclose(corrected, corrected_np)


def test_yxt_correct_bleaching(dtype='int16', smooth=0.99, nthreads=0):
    """Test yxt_correct_bleaching function."""
    # data = numpy.fromfile(HERE + 'Simulation_Channel.bin', dtype='uint16')
    # data.shape = -1, 64, 64  # txy
    data = numpy.fromfile(HERE + 'bleach.bin', 'uint16').reshape((64, 64, -1))
    data = numpy.moveaxis(data, -1, 0).copy()
    if dtype == 'int16':
        data //= 2
        data = data.astype(dtype)

    def correct_bleaching_np(data, smooth=smooth):
        f0 = smooth
        f1 = 1.0 - smooth
        dt = data.dtype
        a = numpy.ascontiguousarray(data).astype('float64')
        mean = numpy.mean(a, axis=2)
        for i in range(1, a.shape[2]):
            a[..., i] = data[..., i] * f1 + a[..., i - 1] * f0
        for i in range(a.shape[2] - 2, -1, -1):
            a[..., i] = a[..., i] * f1 + a[..., i + 1] * f0

        deficit = numpy.reciprocal(a)
        deficit *= mean[..., None]
        deficit = numpy.sqrt(numpy.abs(deficit))
        a *= -1.0
        a += data
        a *= deficit
        a += mean[..., None]
        a = numpy.round(a)
        a = numpy.clip(a, numpy.iinfo(dt).min, numpy.iinfo(dt).max)
        return a.astype(dt), mean

    with timer('correct_bleaching_np'):
        corrected_np, mean_np = correct_bleaching_np(
            numpy.moveaxis(data, 0, -1)
        )

    # non-contiguous
    corrected = data.copy()
    corrected = numpy.moveaxis(corrected, 0, -1)  # need yxt
    # print(corrected.shape, corrected.strides)
    with timer('yxt_correct_bleaching tyx'):
        mean = yxt_correct_bleaching(corrected, smooth, nthreads)

    if 0:
        pyplot.imshow(mean)
        pyplot.figure()
        pyplot.plot(data[:, 10, 10])
        pyplot.plot(corrected[10, 10])
        pyplot.show()

    assert numpy.allclose(corrected, corrected_np)
    assert numpy.allclose(mean, mean_np)

    # contiguous
    corrected = data.copy()
    corrected = numpy.moveaxis(corrected, 0, -1)  # need yxt
    corrected = numpy.ascontiguousarray(corrected)
    # print(corrected.shape, corrected.strides)
    with timer('yxt_correct_bleaching yxt'):
        mean = yxt_correct_bleaching(corrected, smooth, nthreads)

    assert numpy.allclose(corrected, corrected_np)
    assert numpy.allclose(mean, mean_np)


def test_ipcf_nlsp_1dpcf():
    """Test ipcf_nlsp_1dpcf function."""
    if sys.version_info < (3, 7):
        pytest.xfail('results depend on compiler version')
    ipcf = numpy.fromfile(
        HERE + 'Simulation_Channel.ipcf.bin', dtype='float32'
    )
    ipcf.shape = 56, 56, 24, 30

    bins = logbins(8192, 32)
    times = bins2times(bins, frametime=0.01)
    assert len(times) == ipcf.shape[3]
    points = circle(4)
    distances = points2distances(points, pixelsize=0.05)
    assert len(distances) == ipcf.shape[2]

    args = [0.1, 1.0, 0.3 * 0.3]
    bounds = [1e-6, 1e3, 1e-6, 1e3]
    settings = [
        400.0,
        200.0,
        1e-10,
        1e-10,
        1e-10,
        1e-10,
        1e-10,
        1e-10,
        1e-10,
        0,
    ]

    with timer('ipcf_nlsp_1dpcf'):
        x, fx, status = ipcf_nlsp_1dpcf(
            ipcf,
            times,
            distances,
            args,
            bounds,
            ix=None,
            ifx=None,
            status=None,
            settings=settings,
            average=False,
            nthreads=0,
        )

    ipcf_x = numpy.fromfile(
        HERE + 'Simulation_Channel.ipcf_x.bin', dtype='float32'
    )
    ipcf_x.shape = x.shape
    assert_allclose(x, ipcf_x)

    ipcf_fx = numpy.fromfile(
        HERE + 'Simulation_Channel.ipcf_fx.bin', dtype='float32'
    )
    ipcf_fx.shape = fx.shape
    assert_allclose(fx[..., 1:-1], ipcf_fx[..., 1:-1])

    ipcf_status = numpy.fromfile(
        HERE + 'Simulation_Channel.ipcf_status.bin', dtype='float32'
    )
    ipcf_status.shape = status.shape
    # assert_allclose(ipcf_status, status)


def test_yxt_dft():
    """Test yxt_dft function"""
    data = numpy.fromfile(HERE + 'LifetimeHistogram.bin', dtype='float32')
    data.shape = 256, 256, 256

    with timer('yxt_dft'):
        result = icsdll.yxt_dft(numpy.moveaxis(data, 0, -1), samples=5)

    with timer('yxt_dft numpy'):
        expected = numpy.fft.fft(data, axis=0).astype('complex64')

    atol = 1e-6
    assert_allclose(result[0], expected[0].real)
    assert_allclose(result[1], expected[1].real, atol=atol)
    assert_allclose(result[2], expected[1].imag, atol=atol)
    assert_allclose(result[3], expected[2].real, atol=atol)
    assert_allclose(result[4], expected[2].imag, atol=atol)


@contextmanager
def timer(message):
    """Context manager for timing execution speed of body."""
    print(message, end=' ')
    start = time.time()
    yield
    print('took {:1.0f} ms'.format(1000 * (time.time() - start)))


def runall():
    """Run all tests."""
    import inspect

    for func in (
        obj
        for name, obj in inspect.getmembers(sys.modules[__name__])
        if inspect.isfunction(obj) and name.startswith('test_')
    ):
        try:
            func()
        except NotImplementedError:
            pass
    print('All done.')


if __name__ == '__main__':
    if os.environ.get('WINGDB_ACTIVE') or os.environ.get('TERM_PROGRAM'):
        # running in IDE
        sys.path = ['..'] + sys.path

        from matplotlib import pyplot
        import icsdll
        from icsdll import *

        runall()
    else:
        # run pytests
        import icsdll
        from icsdll import *
        import warnings

        # warnings.simplefilter('always')  # noqa
        warnings.filterwarnings('ignore', category=ImportWarning)  # noqa
        argv = sys.argv
        # argv.append('--cov=icsdll')
        argv.append('--verbose')
        sys.exit(pytest.main(argv))
else:
    # pytest
    import icsdll
    from icsdll import *
