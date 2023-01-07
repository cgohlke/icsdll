/* yxt.cpp

Implementation of the yxt class for the ICS library.

Copyright (c) 2016-2023, Christoph Gohlke
This source code is distributed under the BSD 3-Clause license.

Refer to the header file 'ics.h' for documentation and license.

*/

#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <cmath>
#include <limits>

#include "ics.h"
#include "ics.hpp"

/** yxt class C++ API **/

/* Class constructor.

Allocate 3D buffer for caching forward DFTs of specified size.
The size * sizeof(double) should fit comfortably into available memory.

Parameters
----------

size : ssize_t*
    Pointer to three integers specifying the sizes in y, x, and t dimensions
    of the data to be analyzed: [image length, image width, number time
    points]. The number of time points is clipped to the next lower or equal
    power of two.

*/
yxt::yxt(const ssize_t *shape)
{
    a_ = NULL;
    b_ = NULL;
    ysize_ = shape[0];
    xsize_ = shape[1];
    tsize_ = shape[2];

    if ((ysize_ < 1) || (ysize_ > INT32_MAX) || (xsize_ < 1) ||
        (xsize_ > INT32_MAX) || (tsize_ < 8)) {
        throw ICS_VALUE_ERROR;
    }

    if (!ispow2(tsize_)) {
        tsize_ = (ssize_t)pow(2.0, trunc(log2(tsize_)));
    }

    shape_[0] = ysize_;
    shape_[1] = xsize_;
    shape_[2] = tsize_ + MKL_ALIGN_D;

    size_ = shape_[0] * shape_[1] * shape_[2];

    strides_[0] = shape_[2] * shape_[1];
    strides_[1] = shape_[2];
    strides_[2] = 1;

    a_ = (double *)mkl_malloc(size_ * sizeof(double), MKL_ALIGN);
    if (a_ == NULL) {
        throw ICS_MEMORY_ERROR;
    }
}

/* Class destructor: release buffers and handles. */
yxt::~yxt()
{
    if (a_ != NULL) {
        mkl_free(a_);
        a_ = NULL;
    }
    if (b_ != NULL) {
        mkl_free(b_);
        b_ = NULL;
    }
}

/* Access internal buffer a_ */
double *
yxt::yxt_get_buffer(ssize_t *shape, ssize_t *strides)
{
    if (shape != NULL) {
        shape[0] = ysize_;
        shape[1] = xsize_;
        shape[2] = tsize_;
    }
    if (strides != NULL) {
        strides[0] = strides_[0] * sizeof(double);
        strides[1] = strides_[1] * sizeof(double);
        strides[2] = strides_[2] * sizeof(double);
    }
    return a_;
}

/* Calculate pair correlation functions for all pixels in image series.

Parameters
----------

data : Ti*
    Pointer to input 3D array.
    If NULL, the internal buffer is re-used.
    Average along time axis should not be zero, i.e. do not subtract the mean.
channel : Ti*
    Pointer to input 3D array of second channel.
    If not NULL, each pixel from this channel is cross correlated with
    points from the data channel.
strides : ssize_t*
    Pointer to 3 integers defining the strides of the data and channel arrays
    in y, x, and t dimensions.
    Strides are the number of bytes required in each dimension to advance from
    one item to the next within the dimension.
out : To*
    Pointer to 4D output array of cross correlation carpets at each pixel.
    The order of the axes is length, width, npoints, nbins.
    For example, if `points` are circular coordinates, the `out` array must be
    large enough to hold (length-2*radius)*(width-2*radius)*npoints*nbins
    items of type To.
outstrides : ssize_t*
    Pointer to 4 integers defining the strides of the `out` array.
    The last stride must be sizeof(To), i.e. the last axis is contiguous.
    Strides are the number of bytes required in each dimension to advance from
    one item to the next within the dimension.
points : ssize_t*
    Pointer to array of (x, y) coordinates to cross-correlate at each pixel in
    image. For example, this can be the (x, y) coordinates of a circle at
    radius from origin.
npoints : ssize_t
    Half size of `points` array.
bins : ssize_t*
    Pointer to the output of logbins() used to bin cross correlation curves.
    The last item should be half of the internal buffer shape[2].
nbins : ssize_t
    Size of `bins` array. Must be less or equal than half size of time axis.
threshold : double
    Mean intensity threshold for calculating cross correlation curves.
filter : double
    Factor to use for simple exponential smoothing of log-binned cross
    correlation curves.
nthreads : int
    Number of OpenMP threads to use for parallelizing loops along the y axis.
    Set to zero for OpenMP default.

*/
template <typename Ti, typename To>
void
yxt::ipcf(
    const Ti *data,
    const Ti *channel,
    const ssize_t *strides,
    To *out,
    const ssize_t *outstrides,
    const ssize_t *points,
    const ssize_t npoints,
    const ssize_t *bins,
    const ssize_t nbins,
    const double threshold,
    const double filter,
    const int nthreads)
{
    const double threshold_size = threshold * tsize_;
    const bool usechannel = (channel != NULL) && (channel != data);

    if (((data != NULL) || usechannel) && strides == NULL) {
        throw ICS_VALUE_ERROR1;
    }

    if (outstrides == NULL) {
        throw ICS_VALUE_ERROR2;
    }

    if ((out == NULL) || (points == NULL) || (npoints < 1)) {
        throw ICS_VALUE_ERROR3;
    }

    if (validate_bins(bins, nbins, tsize_ / 2) != ICS_OK) {
        throw ICS_VALUE_ERROR4;
    }

    ssize_t *mm = new ssize_t[4];
    minmax(npoints, 2, points, mm);
    const ssize_t xmin = (mm[0] > 0 ? 0 : -mm[0]);
    const ssize_t xmax = (mm[1] < 0 ? 0 : -mm[1]) + xsize_;
    const ssize_t ymin = (mm[2] > 0 ? 0 : -mm[2]);
    const ssize_t ymax = (mm[3] < 0 ? 0 : -mm[3]) + ysize_;
    delete[] mm;

    if ((xmin >= xmax) || (ymin >= ymax))
        return;

    // store indices to symmetric points
    ssize_t *points2 = new ssize_t[npoints];
    for (ssize_t i = 0; i < npoints; i++) {
        const ssize_t x = -points[2 * i];
        const ssize_t y = -points[2 * i + 1];
        points2[i] = -1;
        for (ssize_t j = 0; j < npoints; j++) {
            if ((x == points[2 * j]) && (y == points[2 * j + 1])) {
                points2[i] = j;
                break;
            }
        }
    }

    // initialize MKL for 1D in-place DFT
    DFTI_DESCRIPTOR_HANDLE dfti_handle;
    MKL_LONG status;
    status = DftiCreateDescriptor(
        &dfti_handle, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)tsize_);
    if (status) {
        delete[] points2;
        throw status;
    }
    status = DftiSetValue(
        dfti_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    if (status) {
        delete[] points2;
        throw status;
    }
    status = DftiSetValue(dfti_handle, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT);
    if (status) {
        delete[] points2;
        throw status;
    }
    status = DftiSetValue(dfti_handle, DFTI_PLACEMENT, DFTI_INPLACE);
    if (status) {
        delete[] points2;
        throw status;
    }
    status = DftiCommitDescriptor(dfti_handle);
    if (status) {
        delete[] points2;
        throw status;
    }

    // disable MKL multi-threading to prevent oversubscription
    const int mkl_threads = mkl_get_max_threads();
    mkl_set_num_threads(1);

#pragma omp parallel num_threads(nthreads)
    {
        // allocate thread local buffers
        double *a =
            (double *)mkl_malloc(shape_[2] * sizeof(double), MKL_ALIGN);
        double *b =
            (double *)mkl_malloc(shape_[2] * sizeof(double), MKL_ALIGN);

        if (data != NULL) {
            // copy forward DFT of input time series to buffer
#pragma omp for
            for (ssize_t y = 0; y < ysize_; y++) {
                double *pbuff = a_ + y * strides_[0];
                for (ssize_t x = 0; x < xsize_; x++) {
                    char *pdata =
                        (char *)data + y * strides[0] + x * strides[1];
                    for (ssize_t t = 0; t < tsize_; t++) {
                        pbuff[t] = (double)*(Ti *)pdata;
                        pdata += strides[2];
                    }
                    DftiComputeForward(dfti_handle, pbuff);
                    pbuff += strides_[1];
                }
            }
        }

        // initialize output
#pragma omp for
        for (ssize_t y = 0; y < ymax - ymin; y++) {
            for (ssize_t x = 0; x < xmax - xmin; x++) {
                for (ssize_t p = 0; p < npoints; p++) {
                    To *pout =
                        (To *)(
                            (char *)out
                            + y * outstrides[0]
                            + x * outstrides[1]
                            + p * outstrides[2]);
                    pout[0] = 0.0;
                }
            }
        }

        // loop over all pixels in image
#pragma omp for
        for (ssize_t y = ymin; y < ymax; y++) {
            if ((a == NULL) || (usechannel && (b == NULL)))
                continue;
            for (ssize_t x = xmin; x < xmax; x++) {
                double *origin;
                if (usechannel) {
                    // calculate forward DFT for channel at x, y
                    char *pchannel =
                        (char *)channel + y * strides[0] + x * strides[1];
                    for (ssize_t t = 0; t < tsize_; t++) {
                        b[t] = (double)*(Ti *)pchannel;
                        pchannel += strides[2];
                    }
                    DftiComputeForward(dfti_handle, b);
                    origin = b;
                }
                else {
                    origin = a_ + y * strides_[0] + x * strides_[1];
                }
                if (origin[0] <= threshold_size) {
                    continue;
                }
                const double scale = 1.0 / origin[0];
                // loop over points to cross correlate with origin
                for (ssize_t p = 0; p < npoints; p++) {
                    const double *point =
                        a_ + (y + points[2 * p + 1]) * strides_[0] +
                        (x + points[2 * p]) * strides_[1];
                    if (point[0] <= threshold_size) {
                        continue;
                    }
                    To *pout =
                        (To *)(
                            (char *)out
                            + (y - ymin) * outstrides[0]
                            + (x - xmin) * outstrides[1] +
                            p * outstrides[2]);
                    if (pout[0] != 0.0) {
                        continue;  // already calculated
                    }
                    // multiply point's DFT by complex conjugate of origin's
                    // DFT and store in a
                    // vmzMulByConj((int)(tsize_ / 2 + 1),
                    // (MKL_Complex16*)point, (MKL_Complex16*)origin,
                    // (MKL_Complex16*)a, VML_EP | VML_FTZDAZ_ON |
                    // VML_ERRMODE_IGNORE);
                    complex_multiply(a, point, origin, tsize_ + 2);
                    // compute invers DFT
                    DftiComputeBackward(dfti_handle, a);
                    // average, normalize, and smooth first half of cross
                    // correlation curve
                    anscf(
                        a,
                        pout,
                        outstrides[3],
                        bins,
                        nbins,
                        scale / point[0],
                        -1.0,
                        filter,
                        true);

                    // use symmetry?
                    if (usechannel) {
                        continue;
                    }
                    const ssize_t p2 = points2[p];
                    if (p2 < 0) {
                        continue;  // no symmetric point
                    }
                    const ssize_t x2 = x - points[2 * p2];
                    const ssize_t y2 = y - points[2 * p2 + 1];
                    if ((x2 < xmin) || (x2 >= xmax) || (y2 < ymin) ||
                        (y2 >= ymax)) {
                        continue;  // out of bounds
                    }
                    // reverse, average, normalize, and smooth second half of
                    // cross correlation curve
                    pout =
                        (To *)(
                            (char *)out
                            + (y2 - ymin) * outstrides[0]
                            + (x2 - xmin) * outstrides[1]
                            + p2 * outstrides[2]);
                    anscf(
                        a,
                        pout,
                        outstrides[3],
                        bins,
                        nbins,
                        scale / point[0],
                        -1.0,
                        filter,
                        true,
                        tsize_);
                }
            }
        }

        // de-allocate thread local buffers
        mkl_free(a);
        mkl_free(b);
    }

    DftiFreeDescriptor(&dfti_handle);
    mkl_set_num_threads(mkl_threads);
    delete[] points2;
}

/* Calculate image mean square displacement functions for blocks in image
series.

Parameters
----------

data : Ti*
    Pointer to input 3D array.
    If NULL, the internal buffer is re-used.
    Averages along time axis should not be zero, i.e. do not subtract the mean.
strides : ssize_t*
    Pointer to 3 integers defining the strides of the data array in y, x, and t
    dimensions.
    Strides are the number of bytes required in each dimension to advance from
    one item to the next within the dimension.
data1 : Ti*
    Pointer to second input 3D array.
    If not NULL, the two data arrays is cross-correlated.
strides1 : ssize_t*
    Pointer to 3 integers defining the strides of the data1 array in y, x, and
    t dimensions.
mask : Tm*
    Pointer to a 2D array of integers.
    If NULL, all blocks is analyzed.
maskstrides : ssize_t*
    Pointer to 2 integers defining the strides of the masks array in y and x
    dimensions.
maskmode : int32_t
    Defines how mask is applied.
    One of ICS_MASK_ANY, ICS_MASK_FIRST, ICS_MASK_CENTER, or ICS_MASK_ALL.
    If ICS_MASK_CLEAR is set, sprites not calculated are zeroed.
out : To*
    Pointer to 5D output array of 3D correlation functions at each region.
    The order of the axes is length, width, blocks length, blocks width, nbins.
    Should be initialized since not all sprites might be calculated.
outstrides : ssize_t*
    Pointer to 5 integers defining the strides of the `out` array.
block : ssize_t*
    Pointer to four integers defining overlapping blocks to analyze:
    length, width, delta y between blocks, delta x between blocks.
bins : ssize_t*
    Pointer to the output of logbins() used to bin cross correlation curves.
    The last item should be half of the internal buffer shape[2].
    If NULL, the first `nbins` planes from the cross correlation functions are
    returned.
nbins : ssize_t
    Size of `bins` array. Must be less or equal than half size of the time
    axis.
filter : double
    Factor to use for simple exponential smoothing of log-binned cross
    correlation curves.
nthreads : int
    Number of OpenMP threads to use for parallelizing loops along the y axis.
    Set to zero for OpenMP default.
    For each thread, a rfft3d instance is allocated, requiring
    (number time points)*(block length)*(block width)*sizeof(double) bytes.

*/
template <typename Ti, typename Tm, typename To>
void
yxt::imsd(
    const Ti *data,
    const ssize_t *strides,
    const Ti *data1,
    const ssize_t *strides1,
    const Tm *mask,
    const ssize_t *maskstrides,
    const int32_t maskmode,
    To *out,
    const ssize_t *outstrides,
    const ssize_t *block,
    ssize_t *bins,
    const ssize_t nbins,
    const double filter,
    const int nthreads)
{
    if ((out == NULL) || (outstrides == NULL) || (block == NULL)) {
        throw ICS_VALUE_ERROR;
    }

    if ((block[0] < 1) || (block[0] > ysize_) || !ispow2(block[0]) ||
        (block[1] < 1) || (block[1] > xsize_) || !ispow2(block[1]) ||
        (block[2] < 1) || (block[2] > block[0]) || (block[3] < 1) ||
        (block[3] > block[1])) {
        throw ICS_VALUE_ERROR;
    }

    ssize_t *bins_ = NULL;
    if (bins == NULL) {
        // linear binning
        if ((nbins < 1) || (nbins > tsize_ / 2)) {
            throw ICS_VALUE_ERROR;
        }
        bins_ = (ssize_t *)malloc(nbins * sizeof(ssize_t));
        if (bins_ == NULL) {
            throw ICS_MEMORY_ERROR;
        }
        bins = bins_;
        for (ssize_t i = 0; i < nbins; i++) {
            bins[i] = i + 1;
        }
    }
    else {
        if (validate_bins(bins, nbins, tsize_ / 2) != ICS_OK)
            throw ICS_VALUE_ERROR;
    }

    const bool usechannel = (data1 != NULL) && (data1 != data);

    if ((usechannel) && (b_ == NULL)) {
        b_ = (double *)mkl_malloc(size_ * sizeof(double), MKL_ALIGN);
        if (b_ == NULL) {
            throw ICS_MEMORY_ERROR;
        }
    }

    // disable MKL multi-threading to prevent oversubscription
    const int mkl_threads = mkl_get_max_threads();
    mkl_set_num_threads(1);

#pragma omp parallel num_threads(nthreads)
    {
        // thread-local fft class
        rfft3d *worker;
        try {
            worker = new rfft3d(
                block[0],
                block[1],
                tsize_,
                ICS_AXIS2 | ICS_MODE_FCS | (usechannel ? ICS_MODE_CC : 0));
        }
        catch (...) {
            worker = NULL;
        }

        if (data != NULL) {
            // copy data to a_
#pragma omp for
            for (ssize_t y = 0; y < ysize_; y++) {
                double *pa = a_ + y * strides_[0];
                for (ssize_t x = 0; x < xsize_; x++) {
                    char *pdata =
                        (char *)data + y * strides[0] + x * strides[1];
                    for (ssize_t t = 0; t < tsize_; t++) {
                        pa[t] = (double)*(Ti *)pdata;
                        pdata += strides[2];
                    }
                    pa += strides_[1];
                }
            }
        }

        if (usechannel) {
            // copy channel to b_
#pragma omp for
            for (ssize_t y = 0; y < ysize_; y++) {
                double *pb = b_ + y * strides_[0];
                for (ssize_t x = 0; x < xsize_; x++) {
                    char *pdata =
                        (char *)data1 + y * strides1[0] + x * strides1[1];
                    for (ssize_t t = 0; t < tsize_; t++) {
                        pb[t] = (double)*(Ti *)pdata;
                        pdata += strides1[2];
                    }
                    pb += strides_[1];
                }
            }
        }

        // loop over regions in image
#pragma omp for
        for (ssize_t y = 0; y <= ysize_ - block[0]; y += block[2]) {
            if (worker == NULL)
                continue;
            for (ssize_t x = 0; x <= xsize_ - block[1]; x += block[3]) {
                if (selected(
                        mask,
                        maskstrides,
                        y,
                        x,
                        block[0],
                        block[1],
                        maskmode)) {
                    const double *a = a_ + y * strides_[0] + x * strides_[1];
                    const double *b =
                        usechannel ? b_ + y * strides_[0] + x * strides_[1]
                                   : NULL;
                    To *pout =
                        (To *)(
                            (char *)out
                            + (y / block[2]) * outstrides[0]
                            + (x / block[3]) * outstrides[1]);
                    worker->imsd(
                        a,
                        b,
                        strides_,
                        pout,
                        outstrides + 2,
                        bins,
                        nbins,
                        filter);
                }
                else if (maskmode & ICS_MASK_CLEAR) {
                    const char *pout = (char *)out +
                                       (y / block[2]) * outstrides[0] +
                                       (x / block[3]) * outstrides[1];
                    for (ssize_t i = 0; i < block[0]; i++) {
                        for (ssize_t j = 0; j < block[1]; j++) {
                            for (ssize_t k = 0; k < nbins; k++) {
                                *(To *)(pout
                                        + i * outstrides[2]
                                        + j * outstrides[3]
                                        + k * outstrides[4]) = (To)0.0;
                            }
                        }
                    }
                }
            }
        }

        if (worker != NULL)
            delete worker;
    }

    mkl_set_num_threads(mkl_threads);
    if (bins_ != NULL) {
        free(bins_);
    }
}

/* Calculate line STICS functions for blocks in image series.

Parameters
----------

data : Ti*
    Pointer to input 3D array.
    If NULL, the internal buffer is re-used.
    Averages along time axis should not be zero, i.e. do not subtract the mean.
strides : ssize_t*
    Pointer to 3 integers defining the strides of the data array in y, x, and t
    dimensions.
    Strides are the number of bytes required in each dimension to advance from
    one item to the next within the dimension.
data1 : Ti*
    Pointer to second input 3D array.
    If not NULL, the two data arrays is cross-correlated.
strides1 : ssize_t*
    Pointer to 3 integers defining the strides of the data1 array in y, x, and
    t dimensions.
mask : Tm*
    Pointer to a 2D array of integers.
    If NULL, all blocks is analyzed.
maskstrides : ssize_t*
    Pointer to 2 integers defining the strides of the masks array in y and x
    dimensions.
maskmode : int32_t
    Defines how mask is applied.
    One of ICS_MASK_ANY, ICS_MASK_FIRST, ICS_MASK_CENTER, or ICS_MASK_ALL.
    If ICS_MASK_CLEAR is set, sprites not calculated are zeroed.
out : To*
    Pointer to 5D output array of 2D correlation functions at each region.
    The order of the axes is length, width, number of lines, line length,
    nbins. Should be initialized since not all sprites might be calculated.
outstrides : ssize_t*
    Pointer to 5 integers defining the strides of the `out` array.
lines : ssize_t*
    Pointer to contiguous 3D array of coordinates defining the lines to
    analyze. The order of dimensions is: number of lines, line length, 2.
linesshape : ssize_t*
    Pointer to 3 integers defining the shape of the 3D lines array: number of
    lines, line length, 2. Line length must be a power of two.
blocks : ssize_t*
    Pointer to four integers defining overlapping blocks to analyze:
    length, width, delta y between blocks, delta x between blocks.
    Lines, centered at (length/2, width/2) must fit into blocks.
    If NULL, block size is determened from the line coordinates and
    advancements is 1 pixel.
bins : ssize_t*
    Pointer to the output of logbins() used to bin cross correlation curves.
    The last item should be half of the internal buffer shape[2].
    If NULL, the first `nbins` samples from the cross correlation functions
    are returned.
nbins : ssize_t
    Size of `bins` array. Must be less or equal than half size of the time
    axis.
filter : double
    Factor to use for simple exponential smoothing of log-binned cross
    correlation curves.
nthreads : int
    Number of OpenMP threads to use for parallelizing loops along the y axis.
    Set to zero for OpenMP default.
    For each thread, a rfft2d instance is allocated, requiring
    (number time points)*(line length)*sizeof(double) bytes.

*/
template <typename Ti, typename Tm, typename To>
void
yxt::lstics(
    const Ti *data,
    const ssize_t *strides,
    const Ti *data1,
    const ssize_t *strides1,
    const Tm *mask,
    const ssize_t *maskstrides,
    const int32_t maskmode,
    To *out,
    const ssize_t *outstrides,
    const ssize_t *lines,
    const ssize_t *linesshape,
    const ssize_t *blocks,
    ssize_t *bins,
    const ssize_t nbins,
    const double filter,
    const int nthreads)
{
    if ((out == NULL) || (outstrides == NULL) || (lines == NULL) ||
        (linesshape == NULL)) {
        throw ICS_VALUE_ERROR;
    }

    const ssize_t nlines = linesshape[0];
    const ssize_t linelength = linesshape[1];

    if ((nlines < 1) || (nlines > 1024) || (linelength < 1) ||
        !ispow2(linelength) || (linelength > xsize_ / 2) ||
        (linelength > ysize_ / 2)) {
        throw ICS_VALUE_ERROR;
    }

    ssize_t *mm = new ssize_t[4];
    minmax(nlines * linelength, 2, lines, mm);

    ssize_t *block = new ssize_t[4];
    if (blocks == NULL) {
        block[0] = mm[1] - mm[0];
        block[1] = mm[3] - mm[2];
        block[2] = 1;
        block[3] = 1;
    }
    else {
        block[0] = blocks[0];
        block[1] = blocks[1];
        block[2] = blocks[2];
        block[3] = blocks[3];
    }

    if ((block[0] < 1) || (block[0] > ysize_) || (block[1] < 1) ||
        (block[1] > xsize_) || (block[2] < 1) || (block[2] > block[0]) ||
        (block[3] < 1) || (block[3] > block[1]) ||
        (block[0] / 2 + mm[0] < 0) || (block[0] / 2 + mm[0] >= block[0]) ||
        (block[0] / 2 + mm[1] < 0) || (block[0] / 2 + mm[1] >= block[0]) ||
        (block[1] / 2 + mm[2] < 0) || (block[1] / 2 + mm[2] >= block[1]) ||
        (block[1] / 2 + mm[3] < 0) || (block[1] / 2 + mm[3] >= block[1])) {
        delete[] mm;
        delete[] block;
        throw ICS_VALUE_ERROR;
    }
    delete[] mm;

    ssize_t *bins_ = NULL;
    if (bins == NULL) {
        // linear binning
        if ((nbins < 1) || (nbins > tsize_ / 2)) {
            delete[] block;
            throw ICS_VALUE_ERROR;
        }
        bins_ = (ssize_t *)malloc(nbins * sizeof(ssize_t));
        if (bins_ == NULL) {
            delete[] block;
            throw ICS_MEMORY_ERROR;
        }
        bins = bins_;
        for (ssize_t i = 0; i < nbins; i++) {
            bins[i] = i + 1;
        }
    }
    else {
        if (validate_bins(bins, nbins, tsize_ / 2) != ICS_OK) {
            delete[] block;
            throw ICS_VALUE_ERROR;
        }
    }

    const bool usechannel = (data1 != NULL) && (data1 != data);

    if ((usechannel) && (b_ == NULL)) {
        b_ = (double *)mkl_malloc(size_ * sizeof(double), MKL_ALIGN);
        if (b_ == NULL) {
            delete[] block;
            throw ICS_MEMORY_ERROR;
        }
    }

    // disable MKL multi-threading to prevent oversubscription
    const int mkl_threads = mkl_get_max_threads();
    mkl_set_num_threads(1);

#pragma omp parallel num_threads(nthreads)
    {
        // thread-local fft class
        rfft2d *worker;
        try {
            worker = new rfft2d(
                linelength,
                tsize_,
                ICS_AXIS1 | ICS_MODE_FCS | (usechannel ? ICS_MODE_CC : 0));
        }
        catch (...) {
            worker = NULL;
        }

        if (data != NULL) {
            // copy data to a_
#pragma omp for
            for (ssize_t y = 0; y < ysize_; y++) {
                double *pa = a_ + y * strides_[0];
                for (ssize_t x = 0; x < xsize_; x++) {
                    char *pdata =
                        (char *)data + y * strides[0] + x * strides[1];
                    for (ssize_t t = 0; t < tsize_; t++) {
                        pa[t] = (double)*(Ti *)pdata;
                        pdata += strides[2];
                    }
                    pa += strides_[1];
                }
            }
        }

        if (usechannel) {
            // copy channel to b_
#pragma omp for
            for (ssize_t y = 0; y < ysize_; y++) {
                double *pb = b_ + y * strides_[0];
                for (ssize_t x = 0; x < xsize_; x++) {
                    char *pdata =
                        (char *)data1 + y * strides1[0] + x * strides1[1];
                    for (ssize_t t = 0; t < tsize_; t++) {
                        pb[t] = (double)*(Ti *)pdata;
                        pdata += strides1[2];
                    }
                    pb += strides_[1];
                }
            }
        }

        // loop over regions in image
#pragma omp for
        for (ssize_t y = 0; y <= ysize_ - block[0]; y += block[2]) {
            if (worker == NULL)
                continue;
            for (ssize_t x = 0; x <= xsize_ - block[1]; x += block[3]) {
                if (selected(
                        mask,
                        maskstrides,
                        y,
                        x,
                        block[0],
                        block[1],
                        maskmode)) {
                    const ssize_t offset = (y + block[0] / 2) * strides_[0] +
                                           (x + block[1] / 2) * strides_[1];
                    const double *a = a_ + offset;
                    const double *b = usechannel ? b_ + offset : NULL;
                    ssize_t *pline = (ssize_t *)lines;
                    // loop over lines
                    for (ssize_t i = 0; i < nlines; i++) {
                        To *pout =
                            (To *)(
                                (char *)out
                                + (y / block[2]) * outstrides[0]
                                + (x / block[3]) * outstrides[1]
                                + i * outstrides[2]);
                        worker->lstics(
                            a,
                            b,
                            strides_,
                            pout,
                            outstrides + 3,
                            pline,
                            bins,
                            nbins,
                            filter);
                        pline += linelength + linelength;
                    }
                }
                else if (maskmode & ICS_MASK_CLEAR) {
                    const char *pout = (char *)out +
                                       (y / block[2]) * outstrides[0] +
                                       (x / block[3]) * outstrides[1];
                    for (ssize_t i = 0; i < nlines; i++) {
                        for (ssize_t j = 0; j < linelength; j++) {
                            for (ssize_t k = 0; k < nbins; k++) {
                                *(To *)(pout
                                        + i * outstrides[2]
                                        + j * outstrides[3]
                                        + k * outstrides[4]) = (To)0.0;
                            }
                        }
                    }
                }
            }
        }

        if (worker != NULL)
            delete worker;
    }

    mkl_set_num_threads(mkl_threads);
    if (bins_ != NULL) {
        free(bins_);
    }
    delete[] block;
}

/* Calculate all pair correlation functions for Airy disc.

Parameters
----------

data : Ti*
    Pointer to input 2D array.
    If NULL, the internal buffer is re-used.
    Averages along time axis should not be zero, i.e. do not subtract the mean.
strides : ssize_t*
    Pointer to 2 integers defining the strides of the data array in Airy
    detectors and t dimensions.
    Strides are the number of bytes required in each dimension to advance from
    one item yo the next within the dimension.
out : To*
    Pointer to 3D output array of correlation functions.
    The order of the axes is: airy detectors, airy detectors (-1 if autocorr
    is False), nbins.
    Should be initialized since not all sprites might be calculated.
outstrides : ssize_t*
    Pointer to 3 integers defining the strides of the `out` array.
bins : ssize_t*
    Pointer to the output of logbins() used to bin cross correlation curves.
    The last item should be half of the internal buffer shape[2].
    If NULL, the first `nbins` samples from the cross correlation functions are
    returned.
nbins : ssize_t
    Size of `bins` array. Must be less or equal than half size of the time
    axis.
autocorr : int (bool)
    If True, also calculate autocorrelation functions.
filter : double
    Factor to use for simple exponential smoothing of log-binned cross
    correlation curves.
nthreads : int
    Number of OpenMP threads to use for parallelizing loops.
    Set to zero for OpenMP default.

*/
template <typename Ti, typename To>
void
yxt::apcf(
    const Ti *data,
    const ssize_t *strides,
    To *out,
    const ssize_t *outstrides,
    const ssize_t *bins,
    const ssize_t nbins,
    const int autocorr,
    const double filter,
    const int nthreads)
{
    if ((data == NULL) || (strides == NULL) || (out == NULL) ||
        (outstrides == NULL)) {
        throw ICS_VALUE_ERROR;
    }

    if (validate_bins(bins, nbins, tsize_ / 2) != ICS_OK) {
        throw ICS_VALUE_ERROR;
    }

    // initialize MKL for 1D in-place DFT
    DFTI_DESCRIPTOR_HANDLE dfti_handle;
    MKL_LONG status;
    status = DftiCreateDescriptor(
        &dfti_handle, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)tsize_);
    if (status)
        throw status;
    status = DftiSetValue(
        dfti_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle, DFTI_PLACEMENT, DFTI_INPLACE);
    if (status)
        throw status;
    status = DftiCommitDescriptor(dfti_handle);
    if (status)
        throw status;

    // disable MKL multi-threading to prevent oversubscription
    const int mkl_threads = mkl_get_max_threads();
    mkl_set_num_threads(1);

    const ssize_t xcorr = (autocorr == ICS_FALSE) ? 1 : 0;

#pragma omp parallel num_threads(nthreads)
    {
        // allocate thread local buffers
        double *a =
            (double *)mkl_malloc(shape_[2] * sizeof(double), MKL_ALIGN);

        if (data != NULL) {
            // copy forward DFT of input time series to buffer
#pragma omp for
            for (ssize_t x = 0; x < xsize_; x++) {
                double *pbuff = a_ + x * strides_[1];
                char *pdata = (char *)data + x * strides[0];
                for (ssize_t t = 0; t < tsize_; t++) {
                    pbuff[t] = (double)*(Ti *)pdata;
                    pdata += strides[1];
                }
                DftiComputeForward(dfti_handle, pbuff);
                pbuff += strides_[1];
            }
        }

        // initialize output
        // ...

        // loop over combinations of detectors in airy disk
#pragma omp for
        for (ssize_t i = 0; i < triangular_number(xsize_, autocorr); i++) {
            if (a == NULL)
                continue;

            ssize_t i0, i1;  // indicees of detectors
            triangular_number_coordinates(xsize_, i, &i0, &i1, autocorr);
            // printf("%ti, %ti, %ti\n", i, i0, i1);

            To *pout = (To *)(
                (char *)out + i0 * outstrides[0] + (i1 - xcorr) * outstrides[1]
            );
            const double *p0 = a_ + (i0 * strides_[1] * strides_[2]);
            const double *p1 = a_ + (i1 * strides_[1] * strides_[2]);
            const double scale = 1.0 / (p0[0] * p1[0]);

            // multiply detector0 DFT by complex conjugate of detector1 DFT and
            // store in a
            complex_multiply(a, p0, p1, tsize_ + 2);
            // compute invers DFT
            DftiComputeBackward(dfti_handle, a);
            // average, normalize, and smooth first half of cross correlation
            // curve
            anscf(
                a,
                pout,
                outstrides[2],
                bins,
                nbins,
                scale,
                -1.0,
                filter,
                true);

            if (i0 != i1) {
                // reverse, average, normalize, and smooth second half of cross
                // correlation curve
                pout = (To *)(
                    (char *)out + i1 * outstrides[0] + i0 * outstrides[1]
                );
                anscf(
                    a,
                    pout,
                    outstrides[2],
                    bins,
                    nbins,
                    scale,
                    -1.0,
                    filter,
                    true,
                    tsize_);
            }
        }

        // de-allocate thread local buffers
        mkl_free(a);
    }

    DftiFreeDescriptor(&dfti_handle);
    mkl_set_num_threads(mkl_threads);
}

/* In-place subtract image average and add total average */
template <typename T>
void
yxt_subtract_immobile(
    T *data, const ssize_t *shape, const ssize_t *strides, const int nthreads)
{
    if ((data == NULL) || (shape == NULL) || (strides == NULL)) {
        throw ICS_VALUE_ERROR;
    }

    const ssize_t sizey = shape[0];
    const ssize_t sizex = shape[1];
    const ssize_t sizet = shape[2];
    const ssize_t sizeyx = shape[0] * shape[1];

    const ssize_t strides0 = strides[0];
    const ssize_t strides1 = strides[1];
    const ssize_t strides2 = strides[2];

    // average image
    int64_t sumyxt = 0;
    int64_t *sumyx =
        (int64_t *)mkl_malloc(sizeyx * sizeof(int64_t), MKL_ALIGN);
    if (sumyx == NULL) {
        throw ICS_MEMORY_ERROR;
    }
    for (ssize_t i = 0; i < sizeyx; i++) {
        sumyx[i] = 0;
    }

    if ((strides0 < strides2) && (strides1 < strides2)) {
        // common case of tyx series

#pragma omp parallel num_threads(nthreads)
        {
            // thread local buffer
            int64_t *sumyx_ =
                (int64_t *)mkl_malloc(sizeyx * sizeof(int64_t), MKL_ALIGN);
            if (sumyx_ == NULL) {
                // TODO
            }
            for (ssize_t i = 0; i < sizeyx; i++) {
                sumyx_[i] = 0;
            }

            // sum intensities
#pragma omp for
            for (ssize_t t = 0; t < sizet; t++) {
                int64_t *ps = sumyx_;
                char *py = (char *)data + t * strides2;
                for (ssize_t y = 0; y < sizey; y++) {
                    char *px = py + y * strides0;
                    for (ssize_t x = 0; x < sizex; x++) {
                        *ps++ += (int64_t)(*(T *)px);
                        px += strides1;
                    }
                }
            }

            // reduce to single image, one thread at a time; then wait for all
            // threads
#pragma omp critical
            {
                for (ssize_t i = 0; i < sizeyx; i++) {
                    sumyxt += sumyx_[i];
                    sumyx[i] += sumyx_[i];
                }
            }
#pragma omp barrier

            // average image in single thread; threading overhead?
#pragma omp single
            {
                if ((sumyxt < 9007199254740992L) &&
                    (sumyxt > -9007199254740992L)) {
                    // use double precision for integers less than 2^53 bits
                    const double sumd =
                        (double)sumyxt / (double)(sizeyx * sizet);
                    // #pragma omp for
                    for (ssize_t i = 0; i < sizeyx; i++) {
                        const double d =
                            sumd - (double)sumyx[i] / (double)sizet;
                        sumyx[i] = static_cast<int64_t>(std::round(d));
                    }
                }
                else {
                    // use integer division for large integers
                    const int64_t sumd = sumyxt / (sizeyx * sizet);
                    // #pragma omp for
                    for (ssize_t i = 0; i < sizeyx; i++) {
                        sumyx[i] = sumd - sumyx[i] / sizet;
                    }
                }
            }

            // subtract average image
#pragma omp for
            for (ssize_t t = 0; t < sizet; t++) {
                int64_t *ps = sumyx;
                char *py = (char *)data + t * strides2;
                for (ssize_t y = 0; y < sizey; y++) {
                    char *px = py + y * strides0;
                    for (ssize_t x = 0; x < sizex; x++) {
                        add_int((T *)px, *ps++);
                        px += strides1;
                    }
                }
            }

            mkl_free(sumyx_);
        }
    }
    else {
        // yxt stack
#pragma omp parallel num_threads(nthreads)
        {
            // sum intensities
#pragma omp for
            for (ssize_t y = 0; y < sizey; y++) {
                int64_t *ps = sumyx + y * sizex;
                char *px = (char *)data + y * strides0;
                for (ssize_t x = 0; x < sizex; x++) {
                    char *pt = px + x * strides1;
                    int64_t s = 0;
                    for (ssize_t t = 0; t < sizet; t++) {
                        s += (int64_t)(*(T *)pt);
                        pt += strides2;
                    }
                    *ps++ = s;
                }
            }

            // average image in single thread; threading overhead?
#pragma omp single
            {
                for (ssize_t i = 0; i < sizeyx; i++) {
                    sumyxt += sumyx[i];
                }

                if ((sumyxt < 9007199254740992L) &&
                    (sumyxt > -9007199254740992L)) {
                    // use double precision for integers less than 2^53 bits
                    const double sumd =
                        (double)sumyxt / (double)(sizeyx * sizet);
                    // #pragma omp for
                    for (ssize_t i = 0; i < sizeyx; i++) {
                        const double d =
                            sumd - (double)sumyx[i] / (double)sizet;
                        sumyx[i] = static_cast<int64_t>(std::round(d));
                    }
                }
                else {
                    // use integer division for large integers
                    const int64_t sumd = sumyxt / (sizeyx * sizet);
                    // #pragma omp for
                    for (ssize_t i = 0; i < sizeyx; i++) {
                        sumyx[i] = sumd - sumyx[i] / sizet;
                    }
                }
            }

            // subtract average image
#pragma omp for
            for (ssize_t y = 0; y < sizey; y++) {
                int64_t *ps = sumyx + y * sizex;
                char *px = (char *)data + y * strides0;
                for (ssize_t x = 0; x < sizex; x++) {
                    const int64_t s = *ps;
                    char *pt = px + x * strides1;
                    for (ssize_t t = 0; t < sizet; t++) {
                        add_int((T *)pt, s);
                        pt += strides2;
                    }
                    ps++;
                }
            }
        }
    }
    mkl_free(sumyx);
}

/* In-place subtract smoothed time series, equalizate variance, and add
average.

Depending on filter size, samples at the beginning and end of the time series
should be discarded.

Parameters
----------

data : Ti*
    Pointer to input 3D array.
    If NULL, the internal buffer is re-used.
shape : ssize_t*
    Pointer to 3 integers defining the size of the data array in y, x, and t
    dimensions.
strides : ssize_t*
    Pointer to 3 integers defining the strides of the data array in y, x, and
    t dimensions.
    Strides are the number of bytes required in each dimension to advance from
    one item to the next within the dimension.
mean : double*
    Pointer to 2D output array of mean intensities along time axis.
    Must be large enough to hold shape[0] * shape[1] * sizeof(double).
meanstrides : ssize_t*
    Pointer to 2 integers defining the strides of the `mean` array.
filter : double
    Factor to use for smoothing of time axis.
nthreads : int
    Number of OpenMP threads to use for parallelizing loop over yx dimensions.
    Set to zero for OpenMP default.

*/
template <typename T>
void
yxt_correct_bleaching(
    T *data,
    const ssize_t *shape,
    const ssize_t *strides,
    double *mean,
    const ssize_t *meanstrides,
    const double filter,
    const int nthreads)
{
    if ((data == NULL) || (shape == NULL) || (strides == NULL) ||
        ((meanstrides == NULL) && (mean != NULL))) {
        throw ICS_VALUE_ERROR;
    }
    if ((filter <= 0.0) || (filter >= 1.0) || (shape[0] < 1) ||
        (shape[1] < 1) || (shape[2] < 1)) {
        return;
    }

    const ssize_t sizey = shape[0];
    const ssize_t sizex = shape[1];
    const ssize_t sizet = shape[2];
    const ssize_t stridet = strides[2];
    const double filter1 = 1.0 - filter;

#pragma omp parallel num_threads(nthreads)
    {
        // thread local buffer
        double *a = (double *)mkl_malloc(sizet * sizeof(double), MKL_ALIGN);
        if (a == NULL) {
            throw ICS_MEMORY_ERROR;
        }

        // #pragma omp for collapse(2)
        // for (ssize_t y = 0; y < sizey; y++) {
        //       for (ssize_t x = 0; x < sizex; x++) {

#pragma omp for
        for (ssize_t yx = 0; yx < sizey * sizex; yx++) {
            const ssize_t y = yx / sizex;
            const ssize_t x = yx % sizex;
            char *pdata = (char *)data + y * strides[0] + x * strides[1];
            int64_t sumd = (int64_t)(*((T *)pdata));
            a[0] = (double)sumd;
            for (ssize_t t = 1; t < sizet; t++) {
                pdata += stridet;
                const T d = *((T *)pdata);
                sumd += d;
                a[t] = (double)d * filter1 + a[t - 1] * filter;
            }
            for (ssize_t t = sizet - 2; t >= 0; t--) {
                a[t] = a[t] * filter1 + a[t + 1] * filter;
            }
            // use double precision for integers less than 2^53 bits, else use
            // integer division
            const double meand =
                ((sumd < 9007199254740992L) && (sumd > -9007199254740992L))
                    ? (double)sumd / (double)sizet
                    : (double)(sumd / (int64_t)sizet);
            if (mean != NULL) {
                *((double*)(
                    (char *)mean + y * meanstrides[0] + x * meanstrides[1])
                ) = meand;
            }
            pdata = (char *)data + y * strides[0] + x * strides[1];
            for (ssize_t t = 0; t < sizet; t++) {
                const double d = (double)(*(T *)pdata);
                const double deficit = sqrt(fabs(meand / a[t]));
                add_int(
                    (T *)pdata, std::round(deficit * (d - a[t]) + meand - d));
                pdata += stridet;
            }
        }

        mkl_free(a);
    }
}

/* Calculate 1D DFTs along last dimension of image of time series.

Parameters
----------

data : Ti*
    Pointer to input 3D array (yxt). The DFT is computed for each pixel over
    the last dimension.
shape : ssize_t*
    Pointer to 3 integers defining the size of the data array in y, x, and t
    dimensions.
    shape[2] must be a power of 2.
strides : ssize_t*
    Pointer to 3 integers defining the strides of the data array in y, x, and
    t dimensions.
    Strides are the number of bytes required in each dimension to advance from
    one item
    to the next within the dimension.
out : To*
    Pointer to 3D output array of summed intensities and real & imaginary
    fourier samples for each pixel.
    The returned values are Fdc, F1real, F1imag, F2real, F2imag, etc.
    Must be large enough to hold
    outshape[0] * outshape[1] * outshape[2] * sizeof(To).
outshape : ssize_t*
    Pointer to 3 integers defining the size of the output array in y, x, and
    "fourier" dimensions.
    outshape[0] and outshape[1] must be same as shape[0] and shape[1].
    outshape[2] determines the number of values returned from the DFT.
outstrides : ssize_t*
    Pointer to 3 integers defining the strides of the `out` array.
nthreads : int
    Number of OpenMP threads to use for parallelizing loop over yx dimensions.
    Set to zero for OpenMP default.

*/
template <typename Ti, typename To>
void
yxt_dft(
    const Ti *data,
    const ssize_t *shape,
    const ssize_t *strides,
    To *out,
    const ssize_t *outshape,
    const ssize_t *outstrides,
    const int nthreads)
{
    if ((data == NULL) || (shape == NULL) || (strides == NULL) ||
        (out == NULL) || (outshape == NULL) || (outstrides == NULL)) {
        throw ICS_VALUE_ERROR;
    }
    if ((shape[0] < 1) || (shape[1] < 1) || (shape[2] < 1) ||
        (outshape[2] < 1)) {
        return;
    }

    if ((outshape[0] != shape[0]) || (outshape[1] != shape[1]) ||
        (outshape[2] > shape[2] + 1)) {
        throw ICS_VALUE_ERROR;
    }

    if (!ispow2(shape[2])) {
        throw ICS_VALUE_ERROR;
    }

    const ssize_t sizey = shape[0];
    const ssize_t sizex = shape[1];
    const ssize_t sizet = shape[2];
    const ssize_t sizef = outshape[2];
    const ssize_t stridet = strides[2];

    // initialize MKL for 1D in-place DFT
    DFTI_DESCRIPTOR_HANDLE dfti_handle;
    MKL_LONG status;
    status = DftiCreateDescriptor(
        &dfti_handle, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)sizet);
    if (status)
        throw status;
    status = DftiSetValue(
        dfti_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle, DFTI_PLACEMENT, DFTI_INPLACE);
    if (status)
        throw status;
    status = DftiCommitDescriptor(dfti_handle);
    if (status)
        throw status;

    // disable MKL multi-threading to prevent oversubscription
    const int mkl_threads = mkl_get_max_threads();
    mkl_set_num_threads(1);

#pragma omp parallel num_threads(nthreads)
    {
        // allocate thread local buffers
        double *a = (double *)mkl_malloc(
            sizet * sizeof(double) + MKL_ALIGN_D, MKL_ALIGN);

#pragma omp for
        for (ssize_t y = 0; y < sizey; y++) {
            for (ssize_t x = 0; x < sizex; x++) {
                // copy data to buffer
                char *ptr = (char *)data + y * strides[0] + x * strides[1];
                for (ssize_t t = 0; t < sizet; t++) {
                    a[t] = (double)*(Ti *)ptr;
                    ptr += strides[2];
                }
                DftiComputeForward(dfti_handle, a);
                // copy requested number of values to output
                ptr = (char *)out + y * outstrides[0] + x * outstrides[1];
                *(To *)ptr = (To)a[0];
                for (ssize_t t = 2; t < sizef + 1; t++) {
                    ptr += outstrides[2];
                    *(To *)ptr = (To)a[t];
                }
            }
        }

        // de-allocate thread local buffers
        mkl_free(a);
    }

    DftiFreeDescriptor(&dfti_handle);
    mkl_set_num_threads(mkl_threads);
}

/** yxt class C API **/

yxt_handle
yxt_new(ssize_t *shape)
{
    try {
        return reinterpret_cast<yxt_handle>(new yxt(shape));
    }
    catch (...) {
        return NULL;
    }
}

void
yxt_del(yxt_handle handle)
{
    try {
        delete reinterpret_cast<yxt_handle>(handle);
    }
    catch (...) {
        ;
    }
}

double *
yxt_get_buffer(yxt_handle handle, ssize_t *shape, ssize_t *strides)
{
    return reinterpret_cast<yxt_handle>(handle)->yxt_get_buffer(
        shape, strides);
}

#define YXT_IPCF(function_name, input_type, output_type) \
    int function_name(                                   \
        yxt_handle handle,                               \
        input_type *data,                                \
        input_type *channel,                             \
        ssize_t *strides,                                \
        output_type *out,                                \
        ssize_t *outstrides,                             \
        ssize_t *points,                                 \
        ssize_t npoints,                                 \
        ssize_t *bins,                                   \
        ssize_t nbins,                                   \
        double threshold,                                \
        double filter,                                   \
        int nthreads)                                    \
    {                                                    \
        try {                                            \
            reinterpret_cast<yxt_handle>(handle)->ipcf(  \
                data,                                    \
                channel,                                 \
                strides,                                 \
                out,                                     \
                outstrides,                              \
                points,                                  \
                npoints,                                 \
                bins,                                    \
                nbins,                                   \
                threshold,                               \
                filter,                                  \
                nthreads);                               \
        }                                                \
        catch (const int e) {                            \
            return e;                                    \
        }                                                \
        catch (...) {                                    \
            return ICS_ERROR;                            \
        }                                                \
        return ICS_OK;                                   \
    }

YXT_IPCF(yxt_ipcf_df, double, float)
YXT_IPCF(yxt_ipcf_ff, float, float)
YXT_IPCF(yxt_ipcf_if, int, float)
YXT_IPCF(yxt_ipcf_hf, int16_t, float)
YXT_IPCF(yxt_ipcf_Hf, uint16_t, float)
YXT_IPCF(yxt_ipcf_dd, double, double)
YXT_IPCF(yxt_ipcf_fd, float, double)
YXT_IPCF(yxt_ipcf_id, int, double)
YXT_IPCF(yxt_ipcf_hd, int16_t, double)
YXT_IPCF(yxt_ipcf_Hd, uint16_t, double)

// Compatibility for SimFCS. Deprecated. Note 'float filter'.
int
yxt_crosscorrelate_hf(
    yxt_handle handle,
    int16_t *data,
    int16_t *channel,
    ssize_t *strides,
    float *out,
    ssize_t *outstrides,
    ssize_t *points,
    ssize_t npoints,
    ssize_t *bins,
    ssize_t nbins,
    double threshold,
    float filter,
    int nthreads)
{
    try {
        reinterpret_cast<yxt_handle>(handle)->ipcf(
            data,
            channel,
            strides,
            out,
            outstrides,
            points,
            npoints,
            bins,
            nbins,
            threshold,
            filter,
            nthreads);
    }
    catch (const int e) {
        return e;
    }
    catch (...) {
        return ICS_ERROR;
    }
    return ICS_OK;
}

#define YXT_IMSD(function_name, input_type, output_type) \
    int function_name(                                   \
        yxt_handle handle,                               \
        input_type *data,                                \
        ssize_t *strides,                                \
        input_type *data1,                               \
        ssize_t *strides1,                               \
        int32_t *mask,                                   \
        ssize_t *maskstrides,                            \
        int32_t maskmode,                                \
        output_type *out,                                \
        ssize_t *outstrides,                             \
        ssize_t *block,                                  \
        ssize_t *bins,                                   \
        ssize_t nbins,                                   \
        double filter,                                   \
        int nthreads)                                    \
    {                                                    \
        try {                                            \
            reinterpret_cast<yxt_handle>(handle)->imsd(  \
                data,                                    \
                strides,                                 \
                data1,                                   \
                strides1,                                \
                mask,                                    \
                maskstrides,                             \
                maskmode,                                \
                out,                                     \
                outstrides,                              \
                block,                                   \
                bins,                                    \
                nbins,                                   \
                filter,                                  \
                nthreads);                               \
        }                                                \
        catch (const int e) {                            \
            return e;                                    \
        }                                                \
        catch (...) {                                    \
            return ICS_ERROR;                            \
        }                                                \
        return ICS_OK;                                   \
    }

YXT_IMSD(yxt_imsd_df, double, float)
YXT_IMSD(yxt_imsd_ff, float, float)
YXT_IMSD(yxt_imsd_if, int, float)
YXT_IMSD(yxt_imsd_hf, int16_t, float)
YXT_IMSD(yxt_imsd_Hf, uint16_t, float)
YXT_IMSD(yxt_imsd_dd, double, double)
YXT_IMSD(yxt_imsd_fd, float, double)
YXT_IMSD(yxt_imsd_id, int, double)
YXT_IMSD(yxt_imsd_hd, int16_t, double)
YXT_IMSD(yxt_imsd_Hd, uint16_t, double)

#define YXT_LSTICS(function_name, input_type, output_type) \
    int function_name(                                     \
        yxt_handle handle,                                 \
        input_type *data,                                  \
        ssize_t *strides,                                  \
        input_type *data1,                                 \
        ssize_t *strides1,                                 \
        int32_t *mask,                                     \
        ssize_t *maskstrides,                              \
        int32_t maskmode,                                  \
        output_type *out,                                  \
        ssize_t *outstrides,                               \
        ssize_t *lines,                                    \
        ssize_t *linesshape,                               \
        ssize_t *block,                                    \
        ssize_t *bins,                                     \
        ssize_t nbins,                                     \
        double filter,                                     \
        int nthreads)                                      \
    {                                                      \
        try {                                              \
            reinterpret_cast<yxt_handle>(handle)->lstics(  \
                data,                                      \
                strides,                                   \
                data1,                                     \
                strides1,                                  \
                mask,                                      \
                maskstrides,                               \
                maskmode,                                  \
                out,                                       \
                outstrides,                                \
                lines,                                     \
                linesshape,                                \
                block,                                     \
                bins,                                      \
                nbins,                                     \
                filter,                                    \
                nthreads);                                 \
        }                                                  \
        catch (const int e) {                              \
            return e;                                      \
        }                                                  \
        catch (...) {                                      \
            return ICS_ERROR;                              \
        }                                                  \
        return ICS_OK;                                     \
    }

YXT_LSTICS(yxt_lstics_df, double, float)
YXT_LSTICS(yxt_lstics_ff, float, float)
YXT_LSTICS(yxt_lstics_if, int, float)
YXT_LSTICS(yxt_lstics_hf, int16_t, float)
YXT_LSTICS(yxt_lstics_Hf, uint16_t, float)
YXT_LSTICS(yxt_lstics_dd, double, double)
YXT_LSTICS(yxt_lstics_fd, float, double)
YXT_LSTICS(yxt_lstics_id, int, double)
YXT_LSTICS(yxt_lstics_hd, int16_t, double)
YXT_LSTICS(yxt_lstics_Hd, uint16_t, double)

#define YXT_APCF(function_name, input_type, output_type) \
    int function_name(                                   \
        yxt_handle handle,                               \
        input_type *data,                                \
        ssize_t *strides,                                \
        output_type *out,                                \
        ssize_t *outstrides,                             \
        ssize_t *bins,                                   \
        ssize_t nbins,                                   \
        int autocorr,                                    \
        double filter,                                   \
        int nthreads)                                    \
    {                                                    \
        try {                                            \
            reinterpret_cast<yxt_handle>(handle)->apcf(  \
                data,                                    \
                strides,                                 \
                out,                                     \
                outstrides,                              \
                bins,                                    \
                nbins,                                   \
                autocorr,                                \
                filter,                                  \
                nthreads);                               \
        }                                                \
        catch (const int e) {                            \
            return e;                                    \
        }                                                \
        catch (...) {                                    \
            return ICS_ERROR;                            \
        }                                                \
        return ICS_OK;                                   \
    }

YXT_APCF(yxt_apcf_df, double, float)
YXT_APCF(yxt_apcf_ff, float, float)
YXT_APCF(yxt_apcf_if, int, float)
YXT_APCF(yxt_apcf_hf, int16_t, float)
YXT_APCF(yxt_apcf_Hf, uint16_t, float)
YXT_APCF(yxt_apcf_dd, double, double)
YXT_APCF(yxt_apcf_fd, float, double)
YXT_APCF(yxt_apcf_id, int, double)
YXT_APCF(yxt_apcf_hd, int16_t, double)
YXT_APCF(yxt_apcf_Hd, uint16_t, double)

#define YXT_SUBIMBL(function_name, input_type)                            \
    int function_name(                                                    \
        input_type *data, ssize_t *shape, ssize_t *strides, int nthreads) \
    {                                                                     \
        try {                                                             \
            yxt_subtract_immobile(data, shape, strides, nthreads);        \
        }                                                                 \
        catch (const int e) {                                             \
            return e;                                                     \
        }                                                                 \
        catch (...) {                                                     \
            return ICS_ERROR;                                             \
        }                                                                 \
        return ICS_OK;                                                    \
    }

// YXT_SUBIMBL(yxt_subtract_immobile_d, double)
// YXT_SUBIMBL(yxt_subtract_immobile_f, float)
YXT_SUBIMBL(yxt_subtract_immobile_i, int32_t)
YXT_SUBIMBL(yxt_subtract_immobile_h, int16_t)
YXT_SUBIMBL(yxt_subtract_immobile_H, uint16_t)

#define YXT_CORBLCH(function_name, input_type)                              \
    int function_name(                                                      \
        input_type *data,                                                   \
        ssize_t *shape,                                                     \
        ssize_t *strides,                                                   \
        double *mean,                                                       \
        ssize_t *meanstrides,                                               \
        double filter,                                                      \
        int nthreads)                                                       \
    {                                                                       \
        try {                                                               \
            yxt_correct_bleaching(                                          \
                data, shape, strides, mean, meanstrides, filter, nthreads); \
        }                                                                   \
        catch (const int e) {                                               \
            return e;                                                       \
        }                                                                   \
        catch (...) {                                                       \
            return ICS_ERROR;                                               \
        }                                                                   \
        return ICS_OK;                                                      \
    }

YXT_CORBLCH(yxt_correct_bleaching_i, int32_t)
YXT_CORBLCH(yxt_correct_bleaching_h, int16_t)
YXT_CORBLCH(yxt_correct_bleaching_H, uint16_t)

#define YXT_DFT(function_name, input_type, output_type)                     \
    int function_name(                                                      \
        input_type *data,                                                   \
        ssize_t *shape,                                                     \
        ssize_t *strides,                                                   \
        output_type *out,                                                   \
        ssize_t *outshape,                                                  \
        ssize_t *outstrides,                                                \
        int nthreads)                                                       \
    {                                                                       \
        try {                                                               \
            yxt_dft(                                                        \
                data, shape, strides, out, outshape, outstrides, nthreads); \
        }                                                                   \
        catch (const int e) {                                               \
            return e;                                                       \
        }                                                                   \
        catch (...) {                                                       \
            return ICS_ERROR;                                               \
        }                                                                   \
        return ICS_OK;                                                      \
    }

YXT_DFT(yxt_dft_ff, float, float)
YXT_DFT(yxt_dft_if, int, float)
YXT_DFT(yxt_dft_hf, int16_t, float)
YXT_DFT(yxt_dft_Hf, uint16_t, float)
YXT_DFT(yxt_dft_dd, double, double)
YXT_DFT(yxt_dft_id, int, double)
YXT_DFT(yxt_dft_hd, int16_t, double)
YXT_DFT(yxt_dft_Hd, uint16_t, double)
