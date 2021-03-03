/* rfft.cpp

Implementation of the rfft1d, rfft2d, and rfft3d classes for the ICS library.

The rfft#d classes implement 1D, 2D, and 3D auto- and cross-correlation using
the Intel MKL library.

Copyright (c) 2016-2021, Christoph Gohlke
This source code is distributed under the BSD 3-Clause license.

Refer to the header file 'ics.h' for documentation and license.

*/

#include <math.h>
#include <algorithm>

#include "ics.h"
#include "ics.hpp"

/** rfft1d C++ API **/

/* Class constructor */
rfft1d::rfft1d(const ssize_t shape0, const int mode)
{
    if ((shape0 < 2) || (shape0 > INT32_MAX - 2) || !ispow2(shape0)) {
        throw ICS_VALUE_ERROR;
    }

    dfti_handle_ = 0;
    mode_ = mode;
    n0_ = shape0;

    a_ = NULL;
    b_ = NULL;
    suma_ = 0.0;

    alloc_();

    // initialize MKL FFT descriptor for inplace 1D real DFT
    MKL_LONG status;
    status = DftiCreateDescriptor(
        &dfti_handle_, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)n0_);
    if (status)
        throw status;
    status = DftiSetValue(
        dfti_handle_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle_, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle_, DFTI_PLACEMENT, DFTI_INPLACE);
    if (status)
        throw status;
    status = DftiCommitDescriptor(dfti_handle_);
    if (status)
        throw status;
}

/* Class destructor */
rfft1d::~rfft1d()
{
    free_();

    if (dfti_handle_) {
        DftiFreeDescriptor(&dfti_handle_);
        dfti_handle_ = NULL;
    }
}

/* Allocate internal buffers */
void
rfft1d::alloc_()
{
    a_ = (double *)mkl_malloc((n0_ + MKL_ALIGN_D) * sizeof(double), MKL_ALIGN);
    if (a_ == NULL) {
        throw ICS_MEMORY_ERROR;
    }
    if (mode_ & ICS_MODE_CC) {
        b_ = (double *)mkl_malloc(
            (n0_ + MKL_ALIGN_D) * sizeof(double), MKL_ALIGN);
        if (b_ == NULL) {
            free_();
            throw ICS_MEMORY_ERROR;
        }
    }
}

/* Free internal buffers */
void
rfft1d::free_()
{
    if (a_ != NULL)
        mkl_free(a_);
    if (b_ != NULL)
        mkl_free(b_);

    a_ = NULL;
    b_ = NULL;
}

/* 1D auto correlation */
template <typename Ti, typename To>
void
rfft1d::autocorrelate(const Ti *data, To *out, const ssize_t *strides)
{
    copy_input_(a_, data, strides);
    DftiComputeForward(dfti_handle_, a_);
    const double sum = a_[0];

    complex_multiply(a_, n0_ + MKL_ALIGN_D);

    DftiComputeBackward(dfti_handle_, a_);

    if (mode_ & ICS_MODE_FCS) {
        copy_output_(out, a_, 1.0 / sum / sum, -1.0);
    }
    else {
        copy_output_(out, a_, 1.0 / n0_, 0.0);
    }
}

/* 1D cross correlation */
template <typename Ti, typename To>
void
rfft1d::crosscorrelate(
    const Ti *data0,
    const Ti *data1,
    To *out,
    const ssize_t *strides0,
    const ssize_t *strides1)
{
    // allocate b_ if necessary
    if (b_ == NULL) {
        b_ = (double *)mkl_malloc(
            (n0_ + MKL_ALIGN_D) * sizeof(double), MKL_ALIGN);
        if (b_ == NULL) {
            throw ICS_MEMORY_ERROR;
        }
    }

    if (data0 != NULL) {
        copy_input_(a_, data0, strides0);
        DftiComputeForward(dfti_handle_, a_);
        suma_ = a_[0];
    }

    copy_input_(b_, data1, strides1);
    DftiComputeForward(dfti_handle_, b_);
    const double sumb = b_[0];

    complex_multiply(a_, b_, n0_ + MKL_ALIGN_D);

    DftiComputeBackward(dfti_handle_, b_);

    if (mode_ & ICS_MODE_FCS) {
        copy_output_(out, b_, 1.0 / suma_ / sumb, -1.0);
    }
    else {
        copy_output_(out, b_, 1.0 / n0_, 0.0);
    }
}

/* Copy input array to internal buffer */
template <typename T>
void
rfft1d::copy_input_(double *a, const T *data, const ssize_t *strides)
{
    if (data == NULL) {
        throw ICS_VALUE_ERROR;
    }

    const ssize_t s0 = (strides == NULL) ? sizeof(T) : strides[0];
    char *pdata = (char *)data;

    if (s0 == sizeof(T)) {
        for (ssize_t i = 0; i < n0_; i++) {
            a[i] = (double)data[i];
        }
    }
    else {
        for (ssize_t i = 0; i < n0_; i++) {
            a[i] = (double)*(T *)pdata;
            pdata += s0;
        }
    }
}

/* Shift and scale internal buffer to output array */
template <typename T>
void
rfft1d::copy_output_(
    T *out, double *a, const double scale, const double offset)
{
    if (out == NULL) {
        throw ICS_VALUE_ERROR;
    }

    if (mode_ & ICS_AXIS0) {
        // do not center any axis
        copy(out, a, n0_, scale, offset);
    }
    else {
        // center all axis
        const ssize_t h0 = n0_ / 2;
        copy(out, a + h0, h0, scale, offset);
        copy(out + h0, a, h0, scale, offset);
    }
}

/** rfft2d C++ API **/

/* Class constructor */
rfft2d::rfft2d(const ssize_t shape0, const ssize_t shape1, const int mode)
{
    if ((shape0 < 2) || (shape0 > INT32_MAX) || !ispow2(shape0) ||
        (shape1 < 2) || (shape1 > INT32_MAX - 2) || !ispow2(shape1)) {
        throw ICS_VALUE_ERROR;
    }

    dfti_handle_ = 0;
    mode_ = mode;
    n0_ = shape0;
    n1_ = shape1;
    suma_ = 0.0;

    shape_[0] = n0_;
    shape_[1] = n1_ + MKL_ALIGN_D;

    size_ = shape_[0] * shape_[1];

    rstrides_[0] = 0;
    rstrides_[1] = (MKL_LONG)shape_[1];
    rstrides_[2] = 1;
    cstrides_[0] = 0;
    cstrides_[1] = rstrides_[1] / 2;
    cstrides_[2] = 1;

    a_ = NULL;
    b_ = NULL;

    alloc_();

    MKL_LONG status;
    MKL_LONG length[] = {(MKL_LONG)n0_, (MKL_LONG)n1_};

    status =
        DftiCreateDescriptor(&dfti_handle_, DFTI_DOUBLE, DFTI_REAL, 2, length);
    if (status)
        throw status;
    status = DftiSetValue(
        dfti_handle_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle_, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle_, DFTI_PLACEMENT, DFTI_INPLACE);
    if (status)
        throw status;
    set_forward_strides_();
}

/* Class destructor */
rfft2d::~rfft2d()
{
    free_();

    if (dfti_handle_) {
        DftiFreeDescriptor(&dfti_handle_);
        dfti_handle_ = NULL;
    }
}

/* Allocate internal buffers */
void
rfft2d::alloc_()
{
    // input/output data
    a_ = (double *)mkl_malloc(size_ * sizeof(double), MKL_ALIGN);
    if (a_ == NULL) {
        throw ICS_MEMORY_ERROR;
    }
    if (mode_ & ICS_MODE_CC) {
        b_ = (double *)mkl_malloc(size_ * sizeof(double), MKL_ALIGN);
        if (b_ == NULL) {
            free_();
            throw ICS_MEMORY_ERROR;
        }
    }
}

/* Free internal buffers */
void
rfft2d::free_()
{
    if (a_ != NULL)
        mkl_free(a_);
    if (b_ != NULL)
        mkl_free(b_);

    a_ = NULL;
    b_ = NULL;
}

/* Configure DFTI descriptor for forward transform */
void
rfft2d::set_forward_strides_()
{
    MKL_LONG status;
    status = DftiSetValue(dfti_handle_, DFTI_INPUT_STRIDES, rstrides_);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle_, DFTI_OUTPUT_STRIDES, cstrides_);
    if (status)
        throw status;
    status = DftiCommitDescriptor(dfti_handle_);
    if (status)
        throw status;
}

/* Configure DFTI descriptor for backward transform */
void
rfft2d::set_backward_strides_()
{
    MKL_LONG status;
    status = DftiSetValue(dfti_handle_, DFTI_INPUT_STRIDES, cstrides_);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle_, DFTI_OUTPUT_STRIDES, rstrides_);
    if (status)
        throw status;
    status = DftiCommitDescriptor(dfti_handle_);
    if (status)
        throw status;
}

/* 2D auto correlation */
template <typename Ti, typename To>
void
rfft2d::autocorrelate(const Ti *data, To *out, const ssize_t *strides)
{
    copy_input_(a_, data, strides);
    set_forward_strides_();
    DftiComputeForward(dfti_handle_, a_);
    const double sum = a_[0];

    complex_multiply(a_, size_);

    set_backward_strides_();
    DftiComputeBackward(dfti_handle_, a_);

    if (mode_ & ICS_MODE_FCS) {
        copy_output_(out, a_, 1.0 / sum / sum, -1.0);
    }
    else {
        copy_output_(out, a_, 1.0 / (n0_ * n1_), 0.0);
    }
}

/* 2D cross correlation */
template <typename Ti, typename To>
void
rfft2d::crosscorrelate(
    const Ti *data0,
    const Ti *data1,
    To *out,
    const ssize_t *strides0,
    const ssize_t *strides1)
{
    /* allocate b_ if necessary */
    if (b_ == NULL) {
        b_ = (double *)mkl_malloc(size_ * sizeof(double), MKL_ALIGN);
        if (b_ == NULL) {
            throw ICS_MEMORY_ERROR;
        }
    }

    set_forward_strides_();

    if (data0 != NULL) {
        copy_input_(a_, data0, strides0);
        DftiComputeForward(dfti_handle_, a_);
        suma_ = a_[0];
    }

    copy_input_(b_, data1, strides1);
    DftiComputeForward(dfti_handle_, b_);
    const double sumb = b_[0];

    complex_multiply(a_, b_, size_);

    set_backward_strides_();
    DftiComputeBackward(dfti_handle_, b_);

    if (mode_ & ICS_MODE_FCS) {
        copy_output_(out, b_, 1.0 / suma_ / sumb, -1.0);
    }
    else {
        copy_output_(out, b_, 1.0 / (n0_ * n1_), 0.0);
    }
}

/* Shift and scale internal buffer to output array */
template <typename T>
void
rfft2d::copy_output_(
    T *out, double *a, const double scale, const double offset)
{
    T *pout = out;
    const ssize_t n0 = n0_;
    const ssize_t n1 = n1_;
    const ssize_t s0 = rstrides_[1];
    const ssize_t h0 = n0 / 2;
    const ssize_t h1 = n1 / 2;

    if (out == NULL) {
        throw ICS_VALUE_ERROR;
    }

    if ((mode_ & ICS_AXIS0) && (mode_ & ICS_AXIS1)) {
        // do not center any axis
        pout = out;
        for (ssize_t i = 0; i < n0; i++, pout += n1)
            copy(pout, a + i * s0, n1, scale, offset);
        return;
    }

    if (mode_ & ICS_AXIS0) {
        // do not center axis 0
        pout = out;
        for (ssize_t i = 0; i < n0; i++, pout += n1)
            copy(pout, a + i * s0 + h1, h1, scale, offset);
        pout = out + h1;
        for (ssize_t i = 0; i < n0; i++, pout += n1)
            copy(pout, a + i * s0, h1, scale, offset);
        return;
    }

    if (mode_ & ICS_AXIS1) {
        // do not center axis 1
        pout = out;
        for (ssize_t i = h0; i < n0; i++, pout += n1)
            copy(pout, a + i * s0, n1, scale, offset);
        for (ssize_t i = 0; i < h0; i++, pout += n1)
            copy(pout, a + i * s0, n1, scale, offset);
        return;
    }

    {
        // center all axis
        pout = out;
        for (ssize_t i = h0; i < n0; i++, pout += n1)
            copy(pout, a + i * s0 + h1, h1, scale, offset);
        for (ssize_t i = 0; i < h0; i++, pout += n1)
            copy(pout, a + i * s0 + h1, h1, scale, offset);
        pout = out + h1;
        for (ssize_t i = h0; i < n0; i++, pout += n1)
            copy(pout, a + i * s0, h1, scale, offset);
        for (ssize_t i = 0; i < h0; i++, pout += n1)
            copy(pout, a + i * s0, h1, scale, offset);
    }
}

/* Copy input array to internal buffer */
template <typename T>
void
rfft2d::copy_input_(double *a, const T *data, const ssize_t *strides)
{
    if (data == NULL) {
        throw ICS_VALUE_ERROR;
    }

    const ssize_t s0 = (strides == NULL) ? 0 : strides[0] - n1_ * strides[1];
    const ssize_t s1 = (strides == NULL) ? sizeof(T) : strides[1];
    char *pdata = (char *)data;

    for (ssize_t i = 0; i < n0_; i++) {
        for (ssize_t j = 0; j < n1_; j++) {
            *a++ = (double)(*(T *)pdata);
            pdata += s1;
        }
        for (ssize_t j = 0; j < MKL_ALIGN_D; j++) {
            *a++ = 0.0;
        }
        pdata += s0;
    }
}

/* Copy t-series of selected points from yxt input array to internal buffer */
void
rfft2d::copy_input_(
    double *a,
    const double *data,
    const ssize_t *strides,
    const ssize_t *points)
{
    if ((data == NULL) || (strides == NULL) || (points == NULL)) {
        throw ICS_VALUE_ERROR;
    }
    const ssize_t s = strides[2];

    for (ssize_t i = 0; i < n0_; i++) {  // loop over points
        double *pdata = (double *)data + (points[2 * i + 1] * strides[0]) +
                        (points[2 * i] * strides[1]);
        for (ssize_t j = 0; j < n1_; j++) {  // loop over time axis
            *a++ = *pdata;
            pdata += s;
        }
        for (ssize_t j = 0; j < MKL_ALIGN_D; j++) {
            *a++ = 0.0;
        }
    }
}

/* Special 2D auto correlation for line STICS */
template <typename To>
void
rfft2d::lstics(
    const double *data,
    const double *channel,
    const ssize_t *strides,  // double strides, not byte
    To *out,
    const ssize_t *outstrides,  // byte strides
    const ssize_t *line,
    const ssize_t *bins,
    const ssize_t nbins,
    const double filter)
{
    if ((data == NULL) || (out == NULL) || (strides == NULL) ||
        (outstrides == NULL) || (bins == NULL) || (line == NULL)) {
        throw ICS_VALUE_ERROR;
    }

    double scale = 0.0;

    set_forward_strides_();

    if ((channel != NULL) && (channel != data) && (b_ != NULL)) {
        copy_input_(a_, data, strides, line);
        DftiComputeForward(dfti_handle_, a_);
        copy_input_(b_, channel, strides, line);
        DftiComputeForward(dfti_handle_, b_);
        if ((a_[0] < 1e-9) || (b_[0] < 1e-9))
            return;
        scale = 1.0 / a_[0] / b_[0];
        complex_multiply(b_, a_, size_);
    }
    else {
        copy_input_(a_, data, strides, line);
        DftiComputeForward(dfti_handle_, a_);
        if (a_[0] < 1e-9)
            return;
        scale = 1.0 / a_[0] / a_[0];
        complex_multiply(a_, size_);
    }

    set_backward_strides_();
    DftiComputeBackward(dfti_handle_, a_);

    const ssize_t h0 = n0_ / 2;
    const ssize_t s0 = n1_ + MKL_ALIGN_D;

    // 0 0
    for (ssize_t x = 0, i = h0; i < n0_; x++, i++) {
        To *pout = (To *)((char *)out + x * outstrides[0]);
        anscf(
            a_ + i * s0,
            pout,
            outstrides[1],
            bins,
            nbins,
            scale,
            -1.0,
            filter,
            false);
    }
    // h0 0
    for (ssize_t x = 0, i = 0; i < h0; x++, i++) {
        To *pout = (To *)((char *)out + (x + h0) * outstrides[0]);
        anscf(
            a_ + i * s0,
            pout,
            outstrides[1],
            bins,
            nbins,
            scale,
            -1.0,
            filter,
            false);
    }
}

/** rfft3d C++ API **/

/* Class constructor */
rfft3d::rfft3d(
    const ssize_t shape0,
    const ssize_t shape1,
    const ssize_t shape2,
    const int mode)
{
    if ((shape0 < 2) || (shape0 > INT32_MAX) || !ispow2(shape0) ||
        (shape1 < 2) || (shape1 > INT32_MAX) || !ispow2(shape1) ||
        (shape2 < 2) || (shape2 > INT32_MAX - 2) || !ispow2(shape2)) {
        throw ICS_VALUE_ERROR;
    }

    dfti_handle_ = 0;
    mode_ = mode;
    n0_ = shape0;
    n1_ = shape1;
    n2_ = shape2;
    suma_ = 0.0;

    a_ = NULL;
    b_ = NULL;

    alloc_();

    MKL_LONG status;
    MKL_LONG length[] = {(MKL_LONG)n0_, (MKL_LONG)n1_, (MKL_LONG)n2_};

    status =
        DftiCreateDescriptor(&dfti_handle_, DFTI_DOUBLE, DFTI_REAL, 3, length);
    if (status)
        throw status;
    status = DftiSetValue(
        dfti_handle_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle_, DFTI_PACKED_FORMAT, DFTI_CCE_FORMAT);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle_, DFTI_PLACEMENT, DFTI_INPLACE);
    if (status)
        throw status;

    set_forward_strides_();
}

/* Class destructor */
rfft3d::~rfft3d()
{
    free_();

    if (dfti_handle_) {
        DftiFreeDescriptor(&dfti_handle_);
        dfti_handle_ = NULL;
    }
}

/* Allocate internal buffers */
void
rfft3d::alloc_()
{
    shape_[0] = n0_;
    shape_[1] = n1_;
    shape_[2] = n2_ + MKL_ALIGN_D;

    size_ = shape_[0] * shape_[1] * shape_[2];

    rstrides_[0] = 0;
    rstrides_[1] = (MKL_LONG)(shape_[2] * shape_[1]);
    rstrides_[2] = (MKL_LONG)(shape_[2]);
    rstrides_[3] = 1;
    cstrides_[0] = 0;
    cstrides_[1] = rstrides_[1] / 2;
    cstrides_[2] = rstrides_[2] / 2;
    cstrides_[3] = 1;

    // input/output data
    a_ = (double *)mkl_malloc(size_ * sizeof(double), MKL_ALIGN);
    if (a_ == NULL) {
        throw ICS_MEMORY_ERROR;
    }
    if (mode_ & ICS_MODE_CC) {
        b_ = (double *)mkl_malloc(size_ * sizeof(double), MKL_ALIGN);
        if (b_ == NULL) {
            free_();
            throw ICS_MEMORY_ERROR;
        }
    }
}

/* Free internal buffers */
void
rfft3d::free_()
{
    if (a_ != NULL)
        mkl_free(a_);
    if (b_ != NULL)
        mkl_free(b_);

    a_ = NULL;
    b_ = NULL;
}

/* Configure DFTI descriptor for forward transform */
void
rfft3d::set_forward_strides_()
{
    MKL_LONG status;
    status = DftiSetValue(dfti_handle_, DFTI_INPUT_STRIDES, rstrides_);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle_, DFTI_OUTPUT_STRIDES, cstrides_);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle_, DFTI_PLACEMENT, DFTI_INPLACE);
    if (status)
        throw status;
    status = DftiCommitDescriptor(dfti_handle_);
    if (status)
        throw status;
}

/* Configure DFTI descriptor for backward transform */
void
rfft3d::set_backward_strides_()
{
    MKL_LONG status;
    status = DftiSetValue(dfti_handle_, DFTI_INPUT_STRIDES, cstrides_);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle_, DFTI_OUTPUT_STRIDES, rstrides_);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle_, DFTI_PLACEMENT, DFTI_INPLACE);
    if (status)
        throw status;
    status = DftiCommitDescriptor(dfti_handle_);
    if (status)
        throw status;
}

/* 3D auto correlation */
template <typename Ti, typename To>
void
rfft3d::autocorrelate(const Ti *data, To *out, const ssize_t *strides)
{
    if ((data == NULL) || (out == NULL) || (strides == NULL)) {
        throw ICS_VALUE_ERROR;
    }

    copy_input_(a_, data, strides);
    set_forward_strides_();
    DftiComputeForward(dfti_handle_, a_);
    const double sum = a_[0];

    complex_multiply(a_, size_);

    set_backward_strides_();
    DftiComputeBackward(dfti_handle_, a_);

    if (mode_ & ICS_MODE_FCS) {
        copy_output_(out, a_, 1.0 / sum / sum, -1.0);
    }
    else {
        copy_output_(out, a_, 1.0 / (n0_ * n1_ * n2_), 0.0);
    }
}

/* 3D cross correlation */
template <typename Ti, typename To>
void
rfft3d::crosscorrelate(
    const Ti *data0,
    const Ti *data1,
    To *out,
    const ssize_t *strides0,
    const ssize_t *strides1)
{
    // allocate b_ if necessary
    if (b_ == NULL) {
        b_ = (double *)mkl_malloc(size_ * sizeof(double), MKL_ALIGN);
        if (b_ == NULL) {
            throw ICS_MEMORY_ERROR;
        }
    }

    set_forward_strides_();

    if (data0 != NULL) {
        copy_input_(a_, data0, strides0);
        DftiComputeForward(dfti_handle_, a_);
        suma_ = a_[0];
    }

    copy_input_(b_, data1, strides1);
    DftiComputeForward(dfti_handle_, b_);
    const double sumb = b_[0];

    complex_multiply(a_, b_, size_);

    set_backward_strides_();
    DftiComputeBackward(dfti_handle_, b_);

    if (mode_ & ICS_MODE_FCS) {
        copy_output_(out, b_, 1.0 / suma_ / sumb, -1.0);
    }
    else {
        copy_output_(out, b_, 1.0 / (n0_ * n1_ * n2_), 0.0);
    }
}

/* Copy input array to internal buffer */
template <typename T>
void
rfft3d::copy_input_(double *a, const T *data, const ssize_t *strides)
{
    if (data == NULL) {
        throw ICS_VALUE_ERROR;
    }

    const ssize_t s0 = (strides == NULL) ? 0 : strides[0] - n1_ * strides[1];
    const ssize_t s1 = (strides == NULL) ? 0 : strides[1] - n2_ * strides[2];
    const ssize_t s2 = (strides == NULL) ? sizeof(T) : strides[2];
    char *pdata = (char *)data;

    for (ssize_t i = 0; i < n0_; i++) {
        for (ssize_t j = 0; j < n1_; j++) {
            for (ssize_t k = 0; k < n2_; k++) {
                *a++ = (double)*(T *)pdata;
                pdata += s2;
            }
            for (ssize_t k = 0; k < MKL_ALIGN_D; k++) {
                *a++ = 0.0;
            }
            pdata += s1;
        }
        pdata += s0;
    }
}

/* Shift and scale internal buffer to output array */
template <typename T>
void
rfft3d::copy_output_(
    T *out, double *a, const double scale, const double offset)
{
    T *pout = out;
    const ssize_t n0 = n0_;
    const ssize_t n1 = n1_;
    const ssize_t n2 = n2_;
    const ssize_t s0 = rstrides_[1];
    const ssize_t s1 = rstrides_[2];
    const ssize_t h0 = n0 / 2;
    const ssize_t h1 = n1 / 2;
    const ssize_t h2 = n2 / 2;

    if (out == NULL) {
        throw ICS_VALUE_ERROR;
    }

    if ((mode_ & ICS_AXIS0) && (mode_ & ICS_AXIS1) && (mode_ & ICS_AXIS2)) {
        // do not center any axis
        pout = out;
        for (ssize_t i = 0; i < n0; i++)
            for (ssize_t j = 0; j < n1; j++, pout += n2)
                copy(pout, a + i * s0 + j * s1, n2, scale, offset);
        return;
    }

    if ((mode_ & ICS_AXIS0) && (mode_ & ICS_AXIS1)) {
        throw ICS_NOTIMPLEMENTD_ERROR;
        return;
    }

    if ((mode_ & ICS_AXIS0) && (mode_ & ICS_AXIS2)) {
        throw ICS_NOTIMPLEMENTD_ERROR;
        return;
    }

    if ((mode_ & ICS_AXIS1) && (mode_ & ICS_AXIS2)) {
        throw ICS_NOTIMPLEMENTD_ERROR;
        return;
    }

    if (mode_ & ICS_AXIS1) {
        throw ICS_NOTIMPLEMENTD_ERROR;
        return;
    }

    if (mode_ & ICS_AXIS0) {
        // do not center first axis
        const ssize_t sp = n1 * n2 - h1 * n2;
        // 0 0 0
        pout = out;
        for (ssize_t i = 0; i < n0; i++, pout += sp)
            for (ssize_t j = h1; j < n1; j++, pout += n2)
                copy(pout, a + i * s0 + j * s1 + h2, h2, scale, offset);
        // 0 0 h2
        pout = out + h2;
        for (ssize_t i = 0; i < n0; i++, pout += sp)
            for (ssize_t j = h1; j < n1; j++, pout += n2)
                copy(pout, a + i * s0 + j * s1, h2, scale, offset);
        // 0 h1 0
        pout = out + h1 * n2;
        for (ssize_t i = 0; i < n0; i++, pout += sp)
            for (ssize_t j = 0; j < h1; j++, pout += n2)
                copy(pout, a + i * s0 + j * s1 + h2, h2, scale, offset);
        // 0 h1 h2
        pout = out + h1 * n2 + h2;
        for (ssize_t i = 0; i < n0; i++, pout += sp)
            for (ssize_t j = 0; j < h1; j++, pout += n2)
                copy(pout, a + i * s0 + j * s1, h2, scale, offset);
        return;
    }

    if (mode_ & ICS_AXIS2) {
        // do not center last axis
        const ssize_t sp = n1 * n2 - h1 * n2;
        // 0 0 0
        pout = out;
        for (ssize_t i = h0; i < n0; i++, pout += sp)
            for (ssize_t j = h1; j < n1; j++, pout += n2)
                copy(pout, a + i * s0 + j * s1, n2, scale, offset);
        // 0 h1 0
        pout = out + h1 * n2;
        for (ssize_t i = h0; i < n0; i++, pout += sp)
            for (ssize_t j = 0; j < h1; j++, pout += n2)
                copy(pout, a + i * s0 + j * s1, n2, scale, offset);
        // h0 0 0
        pout = out + h0 * n1 * n2;
        for (ssize_t i = 0; i < h0; i++, pout += sp)
            for (ssize_t j = h1; j < n1; j++, pout += n2)
                copy(pout, a + i * s0 + j * s1, n2, scale, offset);
        // h0 h1 0
        pout = out + h0 * n1 * n2 + h1 * n2;
        for (ssize_t i = 0; i < h0; i++, pout += sp)
            for (ssize_t j = 0; j < h1; j++, pout += n2)
                copy(pout, a + i * s0 + j * s1, n2, scale, offset);
        return;
    }

    {
        // center all axis
        const ssize_t sp = n1 * n2 - h1 * n2;
        // 0 0 0
        pout = out;
        for (ssize_t i = h0; i < n0; i++, pout += sp)
            for (ssize_t j = h1; j < n1; j++, pout += n2)
                copy(pout, a + i * s0 + j * s1 + h2, h2, scale, offset);
        // 0 0 h2
        pout = out + h2;
        for (ssize_t i = h0; i < n0; i++, pout += sp)
            for (ssize_t j = h1; j < n1; j++, pout += n2)
                copy(pout, a + i * s0 + j * s1, h2, scale, offset);
        // 0 h1 0
        pout = out + h1 * n2;
        for (ssize_t i = h0; i < n0; i++, pout += sp)
            for (ssize_t j = 0; j < h1; j++, pout += n2)
                copy(pout, a + i * s0 + j * s1 + h2, h2, scale, offset);
        // 0 h1 h2
        pout = out + h1 * n2 + h2;
        for (ssize_t i = h0; i < n0; i++, pout += sp)
            for (ssize_t j = 0; j < h1; j++, pout += n2)
                copy(pout, a + i * s0 + j * s1, h2, scale, offset);
        // h0 0 0
        pout = out + h0 * n1 * n2;
        for (ssize_t i = 0; i < h0; i++, pout += sp)
            for (ssize_t j = h1; j < n1; j++, pout += n2)
                copy(pout, a + i * s0 + j * s1 + h2, h2, scale, offset);
        // h0 0 h2
        pout = out + h0 * n1 * n2 + h2;
        for (ssize_t i = 0; i < h0; i++, pout += sp)
            for (ssize_t j = h1; j < n1; j++, pout += n2)
                copy(pout, a + i * s0 + j * s1, h2, scale, offset);
        // h0 h1 0
        pout = out + h0 * n1 * n2 + h1 * n2;
        for (ssize_t i = 0; i < h0; i++, pout += sp)
            for (ssize_t j = 0; j < h1; j++, pout += n2)
                copy(pout, a + i * s0 + j * s1 + h2, h2, scale, offset);
        // h0 h1 h2
        pout = out + h0 * n1 * n2 + h1 * n2 + h2;
        for (ssize_t i = 0; i < h0; i++, pout += sp)
            for (ssize_t j = 0; j < h1; j++, pout += n2)
                copy(pout, a + i * s0 + j * s1, h2, scale, offset);
    }
}

/* Special 3D auto correlation for iMSD */
template <typename To>
void
rfft3d::imsd(
    const double *data,
    const double *channel,
    const ssize_t *strides,  // double strides, not byte
    To *out,
    const ssize_t *stridesout,  // byte strides
    const ssize_t *bins,
    const ssize_t nbins,
    const double filter)
{
    if ((data == NULL) || (out == NULL) || (strides == NULL) ||
        (stridesout == NULL) || (bins == NULL)) {
        throw ICS_VALUE_ERROR;
    }

    double scale = 0.0;

    MKL_LONG rstrides[4] = {
        0,
        (MKL_LONG)(strides[0]),
        (MKL_LONG)(strides[1]),
        (MKL_LONG)(strides[2])};
    MKL_LONG status;
    status = DftiSetValue(dfti_handle_, DFTI_INPUT_STRIDES, rstrides);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle_, DFTI_OUTPUT_STRIDES, cstrides_);
    if (status)
        throw status;
    status = DftiSetValue(dfti_handle_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (status)
        throw status;
    status = DftiCommitDescriptor(dfti_handle_);
    if (status)
        throw status;

    if ((channel != NULL) && (channel != data) && (b_ != NULL)) {
        DftiComputeForward(dfti_handle_, (void *)data, (void *)b_);
        DftiComputeForward(dfti_handle_, (void *)channel, (void *)a_);
        if ((a_[0] < 1e-9) || (b_[0] < 1e-9))
            return;
        scale = 1.0 / a_[0] / b_[0];
        complex_multiply(b_, a_, size_);
    }
    else {
        DftiComputeForward(dfti_handle_, (void *)data, (void *)a_);
        if (a_[0] < 1e-9)
            return;
        scale = 1.0 / a_[0] / a_[0];
        complex_multiply(a_, size_);
    }

    set_backward_strides_();
    DftiComputeBackward(dfti_handle_, a_);

    const ssize_t h0 = n0_ / 2;
    const ssize_t h1 = n1_ / 2;
    const ssize_t s0 = (n2_ + MKL_ALIGN_D) * n1_;
    const ssize_t s1 = (n2_ + MKL_ALIGN_D);

    // a_[0] = a_[1];

    // 0 0 0
    for (ssize_t y = 0, i = h0; i < n0_; y++, i++) {
        for (ssize_t x = 0, j = h1; j < n1_; x++, j++) {
            To *pout =
                (To *)((char *)out + y * stridesout[0] + x * stridesout[1]);
            anscf(
                a_ + i * s0 + j * s1,
                pout,
                stridesout[2],
                bins,
                nbins,
                scale,
                -1.0,
                filter,
                false);
        }
    }
    // h0 0 0
    for (ssize_t y = 0, i = 0; i < h0; y++, i++) {
        for (ssize_t x = 0, j = h1; j < n1_; x++, j++) {
            To *pout =
                (To *)((char *)out + (y + h0) * stridesout[0] + x * stridesout[1]);
            anscf(
                a_ + i * s0 + j * s1,
                pout,
                stridesout[2],
                bins,
                nbins,
                scale,
                -1.0,
                filter,
                false);
        }
    }
    // 0 h1 0
    for (ssize_t y = 0, i = h0; i < n0_; y++, i++) {
        for (ssize_t x = 0, j = 0; j < h1; x++, j++) {
            To *pout =
                (To *)((char *)out + y * stridesout[0] + (x + h1) * stridesout[1]);
            anscf(
                a_ + i * s0 + j * s1,
                pout,
                stridesout[2],
                bins,
                nbins,
                scale,
                -1.0,
                filter,
                false);
        }
    }
    // h0 h1 0
    for (ssize_t y = 0, i = 0; i < h0; y++, i++) {
        for (ssize_t x = 0, j = 0; j < h1; x++, j++) {
            To *pout =
                (To *)((char *)out + (y + h0) * stridesout[0] + (x + h1) * stridesout[1]);
            anscf(
                a_ + i * s0 + j * s1,
                pout,
                stridesout[2],
                bins,
                nbins,
                scale,
                -1.0,
                filter,
                false);
        }
    }
}

/* rfft1d C API */

rfft1d_handle
rfft1d_new(ssize_t shape0, int mode)
{
    try {
        return reinterpret_cast<rfft1d_handle>(new rfft1d(shape0, mode));
    }
    catch (...) {
        return NULL;
    }
}

void
rfft1d_del(rfft1d_handle handle)
{
    try {
        delete reinterpret_cast<rfft1d_handle>(handle);
    }
    catch (...) {
        ;
    }
}

void
rfft1d_mode(rfft1d_handle handle, int mode)
{
    try {
        reinterpret_cast<rfft1d_handle>(handle)->set_mode(mode);
    }
    catch (...) {
        ;
    }
}

#define RFFT1D_AUTOCORRELATE(function_name, input_type, output_type) \
    int function_name(                                               \
        rfft1d_handle handle,                                        \
        input_type *data,                                            \
        output_type *out,                                            \
        ssize_t *strides)                                            \
    {                                                                \
        try {                                                        \
            reinterpret_cast<rfft1d_handle>(handle)->autocorrelate(  \
                data, out, strides);                                 \
        }                                                            \
        catch (int e) {                                              \
            return e;                                                \
        }                                                            \
        catch (...) {                                                \
            return ICS_ERROR;                                        \
        }                                                            \
        return ICS_OK;                                               \
    }

RFFT1D_AUTOCORRELATE(rfft1d_autocorrelate_df, double, float)
RFFT1D_AUTOCORRELATE(rfft1d_autocorrelate_ff, float, float)
RFFT1D_AUTOCORRELATE(rfft1d_autocorrelate_if, int, float)
RFFT1D_AUTOCORRELATE(rfft1d_autocorrelate_hf, int16_t, float)
RFFT1D_AUTOCORRELATE(rfft1d_autocorrelate_Hf, uint16_t, float)
RFFT1D_AUTOCORRELATE(rfft1d_autocorrelate_dd, double, double)
RFFT1D_AUTOCORRELATE(rfft1d_autocorrelate_fd, float, double)
RFFT1D_AUTOCORRELATE(rfft1d_autocorrelate_id, int, double)
RFFT1D_AUTOCORRELATE(rfft1d_autocorrelate_hd, int16_t, double)
RFFT1D_AUTOCORRELATE(rfft1d_autocorrelate_Hd, uint16_t, double)

#define RFFT1D_CROSSCORRELATE(function_name, input_type, output_type) \
    int function_name(                                                \
        rfft1d_handle handle,                                         \
        input_type *data0,                                            \
        input_type *data1,                                            \
        output_type *out,                                             \
        ssize_t *strides0,                                            \
        ssize_t *strides1)                                            \
    {                                                                 \
        try {                                                         \
            reinterpret_cast<rfft1d_handle>(handle)->crosscorrelate(  \
                data0, data1, out, strides0, strides1);               \
        }                                                             \
        catch (int e) {                                               \
            return e;                                                 \
        }                                                             \
        catch (...) {                                                 \
            return ICS_ERROR;                                         \
        }                                                             \
        return ICS_OK;                                                \
    }

RFFT1D_CROSSCORRELATE(rfft1d_crosscorrelate_df, double, float)
RFFT1D_CROSSCORRELATE(rfft1d_crosscorrelate_ff, float, float)
RFFT1D_CROSSCORRELATE(rfft1d_crosscorrelate_if, int, float)
RFFT1D_CROSSCORRELATE(rfft1d_crosscorrelate_hf, int16_t, float)
RFFT1D_CROSSCORRELATE(rfft1d_crosscorrelate_Hf, uint16_t, float)
RFFT1D_CROSSCORRELATE(rfft1d_crosscorrelate_dd, double, double)
RFFT1D_CROSSCORRELATE(rfft1d_crosscorrelate_fd, float, double)
RFFT1D_CROSSCORRELATE(rfft1d_crosscorrelate_id, int, double)
RFFT1D_CROSSCORRELATE(rfft1d_crosscorrelate_hd, int16_t, double)
RFFT1D_CROSSCORRELATE(rfft1d_crosscorrelate_Hd, uint16_t, double)

/* rfft2d C API */

rfft2d_handle
rfft2d_new(ssize_t shape0, ssize_t shape1, int mode)
{
    try {
        return reinterpret_cast<rfft2d_handle>(
            new rfft2d(shape0, shape1, mode));
    }
    catch (...) {
        return NULL;
    }
}

void
rfft2d_del(rfft2d_handle handle)
{
    try {
        delete reinterpret_cast<rfft2d_handle>(handle);
    }
    catch (...) {
        ;
    }
}

void
rfft2d_mode(rfft2d_handle handle, int mode)
{
    try {
        reinterpret_cast<rfft2d_handle>(handle)->set_mode(mode);
    }
    catch (...) {
        ;
    }
}

#define RFFT2D_AUTOCORRELATE(function_name, input_type, output_type) \
    int function_name(                                               \
        rfft2d_handle handle,                                        \
        input_type *data,                                            \
        output_type *out,                                            \
        ssize_t *strides)                                            \
    {                                                                \
        try {                                                        \
            reinterpret_cast<rfft2d_handle>(handle)->autocorrelate(  \
                data, out, strides);                                 \
        }                                                            \
        catch (int e) {                                              \
            return e;                                                \
        }                                                            \
        catch (...) {                                                \
            return ICS_ERROR;                                        \
        }                                                            \
        return ICS_OK;                                               \
    }

RFFT2D_AUTOCORRELATE(rfft2d_autocorrelate_df, double, float)
RFFT2D_AUTOCORRELATE(rfft2d_autocorrelate_ff, float, float)
RFFT2D_AUTOCORRELATE(rfft2d_autocorrelate_if, int, float)
RFFT2D_AUTOCORRELATE(rfft2d_autocorrelate_hf, int16_t, float)
RFFT2D_AUTOCORRELATE(rfft2d_autocorrelate_Hf, uint16_t, float)
RFFT2D_AUTOCORRELATE(rfft2d_autocorrelate_dd, double, double)
RFFT2D_AUTOCORRELATE(rfft2d_autocorrelate_fd, float, double)
RFFT2D_AUTOCORRELATE(rfft2d_autocorrelate_id, int, double)
RFFT2D_AUTOCORRELATE(rfft2d_autocorrelate_hd, int16_t, double)
RFFT2D_AUTOCORRELATE(rfft2d_autocorrelate_Hd, uint16_t, double)

#define RFFT2D_CROSSCORRELATE(function_name, input_type, output_type) \
    int function_name(                                                \
        rfft2d_handle handle,                                         \
        input_type *data0,                                            \
        input_type *data1,                                            \
        output_type *out,                                             \
        ssize_t *strides0,                                            \
        ssize_t *strides1)                                            \
    {                                                                 \
        try {                                                         \
            reinterpret_cast<rfft2d_handle>(handle)->crosscorrelate(  \
                data0, data1, out, strides0, strides1);               \
        }                                                             \
        catch (int e) {                                               \
            return e;                                                 \
        }                                                             \
        catch (...) {                                                 \
            return ICS_ERROR;                                         \
        }                                                             \
        return ICS_OK;                                                \
    }

RFFT2D_CROSSCORRELATE(rfft2d_crosscorrelate_df, double, float)
RFFT2D_CROSSCORRELATE(rfft2d_crosscorrelate_ff, float, float)
RFFT2D_CROSSCORRELATE(rfft2d_crosscorrelate_if, int, float)
RFFT2D_CROSSCORRELATE(rfft2d_crosscorrelate_hf, int16_t, float)
RFFT2D_CROSSCORRELATE(rfft2d_crosscorrelate_Hf, uint16_t, float)
RFFT2D_CROSSCORRELATE(rfft2d_crosscorrelate_dd, double, double)
RFFT2D_CROSSCORRELATE(rfft2d_crosscorrelate_fd, float, double)
RFFT2D_CROSSCORRELATE(rfft2d_crosscorrelate_id, int, double)
RFFT2D_CROSSCORRELATE(rfft2d_crosscorrelate_hd, int16_t, double)
RFFT2D_CROSSCORRELATE(rfft2d_crosscorrelate_Hd, uint16_t, double)

#define RFFT2D_LSTICS(function_name, output_type)            \
    int function_name(                                       \
        rfft2d_handle handle,                                \
        double *data,                                        \
        double *channel,                                     \
        ssize_t *strides,                                    \
        output_type *out,                                    \
        ssize_t *outstrides,                                 \
        ssize_t *line,                                       \
        ssize_t *bins,                                       \
        ssize_t nbins,                                       \
        double filter)                                       \
    {                                                        \
        try {                                                \
            reinterpret_cast<rfft2d_handle>(handle)->lstics( \
                data,                                        \
                channel,                                     \
                strides,                                     \
                out,                                         \
                outstrides,                                  \
                line,                                        \
                bins,                                        \
                nbins,                                       \
                filter);                                     \
        }                                                    \
        catch (int e) {                                      \
            return e;                                        \
        }                                                    \
        catch (...) {                                        \
            return ICS_ERROR;                                \
        }                                                    \
        return ICS_OK;                                       \
    }

RFFT2D_LSTICS(rfft2d_lstics_f, float)
RFFT2D_LSTICS(rfft2d_lstics_d, double)

/* rfft3d C API */

rfft3d_handle
rfft3d_new(ssize_t shape0, ssize_t shape1, ssize_t shape2, int mode)
{
    try {
        return reinterpret_cast<rfft3d_handle>(
            new rfft3d(shape0, shape1, shape2, mode));
    }
    catch (...) {
        return NULL;
    }
}

void
rfft3d_del(rfft3d_handle handle)
{
    try {
        delete reinterpret_cast<rfft3d_handle>(handle);
    }
    catch (...) {
        ;
    }
}

void
rfft3d_mode(rfft3d_handle handle, int mode)
{
    try {
        reinterpret_cast<rfft3d_handle>(handle)->set_mode(mode);
    }
    catch (...) {
        ;
    }
}

#define RFFT3D_AUTOCORRELATE(function_name, input_type, output_type) \
    int function_name(                                               \
        rfft3d_handle handle,                                        \
        input_type *data,                                            \
        output_type *out,                                            \
        ssize_t *strides)                                            \
    {                                                                \
        try {                                                        \
            reinterpret_cast<rfft3d_handle>(handle)->autocorrelate(  \
                data, out, strides);                                 \
        }                                                            \
        catch (int e) {                                              \
            return e;                                                \
        }                                                            \
        catch (...) {                                                \
            return ICS_ERROR;                                        \
        }                                                            \
        return ICS_OK;                                               \
    }

RFFT3D_AUTOCORRELATE(rfft3d_autocorrelate_df, double, float)
RFFT3D_AUTOCORRELATE(rfft3d_autocorrelate_ff, float, float)
RFFT3D_AUTOCORRELATE(rfft3d_autocorrelate_if, int, float)
RFFT3D_AUTOCORRELATE(rfft3d_autocorrelate_hf, int16_t, float)
RFFT3D_AUTOCORRELATE(rfft3d_autocorrelate_Hf, uint16_t, float)
RFFT3D_AUTOCORRELATE(rfft3d_autocorrelate_dd, double, double)
RFFT3D_AUTOCORRELATE(rfft3d_autocorrelate_fd, float, double)
RFFT3D_AUTOCORRELATE(rfft3d_autocorrelate_id, int, double)
RFFT3D_AUTOCORRELATE(rfft3d_autocorrelate_hd, int16_t, double)
RFFT3D_AUTOCORRELATE(rfft3d_autocorrelate_Hd, uint16_t, double)

#define RFFT3D_CROSSCORRELATE(function_name, input_type, output_type) \
    int function_name(                                                \
        rfft3d_handle handle,                                         \
        input_type *data0,                                            \
        input_type *data1,                                            \
        output_type *out,                                             \
        ssize_t *strides0,                                            \
        ssize_t *strides1)                                            \
    {                                                                 \
        try {                                                         \
            reinterpret_cast<rfft3d_handle>(handle)->crosscorrelate(  \
                data0, data1, out, strides0, strides1);               \
        }                                                             \
        catch (int e) {                                               \
            return e;                                                 \
        }                                                             \
        catch (...) {                                                 \
            return ICS_ERROR;                                         \
        }                                                             \
        return ICS_OK;                                                \
    }

RFFT3D_CROSSCORRELATE(rfft3d_crosscorrelate_df, double, float)
RFFT3D_CROSSCORRELATE(rfft3d_crosscorrelate_ff, float, float)
RFFT3D_CROSSCORRELATE(rfft3d_crosscorrelate_if, int, float)
RFFT3D_CROSSCORRELATE(rfft3d_crosscorrelate_hf, int16_t, float)
RFFT3D_CROSSCORRELATE(rfft3d_crosscorrelate_Hf, uint16_t, float)
RFFT3D_CROSSCORRELATE(rfft3d_crosscorrelate_dd, double, double)
RFFT3D_CROSSCORRELATE(rfft3d_crosscorrelate_fd, float, double)
RFFT3D_CROSSCORRELATE(rfft3d_crosscorrelate_id, int, double)
RFFT3D_CROSSCORRELATE(rfft3d_crosscorrelate_hd, int16_t, double)
RFFT3D_CROSSCORRELATE(rfft3d_crosscorrelate_Hd, uint16_t, double)

#define RFFT3D_IMSD(function_name, output_type)            \
    int function_name(                                     \
        rfft3d_handle handle,                              \
        double *data,                                      \
        double *channel,                                   \
        ssize_t *strides,                                  \
        output_type *out,                                  \
        ssize_t *outstrides,                               \
        ssize_t *bins,                                     \
        ssize_t nbins,                                     \
        double filter)                                     \
    {                                                      \
        try {                                              \
            reinterpret_cast<rfft3d_handle>(handle)->imsd( \
                data,                                      \
                channel,                                   \
                strides,                                   \
                out,                                       \
                outstrides,                                \
                bins,                                      \
                nbins,                                     \
                filter);                                   \
        }                                                  \
        catch (int e) {                                    \
            return e;                                      \
        }                                                  \
        catch (...) {                                      \
            return ICS_ERROR;                              \
        }                                                  \
        return ICS_OK;                                     \
    }

RFFT3D_IMSD(rfft3d_imsd_f, float)
RFFT3D_IMSD(rfft3d_imsd_d, double)
