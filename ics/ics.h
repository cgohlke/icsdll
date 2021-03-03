/* ics.h */

/*
Copyright (c) 2016-2021, Christoph Gohlke
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* Interface to the ICS DLL.

The ICS Dynamic Link Library implements functions and classes for image
correlation spectroscopy.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2021.3.2

Requirements
------------
* Intel Math Kernel Library 2020
* Microsoft Visual Studio 2019

Notes
-----
To use MKL threading, link to
mkl_intel_lp64.lib;mkl_intel_thread.lib;mkl_core.lib;libiomp5md.lib
else
mkl_intel_lp64.lib;mkl_core.lib;mkl_sequential.lib

Revisions
---------
2021.3.2
  Add deconvolution functions (WIP).
2019.7.10
  API change for nlsp_solve_.
  Fix Visual Studio 2019 compile error.
2018.7.18
  Add function to fit diffusion to ipcf results.
  Add nlsp solver and functions.
2017.11.29
  Add and export psf_ functions.
  Export fft2d library functions.
2017.10.02
  Add yxt_correct_bleaching_ functions.
  Export yxt_apcf_ functions.
  Add apcf function to yxt class.
2017.08.15
  Add radial helper function.
  Export yxt_lstics_ and rfft3d_lstics_ functions.
  Add lstics function to rfft3d and yxt classes.
2017.02.10
  Add yxt_subtract_immobile_ functions to subtract immobile fractions.
2017.01.28
  Add yxt_crosscorrelate_hf compatibility function for SimFCS.
  Change float argument of yxt_pcf_ functions to double (breaking).
  Rename yxt_crosscorrelate_ funtions to yxt_pcf_ (breaking).
  Export yxt_imsd_ and rfft3d_imsd_ functions.
  Add imsd function to rfft3d and yxt classes.
2016.12.30
  Add symmetry speedups to yxt_crosscorrelate function.
  Fix logbins function. Inputs (256, 9) and (32, 4) returned wrong results.
  Simplify average function.
  Disable OpenMP in debug build.
  Add MKL implementations of rfft1d, rfft2d, and rfft3d.
  Remove option to compile with fft2d.
2016.06.22
  Add npoints parameter to circle function.
2016.06.10
  Initial release.

*/

#ifndef ICS_H
#define ICS_H

#include "ics_version.h"

#include <stdint.h>

#if defined(_MSC_VER)
typedef ptrdiff_t ssize_t;
#define SSIZE_MAX INTPTR_MAX
#define SSIZE_MIN INTPTR_MIN
#endif

#define M_PI 3.14159265358979323846 /* pi */

#ifdef ICS_USEDLL
#define ICS_API extern __declspec(dllimport)
#elif defined ICS_EXPORTS
#define ICS_API extern __declspec(dllexport)
#else
#define ICS_API
#endif

#define ICS_FALSE 0
#define ICS_TRUE 1

#define ICS_OK 0
#define ICS_ERROR -1
#define ICS_VALUE_ERROR -2
#define ICS_MEMORY_ERROR -3
#define ICS_NOTIMPLEMENTD_ERROR -4

#define ICS_VALUE_ERROR1 -201
#define ICS_VALUE_ERROR2 -202
#define ICS_VALUE_ERROR3 -203
#define ICS_VALUE_ERROR4 -204
#define ICS_VALUE_ERROR5 -205
#define ICS_VALUE_ERROR6 -206
#define ICS_VALUE_ERROR7 -207
#define ICS_VALUE_ERROR8 -208
#define ICS_VALUE_ERROR9 -209

#define ICS_MODE_DEFAULT 0
#define ICS_MODE_TIME \
    1                  /* do not center correlation results in axis 0 (time) */
#define ICS_MODE_FCS 2 /* normalize correlation results according to FCS */
#define ICS_MODE_CC 4  /* allocate second buffer for cross correlation */

#define ICS_AXIS0 1  /* do not center correlation results in axis 0 */
#define ICS_AXIS1 8  /* do not center correlation results in axis 1 */
#define ICS_AXIS2 16 /* do not center correlation results in axis 2 */

#define ICS_MASK_DEFAULT 0
#define ICS_MASK_ANY 0    /* any one value must be True */
#define ICS_MASK_FIRST 1  /* first mask value must be True */
#define ICS_MASK_CENTER 2 /* center mask value must be True */
#define ICS_MASK_ALL 4    /* all mask values must be True */
#define ICS_MASK_CLEAR 32 /* clear output if not calculated */

#define ICS_RADIUS 1
#define ICS_DIAMETER 2

#define ICS_NLSP_ND 1
#define ICS_NLSP_1DPCF 100

#define ICS_DECONV_DEFAULT 1 /* ICS_DECONV_RICHARDSON_LUCY */
#define ICS_DECONV_RICHARDSON_LUCY 1
#define ICS_DECONV_WIENER 2
#define ICS_DECONV_NOPAD 256

/* C++ API */

#ifdef __cplusplus

#include "mkl.h"
#include "mkl_dfti.h"
#include "mkl_rci.h"
#include "mkl_types.h"
#include "mkl_service.h"

struct rfft1d {
   public:
    rfft1d(const ssize_t shape0, const int mode = 0);

    ~rfft1d();

    void
    set_mode(int mode)
    {
        mode_ = mode;
    }

    template <typename Ti, typename To>
    void
    autocorrelate(const Ti *data, To *out, const ssize_t *strides = NULL);

    template <typename Ti, typename To>
    void
    crosscorrelate(
        const Ti *data0,
        const Ti *data1,
        To *out,
        const ssize_t *strides0 = NULL,
        const ssize_t *strides1 = NULL);

   private:
    DFTI_DESCRIPTOR_HANDLE dfti_handle_;
    int mode_;
    ssize_t n0_;  /* shape of input and output arrays */
    double suma_; /* sum of data in a_ */
    double *a_;   /* input/output buffer */
    double *b_;   /* cross correlation buffer */

    void
    alloc_();

    void
    free_();

    template <typename T>
    void
    copy_input_(double *a, const T *data, const ssize_t *strides);

    template <typename T>
    void
    copy_output_(
        T *out,
        double *a,
        const double scale = 1.0,
        const double offset = 0.0);
};

struct rfft2d {
   public:
    rfft2d(const ssize_t shape0, const ssize_t shape1, const int mode = 0);

    ~rfft2d();

    void
    set_mode(int mode)
    {
        mode_ = mode;
    }

    template <typename Ti, typename To>
    void
    autocorrelate(const Ti *data, To *out, const ssize_t *strides = NULL);

    template <typename Ti, typename To>
    void
    crosscorrelate(
        const Ti *data0,
        const Ti *data1,
        To *out,
        const ssize_t *strides0 = NULL,
        const ssize_t *strides1 = NULL);

    template <typename To>
    void
    lstics(
        const double *data,
        const double *channel,
        const ssize_t *strides,
        To *out,
        const ssize_t *outstrides,
        const ssize_t *line,
        const ssize_t *bins,
        const ssize_t nbins,
        const double filter);

   private:
    DFTI_DESCRIPTOR_HANDLE dfti_handle_;
    int mode_;
    ssize_t n0_, n1_;  /* shape of input and output arrays */
    ssize_t size_;     /* size of internal buffers */
    ssize_t shape_[2]; /* shape of internal buffers */
    MKL_LONG rstrides_[3], cstrides_[3];
    double suma_; /* sum of data in a_ */
    double *a_;   /* input/output buffer */
    double *b_;   /* cross correlation buffer */

    void
    alloc_();

    void
    free_();

    void
    set_forward_strides_();

    void
    set_backward_strides_();

    template <typename T>
    void
    copy_input_(double *a, const T *data, const ssize_t *strides);

    template <typename T>
    void
    copy_output_(
        T *out,
        double *a,
        const double scale = 1.0,
        const double offset = 0.0);

    void
    copy_input_(
        double *a,
        const double *data,
        const ssize_t *strides,
        const ssize_t *points);
};

struct rfft3d {
   public:
    rfft3d(
        const ssize_t shape0,
        const ssize_t shape1,
        const ssize_t shape2,
        const int mode = 0);

    ~rfft3d();

    void
    set_mode(int mode)
    {
        mode_ = mode;
    }

    template <typename Ti, typename To>
    void
    autocorrelate(const Ti *data, To *out, const ssize_t *strides = NULL);

    template <typename Ti, typename To>
    void
    crosscorrelate(
        const Ti *data0,
        const Ti *data1,
        To *out,
        const ssize_t *strides0 = NULL,
        const ssize_t *strides1 = NULL);

    template <typename To>
    void
    imsd(
        const double *data,
        const double *channel,
        const ssize_t *strides,
        To *out,
        const ssize_t *outstrides,
        const ssize_t *bins,
        const ssize_t nbins,
        const double filter);

   private:
    DFTI_DESCRIPTOR_HANDLE dfti_handle_;
    int mode_;
    ssize_t n0_, n1_, n2_; /* shape of input and output arrays */
    ssize_t size_;         /* size of internal buffers */
    ssize_t shape_[3];     /* shape of internal buffers */
    MKL_LONG rstrides_[4], cstrides_[4];
    double suma_; /* sum of data in a_ */
    double *a_;   /* input/output buffer */
    double *b_;   /* cross correlation buffer */

    void
    alloc_();

    void
    free_();

    void
    set_forward_strides_();

    void
    set_backward_strides_();

    template <typename T>
    void
    copy_input_(double *a, const T *data, const ssize_t *strides);

    template <typename T>
    void
    copy_output_(
        T *out,
        double *a,
        const double scale = 1.0,
        const double offset = 0.0);
};

template <typename T>
struct zyx {
   public:
    zyx(const ssize_t *shape);

    ~zyx();

    T *
    zyx_get_buffer(ssize_t *shape = NULL, ssize_t *strides = NULL);

    ssize_t
    size()
    {
        return n0_ * n1_ * n2_;
    };

    template <typename Ti>
    void
    copy_input(const Ti *data, const ssize_t *shape, const ssize_t *strides);

    template <typename To>
    void
    copy_output(To *out, const ssize_t *shape, const ssize_t *strides);

    void
    dfti_forward();

    void
    dfti_forward(zyx<T> &out);

    void
    dfti_backward();

    void
    dfti_backward(zyx<T> &out);

    void
    complex_multiply(zyx<T> &b);

    void
    complex_multiply_conj(zyx<T> &b);

    void
    real_multiply(const T b);

    void
    real_multiply(zyx<T> &b);

    void
    real_divide(const T b);

    void
    real_divide(zyx<T> &b);

    void
    real_divide_by_self(zyx<T> &b);

    void
    real_const(const T c);

   private:
    DFTI_DESCRIPTOR_HANDLE dfti_handle_;
    ssize_t n0_, n1_, n2_; /* shape of input and output arrays */
    ssize_t size_;         /* size of internal buffers */
    ssize_t shape_[3];     /* shape of internal buffers */
    MKL_LONG rstrides_[4], cstrides_[4];
    T suma_; /* sum of data in a_ */
    T *a_;   /* input/output buffer */

    void
    alloc_();

    void
    free_();

    void
    set_forward_strides_(bool inplace = true);

    void
    set_backward_strides_(bool inplace = true);
};

template <typename T>
struct zyx_deconv {
   public:
    zyx_deconv(const ssize_t *shape);

    ~zyx_deconv();

    template <typename Ti>
    void
    set_image(const Ti *data, const ssize_t *shape, const ssize_t *strides);

    template <typename Ti>
    void
    set_psf(const Ti *data, const ssize_t *shape, const ssize_t *strides);

    template <typename To>
    void
    get_image(To *out, const ssize_t *shape, const ssize_t *strides);

    void
    richardson_lucy(const int niter, const int nthreads = 0);

   private:
    zyx<T> *img_;
    zyx<T> *obj_;
    zyx<T> *otf_;
    zyx<T> *tmp_;
};

struct yxt {
   public:
    yxt(const ssize_t *size);
    ~yxt();

    double *
    yxt_get_buffer(ssize_t *shape, ssize_t *strides);

    template <typename Ti, typename To>
    void
    apcf(
        const Ti *data,
        const ssize_t *strides,
        To *out,
        const ssize_t *outstrides,
        const ssize_t *bins,
        const ssize_t nbins,
        const int autocorr = ICS_FALSE,
        const double filter = 0.7,
        const int nthreads = 0);

    template <typename Ti, typename To>
    void
    ipcf(
        const Ti *data,
        const Ti *channel,
        const ssize_t *strides,
        To *out,
        const ssize_t *outstrides,
        const ssize_t *points,
        const ssize_t npoints,
        const ssize_t *bins,
        const ssize_t nbins,
        const double threshold = 0.0,
        const double filter = 0.7,
        const int nthreads = 0);

    template <typename Ti, typename Tm, typename To>
    void
    imsd(
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
        const double filter = 0.7,
        const int nthreads = 0);

    template <typename Ti, typename Tm, typename To>
    void
    lstics(
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
        const double filter = 0.7,
        const int nthreads = 0);

   private:
    ssize_t ysize_, xsize_, tsize_; /* shape of input region/array */
    double *a_;                     /* buffer for data */
    double *b_;                     /* buffer for channel */
    ssize_t size_;                  /* size of internal buffers */
    ssize_t shape_[3];              /* shape of internal buffers */
    ssize_t strides_[3]; /* strides of internal buffers in sizeof(double) */
};

struct nlsp_extra_t {
    void *y;  // f(x) values to fit
    void *x;  // e.g. x axis, constants
};

/* Trust-Region nonlinear least squares problem with boundary constraints */
struct nlsp {
   public:
    nlsp(int model, ssize_t *shape);

    ~nlsp();

    template <typename Ti>
    void
    solve(
        const Ti *data,
        const ssize_t *strides,
        void *extra,
        double *guess,
        double *bounds,
        double *solution);

    template <typename To>
    void
    eval(To *data, const ssize_t *strides);

    void
    set(MKL_INT iter1, MKL_INT iter2, double *eps, double eps_jac_, double rs);

    void
    get(MKL_INT *, MKL_INT *, double *, double *);

   private:
    // user's objective function
    USRFCNXD objective_ = NULL;
    // extra data for objective function
    nlsp_extra_t *extra_ = NULL;
    // TR solver handle
    _TRNSPBC_HANDLE_t handle_bc_ = NULL;
    _TRNSP_HANDLE_t handle_ = NULL;
    // number of function variables
    MKL_INT n_ = 0;
    // dimension of function value
    MKL_INT m_ = 0;
    // maximum number of iterations
    MKL_INT iter1_ = 0;
    // maximum number of iterations of calculation of trial-step
    MKL_INT iter2_ = 0;
    // initial step bound
    double rs_ = 0.0;
    // precisions for stop-criteria
    double eps_[6];
    // precision of the Jacobian matrix calculation
    double eps_jac_;
    // solution vector contains values x for f(x)
    double *x_ = NULL;
    // data to fit
    double *y_ = NULL;
    // function (f(x)) value vector  fvec[i] = (yi  fi(x))
    double *fvec_ = NULL;
    // jacobi matrix
    double *fjac_ = NULL;
    // lower bounds
    double *lw_ = NULL;
    // upper bounds
    double *up_ = NULL;
    // dimension of input data
    int ndim_ = 0;
    // shape of input data
    ssize_t shape_[ICS_NLSP_ND];
};

template <typename Ti, typename To, typename Tx>
void
ipcf_nlsp_1dpcf(
    const Ti *data,
    const ssize_t *shape,
    const ssize_t *strides,
    const Tx *times,
    const Tx *distances,
    const Tx *args,
    const Tx *bounds,
    Tx *x,
    const ssize_t *stridesx,
    To *fx,
    const ssize_t *stridesfx,
    Tx *status,
    const ssize_t *stridesstatus,
    const Tx *settings = NULL,
    const bool average = false,
    const int nthreads = 0);

/* extern objective functions */
extern "C" {
ICS_API void
nlsp_1dpcf(MKL_INT *m, MKL_INT *n, double *x, double *f, void *y);
}

#else
struct zyx;
struct yxt;
struct rfft1d;
struct rfft2d;
struct rfft3d;
struct nlsp;
#endif

/* C API */
typedef struct rfft1d *rfft1d_handle;
typedef struct rfft2d *rfft2d_handle;
typedef struct rfft3d *rfft3d_handle;
typedef struct yxt *yxt_handle;
typedef struct nlsp *nlsp_handle;

#ifdef __cplusplus
extern "C" {
#endif

/* rfft1d class */
ICS_API rfft1d_handle
rfft1d_new(ssize_t shape0, int mode);

ICS_API void rfft1d_del(rfft1d_handle);

ICS_API void
rfft1d_mode(rfft1d_handle, int mode);

ICS_API int
rfft1d_autocorrelate_dd(
    rfft1d_handle, double *data, double *out, ssize_t *strides);

ICS_API int
rfft1d_autocorrelate_df(
    rfft1d_handle, double *data, float *out, ssize_t *strides);

ICS_API int
rfft1d_autocorrelate_ff(
    rfft1d_handle, float *data, float *out, ssize_t *strides);

ICS_API int
rfft1d_autocorrelate_fd(
    rfft1d_handle, float *data, double *out, ssize_t *strides);

ICS_API int
rfft1d_autocorrelate_if(
    rfft1d_handle, int *data, float *out, ssize_t *strides);

ICS_API int
rfft1d_autocorrelate_id(
    rfft1d_handle, int *data, double *out, ssize_t *strides);

ICS_API int
rfft1d_autocorrelate_hf(
    rfft1d_handle, int16_t *data, float *out, ssize_t *strides);

ICS_API int
rfft1d_autocorrelate_hd(
    rfft1d_handle, int16_t *data, double *out, ssize_t *strides);

ICS_API int
rfft1d_autocorrelate_Hf(
    rfft1d_handle, uint16_t *data, float *out, ssize_t *strides);

ICS_API int
rfft1d_autocorrelate_Hd(
    rfft1d_handle, uint16_t *data, double *out, ssize_t *strides);

ICS_API int
rfft1d_crosscorrelate_dd(
    rfft1d_handle,
    double *data0,
    double *data1,
    double *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft1d_crosscorrelate_df(
    rfft1d_handle,
    double *data0,
    double *data1,
    float *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft1d_crosscorrelate_ff(
    rfft1d_handle,
    float *data0,
    float *data1,
    float *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft1d_crosscorrelate_fd(
    rfft1d_handle,
    float *data0,
    float *data1,
    double *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft1d_crosscorrelate_if(
    rfft1d_handle,
    int *data0,
    int *data1,
    float *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft1d_crosscorrelate_id(
    rfft1d_handle,
    int *data0,
    int *data1,
    double *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft1d_crosscorrelate_hf(
    rfft1d_handle,
    int16_t *data0,
    int16_t *data1,
    float *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft1d_crosscorrelate_hd(
    rfft1d_handle,
    int16_t *data0,
    int16_t *data1,
    double *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft1d_crosscorrelate_Hf(
    rfft1d_handle,
    uint16_t *data0,
    uint16_t *data1,
    float *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft1d_crosscorrelate_Hd(
    rfft1d_handle,
    uint16_t *data0,
    uint16_t *data1,
    double *out,
    ssize_t *strides0,
    ssize_t *strides1);

/* rfft2d class */
ICS_API rfft2d_handle
rfft2d_new(ssize_t shape0, ssize_t shape1, int mode);

ICS_API void rfft2d_del(rfft2d_handle);

ICS_API void
rfft2d_mode(rfft2d_handle, int mode);

ICS_API int
rfft2d_autocorrelate_dd(
    rfft2d_handle, double *data, double *out, ssize_t *strides);

ICS_API int
rfft2d_autocorrelate_df(
    rfft2d_handle, double *data, float *out, ssize_t *strides);

ICS_API int
rfft2d_autocorrelate_ff(
    rfft2d_handle, float *data, float *out, ssize_t *strides);

ICS_API int
rfft2d_autocorrelate_fd(
    rfft2d_handle, float *data, double *out, ssize_t *strides);

ICS_API int
rfft2d_autocorrelate_if(
    rfft2d_handle, int *data, float *out, ssize_t *strides);

ICS_API int
rfft2d_autocorrelate_id(
    rfft2d_handle, int *data, double *out, ssize_t *strides);

ICS_API int
rfft2d_autocorrelate_hf(
    rfft2d_handle, int16_t *data, float *out, ssize_t *strides);

ICS_API int
rfft2d_autocorrelate_hd(
    rfft2d_handle, int16_t *data, double *out, ssize_t *strides);

ICS_API int
rfft2d_autocorrelate_Hf(
    rfft2d_handle, uint16_t *data, float *out, ssize_t *strides);

ICS_API int
rfft2d_autocorrelate_Hd(
    rfft2d_handle, uint16_t *data, double *out, ssize_t *strides);

ICS_API int
rfft2d_crosscorrelate_dd(
    rfft2d_handle,
    double *data0,
    double *data1,
    double *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft2d_crosscorrelate_df(
    rfft2d_handle,
    double *data0,
    double *data1,
    float *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft2d_crosscorrelate_ff(
    rfft2d_handle,
    float *data0,
    float *data1,
    float *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft2d_crosscorrelate_fd(
    rfft2d_handle,
    float *data0,
    float *data1,
    double *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft2d_crosscorrelate_if(
    rfft2d_handle,
    int *data0,
    int *data1,
    float *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft2d_crosscorrelate_id(
    rfft2d_handle,
    int *data0,
    int *data1,
    double *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft2d_crosscorrelate_hf(
    rfft2d_handle,
    int16_t *data0,
    int16_t *data1,
    float *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft2d_crosscorrelate_hd(
    rfft2d_handle,
    int16_t *data0,
    int16_t *data1,
    double *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft2d_crosscorrelate_Hf(
    rfft2d_handle,
    uint16_t *data0,
    uint16_t *data1,
    float *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft2d_crosscorrelate_Hd(
    rfft2d_handle,
    uint16_t *data0,
    uint16_t *data1,
    double *out,
    ssize_t *strides0,
    ssize_t *strides1);

/* rfft3d class */
ICS_API rfft3d_handle
rfft3d_new(ssize_t shape0, ssize_t shape1, ssize_t shape2, int mode);

ICS_API void rfft3d_del(rfft3d_handle);

ICS_API void
rfft3d_mode(rfft3d_handle, int mode);

ICS_API int
rfft3d_autocorrelate_dd(
    rfft3d_handle, double *data, double *out, ssize_t *strides);

ICS_API int
rfft3d_autocorrelate_df(
    rfft3d_handle, double *data, float *out, ssize_t *strides);

ICS_API int
rfft3d_autocorrelate_ff(
    rfft3d_handle, float *data, float *out, ssize_t *strides);

ICS_API int
rfft3d_autocorrelate_fd(
    rfft3d_handle, float *data, double *out, ssize_t *strides);

ICS_API int
rfft3d_autocorrelate_if(
    rfft3d_handle, int *data, float *out, ssize_t *strides);

ICS_API int
rfft3d_autocorrelate_id(
    rfft3d_handle, int *data, double *out, ssize_t *strides);

ICS_API int
rfft3d_autocorrelate_hf(
    rfft3d_handle, int16_t *data, float *out, ssize_t *strides);

ICS_API int
rfft3d_autocorrelate_hd(
    rfft3d_handle, int16_t *data, double *out, ssize_t *strides);

ICS_API int
rfft3d_autocorrelate_Hf(
    rfft3d_handle, uint16_t *data, float *out, ssize_t *strides);

ICS_API int
rfft3d_autocorrelate_Hd(
    rfft3d_handle, uint16_t *data, double *out, ssize_t *strides);

ICS_API int
rfft3d_crosscorrelate_dd(
    rfft3d_handle,
    double *data0,
    double *data1,
    double *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft3d_crosscorrelate_df(
    rfft3d_handle,
    double *data0,
    double *data1,
    float *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft3d_crosscorrelate_ff(
    rfft3d_handle,
    float *data0,
    float *data1,
    float *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft3d_crosscorrelate_fd(
    rfft3d_handle,
    float *data0,
    float *data1,
    double *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft3d_crosscorrelate_if(
    rfft3d_handle,
    int *data0,
    int *data1,
    float *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft3d_crosscorrelate_id(
    rfft3d_handle,
    int *data0,
    int *data1,
    double *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft3d_crosscorrelate_hf(
    rfft3d_handle,
    int16_t *data0,
    int16_t *data1,
    float *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft3d_crosscorrelate_hd(
    rfft3d_handle,
    int16_t *data0,
    int16_t *data1,
    double *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft3d_crosscorrelate_Hf(
    rfft3d_handle,
    uint16_t *data0,
    uint16_t *data1,
    float *out,
    ssize_t *strides0,
    ssize_t *strides1);

ICS_API int
rfft3d_crosscorrelate_Hd(
    rfft3d_handle,
    uint16_t *data0,
    uint16_t *data1,
    double *out,
    ssize_t *strides0,
    ssize_t *strides1);

/* yxt class */

ICS_API yxt_handle
yxt_new(ssize_t *shape);

ICS_API void yxt_del(yxt_handle);

ICS_API double *
yxt_get_buffer(yxt_handle handle, ssize_t *shape, ssize_t *strides);

ICS_API int
yxt_apcf_df(
    yxt_handle handle,
    double *data,
    ssize_t *strides,
    float *out,
    ssize_t *outstrides,
    ssize_t *bins,
    ssize_t nbins,
    int autocorr,
    double filter,
    int nthreads);

ICS_API int
yxt_apcf_ff(
    yxt_handle handle,
    float *data,
    ssize_t *strides,
    float *out,
    ssize_t *outstrides,
    ssize_t *bins,
    ssize_t nbins,
    int autocorr,
    double filter,
    int nthreads);

ICS_API int
yxt_apcf_if(
    yxt_handle handle,
    int32_t *data,
    ssize_t *strides,
    float *out,
    ssize_t *outstrides,
    ssize_t *bins,
    ssize_t nbins,
    int autocorr,
    double filter,
    int nthreads);

ICS_API int
yxt_apcf_hf(
    yxt_handle handle,
    int16_t *data,
    ssize_t *strides,
    float *out,
    ssize_t *outstrides,
    ssize_t *bins,
    ssize_t nbins,
    int autocorr,
    double filter,
    int nthreads);

ICS_API int
yxt_apcf_Hf(
    yxt_handle handle,
    uint16_t *data,
    ssize_t *strides,
    float *out,
    ssize_t *outstrides,
    ssize_t *bins,
    ssize_t nbins,
    int autocorr,
    double filter,
    int nthreads);

ICS_API int
yxt_apcf_dd(
    yxt_handle handle,
    double *data,
    ssize_t *strides,
    double *out,
    ssize_t *outstrides,
    ssize_t *bins,
    ssize_t nbins,
    int autocorr,
    double filter,
    int nthreads);

ICS_API int
yxt_apcf_fd(
    yxt_handle handle,
    float *data,
    ssize_t *strides,
    double *out,
    ssize_t *outstrides,
    ssize_t *bins,
    ssize_t nbins,
    int autocorr,
    double filter,
    int nthreads);

ICS_API int
yxt_apcf_id(
    yxt_handle handle,
    int32_t *data,
    ssize_t *strides,
    double *out,
    ssize_t *outstrides,
    ssize_t *bins,
    ssize_t nbins,
    int autocorr,
    double filter,
    int nthreads);

ICS_API int
yxt_apcf_hd(
    yxt_handle handle,
    int16_t *data,
    ssize_t *strides,
    double *out,
    ssize_t *outstrides,
    ssize_t *bins,
    ssize_t nbins,
    int autocorr,
    double filter,
    int nthreads);

ICS_API int
yxt_apcf_Hd(
    yxt_handle handle,
    uint16_t *data,
    ssize_t *strides,
    double *out,
    ssize_t *outstrides,
    ssize_t *bins,
    ssize_t nbins,
    int autocorr,
    double filter,
    int nthreads);

ICS_API int
yxt_ipcf_df(
    yxt_handle handle,
    double *data,
    double *channel,
    ssize_t *strides,
    float *out,
    ssize_t *outstrides,
    ssize_t *points,
    ssize_t npoints,
    ssize_t *bins,
    ssize_t nbins,
    double threshold,
    double filter,
    int nthreads);

ICS_API int
yxt_ipcf_ff(
    yxt_handle handle,
    float *data,
    float *channel,
    ssize_t *strides,
    float *out,
    ssize_t *outstrides,
    ssize_t *points,
    ssize_t npoints,
    ssize_t *bins,
    ssize_t nbins,
    double threshold,
    double filter,
    int nthreads);

ICS_API int
yxt_ipcf_if(
    yxt_handle handle,
    int32_t *data,
    int32_t *channel,
    ssize_t *strides,
    float *out,
    ssize_t *outstrides,
    ssize_t *points,
    ssize_t npoints,
    ssize_t *bins,
    ssize_t nbins,
    double threshold,
    double filter,
    int nthreads);

ICS_API int
yxt_ipcf_hf(
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
    double filter,
    int nthreads);

ICS_API int
yxt_ipcf_Hf(
    yxt_handle handle,
    uint16_t *data,
    uint16_t *channel,
    ssize_t *strides,
    float *out,
    ssize_t *outstrides,
    ssize_t *points,
    ssize_t npoints,
    ssize_t *bins,
    ssize_t nbins,
    double threshold,
    double filter,
    int nthreads);

ICS_API int
yxt_ipcf_dd(
    yxt_handle handle,
    double *data,
    double *channel,
    ssize_t *strides,
    double *out,
    ssize_t *outstrides,
    ssize_t *points,
    ssize_t npoints,
    ssize_t *bins,
    ssize_t nbins,
    double threshold,
    double filter,
    int nthreads);

ICS_API int
yxt_ipcf_fd(
    yxt_handle handle,
    float *data,
    float *channel,
    ssize_t *strides,
    double *out,
    ssize_t *outstrides,
    ssize_t *points,
    ssize_t npoints,
    ssize_t *bins,
    ssize_t nbins,
    double threshold,
    double filter,
    int nthreads);

ICS_API int
yxt_ipcf_id(
    yxt_handle handle,
    int32_t *data,
    int32_t *channel,
    ssize_t *strides,
    double *out,
    ssize_t *outstrides,
    ssize_t *points,
    ssize_t npoints,
    ssize_t *bins,
    ssize_t nbins,
    double threshold,
    double filter,
    int nthreads);

ICS_API int
yxt_ipcf_hd(
    yxt_handle handle,
    int16_t *data,
    int16_t *channel,
    ssize_t *strides,
    double *out,
    ssize_t *outstrides,
    ssize_t *points,
    ssize_t npoints,
    ssize_t *bins,
    ssize_t nbins,
    double threshold,
    double filter,
    int nthreads);

ICS_API int
yxt_ipcf_Hd(
    yxt_handle handle,
    uint16_t *data,
    uint16_t *channel,
    ssize_t *strides,
    double *out,
    ssize_t *outstrides,
    ssize_t *points,
    ssize_t npoints,
    ssize_t *bins,
    ssize_t nbins,
    double threshold,
    double filter,
    int nthreads);

ICS_API int
yxt_imsd_df(
    yxt_handle handle,
    double *data,
    ssize_t *strides,
    double *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    float *out,
    ssize_t *outstrides,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_imsd_ff(
    yxt_handle handle,
    float *data,
    ssize_t *strides,
    float *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    float *out,
    ssize_t *outstrides,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_imsd_if(
    yxt_handle handle,
    int32_t *data,
    ssize_t *strides,
    int32_t *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    float *out,
    ssize_t *outstrides,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_imsd_hf(
    yxt_handle handle,
    int16_t *data,
    ssize_t *strides,
    int16_t *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    float *out,
    ssize_t *outstrides,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_imsd_Hf(
    yxt_handle handle,
    uint16_t *data,
    ssize_t *strides,
    uint16_t *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    float *out,
    ssize_t *outstrides,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_imsd_dd(
    yxt_handle handle,
    double *data,
    ssize_t *strides,
    double *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    double *out,
    ssize_t *outstrides,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_imsd_fd(
    yxt_handle handle,
    float *data,
    ssize_t *strides,
    float *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    double *out,
    ssize_t *outstrides,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_imsd_id(
    yxt_handle handle,
    int32_t *data,
    ssize_t *strides,
    int32_t *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    double *out,
    ssize_t *outstrides,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_imsd_hd(
    yxt_handle handle,
    int16_t *data,
    ssize_t *strides,
    int16_t *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    double *out,
    ssize_t *outstrides,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_imsd_Hd(
    yxt_handle handle,
    uint16_t *data,
    ssize_t *strides,
    uint16_t *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    double *out,
    ssize_t *outstrides,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_lstics_df(
    yxt_handle handle,
    double *data,
    ssize_t *strides,
    double *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    float *out,
    ssize_t *outstrides,
    ssize_t *lines,
    ssize_t *linesshape,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_lstics_ff(
    yxt_handle handle,
    float *data,
    ssize_t *strides,
    float *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    float *out,
    ssize_t *outstrides,
    ssize_t *lines,
    ssize_t *linesshape,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_lstics_if(
    yxt_handle handle,
    int32_t *data,
    ssize_t *strides,
    int32_t *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    float *out,
    ssize_t *outstrides,
    ssize_t *lines,
    ssize_t *linesshape,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_lstics_hf(
    yxt_handle handle,
    int16_t *data,
    ssize_t *strides,
    int16_t *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    float *out,
    ssize_t *outstrides,
    ssize_t *lines,
    ssize_t *linesshape,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_lstics_Hf(
    yxt_handle handle,
    uint16_t *data,
    ssize_t *strides,
    uint16_t *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    float *out,
    ssize_t *outstrides,
    ssize_t *lines,
    ssize_t *linesshape,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_lstics_dd(
    yxt_handle handle,
    double *data,
    ssize_t *strides,
    double *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    double *out,
    ssize_t *outstrides,
    ssize_t *lines,
    ssize_t *linesshape,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_lstics_fd(
    yxt_handle handle,
    float *data,
    ssize_t *strides,
    float *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    double *out,
    ssize_t *outstrides,
    ssize_t *lines,
    ssize_t *linesshape,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_lstics_id(
    yxt_handle handle,
    int32_t *data,
    ssize_t *strides,
    int32_t *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    double *out,
    ssize_t *outstrides,
    ssize_t *lines,
    ssize_t *linesshape,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_lstics_hd(
    yxt_handle handle,
    int16_t *data,
    ssize_t *strides,
    int16_t *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    double *out,
    ssize_t *outstrides,
    ssize_t *lines,
    ssize_t *linesshape,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

ICS_API int
yxt_lstics_Hd(
    yxt_handle handle,
    uint16_t *data,
    ssize_t *strides,
    uint16_t *data1,
    ssize_t *strides1,
    int32_t *mask,
    ssize_t *maskstrides,
    int32_t maskmode,
    double *out,
    ssize_t *outstrides,
    ssize_t *lines,
    ssize_t *linesshape,
    ssize_t *block,
    ssize_t *bins,
    ssize_t nbins,
    double filter,
    int nthreads);

/*
ICS_API int yxt_subtract_immobile_d(double* data, ssize_t* shape, ssize_t*
strides, int nthreads);

ICS_API int yxt_subtract_immobile_f(float* data, ssize_t* shape, ssize_t*
strides, int nthreads);
*/

ICS_API int
yxt_subtract_immobile_i(
    int32_t *data, ssize_t *shape, ssize_t *strides, int nthreads);

ICS_API int
yxt_subtract_immobile_h(
    int16_t *data, ssize_t *shape, ssize_t *strides, int nthreads);

ICS_API int
yxt_subtract_immobile_H(
    uint16_t *data, ssize_t *shape, ssize_t *strides, int nthreads);

ICS_API int
yxt_correct_bleaching_i(
    int32_t *data,
    ssize_t *shape,
    ssize_t *strides,
    double *mean,
    ssize_t *meanstrides,
    double filter,
    int nthreads);

ICS_API int
yxt_correct_bleaching_h(
    int16_t *data,
    ssize_t *shape,
    ssize_t *strides,
    double *mean,
    ssize_t *meanstrides,
    double filter,
    int nthreads);

ICS_API int
yxt_correct_bleaching_H(
    uint16_t *data,
    ssize_t *shape,
    ssize_t *strides,
    double *mean,
    ssize_t *meanstrides,
    double filter,
    int nthreads);

/*
ICS_API int yxt_median_f(float* data, ssize_t* shape, ssize_t* strides, float*
out, ssize_t* outstrides, int radius, int nthreads);

ICS_API int yxt_median_d(double* data, ssize_t* shape, ssize_t* strides,
double* out, ssize_t* outstrides, int radius, int nthreads);
*/

/* ics helper functions */

ICS_API ssize_t
radial(
    ssize_t *points,
    const ssize_t nlines,
    const ssize_t length,
    double *offset,
    const int mode);

ICS_API ssize_t
circle(const ssize_t radius, ssize_t *points, ssize_t npoints);

ICS_API ssize_t
logbins(const ssize_t size, const ssize_t nbins, ssize_t *bins);

ICS_API int
bins2times_f(
    ssize_t *bins, const ssize_t nbins, const float frametime, float *times);

ICS_API int
bins2times_d(
    ssize_t *bins, const ssize_t nbins, const double frametime, double *times);

ICS_API float
points2distances_f(
    const ssize_t *points,
    const ssize_t npoints,
    const float pixelsize,
    float *distances);

ICS_API double
points2distances_d(
    const ssize_t *points,
    const ssize_t npoints,
    const double pixelsize,
    double *distances);

ICS_API int32_t
nextpow2_i(int32_t n);

ICS_API int64_t
nextpow2_q(int64_t n);

/*
ICS_API int smooth_f(ssize_t size, float* data, float filter);
ICS_API int smooth_d(ssize_t size, double* data, double filter);
ICS_API int average_dd(ssize_t size, ssize_t nbins, double* data, ssize_t*
bins, double* out, double scale, double offset); ICS_API int average_df(ssize_t
size, ssize_t nbins, double* data, ssize_t* bins, float* out, double scale,
double offset); ICS_API int average_ff(ssize_t size, ssize_t nbins, float*
data, ssize_t* bins, float* out, double scale, double offset); ICS_API ssize_t
linbins(const ssize_t size, const ssize_t nbins, ssize_t* bins); ICS_API void
minmax_n(ssize_t npoints, ssize_t ndim, ssize_t* points, ssize_t* out);
*/

#ifdef ICS_SIMFCS
/* Compatibility for SimFCS. Deprecated */
ICS_API int
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
    int nthreads);
#endif

/* nlsp class */
ICS_API nlsp_handle
nlsp_new(int model, ssize_t *shape);

ICS_API void nlsp_del(nlsp_handle);

ICS_API int
nlsp_solve_f(
    nlsp_handle handle,
    float *data,
    ssize_t *strides,
    void *extra,
    double *guess,
    double *bounds,
    double *solution);

ICS_API int
nlsp_solve_d(
    nlsp_handle handle,
    double *data,
    ssize_t *strides,
    void *extra,
    double *guess,
    double *bounds,
    double *solution);

ICS_API int
nlsp_eval_f(nlsp_handle handle, float *data, const ssize_t *strides);

ICS_API int
nlsp_eval_d(nlsp_handle handle, double *data, const ssize_t *strides);

ICS_API int
nlsp_get(
    nlsp_handle handle, ssize_t *iter, ssize_t *st_cr, double *r1, double *r2);

ICS_API int
nlsp_set(
    nlsp_handle handle,
    ssize_t iter1,
    ssize_t iter2,
    double *eps,
    double eps_jac,
    double rs);

ICS_API int
ipcf_nlsp_1dpcf_f(
    const float *ipcf,
    const ssize_t *shape,
    const ssize_t *strides,
    const float *times,
    const float *distances,
    const float *args,
    const float *bounds,
    float *ix,
    const ssize_t *stridesx,
    float *ifx,
    const ssize_t *stridesfx,
    float *status,
    const ssize_t *stridesstatus,
    const float *settings,
    const int average,
    const int nthreads);

/* DFT */

ICS_API int
yxt_dft_ff(
    float *data,
    ssize_t *shape,
    ssize_t *strides,
    float *out,
    ssize_t *outshape,
    ssize_t *outstrides,
    int nthreads);

ICS_API int
yxt_dft_if(
    int *data,
    ssize_t *shape,
    ssize_t *strides,
    float *out,
    ssize_t *outshape,
    ssize_t *outstrides,
    int nthreads);

ICS_API int
yxt_dft_hf(
    int16_t *data,
    ssize_t *shape,
    ssize_t *strides,
    float *out,
    ssize_t *outshape,
    ssize_t *outstrides,
    int nthreads);

ICS_API int
yxt_dft_Hf(
    uint16_t *data,
    ssize_t *shape,
    ssize_t *strides,
    float *out,
    ssize_t *outshape,
    ssize_t *outstrides,
    int nthreads);

ICS_API int
yxt_dft_dd(
    double *data,
    ssize_t *shape,
    ssize_t *strides,
    double *out,
    ssize_t *outshape,
    ssize_t *outstrides,
    int nthreads);

ICS_API int
yxt_dft_id(
    int *data,
    ssize_t *shape,
    ssize_t *strides,
    double *out,
    ssize_t *outshape,
    ssize_t *outstrides,
    int nthreads);

ICS_API int
yxt_dft_hd(
    int16_t *data,
    ssize_t *shape,
    ssize_t *strides,
    double *out,
    ssize_t *outshape,
    ssize_t *outstrides,
    int nthreads);

ICS_API int
yxt_dft_Hd(
    uint16_t *data,
    ssize_t *shape,
    ssize_t *strides,
    double *out,
    ssize_t *outshape,
    ssize_t *outstrides,
    int nthreads);

/* Deconvolution */

ICS_API int
zyx_deconv_ff(
    const float *image,
    const ssize_t *shapei,
    const ssize_t *stridesi,
    const float *psf,
    const ssize_t *shapep,
    const ssize_t *stridesp,
    float *out,
    const ssize_t *shapeo,
    const ssize_t *strideso,
    int niter,
    int mode,
    int nthreads);

ICS_API int
zyx_deconv_dd(
    const double *image,
    const ssize_t *shapei,
    const ssize_t *stridesi,
    const double *psf,
    const ssize_t *shapep,
    const ssize_t *stridesp,
    double *out,
    const ssize_t *shapeo,
    const ssize_t *strideso,
    int niter,
    int mode,
    int nthreads);

ICS_API int
zyx_deconv_Hf(
    const uint16_t *image,
    const ssize_t *shapei,
    const ssize_t *stridesi,
    const uint16_t *psf,
    const ssize_t *shapep,
    const ssize_t *stridesp,
    float *out,
    const ssize_t *shapeo,
    const ssize_t *strideso,
    int niter,
    int mode,
    int nthreads);

#ifdef ICS_PSF
/* Point Spread Function */

ICS_API int
psf(int type,
    double *data,
    ssize_t *shape,
    double *uvdim,
    double M,
    double sinalpha,
    double beta,
    double gamma,
    int intsteps);

ICS_API int
psf_obsvol(
    ssize_t dimz,
    ssize_t dimr,
    ssize_t dimd,
    double *obsvol,
    double *ex_psf,
    double *em_psf,
    double *detector);

ICS_API int
psf_pinhole_kernel(int corners, double *out, ssize_t dim, double radius);

ICS_API int
psf_zr2zxy(double *data, double *out, ssize_t dimz, ssize_t dimr);

ICS_API int
psf_gaussian2d(double *out, ssize_t *shape, double *sigma);

ICS_API int
psf_gaussian_sigma(
    double *sz,
    double *sr,
    double lex,
    double lem,
    double NA,
    double n,
    double r,
    int widefield,
    int paraxial);
#endif

#ifdef __cplusplus
}
#endif

#ifdef ICS_FFT2D
/* FFT2D library */
#include "fft2d.h"
#endif

#endif /* ICS_H */
