/* nlsp.cpp

Implementation of fitting functions for the ICS library.

Copyright (c) 2016-2024, Christoph Gohlke
This source code is distributed under the BSD 3-Clause license.

Refer to the header file 'ics.h' for documentation and license.

*/

#include "mkl_types.h"
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <limits>

#include "ics.h"
#include "ics.hpp"

/* Objective function for fitting 1D pCF. */
void
nlsp_1dpcf(MKL_INT *m, MKL_INT *n, double *x, double *f, void *extra)
{
    nlsp_extra_t const *e = (nlsp_extra_t *)extra;
    const double *yy = (double *)e->y;  // function values
    const double *xx = (double *)e->x;  // x axis values and constants

    const double dc = x[0];
    const double D = x[1];
    const double w2 = xx[*m];      // squared waist
    const double d2 = xx[*m + 1];  // squared distance between points

    for (ssize_t i = 0; i < *m; i++) {
        const double df = 4.0 * D * xx[i];       // 4D tau term
        const double g0 = dc / (1.0 + df / w2);  // g(tau) term
        if (df == 0.0) {
            f[i] = g0;
        }
        else {
            f[i] = g0 * exp(-d2 / (df + w2));  // g(tau*gaussian
        }
    }
    if (yy != NULL) {
        for (ssize_t i = 0; i < *m; i++) {
            const double y = yy[i];
            f[i] = (y > 0.0 ? y : 0.0) - f[i];
        }
    }
}

/* Initialize solver of non-linear least squares problem with linear boundary
constraints.

Parameters
----------

model : int
    Select the objective function. One of `ICS_NLSP_`.
shape : ssize_t*
    The shape of the input data. Defines the length of x.

*/
nlsp::nlsp(int model, ssize_t *shape)
{
    switch (model) {
        case ICS_NLSP_1DPCF:
            objective_ = &nlsp_1dpcf;
            n_ = 2;
            ndim_ = 1;
            break;
        default:
            throw ICS_VALUE_ERROR;
    }

    // shape and dimensions
    ssize_t product = 1;
    for (int i = 0; i < ndim_; i++) {
        const ssize_t s = shape[i];
        if (s <= 0) {
            throw ICS_VALUE_ERROR;
        }
        shape_[i] = s;
        product *= s;
    }
    if (product > 2147483647) {
        throw ICS_VALUE_ERROR;
    }
    m_ = (MKL_INT)product;

    // memory allocation
    x_ = (double *)mkl_malloc(sizeof(double) * n_, 64);
    if (x_ == NULL)
        throw ICS_MEMORY_ERROR;
    y_ = (double *)mkl_malloc(sizeof(double) * m_, 64);
    if (y_ == NULL)
        throw ICS_MEMORY_ERROR;
    fvec_ = (double *)mkl_malloc(sizeof(double) * m_, 64);
    if (fvec_ == NULL)
        throw ICS_MEMORY_ERROR;
    fjac_ = (double *)mkl_malloc(sizeof(double) * m_ * n_, 64);
    if (fjac_ == NULL)
        throw ICS_MEMORY_ERROR;
    lw_ = (double *)mkl_malloc(sizeof(double) * n_, 64);
    if (lw_ == NULL)
        throw ICS_MEMORY_ERROR;
    up_ = (double *)mkl_malloc(sizeof(double) * n_, 64);
    if (up_ == NULL)
        throw ICS_MEMORY_ERROR;

    eps_jac_ = 1e-10;
    for (int i = 0; i < 6; i++) {
        eps_[i] = 1e-10;
    }

    iter1_ = 400;
    iter2_ = 200;
    rs_ = 0.0;

    // set initial values
    for (int i = 0; i < m_; i++) {
        fvec_[i] = 0.0;
    }
    for (int i = 0; i < m_ * n_; i++) {
        fjac_[i] = 0.0;
    }

    // set guess and bounds
    for (int i = 0; i < n_; i++) {
        x_[i] = 0.0;
        lw_[i] = -1e9;
        up_[i] = 1e9;
    }

    extra_ = new nlsp_extra_t;
    extra_->x = NULL;
    extra_->y = NULL;
}

/* Class destructor: release buffers and handles. */
nlsp::~nlsp()
{
    delete extra_;

    if (handle_ != NULL)
        dtrnlsp_delete(&handle_);
    if (handle_bc_ != NULL)
        dtrnlspbc_delete(&handle_bc_);
    if (up_ != NULL)
        mkl_free(up_);
    if (lw_ != NULL)
        mkl_free(lw_);
    if (fjac_ != NULL)
        mkl_free(fjac_);
    if (fvec_ != NULL)
        mkl_free(fvec_);
    if (x_ != NULL)
        mkl_free(x_);
    if (y_ != NULL)
        mkl_free(y_);
}

/* Set various solver parameters.

Parameters
----------

iter1 : int
    Specifies the maximum number of iterations.
iter2
    Specifies the maximum number of iterations of trial step calculation.
eps : double*
    Array of size 6 values defining the stopping criteria.
eps_jac : double
    Precision of the Jacobian matrix calculation.
rs : double
    Definition of initial size of the trust region
    (boundary of the trial step).

*/
void
nlsp::set(MKL_INT iter1, MKL_INT iter2, double *eps, double eps_jac, double rs)
{
    rs_ = rs;
    if (iter1 > 0) {
        iter1_ = iter1;
    }
    if (iter2 > 0) {
        iter2_ = iter2;
    }
    if (eps_jac > 0.0) {
        eps_jac_ = eps_jac;
    }
    if (eps != NULL) {
        for (int i = 0; i < 6; i++) {
            if (eps[i] > 0) {
                eps_[i] = eps[i];
            }
        }
    }
}

/* Solves a nonlinear least squares problem using RCI and the Trust-Region
algorithm.

Parameters
----------

...

*/
template <typename Ti>
void
nlsp::solve(
    const Ti *data,
    const ssize_t *strides,
    void *extra,
    double *guess,
    double *bounds,
    double *solution)
{
    // results of input parameter checking
    MKL_INT info[6];
    // reverse communication interface parameter
    MKL_INT RCI_Request = 0;
    // function status
    MKL_INT res = 0;
    // controls of rci cycle
    int successful = 0;

    if (handle_ != NULL) {
        dtrnlsp_delete(&handle_);
        handle_ = NULL;
    }
    if (handle_bc_ != NULL) {
        dtrnlspbc_delete(&handle_bc_);
        handle_bc_ = NULL;
    }

    // copy data to y
    const ssize_t stride0 = strides[0];
    char *pdata = (char *)data;
    if (ndim_ == 1) {
        for (int i = 0; i < m_; i++) {
            y_[i] = (double)(*((Ti *)pdata));
            pdata += stride0;
        }
    }
    else {
        // TODO: implement 2 and 3 dimensions
        throw ICS_NOTIMPLEMENTD_ERROR;
    }
    extra_->y = (void *)y_;
    extra_->x = extra;

    // initialize buffers
    for (int i = 0; i < m_; i++) {
        fvec_[i] = 0.0;
    }
    for (int i = 0; i < m_ * n_; i++) {
        fjac_[i] = 0.0;
    }

    // set initial guess
    if (guess != NULL) {
        for (int i = 0; i < n_; i++) {
            x_[i] = guess[i];
        }
    }

    // set bounds
    if (bounds != NULL) {
        for (int i = 0; i < n_; i++) {
            lw_[i] = bounds[2 * i];
            up_[i] = bounds[2 * i + 1];
        }
    }

    // initialize solver (allocate memory, set initial values)
    if (bounds != NULL) {
        res = dtrnlspbc_init(
            &handle_bc_, &n_, &m_, x_, lw_, up_, eps_, &iter1_, &iter2_, &rs_);
    }
    else {
        res =
            dtrnlsp_init(&handle_, &n_, &m_, x_, eps_, &iter1_, &iter2_, &rs_);
    }
    if (res != TR_SUCCESS) {
        throw res;
    }

    /*
    objective_(&m_, &n_, x_, fvec_, extra_);
    djacobix(objective_, &n_, &m_, fjac_, x_, &eps_jac_, extra_);
    */

    // check the correctness of handle and arrays containing Jacobian matrix,
    // objective function, lower and upper bounds, and stopping criteria
    if (handle_ == NULL) {
        res = dtrnlspbc_check(
            &handle_bc_, &n_, &m_, fjac_, fvec_, lw_, up_, eps_, info);
    }
    else {
        res = dtrnlsp_check(&handle_, &n_, &m_, fjac_, fvec_, eps_, info);
    }
    if (res != TR_SUCCESS) {
        throw res;
    }
    else {
        if (info[0] != 0 ||  // handle is not valid.
            info[1] != 0 ||  // fjac array is not valid.
            info[2] != 0 ||  // fvec array is not valid.
            info[3] != 0 ||  // LW or eps array is not valid.
            ((handle_ == NULL) && (info[4] != 0 ||  // UP array is not valid.
                                   info[5] != 0))   // eps array is not valid.
        ) {
            throw ICS_VALUE_ERROR;
        }
    }

    // RCI cycle
    while (successful == 0) {
        // call TR solver
        if (handle_ == NULL) {
            res = dtrnlspbc_solve(&handle_bc_, fvec_, fjac_, &RCI_Request);
        }
        else {
            res = dtrnlsp_solve(&handle_, fvec_, fjac_, &RCI_Request);
        }
        if (res != TR_SUCCESS) {
            throw res;
        }

        switch (RCI_Request) {
            case 1:
                // recalculate function value
                objective_(&m_, &n_, x_, fvec_, extra_);
                break;
            case 2:
                // compute jacobi matrix
                res = djacobix(
                    objective_, &n_, &m_, fjac_, x_, &eps_jac_, extra_);
                if (res != TR_SUCCESS) {
                    throw res;
                }
                break;
            case -1:
            case -2:
            case -3:
            case -4:
            case -5:
            case -6:
                // exit RCI cycle
                successful = 1;
                break;
        }
    }

    // copy solution
    if (solution != NULL) {
        for (int i = 0; i < n_; i++) {
            solution[i] = x_[i];
        }
    }
}

/* Evaluate function using current solution vector.

Computed values are stored in `data` array.

*/
template <typename To>
void
nlsp::eval(To *data, const ssize_t *strides)
{
    extra_->y = NULL;
    objective_(&m_, &n_, x_, y_, extra_);

    // copy to data
    const ssize_t stride0 = strides[0];
    char *pdata = (char *)data;
    if (ndim_ == 1) {
        for (int i = 0; i < m_; i++) {
            *((To *)pdata) = (To)y_[i];
            pdata += stride0;
        }
    }
    else {
        // TODO: implement 2 and 3 dimensions
        throw ICS_NOTIMPLEMENTD_ERROR;
    }
}

/* Get solution statuses.

Parameters
----------

iter : int
    Contains the current number of iterations.
st_cr : int
    Contains the stop criterion.
r1 : double
    Contains the residual, (||y - f(x)||) given the initial x.
r2 : double
    Contains the final residual, that is, the value of the function
    (||y - f(x)||) of the final x resulting from the algorithm operation.

*/
void
nlsp::get(MKL_INT *iter, MKL_INT *st_cr, double *r1, double *r2)
{
    MKL_INT res;
    if (handle_ == NULL) {
        res = dtrnlspbc_get(&handle_bc_, iter, st_cr, r1, r2);
    }
    else {
        res = dtrnlsp_get(&handle_, iter, st_cr, r1, r2);
    }
    if (res != TR_SUCCESS) {
        throw res;
    }
}

/* Fit 1D pair correlation functions to the output of the yxt::ipcf function.

Parameters
----------

ipcf : Ti*
    Pointer to 4D array of cross correlation carpets at each pixel.
    The order of the axes is length, width, npoints, nbins.
    Pair correlation functions starting with NaN ill not be processed.
shape : ssize_t*
    Pointer to four integers defining the sizes in y, x, points, and bins
    dimensions of the `ipcf` array:
    [image length, image width, number of angles/points, number time points].
strides : ssize_t*
    Pointer to 4 integers defining the strides of the `ipcf` array.
    The last stride must be sizeof(To), i.e. the last axis is contiguous.
    Strides are the number of bytes required in each dimension to advance from
    one item to the next within the dimension.
times : Tx*
    Pointer to array of length shape[3] containing times of the bins.
distances : Tx*
    If `average`, pointer to a single value containing the average squared
    distance.
    If not `average`, pointer to array of length shape[2] containing squared
    distances of the points.
args : Tx*
    Pointer to array of 7 values:
    DCinit : initial guess of DC variable.
    Dinit : initial guess of Diffusion coefficient variable.
    W2 : constant.
bounds : Tx*
    If NULL, a solver without bounds constraints is used.
    If not NULL, a pointer to array of 4 values:
    DClower : lower bound of DC.
    DCupper : upper bound of DC.
    Dlower : lower bound of D.
    Dupper : upper bound of D.
ix : Tx*
    Pointer to 4D output array of fitted DC and D values.
    If not `average`, the array shape is (shape[0], shape[1], shape[2], 2).
    If `average`, the array shape is (shape[0], shape[1], 1, 2).
stridesx : ssize_t*
    Pointer to 4 integers defining the strides of the `ix` array.
ifx : To*
    Pointer to 4D output array of computed function values.
    If not `average`, the array shape is
    (shape[0], shape[1], shape[2], shape[3]).
    If `average`, the array shape is (shape[0], shape[1], 1, shape[3]).
stridesfx : ssize_t*
    Pointer to 4 integers defining the strides of the `ifx` array.
status : Tx*
    Pointer to 4D output array of 4 status values returned by the solver.
    If not `average`, the array shape is (shape[0], shape[1], shape[2], 4).
    If `average`, the array shape is (shape[0], shape[1], 1, 4).
    iter (int):
        Contains the current number of iterations.
    st_cr (int):
        Contains the stop criterion.
    r1:
        Contains the residual, (||y - f(x)||) given the initial x.
    r2:
        Contains the final residual for the final x.
stridesstatus : ssize_t*
    Pointer to 4 integers defining the strides of the `status` array.
settings : Tx*
    If not NULL, pointer to array of 10 values defining solver properties:
    iter1 :
        maximum number of iterations (integer).
    iter2 :
        maximum number of iterations of trial step calculation (integer).
    eps[6] :
        stopping criteria.
    eps_jac :
        precision of Jacobian matrix calculation.
    rs :
        definition of initial size of the trust region (boundary of the trial
        step).
average : bool
    If true (not zero), average the values of the third axis (points) before
    fitting.
    If false (zero), fit all pair correlation functions individually.
nthreads : int
    Number of OpenMP threads to use for parallelizing loops along the first
    axis. Set to zero for OpenMP default. For each thread, a nlsp instance
    is allocated.

*/
template <typename Ti, typename To, typename Tx>
void
ipcf_nlsp_1dpcf(
    const Ti *ipcf,
    const ssize_t *shape,
    const ssize_t *strides,
    const Tx *times,
    const Tx *distances,
    const Tx *args,
    const Tx *bounds,
    Tx *ix,
    const ssize_t *stridesx,
    To *ifx,
    const ssize_t *stridesfx,
    Tx *status,
    const ssize_t *stridesstatus,
    const Tx *settings,
    const bool average,
    const int nthreads)
{
    if ((ipcf == NULL) || (shape == NULL) || (times == NULL)) {
        throw ICS_VALUE_ERROR;
    }

    char *pipcf = (char *)ipcf;
    char *pifx = (char *)ifx;
    char *pix = (char *)ix;
    char *pstatus = (char *)status;

    const bool bix = !((ix == NULL) || (stridesx == NULL));
    const bool bifx = !((ifx == NULL) || (stridesfx == NULL));
    const bool bstatus = !((status == NULL) || (stridesstatus == NULL));

    const ssize_t ntimes = shape[3] - 2;  // skip first and last point
    const double guess[2] = {(double)args[0], (double)args[1]};  // dc, D

    // lower and upper bounds
    double *bounds_ = NULL;
    if (bounds != NULL) {
        bounds_ = new double[4];
        for (ssize_t i = 0; i < 4; i++) {
            bounds_[i] = (double)bounds[i];
        }
    }
    // solver eps
    double *eps_ = NULL;
    if (settings != NULL) {
        eps_ = new double[6];
        for (int i = 0; i < 6; i++) {
            eps_[i] = (double)settings[i + 2];
        }
    }

#pragma omp parallel num_threads(nthreads)
    {
        // thread-local variables
        nlsp *worker;
        try {
            worker = new nlsp(ICS_NLSP_1DPCF, (ssize_t *)(&ntimes));
            if (settings != NULL) {
                worker->set(
                    (MKL_INT)settings[0],
                    (MKL_INT)settings[1],
                    eps_,
                    (double)settings[8],
                    (double)settings[9]);
            }
        }
        catch (...) {
            worker = NULL;
        }

        double *averaged = NULL;
        if (average) {
            averaged = (double *)mkl_malloc(shape[3] * sizeof(double), 64);
        }
        double *solution = (double *)mkl_malloc(2 * sizeof(double), 64);
        double *extra =
            (double *)mkl_malloc((ntimes + 2) * sizeof(double), 64);
        if (extra != NULL) {
            for (int i = 0; i < ntimes; i++) {
                extra[i] = (double)times[i + 1];  // skip first time point
            }
            extra[ntimes] = (double)args[2];           // w2
            extra[ntimes + 1] = (double)distances[0];  // squared distance
        }
#pragma omp for
        for (ssize_t i = 0; i < shape[0]; i++) {
            if ((worker == NULL) || (extra == NULL))
                continue;
            for (ssize_t j = 0; j < shape[1]; j++) {
                if (average) {
                    if (averaged == NULL)
                        continue;

                    // average points/angles
                    const char *pd = pipcf + i * strides[0] + j * strides[1];
                    if (isnan(*((Ti *)pd))) {
                        continue;  // skip masked
                    }
                    for (ssize_t m = 0; m < shape[3]; m++) {
                        averaged[m] = 0.0;
                    }
                    for (ssize_t k = 0; k < shape[2]; k++) {
                        for (ssize_t m = 0; m < shape[3]; m++) {
                            averaged[m] += (double)(*(
                                (Ti *)(pd + k * strides[2] + m * strides[3])));
                        }
                    }
                    for (ssize_t m = 0; m < shape[3]; m++) {
                        averaged[m] /= (double)shape[2];
                    }

                    // solve
                    try {
                        const ssize_t stride = sizeof(double);
                        worker->solve(
                            &averaged[1],
                            &stride,
                            (void *)extra,
                            (double *)guess,
                            (double *)bounds_,
                            solution);  // skip first point
                    }
                    /*
                    catch (int e) {
                        if (bstatus) {
                            const ssize_t offset = i * stridesstatus[0] + j *
                    stridesstatus[1];
                            *((Tx *)(pstatus + offset + stridesstatus[3])) =
                    (Tx)e;
                        }
                        continue;
                    }
                    */
                    catch (...) {
#ifdef _DEBUG
                        throw;
#else
                        continue;
#endif
                    }

                    if (bifx) {
                        // evaluate
                        worker->eval(
                            (To *)(pifx
                                   + i * stridesfx[0]
                                   + j * stridesfx[1]
                                   + stridesfx[3]),
                            &stridesfx[3]);  // skip first point
                    }
                    if (bix) {
                        // save solution
                        const ssize_t offset =
                            i * stridesx[0] + j * stridesx[1];
                        *((Tx *)(pix + offset)) = (Tx)solution[0];
                        *((Tx *)(pix + offset + stridesx[3])) =
                            (Tx)solution[1];
                    }
                    if (bstatus) {
                        MKL_INT iter, st_cr;
                        double r1, r2;
                        worker->get(&iter, &st_cr, &r1, &r2);

                        const ssize_t offset =
                            i * stridesstatus[0] + j * stridesstatus[1];
                        *((Tx *)(pstatus + offset)) = (Tx)iter;
                        *((Tx *)(pstatus + offset + stridesstatus[3])) =
                            (Tx)st_cr;
                        *((Tx *)(pstatus + offset + stridesstatus[3] * 2)) =
                            (Tx)r1;
                        *((Tx *)(pstatus + offset + stridesstatus[3] * 3)) =
                            (Tx)r2;
                    }
                    continue;
                }

                // non-average mode
                for (ssize_t k = 0; k < shape[2]; k++) {
                    const Ti *d = (Ti *)(
                        pipcf
                        + i * strides[0]
                        + j * strides[1]
                        + k * strides[2]
                        + strides[3]);  // skip first point
                    if (isnan(d[0])) {
                        // skip masked
                        continue;
                    }

                    // solve
                    extra[ntimes + 1] = (double)distances[k];
                    try {
                        worker->solve(
                            d,
                            &strides[3],
                            (void *)extra,
                            (double *)guess,
                            (double *)bounds_,
                            solution);
                    }
                    catch (...) {
#ifdef _DEBUG
                        throw;
#else
                        continue;
#endif
                    }
                    if (bifx) {
                        // evaluate
                        worker->eval(
                            (To *)(pifx
                                   + i * stridesfx[0]
                                   + j * stridesfx[1]
                                   + k * stridesfx[2]
                                   + stridesfx[3]),
                            &stridesfx[3]);  // skip first point
                    }
                    if (bix) {
                        // save solution
                        const ssize_t offset = i * stridesx[0] +
                                               j * stridesx[1] +
                                               k * stridesx[2];
                        *((Tx *)(pix + offset)) = (Tx)solution[0];
                        *((Tx *)(pix + offset + stridesx[3])) =
                            (Tx)solution[1];
                    }
                    if (bstatus) {
                        MKL_INT iter, st_cr;
                        double r1, r2;
                        worker->get(&iter, &st_cr, &r1, &r2);

                        const ssize_t offset = i * stridesstatus[0] +
                                               j * stridesstatus[1] +
                                               k * stridesstatus[2];
                        *((Tx *)(pstatus + offset)) = (Tx)iter;
                        *((Tx *)(pstatus + offset + stridesstatus[3])) =
                            (Tx)st_cr;
                        *((Tx *)(pstatus + offset + stridesstatus[3] * 2)) =
                            (Tx)r1;
                        *((Tx *)(pstatus + offset + stridesstatus[3] * 3)) =
                            (Tx)r2;
                    }
                }
            }
        }
        if (worker != NULL)
            delete worker;
        if (extra != NULL)
            mkl_free(extra);
        if (averaged != NULL)
            mkl_free(averaged);
        if (solution != NULL)
            mkl_free(solution);
    }
    if (bounds_ != NULL)
        delete bounds_;
    if (eps_ != NULL)
        delete eps_;

    /* Release internal Intel(R) MKL memory that might be used for
    computations. NOTE: It is important to call the routine below to avoid
    memory leaks unless you disable Intel(R) MKL Memory Manager */
    mkl_free_buffers();
}

/* C API */

nlsp_handle
nlsp_new(int model, ssize_t *shape)
{
    try {
        return reinterpret_cast<nlsp_handle>(new nlsp(model, shape));
    }
    catch (...) {
        return NULL;
    }
}

void
nlsp_del(nlsp_handle handle)
{
    try {
        delete reinterpret_cast<nlsp_handle>(handle);
    }
    catch (...) {
        ;
    }
}

int
nlsp_solve_f(
    nlsp_handle handle,
    float *data,
    ssize_t *strides,
    void *extra,
    double *guess,
    double *bounds,
    double *solution)
{
    try {
        reinterpret_cast<nlsp_handle>(handle)->solve(
            data, strides, extra, guess, bounds, solution);
    }
    catch (int e) {
        return e;
    }
    catch (...) {
        return ICS_ERROR;
    }
    return ICS_OK;
}

int
nlsp_solve_d(
    nlsp_handle handle,
    double *data,
    ssize_t *strides,
    void *extra,
    double *guess,
    double *bounds,
    double *solution)
{
    try {
        reinterpret_cast<nlsp_handle>(handle)->solve(
            data, strides, extra, guess, bounds, solution);
    }
    catch (int e) {
        return e;
    }
    catch (...) {
        return ICS_ERROR;
    }
    return ICS_OK;
}

int
nlsp_eval_f(nlsp_handle handle, float *data, const ssize_t *strides)
{
    try {
        reinterpret_cast<nlsp_handle>(handle)->eval(data, strides);
    }
    catch (int e) {
        return e;
    }
    catch (...) {
        return ICS_ERROR;
    }
    return ICS_OK;
}

int
nlsp_eval_d(nlsp_handle handle, double *data, const ssize_t *strides)
{
    try {
        reinterpret_cast<nlsp_handle>(handle)->eval(data, strides);
    }
    catch (int e) {
        return e;
    }
    catch (...) {
        return ICS_ERROR;
    }
    return ICS_OK;
}

int
nlsp_get(
    nlsp_handle handle, ssize_t *iter, ssize_t *st_cr, double *r1, double *r2)
{
    MKL_INT iter_, st_cr_;
    try {
        reinterpret_cast<nlsp_handle>(handle)->get(&iter_, &st_cr_, r1, r2);
        *iter = (ssize_t)iter_;
        *st_cr = (ssize_t)st_cr_;
    }
    catch (int e) {
        return e;
    }
    catch (...) {
        return ICS_ERROR;
    }
    return ICS_OK;
}

int
nlsp_set(
    nlsp_handle handle,
    ssize_t iter1,
    ssize_t iter2,
    double *eps,
    double eps_jac,
    double rs)
{
    try {
        reinterpret_cast<nlsp_handle>(handle)->set(
            (MKL_INT)iter1, (MKL_INT)iter2, eps, eps_jac, rs);
    }
    catch (int e) {
        return e;
    }
    catch (...) {
        return ICS_ERROR;
    }
    return ICS_OK;
}

int
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
    const int nthreads)
{
    try {
        ipcf_nlsp_1dpcf(
            ipcf,
            shape,
            strides,
            times,
            distances,
            args,
            bounds,
            ix,
            stridesx,
            ifx,
            stridesfx,
            status,
            stridesstatus,
            settings,
            average != 0,
            nthreads);
    }
    catch (int e) {
        return e;
    }
    catch (...) {
        return ICS_ERROR;
    }
    return ICS_OK;
}