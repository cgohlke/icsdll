/* ics.cpp

Implementation of functions for the ICS library.

Copyright (c) 2016-2023, Christoph Gohlke
This source code is distributed under the BSD 3-Clause license.

Refer to the header file 'ics.h' for documentation and license.

*/

#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <limits>

#include "ics.h"
#include "ics.hpp"

extern "C" ICS_API char *ICS_VERSION = ICS_VERSION_STR;

/** Helper functions **/

/* Return next highest power of two of 32/64-bit value */
int32_t
nextpow2_i(int32_t n)
{
    return nextpow2(n);
}

int64_t
nextpow2_q(int64_t n)
{
    return nextpow2(n);
}

/* Append x and y to points if not already in points. */
inline void
circle_add(ssize_t &npoints, ssize_t *points, const ssize_t x, const ssize_t y)
{
    for (ssize_t i = 0; i < 2 * npoints; i += 2) {
        if ((x == points[i]) && (y == points[i + 1])) {
            return;
        }
    }
    points[2 * npoints] = x;
    points[2 * npoints + 1] = y;
    npoints++;
}

/* Return pseudo angle of x, y vector. */
inline double
pseudoangle(const ssize_t x, const ssize_t y)
{
    const double p = 1.0 + (double)x / (double)(abs(x) + abs(y));
    return (y < 0) ? p : -p;
}

/* Bubble sort x, y coordinates by increasing (pseudo)angle. */
void
circle_sort(const ssize_t npoints, ssize_t *points)
{
    bool exchanged;
    do {
        exchanged = false;
        for (ssize_t i = 0; i < 2 * (npoints - 1); i += 2) {
            if (pseudoangle(points[i], points[i + 1]) >
                pseudoangle(points[i + 2], points[i + 3])) {
                ssize_t t;
                t = points[i];
                points[i] = points[i + 2];
                points[i + 2] = t;
                t = points[i + 1];
                points[i + 1] = points[i + 3];
                points[i + 3] = t;
                exchanged = true;
            }
        }
    } while (exchanged);
}

/* Compute x, y integer coordinates of circle of radius.

Return number of points.

Parameters
----------

radius : ssize_t
    Radius of the circle to compute.
points : ssize_t*
    Output array to store the circle's x, y coordinates.
    It must be large enough to hold 2 * npoints values.
    Depending on `npoints` and `radius`, points might contain duplicate
    coordinates.
    If NULL, npoints will be returned without computing any coordinates.
npoints : ssize_t
    Number of circular points/angles to compute.
    If zero, a circle is "drawn" using 4 * floor(sqrt(2) * radius + 0.5)
    points.
*/
ssize_t
circle(const ssize_t radius, ssize_t *points, ssize_t npoints)
{
    if ((radius < 0) || (npoints < 0) || ((radius == 0) && (npoints == 0))) {
        return 0;
    }

    if (points == NULL) {
        if (npoints <= 0) {
            npoints = 4 * (ssize_t)floor(sqrt(2.0) * (double)radius + 0.5);
        }
        return npoints;
    }

    if (radius == 0) {
        if (npoints <= 0) {
            npoints = 4 * (ssize_t)floor(sqrt(2.0) * (double)radius + 0.5);
        }
        for (ssize_t i = 0; i < npoints; i++) {
            points[2 * i] = 0;
            points[2 * i + 1] = 0;
        }
    }

    if (npoints > 0) {
        const double da = 8.0 * atan(1.) / (double)(npoints);
        for (ssize_t i = 0; i < npoints; i++) {
            const double a = (double)i * da;
            points[2 * i] = (ssize_t)nearbyint(cos(a) * (double)radius);
            points[2 * i + 1] = (ssize_t)nearbyint(sin(a) * (double)radius);
        }
    }
    else {
        ssize_t f = 1 - radius;
        ssize_t dx = 1;
        ssize_t dy = -2 * radius;
        ssize_t x = 0;
        ssize_t y = radius;

        circle_add(npoints, points, 0, +radius);
        circle_add(npoints, points, 0, -radius);
        circle_add(npoints, points, +radius, 0);
        circle_add(npoints, points, -radius, 0);

        while (x < y) {
            if (f >= 0) {
                y -= 1;
                dy += 2;
                f += dy;
            }
            x += 1;
            dx += 2;
            f += dx;
            circle_add(npoints, points, +x, +y);
            circle_add(npoints, points, -x, +y);
            circle_add(npoints, points, +x, -y);
            circle_add(npoints, points, -x, -y);
            circle_add(npoints, points, +y, +x);
            circle_add(npoints, points, -y, +x);
            circle_add(npoints, points, +y, -x);
            circle_add(npoints, points, -y, -x);
        }

        circle_sort(npoints, points);
    }

    return npoints;
}

/* Compute distances to origin from integer coordinates. */
template <typename T>
T
points2distances(
    const ssize_t *points,
    const ssize_t npoints,
    const T pixelsize,
    T *distances)
{
    if ((npoints < 1) || (points == NULL) || (distances == NULL)) {
        throw ICS_VALUE_ERROR;
    }
    double average = 0.0;
    for (ssize_t i = 0; i < npoints; i++) {
        const T x = (T)points[2 * i];
        const T y = (T)points[2 * i + 1];
        const T d = (T)(sqrt(x * x + y * y) * pixelsize);
        distances[i] = d;
        average += (double)d;
    }
    return (T)(average / (double)npoints);
}

float
points2distances_f(
    const ssize_t *points,
    const ssize_t npoints,
    const float pixelsize,
    float *distances)
{
    try {
        return points2distances(points, npoints, pixelsize, distances);
    }
    catch (...) {
        return -1.0;
    }
}

double
points2distances_d(
    const ssize_t *points,
    const ssize_t npoints,
    const double pixelsize,
    double *distances)
{
    try {
        return points2distances(points, npoints, pixelsize, distances);
    }
    catch (...) {
        return -1.0;
    }
}

/* Compute x, y integer coordinates of line segments through center of circle.

Return number of points computed.

Parameters
----------

points : ssize_t*
    Output array to store the lines' x, y coordinates.
    It must be large enough to hold 2 * length * nlines values.
length : ssize_t
    Length of lines to compute.
nlines : ssize_t
    Number of lines to compute.
offset : double*
    Offset of coordinates in x and y dimensions.
mode : int
    Defines if line segments are radius (ICS_RADIUS 1) or diameter
    (ICS_DIAMETER 2).
*/
ssize_t
radial(
    ssize_t *points,
    const ssize_t nlines,
    const ssize_t length,
    double *offset,
    const int mode)
{
    if ((length < 1) || (nlines < 0) || (mode < 1) || (mode > 2) ||
        (points == NULL)) {
        return -1;
    }

    const double da = 2.0 * M_PI / (double)nlines / mode;
    const double xoffset = offset == NULL ? 0.0 : offset[0];
    const double yoffset = offset == NULL ? 0.0 : offset[1];
    ssize_t p = 0;

    for (ssize_t i = 0; i < nlines; i++) {
        const double x0 = (double)length * cos(i * da) / mode;
        const double y0 = (double)length * sin(i * da) / mode;
        const double x1 = mode == ICS_DIAMETER ? -x0 : 0.0;
        const double y1 = mode == ICS_DIAMETER ? -y0 : 0.0;
        const double dx = (x0 - x1) / length;
        const double dy = (y0 - y1) / length;
        for (ssize_t j = 0; j < length; j++) {
            points[p++] = (ssize_t)nearbyint(xoffset + x1 + j * dx);
            points[p++] = (ssize_t)nearbyint(yoffset + y1 + j * dy);
        }
    }

    return p / 2;
}

/* Set `bins` array to `nbins` exponentially increasing integers up to `size`.

Return number of integers stored in `bins` array.

For example, `logbins(32, 32)` returns `20` and sets the bins array to
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 25, 28, 32].

*/
ssize_t
logbins(const ssize_t size, const ssize_t nbins, ssize_t *bins)
{
    if ((size <= 2) || (nbins <= 2)) {
        return ICS_VALUE_ERROR;
    }
    const double dx = (log(double(size)) / log(2.0)) / double(nbins - 1);
    ssize_t n = 0;
    bins[n] = 1;
    for (ssize_t i = 1; i < nbins; i++) {
        const ssize_t b = (ssize_t)trunc(pow(2.0, i * dx));
        if (b > bins[n]) {
            bins[++n] = b;
        }
    }
    return n + 1;
}

/* Validate shape array. */
int
validate_shape(const ssize_t *shape, const ssize_t ndims)
{
    if ((shape == NULL) || (ndims < 1))
        return ICS_VALUE_ERROR;
    for (ssize_t i = 1; i < ndims; i++) {
        if (shape[i] <= 0)
            return ICS_VALUE_ERROR;
    }
    return ICS_OK;
}

/* Validate bins array. */
int
validate_bins(const ssize_t *bins, const ssize_t nbins, const ssize_t size)
{
    if ((bins == NULL) || (nbins < 1) || (nbins > size) || (bins[0] < 1) ||
        (bins[nbins - 1] > size))
        return ICS_VALUE_ERROR;
    for (ssize_t i = 1; i < nbins; i++) {
        if (bins[i] <= bins[i - 1])
            return ICS_VALUE_ERROR;
    }
    return ICS_OK;
}

/* Calculate times from bins. */
template <typename T>
void
bins2times(ssize_t *bins, const ssize_t nbins, T frametime, T *times)
{
    if ((times == NULL) || (bins == NULL) || (nbins < 2)) {
        throw ICS_VALUE_ERROR;
    }
    times[0] = ((T)(bins[0] - 1) / (T)2.0) * frametime;
    for (ssize_t i = 1; i < nbins; i++) {
        times[i] = ((T)bins[i - 1] + (T)(bins[i] - 1 - bins[i - 1]) / (T)2.0) *
                   frametime;
    }
}

int
bins2times_f(ssize_t *bins, const ssize_t nbins, float frametime, float *times)
{
    try {
        bins2times(bins, nbins, frametime, times);
    }
    catch (const int e) {
        return e;
    }
    catch (...) {
        return ICS_ERROR;
    }
    return ICS_OK;
}

int
bins2times_d(
    ssize_t *bins, const ssize_t nbins, double frametime, double *times)
{
    try {
        bins2times(bins, nbins, frametime, times);
    }
    catch (const int e) {
        return e;
    }
    catch (...) {
        return ICS_ERROR;
    }
    return ICS_OK;
}

/* Return minimum and maximum in each dimension of flattened array of points.
 */
template <typename T>
void
minmax(const ssize_t npoints, const ssize_t ndim, const T *points, T *out)
{
    for (ssize_t i = 0; i < ndim; i++) {
        out[2 * i] = T.max();
        out[2 * i + 1] = T.min();
    }
    for (ssize_t j = 0; j < npoints; j++) {
        for (ssize_t i = 0; i < ndim; i++) {
            const T v = points[ndim * j + i];
            if (v < out[2 * i])
                out[2 * i] = v;
            if (v > out[2 * i + 1])
                out[2 * i + 1] = v;
        }
    }
}

void
minmax(
    const ssize_t npoints,
    const ssize_t ndim,
    const ssize_t *points,
    ssize_t *out)
{
    for (ssize_t i = 0; i < ndim; i++) {
        out[2 * i] = SSIZE_MAX;
        out[2 * i + 1] = SSIZE_MIN;
    }
    for (ssize_t j = 0; j < npoints; j++) {
        for (ssize_t i = 0; i < ndim; i++) {
            const ssize_t v = points[ndim * j + i];
            if (v < out[2 * i])
                out[2 * i] = v;
            if (v > out[2 * i + 1])
                out[2 * i + 1] = v;
        }
    }
}

void
minmax_n(ssize_t npoints, ssize_t ndim, const ssize_t *points, ssize_t *out)
{
    minmax(npoints, ndim, points, out);
}

/* Return n-th triangular number. */
ssize_t
triangular_number(const ssize_t n, int diag)
{
    ssize_t tn = 0;
    if (diag != 0) {
        for (ssize_t i = 1; i <= n; i++) tn += i;
    }
    else {
        for (ssize_t i = 0; i < n; i++) {
            for (ssize_t j = i + 1; j < n; j++) {
                tn++;
            }
        }
    }
    return tn;
}

/* Return coordinates of n-th triangular number on rectangular grid. */
void
triangular_number_coordinates(
    const ssize_t n, const ssize_t tn, ssize_t *x, ssize_t *y, int diag)
{
    const ssize_t c = (diag != 0) ? 0 : 1;
    ssize_t k = 0;
    for (ssize_t i = 0; i < n; i++) {
        for (ssize_t j = i + c; j < n; j++) {
            if (k == tn) {
                *x = i;
                *y = j;
                return;
            }
            k++;
        }
    }
}

/* Average nd array along axis. Up to 4D. */
template <typename T>
void
average(
    T *data,
    const ssize_t *shape_,
    const ssize_t *strides_,
    T *out,
    const ssize_t *outstrides,
    const int ndim,
    const int axis,
    const int nthreads)
{
    throw ICS_NOTIMPLEMENTD_ERROR;

    if ((data == NULL) || (out == NULL) || (shape_ == NULL) ||
        (strides_ == NULL) || (outstrides == NULL)) {
        throw ICS_VALUE_ERROR1;
    }
    if ((ndim < 1) || (ndim > 4) || (ndim <= axis)) {
        throw ICS_VALUE_ERROR2;
    }
    ssize_t shape[4] = {1, 1, 1, 1};
    ssize_t strides[4] = {0, 0, 0, 0};
    ssize_t strides2[4] = {0, 0, 0, 0};

    for (int i = 0; i < ndim; i++) {
        shape[i] = shape_[i];
        strides[i] = strides_[i];
    }

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
        }
    }
    mkl_free(sumyx_);
}