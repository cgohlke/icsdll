/* ics.hpp

Common inlined functions for the ICS library.

Copyright (c) 2016-2025, Christoph Gohlke
This source code is distributed under the BSD 3-Clause license.

Refer to the header file 'ics.h' for documentation and license.

*/

// number of bytes to align mkl_malloc
#define MKL_ALIGN 64

// Number of doubles to append to last dimension of FFT buffer
// must be >=2 <= MKL_ALIGN/8 and divisible by 2
#define MKL_ALIGN_D 8

/* definitions implemented in ics.cpp */
int
validate_shape(const ssize_t *shape, const ssize_t ndims);
int
validate_bins(const ssize_t *bins, const ssize_t nbins, const ssize_t size);
void
minmax(
    const ssize_t npoints,
    const ssize_t ndim,
    const ssize_t *points,
    ssize_t *out);
ssize_t
triangular_number(const ssize_t n, int diag);
void
triangular_number_coordinates(
    const ssize_t n, const ssize_t tn, ssize_t *x, ssize_t *y, int diag);

/* Return maximum of two numbers */
template <typename T>
inline const T &
max(const T &a, const T &b)
{
    return (a < b) ? b : a;
}

/* Return if number is a power of 2 */
inline int
ispow2(const int n)
{
    return (n > 1) & ((n & (n - 1)) == 0);
}

inline int
ispow2(const ssize_t n)
{
    return (n > 1) & ((n & (n - 1)) == 0);
}

/* Return next highest power of two of 64-bit value */
/* https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2 */
inline int64_t
nextpow2(int64_t n)
{
    if (n <= 0) {
        return 0;
    }
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n++;
    return n;
}

inline int32_t
nextpow2(int32_t n)
{
    if (n <= 0) {
        return 0;
    }
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

/* inline reverse vector */
template <typename T>
inline void
reverse(T *a, const ssize_t size)
{
    for (ssize_t i = 0, j = size - 1; i < size / 2; i++, j--) {
        const T t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
}

/* inline scale vector */
template <typename T>
inline void
normalize(T *a, const ssize_t size, const T scale)
{
    // #pragma omp simd
    for (ssize_t i = 0; i < size; i++) {
        a[i] *= scale;
    }
}

template <typename T>
inline void
normalize(T *a, const ssize_t size, const T scale, const T offset)
{
    // #pragma omp simd
    for (ssize_t i = 0; i < size; i++) {
        a[i] = a[i] * scale + offset;
    }
}

/* copy vector `b` to `a` */
template <typename Ti, typename To>
inline void
copy(To *r, Ti *b, const ssize_t size)
{
    // #pragma omp simd
    for (ssize_t i = 0; i < size; i++) {
        r[i] = (To)b[i];
    }
}

/* copy vector `b` to strided `a` */
template <typename Ti, typename To>
inline void
copy(To *a, Ti *b, const ssize_t size, const ssize_t stride)
{
    char *p = (char *)a;
    for (ssize_t i = 0; i < size; i++) {
        *((To *)p) = (To)b[i];
        p += stride;
    }
}

/* copy reverse of vector `b` to `r` */
template <typename Ti, typename To>
inline void
copy_r(To *r, Ti *b, const ssize_t size)
{
    // #pragma omp simd
    for (ssize_t i = 0, j = size - 1; i < size; i++, j--) {
        r[i] = (To)b[j];
    }
}

/* copy scaled vector `b` to `r` */
template <typename Ti, typename To>
inline void
copy(To *ar, Ti *b, const ssize_t size, const Ti scale)
{
    // #pragma omp simd
    for (ssize_t i = 0; i < size; i++) {
        r[i] = (To)(b[i] * scale);
    }
}

/* copy scaled and shifted vector `b` to `r` */
template <typename Ti, typename To>
inline void
copy(To *r, Ti *b, const ssize_t size, const Ti scale, const Ti offset)
{
    // #pragma omp simd
    for (ssize_t i = 0; i < size; i++) {
        r[i] = (To)(b[i] * scale + offset);
    }
}

/* 1D multiply `a` by its complex conjugate inplace */
inline void
complex_multiply(double *a, const ssize_t size)
{
    // #pragma omp simd
    for (ssize_t i = 0; i < size; i += 2) {
        const double re = a[i];
        const double im = a[i + 1];
        a[i] = re * re + im * im;
        a[i + 1] = 0.0;
    }
}

/* 1D multiply `a` by `b`'s complex conjugate and store in `b` */
inline void
complex_multiply(double *a, double *b, const ssize_t size)
{
    // #pragma omp simd
    for (ssize_t i = 0; i < size; i += 2) {
        const double ar = a[i];
        const double ai = a[i + 1];
        const double br = b[i];
        const double bi = b[i + 1];
        b[i] = ar * br + ai * bi;
        b[i + 1] = ai * br - ar * bi;
    }
}

/* 1D multiply `a` by `b`'s complex conjugate and store in `r` */
inline void
complex_multiply_(
    double *r, const double *a, const double *b, const ssize_t size)
{
    // #pragma omp simd
    for (ssize_t i = 0; i < size; i += 2) {
        const double br = b[i];
        const double bi = b[i + 1];
        const double ar = a[i];
        const double ai = a[i + 1];
        r[i] = ar * br + ai * bi;
        r[i + 1] = ai * br - ar * bi;
    }
}

inline void
complex_multiply(
    double *r, const double *a, const double *b, const ssize_t size)
{
    // #pragma omp simd
    for (ssize_t i = 0; i < size / 2; i++) {
        const double br = *b++;
        const double bi = *b++;
        const double ar = *a++;
        const double ai = *a++;
        *r++ = ar * br + ai * bi;
        *r++ = ai * br - ar * bi;
    }
}

/* 1D multiply `a` by `b`'s complex conjugate and store in `r`.
 * DFTI_PERM_FORMAT only. */
inline void
complex_multiply_perm(
    double *r, const double *a, const double *b, const ssize_t size)
{
    r[0] = a[0] * b[0];
    r[1] = a[1] * b[1];
    for (ssize_t i = 2; i < size; i += 2) {
        const double br = b[i];
        const double bi = b[i + 1];
        const double ar = a[i];
        const double ai = a[i + 1];
        r[i] = ar * br + ai * bi;
        r[i + 1] = ai * br - ar * bi;
    }
}

/* Inline average data into bins */
template <typename T>
inline void
average(T *data, const ssize_t *bins, const ssize_t nbins)
{
    double s = 0.0;
    for (ssize_t i = 0; i < bins[0]; i++) {
        s += data[i];
    }
    data[0] = (T)((s / (double)bins[0]));
    for (ssize_t j = 1; j < nbins; j++) {
        s = 0.0;
        for (ssize_t i = bins[j - 1]; i < bins[j]; i++) {
            s += data[i];
        }
        data[j] = (T)((s / (double)(bins[j] - bins[j - 1])));
    }
}

/* Average data into bins */
template <typename Ti, typename To>
inline void
average(To *out, const Ti *data, const ssize_t *bins, const ssize_t nbins)
{
    double s = 0.0;
    for (ssize_t i = 0; i < bins[0]; i++) {
        s += data[i];
    }
    out[0] = (To)((s / (double)bins[0]));
    for (ssize_t j = 1; j < nbins; j++) {
        s = 0.0;
        for (ssize_t i = bins[j - 1]; i < bins[j]; i++) {
            s += data[i];
        }
        out[j] = (To)((s / (double)(bins[j] - bins[j - 1])));
    }
}

/* Inplace simple exponential smoothing of vector */
template <typename T>
inline void
smooth(T *data, const ssize_t size, const T filter)
{
    if ((filter > 0.0) && (filter < 1.0)) {
        const T filter1 = (T)1.0 - filter;
        for (ssize_t i = 1; i < size; i++) {
            data[i] = data[i] * filter1 + data[i - 1] * filter;
        }
        for (ssize_t i = size - 2; i >= 0; i--) {
            data[i] = data[i] * filter1 + data[i + 1] * filter;
        }
    }
}

/* average, normalize, and smooth correlation function */
template <typename Ti, typename To>
inline void
anscf(
    Ti *a,
    To *out,
    const ssize_t stride,
    const ssize_t *bins,
    const ssize_t nbins,
    const double scale,
    const double offset,
    const double filter,
    const bool skipfirst,
    ssize_t size = 0)
{
    if (size > 0) {
        // use second half of correlation function
        for (ssize_t i = 1, j = size - 1; i < size / 2; i++, j--) {
            a[i] = a[j];
        }
    }

    average(a, bins, nbins);
    normalize(a, nbins, scale, offset);
    if (skipfirst)
        a[0] = a[1];
    smooth(a, nbins, filter);
    copy(out, a, nbins, stride);
}

/* Return if region is selected */
/* A mask value != 0 means a pixel is selected, i.e. not masked */
template <typename Tm>
inline bool
selected(
    const Tm *mask,
    const ssize_t *strides,
    const ssize_t y,
    const ssize_t x,
    const ssize_t length,
    const ssize_t width,
    const int32_t mode)
{
    if ((mask == NULL) || (strides == NULL)) {
        return true;
    }

    if (mode & ICS_MASK_FIRST) {
        const Tm *m = (Tm *)((char *)mask + y * strides[0] + x * strides[1]);
        return m[0] != (Tm)0;
    }

    if (mode & ICS_MASK_CENTER) {
        const Tm *m = (Tm *)(
            (char *)mask
            + (y + length / 2) * strides[0]
            + (x + width / 2) * strides[1]);
        return m[0] != (Tm)0;
    }

    if (mode & ICS_MASK_ALL) {
        for (ssize_t i = y; i < y + length; i++) {
            for (ssize_t j = x; j < x + width; j++) {
                const Tm *m =
                    (Tm *)((char *)mask + i * strides[0] + j * strides[1]);
                if (*m == (Tm)0) {
                    return false;
                }
            }
        }
        return true;
    }

    // ICS_MASK_ANY
    for (ssize_t i = y; i < y + length; i++) {
        for (ssize_t j = x; j < x + width; j++) {
            const Tm *m =
                (Tm *)((char *)mask + i * strides[0] + j * strides[1]);
            if (*m != (Tm)0) {
                return true;
            }
        }
    }
    return false;
}

/* In-place add integer with clamp to integer range */
template <typename Td, typename Ti>
inline void
add_int(Td *data, const Ti value)
{
    const int64_t d = (int64_t)(*data) + (int64_t)value;
    if (d <= static_cast<int64_t>(std::numeric_limits<Td>::min())) {
        *data = std::numeric_limits<Td>::min();
    }
    else if (d >= static_cast<int64_t>(std::numeric_limits<Td>::max())) {
        *data = std::numeric_limits<Td>::max();
    }
    else {
        *data = static_cast<Td>(d);
    }
}

template <typename Tv>
inline void
add_int(double *data, const Tv value)
{
    *data += (double)value;
}

template <typename Tv>
inline void
add_int(float *data, const Tv value)
{
    *data += (float)value;
}

/* In-place add float with clamp to integer range */
template <typename Td, typename Tf>
inline void
add_float(Td *data, const Tf value)
{
    const Tf d = (Tf)(*data) + value;
    if (d <= static_cast<Tf>(std::numeric_limits<Td>::min()) + (Tf)0.5) {
        *data = std::numeric_limits<Td>::min();
    }
    else if (d >= static_cast<Tf>(std::numeric_limits<Td>::max()) - (Tf)0.5) {
        *data = std::numeric_limits<Td>::max();
    }
    else if (d > (Tf)0.0) {
        *data = (Td)(d + 0.5);
    }
    else {
        *data = (Td)(d - 0.5);  // not needed for unsigned
    }
}

template <typename Tf>
inline void
add_float(float *data, const Tf value)
{
    *data += (float)value;
}

template <typename Tf>
inline void
add_float(double *data, const Tf value)
{
    *data += (double)value;
}
