/* test_apcf.cpp */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ics.h"

int
main()
{
    // shape of input data in Airy detectors and t dimensions
    ssize_t shape[3] = {1, 32, 10000000};
    ssize_t nbins = 256;
    double threshold = 0.0;
    double filter = 0.7;
    int nthreads = 0;
    int autocorr = ICS_TRUE;

    int status;
    clock_t t;
    double duration;

    printf(
        "Data shape=(%zd, %zd, %zd) dtype=uint16\n",
        shape[0],
        shape[1],
        shape[2]);
    size_t data_size = shape[0] * shape[1] * shape[2] * sizeof(uint16_t);
    ssize_t strides[3] = {
        /* y */ (ssize_t)sizeof(uint16_t) * shape[2] * shape[1],
        /* x */ (ssize_t)sizeof(uint16_t) * shape[2],
        /* t */ (ssize_t)sizeof(uint16_t),
    };

    // allocate buffer to contain the whole file
    uint16_t *data = (uint16_t *)malloc(data_size);
    if (data == NULL) {
        fputs("Memory error: data", stderr);
        exit(1);
    }

    // open input file
    FILE *pfile;
    errno_t err = fopen_s(&pfile, "Airy_Detectors.bin", "rb");
    if (err != 0) {
        fputs("File open error", stderr);
        exit(2);
    }
    // read file into data array
    size_t result = fread(data, 1, data_size, pfile);
    if (result != data_size) {
        fputs("File read error", stderr);
        exit(3);
    }
    // close file
    fclose(pfile);

    if (1) {
        // correct for bleaching
        ssize_t mean_size = shape[0] * shape[1] * sizeof(double);
        double *mean = (double *)malloc(mean_size);
        if (mean == NULL) {
            fputs("Memory error: mean", stderr);
            exit(1);
        }
        ssize_t meanstrides[2] = {
            (ssize_t)sizeof(double) * shape[1], (ssize_t)sizeof(double)};

        t = clock();
        status = yxt_correct_bleaching_H(
            data, shape, strides, mean, meanstrides, 0.99, nthreads);

        if (status != ICS_OK) {
            fputs("yxt_correct_bleaching_H failed", stderr);
            exit(9);
        }
        t = clock() - t;
        duration = ((double)t) / CLOCKS_PER_SEC;
        printf("Correct bleaching: %f s\n", duration);

        // save filtered
        err = fopen_s(&pfile, "Airy_Detectors.filtered.bin", "wb");
        if (err != 0) {
            fputs("File open error", stderr);
            exit(10);
        }
        result = fwrite(data, 1, data_size, pfile);
        if (result != data_size) {
            fputs("File write error", stderr);
            exit(11);
        }
        fclose(pfile);

        // save mean
        err = fopen_s(&pfile, "Airy_Detectors.mean.bin", "wb");
        if (err != 0) {
            fputs("File open error", stderr);
            exit(10);
        }
        result = fwrite(mean, 1, mean_size, pfile);
        if (result != mean_size) {
            fputs("File write error", stderr);
            exit(11);
        }
        fclose(pfile);

        free(mean);
    }

    // open yxt handle
    yxt_handle handle = yxt_new(shape);
    if (handle == NULL) {
        fputs("Memory error: yxt_handle", stderr);
        exit(5);
    }

    // get handle shape; tsize is cropped to power of two
    yxt_get_buffer(handle, shape, NULL);
    printf(
        "Buffer shape=(%zd, %zd, %zd) dtype=float64\n",
        shape[0],
        shape[1],
        shape[2]);

    // log-bins array
    ssize_t *bins = (ssize_t *)malloc(nbins * sizeof(ssize_t));
    if (bins == NULL) {
        fputs("Memory error: bins", stderr);
        exit(6);
    }
    nbins = logbins(shape[2] / 2, nbins, bins);
    printf("Bins");
    for (int i = 0; i < nbins; i++) {
        printf(" %d", (int)bins[i]);
    }
    printf("\n");

    // output array
    ssize_t shapeout[3] = {
        shape[1], shape[1] - ((autocorr == ICS_FALSE) ? 1 : 0), nbins};
    printf(
        "Output shape=(%zd, %zd, %zd) dtype=float32\n",
        shapeout[0],
        shapeout[1],
        shapeout[2]);
    ssize_t out_size = shapeout[0] * shapeout[1] * shapeout[2];
    float *out = (float *)malloc(out_size * sizeof(float));
    if (out == NULL) {
        fputs("Memory error: out", stderr);
        exit(8);
    }
    ssize_t stridesout[3] = {
        (ssize_t)sizeof(float) * shapeout[2] * shapeout[1],
        (ssize_t)sizeof(float) * shapeout[2],
        (ssize_t)sizeof(float),
    };
    // initialize output
    for (ssize_t i = 0; i < out_size; i++) {
        out[i] = 0.0f;
    }

    // start timer
    t = clock();

    // cross correlate
    status = yxt_apcf_Hf(
        handle,
        data,
        &strides[1],
        out,
        stridesout,
        bins,
        nbins,
        autocorr,
        filter,
        nthreads);

    if (status != ICS_OK) {
        fputs("yxt_apcf_Hf failed", stderr);
        exit(9);
    }

    // print elapsed time
    t = clock() - t;
    duration = ((double)t) / CLOCKS_PER_SEC;
    printf("Duration: %f s\n", duration);

    // open output file
    err = fopen_s(&pfile, "Airy_Detectors.apcf.bin", "wb");
    if (err != 0) {
        fputs("File open error", stderr);
        exit(10);
    }
    // write out array to file
    out_size *= sizeof(float);
    result = fwrite(out, 1, out_size, pfile);
    if (result != out_size) {
        fputs("File write error", stderr);
        exit(11);
    }
    // close file
    fclose(pfile);

    // close handle and free memory
    yxt_del(handle);
    free(data);
    free(bins);
    free(out);

    return 0;
}
