/* test_lstics.cpp */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ics.h"

int
main()
{
    // shape of input data in y, x, t dimensions
    ssize_t shape[3] = {64, 64, 32000};
    // block size and advancements in y and x dimension
    // lines must fit inside block
    ssize_t block[4] = {16, 16, 2, 2};
    // number of lines, line length (must be power of two), 2
    ssize_t linesshape[3] = {16, 8, 2};
    ssize_t nbins = 16;
    double filter = 0.0;  // no smoothing
    int nthreads = 0;

    printf(
        "Data shape=(%zd, %zd, %zd) dtype=uint16\n",
        shape[0],
        shape[1],
        shape[2]);
    size_t data_size = shape[0] * shape[1] * shape[2] * sizeof(uint16_t);
    ssize_t strides[3] = {
        /* y */ (ssize_t)sizeof(uint16_t) * shape[1],
        /* x */ (ssize_t)sizeof(uint16_t),
        /* t */ (ssize_t)sizeof(uint16_t) * shape[1] * shape[0],
    };

    // allocate buffer to contain the whole file
    uint16_t *data = (uint16_t *)malloc(data_size);
    if (data == NULL) {
        fputs("Memory error: data", stderr);
        exit(1);
    }

    // open input file
    FILE *pfile;
    errno_t err = fopen_s(&pfile, "Simulation_Channel.bin", "rb");
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

    // divide by 4
    // for (ssize_t i = 0; i < data_size/2; i++) data[i] /= 4;

    int status;
    clock_t t;
    double duration;

    // subtract immobile fraction inplace
    if (1) {
        t = clock();
        status = yxt_subtract_immobile_H(data, shape, strides, nthreads);
        if (status != ICS_OK) {
            fputs("yxt_subtract_immobile_H failed", stderr);
            exit(7);
        }
        duration = ((double)(clock() - t)) / CLOCKS_PER_SEC;
        printf("Subtract immobile fraction: %f s\n", duration);
    }

    t = clock();

    // line coordinates
    ssize_t *lines = (ssize_t *)malloc(
        linesshape[0] * linesshape[1] * linesshape[2] * sizeof(ssize_t));
    if (lines == NULL) {
        fputs("Memory error: lines", stderr);
        exit(7);
    }
    // double offset[2] = { -1.0, -1.0 };
    ssize_t npoints =
        radial(lines, linesshape[0], linesshape[1], NULL, ICS_RADIUS);
    /*
    printf("Lines");
    for (int i = 0; i < npoints *2; i+=2) {
        printf(" %d,%d", (int)lines[i], (int)lines[i+1]);
    }
    printf("\n");
    */

    // open yxt handle
    yxt_handle handle = yxt_new(shape);
    if (handle == NULL) {
        fputs("Memory error: yxt_handle", stderr);
        exit(4);
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
    // yxt_lstics requires dimension order length, width, number of lines, line
    // length, nbins
    ssize_t outshape[5] = {
        (shape[0] - block[0]) / block[2] + 1,
        (shape[1] - block[1]) / block[3] + 1,
        linesshape[0],
        linesshape[1],
        nbins,
    };
    printf(
        "Output shape=(%zd, %zd, %zd, %zd, %zd) dtype=float32\n",
        outshape[0],
        outshape[1],
        outshape[2],
        outshape[3],
        outshape[4]);
    ssize_t out_size =
        outshape[0] * outshape[1] * outshape[2] * outshape[3] * outshape[4];
    float *out = (float *)malloc(out_size * sizeof(float));
    if (out == NULL) {
        fputs("Memory error: out", stderr);
        exit(5);
    }
    ssize_t outstrides[5] = {
        (ssize_t)sizeof(float) * outshape[4] * outshape[3] * outshape[2] *
            outshape[1],
        (ssize_t)sizeof(float) * outshape[4] * outshape[3] * outshape[2],
        (ssize_t)sizeof(float) * outshape[4] * outshape[3],
        (ssize_t)sizeof(float) * outshape[4],
        (ssize_t)sizeof(float),
    };
    // initialize output to something non-zero
    for (ssize_t i = 0; i < out_size; i++) {
        out[i] = 1.11f;
    }

    // create a mask
    int32_t *mask = (int32_t *)malloc(shape[0] * shape[1] * sizeof(float));
    if (mask == NULL) {
        fputs("Memory error: mask", stderr);
        exit(6);
    }
    ssize_t maskstrides[] = {
        (ssize_t)sizeof(int32_t) * shape[1], (ssize_t)sizeof(int32_t)};
    for (ssize_t i = 0; i < shape[0]; i++) {
        for (ssize_t j = 0; j < shape[1]; j++) {
            mask[i * shape[1] + j] = 1;  // ((i < 32) && (j < 32)) ? 1 : 0;
        }
    }
    // mask[shape[0] * shape[1] - 1] = 1;

    // auto correlate lines
    status = yxt_lstics_Hf(
        handle,
        data,
        strides,
        NULL,  // no second channel
        NULL,
        NULL,  // mask,
        NULL,  // maskstrides,
        ICS_MASK_ANY | ICS_MASK_CLEAR,
        out,
        outstrides,
        lines,
        linesshape,
        block,
        bins,
        nbins,
        filter,
        nthreads);

    if (status != ICS_OK) {
        fputs("yxt_lstics_Hf failed", stderr);
        exit(8);
    }
    duration = ((double)(clock() - t)) / CLOCKS_PER_SEC;
    printf("Duration: %f s\n", duration);

    // open output file
    err = fopen_s(&pfile, "Simulation_Channel.lstics.bin", "wb");
    if (err != 0) {
        fputs("File open error", stderr);
        exit(9);
    }
    // write out array to file
    out_size *= sizeof(float);
    result = fwrite(out, 1, out_size, pfile);
    if (result != out_size) {
        fputs("File write error", stderr);
        exit(10);
    }
    // close file
    fclose(pfile);

    // close handle and free memory
    yxt_del(handle);
    free(data);
    free(out);
    free(mask);

    return 0;
}
