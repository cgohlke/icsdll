/* test_ipcf.cpp */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ics.h"

int
main()
{
    // shape of input data in y, x, t dimensions
    ssize_t shape[3] = {64, 64, 32000};
    ssize_t radius = 6;
    ssize_t nbins = 32;
    double threshold = 0.0;
    double filter = 0.7;
    int nthreads = 0;

    int ret;
    clock_t t;
    double duration;

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

    // read simulation data from file
    FILE *pfile;
    errno_t err = fopen_s(&pfile, "Simulation_Channel.bin", "rb");
    if (err != 0) {
        fputs("File open error", stderr);
        exit(2);
    }
    size_t result = fread(data, 1, data_size, pfile);
    if (result != data_size) {
        fputs("File read error", stderr);
        exit(3);
    }
    fclose(pfile);

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
        exit(5);
    }
    nbins = logbins(shape[2] / 2, nbins, bins);
    printf("Bins");
    for (int i = 0; i < nbins; i++) {
        printf(" %d", (int)bins[i]);
    }
    printf("\n");

    // circle coordinates
    ssize_t npoints = circle(radius, NULL, 0);
    ssize_t *points = (ssize_t *)malloc(2 * npoints * sizeof(ssize_t));
    if (points == NULL) {
        fputs("Memory error: points", stderr);
        exit(6);
    }
    npoints = circle(radius, points, 0);
    printf("Points");
    for (int i = 0; i < npoints * 2; i += 2) {
        printf(" %d,%d", (int)points[i], (int)points[i + 1]);
    }
    printf("\n");

    // output array. skip borders.
    ssize_t shapeout[4] = {
        shape[0] - 2 * radius, shape[1] - 2 * radius, npoints, nbins};
    printf(
        "Output shape=(%zd, %zd, %zd, %zd) dtype=float32\n",
        shapeout[0],
        shapeout[1],
        shapeout[2],
        shapeout[3]);
    ssize_t out_size = shapeout[0] * shapeout[1] * shapeout[2] * shapeout[3];
    float *out = (float *)malloc(out_size * sizeof(float));
    if (out == NULL) {
        fputs("Memory error: out", stderr);
        exit(7);
    }
    ssize_t stridesout[4] = {
        (ssize_t)sizeof(float) * shapeout[3] * shapeout[2] * shapeout[1],
        (ssize_t)sizeof(float) * shapeout[3] * shapeout[2],
        (ssize_t)sizeof(float) * shapeout[3],
        (ssize_t)sizeof(float),
    };
    // initialize output to something non-zero
    for (ssize_t i = 0; i < out_size; i++) {
        out[i] = 1.11f;
    }

    // start timer
    t = clock();

    // cross correlate
    ret = yxt_ipcf_Hf(
        handle,
        data,
        NULL,
        strides,
        out,
        stridesout,
        points,
        npoints,
        bins,
        nbins,
        threshold,
        filter,
        nthreads);
    if (ret != ICS_OK) {
        fputs("yxt_ipcf_Hf failed", stderr);
        exit(8);
    }

    // print elapsed time
    t = clock() - t;
    duration = ((double)t) / CLOCKS_PER_SEC;
    printf("yxt_ipcf_Hf: %f s\n", duration);

    // write out array to file
    err = fopen_s(&pfile, "Simulation_Channel.ipcf.bin", "wb");
    if (err != 0) {
        fputs("File open error", stderr);
        exit(9);
    }
    result = fwrite(out, 1, out_size * sizeof(float), pfile);
    if (result != out_size * sizeof(float)) {
        fputs("File write error", stderr);
        exit(10);
    }
    fclose(pfile);

    /*
      Fit ipcf output
    */
    const float frametime = 0.01f;  // 100 fps
    const float pixelsize = 0.05f;  // um
    const float args[3] = {
        0.1f,        // initial guess of dc variable
        1.0f,        // initial guess of D variable
        0.3f * 0.3f  // w2 squared constant
    };
    const float bounds[4] = {
        1e-6f,  // lower bound of dc
        1e3f,   // upper bound of dc
        1e-6f,  // lower bound of D
        1e3f    // upper bound of D
    };
    // const float *bounds = NULL;
    const float settings[10] = {
        400.0f,  // maximum number of iterations
        200.0f,  // maximum number of iterations of trial step
        1e-10f,  // eps[6] : various stopping criteria
        1e-10f,
        1e-10f,
        1e-10f,
        1e-10f,
        1e-10f,
        1e-10f,  // eps_jac : precision of the Jacobian matrix calculation
        0.0f     // rs : initial size of TR
    };
    // const float *settings = NULL;

    // ix: buffer for fitted parameters; 2 per pcf; single precision
    ssize_t shapex[4] = {shapeout[0], shapeout[1], shapeout[2], 2};
    ssize_t stridesx[4] = {
        (ssize_t)sizeof(float) * shapex[3] * shapex[2] * shapex[1],
        (ssize_t)sizeof(float) * shapex[3] * shapex[2],
        (ssize_t)sizeof(float) * shapex[3],
        (ssize_t)sizeof(float)};
    const ssize_t x_size = shapex[0] * shapex[1] * shapex[2] * shapex[3];
    float *ix = (float *)malloc(x_size * sizeof(float));
    if (ix == NULL) {
        fputs("Memory error: ix", stderr);
        exit(11);
    }
    for (ssize_t i = 0; i < x_size; i++) {
        ix[i] = 1.11f;
    }

    // ifx: fitted data; same shape as ipcf output; single precision
    float *ifx = (float *)malloc(out_size * sizeof(float));
    if (ifx == NULL) {
        fputs("Memory error: ifx", stderr);
        exit(12);
    }
    ssize_t stridesfx[4] = {
        (ssize_t)sizeof(float) * shapeout[3] * shapeout[2] * shapeout[1],
        (ssize_t)sizeof(float) * shapeout[3] * shapeout[2],
        (ssize_t)sizeof(float) * shapeout[3],
        (ssize_t)sizeof(float),
    };
    for (ssize_t i = 0; i < out_size; i++) {
        ifx[i] = 1.11f;
    }

    // solver status
    ssize_t shapestatus[4] = {shapeout[0], shapeout[1], shapeout[2], 4};
    ssize_t stridesstatus[4] = {
        (ssize_t)sizeof(float) * shapestatus[3] * shapestatus[2] *
            shapestatus[1],
        (ssize_t)sizeof(float) * shapestatus[3] * shapestatus[2],
        (ssize_t)sizeof(float) * shapestatus[3],
        (ssize_t)sizeof(float)};
    const ssize_t status_size =
        shapestatus[0] * shapestatus[1] * shapestatus[2] * shapestatus[3];
    float *status = (float *)malloc(status_size * sizeof(float));
    if (status == NULL) {
        fputs("Memory error: status", stderr);
        exit(13);
    }
    for (ssize_t i = 0; i < status_size; i++) {
        status[i] = 1.11f;
    }

    // time axis
    float *times = (float *)malloc(nbins * sizeof(float));
    if (times == NULL) {
        fputs("Memory error: times", stderr);
        exit(14);
    }
    bins2times_f(bins, nbins, frametime, times);
    printf("Times");
    for (int i = 0; i < nbins; i++) {
        printf(" %.3f", times[i]);
    }
    printf("\n");

    // squared distances
    float distance;  // average distance
    float *distances = (float *)malloc(npoints * sizeof(float));
    if (distances == NULL) {
        fputs("Memory error: distances", stderr);
        exit(15);
    }
    distance = points2distances_f(points, npoints, pixelsize, distances);
    printf("Distances");
    for (int i = 0; i < npoints; i++) {
        printf(" %.2f", distances[i]);
        distances[i] *= distances[i];  // square distances
    }
    printf("\nAverage distance: %.3f\n", distance);
    distance *= distance;  // square distance

    t = clock();

    // fit data
    ret = ipcf_nlsp_1dpcf_f(
        out,
        shapeout,
        stridesout,
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
        0,  // not average
        nthreads);
    if (ret != ICS_OK) {
        fputs("ipcf_nlsp_1dpcf_f failed", stderr);
        exit(16);
    }

    t = clock() - t;
    duration = ((double)t) / CLOCKS_PER_SEC;
    printf("ipcf_nlsp_1dpcf_f: %f s\n", duration);

    // write fitted data to file
    err = fopen_s(&pfile, "Simulation_Channel.ipcf_fx.bin", "wb");
    if (err != 0) {
        fputs("File open error", stderr);
        exit(17);
    }
    result = fwrite(ifx, 1, out_size * sizeof(float), pfile);
    if (result != out_size * sizeof(float)) {
        fputs("File write error", stderr);
        exit(18);
    }
    fclose(pfile);

    // write fitted parameters to file
    err = fopen_s(&pfile, "Simulation_Channel.ipcf_x.bin", "wb");
    if (err != 0) {
        fputs("File open error", stderr);
        exit(19);
    }
    result = fwrite(ix, 1, x_size * sizeof(float), pfile);
    if (result != x_size * sizeof(float)) {
        fputs("File write error", stderr);
        exit(20);
    }
    fclose(pfile);

    // write status values to file
    err = fopen_s(&pfile, "Simulation_Channel.ipcf_status.bin", "wb");
    if (err != 0) {
        fputs("File open error", stderr);
        exit(21);
    }
    result = fwrite(status, 1, status_size * sizeof(float), pfile);
    if (result != status_size * sizeof(float)) {
        fputs("File write error", stderr);
        exit(22);
    }
    fclose(pfile);

    /*
       Fit ipcf with averaging.
    */
    // ax: buffer for fitted parameters; 2 per averaged pcf; single precision
    ssize_t shapeax[4] = {shapeout[0], shapeout[1], 1, 2};
    ssize_t stridesax[4] = {
        (ssize_t)sizeof(float) * shapeax[3] * shapeax[2] * shapeax[1],
        (ssize_t)sizeof(float) * shapeax[3] * shapeax[2],
        0,
        (ssize_t)sizeof(float)};
    const ssize_t ax_size = shapeax[0] * shapeax[1] * shapeax[2] * shapeax[3];
    float *ax = (float *)malloc(ax_size * sizeof(float));
    if (ax == NULL) {
        fputs("Memory error: ax", stderr);
        exit(23);
    }
    for (ssize_t i = 0; i < ax_size; i++) {
        ax[i] = 1.11f;
    }

    // afx: f(x) for averaged data
    ssize_t shapeafx[4] = {shapeout[0], shapeout[1], 1, shapeout[3]};
    ssize_t stridesafx[4] = {
        (ssize_t)sizeof(float) * shapeafx[3] * shapeafx[2] * shapeafx[1],
        (ssize_t)sizeof(float) * shapeafx[3] * shapeafx[2],
        0,
        (ssize_t)sizeof(float),
    };
    const ssize_t afx_size =
        shapeafx[0] * shapeafx[1] * shapeafx[2] * shapeafx[3];
    float *afx = (float *)malloc(afx_size * sizeof(float));
    if (afx == NULL) {
        fputs("Memory error: afx", stderr);
        exit(24);
    }
    for (ssize_t i = 0; i < afx_size; i++) {
        afx[i] = 1.11f;
    }

    t = clock();

    // fit data with averaging
    ret = ipcf_nlsp_1dpcf_f(
        out,
        shapeout,
        stridesout,
        times,
        distances,
        args,
        bounds,
        ax,
        stridesax,
        afx,
        stridesafx,
        NULL,
        NULL,
        settings,
        1,  // average
        nthreads);
    if (ret != ICS_OK) {
        fputs("ipcf_nlsp_1dpcf_f failed", stderr);
        exit(25);
    }

    t = clock() - t;
    duration = ((double)t) / CLOCKS_PER_SEC;
    printf("ipcf_nlsp_1dpcf_f averaged: %f s\n", duration);

    // write fitted data to file
    err = fopen_s(&pfile, "Simulation_Channel.ipcf_afx.bin", "wb");
    if (err != 0) {
        fputs("File open error", stderr);
        exit(26);
    }
    result = fwrite(afx, 1, afx_size * sizeof(float), pfile);
    if (result != afx_size * sizeof(float)) {
        fputs("File write error", stderr);
        exit(27);
    }
    fclose(pfile);

    // write fitted parameters to file
    err = fopen_s(&pfile, "Simulation_Channel.ipcf_ax.bin", "wb");
    if (err != 0) {
        fputs("File open error", stderr);
        exit(28);
    }
    result = fwrite(ax, 1, ax_size * sizeof(float), pfile);
    if (result != ax_size * sizeof(float)) {
        fputs("File write error", stderr);
        exit(29);
    }
    fclose(pfile);

    // close handle and free memory
    yxt_del(handle);
    free(data);
    free(bins);
    free(points);
    free(out);
    free(times);
    free(distances);
    free(status);
    free(ifx);
    free(ix);
    free(afx);
    free(ax);

    return 0;
}
