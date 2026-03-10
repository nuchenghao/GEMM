#include "utils.h"

static double gtod_ref_time_sec = 0.0;
double dClock() {
    double the_time, norm_sec;
    struct timeval tv;

    gettimeofday(&tv, NULL);
    if (gtod_ref_time_sec == 0.0)
        gtod_ref_time_sec = (double)tv.tv_sec;

    norm_sec = (double)tv.tv_sec - gtod_ref_time_sec;
    the_time = norm_sec + tv.tv_usec * 1.0e-6;

    return the_time;
}

void rand_fill_matrix_fp32(float *matrix_A, int M, int K) {
    for (int i = 0; i < M * K; i++) {
        matrix_A[i] = (float)drand48() * 0.005;
    }
}
