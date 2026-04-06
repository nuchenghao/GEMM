#include "smegemm.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
void row_packa_output(int m, int k, float *XA, float *result);
#elif defined(__linux__)
#include "kblas.h"
#endif

void buffer_transpose_submatrixa(int Submatrixa_M, int Submatrixa_K, int matrixa_K, uint32_t *matrixa,
                                 uint32_t *MatrixaTileBuffer);

static void test_buffer_transpose_submatrixa(int matrixa_M, int matrixa_K, float *matrixa, float *result) {
    float *restrict atilde_buffer;
    posix_memalign((void **)&atilde_buffer, SME_CACHELINE_SIZE, SUBMATRIX_M * SUBMATRIX_K * sizeof(float));

    int total = 0;
    for (int mi = 0; mi < matrixa_M; mi += SUBMATRIX_M) {
        int Submatrixa_M = min(SUBMATRIX_M, matrixa_M - mi);

        for (int ki = 0; ki < matrixa_K; ki += SUBMATRIX_K) {

            int Submatrixa_K = min(SUBMATRIX_K, matrixa_K - ki);
            buffer_transpose_submatrixa(Submatrixa_M, Submatrixa_K, matrixa_K, (uint32_t *)&matrixa[mi * matrixa_K + ki],
                                        (uint32_t *)atilde_buffer);

            int count = (((Submatrixa_M + 15) / 16) * 16) * Submatrixa_K;
            memcpy(&result[total], atilde_buffer, count * sizeof(float));
            total += count;
        }
    }

    free(atilde_buffer);
}

int unitest_buffer_transpose_submatrixa() {
    int matrixa_M_array[] = {49, 23, 1024, 328, 1078, 2049, 7893, 12345, 87630, 12342, 20480, 1, 1000000};
    int matrixa_K_array[] = {12340, 1, 26, 382, 12340, 12048, 1024, 10204, 10240, 1024, 1024, 1000000, 1};
    int pass = 1;
    for (int i = 0; i < 13; i++) {
        int matrixa_M = matrixa_M_array[i];
        int matrixa_K = matrixa_K_array[i];

        float *matrixa = malloc(matrixa_M * matrixa_K * sizeof(float));
        rand_fill_matrix_fp32(matrixa, matrixa_M, matrixa_K);
        float *correct = malloc(((matrixa_M + 15) / 16) * 16 * matrixa_K * sizeof(float));
        float *answer = malloc(((matrixa_M + 15) / 16) * 16 * matrixa_K * sizeof(float));
#ifdef __APPLE__
        row_packa_output(matrixa_M, matrixa_K, matrixa, correct);
#endif
        test_buffer_transpose_submatrixa(matrixa_M, matrixa_K, matrixa, answer);

        int size = ((matrixa_M + 15) / 16) * 16 * matrixa_K;
        int all_match = 1;
        for (int j = 0; j < size; j++) {
            if (fabsf(correct[j] - answer[j]) > 1e-8f) {
                all_match = 0;
                printf("Mismatch at index %d: correct=%f, answer=%f\n", j, correct[j], answer[j]);
                break;
            }
        }
        if (all_match) {
            free(matrixa);
            free(correct);
            free(answer);
        } else {
            pass = 0;
            free(matrixa);
            free(correct);
            free(answer);
            printf("fail at test %d\n", i);
            break;
        }
        printf("success: %d\n", i);
    }
    if (pass == 1) {
        printf("All tests passed\n");
    }
    exit(0);
}

static double run_blas_sgemm(int matrixa_M, int N, int matrixa_K, float *A, float *B, float *C) {
    memset(C, 0, matrixa_M * N * sizeof(float));
    double start = dClock();
#if defined(__APPLE__) || defined(__linux__)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, matrixa_M, N, matrixa_K, 1.0f, A, matrixa_K, B, N, 0.0f, C, N);
#endif
    double end = dClock();
    return end - start;
}

static double run_sme_sgemm(int matrixa_M, int N, int matrixa_K, float *A, float *B, float *C) {
    memset(C, 0, matrixa_M * N * sizeof(float));
    double start = dClock();
    sme_fp32_gemm(matrixa_M, N, matrixa_K, A, B, C);
    double end = dClock();
    return end - start;
}

static int verify_results(int size, float *ref, float *test) {
    for (int i = 0; i < size; i++) {
        if (fabsf(ref[i] - test[i]) > 1e-3f) {
            printf("  Mismatch at index %d: blas=%f, sme=%f\n", i, ref[i], test[i]);
            return 0;
        }
    }
    return 1;
}

static void bench_group(const char *label, int n, int matrixa_M[], int N[], int matrixa_K[], double speedups[]) {
    printf("\n=== %s (n=%d) ===\n", label, n);
    for (int i = 0; i < n; i++) {
        int m = matrixa_M[i], nn = N[i], k = matrixa_K[i];
        float *A = malloc(m * k * sizeof(float));
        float *B = malloc(k * nn * sizeof(float));
        float *C_blas = malloc(m * nn * sizeof(float));
        float *C_sme = malloc(m * nn * sizeof(float));
        rand_fill_matrix_fp32(A, m, k);
        rand_fill_matrix_fp32(B, k, nn);

        double sme_time = run_sme_sgemm(m, nn, k, A, B, C_sme);
        double blas_time = run_blas_sgemm(m, nn, k, A, B, C_blas);

        int match = verify_results(m * nn, C_blas, C_sme);
        speedups[i] = blas_time / sme_time;

        printf("  [%2d] matrixa_M=%-5d N=%-5d matrixa_K=%-5d  BLAS=%.4fs  SME=%.4fs  "
               "speedup=%.3fx  %s\n",
               i, m, nn, k, blas_time, sme_time, speedups[i], match ? "PASS" : "FAIL");

        free(A);
        free(B);
        free(C_blas);
        free(C_sme);
    }
}

static void compute_mean_var(double *data, int n, double *mean, double *var) {
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += data[i];
    *mean = sum / n;

    double sq_sum = 0.0;
    for (int i = 0; i < n; i++)
        sq_sum += (data[i] - *mean) * (data[i] - *mean);
    *var = sq_sum / n;
}

int unitest_sme_fp32_gemm() {
#if !defined(__APPLE__) && !defined(__linux__)
    printf("No BLAS library available. Only supported on macOS and Linux.\n");
    exit(1);
#endif

    const int N_TESTS = 9;

    int reg_M[] = {4096000, 256, 256, 20480, 20480, 512, 10240, 1024, 128};
    int reg_N[] = {256, 4096000, 256, 20480, 512, 20480, 10240, 1024, 128};
    int reg_K[] = {256, 256, 4096000, 512, 20480, 20480, 10240, 1024, 128};

    int irr_M[] = {12711, 12711, 503, 1271132, 503, 503, 49999, 49999, 1};
    int irr_N[] = {12711, 503, 12711, 503, 1271132, 503, 35713, 1, 35713};
    int irr_K[] = {503, 12711, 12711, 503, 503, 1271132, 1, 35713, 49999};

    double reg_speedups[N_TESTS], irr_speedups[N_TESTS];
    double mean, var;

    bench_group("Regular matrices", N_TESTS, reg_M, reg_N, reg_K, reg_speedups);
    compute_mean_var(reg_speedups, N_TESTS, &mean, &var);
    printf("\n  Regular speedup:  mean=%.4f  var=%.6f  stddev=%.4f\n", mean, var, sqrt(var));

    bench_group("Irregular matrices", N_TESTS, irr_M, irr_N, irr_K, irr_speedups);
    compute_mean_var(irr_speedups, N_TESTS, &mean, &var);
    printf("\n  Irregular speedup: mean=%.4f  var=%.6f  stddev=%.4f\n", mean, var, sqrt(var));

    printf("\n=== Summary ===\n");
    compute_mean_var(reg_speedups, N_TESTS, &mean, &var);
    printf("  Regular:   speedup mean=%.4f  var=%.6f\n", mean, var);
    compute_mean_var(irr_speedups, N_TESTS, &mean, &var);
    printf("  Irregular: speedup mean=%.4f  var=%.6f\n", mean, var);

    return 0;
}
int UnitTest() {

    // unitest_buffer_transpose_submatrixa();

    unitest_sme_fp32_gemm();

    exit(0);
}