#include "smegemm.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MAC
#include <Accelerate/Accelerate.h>
#elif defined(LS)
#include "kblas.h"
#endif

void raw_fp32_gemm(int M, int N, int K, float *Matrixa, float *Matrixb, float *Matrixc) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                Matrixc[i * N + j] += Matrixa[i * K + k] * Matrixb[k * N + j];
            }
        }
    }
}

int main() {

    double start, end;

    // int M = 16, N = 16, K = 64;
    int M = 1024, N = 2048, K = 2048;

    float *Matrixa = (float *)malloc(M * K * sizeof(float));
    float *Matrixb = (float *)malloc(K * N * sizeof(float));
    rand_fill_matrix_fp32(Matrixa, M, K);
    rand_fill_matrix_fp32(Matrixb, K, N);

    float *MatrixcRawFP32Gemm = (float *)malloc(M * N * sizeof(float));
    memset(MatrixcRawFP32Gemm, 0.0, M * N * sizeof(float));
    start = dClock();
    raw_fp32_gemm(M, N, K, Matrixa, Matrixb, MatrixcRawFP32Gemm);
    end = dClock();
    double raw_time = end - start;
    printf("Raw FP32 GEMM time: %f seconds\n", raw_time);

    float *MatrixcSMEFP32Gemm = (float *)malloc(M * N * sizeof(float));
    memset(MatrixcSMEFP32Gemm, 0.0, M * N * sizeof(float));
    start = dClock();
    sme_fp32_gemm(M, N, K, Matrixa, Matrixb, MatrixcSMEFP32Gemm);
    end = dClock();
    double sme_time = end - start;
    printf("SME FP32 GEMM time: %f seconds\n", sme_time);
    int mismatch = 0;
#ifdef MAC
    float *MatrixBlas = (float *)malloc(M * N * sizeof(float));
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
    start = dClock();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, Matrixa, K, Matrixb, N, 0.0f, MatrixBlas, N);
    end = dClock();
    double cblas_time = end - start;
    printf("BLAS FP32 GEMM time: %f seconds\n", cblas_time);

    for (int i = 0; i < M * N; i++) {
        if (fabsf(MatrixBlas[i] - MatrixcSMEFP32Gemm[i]) > 1e-3f) {
            printf("BLAS vs SME mismatch at index %d: blas=%f, sme=%f\n", i, MatrixBlas[i], MatrixcSMEFP32Gemm[i]);
            mismatch = 1;
            break;
        }
    }
    if (!mismatch) {
        printf("BLAS vs SME: all elements match!\n");
    }

    free(MatrixBlas);
#elif defined(LS)
    float *MatrixKBlas = (float *)malloc(M * N * sizeof(float));
    memset(MatrixKBlas, 0, M * N * sizeof(float));
    start = dClock();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, Matrixa, K, Matrixb, N, 0.0f, MatrixKBlas, N);
    end = dClock();
    double kblas_time = end - start;
    printf("BLAS FP32 GEMM time: %f seconds\n", kblas_time);

    for (int i = 0; i < M * N; i++) {
        if (fabsf(MatrixKBlas[i] - MatrixcSMEFP32Gemm[i]) > 1e-3f) {
            printf("BLAS vs SME mismatch at index %d: blas=%f, sme=%f\n", i, MatrixKBlas[i], MatrixcSMEFP32Gemm[i]);
            mismatch = 1;
            break;
        }
    }
    if (!mismatch) {
        printf("BLAS vs SME: all elements match!\n");
    }

    free(MatrixKBlas);
#endif

    for (int i = 0; i < M * N; i++) {
        if (fabsf(MatrixcRawFP32Gemm[i] - MatrixcSMEFP32Gemm[i]) > 1e-3f) {
            printf("Mismatch at index %d: raw=%f, sme=%f\n", i, MatrixcRawFP32Gemm[i], MatrixcSMEFP32Gemm[i]);
            mismatch = 1;
            break;
        }
    }
    if (!mismatch) {
        printf("All elements match! (M=%d, N=%d, K=%d)\n", M, N, K);
        // printf("Speedup: %f\n", raw_time / sme_time);
    }

    free(Matrixa);
    free(Matrixb);
    free(MatrixcRawFP32Gemm);
    free(MatrixcSMEFP32Gemm);
    return 0;
}