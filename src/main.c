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

void raw_fp32_gemm(int matrixa_M, int matrixb_N, int matrixa_K, float *matrixa, float *matrixb, float *Matrixc) {

    for (int i = 0; i < matrixa_M; i++) {
        for (int j = 0; j < matrixb_N; j++) {
            for (int k = 0; k < matrixa_K; k++) {
                Matrixc[i * matrixb_N + j] += matrixa[i * matrixa_K + k] * matrixb[k * matrixb_N + j];
            }
        }
    }
}

int main() {

    double start_time, end_time;

    // int matrixa_M = 16, matrixb_N = 16, matrixa_K = 64;
    int matrixa_M = 7168, matrixb_N = 2048, matrixa_K = 20480;

    float *matrixa = (float *)malloc(matrixa_M * matrixa_K * sizeof(float));
    float *matrixb = (float *)malloc(matrixa_K * matrixb_N * sizeof(float));
    rand_fill_matrix_fp32(matrixa, matrixa_M, matrixa_K);
    rand_fill_matrix_fp32(matrixb, matrixa_K, matrixb_N);

    // float *raw_fp32_gemm_matrixc = (float *)malloc(matrixa_M * matrixb_N * sizeof(float));
    // memset(raw_fp32_gemm_matrixc, 0.0, matrixa_M * matrixb_N * sizeof(float));
    // start_time = dClock();
    // raw_fp32_gemm(matrixa_M, matrixb_N, matrixa_K, matrixa, matrixb, raw_fp32_gemm_matrixc);
    // end_time = dClock();
    // double raw_time = end_time - start_time;
    // printf("Raw FP32 GEMM time: %f seconds\n", raw_time);

    float *sme_fp32_gemm_matrixc = (float *)malloc(matrixa_M * matrixb_N * sizeof(float));
    memset(sme_fp32_gemm_matrixc, 0.0, matrixa_M * matrixb_N * sizeof(float));
    start_time = dClock();
    sme_fp32_gemm(matrixa_M, matrixb_N, matrixa_K, matrixa, matrixb, sme_fp32_gemm_matrixc);
    end_time = dClock();
    double sme_time = end_time - start_time;
    printf("SME FP32 GEMM time: %f seconds\n", sme_time);
    int mismatch = 0;
#ifdef MAC
    float *blas_matrix = (float *)malloc(matrixa_M * matrixb_N * sizeof(float));
    BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
    start_time = dClock();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, matrixa_M, matrixb_N, matrixa_K, 1.0f, matrixa, matrixa_K, matrixb,
                matrixb_N, 0.0f, blas_matrix, matrixb_N);
    end_time = dClock();
    double cblas_time = end_time - start_time;
    printf("BLAS FP32 GEMM time: %f seconds\n", cblas_time);

    for (int i = 0; i < matrixa_M * matrixb_N; i++) {
        if (fabsf(blas_matrix[i] - sme_fp32_gemm_matrixc[i]) > 1e-3f) {
            printf("BLAS vs SME mismatch at index %d: blas=%f, sme=%f\n", i, blas_matrix[i], sme_fp32_gemm_matrixc[i]);
            mismatch = 1;
            break;
        }
    }
    if (!mismatch) {
        printf("BLAS vs SME: all elements match!\n");
        printf("Speedup: %f\n", cblas_time / sme_time);
    }

    free(blas_matrix);
#elif defined(LS)
    float *MatrixKBlas = (float *)malloc(matrixa_M * matrixb_N * sizeof(float));
    memset(MatrixKBlas, 0, matrixa_M * matrixb_N * sizeof(float));
    start_time = dClock();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, matrixa_M, matrixb_N, matrixa_K, 1.0f, matrixa, matrixa_K, matrixb,
                matrixb_N, 0.0f, MatrixKBlas, matrixb_N);
    end_time = dClock();
    double kblas_time = end_time - start_time;
    printf("BLAS FP32 GEMM time: %f seconds\n", kblas_time);

    for (int i = 0; i < matrixa_M * matrixb_N; i++) {
        if (fabsf(MatrixKBlas[i] - sme_fp32_gemm_matrixc[i]) > 1e-3f) {
            printf("BLAS vs SME mismatch at index %d: blas=%f, sme=%f\n", i, MatrixKBlas[i], sme_fp32_gemm_matrixc[i]);
            mismatch = 1;
            break;
        }
    }
    if (!mismatch) {
        printf("BLAS vs SME: all elements match!\n");
        printf("Speedup: %f\n", cblas_time / sme_time);
    }

    free(MatrixKBlas);
#endif

    // for (int i = 0; i < matrixa_M * matrixb_N; i++) {
    //     if (fabsf(raw_fp32_gemm_matrixc[i] - sme_fp32_gemm_matrixc[i]) > 1e-3f) {
    //         printf("Mismatch at index %d: raw=%f, sme=%f\n", i, raw_fp32_gemm_matrixc[i], sme_fp32_gemm_matrixc[i]);
    //         mismatch = 1;
    //         break;
    //     }
    // }
    // if (!mismatch) {
    //     printf("All elements match! (matrixa_M=%d, matrixb_N=%d, matrixa_K=%d)\n", matrixa_M, matrixb_N, matrixa_K);
    //     // printf("Speedup: %f\n", raw_time / sme_time);
    // }

    free(matrixa);
    free(matrixb);
    // free(raw_fp32_gemm_matrixc);
    free(sme_fp32_gemm_matrixc);
    return 0;
}