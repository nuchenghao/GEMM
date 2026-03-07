#include "smegemm.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    int M = 1, N = 4, K = 3;

    float *Matrixa = (float *)malloc(M * K * sizeof(float));
    float *Matrixb = (float *)malloc(K * N * sizeof(float));
    rand_fill_matrix_fp32(Matrixa, M, K);
    rand_fill_matrix_fp32(Matrixb, K, N);

    float *MatrixcRawFP32Gemm = (float *)malloc(M * N * sizeof(float));
    memset(MatrixcRawFP32Gemm, 0.0, M * N * sizeof(float));
    raw_fp32_gemm(M, N, K, Matrixa, Matrixb, MatrixcRawFP32Gemm);

    float *MatrixcSMEFP32Gemm = (float *)malloc(M * N * sizeof(float));
    memset(MatrixcSMEFP32Gemm, 0.0, M * N * sizeof(float));

    sme_fp32_gemm(M, N, K, Matrixa, Matrixb, MatrixcSMEFP32Gemm);

    free(Matrixa);
    free(Matrixb);
    free(MatrixcRawFP32Gemm);
    free(MatrixcSMEFP32Gemm);
    return 0;
}