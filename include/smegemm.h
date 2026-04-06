#ifndef SGEMM_H
#define SGEMM_H

#include <arm_sme.h>
#include <stdint.h>

#define SME_CACHELINE_SIZE 128
#ifdef __APPLE__
#define SUBMATRIX_M 512
#define SUBMATRIX_K 1024
#define SUBMATRIX_N 256
#elif defined(__linux__)
#define SUBMATRIX_M 64
#define SUBMATRIX_K 1023
#define SUBMATRIX_N 64
#endif
void sme_fp32_gemm(int M, int N, int K, float *Matrixa, float *Matrixb, float *Matrixc);
#endif