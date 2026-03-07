#ifndef SGEMM_H
#define SGEMM_H

#include "utils.h"
#include <arm_sme.h>
#include <stdint.h>

#define SME_CACHELINE_SIZE 128

#define Submatrix_M 512
#define Submatrix_K 1024
#define Submatrix_N 256

void sme_fp32_gemm(int M, int N, int K, float *Matrixa, float *Matrixb, float *Matrixc);
#endif