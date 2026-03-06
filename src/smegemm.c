#include "smegemm.h"
#include <stdint.h>

__attribute__((target("+sme"))) __arm_locally_streaming void
BufferSubMatrixAAndTranspose(int Submatrixa_M, int Submatrixa_K, int K,
                             uint8_t *Matrixa, uint8_t *MatrixaTileBuffer) {
  uint64_t CntIn32OfSVL = svcntw();
  printf("%llu\n", CntIn32OfSVL);
}

void sme_fp32_gemm(int M, int N, int K, float *Matrixa, float *Matrixb,
                   float *Matrixc) {
  float *restrict atilde_buffer;
  float *restrict btilde_buffer;
  posix_memalign((void **)&atilde_buffer, SME_CACHELINE_SIZE,
                 Submatrix_M * Submatrix_K * sizeof(float));
  posix_memalign((void **)&btilde_buffer, SME_CACHELINE_SIZE,
                 Submatrix_K * Submatrix_N * sizeof(float));

  for (int mi = 0; mi < M; mi += Submatrix_M) {
    for (int ki = 0; ki < K; ki += Submatrix_K) {
    }
  }
  BufferSubMatrixAAndTranspose(Submatrix_M, Submatrix_K, K, (uint8_t *)Matrixa,
                               (uint8_t *)atilde_buffer);
}