#include "smegemm.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void row_packa_output(int m, int k, float *XA, float *result);
void BufferSubMatrixAAndTranspose(int Submatrixa_M, int Submatrixa_K, int K, uint32_t *Matrixa, uint32_t *MatrixaTileBuffer);

void TestBufferSubMatrixAAndTranspose(int M, int K, float *Matrixa, float *result) {
    float *restrict atilde_buffer;
    posix_memalign((void **)&atilde_buffer, SME_CACHELINE_SIZE, Submatrix_M * Submatrix_K * sizeof(float));

    int total = 0;
    for (int mi = 0; mi < M; mi += Submatrix_M) {
        int Submatrixa_M = min(Submatrix_M, M - mi);

        for (int ki = 0; ki < K; ki += Submatrix_K) {

            int Submatrixa_K = min(Submatrix_K, K - ki);
            BufferSubMatrixAAndTranspose(Submatrixa_M, Submatrixa_K, K, (uint32_t *)&Matrixa[mi * K + ki], (uint32_t *)atilde_buffer);

            int count = (((Submatrixa_M + 15) / 16) * 16) * Submatrixa_K;
            memcpy(&result[total], atilde_buffer, count * sizeof(float));
            total += count;
        }
    }

    free(atilde_buffer);
}

int UnitTest() {
    int MArrayp[] = {1, 23, 1024, 328, 1078, 2049, 7893, 12345, 87630, 12342, 20480, 1, 1000000};
    int KArrayp[] = {16, 1, 26, 382, 12340, 12048, 1024, 10204, 10240, 1024, 1024, 1000000, 1};
    int pass = 1;
    for (int i = 0; i < 13; i++) {
        int M = MArrayp[i];
        int K = KArrayp[i];
        float *Matrixa = malloc(M * K * sizeof(float));
        float *correct = malloc(((M + 15) / 16) * 16 * K * sizeof(float));
        float *answer = malloc(((M + 15) / 16) * 16 * K * sizeof(float));
        row_packa_output(M, K, Matrixa, correct);

        TestBufferSubMatrixAAndTranspose(M, K, Matrixa, answer);

        int size = ((M + 15) / 16) * 16 * K;
        int all_match = 1;
        for (int j = 0; j < size; j++) {
            if (fabsf(correct[j] - answer[j]) > 1e-8f) {
                all_match = 0;
                printf("Mismatch at index %d: correct=%f, answer=%f\n", j, correct[j], answer[j]);
                break;
            }
        }
        if (all_match) {
            free(Matrixa);
            free(correct);
            free(answer);
        } else {
            pass = 0;
            free(Matrixa);
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
    return 0;
}