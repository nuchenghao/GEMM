#include "smegemm.h"
#include "utils.h"
#include <arm_sme.h>
#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__attribute__((target("+sme+nosme2"))) __arm_locally_streaming __arm_new("za") void BufferSubMatrixAAndTranspose(
    int Submatrixa_M, int Submatrixa_K, int K, float *Matrixa, float *MatrixaTileBuffer) {
    int CntIn32OfSVL = svcntw();
    int CntIn32OfSVL4 = CntIn32OfSVL * 4;
    int ZASize = CntIn32OfSVL * CntIn32OfSVL;
    int ZASize2 = ZASize * 2;
    int ZASize3 = ZASize * 3;
    float *CurrentMatrixaTileBuffer = MatrixaTileBuffer;
    float *CurrentMatrixa = Matrixa;
    svbool_t PTrue = svptrue_b32();
    for (int Submatixa_MI = 0; Submatixa_MI < Submatrixa_M; Submatixa_MI += CntIn32OfSVL) {
        for (int Submatrixa_KI = 0; Submatrixa_KI < Submatrixa_K; Submatrixa_KI += CntIn32OfSVL4) {
            svbool_t POf16Rows = svwhilelt_b32(0, min(CntIn32OfSVL, Submatrixa_M - Submatixa_MI));
            svbool_t POfTile0 = svwhilelt_b32(0, min(CntIn32OfSVL, max(Submatrixa_K - Submatrixa_KI, 0)));
            svbool_t POfTile1 = svwhilelt_b32(0, min(CntIn32OfSVL, max(Submatrixa_K - Submatrixa_KI - CntIn32OfSVL, 0)));
            svbool_t POfTile2 = svwhilelt_b32(0, min(CntIn32OfSVL, max(Submatrixa_K - Submatrixa_KI - CntIn32OfSVL * 2, 0)));
            svbool_t POfTile3 = svwhilelt_b32(0, min(CntIn32OfSVL, max(Submatrixa_K - Submatrixa_KI - CntIn32OfSVL * 3, 0)));
            float *Current416Matrix = Matrixa + Submatixa_MI * K + Submatrixa_KI;
            float *Current416MatrixTileBuffer = MatrixaTileBuffer + Submatixa_MI * Submatrixa_K + Submatrixa_KI * CntIn32OfSVL;
            for (int rowi = 0; rowi < CntIn32OfSVL; ++rowi) {
                svbool_t POfCurrentColumn0 = svpsel_lane_b32(POfTile0, POf16Rows, rowi);
                svbool_t POfCurrentColumn1 = svpsel_lane_b32(POfTile1, POf16Rows, rowi);
                svbool_t POfCurrentColumn2 = svpsel_lane_b32(POfTile2, POf16Rows, rowi);
                svbool_t POfCurrentColumn3 = svpsel_lane_b32(POfTile3, POf16Rows, rowi);
                svld1_hor_za32(0, rowi, POfCurrentColumn0, Current416Matrix + rowi * K);
                svld1_hor_za32(1, rowi, POfCurrentColumn1, Current416Matrix + rowi * K + CntIn32OfSVL);
                svld1_hor_za32(2, rowi, POfCurrentColumn2, Current416Matrix + rowi * K + CntIn32OfSVL * 2);
                svld1_hor_za32(3, rowi, POfCurrentColumn3, Current416Matrix + rowi * K + CntIn32OfSVL * 3);
            }
            for (int coli = 0; coli < CntIn32OfSVL; ++coli) {
                svbool_t POfCurrentColumn0 = svpsel_lane_b32(PTrue, POfTile0, coli);
                svbool_t POfCurrentColumn1 = svpsel_lane_b32(PTrue, POfTile1, coli);
                svbool_t POfCurrentColumn2 = svpsel_lane_b32(PTrue, POfTile2, coli);
                svbool_t POfCurrentColumn3 = svpsel_lane_b32(PTrue, POfTile3, coli);
                svst1_ver_za32(0, coli, POfCurrentColumn0, Current416MatrixTileBuffer + coli * CntIn32OfSVL);
                svst1_ver_za32(1, coli, POfCurrentColumn1, Current416MatrixTileBuffer + coli * CntIn32OfSVL + ZASize);
                svst1_ver_za32(2, coli, POfCurrentColumn2, Current416MatrixTileBuffer + coli * CntIn32OfSVL + ZASize2);
                svst1_ver_za32(3, coli, POfCurrentColumn3, Current416MatrixTileBuffer + coli * CntIn32OfSVL + ZASize3);
            }
        }
    }
}

__attribute__((target("+sme+nosme2"))) __arm_locally_streaming __arm_new("za") void FP32SubmatrixSMEMM(
    int Matrixa_M, int Matrixb_N, int Matrixa_K, int Submatrixa_M, int Submatrixb_N, int Submatrixa_K, float *MatrixaTileBuffer,
    float *Point2Submatrixb, float *Point2Submatrixc) {

    int CntIn32OfSVL = svcntw();
    for (int Submatixa_MI = 0; Submatixa_MI < Submatrixa_M; Submatixa_MI += CntIn32OfSVL) {
        svbool_t POfCurrentRow = svwhilelt_b32(0, min(CntIn32OfSVL, Submatrixa_M - Submatixa_MI));

        for (int Submatrixb_NI = 0; Submatrixb_NI < Submatrixb_N; Submatrixb_NI += CntIn32OfSVL) {
            svbool_t POfCurrentColumn = svwhilelt_b32(0, min(CntIn32OfSVL, Submatrixb_N - Submatrixb_NI));
            for (int LoadMatrix_CI = 0; LoadMatrix_CI < CntIn32OfSVL; ++LoadMatrix_CI) {
                svbool_t POFLoadMatrixC = svpsel_lane_b32(POfCurrentColumn, POfCurrentRow, LoadMatrix_CI);
                svld1_hor_za32(0, LoadMatrix_CI, POFLoadMatrixC,
                               Point2Submatrixc + Submatixa_MI * Matrixb_N + Submatrixb_NI + LoadMatrix_CI * Matrixb_N);
            }
            for (int Submatrixa_KI = 0; Submatrixa_KI < Submatrixa_K; Submatrixa_KI += 1) {
                svfloat32_t ZSubmatrixa =
                    svld1_f32(POfCurrentRow, MatrixaTileBuffer + Submatixa_MI * Submatrixa_K + Submatrixa_KI * CntIn32OfSVL);
                svfloat32_t ZSubmatrixb =
                    svld1_f32(POfCurrentColumn, Point2Submatrixb + Submatrixa_KI * Matrixb_N + Submatrixb_NI);

                svmopa_za32_f32_m(0, POfCurrentRow, POfCurrentColumn, ZSubmatrixa, ZSubmatrixb);
            }
            for (int LoadMatrix_CI = 0; LoadMatrix_CI < CntIn32OfSVL; ++LoadMatrix_CI) {
                svbool_t POFLoadMatrixC = svpsel_lane_b32(POfCurrentColumn, POfCurrentRow, LoadMatrix_CI);
                svst1_hor_za32(0, LoadMatrix_CI, POFLoadMatrixC,
                               Point2Submatrixc + Submatixa_MI * Matrixb_N + Submatrixb_NI + LoadMatrix_CI * Matrixb_N);
            }
        }
    }
}

void sme_fp32_gemm(int Matrixa_M, int Matrixb_N, int Matrixa_K, float *Matrixa, float *Matrixb, float *Matrixc) {
    float *restrict MatrixaTileBuffer;
    posix_memalign((void **)&MatrixaTileBuffer, SME_CACHELINE_SIZE, Submatrix_M * Submatrix_K * sizeof(float));

    for (int Matrixa_MI = 0; Matrixa_MI < Matrixa_M; Matrixa_MI += Submatrix_M) {
        int Submatrixa_M = min(Submatrix_M, Matrixa_M - Matrixa_MI);
        for (int Matrixa_KI = 0; Matrixa_KI < Matrixa_K; Matrixa_KI += Submatrix_K) {
            int Submatrixa_K = min(Submatrix_K, Matrixa_K - Matrixa_KI);
            BufferSubMatrixAAndTranspose(Submatrixa_M, Submatrixa_K, Matrixa_K, &Matrixa[Matrixa_MI * Matrixa_K + Matrixa_KI],
                                         MatrixaTileBuffer);
            for (int Matrixb_NI = 0; Matrixb_NI < Matrixb_N; Matrixb_NI += Submatrix_N) {
                int Submatrixb_N = min(Submatrix_N, Matrixb_N - Matrixb_NI);
                FP32SubmatrixSMEMM(Matrixa_M, Matrixb_N, Matrixa_K, Submatrixa_M, Submatrixb_N, Submatrixa_K, MatrixaTileBuffer,
                                   Matrixb + Matrixa_KI * Matrixb_N + Matrixb_NI, Matrixc + Matrixa_MI * Matrixb_N + Matrixb_NI);
            }
        }
    }
    free(MatrixaTileBuffer);
}