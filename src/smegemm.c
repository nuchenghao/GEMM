#include "smegemm.h"
#include "utils.h"
#include <arm_sme.h>
#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__attribute__((target("+sme+nosme2"))) __arm_locally_streaming __arm_new("za") void BufferSubMatrixAAndTranspose(
    int Submatrixa_M, int Submatrixa_K, int K, uint32_t *Matrixa, uint32_t *MatrixaTileBuffer) {
    int CntIn32OfSVL = svcntw();
    int CntIn32OfSVL4 = CntIn32OfSVL * 4;
    int ZASize = CntIn32OfSVL * CntIn32OfSVL;
    int ZASize2 = ZASize * 2;
    int ZASize3 = ZASize * 3;
    uint32_t *CurrentMatrixaTileBuffer = MatrixaTileBuffer;
    uint32_t *CurrentMatrixa = Matrixa;
    svbool_t PTrue = svptrue_b32();
    for (int Submatixa_MI = 0; Submatixa_MI < Submatrixa_M; Submatixa_MI += CntIn32OfSVL) {
        for (int Submatrixa_KI = 0; Submatrixa_KI < Submatrixa_K; Submatrixa_KI += CntIn32OfSVL4) {
            svbool_t POf16Rows = svwhilelt_b32(0, min(CntIn32OfSVL, Submatrixa_M - Submatixa_MI));
            svbool_t POfTile0 = svwhilelt_b32(0, min(CntIn32OfSVL, max(Submatrixa_K - Submatrixa_KI, 0)));
            svbool_t POfTile1 = svwhilelt_b32(0, min(CntIn32OfSVL, max(Submatrixa_K - Submatrixa_KI - CntIn32OfSVL, 0)));
            svbool_t POfTile2 = svwhilelt_b32(0, min(CntIn32OfSVL, max(Submatrixa_K - Submatrixa_KI - CntIn32OfSVL * 2, 0)));
            svbool_t POfTile3 = svwhilelt_b32(0, min(CntIn32OfSVL, max(Submatrixa_K - Submatrixa_KI - CntIn32OfSVL * 3, 0)));
            uint32_t *Current416Matrix = Matrixa + Submatixa_MI * K + Submatrixa_KI;
            uint32_t *Current416MatrixTileBuffer = MatrixaTileBuffer + Submatixa_MI * K + Submatrixa_KI * CntIn32OfSVL;
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

void sme_fp32_gemm(int M, int N, int K, float *Matrixa, float *Matrixb, float *Matrixc) {}
