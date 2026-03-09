#include "smegemm.h"
#include "utils.h"
#include <arm_sme.h>
#include <arm_sve.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

__attribute__((target("+sme+nosme2"))) __arm_locally_streaming __arm_new("za") void buffer_transpose_submatrixa(
    int submatrixa_M, int submatrixa_K, int matrixa_K, float *p2_submatrixa, float *matrixa_tile_buffer) {
    int svl_w_cnt = svcntw();
    int svl_w_cnt_x2 = svl_w_cnt * 2;
    int svl_w_cnt_x3 = svl_w_cnt * 3;
    int svl_w_cnt_x4 = svl_w_cnt * 4;
    int ZA_tile_size = svl_w_cnt * svl_w_cnt;
    int ZA_tile_size_X2 = ZA_tile_size * 2;
    int ZA_tile_size_X3 = ZA_tile_size * 3;
    int ZA_tile_size_X4 = ZA_tile_size * 4;
    int svl_w_cnt_X_submatrixa_K = svl_w_cnt * submatrixa_K;
    int svl_w_cnt_X_matrixa_K = svl_w_cnt * matrixa_K;
    float *p2_current_submatrixa = p2_submatrixa;
    float *aux_p2_current_submatrixa_row = NULL;
    float *aux_p2_current_submatrixa_col = NULL;
    float *p2_current_matrixa_tile_buffer = matrixa_tile_buffer;
    float *aux_p2_current_matrixa_tile_buffer = NULL;
    svbool_t PTrue = svptrue_b32();

    for (int submatrixa_M_idx = 0; submatrixa_M_idx < submatrixa_M; submatrixa_M_idx += svl_w_cnt) {

        aux_p2_current_submatrixa_col = p2_current_submatrixa;
        aux_p2_current_matrixa_tile_buffer = p2_current_matrixa_tile_buffer;
        for (int submatrixa_K_idx = 0; submatrixa_K_idx < submatrixa_K; submatrixa_K_idx += svl_w_cnt_x4) {

            aux_p2_current_submatrixa_row = aux_p2_current_submatrixa_col;

            svbool_t POf16Rows = svwhilelt_b32(0, submatrixa_M - submatrixa_M_idx);
            svbool_t POfTile0 = svwhilelt_b32(0, submatrixa_K - submatrixa_K_idx);
            svbool_t POfTile1 = svwhilelt_b32(0, submatrixa_K - submatrixa_K_idx - svl_w_cnt);
            svbool_t POfTile2 = svwhilelt_b32(0, submatrixa_K - submatrixa_K_idx - svl_w_cnt_x2);
            svbool_t POfTile3 = svwhilelt_b32(0, submatrixa_K - submatrixa_K_idx - svl_w_cnt_x3);

            for (int rowi = 0; rowi < svl_w_cnt; ++rowi) {
                svbool_t POfCurrentColumn0 = svpsel_lane_b32(POfTile0, POf16Rows, rowi);
                svbool_t POfCurrentColumn1 = svpsel_lane_b32(POfTile1, POf16Rows, rowi);
                svbool_t POfCurrentColumn2 = svpsel_lane_b32(POfTile2, POf16Rows, rowi);
                svbool_t POfCurrentColumn3 = svpsel_lane_b32(POfTile3, POf16Rows, rowi);
                svld1_hor_za32(0, rowi, POfCurrentColumn0, aux_p2_current_submatrixa_row);
                svld1_hor_za32(1, rowi, POfCurrentColumn1, aux_p2_current_submatrixa_row + svl_w_cnt);
                svld1_hor_za32(2, rowi, POfCurrentColumn2, aux_p2_current_submatrixa_row + svl_w_cnt_x2);
                svld1_hor_za32(3, rowi, POfCurrentColumn3, aux_p2_current_submatrixa_row + svl_w_cnt_x3);
                aux_p2_current_submatrixa_row += matrixa_K;
            }
            for (int coli = 0; coli < svl_w_cnt; ++coli) {
                svbool_t POfCurrentColumn0 = svpsel_lane_b32(PTrue, POfTile0, coli);
                svbool_t POfCurrentColumn1 = svpsel_lane_b32(PTrue, POfTile1, coli);
                svbool_t POfCurrentColumn2 = svpsel_lane_b32(PTrue, POfTile2, coli);
                svbool_t POfCurrentColumn3 = svpsel_lane_b32(PTrue, POfTile3, coli);
                svst1_ver_za32(0, coli, POfCurrentColumn0, aux_p2_current_matrixa_tile_buffer);
                svst1_ver_za32(1, coli, POfCurrentColumn1, aux_p2_current_matrixa_tile_buffer + ZA_tile_size);
                svst1_ver_za32(2, coli, POfCurrentColumn2, aux_p2_current_matrixa_tile_buffer + ZA_tile_size_X2);
                svst1_ver_za32(3, coli, POfCurrentColumn3, aux_p2_current_matrixa_tile_buffer + ZA_tile_size_X3);
                aux_p2_current_matrixa_tile_buffer += svl_w_cnt;
            }
            aux_p2_current_matrixa_tile_buffer += ZA_tile_size_X3;
            aux_p2_current_submatrixa_col += svl_w_cnt_x4;
        }
        p2_current_matrixa_tile_buffer += svl_w_cnt_X_submatrixa_K;
        p2_current_submatrixa += svl_w_cnt_X_matrixa_K;
    }
}

__attribute__((target("+sme+nosme2"))) __arm_locally_streaming __arm_new("za") void fp32_sme_submatrix_mm(
    int matrixa_M, int matrixb_N, int matrixa_K, int submatrixa_M, int Submatrixb_N, int Submatrixa_K, float *matrixa_tile_buffer,
    float *Point2Submatrixb, float *Point2Submatrixc) {

    int CntIn32OfSVL = svcntw();
    for (int Submatixa_MI = 0; Submatixa_MI < submatrixa_M; Submatixa_MI += CntIn32OfSVL) {
        svbool_t POfCurrentRow = svwhilelt_b32(0, min(CntIn32OfSVL, submatrixa_M - Submatixa_MI));

        for (int Submatrixb_NI = 0; Submatrixb_NI < Submatrixb_N; Submatrixb_NI += CntIn32OfSVL) {
            svbool_t POfCurrentColumn = svwhilelt_b32(0, min(CntIn32OfSVL, Submatrixb_N - Submatrixb_NI));
            for (int LoadMatrix_CI = 0; LoadMatrix_CI < CntIn32OfSVL; ++LoadMatrix_CI) {
                svbool_t POFLoadMatrixC = svpsel_lane_b32(POfCurrentColumn, POfCurrentRow, LoadMatrix_CI);
                svld1_hor_za32(0, LoadMatrix_CI, POFLoadMatrixC,
                               Point2Submatrixc + Submatixa_MI * matrixb_N + Submatrixb_NI + LoadMatrix_CI * matrixb_N);
            }
            for (int Submatrixa_KI = 0; Submatrixa_KI < Submatrixa_K; Submatrixa_KI += 1) {
                svfloat32_t ZSubmatrixa =
                    svld1_f32(POfCurrentRow, matrixa_tile_buffer + Submatixa_MI * Submatrixa_K + Submatrixa_KI * CntIn32OfSVL);
                svfloat32_t ZSubmatrixb =
                    svld1_f32(POfCurrentColumn, Point2Submatrixb + Submatrixa_KI * matrixb_N + Submatrixb_NI);

                svmopa_za32_f32_m(0, POfCurrentRow, POfCurrentColumn, ZSubmatrixa, ZSubmatrixb);
            }
            for (int LoadMatrix_CI = 0; LoadMatrix_CI < CntIn32OfSVL; ++LoadMatrix_CI) {
                svbool_t POFLoadMatrixC = svpsel_lane_b32(POfCurrentColumn, POfCurrentRow, LoadMatrix_CI);
                svst1_hor_za32(0, LoadMatrix_CI, POFLoadMatrixC,
                               Point2Submatrixc + Submatixa_MI * matrixb_N + Submatrixb_NI + LoadMatrix_CI * matrixb_N);
            }
        }
    }
}

void sme_fp32_gemm(int matrixa_M, int matrixb_N, int matrixa_K, float *matrixa, float *matrixb, float *matrixc) {
    float *restrict matrixa_tile_buffer;
    posix_memalign((void **)&matrixa_tile_buffer, SME_CACHELINE_SIZE, SUBMATRIX_M * SUBMATRIX_K * sizeof(float));

    for (int matrixa_M_idx = 0; matrixa_M_idx < matrixa_M; matrixa_M_idx += SUBMATRIX_M) {

        int submatrixa_M = min(SUBMATRIX_M, matrixa_M - matrixa_M_idx);

        for (int Matrixa_KI = 0; Matrixa_KI < matrixa_K; Matrixa_KI += SUBMATRIX_K) {

            int Submatrixa_K = min(SUBMATRIX_K, matrixa_K - Matrixa_KI);

            buffer_transpose_submatrixa(submatrixa_M, Submatrixa_K, matrixa_K, &matrixa[matrixa_M_idx * matrixa_K + Matrixa_KI],
                                        matrixa_tile_buffer);
            for (int matrixb_N_idx = 0; matrixb_N_idx < matrixb_N; matrixb_N_idx += SUBMATRIX_N) {

                int Submatrixb_N = min(SUBMATRIX_N, matrixb_N - matrixb_N_idx);

                fp32_sme_submatrix_mm(matrixa_M, matrixb_N, matrixa_K, submatrixa_M, Submatrixb_N, Submatrixa_K,
                                      matrixa_tile_buffer, matrixb + Matrixa_KI * matrixb_N + matrixb_N_idx,
                                      matrixc + matrixa_M_idx * matrixb_N + matrixb_N_idx);
            }
        }
    }
    free(matrixa_tile_buffer);
}