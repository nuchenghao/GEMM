#ifndef UTILS_H
#define UTILS_H

#include <arm_sve.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

double dClock();

void rand_fill_matrix_fp32(float *matrix_A, int M, int K);

#endif