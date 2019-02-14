#pragma once

#ifndef DENSE_LU_H
#define DENSE_LU_H

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>

//void printMatrix(int m, int n, const double*A, int lda, const char* name);
int denseLUSolver(double *hostA, double *hostB, double *hostX, double *LU, int *Ipiv, int *info, int m);

#endif