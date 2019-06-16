#pragma once

#ifndef DENSE_LU_H
#define DENSE_LU_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>

__device__ void printMatrix(int m, int n, const double*A, int lda, const char* name);
__global__ void printM(int m, int n, const double*A, const char* name);
__global__ void configureEquations(int d, double *devData, double *devEquationData, int *devrh);
__global__ void solveEquations(int d, double *devData, double *devEquationData, int *devrh, double *hypothesisWorkspace);
__global__ void devMemoryCopy(int m, double *src, double *dest, int len);

#endif