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

//void printMatrix(int m, int n, const double*A, int lda, const char* name);
void radonInstance(int threadId, double *data, int equations, double *solvedEquations);
__global__ void configureEquations(double *devData, double *devEquationData, int *devrh);
__global__ void solveEquations(double *devData, double *devEquationData, int *devrh);
__global__ void devMemoryCopy(double *src, double *dest, int len);

#endif