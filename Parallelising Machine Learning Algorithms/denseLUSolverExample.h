#pragma once

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include <cassert>

void printMatrix(int m, int n, const double*A, int lda, const char* name);
int denseLUSolverExample();