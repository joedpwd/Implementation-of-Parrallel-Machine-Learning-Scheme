#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

__global__ void matrixMultiplicationKernel(float* dest, float* A, float* B, int m, int n, int k);