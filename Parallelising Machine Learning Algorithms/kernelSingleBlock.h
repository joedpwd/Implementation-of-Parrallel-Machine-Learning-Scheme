#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int singleBlock();
__global__ void gaussianKernel(float* devGaussianMat, int height, int width);