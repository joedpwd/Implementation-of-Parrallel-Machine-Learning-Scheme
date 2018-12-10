#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int singleLUBlock();
__global__ void gaussianLUKernel(float* devGaussianMat, int height, int width);