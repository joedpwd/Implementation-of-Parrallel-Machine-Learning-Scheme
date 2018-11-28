#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>

int singleThread();
__global__ void gaussianKernelSimple(float* devGaussianMat, int height, int width);