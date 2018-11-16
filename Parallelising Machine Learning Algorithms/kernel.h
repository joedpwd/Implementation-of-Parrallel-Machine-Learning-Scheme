#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>

int main();
__global__ void doit(float *a);