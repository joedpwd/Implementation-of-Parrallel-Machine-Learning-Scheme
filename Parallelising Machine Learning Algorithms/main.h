#pragma once

/*#include "kernelSingleBlock.h"
#include "kernelSingleThread.h"
#include "GaussianLUSingleBlock.h"*/
#include "denseLUSolver.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <mutex>

using namespace std::chrono;

std::mutex mtx;

int main(int argc, char *argv[]);
void RadonMachineInitialise(int d, int h, double *dataPoints);
void RadonMachineInstance(int d, int h, double *dataPoints);
int getMaxAllocation(size_t mem, int d);
void radonInstance(int d, cublasHandle_t *cublas, int threadId, double **d_A, double **d_B, int *piv, int *info, int equations, double *solvedEquations, cudaStream_t *s);