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

int main(int argc, char **argv);
void startRadonMachine(double *dataPoints);
void radonInstance(cusolverDnHandle_t cuSolver, int threadId, double *data, int equations, double *solvedEquations);