#pragma once

/*#include "kernelSingleBlock.h"
#include "kernelSingleThread.h"
#include "GaussianLUSingleBlock.h"*/
#include "denseLUSolver.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

int main(int argc, char **argv);
void radonInstance(double *dataPoints, const int d);