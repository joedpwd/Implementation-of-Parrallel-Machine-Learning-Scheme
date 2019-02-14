#include "kernelMatrixMultiplication.h"

__global__ void matrixMultiplicationKernel(float* dest, float* A, float* B, int m, int n, int k) {
	
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int sum = 0;

	if (col < k && row < m) {
		for (int i = 0; i < n; i++)
			sum += A[row * n + i] * B[i * k + col];

		dest[row * k + col] = sum;
	}
}