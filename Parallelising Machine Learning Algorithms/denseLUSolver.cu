#include "denseLUSolver.h"

//Function printMatrix is taken from:
//https://docs.nvidia.com/cuda/cusolver/index.html#lu_examples
//This provided a reference for using cuSolver to solve equations using LU Decomposition.

const int print = 0;

__device__ void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			double Areg = A[row + col * lda];
			printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg);
		}
	}
}
__global__ void printM(int m, int n, const double*A, const char* name)
{
	int col, row;
	for (row = 0; row < m; row++) {
		for (col = 0; col < n; col++)
			printf("%d - %4.5f\t", row * n + col, *(A + row * n + col));
		printf("\n");
	}
	printf("\n");
}

__global__ void configureEquations(int d, double *devData, double *devEquationData, int *devrh) {
	
	int r = d + 2;
	int m = d + 1;

	int a = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;
	int tid = a + b * (gridDim.x * blockDim.x);
	int c, j, i;

	c = tid;
	while(c < *devrh) {
	
		for (j = 0; j < d; j++) {
			*(devEquationData + j + (m*m) + (c*m * (m + 1))) = -*(devData + j + (c*r*d));
		}

		*(devEquationData + j + (m*m) + (c*m * (m + 1))) = -1;

		for (i = 1; i < r; i++) {
			for (j = 0; j < d; j++) {
				*(devEquationData + (i - 1)*m + j + (c*m * (m + 1))) = *(devData + (i*d) + j + (c*r*d));
			}
			*(devEquationData + (i - 1)*m + j + (c*m * (m + 1))) = 1;
		}

		

		c += blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	}

}

__global__ void devMemoryCopy(int m, double *src, double *dest, int len) {
	if (print) {
		printf("B = (matlab base-1)\n");
		printMatrix(m, 1, src, m, "X");
		printf("=====\n");
	}
	for (int j = 0; j < len; j++) {
		*(dest + j) = *(src + j);
	}
}

__global__ void solveEquations(int d, double *devData, double *devEquationData, int *devrh, double *hypothesisWorkspace) {
	int r = d + 2;
	int m = d + 1;
	
	int a = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;
	int tid = a + b * (gridDim.x * blockDim.x);
	int c, j, i;
	double lambda;
	double *hypothesis;
	double *curData;
	double *curEq;
	c = tid;

	while (c < *devrh) {
		hypothesis = hypothesisWorkspace + c * d;
		curData = (devData + (c*r*d));
		curEq = (devEquationData + (c*m));
		lambda = 1;
		for (i = 0; i < d; i++)
			*(hypothesis+i) = *(curData + i);
		for (i = 0; i < m; i++) {
			if (*(curEq + i) >= 0) {
				lambda += *(curEq + i);
				for (j = 0; j < d; j++)
					*(hypothesis+j) += *(curEq + i) * *(curData + ((i + 1) *d) + j);
			}
		}
		
		for (i = 0; i < d; i++) {
			*(hypothesis+i) /= lambda;
			*(curData + i) = *(hypothesis+i);
		}

		c += blockDim.x * blockDim.y * gridDim.x * gridDim.y;
		
	}

	c = tid;

	__syncthreads();
	
	//Limit copying of data to one thread to prevent race condition.
	if (tid == 0) {
		while (c < *devrh) {
			for (i = 0; i < d; i++) {
				*(devData + i + c * d) = *(devData + i + c * r * d);
				
			}

			c += 1;// blockDim.x * blockDim.y * gridDim.x * gridDim.y;
		}
	}



}