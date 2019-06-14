#include "denseLUSolver.h"

//Function printMatrix is taken from:
//https://docs.nvidia.com/cuda/cusolver/index.html#lu_examples
//Function denseLUSolver has been adapted from the example from
// the same resource.

const int print = 0;
const int d = 12;
const int r = d+2;
const int m = d+1;
const int lda = m;
const int ldb = m;

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

__global__ void configureEquations(double *devData, double *devEquationData, int *devrh) {
	int a = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;
	int tid = a + b * (gridDim.x * blockDim.x);
	int c, j, i;
	double A[m*m];
	double B[m];
	c = tid;
	while(c < *devrh) {
	
		for (j = 0; j < d; j++) {
			B[j] = -*(devData + j + (c*r*d));
		}

		B[j] = -1;

		for (i = 1; i < r; i++) {
			for (j = 0; j < d; j++) {
				A[(i - 1)*m + j] = *(devData + (i*d) + j + (c*r*d));
			}
			A[(i - 1)*m + j] = 1;
		}

		__syncthreads();

		for (i = 0; i < m*m; i++) {
			*(devEquationData + i + (c*m * (m + 1))) = A[i];
		}
		for (i = 0; i < m; i++) {
			*(devEquationData + i + (m*m) + (c*m * (m + 1))) = B[i];
		}

		c += blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	}
}

__global__ void devMemoryCopy(double *src, double *dest, int len) {
	if (print) {
		printf("B = (matlab base-1)\n");
		printMatrix(m, 1, src, m, "X");
		printf("=====\n");
	}
	for (int j = 0; j < len; j++) {
		*(dest + j) = *(src + j);
	}
}

__global__ void solveEquations(double *devData, double *devEquationData, int *devrh) {
	int a = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;
	int tid = a + b * (gridDim.x * blockDim.x);
	int c, j, i;
	double lambda;
	double hypothesis[d];

	double *curData;
	double *curEq;
	c = tid;

	while (c < *devrh) {
		curData = (devData + (c*r*d));
		curEq = (devEquationData + (c*m));
		lambda = 1;
		for (i = 0; i < d; i++)
			hypothesis[i] = *(curData + i);
		for (i = 0; i < m; i++) {
			if (*(curEq + i) >= 0) {
				lambda += *(curEq + i);
				for (j = 0; j < d; j++)
					hypothesis[j] += *(curEq + i) * *(curData + ((i + 1) *d) + j);
			}
		}
		
		for (i = 0; i < d; i++) {
			hypothesis[i] /= lambda;
			*(curData + i) = hypothesis[i];
		}

		c += blockDim.x * blockDim.y * gridDim.x * gridDim.y;
		
	}

	c = tid;

	__syncthreads();
	

	while (c < *devrh) {
		for (i = 0; i < d; i++) {
			*(devData + i + c * d) = *(devData + i + c * r * d);
			/*if (c == 0) {
				printf("%.5f\n", *(devData + i + c * d));
			}*/
		}

		c += blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	}


	//Debuggin purposes
	/*if (threadIdx.x + threadIdx.y*blockDim.x == 0) {
		for (i = 0; i < *devrh*r*d; i++) {
			printf("%.5f\n", *(devData + i));
		}
		printf("\n");
	}*/
}