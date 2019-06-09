#include "denseLUSolver.h"

//Function printMatrix is taken from:
//https://docs.nvidia.com/cuda/cusolver/index.html#lu_examples
//Function denseLUSolver has been adapted from the example from
// the same resource.

const int m = 3;
const int lda = m;
const int ldb = m;
const int print = 0;
const int d = 2;
const int r = d+2;

__device__ void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			double Areg = A[row + col * lda];
			printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg);
		}
	}
}

__global__ void configureEquations(double *devData, double *devEquationData, int *devrh) {
	int tid = threadIdx.x + threadIdx.y*blockDim.x;
	int c, j, i;
	double A[m*m];
	double B[m];
	c = (*devrh) - tid - 1;

	while (c >= 0) {
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

		/*if (tid == 0) {
			printf("%d\n", c);
			printMatrix(m, m, A, m, "A");
			printMatrix(m, m, (devData + (c*m * (m + 1))), m, "A");
		}*/
		
		c -= blockDim.x * blockDim.y;
	}

	
}

void radonInstance(int threadId, double *data, int equations, double *solvedEquations)
{
	/*double *hostA, double *hostB, double *hostX, double *LU, int *Ipiv, int *info, int m*/

	cusolverDnHandle_t cuSolver = NULL;	/*Will be passed to function that will initialise library and allocate resources*/
	cudaStream_t stream = NULL;
	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;	/*Stores Error value for cusolver function calls*/

	/*Used to handle generic cuda errors*/
	cudaError_t c1 = cudaSuccess;
	cudaError_t c2 = cudaSuccess;
	cudaError_t c3 = cudaSuccess;
	cudaError_t c4 = cudaSuccess;

	double *d_A = NULL; /* device copy of A */
	double *d_B = NULL; /* device copy of B */
	int *d_Ipiv = NULL; /* pivoting sequence */
	int *d_info = NULL; /* error info for cuSolverDn */
	int  lwork = 0;     /* size of workspace for suSolverDn */
	double *d_work = NULL; /* device workspace for getrf, will be allocated using lwork */

	const int lda = m;
	const int ldb = m;

	const int pivot = 1; /*By default we will be using pivoting (pivot = 1)*/

	/* Initialise cuSolver*/
	status = cusolverDnCreate(&cuSolver);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	c1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	assert(cudaSuccess == c1);

	status = cusolverDnSetStream(cuSolver, stream);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/*Allocate resources on device and copy A and B to device*/
	//c1 = cudaMalloc((void**)&d_A, sizeof(double) * lda * m);
	//c2 = cudaMalloc((void**)&d_B, sizeof(double) * m);
	c1 = cudaMalloc((void**)&d_Ipiv, sizeof(int) * m);
	c2 = cudaMalloc((void**)&d_info, sizeof(int));
	assert(cudaSuccess == c1);
	assert(cudaSuccess == c2);

	/*if (print)
	{
		printf("example of getrf \n");

		if (pivot) {
			printf("pivot is on : compute P*A = L*U \n");
		}
		else {
			printf("pivot is off: compute A = L*U (not numerically stable)\n");
		}

		printf("A = (matlab base-1)\n");
		printMatrix(m, m, hostA, lda, "A");
		printf("=====\n");

		printf("B = (matlab base-1)\n");
		printMatrix(m, 1, hostB, ldb, "B");
		printf("=====\n");
	}*/


	/*Get the size of the workspace required and store it in lwork.
	Then allocate the workspace and store reference at dwork*/

	status = cusolverDnDgetrf_bufferSize(
		cuSolver,
		m,
		m,
		d_A,
		lda,
		&lwork);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	c1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
	assert(cudaSuccess == c1);

	/* Perform LU Factorisation*/

	for (int i = 0; i < equations; i++) {

		d_A = (data + (i*m * (m + 1)));
		d_B = (data + (m*m) + (i*m * (m + 1)));

		if (pivot) {
			status = cusolverDnDgetrf(
				cuSolver,
				m,
				m,
				d_A,
				lda,
				d_work,
				d_Ipiv,
				d_info);
		}
		else {
			status = cusolverDnDgetrf(
				cuSolver,
				m,
				m,
				d_A,
				lda,
				d_work,
				NULL,
				d_info);
		}

		/* Wait until device has finished */
		c1 = cudaDeviceSynchronize();
		assert(CUSOLVER_STATUS_SUCCESS == status);
		assert(cudaSuccess == c1);

		/* Copy pivot values back to device if pivot is on. Also copy LU array and matrix back to device */
		/*if (pivot) {
			c1 = cudaMemcpy(Ipiv, d_Ipiv, sizeof(int)*m, cudaMemcpyDeviceToHost);
		}
		c2 = cudaMemcpy(LU, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
		c3 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
		assert(cudaSuccess == c1);
		assert(cudaSuccess == c2);
		assert(cudaSuccess == c3);*/

		/*Print values according to operation success*/
		/*if (0 > info) {
			printf("%d-th parameter is wrong \n", info);
			exit(1);
		}
		if (print) {
			if (pivot) {
				printf("pivoting sequence, matlab base-1\n");
				for (int j = 0; j < m; j++) {
					printf("Ipiv(%d) = %d\n", j + 1, Ipiv[j]);
				}
			}
		}
		if (print) {
			printf("L and U = (matlab base-1)\n");
			printMatrix(m, m, LU, lda, "LU");
			printf("=====\n");
		}*/

		/* Using LU decomposition solve for x*/
		if (pivot) {
			status = cusolverDnDgetrs(
				cuSolver,
				CUBLAS_OP_N,
				m,
				1, /* nrhs */
				d_A,
				lda,
				d_Ipiv,
				d_B,
				ldb,
				d_info);
		}
		else {
			status = cusolverDnDgetrs(
				cuSolver,
				CUBLAS_OP_N,
				m,
				1, /* nrhs */
				d_A,
				lda,
				NULL,
				d_B,
				ldb,
				d_info);
		}
		c1 = cudaDeviceSynchronize();
		assert(CUSOLVER_STATUS_SUCCESS == status);
		assert(cudaSuccess == c1);

		//c1 = cudaMemcpy(hostX, d_B, sizeof(double)*m, cudaMemcpyDeviceToHost);
		
		devMemoryCopy<< <1, 1 >>> (d_B, (solvedEquations + (threadId*equations*m) + i*m) , m);
	}

	/* free resources */
	if (d_A) cudaFree(d_A);
	if (d_B) cudaFree(d_B);
	if (d_Ipiv) cudaFree(d_Ipiv);
	if (d_info) cudaFree(d_info);
	if (d_work) cudaFree(d_work);

	if (cuSolver) cusolverDnDestroy(cuSolver);
	if (stream) cudaStreamDestroy(stream);

	//Not
	//cudaDeviceReset();

}

__global__ void devMemoryCopy(double *src, double *dest, int len) {
	for (int j = 0; j < len; j++) {
		*(dest + j) = *(src + j);
	}
}

__global__ void solveEquations(double *devData, double *devEquationData, int *devrh) {
	int tid = threadIdx.x + threadIdx.y*blockDim.x;
	int c, j, i;
	double lambda;
	double hypothesis[d];

	double *curData;
	double *curEq;
	c = tid;

	/*if (threadIdx.x + threadIdx.y*blockDim.x == 0) {
		for (i = 0; i < *devrh*r*d; i++) {
			printf("%.5f\n", *(devData + i));
		}
		printf("\n");
	}

	if (c == 0) {
		printf("%d\n", c);
		printf("%d\n", *devrh);
	}*/
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

		c+= blockDim.x * blockDim.y;
		
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

		c += blockDim.x * blockDim.y;
	}

	/*if (threadIdx.x + threadIdx.y*blockDim.x == 0) {
		for (i = 0; i < *devrh*r*d; i++) {
			printf("%.5f\n", *(devData + i));
		}
		printf("\n");
	}*/
}