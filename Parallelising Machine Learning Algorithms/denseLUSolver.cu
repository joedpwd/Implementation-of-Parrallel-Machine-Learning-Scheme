#include "denseLUSolverExample.h"

//Function printMatrix is taken from:
//https://docs.nvidia.com/cuda/cusolver/index.html#lu_examples
//Function denseLUSolver has been adapted from the example from
// the same resource.

const int m = 2;
const int lda = m;
const int ldb = m;

void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			double Areg = A[row + col * lda];
			printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg);
		}
	}
}

int denseLUSolver(double *hostA, double *hostB, double *hostX, double *LU, int *Ipiv, int *info, int m)
{
	cusolverDnHandle_t cuSolver = NULL;	/*Will be passed to function that will initialise library and allocate resources*/
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
	const int print = 1; /*Print useful information, (taken from example, see above)*/
	
	if (print)
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
	}

	/* Initialise cuSolver*/
	status = cusolverDnCreate(&cuSolver);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/*Allocate resources on device and copy A and B to device*/
	c1 = cudaMalloc((void**)&d_A, sizeof(double) * lda * m);
	c2 = cudaMalloc((void**)&d_B, sizeof(double) * m);
	c3 = cudaMalloc((void**)&d_Ipiv, sizeof(int) * m);
	c4 = cudaMalloc((void**)&d_info, sizeof(int));
	assert(cudaSuccess == c1);
	assert(cudaSuccess == c2);
	assert(cudaSuccess == c3);
	assert(cudaSuccess == c4);

	c1 = cudaMemcpy(d_A, hostA, sizeof(double)*lda*m, cudaMemcpyHostToDevice);
	c2 = cudaMemcpy(d_B, hostB, sizeof(double)*m, cudaMemcpyHostToDevice);
	assert(cudaSuccess == c1);
	assert(cudaSuccess == c2);

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
	if (pivot) {
		c1 = cudaMemcpy(Ipiv, d_Ipiv, sizeof(int)*m, cudaMemcpyDeviceToHost);
	}
	c2 = cudaMemcpy(LU, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
	c3 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
	assert(cudaSuccess == c1);
	assert(cudaSuccess == c2);
	assert(cudaSuccess == c3);

	/*Print values according to operation success*/
	if (0 > info) {
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
	printf("L and U = (matlab base-1)\n");
	printMatrix(m, m, LU, lda, "LU");
	printf("=====\n");

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

	c1 = cudaMemcpy(hostX, d_B, sizeof(double)*m, cudaMemcpyDeviceToHost);
	assert(cudaSuccess == c1);

	printf("X = (matlab base-1)\n");
	printMatrix(m, 1, hostX, ldb, "X");
	printf("=====\n");

	printf("%d\n", info);

	/* free resources */
	if (d_A) cudaFree(d_A);
	if (d_B) cudaFree(d_B);
	if (d_Ipiv) cudaFree(d_Ipiv);
	if (d_info) cudaFree(d_info);
	if (d_work) cudaFree(d_work);

	if (cuSolver) cusolverDnDestroy(cuSolver);

	//Not
	//cudaDeviceReset();
}