/*#include "kernelSingleBlock.h"
#include "kernelSingleThread.h"
#include "GaussianLUSingleBlock.h"*/
#include "denseLUSolver.h"

int main() {
	//r = d + 2
	const int r = 5; //Assuming d = 3
	
	const int h = 1; //Hyper parameter
	
	const int d = 3;

	const int m = 4; //Equivalent to d + 1

	const int rh = pow(r, h);

	/*Read Data in, use 100 for now but later should use more dynamic approach*/
	double data[100][d];

	double luSolverA[100 * m];
	double luSolverB[m];

	double lambda[r];

	const int lda = m;
	const int ldb = m;

	int i;
	int j;

	for (i = 0; i < rh; i++) {
		data[i][0] = rand() % 50;
		data[i][1] = rand() % 20;
		data[i][2] = i * i + 1;
		//printf("%.2lf %.2lf %.2lf\n", data[i][0], data[i][1], data[i][2]);
	}
	//Until we get data just plug in the same data
	for (j = 0; j < d; j++) {
		luSolverB[j] = -data[0][j];
	}

	luSolverB[j] = -1;

	for (i = 1; i < rh; i++) {
		for (j = 0; j < d; j++) {
			luSolverA[(i-1)*m + j] = data[i][j];
		}
		luSolverA[(i - 1)*m + j] = 1;
	}

	luSolverA[9] = 1; //Makes the test matrix non singular

	//double A[lda*m] = { 1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0 };
	//double B[m] = { 1.0, 2.0, 3.0 };
	double X[m]; /* X = A\B */
	double LU[lda*m]; /* L and U */
	int Ipiv[m];      /* host copy of pivoting sequence */
	int info = 0;     /* host copy of error info */

	double *hostA = (double *)(malloc(m*m * sizeof(double)));
	double *hostB = (double *)(malloc(m * sizeof(double)));
	double *hostX = (double *)(malloc(m * sizeof(double)));
	double *hostLU = (double *)(malloc(m*m * sizeof(double)));
	int *hostIpiv = (int *)(malloc(sizeof(int)));
	int *hostInfo = (int *)(malloc(sizeof(int)));

	for (i = 0; i < m*m; i++) {
		*(hostA + i) = luSolverA[i];
		printf("%.2lf\n", luSolverA[i]);
	}
	for (i = 0; i < m; i++)
		*(hostB + i) = luSolverB[i];


	denseLUSolver(hostA, hostB, hostX, hostLU, hostIpiv, hostInfo, m);
}