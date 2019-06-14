#include "main.h"

const int print = 1;

//Get the dimensions of the data
	//Set Valuues
	//h will be a hyper parameter passed to the program.

const int d = 12;

const int r = d + 2; //Assuming d = 2

const int h = 3; //Hyper parameter

const int m = d + 1; //Equivalent to d + 1

const int rh = pow(r, h);

int main(int argc, char **argv) {

	//Used for Iteration
	int i=0;
	int j=0;

	//Size of Data is r^h * d, where d is the no of features
	double *data = (double *)malloc(sizeof(double) * rh * d);


	//Create a vector of threads, one thread per execution of radon machine operation.
	//std::thread *thArray = (std::thread *)malloc(sizeof(std::thread) * pow(r, h - 1));
	std::vector<std::thread> thVect;

	
	//Read data in from CSV, data is stored in long long type and casted back into double type.
	std::ifstream dataFile;
	std::string t;
	std::string::size_type sz;

	dataFile.open("C:/Users/jxd45/Documents/Python Scripts/bigsmall.csv");
	long long *test = (long long *)malloc(sizeof(long long));
	if (dataFile.is_open())
	{
		while (std::getline(dataFile, t))
		{
			//std::cout << t << '\n';
			sz = 0;
			for (j = 0; j < d; j++) {
				t = t.substr(sz);
				*test = std::stoll(t, &sz);
				sz++;
				*(data + (i++)) = *reinterpret_cast<double *>(test);
			}
		}
		dataFile.close();
	}
	else
	{
		std::cout << "Unable to open file";

		return 0;
	}
	
	//Check the GPU capabilities


	//Debugging Purposes
	if (print == 1) {
		for (i = 0; i < d+4; i++) {
			printf("%.5f\n", *(data + i));
		}
		printf("\n");
	}
	
	//Start timer
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	startRadonMachine(data);
	//End Timers
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	
	if (print == 1) {
		for (i = 0; i < d+4; i++) {
			printf("%.5f\n", *(data + i));
		}
		printf("\n");
	}
	
	auto duration = duration_cast<microseconds>(t2 - t1).count();

	std::cout << duration << " microseconds";
	
	
	free(test);
	free(data);
}

void startRadonMachine(double *dataPoints ) {

	int i, j;
	double *devData;
	double *devEquationData;
	double *devSolvedEquations;
	int *devNofEquation;
	int maxThreads = 16;
	int threads;
	int noOfEquations;
	int equationsPerThread;
	cudaError_t c1;
	std::vector<std::thread> thVect;
	cudaStream_t *streams = NULL;

	//Create Streams
	streams = (cudaStream_t *)malloc(256 * sizeof(cudaStream_t));
	
	for (i = 0; i < 256; i++) {
		c1 = cudaStreamCreateWithFlags(streams+i, cudaStreamNonBlocking);
		assert(cudaSuccess == c1);
	}

	const dim3 blockSize(16, 16, 1);
	const dim3 gridSize(8, 8, 1);

	cusolverDnHandle_t cuSolver = NULL;
	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;

	/* Initialise cuSolver*/
	status = cusolverDnCreate(&cuSolver);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	//Allocate space for Equations, solved equations and space for data on the device. Then copy data to device.
	//Allocate A and B (A -> (m * m)), (B->1*m)) for r^h instances
	cudaMalloc(&devEquationData, (sizeof(double) * m * (m + 1))*(rh/r));
	cudaMalloc(&devSolvedEquations, (sizeof(double) * m)*(rh / r));
	cudaMalloc(&devData, sizeof(double) * rh * d);
	cudaMemcpy(devData, dataPoints, sizeof(double) * rh * d, cudaMemcpyHostToDevice);

	//Maintains the number of equations to be solved at each level of the radon tree
	cudaMalloc(&devNofEquation, sizeof(int));
	
	//printM << <1, 1 >> > (m, m, devData, "A");

	for (i = 0; i < h; i++) {
		noOfEquations = pow(r, h - 1 - i);
		cudaMemcpy(devNofEquation, &noOfEquations, sizeof(int), cudaMemcpyHostToDevice);
		configureEquations << < gridSize, blockSize >> > (devData, devEquationData, devNofEquation);
		cudaDeviceSynchronize();
		threads = (noOfEquations > maxThreads ? maxThreads : noOfEquations);
		equationsPerThread = noOfEquations / threads;
		
		//printf("%d threads %d equationsPerThread\n", threads, equationsPerThread);
		
		for (j = 0; j < threads; j++) {
			thVect.push_back(std::thread(radonInstance, cuSolver, j, (devEquationData + (j*equationsPerThread*m * (m + 1))), equationsPerThread, devSolvedEquations, streams + j));
		}
		for (std::thread & th : thVect)
		{
			if (th.joinable())
				th.join();
		}
		thVect.clear();

		solveEquations << < gridSize, blockSize >> > (devData, devSolvedEquations, devNofEquation);
		printM << <1, 1, 0>> > (10, 1, devData, "A");
		cudaDeviceSynchronize();
	}

	cudaMemcpy(dataPoints, devData, sizeof(double) * rh * d, cudaMemcpyDeviceToHost);
	
	if (cuSolver) cusolverDnDestroy(cuSolver);

	for (i = 0; i < 256; i++) {
		c1 = cudaStreamDestroy(*(streams + i));
		assert(cudaSuccess == c1);
	}
	
	c1 = cudaFree(devData);
	assert(cudaSuccess == c1);
	c1 = cudaFree(devSolvedEquations);
	assert(cudaSuccess == c1);
	c1 = cudaFree(devEquationData);
	assert(cudaSuccess == c1);

	free(streams);

}

void radonInstance(cusolverDnHandle_t cuSolver, int threadId, double *data, int equations, double *solvedEquations, cudaStream_t *s)
{
	mtx.lock();
	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;	/*Stores Error value for cusolver function calls*/

	/*Used to handle generic cuda errors*/
	cudaError_t c1 = cudaSuccess;
	cudaError_t c2 = cudaSuccess;

	double *d_A = NULL; /* device copy of A */
	double *d_B = NULL; /* device copy of B */
	int *d_Ipiv = NULL; /* pivoting sequence */
	int *d_info = NULL; /* error info for cuSolverDn */
	int  lwork = 0;     /* size of workspace for suSolverDn */
	double *d_work = NULL; /* device workspace for getrf, will be allocated using lwork */

	const int lda = m;
	const int ldb = m;

	const int pivot = 1; /*By default we will be using pivoting (pivot = 1)*/

	c1 = cudaMalloc((void**)&d_Ipiv, sizeof(int) * m);
	c2 = cudaMalloc((void**)&d_info, sizeof(int));
	assert(cudaSuccess == c1);
	assert(cudaSuccess == c2);
	
	
	status = cusolverDnSetStream(cuSolver, *s);
	assert(CUSOLVER_STATUS_SUCCESS == status);
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
		//printM << <1, 1 ,0, *s >> > (m, m, d_A, "A");
		//printf("\n");
		//printM<<<1,1,0, *s>>>(m, 1, d_B, "B");
		cudaStreamSynchronize(*s);
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
		
		assert(CUSOLVER_STATUS_SUCCESS == status);
		cudaStreamSynchronize(*s);
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
	
		assert(CUSOLVER_STATUS_SUCCESS == status);
		cudaStreamSynchronize(*s);
		devMemoryCopy << <1, 1, 0, *s >> > (d_B, (solvedEquations + (threadId*equations*m) + i * m), m);
		cudaStreamSynchronize(*s);
	}


	/* free resources */
	if (d_Ipiv) cudaFree(d_Ipiv);
	if (d_info) cudaFree(d_info);
	if (d_work) cudaFree(d_work);
	mtx.unlock();
}