#include "main.h"

const int print = 1;

//Get the dimensions of the data
	//Set Valuues
	//h will be a hyper parameter passed to the program.

const int d = 2;

const int r = d + 2; //Assuming d = 2

const int h = 3; //Hyper parameter

const int m = d + 1; //Equivalent to d + 1

const int rh = pow(r, h);

int main(int argc, char **argv) {

	//Used for Iteration
	int i=0;
	int j=0;
	int k=0;

	//Perform rh instances of ML problem

	//Size of Data is r^h * d, where d is the no of features
	double *data = (double *)malloc(sizeof(double) * rh * d);


	//Create a vector of threads, one thread per execution of radon machine operation.
	//std::thread *thArray = (std::thread *)malloc(sizeof(std::thread) * pow(r, h - 1));
	std::vector<std::thread> thVect;

	
	//Read data in from CSV, data is stored in long long type and casted back into double type.
	std::ifstream dataFile;
	std::string t;
	std::string::size_type sz;
	dataFile.open("C:/Users/jxd45/Documents/Python Scripts/csvtest.csv");
	long long *test = (long long *)malloc(sizeof(long long));
	if (dataFile.is_open())
	{
		while (std::getline(dataFile, t))
		{
			//std::cout << t << '\n';
			sz = 0;
			for (j = 0; j < d; j++) {
				*test = std::stoll(t.substr(sz), &sz);
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



	if (print == 1) {
		for (i = 0; i < d* 4; i++) {
			printf("%.5f\n", *(data + i));
		}
		printf("\n");
	}
	
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	startRadonMachine(data);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	if (print == 1) {
		for (i = 0; i < d *4; i++) {
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
	int maxThreads = 4;
	int threads;
	int noOfEquations;
	int equationsPerThread;
	std::vector<std::thread> thVect;

	const dim3 blockSize(16, 16, 1);
	const dim3 gridSize(1, 1, 1);

	cusolverDnHandle_t cuSolver = NULL;
	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;

	/* Initialise cuSolver*/
	status = cusolverDnCreate(&cuSolver);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	//Allocate A and B (A -> (m * m)), (B->1*m)) for r^h instances
	cudaMalloc(&devEquationData, (sizeof(double) * m * (m + 1))*(rh/r));
	cudaMalloc(&devSolvedEquations, (sizeof(double) * m)*(rh / r));
	cudaMalloc(&devData, sizeof(double) * rh * d);
	cudaMemcpy(devData, dataPoints, sizeof(double) * rh * d, cudaMemcpyHostToDevice);

	cudaMalloc(&devNofEquation, sizeof(int));
	

	for (i = 0; i < h; i++) {
		noOfEquations = pow(r, h - 1 - i);
		cudaMemcpy(devNofEquation, &noOfEquations, sizeof(int), cudaMemcpyHostToDevice);
		configureEquations << < gridSize, blockSize >> > (devData, devEquationData, devNofEquation);

		threads = (noOfEquations > maxThreads ? maxThreads : noOfEquations);
		equationsPerThread = noOfEquations / threads;
		printf("%d threads %d equationsPerThread\n", threads, equationsPerThread);

		for (j = 0; j < threads; j++) {
			thVect.push_back(std::thread(radonInstance, cuSolver, j, (devEquationData + (j*equationsPerThread*m * (m + 1))), equationsPerThread, devSolvedEquations));
			//radonInstance((data + (d*j*r)), d);
		}
		for (std::thread & th : thVect)
		{
			// If thread Object is Joinable then Join that thread.
			if (th.joinable())
				th.join();
		}
		thVect.clear();

		solveEquations << < gridSize, blockSize >> > (devData, devSolvedEquations, devNofEquation);
		//Will sort memory out in thread
		/*for (j = 0; j < pow(r, h-1 - i); j++) {
			for(k=0;k<d;k++)
				*(data + (j*d) + k) = *(data + (r*j*d) + k);
		}*/
	}

	cudaMemcpy(dataPoints, devData, sizeof(double) * rh * d, cudaMemcpyDeviceToHost);
	
	if (cuSolver) cusolverDnDestroy(cuSolver);
	/*int i, j;

	const int r = d + 2; //Assuming d = 2

	const int m = d + 1;

	double *hostA = (double *)(malloc(m*m * sizeof(double)));
	double *hostB = (double *)(malloc(m * sizeof(double)));
	double *hostX = (double *)(malloc(m * sizeof(double)));
	double *hostLU = (double *)(malloc(m*m * sizeof(double)));
	int *hostIpiv = (int *)(malloc(m * sizeof(int)));
	int *hostInfo = (int *)(malloc(sizeof(int)));

	long long *Acopy;
	long long *Bcopy;
	long long *Xcopy;

	//Format raw data into correct format for LU factorisation.
	for (j = 0; j < d; j++) {
		hostB[j] = -*(dataPoints+j);
	}

	hostB[j] = -1;

	for (i = 1; i < r; i++) {
		for (j = 0; j < d; j++) {
			hostA[(i - 1)*m + j] = *(dataPoints + (i*d) + j);
		}
		hostA[(i - 1)*m + j] = 1;
	}

	//Perform LU Factorisation
	denseLUSolver(hostA, hostB, hostX, hostLU, hostIpiv, hostInfo, m);*/

	/*for (i = 0; i < m; i++)
		printf("X\t%d\t%.4f\n", i, *(hostX+i));*/

	/*double lambda = 1;
	double *hypothesis = (double *)malloc(sizeof(double)*d);

	//Obtain correct hypothesis for instance. We have chosen to obtain the hypothesis by
	//using the positive index set I. L0 is always 1 (positive) and is therefore always in this set.
	//Therefore the first loop is to copy the first data points for this value into the 
	//hypothesis array. This array is the sum of data points whose index corresponds to the
	//values in the positive index set I.

	for (i = 0; i < d; i++)
		hypothesis[i] = *(dataPoints + i);

	for (i = 0; i < m; i++) {
		if (*(hostX + i) >= 0) {
			lambda += *(hostX + i);
			for (j = 0; j < d; j++)
				hypothesis[j] += *(hostX + i) * *(dataPoints + ((i + 1) *d) + j);
		}
		else
			continue;
	}

	//printf("RESULT\n");

	for (i = 0; i < d; i++) {
		printf("%.5f\n", hypothesis[i]);
	}
	//printf("%.5f\n", lambda);
	for (i = 0; i < d; i++) {
		hypothesis[i] /= lambda;
		//printf("%.5f\n", lambda);
		//printf("%.5f\n", hypothesis[i]);
		*(dataPoints + i) = hypothesis[i];
	}

	Acopy = reinterpret_cast<long long *>(hostA);
	Bcopy = reinterpret_cast<long long *>(hostB);
	Xcopy = reinterpret_cast<long long *>(hostX);

	/*std::ofstream A, B, X;
	A.open("C:/Users/jxd45/Documents/Python Scripts/A.csv", std::ios::out | std::ios::app);
	B.open("C:/Users/jxd45/Documents/Python Scripts/B.csv", std::ios::out | std::ios::app);
	X.open("C:/Users/jxd45/Documents/Python Scripts/X.csv", std::ios::out | std::ios::app);
	if (A.fail() | B.fail() | X.fail()) {
		std::cout << "Unable to open file";
	}
	else {
		for (i = 0; i < m; i++) {
			for (j = 0; j < m-1; j++) {
				A << *(Acopy + i * m + j) << ",";
			}
			A << *(Acopy + i * m + j) << std::endl;
			B << *(Bcopy + i) << std::endl;
			X << *(Xcopy + i) << std::endl;
		}
	}*/

	/*free(hostA);
	free(hostB);
	free(hostX);
	free(hostLU);
	free(hostIpiv);
	free(hostInfo);
	free(hypothesis);*/
}

void radonInstance(cusolverDnHandle_t cuSolver, int threadId, double *data, int equations, double *solvedEquations)
{
	/*double *hostA, double *hostB, double *hostX, double *LU, int *Ipiv, int *info, int m*/

	cudaStream_t stream = NULL;
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

	c1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	assert(cudaSuccess == c1);

	c1 = cudaMalloc((void**)&d_Ipiv, sizeof(int) * m);
	c2 = cudaMalloc((void**)&d_info, sizeof(int));
	assert(cudaSuccess == c1);
	assert(cudaSuccess == c2);

	mtx.lock();
	status = cusolverDnSetStream(cuSolver, stream);
	//c1 = cudaDeviceSynchronize();
	//c1 = cudaStreamSynchronize(stream);
	//assert(cudaSuccess == c1);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	status = cusolverDnDgetrf_bufferSize(
		cuSolver,
		m,
		m,
		d_A,
		lda,
		&lwork);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	//mtx.unlock();

	c1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
	assert(cudaSuccess == c1);

	/* Perform LU Factorisation*/

	for (int i = 0; i < equations; i++) {

		d_A = (data + (i*m * (m + 1)));
		d_B = (data + (m*m) + (i*m * (m + 1)));
		//c1 = cudaDeviceSynchronize();
		c1 = cudaStreamSynchronize(stream);
		assert(cudaSuccess == c1);
		//printM<< <1, 1 >> > (m,m, d_A,"A");
		//printM << <1, 1 >> > (m, m, d_B, "B");
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
		//c1 = cudaDeviceSynchronize();
		c1 = cudaStreamSynchronize(stream);
		assert(CUSOLVER_STATUS_SUCCESS == status);
		assert(cudaSuccess == c1);

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
	
		//c1 = cudaDeviceSynchronize();
		c1 = cudaStreamSynchronize(stream);
		assert(CUSOLVER_STATUS_SUCCESS == status);
		assert(cudaSuccess == c1);
		//c1 = cudaMemcpy(hostX, d_B, sizeof(double)*m, cudaMemcpyDeviceToHost);

		devMemoryCopy << <1, 1 >> > (d_B, (solvedEquations + (threadId*equations*m) + i * m), m);
	}

	mtx.unlock();

	/* free resources */
	if (d_A) cudaFree(d_A);
	if (d_B) cudaFree(d_B);
	if (d_Ipiv) cudaFree(d_Ipiv);
	if (d_info) cudaFree(d_info);
	if (d_work) cudaFree(d_work);


	if (stream) cudaStreamDestroy(stream);

	//Not
	//cudaDeviceReset();

}

/*double lambda = 0;
	double hypothesis[d];

	for (i = 0; i < d; i++)
		hypothesis[i] = 0;

	for (i = 0; i < m; i++) {
		if (*(hostX + i) < 0) {
			lambda += *(hostX + i);
			for (j = 0; j < d; j++)
				hypothesis[j] += *(hostX + i) * data[i + 1][j];
		}
		else
			continue;
	}*/
