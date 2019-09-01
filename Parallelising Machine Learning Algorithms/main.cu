#include "main.h"

const int print = 1;

//Get the dimensions of the data
//Set Valuues
//h will be a hyper parameter passed to the program.

/*const int d = 12;

const int r = d + 2;

const int h = 5; //Hyper parameter

const int m = d + 1; //Equivalent to d + 1

const int rh = pow(r, h);*/

//"C:\Users\jxd45\Documents\Python Scripts\big.csv"

int main(int argc, char *argv[]) {
	
	/*cudaError_t c1;

	size_t *s =  (size_t *)malloc(sizeof(size_t));
	*s = 1000;
	c1 = cudaDeviceGetLimit(s, cudaLimitPrintfFifoSize);
	assert(cudaSuccess == c1);
	c1 = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, *s * 20);
	assert(cudaSuccess == c1);*/

	int d;
	int h;
	std::string inputFile;

	//printf("%d", argc);

	//if (argc == 4) {
		for(int i=0; i < argc; i++)
			std::cout << argv[i] << std::endl;

		inputFile = argv[1];
		d = atoi(argv[2]);
		h = atoi(argv[3]); //Hyper parameter
	//}

	int m = d + 1; //Equivalent to d + 1
	int r = d + 2; //Radon number
	int rh = pow(r, h);

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

	dataFile.open(inputFile); 
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
	RadonMachineInitialise(d,h,data);
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

void RadonMachineInitialise(int d, int h, double *dataPoints ) {
	int m = d + 1;
	int r = d + 2;
	int rh = pow(r, h);
	
	int h1 = h;
	int hmax;
	int numInstances;
	int i, j;
	double *subsetDataPoints;
	int subsetSize;

	size_t problemAllocation = sizeof(double) * rh * d;
	size_t devFree;
	size_t devTotal;
	

	cudaError_t c1;

	c1 = cudaMemGetInfo(&devFree, &devTotal);
	assert(cudaSuccess == c1);

	if (devFree > problemAllocation)
		RadonMachineInstance(d, h, dataPoints);
	else {

	
		hmax = getMaxAllocation(devFree, d);
		while (h1 > hmax) {
			numInstances = pow(r, h1 - hmax);

			for (i = 0; i < numInstances; i++) {
				subsetSize = pow(r, hmax) * d * sizeof(double);
				subsetDataPoints = dataPoints + (i * subsetSize);
				RadonMachineInstance(d, hmax, subsetDataPoints);
			}
			for (i = 0; i < numInstances; i++) {
				//Collapse Memory
				for (j = 0; j < d; j++)
					*(dataPoints + (i*d) + j) = *(dataPoints + (i*subsetSize) + j);
			}
			h1 -= hmax;
			hmax = getMaxAllocation(devFree, d);
		}
		RadonMachineInstance(d, h1, dataPoints);
	}
}

int getMaxAllocation(size_t mem, int d) {
	size_t m1 = 0;
	int h = 0;
	int r = d + 2;
	int rh;
	do {
		rh = pow(r, h);
		m1 = sizeof(double) * rh * d;
		h++;
	} while (m1 < mem);
	
	return h - 1;
}

void RadonMachineInstance(int d, int h, double *dataPoints) {

	int m = d + 1;
	int r = d + 2;
	int rh = pow(r, h);

	int i, j;
	double *devData;
	double *devEquationData;
	double *devSolvedEquations;
	double *hypothesisWorkspace;

	int *pivotArray;
	int *infoArray;
	double **Aarrays;
	double **Barrays;

	double *currentData;
	int *currentPiv;
	int *currentInfo;
	double **currentAarr;
	double **currentBarr;

	int *devNofEquation;
	int maxThreads = 1; //r
	int threads;
	int noOfEquations;
	int equationsPerThread;
	cudaError_t c1;
	std::vector<std::thread> thVect;
	cudaStream_t *streams = NULL;

	c1 = cudaMalloc(&Aarrays, sizeof(double *) * rh/r * d);
	assert(cudaSuccess == c1);
	c1 = cudaMalloc(&Barrays, sizeof(double *) * rh / r * d);
	assert(cudaSuccess == c1);
	c1 = cudaMalloc(&pivotArray, sizeof(int) * rh/r * d * m);
	assert(cudaSuccess == c1);
	c1 = cudaMalloc(&infoArray, sizeof(int) * rh/r * d);
	assert(cudaSuccess == c1);

	cublasStatus_t cblsStat;
	cublasHandle_t *cblsContexts = (cublasHandle_t *)malloc(maxThreads * sizeof(cublasHandle_t));

	//Create Streams
	streams = (cudaStream_t *)malloc(maxThreads * sizeof(cudaStream_t));

	//Allocate size of heap for device
	//c1 = cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeof(double)*d * 16 * 16 * 8 * 8);
	//assert(cudaSuccess == c1);
	//cudaThreadSetLimit(cudaLimitMallocHeapSize, sizeof(double)*d*16*16*8*8);

	for (i = 0; i < maxThreads; i++) {
		c1 = cudaStreamCreateWithFlags(streams + i, cudaStreamNonBlocking);
		assert(cudaSuccess == c1);
		cblsStat = cublasCreate(cblsContexts + i);
		assert(cblsStat == CUBLAS_STATUS_SUCCESS);
	}

	const dim3 blockSize(16, 16, 1);
	const dim3 gridSize(8, 8, 1);

	//Allocate space for Equations, solved equations and space for data on the device. Then copy data to device.
	//Allocate A and B (A -> (m * m)), (B->1*m)) for r^h instances
	c1 = cudaMalloc(&devEquationData, (sizeof(double) * m * (m + 1))*(rh / r));
	assert(cudaSuccess == c1);
	c1 = cudaMalloc(&devSolvedEquations, (sizeof(double) * m)*(rh / r));
	assert(cudaSuccess == c1);
	c1 = cudaMalloc(&hypothesisWorkspace, (sizeof(double) * d)*(rh / r));
	assert(cudaSuccess == c1);
	c1 = cudaMalloc(&devData, sizeof(double) * rh * d);
	assert(cudaSuccess == c1);
	c1 = cudaMemcpy(devData, dataPoints, sizeof(double) * rh * d, cudaMemcpyHostToDevice);
	assert(cudaSuccess == c1);
	//Maintains the number of equations to be solved at each level of the radon tree
	c1 = cudaMalloc(&devNofEquation, sizeof(int));
	assert(cudaSuccess == c1);
	//printM << <1, 1 >> > (m, m, devData, "A");
	printf("DEV %d\n", devEquationData);

	initAarr << < 1, 1,0 >> > (d, Aarrays, Barrays, (double *)5, rh/r); //initAarr << < gridSize, blockSize >> > (d, Aarrays, Barrays, devEquationData, rh/r);
	cudaDeviceSynchronize();
	printPA << <1, 1, 0 >> > (rh/r, 1, Aarrays, "A");
	cudaDeviceSynchronize();
	//printPA << <1, 1, 0 >> > (rh / r, 1, Barrays, "A");
	for (i = 0; i < h; i++) {
		noOfEquations = pow(r, h - 1 - i);
		cudaMemcpy(devNofEquation, &noOfEquations, sizeof(int), cudaMemcpyHostToDevice);
		configureEquations << < gridSize, blockSize >> > (d, devData, devEquationData, devNofEquation);
		cudaDeviceSynchronize();
		threads = (noOfEquations > maxThreads ? maxThreads : noOfEquations);
		equationsPerThread = noOfEquations / threads;

		//printf("%d threads %d equationsPerThread\n", threads, equationsPerThread);
		cudaDeviceSynchronize();
		//printM << <1, 1, 0 >> > (pow(r, h - i)*d, 1, devData, "A");
		cudaDeviceSynchronize();
		for (j = 0; j < threads; j++) {
			//currentData = (devEquationData + (j*equationsPerThread*m * (m + 1)));
			currentInfo = infoArray + j * equationsPerThread;
			currentAarr = Aarrays + j * equationsPerThread;
			currentBarr = Barrays + j * equationsPerThread;
			currentPiv = pivotArray + (j * m * equationsPerThread);
			thVect.push_back(std::thread(radonInstance, d, cblsContexts + j, j, currentAarr, currentBarr, currentPiv, currentInfo, equationsPerThread, devSolvedEquations, streams + j));
		}
		for (std::thread & th : thVect)
		{
			if (th.joinable())
				th.join();
		}
		thVect.clear();
		solveEquations << < gridSize, blockSize >> > (d, devData, devSolvedEquations, devNofEquation, hypothesisWorkspace);
		cudaDeviceSynchronize();
		//printM << <1, 1, 0 >> > (pow(r, h - i), 1, devData, "A");
	}

	cudaMemcpy(dataPoints, devData, sizeof(double) * rh * d, cudaMemcpyDeviceToHost);

	for (i = 0; i < maxThreads; i++) {
		cblsStat = cublasDestroy(*(cblsContexts + i));
		assert(cblsStat == CUBLAS_STATUS_SUCCESS);
		c1 = cudaStreamDestroy(*(streams + i));
		assert(cudaSuccess == c1);
	}

	c1 = cudaFree(devData);
	assert(cudaSuccess == c1);
	c1 = cudaFree(devSolvedEquations);
	assert(cudaSuccess == c1);
	c1 = cudaFree(devEquationData);
	assert(cudaSuccess == c1);
	c1 = cudaFree(hypothesisWorkspace);
	assert(cudaSuccess == c1);
	c1 = cudaFree(streams);
	assert(cudaSuccess == c1);
	c1 = cudaFree(infoArray);
	assert(cudaSuccess == c1);
	c1 = cudaFree(pivotArray);
	assert(cudaSuccess == c1);
	c1 = cudaFree(Barrays);
	assert(cudaSuccess == c1);
	c1 = cudaFree(Aarrays);
	assert(cudaSuccess == c1);

}

void radonInstance(int d, cublasHandle_t *cublas, int threadId, double **d_A, double **d_B, int *piv, int *info, int equations, double *solvedEquations, cudaStream_t *s)
{
	
	int m = d + 1;
	cublasStatus_t cblsStat;
	int i;
	cblsStat = cublasSetStream(*cublas, *s);
	assert(cblsStat == CUBLAS_STATUS_SUCCESS);

	/*Used to handle generic cuda errors*/
	//cudaError_t c1 = cudaSuccess;
	//cudaError_t c2 = cudaSuccess;

	//double *d_A = NULL; /* device copy of A */
	//double *d_B = NULL; /* device copy of B */
	int *d_Ipiv = NULL; /* pivoting sequence */
	int *d_info = NULL; /* error info for cuSolverDn */
	int  lwork = 0;     /* size of workspace for suSolverDn */
	double *d_work = NULL; /* device workspace for getrf, will be allocated using lwork */

	const int lda = m;
	const int ldb = m;
	double alpha = 1.f;

	cblsStat = cublasDgetrfBatched(*cublas, m, d_A, m, piv, info, equations);
	assert(cblsStat == CUBLAS_STATUS_SUCCESS);
	cblsStat = cublasDtrsmBatched(*cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, m, 1, &alpha, d_A, m, d_B , m, equations);
	//printM << <1, 1, 0, *s >> > (m, 1, *d_B, "b");
	assert(cblsStat == CUBLAS_STATUS_SUCCESS);
	cblsStat = cublasDtrsmBatched(*cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, 1, &alpha, d_A, m, d_B , m, equations);
	//printM(m, 1, *d_B, "b");
	assert(cblsStat == CUBLAS_STATUS_SUCCESS);
	//cudaStreamSynchronize(*s);
	//for (i=0; i<equations; i++)
	//	devMemoryCopy << <1, 1, 0, *s >> > (m, *(d_B)+i, (solvedEquations + (threadId*equations*m) + i * m), m);
	//cudaStreamSynchronize(*s);

}