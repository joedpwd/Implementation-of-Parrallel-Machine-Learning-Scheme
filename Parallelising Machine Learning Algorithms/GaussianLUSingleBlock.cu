
#include "GaussianLUSingleBlock.h"

const int Dimensions = 14;
const int Radon = Dimensions + 2;
const int M = Radon

const int threadsX = 32;
const int threadsY = 32;

__global__ void gaussianLUKernel(float* devGaussianMat/*, const int M*/) {

	//Tell one thread to load float devGaussianMat into a shared memory.
	__shared__ float sharedU[M][M];
	__shared__ float sharedL[M][M];
	__shared__ float sharedP[M][M];
	__shared__ float switchRow[M];

	int maxIndex =0;
	int maxVal = 0;


	int i = 0;
	int j = 0;
	int k = 0;
	int t = 0;
	int x = threadIdx.x;
	int y = threadIdx.y;
	float temp;

	if (y < M && x < M)
		sharedU[y][x] = *(devGaussianMat + y * M + x);
	__syncthreads();

	//printf("Block Dim %d\t%d", blockDim.x, blockDim.y);

	if (M > blockDim.y || M > blockDim.x)
		//CHANGE THIS
		printf("Dimensions are too large\n");
	else {

		for (k = 0; k < M; k++) {

			/*if (y == 1 && x == 1) {

				/*for (t = 0; t < width*height; t++) {
					if (t % width == 0)
						printf("\n");
					printf("%.2f\t", *(devGaussianMat + t));
					printf("\n\n");
				}

				for (t = 0; t < width*height; t++) {
					if (t % width == 0)
						printf("\n");
					printf("%.2f\t", sharedMat[t / width][t % width]);
				}
				printf("\n\n");
			}*/
			maxVal = 0;

			for (i = k; i < M; i++) {
				if (abs(sharedU[i][k] > maxVal)
					maxIndex = i;
			}
			if(maxIndex = k)
				continue;
			else {
				if()
			}

			if (sharedMat[i][i] != 1) {
				temp = sharedMat[i][i];
				if (y == i && x <= width)
					sharedMat[y][x] = sharedMat[y][x] / temp;
				//CHECK FOR 0 ON DIAGONAL
			}
			__syncthreads();

			if (((y < i) || ((y > i) && (y < height))) && x < width)
				sharedMat[y][x] = sharedMat[y][x] - sharedMat[y][i] * sharedMat[i][x];
			__syncthreads();
			/*for (k = 0; k < height; k++) {

				if (k == i)
					continue;
				temp = *(devGaussianMat + k * width + i);
				//printf("%d\n", k*width);
				//printf("temp - %.2f\n", temp);
				for (j = 0; j < width; j++) {
					//printf("k-%d,j-%d val-%.2f\n", k, j, *(devGaussianMat + k * width + j));
					//printf("temp - %.2f\n", temp);
					*(devGaussianMat + k * width + j) = *(devGaussianMat + k * width + j) - (temp * *(devGaussianMat + i * width + j));
					//printf("k-%d,j-%d val-%.2f\n", k,j, *(devGaussianMat + k * width + j));
				}
			}*/
			/*for (t = 0; t < width*height; t++) {
				if (t % width == 0)
					printf("\n");
				printf("%.2f\t", *(devGaussianMat + t));
			}*/
		}

		/*if (y == 1 && x == 1) {
			for (t = 0; t < width*height; t++) {
				if (t % width == 0)
					printf("\n");
				printf("%.2f\t", sharedMat[t / width][t % width]);
			}
			printf("\n\n");
		}*/

		if (y < height && x < width)
			//memcpy((devGaussianMat + y * width + x), (sharedMat + y * width + x), sizeof(float));
			*(devGaussianMat + y * width + x) = sharedMat[y][x];


	}

	/*if (y == 1 && x == 1) {
		for (t = 0; t < width*height; t++) {
			if (t % width == 0)
				printf("\n");
			printf("%.5f\t", *(devGaussianMat + t));
		}
		printf("\n\n");
	}*/

}
int singleBlock()
{

	//TODO
	//Functionality to read a hypothesis, convert to matrix for guassian elimination. Get Dimensions
	//Allocate memory for 2-D arrays

	int width = Radon;
	int height = Dimensions + 1;
	float r;
	cudaEvent_t start, stop;

	float* hostGaussianMat = (float *)malloc(height * width * sizeof(float));

	float* devGaussianMat;

	cudaMalloc(&(devGaussianMat), height * width * sizeof(float));

	dim3 grid(1, 1, 1);
	dim3 block(threadsY, threadsX, 1);
	int i = 0;

	//For now we'll just fill matrices with dummy values

	for (i = 0; i < height*width; i++)
		*(hostGaussianMat + i) = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / 10000.0));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMemcpy(devGaussianMat, hostGaussianMat, height * width * sizeof(float), cudaMemcpyHostToDevice);

	gaussianLUKernel << <grid, block >> > (devGaussianMat/*, Radon*/);

	cudaMemcpy(hostGaussianMat, devGaussianMat, height * width * sizeof(float), cudaMemcpyDeviceToHost);

	/*for (i = 0; i < width*height; i++) {
		if (i % width == 0)
			printf("\n");
		printf("%.2f\t", *(hostGaussianMat + i));
	}*/

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);

	printf("Time %.2f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(devGaussianMat);

	free(hostGaussianMat);
	return 0;
}
